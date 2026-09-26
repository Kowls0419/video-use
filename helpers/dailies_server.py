"""Dailies — interactive video review server (the visual feedback tool).

Serves a rendered preview to a local web page where the user scrubs the video,
drops timestamped comments, and draws pen/arrow/box annotations directly on the
frame. Every note is written STRAIGHT into `<edit>/review/` — a compact JSON plus
one flattened PNG per annotation (the video frame with the drawing baked in). The
assistant then reads that JSON and those PNGs directly, instead of the user having
to type a long "at 1:32, text is cropped on the right, move it up" description.

This is the interactive successor to `timeline_view.py` (a static filmstrip PNG):
same edit-dir auto-resolution, but live and two-way. It feeds the `reflect` skill's
learning loop — each round's note count is the convergence signal.

Usage:
    python helpers/dailies_server.py <preview.mp4>
    python helpers/dailies_server.py <preview.mp4> --port 8756 --round 3
    python helpers/dailies_server.py <preview.mp4> --edit-dir /path/to/edit

Output (round NN auto-increments per video unless --round is given):
    <edit>/review/<stem>_rNN.json
    <edit>/review/frames/<stem>_rNN_<idx>_<t>.png

Stdlib only. Ctrl-C to stop; the JSON is complete after every note (safe to kill).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "dailies.html"


def probe_fps(path: Path) -> float:
    """Frames per second via ffprobe's r_frame_rate (num/den). Defaults to 24."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1",
         str(path)],
        capture_output=True, text=True).stdout.strip()
    if "/" in out:
        num, den = out.split("/")[:2]
        try:
            n, d = float(num), float(den)
            if d:
                return n / d
        except ValueError:
            pass
    return 24.0


def resolve_edit_dir(video: Path, override: Path | None) -> Path:
    """Where session outputs live. Mirrors timeline_view.py's convention:
    if the video already sits in an `edit/` dir, use it; else <parent>/edit."""
    if override:
        return override.expanduser().resolve()
    parent = video.parent
    if parent.name == "edit":
        return parent
    cand = parent / "edit"
    return cand if cand.exists() else parent


def next_round(review_dir: Path, stem: str) -> int:
    """Highest existing <stem>_rNN.json + 1, or 1 if none."""
    hi = 0
    for p in review_dir.glob(f"{stem}_r*.json"):
        m = re.search(r"_r(\d+)\.json$", p.name)
        if m:
            hi = max(hi, int(m.group(1)))
    return hi + 1


def timecode(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:06.3f}"


class ReviewState:
    """Holds paths + the growing note list; serializes on every note."""

    def __init__(self, video: Path, edit_dir: Path, rnd: int, fps: float):
        self.video = video
        self.fps = fps
        self.round = rnd
        self.stem = video.stem
        self.review_dir = edit_dir / "review"
        self.frames_dir = self.review_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.review_dir / f"{self.stem}_r{rnd:02d}.json"
        self.lock = threading.Lock()
        self.notes: list[dict] = []
        if self.json_path.exists():  # resume a round in progress
            try:
                self.notes = json.loads(self.json_path.read_text()).get("notes", [])
            except (json.JSONDecodeError, OSError):
                self.notes = []

    def doc(self) -> dict:
        return {
            "video": self.video.name,
            "round": self.round,
            "fps": round(self.fps, 4),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": self.notes,
        }

    def add_note(self, t: float, comment: str, tool: str, strokes,
                 frame_png_b64: str | None) -> dict:
        with self.lock:
            i = len(self.notes) + 1
            frame_rel = None
            if frame_png_b64:
                header, _, data = frame_png_b64.partition(",")  # strip data: URL prefix
                raw = base64.b64decode(data or header)
                fname = f"{self.stem}_r{self.round:02d}_{i:02d}_{t:.2f}.png"
                (self.frames_dir / fname).write_bytes(raw)
                frame_rel = f"frames/{fname}"
            note = {
                "i": i,
                "t": round(t, 3),
                "tc": timecode(t),
                "comment": comment.strip(),
                "tool": tool,
                "strokes": strokes or [],
                "frame": frame_rel,
            }
            self.notes.append(note)
            self.json_path.write_text(
                json.dumps(self.doc(), ensure_ascii=False, indent=2))
            return note


def make_handler(state: ReviewState):
    video_path = state.video
    video_size = video_path.stat().st_size

    class Handler(BaseHTTPRequestHandler):
        # keep the console quiet except for our own prints
        def log_message(self, *args):  # noqa: D401
            pass

        def _send(self, code, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                html = HTML_PATH.read_text(encoding="utf-8")
                cfg = json.dumps({
                    "video": video_path.name,
                    "fps": state.fps,
                    "round": state.round,
                    "existing": state.notes,
                })
                html = html.replace("/*__CONFIG__*/null", cfg)
                self._send(HTTPStatus.OK, html.encode("utf-8"),
                           "text/html; charset=utf-8")
            elif path == "/video":
                self._serve_video()
            elif path == "/review":
                body = json.dumps(state.doc(), ensure_ascii=False).encode("utf-8")
                self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/note":
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(HTTPStatus.BAD_REQUEST, b'{"error":"bad json"}',
                           "application/json")
                return
            note = state.add_note(
                t=float(payload.get("t", 0.0)),
                comment=str(payload.get("comment", "")),
                tool=str(payload.get("tool", "")),
                strokes=payload.get("strokes"),
                frame_png_b64=payload.get("frame_png"),
            )
            print(f"  note #{note['i']} @ {note['tc']}  {note['comment']!r}"
                  + (f"  [{note['frame']}]" if note["frame"] else ""))
            body = json.dumps({"ok": True, "note": note},
                              ensure_ascii=False).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")

        def _serve_video(self):
            """Stream the mp4 with HTTP Range support so <video> can seek/scrub.
            Without 206 Partial Content most browsers refuse to scrub."""
            rng = self.headers.get("Range")
            ctype = "video/mp4"
            if not rng:
                with open(video_path, "rb") as f:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(video_size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    if self.command != "HEAD":
                        self._copy(f, video_size)
                return
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if not m:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{video_size}")
                self.end_headers()
                return
            start = int(m.group(1)) if m.group(1) else 0
            end = int(m.group(2)) if m.group(2) else video_size - 1
            end = min(end, video_size - 1)
            start = min(start, end)
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{video_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            if self.command != "HEAD":
                with open(video_path, "rb") as f:
                    f.seek(start)
                    self._copy(f, length)

        def _copy(self, f, length: int, chunk: int = 256 * 1024):
            remaining = length
            while remaining > 0:
                buf = f.read(min(chunk, remaining))
                if not buf:
                    break
                try:
                    self.wfile.write(buf)
                except (BrokenPipeError, ConnectionResetError):
                    break  # browser closed the range early — normal while scrubbing
                remaining -= len(buf)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Dailies — interactive video review server")
    ap.add_argument("video", type=Path, help="Rendered preview mp4 to review")
    ap.add_argument("--port", type=int, default=8756)
    ap.add_argument("--round", type=int, default=None,
                    help="Review round number (default: auto-increment per video)")
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Override the edit/ output dir (default: auto from video path)")
    ap.add_argument("--no-open", action="store_true",
                    help="Do not auto-open the browser")
    args = ap.parse_args()

    video = args.video.expanduser().resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    if not HTML_PATH.exists():
        sys.exit(f"dailies.html missing next to this script: {HTML_PATH}")

    edit_dir = resolve_edit_dir(video, args.edit_dir)
    review_dir = edit_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    fps = probe_fps(video)
    rnd = args.round if args.round is not None else next_round(review_dir, video.stem)

    state = ReviewState(video, edit_dir, rnd, fps)
    handler = make_handler(state)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"

    print(f"Dailies review server  →  {url}")
    print(f"  video : {video.name}  ({fps:g} fps)")
    print(f"  round : r{rnd:02d}")
    print(f"  writes: {state.json_path}")
    print(f"          {state.frames_dir}/")
    print("  keys  : Space play/pause · ←/→ frame · Shift+←/→ 1s · Enter comment")
    print("  Ctrl-C to stop (JSON is saved after every note).")

    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped. {len(state.notes)} note(s) → {state.json_path}")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
