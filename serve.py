#!/usr/bin/env python3
"""Run the project page locally, including byte-range video seeking."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        path = Path(self.translate_path(self.path))
        requested = self.headers.get("Range")
        if not requested or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
        size = path.stat().st_size
        if not match or not size:
            return self.unsatisfiable(size)
        first, last = match.groups()
        if not first and not last:
            return self.unsatisfiable(size)
        if first:
            start = int(first)
            end = min(int(last) if last else size - 1, size - 1)
        else:
            start = max(0, size - int(last))
            end = size - 1
        if start >= size or start > end:
            return self.unsatisfiable(size)
        source = path.open("rb")
        source.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(self.remaining))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return source

    def unsatisfiable(self, size):
        self.send_response(416)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return None

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            while self.remaining > 0:
                chunk = source.read(min(self.remaining, 64 * 1024))
                if not chunk:
                    break
                outputfile.write(chunk)
                self.remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browsers cancel old video requests when switching examples.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    options = parser.parse_args()
    server = ThreadingHTTPServer((options.host, options.port), partial(RangeHandler, directory=str(ROOT)))
    print(f"GRACE demo: http://{options.host}:{options.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
