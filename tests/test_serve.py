"""Exercise browser-style media requests over a real local HTTP connection."""
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import RangeHandler


class QuietHandler(RangeHandler):
    def log_message(self, *args):
        pass


class RangeServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.payload = bytes(range(256)) * 512
        (Path(cls.temp.name) / "video.mp4").write_bytes(cls.payload)
        (Path(cls.temp.name) / "empty.mp4").write_bytes(b"")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=cls.temp.name))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(self, byte_range=None, method="GET", path="/video.mp4"):
        conn = HTTPConnection(*self.server.server_address, timeout=5)
        try:
            conn.request(method, path, headers={"Range": byte_range} if byte_range else {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_full_response_and_content_type(self):
        status, headers, body = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(body, self.payload)
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        self.assertEqual(headers.get("Content-Type", headers.get("Content-type")), "video/mp4")

    def test_bounded_open_and_suffix_ranges(self):
        size = len(self.payload)
        for header, start, end in (("bytes=100-90000", 100, 90000), ("bytes=90000-", 90000, size-1),
                                   ("bytes=-100", size-100, size-1), ("bytes=0-999999", 0, size-1)):
            with self.subTest(header=header):
                status, headers, body = self.request(header)
                self.assertEqual(status, 206)
                self.assertEqual(body, self.payload[start:end+1])
                self.assertEqual(headers["Content-Length"], str(end-start+1))
                self.assertEqual(headers["Content-Range"], f"bytes {start}-{end}/{size}")

    def test_unsatisfiable_or_unsupported_range(self):
        for header in ("bytes=999999-", "bytes=5-2", "bytes=-0", "bytes=-", "bytes=0-1,3-4", "invalid"):
            with self.subTest(header=header):
                status, headers, body = self.request(header)
                self.assertEqual(status, 416)
                self.assertEqual(headers["Content-Range"], f"bytes */{len(self.payload)}")
                self.assertEqual(body, b"")
        status, headers, body = self.request("bytes=0-", path="/empty.mp4")
        self.assertEqual(status, 416)
        self.assertEqual(headers["Content-Range"], "bytes */0")

    def test_head_and_missing_file(self):
        status, headers, body = self.request("bytes=10-19", method="HEAD")
        self.assertEqual(status, 206)
        self.assertEqual(headers["Content-Length"], "10")
        self.assertEqual(body, b"")
        self.assertEqual(self.request("bytes=0-", path="/missing.mp4")[0], 404)
