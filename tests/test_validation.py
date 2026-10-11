"""Protect failure cases that a successful demo checkout cannot exercise."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate


class LinkValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        with patch("validate.subprocess.check_output", return_value=b"assets/video.mp4\0"):
            self.check = validate.Checker(self.root, tracked_assets=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_sparse_assets_require_a_tracked_path(self):
        self.check.reference("assets/video.mp4")
        self.assertEqual(self.check.errors, [])
        self.check.reference("assets/missing.mp4")
        self.assertTrue(self.check.errors)

    def test_missing_source_files_and_traversal_are_rejected(self):
        self.check.reference("app.js")
        self.check.reference("../outside.mp4")
        self.assertEqual(len(self.check.errors), 2)

    def test_query_strings_and_local_anchors(self):
        (self.root / "app.js").write_text("// example\n")
        self.check.reference("app.js?v=1")
        self.check.reference("#gallery", anchors={"gallery"})
        self.assertEqual(self.check.errors, [])
        self.check.reference("#absent", anchors={"gallery"})
        self.assertEqual(len(self.check.errors), 1)


class CorruptDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "assets").mkdir()
        with patch("validate.subprocess.check_output", return_value=b""):
            self.check = validate.Checker(self.root)
        self.check.reference = lambda *args, **kwargs: None

    def tearDown(self):
        self.temp.cleanup()

    def test_stale_browser_payload_and_outcomes_are_rejected(self):
        catalog = json.loads((ROOT / "catalog.json").read_text())
        (self.root / "assets/data.js").write_text("window.GRACE_DEMO = " + json.dumps(catalog) + ";")
        catalog["cases"][0]["methods"][-1]["recorded_action_steps"] = 450
        (self.root / "catalog.json").write_text(json.dumps(catalog))
        for group in ("main", "supplemental"):
            (self.root / f"{group}_showcase.json").write_text((ROOT / f"{group}_showcase.json").read_text())
        validate.validate_data(self.check)
        self.assertTrue(any("differs from catalog" in error for error in self.check.errors))
        self.assertTrue(any("completion summary" in error for error in self.check.errors))
        self.assertTrue(any("is stale" in error for error in self.check.errors))
