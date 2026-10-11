"""Updates to the canonical catalog must reach every representation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sync_catalog


class CatalogSyncTest(unittest.TestCase):
    def test_canonical_change_updates_browser_json_and_markdown(self):
        catalog = json.loads((ROOT / "catalog.json").read_text())
        catalog["cases"][0]["methods"][-1]["recorded_action_steps"] = 400
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "catalog.json").write_text(json.dumps(catalog))
            generated = sync_catalog.generated(root)
        self.assertEqual(len(generated), 5)
        browser = json.loads(generated["assets/data.js"].split(" = ", 1)[1].rstrip(";\n"))
        subset = json.loads(generated["main_showcase.json"])
        self.assertEqual(browser["cases"][0]["methods"][-1]["recorded_action_steps"], 400)
        self.assertEqual(subset["cases"][0]["methods"][-1]["recorded_action_steps"], 400)
        self.assertIn("| GRACE | Success | 400 |", generated["MAIN_SHOWCASE.md"])
        for name, content in sync_catalog.generated().items():
            self.assertEqual(content, (ROOT / name).read_text(), name)
