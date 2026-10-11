#!/usr/bin/env python3
"""Check local references and published data without downloading media or using npm."""
import argparse
from fractions import Fraction
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MEDIA = {".mp4", ".jpg", ".jpeg", ".png", ".webp", ".ttf", ".pdf", ".zip"}


class Checker:
    def __init__(self, root=ROOT, tracked_assets=False):
        self.root = root.resolve()
        self.tracked_assets = tracked_assets
        self.errors = []
        self.references = set()
        self.tracked = set(subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=self.root).decode().split("\0"))

    def require(self, condition, message):
        if not condition:
            self.errors.append(message)

    def reference(self, value, source="index.html", anchors=None):
        if not isinstance(value, str) or not value:
            self.require(False, f"{source}: empty asset reference")
            return
        url = urlsplit(value)
        if url.scheme or url.netloc:
            return  # Offline checks intentionally do not probe external services.
        if not url.path:
            if url.fragment and anchors is not None:
                self.require(unquote(url.fragment) in anchors,
                             f"{source}: unknown anchor #{url.fragment}")
            return
        path = (self.root / source).parent / unquote(url.path)
        path = path.resolve()
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError:
            self.require(False, f"{source}: reference escapes repository: {value}")
            return
        self.references.add(relative)
        sparse_media = self.tracked_assets and path.suffix.lower() in MEDIA
        self.require(path.is_file() or (path.is_dir() and (path / "index.html").is_file())
                     or (sparse_media and relative in self.tracked),
                     f"{source}: missing local target {value}")
        if path.is_file() and not sparse_media:
            self.require(path.stat().st_size > 0, f"{source}: empty file {value}")

    def documents(self):
        parser = PageParser()
        parser.feed((self.root / "index.html").read_text(encoding="utf-8"))
        self.require(len(parser.ids) == len(set(parser.ids)), "index.html: duplicate element IDs")
        for target in parser.targets:
            self.reference(target, anchors=set(parser.ids))
        for path in self.root.rglob("*.css"):
            if ".git" in path.parts:
                continue
            for target in re.findall(r"url\(\s*['\"]?([^)'\"]+)", path.read_text(encoding="utf-8")):
                self.reference(target.strip(), path.relative_to(self.root).as_posix())
        for path in self.root.rglob("*.md"):
            if ".git" in path.parts:
                continue
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                self.reference(target.split()[0].strip("<>"), path.relative_to(self.root).as_posix())


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.targets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        for key in ("href", "src", "poster"):
            if attrs.get(key):
                self.targets.append(attrs[key])
        if tag == "meta" and attrs.get("property") == "og:image":
            self.targets.append(attrs["content"])


def json_assignment(path, variable):
    text = path.read_text(encoding="utf-8")
    marker = f"window.{variable} = "
    if marker not in text:
        raise ValueError(f"{path}: missing {marker}")
    return json.loads(text.split(marker, 1)[1].strip().removesuffix(";"))


def positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def unique(check, values, name):
    check.require(len(values) == len(set(values)), f"{name}: duplicate identifiers")


def validate_data(check):
    catalog = json.loads((check.root / "catalog.json").read_text())
    browser_data = json_assignment(check.root / "assets/data.js", "GRACE_DEMO")
    check.require(catalog == browser_data, "assets/data.js differs from catalog.json; run scripts/sync_catalog.py")
    cases = catalog["cases"]
    check.require(bool(cases), "catalog: expected at least one example")
    unique(check, [case["case_id"] for case in cases], "cases")
    tasks = [task["key"] for task in catalog["tasks"]]
    unique(check, tasks, "tasks")
    unique(check, catalog["methods"], "methods")
    check.require(positive_number(catalog["fps"]), "catalog: invalid fps")
    video_count = 0
    for case in cases:
        name = case["case_id"]
        check.require(case["task_key"] in tasks, f"{name}: unknown task")
        check.require(case["showcase_group"] in {"main", "supplemental"}, f"{name}: unknown group")
        names = [method["method"] for method in case["methods"]]
        check.require(names == catalog["methods"], f"{name}: missing, duplicated, or reordered methods")
        unique(check, [method["slug"] for method in case["methods"]], f"{name} method slugs")
        for key in ("overview_video", "poster", "thumbnail"):
            check.reference(case[key])
        video_count += 1
        for method in case["methods"]:
            label = f"{name}/{method['method']}"
            check.reference(method["video"])
            check.reference(method["poster"])
            check.require(type(method["success"]) is bool, f"{label}: success must be boolean")
            check.require(positive_number(method["recorded_action_steps"]) and
                          method["recorded_action_steps"] < method["rendered_frames"],
                          f"{label}: invalid step/frame counts")
            video_count += 1
        ours = next(method for method in case["methods"] if method["method"] == "GRACE")
        failed = [method["method"] for method in case["methods"] if not method["success"] and method["method"] != "GRACE"]
        check.require(case["failed_comparators"] == failed and case["advantage"]["failed_comparators"] == failed,
                      f"{name}: failed comparator summary differs from outcomes")
        check.require(case["advantage"]["grace_completion_step"] ==
                      (ours["recorded_action_steps"] if ours["success"] else None),
                      f"{name}: GRACE completion summary differs from outcome")
        successful = [method for method in case["methods"] if method["method"] != "GRACE" and method["success"]]
        step = ours["recorded_action_steps"]
        earliest = min((method["recorded_action_steps"] for method in successful), default=None)
        advantage = case["advantage"]
        expected_faster = [method for method in successful if method["recorded_action_steps"] > step]
        check.require([item["method"] for item in advantage["faster_than"]] ==
                      [method["method"] for method in expected_faster], f"{name}: faster-than summary differs from outcomes")
        for summary, method in zip(advantage["faster_than"], expected_faster):
            check.require(summary["completion_step"] == method["recorded_action_steps"] and
                          summary["steps_saved"] == method["recorded_action_steps"] - step,
                          f"{name}: incorrect steps saved")
        check.require(advantage["tied_completion"] ==
                      [method["method"] for method in successful if method["recorded_action_steps"] == step],
                      f"{name}: tied completion summary differs from outcomes")
        check.require(advantage["earliest_comparator_completion_step"] == earliest,
                      f"{name}: earliest comparator summary differs from outcomes")
        check.require(advantage["earliest_successful_completion"] ==
                      (ours["success"] and (earliest is None or step <= earliest)),
                      f"{name}: earliest success claim differs from outcomes")
        check.require(advantage["strictly_earliest_successful_completion"] ==
                      (ours["success"] and earliest is not None and step < earliest),
                      f"{name}: strictly earliest claim differs from outcomes")
        check.require(advantage["majority_comparators_fail"] == (len(failed) > (len(names) - 1) / 2),
                      f"{name}: majority failure claim differs from outcomes")
    check.require(video_count == catalog["video_count"], "catalog: incorrect video_count")
    check.require(len(catalog["highlights"]) == catalog["presentation_clip_count"], "catalog: incorrect highlight count")
    ids = {case["case_id"] for case in cases}
    for highlight in catalog["highlights"]:
        check.require(highlight["case_id"] in ids, "highlight: unknown case")
        check.reference(highlight["video"])
        check.reference(highlight["poster"])
    check.require(catalog["hero"]["case_id"] in ids, "hero: unknown case")
    for frame in catalog["hero"]["frames"]:
        check.reference(frame["image"])
    for group in ("main", "supplemental"):
        subset = json.loads((check.root / f"{group}_showcase.json").read_text())
        expected = {"schema_version": catalog["schema_version"], "method": catalog["method"],
                    "showcase_group": group, "cases": [case for case in cases if case["showcase_group"] == group]}
        check.require(subset == expected, f"{group}_showcase.json is stale; run scripts/sync_catalog.py")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked-assets", action="store_true",
                        help="Allow media omitted by sparse checkout if present in the Git index.")
    options = parser.parse_args()
    try:
        check = Checker(tracked_assets=options.tracked_assets)
        check.documents()
        validate_data(check)
    except (KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1
    if check.errors:
        print("\n".join(f"ERROR: {error}" for error in check.errors), file=sys.stderr)
        return 1
    print(f"Validated data and {len(check.references)} local references"
          + (" (media checked against Git index)." if options.tracked_assets else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
