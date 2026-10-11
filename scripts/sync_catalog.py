#!/usr/bin/env python3
"""Regenerate browser data, showcase JSON, and Markdown reports from catalog.json."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def showcase_markdown(group, cases):
    lines = [f"# GRACE {group.title()} Showcase", "", "Watch the movement. Compare the outcomes.", ""]
    for case in cases:
        description = case["description"]
        title = description[:1].upper() + description[1:]
        lines.extend([
            f"## {title} — Example {case['example_number']}", "",
            f"[Watch the synchronized comparison]({case['overview_video']})", "",
            "| Method | Outcome | Steps to first success | Video |",
            "| --- | --- | ---: | --- |",
        ])
        for method in case["methods"]:
            outcome = "Success" if method["success"] else "Unsuccessful"
            steps = str(method["recorded_action_steps"]) if method["success"] else "—"
            lines.append(f"| {method['method']} | {outcome} | {steps} | [Watch]({method['video']}) |")
        lines.append("")
    return "\n".join(lines)


def generated(root=ROOT):
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    files = {"assets/data.js": "window.GRACE_DEMO = " +
             json.dumps(catalog, ensure_ascii=False, separators=(",", ":")) + ";\n"}
    for group in ("main", "supplemental"):
        cases = [case for case in catalog["cases"] if case["showcase_group"] == group]
        payload = {"schema_version": catalog["schema_version"], "method": catalog["method"],
                   "showcase_group": group, "cases": cases}
        files[f"{group}_showcase.json"] = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        files[f"{group.upper()}_SHOWCASE.md"] = showcase_markdown(group, cases)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if any generated file is stale.")
    args = parser.parse_args()
    try:
        expected = generated()
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    stale = [name for name, content in expected.items()
             if not (ROOT / name).exists() or (ROOT / name).read_text(encoding="utf-8") != content]
    if args.check:
        print("Stale generated files: " + ", ".join(stale) if stale else "Generated catalog files are up to date.")
        return int(bool(stale))
    for name, content in expected.items():
        (ROOT / name).write_text(content, encoding="utf-8")
    print("Updated " + ", ".join(expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
