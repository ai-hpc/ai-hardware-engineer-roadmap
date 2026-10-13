#!/usr/bin/env python3
"""Inventory curriculum structure. Keyword signals guide review; they do not grade quality.

Run from any directory: python3 scripts/audit_curriculum.py [--json]
Includes new nonignored Markdown files, excludes generated docs/site trees.
"""
import argparse
import collections
import json
from pathlib import Path
import re

from check_links import markdown_files, prose_only

SIGNALS = {
    "prerequisite_or_diagnostic": r"prerequisit|entry diagnostic|before you|before starting",
    "practice": r"\blab\b|exercise|build it|projects?",
    "assessment": r"exit criteria|self.check|checkpoint|rubric|acceptance criteria",
    "measurement": r"measur|metrics|benchmark",
    "external_sources": r"https?://",
}


def inventory(root):
    rows = []
    for relative in markdown_files(str(root)):
        path = root / relative
        if not path.is_file():
            continue
        prose = prose_only(path.read_text(encoding="utf-8"))
        is_entry = relative.startswith("Phase ") and path.name in ("Guide.md", "README.md") and "/demos/" not in relative
        rows.append({
            "path": relative,
            "phase": relative.split("/")[0] if relative.startswith("Phase ") else "Repository / projects",
            "course_entry": is_entry,
            "words": len(prose.split()),
            "signals": {key: bool(re.search(pattern, prose, re.I)) for key, pattern in SIGNALS.items()},
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = inventory(root)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"{len(rows)} source Markdown pages; {sum(r['course_entry'] for r in rows)} course entry pages")
    for phase, count in sorted(collections.Counter(r["phase"] for r in rows).items()):
        print(f"{count:4}  {phase}")
    print("\nEntry pages missing review signals (inspect manually; absence is not necessarily a defect):")
    for row in rows:
        if row["course_entry"]:
            missing = [key for key, present in row["signals"].items() if not present]
            if missing:
                print(row["path"] + ": " + ", ".join(missing))


if __name__ == "__main__":
    main()
