#!/usr/bin/env python3
"""Report markdown links whose target file does not exist.

Usage: check_links.py [ROOT]

Walks every tracked (or, outside a git checkout, every discovered) ``*.md`` file
under ROOT and resolves each relative link target against the filesystem.
Anchors and external schemes are ignored. Exits non-zero if anything is broken,
so it can be used as a CI gate.
"""
import os
import re
import subprocess
import sys
import urllib.parse

SKIP_PREFIXES = ("_site/", "docs/")
# Targets normally contain no whitespace. Unencoded paths with literal spaces are
# also valid and are the easy ones to miss in a rename, so they get a second
# pattern restricted to *.md targets — broad enough to catch them, narrow enough
# not to match the link-shaped math and code that appears in lecture prose.
LINK = re.compile(r'\]\(([^)\s]+?)(?:\s+"[^"]*")?\)')
LINK_WITH_SPACES = re.compile(r'\]\(([^)]+?\.md)(?:#[^)]*)?\)')


def markdown_files(root):
    try:
        out = subprocess.run(
            ["git", "-C", root, "ls-files", "*.md"],
            capture_output=True, text=True, check=True,
        ).stdout
        files = [f for f in out.split("\n") if f]
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = []
    if not files:
        files = [
            os.path.relpath(os.path.join(dirpath, name), root)
            for dirpath, _, names in os.walk(root)
            for name in names
            if name.endswith(".md")
        ]
    return [f for f in files if not f.startswith(SKIP_PREFIXES)]


def broken_links(root):
    broken = set()
    for rel in markdown_files(root):
        path = os.path.join(root, rel)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        base = os.path.dirname(path)
        for pattern in (LINK, LINK_WITH_SPACES):
            for match in pattern.finditer(text):
                href = match.group(1)
                if href.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = urllib.parse.unquote(href.split("#")[0])
                if not target:
                    continue
                if not os.path.exists(os.path.normpath(os.path.join(base, target))):
                    broken.add(f"{rel} -> {href}")
    return broken


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    broken = broken_links(root)
    for entry in sorted(broken):
        print(entry)
    print(f"\n{len(broken)} broken link(s)", file=sys.stderr)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
