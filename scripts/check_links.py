#!/usr/bin/env python3
"""Report markdown links whose target file does not exist.

Usage: check_links.py [ROOT]

Walks tracked and untracked, nonignored (or, outside a git checkout, discovered) ``*.md`` files
under ROOT and resolves each relative link target against the filesystem.
Anchors and external schemes are ignored. Exits non-zero if anything is broken,
so it can be used as a CI gate.
"""
import argparse
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
LINK_ANGLE = re.compile(r'\]\(<([^>]+)>(?:\s+"[^"]*")?\)')


def prose_only(text):
    """Skip fenced and inline code, where array indexing resembles Markdown links."""
    lines, fence_char, fence_length = [], None, 0
    for line in text.splitlines():
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            marker = match.group(1)
            if fence_char is None:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length and not line[match.end():].strip():
                fence_char = None
            continue
        if fence_char is None:
            lines.append(line)
    return re.sub(r'(`+)(?!`)(.*?)\1(?!`)', '', '\n'.join(lines), flags=re.DOTALL)


def markdown_files(root):
    try:
        out = subprocess.run(
            ["git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.md"],
            capture_output=True, text=True, check=True,
        ).stdout
        files = sorted(set(f for f in out.split("\0") if f))
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
            with open(path, encoding="utf-8") as source:
                text = prose_only(source.read())
        except OSError:
            continue
        base = os.path.dirname(path)
        for pattern in (LINK, LINK_WITH_SPACES, LINK_ANGLE):
            for match in pattern.finditer(text):
                href = match.group(1).strip('<>')
                if urllib.parse.urlsplit(href).scheme or href.startswith(("#", "//")):
                    continue
                target = urllib.parse.unquote(href.split("#")[0])
                if not target:
                    continue
                if not os.path.exists(os.path.normpath(os.path.join(base, target))):
                    broken.add(f"{rel} -> {href}")
    return broken


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".",
                        help="existing directory to inspect (default: current directory)")
    args = parser.parse_args(argv)
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        parser.error(f"root must be an existing directory: {root}")
    broken = broken_links(root)
    for entry in sorted(broken):
        print(entry)
    print(f"\n{len(broken)} broken link(s)", file=sys.stderr)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
