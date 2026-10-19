#!/usr/bin/env python3
"""Copy public course pages and runnable source assets into docs/.

Git checkouts include tracked and nonignored new files. Source archives use a
filesystem walk. Discovery must succeed before an existing docs tree is replaced.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT_EXCLUDED = {"docs", "_site", "site"}
EXCLUDED = {"build", "dist", "node_modules", "__pycache__",
            "venv", "env", "env_isaaclab"}
COMMON_SUFFIXES = {".md", ".png", ".jpg", ".jpeg", ".svg", ".gif", ".py", ".c",
                   ".cpp", ".cu", ".h", ".hpp", ".bb", ".bbappend", ".conf", ".css"}
COURSE_SUFFIXES = {".mlir", ".ll", ".sv", ".v", ".json", ".dts", ".html", ".js",
                   ".sh", ".cl", ".cmake", ".toml", ".yaml", ".yml", ".lock"}


def source_path(path: Path) -> bool:
    return path.parts[0] not in ROOT_EXCLUDED and not any(
        part.startswith(".") or part in EXCLUDED for part in path.parts)


def candidates(root: Path) -> list[Path]:
    if (root / ".git").exists():  # includes worktrees, whose .git is a file
        # A broken Git checkout is an error, not an excuse to publish ignored files.
        result = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached",
                                 "--others", "--exclude-standard"], check=True,
                                text=True, capture_output=True)
        return sorted({Path(name) for name in result.stdout.split("\0") if name})
    paths = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = [name for name in dirs if not name.startswith(".") and name not in EXCLUDED
                   and not (Path(directory) == root and name in ROOT_EXCLUDED)]
        paths.extend((Path(directory) / name).relative_to(root) for name in names)
    return sorted(paths)


def publish_asset(path: Path) -> bool:
    if not source_path(path):
        return False
    course = path.parts[0].startswith("Phase ") or path.parts[0] == "Projects"
    if path.suffix.lower() in COMMON_SUFFIXES:
        return True
    if course and (path.suffix.lower() in COURSE_SUFFIXES
                   or path.name in {"Dockerfile", "Makefile", "CMakeLists.txt", "LICENSE"}
                   or (path.name.startswith("requirements") and path.suffix == ".txt")
                   or ("evidence" in path.parts and path.suffix == ".txt")):
        return True
    return (path in {Path("LICENSE"), Path("robots.txt"), Path("llms.txt"),
                     Path("requirements-docs.txt"), Path("mkdocs.yml")}
            or (path.parts[0] == "scripts" and path.suffix == ".sh")
            or (path.parts[0] == "Curriculum-Reviews" and path.suffix == ".json")
            or (path.parts[:2] == ("Assets", "js")
                and (path.suffix == ".js" or path.parts[:4] in {
                    ("Assets", "js", "vendor", "mermaid"),
                    ("Assets", "js", "vendor", "mathjax"),
                })))


def prepare(root: Path) -> int:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("source root must be an existing directory")
    selected = [path for path in candidates(root)
                if publish_asset(path) and (root / path).is_file()]
    if not selected:
        raise ValueError("no public course files found; refusing to publish an empty tree")
    for path in selected:
        if not (root / path).resolve().is_relative_to(root):
            raise ValueError(f"source asset resolves outside the repository: {path}")
    docs = root / "docs"
    with tempfile.TemporaryDirectory(prefix=".docs-stage-", dir=root) as directory:
        stage = Path(directory)
        for path in selected:
            destination = stage / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / path, destination)
        if docs.exists():
            shutil.rmtree(docs)
        stage.rename(docs)
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        count = prepare(args.root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Documentation preparation failed: {error}\n")
    print(f"Prepared {count} public source files at {args.root.resolve() / 'docs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
