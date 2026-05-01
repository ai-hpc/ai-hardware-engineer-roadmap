#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$ROOT_DIR/docs"

rm -rf "$DOCS_DIR"
mkdir -p "$DOCS_DIR"

copy_path() {
  local path="$1"
  mkdir -p "$DOCS_DIR/$(dirname "$path")"
  cp "$ROOT_DIR/$path" "$DOCS_DIR/$path"
}

while IFS= read -r -d '' path; do
  copy_path "$path"
done < <(
  git -C "$ROOT_DIR" ls-files -z -- \
    '*.md' \
    '*.png' \
    '*.jpg' \
    '*.jpeg' \
    '*.svg' \
    '*.gif' \
    '*.py' \
    '*.css'
)

for static_file in robots.txt llms.txt; do
  if [[ -f "$ROOT_DIR/$static_file" ]]; then
    copy_path "$static_file"
  fi
done

while IFS= read -r -d '' path; do
  copy_path "$path"
done < <(
  git -C "$ROOT_DIR" ls-files -z -o --exclude-standard -- \
    '*.md' \
    '*.png' \
    '*.jpg' \
    '*.jpeg' \
    '*.svg' \
    '*.gif' \
    '*.py' \
    '*.css'
)

echo "Prepared docs tree at $DOCS_DIR"
