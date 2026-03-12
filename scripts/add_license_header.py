#!/usr/bin/env python3
# GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
# Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
# Upstream is included in this repo as a git submodule at: rkhs_ba/
# Upstream file: N/A (no 1:1 mapping claimed)
#
# This GCVO repository is NOT a verbatim copy of RKHS_BA. It includes substantial modifications,
# refactors, and additional original content (e.g., solver/optimization changes and new utilities).
#
# References:
# - RKHS_BA paper: R. Zhang et al., IEEE TPAMI 2025, doi: 10.1109/TPAMI.2025.3593521
# - GCVO paper: R. Zhang et al., CVPR 2026 (see repo README for details)
#
# License: RKHS_BA is MIT-licensed (see rkhs_ba/ for the upstream LICENSE). This repo’s license is in
# the root LICENSE file. Contact (GCVO modifications): ray.zhang@tri.global

"""
add_license_header.py

Prepend a license header from a text file to all source files in the repo.
Skips files that already contain the header. Wraps the header in the
appropriate comment style for each file type.

Examples:
  # Dry run (preview changes)
  python3 scripts/add_license_header.py --license LICENSE_HEADER.txt --dry-run

  # Apply to all source files under gcvo/
  python3 scripts/add_license_header.py --license LICENSE_HEADER.txt --root gcvo/

  # Apply to entire repo (default root is .)
  python3 scripts/add_license_header.py --license LICENSE_HEADER.txt

  # Remove headers instead of adding them
  python3 scripts/add_license_header.py --license LICENSE_HEADER.txt --remove
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

# Extensions to process, grouped by comment style.
C_STYLE_EXTS = {
    ".h", ".hpp", ".hh", ".hxx",
    ".c", ".cc", ".cpp", ".cxx",
    ".cu", ".cuh",
    ".inl",
}
HASH_STYLE_EXTS = {".py", ".cmake", ".sh", ".bash"}
HASH_STYLE_NAMES = {"CMakeLists.txt"}

ALL_EXTS = C_STYLE_EXTS | HASH_STYLE_EXTS
ALL_NAMES = HASH_STYLE_NAMES

IGNORE_DIRS = {
    ".git", "build", "build_debug", "build_release", "build_renamed",
    "cmake-build-debug", "cmake-build-release",
    "__pycache__", ".cache", "third_party", "rkhs_ba",
    "Testing", "external", "llm_outputs",
}


def comment_style(p: Path) -> str:
    if p.suffix in C_STYLE_EXTS:
        return "c"
    if p.suffix in HASH_STYLE_EXTS or p.name in HASH_STYLE_NAMES:
        return "hash"
    return "none"


def wrap_header(raw: str, style: str) -> str:
    """Wrap raw license text in the appropriate comment block."""
    lines = raw.rstrip("\n").splitlines()
    if style == "c":
        body = "\n".join(f" * {ln}".rstrip() for ln in lines)
        return f"/*\n{body}\n */\n\n"
    if style == "hash":
        body = "\n".join(f"# {ln}".rstrip() for ln in lines)
        return f"{body}\n\n"
    return ""


def is_ignored(p: Path) -> bool:
    return any(part in IGNORE_DIRS for part in p.parts)


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1")


def find_header_in_text(text: str, first_line: str) -> bool:
    """Check if the license header (identified by its first line) is present."""
    return first_line in text


def insert_header(text: str, header: str, path: Path) -> str:
    """Insert header at top, preserving shebang / #pragma once ordering."""
    if not header:
        return text

    lines = text.splitlines(True)
    prefix: List[str] = []
    rest_start = 0

    # Preserve shebang
    if lines and lines[0].startswith("#!"):
        prefix.append(lines[0])
        rest_start = 1
        # Preserve encoding cookie (Python)
        if rest_start < len(lines) and re.search(r"coding[:=]\s*[-\w.]+", lines[rest_start]):
            prefix.append(lines[rest_start])
            rest_start += 1

    rest = "".join(lines[rest_start:])
    return "".join(prefix) + header + rest


def remove_header(text: str, wrapped_header: str) -> str:
    """Remove the wrapped header from text if present."""
    if wrapped_header in text:
        return text.replace(wrapped_header, "", 1)
    return text


def iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        if is_ignored(rel):
            continue
        if p.suffix in ALL_EXTS or p.name in ALL_NAMES:
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description="Add or remove a license header to/from all source files.")
    ap.add_argument("--license", required=True, help="Path to plain-text license header file.")
    ap.add_argument("--root", default=".", help="Root directory to walk (default: current dir).")
    ap.add_argument("--dry-run", action="store_true", help="Print what would change without modifying files.")
    ap.add_argument("--remove", action="store_true", help="Remove the header instead of adding it.")
    args = ap.parse_args()

    license_path = Path(args.license)
    if not license_path.exists():
        print(f"Error: license file not found: {license_path}", file=sys.stderr)
        return 1

    raw_header = license_path.read_text(encoding="utf-8")
    if not raw_header.strip():
        print("Error: license file is empty.", file=sys.stderr)
        return 1

    # Use first non-empty line as the detection marker.
    first_line = ""
    for ln in raw_header.strip().splitlines():
        if ln.strip():
            first_line = ln.strip()
            break

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Error: root is not a directory: {root}", file=sys.stderr)
        return 1

    added = 0
    removed = 0
    skipped = 0
    total = 0

    for fpath in iter_files(root):
        total += 1
        style = comment_style(fpath)
        if style == "none":
            continue

        text = read_text(fpath)
        wrapped = wrap_header(raw_header, style)
        has_header = find_header_in_text(text, first_line)

        if args.remove:
            if not has_header:
                skipped += 1
                continue
            new_text = remove_header(text, wrapped)
            if new_text == text:
                # Wrapped form didn't match exactly; skip to avoid corruption.
                print(f"  skip (header form mismatch): {fpath.relative_to(root)}")
                skipped += 1
                continue
            if args.dry_run:
                print(f"  would remove: {fpath.relative_to(root)}")
            else:
                fpath.write_text(new_text, encoding="utf-8")
                print(f"  removed: {fpath.relative_to(root)}")
            removed += 1
        else:
            if has_header:
                skipped += 1
                continue
            new_text = insert_header(text, wrapped, fpath)
            if args.dry_run:
                print(f"  would add: {fpath.relative_to(root)}")
            else:
                fpath.write_text(new_text, encoding="utf-8")
                print(f"  added: {fpath.relative_to(root)}")
            added += 1

    action = "remove" if args.remove else "add"
    count = removed if args.remove else added
    print(f"\n[done] {action}: {count}, skipped: {skipped}, total files: {total}, dry_run: {args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
