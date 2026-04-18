#!/usr/bin/env python3
"""
align.py — sync file birth dates from markdown frontmatter.

Usage: python align.py <directory> [--dry-run]

Reads `created` or `date` from each .md file's YAML frontmatter and
overwrites the file's birth date (bday) with that value via SetFile.
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
DATE_FIELDS = ("created", "date")


def parse_frontmatter(text: str) -> dict:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    block = m.group(1)
    result = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def extract_date(fm: dict) -> datetime | None:
    for field in DATE_FIELDS:
        raw = fm.get(field)
        if not raw:
            continue
        raw = raw.strip().strip("\"'")
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return None


def get_birth_date(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_birthtime)


def set_birth_date(path: Path, dt: datetime, dry_run: bool) -> bool:
    # SetFile -d expects "MM/DD/YYYY HH:MM:SS"
    setfile_date = dt.strftime("%m/%d/%Y %H:%M:%S")
    if dry_run:
        return True
    result = subprocess.run(
        ["SetFile", "-d", setfile_date, str(path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Align .md file birth dates from frontmatter.")
    parser.add_argument("directory", type=Path, help="Directory containing .md files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(root.rglob("*.md"))
    if not md_files:
        print("No .md files found.")
        return

    results = []  # (rel_path, date_str | None, status)

    for path in md_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        fm = parse_frontmatter(text)
        dt = extract_date(fm)
        rel = path.relative_to(root)

        if dt is None:
            results.append((rel, None, "no date"))
            continue

        current = get_birth_date(path)
        if current.date() == dt.date():
            results.append((rel, dt.strftime("%Y-%m-%d"), "same"))
            continue

        ok = set_birth_date(path, dt, args.dry_run)
        results.append((rel, dt.strftime("%Y-%m-%d"), "ok" if ok else "error"))

    # --- report ---
    label = "[dry-run] " if args.dry_run else ""
    print(f"\n{label}set_date.py — {root}\n")

    col = max((len(str(r[0])) for r in results), default=20)
    STATUS = {"ok": "✓", "same": "=", "no date": "—", "error": "✗"}
    for rel, date_str, status in results:
        date_col = date_str if date_str else "----------"
        print(f"  {str(rel):<{col}}  {date_col}  {STATUS[status]}")

    n_ok = sum(1 for r in results if r[2] == "ok")
    n_same = sum(1 for r in results if r[2] == "same")
    n_skip = sum(1 for r in results if r[2] == "no date")
    n_err = sum(1 for r in results if r[2] == "error")
    print(f"\n  총 {len(results)}개  |  적용 {n_ok}  |  동일 {n_same}  |  날짜없음 {n_skip}  |  오류 {n_err}")


if __name__ == "__main__":
    main()
