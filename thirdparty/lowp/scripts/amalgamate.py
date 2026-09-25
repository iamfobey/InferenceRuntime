#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]\s*$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the lowp single-header distribution.")
    parser.add_argument("--source-root", type=Path, required=True, help="Path to include/lowp")
    parser.add_argument("--entry", type=Path, required=True, help="Entry header relative to --source-root")
    parser.add_argument("--output", type=Path, required=True, help="Generated single header")
    parser.add_argument("--check", action="store_true", help="Fail if --output is not up to date")
    return parser.parse_args()


def resolve_internal_include(current: Path, include_name: str, source_root: Path) -> Path | None:
    if include_name.startswith("lowp/"):
        candidate = source_root / include_name.removeprefix("lowp/")
        return candidate if candidate.is_file() else None

    candidate = current.parent / include_name
    if candidate.is_file() and source_root in candidate.resolve().parents:
        return candidate

    return None


def expand_header(path: Path, source_root: Path, visited: set[Path], external_includes: set[str]) -> list[str]:
    path = path.resolve()
    if path in visited:
        return []

    visited.add(path)
    relative = path.relative_to(source_root.resolve()).as_posix()
    output = [f"// ---- lowp/{relative} ----"]

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = raw_line.strip()
        if stripped == "#pragma once":
            continue

        match = INCLUDE_RE.match(raw_line)
        if match:
            include_name = match.group(2)
            internal = resolve_internal_include(path, include_name, source_root.resolve())
            if internal is not None:
                output.extend(expand_header(internal, source_root, visited, external_includes))
                continue

            normalized = stripped
            if normalized in external_includes:
                continue
            external_includes.add(normalized)

        output.append(raw_line)

    output.append("")
    return output


def generate(source_root: Path, entry: Path) -> str:
    source_root = source_root.resolve()
    entry_path = (source_root / entry).resolve()

    if not entry_path.is_file():
        raise FileNotFoundError(f"Entry header not found: {entry_path}")

    if source_root not in entry_path.parents:
        raise ValueError("Entry header must be inside source root")

    body = expand_header(entry_path, source_root, set(), set())
    lines = [
        "// Generated file. Edit include/lowp/* and run scripts/amalgamate.py instead.",
        "#pragma once",
        "",
        *body,
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    generated = generate(args.source_root, args.entry)

    if args.check:
        if not args.output.is_file():
            print(f"Single header is missing: {args.output}", file=sys.stderr)
            return 1
        if args.output.read_text(encoding="utf-8") != generated:
            print(f"Single header is out of date: {args.output}", file=sys.stderr)
            return 1
        print(f"Single header is up to date: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8", newline="\n")
    print(f"Generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
