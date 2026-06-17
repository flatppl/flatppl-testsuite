"""CLI entry: `python -m flatppl_testsuite [--id ID]... [--oracle roofit|pyhs3]...`"""

from __future__ import annotations

import argparse
import sys

from .runner import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flatppl_testsuite")
    parser.add_argument("--id", action="append", default=[], help="fixture test_id (repeatable)")
    parser.add_argument("--oracle", action="append", default=[], choices=["roofit", "pyhs3"],
                        help="additionally cross-check against this oracle (repeatable)")
    args = parser.parse_args(argv)

    selected = set(args.id) or None
    results = run(selected=selected, oracles=tuple(args.oracle))

    failed = 0
    for r in results:
        line = f"{r.status.upper():7} {r.test_id}::{r.check_id}"
        if r.tag:
            line += f"  [{r.tag}]"
        if r.message:
            line += f"  {r.message}"
        print(line)
        if r.status == "failed":
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
