from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualization placeholder. Real plots should be generated from evaluation outputs."
    )
    parser.add_argument("--all", action="store_true", help="Reserved for future real plot generation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    print(
        "Error: visualization generation is not implemented yet. "
        "Run real evaluation first, then implement plots from saved metrics.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
