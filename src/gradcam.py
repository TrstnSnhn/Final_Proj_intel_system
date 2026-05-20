from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grad-CAM entrypoint placeholder. Real Grad-CAM is planned for Phase 2B."
    )
    parser.add_argument("--checkpoint", help="Path to a trained checkpoint.")
    parser.add_argument("--class-map", help="Path to class mapping JSON.")
    parser.add_argument("--image", help="Path to an image to explain.")
    parser.add_argument("--output", help="Output image path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    print(
        "Error: real Grad-CAM generation is not implemented yet. "
        "Train a checkpoint first, then implement Phase 2B Grad-CAM using grad-cam.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
