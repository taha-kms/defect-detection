"""
Unified CLI entry point for the project.

Run:
  python -m src train   --help
  python -m src eval    --help
  python -m src report  --help
  python -m src prepare --help
"""

from __future__ import annotations
import sys

USAGE = """\
Usage:
  python -m src <train|eval|report|prepare> [args]

Examples:
  python -m src train   --help
  python -m src eval    --help
  python -m src report  --help
  python -m src prepare --help
"""


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def _delegate(module, prog: str, rest: list[str]) -> int:
    old_argv = sys.argv
    try:
        sys.argv = [prog, *rest]
        return int(module.main())
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    cmd, *rest = argv

    if cmd == "train":
        from src import train as _train
        return _delegate(_train, "python -m src train", rest)

    if cmd == "eval":
        from src import eval as _eval  # noqa: A001
        return _delegate(_eval, "python -m src eval", rest)

    if cmd == "report":
        from src.reporting import report as _report
        return _delegate(_report, "python -m src report", rest)

    if cmd == "prepare":
        from src import prepare as _prepare
        return _delegate(_prepare, "python -m src prepare", rest)

    _die(f"Unknown subcommand: {cmd!r}\nExpected one of: train, eval, report, prepare")
    return 2


if __name__ == "__main__":
    sys.exit(main())
