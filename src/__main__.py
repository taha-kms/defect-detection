from __future__ import annotations
import sys


# COMMANDS = {"prepare", "train", "test", "eval", "report"}

def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _die(
            "Usage: python -m src <train|eval|report> [args]\n"
            "Examples:\n"
            "  python -m src train  --help\n"
            "  python -m src eval   --help\n"
            "  python -m src report --help"
        )

    cmd, *rest = argv

    if cmd == "train":
        from src import train as _train
        # If src.train.main accepts argv, pass it; otherwise call without and let it read sys.argv.
        return int(_train.main(rest) if hasattr(_train, "main") else _train.main())

    if cmd == "eval":
        from src import eval as _eval  # noqa: A001
        return int(_eval.main(rest) if hasattr(_eval, "main") else _eval.main())

    if cmd == "report":
        from src.reporting import report as _report
        return int(_report.main(rest) if hasattr(_report, "main") else _report.main())

    _die(f"Unknown subcommand: {cmd!r}\nExpected one of: train, eval, report")

if __name__ == "__main__":
    sys.exit(main())