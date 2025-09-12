from importlib import import_module
import sys

COMMANDS = {"prepare", "train", "test", "eval", "report"}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python -m src <prepare|train|test|eval|report> [args]")
        sys.exit(1)

    cmd = sys.argv[1]
    module = import_module(f"src.cli.{cmd}")
    module.main()

if __name__ == "__main__":
    main()

    