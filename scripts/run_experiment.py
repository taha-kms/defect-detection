
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run multi-class multi-model experiments")
    parser.add_argument("--models", nargs="+", required=True, help="Models to run (e.g., padim patchcore)")
    parser.add_argument("--classes", nargs="+", required=True, help="MVTec AD classes to run")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    for model in args.models:
        for cls in args.classes:
            # Train
            run([
                sys.executable, "-m", "src.train",
                "--model", model,
                "--class_name", cls,
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
            ])
            # Evaluate
            run([
                sys.executable, "-m", "src.eval",
                "--model", model,
                "--class_name", cls,
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
            ])

    print("All runs completed.")


if __name__ == "__main__":
    main()