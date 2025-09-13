
import argparse
from email import parser
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
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    for model in args.models:
        for cls in args.classes:
            # Train
            run([
                sys.executable, 
                "-m", 
                "src.train",
                "--model", model, 
                "--class_name", cls,
                "--config", args.config, *sum([["--extra", e] for e in args.extra], []),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers)
            ])
            # Evaluate
            run([
                "-m", 
                "src.eval",
                "--model", model, 
                "--class_name", cls,
                "--config", args.config, *sum([["--extra", e] for e in args.extra], []),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers)
            ])

    print("All runs completed.")


if __name__ == "__main__":
    main()