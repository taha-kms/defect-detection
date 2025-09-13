"""
Dataset preparation / inspection utilities for MVTec-style datasets.

Typical layouts supported (auto-detected):
  <DATA_DIR>/<class>/{train,test,ground_truth}/...
  <DATA_DIR>/mvtec/<class>/...
  <DATA_DIR>/mvtec_ad/<class>/...

Examples:
  python -m src prepare --data-dir ./data --list-classes
  python -m src prepare --data-dir ./data --verify
  python -m src prepare --data-dir ./data --write-classfile classes.txt
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple


# ------------------------ filesystem helpers ------------------------ #

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _looks_like_class_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    # MVTec classes usually contain at least one of these
    for sub in ("train", "test", "ground_truth"):
        if (p / sub).exists():
            return True
    return False


def _detect_root(data_dir: Path) -> Path:
    """
    Try to find the directory that directly contains the class folders.
    We accept DATA_DIR itself or nested mvtec/mvtec_ad.
    """
    candidates = [data_dir, data_dir / "mvtec", data_dir / "mvtec_ad"]
    for root in candidates:
        if root.exists() and any(_looks_like_class_dir(c) for c in root.iterdir()):
            return root
    # fallback: most likely DATA_DIR; caller will handle "no classes found"
    return data_dir


def _iter_classes(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if _looks_like_class_dir(p)]


def _count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _verify_class(cdir: Path) -> Tuple[bool, List[str]]:
    """
    Basic structure checks for a class directory:
      - has train/ and test/
      - train/good exists and has images
      - test has at least one image
      - if ground_truth exists, masks roughly correspond to non-good defects
    """
    problems: List[str] = []
    train = cdir / "train"
    test = cdir / "test"
    gt = cdir / "ground_truth"

    if not train.exists():
        problems.append("missing 'train/'")
    if not test.exists():
        problems.append("missing 'test/'")

    # train/good count
    tgood = train / "good"
    if not tgood.exists():
        problems.append("missing 'train/good/'")
    tg_n = _count_images(tgood)

    # test counts (all subfolders)
    test_total = 0
    test_defect_total = 0
    if test.exists():
        for sub in sorted(p for p in test.iterdir() if p.is_dir()):
            n = _count_images(sub)
            test_total += n
            if sub.name != "good":
                test_defect_total += n

    if tg_n == 0:
        problems.append("no images in 'train/good/'")
    if test_total == 0:
        problems.append("no images in 'test/'")

    # ground truth rough correspondence
    if gt.exists():
        gt_total = _count_images(gt)
        if test_defect_total and gt_total == 0:
            problems.append("ground_truth/ has no masks, but test has defect images")
        # We don't enforce 1:1 matching here (file names/types vary), just sanity-check scale
        if test_defect_total and gt_total and (gt_total < test_defect_total * 0.3):
            problems.append(
                f"suspicious few masks: {gt_total} vs ~{test_defect_total} defect images"
            )

    return (len(problems) == 0, problems)


# ------------------------------ CLI ------------------------------ #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src prepare",
        description="Prepare/inspect MVTec-style datasets.",
    )
    p.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Path to dataset directory (e.g., ./data or ./data/mvtec).",
    )
    p.add_argument(
        "--list-classes",
        action="store_true",
        help="List detected classes under the dataset root.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Verify basic structure for each class and report issues.",
    )
    p.add_argument(
        "--write-classfile",
        type=Path,
        default=None,
        help="Optional path to write class names (one per line).",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = _detect_root(args.data_dir)
    classes = _iter_classes(root)

    if not classes:
        print(f"[prepare] No classes found under: {root}")
        print("Hint: expected layout like <root>/<class>/{train,test,ground_truth}/")
        return 2

    # --list-classes
    if args.list_classes:
        print(f"[prepare] Dataset root: {root}")
        for c in classes:
            print(c.name)

    # --write-classfile
    if args.write_classfile:
        args.write_classfile.parent.mkdir(parents=True, exist_ok=True)
        with args.write_classfile.open("w", encoding="utf-8") as f:
            for c in classes:
                f.write(c.name + "\n")
        print(f"[prepare] Wrote classes → {args.write_classfile}")

    # --verify
    if args.verify:
        print(f"[prepare] Verifying classes under: {root}")
        ok_total = 0
        for c in classes:
            ok, problems = _verify_class(c)
            if ok:
                ok_total += 1
                print(f"  ✓ {c.name}")
            else:
                print(f"  ✗ {c.name}")
                for msg in problems:
                    print(f"     - {msg}")
        print(f"[prepare] Summary: {ok_total}/{len(classes)} classes passed basic checks.")
        # Non-zero exit code if any class failed checks
        if ok_total != len(classes):
            return 3

    # If no actionable flags were given, show a quick summary
    if not (args.list_classes or args.verify or args.write_classfile):
        print(f"[prepare] Dataset root: {root}")
        print(f"[prepare] Detected classes ({len(classes)}): " + ", ".join(c.name for c in classes))
        print("[prepare] Use --list-classes, --verify, or --write-classfile to perform actions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
