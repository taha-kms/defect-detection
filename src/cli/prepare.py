import argparse
from pathlib import Path
from typing import List

from src.utils import env

ALL_CLASSES = {
    "bottle","cable","capsule","carpet","grid",
    "hazelnut","leather","metal_nut","pill","screw",
    "tile","toothbrush","transistor","wood","zipper",
}


def _find_available_classes(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _check_class_layout(root: Path, cls: str) -> list[str]:
    problems = []
    croot = root / cls
    expect = [
        croot / "train" / "good",
        croot / "test" / "good",
    ]
    for p in expect:
        if not p.exists():
            problems.append(f"missing: {p}")
    # quick spot check for any defect folder having masks
    test_dir = croot / "test"
    if test_dir.exists():
        for defect_dir in [d for d in test_dir.iterdir() if d.is_dir() and d.name != "good" and d.name != "ground_truth"]:
            mask_dir = croot / "ground_truth" / defect_dir.name
            if not mask_dir.exists():
                problems.append(f"missing masks dir for '{defect_dir.name}': {mask_dir}")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Validate env and dataset for MVTec AD")
    parser.add_argument("--dataset", choices=["mvtec"], required=True)
    parser.add_argument("--classes", nargs="*", default=[])
    parser.add_argument("--list-classes", action="store_true", help="List classes found under DATA_DIR/mvtec_ad")
    args = parser.parse_args()

    data_root = env.DATA_DIR
    print(f"DATA_DIR: {env.DATA_DIR}")
    print(f"Looking for MVTec AD at: {data_root}")

    if args.list_classes:
        found = _find_available_classes(data_root)
        if not found:
            print("No classes found. Ensure dataset is extracted to data/<class>/ ...")
            return
        print("Classes available:", ", ".join(found))
        return

    classes = args.classes or ["bottle", "cable", "screw", "leather", "tile", "grid"]
    for cls in classes:
        if cls not in ALL_CLASSES:
            print(f"Unknown class '{cls}'. Skipping (valid: {sorted(ALL_CLASSES)})")
            continue
        issues = _check_class_layout(data_root, cls)
        if issues:
            print(f"{cls}:")
            for i in issues:
                print(f"   - {i}")
        else:
            print(f"{cls}: structure looks OK")

    print("DEVICE:", env.DEVICE)
    print("IMAGE_SIZE:", env.IMAGE_SIZE, "| BACKBONE:", env.BACKBONE)
    print("RUNS_DIR:", env.RUNS_DIR)


if __name__ == "__main__":
    main()