#!/usr/bin/env python3
"""Convert VisDrone DET annotations to YOLO HBB labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image


SPLITS = ("VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# VisDrone category ids are 1-based. Category 0 is ignored region.
# YOLO class ids become 0-based: 1 pedestrian -> 0, ..., 10 motor -> 9.
VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO labels.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/VisDrone"),
        help="VisDrone dataset root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        help="Split directories to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing labels/*.txt files.",
    )
    parser.add_argument(
        "--keep-ignored",
        action="store_true",
        help="Keep category 0 ignored regions as class 0. Not recommended for standard training.",
    )
    return parser.parse_args()


def image_size(image_dir: Path, stem: str) -> tuple[int, int] | None:
    for suffix in IMAGE_SUFFIXES:
        image_path = image_dir / f"{stem}{suffix}"
        if image_path.exists():
            with Image.open(image_path) as im:
                return im.size
    return None


def convert_file(
    ann_path: Path,
    label_path: Path,
    width: int,
    height: int,
    keep_ignored: bool,
) -> tuple[Counter[int], int, int]:
    labels = []
    class_counts: Counter[int] = Counter()
    ignored = 0
    bad = 0

    for line_no, line in enumerate(ann_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 8:
            bad += 1
            continue
        try:
            x, y, w, h = (float(parts[i]) for i in range(4))
            category = int(float(parts[5]))
        except ValueError:
            bad += 1
            continue

        if w <= 0 or h <= 0:
            bad += 1
            continue
        if category == 0 and not keep_ignored:
            ignored += 1
            continue
        if category < 0 or category > 10:
            bad += 1
            continue

        cls = category if keep_ignored else category - 1
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(width), x + w)
        y2 = min(float(height), y + h)
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            bad += 1
            continue

        xc = (x1 + x2) / 2.0 / width
        yc = (y1 + y2) / 2.0 / height
        bw = box_w / width
        bh = box_h / height
        labels.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        class_counts[cls] += 1

    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return class_counts, ignored, bad


def convert_split(root: Path, split: str, overwrite: bool, keep_ignored: bool) -> dict:
    split_dir = root / split
    image_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    label_dir = split_dir / "labels"
    if not image_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError(f"Missing images or annotations directory in {split_dir}")
    label_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[int] = Counter()
    converted = 0
    skipped_existing = 0
    missing_images = []
    ignored_total = 0
    bad_total = 0

    for ann_path in sorted(ann_dir.glob("*.txt")):
        label_path = label_dir / ann_path.name
        if label_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        size = image_size(image_dir, ann_path.stem)
        if size is None:
            missing_images.append(ann_path.stem)
            continue
        width, height = size
        counts, ignored, bad = convert_file(ann_path, label_path, width, height, keep_ignored)
        class_counts.update(counts)
        ignored_total += ignored
        bad_total += bad
        converted += 1

    return {
        "split": split,
        "converted": converted,
        "skipped_existing": skipped_existing,
        "missing_images": missing_images,
        "ignored": ignored_total,
        "bad": bad_total,
        "class_counts": class_counts,
        "label_dir": str(label_dir),
    }


def print_summary(stats: dict) -> None:
    print(f"\n[{stats['split']}]")
    print(f"converted files: {stats['converted']}")
    print(f"skipped existing: {stats['skipped_existing']}")
    print(f"ignored regions skipped: {stats['ignored']}")
    print(f"bad boxes/lines skipped: {stats['bad']}")
    print(f"labels saved to: {stats['label_dir']}")
    if stats["missing_images"]:
        print(f"missing images: {len(stats['missing_images'])}")
        for stem in stats["missing_images"][:20]:
            print(f"  {stem}")
        if len(stats["missing_images"]) > 20:
            print(f"  ... {len(stats['missing_images']) - 20} more")
    print("per-class instances:")
    for cls, name in enumerate(VISDRONE_NAMES):
        print(f"  {cls}: {name}: {stats['class_counts'].get(cls, 0)}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    for split in args.splits:
        stats = convert_split(root, split, args.overwrite, args.keep_ignored)
        print_summary(stats)


if __name__ == "__main__":
    main()
