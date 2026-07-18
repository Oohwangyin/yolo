#!/usr/bin/env python3
"""Create the VEDAI first-fold split from existing YOLO-format labels."""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_NAMES = ["car", "pick-up", "camping car", "truck", "vehicle", "tractor", "boat", "van"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a YOLO-format VEDAI_NEW dataset from existing YOLO labels and "
            "fold01/fold01test split files. RGB images and labels are linked instead "
            "of copied, and names like 00000000_co.png are linked as 00000000.png."
        )
    )
    parser.add_argument("--src", type=Path, default=Path("/root/autodl-tmp/datasets/VEDAI"), help="VEDAI root.")
    parser.add_argument(
        "--dst", type=Path, default=Path("/root/autodl-tmp/datasets/VEDAI_NEW"), help="Output YOLO dataset root."
    )
    parser.add_argument("--yaml-out", type=Path, default=Path("VEDAI_NEW.yaml"), help="Output YOLO data YAML path.")
    parser.add_argument("--fold", type=str, default="01", help="Fold id, e.g. 01 for fold01.txt/fold01test.txt.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the destination directory first if it already exists.",
    )
    return parser.parse_args()


def first_existing(paths: list[Path], description: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("Missing " + description + ":\n" + "\n".join(str(p) for p in paths))


def check_paths(src: Path, fold: str) -> tuple[Path, Path, Path, Path]:
    image_dir = first_existing([src / "images", src / "Vehicules1024"], "image directory")
    label_dir = first_existing([src / "labels", src / "Annotations1024"], "label directory")
    train_file = first_existing(
        [src / f"fold{fold}.txt", label_dir / f"fold{fold}.txt", src / "Annotations1024" / f"fold{fold}.txt"],
        f"fold{fold}.txt",
    )
    test_file = first_existing(
        [
            src / f"fold{fold}test.txt",
            label_dir / f"fold{fold}test.txt",
            src / "Annotations1024" / f"fold{fold}test.txt",
        ],
        f"fold{fold}test.txt",
    )
    return image_dir, label_dir, train_file, test_file


def prepare_dst(dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite and any(dst.iterdir()):
            raise FileExistsError(
                f"Destination exists and is not empty: {dst}\n"
                "Use --overwrite if you want to recreate it."
            )
        if overwrite:
            shutil.rmtree(dst)
    for split in ("train", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)


def symlink_file(src: Path, dst: Path) -> None:
    """Create a relative file symlink so fold directories do not duplicate dataset files."""
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Destination already exists: {dst}")
    relative_src = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(relative_src)


def normalize_id(value: str) -> str:
    stem = Path(value.strip()).stem
    if stem.lower().endswith("_co"):
        stem = stem[:-3]
    return stem


def read_split(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(normalize_id(line))
    if not ids:
        raise RuntimeError(f"Split file is empty: {path}")
    return ids


def build_image_index(image_dir: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    image_priority: dict[str, int] = {}

    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lower_stem = path.stem.lower()
        if lower_stem.endswith("_ir"):
            continue

        key = normalize_id(path.stem)
        priority = 2 if lower_stem.endswith("_co") else 1
        old = image_index.get(key)
        if old is None or priority > image_priority[key]:
            image_index[key] = path
            image_priority[key] = priority
        elif priority == image_priority[key] and old.resolve() != path.resolve():
            raise ValueError(f"Duplicate RGB image id {key}:\n{old}\n{path}")

    if not image_index:
        raise RuntimeError(f"No RGB images found in {image_dir}")
    return image_index


def build_label_index(label_dir: Path) -> dict[str, Path]:
    label_index: dict[str, Path] = {}
    for path in label_dir.rglob("*.txt"):
        if path.name.lower().startswith("fold"):
            continue
        key = normalize_id(path.stem)
        old = label_index.get(key)
        if old is not None and old.resolve() != path.resolve():
            raise ValueError(f"Duplicate label id {key}:\n{old}\n{path}")
        label_index[key] = path
    if not label_index:
        raise RuntimeError(f"No YOLO label txt files found in {label_dir}")
    return label_index


def summarize_yolo_label(label_path: Path) -> tuple[Counter[int], int]:
    class_counts: Counter[int] = Counter()
    invalid_lines = 0
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            invalid_lines += 1
            print(f"  invalid label line ignored: {label_path}:{line_no}")
            continue
        try:
            cls = int(float(parts[0]))
            [float(x) for x in parts[1:5]]
        except ValueError:
            invalid_lines += 1
            print(f"  invalid label line ignored: {label_path}:{line_no}")
            continue
        class_counts[cls] += 1
    return class_counts, invalid_lines


def convert_split(
    split_name: str,
    image_ids: list[str],
    image_index: dict[str, Path],
    label_index: dict[str, Path],
    dst: Path,
) -> dict:
    class_counts: Counter[int] = Counter()
    missing_images: list[str] = []
    missing_labels = 0
    empty_labels = 0
    invalid_label_lines = 0
    linked_images = 0

    for stem in image_ids:
        src_image = image_index.get(stem)
        if src_image is None:
            missing_images.append(stem)
            continue

        dst_image = dst / "images" / split_name / f"{stem}{src_image.suffix.lower()}"
        symlink_file(src_image, dst_image)
        linked_images += 1

        label_path = dst / "labels" / split_name / f"{stem}.txt"
        src_label = label_index.get(stem)
        if src_label is None:
            missing_labels += 1
            label_path.write_text("", encoding="utf-8")
        else:
            symlink_file(src_label, label_path)

        counts, invalid = summarize_yolo_label(label_path)
        class_counts.update(counts)
        invalid_label_lines += invalid
        if sum(counts.values()) == 0:
            empty_labels += 1

    return {
        "requested_images": len(image_ids),
        "images": linked_images,
        "instances": sum(class_counts.values()),
        "class_counts": class_counts,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "invalid_label_lines": invalid_label_lines,
    }


def write_yaml(yaml_out: Path, dst: Path) -> None:
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {dst.resolve().as_posix()}",
        "train: images/train",
        "val: images/test",
        "test: images/test",
        "",
        "names:",
    ]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(DEFAULT_NAMES))
    yaml_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(split: str, stats: dict) -> None:
    print(f"\n{split}: {stats['images']}/{stats['requested_images']} images, {stats['instances']} instances")
    if stats["missing_images"]:
        print(f"  missing RGB images: {len(stats['missing_images'])}")
        for stem in stats["missing_images"][:20]:
            print(f"    {stem}")
        if len(stats["missing_images"]) > 20:
            print(f"    ... {len(stats['missing_images']) - 20} more")
    print(f"  missing labels: {stats['missing_labels']}")
    print(f"  empty labels: {stats['empty_labels']}")
    print(f"  invalid label lines: {stats['invalid_label_lines']}")
    for cls, name in enumerate(DEFAULT_NAMES):
        print(f"  {cls}: {name}: {stats['class_counts'].get(cls, 0)}")
    extra_classes = sorted(cls for cls in stats["class_counts"] if cls >= len(DEFAULT_NAMES) or cls < 0)
    if extra_classes:
        print("  extra class ids:")
        for cls in extra_classes:
            print(f"    {cls}: {stats['class_counts'][cls]}")


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    yaml_out = args.yaml_out.resolve()
    image_dir, label_dir, train_file, test_file = check_paths(src, args.fold)
    image_index = build_image_index(image_dir)
    label_index = build_label_index(label_dir)

    train_ids = read_split(train_file)
    test_ids = read_split(test_file)
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"Train/test split overlap detected, first examples: {overlap[:10]}")

    prepare_dst(dst, args.overwrite)
    train_stats = convert_split("train", train_ids, image_index, label_index, dst)
    test_stats = convert_split("test", test_ids, image_index, label_index, dst)
    write_yaml(yaml_out, dst)

    print(f"Source: {src}")
    print(f"Images: {image_dir}")
    print(f"Labels: {label_dir}")
    print(f"Destination: {dst}")
    print(f"YAML: {yaml_out}")
    print(f"Fold: {args.fold}")
    print_summary("train", train_stats)
    print_summary("test", test_stats)


if __name__ == "__main__":
    main()
