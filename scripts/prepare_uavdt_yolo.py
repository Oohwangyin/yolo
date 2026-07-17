"""Convert the UAVDT detection benchmark to an Ultralytics YOLO dataset.

Expected source layout::

    UAVDT/
    |-- UAV-benchmark-M/
    |-- UAV-benchmark-MOTD_v1.0/GT/
    `-- M_attr/{train,test}/

The official train/test split is read from ``M_attr``. A validation split is
created from complete training sequences, never from randomly selected frames.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


CLASS_MAP = {1: 0, 2: 1, 3: 2}  # UAVDT: car, truck, bus -> YOLO zero-based IDs
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Root containing the three extracted UAVDT folders.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory. Defaults to <root>/yolo.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of official train sequences used for validation.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for the sequence-level validation split.")
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How images are placed in the YOLO tree. Hard links avoid duplicating image data.",
    )
    return parser.parse_args()


def sequence_names(attr_dir: Path) -> list[str]:
    """Return sequence names from files such as M0101_attr.txt."""
    names = []
    for path in sorted(attr_dir.glob("*_attr.txt")):
        names.append(path.stem.removesuffix("_attr"))
    if not names:
        raise FileNotFoundError(f"No *_attr.txt files found in {attr_dir}")
    return names


def split_sequences(train_sequences: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    """Split complete official training sequences into train and validation sets."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if len(train_sequences) < 2:
        raise ValueError("At least two official training sequences are required to create a validation split.")

    val_count = min(max(round(len(train_sequences) * val_ratio), 1), len(train_sequences) - 1)
    val_set = set(random.Random(seed).sample(train_sequences, val_count))
    train = sorted(sequence for sequence in train_sequences if sequence not in val_set)
    val = sorted(val_set)
    return train, val


def frame_id_from_image(path: Path) -> int:
    """Extract the numeric frame ID from names such as img000001.jpg."""
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot obtain a frame ID from image name: {path.name}")
    return int(match.group(1))


def read_detection_labels(gt_path: Path, image_width: int, image_height: int) -> dict[int, list[str]]:
    """Convert one UAVDT *_gt_whole.txt file to normalized YOLO labels grouped by frame."""
    labels: dict[int, list[str]] = defaultdict(list)
    with gt_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            fields = [field.strip() for field in line.strip().split(",")]
            if not fields or fields == [""]:
                continue
            if len(fields) < 9:
                raise ValueError(f"{gt_path}:{line_number} has {len(fields)} fields; expected at least 9.")

            frame_id = int(float(fields[0]))
            left, top, width, height = map(float, fields[2:6])
            category = int(float(fields[8]))
            if category not in CLASS_MAP:
                continue

            x1 = max(0.0, min(left, float(image_width)))
            y1 = max(0.0, min(top, float(image_height)))
            x2 = max(0.0, min(left + width, float(image_width)))
            y2 = max(0.0, min(top + height, float(image_height)))
            if x2 <= x1 or y2 <= y1:
                continue

            x_center = ((x1 + x2) / 2.0) / image_width
            y_center = ((y1 + y2) / 2.0) / image_height
            box_width = (x2 - x1) / image_width
            box_height = (y2 - y1) / image_height
            labels[frame_id].append(
                f"{CLASS_MAP[category]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )
    return labels


def place_image(source: Path, destination: Path, mode: str) -> None:
    """Place an image without silently overwriting an existing file."""
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def convert_sequence(
    sequence: str,
    split: str,
    image_root: Path,
    gt_root: Path,
    output: Path,
    image_mode: str,
) -> tuple[int, int]:
    """Convert one complete video sequence and return image/box counts."""
    source_dir = image_root / sequence
    gt_path = gt_root / f"{sequence}_gt_whole.txt"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing image sequence directory: {source_dir}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing detection annotation: {gt_path}")

    images = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found in {source_dir}")

    with Image.open(images[0]) as image:
        image_width, image_height = image.size
    labels_by_frame = read_detection_labels(gt_path, image_width, image_height)

    image_destination = output / "images" / split / sequence
    label_destination = output / "labels" / split / sequence
    image_destination.mkdir(parents=True, exist_ok=True)
    label_destination.mkdir(parents=True, exist_ok=True)

    box_count = 0
    for image_path in images:
        place_image(image_path, image_destination / image_path.name, image_mode)
        frame_id = frame_id_from_image(image_path)
        frame_labels = labels_by_frame.get(frame_id, [])
        box_count += len(frame_labels)
        (label_destination / f"{image_path.stem}.txt").write_text(
            "\n".join(frame_labels) + ("\n" if frame_labels else ""), encoding="utf-8"
        )
    return len(images), box_count


def write_sequence_manifest(output: Path, split: str, sequences: list[str]) -> None:
    """Record the exact sequence split for reproducibility."""
    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / f"{split}_sequences.txt").write_text("\n".join(sequences) + "\n", encoding="utf-8")


def write_dataset_yaml(output: Path) -> Path:
    """Write an Ultralytics dataset YAML with an absolute dataset root."""
    yaml_path = output / "UAVDT.yaml"
    root = output.resolve().as_posix()
    yaml_path.write_text(
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: car\n"
        "  1: truck\n"
        "  2: bus\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output or root / "yolo").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Use a new directory or remove it explicitly.")

    image_root = root / "UAV-benchmark-M"
    gt_root = root / "UAV-benchmark-MOTD_v1.0" / "GT"
    attr_root = root / "M_attr"
    official_train = sequence_names(attr_root / "train")
    official_test = sequence_names(attr_root / "test")
    train, val = split_sequences(official_train, args.val_ratio, args.seed)
    splits = {"train": train, "val": val, "test": sorted(official_test)}

    overlap = (set(train) & set(val)) | (set(train) & set(official_test)) | (set(val) & set(official_test))
    if overlap:
        raise ValueError(f"Sequence leakage detected across splits: {sorted(overlap)}")

    output.mkdir(parents=True, exist_ok=True)
    for split, sequences in splits.items():
        write_sequence_manifest(output, split, sequences)
        image_count = 0
        box_count = 0
        for sequence in sequences:
            sequence_images, sequence_boxes = convert_sequence(
                sequence, split, image_root, gt_root, output, args.image_mode
            )
            image_count += sequence_images
            box_count += sequence_boxes
        print(f"{split}: {len(sequences)} sequences, {image_count} images, {box_count} boxes")

    yaml_path = write_dataset_yaml(output)
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
