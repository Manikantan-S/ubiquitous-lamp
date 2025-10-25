"""Preprocess raw datasets into a unified spoilage-quality format."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable

from PIL import Image


@dataclass
class ImageRecord:
    image_id: str
    source: str
    filepath: str
    label: str
    spoilage_score: float
    split: str


QUALITY_MAPPING = {
    "fresh": "fresh",
    "good": "fresh",
    "ripe": "fresh",
    "early_blight": "early_spoilage",
    "early_spoilage": "early_spoilage",
    "mid_spoilage": "early_spoilage",
    "late_blight": "spoiled",
    "rot": "spoiled",
    "rotten": "spoiled",
    "spoiled": "spoiled",
}


def derive_spoilage_score(label: str) -> float:
    mapping = {"fresh": 0.1, "early_spoilage": 0.5, "spoiled": 0.9}
    return mapping[label]


def discover_images(raw_root: Path) -> Iterable[ImageRecord]:
    supported_extensions = {".jpg", ".jpeg", ".png"}
    for image_path in raw_root.rglob("*"):
        if image_path.suffix.lower() not in supported_extensions:
            continue
        parts = image_path.relative_to(raw_root).parts
        if len(parts) < 2:
            continue
        source = parts[0]
        raw_label = parts[1].lower()
        label = QUALITY_MAPPING.get(raw_label)
        if label is None:
            continue
        image_id = image_path.stem
        split = "train"
        yield ImageRecord(
            image_id=image_id,
            source=source,
            filepath=str(image_path),
            label=label,
            spoilage_score=derive_spoilage_score(label),
            split=split,
        )


def normalise_images(records: Iterable[ImageRecord], output_root: Path) -> Dict[str, ImageRecord]:
    output_images = output_root / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, ImageRecord] = {}
    for record in records:
        src = Path(record.filepath)
        dest = output_images / f"{record.image_id}{src.suffix.lower()}"
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            rgb.save(dest, quality=95)
        manifest[record.image_id] = ImageRecord(
            image_id=record.image_id,
            source=record.source,
            filepath=str(dest.relative_to(output_root)),
            label=record.label,
            spoilage_score=record.spoilage_score,
            split=record.split,
        )
    return manifest


def assign_splits(manifest: Dict[str, ImageRecord], seed: int = 42) -> None:
    rng = random.Random(seed)
    records = list(manifest.values())
    rng.shuffle(records)
    total = len(records)
    train_cutoff = int(total * 0.7)
    val_cutoff = int(total * 0.85)
    for idx, record in enumerate(records):
        if idx < train_cutoff:
            split = "train"
        elif idx < val_cutoff:
            split = "val"
        else:
            split = "test"
        manifest[record.image_id].split = split


def materialise_split_dirs(manifest: Dict[str, ImageRecord], output_root: Path) -> None:
    for split in {"train", "val", "test"}:
        split_dir = output_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for label in {record.label for record in manifest.values()}:
            (split_dir / label).mkdir(parents=True, exist_ok=True)

    for record in manifest.values():
        src = output_root / record.filepath
        dst = output_root / record.split / record.label / src.name
        shutil.copy(src, dst)


def write_manifest(manifest: Dict[str, ImageRecord], output_root: Path) -> None:
    data = [asdict(record) for record in manifest.values()]
    with (output_root / "annotations.json").open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess produce quality datasets")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parsed = parser.parse_args(args=args)

    records = list(discover_images(parsed.raw_root))
    if not records:
        raise RuntimeError(
            f"No supported images found in {parsed.raw_root}. Check dataset placement."
        )

    parsed.output_root.mkdir(parents=True, exist_ok=True)
    manifest = normalise_images(records, parsed.output_root)
    assign_splits(manifest, seed=parsed.seed)
    materialise_split_dirs(manifest, parsed.output_root)
    write_manifest(manifest, parsed.output_root)
    print(json.dumps({"status": "ok", "samples": len(manifest)}))


if __name__ == "__main__":
    main()
