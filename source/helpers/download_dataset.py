"""
Download a person + luggage dataset for YOLO training.

Pulls from one or more FiftyOne zoo sources (COCO 2017, Open Images V7),
applies filtering, remapping, balancing, and optional supplement merging,
then exports in YOLOv5 format.

Usage:
    python download_dataset.py                              # defaults (OID + COCO, 2:1)
    python download_dataset.py --max-samples 30000          # cap total luggage images
    python download_dataset.py --source-ratio 3.0           # heavier OID weighting
    python download_dataset.py --supplement ./overhead       # + local YOLO dataset
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

TARGET_CLASSES = ["person", "luggage"]


@dataclass
class ZooSource:
    """Configuration for a FiftyOne zoo dataset source."""

    zoo_name: str
    dataset_prefix: str
    classes: list[str]
    remap: dict[str, str]
    luggage_classes: list[str] = field(default_factory=list)

    @property
    def person_classes(self) -> list[str]:
        return [c for c in self.classes if c not in self.luggage_classes]

    def download(
        self,
        split: str,
        max_samples: int | None = None,
        classes: list[str] | None = None,
        name_suffix: str = "",
    ) -> fo.Dataset:
        download_classes = classes if classes is not None else self.classes
        name = f"{self.dataset_prefix}-{split}-luggage-watch{name_suffix}"

        if fo.dataset_exists(name):
            fo.delete_dataset(name)

        kwargs = dict(
            dataset_name=name,
            label_types=["detections"],
            classes=download_classes,
            only_matching=True,
            split=split,
        )
        if max_samples is not None:
            kwargs["max_samples"] = max_samples
        return foz.load_zoo_dataset(self.zoo_name, **kwargs)

    def remap_labels(self, dataset: fo.Dataset) -> fo.Dataset:
        mapped = dataset.clone()
        for sample in mapped.iter_samples(autosave=True):
            sample.ground_truth.detections = [
                det
                for det in sample.ground_truth.detections
                if det.label in self.remap
                and not (setattr(det, "label", self.remap[det.label]))  # remap in-place
            ]
        return mapped

    @property
    def zoo_dataset_names(self) -> list[str]:
        names = []
        for split in ("train", "validation"):
            for suffix in ("", "-luggage", "-person"):
                names.append(f"{self.dataset_prefix}-{split}-luggage-watch{suffix}")
        return names


COCO = ZooSource(
    zoo_name="coco-2017",
    dataset_prefix="coco-2017",
    classes=["person", "backpack", "handbag", "suitcase"],
    remap={
        "person": "person",
        "backpack": "luggage",
        "handbag": "luggage",
        "suitcase": "luggage",
    },
    luggage_classes=["backpack", "handbag", "suitcase"],
)

_OI_PERSON_CLASSES = ["Person", "Man", "Woman", "Boy", "Girl", "Human body"]
_OI_LUGGAGE_CLASSES = [
    "Backpack",
    "Suitcase",
    "Handbag",
    "Briefcase",
    "Luggage and bags"
]

OPEN_IMAGES = ZooSource(
    zoo_name="open-images-v7",
    dataset_prefix="open-images-v7",
    classes=_OI_PERSON_CLASSES + _OI_LUGGAGE_CLASSES,
    remap={
        **{c: "person" for c in _OI_PERSON_CLASSES},
        **{c: "luggage" for c in _OI_LUGGAGE_CLASSES},
    },
    luggage_classes=_OI_LUGGAGE_CLASSES,
)


# Dataset transforms


def filter_small_detections(
    dataset: fo.Dataset,
    min_area: float,
    classes: list[str] | None = None,
) -> fo.Dataset:
    """Remove detections below a minimum bounding box area.

    Area is relative (width * height, both normalised to [0, 1]).
    0.001 ~= 32x32 px in a 1024x1024 image.
    """
    filtered = dataset.clone()
    removed_count = 0

    for sample in filtered.iter_samples(autosave=True):
        kept = []
        for det in sample.ground_truth.detections:
            _, _, w, h = det.bounding_box
            if w * h < min_area and (classes is None or det.label in classes):
                removed_count += 1
                continue
            kept.append(det)
        sample.ground_truth.detections = kept

    non_empty = filtered.match(F("ground_truth.detections").length() > 0)
    dropped_images = len(filtered) - len(non_empty)

    print(f"    Removed {removed_count} small detections (area < {min_area})")
    print(f"    Dropped {dropped_images} now-empty images")

    return non_empty


def drop_missing_media_samples(dataset: fo.Dataset, context: str = "") -> fo.Dataset:
    """Drop samples whose file paths no longer exist on disk."""
    missing_ids = []
    for sample in dataset.iter_samples(progress=False):
        if not Path(sample.filepath).is_file():
            missing_ids.append(sample.id)

    if not missing_ids:
        return dataset

    cleaned = dataset.exclude(missing_ids)
    label = f" for {context}" if context else ""
    print(
        f"    Removed {len(missing_ids)} samples with missing image files{label}"
    )
    return cleaned


def oversample_classes(
    dataset: fo.Dataset,
    weights: dict[str, int],
) -> fo.Dataset:
    """Duplicate images containing specific classes.

    A weight of 3 means 3x total (2 extra copies). Weight <= 1 is a no-op.
    """
    classes_to_boost = {c: w for c, w in weights.items() if w > 1}
    if not classes_to_boost:
        return dataset

    boosted = dataset.clone()
    for cls, weight in classes_to_boost.items():
        view = boosted.match(
            F("ground_truth.detections").filter(F("label") == cls).length() > 0
        )
        n_originals = len(view)
        if n_originals == 0:
            print(f"    No images contain '{cls}' — skipping")
            continue

        extra = weight - 1
        print(
            f"    Oversampling '{cls}': {n_originals} x{weight} (+{n_originals * extra})"
        )
        for _ in range(extra):
            for sample in view:
                boosted.add_sample(sample.copy())

    print(f"    Dataset size after oversampling: {len(boosted)} images")
    return boosted


def count_label(view, label: str) -> int:
    """Count total instances of a label across all samples in a view."""
    return sum(
        sum(1 for d in s.ground_truth.detections if d.label == label) for s in view
    )


def balance_classes(dataset: fo.Dataset, ratio: float, seed: int = 42) -> fo.Dataset:
    """Down-sample person-only images so person:luggage instance ratio ~= ratio."""
    luggage_view = dataset.match(
        F("ground_truth.detections").filter(F("label") == "luggage").length() > 0
    )
    person_only_view = dataset.match(
        F("ground_truth.detections").filter(F("label") == "luggage").length() == 0
    )

    luggage_count = count_label(luggage_view, "luggage")
    person_in_luggage = count_label(luggage_view, "person")
    person_budget = max(0, int(luggage_count * ratio) - person_in_luggage)

    print(f"    Luggage instances             : {luggage_count}")
    print(f"    Person instances (luggage imgs): {person_in_luggage}")
    print(f"    Person budget (person-only)   : {person_budget}")

    person_only_ids = [s.id for s in person_only_view]
    random.seed(seed)
    random.shuffle(person_only_ids)

    keep_ids, accumulated = [], 0
    for sid in person_only_ids:
        if accumulated >= person_budget:
            break
        n = sum(1 for d in dataset[sid].ground_truth.detections if d.label == "person")
        keep_ids.append(sid)
        accumulated += n

    balanced = dataset.select([s.id for s in luggage_view] + keep_ids)

    total_person = count_label(balanced, "person")
    total_luggage = count_label(balanced, "luggage")
    actual_ratio = total_person / total_luggage if total_luggage else float("inf")

    print(f"    Balanced: {len(balanced)} images, ratio {actual_ratio:.2f}:1")

    return balanced


def merge_supplement(
    dataset: fo.Dataset, supplement_dir: Path, split: str
) -> fo.Dataset:
    """Merge a local YOLOv5-format dataset into the main dataset."""
    yaml_path = supplement_dir / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Expected dataset.yaml at {yaml_path}. "
            f"Ensure the supplement directory is in YOLOv5 format."
        )

    supp_name = f"supplement-{split}-{dataset.name}"
    supp = fo.Dataset.from_dir(
        dataset_dir=str(supplement_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        name=supp_name,
        label_field="ground_truth",
    )

    invalid = set(supp.distinct("ground_truth.detections.label")) - set(TARGET_CLASSES)
    if invalid:
        raise ValueError(
            f"Supplement contains unknown classes: {invalid}. Expected: {TARGET_CLASSES}"
        )

    before = len(dataset)
    dataset.merge_samples(supp)
    print(f"    Merged {len(dataset) - before} supplement images")

    fo.delete_dataset(supp_name)
    return dataset


def materialise_duplicate_media(
    dataset: fo.Dataset,
    export_dir: Path,
    split: str,
) -> tuple[fo.Dataset, Path | None]:
    """Give duplicated samples unique media paths so export keeps all copies."""
    filepath_counts = Counter(dataset.values("filepath"))
    duplicates_to_copy = sum(count - 1 for count in filepath_counts.values() if count > 1)
    if duplicates_to_copy == 0:
        return dataset, None

    materialised = dataset.clone()
    materialised_dir = export_dir / ".materialised_media" / split
    if materialised_dir.exists():
        shutil.rmtree(materialised_dir, ignore_errors=True)
    materialised_dir.mkdir(parents=True, exist_ok=True)

    seen = Counter()
    copied = 0
    for sample in materialised.iter_samples(autosave=True, progress=False):
        source_path = sample.filepath
        if filepath_counts[source_path] <= 1:
            continue

        seen[source_path] += 1
        if seen[source_path] == 1:
            continue

        src = Path(source_path)
        dst = materialised_dir / f"{sample.id}{src.suffix}"
        shutil.copy2(src, dst)
        sample.filepath = str(dst)
        copied += 1

    print(f"    materialised {copied} duplicated samples for export")
    return materialised, materialised_dir


def export_yolo(dataset: fo.Dataset, export_dir: Path, split: str) -> None:
    """Export to YOLOv5 format."""
    export_dataset, materialised_dir = materialise_duplicate_media(
        dataset, export_dir, split
    )
    temp_dataset_name = export_dataset.name if export_dataset is not dataset else None

    try:
        export_dataset.export(
            export_dir=str(export_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            split=split,
            classes=TARGET_CLASSES,
            label_field="ground_truth",
        )
    finally:
        if temp_dataset_name and fo.dataset_exists(temp_dataset_name):
            fo.delete_dataset(temp_dataset_name)
        if materialised_dir and materialised_dir.exists():
            shutil.rmtree(materialised_dir, ignore_errors=True)


def write_dataset_yaml(export_dir: Path) -> None:
    """Overwrite the FiftyOne-generated YAML with the correct one for training."""
    yaml_path = export_dir / "dataset.yaml"
    yaml_path.write_text(
        "names:\n"
        "  0: person\n"
        "  1: luggage\n"
        "path: /app/data\n"
        "train: ./images/train/\n"
        "val: ./images/val/\n"
    )


# Cleanup


class DatasetTracker:
    """Track temporary FiftyOne datasets for cleanup."""

    def __init__(self):
        self._names: list[str] = []

    def track(self, dataset: fo.Dataset) -> fo.Dataset:
        self._names.append(dataset.name)
        return dataset

    def cleanup(self):
        for name in self._names:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)
        self._names.clear()


# CLI & pipeline


def parse_luggage_weights(raw: str | None) -> dict[str, int]:
    weights = {c: 1 for c in COCO.luggage_classes}
    if not raw:
        return weights
    for item in raw.split(","):
        key, val = item.strip().split("=")
        key = key.strip()
        if key not in COCO.luggage_classes:
            raise argparse.ArgumentTypeError(
                f"Unknown luggage sub-class '{key}'. Valid: {COCO.luggage_classes}"
            )
        weights[key] = int(val)
    return weights


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Download person + luggage dataset for YOLO training"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root output directory (default: training/data)",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap images per split (useful for quick tests)",
    )
    ap.add_argument(
        "--class-ratio",
        type=float,
        default=2.0,
        help="Target person:luggage instance ratio (default: 2.0)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subsampling (default: 42)",
    )
    ap.add_argument(
        "--min-area",
        type=float,
        default=0.001,
        help="Min normalised bbox area for luggage detections (default: 0.001)",
    )
    ap.add_argument(
        "--source-ratio",
        type=float,
        default=2.0,
        help=(
            "OID:COCO download ratio. Controls how --max-samples is split "
            "between Open Images and COCO. 2.0 means 2x as many OID images "
            "as COCO (default: 2.0)"
        ),
    )
    ap.add_argument(
        "--supplement",
        type=Path,
        default=None,
        help="Path to a supplementary YOLOv5-format dataset to merge in",
    )
    ap.add_argument(
        "--luggage-weights",
        type=str,
        default=None,
        help="Per-subclass oversampling, e.g. 'backpack=1,handbag=3,suitcase=1'",
    )
    return ap.parse_args()


def download_and_filter(
    source: ZooSource,
    split: str,
    max_samples: int | None,
    min_area: float,
    tracker: DatasetTracker,
    classes: list[str] | None = None,
    name_suffix: str = "",
) -> fo.Dataset:
    """Download a zoo source and filter small detections (keeps original labels)."""
    label = f"{source.zoo_name} {split}"
    if name_suffix:
        label += f" ({name_suffix.strip('-')})"
    print(f"  Downloading {label} …")
    raw = source.download(split, max_samples=max_samples, classes=classes, name_suffix=name_suffix)

    if min_area > 0:
        print(f"  Filtering small detections (min_area={min_area}) …")
        raw = tracker.track(
            filter_small_detections(raw, min_area, source.luggage_classes)
        )

    raw = tracker.track(
        drop_missing_media_samples(raw, context=f"{source.zoo_name}/{split}")
    )

    return raw


def process_split(
    split: str,
    args: argparse.Namespace,
    luggage_weights: dict[str, int],
    tracker: DatasetTracker,
    export_dir: Path,
) -> None:
    """Run the full pipeline for a single split."""
    yolo_split = "val" if split == "validation" else split

    print(f"\n{'=' * 60}")
    print(f"  Processing {split} split")
    print(f"{'=' * 60}")

    # 1. Split max_samples between OID and COCO by source ratio
    sr = args.source_ratio
    coco_max = int(args.max_samples / (sr + 1)) if args.max_samples else None
    oi_max = int(args.max_samples * sr / (sr + 1)) if args.max_samples else None

    if args.max_samples:
        print(f"  Source split: COCO={coco_max}, OID={oi_max} (ratio {sr}:1)")

    # 2. Download luggage images from COCO
    raw_luggage = download_and_filter(
        COCO, split, coco_max, args.min_area, tracker,
        classes=COCO.luggage_classes, name_suffix="-luggage",
    )

    # 3. Oversample luggage sub-classes (before remap)
    if any(w > 1 for w in luggage_weights.values()):
        print(f"  Oversampling luggage sub-classes …")
        raw_luggage = tracker.track(oversample_classes(raw_luggage, luggage_weights))

    # 4. Remap to target classes
    print(f"  Remapping COCO labels → {TARGET_CLASSES} …")
    dataset = tracker.track(COCO.remap_labels(raw_luggage))

    # 5. Download and merge Open Images luggage
    oi_raw = download_and_filter(
        OPEN_IMAGES, split, oi_max, args.min_area, tracker,
        classes=OPEN_IMAGES.luggage_classes, name_suffix="-luggage",
    )
    print(f"  Remapping Open Images labels → {TARGET_CLASSES} …")
    oi = tracker.track(OPEN_IMAGES.remap_labels(oi_raw))
    dataset.merge_samples(oi, key_field="id")
    print(f"    Merged {len(oi)} Open Images samples")
    fo.delete_dataset(oi.name)

    # 6. Calculate person budget from luggage images
    luggage_count = count_label(dataset, "luggage")
    person_in_luggage = count_label(dataset, "person")
    person_budget = max(0, int(luggage_count * args.class_ratio) - person_in_luggage)

    print(f"    Luggage instances             : {luggage_count}")
    print(f"    Person instances (luggage imgs): {person_in_luggage}")
    print(f"    Person budget (person-only)   : {person_budget}")

    # 7. Download only as many person-only images as the ratio requires
    if person_budget > 0:
        # Estimate ~2.5 person instances per image, add headroom
        person_img_cap = int(person_budget / 2) + 100
        raw_person = download_and_filter(
            COCO, split, person_img_cap, args.min_area, tracker,
            classes=COCO.person_classes, name_suffix="-person",
        )
        person_mapped = tracker.track(COCO.remap_labels(raw_person))
        dataset.merge_samples(person_mapped, key_field="id")
        print(f"    Merged {len(person_mapped)} person-only images")

    # 8. Balance to exact ratio (trims any excess person images)
    print(f"  Balancing classes (target ratio {args.class_ratio}:1) …")
    balanced = balance_classes(dataset, ratio=args.class_ratio, seed=args.seed)

    # 9. Merge local supplement (after balancing — preserves all supplement data)
    if args.supplement:
        print(f"  Merging supplement from {args.supplement} …")
        balanced = merge_supplement(balanced, args.supplement, yolo_split)

    balanced = drop_missing_media_samples(balanced, context=f"final {split}")

    # 10. Export
    print(f"  Exporting {len(balanced)} samples as YOLOv5/{yolo_split} …")
    export_yolo(balanced, export_dir, yolo_split)


def main():
    args = parse_args()
    luggage_weights = parse_luggage_weights(args.luggage_weights)

    export_dir = args.out.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    tracker = DatasetTracker()

    try:
        for split in ("train", "validation"):
            process_split(split, args, luggage_weights, tracker, export_dir)

        write_dataset_yaml(export_dir)
    finally:
        # Clean up all zoo + intermediate datasets, even after errors.
        tracker.cleanup()
        for source in (COCO, OPEN_IMAGES):
            for name in source.zoo_dataset_names:
                if fo.dataset_exists(name):
                    fo.delete_dataset(name)

    print(f"\n{'=' * 60}")
    print(f"  Done!  Dataset written to: {export_dir}")
    print(f"{'=' * 60}")
    print(f"\n  Classes: {TARGET_CLASSES}")


if __name__ == "__main__":
    main()
