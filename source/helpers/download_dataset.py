"""
Download a person + luggage dataset from COCO using FiftyOne.

Luggage is an aggregate class combining the COCO classes:
  - backpack
  - handbag
  - suitcase

Exports train and val splits in YOLOv5 format (compatible with YOLO11 and YOLO26).
Generates a data.yaml consumed by Ultralytics training.

Usage:
    python download_dataset.py                         # defaults
    python download_dataset.py --out ../data           # custom output dir
    python download_dataset.py --max-samples 5000      # cap per split
"""

import argparse
import random
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# COCO classes that map to "luggage"
LUGGAGE_CLASSES = ["backpack", "handbag", "suitcase"]
SOURCE_CLASSES = ["person"] + LUGGAGE_CLASSES

# Target label mapping  (class_id → name)
TARGET_CLASSES = ["person", "luggage"]
REMAP = {c: "luggage" for c in LUGGAGE_CLASSES}
REMAP["person"] = "person"


def download_split(split: str, max_samples: int | None = None) -> fo.Dataset:
    """Download a COCO split filtered to the classes we need."""
    kwargs = dict(
        dataset_name=f"coco-2017-{split}-luggage-watch",
        label_types=["detections"],
        classes=SOURCE_CLASSES,
        only_matching=True,  # keep only images with >=1 target class
        split=split,
    )
    if max_samples is not None:
        kwargs["max_samples"] = max_samples

    # Load (downloads on first run, cached afterwards)
    dataset = foz.load_zoo_dataset("coco-2017", **kwargs)
    return dataset


def oversample_classes(
    dataset: fo.Dataset,
    weights: dict[str, int],
    seed: int = 42,
) -> fo.Dataset:
    """Duplicate samples containing specific classes to increase their
    representation in the dataset.

    Parameters
    ----------
    dataset : fo.Dataset
        Dataset with original COCO labels (before remapping).
    weights : dict[str, int]
        Mapping of class name -> integer weight.  A weight of 3 means images
        containing that class will appear 3* (i.e. 2 extra copies).  Classes
        with weight <= 1 are left untouched.
    seed : int
        Random seed (unused currently, reserved for future stochastic sampling).
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
            print(f"    ⚠ No images contain '{cls}' — skipping oversample")
            continue

        extra_copies = weight - 1
        print(
            f"    Oversampling '{cls}': {n_originals} images * {weight} "
            f"(+{n_originals * extra_copies} copies)"
        )
        for _ in range(extra_copies):
            for sample in view:
                dup = sample.copy()
                boosted.add_sample(dup)

    print(f"    Dataset size after oversampling: {len(boosted)} images")
    return boosted


def remap_labels(dataset: fo.Dataset) -> fo.Dataset:
    """Merge luggage sub-classes into a single 'luggage' class and drop
    any other classes that may have leaked through."""
    # Clone so we don't mutate the zoo cache
    mapped = dataset.clone()

    for sample in mapped.iter_samples(autosave=True):
        new_dets = []
        for det in sample.ground_truth.detections:
            if det.label in REMAP:
                det.label = REMAP[det.label]
                new_dets.append(det)
            # else: drop detection (shouldn't happen with only_matching)
        sample.ground_truth.detections = new_dets

    return mapped


def balance_classes(dataset: fo.Dataset, ratio: float, seed: int = 42) -> fo.Dataset:
    """Down-sample person-only images so person:luggage instance ratio ≈ `ratio`.

    Strategy
    --------
    1. Split the dataset into images that contain ≥1 luggage detection
       ('luggage images') and those with only person detections
       ('person-only images').
    2. Count luggage instances across all luggage images.
    3. Calculate how many person instances are allowed:
           allowed_person = luggage_instances * ratio
    4. Subtract person instances already present in luggage images.
    5. Greedily keep person-only images (shuffled) until the remaining
       person-instance budget is exhausted.
    6. Return the balanced dataset.

    Parameters
    ----------
    dataset : fo.Dataset
        Remapped dataset (labels are 'person' / 'luggage').
    ratio : float
        Target person-to-luggage *instance* ratio (e.g. 2.0 → 2 persons
        per 1 luggage instance).
    seed : int
        Random seed for reproducible subsampling.
    """
    # identify luggage vs person-only images
    luggage_view = dataset.match(
        F("ground_truth.detections").filter(F("label") == "luggage").length() > 0
    )
    person_only_view = dataset.match(
        F("ground_truth.detections").filter(F("label") == "luggage").length() == 0
    )

    # count instances
    luggage_instance_count = sum(
        sum(1 for d in s.ground_truth.detections if d.label == "luggage")
        for s in luggage_view
    )
    person_in_luggage_imgs = sum(
        sum(1 for d in s.ground_truth.detections if d.label == "person")
        for s in luggage_view
    )

    allowed_person_total = int(luggage_instance_count * ratio)
    person_budget = max(0, allowed_person_total - person_in_luggage_imgs)

    print(f"    Luggage instances            : {luggage_instance_count}")
    print(f"    Person instances (luggage imgs): {person_in_luggage_imgs}")
    print(f"    Person budget (person-only)   : {person_budget}")

    # subsample person-only images
    person_only_ids = [s.id for s in person_only_view]
    random.seed(seed)
    random.shuffle(person_only_ids)

    keep_ids = []
    accumulated = 0
    for sid in person_only_ids:
        if accumulated >= person_budget:
            break
        sample = dataset[sid]
        n_person = sum(1 for d in sample.ground_truth.detections if d.label == "person")
        keep_ids.append(sid)
        accumulated += n_person

    # merge luggage images + kept person-only images
    all_keep_ids = [s.id for s in luggage_view] + keep_ids
    balanced = dataset.select(all_keep_ids)

    # Stats
    total_person = sum(
        sum(1 for d in s.ground_truth.detections if d.label == "person")
        for s in balanced
    )
    total_luggage = sum(
        sum(1 for d in s.ground_truth.detections if d.label == "luggage")
        for s in balanced
    )
    actual_ratio = total_person / total_luggage if total_luggage else float("inf")

    print(f"    Balanced dataset: {len(balanced)} images")
    print(f"    Person instances : {total_person}")
    print(f"    Luggage instances: {total_luggage}")
    print(f"    Actual ratio     : {actual_ratio:.2f}:1")

    return balanced


def export_yolo(dataset: fo.Dataset, export_dir: Path, split: str) -> None:
    """Export a FiftyOne dataset to YOLOv5 format under export_dir/<split>."""
    dataset.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        classes=TARGET_CLASSES,
        label_field="ground_truth",
    )


def cleanup_fiftyone_datasets(*names: str) -> None:
    """Delete temporary FiftyOne datasets from the local DB."""
    for name in names:
        if fo.dataset_exists(name):
            fo.delete_dataset(name)


def main():
    ap = argparse.ArgumentParser(
        description="Download person + luggage COCO subset for YOLO training"
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
        help="Cap the number of images per split (useful for quick tests)",
    )
    ap.add_argument(
        "--ratio",
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
        "--luggage-weights",
        type=str,
        default=None,
        help=(
            "Per-subclass oversampling weights as comma-separated key=value "
            "pairs, e.g. 'backpack=1,handbag=3,suitcase=1'. Classes with "
            "weight > 1 will have their images duplicated that many times."
        ),
    )
    args = ap.parse_args()

    # Parse luggage weights
    luggage_weights: dict[str, int] = {c: 1 for c in LUGGAGE_CLASSES}
    if args.luggage_weights:
        for item in args.luggage_weights.split(","):
            key, val = item.strip().split("=")
            key = key.strip()
            if key not in LUGGAGE_CLASSES:
                ap.error(f"Unknown luggage sub-class '{key}'. "
                         f"Valid: {LUGGAGE_CLASSES}")
            luggage_weights[key] = int(val)

    export_dir: Path = args.out.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    cloned_names: list[str] = []

    for split in ("train", "validation"):
        yolo_split = "val" if split == "validation" else split
        print(f"\n{'='*60}")
        print(f"  Downloading COCO 2017 {split} split …")
        print(f"{'='*60}")
        raw = download_split(split, max_samples=args.max_samples)

        if any(w > 1 for w in luggage_weights.values()):
            print(f"  Oversampling luggage sub-classes {luggage_weights} …")
            raw = oversample_classes(raw, luggage_weights, seed=args.seed)
            cloned_names.append(raw.name)

        print(f"  Remapping labels → {TARGET_CLASSES} …")
        mapped = remap_labels(raw)
        cloned_names.append(mapped.name)

        print(f"  Balancing classes (target ratio {args.ratio}:1) …")
        balanced = balance_classes(mapped, ratio=args.ratio, seed=args.seed)

        print(f"  Exporting {len(balanced)} samples as YOLOv5/{yolo_split} …")
        export_yolo(balanced, export_dir, yolo_split)

        # Free the cloned dataset from FiftyOne DB
        fo.delete_dataset(mapped.name)

    # Clean up source zoo datasets from FiftyOne DB
    cleanup_fiftyone_datasets(
        "coco-2017-train-luggage-watch",
        "coco-2017-validation-luggage-watch",
    )

    print(f"\n{'='*60}")
    print(f"  Done!  Dataset written to: {export_dir}")
    print(f"{'='*60}")
    print(f"\n  Classes: {TARGET_CLASSES}")


if __name__ == "__main__":
    main()
