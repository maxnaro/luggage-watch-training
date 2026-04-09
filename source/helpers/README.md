# Dataset Helpers

Scripts for downloading, preparing, and loading the training dataset.

## Prerequisites

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

This includes `fiftyone`, which manages dataset downloads and metadata via a local MongoDB instance.

## Downloading the dataset

`download_dataset.py` pulls images from COCO 2017 and Open Images V7, filters and remaps them into two target classes (`person`, `luggage`), and exports in YOLOv5 format.

### Basic usage

```bash
python source/helpers/download_dataset.py --out ./dataset --max-samples 30000
```

This downloads ~30,000 luggage images split between OID and COCO (2:1 by default), calculates the required number of person-only images from the class ratio, and exports everything to `./dataset/`.

### Arguments

| Argument            | Default         | Description                                                                                 |
| ------------------- | --------------- | ------------------------------------------------------------------------------------------- |
| `--out`             | `training/data` | Output directory for the YOLOv5 dataset                                                     |
| `--max-samples`     | None (all)      | Cap on total luggage images per split, divided between sources by `--source-ratio`          |
| `--source-ratio`    | `2.0`           | OID:COCO download ratio. `2.0` means 2x as many Open Images samples as COCO                 |
| `--class-ratio`     | `2.0`           | Target person:luggage instance ratio. Person-only images are downloaded to fill this budget |
| `--min-area`        | `0.001`         | Minimum normalised bounding box area. Smaller annotations are removed as label noise        |
| `--seed`            | `42`            | Random seed for reproducible subsampling                                                    |
| `--luggage-weights` | None            | Per-subclass oversampling, e.g. `backpack=3,handbag=2,suitcase=1`                           |
| `--supplement`      | None            | Path to a local YOLOv5-format dataset to merge in (e.g. overhead CCTV data)                 |

### Examples

```bash
# Default ratio, cap at 30k luggage images (20k OID + 10k COCO)
python source/helpers/download_dataset.py --out ./dataset --max-samples 30000

# Heavier OID weighting, oversample handbags
python source/helpers/download_dataset.py --out ./dataset \
  --max-samples 30000 \
  --source-ratio 3.0 \
  --luggage-weights "backpack=2,handbag=3,suitcase=1"

# Low person:luggage ratio for luggage-focused training
python source/helpers/download_dataset.py --out ./dataset \
  --max-samples 30000 \
  --class-ratio 0.5

# Merge a local overhead dataset
python source/helpers/download_dataset.py --out ./dataset \
  --max-samples 30000 \
  --supplement ./data
```

### Pipeline

The download script runs the following steps per split:

1. Download luggage images from COCO (`--max-samples` split by `--source-ratio`)
2. Download luggage images from Open Images V7
3. Filter small detections below `--min-area`
4. Oversample luggage sub-classes (if `--luggage-weights` provided)
5. Remap source labels to target classes (`person`, `luggage`)
6. Calculate person instance budget from `--class-ratio`
7. Download only the required number of person-only images
8. Balance to exact class ratio
9. Merge local supplement data (if `--supplement` provided)
10. Export as YOLOv5 format

### Source classes

**COCO 2017:** `backpack`, `handbag`, `suitcase` -> `luggage`; `person` -> `person`

**Open Images V7:**
- Person classes: `Person`, `Man`, `Woman`, `Boy`, `Girl`, `Human body` -> `person`
- Luggage classes: `Backpack`, `Suitcase`, `Handbag`, `Briefcase`, `Luggage and bags`, `Bag` -> `luggage`

## Creating the Docker volume

The training container expects the dataset mounted at `/app/data`. Use the provided PowerShell script to create a Docker volume from the downloaded dataset:

```powershell
.\source\helpers\create_docker_volume.ps1 -Volume training-data -Path .\dataset
```

This creates a named Docker volume and copies the dataset into it. The volume can then be mounted when running the training container:

```bash
docker run --gpus all \
  -v training-data:/app/data \
  -v training-runs:/app/runs \
  -v training-model:/app/model \
  -v ./source/config.json:/app/config.json \
  luggage-watch-training train
```
