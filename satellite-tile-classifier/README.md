**Data Science Projects**

# Satellite Tile Classifier — Offline Land-Use Classification

## Project Structure
satellite-tile-classifier/
├── DESIGN_NOTE.md # Part 1: Design note
├── PROBLEM_SOLVING.md # Part 3: Problem-solving answers
├── README.md # This file (setup + run instructions)
├── requirements.txt # Python dependencies
├── train.py # Step 1: Train the model
├── evaluate.py # Step 3: Evaluate against ground truth
├── classify_batch.py # Batch-classify a directory of tiles
├── app/
│ ├── init.py
│ ├── config.py # Paths, class names, thresholds
│ ├── classifier.py # Model loading + inference
│ ├── database.py # SQLite result store
│ └── main.py # FastAPI service (Part 2)
├── data/ 
│ ├── candidate_tiles/ # Training data (class-named folders)
│ ├── eval_set/ # Unlabelled tiles for evaluation
│ └── eval_labels.csv # Ground truth for eval set
├── model/
│ └── classifier.pt # Created by training
└── results.db # Created at runtime


## Step-by-Step: Setup, Train, Run, Evaluate

### Step 0: Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch (CPU), torchvision, FastAPI, and uvicorn. All offline-capable, no cloud APIs.

### Step 1: Place your data

Create the `data/` directory structure:
data/
├── candidate_tiles/
│ ├── AnnualCrop/
│ │ └── AnnualCrop_1.png
│ ├── Forest/
│ │ └── Forest_1.png
│ ├── Highway/
│ ├── Industrial/
│ ├── Residential/
│ ├── River/
│ └── SeaLake/
├── eval_set/
│ ├── tile_001.png
│ ├── tile_002.png
│ └── ...
└── eval_labels.csv


### Step 2: Train the model

```bash
python train.py
```

**What this does:**

1. Loads a pre-trained ResNet-18 (ImageNet weights downloaded once, then cached locally).
2. Replaces the final classification layer for our 7 land-use classes.
3. Freezes early layers (conv1, layer1, layer2) — only fine-tunes layer3, layer4, and fc.
4. Trains for 30 epochs with data augmentation (flips, rotations, color jitter).
5. Saves the trained model to `model/classifier.pt`.

**Expected output:**
Training samples: 210
Classes: ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'River', 'SeaLake']
Parameters: 5,765,703 trainable / 11,182,919 total
Epoch 1/30 loss=1.8432 acc=0.2381
Epoch 2/30 loss=1.2100 acc=0.5190
...
Epoch 30/30 loss=0.1200 acc=0.9810
Model saved to model/classifier.pt


Training takes ~2-5 minutes on CPU.

**IMPORTANT NOTE about first run:** The very first time you run this, PyTorch will download the ResNet-18 ImageNet weights (~45 MB). This requires internet. After that first download, the weights are cached in `~/.cache/torch/` and everything runs fully offline.

### Step 3: Evaluate accuracy

```bash
python evaluate.py
```

This classifies all tiles in `eval_set/` and compares against `eval_labels.csv`. Prints overall accuracy, per-class accuracy, and a confusion matrix.

### Step 4: Start the API service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Step 5: Use the API

**Classify a tile (upload):**

```bash
curl -X POST http://localhost:8000/classify -F "tile=@data/eval_set/tile_001.png"
```

**Classify a tile (local path):**

```bash
curl -X POST "http://localhost:8000/classify/path?file_path=data/eval_set/tile_001.png"
```

**Query results:**

```bash
# All results
curl http://localhost:8000/results

# Filter by class
curl "http://localhost:8000/results?label=Forest"

# Low-confidence tiles needing review
curl "http://localhost:8000/results?needs_review=true"

# High-confidence only
curl "http://localhost:8000/results?min_confidence=0.8"

# Summary counts
curl http://localhost:8000/summary
```

**Batch classify (CLI, no server needed):**

```bash
python classify_batch.py data/eval_set
```

## What's Stubbed / Skipped

- **Authentication/multi-user:** Not needed for a single-machine offline deployment.
- **Tile geo-metadata parsing:** Would extract lat/lon from filenames if they encoded it.
- **Web dashboard:** A React/HTML frontend for visual tile review — useful but not core.
- **Model retraining pipeline:** Would allow accumulating analyst corrections as new training data.
- **CSV export endpoint:** Straightforward addition (`/export/csv`).
- **Multi-band support:** EuroSAT RGB assumed; real Sentinel-2 has 13 bands.