# Part 2 — Working Slice

## What I Built

The core path: POST a tile image → classify it → store the result → return the prediction. That's implemented end-to-end and runs.

The API is FastAPI, the model is a fine-tuned ResNet-18 running on CPU via PyTorch, and results go into a SQLite database. There's also a batch script that classifies a whole directory of tiles from the command line without needing the server.

## How It Works

When a tile hits `/classify`, the server saves the raw image to disk (so we have it for audit or reprocessing), resizes it to 64x64, normalizes it, and runs a forward pass through the model. The softmax output gives us a probability for each of the 7 classes. We take the top one, check if the confidence is above our threshold, write everything to the database, and return the result as JSON.

The query side has endpoints to list results with filters (class, confidence range, review flag), look up a specific tile, and get a count summary by class.

## Training

I took a ResNet-18 pretrained on ImageNet, swapped the final layer for a 7-class head, froze the early layers, and fine-tuned `layer3` + `layer4` + `fc` on the candidate tiles. Data augmentation (random flips, rotation, color jitter) was important given only ~30 images per class. Trained for 30 epochs with Adam, took about 3 minutes on CPU.

## Results

Evaluated on the 210-tile eval set against `eval_labels.csv`.

**Overall: 195/210 = 92.9%**

| Class        | Correct/Total | Accuracy |
|--------------|---------------|----------|
| AnnualCrop   | 29/30         | 96.7%    |
| Forest       | 27/30         | 90.0%    |
| Highway      | 29/30         | 96.7%    |
| Industrial   | 27/30         | 90.0%    |
| Residential  | 30/30         | 100.0%   |
| River        | 24/30         | 80.0%    |
| SeaLake      | 29/30         | 96.7%    |

River is the weakest at 80%, almost certainly confused with SeaLake — both are water, and in a 64×64 tile you often can't tell whether you're looking at a river or the edge of a lake without seeing the broader shape. Residential at 100% makes sense — buildings have very distinctive texture in satellite imagery.

## How to Run It

```bash
pip install -r requirements.txt
# torch, torchvision, fastapi, uvicorn, pillow

# fine-tune the model
python train.py          # ~3 min

# check accuracy against eval_labels.csv
python evaluate.py

# start the API
uvicorn app.main:app --port 8000

# classify a tile
curl -X POST http://localhost:8000/classify \
  -F "tile=@data/eval_set/tile_001.png"

# query stored results
curl "http://localhost:8000/results?label=Forest&min_confidence=0.8"
curl "http://localhost:8000/results?needs_review=true"
curl http://localhost:8000/summary

# or batch classify without the server
python classify_batch.py data/eval_set
```

## What I Skipped

Auth, a web dashboard, CSV export, geo-metadata parsing, a model retraining pipeline, and multi-band Sentinel-2 support. All would be natural next steps but aren't the core classification path.