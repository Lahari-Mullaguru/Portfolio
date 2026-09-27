"""
Evaluation script: classify eval set tiles and compare against eval_labels.csv.

Usage:
    python evaluate.py

Expects:
    data/eval_set/tile_001.png, tile_002.png, ...
    data/eval_labels.csv (columns: filename, true_label)
    model/classifier.pt

Prints:
    Per-class accuracy, confusion matrix, overall accuracy.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

from app.config import EVAL_SET_DIR, EVAL_LABELS_PATH, CLASS_NAMES
from app.classifier import classify_tile, load_model


def main():
    if not EVAL_LABELS_PATH.exists():
        print(f"ERROR: eval_labels.csv not found at {EVAL_LABELS_PATH}")
        sys.exit(1)

    if not EVAL_SET_DIR.exists():
        print(f"ERROR: eval set directory not found at {EVAL_SET_DIR}")
        sys.exit(1)

    # Load the model once
    load_model()

    # Read ground truth
    labels = {}
    with open(EVAL_LABELS_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["filename"].strip()] = row["true_label"].strip()

    print(f"Eval set: {len(labels)} tiles")
    print(f"Classes: {sorted(set(labels.values()))}")
    print()

    # Classify each tile
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    errors = []

    for filename, true_label in sorted(labels.items()):
        tile_path = EVAL_SET_DIR / filename

        if not tile_path.exists():
            print(f"  SKIP {filename} (file not found)")
            continue

        result = classify_tile(tile_path)
        pred = result["predicted_label"]
        conf = result["confidence"]

        per_class_total[true_label] += 1
        confusion[true_label][pred] += 1
        total += 1

        if pred == true_label:
            correct += 1
            per_class_correct[true_label] += 1
        else:
            errors.append((filename, true_label, pred, conf))

        # Progress indicator
        if total % 200 == 0:
            print(f"  Classified {total}/{len(labels)} tiles...")

    print(f"\n{'=' * 60}")
    print(f"OVERALL ACCURACY: {correct}/{total} = {correct / total:.1%}")
    print(f"{'=' * 60}\n")

    # Per-class accuracy
    print("Per-class accuracy:")
    print(f"  {'Class':<15}{'Correct':>8}{'Total':>8}{'Accuracy':>10}")
    print(f"  {'-' * 43}")
    for cls in CLASS_NAMES:
        c = per_class_correct[cls]
        t = per_class_total[cls]
        acc = c / t if t else 0
        print(f"  {cls:<15}{c:>8}{t:>8}{acc:>10.1%}")

    # Confusion matrix
    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = f"  {'':15}" + "".join(f"{c[:7]:>8}" for c in CLASS_NAMES)
    print(header)
    for true_cls in CLASS_NAMES:
        row = f"  {true_cls:15}"
        for pred_cls in CLASS_NAMES:
            row += f"{confusion[true_cls][pred_cls]:>8}"
        print(row)

    # Worst errors
    if errors:
        print(f"\nMisclassifications ({len(errors)} total):")
        for fn, true, pred, conf in errors[:20]:
            print(f"  {fn}: true={true}, pred={pred}, conf={conf:.3f}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()