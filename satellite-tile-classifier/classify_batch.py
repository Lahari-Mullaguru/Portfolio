"""
Batch classification: classify all tiles in a directory and store results.
Useful as a CLI alternative to the REST API.

Usage:
    python classify_batch.py <tile_directory>
    python classify_batch.py data/eval_set
"""

import sys
import uuid
from pathlib import Path

from app.config import CONFIDENCE_THRESHOLD, MODEL_VERSION
from app.classifier import classify_tile, load_model
from app.database import init_db, insert_result


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_batch.py <tile_directory>")
        sys.exit(1)

    tile_dir = Path(sys.argv[1])

    init_db()
    load_model()

    tiles = sorted(tile_dir.glob("*.png"))
    print(f"Found {len(tiles)} tiles in {tile_dir}")

    for i, tile_path in enumerate(tiles, 1):
        result = classify_tile(tile_path)
        tile_id = str(uuid.uuid4())
        needs_review = result["confidence"] < CONFIDENCE_THRESHOLD

        insert_result(
            tile_id=tile_id,
            filename=tile_path.name,
            predicted_label=result["predicted_label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            needs_review=needs_review,
            model_version=MODEL_VERSION,
        )

        flag = " [REVIEW]" if needs_review else ""
        print(
            f"  [{i:3d}/{len(tiles)}] {tile_path.name} -> "
            f"{result['predicted_label']} ({result['confidence']:.3f}){flag}"
        )

    print(f"\nDone. {len(tiles)} results stored in results.db")


if __name__ == "__main__":
    main()