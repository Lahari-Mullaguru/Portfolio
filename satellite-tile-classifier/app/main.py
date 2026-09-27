"""
FastAPI service for offline satellite tile classification.

Core endpoint: POST /classify (takes a tile image, classifies it, stores result)
Query endpoints for analyst use.
"""

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query

from app.config import CONFIDENCE_THRESHOLD, MODEL_VERSION, TILE_STORE_DIR
from app.classifier import classify_tile
from app.database import init_db, insert_result, query_results, count_by_label, get_result

app = FastAPI(
    title="Satellite Tile Classifier",
    description="Offline land-use classification for satellite image tiles",
    version="2.1.0",
)


@app.on_event("startup")
def startup():
    init_db()
    TILE_STORE_DIR.mkdir(parents=True, exist_ok=True)


# Core endpoint
@app.post("/classify")
async def classify(tile: UploadFile = File(...)):
    """Accept a tile image, classify it, store the result, return prediction."""
    tile_id = str(uuid.uuid4())
    ext = Path(tile.filename).suffix or ".png"
    saved_path = TILE_STORE_DIR / f"{tile_id}{ext}"

    with open(saved_path, "wb") as f:
        shutil.copyfileobj(tile.file, f)

    try:
        result = classify_tile(saved_path)
    except Exception as e:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

    needs_review = result["confidence"] < CONFIDENCE_THRESHOLD

    insert_result(
        tile_id=tile_id,
        filename=tile.filename,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        needs_review=needs_review,
        model_version=MODEL_VERSION,
    )

    return {
        "tile_id": tile_id,
        "filename": tile.filename,
        "predicted_label": result["predicted_label"],
        "confidence": round(result["confidence"], 4),
        "needs_review": needs_review,
        "probabilities": result["probabilities"],
        "model_version": MODEL_VERSION,
    }


# Classify from local path (useful for batch / CLI)
@app.post("/classify/path")
async def classify_by_path(file_path: str):
    """Classify a tile already on disk (by absolute or relative path)."""
    p = Path(file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    tile_id = str(uuid.uuid4())
    dest = TILE_STORE_DIR / f"{tile_id}{p.suffix}"
    shutil.copy2(p, dest)

    try:
        result = classify_tile(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

    needs_review = result["confidence"] < CONFIDENCE_THRESHOLD

    insert_result(
        tile_id=tile_id,
        filename=p.name,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        needs_review=needs_review,
        model_version=MODEL_VERSION,
    )

    return {
        "tile_id": tile_id,
        "filename": p.name,
        "predicted_label": result["predicted_label"],
        "confidence": round(result["confidence"], 4),
        "needs_review": needs_review,
        "probabilities": result["probabilities"],
        "model_version": MODEL_VERSION,
    }


# Query endpoints
@app.get("/results")
async def list_results(
    label: Optional[str] = Query(None, description="Filter by predicted label"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    max_confidence: Optional[float] = Query(None, ge=0, le=1),
    needs_review: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Query stored classification results with optional filters."""
    return query_results(
        label=label,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        needs_review=needs_review,
        limit=limit,
        offset=offset,
    )


@app.get("/results/{tile_id}")
async def get_tile_result(tile_id: str):
    """Get classification result for a specific tile."""
    result = get_result(tile_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Tile not found")
    return result


@app.get("/summary")
async def summary():
    """Count of classifications per land-use class."""
    return count_by_label()


@app.get("/health")
async def health():
    """Basic health check confirms the service is alive."""
    return {"status": "ok", "model_version": MODEL_VERSION}