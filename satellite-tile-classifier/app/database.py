"""
SQLite result store.

Each classification becomes a row with the predicted label, confidence,
full probability vector, review flag, model version, and timestamp.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import DB_PATH


def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent read perf
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    conn = _get_conn(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            tile_id         TEXT PRIMARY KEY,
            filename        TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence      REAL NOT NULL,
            probabilities   TEXT NOT NULL,
            needs_review    INTEGER NOT NULL DEFAULT 0,
            model_version   TEXT NOT NULL,
            classified_at   TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_predicted_label
        ON classifications (predicted_label)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_needs_review
        ON classifications (needs_review)
    """)
    conn.commit()
    conn.close()


def insert_result(
    tile_id: str,
    filename: str,
    predicted_label: str,
    confidence: float,
    probabilities: dict[str, float],
    needs_review: bool,
    model_version: str,
) -> None:
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO classifications
        (tile_id, filename, predicted_label, confidence, probabilities, needs_review, model_version, classified_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            tile_id,
            filename,
            predicted_label,
            round(confidence, 4),
            json.dumps(probabilities),
            int(needs_review),
            model_version,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def query_results(
    label: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    needs_review: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    conn = _get_conn()

    clauses = []
    params: list = []

    if label:
        clauses.append("predicted_label = ?")
        params.append(label)
    if min_confidence is not None:
        clauses.append("confidence >= ?")
        params.append(min_confidence)
    if max_confidence is not None:
        clauses.append("confidence <= ?")
        params.append(max_confidence)
    if needs_review is not None:
        clauses.append("needs_review = ?")
        params.append(int(needs_review))

    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"""
        SELECT * FROM classifications
        {where}
        ORDER BY classified_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    results = []
    for row in rows:
        d = dict(row)
        d["probabilities"] = json.loads(d["probabilities"])
        d["needs_review"] = bool(d["needs_review"])
        results.append(d)
    return results


def count_by_label() -> dict[str, int]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT predicted_label, COUNT(*) as cnt FROM classifications GROUP BY predicted_label"
    ).fetchall()
    conn.close()
    return {row["predicted_label"]: row["cnt"] for row in rows}


def get_result(tile_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM classifications WHERE tile_id = ?", (tile_id,)
    ).fetchone()
    conn.close()

    if row is None:
        return None

    d = dict(row)
    d["probabilities"] = json.loads(d["probabilities"])
    d["needs_review"] = bool(d["needs_review"])
    return d