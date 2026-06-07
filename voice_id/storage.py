"""
SQLite storage for speaker profiles.

Schema supports multi-profile out of the box even though Phase 1 ships with
a single-profile UX — avoids a future migration. Canonical embedding per
profile is computed as the mean of all stored sample embeddings; we store
the raw 256-dim float32 vectors as BLOBs (1024 bytes each) rather than
pre-averaging so users can "add another sample" without recomputing.

Pattern mirrors memory.py:21-28 (data/ directory, WAL journal, lazy
CREATE TABLE IF NOT EXISTS on first call).
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import TypedDict

import numpy as np

from .embedding import compute_embedding

log = logging.getLogger("jarvis.voice_id.storage")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "voice_profiles.db"

# Cache of the canonical (mean) embedding, keyed by profile name (None = "the
# only profile"). Computing it means reading every sample BLOB and averaging —
# safe to cache because it changes ONLY on (re-)enrollment, and every writer
# below calls invalidate_canonical(). This is the *right* thing to cache: the
# enrolled identity is immutable between enrollments. (The incoming utterance,
# by contrast, must be re-embedded every time — see verify.verify_speaker.)
_canonical_cache: dict[str | None, "tuple[int, np.ndarray] | None"] = {}


def invalidate_canonical() -> None:
    """Drop the cached canonical embedding(s). Called after any write to the
    profiles/samples tables so the next verify recomputes from fresh data."""
    _canonical_cache.clear()


class StatusDict(TypedDict):
    enrolled: bool
    name: str | None
    sample_count: int


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = _get_db()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY,
                profile_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_samples_profile ON samples(profile_id);
        """)
        conn.commit()
    finally:
        conn.close()


def enroll_sample(audio_bytes: bytes, name: str) -> int:
    """Compute embedding and store as a sample under `name`. Creates the
    profile row if it doesn't exist. Returns the new total sample_count.

    Raises voice_id.wav.AudioTooShortError if the clip is too short.
    """
    init_db()
    embedding = compute_embedding(audio_bytes)
    now = time.time()
    conn = _get_db()
    try:
        cur = conn.execute("SELECT id FROM profiles WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            profile_id = row["id"]
        else:
            cur = conn.execute(
                "INSERT INTO profiles (name, created_at) VALUES (?, ?)",
                (name, now),
            )
            profile_id = cur.lastrowid
        conn.execute(
            "INSERT INTO samples (profile_id, embedding, created_at) VALUES (?, ?, ?)",
            (profile_id, embedding.tobytes(), now),
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) AS n FROM samples WHERE profile_id = ?", (profile_id,)).fetchone()["n"]
        log.info(f"Enrolled sample for {name!r} (total: {count})")
        invalidate_canonical()  # samples table changed — drop the cached mean
        return count
    finally:
        conn.close()


def get_canonical_embedding(name: str | None = None) -> tuple[int, np.ndarray] | None:
    """Return (profile_id, mean_embedding) for the named profile, or the only
    profile if `name` is None. Returns None if no profile exists.

    Read-through cache: the canonical embedding only changes on (re-)enrollment,
    so we compute it once and reuse it across every runtime verification. This
    is the hot-path win that replaces the old per-connection trust cache —
    cheap to recompute when it matters, free when it doesn't.
    """
    if name in _canonical_cache:
        return _canonical_cache[name]
    result = _compute_canonical_embedding(name)
    _canonical_cache[name] = result
    return result


def _compute_canonical_embedding(name: str | None = None) -> tuple[int, np.ndarray] | None:
    """Read all sample BLOBs for the profile and return (profile_id,
    unit-normalized mean embedding). Returns None if no profile/samples exist.
    Uncached — callers should go through get_canonical_embedding().
    """
    init_db()
    conn = _get_db()
    try:
        if name is None:
            row = conn.execute("SELECT id, name FROM profiles LIMIT 1").fetchone()
        else:
            row = conn.execute("SELECT id, name FROM profiles WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        profile_id = row["id"]
        blobs = conn.execute("SELECT embedding FROM samples WHERE profile_id = ?", (profile_id,)).fetchall()
        if not blobs:
            return None
        arrays = [np.frombuffer(b["embedding"], dtype=np.float32) for b in blobs]
        mean = np.mean(np.stack(arrays), axis=0)
        # Re-normalize (resemblyzer embeddings are unit-norm; the mean is not).
        # Shape is invariant (EMBEDDING_DIM,) by construction of the embeddings table.
        mean = mean / np.linalg.norm(mean)
        return profile_id, mean.astype(np.float32, copy=False)
    finally:
        conn.close()


def get_status() -> StatusDict:
    """Fast status check for GET /api/voice/status and settings UI."""
    init_db()
    conn = _get_db()
    try:
        row = conn.execute(
            """SELECT p.name AS name, COUNT(s.id) AS sample_count
               FROM profiles p LEFT JOIN samples s ON s.profile_id = p.id
               GROUP BY p.id LIMIT 1"""
        ).fetchone()
        if not row:
            return {"enrolled": False, "name": None, "sample_count": 0}
        return {
            "enrolled": row["sample_count"] > 0,
            "name": row["name"],
            "sample_count": row["sample_count"],
        }
    finally:
        conn.close()


def is_enrolled() -> bool:
    """Cheap boolean gate for the voice_handler hot path.

    Kept separate from get_status so the voice handler doesn't pay for a
    GROUP BY on every user utterance.
    """
    init_db()
    conn = _get_db()
    try:
        row = conn.execute("SELECT 1 FROM samples LIMIT 1").fetchone()
        return row is not None
    finally:
        conn.close()


def clear_profile(name: str | None = None) -> None:
    """Wipe the named profile (or the only one if name is None).

    Samples are cascade-deleted via the foreign key.
    """
    init_db()
    conn = _get_db()
    try:
        if name is None:
            conn.execute("DELETE FROM samples")
            conn.execute("DELETE FROM profiles")
        else:
            conn.execute("DELETE FROM profiles WHERE name = ?", (name,))
        conn.commit()
        invalidate_canonical()  # profile gone — drop the cached mean
        log.info(f"Cleared profile: {name or '(all)'}")
    finally:
        conn.close()
