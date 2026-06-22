"""
Speaker verification — compare an incoming utterance against the enrolled
canonical embedding.

EVERY utterance is verified. There is deliberately NO "this connection is
trusted for N seconds" cache: that window let anyone in mic range speak
commands right after the owner, with no re-check and no log line. The only
thing cached is the canonical (enrolled) embedding, which lives in storage.py
and is safe to cache because it changes only on re-enrollment. The incoming
voice is re-embedded every time — ~20ms on CPU, negligible next to the LLM
call that follows a recognized command.
"""

import logging
from dataclasses import dataclass

import numpy as np

from .embedding import compute_embedding
from .storage import get_canonical_embedding

log = logging.getLogger("jarvis.voice_id.verify")

# Cosine similarity threshold. Resemblyzer's textbook same-speaker range is
# 0.75–0.95, cross-speaker 0.0–0.5. The old 0.65 was set low to compensate
# for runtime clips scoring far below enrollment — but that gap turned out to
# be a BUG, not a fact of life: compute_embedding was skipping preprocess_wav,
# so trailing silence in snapshot(N) clips diluted the embedding and dragged
# the owner's runtime score down into impostor range (family members got in).
# With preprocessing fixed (embedding.py), runtime and enrollment now embed
# consistently, so the bands separate again and we can afford a stricter gate.
#
# VALIDATED 2026-06-21 against a live family repro (logged to
# /tmp/jarvis-backend.log) after a clean 3-sample re-enrollment:
#   owner accepts: 0.733–0.857 (floor 0.733)
#   3 kids:        0.553–0.662 (ceiling 0.662)
#   gap +0.071, midpoint 0.698 — fully separable, zero overlap.
# Set to 0.72 (above the midpoint) to favor keeping kids out over owner
# convenience: a false reject just costs a repeated command; a false accept
# means a kid drives JARVIS. TRADE-OFF: at 0.72 the owner's margin is only
# ~0.013 on a thin 3-sample profile, so occasional self-rejects are expected.
# The fix for that is NOT lowering this — it's ENROLLING MORE SAMPLES (8–10):
# a richer canonical raises the owner's floor and widens the margin, after
# which 0.72 sits comfortably. Every decision logs its similarity at INFO;
# re-measure from the log if the distribution shifts.
VERIFY_THRESHOLD = 0.72


@dataclass(frozen=True, slots=True)
class VerifyResult:
    recognized: bool
    similarity: float
    profile_id: int | None


def should_verify_speaker(msg: dict) -> bool:
    """Decide whether an incoming WebSocket message is subject to speaker
    verification policy.

    Rules:
    - source == "text" → skip (typed input; auth token is the gate here,
      not voice ID, which is specifically about "voice within mic range")
    - Otherwise → True (voice path)

    The caller is responsible for handling the not-enrolled case — typically
    by rejecting the command and prompting the user to enroll. Returning True
    here means "this is voice; the policy applies."
    """
    return msg.get("source") != "text"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for unit-norm embeddings (dot product suffices,
    but we normalize anyway in case the mean embedding drifted)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def verify_speaker(audio_bytes: bytes) -> VerifyResult:
    """Verify a single utterance against the enrolled canonical embedding.

    Always re-embeds the incoming audio and compares — no trust-by-connection
    shortcut. The canonical embedding is cached in storage.py, so the only
    per-call cost is the ~20ms incoming embedding.

    Every decision logs its similarity at INFO so the runtime score
    distribution (and any false-accepts) is observable in the server log.
    """
    canonical = get_canonical_embedding()
    if canonical is None:
        # No profile enrolled — caller should gate on is_enrolled() first,
        # but this is a defensive fallback.
        return VerifyResult(recognized=False, similarity=0.0, profile_id=None)

    profile_id, canonical_emb = canonical
    incoming_emb = compute_embedding(audio_bytes)
    similarity = _cosine_similarity(canonical_emb, incoming_emb)
    recognized = similarity >= VERIFY_THRESHOLD

    verdict = "ACCEPT" if recognized else "REJECT"
    log.info(f"Voice {verdict} similarity={similarity:.3f} (threshold={VERIFY_THRESHOLD})")

    return VerifyResult(
        recognized=recognized,
        similarity=similarity,
        profile_id=profile_id if recognized else None,
    )
