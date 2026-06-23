"""
Speaker identification for JARVIS.

Public API (backend, consumed by api/voice.py and server.py voice_handler):
- enroll_sample(audio_bytes, name) -> int           # returns new sample_count
- verify_speaker(audio_bytes) -> VerifyResult        # verifies EVERY utterance
- get_status() -> StatusDict
- is_enrolled() -> bool                              # fast bool for the voice handler gate
- clear_profile() -> None
- rebuild_embeddings() -> int                        # re-embed stored samples from retained audio
- maybe_rebuild_on_pipeline_change() -> int          # startup self-heal when the pipeline changed
"""

from .storage import (
    clear_profile,
    enroll_sample,
    get_status,
    is_enrolled,
    maybe_rebuild_on_pipeline_change,
    rebuild_embeddings,
)
from .verify import VerifyResult, should_verify_speaker, verify_speaker

__all__ = [
    "VerifyResult",
    "clear_profile",
    "enroll_sample",
    "get_status",
    "is_enrolled",
    "maybe_rebuild_on_pipeline_change",
    "rebuild_embeddings",
    "should_verify_speaker",
    "verify_speaker",
]
