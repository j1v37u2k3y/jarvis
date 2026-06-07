"""
Speaker identification for JARVIS.

Public API (backend, consumed by api/voice.py and server.py voice_handler):
- enroll_sample(audio_bytes, name) -> int           # returns new sample_count
- verify_speaker(audio_bytes) -> VerifyResult        # verifies EVERY utterance
- get_status() -> StatusDict
- is_enrolled() -> bool                              # fast bool for the voice handler gate
- clear_profile() -> None
"""

from .storage import clear_profile, enroll_sample, get_status, is_enrolled
from .verify import VerifyResult, should_verify_speaker, verify_speaker

__all__ = [
    "VerifyResult",
    "clear_profile",
    "enroll_sample",
    "get_status",
    "is_enrolled",
    "should_verify_speaker",
    "verify_speaker",
]
