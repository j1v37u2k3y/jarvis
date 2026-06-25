"""
Resemblyzer wrapper — computes 256-dim speaker embeddings.

The VoiceEncoder model is loaded once, lazily, on first call. ~14MB
download on first import; ~20ms per embedding on CPU after that.
"""

import logging
import threading
from importlib.metadata import version as _pkg_version

import numpy as np

from .wav import TARGET_SAMPLE_RATE, AudioTooShortError, decode

log = logging.getLogger("jarvis.voice_id.embedding")

EMBEDDING_DIM = 256

# Bump this whenever compute_embedding's behavior changes in a way that shifts
# the embedding space (e.g. adding/removing preprocess_wav, changing the VAD
# trim, swapping the model). It's folded into pipeline_fingerprint() together
# with the installed resemblyzer version; when the fingerprint changes, stored
# profiles are auto-re-embedded from their retained audio on startup (see
# storage.maybe_rebuild_on_pipeline_change) — no re-recording at the mic.
#   v1 = raw audio -> embed_utterance (the false-accept bug)
#   v2 = preprocess_wav (normalize + VAD-trim) -> embed_utterance
PIPELINE_VERSION = "v2-preprocess_wav"


def pipeline_fingerprint() -> str:
    """Stable identifier for the current embedding pipeline. Changes when the
    embedding space shifts (PIPELINE_VERSION bump or a resemblyzer upgrade),
    which triggers an auto re-embed of stored samples from retained audio."""
    try:
        rv = _pkg_version("resemblyzer")
    except Exception:  # noqa: BLE001 — any metadata failure -> opaque token, still stable per env
        rv = "unknown"
    return f"{PIPELINE_VERSION}+resemblyzer:{rv}"


# Lazy-loaded singleton so importing this module is cheap — the model
# only loads when we actually compute an embedding.
_encoder = None
_encoder_lock = threading.Lock()


def _get_encoder():
    """Load VoiceEncoder lazily. Thread-safe."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is None:
            from resemblyzer import VoiceEncoder

            log.info("Loading resemblyzer VoiceEncoder (first use)")
            _encoder = VoiceEncoder(verbose=False)
    return _encoder


def compute_embedding(audio_bytes: bytes) -> np.ndarray:
    """Decode audio, preprocess, and produce a 256-dim float32 embedding.

    resemblyzer's ``embed_utterance`` expects a *preprocessed* waveform — its
    own docstring says so — meaning volume-normalized and VAD silence-trimmed
    (see ``resemblyzer.preprocess_wav``). We previously handed it the raw
    decoded audio, which let trailing silence in a runtime ``snapshot(N)`` clip
    dilute the embedding toward a generic "quiet room" centroid and collapse
    the gap between the enrolled owner and other speakers — i.e. false accepts.
    Enrollment escaped this because those clips are VAD-cropped on the frontend;
    runtime snapshots are not. Routing both through preprocess_wav makes the two
    paths consistent and re-opens the separation.

    Raises ``wav.AudioTooShortError`` if the clip is too short to embed — which
    now includes the case where VAD trimming leaves too little voiced audio
    (e.g. a snapshot that was almost entirely silence). The voice handler wraps
    verification in try/except and drops the command, so this fails closed.
    """
    from resemblyzer import preprocess_wav

    audio = decode(audio_bytes)
    # Volume-normalize + trim long silences (webrtcvad). source_sr is omitted
    # because decode() already returns 16kHz, so no resampling happens here.
    processed = preprocess_wav(audio)
    if len(processed) < TARGET_SAMPLE_RATE // 2:  # <0.5s voiced after trimming
        raise AudioTooShortError(
            f"Only {len(processed) / TARGET_SAMPLE_RATE:.2f}s of voiced audio after VAD trim — "
            "no usable speech in the clip"
        )
    encoder = _get_encoder()
    embedding = encoder.embed_utterance(processed)
    return embedding.astype(np.float32, copy=False)
