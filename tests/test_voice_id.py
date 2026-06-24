"""
Speaker identification — unit tests for the voice_id package.

We use synthetic audio (different signal families act as different
"speakers") rather than real recordings. Real-voice smoke is done
manually via the enrollment UI after PR #2.
"""

import io

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Isolate each test: point DB_PATH at a tmp file, reset caches
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_voice_id(tmp_path, monkeypatch):
    """Redirect voice_id.storage.DB_PATH to a per-test tmp file, clear the
    canonical-embedding cache, and reset server._AUTH_TOKEN (other tests
    leave it set).
    """
    import server
    from voice_id import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "voice_profiles.db")
    monkeypatch.setattr(server, "_AUTH_TOKEN", "")
    storage.invalidate_canonical()
    # Force the resemblyzer encoder to persist across tests for speed — it
    # takes ~0.5s to load. Don't reset embedding._encoder.
    yield
    storage.invalidate_canonical()


# ---------------------------------------------------------------------------
# Synthetic "speakers"
# ---------------------------------------------------------------------------


def _synth_wav(kind: str, seed: int) -> bytes:
    """Generate a 2s WAV clip. Each `kind` is a different synthetic 'speaker'."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2, 32000, dtype=np.float32)
    if kind == "speaker_a":
        # Pink-ish noise centered on 200Hz
        x = 0.3 * rng.standard_normal(32000).astype(np.float32)
        x += 0.2 * np.sin(2 * np.pi * 200 * t)
    elif kind == "speaker_b":
        # Much higher fundamental + different noise
        x = 0.1 * rng.standard_normal(32000).astype(np.float32)
        x += 0.4 * np.sin(2 * np.pi * 440 * t)
        x += 0.2 * np.sin(2 * np.pi * 880 * t)
    else:
        raise ValueError(f"unknown speaker kind: {kind}")
    buf = io.BytesIO()
    sf.write(buf, x, 16000, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Storage + enrollment
# ---------------------------------------------------------------------------


class TestEnrollment:
    def test_enroll_stores_sample(self, isolated_voice_id):
        from voice_id import enroll_sample, get_status, is_enrolled

        assert not is_enrolled()
        assert get_status() == {"enrolled": False, "name": None, "sample_count": 0}

        count = enroll_sample(_synth_wav("speaker_a", 1), "tom")
        assert count == 1
        assert is_enrolled()
        assert get_status() == {"enrolled": True, "name": "tom", "sample_count": 1}

    def test_enroll_multiple_samples_increments_count(self, isolated_voice_id):
        from voice_id import enroll_sample, get_status

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        enroll_sample(_synth_wav("speaker_a", 2), "tom")
        count = enroll_sample(_synth_wav("speaker_a", 3), "tom")
        assert count == 3
        assert get_status()["sample_count"] == 3

    def test_audio_too_short_rejected(self, isolated_voice_id):
        from voice_id import enroll_sample
        from voice_id.wav import AudioTooShortError

        # 0.5 seconds at 16kHz — below the 1s floor
        short = np.zeros(8000, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, short, 16000, format="WAV")

        with pytest.raises(AudioTooShortError):
            enroll_sample(buf.getvalue(), "tom")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TestVerification:
    def test_same_speaker_passes(self, isolated_voice_id):
        from voice_id import enroll_sample, verify_speaker
        from voice_id.verify import VERIFY_THRESHOLD

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        enroll_sample(_synth_wav("speaker_a", 2), "tom")

        result = verify_speaker(_synth_wav("speaker_a", 99))
        assert result.recognized, f"same speaker rejected (sim={result.similarity:.3f}, threshold={VERIFY_THRESHOLD})"
        assert result.similarity > VERIFY_THRESHOLD
        assert result.profile_id is not None

    def test_different_speaker_rejected(self, isolated_voice_id):
        from voice_id import enroll_sample, verify_speaker
        from voice_id.verify import VERIFY_THRESHOLD

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        enroll_sample(_synth_wav("speaker_a", 2), "tom")

        result = verify_speaker(_synth_wav("speaker_b", 99))
        assert not result.recognized, f"different speaker passed (sim={result.similarity:.3f})"
        assert result.similarity < VERIFY_THRESHOLD
        assert result.profile_id is None

    def test_no_profile_returns_false_without_error(self, isolated_voice_id):
        from voice_id import verify_speaker

        result = verify_speaker(_synth_wav("speaker_a", 1))
        assert not result.recognized
        assert result.similarity == 0.0
        assert result.profile_id is None

    def test_every_utterance_is_reverified(self, isolated_voice_id):
        """THE security regression test. Before this fix, a positive verify
        cached "trusted" for 30s per connection, so an impostor speaking right
        after the owner sailed through. Now every utterance is scored on its
        own merits — the owner passing does NOT grant a trust window to the
        next voice.
        """
        from voice_id import enroll_sample, verify_speaker

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        enroll_sample(_synth_wav("speaker_a", 2), "tom")

        owner = verify_speaker(_synth_wav("speaker_a", 99))
        assert owner.recognized, f"owner rejected (sim={owner.similarity:.3f})"

        # Impostor immediately after the owner — must be judged fresh, not
        # waved through on the owner's coattails.
        impostor = verify_speaker(_synth_wav("speaker_b", 99))
        assert not impostor.recognized, (
            f"impostor passed right after owner (sim={impostor.similarity:.3f}) — the 30s trust window is back"
        )
        assert impostor.profile_id is None


# ---------------------------------------------------------------------------
# Preprocessing robustness — the family-false-accept fix
# ---------------------------------------------------------------------------


class TestPreprocessingRobustness:
    """compute_embedding routes audio through resemblyzer.preprocess_wav, which
    volume-normalizes and VAD-trims silence. This is the fix for the bug where
    family members cleared the gate: runtime snapshot(N) clips include trailing
    silence (Web Speech finalizes ~1-2s after the speaker stops), and feeding
    that raw to embed_utterance diluted the owner's embedding toward a generic
    centroid — collapsing the gap that separates the owner from impostors.
    """

    @staticmethod
    def _synth_a_with_silence(seed: int, silence_seconds: float) -> bytes:
        """speaker_a from _synth_wav with trailing silence appended — mimics a
        runtime snapshot that caught speech plus the trailing endpoint gap.
        Reuses speaker_a's exact signal so the only difference is the silence."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 2, 32000, dtype=np.float32)
        x = 0.3 * rng.standard_normal(32000).astype(np.float32)
        x += 0.2 * np.sin(2 * np.pi * 200 * t)
        x = np.concatenate([x, np.zeros(int(silence_seconds * 16000), dtype=np.float32)])
        buf = io.BytesIO()
        sf.write(buf, x.astype(np.float32), 16000, format="WAV")
        return buf.getvalue()

    def test_trailing_silence_does_not_corrupt_embedding(self, isolated_voice_id):
        """Same utterance, with vs without 2s of trailing silence, must embed to
        nearly the same vector. Under the old raw-embed path this drifted ~0.10
        — enough to drop the owner into impostor range; preprocess_wav trims the
        silence so it stays ~0.998. Guards against reverting embedding.py to a
        bare embed_utterance call.
        """
        from voice_id.embedding import compute_embedding

        clean = _synth_wav("speaker_a", 1)
        padded = self._synth_a_with_silence(1, silence_seconds=2.0)

        a = compute_embedding(clean)
        b = compute_embedding(padded)
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        assert sim >= 0.97, (
            f"trailing silence shifted the embedding (sim={sim:.3f}) — "
            "compute_embedding must apply preprocess_wav to VAD-trim silence"
        )

    def test_silence_only_clip_is_rejected_not_crashed(self, isolated_voice_id):
        """A snapshot with no real speech trims to ~nothing. compute_embedding
        must raise AudioTooShortError (the voice handler catches it and drops the
        command → fail-closed) rather than embedding silence as a valid speaker.
        """
        from voice_id.embedding import compute_embedding
        from voice_id.wav import AudioTooShortError

        # 3s of near-silence (tiny dither so it's valid PCM, longer than the 1s
        # raw-duration floor — the rejection must come from the VAD trim, not
        # from decode()'s length check).
        rng = np.random.default_rng(7)
        quiet = (1e-4 * rng.standard_normal(48000)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, quiet, 16000, format="WAV")

        with pytest.raises(AudioTooShortError):
            compute_embedding(buf.getvalue())


# ---------------------------------------------------------------------------
# Canonical-embedding cache (the only thing we cache, and only because it
# changes solely on re-enrollment)
# ---------------------------------------------------------------------------


class TestCanonicalCache:
    def test_canonical_is_cached_after_first_read(self, isolated_voice_id):
        from voice_id import enroll_sample, storage, verify_speaker

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        assert storage._canonical_cache == {}, "enroll must leave the cache empty (invalidated)"

        verify_speaker(_synth_wav("speaker_a", 2))
        assert None in storage._canonical_cache, "verify should populate the canonical cache"

    def test_enroll_invalidates_canonical_cache(self, isolated_voice_id):
        from voice_id import enroll_sample, storage, verify_speaker

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        verify_speaker(_synth_wav("speaker_a", 2))
        assert None in storage._canonical_cache

        # Adding a sample changes the canonical mean — cache must drop so the
        # next verify recomputes against the new profile.
        enroll_sample(_synth_wav("speaker_a", 3), "tom")
        assert storage._canonical_cache == {}, "adding a sample must invalidate the canonical cache"

    def test_clear_profile_invalidates_canonical_cache(self, isolated_voice_id):
        from voice_id import clear_profile, enroll_sample, storage, verify_speaker

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        verify_speaker(_synth_wav("speaker_a", 2))
        assert None in storage._canonical_cache

        clear_profile()
        assert storage._canonical_cache == {}, "clearing the profile must invalidate the canonical cache"


# ---------------------------------------------------------------------------
# Persistent enrollment — raw-audio retention + pipeline-change auto-rebuild
# ---------------------------------------------------------------------------


class TestPersistentEnrollment:
    """Enrollment retains the raw WAV so the profile can be re-embedded from
    audio after a pipeline change — the owner enrolls once, the system self-heals
    on the next start instead of forcing a re-record at the mic.
    """

    @staticmethod
    def _sample_audio_present(name: str = "tom") -> tuple[int, int]:
        """Returns (samples_with_audio, total_samples) for the named profile."""
        import sqlite3

        from voice_id import storage

        conn = sqlite3.connect(str(storage.DB_PATH))
        try:
            total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            with_audio = conn.execute("SELECT COUNT(*) FROM samples WHERE audio IS NOT NULL").fetchone()[0]
            return with_audio, total
        finally:
            conn.close()

    def test_enroll_retains_raw_audio(self, isolated_voice_id):
        from voice_id import enroll_sample

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        enroll_sample(_synth_wav("speaker_a", 2), "tom")
        with_audio, total = self._sample_audio_present()
        assert total == 2
        assert with_audio == 2, "every enrolled sample must retain its raw audio for rebuild"

    def test_enroll_stamps_pipeline_fingerprint(self, isolated_voice_id):
        import sqlite3

        from voice_id import enroll_sample, storage
        from voice_id.embedding import pipeline_fingerprint

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        conn = sqlite3.connect(str(storage.DB_PATH))
        try:
            conn.row_factory = sqlite3.Row
            stored = conn.execute("SELECT value FROM meta WHERE key = 'pipeline_fingerprint'").fetchone()
        finally:
            conn.close()
        assert stored is not None
        assert stored["value"] == pipeline_fingerprint()

    def test_rebuild_reembeds_from_audio(self, isolated_voice_id):
        """rebuild_embeddings recomputes embeddings from retained audio. After
        zeroing a stored embedding, rebuild restores a valid (non-zero) one that
        still verifies the owner."""
        import sqlite3

        import numpy as np

        from voice_id import enroll_sample, rebuild_embeddings, storage, verify_speaker
        from voice_id.embedding import EMBEDDING_DIM

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        enroll_sample(_synth_wav("speaker_a", 2), "tom")

        # Corrupt a stored embedding to prove rebuild actually recomputes it.
        zero = np.zeros(EMBEDDING_DIM, dtype=np.float32).tobytes()
        conn = sqlite3.connect(str(storage.DB_PATH))
        try:
            conn.execute("UPDATE samples SET embedding = ? WHERE id = (SELECT MIN(id) FROM samples)", (zero,))
            conn.commit()
        finally:
            conn.close()
        storage.invalidate_canonical()

        rebuilt = rebuild_embeddings()
        assert rebuilt == 2

        # The corrupted embedding is restored: owner still verifies.
        result = verify_speaker(_synth_wav("speaker_a", 99))
        assert result.recognized, f"owner rejected after rebuild (sim={result.similarity:.3f})"

    def test_pipeline_change_triggers_auto_rebuild(self, isolated_voice_id, monkeypatch):
        from voice_id import enroll_sample, maybe_rebuild_on_pipeline_change, storage

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        enroll_sample(_synth_wav("speaker_a", 2), "tom")

        # Same pipeline → no-op.
        assert maybe_rebuild_on_pipeline_change() == 0

        # Simulate a pipeline change: the fingerprint now differs from what was
        # stamped at enrollment, so startup should re-embed from retained audio.
        monkeypatch.setattr(storage, "pipeline_fingerprint", lambda: "v999-changed+resemblyzer:test")
        assert maybe_rebuild_on_pipeline_change() == 2

        # Fingerprint is now stamped current → second call is a no-op.
        assert maybe_rebuild_on_pipeline_change() == 0

    def test_prune_removes_audioless_keeps_audio(self, isolated_voice_id):
        """prune_unrebuildable drops samples with no retained audio (legacy,
        unrebuildable) and keeps the audio-bearing ones — the cleanup for a
        profile that mixed pre-retention and retention-era samples."""
        import sqlite3

        from voice_id import enroll_sample, get_status, prune_unrebuildable, storage

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        enroll_sample(_synth_wav("speaker_a", 2), "tom")
        # Inject two legacy rows with NULL audio under the same profile.
        conn = sqlite3.connect(str(storage.DB_PATH))
        try:
            pid = conn.execute("SELECT id FROM profiles WHERE name='tom'").fetchone()[0]
            import numpy as np

            blob = np.zeros(256, dtype=np.float32).tobytes()
            conn.execute(
                "INSERT INTO samples (profile_id, embedding, audio, created_at) VALUES (?, ?, NULL, 0)", (pid, blob)
            )
            conn.execute(
                "INSERT INTO samples (profile_id, embedding, audio, created_at) VALUES (?, ?, NULL, 0)", (pid, blob)
            )
            conn.commit()
        finally:
            conn.close()
        assert get_status()["sample_count"] == 4

        pruned = prune_unrebuildable()
        assert pruned == 2
        status = get_status()
        assert status["sample_count"] == 2, "only the 2 audio-bearing samples should remain"
        assert status["enrolled"]

    def test_prune_drops_profile_when_all_audioless(self, isolated_voice_id):
        """If pruning leaves a profile with zero samples, the profile row is
        removed too so status reports not-enrolled (no phantom profile)."""
        import sqlite3

        from voice_id import enroll_sample, get_status, prune_unrebuildable, storage

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        # Strip audio from the only sample → it's now unrebuildable legacy.
        conn = sqlite3.connect(str(storage.DB_PATH))
        try:
            conn.execute("UPDATE samples SET audio = NULL")
            conn.commit()
        finally:
            conn.close()

        assert prune_unrebuildable() == 1
        assert get_status() == {"enrolled": False, "name": None, "sample_count": 0}

    def test_no_rebuild_without_retained_audio(self, isolated_voice_id, monkeypatch):
        """A profile whose samples have no retained audio (pre-retention DB)
        can't be rebuilt — maybe_rebuild is a no-op even if the pipeline changed."""
        import sqlite3

        from voice_id import enroll_sample, maybe_rebuild_on_pipeline_change, storage

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        # Strip retained audio to emulate a pre-retention sample.
        conn = sqlite3.connect(str(storage.DB_PATH))
        try:
            conn.execute("UPDATE samples SET audio = NULL")
            conn.commit()
        finally:
            conn.close()

        monkeypatch.setattr(storage, "pipeline_fingerprint", lambda: "v999-changed+resemblyzer:test")
        assert maybe_rebuild_on_pipeline_change() == 0


# ---------------------------------------------------------------------------
# Clear profile
# ---------------------------------------------------------------------------


class TestClearProfile:
    def test_clear_resets_status(self, isolated_voice_id):
        from voice_id import clear_profile, enroll_sample, get_status, is_enrolled

        enroll_sample(_synth_wav("speaker_a", 1), "tom")
        assert is_enrolled()

        clear_profile()
        assert not is_enrolled()
        assert get_status() == {"enrolled": False, "name": None, "sample_count": 0}


# ---------------------------------------------------------------------------
# REST API surface (FastAPI TestClient)
# ---------------------------------------------------------------------------


class TestRESTEndpoints:
    def test_status_endpoint_returns_not_enrolled(self, isolated_voice_id):
        from fastapi.testclient import TestClient

        import server

        client = TestClient(server.app)
        r = client.get("/api/voice/status", headers={"Authorization": "Bearer "})
        assert r.status_code == 200
        assert r.json() == {"enrolled": False, "name": None, "sample_count": 0}

    def test_enroll_via_upload(self, isolated_voice_id):
        from fastapi.testclient import TestClient

        import server

        client = TestClient(server.app)
        wav = _synth_wav("speaker_a", 1)
        r = client.post(
            "/api/voice/enroll",
            headers={"Authorization": "Bearer "},
            data={"name": "tom"},
            files={"audio": ("sample.wav", wav, "audio/wav")},
        )
        assert r.status_code == 200, r.text
        assert r.json() == {"success": True, "name": "tom", "sample_count": 1}

        r = client.get("/api/voice/status", headers={"Authorization": "Bearer "})
        assert r.json()["enrolled"] is True

    def test_enroll_rejects_short_audio(self, isolated_voice_id):
        from fastapi.testclient import TestClient

        import server

        client = TestClient(server.app)
        short = np.zeros(4000, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, short, 16000, format="WAV")
        r = client.post(
            "/api/voice/enroll",
            headers={"Authorization": "Bearer "},
            data={"name": "tom"},
            files={"audio": ("sample.wav", buf.getvalue(), "audio/wav")},
        )
        assert r.status_code == 400
        assert r.json()["success"] is False

    def test_test_endpoint_scores_without_gating(self, isolated_voice_id):
        from fastapi.testclient import TestClient

        import server
        from voice_id.verify import VERIFY_THRESHOLD

        client = TestClient(server.app)
        client.post(
            "/api/voice/enroll",
            headers={"Authorization": "Bearer "},
            data={"name": "tom"},
            files={"audio": ("a.wav", _synth_wav("speaker_a", 1), "audio/wav")},
        )

        r = client.post(
            "/api/voice/test",
            headers={"Authorization": "Bearer "},
            files={"audio": ("b.wav", _synth_wav("speaker_b", 99), "audio/wav")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "similarity" in body
        assert body["recognized"] is False
        # Pinned to the live constant so it can't silently drift like the
        # hardcoded 0.75 did when the threshold dropped to 0.65.
        assert body["threshold"] == VERIFY_THRESHOLD

    def test_clear_endpoint_resets(self, isolated_voice_id):
        from fastapi.testclient import TestClient

        import server

        client = TestClient(server.app)
        client.post(
            "/api/voice/enroll",
            headers={"Authorization": "Bearer "},
            data={"name": "tom"},
            files={"audio": ("a.wav", _synth_wav("speaker_a", 1), "audio/wav")},
        )
        r = client.delete("/api/voice/enroll", headers={"Authorization": "Bearer "})
        assert r.status_code == 200
        assert r.json() == {"success": True}

        r = client.get("/api/voice/status", headers={"Authorization": "Bearer "})
        assert r.json()["enrolled"] is False


# ---------------------------------------------------------------------------
# Voice handler gate behavior (WebSocket-level)
# ---------------------------------------------------------------------------


class TestShouldVerifySpeaker:
    """The policy function that the voice_handler gate calls.

    Regression coverage for: "typed text input bypasses the speaker gate"
    and "voice input still needs verification post-enrollment."
    """

    def test_voice_is_gated_even_when_not_enrolled(self, isolated_voice_id):
        """Pre-2026-05: this returned False (soft bootstrap), letting unverified
        voice reach the LLM. Now: voice always trips the gate; the server
        rejects when no profile is enrolled instead of soft-bootstrapping."""
        from voice_id import should_verify_speaker

        msg = {"type": "transcript", "text": "hi", "isFinal": True}
        assert should_verify_speaker(msg) is True, "voice path must hit the gate regardless of enrollment"

    def test_skips_text_input_even_when_enrolled(self, isolated_voice_id):
        """The bug that shipped in voice-id v1: text input got blocked
        with "I need to hear you to confirm, sir. Try refreshing…"
        because the gate checked only is_enrolled(). Fix added
        source="text" check."""
        from voice_id import enroll_sample, should_verify_speaker

        enroll_sample(_synth_wav("speaker_a", 1), "tom")

        text_msg = {"type": "transcript", "text": "hi", "isFinal": True, "source": "text"}
        assert should_verify_speaker(text_msg) is False, "typed input must skip the voice gate"

    def test_verifies_voice_input_when_enrolled(self, isolated_voice_id):
        from voice_id import enroll_sample, should_verify_speaker

        enroll_sample(_synth_wav("speaker_a", 1), "tom")

        voice_msg = {"type": "transcript", "text": "hi", "isFinal": True}
        assert should_verify_speaker(voice_msg) is True, "voice input must still hit the gate"

    def test_unknown_source_is_treated_as_voice(self, isolated_voice_id):
        """Defensive: any source value other than "text" falls into voice
        verification. Prevents a future refactor from coining a new source
        label (e.g. "api") that silently bypasses the check."""
        from voice_id import enroll_sample, should_verify_speaker

        enroll_sample(_synth_wav("speaker_a", 1), "tom")

        for source in ["voice", "api", "whatever", ""]:
            msg = {"type": "transcript", "text": "hi", "isFinal": True, "source": source}
            assert should_verify_speaker(msg) is True, f"source={source!r} must not bypass"
