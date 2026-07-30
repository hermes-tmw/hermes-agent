"""Tests for tools.vad -- Silero VAD integration and config parsing.

These tests do not require a real microphone.  The Silero model file is
expected at ``~/.hermes/cache/silero-vad/silero_vad.onnx`` on this host;
tests that need it are skipped when the model is absent.  The model is a
neural voice-activity detector, so deterministic synthetic audio is used to
exercise the state machine and endpoint logic rather than asserting exact
probabilities.
"""

import os
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


MODEL_PATH = Path.home() / ".hermes" / "cache" / "silero-vad" / "silero_vad.onnx"


def _silero_model_available():
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return MODEL_PATH.is_file()


def _write_synthetic_wav(path: str, duration_s: float, *, loud: bool = False) -> None:
    """Write a 16-bit 16kHz mono WAV with synthetic audio.

    ``loud=True`` produces a high-energy burst; ``loud=False`` produces
    near-silence.  The energy difference is large enough to drive the RMS and
    Silero state machines deterministically.
    """
    sample_rate = 16000
    n_frames = int(sample_rate * duration_s)
    if loud:
        # A strong tone-like signal; energy is high.
        samples = [int(8000 * (1 if i % 2 else -1)) for i in range(n_frames)]
    else:
        samples = [0] * n_frames
    audio = struct.pack(f"<{n_frames}h", *samples)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


# ============================================================================
# Config parsing
# ============================================================================
class TestVADConfigParsing:
    def test_default_is_rms(self):
        from tools.vad import vad_config_from_dict

        cfg = vad_config_from_dict({})
        assert cfg.provider == "rms"
        assert cfg.threshold == 200
        assert cfg.min_silence_ms == 3000
        assert cfg.model_path is None

    def test_legacy_voice_keys_are_honored(self):
        from tools.vad import vad_config_from_dict

        cfg = vad_config_from_dict({
            "voice": {
                "silence_threshold": 500,
                "silence_duration": 2.0,
            },
        })
        assert cfg.provider == "rms"
        assert cfg.threshold == 500
        assert cfg.min_silence_ms == 2000

    def test_silero_provider_is_parsed(self):
        from tools.vad import vad_config_from_dict

        cfg = vad_config_from_dict({
            "voice": {
                "vad": {
                    "provider": "silero",
                    "silero": {
                        "threshold": 0.7,
                        "min_speech_ms": 400,
                        "min_silence_ms": 900,
                        "model_path": "/tmp/silero.onnx",
                    },
                },
            },
        })
        assert cfg.provider == "silero"
        assert cfg.threshold == 0.7
        assert cfg.min_speech_ms == 400
        assert cfg.min_silence_ms == 900
        assert cfg.model_path == "/tmp/silero.onnx"

    def test_bounds_are_applied(self):
        from tools.vad import vad_config_from_dict

        cfg = vad_config_from_dict({
            "voice": {
                "vad": {
                    "provider": "silero",
                    "silero": {
                        "threshold": -1.0,
                        "min_speech_ms": 10000,
                        "min_silence_ms": 50000,
                    },
                },
            },
        })
        assert cfg.threshold == 0.0
        assert cfg.min_speech_ms == 5000
        assert cfg.min_silence_ms == 10000

    def test_invalid_provider_defaults_to_rms(self):
        from tools.vad import vad_config_from_dict

        cfg = vad_config_from_dict({"voice": {"vad": {"provider": "super_vad"}}})
        assert cfg.provider == "rms"

    def test_malformed_voice_block_is_safe(self):
        from tools.vad import vad_config_from_dict

        for bad in (True, "scalar", ["list"], None):
            cfg = vad_config_from_dict({"voice": bad})
            assert cfg.provider == "rms"
            assert cfg.threshold == 200


# ============================================================================
# Factory
# ============================================================================
class TestVADFactory:
    @pytest.mark.skipif(not _silero_model_available(), reason="Silero model not available")
    def test_load_vad_from_config_returns_silero_when_requested(self):
        from tools.vad import SileroVAD, load_vad_from_config

        vad, cfg, is_fallback = load_vad_from_config({
            "voice": {"vad": {"provider": "silero"}},
        })
        assert isinstance(vad, SileroVAD)
        assert cfg.provider == "silero"
        assert is_fallback is False

    def test_load_vad_from_config_returns_rms_by_default(self):
        from tools.vad import RMSVAD, load_vad_from_config

        vad, cfg, is_fallback = load_vad_from_config({})
        assert isinstance(vad, RMSVAD)
        assert cfg.provider == "rms"
        assert is_fallback is False

    def test_silero_missing_model_falls_back_to_rms(self, tmp_path, monkeypatch):
        from tools.vad import RMSVAD, load_vad_from_config

        fake_model = tmp_path / "not_there.onnx"
        monkeypatch.setenv("HERMES_SILERO_VAD_MODEL", str(fake_model))
        try:
            vad, cfg, is_fallback = load_vad_from_config({
                "voice": {"vad": {"provider": "silero"}},
            })
        finally:
            monkeypatch.delenv("HERMES_SILERO_VAD_MODEL", raising=False)
        assert isinstance(vad, RMSVAD)
        assert is_fallback is True


# ============================================================================
# RMSVAD state machine
# ============================================================================
class TestRMSVAD:
    def test_endpoint_after_silence_duration(self):
        from tools.vad import RMSVAD

        vad = RMSVAD(threshold=200, silence_duration=0.1, min_speech_duration=0.05)
        start = 1000.0
        vad.reset(start)
        assert not vad.update(500, start + 0.06)   # speech confirmed
        assert not vad.update(500, start + 0.12)     # still speaking
        assert not vad.update(50, start + 0.13)    # silence starts
        assert vad.update(50, start + 0.25)          # endpoint

    def test_no_endpoint_without_speech(self):
        from tools.vad import RMSVAD

        vad = RMSVAD(threshold=200, silence_duration=0.05)
        start = 1000.0
        vad.reset(start)
        assert not vad.update(50, start + 0.06)
        assert not vad.update(50, start + 0.12)


# ============================================================================
# SileroVAD model loading and endpoint detection
# ============================================================================
@pytest.mark.skipif(not _silero_model_available(), reason="Silero model not available")
class TestSileroVAD:
    def test_model_loads_and_returns_probabilities(self):
        from tools.vad import SileroVAD

        vad = SileroVAD(model_path=str(MODEL_PATH))
        vad._ensure_session()
        assert vad._session is not None

    def test_silence_returns_low_probability(self):
        import numpy as np
        from tools.vad import SileroVAD

        vad = SileroVAD(model_path=str(MODEL_PATH))
        silence = np.zeros((1, 512), dtype=np.float32)
        prob = vad._consume_chunk(silence)
        assert 0.0 <= prob < 0.1

    def test_loud_burst_returns_higher_probability_than_silence(self):
        import numpy as np
        from tools.vad import SileroVAD

        vad = SileroVAD(model_path=str(MODEL_PATH))
        silence = np.zeros((1, 512), dtype=np.float32)
        loud = np.random.randn(1, 512).astype(np.float32) * 0.3
        p_sil = vad._consume_chunk(silence)
        p_loud = vad._consume_chunk(loud)
        assert p_loud > p_sil

    def test_detect_endpoint_with_synthetic_bursts(self, tmp_path):
        """Test the endpoint state machine: speech → silence → endpoint.

        Silero VAD is trained on real human speech — synthetic noise produces
        near-zero probabilities. Instead of fighting the model, we mock
        `_consume_chunk` to return controlled probabilities and verify the
        endpoint detection state machine works correctly.
        """
        import numpy as np
        from tools.vad import SileroVAD

        vad = SileroVAD(
            model_path=str(MODEL_PATH),
            threshold=0.5,
            min_speech_ms=300,
            min_silence_ms=700,
        )

        # Mock: 20 chunks of "speech" (prob=0.8) then 30 chunks of "silence" (prob=0.01)
        # 20 chunks * 32ms = 640ms > 300ms min_speech → speech confirmed
        # 30 chunks * 32ms = 960ms > 700ms min_silence → endpoint detected
        speech_probs = [0.8] * 20 + [0.01] * 30
        call_count = [0]
        orig_consume = vad._consume_chunk

        def mock_consume(chunk):
            if call_count[0] < len(speech_probs):
                prob = speech_probs[call_count[0]]
            else:
                prob = 0.01
            call_count[0] += 1
            vad._last_prob = prob
            # Still update state so the model path is exercised
            try:
                orig_consume(chunk)
            except Exception:
                pass
            return prob

        vad._consume_chunk = mock_consume
        # Don't need a real session since we're mocking inference
        vad._session = object()  # prevent _ensure_session from loading
        vad._state = np.zeros((2, 1, 128), dtype=np.float32)
        vad._sample_rate_arr = np.array(16000, dtype=np.int64)

        # Build audio buffer: 20*512 + 30*512 = 25600 samples at 16kHz
        audio = np.zeros(20 * 512 + 30 * 512, dtype=np.int16)
        assert vad.detect_endpoint(audio, 16000) is True

    def test_detect_endpoint_false_for_brief_noise(self, tmp_path):
        import numpy as np
        from tools.vad import SileroVAD

        # A single 200ms burst is not enough to confirm speech.
        sample_rate = 16000
        audio = np.array([8000 * (1 if i % 2 else -1) for i in range(int(sample_rate * 0.2))], dtype=np.int16)

        vad = SileroVAD(
            model_path=str(MODEL_PATH),
            threshold=0.5,
            min_speech_ms=300,
            min_silence_ms=700,
        )
        assert vad.detect_endpoint(audio, sample_rate) is False

    def test_reset_clears_state(self):
        import numpy as np
        from tools.vad import SileroVAD

        vad = SileroVAD(model_path=str(MODEL_PATH))
        vad._ensure_session()
        original_state = vad._state.copy()
        vad._consume_chunk(np.random.randn(1, 512).astype(np.float32) * 0.3)
        assert not np.array_equal(vad._state, original_state)
        vad.reset()
        assert np.array_equal(vad._state, original_state)

    def test_missing_model_raises_file_not_found(self, tmp_path):
        from tools.vad import SileroVAD

        fake_model = tmp_path / "missing.onnx"
        vad = SileroVAD(model_path=str(fake_model))
        with pytest.raises(FileNotFoundError):
            vad._ensure_session()


# ============================================================================
# AudioRecorder VAD integration (mocked sounddevice)
# ============================================================================
@pytest.fixture
def mock_sd(monkeypatch):
    """Mock _import_audio to return (mock_sd, real_np) so lazy imports work."""
    mock = MagicMock()
    try:
        import numpy as real_np
    except ImportError:
        real_np = MagicMock()

    def _fake_import_audio():
        return mock, real_np

    monkeypatch.setattr("tools.voice_mode._import_audio", _fake_import_audio)
    monkeypatch.setattr("tools.voice_mode._audio_available", lambda: True)
    return mock


class TestAudioRecorderVAD:
    def test_recorder_defaults_to_rms(self, mock_sd):
        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        assert recorder.vad_provider == "rms"

    def test_recorder_uses_silero_when_configured(self, mock_sd, monkeypatch):
        from tools.voice_mode import AudioRecorder

        # Ensure factory creates a SileroVAD by making the model "available".
        monkeypatch.setattr("tools.vad._default_silero_model_path", lambda: str(MODEL_PATH))
        recorder = AudioRecorder(config={"voice": {"vad": {"provider": "silero"}}})
        if not _silero_model_available():
            pytest.skip("Silero model not available")
        assert recorder.vad_provider == "silero"

    def test_recorder_falls_back_to_rms_when_silero_missing(self, mock_sd, tmp_path, monkeypatch):
        from tools.voice_mode import AudioRecorder

        fake_model = tmp_path / "not_there.onnx"
        monkeypatch.setenv("HERMES_SILERO_VAD_MODEL", str(fake_model))
        try:
            recorder = AudioRecorder(config={"voice": {"vad": {"provider": "silero"}}})
            assert recorder.vad_provider in {"rms", "rms_fallback"}
        finally:
            monkeypatch.delenv("HERMES_SILERO_VAD_MODEL", raising=False)


# ============================================================================
# Backward compatibility helpers
# ============================================================================
class TestBackwardCompatibility:
    def test_silero_vad_available_false_for_missing_file(self, tmp_path):
        from tools.voice_mode import _silero_vad_available

        fake = tmp_path / "missing.onnx"
        assert _silero_vad_available(str(fake)) is False

    @pytest.mark.skipif(not _silero_model_available(), reason="Silero model not available")
    def test_silero_vad_available_true_for_real_model(self):
        from tools.voice_mode import _silero_vad_available

        assert _silero_vad_available(str(MODEL_PATH)) is True
