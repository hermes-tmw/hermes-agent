"""Tests for tools.audio_frontend.

These tests avoid requiring a real microphone.  ``sounddevice`` is mocked so
frontend tests can run on headless CI hosts.
"""

from __future__ import annotations

import threading
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_sd(monkeypatch):
    """Mock sounddevice/numpy for tests that import the local frontend."""
    mock = MagicMock()
    mock_np = MagicMock()
    mock_np.int16 = "int16"
    mock_np.frombuffer.return_value = mock_np.array([0, 1, 2])
    mock.get_stream.return_value = None
    mock.InputStream.return_value = MagicMock(start=MagicMock())

    def _fake_import():
        return mock, mock_np

    monkeypatch.setattr("tools.audio_frontend._import_audio", _fake_import)
    monkeypatch.setattr("tools.audio_frontend._audio_available", lambda: True)
    return mock, mock_np


class TestAudioFrontendABC:
    def test_abc_cannot_be_instantiated(self):
        from tools.audio_frontend import AudioFrontend

        with pytest.raises(TypeError):
            AudioFrontend()

    def test_abstract_methods_must_be_implemented(self):
        from tools.audio_frontend import AudioFrontend

        class Partial(AudioFrontend):
            pass

        with pytest.raises(TypeError):
            Partial()


class TestLocalAudioFrontend:
    def test_local_frontend_implements_properties(self, mock_sd):
        from tools.audio_frontend import LocalAudioFrontend

        fe = LocalAudioFrontend()
        assert fe.supports_barge_in is False
        assert fe.is_playing is False

    def test_local_frontend_capture_lifecycle(self, mock_sd):
        from tools.audio_frontend import LocalAudioFrontend

        fe = LocalAudioFrontend()
        chunks: List[bytes] = []
        fe.start_capture(chunks.append)
        assert fe._recording is True

        # Simulate a sounddevice callback delivering a chunk.
        sd, _ = mock_sd
        callback = sd.InputStream.call_args[1]["callback"]
        indata = MagicMock()
        indata.tobytes.return_value = b"1234"
        callback(indata, 64, None, None)
        assert chunks == [b"1234"]

        fe.stop_capture()
        assert fe._recording is False

    def test_local_frontend_stop_capture_does_not_close_stream(self, mock_sd):
        from tools.audio_frontend import LocalAudioFrontend

        fe = LocalAudioFrontend()
        fe.start_capture(lambda x: None)
        stream = fe._stream
        fe.stop_capture()
        assert fe._stream is stream

    def test_local_frontend_shutdown_closes_stream(self, mock_sd):
        from tools.audio_frontend import LocalAudioFrontend

        fe = LocalAudioFrontend()
        fe.start_capture(lambda x: None)
        fe.shutdown()
        assert fe._stream is None
        assert fe._recording is False

    def test_local_frontend_playback_uses_sounddevice(self, mock_sd):
        from tools.audio_frontend import LocalAudioFrontend

        fe = LocalAudioFrontend()
        sd, _ = mock_sd
        fe.play_audio(b"\x00\x01\x02\x03")
        sd.play.assert_called_once()

    def test_local_frontend_stop_playback_clears_state(self, mock_sd):
        from tools.audio_frontend import LocalAudioFrontend

        fe = LocalAudioFrontend()
        fe.play_audio(b"\x00\x01\x02\x03")
        fe.stop_playback()
        assert fe.is_playing is False

    def test_local_frontend_missing_audio_libs_raises(self, monkeypatch):
        monkeypatch.setattr(
            "tools.audio_frontend._import_audio",
            lambda: (_ for _ in ()).throw(ImportError("no sounddevice")),
        )
        from tools.audio_frontend import LocalAudioFrontend

        with pytest.raises(RuntimeError):
            LocalAudioFrontend()


class TestMockAudioFrontend:
    def test_mock_supports_barge_in(self):
        from tools.audio_frontend import MockAudioFrontend

        fe = MockAudioFrontend()
        assert fe.supports_barge_in is True

    def test_mock_capture_callback_fires(self):
        from tools.audio_frontend import MockAudioFrontend

        fe = MockAudioFrontend()
        chunks: List[bytes] = []
        fe.start_capture(chunks.append)
        fe.feed_synthetic_chunk(b"abcd")
        fe.feed_synthetic_chunk(b"efgh")
        assert chunks == [b"abcd", b"efgh"]
        assert fe.captured == [b"abcd", b"efgh"]

    def test_mock_stop_capture_clears_callback(self):
        from tools.audio_frontend import MockAudioFrontend

        fe = MockAudioFrontend()
        received: List[bytes] = []
        fe.start_capture(received.append)
        fe.stop_capture()
        fe.feed_synthetic_chunk(b"ijkl")
        assert received == []

    def test_mock_playback_stores_audio(self):
        from tools.audio_frontend import MockAudioFrontend

        fe = MockAudioFrontend()
        audio = b"\x00\x01\x02\x03"
        # play_audio is blocking in MockAudioFrontend; run it in a thread so we
        # can observe the state and then stop it.
        t = threading.Thread(target=fe.play_audio, args=(audio,))
        t.daemon = True
        t.start()
        deadline = time.monotonic() + 0.5
        while not fe.is_playing and time.monotonic() < deadline:
            time.sleep(0.01)
        assert fe.is_playing is True
        assert fe._last_played_audio == audio
        fe.stop_playback()
        t.join(timeout=1.0)
        assert fe.is_playing is False

    def test_mock_stop_playback_clears_state(self):
        from tools.audio_frontend import MockAudioFrontend

        fe = MockAudioFrontend()
        fe._playing = True
        fe._last_played_audio = b"audio"
        fe.stop_playback()
        assert fe.is_playing is False
        assert fe._last_played_audio == b""


class TestAudioRecorderIntegration:
    def test_audio_recorder_works_with_mock_frontend(self):
        from tools.audio_frontend import MockAudioFrontend
        from tools.voice_mode import AudioRecorder

        frontend = MockAudioFrontend()
        recorder = AudioRecorder(frontend=frontend)
        assert recorder.vad_provider == "rms"
        assert recorder._frontend is frontend

    def test_mock_frontend_stop_playback_safe_when_not_playing(self):
        from tools.audio_frontend import MockAudioFrontend

        fe = MockAudioFrontend()
        fe.stop_playback()
        assert fe.is_playing is False


class TestAudioFrontendModuleConstants:
    def test_constants_are_sane(self):
        from tools.audio_frontend import CHANNELS, DTYPE, SAMPLE_RATE, SAMPLE_WIDTH

        assert SAMPLE_RATE == 16000
        assert CHANNELS == 1
        assert SAMPLE_WIDTH == 2
        assert DTYPE == "int16"
