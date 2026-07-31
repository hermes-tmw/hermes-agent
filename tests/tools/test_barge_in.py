"""Tests for barge-in (interruption) behavior in the voice pipeline.

These tests use MockAudioFrontend and a mocked LocalAudioFrontend to verify:
* Barge-in fires on remote frontends when speech is detected during playback.
* Local frontends (supports_barge_in=False) do not trigger barge-in.
* Barge-in calls frontend.stop_playback() and cancels ongoing generation.
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def loud_pcm():
    """Return a 400 ms chunk of loud 16-bit 16 kHz mono PCM."""
    sample_rate = 16000
    duration_s = 0.4
    n_frames = int(sample_rate * duration_s)
    samples = [8000 * (1 if i % 2 else -1) for i in range(n_frames)]
    return struct.pack(f"<{n_frames}h", *samples)


@pytest.fixture
def mock_sd(monkeypatch):
    """Mock sounddevice/numpy for tests that import the local frontend."""
    mock = MagicMock()
    mock_np = MagicMock()
    mock_np.int16 = "int16"
    mock_np.frombuffer.return_value = mock_np.array([0, 1, 2])
    mock_np.sqrt.return_value = 0.0
    mock.get_stream.return_value = None
    mock.InputStream.return_value = MagicMock(start=MagicMock())

    def _fake_import():
        return mock, mock_np

    monkeypatch.setattr("tools.audio_frontend._import_audio", _fake_import)
    monkeypatch.setattr("tools.audio_frontend._audio_available", lambda: True)
    monkeypatch.setattr("tools.voice_mode._import_audio", _fake_import)
    monkeypatch.setattr("tools.voice_mode._audio_available", lambda: True)
    return mock, mock_np


# ============================================================================
# BargeInHandler unit tests
# ============================================================================
class TestBargeInHandler:
    def test_barge_in_fires_for_remote_frontend(self, loud_pcm):
        from tools.audio_frontend import MockAudioFrontend
        from tools.vad import RMSVAD
        from tools.voice_mode import BargeInHandler

        frontend = MockAudioFrontend()
        vad = RMSVAD(threshold=200, silence_duration=3.0, min_speech_duration=0.05)
        called = threading.Event()

        handler = BargeInHandler(
            frontend=frontend,
            vad=vad,
            config={"voice": {"barge_in": True}},
            on_barge_in=called.set,
        )

        # Simulate TTS playback in a thread so is_playing is True.
        # Simulate TTS playback in a thread so is_playing is True.
        t = threading.Thread(target=frontend.play_audio, args=(b"\x00" * 16000,))
        t.daemon = True
        t.start()
        deadline = time.monotonic() + 1.0
        while not frontend.is_playing and time.monotonic() < deadline:
            time.sleep(0.01)

        handler.reset()
        # Feed loud audio for > 300 ms; use several chunks so sustained speech
        # exceeds the default min_speech_ms.
        chunk_size = 16000 * 2 * 100 // 1000  # 100 ms of int16 mono
        for _ in range(10):
            handler.feed_chunk(loud_pcm[:chunk_size])
            time.sleep(0.02)

        assert called.is_set(), "barge-in callback should fire on remote frontend"
        assert handler.cancelled.is_set()
        frontend.stop_playback()
        t.join(timeout=1.0)

    def test_barge_in_skipped_for_local_frontend(self, mock_sd, loud_pcm):
        from tools.audio_frontend import LocalAudioFrontend
        from tools.vad import RMSVAD
        from tools.voice_mode import BargeInHandler

        frontend = LocalAudioFrontend()
        vad = RMSVAD(threshold=200, silence_duration=3.0, min_speech_duration=0.05)
        called = threading.Event()

        handler = BargeInHandler(
            frontend=frontend,
            vad=vad,
            config={"voice": {"barge_in": True}},
            on_barge_in=called.set,
        )

        handler.reset()
        # Even if the frontend happens to report is_playing True, supports_barge_in
        # is False so barge-in should never fire.
        frontend._is_playing_flag.set()
        for _ in range(10):
            handler.feed_chunk(loud_pcm[:3200])
            time.sleep(0.02)

        assert not called.is_set(), "local frontend must not trigger barge-in"
        assert not handler.cancelled.is_set()

    def test_barge_in_disabled_in_config(self, loud_pcm):
        from tools.audio_frontend import MockAudioFrontend
        from tools.vad import RMSVAD
        from tools.voice_mode import BargeInHandler

        frontend = MockAudioFrontend()
        vad = RMSVAD(threshold=200, silence_duration=3.0, min_speech_duration=0.05)
        called = threading.Event()

        handler = BargeInHandler(
            frontend=frontend,
            vad=vad,
            config={"voice": {"barge_in": False}},
            on_barge_in=called.set,
        )

        # Simulate TTS playback in a thread so is_playing is True.
        t = threading.Thread(target=frontend.play_audio, args=(b"\x00" * 16000,))
        t.daemon = True
        t.start()
        deadline = time.monotonic() + 1.0
        while not frontend.is_playing and time.monotonic() < deadline:
            time.sleep(0.01)

        handler.reset()
        for _ in range(10):
            handler.feed_chunk(loud_pcm[:3200])
            time.sleep(0.02)

        assert not called.is_set(), "disabled barge-in should not fire"
        frontend.stop_playback()
        t.join(timeout=1.0)

    def test_barge_in_calls_stop_playback(self, loud_pcm):
        from tools.audio_frontend import MockAudioFrontend
        from tools.vad import RMSVAD
        from tools.voice_mode import BargeInHandler

        frontend = MockAudioFrontend()
        vad = RMSVAD(threshold=200, silence_duration=3.0, min_speech_duration=0.05)

        def on_barge_in():
            frontend.stop_playback()

        handler = BargeInHandler(
            frontend=frontend,
            vad=vad,
            config={"voice": {"barge_in": True}},
            on_barge_in=on_barge_in,
        )

        t = threading.Thread(target=frontend.play_audio, args=(b"\x00" * 16000,))
        t.daemon = True
        t.start()
        deadline = time.monotonic() + 1.0
        while not frontend.is_playing and time.monotonic() < deadline:
            time.sleep(0.01)

        handler.reset()
        for _ in range(10):
            handler.feed_chunk(loud_pcm[:3200])
            time.sleep(0.02)

        deadline = time.monotonic() + 1.0
        while frontend.is_playing and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not frontend.is_playing, "stop_playback should halt playback"
        t.join(timeout=1.0)

    def test_barge_in_cancels_generation_token(self, loud_pcm):
        from tools.audio_frontend import MockAudioFrontend
        from tools.vad import RMSVAD
        from tools.voice_mode import BargeInHandler

        frontend = MockAudioFrontend()
        vad = RMSVAD(threshold=200, silence_duration=3.0, min_speech_duration=0.05)
        token = MagicMock()

        handler = BargeInHandler(
            frontend=frontend,
            vad=vad,
            config={"voice": {"barge_in": True}},
            on_barge_in=token.cancel,
        )

        t = threading.Thread(target=frontend.play_audio, args=(b"\x00" * 16000,))
        t.daemon = True
        t.start()
        deadline = time.monotonic() + 1.0
        while not frontend.is_playing and time.monotonic() < deadline:
            time.sleep(0.01)

        handler.reset()
        for _ in range(10):
            handler.feed_chunk(loud_pcm[:3200])
            time.sleep(0.02)

        token.cancel.assert_called()
        frontend.stop_playback()
        t.join(timeout=1.0)


# ============================================================================
# Config helpers
# ============================================================================
class TestBargeInConfig:
    def test_barge_in_enabled_default(self):
        from tools.voice_mode import barge_in_enabled

        assert barge_in_enabled({}) is True
        assert barge_in_enabled({"voice": {}}) is True

    def test_barge_in_enabled_false(self):
        from tools.voice_mode import barge_in_enabled

        assert barge_in_enabled({"voice": {"barge_in": False}}) is False
        assert barge_in_enabled({"voice": {"barge_in": "false"}}) is False

    def test_barge_in_threshold_default(self):
        from tools.voice_mode import barge_in_threshold

        assert barge_in_threshold({}) == 0.5

    def test_barge_in_threshold_clamped(self):
        from tools.voice_mode import barge_in_threshold

        assert barge_in_threshold({"voice": {"barge_in": {"threshold": 1.5}}}) == 1.0
        assert barge_in_threshold({"voice": {"barge_in": {"threshold": -0.2}}}) == 0.0


# ============================================================================
# VAD has_speech property
# ============================================================================
class TestVADHasSpeech:
    def test_rms_has_speech_after_confirmed(self):
        from tools.vad import RMSVAD

        vad = RMSVAD(threshold=200, silence_duration=3.0, min_speech_duration=0.05)
        vad.reset(0.0)
        assert not vad.has_speech
        # Speech must be sustained for min_speech_duration (0.05s) before has_speech flips.
        vad.update(500, 0.0)
        vad.update(500, 0.02)
        vad.update(500, 0.04)
        assert not vad.has_speech
        vad.update(500, 0.08)
        assert vad.has_speech

    def test_silero_has_speech_tracks_state(self):
        from tools.vad import SileroVAD

        vad = SileroVAD(model_path="/dev/null/nonexistent.onnx")
        # Bypass model loading; set internal state directly.
        vad._has_speech = True
        assert vad.has_speech


# ============================================================================
# WebSocket frontend barge-in support
# ============================================================================
class TestWebSocketFrontendBargeIn:
    def test_web_socket_frontend_supports_barge_in(self):
        from gateway.voice_ws_handler import WebSocketAudioFrontend

        class FakeWS:
            pass

        fe = WebSocketAudioFrontend(FakeWS(), None)
        assert fe.supports_barge_in is True

    def test_web_socket_playback_is_interruptible(self):
        import asyncio
        from unittest.mock import AsyncMock

        from gateway.voice_ws_handler import WebSocketAudioFrontend

        ws = AsyncMock()

        async def _run():
            loop = asyncio.get_running_loop()
            fe = WebSocketAudioFrontend(ws, loop)

            async def play():
                await fe.play_audio(b"\x00" * 16000 * 2)

            t = threading.Thread(target=lambda: asyncio.run(play()))
            t.daemon = True
            t.start()
            deadline = time.monotonic() + 1.0
            while not fe.is_playing and time.monotonic() < deadline:
                time.sleep(0.01)
            fe.stop_playback()
            t.join(timeout=1.0)
            assert not fe.is_playing

        asyncio.run(_run())


# ============================================================================
# Local frontend no barge-in support
# ============================================================================
class TestLocalFrontendNoBargeIn:
    def test_local_frontend_supports_barge_in_is_false(self, mock_sd):
        from tools.audio_frontend import LocalAudioFrontend

        fe = LocalAudioFrontend()
        assert fe.supports_barge_in is False
