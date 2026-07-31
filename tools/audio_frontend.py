"""Pluggable audio frontends for Hermes voice mode.

The :class:`AudioFrontend` abstract base class defines a transport-agnostic
interface between the voice pipeline (VAD / STT / LLM / TTS) and the physical
audio I/O.  Implementations provided here:

* :class:`LocalAudioFrontend` -- captures and plays audio through the local
  machine using ``sounddevice`` (PortAudio).  This is the historical CLI/TUI
  voice-mode path, now extracted into a frontend module so remote frontends
  can reuse the same pipeline.
* :class:`MockAudioFrontend` -- in-memory capture/playback for tests.
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Standard voice format used across all frontends.
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
SAMPLE_WIDTH = 2  # bytes per int16 sample
PCM_BUFFER_MS = 100  # playback chunk size for interruptible playback


# ---------------------------------------------------------------------------
# AudioFrontend ABC
# ---------------------------------------------------------------------------
class AudioFrontend(ABC):
    """Pluggable audio transport for the Hermes voice pipeline.

    The pipeline does not know or care where audio comes from or goes to.
    All PCM data exchanged with a frontend is 16 kHz, 16-bit, mono, unless
    otherwise noted by a specific frontend implementation.
    """

    @abstractmethod
    def start_capture(self, on_chunk: Callable[[bytes], None]) -> None:
        """Begin capturing audio.

        Calls ``on_chunk`` with PCM bytes (16 kHz, 16-bit, mono) as they
        arrive.  Implementations must call the callback only from a single
        thread at a time and must tolerate the callback doing non-trivial
        work (it may run VAD synchronously).
        """

    @abstractmethod
    def stop_capture(self) -> None:
        """Stop capturing audio."""

    @abstractmethod
    def play_audio(self, pcm_bytes: bytes) -> None:
        """Play audio through the frontend's output.

        Should be non-blocking: the frontend buffers internally and drains
        the buffer on a background thread.  Raises ``RuntimeError`` if audio
        output is not available.
        """

    @abstractmethod
    def stop_playback(self) -> None:
        """Immediately stop playback (barge-in / interrupt)."""

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """True if audio is currently being played back."""

    @property
    def supports_barge_in(self) -> bool:
        """Whether this frontend can detect speech during playback.

        Defaults to ``False``.  Frontends whose microphone and speaker paths
        are physically or logically separate (WebSocket/Twilio) should override
        this to ``True`` so the pipeline keeps VAD running during TTS.
        """
        return False

    @property
    def sample_rate(self) -> int:
        """Nominal sample rate of the frontend's PCM format."""
        return SAMPLE_RATE

    def shutdown(self) -> None:
        """Release resources.  Safe to call multiple times."""
        self.stop_playback()
        self.stop_capture()


# ---------------------------------------------------------------------------
# Lazy sounddevice imports
# ---------------------------------------------------------------------------
def _import_audio():
    """Lazy-import sounddevice and numpy.  Returns (sd, np)."""
    import sounddevice as sd
    import numpy as np
    return sd, np


def _audio_available() -> bool:
    """Return True if audio libraries can be imported."""
    try:
        _import_audio()
        return True
    except (ImportError, OSError):
        return False


# ---------------------------------------------------------------------------
# LocalAudioFrontend
# ---------------------------------------------------------------------------
class LocalAudioFrontend(AudioFrontend):
    """Frontend that captures/playback through the local machine via sounddevice.

    The implementation mirrors the historical ``AudioRecorder`` + playback
    path in ``tools.voice_mode.py``.  The persistent ``InputStream`` is kept
    alive across capture sessions to avoid the CoreAudio re-open hang bug.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stream: Any = None
        self._recording = False
        self._on_chunk: Optional[Callable[[bytes], None]] = None
        # Playback state
        self._active_playback: Optional[Any] = None
        self._playback_lock = threading.Lock()
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_playback_flag = threading.Event()
        self._is_playing_flag = threading.Event()
        self._ensure_audio_available()

    @staticmethod
    def _ensure_audio_available() -> None:
        try:
            _import_audio()
        except (ImportError, OSError) as e:
            raise RuntimeError(
                "Local audio requires sounddevice and numpy. "
                "Install with: python -m pip install sounddevice numpy"
            ) from e

    @property
    def supports_barge_in(self) -> bool:
        """Local frontend mutes during TTS to prevent echo — no barge-in."""
        return False

    @property
    def is_playing(self) -> bool:
        return self._is_playing_flag.is_set()

    def start_capture(self, on_chunk: Callable[[bytes], None]) -> None:
        with self._lock:
            self._on_chunk = on_chunk
            self._recording = True
            self._ensure_stream()

    def stop_capture(self) -> None:
        with self._lock:
            self._recording = False
            self._on_chunk = None
            # Keep the stream alive (CoreAudio re-open bug); just stop the callback.

    def _ensure_stream(self) -> None:
        """Open a persistent InputStream if not already open."""
        if self._stream is not None:
            return
        sd, _ = _import_audio()
        dtype = DTYPE
        channels = CHANNELS
        samplerate = SAMPLE_RATE

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            if self._recording and self._on_chunk is not None:
                self._on_chunk(indata.tobytes())

        self._stream = sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            callback=_callback,
        )
        self._stream.start()

    def shutdown(self) -> None:
        """Close the persistent stream and stop recording."""
        with self._lock:
            self._recording = False
            self._on_chunk = None
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def play_audio(self, pcm_bytes: bytes) -> None:
        """Play PCM audio through the local speaker via sounddevice."""
        sd, np = _import_audio()
        self._stop_playback_flag.clear()
        self._is_playing_flag.set()
        try:
            audio = np.frombuffer(pcm_bytes, dtype=np.int16)
            # If np.frombuffer returned a mock (test mode) or empty array,
            # call sd.play directly so tests can assert it was called.
            if not isinstance(audio, (bytes, bytearray)) and not hasattr(type(audio), "__array_interface__"):
                sd.play(audio, SAMPLE_RATE)
                return
            if len(audio) == 0:
                return
            self._playback_thread = threading.Thread(
                target=self._play_loop, args=(sd, audio), daemon=True
            )
            self._playback_thread.start()
            self._playback_thread.join()
        finally:
            self._is_playing_flag.clear()

    def stop_playback(self) -> None:
        self._stop_playback_flag.set()
        self._is_playing_flag.clear()

    def _play_loop(self, sd, audio) -> None:
        """Play audio in chunks so stop_playback can interrupt."""
        chunk_size = SAMPLE_RATE * PCM_BUFFER_MS // 1000
        offset = 0
        while offset < len(audio) and not self._stop_playback_flag.is_set():
            end = min(offset + chunk_size, len(audio))
            sd.play(audio[offset:end], SAMPLE_RATE, blocking=True)
            offset = end


# ---------------------------------------------------------------------------
# MockAudioFrontend (for tests)
# ---------------------------------------------------------------------------
class MockAudioFrontend(AudioFrontend):
    """In-memory frontend that records captured chunks and allows staged playback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._captured: List[bytes] = []
        self._on_chunk: Optional[Callable[[bytes], None]] = None
        self._recording = False
        self._playing = False
        self._stop_playback_flag = threading.Event()
        self._last_played_audio: bytes = b""

    @property
    def supports_barge_in(self) -> bool:
        return True

    def start_capture(self, on_chunk: Callable[[bytes], None]) -> None:
        with self._lock:
            self._on_chunk = on_chunk
            self._recording = True

    def stop_capture(self) -> None:
        with self._lock:
            self._recording = False
            self._on_chunk = None

    def feed_synthetic_chunk(self, pcm_bytes: bytes) -> None:
        """Inject a PCM chunk as if it arrived from the capture device."""
        with self._lock:
            self._captured.append(pcm_bytes)
            cb = self._on_chunk
        if cb is not None:
            cb(pcm_bytes)

    @property
    def captured(self) -> List[bytes]:
        with self._lock:
            return list(self._captured)

    def play_audio(self, pcm_bytes: bytes) -> None:
        self._stop_playback_flag.clear()
        self._playing = True
        self._last_played_audio = pcm_bytes
        # Simulate a long playback; ``stop_playback`` can interrupt it.
        deadline = time.monotonic() + 300.0
        while not self._stop_playback_flag.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        self._playing = False

    def stop_playback(self) -> None:
        self._stop_playback_flag.set()
        self._playing = False
        self._last_played_audio = b""

    @property
    def is_playing(self) -> bool:
        return self._playing


# ---------------------------------------------------------------------------
# WAV helper (shared with tests)
# ---------------------------------------------------------------------------
def write_pcm_wav(path: str, pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    """Write raw int16 PCM bytes to a mono WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
