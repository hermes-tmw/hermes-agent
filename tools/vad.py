"""Voice Activity Detection (VAD) implementations for Hermes voice mode.

This module is a companion to ``tools.voice_mode``.  It intentionally stays
separate so that the ONNX + numpy code can be imported lazily and failures are
isolated from the core recorder path.  The public surface is tiny:

* ``SileroVAD`` — neural voice activity detector using the Silero VAD ONNX model.
* ``RMSVAD`` — backward-compatible energy-based detector (the historical default).
* ``load_vad_from_config(config_dict)`` — factory that picks the provider,
  applies defaults, and falls back to RMS if Silero is unavailable.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default model cache layout
# ---------------------------------------------------------------------------
DEFAULT_SILERO_MODEL_PATH = Path.home() / ".hermes" / "cache" / "silero-vad" / "silero_vad.onnx"


def _default_silero_model_path() -> str:
    """Return the canonical Silero VAD model path.

    The model is expected to have been downloaded to ``~/.hermes/cache/silero-vad/``
    before use (the task spec says it is installed in parallel).  Callers get a
    clear warning + RMS fallback when the file is missing.
    """
    return os.environ.get("HERMES_SILERO_VAD_MODEL", str(DEFAULT_SILERO_MODEL_PATH))


# ---------------------------------------------------------------------------
# VAD configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VADConfig:
    """Normalized voice-activity-detection settings used by ``AudioRecorder``."""

    provider: str  # "silero" or "rms"
    threshold: float  # speech probability (Silero) or RMS level (RMS)
    min_speech_ms: int
    min_silence_ms: int
    model_path: Optional[str]

    def __post_init__(self) -> None:
        if self.provider not in {"silero", "rms"}:
            raise ValueError(f"Unknown VAD provider {self.provider!r}")


@dataclass(frozen=True)
class _SileroParams:
    """Silero-specific runtime parameters extracted from ``VADConfig``."""

    threshold: float
    min_speech_ms: int
    min_silence_ms: int


@dataclass(frozen=True)
class _RMSParams:
    """RMS-specific runtime parameters used when Silero is unavailable."""

    threshold: int
    silence_duration_s: float
    min_speech_duration_s: float = 0.3
    max_dip_tolerance_s: float = 0.3


# ---------------------------------------------------------------------------
# Config parsing helpers
# ---------------------------------------------------------------------------
def _voice_cfg_dict(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Shape-safe accessor for the nested ``voice.vad`` block.

    Mirrors the defensive style in ``hermes_cli.voice`` and ``tui_gateway.server``:
    malformed scalar/list/None shapes collapse to an empty dict so callers can
    use ``.get("...")`` without AttributeError.
    """
    if not isinstance(config, dict):
        return {}
    voice = config.get("voice")
    if not isinstance(voice, dict):
        return {}
    vad = voice.get("vad")
    return vad if isinstance(vad, dict) else {}


def _safe_float(value: Any, default: float, lo: float, hi: float) -> float:
    """Return a bounded float, falling back to ``default`` for invalid values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(lo, min(hi, float(value)))


def _safe_int(value: Any, default: int, lo: int, hi: int) -> int:
    """Return a bounded int, falling back to ``default`` for invalid values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(lo, min(hi, int(value)))


def _safe_str(value: Any) -> Optional[str]:
    """Return a non-empty string or ``None``."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def vad_config_from_dict(config: Optional[Dict[str, Any]]) -> VADConfig:
    """Build a ``VADConfig`` from the loaded Hermes config dict.

    Defaults preserve backward compatibility: if ``voice.vad.provider`` is
    absent or unrecognized, we use RMS with the historical ``silence_threshold``
    and ``silence_duration`` values from ``DEFAULT_CONFIG``.
    """
    vad_cfg = _voice_cfg_dict(config)
    provider = "rms"
    raw_provider = vad_cfg.get("provider")
    if isinstance(raw_provider, str) and raw_provider.lower().strip() in {"silero", "rms"}:
        provider = raw_provider.lower().strip()

    # Silero block defaults
    silero_block = vad_cfg.get("silero")
    if not isinstance(silero_block, dict):
        silero_block = {}

    silero_threshold = _safe_float(silero_block.get("threshold"), 0.5, 0.0, 1.0)
    silero_min_speech_ms = _safe_int(silero_block.get("min_speech_ms"), 300, 0, 5000)
    silero_min_silence_ms = _safe_int(silero_block.get("min_silence_ms"), 700, 0, 10000)
    silero_model_path = _safe_str(silero_block.get("model_path")) or _default_silero_model_path()

    # RMS block defaults (also read legacy top-level voice keys for back-compat)
    rms_block = vad_cfg.get("rms")
    if not isinstance(rms_block, dict):
        rms_block = {}

    voice_cfg = {}
    if isinstance(config, dict):
        voice = config.get("voice")
        if isinstance(voice, dict):
            voice_cfg = voice

    rms_threshold = _safe_int(
        rms_block.get("threshold") if "threshold" in rms_block else voice_cfg.get("silence_threshold"),
        200, 0, 32767,
    )
    rms_duration = _safe_float(
        rms_block.get("silence_duration") if "silence_duration" in rms_block else voice_cfg.get("silence_duration"),
        3.0, 0.1, 60.0,
    )

    if provider == "silero":
        return VADConfig(
            provider="silero",
            threshold=silero_threshold,
            min_speech_ms=silero_min_speech_ms,
            min_silence_ms=silero_min_silence_ms,
            model_path=silero_model_path,
        )

    return VADConfig(
        provider="rms",
        threshold=float(rms_threshold),
        min_speech_ms=300,
        min_silence_ms=int(rms_duration * 1000),
        model_path=None,
    )


# ---------------------------------------------------------------------------
# VAD base helpers
# ---------------------------------------------------------------------------
def _resample_int16_to_float32_16khz(audio: Any, sample_rate: int) -> Any:
    """Convert int16 PCM to float32 normalized [-1, 1] at 16 kHz.

    If ``audio`` is already float32 at 16 kHz it is returned unchanged.
    Resampling uses a simple linear interpolation; quality is adequate for VAD.
    """
    import numpy as np

    arr = np.asarray(audio)
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    else:
        arr = arr.astype(np.float32)

    # Flatten if necessary (InputStream gives shape (frames, channels)).
    if arr.ndim > 1:
        arr = arr.reshape(-1)

    if sample_rate == 16000:
        return arr

    # Linear resample to 16 kHz.
    target_len = int(round(len(arr) * 16000 / sample_rate))
    if target_len == 0:
        return np.zeros(0, dtype=np.float32)
    old_indices = np.linspace(0, len(arr) - 1, target_len)
    indices = old_indices.astype(np.int32)
    frac = old_indices - indices
    next_indices = np.minimum(indices + 1, len(arr) - 1)
    return arr[indices] * (1.0 - frac) + arr[next_indices] * frac


# ---------------------------------------------------------------------------
# Silero VAD
# ---------------------------------------------------------------------------
class SileroVAD:
    """Neural voice activity detector using the Silero VAD ONNX model.

    The model consumes non-overlapping 512-sample chunks (32 ms at 16 kHz).
    Each forward pass updates an internal LSTM state, so the same instance must
    be used across the lifetime of a single utterance.

    The public ``detect_endpoint(audio_buffer, sample_rate)`` method runs the
    model over ``audio_buffer`` and returns ``True`` when speech has started
    and then been followed by ``min_silence_ms`` of silence.
    """

    # Model window size in samples at 16 kHz.
    WINDOW_SAMPLES = 512

    def __init__(
        self,
        model_path: str,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 300,
        min_silence_ms: int = 700,
    ) -> None:
        self._model_path = model_path
        self._threshold = float(threshold)
        self._min_speech_ms = int(min_speech_ms)
        self._min_silence_ms = int(min_silence_ms)

        # Lazy state — created on first use so import failures happen inside
        # the factory, not during ``AudioRecorder`` construction.
        self._session: Any = None
        self._state: Any = None
        self._sample_rate_arr: Any = None

        # Endpoint tracking
        self._has_speech = False
        self._speech_start_ms: float = 0.0
        self._silence_start_ms: Optional[float] = None
        self._ms_processed: float = 0.0
        self._last_prob: float = 0.0

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def has_speech(self) -> bool:
        """True once speech has been confirmed for the current utterance."""
        return self._has_speech

    def _ensure_session(self) -> None:
        """Create the ONNX InferenceSession on first use."""
        if self._session is not None:
            return
        import numpy as np
        import onnxruntime as ort

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"Silero VAD model not found at {self._model_path}. "
                "Run the Silero VAD install step or set HERMES_SILERO_VAD_MODEL."
            )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            self._model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sample_rate_arr = np.array(16000, dtype=np.int64)

    def reset(self) -> None:
        """Clear endpoint state and LSTM hidden state for a new utterance."""
        self._has_speech = False
        self._speech_start_ms = 0.0
        self._silence_start_ms = None
        self._ms_processed = 0.0
        self._last_prob = 0.0
        if self._state is not None:
            import numpy as np

            self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def _consume_chunk(self, chunk: Any) -> float:
        """Run a single 512-sample 16 kHz float32 chunk through the model."""
        import numpy as np

        self._ensure_session()
        assert self._session is not None
        # ONNX model expects shape (batch=1, samples=512).
        if chunk.shape[0] != 1 or chunk.shape[1] != self.WINDOW_SAMPLES:
            raise ValueError(
                f"Silero VAD chunk must have shape (1, {self.WINDOW_SAMPLES}), got {chunk.shape}"
            )
        out = self._session.run(
            None,
            {
                "input": chunk,
                "state": self._state,
                "sr": self._sample_rate_arr,
            },
        )
        self._state = out[1]
        prob = float(out[0].item())
        self._last_prob = prob
        return prob

    def _process_window(self, window: Any) -> bool:
        """Process one resampled window and update endpoint state.

        Returns True if an endpoint has been detected.
        """
        import numpy as np

        chunk = window.reshape(1, -1).astype(np.float32)
        prob = self._consume_chunk(chunk)
        window_ms = self.WINDOW_SAMPLES / 16.0  # 512 / 16000 * 1000
        self._ms_processed += window_ms

        is_speech = prob > self._threshold
        if is_speech:
            self._silence_start_ms = None
            if not self._has_speech:
                if self._speech_start_ms == 0.0:
                    self._speech_start_ms = self._ms_processed - window_ms
                # Confirm speech only after sustained speech.
                if (self._ms_processed - self._speech_start_ms) >= self._min_speech_ms:
                    self._has_speech = True
                    logger.debug("Silero speech confirmed (%.0f ms)", self._ms_processed)
        else:
            if self._has_speech:
                if self._silence_start_ms is None:
                    self._silence_start_ms = self._ms_processed
                elif (self._ms_processed - self._silence_start_ms) >= self._min_silence_ms:
                    logger.debug("Silero endpoint detected at %.0f ms", self._ms_processed)
                    return True
            else:
                # No speech yet; reset the speech_start tracker after a short
                # silence so random pops don't accumulate into a phantom utterance.
                self._speech_start_ms = 0.0

        return False

    def detect_endpoint(self, audio_buffer: Any, sample_rate: int = 16000) -> bool:
        """Process ``audio_buffer`` and return True when speech ends.

        ``audio_buffer`` may be int16 or float32, mono, at any sample rate.
        Each call updates the internal LSTM state; for a single utterance the
        same ``SileroVAD`` instance should be used until endpoint is found.
        """
        import numpy as np

        self._ensure_session()
        audio = _resample_int16_to_float32_16khz(audio_buffer, sample_rate)
        if len(audio) == 0:
            return False

        # Pad to a multiple of WINDOW_SAMPLES if needed.
        n = len(audio)
        pad = (self.WINDOW_SAMPLES - (n % self.WINDOW_SAMPLES)) % self.WINDOW_SAMPLES
        if pad:
            audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])

        for i in range(0, len(audio), self.WINDOW_SAMPLES):
            window = audio[i : i + self.WINDOW_SAMPLES]
            if self._process_window(window):
                return True
        return False


# ---------------------------------------------------------------------------
# RMS VAD (backward-compatible)
# ---------------------------------------------------------------------------
class RMSVAD:
    """Energy-based voice activity detector matching the historical behavior."""

    def __init__(
        self,
        *,
        threshold: int = 200,
        silence_duration: float = 3.0,
        min_speech_duration: float = 0.3,
        max_dip_tolerance: float = 0.3,
    ) -> None:
        self._threshold = int(threshold)
        self._silence_duration = float(silence_duration)
        self._min_speech_duration = float(min_speech_duration)
        self._max_dip_tolerance = float(max_dip_tolerance)

        # State (reset per utterance)
        self._has_spoken = False
        self._speech_start: float = 0.0
        self._dip_start: float = 0.0
        self._silence_start: float = 0.0
        self._resume_start: float = 0.0
        self._resume_dip_start: float = 0.0
        self._start_time: float = 0.0

    @property
    def has_speech(self) -> bool:
        """True once speech has been confirmed for the current utterance."""
        return self._has_spoken

    def reset(self, start_time: Optional[float] = None) -> None:
        """Reset all tracking state for a new utterance."""
        self._has_spoken = False
        self._speech_start = 0.0
        self._dip_start = 0.0
        self._silence_start = 0.0
        self._resume_start = 0.0
        self._resume_dip_start = 0.0
        self._start_time = start_time if start_time is not None else time.monotonic()

    def update(self, rms: int, now: Optional[float] = None) -> bool:
        """Feed one RMS observation and return True if an endpoint is reached."""
        if now is None:
            now = time.monotonic()
        elapsed = now - self._start_time

        if rms > self._threshold:
            self._dip_start = 0.0
            if self._speech_start == 0.0:
                self._speech_start = now
            elif not self._has_spoken and (now - self._speech_start) >= self._min_speech_duration:
                self._has_spoken = True

            if not self._has_spoken:
                self._silence_start = 0.0
            else:
                self._resume_dip_start = 0.0
                if self._resume_start == 0.0:
                    self._resume_start = now
                elif (now - self._resume_start) >= self._min_speech_duration:
                    self._silence_start = 0.0
                    self._resume_start = 0.0
        elif self._has_spoken:
            if self._resume_start > 0:
                if self._resume_dip_start == 0.0:
                    self._resume_dip_start = now
                elif (now - self._resume_dip_start) >= self._max_dip_tolerance:
                    self._resume_start = 0.0
                    self._resume_dip_start = 0.0
        elif self._speech_start > 0:
            if self._dip_start == 0.0:
                self._dip_start = now
            elif (now - self._dip_start) >= self._max_dip_tolerance:
                self._speech_start = 0.0
                self._dip_start = 0.0

        if self._has_spoken and rms <= self._threshold:
            if self._silence_start == 0.0:
                self._silence_start = now
            elif (now - self._silence_start) >= self._silence_duration:
                return True
        elif not self._has_spoken and elapsed >= 15.0:
            # Historical max_wait guard lives in AudioRecorder, but we mirror
            # a sane fallback here so standalone RMSVAD callers get it too.
            return True

        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def load_vad_from_config(
    config: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Return ``(vad_instance, config, is_fallback)`` for the given config.

    * ``vad_instance`` is either a ``SileroVAD`` or ``RMSVAD``.
    * ``config`` is the normalized ``VADConfig``.
    * ``is_fallback`` is True when Silero was requested but unavailable.
    """
    cfg = vad_config_from_dict(config)
    if cfg.provider == "silero":
        try:
            vad = SileroVAD(
                model_path=cfg.model_path or _default_silero_model_path(),
                threshold=cfg.threshold,
                min_speech_ms=cfg.min_speech_ms,
                min_silence_ms=cfg.min_silence_ms,
            )
            # Touch the model file/ONNX session once so any failure surfaces now.
            vad._ensure_session()
            return vad, cfg, False
        except Exception as e:
            logger.warning(
                "Silero VAD unavailable (%s), falling back to RMS VAD", e
            )
            # Fall through to RMS.

    rms_config = cfg if cfg.provider == "rms" else vad_config_from_dict({})
    # If we are falling back from a Silero request, prefer the legacy RMS
    # defaults from the config dict when present.
    rms_cfg = vad_config_from_dict(config)
    rms_params = _RMSParams(
        threshold=int(rms_cfg.threshold),
        silence_duration_s=rms_cfg.min_silence_ms / 1000.0,
    )
    # is_fallback is True only when we fell back from a Silero request,
    # not when RMS was the default/explicit choice.
    return RMSVAD(
        threshold=rms_params.threshold,
        silence_duration=rms_params.silence_duration_s,
    ), cfg, cfg.provider == "silero"


# ---------------------------------------------------------------------------
# Backward-compatible helpers kept for callers that import voice_mode only
# ---------------------------------------------------------------------------
def _silero_vad_available(model_path: Optional[str] = None) -> bool:
    """Return True if the Silero model file and onnxruntime are loadable."""
    path = model_path or _default_silero_model_path()
    if not os.path.isfile(path):
        return False
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True
