"""WebSocket voice session handler for the API server gateway.

This module implements the server side of the Hermes voice WebSocket protocol:

* Accept one concurrent voice session per adapter instance.
* Require API-key auth plus optional Tailscale source-IP gating.
* Stream 16 kHz / 16-bit / mono PCM from the client through VAD.
* On endpoint detection: transcribe the utterance, run it through the agent,
  synthesize the response to PCM, and stream the audio back to the client.
* Emit JSON status/transcript messages so the client can render UI state.

It is intentionally isolated from ``api_server.py`` so that test environments
which do not need voice support can import the adapter without pulling in heavy
voice dependencies.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.platforms.api_server import APIServerAdapter  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

# Audio format shared with ``tools.audio_frontend``.
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # int16
PCM_BUFFER_MS = 100  # playback chunk size for interruptible playback
CHANNELS = 1
PCM_BUFFER_MS = 100  # playback chunk size for interruptible WebSocket streaming


class CancelToken:
    """Simple cancellation token shared between the generation task and barge-in."""

    def __init__(self) -> None:
        self._cancelled = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled


class WebSocketAudioFrontend:
    """AudioFrontend-compatible bridge that feeds a WebSocket receive loop."""

    sample_rate = SAMPLE_RATE

    def __init__(self, ws: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._ws = ws
        self._loop = loop
        self._on_chunk: Optional[Any] = None
        self._lock = asyncio.Lock()
        self._playing = False
        self._stop_playback_flag = asyncio.Event()

    @property
    def supports_barge_in(self) -> bool:
        return True

    def start_capture(self, on_chunk) -> None:
        self._on_chunk = on_chunk

    def stop_capture(self) -> None:
        self._on_chunk = None

    @property
    def is_playing(self) -> bool:
        return self._playing

    async def _feed_chunk(self, pcm_bytes: bytes) -> None:
        cb = self._on_chunk
        if cb is not None:
            if asyncio.iscoroutinefunction(cb):
                await cb(pcm_bytes)
            else:
                await self._loop.run_in_executor(None, cb, pcm_bytes)

    async def play_audio(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        self._playing = True
        self._stop_playback_flag.clear()
        try:
            chunk_size = SAMPLE_RATE * SAMPLE_WIDTH * PCM_BUFFER_MS // 1000
            offset = 0
            while offset < len(pcm_bytes):
                if self._stop_playback_flag.is_set():
                    break
                end = min(offset + chunk_size, len(pcm_bytes))
                chunk = pcm_bytes[offset:end]
                if asyncio.iscoroutinefunction(self._ws.send_bytes):
                    await self._ws.send_bytes(chunk)
                else:
                    self._ws.send_bytes(chunk)
                offset = end
        finally:
            self._playing = False

    def stop_playback(self) -> None:
        self._stop_playback_flag.set()
        self._playing = False

    def shutdown(self) -> None:
        self.stop_capture()
        self.stop_playback()

class VoiceSession:
    """One WebSocket-backed voice conversation."""

    def __init__(
        self,
        connection_id: str,
        ws: Any,
        adapter: Any,
        config: Dict[str, Any],
    ) -> None:
        self.connection_id = connection_id
        self._ws = ws
        self._adapter = adapter
        self._config = config
        self._loop = asyncio.get_event_loop()
        self._frontend = WebSocketAudioFrontend(ws, self._loop)
        self._vad: Any = None
        self._recorder: Any = None
        self._frames: List[bytes] = []
        self._vad_lock = asyncio.Lock()
        self._closed = False
        self._stt_config: Dict[str, Any] = {}
        self._tts_config: Dict[str, Any] = {}
        self._barge_in: Any = None
        self._gen_token: Any = None

    async def send_json(self, payload: Dict[str, Any]) -> None:
        """Send a JSON text message to the client."""
        try:
            if self._closed:
                return
            if asyncio.iscoroutinefunction(self._ws.send_str):
                await self._ws.send_str(json.dumps(payload))
            else:
                self._ws.send_str(json.dumps(payload))
        except Exception as e:
            logger.debug("Voice session %s send_json failed: %s", self.connection_id, e)

    async def run(self) -> None:
        """Main loop: receive messages, route to VAD, and handle endpoint."""
        from tools.audio_frontend import MockAudioFrontend

        if self._closed:
            return

        # Initialize VAD lazily; fall back to RMS if Silero is unavailable.
        self._vad, self._vad_config, _ = self._load_vad()

        # Use MockAudioFrontend as the capture-side stand-in because the actual
        # PCM arrives over the WebSocket, not from a local sounddevice. The
        # AudioRecorder only needs *a* frontend with start_capture / stop_capture
        # semantics; MockAudioFrontend is lightweight and barge-in capable.
        frontend = MockAudioFrontend()
        frontend.start_capture(self._on_chunk)
        self._recorder = self._create_recorder(frontend)

        await self.send_json({
            "type": "status",
            "status": "ready",
            "vad": getattr(self._vad, "__class__", object).__name__,
        })

        try:
            async for msg in self._ws:
                if self._closed:
                    break
                if msg.type == 1:  # WSMsgType.TEXT
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    await self._handle_text_message(data)
                elif msg.type == 2:  # WSMsgType.BINARY
                    await self._handle_binary_message(msg.data)
                elif msg.type in (257, 258):  # WSMsgType.CLOSE / CLOSING
                    break
        except Exception as e:
            logger.warning("Voice session %s receive loop ended: %s", self.connection_id, e)
        finally:
            await self.close()

    def _create_recorder(self, frontend: Any) -> Any:
        from tools.voice_mode import AudioRecorder
        return AudioRecorder(config=self._config, frontend=frontend)

    def _load_vad(self) -> tuple:
        from tools.vad import load_vad_from_config
        return load_vad_from_config(self._config)

    def _on_chunk(self, pcm_bytes: bytes) -> None:
        """Called by MockAudioFrontend on each captured chunk."""
        self._frames.append(pcm_bytes)
        if self._barge_in is not None:
            self._barge_in.feed_chunk(pcm_bytes)
        if self._vad is None:
            return
        try:
            import numpy as np
            arr = np.frombuffer(pcm_bytes, dtype=np.int16)
            cls_name = type(self._vad).__name__
            if cls_name == "SileroVAD":
                endpoint = self._vad.detect_endpoint(arr.copy(), SAMPLE_RATE)
            elif cls_name == "RMSVAD":
                rms = int(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
                endpoint = self._vad.update(rms)
            else:
                endpoint = False
            if endpoint:
                # Schedule pipeline on the event loop without blocking the
                # frontend capture thread.
                asyncio.run_coroutine_threadsafe(self._on_endpoint(), self._loop)
        except Exception as e:
            logger.debug("VAD chunk failed in session %s: %s", self.connection_id, e)

    def _install_barge_in(self) -> None:
        """Create a fresh barge-in detector for the upcoming playback."""
        from tools.voice_mode import BargeInHandler

        if not getattr(self._frontend, "supports_barge_in", False):
            return
        if self._vad is None:
            return
        self._barge_in = BargeInHandler(
            frontend=self._frontend,
            vad=self._vad,
            config=self._config,
            on_barge_in=self._on_barge_in,
        )
        self._barge_in.reset()

    def _on_barge_in(self) -> None:
        """Interrupt playback and cancel generation when barge-in fires."""
        try:
            self._frontend.stop_playback()
        except Exception as e:
            logger.debug("stop_playback in barge-in failed: %s", e)
        if self._gen_token is not None:
            self._gen_token.cancel()
        try:
            asyncio.run_coroutine_threadsafe(self._restart_turn(), self._loop)
        except Exception as e:
            logger.debug("Barge-in restart scheduling failed: %s", e)

    async def _restart_turn(self) -> None:
        """Reset VAD and start a new STT cycle with the buffered audio."""
        if self._closed:
            return
        self._clear_vad_after_endpoint()
        if self._barge_in is not None:
            self._barge_in.reset()
        await self.send_json({"type": "status", "status": "interrupted"})
        await self._on_endpoint()

    async def _handle_text_message(self, data: Dict[str, Any]) -> None:
        """Handle control / ping messages from the client."""
        msg_type = data.get("type")
        if msg_type == "ping":
            await self.send_json({"type": "pong", "ts": data.get("ts")})
        elif msg_type == "stop_playback":
            self._frontend.stop_playback()
            await self.send_json({"type": "status", "status": "playback_stopped"})
        else:
            logger.debug("Unknown voice text message type %r", msg_type)

    async def _handle_binary_message(self, pcm_bytes: bytes) -> None:
        """Feed a PCM chunk into the capture pipeline."""
        if self._recorder is None:
            return
        try:
            # Inject the chunk as if it came from the local microphone.
            self._recorder._frontend.feed_synthetic_chunk(pcm_bytes)
        except Exception as e:
            logger.debug("Failed to feed PCM chunk in session %s: %s", self.connection_id, e)

    def _clear_vad_after_endpoint(self) -> None:
        """Reset VAD state so the next utterance starts fresh."""
        if self._vad is None:
            return
        try:
            if hasattr(self._vad, "reset"):
                self._vad.reset()
        except Exception as e:
            logger.debug("VAD reset failed: %s", e)

    async def _on_endpoint(self) -> None:
        """Transcribe, run agent, and stream TTS when VAD endpoint fires."""
        async with self._vad_lock:
            if self._closed:
                return
            frames = self._frames
            self._frames = []
            if not frames:
                return
            pcm = b"".join(frames)
            wav_path = self._write_wav(pcm)
            if not wav_path:
                return

        self._clear_vad_after_endpoint()

        await self.send_json({"type": "status", "status": "transcribing"})
        transcript_result = await self._transcribe(wav_path)
        transcript = transcript_result.get("transcript", "").strip()

        await self.send_json({
            "type": "transcript",
            "speaker": "you",
            "text": transcript,
            "ts": self._format_ts(),
        })

        if not transcript:
            os.unlink(wav_path)
            return

        await self.send_json({"type": "status", "status": "thinking"})
        self._gen_token = CancelToken()
        response_text = await self._run_agent(transcript, self._gen_token)

        await self.send_json({
            "type": "transcript",
            "speaker": "hermes",
            "text": response_text,
            "ts": self._format_ts(),
        })

        await self.send_json({"type": "status", "status": "speaking"})
        self._install_barge_in()
        await self._synthesize_and_stream(response_text)

        try:
            os.unlink(wav_path)
        except OSError:
            pass

    async def _transcribe(self, wav_path: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        from tools.transcription_tools import transcribe_audio
        try:
            return await loop.run_in_executor(None, transcribe_audio, wav_path)
        except Exception as e:
            logger.error("Transcription failed in voice session %s: %s", self.connection_id, e)
            return {"success": False, "transcript": "", "error": str(e)}

    async def _run_agent(self, transcript: str, token: CancelToken) -> str:
        loop = asyncio.get_running_loop()
        try:
            result, _ = await self._adapter._run_agent(
                user_message=transcript,
                conversation_history=[],
                session_id=f"voice:{self.connection_id}",
            )
            content = ""
            if isinstance(result, dict):
                content = result.get("content", "") or ""
                if not content:
                    # Some agent paths put the reply in different keys.
                    for key in ("response", "text", "message", "reply"):
                        val = result.get(key)
                        if isinstance(val, str) and val:
                            content = val
                            break
            else:
                content = str(result)
            if not content:
                content = "I'm not sure how to respond to that."
            return content
        except Exception as e:
            logger.error("Agent run failed in voice session %s: %s", self.connection_id, e, exc_info=True)
            return "Sorry, something went wrong while processing your request."

    async def _synthesize_and_stream(self, text: str) -> None:
        from tools.tts_tool import text_to_speech_tool
        loop = asyncio.get_running_loop()
        if self._gen_token is not None and self._gen_token.cancelled:
            logger.debug("Skipping TTS for interrupted generation in session %s", self.connection_id)
            await self.send_json({"type": "status", "status": "idle"})
            return
        try:
            raw = await loop.run_in_executor(None, text_to_speech_tool, text)
            tts_result = json.loads(raw)
        except Exception as e:
            logger.error("TTS failed in voice session %s: %s", self.connection_id, e)
            await self.send_json({"type": "status", "status": "tts_failed", "error": str(e)})
            return

        if not tts_result.get("success"):
            await self.send_json({
                "type": "status",
                "status": "tts_failed",
                "error": tts_result.get("error", "TTS failed"),
            })
            return

        file_path = tts_result.get("file_path", "")
        if not file_path or not os.path.isfile(file_path):
            await self.send_json({"type": "status", "status": "tts_failed", "error": "no audio file"})
            return

        pcm = self._file_to_pcm(file_path)
        if pcm:
            await self._frontend.play_audio(pcm)
        await self.send_json({"type": "status", "status": "idle"})

        try:
            os.unlink(file_path)
        except OSError:
            pass

    def _file_to_pcm(self, file_path: str) -> bytes:
        """Best-effort conversion of any supported audio file to 16 kHz mono PCM."""
        try:
            import sounddevice as sd
            import numpy as np
            data, fs = sd.read(file_path, dtype="float32")  # type: ignore[attr-defined]
            if data.ndim > 1:
                data = data.mean(axis=1)
            target_len = int(round(len(data) * SAMPLE_RATE / fs))
            if target_len == 0:
                return b""
            old_indices = np.linspace(0, len(data) - 1, target_len)
            indices = old_indices.astype(np.int32)
            frac = old_indices - indices
            next_indices = np.minimum(indices + 1, len(data) - 1)
            resampled = data[indices] * (1.0 - frac) + data[next_indices] * frac
            int16 = (resampled * 32767.0).astype(np.int16)
            return int16.tobytes()
        except Exception:
            pass

        # Fallback: if file is already WAV, read it directly if the format matches.
        if file_path.lower().endswith(".wav"):
            try:
                with wave.open(file_path, "rb") as wf:
                    if wf.getsampwidth() == SAMPLE_WIDTH and wf.getnchannels() == CHANNELS and wf.getframerate() == SAMPLE_RATE:
                        return wf.readframes(wf.getnframes())
            except Exception:
                pass
        return b""

    def _write_wav(self, pcm_bytes: bytes) -> Optional[str]:
        if not pcm_bytes:
            return None
        try:
            tmp_dir = Path(tempfile.gettempdir()) / "hermes_voice"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            path = tmp_dir / f"ws_voice_{self.connection_id}_{int(time.time())}.wav"
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm_bytes)
            return str(path)
        except Exception as e:
            logger.error("Failed to write voice WAV: %s", e)
            return None

    @staticmethod
    def _format_ts() -> str:
        now = time.localtime()
        return f"{now.tm_hour}:{now.tm_min:02d}:{now.tm_sec:02d}"

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._recorder is not None:
                self._recorder.cancel()
                self._recorder.shutdown()
        except Exception as e:
            logger.debug("Recorder shutdown failed: %s", e)
        self._frontend.shutdown()
        try:
            if not self._ws.closed:
                await self._ws.close()
        except Exception as e:
            logger.debug("WebSocket close failed: %s", e)


def _is_tailscale_ip(ip_str: str) -> bool:
    """Return True for Tailscale CGNAT addresses (100.64.0.0/10)."""
    try:
        addr = ipaddress.ip_address(ip_str)
        return addr in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        return False


def _client_ip(request: Any) -> str:
    """Best-effort client IP, preferring X-Forwarded-For only when plausible."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    real_ip = request.headers.get("X-Real-IP", "")
    if real_ip:
        return real_ip.strip()
    peer = request.transport.get_extra_info("peername") if request.transport else None
    if isinstance(peer, (tuple, list)) and peer:
        return str(peer[0])
    return ""


def _voice_websocket_config(adapter: Any) -> Dict[str, Any]:
    """Read the ``voice.websocket`` block from the adapter config / env."""
    cfg = {}
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        pass
    voice_cfg = cfg.get("voice") if isinstance(cfg, dict) else {}
    if not isinstance(voice_cfg, dict):
        voice_cfg = {}
    ws_cfg = voice_cfg.get("websocket") if isinstance(voice_cfg, dict) else {}
    if not isinstance(ws_cfg, dict):
        ws_cfg = {}

    extra = (adapter._config.extra or {}) if hasattr(adapter, "_config") else {}
    voice_extra = extra.get("voice", {})
    enabled = ws_cfg.get("enabled", voice_extra.get("enabled", True))
    require_tailscale = ws_cfg.get("require_tailscale", voice_extra.get("require_tailscale", True))
    path = ws_cfg.get("path", voice_extra.get("path", "/voice"))
    return {
        "enabled": bool(enabled) if isinstance(enabled, bool) else str(enabled).lower() in {"true", "1", "yes", "on"},
        "require_tailscale": bool(require_tailscale) if isinstance(require_tailscale, bool) else str(require_tailscale).lower() in {"true", "1", "yes", "on"},
        "path": str(path) if path else "/voice",
    }


async def handle_voice_ws(adapter: Any, request: Any) -> Any:
    """Entry point called by ``APIServerAdapter._handle_voice_ws``."""
    from aiohttp import web, WSMsgType

    cfg = _voice_websocket_config(adapter)
    if not cfg["enabled"]:
        return web.json_response({"error": "Voice WebSocket is disabled"}, status=404)

    # Auth: reuse the API server's Bearer-token check.
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    client_ip = _client_ip(request)
    if cfg["require_tailscale"] and not _is_tailscale_ip(client_ip):
        logger.warning(
            "Voice WebSocket rejected non-Tailscale client ip=%s path=%s",
            client_ip, request.path_qs,
        )
        return web.json_response({"error": "Voice WebSocket requires Tailscale"}, status=403)

    logger.info("Voice WebSocket accepted from %s", client_ip)

    ws = web.WebSocketResponse(heartbeat=30.0, autoping=True)
    await ws.prepare(request)

    connection_id = str(uuid.uuid4())
    session = VoiceSession(connection_id, ws, adapter, {})

    async with adapter._voice_sessions_lock:
        if adapter._voice_sessions:
            await ws.send_str(json.dumps({"type": "status", "status": "busy", "message": "Only one voice session at a time"}))
            await ws.close()
            return ws
        adapter._voice_sessions[connection_id] = session

    try:
        await session.run()
    finally:
        async with adapter._voice_sessions_lock:
            adapter._voice_sessions.pop(connection_id, None)
        await session.close()

    return ws
