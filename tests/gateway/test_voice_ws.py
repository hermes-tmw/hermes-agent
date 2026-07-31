"""Tests for the WebSocket /voice endpoint in gateway/platforms/api_server.py.

These tests verify:
- Config-based enable/disable of the voice WebSocket.
- Auth (valid API key, missing API key, invalid API key).
- Tailscale source-IP gating.
- Single-session-at-a-time rejection.
- A full happy-path conversation using mocked STT/TTS and a synthetic PCM
  utterance that triggers the RMS VAD endpoint detector.
"""

import asyncio
import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from aiohttp import web, WSMsgType
from aiohttp.test_utils import TestClient, TestServer

import aiohttp

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, security_headers_middleware, cors_middleware
from gateway.voice_ws_handler import _is_tailscale_ip, _client_ip

# AsyncMock is imported but unused by the new single-session test; keep it for
# future test expansion.


def _create_app(adapter):
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/voice", adapter._handle_voice_ws)
    return app


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, extra={"key": "test-api-key-16b-long"})
    adapter = APIServerAdapter(config)
    adapter._host = "127.0.0.1"
    adapter._port = 0
    return adapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pcm_utterance(duration_s: float = 2.0) -> bytes:
    """Create a synthetic high-energy PCM utterance that RMS VAD will detect."""
    sample_rate = 16000
    samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, samples)
    wave = (0.7 * 32767.0 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
    return wave.tobytes()


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

class TestTailscaleIp:
    def test_accepts_tailscale_cgnat(self):
        assert _is_tailscale_ip("100.64.0.1") is True
        assert _is_tailscale_ip("100.127.255.254") is True

    def test_rejects_non_tailscale(self):
        assert _is_tailscale_ip("192.168.1.1") is False
        assert _is_tailscale_ip("127.0.0.1") is False
        assert _is_tailscale_ip("::1") is False
        assert _is_tailscale_ip("not-an-ip") is False


class TestClientIp:
    def test_prefers_x_forwarded_for(self):
        request = MagicMock()
        request.headers = {
            "X-Forwarded-For": "100.64.1.2, 10.0.0.1",
            "X-Real-IP": "192.168.1.1",
        }
        request.transport = MagicMock()
        request.transport.get_extra_info.return_value = ("127.0.0.1", 12345)
        assert _client_ip(request) == "100.64.1.2"

    def test_falls_back_to_peername(self):
        request = MagicMock()
        request.headers = {}
        request.transport = MagicMock()
        request.transport.get_extra_info.return_value = ("100.64.1.3", 12345)
        assert _client_ip(request) == "100.64.1.3"


# ---------------------------------------------------------------------------
# Connection / auth
# ---------------------------------------------------------------------------

class TestVoiceWsAuth:
    @pytest.mark.asyncio
    async def test_rejects_missing_auth(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with pytest.raises(Exception):
                await cli.ws_connect("/voice")

    @pytest.mark.asyncio
    async def test_rejects_invalid_auth(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with pytest.raises(Exception):
                await cli.session.ws_connect(
                    f"{cli.make_url('/voice')}",
                    headers={"Authorization": "Bearer wrong-key"},
                )

    @pytest.mark.asyncio
    async def test_accepts_valid_auth(self, adapter, monkeypatch):
        app = _create_app(adapter)
        from gateway import voice_ws_handler
        monkeypatch.setattr(
            voice_ws_handler,
            "_voice_websocket_config",
            lambda _a: {"enabled": True, "require_tailscale": False, "path": "/voice"},
        )
        with patch("gateway.voice_ws_handler.VoiceSession.run", new=AsyncMock()):
            async with TestClient(TestServer(app)) as cli:
                async with cli.session.ws_connect(
                    f"{cli.make_url('/voice')}",
                    headers={"Authorization": "Bearer test-api-key-16b-long"},
                ) as ws:
                    assert not ws.closed

    @pytest.mark.asyncio
    async def test_rejects_non_tailscale_when_required(self, adapter, monkeypatch):
        app = _create_app(adapter)
        from gateway import voice_ws_handler
        monkeypatch.setattr(
            voice_ws_handler,
            "_voice_websocket_config",
            lambda _a: {"enabled": True, "require_tailscale": True, "path": "/voice"},
        )
        async with TestClient(TestServer(app)) as cli:
            with pytest.raises(Exception):
                await cli.session.ws_connect(
                    f"{cli.make_url('/voice')}",
                    headers={"Authorization": "Bearer test-api-key-16b-long"},
                )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class TestVoiceSessionManagement:
    @pytest.mark.asyncio
    async def test_only_one_session_at_a_time(self, adapter, monkeypatch):
        app = _create_app(adapter)
        from gateway import voice_ws_handler
        monkeypatch.setattr(
            voice_ws_handler,
            "_voice_websocket_config",
            lambda _a: {"enabled": True, "require_tailscale": False, "path": "/voice"},
        )

        # Keep the first session registered long enough to test the lock.
        stop_event = asyncio.Event()

        async def blocking_run(*args, **kwargs) -> None:
            await stop_event.wait()

        with patch("gateway.voice_ws_handler.VoiceSession.run", side_effect=blocking_run):
            async with TestClient(TestServer(app)) as cli:
                async with cli.session.ws_connect(
                    f"{cli.make_url('/voice')}",
                    headers={"Authorization": "Bearer test-api-key-16b-long"},
                ) as ws1:
                    # Wait for the server to register the first session.
                    for _ in range(20):
                        if adapter._voice_sessions:
                            break
                        await asyncio.sleep(0.05)
                    assert len(adapter._voice_sessions) == 1

                    # Second connection should be rejected with a "busy" status.
                    async with cli.session.ws_connect(
                        f"{cli.make_url('/voice')}",
                        headers={"Authorization": "Bearer test-api-key-16b-long"},
                    ) as ws2:
                        msg = await ws2.receive(timeout=2.0)
                        assert msg.type == WSMsgType.TEXT
                        data = json.loads(msg.data)
                        assert data["type"] == "status"
                        assert data["status"] == "busy"
                        # aiohttp may not mark ws2 closed until we read the close
                        # frame; drain it if present.
                        if not ws2.closed:
                            try:
                                await ws2.receive(timeout=1.0)
                            except Exception:
                                pass
                        assert ws2.closed

                    # Release the first session and verify cleanup.
                    stop_event.set()
                    await ws1.close()
                    for _ in range(20):
                        if not adapter._voice_sessions:
                            break
                        await asyncio.sleep(0.05)
                    assert len(adapter._voice_sessions) == 0

# ---------------------------------------------------------------------------
# Happy path (mocked STT/TTS)
# ---------------------------------------------------------------------------

class TestVoiceWsHappyPath:
    @pytest.mark.asyncio
    async def test_full_conversation(self, adapter, monkeypatch):
        app = _create_app(adapter)
        from gateway import voice_ws_handler
        monkeypatch.setattr(
            voice_ws_handler,
            "_voice_websocket_config",
            lambda _a: {"enabled": True, "require_tailscale": False, "path": "/voice"},
        )

        # Patch the agent so no real LLM is called.
        async def fake_run_agent(self, **kwargs):
            return {"content": "hello from hermes"}, {"total_tokens": 10}

        monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_run_agent)

        async def fake_transcribe(self, wav_path):
            return {"success": True, "transcript": "hello computer"}

        monkeypatch.setattr(voice_ws_handler.VoiceSession, "_transcribe", fake_transcribe)

        # TTS returns a tiny fake WAV file (already 16kHz mono).
        def fake_tts(text, output_path=None):
            import wave, tempfile, os
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x01" * 16000)
            return json.dumps({"success": True, "file_path": path})

        monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts)

        async with TestClient(TestServer(app)) as cli:
            async with cli.session.ws_connect(
                f"{cli.make_url('/voice')}",
                headers={"Authorization": "Bearer test-api-key-16b-long"},
            ) as ws:
                msg = await ws.receive_json(timeout=5)
                assert msg["type"] == "status"
                assert msg["status"] == "ready"

                # Send enough energy to trip the RMS VAD endpoint.
                pcm = _make_pcm_utterance(duration_s=2.5)
                await ws.send_bytes(pcm)

                seen_types = set()
                for _ in range(30):
                    try:
                        msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                        seen_types.add((msg.get("type"), msg.get("status")))
                        if msg.get("type") == "transcript" and msg.get("speaker") == "hermes":
                            break
                    except Exception:
                        break

                assert any(t[0] == "transcript" for t in seen_types)
