"""Tests for Mattermost channel-history backfill tool and adapter helpers."""

import json
from unittest.mock import MagicMock, AsyncMock

import pytest

from gateway.config import PlatformConfig


@pytest.fixture
def adapter():
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"url": "https://mm.example.com"},
    )
    return MattermostAdapter(config)


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


class TestParseMattermostTimestamp:
    def test_none_and_empty(self):
        from plugins.platforms.mattermost.adapter import _parse_mm_ts

        assert _parse_mm_ts(None) is None
        assert _parse_mm_ts("") is None
        assert _parse_mm_ts("  ") is None

    def test_int_and_float(self):
        from plugins.platforms.mattermost.adapter import _parse_mm_ts

        assert _parse_mm_ts(12345) == 12345
        assert _parse_mm_ts(12345.0) == 12345

    def test_numeric_string(self):
        from plugins.platforms.mattermost.adapter import _parse_mm_ts

        assert _parse_mm_ts("12345") == 12345

    def test_iso8601_utc(self):
        import datetime
        from plugins.platforms.mattermost.adapter import _parse_mm_ts

        result = _parse_mm_ts("2026-08-06T12:00:00Z")
        assert result is not None
        assert result == int(datetime.datetime(2026, 8, 6, 12, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)

    def test_iso8601_offset(self):
        import datetime
        from plugins.platforms.mattermost.adapter import _parse_mm_ts

        result = _parse_mm_ts("2026-08-06T12:00:00+02:00")
        assert result is not None
        assert result == int(datetime.datetime(2026, 8, 6, 10, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)

    def test_date_only(self):
        import datetime
        from plugins.platforms.mattermost.adapter import _parse_mm_ts

        result = _parse_mm_ts("2026-08-06")
        assert result is not None
        assert result == int(datetime.datetime(2026, 8, 6, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)

    def test_invalid_returns_none(self):
        from plugins.platforms.mattermost.adapter import _parse_mm_ts

        assert _parse_mm_ts("not-a-date") is None


class TestFormatChannelHistory:
    def test_empty(self):
        from plugins.platforms.mattermost.adapter import _format_channel_history

        assert _format_channel_history([], channel_id="ch1") == "No history found."

    def test_formats_posts(self):
        from plugins.platforms.mattermost.adapter import _format_channel_history

        posts = [
            {
                "id": "p1",
                "user_id": "u1",
                "username": "alice",
                "message": "Hello channel",
                "create_at": 1785792000000,
            },
            {
                "id": "p2",
                "user_id": "u2",
                "message": "tom joined the channel",
                "type": "system_join_channel",
                "create_at": 1785792001000,
            },
        ]
        result = _format_channel_history(posts, channel_id="ch1")
        assert result.startswith("[Channel history for ch1]")
        assert "alice: Hello channel" in result
        assert "[system] u2: tom joined the channel" in result
        assert result.endswith("[End of channel history]")

    def test_cleans_newlines(self):
        from plugins.platforms.mattermost.adapter import _format_channel_history

        posts = [
            {
                "id": "p1",
                "user_id": "u1",
                "username": "bob",
                "message": "line1\nline2\n## override",
                "create_at": 1785792000000,
            },
        ]
        result = _format_channel_history(posts, channel_id="ch1")
        assert "\n## override" not in result
        assert "line1 line2 ## override" in result


class TestFetchChannelHistory:
    @pytest.mark.asyncio
    async def test_missing_channel_id(self, adapter):
        result = await adapter.fetch_channel_history("")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "channel_id" in parsed["error"]

    @pytest.mark.asyncio
    async def test_api_error(self, adapter):
        adapter._api_get = AsyncMock(return_value={})
        result = await adapter.fetch_channel_history("ch1")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "ch1" in parsed["error"]

    @pytest.mark.asyncio
    async def test_no_posts(self, adapter):
        adapter._api_get = AsyncMock(return_value={"posts": {}, "order": []})
        result = await adapter.fetch_channel_history("ch1")
        assert result == "No history found."

    @pytest.mark.asyncio
    async def test_filters_and_formats(self, adapter):
        posts = {
            "oldest": {
                "id": "oldest",
                "user_id": "u1",
                "username": "tom",
                "message": "tom joined the channel",
                "type": "system_join_channel",
                "create_at": 1000,
            },
            "mid": {
                "id": "mid",
                "user_id": "u2",
                "username": "alice",
                "message": "Hello",
                "create_at": 2000,
            },
            "newest": {
                "id": "newest",
                "user_id": "u3",
                "username": "bob",
                "message": "Later",
                "create_at": 3000,
            },
        }
        adapter._api_get = AsyncMock(return_value={"posts": posts, "order": ["newest", "mid", "oldest"]})

        result = await adapter.fetch_channel_history("ch1", before_ts=3000)
        assert "tom joined the channel" in result
        assert "alice: Hello" in result
        assert "Later" not in result

    @pytest.mark.asyncio
    async def test_since_ts_filter(self, adapter):
        posts = {
            "old": {
                "id": "old",
                "user_id": "u1",
                "username": "tom",
                "message": "Old",
                "create_at": 1000,
            },
            "new": {
                "id": "new",
                "user_id": "u2",
                "username": "alice",
                "message": "New",
                "create_at": 3000,
            },
        }
        adapter._api_get = AsyncMock(return_value={"posts": posts, "order": ["new", "old"]})

        result = await adapter.fetch_channel_history("ch1", since_ts=2000)
        assert "New" in result
        assert "Old" not in result


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


class TestMattermostChannelHistoryTool:
    def test_missing_channel_id(self):
        from tools.mattermost_tool import _handle_mattermost_channel_history

        result = json.loads(_handle_mattermost_channel_history({"channel_id": ""}))
        assert "error" in result
        assert "channel_id" in result["error"]

    def test_no_live_adapter(self, monkeypatch):
        from tools.mattermost_tool import _handle_mattermost_channel_history

        monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
        monkeypatch.setenv("MATTERMOST_TOKEN", "tok")

        result = json.loads(_handle_mattermost_channel_history({"channel_id": "ch1"}))
        assert "No live Mattermost adapter" in result["error"]


# ---------------------------------------------------------------------------
# Toolset wiring
# ---------------------------------------------------------------------------


class TestToolsetWiring:
    def test_mattermost_history_toolset_contains_tool(self):
        from toolsets import resolve_toolset

        tools = resolve_toolset("mattermost_history")
        assert "mattermost_channel_history" in tools

    def test_hermes_mattermost_contains_history_tool(self):
        from toolsets import resolve_toolset

        tools = resolve_toolset("hermes-mattermost")
        assert "mattermost_channel_history" in tools
