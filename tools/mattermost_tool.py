"""Mattermost on-request channel-history backfill tool.

Provides a single LLM-callable tool, ``mattermost_channel_history``, that fetches
prior posts from a Mattermost channel when the user explicitly asks for them.
This is a manual backfill tool only — it does NOT run automatically on channel
join or on every message.

Authentication and server URL are read from the active profile's
``MATTERMOST_TOKEN`` and ``MATTERMOST_URL`` (mirrors the adapter). The tool
requires a live Mattermost adapter in the running gateway; there is no
standalone cron fallback because history fetch is an interactive, on-request
operation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from agent.secret_scope import get_secret
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


MATTERMOST_CHANNEL_HISTORY_SCHEMA = {
    "name": "mattermost_channel_history",
    "description": (
        "Fetch prior messages from a Mattermost channel on explicit request.\n\n"
        "Use this when the user asks something like 'what was said in #X before I joined'.\n"
        "It does NOT backfill automatically — it only runs when you call it.\n\n"
        "Provide the Mattermost channel_id (not the human-readable name). "
        "You can use before_ts / since_ts to bound the window; timestamps can be:\n"
        "  - epoch milliseconds (Mattermost create_at format),\n"
        "  - ISO-8601 strings (e.g. 2026-08-06T12:00:00Z),\n"
        "  - YYYY-MM-DD dates (interpreted as UTC midnight).\n"
        "Returns formatted history: one line per post with timestamp, sender, and message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": "Mattermost channel ID (e.g. 7s1jnktiz3npt85kmcwgwem6nh).",
            },
            "before_ts": {
                "type": "string",
                "description": "Optional: include posts strictly before this timestamp.",
            },
            "since_ts": {
                "type": "string",
                "description": "Optional: include posts at or after this timestamp.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "default": 200,
                "description": "Maximum posts to fetch (1-200, default 200).",
            },
        },
        "required": ["channel_id"],
    },
}


def _check_mattermost_tool_requirements() -> bool:
    """Available when Mattermost token and URL are configured."""
    token = (get_secret("MATTERMOST_TOKEN", "") or "").strip()
    url = (get_secret("MATTERMOST_URL", "") or "").strip()
    return bool(token and url)


def _get_live_adapter() -> Optional[Any]:
    """Return the live in-process Mattermost adapter, if any."""
    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        if runner is None:
            return None
        from gateway.config import Platform

        return runner.adapters.get(Platform.MATTERMOST)
    except Exception:
        return None


def _handle_mattermost_channel_history(args: Dict[str, Any], **kw) -> str:
    """Sync handler bridge to the async adapter method."""
    channel_id = str(args.get("channel_id") or "").strip()
    if not channel_id:
        return tool_error("channel_id is required")
    # Defense-in-depth: validate channel_id format before touching the adapter.
    from plugins.platforms.mattermost.adapter import _is_valid_mm_channel_id
    if not _is_valid_mm_channel_id(channel_id):
        return tool_error(f"Invalid Mattermost channel_id: {channel_id}")

    adapter = _get_live_adapter()
    if adapter is None:
        return tool_error(
            "No live Mattermost adapter in the running gateway. "
            "The gateway must be connected to Mattermost to fetch history."
        )

    before_ts = args.get("before_ts")
    since_ts = args.get("since_ts")
    limit = args.get("limit", 200)

    try:
        from model_tools import _run_async

        result = _run_async(
            adapter.fetch_channel_history(
                channel_id=channel_id,
                before_ts=before_ts,
                since_ts=since_ts,
                limit=limit,
            )
        )
    except Exception as exc:
        logger.exception("mattermost_channel_history failed")
        return tool_error(f"Failed to fetch Mattermost history: {exc}")

    # fetch_channel_history returns either a formatted string or an error JSON.
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return tool_result(result)
        if isinstance(parsed, dict) and "error" in parsed:
            return tool_error(parsed["error"])
        return tool_result(result)
    return tool_result(result)


registry.register(
    name="mattermost_channel_history",
    toolset="mattermost",
    schema=MATTERMOST_CHANNEL_HISTORY_SCHEMA,
    handler=_handle_mattermost_channel_history,
    check_fn=_check_mattermost_tool_requirements,
    requires_env=["MATTERMOST_URL", "MATTERMOST_TOKEN"],
    emoji="💬",
)
