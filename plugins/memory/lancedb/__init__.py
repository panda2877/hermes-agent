"""LanceDB memory provider — MemoryProvider plugin for LanceDB-backed storage.

Provides:
  - Soul loading (profile identity) from LanceDB, with file fallback
  - User profile loading from LanceDB, with file fallback
  - Memory listing and semantic search via LanceDB API
  - Built-in memory write mirroring (on_memory_write)

Config:
  No extra config needed — reuses the existing LanceDB Memory Service
  (port 9091) configured via ``LANCE_MEMORY_API_BASE`` env var or
  the default ``http://127.0.0.1:9091/api/v1``.

  To activate, set in config.yaml:
    memory:
      provider: lancedb
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ── Reuse the existing HTTP client ────────────────────────────────────────
# The lancedb_client module lives in agent/ so both prompt_builder and
# this plugin share the same code path. It handles retries, fallback
# files, and agent-specific path resolution.
try:
    from agent.lancedb_client import (
        fetch_soul,
        fetch_user,
        fetch_memories,
        search_memories,
        is_api_available,
    )
    HAS_CLIENT = True
except ImportError:
    logger.warning("agent.lancedb_client not available; LanceDB provider disabled")
    HAS_CLIENT = False


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

MEMORY_SEARCH_SCHEMA = {
    "name": "memory_search",
    "description": (
        "Semantic search across all memories stored in LanceDB. "
        "Use this for cross-referencing past conversations, retrieving "
        "specific facts, or finding related context. "
        "Returns relevant memory snippets sorted by similarity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query — describe what you're looking for.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default: 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

MEMORY_LIST_SCHEMA = {
    "name": "memory_list",
    "description": (
        "List all memories stored in LanceDB for the current agent. "
        "Useful for reviewing what the agent remembers."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class LanceDBMemoryProvider(MemoryProvider):
    """MemoryProvider backed by LanceDB HTTP API."""

    @property
    def name(self) -> str:
        return "lancedb"

    def is_available(self) -> bool:
        return HAS_CLIENT

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        # Determine agent identity from kwargs (set by run_agent.py)
        self._agent_identity = kwargs.get("agent_identity", "")
        self._hermes_home = kwargs.get("hermes_home", "")
        logger.info(
            "LanceDB provider initialized for agent=%s session=%s",
            self._agent_identity, session_id,
        )

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Fetch relevant memories before each turn."""
        agent = self._agent_identity or ""
        if not agent:
            return ""
        try:
            results = search_memories(query, owner=agent, limit=5)
            if results:
                parts = [f"<recalled_memory>{r}</recalled_memory>" for r in results]
                return "\n".join(parts)
        except Exception as e:
            logger.debug("LanceDB prefetch failed: %s", e)
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Sync a turn to LanceDB (no-op for now — handled by memory tool)."""
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [MEMORY_SEARCH_SCHEMA, MEMORY_LIST_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "memory_search":
            return self._handle_search(args)
        elif tool_name == "memory_list":
            return self._handle_list()
        return tool_error(f"LanceDB provider does not handle tool '{tool_name}'")

    def _handle_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        limit = args.get("limit", 5)
        agent = self._agent_identity or ""
        try:
            results = search_memories(query, owner=agent, limit=limit)
            if results:
                return json.dumps({"results": results, "count": len(results)})
            return json.dumps({"results": [], "count": 0, "message": "No relevant memories found."})
        except Exception as e:
            return tool_error(f"Memory search failed: {e}")

    def _handle_list(self) -> str:
        agent = self._agent_identity or ""
        try:
            memories = fetch_memories(agent)
            if memories:
                return json.dumps({"memories": memories, "count": len(memories)})
            return json.dumps({"memories": [], "count": 0, "message": "No memories stored."})
        except Exception as e:
            return tool_error(f"Memory list failed: {e}")

    def shutdown(self) -> None:
        logger.info("LanceDB provider shut down")

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes to LanceDB.

        Called by MemoryManager when the built-in memory tool writes an entry.
        Since LanceDB is the primary store (not a mirror), this is a no-op —
        the built-in memory tool already writes to LanceDB directly.
        """
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Nothing to do on session end — LanceDB is persistent."""
        pass


def register(ctx):
    """Register LanceDB as a memory provider plugin."""
    ctx.register_memory_provider(LanceDBMemoryProvider())