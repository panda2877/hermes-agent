"""Lightweight HTTP client for LanceDB Memory Service (port 9091).

Used by prompt_builder.py and memory_tool.py to fetch souls / user / memories
from LanceDB, with file-based fallback for soul/user only (not memories).

Usage:
    from agent.lancedb_client import fetch_soul, fetch_user, fetch_memories
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LANCE_API_BASE = os.environ.get(
    "LANCE_MEMORY_API_BASE",
    "http://127.0.0.1:9091/api/v1",
)
LANCE_API_TIMEOUT = 5  # seconds per attempt
LANCE_PAGE_LIMIT = 100  # max results per request
LANCE_STARTUP_RETRIES = 5  # retry count when LanceDB not yet ready (server startup race)
LANCE_RETRY_DELAY = 2  # seconds between retries

# Fallback file paths (soul/user only — memories are LanceDB-only)
FALLBACK_SOUL_PATH = Path("/home/agentuser/.hermes/SOUL.md")
FALLBACK_USER_PATH = Path("/home/agentuser/.hermes/memories/USER.md")

# Module-level flag: only retry on the first-ever call to _api_get
_had_successful_call = False


def _api_get(path: str) -> Optional[dict]:
    """GET from LanceDB API. Retries on first call if service not yet ready.

    After one successful call, subsequent calls fail fast (no retry).
    Returns None on any failure.
    """
    global _had_successful_call
    url = f"{LANCE_API_BASE}{path}"
    max_attempts = 1 if _had_successful_call else LANCE_STARTUP_RETRIES

    for attempt in range(1, max_attempts + 1):
        try:
            resp = urllib.request.urlopen(url, timeout=LANCE_API_TIMEOUT)
            _had_successful_call = True
            return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError, json.JSONDecodeError) as e:
            if attempt < max_attempts:
                logger.debug("LanceDB not ready yet (attempt %d/%d): %s",
                             attempt, max_attempts, e)
                time.sleep(LANCE_RETRY_DELAY)
            else:
                logger.debug("LanceDB API unavailable (%s): %s", url, e)
    return None


def _api_post(path: str, body: dict) -> Optional[dict]:
    """POST to LanceDB API. Returns None on any failure."""
    url = f"{LANCE_API_BASE}{path}"
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=LANCE_API_TIMEOUT)
        return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, OSError, json.JSONDecodeError) as e:
        logger.debug("LanceDB API POST unavailable (%s): %s", url, e)
        return None


def _sync_file(path: Path, content: str):
    """Write content to fallback file, ensuring parent dir exists."""
    if not content:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        logger.debug("Synced fallback file: %s (%d chars)", path, len(content))
    except OSError as e:
        logger.debug("Failed to sync fallback file %s: %s", path, e)


def _soul_fallback_path(agent_name: str) -> Path:
    """Determine the fallback SOUL.md path for a given agent.

    Silver Moon (main agent, name='hermes' or profile='home'):
      → /home/agentuser/.hermes/SOUL.md
    Sub-agents (named profiles like xingruyin, ziling, wensiyue):
      → /home/agentuser/.hermes/profiles/<name>/SOUL.md
    """
    if agent_name in ("hermes", "home", None, ""):
        return FALLBACK_SOUL_PATH
    return Path(f"/home/agentuser/.hermes/profiles/{agent_name}/SOUL.md")


def fetch_soul(agent_name: str) -> Optional[str]:
    """Fetch soul content for the given agent from LanceDB.

    On success, syncs to fallback SOUL.md file.
    On failure, falls back to reading the file.

    Returns the soul text content, or None if unavailable.
    """
    data = _api_get(f"/memories?category=soul&owner={agent_name}&limit=3")
    if data and data.get("count", 0) > 0:
        results = data["results"]
        results.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        content = results[0].get("content", "")
        if content:
            logger.info("Loaded soul from LanceDB for agent=%s (%d chars)",
                        agent_name, len(content))
            # Sync to agent-specific fallback file
            _sync_file(_soul_fallback_path(agent_name), content)
            return content

    # Fallback: read from agent-specific file, then global
    fallback_paths = [_soul_fallback_path(agent_name), FALLBACK_SOUL_PATH]
    for fb_path in fallback_paths:
        try:
            if fb_path.exists():
                content = fb_path.read_text().strip()
                if content:
                    logger.info("Loaded soul from fallback file: %s (%d chars)",
                                fb_path, len(content))
                    return content
        except OSError as e:
            logger.debug("Failed to read fallback soul file %s: %s", fb_path, e)
    return None


def fetch_user() -> Optional[str]:
    """Fetch merged user profile from LanceDB.

    On success, syncs to fallback USER.md file.
    On failure, falls back to reading the file.

    Returns the user content text, or None if unavailable.
    """
    data = _api_get("/memories?category=user&limit=3")
    if data and data.get("count", 0) > 0:
        results = data["results"]
        results.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        content = results[0].get("content", "")
        if content:
            logger.info("Loaded user profile from LanceDB (%d chars)", len(content))
            # Sync to fallback file for offline backup
            _sync_file(FALLBACK_USER_PATH, content)
            return content

    # Fallback: read from file
    try:
        if FALLBACK_USER_PATH.exists():
            content = FALLBACK_USER_PATH.read_text().strip()
            if content:
                logger.info("Loaded user from fallback file (%d chars)", len(content))
                return content
    except OSError as e:
        logger.debug("Failed to read fallback user file: %s", e)
    return None


def fetch_memories(agent_name: str) -> list:
    """Fetch memory entries for an agent from LanceDB only.

    Returns list of content strings (oldest first), or empty list on failure.
    No file fallback — memories are LanceDB-only.
    """
    data = _api_get(f"/memories?category=memory&owner={agent_name}&limit={LANCE_PAGE_LIMIT}")
    if data and data.get("count", 0) > 0:
        entries = [r["content"] for r in data["results"] if r.get("content")]
        logger.info("Loaded %d memories from LanceDB for agent=%s",
                    len(entries), agent_name)
        return entries
    logger.debug("No memories found in LanceDB for agent=%s", agent_name)
    return []


def search_memories(query: str, owner: Optional[str] = None,
                    limit: int = 5) -> list[str]:
    """Semantic search memories via LanceDB.

    Returns a list of relevant content strings (empty if unavailable).
    """
    body = {"query": query, "limit": limit}
    if owner:
        body["owner"] = owner
    data = _api_post("/search", body)
    if data and data.get("count", 0) > 0:
        return [r["content"] for r in data["results"] if r.get("content")]
    return []


def log_injection(agent_name: str, injected: list[dict]):
    """Log what was injected into an agent's context (best-effort, no fallback)."""
    try:
        data = json.dumps({"agent": agent_name, "injected": injected}).encode()
        req = urllib.request.Request(
            f"{LANCE_API_BASE}/injection/{agent_name}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception as e:
        logger.debug("Failed to log injection: %s", e)


def is_api_available() -> bool:
    """Quick health check — returns True if LanceDB API is reachable."""
    data = _api_get("/status")
    return data is not None and data.get("status") == "ok"


def write_memory(agent_name: str, content: str, category: str = "memory") -> bool:
    """Write a memory entry to LanceDB.

    Args:
        agent_name: The agent profile name (e.g. "xingruyin", "ziling").
        content: The memory content string to store.
        category: "memory" (default) or "user".

    Returns:
        True if the write succeeded, False otherwise.
    """
    if not agent_name or not content:
        return False
    data = _api_post(
        f"/memories/{agent_name}",
        {"content": content, "category": category},
    )
    return data is not None