"""Tests for the snapshot memory provider (plugins/memory/snapshot/)."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from plugins.memory.snapshot.context_extractor import extract_task_context
from plugins.memory.snapshot.snapshot_store import SnapshotStore
from plugins.memory.snapshot.provider import SnapshotMemoryProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_messages(*pairs):
    """Build a minimal message list from alternating user/assistant strings."""
    msgs = []
    for i, text in enumerate(pairs):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": text})
    return msgs


# ---------------------------------------------------------------------------
# context_extractor — finished field
# ---------------------------------------------------------------------------

class TestExtractTaskContextFinished:
    """finished is True when all checklist items are done."""

    def test_no_checklist_returns_false(self):
        msgs = make_messages("帮我实现用户认证模块")
        result = extract_task_context(msgs)
        assert result["finished"] is False

    def test_partial_checklist_returns_false(self):
        msgs = make_messages(
            "任务：完成登录模块\n"
            "- [ ] 实现 JWT 签名\n"
            "- [x] 实现登录 API"
        )
        result = extract_task_context(msgs)
        assert result["finished"] is False

    def test_all_done_returns_true(self):
        msgs = make_messages(
            "任务：完成登录模块\n"
            "- [x] 完成 JWT 签名\n"
            "- [x] 实现登录 API"
        )
        result = extract_task_context(msgs)
        assert result["finished"] is True

    def test_empty_messages_returns_finished_false(self):
        result = extract_task_context([])
        assert result["finished"] is False


# ---------------------------------------------------------------------------
# context_extractor — task_goals
# ---------------------------------------------------------------------------

class TestExtractTaskGoals:
    def test_simple_task(self):
        msgs = make_messages("需要你帮我实现用户认证模块")
        result = extract_task_context(msgs)
        assert len(result["task_goals"]) > 0

    def test_no_goal(self):
        msgs = make_messages("hello")
        result = extract_task_context(msgs)
        assert result["task_goals"] == []

    def test_trivial_hello(self):
        msgs = make_messages("hi")
        result = extract_task_context(msgs)
        assert result["task_goals"] == []


# ---------------------------------------------------------------------------
# _generate_task_id
# ---------------------------------------------------------------------------

class TestGenerateTaskId:
    def _provider(self):
        p = SnapshotMemoryProvider()
        p._session_id = "test-session"
        return p

    def test_no_goals_returns_trivial(self):
        p = self._provider()
        assert p._generate_task_id([]) == "trivial"

    def test_empty_goal_returns_trivial(self):
        p = self._provider()
        assert p._generate_task_id(["  "]) == "trivial"

    def test_task_id_is_deterministic(self):
        p = self._provider()
        id1 = p._generate_task_id(["实现用户认证模块"])
        id2 = p._generate_task_id(["实现用户认证模块"])
        assert id1 == id2

    def test_task_id_contains_hash(self):
        p = self._provider()
        task_id = p._generate_task_id(["实现用户认证模块"])
        # Should be "实现用户认证模块_<8char_hash>"
        assert "_" in task_id
        assert len(task_id.split("_")[-1]) == 8

    def test_task_id_truncates_long_goal(self):
        p = self._provider()
        long_goal = "实现用户认证模块" * 20
        task_id = p._generate_task_id([long_goal])
        # Truncated to 50 chars + "_" + 8 hash
        base = task_id.rsplit("_", 1)[0]
        assert len(base) <= 50


# ---------------------------------------------------------------------------
# snapshot_store — write / load
# ---------------------------------------------------------------------------

class TestSnapshotStoreWriteLoad:
    @pytest.fixture
    def store(self, tmp_path):
        return SnapshotStore(snapshot_dir=tmp_path, agent_identity="tester", ttl_days=7)

    def test_write_and_load(self, store):
        data = {
            "schema_version": 2,
            "session_id": "s1",
            "task_id": "test-task_abc12345",
            "finished": False,
            "task_goals": ["实现认证"],
            "checklist": [{"content": "JWT", "done": False}],
        }
        start_ms = 1746163590123
        path = store.write(start_ms, data)

        loaded = store.load(start_ms)
        assert loaded is not None
        assert loaded["session_id"] == "s1"
        assert loaded["task_id"] == "test-task_abc12345"
        assert loaded["finished"] is False

    def test_load_nonexistent_returns_none(self, store):
        assert store.load(9999999) is None


# ---------------------------------------------------------------------------
# snapshot_store — load_latest_unfinished
# ---------------------------------------------------------------------------

class TestLoadLatestUnfinished:
    @pytest.fixture
    def store(self, tmp_path):
        return SnapshotStore(snapshot_dir=tmp_path, agent_identity="tester", ttl_days=7)

    def _write(self, store, session_start_ms, finished, task_id="test_abc12345"):
        data = {
            "schema_version": 2,
            "session_id": f"s-{session_start_ms}",
            "task_id": task_id,
            "finished": finished,
            "task_goals": ["测试目标"],
            "checklist": [],
        }
        store.write(session_start_ms, data)

    def test_returns_none_when_no_snapshots(self, store):
        assert store.load_latest_unfinished() is None

    def test_skips_finished_snapshots(self, store):
        t = int(time.time() * 1000)
        self._write(store, t, finished=True)
        self._write(store, t + 1000, finished=True)
        assert store.load_latest_unfinished() is None

    def test_returns_unfinished_over_finished(self, store):
        t = int(time.time() * 1000)
        self._write(store, t, finished=True)
        time.sleep(0.01)
        self._write(store, t + 1000, finished=False)
        result = store.load_latest_unfinished()
        assert result is not None
        assert result["finished"] is False

    def test_returns_most_recent_unfinished_by_mtime(self, store):
        t = int(time.time() * 1000)
        # Write older first, newer second (reverse order to test mtime ordering)
        time.sleep(0.01)
        self._write(store, t + 1000, finished=False)
        time.sleep(0.01)
        self._write(store, t + 2000, finished=False)
        time.sleep(0.01)
        self._write(store, t + 3000, finished=False)
        result = store.load_latest_unfinished()
        # Most recent by mtime
        assert result["session_id"] == f"s-{t + 3000}"

    def test_backwards_compat_missing_finished_field(self, store):
        """Snapshots without finished field default to unfinished."""
        t = int(time.time() * 1000)
        data = {
            "schema_version": 1,
            "session_id": "old-s",
            "task_id": "old-task",
            # no "finished" field
            "task_goals": ["旧目标"],
            "checklist": [],
        }
        store.write(t, data)
        result = store.load_latest_unfinished()
        assert result is not None
        assert result["session_id"] == "old-s"
        # Defaults to unfinished (False)
        assert result.get("finished", False) is False


# ---------------------------------------------------------------------------
# provider — on_session_end writes task_id and finished
# ---------------------------------------------------------------------------

class TestProviderOnSessionEnd:
    @pytest.fixture
    def provider(self, tmp_path):
        p = SnapshotMemoryProvider(config={
            "snapshot_dir": str(tmp_path),
            "ttl_days": 7,
        })
        p._session_id = "session-end-test"
        p._session_start_time = time.time()
        p._agent_identity = "test-agent"
        p._ttl_days = 7
        p._max_tokens = 4000
        p._store = SnapshotStore(
            snapshot_dir=tmp_path,
            agent_identity="test-agent",
            ttl_days=7,
        )
        return p

    def test_writes_task_id_and_finished(self, provider):
        msgs = make_messages("任务：完成支付模块\n- [x] 调通支付 API\n- [ ] 对账逻辑")
        provider.on_session_end(msgs)

        snapshots = provider._store.list_snapshots()
        assert len(snapshots) == 1
        latest = provider._store.load_latest()
        assert latest["task_id"] != "trivial"
        assert latest["finished"] is False
        assert latest["schema_version"] == 2

    def test_trivial_session_writes_trivial_task_id(self, provider):
        msgs = make_messages("hi")
        provider.on_session_end(msgs)

        latest = provider._store.load_latest()
        assert latest["task_id"] == "trivial"

    def test_all_done_sets_finished_true(self, provider):
        msgs = make_messages(
            "任务：完成模块\n- [x] 完成 A\n- [x] 完成 B"
        )
        provider.on_session_end(msgs)

        latest = provider._store.load_latest()
        assert latest["finished"] is True

    def test_empty_messages_does_not_write(self, provider):
        provider.on_session_end([])
        assert provider._store.load_latest() is None


# ---------------------------------------------------------------------------
# provider — system_prompt_block loading strategy
# ---------------------------------------------------------------------------

class TestProviderSystemPromptBlock:
    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def provider(self, tmp_dir):
        p = SnapshotMemoryProvider(config={
            "snapshot_dir": str(tmp_dir),
            "ttl_days": 7,
        })
        p._session_id = "load-test-session"
        p._session_start_time = time.time()
        p._agent_identity = "test-agent"
        p._ttl_days = 7
        p._max_tokens = 4000
        p._store = SnapshotStore(
            snapshot_dir=tmp_dir,
            agent_identity="test-agent",
            ttl_days=7,
        )
        return p

    def _write_snapshot(self, provider, session_start_ms, finished, task_id="test_abc12345"):
        data = {
            "schema_version": 2,
            "session_id": f"s-{session_start_ms}",
            "task_id": task_id,
            "finished": finished,
            "task_goals": ["测试目标"],
            "checklist": [],
            "messages": [{"role": "user", "content": "hello"}],
        }
        provider._store.write(session_start_ms, data)

    def test_no_snapshot_returns_empty(self, provider):
        block = provider.system_prompt_block()
        assert block == ""

    def test_loads_unfinished_when_available(self, provider):
        t = int(time.time() * 1000)
        time.sleep(0.01)
        self._write_snapshot(provider, t, finished=True)      # older, finished
        time.sleep(0.01)
        self._write_snapshot(provider, t + 1000, finished=False)  # newer, unfinished
        block = provider.system_prompt_block()
        assert f"s-{t + 1000}" in block

    def test_falls_back_to_latest_by_mtime_when_all_finished(self, provider):
        t = int(time.time() * 1000)
        time.sleep(0.01)
        self._write_snapshot(provider, t, finished=True)
        time.sleep(0.01)
        self._write_snapshot(provider, t + 1000, finished=True)
        block = provider.system_prompt_block()
        # Should fall back to latest by mtime
        assert f"s-{t + 1000}" in block

    def test_block_contains_task_id_and_finished(self, provider):
        t = int(time.time() * 1000)
        self._write_snapshot(provider, t, finished=False, task_id="实现支付模块_abc12345")
        block = provider.system_prompt_block()
        assert "实现支付模块" in block
        assert "Done:" in block or "done:" in block
