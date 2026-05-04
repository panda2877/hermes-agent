"""
Tasks Store — SQLite 任务状态追踪，与 CheckpointStore 共用同一 DB。

Schema:
  tasks(id, title, description, status, priority, assignee,
        mission_id, created_at, updated_at)
  agents(id, profile, name, role, capabilities, status, registered_at)

Status enum: backlog | inbox | assigned | in_progress | review | done
Priority enum: P0 | P1 | P2 | P3
Agent status: idle | busy | offline
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _get_db_path() -> Path:
    """获取 tasks.db 路径（公共看板目录，所有 profile 共用）。
    
    公共看板位于 HERMES_HOME/../kanban/（即 ~/.hermes/kanban/），
    不绑定任何 profile，所有 agent 共享同一份任务状态。
    """
    import os
    from pathlib import Path as P

    # 优先使用环境变量指定的全局 Hermes 根目录
    hermes_root = os.environ.get("HERMES_HOME_ROOT", "")
    if hermes_root:
        return P(hermes_root) / "kanban" / "tasks.db"

    # 使用硬编码的全局 Hermes 根目录（不依赖 HOME/PROFILE 环境变量）
    return Path("/home/agentuser/.hermes") / "kanban" / "tasks.db"


_TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT,
    status      TEXT NOT NULL DEFAULT 'backlog',
    priority    TEXT NOT NULL DEFAULT 'P2',
    assignee    TEXT,
    mission_id  TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status    ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_assignee ON tasks(assignee);
CREATE INDEX IF NOT EXISTS idx_tasks_mission  ON tasks(mission_id);

CREATE TABLE IF NOT EXISTS agents (
    id            TEXT PRIMARY KEY,
    profile       TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'generalist',
    capabilities  TEXT NOT NULL DEFAULT '[]',
    status        TEXT NOT NULL DEFAULT 'idle',
    registered_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agents_profile ON agents(profile);
CREATE INDEX IF NOT EXISTS idx_agents_status  ON agents(status);
"""


def _ensure_db() -> None:
    """确保 tasks.db 和表结构存在。"""
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.executescript(_TASKS_SCHEMA)
    conn.commit()
    conn.close()


def create_task(
    task_id: str,
    title: str,
    description: str = "",
    status: str = "backlog",
    priority: str = "P2",
    assignee: str = "",
    mission_id: str = "",
) -> str:
    """创建新任务，返回 task_id。"""
    _ensure_db()
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO tasks
            (id, title, description, status, priority, assignee, mission_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, title, description, status, priority, assignee, mission_id, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return task_id


def update_task_status(task_id: str, status: str) -> bool:
    """更新任务状态，返回是否成功。"""
    _ensure_db()
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        cur = conn.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, task_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_task(task_id: str) -> Optional[dict]:
    """获取单个任务，返回 dict 或 None。"""
    _ensure_db()
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        row = conn.execute(
            "SELECT id, title, description, status, priority, assignee, mission_id, created_at, updated_at FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "status": row[3],
            "priority": row[4],
            "assignee": row[5],
            "mission_id": row[6],
            "created_at": row[7],
            "updated_at": row[8],
        }
    finally:
        conn.close()


def list_tasks(status: Optional[str] = None, mission_id: Optional[str] = None) -> list[dict]:
    """列出任务，可按 status 或 mission_id 过滤。"""
    _ensure_db()
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        query = "SELECT id, title, description, status, priority, assignee, mission_id, created_at, updated_at FROM tasks WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if mission_id:
            query += " AND mission_id = ?"
            params.append(mission_id)
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": r[0], "title": r[1], "description": r[2],
                "status": r[3], "priority": r[4], "assignee": r[5],
                "mission_id": r[6], "created_at": r[7], "updated_at": r[8],
            }
            for r in rows
        ]
    finally:
        conn.close()


def upsert_from_obsidian(task_id: str, title: str, status: str, priority: str = "P2", mission_id: str = "") -> None:
    """从 Obsidian frontmatter 同步任务（存在则更新，不存在则创建）。"""
    _ensure_db()
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        conn.execute(
            """
            INSERT INTO tasks (id, title, status, priority, mission_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                status = excluded.status,
                priority = excluded.priority,
                mission_id = excluded.mission_id,
                updated_at = excluded.updated_at
            """,
            (task_id, title, status, priority, mission_id, now, now),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------

def register_agent(
    profile: str,
    name: str,
    role: str = "generalist",
    capabilities: list[str] | None = None,
) -> str:
    """注册 agent，返回 agent_id。"""
    _ensure_db()
    now = datetime.now(timezone.utc).isoformat()
    agent_id = f"agent-{profile}"
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        conn.execute(
            """
            INSERT INTO agents (id, profile, name, role, capabilities, status, registered_at)
            VALUES (?, ?, ?, ?, ?, 'idle', ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                role = excluded.role,
                capabilities = excluded.capabilities,
                status = 'idle'
            """,
            (agent_id, profile, name, role, json.dumps(capabilities or [], ensure_ascii=False), now),
        )
        conn.commit()
    finally:
        conn.close()
    return agent_id


def update_agent_status(profile: str, status: str) -> bool:
    """更新 agent 状态（idle/busy/offline）。"""
    _ensure_db()
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        cur = conn.execute(
            "UPDATE agents SET status = ? WHERE profile = ?",
            (status, profile),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_agent(profile: str) -> Optional[dict]:
    """获取 agent 信息。"""
    _ensure_db()
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        row = conn.execute(
            "SELECT id, profile, name, role, capabilities, status, registered_at FROM agents WHERE profile = ?",
            (profile,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "profile": row[1], "name": row[2], "role": row[3],
            "capabilities": json.loads(row[4]),
            "status": row[5], "registered_at": row[6],
        }
    finally:
        conn.close()


def list_agents(status: Optional[str] = None) -> list[dict]:
    """列出所有 agent，可按状态过滤。"""
    _ensure_db()
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        query = "SELECT id, profile, name, role, capabilities, status, registered_at FROM agents WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": r[0], "profile": r[1], "name": r[2], "role": r[3],
                "capabilities": json.loads(r[4]),
                "status": r[5], "registered_at": r[6],
            }
            for r in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Task Assignment
# ---------------------------------------------------------------------------

def assign_task(task_id: str, assignee: str) -> bool:
    """分派任务给 agent，同时更新任务状态为 assigned。"""
    _ensure_db()
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        cur = conn.execute(
            "UPDATE tasks SET assignee = ?, status = 'assigned', updated_at = ? WHERE id = ?",
            (assignee, now, task_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def unassign_task(task_id: str) -> bool:
    """取消任务分派（保留 assignee 历史）。"""
    _ensure_db()
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        cur = conn.execute(
            "UPDATE tasks SET assignee = '', status = 'backlog', updated_at = ? WHERE id = ?",
            (now, task_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_tasks_by_agent(assignee: str, status: Optional[str] = None) -> list[dict]:
    """获取某 agent 负责的任务。"""
    _ensure_db()
    conn = sqlite3.connect(str(_get_db_path()), timeout=10)
    try:
        query = """
            SELECT id, title, description, status, priority, assignee, mission_id, created_at, updated_at
            FROM tasks WHERE assignee = ?
        """
        params: list = [assignee]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": r[0], "title": r[1], "description": r[2],
                "status": r[3], "priority": r[4], "assignee": r[5],
                "mission_id": r[6], "created_at": r[7], "updated_at": r[8],
            }
            for r in rows
        ]
    finally:
        conn.close()

