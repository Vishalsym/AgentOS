from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from sentari.pcb import AgentPCB, AgentState


class AgentRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create(self, pcb: AgentPCB) -> None:
        self._conn.execute(
            """INSERT INTO agents
               (agent_id, parent_id, state, priority, quota_total, quota_used, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pcb.agent_id, pcb.parent_id, pcb.state.value, pcb.priority,
                pcb.quota_total, pcb.quota_used, pcb.created_at, pcb.updated_at,
            ),
        )
        self._conn.commit()

    def save(self, pcb: AgentPCB) -> None:
        self._conn.execute(
            """UPDATE agents SET state=?, priority=?, quota_total=?, quota_used=?, updated_at=?
               WHERE agent_id=?""",
            (pcb.state.value, pcb.priority, pcb.quota_total, pcb.quota_used, pcb.updated_at, pcb.agent_id),
        )
        self._conn.commit()

    def get(self, agent_id: str) -> AgentPCB | None:
        row = self._conn.execute("SELECT * FROM agents WHERE agent_id=?", (agent_id,)).fetchone()
        return self._row_to_pcb(row) if row else None

    def list_by_state(self, state: AgentState) -> list[AgentPCB]:
        rows = self._conn.execute("SELECT * FROM agents WHERE state=?", (state.value,)).fetchall()
        return [self._row_to_pcb(r) for r in rows]

    def list_all(self) -> list[AgentPCB]:
        rows = self._conn.execute("SELECT * FROM agents").fetchall()
        return [self._row_to_pcb(r) for r in rows]

    @staticmethod
    def _row_to_pcb(row: sqlite3.Row) -> AgentPCB:
        return AgentPCB(
            agent_id=row["agent_id"],
            parent_id=row["parent_id"],
            state=AgentState(row["state"]),
            priority=row["priority"],
            quota_total=row["quota_total"],
            quota_used=row["quota_used"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class SyscallLogRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def log(self, agent_id: str, syscall_type: str, arguments: dict[str, Any], result: str) -> int:
        cur = self._conn.execute(
            """INSERT INTO syscall_log (agent_id, syscall_type, arguments, result, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (agent_id, syscall_type, json.dumps(arguments, default=str), result, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def list_for_agent(self, agent_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM syscall_log WHERE agent_id=? ORDER BY log_id", (agent_id,)
        ).fetchall()

    def list_all(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM syscall_log ORDER BY log_id").fetchall()


class ResourceRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def record_hold(self, resource_key: str, holder_agent_id: str) -> None:
        self._conn.execute(
            """INSERT INTO resource_allocation (resource_key, holder_agent_id, waiter_agent_id, created_at)
               VALUES (?, ?, NULL, ?)""",
            (resource_key, holder_agent_id, time.time()),
        )
        self._conn.commit()

    def record_wait(self, resource_key: str, waiter_agent_id: str) -> None:
        self._conn.execute(
            """INSERT INTO resource_allocation (resource_key, holder_agent_id, waiter_agent_id, created_at)
               VALUES (?, NULL, ?, ?)""",
            (resource_key, waiter_agent_id, time.time()),
        )
        self._conn.commit()

    def clear_for_resource(self, resource_key: str) -> None:
        self._conn.execute("DELETE FROM resource_allocation WHERE resource_key=?", (resource_key,))
        self._conn.commit()

    def clear_for_agent(self, agent_id: str) -> None:
        self._conn.execute(
            "DELETE FROM resource_allocation WHERE holder_agent_id=? OR waiter_agent_id=?",
            (agent_id, agent_id),
        )
        self._conn.commit()

    def list_all(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM resource_allocation").fetchall()


class ReputationRepo:
    """Backing store for reputation-driven adaptive admission control
    (novel mechanism #5, README/TRACEABILITY extension). Tracks outcomes
    per `agent_type`, a caller-declared label that -- unlike `agent_id` --
    is allowed to recur across many admissions."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def _ensure_row(self, agent_type: str) -> None:
        self._conn.execute(
            """INSERT INTO agent_reputation (agent_type, admissions, kills, quota_exhaustions, updated_at)
               VALUES (?, 0, 0, 0, ?)
               ON CONFLICT(agent_type) DO NOTHING""",
            (agent_type, time.time()),
        )

    def record_admission(self, agent_type: str) -> None:
        self._ensure_row(agent_type)
        self._conn.execute(
            "UPDATE agent_reputation SET admissions = admissions + 1, updated_at = ? WHERE agent_type = ?",
            (time.time(), agent_type),
        )
        self._conn.commit()

    def record_kill(self, agent_type: str) -> None:
        self._ensure_row(agent_type)
        self._conn.execute(
            "UPDATE agent_reputation SET kills = kills + 1, updated_at = ? WHERE agent_type = ?",
            (time.time(), agent_type),
        )
        self._conn.commit()

    def record_quota_exhaustion(self, agent_type: str) -> None:
        self._ensure_row(agent_type)
        self._conn.execute(
            "UPDATE agent_reputation SET quota_exhaustions = quota_exhaustions + 1, updated_at = ? "
            "WHERE agent_type = ?",
            (time.time(), agent_type),
        )
        self._conn.commit()

    def get(self, agent_type: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM agent_reputation WHERE agent_type = ?", (agent_type,)
        ).fetchone()

    def risk_score(self, agent_type: str) -> float:
        """Fraction of this type's past admissions that ended badly (killed
        or quota-exhausted), in [0.0, 1.0]. 0.0 for a type with no history
        -- "innocent until proven otherwise", not a default penalty."""
        row = self.get(agent_type)
        if row is None or row["admissions"] == 0:
            return 0.0
        bad = row["kills"] + row["quota_exhaustions"]
        return min(1.0, bad / row["admissions"])


class KnowledgeBaseRepo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM knowledge_base WHERE kb_key=?", (key,)).fetchone()
        return row["value"] if row else None

    def get_entry(self, key: str) -> sqlite3.Row | None:
        """Full row including writer, evidence, and timestamp -- for
        hallucination-interception verification (novel mechanism #3).
        `get()` above is unchanged/untouched so every existing caller keeps
        seeing exactly the bare value it always has."""
        return self._conn.execute("SELECT * FROM knowledge_base WHERE kb_key=?", (key,)).fetchone()

    def set(self, key: str, value: str, writer_agent_id: str, evidence: str | None = None) -> None:
        self._conn.execute(
            """INSERT INTO knowledge_base (kb_key, value, last_writer_agent_id, updated_at, evidence)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(kb_key) DO UPDATE SET
                   value=excluded.value,
                   last_writer_agent_id=excluded.last_writer_agent_id,
                   updated_at=excluded.updated_at,
                   evidence=excluded.evidence""",
            (key, value, writer_agent_id, time.time(), evidence),
        )
        self._conn.commit()
