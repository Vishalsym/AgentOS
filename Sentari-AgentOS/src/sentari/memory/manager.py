from __future__ import annotations

from typing import Any


class MemoryIsolationError(PermissionError):
    """Raised when an agent attempts to read or write another agent's
    isolated context (FR-10)."""


class MemoryManager:
    """Per-agent isolated context plus a syscall-mediated shared knowledge
    base (FR-10, FR-11). The isolated context is process-local (in-memory
    only, never persisted cross-agent); the knowledge base is the one region
    every agent can reach, and only via explicit memory_read/memory_write
    syscalls."""

    def __init__(self, kb_repo):
        self._contexts: dict[str, dict[str, Any]] = {}
        self._kb_repo = kb_repo

    def create_context(self, agent_id: str) -> None:
        self._contexts.setdefault(agent_id, {})

    def read_context(self, agent_id: str, requester_id: str) -> dict[str, Any]:
        if requester_id != agent_id:
            raise MemoryIsolationError(
                f"agent {requester_id} may not read agent {agent_id}'s isolated context"
            )
        return dict(self._contexts.get(agent_id, {}))

    def write_context(self, agent_id: str, requester_id: str, key: str, value: Any) -> None:
        if requester_id != agent_id:
            raise MemoryIsolationError(
                f"agent {requester_id} may not write agent {agent_id}'s isolated context"
            )
        self._contexts.setdefault(agent_id, {})[key] = value

    def kb_read(self, key: str) -> str | None:
        return self._kb_repo.get(key)

    def kb_read_entry(self, key: str) -> dict[str, Any] | None:
        """Full entry (value, writer, evidence, timestamp) for hallucination-
        interception verification (novel mechanism #3) -- a plain dict
        rather than the sqlite3.Row directly, so callers outside the
        persistence layer don't need to know about that type."""
        row = self._kb_repo.get_entry(key)
        if row is None:
            return None
        return {
            "value": row["value"],
            "writer_agent_id": row["last_writer_agent_id"],
            "evidence": row["evidence"],
            "updated_at": row["updated_at"],
        }

    def kb_write(self, agent_id: str, key: str, value: str, evidence: str | None = None) -> None:
        """`evidence` is an optional, caller-supplied justification for
        `value` -- e.g. the literal tool output it was derived from. Purely
        additive: omitting it (the default) behaves exactly as before."""
        self._kb_repo.set(key, value, agent_id, evidence=evidence)
