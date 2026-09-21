from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class DuplicateAgentError(ValueError):
    """Raised when admitting an agent_id that is already registered. A
    process, once admitted, occupies its id for the lifetime of the kernel
    -- even after it's TERMINATED/KILLED -- exactly like a PID an OS won't
    silently hand to two processes at once. Callers that want to "restart"
    a killed agent must admit a fresh id (or restart the kernel), not reuse
    the old one; without this check the reuse would instead surface as an
    opaque sqlite3.IntegrityError from the agents table's PRIMARY KEY."""


class AgentState(StrEnum):
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    TERMINATED = "TERMINATED"
    KILLED = "KILLED"


# Legal state transitions (FR-2), enforced by AgentPCB.transition_to.
_ALLOWED_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.NEW: {AgentState.READY},
    AgentState.READY: {AgentState.RUNNING, AgentState.TERMINATED, AgentState.KILLED},
    AgentState.RUNNING: {
        AgentState.READY,  # preempted (FR-15)
        AgentState.BLOCKED,  # waiting on a resource held by a peer
        AgentState.TERMINATED,  # task complete / quota exhausted gracefully
        AgentState.KILLED,  # kernel-enforced kill
    },
    AgentState.BLOCKED: {
        AgentState.READY,  # resource granted -- must re-arbitrate for a CPU turn
        AgentState.KILLED,
    },
    AgentState.TERMINATED: set(),
    AgentState.KILLED: set(),
}


@dataclass
class AgentPCB:
    """Process Control Block for an agent (FR-1)."""

    agent_id: str
    priority: int
    quota_total: int
    quota_used: int = 0
    parent_id: str | None = None
    state: AgentState = AgentState.NEW
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Reputation-tracking label (novel mechanism #5) -- unlike agent_id,
    # this is allowed to recur across many admissions (e.g. every worker
    # spawned from the same template), so historical outcomes for the
    # *type* can inform admission control for future instances of it.
    # Defaults to agent_id, so existing callers that never set it behave
    # exactly as before (each agent is its own unique, one-off "type").
    agent_type: str = ""
    # Semantic task-value score (novel mechanism #1), in [0.0, 1.0] or None
    # for "no opinion". Optionally set at admission time -- typically from
    # a one-off LLM/classifier judgment of how costly losing this agent's
    # work would be (see deadlock/semantic_scoring.py) -- and read back
    # synchronously by DeadlockDetector at victim-selection time. Deliberately
    # NOT computed inline during deadlock detection: that path is
    # synchronous and benchmarked at sub-millisecond resolution times: an
    # LLM call there would make the detector itself the slow part of the
    # kernel it's supposed to keep fast.
    task_value: float | None = None
    # Probabilistic Banker's Algorithm claims (novel mechanism #2):
    # resource_key -> estimated probability [0.0, 1.0] this agent will
    # eventually request it, even though it doesn't hold or need it yet.
    # Optional and empty by default -- an agent that declares nothing is
    # simply invisible to the proactive avoidance check (which then only
    # ever sees risk 0.0 for it), not treated as risky or safe by default.
    declared_resource_claims: dict[str, float] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        priority: int,
        quota_total: int,
        parent_id: str | None = None,
        agent_id: str | None = None,
        agent_type: str | None = None,
        task_value: float | None = None,
        declared_resource_claims: dict[str, float] | None = None,
    ) -> AgentPCB:
        resolved_id = agent_id or uuid.uuid4().hex
        return cls(
            agent_id=resolved_id,
            priority=priority,
            quota_total=quota_total,
            parent_id=parent_id,
            agent_type=agent_type or resolved_id,
            task_value=task_value,
            declared_resource_claims=declared_resource_claims or {},
        )

    @property
    def quota_remaining(self) -> int:
        return max(0, self.quota_total - self.quota_used)

    def transition_to(self, new_state: AgentState) -> None:
        if new_state == self.state:
            return
        allowed = _ALLOWED_TRANSITIONS[self.state]
        if new_state not in allowed:
            raise ValueError(
                f"illegal transition for agent {self.agent_id}: {self.state} -> {new_state}"
            )
        self.state = new_state
        self.updated_at = time.time()
