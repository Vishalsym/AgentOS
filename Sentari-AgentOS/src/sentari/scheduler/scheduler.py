from __future__ import annotations

import asyncio
import time
from enum import StrEnum

from sentari.pcb import AgentPCB, AgentState, DuplicateAgentError


class SchedulingPolicy(StrEnum):
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"


DEFAULT_AGING_THRESHOLD_SECONDS = 2.0
DEFAULT_AGING_BOOST = 1

# Reputation-driven admission control (novel mechanism #5): don't penalize
# a type until it has a meaningful sample, and cap how much a bad track
# record can worsen priority so one recurring troublemaker can't be pushed
# to effectively-infinite deprioritization.
DEFAULT_MIN_REPUTATION_SAMPLE = 3
DEFAULT_MAX_REPUTATION_PENALTY = 20


class Scheduler:
    """Turn-based scheduler (FR-4, FR-5, FR-6).
    asyncio is cooperative, not preemptive at the OS level, so "the CPU"
    here is a single admission slot: only one agent may be RUNNING at a
    time, and a "turn" is exactly one syscall's execution -- an agent must
    re-request a turn (and re-enter the priority/aging queue) for every
    syscall it makes, which is what bounds how long any agent can hold the
    CPU without yielding (FR-5). True forced preemption of an in-flight
    syscall is layered on top of this by the syscall layer via
    asyncio.wait_for + task cancellation.
    """

    def __init__(
        self,
        agent_repo,
        policy: SchedulingPolicy = SchedulingPolicy.PRIORITY,
        aging_threshold_seconds: float = DEFAULT_AGING_THRESHOLD_SECONDS,
        aging_boost: int = DEFAULT_AGING_BOOST,
        reputation_repo=None,
        min_reputation_sample: int = DEFAULT_MIN_REPUTATION_SAMPLE,
        max_reputation_penalty: int = DEFAULT_MAX_REPUTATION_PENALTY,
    ):
        self._repo = agent_repo
        self.policy = policy
        self._aging_threshold = aging_threshold_seconds
        self._aging_boost = aging_boost
        self._reputation = reputation_repo
        self._min_reputation_sample = min_reputation_sample
        self._max_reputation_penalty = max_reputation_penalty
        self._agents: dict[str, AgentPCB] = {}
        self._ready: list[str] = []
        self._enqueued_at: dict[str, float] = {}
        self._current: str | None = None
        self._cv = asyncio.Condition()
        self.last_reputation_adjustment: dict[str, int] = {}
        # Guards against concurrent acquire_turn() calls for the *same*
        # agent_id (e.g. an agent spawning several children in parallel,
        # each a separate kernel.syscall() call for the same parent id).
        # _ready holds at most one entry per agent_id, so without this,
        # every concurrent caller past the first would find its own
        # agent_id "already queued", skip re-adding itself, and then --
        # once the first caller's turn consumed and removed that single
        # entry -- have nothing left to ever become eligible again
        # (permanently stranded). This flag makes additional same-agent
        # callers wait for the *entire* acquire/dispatch/release cycle to
        # finish, not just the acquire step, before they enter the shared
        # ready-queue machinery themselves.
        self._turn_in_use: dict[str, bool] = {}

    def admit(self, pcb: AgentPCB) -> None:
        """Register the agent. It only enters the active dispatch
        queue once it actually requests a turn via acquire_turn -- an
        admitted-but-idle agent must never be able to block others from
        being selected, no matter how high its priority.

        Reputation-driven adjustment (novel mechanism #5): if this PCB's
        `agent_type` has enough prior admissions on record and a nonzero
        risk score (fraction that were killed or quota-exhausted), its
        starting priority is proportionally worsened *before* it ever
        enters the ready queue -- a feedback loop from the kernel's own
        audit history to admission control, not a value a caller sets
        directly. `last_reputation_adjustment` records what happened (0 if
        no adjustment), so a caller/test can see the decision was made,
        not just its effect."""
        if pcb.agent_id in self._agents:
            raise DuplicateAgentError(
                f"agent_id '{pcb.agent_id}' is already registered "
                f"(state={self._agents[pcb.agent_id].state.value}) -- admit a fresh id instead"
            )

        penalty = 0
        if self._reputation is not None:
            row = self._reputation.get(pcb.agent_type)
            if row is not None and row["admissions"] >= self._min_reputation_sample:
                risk = self._reputation.risk_score(pcb.agent_type)
                penalty = round(risk * self._max_reputation_penalty)
                pcb.priority += penalty
            self._reputation.record_admission(pcb.agent_type)
        self.last_reputation_adjustment[pcb.agent_id] = penalty

        pcb.transition_to(AgentState.READY)
        self._agents[pcb.agent_id] = pcb
        self._repo.create(pcb)

    def get(self, agent_id: str) -> AgentPCB:
        return self._agents[agent_id]

    @property
    def current_agent_id(self) -> str | None:
        """The agent currently holding the single dispatch slot ("the CPU"), if any."""
        return self._current

    @property
    def ready_queue(self) -> list[str]:
        """Agent ids currently contending for the next turn, in queue order."""
        return list(self._ready)

    def _apply_aging(self) -> None:
        now = time.monotonic()
        for agent_id in self._ready:
            waited = now - self._enqueued_at[agent_id]
            if waited > self._aging_threshold:
                pcb = self._agents[agent_id]
                if pcb.priority > 0:
                    pcb.priority -= self._aging_boost
                self._enqueued_at[agent_id] = now

    def _peek_next(self) -> str | None:
        if not self._ready:
            return None
        self._apply_aging()
        if self.policy is SchedulingPolicy.PRIORITY:
            return min(
                self._ready, key=lambda aid: (self._agents[aid].priority, self._enqueued_at[aid])
            )
        return self._ready[0]

    async def acquire_turn(self, agent_id: str) -> None:
        pcb = self._agents[agent_id]

        async with self._cv:
            # Wait for any other concurrent acquire_turn call for this same
            # agent_id to fully finish its turn (through release_turn/
            # force_yield) before this one touches the shared ready queue.
            await self._cv.wait_for(lambda: not self._turn_in_use.get(agent_id, False))
            self._turn_in_use[agent_id] = True

            if agent_id not in self._ready and self._current != agent_id:
                self._ready.append(agent_id)
                self._enqueued_at[agent_id] = time.monotonic()

            def can_run() -> bool:
                if pcb.state in (AgentState.KILLED, AgentState.TERMINATED):
                    return True
                if self._current is not None and self._current != agent_id:
                    return False
                return self._peek_next() == agent_id

            await self._cv.wait_for(can_run)

            if pcb.state in (AgentState.KILLED, AgentState.TERMINATED):
                if agent_id in self._ready:
                    self._ready.remove(agent_id)
                self._turn_in_use[agent_id] = False
                self._cv.notify_all()
                return

            if agent_id in self._ready:
                self._ready.remove(agent_id)
            self._current = agent_id
            pcb.transition_to(AgentState.RUNNING)
            self._repo.save(pcb)

    async def release_turn(self, agent_id: str, next_state: AgentState = AgentState.READY) -> None:
        """Give up the CPU. This only updates PCB state -- it does NOT
        re-enter the agent into the active dispatch queue. READY here means
        "available to be scheduled next time its driver asks", not "still
        contending right now"; re-joining the dispatch queue happens only
        via a fresh acquire_turn call. Conflating the two would let a
        released agent perpetually win priority selection against agents
        that are actually waiting, even though nothing is asking for it."""
        async with self._cv:
            pcb = self._agents[agent_id]
            if self._current == agent_id:
                self._current = None
            if pcb.state not in (AgentState.KILLED, AgentState.TERMINATED):
                pcb.transition_to(next_state)
                self._repo.save(pcb)
            self._turn_in_use[agent_id] = False
            self._cv.notify_all()

    async def force_yield(self, agent_id: str) -> None:
        """Unconditionally drop this agent from the CPU slot and the
        dispatch queue, regardless of what it was doing. Used by the Kill
        Manager (FR-16): a kill can happen while an agent is RUNNING,
        BLOCKED, or merely queued, and in every case the scheduler must
        stop treating it as a live contender so other agents aren't stuck
        behind a killed one forever."""
        async with self._cv:
            if self._current == agent_id:
                self._current = None
            if agent_id in self._ready:
                self._ready.remove(agent_id)
            self._turn_in_use[agent_id] = False
            self._cv.notify_all()
