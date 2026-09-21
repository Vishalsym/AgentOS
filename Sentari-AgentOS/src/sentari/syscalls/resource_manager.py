from __future__ import annotations

import asyncio

from sentari.deadlock.detector import DeadlockDetector
from sentari.deadlock.probabilistic_banker import ResourceRequestUnsafeError
from sentari.pcb import AgentState


class ResourceManager:
    """Mediates exclusive access to shared resources requested through
    tool_call's `resource_key` argument.

    Acquiring a resource already held by another agent registers a wait-for
    edge (FR-12) and triggers incremental cycle detection (FR-13); a
    detected cycle results in the least-important agent in it being killed
    (FR-14), which in turn releases everything that victim held (FR-17).
    """

    def __init__(
        self, resource_repo, detector: DeadlockDetector, scheduler, kill_manager, probabilistic_banker=None
    ):
        self._repo = resource_repo
        self._detector = detector
        self._scheduler = scheduler
        self._kill_manager = kill_manager
        self._banker = probabilistic_banker
        self._held: dict[str, str] = {}
        self._waiters: dict[str, list[tuple[str, asyncio.Event]]] = {}

    async def acquire(self, agent_id: str, resource_key: str) -> None:
        holder = self._held.get(resource_key)
        if holder is None or holder == agent_id:
            self._grant(resource_key, agent_id)
            return

        victim_id = self._detector.add_wait_edge(agent_id, holder)

        if victim_id is not None:
            await self._kill_manager.kill(victim_id, reason="deadlock_victim")
            if victim_id == agent_id:
                raise RuntimeError(f"agent {agent_id} was selected as the deadlock victim")
            new_holder = self._held.get(resource_key)
            if new_holder is None or new_holder == agent_id:
                # killing the victim freed exactly the resource we wanted.
                self._grant(resource_key, agent_id)
                return
            # victim wasn't the direct holder (a longer cycle) -- still
            # need to wait for the actual holder, fall through below.

        # Proactive avoidance (novel mechanism #2), opt-in: no cycle exists
        # yet (the reactive detector above would already have caught and
        # resolved one), but if the holder has declared a high enough
        # estimated chance of eventually wanting something we already hold,
        # refuse this wait now rather than let a likely AB-BA cycle form
        # and rely on the reactive detector to clean it up afterward.
        if self._banker is not None:
            holder_pcb = self._scheduler.get(holder)
            held_by_waiter = {rk for rk, aid in self._held.items() if aid == agent_id}
            check = self._banker.check(
                agent_id, holder, resource_key, holder_pcb.declared_resource_claims, held_by_waiter
            )
            if not check.safe:
                self._detector.graph.remove_edges_for(agent_id)
                raise ResourceRequestUnsafeError(check.reason)

        self._repo.record_wait(resource_key, agent_id)
        event = asyncio.Event()
        self._waiters.setdefault(resource_key, []).append((agent_id, event))

        # Release the CPU turn while blocked (RUNNING -> BLOCKED). This
        # matters: if a blocked agent kept holding the scheduler's single
        # dispatch slot, the peer it's waiting on could never itself get
        # dispatched to make the reciprocal wait-edge -- the cycle would
        # never even form, let alone get detected.
        await self._scheduler.release_turn(agent_id, AgentState.BLOCKED)
        await event.wait()

        pcb = self._scheduler.get(agent_id)
        if pcb.state is AgentState.KILLED:
            raise RuntimeError(f"agent {agent_id} was killed while waiting for {resource_key}")

        # Resource granted (agent is READY again, see _grant) -- rejoin the
        # dispatch queue for a fresh turn before resuming execution.
        await self._scheduler.acquire_turn(agent_id)

    def _grant(self, resource_key: str, agent_id: str) -> None:
        self._held[resource_key] = agent_id
        self._repo.record_hold(resource_key, agent_id)
        self._detector.graph.remove_edges_for(agent_id)
        pcb = self._scheduler.get(agent_id)
        if pcb.state is AgentState.BLOCKED:
            pcb.transition_to(AgentState.READY)

    def release(self, agent_id: str, resource_key: str) -> None:
        if self._held.get(resource_key) != agent_id:
            return
        del self._held[resource_key]
        self._repo.clear_for_resource(resource_key)
        waiters = self._waiters.get(resource_key, [])
        if waiters:
            next_agent_id, event = waiters.pop(0)
            self._grant(resource_key, next_agent_id)
            event.set()

    def release_all(self, agent_id: str) -> None:
        """Release every resource an agent holds and wake any of its own
        pending waits, so a killed agent's acquire() call unblocks instead
        of hanging forever (FR-17)."""
        for resource_key, holder in list(self._held.items()):
            if holder == agent_id:
                self.release(agent_id, resource_key)
        for resource_key, waiters in list(self._waiters.items()):
            remaining = []
            for waiter_id, event in waiters:
                if waiter_id == agent_id:
                    event.set()
                else:
                    remaining.append((waiter_id, event))
            self._waiters[resource_key] = remaining
        self._detector.graph.remove_edges_for(agent_id)
        self._repo.clear_for_agent(agent_id)
