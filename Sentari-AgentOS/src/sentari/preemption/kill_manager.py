from __future__ import annotations

from sentari.notifications.notifier import LogNotifier, NotificationEvent, Notifier
from sentari.pcb import AgentState


class KillManager:
    """Preemption & Kill Manager (FR-15, FR-16, FR-17).

    - terminate_quota_exhausted: an agent that has used up its budget can
      never make further progress, so it is moved to TERMINATED (a graceful
      end-of-budget outcome, distinct from a kernel-enforced KILLED).
    - kill: force-kill on blocked-timeout or as a deadlock victim (FR-16).

    Both paths release every resource the agent holds (FR-17) via the bound
    ResourceManager. `bind_resource_manager` exists because ResourceManager
    and KillManager depend on each other (kill releases resources; resource
    acquisition can trigger a kill) -- the kernel wires the cycle together
    after both are constructed.

    Both paths also fire a `NotificationEvent` through the configured
    `Notifier` (default: log-only) -- these are the two outcomes an
    operator would plausibly want to know about proactively, distinct from
    the syscall_log's record of every mediated call.
    """

    def __init__(self, scheduler, agent_repo, notifier: Notifier | None = None, reputation_repo=None):
        self._scheduler = scheduler
        self._repo = agent_repo
        self._resources = None
        self._notifier = notifier or LogNotifier()
        self._reputation = reputation_repo

    def bind_resource_manager(self, resource_manager) -> None:
        self._resources = resource_manager

    async def terminate_quota_exhausted(self, agent_id: str) -> None:
        pcb = self._scheduler.get(agent_id)
        if pcb.state in (AgentState.TERMINATED, AgentState.KILLED):
            return
        pcb.transition_to(AgentState.TERMINATED)
        self._repo.save(pcb)
        await self._scheduler.force_yield(agent_id)
        self._release_all(agent_id)
        if self._reputation is not None:
            self._reputation.record_quota_exhaustion(pcb.agent_type)
        await self._notifier.notify(
            NotificationEvent(
                event_type="quota_exhausted",
                agent_id=agent_id,
                reason="quota_exhausted",
                message=f"agent '{agent_id}' terminated: exhausted its quota ({pcb.quota_used}/{pcb.quota_total})",
                metadata={"quota_used": pcb.quota_used, "quota_total": pcb.quota_total},
            )
        )

    async def kill(self, agent_id: str, reason: str) -> None:
        pcb = self._scheduler.get(agent_id)
        if pcb.state in (AgentState.TERMINATED, AgentState.KILLED):
            return
        pcb.transition_to(AgentState.KILLED)
        self._repo.save(pcb)
        await self._scheduler.force_yield(agent_id)
        self._release_all(agent_id)
        if self._reputation is not None:
            self._reputation.record_kill(pcb.agent_type)
        await self._notifier.notify(
            NotificationEvent(
                event_type="agent_killed",
                agent_id=agent_id,
                reason=reason,
                message=f"agent '{agent_id}' was force-killed ({reason})",
            )
        )

    def _release_all(self, agent_id: str) -> None:
        if self._resources is not None:
            self._resources.release_all(agent_id)
