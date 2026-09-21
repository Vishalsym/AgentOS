from __future__ import annotations

from typing import Any

from sentari.deadlock.detector import DeadlockDetector
from sentari.memory.manager import MemoryManager
from sentari.pcb import AgentPCB
from sentari.persistence.db import connect
from sentari.persistence.repositories import (
    AgentRepo,
    KnowledgeBaseRepo,
    ReputationRepo,
    ResourceRepo,
    SyscallLogRepo,
)
from sentari.preemption.kill_manager import KillManager
from sentari.providers.mock import MockProvider
from sentari.scheduler.scheduler import Scheduler, SchedulingPolicy
from sentari.syscalls.layer import SyscallLayer, SyscallRequest, SyscallResponse, SyscallType
from sentari.syscalls.resource_manager import ResourceManager

DEFAULT_CHILD_QUOTA_SHARE = 0.5


class Kernel:
    """Wires the PCB store, scheduler, syscall layer, memory manager,
    deadlock detector and persistence into a single async API. This is the
    embedding point for wrapping an external agent framework (LangGraph,
    CrewAI): route that framework's tool execution through `Kernel.syscall`
    instead of calling tools directly."""

    def __init__(
        self,
        db_path: str = ":memory:",
        provider: Any = None,
        policy: SchedulingPolicy = SchedulingPolicy.PRIORITY,
        notifier: Any = None,
        kb_verifier: Any = None,
        kb_verification_strict: bool = False,
        probabilistic_banker: Any = None,
        token_aware_quota: bool = False,
    ):
        self._conn = connect(db_path)
        self.agent_repo = AgentRepo(self._conn)
        self.syscall_log_repo = SyscallLogRepo(self._conn)
        self.resource_repo = ResourceRepo(self._conn)
        self.kb_repo = KnowledgeBaseRepo(self._conn)
        self.reputation_repo = ReputationRepo(self._conn)

        self.scheduler = Scheduler(self.agent_repo, policy=policy, reputation_repo=self.reputation_repo)
        self.memory = MemoryManager(self.kb_repo)
        self.detector = DeadlockDetector(
            priority_fn=lambda aid: self.scheduler.get(aid).priority,
            value_fn=lambda aid: self.scheduler.get(aid).task_value,
        )
        self.kill_manager = KillManager(
            self.scheduler, self.agent_repo, notifier=notifier, reputation_repo=self.reputation_repo
        )
        self.resources = ResourceManager(
            self.resource_repo,
            self.detector,
            self.scheduler,
            self.kill_manager,
            probabilistic_banker=probabilistic_banker,
        )
        self.kill_manager.bind_resource_manager(self.resources)

        self.provider = provider or MockProvider()
        self.syscalls = SyscallLayer(
            scheduler=self.scheduler,
            memory_manager=self.memory,
            resource_manager=self.resources,
            syscall_log_repo=self.syscall_log_repo,
            provider=self.provider,
            spawn_fn=self._spawn_child,
            kill_manager=self.kill_manager,
            kb_verifier=kb_verifier,
            kb_verification_strict=kb_verification_strict,
            token_aware_quota=token_aware_quota,
        )

    def admit(
        self,
        priority: int,
        quota_total: int,
        parent_id: str | None = None,
        agent_id: str | None = None,
        agent_type: str | None = None,
        task_value: float | None = None,
        declared_resource_claims: dict[str, float] | None = None,
    ) -> AgentPCB:
        pcb = AgentPCB.new(
            priority=priority,
            quota_total=quota_total,
            parent_id=parent_id,
            agent_id=agent_id,
            agent_type=agent_type,
            declared_resource_claims=declared_resource_claims,
            task_value=task_value,
        )
        self.scheduler.admit(pcb)
        self.memory.create_context(pcb.agent_id)
        return pcb

    def _spawn_child(self, parent_id: str, arguments: dict[str, Any]) -> AgentPCB:
        parent = self.scheduler.get(parent_id)
        share = arguments.get("quota_share", DEFAULT_CHILD_QUOTA_SHARE)
        child_quota = max(1, int(parent.quota_remaining * share))
        return self.admit(
            priority=parent.priority,
            quota_total=child_quota,
            parent_id=parent_id,
            agent_type=parent.agent_type,
        )

    async def syscall(
        self, agent_id: str, syscall_type: SyscallType, arguments: dict[str, Any] | None = None
    ) -> SyscallResponse:
        request = SyscallRequest(agent_id=agent_id, syscall_type=syscall_type, arguments=arguments or {})
        return await self.syscalls.on_syscall(request)

    async def interrupt(self, agent_id: str, reason: str = "human_interrupt") -> bool:
        """Bounded-latency human interrupt (novel mechanism #4): cancel
        `agent_id`'s in-flight syscall right now and force-kill it,
        independent of the configured execution_timeout. See
        SyscallLayer.interrupt for the full contract."""
        return await self.syscalls.interrupt(agent_id, reason=reason)

    def close(self) -> None:
        self._conn.close()
