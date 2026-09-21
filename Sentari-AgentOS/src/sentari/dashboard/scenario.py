"""The same scripted scenario as scripts/demo.py, adapted to narrate itself
into a DashboardState instead of printing with rich -- driven by the real
kernel, not fake/staged data."""

from __future__ import annotations

import asyncio

from sentari.kernel import Kernel
from sentari.pcb import AgentState
from sentari.providers.mock import MockProvider
from sentari.syscalls.layer import SyscallLayer, SyscallRequest, SyscallType

from .state import DashboardState, build_default_notifier

STEP_PAUSE = 1.6


async def run_scenario(state: DashboardState) -> None:
    kernel = Kernel(provider=MockProvider(), notifier=build_default_notifier())
    state.kernel = kernel

    await state.narrate("Admitting three agents: alpha, beta, gamma.", "admission")
    alpha = kernel.admit(priority=1, quota_total=6, agent_id="alpha")
    beta = kernel.admit(priority=5, quota_total=6, agent_id="beta")
    gamma = kernel.admit(priority=10, quota_total=3, agent_id="gamma")
    await asyncio.sleep(STEP_PAUSE)

    await state.narrate("alpha makes a normal, mediated tool call.", "syscall")
    await kernel.syscall(alpha.agent_id, SyscallType.TOOL_CALL, {"prompt": "plan the next step"})
    await asyncio.sleep(STEP_PAUSE)

    await state.narrate("alpha grabs resource X, beta grabs resource Y.", "deadlock-setup")
    await kernel.resources.acquire(alpha.agent_id, "X")
    await kernel.resources.acquire(beta.agent_id, "Y")
    await asyncio.sleep(STEP_PAUSE)

    await state.narrate(
        "Now alpha wants Y (held by beta) and beta wants X (held by alpha), at the same time.",
        "deadlock",
    )

    async def alpha_wants_y() -> None:
        await kernel.scheduler.acquire_turn(alpha.agent_id)
        await kernel.resources.acquire(alpha.agent_id, "Y")

    async def beta_wants_x() -> None:
        await kernel.scheduler.acquire_turn(beta.agent_id)
        await kernel.resources.acquire(beta.agent_id, "X")

    await asyncio.gather(alpha_wants_y(), beta_wants_x(), return_exceptions=True)

    killed = [p.agent_id for p in kernel.agent_repo.list_all() if p.state is AgentState.KILLED]
    victim = killed[0] if killed else "?"
    survivor = "beta" if victim == "alpha" else "alpha"
    await state.set_cycle_notice({"a": "alpha", "b": "beta", "victim": victim, "survivor": survivor})
    await state.log_event(
        "KERNEL",
        f"deadlock cycle alpha<->beta detected -- killed {victim} (lower priority); {survivor} continues",
        "danger",
    )
    await kernel.scheduler.release_turn(survivor)
    await asyncio.sleep(STEP_PAUSE)

    await state.narrate("A new agent, runaway, calls a tool that never returns in time.", "preempt")
    runaway = kernel.admit(priority=3, quota_total=6, agent_id="runaway")
    short_timeout_layer = SyscallLayer(
        scheduler=kernel.scheduler,
        memory_manager=kernel.memory,
        resource_manager=kernel.resources,
        syscall_log_repo=kernel.syscall_log_repo,
        provider=kernel.provider,
        spawn_fn=kernel._spawn_child,
        kill_manager=kernel.kill_manager,
        execution_timeout=0.2,
    )

    async def stuck_tool() -> None:
        await asyncio.sleep(5)

    await short_timeout_layer.on_syscall(
        SyscallRequest(agent_id=runaway.agent_id, syscall_type=SyscallType.TOOL_CALL, arguments={"fn": stuck_tool})
    )
    await state.log_event(
        "KERNEL", "runaway missed its 0.2s time budget -- preempted back to READY, not killed", "warn"
    )
    await asyncio.sleep(STEP_PAUSE)

    await state.narrate(f"gamma has a tiny quota ({gamma.quota_total}) -- burning through it.", "quota")
    for _ in range(gamma.quota_total + 1):
        await kernel.syscall(gamma.agent_id, SyscallType.YIELD)
    await state.log_event("KERNEL", "gamma exhausted its quota -- terminated gracefully", "dim")
    await asyncio.sleep(STEP_PAUSE)

    await state.finish()
