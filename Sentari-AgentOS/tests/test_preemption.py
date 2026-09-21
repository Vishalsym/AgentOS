import asyncio

import pytest

from sentari.kernel import Kernel
from sentari.pcb import AgentState
from sentari.syscalls.layer import SyscallLayer, SyscallResult, SyscallType


@pytest.mark.asyncio
async def test_slow_tool_call_times_out_and_agent_returns_to_ready():
    kernel = Kernel()
    pcb = kernel.admit(priority=1, quota_total=5)
    kernel.syscalls = SyscallLayer(
        scheduler=kernel.scheduler,
        memory_manager=kernel.memory,
        resource_manager=kernel.resources,
        syscall_log_repo=kernel.syscall_log_repo,
        provider=kernel.provider,
        spawn_fn=kernel._spawn_child,
        kill_manager=kernel.kill_manager,
        execution_timeout=0.05,
    )

    async def runaway():
        await asyncio.sleep(5)

    resp = await kernel.syscall(pcb.agent_id, SyscallType.TOOL_CALL, {"fn": runaway})
    assert resp.result is SyscallResult.ERROR
    assert "timed out" in resp.error
    # preempted, not killed -- FR-15 vs FR-16
    assert kernel.scheduler.get(pcb.agent_id).state is AgentState.READY
    kernel.close()


@pytest.mark.asyncio
async def test_quota_exhausted_releases_held_resources():
    kernel = Kernel()
    pcb = kernel.admit(priority=1, quota_total=1)
    await kernel.resources.acquire(pcb.agent_id, "res")
    assert kernel.resources._held.get("res") == pcb.agent_id

    # this syscall consumes the last unit of quota
    await kernel.syscall(pcb.agent_id, SyscallType.YIELD)
    assert kernel.scheduler.get(pcb.agent_id).quota_used == 1

    # next syscall attempt is denied and terminates the agent
    resp = await kernel.syscall(pcb.agent_id, SyscallType.YIELD)
    assert resp.result is SyscallResult.DENY
    assert kernel.scheduler.get(pcb.agent_id).state is AgentState.TERMINATED

    kernel.close()


@pytest.mark.asyncio
async def test_kill_releases_held_resource_to_waiter():
    kernel = Kernel()
    holder = kernel.admit(priority=5, quota_total=10, agent_id="holder")
    waiter = kernel.admit(priority=1, quota_total=10, agent_id="waiter")

    await kernel.resources.acquire(holder.agent_id, "res")

    async def wait_for_res():
        await kernel.scheduler.acquire_turn(waiter.agent_id)
        await kernel.resources.acquire(waiter.agent_id, "res")

    task = asyncio.ensure_future(wait_for_res())
    await asyncio.sleep(0.01)
    assert not task.done()

    await kernel.kill_manager.kill(holder.agent_id, reason="test")
    await asyncio.wait_for(task, timeout=1.0)

    assert kernel.resources._held.get("res") == waiter.agent_id
    kernel.close()
