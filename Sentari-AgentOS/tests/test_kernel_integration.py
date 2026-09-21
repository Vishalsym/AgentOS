import asyncio

import pytest

from sentari.kernel import Kernel
from sentari.pcb import AgentState
from sentari.syscalls.layer import SyscallResult, SyscallType


@pytest.mark.asyncio
async def test_end_to_end_spawn_deadlock_and_consistent_persisted_state():
    kernel = Kernel(db_path=":memory:")

    parent = kernel.admit(priority=2, quota_total=20, agent_id="parent")
    spawn_resp = await kernel.syscall(parent.agent_id, SyscallType.SPAWN, {"quota_share": 0.5})
    assert spawn_resp.result is SyscallResult.OK
    child_id = spawn_resp.value
    assert kernel.scheduler.get(child_id).parent_id == "parent"

    important = kernel.admit(priority=1, quota_total=10, agent_id="important")
    unimportant = kernel.admit(priority=20, quota_total=10, agent_id="unimportant")

    await kernel.resources.acquire(important.agent_id, "lock_x")
    await kernel.resources.acquire(unimportant.agent_id, "lock_y")

    async def important_wants_y():
        await kernel.syscall(
            important.agent_id, SyscallType.TOOL_CALL, {"resource_key": "lock_y", "prompt": "p"}
        )

    async def unimportant_wants_x():
        await kernel.syscall(
            unimportant.agent_id, SyscallType.TOOL_CALL, {"resource_key": "lock_x", "prompt": "p"}
        )

    await asyncio.wait_for(
        asyncio.gather(important_wants_y(), unimportant_wants_x(), return_exceptions=True),
        timeout=2.0,
    )

    assert kernel.scheduler.get("unimportant").state is AgentState.KILLED
    assert kernel.scheduler.get("important").state is not AgentState.KILLED

    # FR-17: killed agent holds no resources afterward.
    remaining_alloc = [r for r in kernel.resource_repo.list_all() if r["holder_agent_id"] == "unimportant"]
    assert remaining_alloc == []

    # FR-9: every syscall made it into the audit log.
    log_rows = kernel.syscall_log_repo.list_all()
    assert len(log_rows) >= 2
    assert any(row["syscall_type"] == "spawn" for row in log_rows)

    # DB state is reachable straight from persistence, independent of the
    # in-memory scheduler cache -- proves auditability from the log/DB alone.
    persisted_unimportant = kernel.agent_repo.get("unimportant")
    assert persisted_unimportant.state is AgentState.KILLED

    kernel.close()


@pytest.mark.asyncio
async def test_twenty_agents_can_be_admitted_and_run_a_syscall():
    kernel = Kernel()
    agent_ids = []
    for i in range(20):
        pcb = kernel.admit(priority=i % 5, quota_total=5, agent_id=f"agent-{i}")
        agent_ids.append(pcb.agent_id)

    responses = await asyncio.gather(
        *(kernel.syscall(aid, SyscallType.YIELD) for aid in agent_ids)
    )
    assert all(r.result is SyscallResult.OK for r in responses)
    assert len(kernel.agent_repo.list_all()) == 20
    kernel.close()
