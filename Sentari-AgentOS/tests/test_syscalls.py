import pytest

from sentari.kernel import Kernel
from sentari.pcb import AgentState
from sentari.syscalls.layer import SyscallResult, SyscallType


@pytest.mark.asyncio
async def test_tool_call_returns_ok_and_response_value():
    kernel = Kernel()
    pcb = kernel.admit(priority=1, quota_total=5)
    resp = await kernel.syscall(pcb.agent_id, SyscallType.TOOL_CALL, {"prompt": "ping"})
    assert resp.result is SyscallResult.OK
    assert "ping" in resp.value
    kernel.close()


@pytest.mark.asyncio
async def test_syscall_charges_quota_and_logs():
    kernel = Kernel()
    pcb = kernel.admit(priority=1, quota_total=5)
    await kernel.syscall(pcb.agent_id, SyscallType.YIELD)
    assert kernel.scheduler.get(pcb.agent_id).quota_used == 1

    entries = kernel.syscall_log_repo.list_for_agent(pcb.agent_id)
    assert len(entries) == 1
    assert entries[0]["result"] == "OK"
    kernel.close()


@pytest.mark.asyncio
async def test_quota_exhaustion_denies_and_terminates_agent():
    kernel = Kernel()
    pcb = kernel.admit(priority=1, quota_total=1)
    resp1 = await kernel.syscall(pcb.agent_id, SyscallType.YIELD)
    assert resp1.result is SyscallResult.OK

    resp2 = await kernel.syscall(pcb.agent_id, SyscallType.YIELD)
    assert resp2.result is SyscallResult.DENY
    assert kernel.scheduler.get(pcb.agent_id).state is AgentState.TERMINATED
    kernel.close()


@pytest.mark.asyncio
async def test_memory_read_write_via_syscalls_is_self_scoped():
    kernel = Kernel()
    pcb = kernel.admit(priority=1, quota_total=5)
    await kernel.syscall(
        pcb.agent_id, SyscallType.MEMORY_WRITE, {"key": "note", "value": "hello"}
    )
    resp = await kernel.syscall(pcb.agent_id, SyscallType.MEMORY_READ, {})
    assert resp.value == {"note": "hello"}
    kernel.close()


@pytest.mark.asyncio
async def test_kb_read_write_via_syscalls_is_shared():
    kernel = Kernel()
    writer = kernel.admit(priority=1, quota_total=5)
    reader = kernel.admit(priority=1, quota_total=5)
    await kernel.syscall(
        writer.agent_id, SyscallType.MEMORY_WRITE, {"scope": "kb", "key": "shared", "value": "v1"}
    )
    resp = await kernel.syscall(reader.agent_id, SyscallType.MEMORY_READ, {"scope": "kb", "key": "shared"})
    assert resp.value == "v1"
    kernel.close()


@pytest.mark.asyncio
async def test_spawn_creates_child_with_bounded_quota():
    kernel = Kernel()
    parent = kernel.admit(priority=3, quota_total=10)
    resp = await kernel.syscall(parent.agent_id, SyscallType.SPAWN, {"quota_share": 0.5})
    assert resp.result is SyscallResult.OK
    child_id = resp.value
    child = kernel.scheduler.get(child_id)
    assert child.parent_id == parent.agent_id
    assert 0 < child.quota_total <= parent.quota_total
    kernel.close()


@pytest.mark.asyncio
async def test_a_failing_tool_does_not_crash_the_kernel():
    kernel = Kernel()
    pcb = kernel.admit(priority=1, quota_total=5)

    async def boom():
        raise RuntimeError("tool exploded")

    resp = await kernel.syscall(pcb.agent_id, SyscallType.TOOL_CALL, {"fn": boom})
    assert resp.result is SyscallResult.ERROR
    assert "tool exploded" in resp.error
    # agent survives, quota still charged, kernel unaffected
    assert kernel.scheduler.get(pcb.agent_id).quota_used == 1
    kernel.close()
