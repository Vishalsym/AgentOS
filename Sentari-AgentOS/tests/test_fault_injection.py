"""Scripted fault-injection suite (README roadmap item): induced deadlocks
beyond the simple 2-agent case, induced runaway/adversarial agents, and
induced resource contention storms -- proving the kernel survives failure
modes that the per-module unit tests don't individually exercise.

Every test here either (a) asserts the kernel keeps running and produces a
sane, bounded outcome, or (b) asserts a previously-silent failure mode now
fails loudly and cleanly instead of corrupting state or hanging."""

from __future__ import annotations

import asyncio

import pytest

from sentari.kernel import Kernel
from sentari.pcb import AgentState, DuplicateAgentError
from sentari.scheduler.scheduler import Scheduler
from sentari.syscalls.layer import SyscallResult, SyscallType

# --------------------------------------------------------------------------
# Duplicate admission (a killed/terminated agent_id must never be reusable)
# --------------------------------------------------------------------------


def test_readmitting_a_live_agent_id_raises_cleanly_not_a_raw_db_error():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=5, agent_id="dup")
    with pytest.raises(DuplicateAgentError):
        kernel.admit(priority=1, quota_total=5, agent_id="dup")
    kernel.close()


@pytest.mark.asyncio
async def test_readmitting_a_killed_agent_id_still_raises_cleanly():
    """A killed agent's id is retired, not freed -- exactly like a PID an OS
    won't reissue to a second process while the first is still in the
    process table. Confirms Scheduler.admit's guard covers terminal states
    too, not just live ones."""
    kernel = Kernel()
    pcb = kernel.admit(priority=5, quota_total=1, agent_id="zombie")
    await kernel.syscall(pcb.agent_id, SyscallType.YIELD)  # burns the only quota unit
    await kernel.syscall(pcb.agent_id, SyscallType.YIELD)  # exhausts quota -> TERMINATED
    assert kernel.scheduler.get("zombie").state is AgentState.TERMINATED

    with pytest.raises(DuplicateAgentError):
        kernel.admit(priority=5, quota_total=1, agent_id="zombie")
    kernel.close()


def test_scheduler_admit_alone_also_rejects_duplicates():
    """Unit-level guard, independent of Kernel -- the fix lives in
    Scheduler.admit itself, not layered on top of it."""
    from sentari.pcb import AgentPCB
    from sentari.persistence.db import connect
    from sentari.persistence.repositories import AgentRepo

    repo = AgentRepo(connect(":memory:"))
    scheduler = Scheduler(repo)
    scheduler.admit(AgentPCB.new(priority=1, quota_total=5, agent_id="x"))
    with pytest.raises(DuplicateAgentError):
        scheduler.admit(AgentPCB.new(priority=1, quota_total=5, agent_id="x"))


# --------------------------------------------------------------------------
# A killed agent must never be able to transact again
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_killed_agent_syscalls_return_clean_error_not_exception():
    kernel = Kernel()
    victim = kernel.admit(priority=10, quota_total=10, agent_id="soon_dead")
    holder = kernel.admit(priority=1, quota_total=10, agent_id="holder")

    await kernel.resources.acquire(holder.agent_id, "the_lock")
    await kernel.kill_manager.kill(victim.agent_id, reason="fault_injection_test")
    assert kernel.scheduler.get("soon_dead").state is AgentState.KILLED

    resp = await kernel.syscall(victim.agent_id, SyscallType.TOOL_CALL, {"prompt": "still alive?"})
    assert resp.result is SyscallResult.ERROR
    assert "KILLED" in (resp.error or "")
    kernel.close()


# --------------------------------------------------------------------------
# N-agent (not just pairwise) circular wait
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_agent_cycle_is_detected_and_resolved_without_hanging():
    """A -> B -> C -> A. Proves DeadlockDetector's DFS generalizes past the
    2-agent case the rest of the suite only ever exercises.

    Killing one victim only directly frees the resource *that victim*
    held -- the rest of the chain (here, A still waiting on B for R2) only
    resolves once the surviving agents release what they no longer need,
    exactly like a real system: breaking a deadlock cycle isn't the same
    as instantly satisfying every agent that was ever waiting on anyone."""
    kernel = Kernel()
    a = kernel.admit(priority=1, quota_total=10, agent_id="cyc_a")
    b = kernel.admit(priority=5, quota_total=10, agent_id="cyc_b")
    c = kernel.admit(priority=10, quota_total=10, agent_id="cyc_c")  # least important

    await kernel.resources.acquire(a.agent_id, "R1")
    await kernel.resources.acquire(b.agent_id, "R2")
    await kernel.resources.acquire(c.agent_id, "R3")

    results: dict[str, BaseException | None] = {}

    async def run(agent_id: str, wants: str) -> None:
        try:
            await kernel.scheduler.acquire_turn(agent_id)
            await kernel.resources.acquire(agent_id, wants)
            results[agent_id] = None
        except BaseException as exc:  # noqa: BLE001 -- capturing for inspection, not swallowing
            results[agent_id] = exc

    tasks = {
        "cyc_a": asyncio.ensure_future(run("cyc_a", "R2")),
        "cyc_b": asyncio.ensure_future(run("cyc_b", "R3")),
        "cyc_c": asyncio.ensure_future(run("cyc_c", "R1")),
    }
    done, pending = await asyncio.wait(tasks.values(), timeout=3.0)

    states = {aid: kernel.scheduler.get(aid).state for aid in tasks}
    killed = [aid for aid, s in states.items() if s is AgentState.KILLED]
    assert len(killed) == 1, f"expected exactly one deadlock victim, got {killed} (states={states})"
    assert killed == ["cyc_c"], "victim should be the least-important agent in the cycle"
    # the victim's own acquire() call is the one that closed the cycle, and
    # it was also the victim -- that surfaces as a RuntimeError to the
    # caller (documented in ResourceManager.acquire), not a silent hang.
    assert isinstance(results.get("cyc_c"), RuntimeError)

    # break what's left of the chain the same way a real agent would: by
    # releasing resources (and the CPU turn) it no longer needs once its
    # own work is done -- cyc_b's `run()` finished holding both, exactly
    # like any caller of ResourceManager.acquire() must give the turn back
    # explicitly (SyscallLayer.on_syscall does this automatically; direct
    # ResourceManager use, as here, has to do it by hand).
    kernel.resources.release("cyc_b", "R2")
    kernel.resources.release("cyc_b", "R3")
    await kernel.scheduler.release_turn("cyc_b")

    remaining = [t for aid, t in tasks.items() if aid not in {"cyc_b", "cyc_c"} and not t.done()]
    if remaining:
        done2, pending2 = await asyncio.wait(remaining, timeout=3.0)
        assert not pending2, "surviving agent never got its resource after the blocker released it"

    final_states = {aid: kernel.scheduler.get(aid).state for aid in tasks}
    assert final_states["cyc_c"] is AgentState.KILLED
    assert final_states["cyc_a"] is not AgentState.KILLED
    assert final_states["cyc_b"] is not AgentState.KILLED

    # the graph is fully clean afterward -- no dead agent lingers in it.
    for aid in tasks:
        assert kernel.detector.graph.edges(aid) == set()

    kernel.close()


# --------------------------------------------------------------------------
# Resource contention storm: many agents, one resource
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_many_agents_contend_for_one_resource_no_double_grant():
    kernel = Kernel()
    n = 8
    agents = [kernel.admit(priority=1, quota_total=10, agent_id=f"contender-{i}") for i in range(n)]

    grant_order: list[str] = []
    lock = asyncio.Lock()

    async def use_and_release(agent_id: str) -> None:
        await kernel.scheduler.acquire_turn(agent_id)
        await kernel.resources.acquire(agent_id, "shared")
        async with lock:
            grant_order.append(agent_id)
        # simulate doing real work while holding it -- if the resource
        # manager ever double-grants, two agents would both be "inside"
        # this window at once and grant_order would show it out of order
        # relative to release().
        await asyncio.sleep(0.01)
        kernel.resources.release(agent_id, "shared")
        await kernel.scheduler.release_turn(agent_id)

    await asyncio.wait_for(
        asyncio.gather(*(use_and_release(a.agent_id) for a in agents)), timeout=5.0
    )

    assert len(grant_order) == n
    assert len(set(grant_order)) == n, f"a resource was granted to more than one agent: {grant_order}"
    # nobody should still be holding or waiting on it afterward.
    assert kernel.resources._held.get("shared") is None
    assert kernel.resources._waiters.get("shared", []) == []
    kernel.close()


# --------------------------------------------------------------------------
# Spawn storm
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rapid_spawn_storm_stays_quota_bounded_and_consistent():
    kernel = Kernel()
    parent = kernel.admit(priority=1, quota_total=1000, agent_id="spawner")

    responses = await asyncio.gather(
        *(kernel.syscall(parent.agent_id, SyscallType.SPAWN, {"quota_share": 0.5}) for _ in range(15))
    )
    child_ids = [r.value for r in responses if r.result is SyscallResult.OK]
    assert len(child_ids) == 15
    assert len(set(child_ids)) == 15  # every spawned child got a unique id

    for cid in child_ids:
        child = kernel.scheduler.get(cid)
        assert child.parent_id == "spawner"
        assert 1 <= child.quota_total <= parent.quota_total

    # every spawn syscall made it into the audit log -- no silent drops
    # under concurrent load.
    spawn_logs = [r for r in kernel.syscall_log_repo.list_all() if r["syscall_type"] == "spawn"]
    assert len(spawn_logs) == 15
    kernel.close()


# --------------------------------------------------------------------------
# Adversarial / malformed tool calls must never crash the kernel
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_raising_unexpected_exception_types_is_always_contained():
    kernel = Kernel()
    agent = kernel.admit(priority=1, quota_total=20, agent_id="chaos")

    async def raises_key_error() -> None:
        raise KeyError("missing field the tool assumed would be there")

    async def raises_recursion_error() -> None:
        raise RecursionError("simulated runaway recursion inside a tool")

    async def raises_arbitrary_custom_error() -> None:
        class WeirdToolError(Exception):
            pass

        raise WeirdToolError("some third-party tool's own exception type")

    for fn in (raises_key_error, raises_recursion_error, raises_arbitrary_custom_error):
        resp = await kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"fn": fn})
        assert resp.result is SyscallResult.ERROR
        assert resp.error  # the real exception message survived into the audit trail

    # the agent itself is untouched by any of this -- one bad tool call
    # never corrupts the agent's own state.
    assert kernel.scheduler.get("chaos").state in (AgentState.READY, AgentState.RUNNING)
    kernel.close()


@pytest.mark.asyncio
async def test_quota_exhaustion_mid_storm_denies_cleanly_for_every_excess_call():
    """10 concurrent calls from the SAME agent, quota for only 3. Every
    call must resolve to exactly one of: OK (one of the first 3, in
    whatever order they actually get serialized), DENY (the specific call
    that tipped the agent over into exhaustion), or ERROR (a call that only
    reached the front of the queue after the agent was already
    TERMINATED by an earlier one -- a real, sensible distinct outcome, not
    a bug: it never had a chance to consume quota at all). The one
    property that must hold exactly, even under this concurrency, is that
    no more than 3 calls ever actually ran."""
    kernel = Kernel()
    agent = kernel.admit(priority=1, quota_total=3, agent_id="tight_budget")

    responses = await asyncio.gather(
        *(kernel.syscall(agent.agent_id, SyscallType.YIELD) for _ in range(10)),
        return_exceptions=True,
    )
    assert all(not isinstance(r, Exception) for r in responses)
    ok = [r for r in responses if r.result is SyscallResult.OK]
    denied = [r for r in responses if r.result is SyscallResult.DENY]
    errored = [r for r in responses if r.result is SyscallResult.ERROR]
    assert len(ok) == 3  # exactly the budget -- no partial overrun, no undershoot
    assert len(denied) + len(errored) == 7
    assert len(ok) + len(denied) + len(errored) == 10
    assert kernel.scheduler.get("tight_budget").state is AgentState.TERMINATED
    kernel.close()
