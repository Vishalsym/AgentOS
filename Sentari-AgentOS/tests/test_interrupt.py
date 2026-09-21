"""Tests for the bounded-latency human interrupt primitive (novel mechanism
#4): a human operator or automated safety monitor can force-stop a specific
agent's in-flight syscall *right now*, with a measured latency bound, not
waiting up to `execution_timeout` for the normal preemption path.

Includes an actual timing test (not just a functional one) -- the entire
point of this mechanism is that its reaction time is a claim you can
measure, so it should be measured here rather than only asserted."""

from __future__ import annotations

import asyncio
import time

import pytest

from sentari.kernel import Kernel
from sentari.pcb import AgentState
from sentari.syscalls.layer import SyscallResult, SyscallType


@pytest.mark.asyncio
async def test_interrupt_cancels_a_long_running_call_far_before_its_timeout():
    """The agent's tool would normally run for 30s (default timeout is also
    30s, so the ordinary timeout-based preemption wouldn't fire for a long
    time); interrupt() must stop it almost immediately regardless."""
    kernel = Kernel()
    agent = kernel.admit(priority=1, quota_total=5, agent_id="stuck_agent")

    async def slow_tool() -> str:
        await asyncio.sleep(30)
        return "should never get here"

    call_task = asyncio.ensure_future(
        kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"fn": slow_tool})
    )
    await asyncio.sleep(0.05)  # let the call actually start and register as in-flight

    start = time.perf_counter()
    was_interrupted = await kernel.interrupt(agent.agent_id, reason="operator_stop")
    interrupt_call_latency = time.perf_counter() - start

    response = await asyncio.wait_for(call_task, timeout=2.0)
    total_latency = time.perf_counter() - start

    assert was_interrupted is True
    assert response.result is SyscallResult.ERROR
    assert "interrupted" in response.error.lower()
    assert kernel.scheduler.get("stuck_agent").state is AgentState.KILLED

    # the actual claim under test: reaction time is a small fraction of a
    # second, not anywhere near the tool's 30s sleep or the kernel's 30s
    # default execution_timeout.
    assert interrupt_call_latency < 0.5
    assert total_latency < 1.0
    kernel.close()


@pytest.mark.asyncio
async def test_interrupt_on_an_idle_agent_still_kills_it_cleanly():
    """No in-flight call to cancel -- interrupt() must still work (a human
    might want to stop an agent that's merely about to act, or between
    calls), and report that there was nothing in-flight to cancel."""
    kernel = Kernel()
    agent = kernel.admit(priority=1, quota_total=5, agent_id="idle_agent")

    had_inflight = await kernel.interrupt(agent.agent_id, reason="preemptive_stop")

    assert had_inflight is False
    assert kernel.scheduler.get("idle_agent").state is AgentState.KILLED
    kernel.close()


@pytest.mark.asyncio
async def test_interrupted_agent_releases_its_held_resources():
    """Interrupt reuses KillManager.kill() under the hood -- confirm the
    full FR-17 resource-release guarantee still applies, not just the
    cancellation itself."""
    kernel = Kernel()
    agent = kernel.admit(priority=1, quota_total=5, agent_id="holder")
    waiter = kernel.admit(priority=1, quota_total=5, agent_id="waiter")

    await kernel.resources.acquire(agent.agent_id, "the_lock")

    async def slow_tool() -> None:
        await asyncio.sleep(30)

    call_task = asyncio.ensure_future(
        kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"fn": slow_tool})
    )
    await asyncio.sleep(0.05)

    await kernel.interrupt(agent.agent_id, reason="operator_stop")
    await asyncio.wait_for(call_task, timeout=2.0)

    # the lock is free again -- a waiter can now acquire it.
    await kernel.scheduler.acquire_turn(waiter.agent_id)
    await asyncio.wait_for(kernel.resources.acquire(waiter.agent_id, "the_lock"), timeout=1.0)
    assert kernel.resources._held.get("the_lock") == "waiter"
    kernel.close()


@pytest.mark.asyncio
async def test_interrupting_one_agent_does_not_affect_unrelated_agents():
    kernel = Kernel()
    victim = kernel.admit(priority=1, quota_total=5, agent_id="victim")
    bystander = kernel.admit(priority=1, quota_total=5, agent_id="bystander")

    async def slow_tool() -> None:
        await asyncio.sleep(30)

    victim_call = asyncio.ensure_future(
        kernel.syscall(victim.agent_id, SyscallType.TOOL_CALL, {"fn": slow_tool})
    )
    await asyncio.sleep(0.05)

    await kernel.interrupt(victim.agent_id, reason="operator_stop")
    await asyncio.wait_for(victim_call, timeout=2.0)

    # the bystander, uninvolved, transacts completely normally afterward.
    resp = await kernel.syscall(bystander.agent_id, SyscallType.YIELD)
    assert resp.result is SyscallResult.OK
    assert kernel.scheduler.get("bystander").state is not AgentState.KILLED
    kernel.close()


@pytest.mark.asyncio
async def test_interrupt_notifies_and_updates_reputation_like_any_other_kill():
    """interrupt() is built on the same KillManager.kill() path as a
    deadlock-victim kill -- confirm it participates identically in the
    notification and reputation systems, not a bolted-on side channel."""
    from sentari.notifications.notifier import LogNotifier

    notifier = LogNotifier()
    kernel = Kernel(notifier=notifier)
    kernel.admit(priority=1, quota_total=5, agent_id="a1", agent_type="interruptible_worker")

    await kernel.interrupt("a1", reason="operator_stop")

    kill_events = [e for e in notifier.events if e.event_type == "agent_killed"]
    assert len(kill_events) == 1
    assert kill_events[0].reason == "operator_stop"

    row = kernel.reputation_repo.get("interruptible_worker")
    assert row["kills"] == 1
    kernel.close()


@pytest.mark.asyncio
async def test_interrupting_an_already_killed_agent_is_a_safe_no_op():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=5, agent_id="a1")
    await kernel.interrupt("a1", reason="first")
    # second interrupt on an already-KILLED agent must not raise or
    # double-count -- KillManager.kill() already guards against this.
    had_inflight = await kernel.interrupt("a1", reason="second")
    assert had_inflight is False
    assert kernel.scheduler.get("a1").state is AgentState.KILLED
    kernel.close()
