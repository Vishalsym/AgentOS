import asyncio

import pytest

from sentari.deadlock.detector import DeadlockDetector
from sentari.deadlock.wait_graph import WaitForGraph
from sentari.kernel import Kernel
from sentari.pcb import AgentState
from sentari.syscalls.layer import SyscallType


def test_wait_graph_detects_two_agent_cycle():
    graph = WaitForGraph()
    graph.add_edge("a", "b")
    assert not graph.has_cycle("a")
    graph.add_edge("b", "a")
    assert graph.has_cycle("b")
    cycle = graph.extract_cycle("b")
    assert set(cycle) == {"a", "b"}


def test_wait_graph_no_false_positive_on_diamond():
    graph = WaitForGraph()
    graph.add_edge("a", "b")
    graph.add_edge("a", "c")
    graph.add_edge("b", "d")
    graph.add_edge("c", "d")
    assert not graph.has_cycle("a")


def test_wait_graph_remove_edges_for_agent():
    graph = WaitForGraph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "a")
    graph.remove_edges_for("a")
    assert not graph.has_cycle("b")


def test_detector_picks_least_important_agent_as_victim():
    priorities = {"a": 1, "b": 10}  # b is numerically higher => less important
    detector = DeadlockDetector(priority_fn=lambda aid: priorities[aid])
    assert detector.add_wait_edge("a", "b") is None
    victim = detector.add_wait_edge("b", "a")
    assert victim == "b"


# ---- semantic-value-aware victim selection (novel mechanism #1) ----------


def test_no_value_fn_is_byte_for_byte_the_original_priority_only_behavior():
    """Backward-compat guarantee: omitting value_fn entirely must produce
    the identical outcome as before this feature existed."""
    priorities = {"a": 1, "b": 10}
    detector = DeadlockDetector(priority_fn=lambda aid: priorities[aid])
    detector.add_wait_edge("a", "b")
    assert detector.add_wait_edge("b", "a") == "b"


def test_semantic_value_can_override_static_priority():
    """b is numerically less important (would normally be the victim), but
    b's task_value is far higher (e.g. hours of irreplaceable work) than
    a's -- the semantic score should win, sparing b and killing a instead."""
    priorities = {"a": 1, "b": 10}
    values = {"a": 0.1, "b": 0.95}
    detector = DeadlockDetector(
        priority_fn=lambda aid: priorities[aid], value_fn=lambda aid: values[aid]
    )
    detector.add_wait_edge("a", "b")
    victim = detector.add_wait_edge("b", "a")
    assert victim == "a"  # spared b despite b's worse static priority


def test_value_fn_returning_none_for_everyone_falls_back_to_priority():
    priorities = {"a": 1, "b": 10}
    detector = DeadlockDetector(
        priority_fn=lambda aid: priorities[aid], value_fn=lambda aid: None
    )
    detector.add_wait_edge("a", "b")
    victim = detector.add_wait_edge("b", "a")
    assert victim == "b"  # identical to no value_fn at all


def test_value_fn_partial_coverage_unscored_agent_treated_as_neutral():
    """Only one agent in the cycle has a semantic score; the unscored one
    is treated as neutral (0.0 expendability contribution), so a
    positively-valued scored agent is still spared over it."""
    priorities = {"a": 5, "b": 5}  # tied priority -- semantic score must decide
    values = {"a": 0.8, "b": None}
    detector = DeadlockDetector(
        priority_fn=lambda aid: priorities[aid], value_fn=lambda aid: values[aid]
    )
    detector.add_wait_edge("a", "b")
    victim = detector.add_wait_edge("b", "a")
    assert victim == "b"  # a's positive value outweighs b's neutral (unscored) status


@pytest.mark.asyncio
async def test_two_agent_resource_deadlock_kills_one_and_frees_the_other():
    kernel = Kernel()
    high = kernel.admit(priority=1, quota_total=10, agent_id="high")  # more important
    low = kernel.admit(priority=10, quota_total=10, agent_id="low")  # less important

    # high grabs resource A, low grabs resource B.
    await kernel.resources.acquire(high.agent_id, "A")
    await kernel.resources.acquire(low.agent_id, "B")

    async def high_wants_b():
        await kernel.scheduler.acquire_turn(high.agent_id)
        await kernel.resources.acquire(high.agent_id, "B")

    async def low_wants_a():
        await kernel.scheduler.acquire_turn(low.agent_id)
        await kernel.resources.acquire(low.agent_id, "A")

    task_high = asyncio.ensure_future(high_wants_b())
    task_low = asyncio.ensure_future(low_wants_a())

    done, pending = await asyncio.wait(
        {task_high, task_low}, timeout=2.0, return_when=asyncio.ALL_COMPLETED
    )
    assert not pending, "deadlock was not resolved -- a task is still hanging"

    # exactly one of them should have been killed as the deadlock victim
    states = {kernel.scheduler.get("high").state, kernel.scheduler.get("low").state}
    assert AgentState.KILLED in states
    # the low-importance agent is the one expected to be sacrificed
    assert kernel.scheduler.get("low").state is AgentState.KILLED
    assert kernel.scheduler.get("high").state is not AgentState.KILLED

    kernel.close()


@pytest.mark.asyncio
async def test_task_value_flips_the_real_kernel_deadlock_outcome():
    """Same scenario and same static priorities as the test above -- 'low'
    is numerically less important and would normally be the victim -- but
    'low' is given a high task_value (e.g. it already did a lot of
    irreplaceable work) and 'high' is given a low one. The real kernel,
    through Kernel.admit(task_value=...) -> DeadlockDetector, must now
    spare 'low' and kill 'high' instead."""
    kernel = Kernel()
    high = kernel.admit(priority=1, quota_total=10, agent_id="high", task_value=0.1)
    low = kernel.admit(priority=10, quota_total=10, agent_id="low", task_value=0.9)

    await kernel.resources.acquire(high.agent_id, "A")
    await kernel.resources.acquire(low.agent_id, "B")

    async def high_wants_b():
        await kernel.scheduler.acquire_turn(high.agent_id)
        await kernel.resources.acquire(high.agent_id, "B")

    async def low_wants_a():
        await kernel.scheduler.acquire_turn(low.agent_id)
        await kernel.resources.acquire(low.agent_id, "A")

    task_high = asyncio.ensure_future(high_wants_b())
    task_low = asyncio.ensure_future(low_wants_a())

    done, pending = await asyncio.wait(
        {task_high, task_low}, timeout=2.0, return_when=asyncio.ALL_COMPLETED
    )
    assert not pending

    assert kernel.scheduler.get("high").state is AgentState.KILLED
    assert kernel.scheduler.get("low").state is not AgentState.KILLED
    kernel.close()


@pytest.mark.asyncio
async def test_deadlock_via_syscalls_end_to_end():
    kernel = Kernel()
    a1 = kernel.admit(priority=1, quota_total=10, agent_id="a1")
    a2 = kernel.admit(priority=5, quota_total=10, agent_id="a2")

    await kernel.resources.acquire(a1.agent_id, "res1")
    await kernel.resources.acquire(a2.agent_id, "res2")

    async def a1_call():
        return await kernel.syscall(
            a1.agent_id, SyscallType.TOOL_CALL, {"resource_key": "res2", "prompt": "x"}
        )

    async def a2_call():
        return await kernel.syscall(
            a2.agent_id, SyscallType.TOOL_CALL, {"resource_key": "res1", "prompt": "y"}
        )

    results = await asyncio.wait_for(asyncio.gather(a1_call(), a2_call(), return_exceptions=True), timeout=2.0)
    # neither call should hang; the killed agent's syscall resolves with an
    # ERROR response (via the exception path in on_syscall), not a raised
    # exception escaping to the caller.
    for r in results:
        assert not isinstance(r, Exception)

    states = {kernel.scheduler.get("a1").state, kernel.scheduler.get("a2").state}
    assert AgentState.KILLED in states

    kernel.close()
