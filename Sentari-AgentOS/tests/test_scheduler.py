import asyncio

import pytest

from sentari.pcb import AgentPCB, AgentState
from sentari.persistence.db import connect
from sentari.persistence.repositories import AgentRepo
from sentari.scheduler.scheduler import Scheduler, SchedulingPolicy


def make_scheduler(**kwargs) -> Scheduler:
    conn = connect(":memory:")
    return Scheduler(AgentRepo(conn), **kwargs)


@pytest.mark.asyncio
async def test_higher_priority_agent_runs_first_when_both_contend():
    sched = make_scheduler(policy=SchedulingPolicy.PRIORITY)
    sched.admit(AgentPCB.new(priority=10, quota_total=5, agent_id="low"))
    sched.admit(AgentPCB.new(priority=1, quota_total=5, agent_id="high"))

    # "low" asks for the CPU first and has to queue behind nothing yet, but
    # before it's granted, "high" also asks -- priority must win.
    low_task = asyncio.ensure_future(sched.acquire_turn("low"))
    await asyncio.sleep(0.01)
    assert sched.get("low").state.value == "RUNNING"  # low was alone, ran immediately
    high_task = asyncio.ensure_future(sched.acquire_turn("high"))
    await asyncio.sleep(0.01)
    assert not high_task.done()  # low is still holding the CPU

    await sched.release_turn("low")
    await low_task
    await high_task
    assert sched.get("high").state.value == "RUNNING"


@pytest.mark.asyncio
async def test_single_agent_full_turn_cycle():
    sched = make_scheduler()
    pcb = AgentPCB.new(priority=5, quota_total=5, agent_id="a1")
    sched.admit(pcb)
    assert pcb.state is AgentState.READY

    await sched.acquire_turn("a1")
    assert pcb.state is AgentState.RUNNING

    await sched.release_turn("a1")
    assert pcb.state is AgentState.READY


@pytest.mark.asyncio
async def test_priority_selection_prefers_lower_number():
    sched = make_scheduler(policy=SchedulingPolicy.PRIORITY)
    sched.admit(AgentPCB.new(priority=1, quota_total=5, agent_id="blocker"))
    sched.admit(AgentPCB.new(priority=10, quota_total=5, agent_id="low_importance"))
    sched.admit(AgentPCB.new(priority=1, quota_total=5, agent_id="high_importance"))

    await sched.acquire_turn("blocker")  # occupies the CPU so the others must queue
    low_task = asyncio.ensure_future(sched.acquire_turn("low_importance"))
    await asyncio.sleep(0.01)
    high_task = asyncio.ensure_future(sched.acquire_turn("high_importance"))
    await asyncio.sleep(0.01)

    assert sched._peek_next() == "high_importance"

    await sched.release_turn("blocker")
    await high_task
    assert sched.get("high_importance").state.value == "RUNNING"
    assert not low_task.done()

    await sched.release_turn("high_importance")
    await low_task


@pytest.mark.asyncio
async def test_round_robin_preserves_queue_order():
    sched = make_scheduler(policy=SchedulingPolicy.ROUND_ROBIN)
    sched.admit(AgentPCB.new(priority=1, quota_total=5, agent_id="blocker"))
    sched.admit(AgentPCB.new(priority=1, quota_total=5, agent_id="first"))
    sched.admit(AgentPCB.new(priority=1, quota_total=5, agent_id="second"))

    await sched.acquire_turn("blocker")
    first_task = asyncio.ensure_future(sched.acquire_turn("first"))
    await asyncio.sleep(0.01)
    second_task = asyncio.ensure_future(sched.acquire_turn("second"))
    await asyncio.sleep(0.01)

    assert sched._peek_next() == "first"

    await sched.release_turn("blocker")
    await first_task
    assert sched.get("first").state.value == "RUNNING"

    await sched.release_turn("first")
    await second_task


@pytest.mark.asyncio
async def test_aging_boosts_starved_agent_priority():
    sched = make_scheduler(policy=SchedulingPolicy.PRIORITY, aging_threshold_seconds=0.0)
    sched.admit(AgentPCB.new(priority=1, quota_total=5, agent_id="blocker"))
    sched.admit(AgentPCB.new(priority=10, quota_total=5, agent_id="starved"))

    await sched.acquire_turn("blocker")
    starved_task = asyncio.ensure_future(sched.acquire_turn("starved"))
    await asyncio.sleep(0.01)
    starting_priority = sched.get("starved").priority

    sched._apply_aging()
    assert sched.get("starved").priority < starting_priority

    await sched.release_turn("blocker")
    await starved_task


@pytest.mark.asyncio
async def test_only_one_agent_runs_at_a_time():
    sched = make_scheduler()
    sched.admit(AgentPCB.new(priority=1, quota_total=5, agent_id="a1"))
    sched.admit(AgentPCB.new(priority=1, quota_total=5, agent_id="a2"))

    await sched.acquire_turn("a1")
    assert sched.get("a1").state is AgentState.RUNNING

    second_turn = asyncio.ensure_future(sched.acquire_turn("a2"))
    await asyncio.sleep(0.01)
    assert not second_turn.done()
    assert sched.get("a2").state is not AgentState.RUNNING

    await sched.release_turn("a1")
    await second_turn
    assert sched.get("a2").state is AgentState.RUNNING
