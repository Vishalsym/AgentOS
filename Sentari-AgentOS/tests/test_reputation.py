"""Tests for reputation-driven adaptive admission control (novel mechanism
#5): the kernel's own audit history of a recurring agent_type feeds back
into admission control for future instances of that type -- a closed loop
real operating systems don't have, because ordinary processes don't have a
"track record" the way a recurring class of AI agent does.

Covers the repository in isolation, the scheduler's admission-time
adjustment, and a full kernel-level scenario where a type's repeated
deadlock-victim history measurably worsens a brand new instance's starting
priority before it does anything at all."""

from __future__ import annotations

import pytest

from sentari.kernel import Kernel
from sentari.pcb import AgentPCB
from sentari.persistence.db import connect
from sentari.persistence.repositories import ReputationRepo
from sentari.scheduler.scheduler import Scheduler
from sentari.syscalls.layer import SyscallType


def test_reputation_repo_tracks_admissions_and_outcomes():
    repo = ReputationRepo(connect(":memory:"))
    assert repo.risk_score("worker") == 0.0  # no history yet -- innocent by default

    repo.record_admission("worker")
    repo.record_admission("worker")
    repo.record_admission("worker")
    repo.record_kill("worker")

    row = repo.get("worker")
    assert row["admissions"] == 3
    assert row["kills"] == 1
    assert row["quota_exhaustions"] == 0
    assert repo.risk_score("worker") == pytest.approx(1 / 3)


def test_reputation_repo_risk_score_is_capped_at_one():
    repo = ReputationRepo(connect(":memory:"))
    repo.record_admission("chaos")
    repo.record_kill("chaos")
    repo.record_kill("chaos")  # more bad outcomes recorded than admissions
    assert repo.risk_score("chaos") == 1.0


def test_scheduler_does_not_penalize_below_minimum_sample_size():
    """A type with only 1-2 prior admissions hasn't proven anything yet --
    penalizing on a tiny sample would be noisy and unfair."""
    repo = ReputationRepo(connect(":memory:"))
    scheduler = Scheduler(AgentRepoStub(), reputation_repo=repo, min_reputation_sample=3)

    repo.record_admission("flaky")
    repo.record_kill("flaky")  # 1/1 = 100% risk, but sample size is only 1

    pcb = AgentPCB.new(priority=10, quota_total=5, agent_id="a1", agent_type="flaky")
    scheduler.admit(pcb)
    assert pcb.priority == 10  # unchanged -- sample too small to act on
    assert scheduler.last_reputation_adjustment["a1"] == 0


def test_scheduler_penalizes_priority_proportional_to_risk_once_sampled():
    repo = ReputationRepo(connect(":memory:"))
    scheduler = Scheduler(
        AgentRepoStub(), reputation_repo=repo, min_reputation_sample=3, max_reputation_penalty=20
    )

    for _ in range(4):
        repo.record_admission("flaky")
    repo.record_kill("flaky")
    repo.record_kill("flaky")  # 2/4 = 0.5 risk score

    pcb = AgentPCB.new(priority=5, quota_total=5, agent_id="a1", agent_type="flaky")
    scheduler.admit(pcb)

    expected_penalty = round(0.5 * 20)  # 10
    assert pcb.priority == 5 + expected_penalty
    assert scheduler.last_reputation_adjustment["a1"] == expected_penalty


def test_scheduler_never_penalizes_a_type_with_a_clean_record():
    repo = ReputationRepo(connect(":memory:"))
    scheduler = Scheduler(AgentRepoStub(), reputation_repo=repo, min_reputation_sample=3)

    for _ in range(10):
        repo.record_admission("reliable")
    # zero kills, zero quota exhaustions recorded

    pcb = AgentPCB.new(priority=1, quota_total=5, agent_id="a1", agent_type="reliable")
    scheduler.admit(pcb)
    assert pcb.priority == 1
    assert scheduler.last_reputation_adjustment["a1"] == 0


@pytest.mark.asyncio
async def test_full_kernel_scenario_a_troubled_type_is_deprioritized_next_time():
    """End to end, using the real Kernel: repeatedly admit-and-kill several
    agents that all share agent_type='rogue_worker', then admit a *brand
    new, never-before-seen* agent of the same type and confirm its
    starting priority is already worse -- purely from the type's history,
    before it has done anything at all."""
    kernel = Kernel()

    for i in range(4):
        pcb = kernel.admit(priority=5, quota_total=10, agent_id=f"rogue-{i}", agent_type="rogue_worker")
        await kernel.kill_manager.kill(pcb.agent_id, reason="fault_injection_test")

    row = kernel.reputation_repo.get("rogue_worker")
    assert row["admissions"] == 4
    assert row["kills"] == 4

    fresh = kernel.admit(priority=5, quota_total=10, agent_id="rogue-fresh", agent_type="rogue_worker")
    assert fresh.priority > 5  # worse (numerically higher) than the declared priority=5
    assert kernel.scheduler.last_reputation_adjustment["rogue-fresh"] > 0
    kernel.close()


@pytest.mark.asyncio
async def test_unrelated_agent_type_is_unaffected_by_a_different_types_history():
    """Reputation is per-type, not global -- a well-behaved type's agents
    must never be penalized for a *different* type's bad history."""
    kernel = Kernel()

    for i in range(4):
        pcb = kernel.admit(priority=5, quota_total=10, agent_id=f"rogue-{i}", agent_type="rogue_worker")
        await kernel.kill_manager.kill(pcb.agent_id, reason="fault_injection_test")

    clean = kernel.admit(priority=5, quota_total=10, agent_id="clean-1", agent_type="trusted_worker")
    assert clean.priority == 5
    assert kernel.scheduler.last_reputation_adjustment["clean-1"] == 0
    kernel.close()


@pytest.mark.asyncio
async def test_spawned_children_inherit_parent_agent_type_for_reputation():
    kernel = Kernel()
    parent = kernel.admit(priority=1, quota_total=20, agent_id="parent", agent_type="spawner_type")
    from sentari.syscalls.layer import SyscallType

    resp = await kernel.syscall(parent.agent_id, SyscallType.SPAWN, {"quota_share": 0.5})
    child = kernel.scheduler.get(resp.value)
    assert child.agent_type == "spawner_type"
    kernel.close()


class AgentRepoStub:
    """Minimal stand-in for AgentRepo -- these scheduler-level tests don't
    need real persistence, just something admit() can call .create()/.save() on."""

    def create(self, pcb):
        pass

    def save(self, pcb):
        pass


@pytest.mark.asyncio
async def test_reputation_does_not_block_or_deny_only_deprioritizes():
    """Confirm this is purely advisory priority adjustment, never a hard
    denial -- even a maximally bad track record still gets admitted and
    can still eventually run (possibly after aging catches up), which
    matters for the "innocent until proven, correction not exile" design
    intent."""
    kernel = Kernel()
    for i in range(5):
        pcb = kernel.admit(priority=1, quota_total=10, agent_id=f"bad-{i}", agent_type="worst_type")
        await kernel.kill_manager.kill(pcb.agent_id, reason="fault_injection_test")

    newest = kernel.admit(priority=1, quota_total=10, agent_id="bad-newest", agent_type="worst_type")
    # admitted (not rejected) and can still actually run a syscall.
    resp = await kernel.syscall(newest.agent_id, SyscallType.YIELD)
    assert resp.result.value == "OK"
    kernel.close()
