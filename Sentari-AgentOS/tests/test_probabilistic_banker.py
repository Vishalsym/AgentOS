"""Tests for the probabilistic Banker's Algorithm (novel mechanism #2):
proactive deadlock AVOIDANCE based on declared future-resource-claim
probabilities, distinct from (and complementary to) the reactive
DeadlockDetector tested elsewhere.

Covers the risk-estimation math in isolation, and a full kernel-level
scenario where a wait that *would* have formed a real AB-BA deadlock is
instead refused proactively -- before any cycle, kill, or victim selection
ever happens -- because both sides declared a high probability of wanting
what the other holds."""

from __future__ import annotations

import asyncio

import pytest

from sentari.deadlock.probabilistic_banker import ProbabilisticBanker, ResourceRequestUnsafeError
from sentari.kernel import Kernel
from sentari.pcb import AgentState

# --------------------------------------------------------------------------
# ProbabilisticBanker.check in isolation
# --------------------------------------------------------------------------


def test_no_declared_claims_is_always_safe():
    banker = ProbabilisticBanker()
    check = banker.check("waiter", "holder", "R", holder_claims={}, held_by_waiter={"X"})
    assert check.safe is True
    assert check.risk == 0.0


def test_waiter_holding_nothing_is_always_safe_regardless_of_holder_claims():
    banker = ProbabilisticBanker()
    check = banker.check("waiter", "holder", "R", holder_claims={"X": 0.99}, held_by_waiter=set())
    assert check.safe is True


def test_high_reciprocal_claim_on_a_held_resource_is_unsafe():
    banker = ProbabilisticBanker(unsafe_risk_threshold=0.7)
    check = banker.check("waiter", "holder", "R", holder_claims={"X": 0.9}, held_by_waiter={"X"})
    assert check.safe is False
    assert check.risk == pytest.approx(0.9)


def test_low_reciprocal_claim_stays_safe():
    banker = ProbabilisticBanker(unsafe_risk_threshold=0.7)
    check = banker.check("waiter", "holder", "R", holder_claims={"X": 0.2}, held_by_waiter={"X"})
    assert check.safe is True
    assert check.risk == pytest.approx(0.2)


def test_risk_combines_across_multiple_held_resources_independence_assumption():
    """Waiter holds two resources the holder has *some* declared interest
    in; risk should reflect "at least one" under the documented
    independence assumption: 1 - (1-0.5)(1-0.5) = 0.75."""
    banker = ProbabilisticBanker(unsafe_risk_threshold=0.7)
    check = banker.check(
        "waiter", "holder", "R", holder_claims={"X": 0.5, "Y": 0.5}, held_by_waiter={"X", "Y"}
    )
    assert check.risk == pytest.approx(0.75)
    assert check.safe is False


def test_threshold_is_configurable():
    lenient = ProbabilisticBanker(unsafe_risk_threshold=0.95)
    strict = ProbabilisticBanker(unsafe_risk_threshold=0.3)
    check_lenient = lenient.check("w", "h", "R", holder_claims={"X": 0.5}, held_by_waiter={"X"})
    check_strict = strict.check("w", "h", "R", holder_claims={"X": 0.5}, held_by_waiter={"X"})
    assert check_lenient.safe is True
    assert check_strict.safe is False


# --------------------------------------------------------------------------
# Full kernel-level scenario
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proactive_avoidance_refuses_a_wait_that_would_become_a_real_deadlock():
    """Same classic AB-BA setup used throughout test_deadlock.py, but this
    time both agents declared (at admission) a high probability of wanting
    the resource the other holds. The wait must be refused proactively --
    with NO kill, NO victim, and the wait-for graph left clean -- instead
    of the usual reactive path where a real cycle forms and gets resolved
    by killing someone."""
    banker = ProbabilisticBanker(unsafe_risk_threshold=0.7)
    kernel = Kernel(probabilistic_banker=banker)
    alpha = kernel.admit(
        priority=1, quota_total=10, agent_id="alpha", declared_resource_claims={"Y": 0.9}
    )
    beta = kernel.admit(
        priority=5, quota_total=10, agent_id="beta", declared_resource_claims={"X": 0.9}
    )

    await kernel.resources.acquire(alpha.agent_id, "X")
    await kernel.resources.acquire(beta.agent_id, "Y")

    # alpha tries to grab Y (held by beta, who has a declared 90% chance
    # of wanting X, which alpha holds) -- must be refused proactively.
    await kernel.scheduler.acquire_turn(alpha.agent_id)
    with pytest.raises(ResourceRequestUnsafeError):
        await kernel.resources.acquire(alpha.agent_id, "Y")
    await kernel.scheduler.release_turn(alpha.agent_id)

    # nobody was killed -- this was avoidance, not reactive resolution.
    assert kernel.scheduler.get("alpha").state is not AgentState.KILLED
    assert kernel.scheduler.get("beta").state is not AgentState.KILLED
    # the wait-for graph has no lingering edge from the refused attempt.
    assert kernel.detector.graph.edges("alpha") == set()
    # beta still holds Y undisturbed; alpha never got it.
    assert kernel.resources._held.get("Y") == "beta"
    assert kernel.resources._held.get("X") == "alpha"
    kernel.close()


@pytest.mark.asyncio
async def test_low_declared_risk_falls_through_to_normal_reactive_resolution():
    """Same setup, but the claims are declared with LOW probability this
    time -- the proactive check should let it through, and the *existing*
    reactive detector resolves the resulting real deadlock exactly as it
    does everywhere else in the suite (a real kill, one survivor)."""
    banker = ProbabilisticBanker(unsafe_risk_threshold=0.7)
    kernel = Kernel(probabilistic_banker=banker)
    alpha = kernel.admit(
        priority=1, quota_total=10, agent_id="alpha", declared_resource_claims={"Y": 0.1}
    )
    beta = kernel.admit(
        priority=10, quota_total=10, agent_id="beta", declared_resource_claims={"X": 0.1}
    )

    await kernel.resources.acquire(alpha.agent_id, "X")
    await kernel.resources.acquire(beta.agent_id, "Y")

    async def alpha_wants_y():
        await kernel.scheduler.acquire_turn(alpha.agent_id)
        await kernel.resources.acquire(alpha.agent_id, "Y")

    async def beta_wants_x():
        await kernel.scheduler.acquire_turn(beta.agent_id)
        await kernel.resources.acquire(beta.agent_id, "X")

    task_a = asyncio.ensure_future(alpha_wants_y())
    task_b = asyncio.ensure_future(beta_wants_x())
    done, pending = await asyncio.wait({task_a, task_b}, timeout=2.0)
    assert not pending

    # a real cycle formed and was reactively resolved -- unlike the high-
    # risk test above, someone WAS killed here.
    assert kernel.scheduler.get("beta").state is AgentState.KILLED
    kernel.close()


@pytest.mark.asyncio
async def test_no_banker_configured_is_byte_for_byte_the_original_behavior():
    """Backward-compat guarantee: a kernel with no probabilistic_banker at
    all reproduces the exact reactive-only behavior tested throughout
    test_deadlock.py -- nothing here changes default kernel behavior."""
    kernel = Kernel()  # no probabilistic_banker
    alpha = kernel.admit(priority=1, quota_total=10, agent_id="alpha")
    beta = kernel.admit(priority=10, quota_total=10, agent_id="beta")

    await kernel.resources.acquire(alpha.agent_id, "X")
    await kernel.resources.acquire(beta.agent_id, "Y")

    async def alpha_wants_y():
        await kernel.scheduler.acquire_turn(alpha.agent_id)
        await kernel.resources.acquire(alpha.agent_id, "Y")

    async def beta_wants_x():
        await kernel.scheduler.acquire_turn(beta.agent_id)
        await kernel.resources.acquire(beta.agent_id, "X")

    task_a = asyncio.ensure_future(alpha_wants_y())
    task_b = asyncio.ensure_future(beta_wants_x())
    done, pending = await asyncio.wait({task_a, task_b}, timeout=2.0)
    assert not pending
    assert kernel.scheduler.get("beta").state is AgentState.KILLED
    kernel.close()


@pytest.mark.asyncio
async def test_banker_configured_but_no_claims_declared_never_intervenes():
    """Opt-in at the per-agent level too -- configuring a banker
    kernel-wide doesn't retroactively make undeclared agents risky."""
    banker = ProbabilisticBanker(unsafe_risk_threshold=0.5)
    kernel = Kernel(probabilistic_banker=banker)
    alpha = kernel.admit(priority=1, quota_total=10, agent_id="alpha")  # no claims declared
    beta = kernel.admit(priority=10, quota_total=10, agent_id="beta")  # no claims declared

    await kernel.resources.acquire(alpha.agent_id, "X")
    await kernel.resources.acquire(beta.agent_id, "Y")

    async def alpha_wants_y():
        await kernel.scheduler.acquire_turn(alpha.agent_id)
        await kernel.resources.acquire(alpha.agent_id, "Y")

    async def beta_wants_x():
        await kernel.scheduler.acquire_turn(beta.agent_id)
        await kernel.resources.acquire(beta.agent_id, "X")

    task_a = asyncio.ensure_future(alpha_wants_y())
    task_b = asyncio.ensure_future(beta_wants_x())
    done, pending = await asyncio.wait({task_a, task_b}, timeout=2.0)
    assert not pending
    # still resolved reactively -- no declared claims means no proactive
    # refusal was possible, exactly as intended.
    assert kernel.scheduler.get("beta").state is AgentState.KILLED
    kernel.close()
