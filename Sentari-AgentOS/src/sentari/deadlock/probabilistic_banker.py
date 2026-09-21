"""Probabilistic generalization of the Banker's Algorithm (novel mechanism
#2) for proactive deadlock AVOIDANCE, complementing (not replacing) the
existing reactive `DeadlockDetector`.

The classical Banker's Algorithm requires every process to declare its
EXACT maximum future resource claim in advance, and proves a request keeps
the system in a provably safe state before granting it. LLM agents can't
do that: an agent frequently doesn't know precisely what it will need
until it gets there, and it can spawn children with their own needs
mid-flight -- there is no fixed, known-in-advance claim matrix the way
real-time OS theory assumes.

This module instead lets an agent declare a *probability* it will
eventually want a given resource (`AgentPCB.declared_resource_claims`),
and uses that, at the moment another agent is about to BLOCK waiting on a
resource, to estimate the risk that this specific wait is the first half
of a soon-to-form AB-BA mutual-wait cycle -- and can refuse to grant it
proactively, before any real cycle exists for the reactive detector to
catch. This is deliberately an *additional*, opt-in layer: with no
ProbabilisticBanker configured, or no claims declared, behavior is
unchanged, and the reactive detector still catches and resolves any
deadlock that does form (correctly, and fast, per the benchmark suite).
"""

from __future__ import annotations

from dataclasses import dataclass


class ResourceRequestUnsafeError(RuntimeError):
    """Raised when a prospective wait is refused proactively because the
    estimated mutual-wait risk exceeds the configured threshold. Contained
    by SyscallLayer.on_syscall like any other tool failure -- the caller
    gets a clean ERROR response, not a crashed kernel, and is free to
    retry with a different strategy (e.g. request resources in a different
    order, or back off)."""


@dataclass
class AvoidanceCheck:
    safe: bool
    risk: float
    reason: str


class ProbabilisticBanker:
    """`unsafe_risk_threshold`: refuse a wait once the estimated
    probability of a reciprocal claim reaches this level. Lower = more
    cautious (refuses more waits, including some that would have been
    fine); higher = closer to never intervening (falls back entirely to
    the reactive detector)."""

    def __init__(self, unsafe_risk_threshold: float = 0.7):
        self.unsafe_risk_threshold = unsafe_risk_threshold

    def check(
        self,
        waiter_id: str,
        holder_id: str,
        resource_key: str,
        holder_claims: dict[str, float],
        held_by_waiter: set[str],
    ) -> AvoidanceCheck:
        """The waiter definitely wants `resource_key` right now (this is a
        live request, probability 1.0 by construction) -- the only
        uncertain side of a prospective AB-BA cycle is whether the holder
        will eventually reciprocate by wanting something the waiter
        currently holds. Risk = P(holder eventually wants at least one
        resource the waiter holds), estimated under an explicit
        independence assumption across the waiter's held resources
        (deliberately simple and conservative, not a claim of statistical
        rigor -- documented, not hidden): P(at least one) = 1 - prod(1-p_i).
        """
        reciprocal_probs = [holder_claims.get(rk, 0.0) for rk in held_by_waiter]
        if not reciprocal_probs:
            return AvoidanceCheck(
                True, 0.0, "waiter holds nothing the holder has any declared interest in"
            )
        prob_none_reciprocate = 1.0
        for p in reciprocal_probs:
            prob_none_reciprocate *= 1.0 - max(0.0, min(1.0, p))
        risk = 1.0 - prob_none_reciprocate

        if risk >= self.unsafe_risk_threshold:
            return AvoidanceCheck(
                False,
                risk,
                f"holder '{holder_id}' has an estimated {risk:.0%} chance of eventually wanting "
                f"a resource waiter '{waiter_id}' already holds -- refusing this wait proactively "
                f"to avoid a likely AB-BA cycle before it forms",
            )
        return AvoidanceCheck(True, risk, f"estimated mutual-wait risk {risk:.0%} is below threshold")
