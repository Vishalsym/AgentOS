"""Optional LLM-backed task-value scoring for semantic-value-aware deadlock
resolution (novel mechanism #1).

Deliberately NOT called during deadlock detection itself -- `DeadlockDetector
.add_wait_edge` is synchronous and, per `scripts/benchmark.py`, resolves a
real cycle in well under a millisecond; an LLM round-trip there (hundreds of
ms) would make semantic scoring cost more than the thing it's protecting.
Instead, score the agent's task *once*, before admission, and pass the
result as `Kernel.admit(..., task_value=...)` -- the PCB carries the score,
and DeadlockDetector just reads it back synchronously if a cycle ever forms.

Usage:
    from sentari.deadlock.semantic_scoring import score_task_value

    value = await score_task_value(provider, "draft the Q3 board memo")
    kernel.admit(priority=5, quota_total=20, agent_id="writer", task_value=value)
"""

from __future__ import annotations

import re

_SCORING_PROMPT = (
    "On a scale from 0.0 to 1.0, how costly would it be to lose all progress "
    "on the following task if it were interrupted right now and had to be "
    "restarted from scratch by a different agent? 0.0 means trivially "
    "restartable with no real loss; 1.0 means severe, hard-to-recover loss "
    "(e.g. expensive research, a long multi-step process, or content a user "
    "is actively waiting on). Respond with ONLY the number, nothing else.\n\n"
    "Task: {task_description}"
)

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _parse_score(raw: str) -> float:
    match = _NUMBER_RE.search(raw)
    if not match:
        raise ValueError(f"could not parse a task-value score out of provider response: {raw!r}")
    value = float(match.group(1))
    return max(0.0, min(1.0, value))


async def score_task_value(provider, task_description: str) -> float:
    """Ask any `LLMProvider` (real or mock) to rate how costly losing this
    task's progress would be, in [0.0, 1.0]. Robust to a provider that
    doesn't answer with *only* a bare number (common in practice) by
    extracting the first number in the response; if none is found, this
    raises rather than silently guessing -- callers should treat a scoring
    failure as "no opinion" (pass task_value=None) rather than fabricate one.
    """
    response = await provider.complete(_SCORING_PROMPT.format(task_description=task_description))
    return _parse_score(response)
