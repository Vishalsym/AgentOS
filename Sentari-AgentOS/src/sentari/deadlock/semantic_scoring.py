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
from collections.abc import Callable

_SCORING_PROMPT = (
    "On a scale from 0.0 to 1.0, how costly would it be to lose all progress "
    "on the following task if it were interrupted right now and had to be "
    "restarted from scratch by a different agent? 0.0 means trivially "
    "restartable with no real loss; 1.0 means severe, hard-to-recover loss "
    "(e.g. expensive research, a long multi-step process, or content a user "
    "is actively waiting on).\n\n"
    "Task: {task_description}\n\n"
    "Respond with ONLY your final score, on its own line, formatted exactly "
    "like this example (a decimal with two places, nothing else on that "
    "line -- no words, no restating the scale, no explanation before or "
    "after it):\n"
    "SCORE: 0.73"
)

# A genuine answer from this prompt is a decimal ("0.73"), and the model was
# told to prefix it with "SCORE:". Real-world models frequently ignore
# "respond with ONLY the number" and explain first -- often restating the
# 0.0-1.0 scale itself, which used to get mistaken for the answer by a
# naive "first number in the response" parse (the actual bug reported: two
# very different tasks both scored 1, almost certainly because the model's
# opening reasoning both times echoed a number from the scale/instructions,
# not its real judgment). This parser now, in order of preference:
#   1. looks for an explicit "SCORE: <number>" line,
#   2. else takes the LAST decimal number in the response (a trailing
#      conclusion is far more likely to be the real answer than a number
#      mentioned while restating the question),
#   3. else takes the LAST bare integer, as a last resort.
_SCORE_LABEL_RE = re.compile(r"SCORE\s*:?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_DECIMAL_RE = re.compile(r"\d+\.\d+")
_INTEGER_RE = re.compile(r"\d+")


def _parse_score(raw: str) -> float:
    labeled = _SCORE_LABEL_RE.findall(raw)
    if labeled:
        return max(0.0, min(1.0, float(labeled[-1])))
    decimals = _DECIMAL_RE.findall(raw)
    if decimals:
        return max(0.0, min(1.0, float(decimals[-1])))
    integers = _INTEGER_RE.findall(raw)
    if integers:
        return max(0.0, min(1.0, float(integers[-1])))
    raise ValueError(f"could not parse a task-value score out of provider response: {raw!r}")


async def score_task_value(
    provider,
    task_description: str,
    on_raw_response: Callable[[str], None] | None = None,
) -> float:
    """Ask any `LLMProvider` (real or mock) to rate how costly losing this
    task's progress would be, in [0.0, 1.0]. If none of the parse
    strategies above find a number, this raises rather than silently
    guessing -- callers should treat a scoring failure as "no opinion"
    (pass task_value=None) rather than fabricate one.

    `on_raw_response`, if given, is called with the provider's exact raw
    text (before parsing) -- useful to log/display for transparency or to
    debug a bad score, without changing this function's return type."""
    response = await provider.complete(_SCORING_PROMPT.format(task_description=task_description))
    if on_raw_response is not None:
        on_raw_response(response)
    return _parse_score(response)
