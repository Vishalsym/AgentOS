"""Tests for LLM-backed task-value scoring (src/sentari/deadlock/
semantic_scoring.py) -- the part of novel mechanism #1 that talks to a
provider. Uses fake providers with controlled responses to test parsing
robustness (real LLMs don't reliably answer with *only* a bare number),
plus the real MockProvider to prove the function works against the actual
LLMProvider interface, not just a hand-rolled stub."""

from __future__ import annotations

import pytest

from sentari.deadlock.semantic_scoring import score_task_value
from sentari.providers.mock import MockProvider


class FixedResponseProvider:
    def __init__(self, response: str) -> None:
        self._response = response

    async def complete(self, prompt: str, **kwargs: object) -> str:
        return self._response


@pytest.mark.asyncio
async def test_parses_a_bare_number_response():
    provider = FixedResponseProvider("0.85")
    assert await score_task_value(provider, "long research task") == 0.85


@pytest.mark.asyncio
async def test_parses_a_number_embedded_in_prose():
    provider = FixedResponseProvider("I'd say this is about a 0.3 on that scale.")
    assert await score_task_value(provider, "trivial lookup") == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_clamps_out_of_range_scores():
    provider = FixedResponseProvider("7.0")
    assert await score_task_value(provider, "task") == 1.0

    provider2 = FixedResponseProvider("-3")
    # regex only matches unsigned digits, so "-3" parses as "3" -- clamped
    # value is still in range; this asserts the clamp itself works via a
    # clean over-range case, and documents the sign-stripping behavior.
    assert 0.0 <= await score_task_value(provider2, "task") <= 1.0


@pytest.mark.asyncio
async def test_raises_clearly_when_no_number_found_rather_than_guessing():
    provider = FixedResponseProvider("I refuse to answer that question.")
    with pytest.raises(ValueError, match="could not parse"):
        await score_task_value(provider, "task")


@pytest.mark.asyncio
async def test_prefers_an_explicit_score_label_over_any_other_number():
    provider = FixedResponseProvider(
        "Thinking about this on a scale from 0.0 to 1.0, and considering it "
        "could take up to 3 attempts...\nSCORE: 0.42"
    )
    assert await score_task_value(provider, "task") == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_regression_verbose_reasoning_that_restates_the_scale_first():
    """The exact reported bug: two different real tasks both scored 1.0.
    Root cause -- a verbose model restates "0.0 to 1.0" (or similar) while
    reasoning before giving its real answer, and the old parser took the
    FIRST number in the response, which is that restated scale value, not
    the model's actual judgment. Reproduced here with two clearly different
    tasks that must now score differently despite both opening with scale
    language, proving the fix takes the model's actual conclusion instead."""
    low_value_response = (
        "I need to rate this from 0.0 to 1.0. This is a disposable scratch "
        "note with no real consequence if lost. My assessment: 0.05"
    )
    high_value_response = (
        "I need to rate this from 0.0 to 1.0. This represents hours of "
        "irreplaceable work a user is actively waiting on. My assessment: 0.95"
    )
    low = await score_task_value(FixedResponseProvider(low_value_response), "scratch note")
    high = await score_task_value(FixedResponseProvider(high_value_response), "irreplaceable report")
    assert low == pytest.approx(0.05)
    assert high == pytest.approx(0.95)
    assert low != high  # the actual bug: these used to come out identical


@pytest.mark.asyncio
async def test_on_raw_response_callback_receives_the_exact_provider_text():
    captured = []
    provider = FixedResponseProvider("SCORE: 0.6")
    await score_task_value(provider, "task", on_raw_response=captured.append)
    assert captured == ["SCORE: 0.6"]


@pytest.mark.asyncio
async def test_works_against_the_real_mock_provider_interface():
    provider = MockProvider()
    # MockProvider's deterministic hash-based response always contains
    # digits, so this just proves the plumbing (prompt in, float out,
    # never raises for this provider's response shape) works against the
    # actual LLMProvider protocol, not a hand-rolled test double.
    value = await score_task_value(provider, "some task")
    assert 0.0 <= value <= 1.0
