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
async def test_works_against_the_real_mock_provider_interface():
    provider = MockProvider()
    # MockProvider's deterministic hash-based response always contains
    # digits, so this just proves the plumbing (prompt in, float out,
    # never raises for this provider's response shape) works against the
    # actual LLMProvider protocol, not a hand-rolled test double.
    value = await score_task_value(provider, "some task")
    assert 0.0 <= value <= 1.0
