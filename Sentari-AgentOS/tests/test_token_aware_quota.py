"""Tests for token-aware quota (built for the live-provider demo): when
enabled, a tool_call with a prompt is charged by REAL tokens (input +
output, read back from the provider) instead of a flat 1-per-call, and its
max_tokens is capped at the agent's remaining budget so a single call
can't blow through it. If that cap cuts the response short, the partial
text is still returned, flagged as truncated.

Uses MockProvider.complete_metered (deterministic, offline) so this suite
needs no real API key; the exact same code path is what actually runs
against AnthropicProvider.complete_metered when a real key is present."""

from __future__ import annotations

import pytest

from sentari.kernel import Kernel
from sentari.providers.mock import MockProvider
from sentari.syscalls.layer import SyscallResult, SyscallType


@pytest.mark.asyncio
async def test_default_kernel_is_unaffected_flat_call_count_quota():
    """Backward-compat guarantee: token_aware_quota defaults to False --
    a tool_call with a prompt still costs exactly 1 unit, exactly as
    every other test in this suite already assumes."""
    kernel = Kernel()
    agent = kernel.admit(priority=1, quota_total=5, agent_id="a1")
    resp = await kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"prompt": "hello"})
    assert resp.result is SyscallResult.OK
    assert resp.truncated is False
    assert kernel.scheduler.get("a1").quota_used == 1
    kernel.close()


@pytest.mark.asyncio
async def test_token_aware_quota_charges_real_tokens_not_a_flat_one():
    kernel = Kernel(token_aware_quota=True)
    agent = kernel.admit(priority=1, quota_total=2000, agent_id="a1")
    resp = await kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"prompt": "hello world"})
    assert resp.result is SyscallResult.OK
    assert resp.truncated is False

    pcb = kernel.scheduler.get("a1")
    # MockProvider's simulated full response is 40 "words"/tokens, plus
    # a couple of input tokens for "hello world" -- nowhere near the flat
    # value of 1, and reflects what was actually generated.
    assert pcb.quota_used > 1
    assert pcb.quota_used == 40 + 2  # 40 output words + 2 input words, deterministic
    kernel.close()


@pytest.mark.asyncio
async def test_token_aware_quota_truncates_when_budget_is_small():
    """The user's exact scenario: a tight token budget (well under what a
    full response needs) should stop generation at the cap and hand back
    the partial result with a clear truncated flag -- not silently return
    an incomplete answer, and not deny the call outright either."""
    kernel = Kernel(token_aware_quota=True)
    agent = kernel.admit(priority=1, quota_total=10, agent_id="a1")  # well under the 40-word "full" reply

    resp = await kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"prompt": "tell me everything"})
    assert resp.result is SyscallResult.OK  # not denied -- a real partial result came back
    assert resp.truncated is True
    assert resp.value  # non-empty partial text
    assert len(resp.value.split()) <= 10  # capped at (roughly) the remaining budget

    pcb = kernel.scheduler.get("a1")
    assert pcb.quota_used >= pcb.quota_total  # the cap was fully consumed
    kernel.close()


@pytest.mark.asyncio
async def test_a_second_call_after_budget_exhausted_is_denied_not_truncated():
    """Once the budget is genuinely gone, the *next* call hits the normal
    admission-time DENY path -- truncation only applies to the call that
    was actually still allowed to run."""
    kernel = Kernel(token_aware_quota=True)
    agent = kernel.admit(priority=1, quota_total=5, agent_id="a1")

    first = await kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"prompt": "hi"})
    assert first.truncated is True  # 5 tokens is far under the 40-word full reply

    second = await kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"prompt": "hi again"})
    assert second.result is SyscallResult.DENY
    assert "quota exhausted" in second.error
    assert second.truncated is False  # denied outright, never ran at all
    kernel.close()


@pytest.mark.asyncio
async def test_generous_budget_never_truncates():
    kernel = Kernel(token_aware_quota=True)
    agent = kernel.admit(priority=1, quota_total=2000, agent_id="a1")
    resp = await kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"prompt": "hi"})
    assert resp.truncated is False
    assert kernel.scheduler.get("a1").quota_used < 2000
    kernel.close()


@pytest.mark.asyncio
async def test_token_aware_quota_has_no_effect_on_non_prompt_tool_calls():
    """A tool_call using `fn` (not `prompt`) has no tokens to meter -- it
    must fall back to the ordinary flat cost, not error out or charge 0."""
    kernel = Kernel(token_aware_quota=True)
    agent = kernel.admit(priority=1, quota_total=5, agent_id="a1")

    async def plain_tool() -> str:
        return "did a thing"

    resp = await kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"fn": plain_tool})
    assert resp.result is SyscallResult.OK
    assert resp.value == "did a thing"
    assert resp.truncated is False
    assert kernel.scheduler.get("a1").quota_used == 1
    kernel.close()


@pytest.mark.asyncio
async def test_token_aware_quota_with_a_provider_lacking_complete_metered_falls_back():
    """A provider that only implements the plain complete() (no metering
    support) must not crash token-aware mode -- it just behaves like the
    flat-cost model for that call."""

    class BareProvider:
        async def complete(self, prompt: str, **kwargs: object) -> str:
            return "plain response, no metering here"

    kernel = Kernel(provider=BareProvider(), token_aware_quota=True)
    agent = kernel.admit(priority=1, quota_total=5, agent_id="a1")
    resp = await kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"prompt": "hi"})
    assert resp.result is SyscallResult.OK
    assert resp.value == "plain response, no metering here"
    assert kernel.scheduler.get("a1").quota_used == 1
    kernel.close()


@pytest.mark.asyncio
async def test_mock_provider_complete_metered_directly():
    provider = MockProvider()
    generous = await provider.complete_metered("a prompt", max_tokens=1000)
    assert generous.truncated is False
    assert generous.output_tokens == 40
    assert generous.input_tokens == 2

    tight = await provider.complete_metered("a prompt", max_tokens=5)
    assert tight.truncated is True
    assert tight.output_tokens == 5
    assert len(tight.text.split()) == 5
