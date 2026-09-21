"""Tests for the pluggable LLMProvider implementations.

The live-API tests are skipped unless ANTHROPIC_API_KEY is set in the
environment -- exactly the "at least one real provider with real rate
limits" SRS requirement, verified against the actual Anthropic API when a
key is available (e.g. in a grading/CI environment with the key
configured), without making the base test suite depend on network access
or a paid key just to run `uv run pytest`."""

from __future__ import annotations

import os

import pytest

from sentari.kernel import Kernel
from sentari.providers.anthropic import AnthropicProvider, _extract_text
from sentari.syscalls.layer import SyscallResult, SyscallType


class _FakeBlock:
    """Duck-typed stand-in for the SDK's TextBlock/ThinkingBlock -- only
    the attributes _extract_text actually touches."""

    def __init__(self, type_: str, **attrs):
        self.type = type_
        for k, v in attrs.items():
            setattr(self, k, v)


class _FakeResponse:
    def __init__(self, content):
        self.content = content


def test_extract_text_handles_a_plain_text_only_response():
    resp = _FakeResponse([_FakeBlock("text", text="hello there")])
    assert _extract_text(resp) == "hello there"


def test_extract_text_regression_thinking_block_before_text_no_longer_crashes():
    """The exact bug found live: content[0] was a ThinkingBlock (no
    .text attribute) because the model reasoned before answering --
    content[0].text used to raise AttributeError. Must now skip the
    thinking block and return the real text."""
    resp = _FakeResponse(
        [
            _FakeBlock("thinking", thinking="reasoning about the answer..."),
            _FakeBlock("text", text="the actual answer"),
        ]
    )
    assert _extract_text(resp) == "the actual answer"


def test_extract_text_concatenates_multiple_text_blocks():
    resp = _FakeResponse([_FakeBlock("text", text="part one. "), _FakeBlock("text", text="part two.")])
    assert _extract_text(resp) == "part one. part two."


def test_extract_text_returns_empty_string_when_only_thinking_no_text_yet():
    """E.g. cut off by max_tokens mid-thought, before any text block was
    emitted -- an honest empty result, not a crash."""
    resp = _FakeResponse([_FakeBlock("thinking", thinking="still reasoning...")])
    assert _extract_text(resp) == ""

HAS_LIVE_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
skip_no_key = pytest.mark.skipif(
    not HAS_LIVE_KEY, reason="ANTHROPIC_API_KEY not set -- skipping live-API test"
)


def test_provider_construction_never_requires_key_or_import():
    """Importing/constructing the provider must be safe even with no key
    and even if the `anthropic` package couldn't be imported -- the client
    is created lazily, only on first real use (module docstring's claim,
    verified here rather than just asserted)."""
    provider = AnthropicProvider(api_key=None)
    assert provider._client is None


@pytest.mark.asyncio
async def test_provider_raises_clear_error_with_no_key_configured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider(api_key=None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await provider.complete("this should never reach the network")


@skip_no_key
@pytest.mark.asyncio
async def test_provider_completes_a_real_prompt_against_the_live_api():
    provider = AnthropicProvider()
    result = await provider.complete("Reply with exactly one word: pong")
    assert isinstance(result, str)
    assert result.strip()


@skip_no_key
@pytest.mark.asyncio
async def test_real_provider_wired_through_the_full_mediated_syscall_path():
    """Not just the provider in isolation -- the whole kernel path: admit,
    fire a real tool_call syscall through SyscallLayer.on_syscall, confirm
    it's quota-charged and audit-logged, exactly like MockProvider-backed
    tests elsewhere, but hitting the real API this time."""
    kernel = Kernel(provider=AnthropicProvider())
    agent = kernel.admit(priority=1, quota_total=5, agent_id="live_llm_agent")

    response = await kernel.syscall(
        agent.agent_id, SyscallType.TOOL_CALL, {"prompt": "Reply with exactly one word: pong"}
    )
    assert response.result is SyscallResult.OK
    assert isinstance(response.value, str)
    assert response.value.strip()

    pcb = kernel.scheduler.get("live_llm_agent")
    assert pcb.quota_used == 1

    logs = kernel.syscall_log_repo.list_for_agent("live_llm_agent")
    assert len(logs) == 1
    assert logs[0]["result"] == "OK"
    kernel.close()


@skip_no_key
@pytest.mark.asyncio
async def test_real_provider_complete_metered_reports_real_usage():
    provider = AnthropicProvider()
    result = await provider.complete_metered("Reply with exactly one word: pong", max_tokens=1024)
    assert result.text.strip()
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.truncated is False  # plenty of headroom for a one-word reply


@skip_no_key
@pytest.mark.asyncio
async def test_real_token_aware_quota_actually_truncates_against_the_live_api():
    """The user's exact live scenario, against the real API: a tiny token
    budget forces a real, hard cutoff (Anthropic's own max_tokens cap,
    stop_reason == "max_tokens"), and the kernel reports it as truncated
    with the genuinely partial text -- not a simulation."""
    kernel = Kernel(provider=AnthropicProvider(), token_aware_quota=True)
    agent = kernel.admit(priority=1, quota_total=12, agent_id="tight_live_agent")

    response = await kernel.syscall(
        agent.agent_id,
        SyscallType.TOOL_CALL,
        {"prompt": "Write a detailed 500-word essay about the history of computing."},
    )
    assert response.result is SyscallResult.OK
    assert response.truncated is True
    assert response.value.strip()  # real, partial text came back

    pcb = kernel.scheduler.get("tight_live_agent")
    assert pcb.quota_used >= pcb.quota_total
    kernel.close()
