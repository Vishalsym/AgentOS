"""Tests for hallucination interception on mediated shared-KB reads (novel
mechanism #3): a kb_write can carry evidence for what it claims; a kb_read
can optionally verify the value is actually supported by that evidence
before letting it "commit" into the reading agent's own context.

Covers the heuristic verifier in isolation (including its honest
limitation -- it's lexical, not semantic, so genuine paraphrase can read as
unverified), the LLM-backed verifier against fake and real (Mock) providers,
and the full mediated syscall path in both permissive and strict modes."""

from __future__ import annotations

import pytest

from sentari.kernel import Kernel
from sentari.memory.hallucination import heuristic_verify, llm_verify
from sentari.providers.mock import MockProvider
from sentari.syscalls.layer import SyscallResult, SyscallType

# --------------------------------------------------------------------------
# heuristic_verify in isolation
# --------------------------------------------------------------------------


def test_heuristic_rejects_a_value_with_no_evidence_at_all():
    result = heuristic_verify("the sky is green", None)
    assert result.verified is False
    assert "no evidence" in result.reason


def test_heuristic_verifies_a_short_value_present_verbatim_in_evidence():
    result = heuristic_verify("142.50", "tool output: ACME closed at 142.50 today")
    assert result.verified is True


def test_heuristic_rejects_a_short_value_absent_from_evidence():
    """The classic hallucination shape: a specific fabricated number that
    doesn't actually appear anywhere in the cited source."""
    result = heuristic_verify("999.99", "tool output: ACME closed at 142.50 today")
    assert result.verified is False


def test_heuristic_verifies_a_longer_claim_with_strong_lexical_overlap():
    result = heuristic_verify(
        "the deployment failed due to a timeout connecting to the database",
        "error log: deployment failed - timeout while connecting to database host db-1",
    )
    assert result.verified is True


def test_heuristic_rejects_a_longer_claim_that_fabricates_details():
    result = heuristic_verify(
        "the deployment failed because the CEO cancelled the budget",
        "error log: deployment failed - timeout while connecting to database host db-1",
    )
    assert result.verified is False


def test_heuristic_honest_limitation_pure_paraphrase_can_read_as_unverified():
    """Documented, expected limitation: this is a lexical-overlap check,
    not a semantic one. A faithful paraphrase using different words for the
    same facts can legitimately fail the heuristic -- which is exactly why
    llm_verify exists as the stronger alternative. This test exists so the
    limitation is asserted, not silently discovered later."""
    result = heuristic_verify(
        "revenue grew by 12 percent year over year",
        "Q3 report: revenue increased 12% YoY compared to the prior year",
    )
    assert result.verified is False  # lexical overlap alone isn't enough here


# --------------------------------------------------------------------------
# llm_verify
# --------------------------------------------------------------------------


class FixedResponseProvider:
    def __init__(self, response: str) -> None:
        self._response = response

    async def complete(self, prompt: str, **kwargs: object) -> str:
        return self._response


@pytest.mark.asyncio
async def test_llm_verify_can_confirm_what_the_heuristic_missed():
    """The exact paraphrase case above -- an LLM verifier (even a stubbed
    one standing in for a real judgment) correctly recognizes support that
    pure lexical overlap couldn't."""
    provider = FixedResponseProvider("yes, the evidence supports the claim")
    result = await llm_verify(
        provider,
        "revenue grew by 12 percent year over year",
        "Q3 report: revenue increased 12% YoY compared to the prior year",
    )
    assert result.verified is True


@pytest.mark.asyncio
async def test_llm_verify_rejects_when_provider_says_no():
    provider = FixedResponseProvider("no, this is not supported")
    result = await llm_verify(provider, "claim", "unrelated evidence")
    assert result.verified is False


@pytest.mark.asyncio
async def test_llm_verify_rejects_with_no_evidence_without_even_calling_the_provider():
    calls = []

    class TrackingProvider:
        async def complete(self, prompt: str, **kwargs: object) -> str:
            calls.append(prompt)
            return "yes"

    result = await llm_verify(TrackingProvider(), "claim", None)
    assert result.verified is False
    assert calls == []  # short-circuited -- no wasted API call for an unsupported claim


@pytest.mark.asyncio
async def test_llm_verify_works_against_the_real_mock_provider():
    result = await llm_verify(MockProvider(), "some claim", "some evidence")
    assert isinstance(result.verified, bool)


# --------------------------------------------------------------------------
# Full mediated syscall path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_write_with_evidence_and_permissive_read_returns_verification_info():
    kernel = Kernel(kb_verifier=heuristic_verify)  # permissive (default: not strict)
    writer = kernel.admit(priority=1, quota_total=10, agent_id="writer")
    reader = kernel.admit(priority=1, quota_total=10, agent_id="reader")

    await kernel.syscall(
        writer.agent_id,
        SyscallType.MEMORY_WRITE,
        {"scope": "kb", "key": "price", "value": "142.50", "evidence": "tool output: closed at 142.50"},
    )

    resp = await kernel.syscall(
        reader.agent_id, SyscallType.MEMORY_READ, {"scope": "kb", "key": "price", "verify": True}
    )
    assert resp.result is SyscallResult.OK
    assert resp.value["value"] == "142.50"
    assert resp.value["verified"] is True
    kernel.close()


@pytest.mark.asyncio
async def test_kb_write_without_evidence_permissive_read_flags_unverified_but_still_returns_value():
    kernel = Kernel(kb_verifier=heuristic_verify)
    writer = kernel.admit(priority=1, quota_total=10, agent_id="writer")
    reader = kernel.admit(priority=1, quota_total=10, agent_id="reader")

    await kernel.syscall(
        writer.agent_id, SyscallType.MEMORY_WRITE, {"scope": "kb", "key": "rumor", "value": "999.99"}
    )
    resp = await kernel.syscall(
        reader.agent_id, SyscallType.MEMORY_READ, {"scope": "kb", "key": "rumor", "verify": True}
    )
    assert resp.result is SyscallResult.OK  # permissive mode: not blocked
    assert resp.value["verified"] is False
    assert resp.value["value"] == "999.99"  # value still returned -- caller decides what to do
    kernel.close()


@pytest.mark.asyncio
async def test_strict_mode_blocks_an_unsupported_read_with_a_clean_error():
    kernel = Kernel(kb_verifier=heuristic_verify, kb_verification_strict=True)
    writer = kernel.admit(priority=1, quota_total=10, agent_id="writer")
    reader = kernel.admit(priority=1, quota_total=10, agent_id="reader")

    await kernel.syscall(
        writer.agent_id, SyscallType.MEMORY_WRITE, {"scope": "kb", "key": "rumor", "value": "999.99"}
    )
    resp = await kernel.syscall(
        reader.agent_id, SyscallType.MEMORY_READ, {"scope": "kb", "key": "rumor", "verify": True}
    )
    assert resp.result is SyscallResult.ERROR
    assert "verification" in resp.error.lower()

    # the reading agent itself survives the block -- same containment
    # guarantee as any other kernel-caught failure.
    from sentari.pcb import AgentState

    assert kernel.scheduler.get("reader").state is not AgentState.KILLED
    kernel.close()


@pytest.mark.asyncio
async def test_strict_mode_still_allows_a_verified_read_through():
    kernel = Kernel(kb_verifier=heuristic_verify, kb_verification_strict=True)
    writer = kernel.admit(priority=1, quota_total=10, agent_id="writer")
    reader = kernel.admit(priority=1, quota_total=10, agent_id="reader")

    await kernel.syscall(
        writer.agent_id,
        SyscallType.MEMORY_WRITE,
        {"scope": "kb", "key": "price", "value": "142.50", "evidence": "closed at 142.50"},
    )
    resp = await kernel.syscall(
        reader.agent_id, SyscallType.MEMORY_READ, {"scope": "kb", "key": "price", "verify": True}
    )
    assert resp.result is SyscallResult.OK
    assert resp.value["verified"] is True
    kernel.close()


@pytest.mark.asyncio
async def test_no_verifier_configured_is_byte_for_byte_the_original_behavior():
    """Backward-compat guarantee: a kernel with no kb_verifier at all
    (the default) must behave exactly as before this feature existed --
    kb_read returns the bare value, verify=True or not."""
    kernel = Kernel()  # no kb_verifier
    writer = kernel.admit(priority=1, quota_total=10, agent_id="writer")
    reader = kernel.admit(priority=1, quota_total=10, agent_id="reader")

    await kernel.syscall(
        writer.agent_id, SyscallType.MEMORY_WRITE, {"scope": "kb", "key": "k", "value": "v"}
    )
    resp = await kernel.syscall(
        reader.agent_id, SyscallType.MEMORY_READ, {"scope": "kb", "key": "k", "verify": True}
    )
    assert resp.result is SyscallResult.OK
    assert resp.value == "v"  # bare string, not a dict -- unchanged shape
    kernel.close()


@pytest.mark.asyncio
async def test_verifier_configured_but_verify_not_requested_is_also_unchanged():
    """Opt-in at the *call* level too -- a verifier being configured
    kernel-wide doesn't force every read to pay for verification."""
    kernel = Kernel(kb_verifier=heuristic_verify)
    writer = kernel.admit(priority=1, quota_total=10, agent_id="writer")
    reader = kernel.admit(priority=1, quota_total=10, agent_id="reader")

    await kernel.syscall(
        writer.agent_id, SyscallType.MEMORY_WRITE, {"scope": "kb", "key": "k", "value": "v"}
    )
    resp = await kernel.syscall(reader.agent_id, SyscallType.MEMORY_READ, {"scope": "kb", "key": "k"})
    assert resp.value == "v"
    kernel.close()


@pytest.mark.asyncio
async def test_reading_a_missing_key_with_verification_requested_returns_none_not_an_error():
    kernel = Kernel(kb_verifier=heuristic_verify)
    reader = kernel.admit(priority=1, quota_total=10, agent_id="reader")
    resp = await kernel.syscall(
        reader.agent_id, SyscallType.MEMORY_READ, {"scope": "kb", "key": "nonexistent", "verify": True}
    )
    assert resp.result is SyscallResult.OK
    assert resp.value is None
    kernel.close()


@pytest.mark.asyncio
async def test_async_llm_verifier_works_through_the_full_mediated_path():
    """Confirm an async verifier (not just the sync heuristic) works
    end-to-end through on_syscall's dispatch -- inspect.isawaitable
    handling, not just the sync code path."""

    async def always_reject(value: str, evidence: str | None):
        from sentari.memory.hallucination import VerificationResult

        return VerificationResult(False, "async verifier says no, always")

    kernel = Kernel(kb_verifier=always_reject)
    writer = kernel.admit(priority=1, quota_total=10, agent_id="writer")
    reader = kernel.admit(priority=1, quota_total=10, agent_id="reader")

    await kernel.syscall(
        writer.agent_id,
        SyscallType.MEMORY_WRITE,
        {"scope": "kb", "key": "k", "value": "v", "evidence": "solid evidence"},
    )
    resp = await kernel.syscall(
        reader.agent_id, SyscallType.MEMORY_READ, {"scope": "kb", "key": "k", "verify": True}
    )
    assert resp.result is SyscallResult.OK
    assert resp.value["verified"] is False
    assert "async verifier" in resp.value["reason"]
    kernel.close()
