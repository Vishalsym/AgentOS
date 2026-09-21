"""Hallucination interception for mediated shared-memory reads (novel
mechanism #3).

Memory isolation (FR-10, FR-11) already stops an agent from *directly*
touching another agent's private context -- but the shared knowledge base
is, by design, a channel every agent can read from. Nothing currently
checks whether what's written there is actually *true* before another
agent's context absorbs it. This module adds that check at the mediation
layer: a `kb_write` can optionally carry `evidence` (what the value was
actually derived from), and a `kb_read` can optionally verify the value is
still supported by that evidence before the read is allowed to "commit" --
turning hallucination containment into a kernel-mediation concern instead
of a prompt-engineering one.

Two verifiers are provided:
  - `heuristic_verify`: synchronous, deterministic, no LLM call -- a real,
    working default based on lexical overlap between the value and its
    cited evidence.
  - `llm_verify`: asks an LLMProvider whether the evidence actually
    supports the value. Async, so it's only appropriate where a caller
    explicitly opts into the extra latency (memory reads are not the
    sub-millisecond-critical path deadlock detection is).

Both are entirely opt-in at the SyscallLayer level (`kb_verifier=None` by
default) -- omitting them reproduces the exact prior kb_read behavior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class VerificationResult:
    verified: bool
    reason: str


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def heuristic_verify(value: str, evidence: str | None) -> VerificationResult:
    """No evidence recorded at all -> unverified (this is the whole point:
    an unsupported claim should not be silently trusted just because it's
    sitting in the shared KB). Otherwise, verified if a meaningful share of
    the value's own content-word tokens actually appear in the cited
    evidence -- a real, if simple, lexical-support check, not a rubber
    stamp. Short/numeric values (e.g. "142.50") use substring containment
    instead, since token-overlap on a single short token is too coarse."""
    if not evidence or not evidence.strip():
        return VerificationResult(False, "no evidence was recorded for this value at write time")

    value_stripped = value.strip()
    evidence_lower = evidence.lower()

    if len(value_stripped) <= 12 or " " not in value_stripped:
        if value_stripped.lower() in evidence_lower:
            return VerificationResult(True, "value appears verbatim in the cited evidence")
        return VerificationResult(False, "short/atomic value does not appear in the cited evidence")

    value_tokens = _tokenize(value_stripped)
    if not value_tokens:
        return VerificationResult(False, "value has no meaningful content to check")
    evidence_tokens = _tokenize(evidence)
    overlap = value_tokens & evidence_tokens
    coverage = len(overlap) / len(value_tokens)
    if coverage >= 0.6:
        return VerificationResult(True, f"{coverage:.0%} of the value's content words are backed by the evidence")
    return VerificationResult(False, f"only {coverage:.0%} of the value's content words appear in the evidence")


_LLM_VERIFY_PROMPT = (
    "A shared knowledge base entry claims the following:\n\n"
    'CLAIM: "{value}"\n\n'
    "It cites this as supporting evidence:\n\n"
    'EVIDENCE: "{evidence}"\n\n'
    "Does the evidence actually support the claim, with no unsupported "
    'additions? Respond with ONLY "yes" or "no".'
)


async def llm_verify(provider, value: str, evidence: str | None) -> VerificationResult:
    if not evidence or not evidence.strip():
        return VerificationResult(False, "no evidence was recorded for this value at write time")
    response = await provider.complete(_LLM_VERIFY_PROMPT.format(value=value, evidence=evidence))
    verified = response.strip().lower().startswith("y")
    return VerificationResult(verified, f"LLM verdict: {response.strip()!r}")


class HallucinationSuspectedError(RuntimeError):
    """Raised (in strict mode) when a mediated kb_read's verifier finds the
    value is not supported by its recorded evidence. Contained by
    SyscallLayer.on_syscall exactly like any other tool failure -- the
    reading agent gets a clean ERROR response, not a crashed kernel."""
