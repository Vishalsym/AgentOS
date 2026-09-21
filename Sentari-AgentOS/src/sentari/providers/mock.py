from __future__ import annotations

import hashlib

from sentari.providers.base import CompletionResult

# Deterministic "intended full response" length for complete_metered's
# simulation -- if the caller's max_tokens is smaller than this, the mock
# response is truncated to exactly max_tokens and reported as such, so
# token-aware-quota tests can exercise real truncation behavior offline.
_SIMULATED_FULL_OUTPUT_TOKENS = 40


class MockProvider:
    """Deterministic, offline LLM provider. Default for tests and CI so the
    kernel's behavior can be verified without network access or real
    rate limits."""

    async def complete(self, prompt: str, **kwargs: object) -> str:
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return f"[mock-response {digest}] {prompt[:50]}"

    async def complete_metered(self, prompt: str, max_tokens: int) -> CompletionResult:
        """A "token" here is simulated as one word, so behavior is easy to
        reason about and assert on in tests -- not a claim of matching any
        real tokenizer. The simulated "intended full response" is always
        exactly `_SIMULATED_FULL_OUTPUT_TOKENS` words (deterministic,
        independent of prompt length), so a test can reliably choose a
        max_tokens value above or below it to force either outcome."""
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        full_words = [f"{digest}-word{i}" for i in range(_SIMULATED_FULL_OUTPUT_TOKENS)]
        input_tokens = max(1, len(prompt.split()))

        if len(full_words) > max_tokens:
            text = " ".join(full_words[:max_tokens])
            return CompletionResult(
                text=text, input_tokens=input_tokens, output_tokens=max_tokens, truncated=True
            )
        return CompletionResult(
            text=" ".join(full_words),
            input_tokens=input_tokens,
            output_tokens=len(full_words),
            truncated=False,
        )
