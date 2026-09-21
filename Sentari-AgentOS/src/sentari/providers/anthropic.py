from __future__ import annotations

import os

from sentari.providers.base import CompletionResult


class AnthropicProvider:
    """Real Anthropic Messages API provider. Satisfies the SRS constraint of
    at least one real provider with real rate limits. Requires
    ANTHROPIC_API_KEY; the client is created lazily so importing this module
    never requires the key or the `anthropic` package to be present unless
    it's actually used."""

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None):
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def complete(self, prompt: str, **kwargs: object) -> str:
        client = self._ensure_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_text(response)

    async def complete_metered(self, prompt: str, max_tokens: int) -> CompletionResult:
        """Same call as `complete`, but reads the real per-call token usage
        back off the response (`response.usage`) and detects a hard cutoff
        via `response.stop_reason == "max_tokens"` -- both genuine signals
        from the Anthropic API, not estimated. `max_tokens` is clamped to
        at least 1 since the API rejects 0 -- the caller (SyscallLayer) is
        responsible for never calling this with a non-positive remaining
        budget in the first place (the ordinary admission-time quota check
        already denies that case before dispatch is ever reached)."""
        client = self._ensure_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=max(1, max_tokens),
            messages=[{"role": "user", "content": prompt}],
        )
        return CompletionResult(
            text=_extract_text(response),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            truncated=response.stop_reason == "max_tokens",
        )


def _extract_text(response) -> str:
    """A Messages API response's `content` is a list of typed blocks, not
    always just one text block -- extended-thinking-capable models (like
    Sonnet 5, used by default here) can prepend a `thinking` block before
    the actual `text` block, and a response cut off by max_tokens mid-
    thought can even end with only a thinking block and no text yet.
    Blindly indexing content[0].text breaks the moment a non-text block
    comes first (a real bug this fixes, not a hypothetical one) -- so
    this instead concatenates every actual text block and ignores the
    rest, returning "" rather than raising if no text was produced yet."""
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
