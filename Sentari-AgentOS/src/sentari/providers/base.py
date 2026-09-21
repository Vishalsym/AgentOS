from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMProvider(Protocol):
    """Pluggable LLM backend. All calls to a provider must go through the
    syscall layer's `tool_call` -- agents never hold a reference to one
    directly (FR-7, NFR-Security)."""

    async def complete(self, prompt: str, **kwargs: object) -> str: ...


@dataclass
class CompletionResult:
    """Real, token-metered completion result (token-aware quota mode).
    `truncated=True` means the provider's own `max_tokens` cap cut the
    response short -- the exact real-API signal, not a guess -- so the
    kernel can charge quota by *actual* tokens spent and report a clean
    "token limit reached, partial result" outcome instead of silently
    returning a truncated answer as if it were complete."""

    text: str
    input_tokens: int
    output_tokens: int
    truncated: bool


class MeteredLLMProvider(Protocol):
    """Optional extension of LLMProvider: a provider that can report real
    per-call token usage and honor a hard max_tokens cap. Checked for via
    `hasattr(provider, "complete_metered")`, not required by every
    provider -- token-aware quota mode (SyscallLayer(token_aware_quota=True))
    simply falls back to the plain, call-counted quota model for a
    provider that doesn't implement it."""

    async def complete_metered(self, prompt: str, max_tokens: int) -> CompletionResult: ...
