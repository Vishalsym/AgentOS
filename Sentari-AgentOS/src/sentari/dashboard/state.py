from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

from sentari.notifications.notifier import CompositeNotifier, LogNotifier, Notifier, WebhookNotifier


def build_default_notifier() -> Notifier:
    """LogNotifier is always on. If SENTARI_WEBHOOK_URL is set in the
    environment, kills and quota-exhaustion events are also POSTed there
    (e.g. a Slack incoming webhook) -- opt-in, zero code changes needed to
    use it, and a dead/misconfigured webhook never affects the kernel
    (WebhookNotifier contains its own failures)."""
    log_notifier = LogNotifier()
    webhook_url = os.environ.get("SENTARI_WEBHOOK_URL")
    if webhook_url:
        return CompositeNotifier([log_notifier, WebhookNotifier(webhook_url)])
    return log_notifier


def build_default_provider() -> Any:
    """MockProvider (deterministic, offline, free) unless ANTHROPIC_API_KEY
    is set in the environment, in which case the dashboard's live kernel
    uses the real AnthropicProvider instead -- so firing a tool_call with a
    prompt from the Cockpit/agent-drawer UI hits the real API and the
    response you see on screen is genuinely Claude's, not a canned mock.
    Zero code changes needed to use it: just set the env var before
    starting the dashboard. Falls back to the mock (never crashes) if the
    key turns out to be missing/invalid when a real call is first made --
    see AnthropicProvider's own lazy-client error handling."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from sentari.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    from sentari.providers.mock import MockProvider

    return MockProvider()


@dataclass
class DashboardState:
    """Shared, lock-protected state between the running demo scenario
    (a background asyncio task) and the FastAPI request handlers polling
    it. The kernel itself is the source of truth for agent/quota state;
    this only tracks narration and kernel-level decisions (kills,
    preemptions) that aren't syscalls and so wouldn't otherwise show up
    in the syscall_log."""

    kernel: Any = None
    scenario: str = "standard"
    narration: str = "Starting the kernel..."
    phase: str = "boot"
    done: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)
    cycle_notice: dict[str, str] | None = None
    comparison: dict[str, Any] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def narrate(self, text: str, phase: str) -> None:
        async with self.lock:
            self.narration = text
            self.phase = phase

    async def log_event(self, source: str, text: str, level: str = "info") -> None:
        async with self.lock:
            self.events.append({"time": time.time(), "source": source, "text": text, "level": level})

    async def set_cycle_notice(self, notice: dict[str, str] | None) -> None:
        async with self.lock:
            self.cycle_notice = notice

    async def set_comparison(self, comparison: dict[str, Any] | None) -> None:
        async with self.lock:
            self.comparison = comparison

    async def finish(self) -> None:
        async with self.lock:
            self.done = True
            self.narration = "Scenario complete. Everything below is live kernel state, read from SQLite."
            self.phase = "done"
