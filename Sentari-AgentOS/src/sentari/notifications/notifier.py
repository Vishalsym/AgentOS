from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("sentari.notifications")


@dataclass
class NotificationEvent:
    """A significant, human-worth-knowing-about kernel decision. Distinct
    from the syscall_log (which records every mediated call, success or
    not): a NotificationEvent only fires for outcomes an operator would
    plausibly want pushed to them proactively -- an agent was killed, a
    deadlock was resolved, an agent ran out of budget."""

    event_type: str  # "agent_killed" | "quota_exhausted"
    agent_id: str
    message: str
    reason: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class Notifier(Protocol):
    async def notify(self, event: NotificationEvent) -> None: ...


class LogNotifier:
    """Default notifier: writes to the standard `logging` module. Zero
    configuration, zero external dependencies, always on -- every other
    notifier is opt-in on top of this, never a replacement for it, so an
    operator with no webhook/SMTP configured still sees these events
    somewhere (the process's own logs)."""

    def __init__(self) -> None:
        self.events: list[NotificationEvent] = []

    async def notify(self, event: NotificationEvent) -> None:
        self.events.append(event)
        logger.warning(
            "[%s] agent=%s reason=%s -- %s", event.event_type, event.agent_id, event.reason, event.message
        )


class WebhookNotifier:
    """POSTs the event as JSON to a configured URL (e.g. a Slack incoming
    webhook, or a generic ops endpoint). Uses `httpx` lazily -- imported
    only on first real use, so the base package never requires it just to
    construct a WebhookNotifier (matches AnthropicProvider's lazy-client
    pattern). A failed delivery is logged and swallowed, never raised --
    an unreachable webhook must not be able to affect kernel behavior."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self._url = url
        self._timeout = timeout

    async def notify(self, event: NotificationEvent) -> None:
        try:
            import httpx

            payload = {
                "event_type": event.event_type,
                "agent_id": event.agent_id,
                "message": event.message,
                "reason": event.reason,
                "timestamp": event.timestamp,
                "metadata": event.metadata,
            }
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, json=payload)
                resp.raise_for_status()
        except Exception:  # noqa: BLE001 -- a notification sink must never break the kernel
            logger.exception(
                "WebhookNotifier: failed to deliver event %s for agent %s", event.event_type, event.agent_id
            )


class CompositeNotifier:
    """Fan a single event out to multiple sinks (e.g. LogNotifier always,
    plus a WebhookNotifier if one is configured). Each sink's failure is
    isolated from the others, same containment guarantee as a single sink."""

    def __init__(self, notifiers: list[Notifier]) -> None:
        self._notifiers = notifiers

    async def notify(self, event: NotificationEvent) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.notify(event)
            except Exception:  # noqa: BLE001 -- one sink's failure must not skip the rest
                logger.exception("CompositeNotifier: sink %r failed on event %s", notifier, event.event_type)
