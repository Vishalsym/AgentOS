"""Tests for the notification system (README roadmap: dashboard/email
notifications). Covers the sinks in isolation, the KillManager wiring that
actually fires them, and a full kernel-level deadlock scenario proving a
real kill produces a real notification -- not just a unit test of the
notifier class alone."""

from __future__ import annotations

import httpx
import pytest

from sentari.kernel import Kernel
from sentari.notifications.notifier import (
    CompositeNotifier,
    LogNotifier,
    NotificationEvent,
    WebhookNotifier,
)
from sentari.pcb import AgentState
from sentari.persistence.db import connect
from sentari.persistence.repositories import AgentRepo
from sentari.preemption.kill_manager import KillManager
from sentari.scheduler.scheduler import Scheduler


@pytest.mark.asyncio
async def test_log_notifier_captures_events():
    notifier = LogNotifier()
    event = NotificationEvent(event_type="agent_killed", agent_id="x", message="test")
    await notifier.notify(event)
    assert notifier.events == [event]


@pytest.mark.asyncio
async def test_kill_manager_fires_kill_notification():
    repo = AgentRepo(connect(":memory:"))
    scheduler = Scheduler(repo)
    notifier = LogNotifier()
    km = KillManager(scheduler, repo, notifier=notifier)

    from sentari.pcb import AgentPCB

    pcb = AgentPCB.new(priority=5, quota_total=10, agent_id="victim")
    scheduler.admit(pcb)

    await km.kill("victim", reason="deadlock_victim")

    assert len(notifier.events) == 1
    event = notifier.events[0]
    assert event.event_type == "agent_killed"
    assert event.agent_id == "victim"
    assert event.reason == "deadlock_victim"


@pytest.mark.asyncio
async def test_kill_manager_fires_quota_exhausted_notification():
    repo = AgentRepo(connect(":memory:"))
    scheduler = Scheduler(repo)
    notifier = LogNotifier()
    km = KillManager(scheduler, repo, notifier=notifier)

    from sentari.pcb import AgentPCB

    pcb = AgentPCB.new(priority=5, quota_total=3, agent_id="broke")
    pcb.quota_used = 3
    scheduler.admit(pcb)

    await km.terminate_quota_exhausted("broke")

    assert len(notifier.events) == 1
    event = notifier.events[0]
    assert event.event_type == "quota_exhausted"
    assert event.metadata == {"quota_used": 3, "quota_total": 3}


@pytest.mark.asyncio
async def test_kill_manager_does_not_double_notify_an_already_dead_agent():
    repo = AgentRepo(connect(":memory:"))
    scheduler = Scheduler(repo)
    notifier = LogNotifier()
    km = KillManager(scheduler, repo, notifier=notifier)

    from sentari.pcb import AgentPCB

    pcb = AgentPCB.new(priority=5, quota_total=10, agent_id="already_dead")
    scheduler.admit(pcb)
    await km.kill("already_dead", reason="first")
    await km.kill("already_dead", reason="second")  # no-op: already KILLED

    assert len(notifier.events) == 1
    assert notifier.events[0].reason == "first"


@pytest.mark.asyncio
async def test_webhook_notifier_posts_json_payload(monkeypatch):
    captured = {}

    async def fake_post(self, url, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    notifier = WebhookNotifier("https://example.invalid/hook")
    event = NotificationEvent(event_type="agent_killed", agent_id="x", message="killed", reason="deadlock_victim")
    await notifier.notify(event)

    assert captured["url"] == "https://example.invalid/hook"
    assert captured["json"]["event_type"] == "agent_killed"
    assert captured["json"]["agent_id"] == "x"
    assert captured["json"]["reason"] == "deadlock_victim"


@pytest.mark.asyncio
async def test_webhook_notifier_failure_is_contained_not_raised(monkeypatch):
    async def failing_post(self, url, json=None, **kwargs):
        raise httpx.ConnectError("simulated network failure", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", failing_post)

    notifier = WebhookNotifier("https://unreachable.invalid/hook")
    event = NotificationEvent(event_type="agent_killed", agent_id="x", message="killed")
    await notifier.notify(event)  # must not raise


@pytest.mark.asyncio
async def test_composite_notifier_isolates_one_sink_failure_from_others():
    good = LogNotifier()

    class BrokenNotifier:
        async def notify(self, event: NotificationEvent) -> None:
            raise RuntimeError("this sink is broken")

    composite = CompositeNotifier([BrokenNotifier(), good])
    event = NotificationEvent(event_type="agent_killed", agent_id="x", message="killed")
    await composite.notify(event)  # must not raise

    assert good.events == [event]


@pytest.mark.asyncio
async def test_full_kernel_deadlock_produces_a_real_notification():
    """End-to-end: a genuine 2-agent circular wait, resolved by the real
    DeadlockDetector/ResourceManager, must produce exactly one
    'agent_killed' notification for the actual victim -- not a mocked
    kill, the real kernel path."""
    notifier = LogNotifier()
    kernel = Kernel(notifier=notifier)
    high = kernel.admit(priority=1, quota_total=10, agent_id="high")
    low = kernel.admit(priority=10, quota_total=10, agent_id="low")

    await kernel.resources.acquire(high.agent_id, "A")
    await kernel.resources.acquire(low.agent_id, "B")

    import asyncio

    async def high_wants_b() -> None:
        await kernel.scheduler.acquire_turn(high.agent_id)
        await kernel.resources.acquire(high.agent_id, "B")

    async def low_wants_a() -> None:
        await kernel.scheduler.acquire_turn(low.agent_id)
        await kernel.resources.acquire(low.agent_id, "A")

    await asyncio.wait_for(
        asyncio.gather(high_wants_b(), low_wants_a(), return_exceptions=True), timeout=2.0
    )

    assert kernel.scheduler.get("low").state is AgentState.KILLED
    kill_events = [e for e in notifier.events if e.event_type == "agent_killed"]
    assert len(kill_events) == 1
    assert kill_events[0].agent_id == "low"
    assert kill_events[0].reason == "deadlock_victim"
    kernel.close()
