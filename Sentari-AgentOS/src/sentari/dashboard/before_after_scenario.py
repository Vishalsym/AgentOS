"""The narrated, dashboard-driven twin of scripts/before_after_demo.py: the
same circular resource dependency, run first with raw asyncio (no kernel --
it genuinely hangs) and then through the real kernel (resolved in
milliseconds), narrating both halves into a DashboardState instead of
printing with rich."""

from __future__ import annotations

import asyncio
import time

from sentari.kernel import Kernel
from sentari.pcb import AgentState

from .state import DashboardState, build_default_notifier

STEP_PAUSE = 1.4
HANG_CAP = 4.0


async def run_before_after_scenario(state: DashboardState) -> None:
    # ---- BEFORE: raw asyncio, no kernel at all -----------------------
    state.kernel = None
    await state.narrate(
        "BEFORE: alpha grabs lock X, beta grabs lock Y, then each wants the other's lock -- "
        "with nothing mediating this, using plain asyncio.Lock() (no kernel involved yet).",
        "before-setup",
    )
    await state.log_event(
        "UNMANAGED", "alpha holds X, beta holds Y -- both now want the lock the other is holding", "warn"
    )
    await asyncio.sleep(STEP_PAUSE)

    lock_x = asyncio.Lock()
    lock_y = asyncio.Lock()

    async def worker_alpha() -> None:
        async with lock_x:
            await asyncio.sleep(0.05)
            async with lock_y:  # blocks forever -- beta holds Y and wants X
                pass

    async def worker_beta() -> None:
        async with lock_y:
            await asyncio.sleep(0.05)
            async with lock_x:  # blocks forever -- alpha holds X and wants Y
                pass

    start = time.perf_counter()
    task_a = asyncio.ensure_future(worker_alpha())
    task_b = asyncio.ensure_future(worker_beta())

    while True:
        elapsed = time.perf_counter() - start
        if task_a.done() and task_b.done():
            break
        if elapsed >= HANG_CAP:
            break
        await state.narrate(
            f"BEFORE: still stuck -- {elapsed:.1f}s elapsed. Nothing is watching for this; "
            "raw asyncio has no scheduler, no wait-for graph, no deadlock detector.",
            "before-hang",
        )
        await asyncio.sleep(0.3)

    unmanaged_elapsed = time.perf_counter() - start
    task_a.cancel()
    task_b.cancel()
    await asyncio.gather(task_a, task_b, return_exceptions=True)

    await state.log_event(
        "UNMANAGED",
        f"still deadlocked after {unmanaged_elapsed:.2f}s -- cut off here only so the demo ends; "
        "an unmanaged pipeline has no such cap and hangs forever",
        "danger",
    )
    await state.narrate(
        f"BEFORE: still deadlocked after {unmanaged_elapsed:.2f}s. We forcibly cut it off here only "
        "so this demo ends -- a real unmanaged pipeline has no timeout and hangs forever.",
        "before-result",
    )
    await asyncio.sleep(STEP_PAUSE * 1.3)

    # ---- AFTER: the exact same conflict, mediated by the real kernel --
    await state.narrate(
        "AFTER: the identical conflict, now mediated by the real Sentari kernel.", "after-setup"
    )
    kernel = Kernel(notifier=build_default_notifier())
    state.kernel = kernel
    alpha = kernel.admit(priority=1, quota_total=6, agent_id="alpha")
    beta = kernel.admit(priority=5, quota_total=6, agent_id="beta")
    await asyncio.sleep(STEP_PAUSE * 0.6)

    await kernel.resources.acquire(alpha.agent_id, "X")
    await kernel.resources.acquire(beta.agent_id, "Y")
    await state.narrate(
        "AFTER: alpha holds X, beta holds Y (via kernel.resources this time). "
        "Now each wants the other's resource, at the same time.",
        "after-deadlock",
    )
    await asyncio.sleep(STEP_PAUSE * 0.6)

    async def alpha_wants_y() -> None:
        await kernel.scheduler.acquire_turn(alpha.agent_id)
        await kernel.resources.acquire(alpha.agent_id, "Y")

    async def beta_wants_x() -> None:
        await kernel.scheduler.acquire_turn(beta.agent_id)
        await kernel.resources.acquire(beta.agent_id, "X")

    start2 = time.perf_counter()
    await asyncio.gather(alpha_wants_y(), beta_wants_x(), return_exceptions=True)
    managed_elapsed = time.perf_counter() - start2

    killed = [p.agent_id for p in kernel.agent_repo.list_all() if p.state is AgentState.KILLED]
    victim = killed[0] if killed else "?"
    survivor = "beta" if victim == "alpha" else "alpha"
    await kernel.scheduler.release_turn(survivor)

    await state.set_cycle_notice({"a": "alpha", "b": "beta", "victim": victim, "survivor": survivor})
    await state.log_event(
        "KERNEL",
        f"cycle detected and resolved in {managed_elapsed * 1000:.1f}ms -- killed {victim} "
        f"(lower priority); {survivor} continues",
        "danger",
    )
    await state.set_comparison(
        {
            "hang_cap": HANG_CAP,
            "unmanaged_elapsed": unmanaged_elapsed,
            "managed_elapsed_ms": managed_elapsed * 1000,
            "victim": victim,
            "survivor": survivor,
        }
    )
    await state.narrate(
        f"Resolved in {managed_elapsed * 1000:.1f}ms by the kernel's deadlock detector -- vs. "
        f"{unmanaged_elapsed:.1f}s+ (and unbounded) with no kernel at all.",
        "comparison",
    )
    await asyncio.sleep(STEP_PAUSE)

    await state.finish()
