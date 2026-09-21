"""Before/After: the same circular resource dependency, with and without
Sentari mediating it.

"Before" spins up two coroutines that grab two shared locks in opposite
order using nothing but raw asyncio primitives -- the exact pattern that
happens when two agents in an unmanaged pipeline (LangGraph, CrewAI, or
hand-rolled orchestration) each hold one resource and want the other. There
is no scheduler and no deadlock detector here, so it genuinely deadlocks. We
cap the wait at a fixed safety timeout purely so this demo terminates -- an
real unmanaged system has no such cap; the process just hangs until a human
notices and kills it.

"After" runs the identical intent -- two agents wanting two resources in
conflicting order -- through the real Sentari kernel (the same
Kernel/Scheduler/ResourceManager/DeadlockDetector exercised by the test
suite, not a staged re-enactment). The kernel detects the wait-for cycle the
instant it forms and kills the lower-priority agent to break it,
automatically, with a full audit trail.

Usage:
    uv run python scripts/before_after_demo.py
    uv run python scripts/before_after_demo.py --fast           # skip narration pacing
    uv run python scripts/before_after_demo.py --hang-cap 3     # shorter forced timeout
"""

from __future__ import annotations

import argparse
import asyncio
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sentari.kernel import Kernel
from sentari.pcb import AgentState

console = Console()

DEFAULT_HANG_CAP = 4.0


async def pause(seconds: float, fast: bool) -> None:
    if not fast:
        await asyncio.sleep(seconds)


# --------------------------------------------------------------- "before" -


async def run_unmanaged_deadlock(hang_cap: float) -> float:
    """Two coroutines, two plain asyncio.Lock()s, zero mediation. alpha
    grabs X then wants Y; beta grabs Y then wants X, at the same time. This
    is a textbook AB-BA lock-ordering deadlock -- it does not resolve on its
    own, ever. We only escape via the caller's asyncio.wait_for timeout,
    which stands in for "a human eventually kills the hung process"."""
    lock_x = asyncio.Lock()
    lock_y = asyncio.Lock()

    async def worker_alpha() -> None:
        async with lock_x:
            await asyncio.sleep(0.05)
            async with lock_y:  # blocks forever: beta is holding Y and wants X
                pass

    async def worker_beta() -> None:
        async with lock_y:
            await asyncio.sleep(0.05)
            async with lock_x:  # blocks forever: alpha is holding X and wants Y
                pass

    start = time.perf_counter()
    task_a = asyncio.ensure_future(worker_alpha())
    task_b = asyncio.ensure_future(worker_beta())

    with console.status("[bold yellow]alpha and beta are stuck waiting on each other...", spinner="dots"):
        try:
            await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=hang_cap)
        except TimeoutError:
            pass

    elapsed = time.perf_counter() - start
    for t in (task_a, task_b):
        t.cancel()
    await asyncio.gather(task_a, task_b, return_exceptions=True)
    return elapsed


# ---------------------------------------------------------------- "after" -


async def run_kernel_managed_deadlock() -> tuple[float, str, str]:
    """The same conflicting-resource-order intent, mediated by the real
    Sentari kernel instead of raw locks."""
    kernel = Kernel()
    alpha = kernel.admit(priority=1, quota_total=6, agent_id="alpha")
    beta = kernel.admit(priority=5, quota_total=6, agent_id="beta")

    await kernel.resources.acquire(alpha.agent_id, "X")
    await kernel.resources.acquire(beta.agent_id, "Y")

    async def alpha_wants_y() -> None:
        await kernel.scheduler.acquire_turn(alpha.agent_id)
        await kernel.resources.acquire(alpha.agent_id, "Y")

    async def beta_wants_x() -> None:
        await kernel.scheduler.acquire_turn(beta.agent_id)
        await kernel.resources.acquire(beta.agent_id, "X")

    start = time.perf_counter()
    await asyncio.gather(alpha_wants_y(), beta_wants_x(), return_exceptions=True)
    elapsed = time.perf_counter() - start

    killed = [p.agent_id for p in kernel.agent_repo.list_all() if p.state is AgentState.KILLED]
    victim = killed[0] if killed else "?"
    survivor = "beta" if victim == "alpha" else "alpha"
    await kernel.scheduler.release_turn(survivor)
    kernel.close()
    return elapsed, victim, survivor


# ------------------------------------------------------------------- main -


def comparison_table(
    hang_cap: float, unmanaged_elapsed: float, managed_elapsed: float, victim: str, survivor: str
) -> Table:
    table = Table(title="Same deadlock, mediated vs. unmediated", title_style="bold")
    table.add_column("Scenario")
    table.add_column("Outcome")
    table.add_column("Time to resolution")
    table.add_column("Who intervened")

    table.add_row(
        "Raw asyncio (no kernel)",
        Text("HUNG -- forcibly cut off by this demo", style="bold red"),
        f"still stuck after {unmanaged_elapsed:.2f}s (capped at {hang_cap:.0f}s; unbounded in production)",
        "nobody -- needs a human to notice and kill the process",
    )
    table.add_row(
        "Sentari-managed",
        Text(f"RESOLVED -- {victim} killed, {survivor} continued", style="bold green"),
        f"{managed_elapsed * 1000:.1f}ms",
        "kernel's incremental wait-for-graph cycle detector, automatically",
    )
    return table


async def main(fast: bool, hang_cap: float) -> None:
    console.print(
        Panel.fit("Sentari Agent-OS -- Before / After: a circular resource dependency", style="bold magenta")
    )

    console.rule("[bold]1. BEFORE -- unmanaged (raw asyncio, no kernel)")
    console.print(
        "alpha grabs lock X, beta grabs lock Y; then alpha wants Y (held by beta) while beta wants X\n"
        "(held by alpha) -- at the same time. Nothing is watching for this. It does not resolve on its own."
    )
    await pause(0.8, fast)
    unmanaged_elapsed = await run_unmanaged_deadlock(hang_cap)
    console.print(
        f"[bold red]Still deadlocked after {unmanaged_elapsed:.2f}s.[/bold red] "
        f"We cut it off here only so the demo ends -- a real unmanaged pipeline has no timeout "
        f"and would hang forever, or until whatever process supervisor eventually kills it."
    )
    await pause(1.2, fast)

    console.rule("[bold]2. AFTER -- the exact same conflict, mediated by the Sentari kernel")
    console.print(
        "Same setup: alpha holds X and wants Y, beta holds Y and wants X, at the same time -- but now\n"
        "every acquire() goes through kernel.resources, which maintains a live wait-for graph and runs\n"
        "cycle detection on every new edge (FR-12, FR-13)."
    )
    await pause(0.8, fast)
    managed_elapsed, victim, survivor = await run_kernel_managed_deadlock()
    console.print(
        f"[bold green]Resolved in {managed_elapsed * 1000:.1f}ms.[/bold green] "
        f"The kernel detected the cycle the instant the second wait-edge formed, selected the "
        f"lower-priority agent ([bold]{victim}[/bold]) as the victim, force-killed it and released "
        f"everything it held (FR-14, FR-16, FR-17); [bold]{survivor}[/bold] continued uninterrupted."
    )
    await pause(1.2, fast)

    console.rule("[bold]3. Side by side")
    console.print(comparison_table(hang_cap, unmanaged_elapsed, managed_elapsed, victim, survivor))

    console.print(
        Panel.fit(
            "Same failure mode. Unmanaged: silent, permanent hang.\n"
            "Sentari-managed: detected, resolved, and logged automatically in milliseconds.",
            style="bold green",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="skip narration pacing delays (for CI/automated runs)")
    parser.add_argument(
        "--hang-cap",
        type=float,
        default=DEFAULT_HANG_CAP,
        help=f"seconds to let the unmanaged scenario hang before cutting it off (default {DEFAULT_HANG_CAP})",
    )
    args = parser.parse_args()
    asyncio.run(main(args.fast, args.hang_cap))
