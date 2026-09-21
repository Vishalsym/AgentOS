"""Benchmark suite (README roadmap item): real, measured numbers for the
claims the rest of the project only argues architecturally --

  1. Mediation overhead   -- kernel.syscall() vs calling the same function
                              directly, no kernel involved (NFR-Performance:
                              "<5% syscall overhead").
  2. Fairness              -- N equal-priority agents contending for the
                              single dispatch slot: how evenly are turns
                              actually distributed? (FR-4)
  3. Starvation prevention -- one low-priority agent among several
                              high-priority ones, held to real wall-clock
                              time so priority aging (FR-6) has to do its
                              job, not just pass a unit test in isolation.
  4. Deadlock resolution   -- N trials of the real kernel resolving a
                              2-agent circular wait, vs. the same conflict
                              with no kernel at all (reuses
                              before_after_demo.py's unmanaged/managed
                              functions directly, not a re-implementation).

Every number below comes from actually running the real Kernel/Scheduler/
ResourceManager -- nothing here is simulated or hand-computed.

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --out benchmark_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentari.kernel import Kernel
from sentari.syscalls.layer import SyscallResult, SyscallType

sys.path.insert(0, str(Path(__file__).parent))
from before_after_demo import run_kernel_managed_deadlock, run_unmanaged_deadlock  # noqa: E402

console = Console()


async def noop() -> str:
    return "ok"


# --------------------------------------------------------------- overhead -


async def bench_overhead(n: int) -> dict:
    """Same no-op async function, called N times directly vs N times through
    kernel.syscall(TOOL_CALL). Single agent, no contention -- isolates pure
    mediation cost (admission check, quota accounting, audit log write)
    from any scheduling/queueing effects."""
    start = time.perf_counter()
    for _ in range(n):
        await noop()
    raw_elapsed = time.perf_counter() - start

    kernel = Kernel()
    agent = kernel.admit(priority=1, quota_total=n + 5, agent_id="bench_overhead")

    start = time.perf_counter()
    for _ in range(n):
        resp = await kernel.syscall(agent.agent_id, SyscallType.TOOL_CALL, {"fn": noop})
        assert resp.result is SyscallResult.OK
    mediated_elapsed = time.perf_counter() - start
    kernel.close()

    overhead_pct = ((mediated_elapsed - raw_elapsed) / raw_elapsed) * 100 if raw_elapsed > 0 else float("nan")
    overhead_per_call_us = ((mediated_elapsed - raw_elapsed) / n) * 1e6
    # A bare no-op function call is a pathological baseline -- no real agent
    # syscall is ever that cheap; what it actually replaces is a network
    # call to an LLM/tool API, typically 100s of ms. Reporting overhead only
    # against the no-op makes an intrinsically-larger-than-a-function-call
    # cost (a SQLite write + admission/quota bookkeeping) look catastrophic
    # when it is, in practice, noise next to the call it's mediating.
    realistic_llm_call_us = 300_000  # ~300ms, a typical LLM completion latency
    overhead_pct_of_realistic_call = (overhead_per_call_us / realistic_llm_call_us) * 100
    return {
        "n_calls": n,
        "raw_total_s": raw_elapsed,
        "mediated_total_s": mediated_elapsed,
        "raw_avg_us": (raw_elapsed / n) * 1e6,
        "mediated_avg_us": (mediated_elapsed / n) * 1e6,
        "mediation_overhead_per_call_us": overhead_per_call_us,
        "overhead_pct_of_noop_baseline": overhead_pct,
        "overhead_pct_of_realistic_llm_call": overhead_pct_of_realistic_call,
    }


# --------------------------------------------------------------- fairness -


async def bench_fairness(n_agents: int, calls_per_agent: int) -> dict:
    """N agents, identical priority, all firing YIELD syscalls concurrently
    for the whole run. With no resource contention involved, the only thing
    that can make one agent's average turn-latency worse than another's is
    scheduler unfairness -- so the spread across agents' mean latencies is
    the fairness signal."""
    kernel = Kernel()
    agents = [
        kernel.admit(priority=5, quota_total=calls_per_agent + 2, agent_id=f"fair-{i}")
        for i in range(n_agents)
    ]

    per_agent_latencies: dict[str, list[float]] = {a.agent_id: [] for a in agents}

    async def worker(agent_id: str) -> None:
        for _ in range(calls_per_agent):
            t0 = time.perf_counter()
            resp = await kernel.syscall(agent_id, SyscallType.YIELD)
            per_agent_latencies[agent_id].append(time.perf_counter() - t0)
            assert resp.result is SyscallResult.OK

    await asyncio.gather(*(worker(a.agent_id) for a in agents))

    means = {aid: statistics.mean(lat) for aid, lat in per_agent_latencies.items()}
    overall_mean = statistics.mean(means.values())
    overall_stdev = statistics.pstdev(means.values()) if len(means) > 1 else 0.0
    # Jain's fairness index over each agent's total served turns -- 1.0 is
    # perfectly fair; here every agent completes all its calls (nothing is
    # denied), so this mainly guards against a scheduler bug that silently
    # starves one agent's requests entirely.
    served = [len(lat) for lat in per_agent_latencies.values()]
    jain = (sum(served) ** 2) / (len(served) * sum(x * x for x in served)) if served else 0.0

    kernel.close()
    return {
        "n_agents": n_agents,
        "calls_per_agent": calls_per_agent,
        "per_agent_mean_latency_us": {aid: m * 1e6 for aid, m in means.items()},
        "overall_mean_latency_us": overall_mean * 1e6,
        "overall_latency_stdev_us": overall_stdev * 1e6,
        "coefficient_of_variation_pct": (overall_stdev / overall_mean * 100) if overall_mean else 0.0,
        "jain_fairness_index_on_turns_served": jain,
    }


# --------------------------------------------------------- anti-starvation -


async def bench_starvation_prevention(duration_s: float, low_priority: int, n_high: int = 4) -> dict:
    """One low-priority ("unimportant") agent among several high-priority
    ones, all hammering YIELD as fast as possible for a real wall-clock
    window long enough for FR-6's aging to matter. Reports the low-priority
    agent's share of total turns -- the claim under test is "non-zero and
    growing", not "equal", since it starts out least important by design.

    IMPORTANT, and this is a real finding from running this benchmark, not
    a hypothetical: aging boosts priority by a fixed +1 per
    DEFAULT_AGING_THRESHOLD_SECONDS (2.0s) of continuous waiting. That is a
    *linear* correction, so closing a gap of G priority levels against a
    swarm of always-ready priority-1 agents takes roughly G * 2.0s. For a
    modest gap (this benchmark's default) that's survivable; for an
    extreme one (e.g. priority 50 against priority 1, a ~98s convergence
    time) the agent is, for all practical purposes, starved within any
    demo-length window even though the mechanism is technically working.
    This is a genuine, previously-undocumented limitation of FR-6 as
    currently implemented -- see the benchmark report's note."""
    kernel = Kernel()
    high_agents = [
        kernel.admit(priority=1, quota_total=10_000, agent_id=f"hp-{i}") for i in range(n_high)
    ]
    low_agent = kernel.admit(priority=low_priority, quota_total=10_000, agent_id="starved-candidate")

    counts: dict[str, int] = {a.agent_id: 0 for a in high_agents}
    counts[low_agent.agent_id] = 0
    stop_at = time.perf_counter() + duration_s

    async def hammer(agent_id: str) -> None:
        while time.perf_counter() < stop_at:
            resp = await kernel.syscall(agent_id, SyscallType.YIELD)
            if resp.result is SyscallResult.OK:
                counts[agent_id] += 1
            else:
                break  # quota exhausted (shouldn't happen at 10_000, but be safe)

    await asyncio.gather(*(hammer(a.agent_id) for a in high_agents), hammer(low_agent.agent_id))

    total = sum(counts.values())
    low_share_pct = (counts[low_agent.agent_id] / total * 100) if total else 0.0
    kernel.close()
    return {
        "duration_s": duration_s,
        "low_priority_value": low_priority,
        "estimated_convergence_s": (low_priority - 1) * 2.0,
        "turns_per_agent": counts,
        "total_turns": total,
        "low_priority_agent_turns": counts[low_agent.agent_id],
        "low_priority_agent_share_pct": low_share_pct,
        "starved": counts[low_agent.agent_id] == 0,
    }


# ------------------------------------------------------------- deadlock ---


async def bench_deadlock_resolution(n_trials: int, hang_cap: float) -> dict:
    managed_ms = []
    for _ in range(n_trials):
        elapsed, _victim, _survivor = await run_kernel_managed_deadlock()
        managed_ms.append(elapsed * 1000)

    unmanaged_s = await run_unmanaged_deadlock(hang_cap)

    return {
        "n_trials": n_trials,
        "managed_ms_all_trials": managed_ms,
        "managed_ms_mean": statistics.mean(managed_ms),
        "managed_ms_median": statistics.median(managed_ms),
        "managed_ms_max": max(managed_ms),
        "unmanaged_hang_cap_s": hang_cap,
        "unmanaged_still_stuck_after_s": unmanaged_s,
        "speedup_factor_vs_hang_cap": (hang_cap * 1000) / statistics.mean(managed_ms),
    }


# ------------------------------------------------------------------- main -


def print_report(results: dict) -> None:
    console.print(Panel.fit("Sentari Agent-OS -- Benchmark Suite", style="bold magenta"))

    o = results["overhead"]
    t1 = Table(title="1. Mediation overhead (single agent, no contention)", title_style="bold")
    t1.add_column("metric")
    t1.add_column("value")
    t1.add_row("calls measured", str(o["n_calls"]))
    t1.add_row("raw call (no kernel)", f'{o["raw_avg_us"]:.1f} us/call')
    t1.add_row("mediated (kernel.syscall)", f'{o["mediated_avg_us"]:.1f} us/call')
    t1.add_row("added cost per call", f'{o["mediation_overhead_per_call_us"]:.1f} us')
    t1.add_row("overhead vs. a bare no-op call", f'{o["overhead_pct_of_noop_baseline"]:,.0f}%  (see note)')
    t1.add_row("overhead vs. a realistic ~300ms LLM call", f'{o["overhead_pct_of_realistic_llm_call"]:.3f}%')
    console.print(t1)
    console.print(
        "[dim]  note: the no-op baseline is a pathological comparison -- no real agent syscall is\n"
        "  ever that cheap. The mediation cost is essentially fixed (~1 SQLite write + bookkeeping),\n"
        "  so as a share of what a syscall actually replaces (a real tool/LLM call) it's negligible.[/dim]"
    )

    f = results["fairness"]
    t2 = Table(title="2. Fairness across equal-priority agents", title_style="bold")
    t2.add_column("metric")
    t2.add_column("value")
    t2.add_row("agents", str(f["n_agents"]))
    t2.add_row("calls per agent", str(f["calls_per_agent"]))
    t2.add_row("mean per-call latency", f'{f["overall_mean_latency_us"]:.1f} us')
    t2.add_row("stdev across agents", f'{f["overall_latency_stdev_us"]:.1f} us')
    t2.add_row("coefficient of variation", f'{f["coefficient_of_variation_pct"]:.1f}%')
    t2.add_row("Jain's fairness index (turns served)", f'{f["jain_fairness_index_on_turns_served"]:.4f}')
    console.print(t2)

    starvation_cases = (
        ("modest priority gap", results["starvation_modest"]),
        ("extreme priority gap", results["starvation_extreme"]),
    )
    for label, s in starvation_cases:
        t3 = Table(title=f"3. Starvation prevention -- {label} (aging, FR-6)", title_style="bold")
        t3.add_column("metric")
        t3.add_column("value")
        t3.add_row("window", f'{s["duration_s"]:.1f}s')
        t3.add_row("low agent's priority (1 = most important)", str(s["low_priority_value"]))
        t3.add_row("estimated aging convergence time", f'{s["estimated_convergence_s"]:.0f}s')
        t3.add_row("total turns dispatched", str(s["total_turns"]))
        t3.add_row("low-priority agent's turns", str(s["low_priority_agent_turns"]))
        t3.add_row("low-priority agent's share", f'{s["low_priority_agent_share_pct"]:.2f}%')
        starved_label = "[bold red]YES[/bold red]" if s["starved"] else "[bold green]no[/bold green]"
        t3.add_row("starved (0 turns)?", starved_label)
        console.print(t3)
    console.print(
        "[dim]  finding: aging boosts priority by a fixed +1 per ~2.0s of continuous waiting -- a\n"
        "  linear correction. Under a modest gap it demonstrably prevents starvation within a short\n"
        "  window; under an extreme gap (e.g. 50 vs. 1) convergence takes ~98s, so the agent is, for\n"
        "  any demo-length window, effectively starved even though the mechanism is technically\n"
        "  working. This is a real, previously-undocumented limitation of FR-6, not a bug in this\n"
        "  benchmark -- worth a design note (e.g. a non-linear/adaptive aging boost) if pursued further.[/dim]"
    )

    d = results["deadlock"]
    t4 = Table(title="4. Deadlock resolution: managed vs. unmanaged", title_style="bold")
    t4.add_column("metric")
    t4.add_column("value")
    t4.add_row("trials (managed)", str(d["n_trials"]))
    t4.add_row("managed: mean resolution time", f'{d["managed_ms_mean"]:.2f} ms')
    t4.add_row("managed: median", f'{d["managed_ms_median"]:.2f} ms')
    t4.add_row("managed: worst trial", f'{d["managed_ms_max"]:.2f} ms')
    t4.add_row(
        "unmanaged: still stuck after",
        f'{d["unmanaged_still_stuck_after_s"]:.2f}s (capped; unbounded for real)',
    )
    t4.add_row("speedup vs. the cap alone", f'{d["speedup_factor_vs_hang_cap"]:.0f}x')
    console.print(t4)

    console.print(
        Panel.fit(
            "All four numbers above came from the real Kernel/Scheduler/ResourceManager/\n"
            "DeadlockDetector -- the same code exercised by `uv run pytest -v`, not a mockup.",
            style="bold green",
        )
    )


async def main(out_path: Path | None) -> None:
    results = {
        "overhead": await bench_overhead(n=500),
        "fairness": await bench_fairness(n_agents=6, calls_per_agent=25),
        "starvation_modest": await bench_starvation_prevention(duration_s=5.0, low_priority=5),
        "starvation_extreme": await bench_starvation_prevention(duration_s=2.5, low_priority=50),
        "deadlock": await bench_deadlock_resolution(n_trials=20, hang_cap=1.0),
    }
    print_report(results)
    if out_path:
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        console.print(f"\nFull results written to [bold]{out_path}[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="write full JSON results to this path")
    args = parser.parse_args()
    asyncio.run(main(args.out))
