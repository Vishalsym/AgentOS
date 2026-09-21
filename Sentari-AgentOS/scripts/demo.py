"""Live terminal walkthrough of the Sentari Agent-OS kernel.

Runs real scenarios through the actual kernel (no mocking of kernel
behavior -- only the LLM call is a MockProvider) and narrates what's
happening, with a live-updating table during the deadlock scenario.

Usage:
    uv run python scripts/demo.py            # paced for a live audience
    uv run python scripts/demo.py --fast      # no pacing delays (CI/smoke)
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sentari.kernel import Kernel
from sentari.pcb import AgentState
from sentari.syscalls.layer import SyscallLayer, SyscallType

console = Console()

STATE_STYLE = {
    AgentState.RUNNING: "bold green",
    AgentState.READY: "cyan",
    AgentState.BLOCKED: "yellow",
    AgentState.KILLED: "bold red",
    AgentState.TERMINATED: "dim white",
    AgentState.NEW: "white",
}

RESULT_STYLE = {"OK": "green", "DENY": "yellow", "ERROR": "red", "WAIT": "cyan"}


def agents_table(kernel: Kernel, title: str) -> Table:
    table = Table(title=title, title_style="bold")
    table.add_column("agent_id")
    table.add_column("parent")
    table.add_column("state")
    table.add_column("priority")
    table.add_column("quota")
    for pcb in sorted(kernel.agent_repo.list_all(), key=lambda p: p.agent_id):
        table.add_row(
            pcb.agent_id,
            pcb.parent_id or "-",
            Text(pcb.state.value, style=STATE_STYLE.get(pcb.state, "white")),
            str(pcb.priority),
            f"{pcb.quota_used}/{pcb.quota_total}",
        )
    return table


def log_table(kernel: Kernel, limit: int = 15) -> Table:
    table = Table(title="syscall_log -- full audit trail (read from SQLite)", title_style="bold")
    table.add_column("#")
    table.add_column("agent")
    table.add_column("syscall")
    table.add_column("result")
    for row in kernel.syscall_log_repo.list_all()[-limit:]:
        table.add_row(
            str(row["log_id"]),
            row["agent_id"],
            row["syscall_type"],
            Text(row["result"], style=RESULT_STYLE.get(row["result"], "white")),
        )
    return table


async def pause(seconds: float, fast: bool) -> None:
    if not fast:
        await asyncio.sleep(seconds)


async def main(fast: bool) -> None:
    kernel = Kernel()
    console.print(Panel.fit("Sentari Agent-OS -- Live Kernel Demo", style="bold magenta"))

    # 1. Admission -----------------------------------------------------
    console.rule("[bold]1. Admission  (FR-1, FR-2)")
    alpha = kernel.admit(priority=1, quota_total=6, agent_id="alpha")
    beta = kernel.admit(priority=5, quota_total=6, agent_id="beta")
    gamma = kernel.admit(priority=10, quota_total=3, agent_id="gamma")
    console.print(agents_table(kernel, "Agents admitted"))
    await pause(1.0, fast)

    # 2. Ordinary syscall ------------------------------------------------
    console.rule("[bold]2. A normal tool_call  (FR-7, FR-8, FR-9)")
    resp = await kernel.syscall(alpha.agent_id, SyscallType.TOOL_CALL, {"prompt": "plan the next step"})
    console.print(f"alpha's tool_call -> [green]{resp.result.value}[/green]: {resp.value}")
    console.print(agents_table(kernel, "After alpha's syscall"))
    await pause(1.0, fast)

    # 3. Deadlock ----------------------------------------------------------
    console.rule("[bold]3. Deadlock detection & recovery  (FR-12..14, 16, 17)")
    console.print("alpha grabs resource X, beta grabs resource Y ...")
    await kernel.resources.acquire(alpha.agent_id, "X")
    await kernel.resources.acquire(beta.agent_id, "Y")
    console.print(agents_table(kernel, "Resources held"))
    await pause(0.8, fast)

    console.print("Now alpha wants Y (held by beta) and beta wants X (held by alpha), at the same time ...")

    async def alpha_wants_y() -> None:
        await kernel.scheduler.acquire_turn(alpha.agent_id)
        await kernel.resources.acquire(alpha.agent_id, "Y")

    async def beta_wants_x() -> None:
        await kernel.scheduler.acquire_turn(beta.agent_id)
        await kernel.resources.acquire(beta.agent_id, "X")

    with Live(agents_table(kernel, "Resolving deadlock..."), console=console, refresh_per_second=6) as live:
        task_a = asyncio.ensure_future(alpha_wants_y())
        task_b = asyncio.ensure_future(beta_wants_x())
        for _ in range(15):
            await asyncio.sleep(0.01 if fast else 0.15)
            live.update(agents_table(kernel, "Resolving deadlock..."))
            if task_a.done() and task_b.done():
                break
        await asyncio.gather(task_a, task_b, return_exceptions=True)
        live.update(agents_table(kernel, "Deadlock resolved"))

    killed = [p.agent_id for p in kernel.agent_repo.list_all() if p.state is AgentState.KILLED]
    console.print(
        f"[bold red]Kernel detected the cycle and killed the lower-priority agent: "
        f"{killed[0] if killed else '?'}[/bold red] -- the other proceeds normally."
    )
    # We drove the scheduler directly (not through the syscall layer, which
    # would release the turn itself), so the survivor must give up the CPU
    # turn by hand before the next section needs it.
    survivor_id = "beta" if killed == ["alpha"] else "alpha"
    await kernel.scheduler.release_turn(survivor_id)
    await pause(1.0, fast)

    # 4. Preemption on timeout ---------------------------------------------
    console.rule("[bold]4. Preemption on time-slice exceeded  (FR-15)")
    runaway = kernel.admit(priority=3, quota_total=6, agent_id="runaway")
    short_timeout_layer = SyscallLayer(
        scheduler=kernel.scheduler,
        memory_manager=kernel.memory,
        resource_manager=kernel.resources,
        syscall_log_repo=kernel.syscall_log_repo,
        provider=kernel.provider,
        spawn_fn=kernel._spawn_child,
        kill_manager=kernel.kill_manager,
        execution_timeout=0.2,
    )

    async def stuck_tool() -> None:
        await asyncio.sleep(5)

    console.print("runaway calls a tool that never returns in time (0.2s budget) ...")
    from sentari.syscalls.layer import SyscallRequest

    resp = await short_timeout_layer.on_syscall(
        SyscallRequest(agent_id=runaway.agent_id, syscall_type=SyscallType.TOOL_CALL, arguments={"fn": stuck_tool})
    )
    console.print(f"result: [yellow]{resp.result.value}[/yellow] ({resp.error})")
    console.print(agents_table(kernel, "runaway is preempted, not killed -- back to READY"))
    await pause(1.0, fast)

    # 5. Quota exhaustion ----------------------------------------------------
    console.rule("[bold]5. Quota exhaustion  (FR-8, FR-17)")
    console.print(f"gamma has a tiny quota ({gamma.quota_total}); burning through it with repeated yields ...")
    for _ in range(gamma.quota_total + 1):
        resp = await kernel.syscall(gamma.agent_id, SyscallType.YIELD)
        console.print(f"  gamma yield -> {resp.result.value}")
    console.print(agents_table(kernel, "gamma is TERMINATED once its budget runs out"))
    await pause(1.0, fast)

    # 6. Audit trail -----------------------------------------------------
    console.rule("[bold]6. Audit trail  (FR-9, Auditability)")
    console.print(log_table(kernel))
    console.print(agents_table(kernel, "Final agent states -- read straight from SQLite, not memory"))

    console.print(
        Panel.fit(
            "Demo complete. Every state transition above ran through the real kernel and SQLite --\n"
            "the same code paths exercised by the automated test suite (uv run pytest -v).",
            style="bold green",
        )
    )
    kernel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="skip pacing delays (for CI/automated runs)")
    args = parser.parse_args()
    asyncio.run(main(args.fast))
