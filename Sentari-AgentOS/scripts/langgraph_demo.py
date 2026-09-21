"""Live terminal walkthrough of the Sentari <-> LangGraph adapter.

Builds a real, compiled LangGraph `StateGraph` with a real `ToolNode`, and
shows the same tool call running two ways: once completely unmediated (as
LangGraph would run it on its own), and once routed through the real
Sentari kernel via `awrap_tool_call` -- quota-checked and audit-logged --
finishing with a quota-exhaustion denial to prove the mediation is real,
not cosmetic.

Usage:
    uv run python scripts/langgraph_demo.py
    uv run python scripts/langgraph_demo.py --fast

Requires the optional 'langgraph' dependency group: uv sync --extra langgraph
"""

from __future__ import annotations

import argparse
import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentari.adapters.langgraph_adapter import make_sentari_tool_wrapper
from sentari.kernel import Kernel

console = Console()


@tool
def lookup_price(ticker: str) -> str:
    """Look up a (fake) stock price for a ticker symbol."""
    fake_prices = {"ACME": "142.50", "SENT": "9001.00"}
    return f"{ticker}: ${fake_prices.get(ticker.upper(), '??')}"


def _graph(tool_node: ToolNode):
    g = StateGraph(MessagesState)
    g.add_node("tools", tool_node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile()


def _call(ticker: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "lookup_price", "args": {"ticker": ticker}, "id": call_id, "type": "tool_call"}],
    )


def log_table(kernel: Kernel) -> Table:
    table = Table(title="syscall_log (real Sentari kernel, read from SQLite)", title_style="bold")
    table.add_column("#")
    table.add_column("agent")
    table.add_column("syscall")
    table.add_column("result")
    for row in kernel.syscall_log_repo.list_all():
        style = {"OK": "green", "DENY": "yellow", "ERROR": "red"}.get(row["result"], "white")
        result_cell = f"[{style}]{row['result']}[/{style}]"
        table.add_row(str(row["log_id"]), row["agent_id"], row["syscall_type"], result_cell)
    return table


async def pause(seconds: float, fast: bool) -> None:
    if not fast:
        await asyncio.sleep(seconds)


async def main(fast: bool) -> None:
    console.print(Panel.fit("Sentari <-> LangGraph Adapter -- Live Walkthrough", style="bold magenta"))

    # 1. Baseline -- LangGraph on its own, no Sentari involved -------------
    console.rule("[bold]1. Baseline: LangGraph running completely unmediated")
    plain_node = ToolNode([lookup_price])
    plain_graph = _graph(plain_node)
    result = await plain_graph.ainvoke({"messages": [_call("ACME", "c1")]})
    console.print(f"tool result: [cyan]{result['messages'][-1].content}[/cyan]")
    console.print(
        "Nothing governed that call -- no quota, no audit trail, no way to know it happened\n"
        "except by reading the graph's own message log."
    )
    await pause(1.2, fast)

    # 2. The same graph, now mediated by the real Sentari kernel -----------
    console.rule("[bold]2. The identical tool, now mediated by the real Sentari kernel")
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=2, agent_id="langgraph_trader")
    mediated_node = ToolNode(
        [lookup_price],
        awrap_tool_call=make_sentari_tool_wrapper(kernel, agent_id="langgraph_trader"),
    )
    mediated_graph = _graph(mediated_node)

    for i, ticker in enumerate(("ACME", "SENT"), start=1):
        result = await mediated_graph.ainvoke({"messages": [_call(ticker, f"call{i}")]})
        console.print(f"call {i} ({ticker}): [cyan]{result['messages'][-1].content}[/cyan]")
        await pause(0.5, fast)

    pcb = kernel.scheduler.get("langgraph_trader")
    console.print(f"agent quota after 2 calls: {pcb.quota_used}/{pcb.quota_total} -- state={pcb.state.value}")
    await pause(1.0, fast)

    # 3. Quota exhaustion, still inside the graph ---------------------------
    console.rule("[bold]3. A third call exceeds quota -- Sentari denies it, LangGraph stays intact")
    result = await mediated_graph.ainvoke({"messages": [_call("ACME", "call3")]})
    denial = result["messages"][-1]
    console.print(f"call 3 status: [bold red]{denial.status}[/bold red]")
    console.print(f"message content: {denial.content}")
    console.print(
        "The graph did not crash -- it received a normal LangGraph ToolMessage,\n"
        "exactly like any other tool failure, and can react to it on the next turn."
    )
    await pause(1.0, fast)

    # 4. Audit trail ----------------------------------------------------------
    console.rule("[bold]4. Full audit trail, straight from SQLite")
    console.print(log_table(kernel))

    console.print(
        Panel.fit(
            "Every mediated call above went through the exact same SyscallLayer.on_syscall\n"
            "path as the dashboard and MCP bridge -- LangGraph never touched the tool directly.",
            style="bold green",
        )
    )
    kernel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="skip narration pacing delays (for CI/automated runs)")
    args = parser.parse_args()
    asyncio.run(main(args.fast))
