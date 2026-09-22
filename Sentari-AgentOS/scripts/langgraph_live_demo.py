"""Real cross-framework contention, live, no simulation.

A real LangGraph graph calls a tool that asks the ALREADY-RUNNING dashboard's
kernel to genuinely generate content (via whatever LLMProvider that kernel is
configured with -- the real AnthropicProvider if ANTHROPIC_API_KEY is set)
and write it to a shared file. The resource lock is held for the entire real
generation call, not a fake delay -- see app.py's "generate_and_write" tool,
which only releases the lock after the real `provider.complete()` call
returns.

While this script is running (a few real seconds, if a real key is
configured), open your actual Claude Desktop chat -- the one connected via
scripts/mcp_bridge.py -- and ask it to write to the SAME file. Watch it
genuinely block, live, on the dashboard: a real LangGraph-orchestrated call
and a real external Claude Desktop call, contending on one real resource,
mediated by one real kernel.

Requires: the dashboard already running (uv run python scripts/dashboard.py).

Usage:
    uv run python scripts/langgraph_live_demo.py
    uv run python scripts/langgraph_live_demo.py --path shared_report.md \
        --topic "the history of operating systems"
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from rich.console import Console
from rich.panel import Panel

console = Console()
DASHBOARD_URL = "http://127.0.0.1:8000"
AGENT_ID = "langgraph_writer"


@tool
async def generate_and_write_report(topic: str, path: str) -> str:
    """Ask the shared Sentari kernel to really generate content about
    `topic` and write it to `path` in mcp_workspace/ -- the resource lock
    is held for the whole real generation, so anything else targeting the
    same file genuinely contends with it, not with a placeholder delay."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{DASHBOARD_URL}/api/mcp/tool_call",
            json={
                "agent_id": AGENT_ID,
                "tool": "generate_and_write",
                "path": path,
                "prompt": f"Write a detailed, well-structured 400-600 word explanation of: {topic}",
            },
        )
    data = resp.json()
    if data["result"] != "OK":
        return f"DENIED/ERROR: {data.get('error')}"
    return str(data["value"])


def _graph():
    node = ToolNode([generate_and_write_report])
    g = StateGraph(MessagesState)
    g.add_node("tools", node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile()


async def main(path: str, topic: str) -> None:
    console.print(
        Panel.fit(
            "LangGraph -> real Sentari kernel -> real generation, holding a real resource lock\n"
            f"Target file: mcp_workspace/{path}\n\n"
            "While this is running, ask Claude Desktop (via MCP) to write to the SAME file --\n"
            "watch it genuinely block, live, on the dashboard.",
            style="bold magenta",
        )
    )

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.get(f"{DASHBOARD_URL}/api/state")
    except httpx.ConnectError:
        console.print("[bold red]Dashboard isn't reachable at 127.0.0.1:8000 -- start it first:[/bold red]")
        console.print("  uv run python scripts/dashboard.py")
        return

    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "generate_and_write_report",
                "args": {"topic": topic, "path": path},
                "id": "call1",
                "type": "tool_call",
            }
        ],
    )
    graph = _graph()
    console.print(
        f"[yellow]LangGraph agent is generating and writing '{path}' now -- "
        f"the file is locked for real until this finishes...[/yellow]"
    )
    start = time.perf_counter()
    result = await graph.ainvoke({"messages": [ai]})
    elapsed = time.perf_counter() - start
    console.print(f"[bold green]Done in {elapsed:.1f}s.[/bold green] Result:")
    console.print(result["messages"][-1].content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default="shared_report.md", help="file in mcp_workspace/ to target")
    parser.add_argument("--topic", default="the history of operating systems", help="what to generate about")
    args = parser.parse_args()
    asyncio.run(main(args.path, args.topic))
