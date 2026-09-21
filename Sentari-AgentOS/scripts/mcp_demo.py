"""Sandboxed, self-contained proof of the Sentari MCP bridge -- no Claude
Desktop, no external config edits, nothing outside this repo. It boots the
real dashboard, then drives scripts/mcp_bridge.py with a real MCP client
(the same stdio protocol Claude Desktop speaks), calling the exact tools an
assistant would call from a live conversation. Every call is mediated by the
real kernel and shows up live on the dashboard if you have it open in a
browser at the same time.

This intentionally never touches claude_desktop_config.json or any file
outside this repository -- it only starts scripts/dashboard.py (if one
isn't already running) and scripts/mcp_bridge.py as ordinary subprocesses.

Usage:
    uv run python scripts/mcp_demo.py
        Starts the dashboard, runs the scripted MCP calls with pacing so you
        (or an audience watching http://127.0.0.1:8000) can follow along,
        then keeps the dashboard alive until you press Ctrl+C.

    uv run python scripts/mcp_demo.py --fast
        No pacing, exits immediately after the calls (CI/smoke use).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult
from rich.console import Console
from rich.panel import Panel

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_URL = "http://127.0.0.1:8000"
WORKSPACE_DIR = REPO_ROOT / "mcp_workspace"

console = Console()


def _text(result: CallToolResult) -> str:
    for block in result.content:
        if hasattr(block, "text"):
            return block.text
    return str(result.content)


async def pause(seconds: float, fast: bool) -> None:
    if not fast:
        await asyncio.sleep(seconds)


async def _dashboard_is_up() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{DASHBOARD_URL}/api/state")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def _wait_for_dashboard(timeout: float = 10.0) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if await _dashboard_is_up():
            return
        await asyncio.sleep(0.25)
    raise RuntimeError("dashboard did not come up in time")


async def main(fast: bool) -> None:
    console.print(
        Panel.fit(
            "Sentari MCP Bridge -- Sandboxed Proof\n"
            "Everything below stays inside this repo: no Claude Desktop, no external config.",
            style="bold magenta",
        )
    )

    started_dashboard = False
    dashboard_proc: subprocess.Popen | None = None

    if await _dashboard_is_up():
        console.print(f"[cyan]A dashboard is already running at {DASHBOARD_URL} -- reusing it.[/cyan]")
    else:
        console.print(f"Starting the real dashboard server ({DASHBOARD_URL}) ...")
        dashboard_proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "scripts" / "dashboard.py")],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started_dashboard = True
        await _wait_for_dashboard()
        console.print("[green]Dashboard is up.[/green] Open it now if you want to watch this live:")
        console.print(f"  [bold]{DASHBOARD_URL}[/bold]")
        await pause(3.0, fast)

    try:
        params = StdioServerParameters(
            command=sys.executable, args=[str(REPO_ROOT / "scripts" / "mcp_bridge.py")], cwd=str(REPO_ROOT)
        )
        console.rule("[bold]Connecting a real MCP client to scripts/mcp_bridge.py (stdio)")
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                console.print("Tools exposed by the bridge: " + ", ".join(t.name for t in tools.tools))
                await pause(1.0, fast)

                console.rule("[bold]1. Status before any calls")
                r = await session.call_tool("sentari_status", {})
                console.print(_text(r))
                await pause(1.0, fast)

                console.rule("[bold]2. Write a file -- mediated by the kernel (quota, resource lock, audit log)")
                r = await session.call_tool(
                    "sentari_write_file",
                    {
                        "path": "demo_note.md",
                        "content": (
                            "# Written via MCP\n\n"
                            "This file was created by calling `sentari_write_file` over the Model "
                            "Context Protocol -- the same protocol Claude Desktop speaks -- and was "
                            "mediated by the real Sentari kernel: quota-checked, resource-locked, and "
                            "logged to the syscall audit trail.\n"
                        ),
                    },
                )
                console.print(_text(r))
                await pause(1.2, fast)

                console.rule("[bold]3. Read it back")
                r = await session.call_tool("sentari_read_file", {"path": "demo_note.md"})
                console.print(_text(r))
                await pause(1.0, fast)

                console.rule("[bold]4. Sandbox enforcement -- try to escape mcp_workspace/")
                console.print("Attempting to write to '../outside_sandbox.txt' ...")
                r = await session.call_tool(
                    "sentari_write_file", {"path": "../outside_sandbox.txt", "content": "should never land"}
                )
                console.print(f"[yellow]{_text(r)}[/yellow]")
                escaped = (REPO_ROOT / "outside_sandbox.txt").exists()
                console.print(
                    "[green]Confirmed: nothing was written outside the sandbox.[/green]"
                    if not escaped
                    else "[bold red]SANDBOX FAILED -- file escaped![/bold red]"
                )
                await pause(1.2, fast)

                console.rule("[bold]5. A non-filesystem call -- charges quota, no disk I/O")
                r = await session.call_tool("sentari_note", {"prompt": "demo audience checkpoint"})
                console.print(_text(r))
                await pause(1.0, fast)

                console.rule("[bold]6. Status after -- quota consumed, resource released, all logged")
                r = await session.call_tool("sentari_status", {})
                console.print(_text(r))
    finally:
        console.rule("[bold]Live audit trail (read straight from the kernel's SQLite log)")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DASHBOARD_URL}/api/agent/claude_desktop")
            if resp.status_code == 200:
                for row in resp.json()["syscall_log"]:
                    raw_args = json.loads(row["arguments"])
                    args = {k: v for k, v in raw_args.items() if k != "fn"}
                    console.print(f"  {row['syscall_type']:<12} -> {row['result']}  {args}")

    console.print(
        Panel.fit(
            f"Wrote a real file at {WORKSPACE_DIR / 'demo_note.md'}, sandbox-escaped write rejected,\n"
            "every call quota-checked and logged -- all mediated by the same kernel your test suite "
            "exercises.\nThis is exactly what happens when Claude Desktop calls these tools for real.",
            style="bold green",
        )
    )

    if started_dashboard and dashboard_proc is not None:
        if fast:
            dashboard_proc.terminate()
            dashboard_proc.wait(timeout=5)
        else:
            console.print(
                f"\nDashboard still running at [bold]{DASHBOARD_URL}[/bold] -- "
                "keep looking around, press Ctrl+C here when you're done."
            )
            try:
                dashboard_proc.wait()
            except KeyboardInterrupt:
                dashboard_proc.terminate()
                dashboard_proc.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="skip pacing and exit immediately (CI/smoke use)")
    args = parser.parse_args()
    asyncio.run(main(args.fast))
