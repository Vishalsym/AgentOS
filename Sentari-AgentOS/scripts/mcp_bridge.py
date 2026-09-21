"""Sentari MCP bridge -- exposes the live Sentari kernel as tools a real MCP
client (Claude Desktop) can call directly from an actual conversation.

Every write_file / read_file / note call you trigger by chatting normally in
Claude Desktop is forwarded here over stdio (MCP), then relayed as a plain
HTTP request to the dashboard's kernel-mediated /api/mcp/tool_call endpoint.
That means it runs through the exact same SyscallLayer.on_syscall code path
as every other syscall in this project -- admission, quota check, resource
mediation (with real deadlock detection if it ever contends with another
agent), and audit logging -- and shows up live on the dashboard while you
watch, because the dashboard process holds the one real Kernel instance.

Requires the dashboard to already be running in another terminal:
    uv run python scripts/dashboard.py

Then register this script with Claude Desktop: Settings > Developer > Edit
Config, and add to claude_desktop_config.json (adjust the repo path):

    {
      "mcpServers": {
        "sentari": {
          "command": "uv",
          "args": [
            "run", "--project", "G:/SentariOS/Sentari-AgentOS",
            "python", "scripts/mcp_bridge.py"
          ]
        }
      }
    }

Restart Claude Desktop, then ask it to write or read a file "using Sentari" --
it will call these tools. Files are sandboxed to <repo>/mcp_workspace/.

Requires the optional 'mcp' dependency group: uv sync --extra mcp
"""

from __future__ import annotations

import os

import httpx
from mcp.server.mcpserver import MCPServer

DASHBOARD_URL = os.environ.get("SENTARI_DASHBOARD_URL", "http://127.0.0.1:8000")
AGENT_ID = os.environ.get("SENTARI_MCP_AGENT_ID", "claude_desktop")

mcp = MCPServer("sentari")


async def _call_kernel(tool: str, **kwargs: str) -> str:
    payload = {"agent_id": AGENT_ID, "tool": tool, **kwargs}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{DASHBOARD_URL}/api/mcp/tool_call", json=payload)
    except httpx.ConnectError:
        return (
            "Sentari kernel is unreachable -- start it first with "
            "`uv run python scripts/dashboard.py`, then try again."
        )
    if resp.status_code != 200:
        return f"Sentari bridge error ({resp.status_code}): {resp.text}"
    data = resp.json()
    if data["result"] == "OK":
        return str(data["value"])
    return f"Sentari kernel returned {data['result']}: {data.get('error') or 'no further detail'}"


@mcp.tool()
async def sentari_write_file(path: str, content: str) -> str:
    """Write a file through the Sentari kernel: quota-checked, resource-locked
    (so a concurrent conflicting write is deadlock-safe), and audit-logged.
    `path` is relative to the sandboxed mcp_workspace/ directory in the repo."""
    return await _call_kernel("write_file", path=path, content=content)


@mcp.tool()
async def sentari_read_file(path: str) -> str:
    """Read a file through the Sentari kernel (mediated + audit-logged).
    `path` is relative to the sandboxed mcp_workspace/ directory."""
    return await _call_kernel("read_file", path=path)


@mcp.tool()
async def sentari_note(prompt: str) -> str:
    """Log a mediated tool_call syscall through the Sentari kernel without
    touching the filesystem -- charges this agent's quota and appends to the
    audit trail, useful for demonstrating enforcement from a live chat."""
    return await _call_kernel("note", prompt=prompt)


@mcp.tool()
async def sentari_status() -> str:
    """Report this agent's live state in the Sentari kernel: PCB state,
    priority, quota used/remaining, and any resources currently held."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{DASHBOARD_URL}/api/agent/{AGENT_ID}")
    except httpx.ConnectError:
        return "Sentari kernel is unreachable -- start it with `uv run python scripts/dashboard.py`."
    if resp.status_code == 404:
        return f"Agent '{AGENT_ID}' has not made any Sentari calls yet in the current dashboard session."
    if resp.status_code != 200:
        return f"Sentari bridge error ({resp.status_code}): {resp.text}"
    a = resp.json()
    held = ", ".join(a["held_resources"]) or "none"
    return (
        f"state={a['state']} priority={a['priority']} "
        f"quota={a['quota_used']}/{a['quota_total']} (remaining {a['quota_remaining']}) "
        f"held_resources=[{held}]"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
