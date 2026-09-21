"""FastAPI control dashboard -- the "Optional observability dashboard"
roadmap item, matching the approved Design Doc wireframe (3-panel layout:
Active Agents / Wait-for Graph + Quota Usage / Syscall Log), plus a
cockpit control panel for driving the kernel interactively: admit agents,
fire syscalls, force-kill, and manually acquire/release resources so you
can build your own deadlock live instead of only watching the scripted one.

Run with: uv run python scripts/dashboard.py, then open http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from sentari.pcb import AgentState
from sentari.syscalls.layer import SyscallRequest, SyscallType

from .before_after_scenario import run_before_after_scenario
from .live_scenario import run_live_scenario
from .scenario import run_scenario
from .state import DashboardState

STATIC_DIR = Path(__file__).parent / "static"
RESULT_LEVEL = {"OK": "ok", "DENY": "warn", "ERROR": "danger", "WAIT": "info"}

# Sandbox root for the MCP bridge (scripts/mcp_bridge.py): every write_file/
# read_file call from an external MCP client (e.g. Claude Desktop) is
# confined to this directory so a real chat conversation can never touch
# arbitrary paths on disk through the kernel.
MCP_WORKSPACE_DIR = Path(__file__).resolve().parents[3] / "mcp_workspace"


def _resolve_mcp_path(rel_path: str | None) -> Path:
    if not rel_path or not rel_path.strip():
        raise ValueError("path is required")
    MCP_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    workspace = MCP_WORKSPACE_DIR.resolve()
    candidate = (workspace / rel_path).resolve()
    if candidate == workspace:
        raise ValueError(f"path '{rel_path}' resolves to the workspace root, not a file")
    if workspace not in candidate.parents:
        raise ValueError(f"path '{rel_path}' escapes the sandboxed mcp_workspace directory")
    if candidate.exists() and candidate.is_dir():
        raise ValueError(f"path '{rel_path}' is a directory, not a file")
    return candidate

SCENARIOS = {
    "standard": run_scenario,
    "before_after": run_before_after_scenario,
    "live": run_live_scenario,
}
DEFAULT_SCENARIO = "live"

dashboard_state = DashboardState()
_scenario_task: asyncio.Task[None] | None = None


def _start_scenario(name: str = DEFAULT_SCENARIO) -> None:
    global dashboard_state, _scenario_task
    name = name if name in SCENARIOS else DEFAULT_SCENARIO
    dashboard_state = DashboardState()
    dashboard_state.scenario = name
    _scenario_task = asyncio.create_task(SCENARIOS[name](dashboard_state))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _start_scenario()
    yield


app = FastAPI(title="Sentari Control Dashboard", lifespan=lifespan)


# ---------------------------------------------------------------- pages ---


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/scenarios")
async def list_scenarios() -> JSONResponse:
    return JSONResponse({"scenarios": list(SCENARIOS.keys()), "default": DEFAULT_SCENARIO})


@app.post("/api/restart")
async def restart(scenario: str = DEFAULT_SCENARIO) -> JSONResponse:
    _start_scenario(scenario)
    return JSONResponse({"ok": True, "scenario": dashboard_state.scenario})


# ------------------------------------------------------------- read state -


@app.get("/api/state")
async def get_state() -> JSONResponse:
    state = dashboard_state
    kernel = state.kernel

    async with state.lock:
        narration, phase, done = state.narration, state.phase, state.done
        events = list(state.events)
        cycle_notice = state.cycle_notice
        comparison = state.comparison
        scenario = state.scenario

    if kernel is None:
        return JSONResponse(
            {
                "narration": narration,
                "phase": phase,
                "done": done,
                "scenario": scenario,
                "agents": [],
                "wait_edges": [],
                "resources": [],
                "events": events,
                "cycle_notice": cycle_notice,
                "comparison": comparison,
                "cpu": None,
                "ready_queue": [],
                "stats": {"counts": {}, "total_agents": 0, "total_syscalls": 0, "throughput": 0},
            }
        )

    all_pcbs = sorted(kernel.agent_repo.list_all(), key=lambda p: p.agent_id)
    agents: list[dict[str, Any]] = [
        {
            "agent_id": pcb.agent_id,
            "parent_id": pcb.parent_id,
            "state": pcb.state.value,
            "priority": pcb.priority,
            "quota_used": pcb.quota_used,
            "quota_total": pcb.quota_total,
        }
        for pcb in all_pcbs
    ]

    wait_edges = [{"waiter": w, "holder": h} for w, h in kernel.detector.graph.all_edges()]

    resources = [
        {
            "resource_key": row["resource_key"],
            "holder_agent_id": row["holder_agent_id"],
            "waiter_agent_id": row["waiter_agent_id"],
        }
        for row in kernel.resource_repo.list_all()
    ]

    log_rows = kernel.syscall_log_repo.list_all()
    syscall_events = [
        {
            "time": row["timestamp"],
            "source": row["agent_id"],
            "text": f"{row['syscall_type']} -> {row['result']}",
            "level": RESULT_LEVEL.get(row["result"], "info"),
            "args": row["arguments"],
        }
        for row in log_rows
    ]

    merged_events = sorted(syscall_events + events, key=lambda e: e["time"])[-40:]

    now = time.time()
    counts: dict[str, int] = {}
    for pcb in all_pcbs:
        counts[pcb.state.value] = counts.get(pcb.state.value, 0) + 1
    throughput = sum(1 for row in log_rows if now - row["timestamp"] <= 5.0)

    return JSONResponse(
        {
            "narration": narration,
            "phase": phase,
            "done": done,
            "scenario": scenario,
            "agents": agents,
            "wait_edges": wait_edges,
            "resources": resources,
            "events": merged_events,
            "cycle_notice": cycle_notice,
            "comparison": comparison,
            "cpu": kernel.scheduler.current_agent_id,
            "ready_queue": kernel.scheduler.ready_queue,
            "stats": {
                "counts": counts,
                "total_agents": len(all_pcbs),
                "total_syscalls": len(log_rows),
                "throughput": throughput,
                "quota_used": sum(p.quota_used for p in all_pcbs),
                "quota_total": sum(p.quota_total for p in all_pcbs),
            },
        }
    )


@app.get("/api/agent/{agent_id}")
async def agent_detail(agent_id: str) -> JSONResponse:
    kernel = dashboard_state.kernel
    if kernel is None:
        return JSONResponse({"error": "kernel not ready"}, status_code=503)

    pcb = kernel.agent_repo.get(agent_id)
    if pcb is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    all_alloc = kernel.resource_repo.list_all()
    held = [r["resource_key"] for r in all_alloc if r["holder_agent_id"] == agent_id]
    waiting_on = [r["resource_key"] for r in all_alloc if r["waiter_agent_id"] == agent_id]

    log_rows = kernel.syscall_log_repo.list_for_agent(agent_id)

    return JSONResponse(
        {
            "agent_id": pcb.agent_id,
            "parent_id": pcb.parent_id,
            "state": pcb.state.value,
            "priority": pcb.priority,
            "quota_used": pcb.quota_used,
            "quota_total": pcb.quota_total,
            "quota_remaining": pcb.quota_remaining,
            "created_at": pcb.created_at,
            "updated_at": pcb.updated_at,
            "held_resources": held,
            "waiting_on": waiting_on,
            "syscall_log": [
                {
                    "time": row["timestamp"],
                    "syscall_type": row["syscall_type"],
                    "arguments": row["arguments"],
                    "result": row["result"],
                }
                for row in log_rows
            ],
        }
    )


# --------------------------------------------------------- cockpit actions -


class AdmitRequest(BaseModel):
    agent_id: str | None = None
    priority: int = 5
    quota_total: int = 5


class SyscallRequestBody(BaseModel):
    agent_id: str
    syscall_type: str
    resource_key: str | None = None
    prompt: str | None = None
    key: str | None = None
    value: str | None = None
    scope: str | None = None


class ResourceRequestBody(BaseModel):
    agent_id: str
    resource_key: str


class KillRequestBody(BaseModel):
    agent_id: str
    reason: str | None = "manual"


class McpToolCallBody(BaseModel):
    agent_id: str = "claude_desktop"
    tool: str  # "write_file" | "read_file" | "note"
    path: str | None = None
    content: str | None = None
    prompt: str | None = None
    priority: int = 5
    quota_total: int = 500


def _kernel_or_error() -> tuple[Any, JSONResponse | None]:
    kernel = dashboard_state.kernel
    if kernel is None:
        return None, JSONResponse({"error": "kernel not ready yet"}, status_code=503)
    return kernel, None


@app.post("/api/admit")
async def admit_agent(body: AdmitRequest) -> JSONResponse:
    kernel, err = _kernel_or_error()
    if err:
        return err
    try:
        pcb = kernel.admit(priority=body.priority, quota_total=body.quota_total, agent_id=body.agent_id or None)
    except Exception as exc:  # noqa: BLE001 -- surface as a normal API error, don't crash the server
        return JSONResponse({"error": str(exc)}, status_code=400)
    await dashboard_state.log_event(
        "COCKPIT", f"admitted {pcb.agent_id} (priority={body.priority}, quota={body.quota_total})", "info"
    )
    return JSONResponse({"agent_id": pcb.agent_id})


def _build_arguments(body: SyscallRequestBody) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if body.resource_key:
        args["resource_key"] = body.resource_key
    if body.prompt:
        args["prompt"] = body.prompt
    if body.scope:
        args["scope"] = body.scope
    if body.key is not None:
        args["key"] = body.key
    if body.value is not None:
        args["value"] = body.value
    if body.syscall_type == "tool_call" and "prompt" not in args and "resource_key" not in args:
        args["prompt"] = "manual cockpit trigger"
    return args


@app.post("/api/syscall")
async def trigger_syscall(body: SyscallRequestBody) -> JSONResponse:
    kernel, err = _kernel_or_error()
    if err:
        return err
    try:
        syscall_type = SyscallType(body.syscall_type)
    except ValueError:
        return JSONResponse({"error": f"unknown syscall_type: {body.syscall_type}"}, status_code=400)

    request = SyscallRequest(agent_id=body.agent_id, syscall_type=syscall_type, arguments=_build_arguments(body))
    response = await kernel.syscalls.on_syscall(request)
    return JSONResponse(
        {
            "result": response.result.value,
            "value": str(response.value),
            "error": response.error,
            "truncated": response.truncated,
        }
    )


@app.post("/api/acquire")
async def acquire_resource(body: ResourceRequestBody) -> JSONResponse:
    kernel, err = _kernel_or_error()
    if err:
        return err

    async def _do() -> None:
        killed_before = {p.agent_id for p in kernel.agent_repo.list_all() if p.state is AgentState.KILLED}
        try:
            await kernel.scheduler.acquire_turn(body.agent_id)
            await kernel.resources.acquire(body.agent_id, body.resource_key)
            await dashboard_state.log_event(body.agent_id, f"acquired resource '{body.resource_key}'", "ok")
        except Exception as exc:  # noqa: BLE001 -- report into the cockpit log, don't crash the task
            msg = f"acquire '{body.resource_key}' failed: {exc}"
            await dashboard_state.log_event(body.agent_id, msg, "danger")
        finally:
            # A cycle may have been detected and resolved during this
            # acquire (scripted or manual, doesn't matter) -- surface it in
            # the Wait-For Graph banner regardless of who triggered it.
            killed_after = {p.agent_id for p in kernel.agent_repo.list_all() if p.state is AgentState.KILLED}
            newly_killed = killed_after - killed_before
            if newly_killed:
                victim = next(iter(newly_killed))
                await dashboard_state.set_cycle_notice(
                    {"a": body.agent_id, "b": victim, "victim": victim, "survivor": body.agent_id}
                )
                await dashboard_state.log_event(
                    "KERNEL", f"deadlock cycle resolved -- killed {victim} to break it", "danger"
                )
            try:
                pcb = kernel.scheduler.get(body.agent_id)
                if pcb.state is AgentState.RUNNING:
                    await kernel.scheduler.release_turn(body.agent_id)
            except KeyError:
                pass

    asyncio.create_task(_do())
    await dashboard_state.log_event(body.agent_id, f"requesting resource '{body.resource_key}'...", "info")
    return JSONResponse({"started": True})


@app.post("/api/release")
async def release_resource(body: ResourceRequestBody) -> JSONResponse:
    kernel, err = _kernel_or_error()
    if err:
        return err
    kernel.resources.release(body.agent_id, body.resource_key)
    await dashboard_state.log_event(body.agent_id, f"released resource '{body.resource_key}'", "info")
    return JSONResponse({"ok": True})


@app.post("/api/kill")
async def kill_agent(body: KillRequestBody) -> JSONResponse:
    kernel, err = _kernel_or_error()
    if err:
        return err
    await kernel.kill_manager.kill(body.agent_id, reason=body.reason or "manual")
    await dashboard_state.log_event("COCKPIT", f"force-killed {body.agent_id}", "danger")
    return JSONResponse({"ok": True})


# ------------------------------------------------------- MCP bridge (external clients like Claude Desktop) -


@app.post("/api/mcp/tool_call")
async def mcp_tool_call(body: McpToolCallBody) -> JSONResponse:
    """Entry point for scripts/mcp_bridge.py: a real external MCP client
    (e.g. Claude Desktop) reaches the kernel through this single endpoint.
    It runs through the exact same SyscallLayer.on_syscall used everywhere
    else -- admission/quota check, turn arbitration, resource mediation
    (with deadlock detection if it contends with another agent), timeout,
    and audit logging -- nothing here is a separate code path."""
    kernel, err = _kernel_or_error()
    if err:
        return err

    try:
        kernel.scheduler.get(body.agent_id)
    except KeyError:
        kernel.admit(priority=body.priority, quota_total=body.quota_total, agent_id=body.agent_id)
        await dashboard_state.log_event(
            "MCP", f"admitted external agent '{body.agent_id}' via MCP bridge", "info"
        )

    try:
        if body.tool == "write_file":
            target = _resolve_mcp_path(body.path)
            content = body.content or ""

            async def _fn(_target=target, _content=content) -> str:
                _target.parent.mkdir(parents=True, exist_ok=True)
                _target.write_text(_content, encoding="utf-8")
                return f"wrote {len(_content)} chars to mcp_workspace/{_target.name}"

            arguments = {"resource_key": f"mcp:{target}", "fn": _fn}
        elif body.tool == "read_file":
            target = _resolve_mcp_path(body.path)

            async def _fn(_target=target) -> str:
                return _target.read_text(encoding="utf-8") if _target.exists() else ""

            arguments = {"resource_key": f"mcp:{target}", "fn": _fn}
        elif body.tool == "note":
            arguments = {"prompt": body.prompt or "note"}
        else:
            return JSONResponse({"error": f"unknown tool: {body.tool}"}, status_code=400)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    response = await kernel.syscall(body.agent_id, SyscallType.TOOL_CALL, arguments)
    return JSONResponse({"result": response.result.value, "value": str(response.value), "error": response.error})


@app.get("/api/mcp/files")
async def list_mcp_files() -> JSONResponse:
    """What's actually on disk in the MCP sandbox right now -- lets the
    dashboard show which file(s) an MCP client (the console below, or a real
    Claude Desktop session) is working on, not just an abstract resource_key
    string in the syscall log."""
    if not MCP_WORKSPACE_DIR.exists():
        return JSONResponse({"files": []})
    files = [
        {"path": str(p.relative_to(MCP_WORKSPACE_DIR)).replace("\\", "/"), "size": p.stat().st_size}
        for p in sorted(MCP_WORKSPACE_DIR.rglob("*"))
        if p.is_file()
    ]
    return JSONResponse({"files": files})


@app.get("/api/mcp/files/{file_path:path}")
async def read_mcp_file(file_path: str) -> JSONResponse:
    try:
        target = _resolve_mcp_path(file_path)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not target.exists() or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return JSONResponse({"error": "binary file, cannot preview"}, status_code=415)
    return JSONResponse({"path": file_path, "content": content, "size": target.stat().st_size})


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
