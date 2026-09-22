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
import json
import re
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
            # task_value lives only on the live in-memory PCB (like
            # agent_type/declared_resource_claims, it isn't persisted to
            # SQLite) -- read it from the scheduler, not agent_repo.
            "task_value": _task_value_or_none(kernel, pcb.agent_id),
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
            "task_value": _task_value_or_none(kernel, pcb.agent_id),
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
    task_description: str | None = None


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


class TeamStartBody(BaseModel):
    project: str


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


def _task_value_or_none(kernel: Any, agent_id: str) -> float | None:
    try:
        return kernel.scheduler.get(agent_id).task_value
    except KeyError:
        return None


@app.post("/api/admit")
async def admit_agent(body: AdmitRequest) -> JSONResponse:
    kernel, err = _kernel_or_error()
    if err:
        return err

    task_value: float | None = None
    if body.task_description:
        # Real LLM judgment (novel mechanism #1: semantic-value-aware
        # deadlock resolution) -- scored once, here, at admission time,
        # never during deadlock detection itself (that path stays
        # synchronous and sub-millisecond; see deadlock/semantic_scoring.py).
        raw_holder: list[str] = []
        try:
            from sentari.deadlock.semantic_scoring import score_task_value

            task_value = await score_task_value(
                kernel.provider, body.task_description, on_raw_response=raw_holder.append
            )
            raw = raw_holder[0] if raw_holder else ""
            await dashboard_state.log_event(
                "COCKPIT",
                f"scored '{body.task_description[:60]}' -> task_value={task_value:.2f} "
                f"(raw response: {raw[:200]!r})",
                "info",
            )
        except Exception as exc:  # noqa: BLE001 -- a scoring failure must not block admission
            raw = raw_holder[0] if raw_holder else "<no response captured>"
            await dashboard_state.log_event(
                "COCKPIT", f"task-value scoring failed: {exc} (raw response: {raw[:200]!r})", "warn"
            )

    try:
        pcb = kernel.admit(
            priority=body.priority,
            quota_total=body.quota_total,
            agent_id=body.agent_id or None,
            task_value=task_value,
        )
    except Exception as exc:  # noqa: BLE001 -- surface as a normal API error, don't crash the server
        return JSONResponse({"error": str(exc)}, status_code=400)
    await dashboard_state.log_event(
        "COCKPIT", f"admitted {pcb.agent_id} (priority={body.priority}, quota={body.quota_total})", "info"
    )
    return JSONResponse({"agent_id": pcb.agent_id, "task_value": task_value})


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


@app.post("/api/interrupt")
async def interrupt_agent(body: KillRequestBody) -> JSONResponse:
    """Unlike /api/kill (which only updates PCB state and releases
    resources), this cancels the agent's actual in-flight asyncio Task --
    if it's mid-syscall on a real, slow network call (e.g. a real
    Anthropic completion), that real call is genuinely aborted right now,
    not left running in the background until it naturally finishes. See
    SyscallLayer.interrupt / Kernel.interrupt (novel mechanism #4,
    bounded-latency human interrupt)."""
    kernel, err = _kernel_or_error()
    if err:
        return err
    had_inflight = await kernel.interrupt(body.agent_id, reason=body.reason or "manual_interrupt")
    await dashboard_state.log_event(
        "COCKPIT",
        f"interrupted {body.agent_id}"
        + (" (cancelled a real in-flight call)" if had_inflight else " (was idle, killed anyway)"),
        "danger",
    )
    return JSONResponse({"ok": True, "had_inflight_call": had_inflight})


# ---------------------------------------------------- 3-agent real team demo -


async def _mediated_call(kernel: Any, agent_id: str, resource_key: str, fn: Any) -> Any:
    """One resource-mediated TOOL_CALL syscall -- the single building block
    every stage below uses (gating on a teammate, reading a teammate's
    real output, and generating+writing your own all go through this exact
    same, already-tested kernel.syscall path -- no raw scheduler/resource-
    manager calls in this orchestration, so there's no risk of the turn-
    management self-conflicts a hand-rolled acquire/release sequence could
    introduce)."""
    response = await kernel.syscall(agent_id, SyscallType.TOOL_CALL, {"resource_key": resource_key, "fn": fn})
    if response.result.value != "OK":
        raise RuntimeError(f"{agent_id}'s call on '{resource_key}' failed: {response.error}")
    return response.value


async def _real_write(
    kernel: Any, agent_id: str, filename: str, prompt: str, resource_key: str | None = None
) -> str:
    """A real provider.complete() call, holding `resource_key` for the
    entire real generation, then a real write to mcp_workspace/filename.
    If `resource_key` isn't given, the file's own path is used (matching
    the MCP bridge's generate_and_write tool); passing an explicit
    resource_key (e.g. "stage:design") is what lets a *different* agent
    gate on this stage finishing without needing to know the filename."""
    target = _resolve_mcp_path(filename)
    key = resource_key or f"mcp:{target}"

    async def _fn() -> str:
        content = await kernel.provider.complete(prompt)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return content

    return await _mediated_call(kernel, agent_id, key, _fn)


async def _real_read(kernel: Any, agent_id: str, filename: str) -> str:
    target = _resolve_mcp_path(filename)

    async def _fn() -> str:
        return target.read_text(encoding="utf-8") if target.exists() else ""

    return await _mediated_call(kernel, agent_id, f"mcp:{target}", _fn)


async def _gate(kernel: Any, agent_id: str, gate_key: str) -> None:
    """Block (genuinely -- real wait-for-graph edge, real BLOCKED state)
    until whoever currently holds `gate_key` releases it, then continue.
    A trivial no-op call under the hood; its only purpose is the
    acquire/release semantics."""

    async def _fn() -> None:
        return None

    await _mediated_call(kernel, agent_id, gate_key, _fn)


MAX_TEAM_SIZE = 5

_TEAM_PLAN_PROMPT = (
    "You are decomposing a software project into a small real team of AI agents "
    "that will each do REAL work (their own LLM generation, written to their own "
    "file). Project: {project}\n\n"
    f"Decide how many agents this genuinely needs -- between 1 and {MAX_TEAM_SIZE}. "
    "Do not default to any fixed number or fixed role names; pick whatever roles "
    "actually fit this specific project. For each agent give a short name "
    "(lowercase, letters/digits/underscores only, no spaces, e.g. 'ui_designer' "
    "or 'api_dev'), a one-sentence task, and a list of the OTHER agents' names "
    "it must wait for and read the real output of before it can start (empty "
    "list if it can start immediately). Dependencies must not be circular.\n\n"
    "Respond with ONLY a JSON array between the markers <PLAN> and </PLAN>, "
    "nothing else outside those markers, in exactly this shape:\n"
    "<PLAN>\n"
    '[{{"name": "ui_designer", "task": "design the screens and user flow", "depends_on": []}}, '
    '{{"name": "backend_dev", "task": "design the API and data model", "depends_on": []}}, '
    '{{"name": "reviewer", "task": "check the API and screens are consistent with each other", '
    '"depends_on": ["ui_designer", "backend_dev"]}}]\n'
    "</PLAN>"
)

_PLAN_BLOCK_RE = re.compile(r"<PLAN>(.*?)</PLAN>", re.DOTALL | re.IGNORECASE)
_NAME_SANITIZE_RE = re.compile(r"[^a-z0-9_]+")


def _sanitize_agent_name(raw: str, fallback_index: int) -> str:
    name = _NAME_SANITIZE_RE.sub("_", raw.strip().lower()).strip("_")
    return name or f"agent_{fallback_index}"


async def _plan_team(kernel: Any, project: str) -> list[dict[str, Any]]:
    """One real LLM call that decides the team's own shape -- how many
    agents, what each is called, and who waits on whom -- instead of this
    file hardcoding role names/count. The model's raw JSON is validated and
    sanitized (names made filesystem/resource-key-safe, unknown/self
    dependency references dropped, size clamped to MAX_TEAM_SIZE) but never
    silently padded or renamed beyond that -- if the model asks for 2
    agents, you get 2. A genuinely circular dependency isn't detected here;
    it's left to the kernel's own real DeadlockDetector to catch and break
    live, the same as any other resource cycle."""
    raw = await kernel.provider.complete(_TEAM_PLAN_PROMPT.format(project=project))
    match = _PLAN_BLOCK_RE.search(raw)
    if not match:
        raise ValueError(f"planner did not return a <PLAN>...</PLAN> block: {raw!r}")
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"planner's PLAN block was not valid JSON: {match.group(1)!r}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"planner's PLAN block was not a non-empty JSON array: {parsed!r}")

    parsed = parsed[:MAX_TEAM_SIZE]
    seen: set[str] = set()
    plan: list[dict[str, Any]] = []
    for i, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            continue
        name = _sanitize_agent_name(str(entry.get("name", "")), i)
        while name in seen:
            name = f"{name}_{i}"
        seen.add(name)
        task = str(entry.get("task") or "contribute to the project").strip()
        deps_raw = entry.get("depends_on") or []
        deps = [_sanitize_agent_name(str(d), 0) for d in deps_raw if isinstance(d, str)]
        plan.append({"name": name, "task": task, "depends_on": deps})

    valid_names = {a["name"] for a in plan}
    for agent in plan:
        agent["depends_on"] = [d for d in agent["depends_on"] if d in valid_names and d != agent["name"]]

    if not plan:
        raise ValueError(f"planner produced no usable agents from: {raw!r}")
    return plan


async def _run_team(kernel: Any, project: str, plan: list[dict[str, Any]]) -> None:
    """Every agent in `plan` (an LLM-decided list, not hardcoded roles/
    count) is admitted, then all of them race concurrently via
    asyncio.gather. The only thing enforcing "X waits for Y" is the
    kernel's own resource lock on `stage:{Y}` -- gate on each dependency's
    stage key, then really read that dependency's own output file, then
    generate and write your own. Remove the gates and every agent would
    just race independently; the ordering you see live is the kernel doing
    real work, not narration.

    Each agent's own `stage:{name}` key is claimed for it up front, in a
    strictly sequential loop, BEFORE any concurrent work starts. This
    matters: without it, a downstream agent's gate-acquire on
    `stage:{producer}` and the producer's own first acquire of that same
    key are two independent coroutines racing on a freely-available lock --
    whichever happens to reach ResourceManager.acquire first wins it, so a
    consumer scheduled slightly ahead of a still-blocked producer could
    grab its own dependency's key before the producer ever does, sail
    through with empty context, and defeat the whole gate. Pre-claiming
    sequentially removes that race entirely: by the time asyncio.gather
    starts the real concurrent generation, every stage key is already held
    by its rightful owner, so every gate-acquire downstream is a genuine,
    deterministic wait -- not a coin flip."""
    for agent in plan:
        kernel.admit(priority=5, quota_total=2000, agent_id=agent["name"], agent_type="team_member")
    for agent in plan:
        await kernel.resources.acquire(agent["name"], f"stage:{agent['name']}")
    summary = "; ".join(
        f"{a['name']} (waits on: {', '.join(a['depends_on']) or 'none'})" for a in plan
    )
    await dashboard_state.log_event("TEAM", f"team plan for '{project}': {summary}", "info")

    async def run_stage(agent: dict[str, Any]) -> None:
        name = agent["name"]
        try:
            for dep in agent["depends_on"]:
                await _gate(kernel, name, f"stage:{dep}")
            context_parts = []
            for dep in agent["depends_on"]:
                text = await _real_read(kernel, name, f"team_{dep}.md")
                if text:
                    context_parts.append(f"=== {dep}'s real output ===\n{text}")
            context = "\n\n".join(context_parts)
            prompt = (
                f"You are the '{name}' agent on a real small software team. Project: {project}\n"
                f"Your task: {agent['task']}\n\n"
                + (f"Your teammates' real output so far:\n\n{context}\n\n" if context else "")
                + "If your task is to build/design/implement something, write REAL, working "
                "code for it (with filenames/paths as comments and enough of the surrounding "
                "file -- imports, function signatures, etc. -- that it's a genuine artifact, "
                "not pseudocode or a description of what you would write). If your task is "
                "inherently a review/verification, write concrete findings that reference "
                "specific code or decisions from your teammates' real output above, not "
                "generic advice. Do not just describe your contribution in prose -- produce it."
            )
            await _real_write(kernel, name, f"team_{name}.md", prompt)
        finally:
            kernel.resources.release(name, f"stage:{name}")

    results = await asyncio.gather(*(run_stage(agent) for agent in plan), return_exceptions=True)
    for agent, result in zip(plan, results, strict=True):
        if isinstance(result, Exception):
            await dashboard_state.log_event("TEAM", f"{agent['name']}'s stage failed: {result}", "danger")
    files = ", ".join(f"team_{a['name']}.md" for a in plan)
    await dashboard_state.log_event(
        "TEAM", f"team run complete -- see {files} in the MCP workspace files list", "ok"
    )


@app.post("/api/team/start")
async def start_team(body: TeamStartBody) -> JSONResponse:
    kernel, err = _kernel_or_error()
    if err:
        return err
    project = (body.project or "").strip()
    if not project:
        return JSONResponse({"error": "project description is required"}, status_code=400)

    if getattr(kernel, "_team_started", False):
        return JSONResponse(
            {"error": "A team has already run in this session -- restart the kernel first (↻ Restart kernel)."},
            status_code=400,
        )

    await dashboard_state.log_event("TEAM", f"planning a team for: {project}", "info")
    try:
        plan = await _plan_team(kernel, project)
    except ValueError as exc:
        await dashboard_state.log_event("TEAM", f"planning failed: {exc}", "danger")
        return JSONResponse({"error": f"team planning failed: {exc}"}, status_code=502)

    kernel._team_started = True
    asyncio.create_task(_run_team(kernel, project, plan))
    return JSONResponse({"started": True, "plan": plan})


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
        elif body.tool == "generate_and_write":
            # Real cross-framework contention demo: the resource lock spans
            # the ACTUAL LLM generation, not a fake delay -- whatever
            # LLMProvider this kernel is configured with (MockProvider or
            # the real AnthropicProvider) generates body.prompt's content,
            # and only once that real call returns does the file get
            # written and the lock released. A second caller (e.g. the
            # MCP bridge from a real Claude Desktop chat) targeting the
            # same path genuinely blocks for as long as generation
            # actually takes -- no scripted sleep anywhere in this path.
            target = _resolve_mcp_path(body.path)
            prompt = body.prompt or "Write a short note."

            async def _fn(_target=target, _prompt=prompt) -> str:
                content = await kernel.provider.complete(_prompt)
                _target.parent.mkdir(parents=True, exist_ok=True)
                _target.write_text(content, encoding="utf-8")
                return f"generated {len(content)} chars and wrote to mcp_workspace/{_target.name}"

            arguments = {"resource_key": f"mcp:{target}", "fn": _fn}
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
    import os

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
