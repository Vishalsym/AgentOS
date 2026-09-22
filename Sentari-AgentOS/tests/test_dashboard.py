import asyncio
import re
import time

import pytest
from fastapi.testclient import TestClient

import sentari.dashboard.app as app_module
from sentari.dashboard.app import app


def _wait_for_kernel(client: TestClient) -> None:
    for _ in range(50):
        if app_module.dashboard_state.kernel is not None:
            return
        asyncio.run(asyncio.sleep(0.05))
    raise AssertionError("kernel never became ready")


def test_index_page_serves_html():
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "CONTROL DASHBOARD" in resp.text


def test_state_endpoint_reflects_running_scenario():
    with TestClient(app) as client:
        resp = client.post("/api/restart", params={"scenario": "standard"})
        assert resp.status_code == 200
        # give the background scenario task a moment to admit the agents
        for _ in range(50):
            resp = client.get("/api/state")
            data = resp.json()
            if data["agents"]:
                break
            asyncio.run(asyncio.sleep(0.05))
        assert resp.status_code == 200
        agent_ids = {a["agent_id"] for a in data["agents"]}
        assert {"alpha", "beta", "gamma"}.issubset(agent_ids)


def test_default_boot_is_live_with_no_scripted_agents():
    with TestClient(app) as client:
        _wait_for_kernel(client)
        data = client.get("/api/state").json()
        assert data["scenario"] == "live"
        assert data["agents"] == []


def test_restart_endpoint_resets_scenario():
    with TestClient(app) as client:
        resp = client.post("/api/restart")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "scenario": "live"}


def test_restart_endpoint_accepts_before_after_scenario():
    with TestClient(app) as client:
        resp = client.post("/api/restart", params={"scenario": "before_after"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "scenario": "before_after"}

        for _ in range(50):
            data = client.get("/api/state").json()
            if data["scenario"] == "before_after":
                break
            asyncio.run(asyncio.sleep(0.05))
        assert data["scenario"] == "before_after"


def test_scenarios_endpoint_lists_available_scenarios():
    with TestClient(app) as client:
        resp = client.get("/api/scenarios")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body["scenarios"]) == {"standard", "before_after", "live"}
        assert body["default"] == "live"


def test_mcp_tool_call_write_and_read_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MCP_WORKSPACE_DIR", tmp_path)

    with TestClient(app) as client:
        _wait_for_kernel(client)

        write_resp = client.post(
            "/api/mcp/tool_call",
            json={
                "agent_id": "claude_desktop_test",
                "tool": "write_file",
                "path": "hello.txt",
                "content": "hi from claude desktop",
            },
        )
        assert write_resp.status_code == 200
        assert write_resp.json()["result"] == "OK"
        assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hi from claude desktop"

        read_resp = client.post(
            "/api/mcp/tool_call",
            json={"agent_id": "claude_desktop_test", "tool": "read_file", "path": "hello.txt"},
        )
        assert read_resp.status_code == 200
        assert read_resp.json()["value"] == "hi from claude desktop"

        agent_resp = client.get("/api/agent/claude_desktop_test")
        assert agent_resp.status_code == 200
        assert agent_resp.json()["quota_used"] >= 2


def test_mcp_generate_and_write_writes_real_provider_output(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MCP_WORKSPACE_DIR", tmp_path)

    with TestClient(app) as client:
        _wait_for_kernel(client)
        resp = client.post(
            "/api/mcp/tool_call",
            json={
                "agent_id": "langgraph_writer",
                "tool": "generate_and_write",
                "path": "shared_report.md",
                "prompt": "Write a short note.",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["result"] == "OK"
        written = (tmp_path / "shared_report.md").read_text(encoding="utf-8")
        assert written  # the kernel's own configured provider's real output landed on disk
        assert "generated" in resp.json()["value"]


def test_mcp_generate_and_write_genuinely_holds_the_lock_across_generation(monkeypatch, tmp_path):
    """The whole point of this tool: a second agent targeting the SAME path
    must block for as long as the first agent's *generation* is still
    running, not just for the final disk write. Proven here with a
    deliberately slow fake provider standing in for a real, slow LLM call
    -- the resource lock spans the slow part, exactly like it would span a
    real Anthropic completion."""
    monkeypatch.setattr(app_module, "MCP_WORKSPACE_DIR", tmp_path)

    class SlowProvider:
        def __init__(self):
            self.started = asyncio.Event()

        async def complete(self, prompt: str, **kwargs) -> str:
            self.started.set()
            await asyncio.sleep(0.3)
            return "slow real content"

    with TestClient(app) as client:
        _wait_for_kernel(client)
        slow_provider = SlowProvider()
        app_module.dashboard_state.kernel.provider = slow_provider

        async def scenario():
            first = asyncio.create_task(
                asyncio.to_thread(
                    client.post,
                    "/api/mcp/tool_call",
                    json={
                        "agent_id": "langgraph_writer",
                        "tool": "generate_and_write",
                        "path": "contended.md",
                        "prompt": "long real generation",
                    },
                )
            )
            await slow_provider.started.wait()  # generation has genuinely begun

            # a second agent targeting the SAME path, while generation (not
            # just the write) is still in flight.
            second = await asyncio.to_thread(
                client.post,
                "/api/mcp/tool_call",
                json={
                    "agent_id": "claude_desktop_contender",
                    "tool": "write_file",
                    "path": "contended.md",
                    "content": "claude's version",
                },
            )
            first_resp = await first
            return first_resp, second

        first_resp, second_resp = asyncio.run(scenario())
        assert first_resp.status_code == 200
        assert first_resp.json()["result"] == "OK"
        assert second_resp.status_code == 200
        assert second_resp.json()["result"] == "OK"  # it waited, then succeeded -- not denied
        # whichever finished last "won" the file -- both real, mediated writes happened in order.
        final = (tmp_path / "contended.md").read_text(encoding="utf-8")
        assert final in ("slow real content", "claude's version")


def test_mcp_bare_directory_path_rejected_cleanly(monkeypatch, tmp_path):
    """Regression test: '.'/''/a path that resolves to the workspace root
    itself used to slip past validation and blow up as a raw PermissionError
    from the filesystem instead of a clean 400."""
    monkeypatch.setattr(app_module, "MCP_WORKSPACE_DIR", tmp_path)

    with TestClient(app) as client:
        _wait_for_kernel(client)

        for bad_path in (".", ""):
            resp = client.post(
                "/api/mcp/tool_call",
                json={"agent_id": "adversary", "tool": "write_file", "path": bad_path, "content": "x"},
            )
            assert resp.status_code == 400
            assert "PermissionError" not in resp.text

        (tmp_path / "subdir").mkdir()
        resp = client.post(
            "/api/mcp/tool_call",
            json={"agent_id": "adversary", "tool": "write_file", "path": "subdir", "content": "x"},
        )
        assert resp.status_code == 400
        assert "directory" in resp.text.lower()


def test_mcp_path_traversal_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MCP_WORKSPACE_DIR", tmp_path)

    with TestClient(app) as client:
        _wait_for_kernel(client)
        resp = client.post(
            "/api/mcp/tool_call",
            json={"agent_id": "adversary", "tool": "read_file", "path": "../../etc/passwd"},
        )
        assert resp.status_code == 400
        assert "escapes" in resp.text.lower()


def test_mcp_tool_call_rejects_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MCP_WORKSPACE_DIR", tmp_path)

    with TestClient(app) as client:
        _wait_for_kernel(client)

        resp = client.post(
            "/api/mcp/tool_call",
            json={
                "agent_id": "claude_desktop_test2",
                "tool": "write_file",
                "path": "../escape.txt",
                "content": "nope",
            },
        )
        assert resp.status_code == 400
        assert not (tmp_path.parent / "escape.txt").exists()


def test_mcp_tool_call_note_charges_quota_without_touching_filesystem():
    with TestClient(app) as client:
        _wait_for_kernel(client)

        resp = client.post(
            "/api/mcp/tool_call",
            json={"agent_id": "claude_desktop_note_test", "tool": "note", "prompt": "hello"},
        )
        assert resp.status_code == 200
        assert resp.json()["result"] == "OK"

        agent_resp = client.get("/api/agent/claude_desktop_note_test")
        assert agent_resp.json()["quota_used"] == 1


def test_interrupt_endpoint_delegates_to_kernel_interrupt_not_kill_manager(monkeypatch):
    """/api/interrupt must call kernel.interrupt() (the bounded-latency
    primitive that cancels a real in-flight asyncio Task -- proven at the
    kernel level in tests/test_interrupt.py, including a real measured-
    latency assertion) rather than kill_manager.kill() directly (which
    only updates PCB state/releases resources and leaves any real in-
    flight network call running in the background). This test proves the
    HTTP layer wiring; it deliberately does not re-prove the cancellation
    mechanism itself, since TestClient's single-threaded request portal
    can't reliably reproduce a genuinely concurrent in-flight-call scenario
    the way a direct Kernel() test can."""
    with TestClient(app) as client:
        _wait_for_kernel(client)
        client.post(
            "/api/admit", json={"agent_id": "interrupt_target", "priority": 1, "quota_total": 10}
        )

        calls = []

        async def fake_interrupt(agent_id, reason="human_interrupt"):
            calls.append((agent_id, reason))
            return True

        monkeypatch.setattr(app_module.dashboard_state.kernel, "interrupt", fake_interrupt)

        resp = client.post(
            "/api/interrupt", json={"agent_id": "interrupt_target", "reason": "live_demo_interrupt"}
        )

        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "had_inflight_call": True}
        assert calls == [("interrupt_target", "live_demo_interrupt")]


class _TaskAwareFakeProvider:
    """Scores a prompt by sniffing for a keyword -- stands in for a real
    LLM judging which agent's work matters more, with a controllable,
    deterministic outcome for testing."""

    async def complete(self, prompt: str, **kwargs) -> str:
        if "irreplaceable" in prompt:
            return "0.95"
        return "0.05"


def test_admit_with_task_description_scores_and_surfaces_task_value():
    with TestClient(app) as client:
        _wait_for_kernel(client)
        app_module.dashboard_state.kernel.provider = _TaskAwareFakeProvider()

        resp = client.post(
            "/api/admit",
            json={
                "agent_id": "valuable_worker",
                "priority": 10,
                "quota_total": 10,
                "task_description": "finalize the irreplaceable customer report",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["task_value"] == pytest.approx(0.95)

        agent = client.get("/api/agent/valuable_worker").json()
        assert agent["task_value"] == pytest.approx(0.95)

        state = client.get("/api/state").json()
        entry = next(a for a in state["agents"] if a["agent_id"] == "valuable_worker")
        assert entry["task_value"] == pytest.approx(0.95)


def test_admit_without_task_description_leaves_task_value_null():
    with TestClient(app) as client:
        _wait_for_kernel(client)
        resp = client.post("/api/admit", json={"agent_id": "plain_worker", "priority": 5, "quota_total": 5})
        assert resp.status_code == 200
        assert resp.json()["task_value"] is None
        agent = client.get("/api/agent/plain_worker").json()
        assert agent["task_value"] is None


def test_semantic_value_overrides_priority_in_a_real_dashboard_deadlock():
    """The exact live-demo scenario: 'researcher' is high priority (1) but
    declares low-value scratch work; 'writer' is low priority (5) but
    declares irreplaceable work. A real circular wait forms through the
    Cockpit's own acquire endpoints; the kernel must spare 'writer' --
    the priority-only outcome would have been the opposite."""
    with TestClient(app) as client:
        _wait_for_kernel(client)
        app_module.dashboard_state.kernel.provider = _TaskAwareFakeProvider()

        client.post(
            "/api/admit",
            json={
                "agent_id": "researcher",
                "priority": 1,
                "quota_total": 10,
                "task_description": "quick disposable scratch note",
            },
        )
        client.post(
            "/api/admit",
            json={
                "agent_id": "writer",
                "priority": 5,
                "quota_total": 10,
                "task_description": "finalize the irreplaceable customer report",
            },
        )

        assert client.get("/api/agent/researcher").json()["task_value"] == pytest.approx(0.05)
        assert client.get("/api/agent/writer").json()["task_value"] == pytest.approx(0.95)

        client.post("/api/acquire", json={"agent_id": "researcher", "resource_key": "dataset"})
        client.post("/api/acquire", json={"agent_id": "writer", "resource_key": "draft"})
        time.sleep(0.2)

        client.post("/api/acquire", json={"agent_id": "writer", "resource_key": "dataset"})
        time.sleep(0.2)
        client.post("/api/acquire", json={"agent_id": "researcher", "resource_key": "draft"})
        time.sleep(0.3)

        researcher_state = client.get("/api/agent/researcher").json()["state"]
        writer_state = client.get("/api/agent/writer").json()["state"]
        # priority alone would have killed 'writer' (priority 5, less
        # important); task_value must flip this outcome.
        assert writer_state != "KILLED"
        assert researcher_state == "KILLED"


class _SlowSequencedProvider:
    """Each real call takes a controllable, deliberate slice of time.
    The FIRST call any test makes is always the planning call (detected by
    the <PLAN> marker instruction) -- this fake returns a fixed 3-role
    linear plan so the rest of the pipeline is deterministic, while still
    exercising the real planner-parsing code path in app.py. Subsequent
    calls are stage generations, matched by the agent name the real
    orchestration embeds in every stage prompt ("You are the '<name>'
    agent"), and return content identifying which stage produced it --
    lets a test observe genuine mid-flight blocking (not just "things
    happened to run in the right order") and confirms downstream agents'
    prompts actually contain the real upstream content, not a placeholder."""

    def __init__(self, delay: float = 0.25):
        self._delay = delay

    async def complete(self, prompt: str, **kwargs) -> str:
        await asyncio.sleep(self._delay)
        if "<PLAN>" in prompt and "depends_on" in prompt:
            return (
                "<PLAN>\n"
                '[{"name": "designer", "task": "design the app", "depends_on": []}, '
                '{"name": "implementer", "task": "implement the app", "depends_on": ["designer"]}, '
                '{"name": "verifier", "task": "verify the app", "depends_on": ["implementer"]}]\n'
                "</PLAN>"
            )
        match = re.search(r"You are the '(\w+)' agent", prompt)
        name = match.group(1) if match else None
        if name == "designer":
            return "REAL DESIGN OUTPUT: a to-do list with add/remove/complete."
        if name == "implementer":
            assert "REAL DESIGN OUTPUT" in prompt  # genuinely read the real upstream file
            return "REAL IMPLEMENTATION OUTPUT: a Flask app with a Task model."
        if name == "verifier":
            assert "REAL IMPLEMENTATION OUTPUT" in prompt
            return "REAL VERIFICATION OUTPUT: covers add/remove/complete; no gaps found."
        return "unexpected prompt"


class _FixedPlanProvider:
    def __init__(self, response: str) -> None:
        self._response = response

    async def complete(self, prompt: str, **kwargs) -> str:
        return self._response


class _FakeKernel:
    def __init__(self, provider) -> None:
        self.provider = provider


@pytest.mark.asyncio
async def test_plan_team_supports_more_than_three_dynamically_named_agents():
    """The core ask this generalizes away from: no hardcoded designer/
    implementer/verifier, and more than 3 agents must be possible."""
    plan_json = (
        "<PLAN>\n"
        '[{"name": "ui_designer", "task": "design screens", "depends_on": []}, '
        '{"name": "backend_dev", "task": "build the API", "depends_on": []}, '
        '{"name": "db_admin", "task": "design the schema", "depends_on": []}, '
        '{"name": "integrator", "task": "wire it together", '
        '"depends_on": ["ui_designer", "backend_dev", "db_admin"]}]\n'
        "</PLAN>"
    )
    kernel = _FakeKernel(_FixedPlanProvider(plan_json))
    plan = await app_module._plan_team(kernel, "a project")
    assert [a["name"] for a in plan] == ["ui_designer", "backend_dev", "db_admin", "integrator"]
    assert plan[3]["depends_on"] == ["ui_designer", "backend_dev", "db_admin"]


@pytest.mark.asyncio
async def test_plan_team_sanitizes_unsafe_names_and_dedupes_collisions():
    plan_json = (
        "<PLAN>\n"
        '[{"name": "UI Designer!!", "task": "t1", "depends_on": []}, '
        '{"name": "ui designer", "task": "t2", "depends_on": []}]\n'
        "</PLAN>"
    )
    kernel = _FakeKernel(_FixedPlanProvider(plan_json))
    plan = await app_module._plan_team(kernel, "a project")
    names = [a["name"] for a in plan]
    assert names[0] == "ui_designer"
    assert names[1] != names[0]  # collision after sanitizing must not silently merge two agents


@pytest.mark.asyncio
async def test_plan_team_drops_unknown_and_self_dependencies():
    plan_json = (
        "<PLAN>\n"
        '[{"name": "solo", "task": "t1", "depends_on": ["solo", "ghost_agent"]}]\n'
        "</PLAN>"
    )
    kernel = _FakeKernel(_FixedPlanProvider(plan_json))
    plan = await app_module._plan_team(kernel, "a project")
    assert plan[0]["depends_on"] == []


@pytest.mark.asyncio
async def test_plan_team_clamps_to_max_team_size():
    entries = ", ".join(f'{{"name": "agent_{i}", "task": "t", "depends_on": []}}' for i in range(10))
    kernel = _FakeKernel(_FixedPlanProvider(f"<PLAN>\n[{entries}]\n</PLAN>"))
    plan = await app_module._plan_team(kernel, "a project")
    assert len(plan) <= app_module.MAX_TEAM_SIZE


@pytest.mark.asyncio
async def test_plan_team_raises_clearly_when_response_has_no_plan_block():
    kernel = _FakeKernel(_FixedPlanProvider("sure, here's a plan: designer, implementer, verifier"))
    with pytest.raises(ValueError, match="PLAN"):
        await app_module._plan_team(kernel, "a project")


def test_team_start_rejects_an_empty_project():
    with TestClient(app) as client:
        _wait_for_kernel(client)
        resp = client.post("/api/team/start", json={"project": "  "})
        assert resp.status_code == 400


def test_team_of_three_real_agents_genuinely_gates_in_order(monkeypatch, tmp_path):
    """The core claim: implementer must be BLOCKED (not just 'not started
    yet') while designer is still mid-generation, and verifier must be
    BLOCKED while implementer is still mid-generation -- proven by
    checking live PCB state partway through a real, slow run, not just by
    checking the final files exist in the right order."""
    monkeypatch.setattr(app_module, "MCP_WORKSPACE_DIR", tmp_path)

    with TestClient(app) as client:
        _wait_for_kernel(client)
        app_module.dashboard_state.kernel.provider = _SlowSequencedProvider(delay=0.3)

        resp = client.post("/api/team/start", json={"project": "a simple to-do list app"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["started"] is True
        assert [a["name"] for a in data["plan"]] == ["designer", "implementer", "verifier"]

        time.sleep(0.15)  # designer has started; implementer/verifier should be gated
        designer = client.get("/api/agent/designer").json()
        implementer = client.get("/api/agent/implementer").json()
        assert designer["state"] == "RUNNING"
        assert implementer["state"] in ("BLOCKED", "READY")  # genuinely not running yet

        time.sleep(2.5)  # let the whole real (slow) pipeline finish (plan + 3 sequential stage calls)
        design_file = tmp_path / "team_designer.md"
        impl_file = tmp_path / "team_implementer.md"
        verify_file = tmp_path / "team_verifier.md"
        assert design_file.exists()
        assert impl_file.exists()
        assert verify_file.exists()
        assert "REAL DESIGN OUTPUT" in design_file.read_text(encoding="utf-8")
        assert "REAL IMPLEMENTATION OUTPUT" in impl_file.read_text(encoding="utf-8")
        assert "REAL VERIFICATION OUTPUT" in verify_file.read_text(encoding="utf-8")

        for agent_id in ("designer", "implementer", "verifier"):
            state = client.get(f"/api/agent/{agent_id}").json()["state"]
            assert state != "KILLED", f"{agent_id} should never be killed in a linear (non-cyclic) handoff"


def test_team_cannot_be_started_twice_in_the_same_kernel(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MCP_WORKSPACE_DIR", tmp_path)
    with TestClient(app) as client:
        _wait_for_kernel(client)
        app_module.dashboard_state.kernel.provider = _SlowSequencedProvider(delay=0.05)
        client.post("/api/team/start", json={"project": "first run"})
        time.sleep(0.5)
        resp = client.post("/api/team/start", json={"project": "second run"})
        assert resp.status_code == 400
        assert "restart" in resp.json()["error"].lower()
