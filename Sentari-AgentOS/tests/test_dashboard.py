import asyncio

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
