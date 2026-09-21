from sentari.pcb import AgentPCB, AgentState
from sentari.persistence.db import connect
from sentari.persistence.repositories import (
    AgentRepo,
    KnowledgeBaseRepo,
    ResourceRepo,
    SyscallLogRepo,
)


def test_agent_repo_roundtrip():
    conn = connect(":memory:")
    repo = AgentRepo(conn)
    pcb = AgentPCB.new(priority=3, quota_total=5)
    pcb.transition_to(AgentState.READY)
    repo.create(pcb)

    fetched = repo.get(pcb.agent_id)
    assert fetched is not None
    assert fetched.agent_id == pcb.agent_id
    assert fetched.state is AgentState.READY
    assert fetched.priority == 3

    pcb.quota_used = 2
    pcb.transition_to(AgentState.RUNNING)
    repo.save(pcb)
    fetched2 = repo.get(pcb.agent_id)
    assert fetched2.quota_used == 2
    assert fetched2.state is AgentState.RUNNING


def test_agent_repo_list_by_state():
    conn = connect(":memory:")
    repo = AgentRepo(conn)
    p1 = AgentPCB.new(priority=1, quota_total=5)
    p1.transition_to(AgentState.READY)
    p2 = AgentPCB.new(priority=2, quota_total=5)
    p2.transition_to(AgentState.READY)
    repo.create(p1)
    repo.create(p2)

    ready = repo.list_by_state(AgentState.READY)
    assert {p.agent_id for p in ready} == {p1.agent_id, p2.agent_id}


def test_syscall_log_repo():
    conn = connect(":memory:")
    AgentRepo(conn).create(_ready_pcb("a1"))
    log_repo = SyscallLogRepo(conn)
    log_repo.log("a1", "tool_call", {"resource_key": "db"}, "OK")
    log_repo.log("a1", "yield", {}, "OK")

    entries = log_repo.list_for_agent("a1")
    assert len(entries) == 2
    assert entries[0]["syscall_type"] == "tool_call"
    assert entries[1]["result"] == "OK"


def test_resource_repo():
    conn = connect(":memory:")
    AgentRepo(conn).create(_ready_pcb("a1"))
    AgentRepo(conn).create(_ready_pcb("a2"))
    repo = ResourceRepo(conn)
    repo.record_hold("db", "a1")
    repo.record_wait("db", "a2")

    rows = repo.list_all()
    assert len(rows) == 2

    repo.clear_for_agent("a2")
    rows_after = repo.list_all()
    assert len(rows_after) == 1
    assert rows_after[0]["holder_agent_id"] == "a1"


def test_knowledge_base_repo_upsert():
    conn = connect(":memory:")
    AgentRepo(conn).create(_ready_pcb("a1"))
    kb = KnowledgeBaseRepo(conn)
    assert kb.get("shared_key") is None

    kb.set("shared_key", "v1", "a1")
    assert kb.get("shared_key") == "v1"

    kb.set("shared_key", "v2", "a1")
    assert kb.get("shared_key") == "v2"


def _ready_pcb(agent_id: str) -> AgentPCB:
    pcb = AgentPCB.new(priority=1, quota_total=10, agent_id=agent_id)
    pcb.transition_to(AgentState.READY)
    return pcb
