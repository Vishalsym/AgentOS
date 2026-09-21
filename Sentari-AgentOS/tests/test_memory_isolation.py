import pytest

from sentari.memory.manager import MemoryIsolationError, MemoryManager
from sentari.pcb import AgentPCB, AgentState
from sentari.persistence.db import connect
from sentari.persistence.repositories import AgentRepo, KnowledgeBaseRepo


def make_manager() -> MemoryManager:
    conn = connect(":memory:")
    return MemoryManager(KnowledgeBaseRepo(conn))


def make_manager_with_agent(agent_id: str) -> MemoryManager:
    conn = connect(":memory:")
    pcb = AgentPCB.new(priority=1, quota_total=5, agent_id=agent_id)
    pcb.transition_to(AgentState.READY)
    AgentRepo(conn).create(pcb)
    return MemoryManager(KnowledgeBaseRepo(conn))


def test_agent_can_read_write_own_context():
    mem = make_manager()
    mem.create_context("a1")
    mem.write_context("a1", requester_id="a1", key="k", value="v")
    assert mem.read_context("a1", requester_id="a1") == {"k": "v"}


def test_agent_cannot_read_other_agents_context():
    mem = make_manager()
    mem.create_context("a1")
    mem.write_context("a1", requester_id="a1", key="secret", value="v")
    with pytest.raises(MemoryIsolationError):
        mem.read_context("a1", requester_id="a2")


def test_agent_cannot_write_other_agents_context():
    mem = make_manager()
    mem.create_context("a1")
    with pytest.raises(MemoryIsolationError):
        mem.write_context("a1", requester_id="a2", key="k", value="v")


def test_shared_knowledge_base_is_readable_by_any_agent():
    mem = make_manager_with_agent("writer")
    mem.kb_write("writer", "shared_key", "shared_value")
    assert mem.kb_read("shared_key") == "shared_value"
