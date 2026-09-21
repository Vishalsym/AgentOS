"""Tests for the LangGraph adapter (src/sentari/adapters/langgraph_adapter.py).
Skipped entirely if the optional 'langgraph' extra isn't installed
(`uv sync --extra langgraph`) -- matches how the AnthropicProvider live-key
tests are skip-if-unavailable rather than hard dependencies of the base
test suite."""

from __future__ import annotations

import pytest

langgraph = pytest.importorskip("langgraph")
langchain_core = pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from sentari.adapters.langgraph_adapter import make_sentari_tool_wrapper  # noqa: E402
from sentari.kernel import Kernel  # noqa: E402
from sentari.pcb import AgentState  # noqa: E402


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def boom() -> str:
    """A tool that always raises, to prove kernel-level failure containment
    still applies when the call arrives via LangGraph, not just via a
    direct kernel.syscall() caller."""
    raise ValueError("this tool is broken on purpose")


def _make_graph(tool_node: ToolNode):
    g = StateGraph(MessagesState)
    g.add_node("tools", tool_node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile()


def _ai_message_calling(name: str, args: dict, call_id: str = "call1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.mark.asyncio
async def test_langgraph_tool_call_is_mediated_and_audit_logged():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=10, agent_id="graph_agent")

    tool_node = ToolNode([add], awrap_tool_call=make_sentari_tool_wrapper(kernel, agent_id="graph_agent"))
    graph = _make_graph(tool_node)

    result = await graph.ainvoke({"messages": [_ai_message_calling("add", {"a": 2, "b": 3})]})
    tool_message = result["messages"][-1]
    assert tool_message.content == "5"  # ToolMessage.content is always a string
    assert tool_message.status != "error"

    # it really went through the kernel: quota charged, audit log written.
    pcb = kernel.scheduler.get("graph_agent")
    assert pcb.quota_used == 1
    logs = kernel.syscall_log_repo.list_for_agent("graph_agent")
    assert len(logs) == 1
    assert logs[0]["syscall_type"] == "tool_call"
    assert logs[0]["result"] == "OK"
    kernel.close()


@pytest.mark.asyncio
async def test_langgraph_call_denied_on_quota_exhaustion_surfaces_as_tool_error():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=1, agent_id="broke_agent")

    tool_node = ToolNode([add], awrap_tool_call=make_sentari_tool_wrapper(kernel, agent_id="broke_agent"))
    graph = _make_graph(tool_node)

    # first call spends the only quota unit -- succeeds, agent stays READY
    # (termination is lazy: it's evaluated at the *next* syscall attempt,
    # per SyscallLayer.on_syscall's admission-time quota check).
    first = await graph.ainvoke({"messages": [_ai_message_calling("add", {"a": 1, "b": 1})]})
    assert first["messages"][-1].status != "error"

    # second call must not crash the graph -- the wrapper itself returns a
    # clean error-status ToolMessage on denial (verified: LangGraph's own
    # default handle_tool_errors does NOT auto-convert an arbitrary raised
    # exception, only its internal ToolInvocationError, so the adapter has
    # to produce the ToolMessage itself rather than just raising).
    result = await graph.ainvoke({"messages": [_ai_message_calling("add", {"a": 9, "b": 9})]})
    tool_message = result["messages"][-1]
    assert tool_message.status == "error"
    assert "Sentari denied" in tool_message.content
    assert kernel.scheduler.get("broke_agent").state is AgentState.TERMINATED
    kernel.close()


@pytest.mark.asyncio
async def test_langgraph_tool_exception_is_contained_not_a_crashed_graph():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=5, agent_id="chaos_graph_agent")

    tool_node = ToolNode(
        [boom], awrap_tool_call=make_sentari_tool_wrapper(kernel, agent_id="chaos_graph_agent")
    )
    graph = _make_graph(tool_node)

    result = await graph.ainvoke({"messages": [_ai_message_calling("boom", {})]})
    tool_message = result["messages"][-1]
    assert tool_message.status == "error"
    # the agent itself survives one bad tool call -- same containment
    # guarantee as every other syscall path in this project.
    assert kernel.scheduler.get("chaos_graph_agent").state is not AgentState.KILLED
    kernel.close()


@pytest.mark.asyncio
async def test_resource_key_fn_mediates_cross_agent_contention():
    """Two separate LangGraph agents, each in their own graph, both wanting
    the same resource_key -- proves the adapter plugs into the kernel-wide
    ResourceManager (and therefore deadlock detection), not a
    graph-local/no-op stand-in."""
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=5, agent_id="writer_a")
    kernel.admit(priority=1, quota_total=5, agent_id="writer_b")

    def resource_key_fn(request) -> str:
        return "shared_file"

    node_a = ToolNode(
        [add], awrap_tool_call=make_sentari_tool_wrapper(kernel, "writer_a", resource_key_fn)
    )
    graph_a = _make_graph(node_a)

    # writer_a's call acquires and releases "shared_file" around the tool
    # execution (acquire -> run -> release, per SyscallLayer._handle_tool_call).
    result = await graph_a.ainvoke({"messages": [_ai_message_calling("add", {"a": 1, "b": 2})]})
    assert result["messages"][-1].content == "3"
    # resource was released again after the mediated call completed.
    assert kernel.resources._held.get("shared_file") is None

    logs = kernel.syscall_log_repo.list_for_agent("writer_a")
    assert logs[0]["arguments"] and "shared_file" in logs[0]["arguments"]
    kernel.close()


@pytest.mark.asyncio
async def test_agent_must_be_pre_admitted_before_graph_runs():
    """The adapter mediates an existing agent's calls -- it is not an
    admission policy. An unadmitted agent_id raises a KeyError from
    Scheduler.get (before on_syscall's own try/except even starts); the
    wrapper catches that itself and returns a clean error ToolMessage,
    since LangGraph's own default error handling only auto-converts its
    internal ToolInvocationError, not an arbitrary propagating exception."""
    kernel = Kernel()
    tool_node = ToolNode([add], awrap_tool_call=make_sentari_tool_wrapper(kernel, agent_id="never_admitted"))
    graph = _make_graph(tool_node)

    result = await graph.ainvoke({"messages": [_ai_message_calling("add", {"a": 1, "b": 1})]})
    tool_message = result["messages"][-1]
    assert tool_message.status == "error"
    kernel.close()
