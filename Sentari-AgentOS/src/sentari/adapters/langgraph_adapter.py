"""Sentari <-> LangGraph adapter (README roadmap item).

Routes every tool call LangGraph's `ToolNode` would otherwise execute
directly through the real Sentari kernel instead, using LangGraph's own
documented interception point (`ToolNode(..., awrap_tool_call=...)`, added
in langgraph-prebuilt) rather than monkeypatching or reimplementing tool
dispatch. This is what "wraps your existing framework instead of replacing
it" means concretely for LangGraph: LangGraph still owns the graph, the
state machine, and the tool schema; Sentari only mediates the moment a
tool call actually executes -- quota-checked, optionally resource-locked
(deadlock-detectable if it collides with another agent), and audit-logged,
through the exact same `SyscallLayer.on_syscall` path every other syscall
in this project uses, whether it came from the dashboard, the MCP bridge,
or here.

Usage:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.prebuilt import ToolNode
    from sentari.kernel import Kernel
    from sentari.adapters.langgraph_adapter import make_sentari_tool_wrapper

    kernel = Kernel()
    kernel.admit(priority=1, quota_total=50, agent_id="graph_agent")

    tool_node = ToolNode(
        my_tools,
        awrap_tool_call=make_sentari_tool_wrapper(kernel, agent_id="graph_agent"),
    )
    graph = StateGraph(MessagesState)
    graph.add_node("tools", tool_node)
    ...

Requires the optional 'langgraph' dependency group: uv sync --extra langgraph
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sentari.kernel import Kernel
from sentari.syscalls.layer import SyscallResult, SyscallType

if TYPE_CHECKING:
    from langgraph.prebuilt.tool_node import ToolCallRequest


def make_sentari_tool_wrapper(
    kernel: Kernel,
    agent_id: str,
    resource_key_fn: Callable[[ToolCallRequest], str | None] | None = None,
):
    """Build an `awrap_tool_call` callable for `langgraph.prebuilt.ToolNode`.

    Every tool call LangGraph would otherwise execute directly is instead
    routed through `kernel.syscall(agent_id, TOOL_CALL, ...)`:

      - denied if `agent_id`'s quota is exhausted, or if it's currently
        KILLED/TERMINATED (e.g. a prior deadlock made it a victim);
      - resource-locked (and therefore deadlock-detectable against any
        *other* agent -- including ones outside this graph entirely, since
        it's the same kernel-wide ResourceManager) if `resource_key_fn` is
        given and returns a key for this call -- e.g. to serialize every
        graph agent's writes to one shared file or rate-limited API;
      - logged into the kernel's audit trail exactly like any other
        mediated call, so a LangGraph run shows up in the dashboard/
        syscall_log the same way an MCP or dashboard-driven agent does.

    On denial, the wrapper *returns* an error-status `ToolMessage` rather
    than raising -- this deliberately matches LangGraph's own idiom for a
    failed tool call (surfaced in the graph's message state, so the next
    LLM turn can see and react to it) instead of forcing every caller of
    `graph.ainvoke()` to wrap it in a Sentari-specific try/except. Verified
    against the installed langgraph-prebuilt's actual behavior: its default
    `handle_tool_errors` only auto-converts its own internal
    `ToolInvocationError`, so a raised custom exception from this wrapper
    would otherwise propagate raw out of `ainvoke()` -- not the seamless
    "wraps your framework" integration this adapter is for.

    `agent_id` must already be admitted (`kernel.admit(...)`) before the
    graph runs -- the adapter mediates an existing agent's calls, it
    doesn't decide admission policy for you. Calling it for an unadmitted
    id is itself reported the same clean way (an error ToolMessage), not
    as a raw KeyError escaping the graph.
    """

    async def wrapper(request: ToolCallRequest, execute: Callable[[ToolCallRequest], Any]) -> Any:
        from langchain_core.messages import ToolMessage

        tool_call = request.tool_call

        def _error_message(text: str) -> ToolMessage:
            return ToolMessage(
                content=text,
                name=tool_call.get("name", "<unknown tool>"),
                tool_call_id=tool_call["id"],
                status="error",
            )

        async def _run() -> Any:
            return await execute(request)

        arguments: dict[str, Any] = {"fn": _run}
        if resource_key_fn is not None:
            resource_key = resource_key_fn(request)
            if resource_key:
                arguments["resource_key"] = resource_key

        try:
            response = await kernel.syscall(agent_id, SyscallType.TOOL_CALL, arguments)
        except KeyError:
            return _error_message(f"Sentari has no admitted agent '{agent_id}' -- call kernel.admit() first.")

        if response.result is not SyscallResult.OK:
            return _error_message(
                f"Sentari denied this tool call for agent '{agent_id}': "
                f"{response.result.value} -- {response.error}"
            )
        return response.value

    return wrapper
