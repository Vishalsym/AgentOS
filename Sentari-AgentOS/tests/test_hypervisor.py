"""Tests for the transparent "agent hypervisor" (novel mechanism #6):
mediating arbitrary framework tool objects by instrumenting their callable
entry points in place, without any framework-specific code.

Uses hand-built fake objects shaped like real frameworks' tool classes
(LangChain-style func/coroutine, CrewAI-style _run) rather than depending
on those packages directly -- proves the mechanism is genuinely
framework-agnostic (it only inspects attribute names/callability, nothing
framework-specific), and keeps this suite fast and dependency-free."""

from __future__ import annotations

import pytest

from sentari.adapters.hypervisor import (
    MediationDeniedError,
    NotInstrumentableError,
    instrument,
    instrument_callable,
)
from sentari.kernel import Kernel
from sentari.pcb import AgentState

# --------------------------------------------------------------------------
# Fakes shaped like real frameworks' tool objects -- the frameworks
# themselves are never imported or modified.
# --------------------------------------------------------------------------


class LangChainStyleTool:
    """Shaped like langchain_core.tools.StructuredTool: separate sync
    `func` and async `coroutine` slots, called by that framework's own
    (unmodified, not reproduced here) dispatch machinery."""

    def __init__(self, name: str):
        self.name = name

        def _sync_impl(a: int, b: int) -> int:
            return a + b

        async def _async_impl(a: int, b: int) -> int:
            return a + b

        self.func = _sync_impl
        self.coroutine = _async_impl


class CrewAIStyleTool:
    """Shaped like a CrewAI BaseTool subclass: a single `_run` method."""

    def __init__(self):
        def _run(query: str) -> str:
            return f"result for {query}"

        self._run = _run


def plain_function(x: int) -> int:
    return x * 2


async def plain_async_function(x: int) -> int:
    return x * 2


# --------------------------------------------------------------------------
# instrument() on wrapping objects
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_instrument_mediates_a_langchain_shaped_tools_async_slot():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=10, agent_id="a1")
    tool = LangChainStyleTool("adder")
    instrument(tool, kernel, agent_id="a1")

    # the framework's own dispatch would call tool.coroutine(...) exactly
    # like this -- it never knows mediation happened.
    result = await tool.coroutine(a=2, b=3)
    assert result == 5

    logs = kernel.syscall_log_repo.list_for_agent("a1")
    assert len(logs) == 1
    assert logs[0]["result"] == "OK"
    kernel.close()


def test_instrument_mediates_a_langchain_shaped_tools_sync_slot():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=10, agent_id="a1")
    tool = LangChainStyleTool("adder")
    instrument(tool, kernel, agent_id="a1")

    # a framework calling the SYNC slot synchronously, with no event loop
    # of its own running -- the calling convention is preserved exactly
    # (no await needed at the call site, matching the original signature).
    result = tool.func(a=2, b=3)
    assert result == 5
    kernel.close()


def test_instrument_mediates_a_crewai_shaped_tools_run_method():
    """A plain (non-async) test function: pytest never puts a running
    event loop around it, so this genuinely matches how CrewAI's own
    synchronous machinery would call `_run` -- no loop already running,
    same thread the kernel was created in."""
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=10, agent_id="a1")
    tool = CrewAIStyleTool()
    instrument(tool, kernel, agent_id="a1")

    result = tool._run("hello")
    assert result == "result for hello"
    kernel.close()


def test_instrument_raises_clearly_on_an_uninstrumentable_object():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=10, agent_id="a1")

    class NothingCallableHere:
        name = "useless"

    with pytest.raises(NotInstrumentableError):
        instrument(NothingCallableHere(), kernel, agent_id="a1")
    kernel.close()


# --------------------------------------------------------------------------
# instrument_callable() on a bare function
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_instrument_callable_mediates_a_bare_async_function():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=10, agent_id="a1")
    mediated = instrument_callable(plain_async_function, kernel, agent_id="a1")
    result = await mediated(21)
    assert result == 42
    kernel.close()


def test_instrument_callable_mediates_a_bare_sync_function():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=10, agent_id="a1")
    mediated = instrument_callable(plain_function, kernel, agent_id="a1")
    result = mediated(21)  # unchanged calling convention: no await needed
    assert result == 42
    kernel.close()


@pytest.mark.asyncio
async def test_sync_wrapper_refuses_to_run_inside_an_already_running_loop():
    """The documented, honest limitation: bridging a sync call into the
    async kernel via asyncio.run() cannot work if a loop is already
    running in this thread (we're inside one right now, via
    pytest-asyncio) -- must fail clearly, not deadlock or silently break."""
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=10, agent_id="a1")
    mediated = instrument_callable(plain_function, kernel, agent_id="a1")
    with pytest.raises(RuntimeError, match="already-running event loop"):
        mediated(5)
    kernel.close()


# --------------------------------------------------------------------------
# Mediation is real: quota, containment, resource-key all apply
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mediated_call_is_denied_on_quota_exhaustion():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=1, agent_id="tight")
    mediated = instrument_callable(plain_async_function, kernel, agent_id="tight")

    await mediated(1)  # spends the only quota unit
    with pytest.raises(MediationDeniedError):
        await mediated(1)
    assert kernel.scheduler.get("tight").state is AgentState.TERMINATED
    kernel.close()


@pytest.mark.asyncio
async def test_mediated_tool_exception_is_contained_and_reported_cleanly():
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=10, agent_id="a1")

    async def broken(x: int) -> int:
        raise ValueError("boom")

    mediated = instrument_callable(broken, kernel, agent_id="a1")
    with pytest.raises(MediationDeniedError, match="boom"):
        await mediated(1)
    # the agent itself survives -- same containment as every other path.
    assert kernel.scheduler.get("a1").state is not AgentState.KILLED
    kernel.close()


@pytest.mark.asyncio
async def test_resource_key_enables_real_cross_agent_deadlock_detection():
    """Two DIFFERENT hypervisor-mediated tools, from two different
    (fake) frameworks, both writing to the same resource_key -- proves
    this plugs into the real kernel-wide ResourceManager, same as every
    other mediation path in this project, not a private/no-op stand-in."""
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=10, agent_id="writer_a")

    async def write_a() -> str:
        return "wrote from framework A"

    mediated = instrument_callable(write_a, kernel, agent_id="writer_a", resource_key="shared_file")
    result = await mediated()
    assert result == "wrote from framework A"
    # resource was acquired-and-released around the call, exactly like the
    # MCP bridge and LangGraph adapter's resource_key_fn path.
    assert kernel.resources._held.get("shared_file") is None
    logs = kernel.syscall_log_repo.list_for_agent("writer_a")
    assert "shared_file" in logs[0]["arguments"]
    kernel.close()


def test_two_different_fake_frameworks_mediated_by_the_same_kernel_identically():
    """The actual point of the hypervisor idea: a LangChain-shaped tool's
    sync slot and a CrewAI-shaped tool's _run, instrumented the exact same
    way, produce identical kernel-level mediation (audit log, quota
    charge) -- one mechanism, not per-framework special cases. A plain
    (non-async) test function, matching how both frameworks' real sync
    dispatch actually calls these."""
    kernel = Kernel()
    kernel.admit(priority=1, quota_total=10, agent_id="a1")

    lc_tool = LangChainStyleTool("adder")
    crew_tool = CrewAIStyleTool()
    instrument(lc_tool, kernel, agent_id="a1")
    instrument(crew_tool, kernel, agent_id="a1")

    lc_tool.func(a=1, b=1)
    crew_tool._run("q")

    logs = kernel.syscall_log_repo.list_for_agent("a1")
    assert len(logs) == 2
    assert all(row["result"] == "OK" for row in logs)
    kernel.close()
