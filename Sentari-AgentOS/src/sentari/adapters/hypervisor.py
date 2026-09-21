"""Transparent "agent hypervisor" (novel mechanism #6).

`adapters/langgraph_adapter.py` plugs into one specific framework's own
documented extension point (`ToolNode(..., awrap_tool_call=...)`). This
module takes the more general approach: instead of a per-framework adapter,
it monkey-patches a tool object's *callable entry point(s)* in place --
the one thing every framework's dispatch mechanism eventually calls,
regardless of what that framework is -- so the same mechanism mediates
LangChain/LangGraph tools, CrewAI tools, a raw function handed to a
hand-rolled OpenAI Assistants-style loop, or any future framework shaped
similarly, with zero framework-specific code and zero changes to that
framework's own source. The framework never learns Sentari is there.

Honest limitation, stated rather than hidden: mediating a genuinely
*synchronous* call path requires running the kernel's async syscall via
`asyncio.run()` internally, which only works when no other event loop is
already running in that thread. Every async entry point (the common case
for the frameworks this project targets -- LangGraph, CrewAI's async
tools) has no such constraint; see `test_hypervisor.py` for both cases
proven directly, including the sync path's limitation.

Usage:
    from sentari.adapters.hypervisor import instrument

    my_tool = SomeFrameworksToolClass(...)   # unmodified, from any framework
    instrument(my_tool, kernel, agent_id="agent1")
    # register/use my_tool with its framework exactly as before -- that
    # framework's own code needs no changes and no awareness of Sentari.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any

from sentari.kernel import Kernel
from sentari.syscalls.layer import SyscallResult, SyscallType

# Attribute names, in the order checked, that different frameworks use as a
# tool object's actual callable entry point. Every match found is
# instrumented (a tool object commonly exposes both a sync and an async
# path, e.g. LangChain's StructuredTool.func/.coroutine).
KNOWN_CALLABLE_ATTRS: tuple[str, ...] = ("coroutine", "func", "_arun", "_run", "arun", "run")


class NotInstrumentableError(TypeError):
    """Raised when `instrument()` finds no recognizable callable entry
    point on the given object, rather than silently doing nothing (which
    would look indistinguishable from successful mediation)."""


class MediationDeniedError(RuntimeError):
    """Raised when the kernel denies or errors a hypervisor-mediated call.
    Unlike the LangGraph adapter (which can translate a denial into that
    framework's own ToolMessage idiom), a framework-agnostic wrapper has
    no shared idiom to translate into -- so this is a plain, real
    exception, exactly what a synchronous tool call raising an error
    already looks like to any calling framework."""


def instrument(
    tool_obj: Any,
    kernel: Kernel,
    agent_id: str,
    resource_key: str | None = None,
) -> Any:
    """Monkey-patch every recognized callable attribute on `tool_obj` in
    place; returns the same object (mutated), so the call site can either
    use the return value or ignore it -- the original reference is already
    mediated. Raises NotInstrumentableError if nothing recognizable was
    found; for a bare function with no wrapping object, use
    `instrument_callable` instead and use its *return value* in place of
    the original function."""
    found_any = False
    for attr in KNOWN_CALLABLE_ATTRS:
        original = getattr(tool_obj, attr, None)
        if original is None or not callable(original):
            continue
        mediated = _wrap(original, kernel, agent_id, resource_key)
        try:
            setattr(tool_obj, attr, mediated)
        except AttributeError:
            continue  # read-only attribute on this particular object
        found_any = True
    if not found_any:
        raise NotInstrumentableError(
            f"no recognizable callable attribute ({', '.join(KNOWN_CALLABLE_ATTRS)}) found on "
            f"{tool_obj!r} -- for a bare function, use instrument_callable() instead"
        )
    return tool_obj


def instrument_callable(
    fn: Any, kernel: Kernel, agent_id: str, resource_key: str | None = None
) -> Any:
    """For a bare function with no wrapping tool object: returns a new,
    mediated callable with the exact same calling convention (sync stays
    sync, async stays async) as `fn` -- use the *return value* wherever
    `fn` would have been registered/passed to a framework."""
    return _wrap(fn, kernel, agent_id, resource_key)


def _wrap(original: Any, kernel: Kernel, agent_id: str, resource_key: str | None) -> Any:
    is_async = inspect.iscoroutinefunction(original)

    async def _mediate(*args: Any, **kwargs: Any) -> Any:
        async def _run() -> Any:
            if is_async:
                return await original(*args, **kwargs)
            return original(*args, **kwargs)

        arguments: dict[str, Any] = {"fn": _run}
        if resource_key:
            arguments["resource_key"] = resource_key
        response = await kernel.syscall(agent_id, SyscallType.TOOL_CALL, arguments)
        if response.result is not SyscallResult.OK:
            name = getattr(original, "__name__", repr(original))
            raise MediationDeniedError(
                f"Sentari denied mediated call to '{name}' for agent '{agent_id}': "
                f"{response.result.value} -- {response.error}"
            )
        return response.value

    if is_async:
        return functools.wraps(original)(_mediate)

    @functools.wraps(original)
    def _mediate_sync(*args: Any, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_mediate(*args, **kwargs))
        raise RuntimeError(
            "instrument()'s synchronous entry point was called from inside an already-"
            "running event loop -- call the async entry point directly instead (this is "
            "an inherent limitation of bridging a sync call into an async kernel, not a bug)"
        )

    return _mediate_sync
