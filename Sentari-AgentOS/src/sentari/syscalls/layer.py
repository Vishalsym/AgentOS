from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sentari.pcb import AgentPCB, AgentState


class SyscallType(StrEnum):
    TOOL_CALL = "tool_call"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    SPAWN = "spawn"
    YIELD = "yield"


class SyscallResult(StrEnum):
    OK = "OK"
    WAIT = "WAIT"
    DENY = "DENY"
    ERROR = "ERROR"


@dataclass
class SyscallRequest:
    agent_id: str
    syscall_type: SyscallType
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyscallResponse:
    result: SyscallResult
    value: Any = None
    error: str | None = None
    # True when a token-aware-quota prompt call (see SyscallLayer's
    # token_aware_quota) was cut short by hitting its remaining-token cap.
    # `value` still holds whatever text was actually generated up to that
    # point -- a real partial result, not withheld.
    truncated: bool = False


DEFAULT_EXECUTION_TIMEOUT = 30.0


def default_cost(request: SyscallRequest) -> int:
    return 1


class SyscallLayer:
    """The sole mediated entry point to external resources (FR-7). Every
    call is validated against quota before it runs and logged after it
    finishes, win or lose (FR-8, FR-9). This implements the Design Doc's
    `on_syscall` algorithm."""

    def __init__(
        self,
        scheduler,
        memory_manager,
        resource_manager,
        syscall_log_repo,
        provider,
        spawn_fn: Callable[[str, dict[str, Any]], AgentPCB],
        kill_manager,
        execution_timeout: float = DEFAULT_EXECUTION_TIMEOUT,
        cost_fn: Callable[[SyscallRequest], int] = default_cost,
        kb_verifier: Callable[..., Any] | None = None,
        kb_verification_strict: bool = False,
        token_aware_quota: bool = False,
    ):
        self._scheduler = scheduler
        self._memory = memory_manager
        self._resources = resource_manager
        self._log_repo = syscall_log_repo
        self._provider = provider
        self._spawn_fn = spawn_fn
        self._kill_manager = kill_manager
        self._timeout = execution_timeout
        self._cost_fn = cost_fn
        # Token-aware quota: when True, a tool_call with a "prompt" charges
        # quota by REAL tokens spent (input+output, read back from the
        # provider) instead of a flat 1-per-call, and caps the call's
        # max_tokens at the agent's remaining budget so a single call can't
        # blow through it -- quota_total/quota_used are then interpreted as
        # a token budget, not a call count. False by default: zero change
        # to the flat call-counted behavior every existing test relies on.
        self._token_aware_quota = token_aware_quota
        self._last_token_result: dict[str, Any] = {}
        # Hallucination interception for shared-KB reads (novel mechanism
        # #3). None by default -- fully opt-in, reproduces the exact prior
        # kb_read behavior when not configured. `kb_verifier` may be sync
        # or async: `(value, evidence) -> VerificationResult |
        # Awaitable[VerificationResult]` (see memory/hallucination.py).
        self._kb_verifier = kb_verifier
        self._kb_verification_strict = kb_verification_strict
        # Bounded-latency human interrupt (novel mechanism #4): a handle to
        # each agent's currently in-flight dispatch task, so `interrupt()`
        # can cancel it *immediately* -- not wait up to `execution_timeout`
        # for the normal preemption path to notice. This is what makes a
        # human "stop that agent now" a measurable systems primitive
        # instead of a UI convenience riding on the existing timeout.
        self._inflight: dict[str, asyncio.Task] = {}

    async def on_syscall(self, request: SyscallRequest) -> SyscallResponse:
        pcb = self._scheduler.get(request.agent_id)

        # Cheap early-exit: if this agent is already unmistakably
        # exhausted, skip queueing for a turn at all. This is an
        # optimization only, not the enforcement itself -- it's racy
        # under concurrent same-agent calls (they can all read quota_used
        # before any of them has incremented it), which is exactly why the
        # authoritative check below exists.
        if pcb.quota_used >= pcb.quota_total:
            await self._kill_manager.terminate_quota_exhausted(pcb.agent_id)
            response = SyscallResponse(SyscallResult.DENY, error="quota exhausted")
            self._log(request, response)
            return response

        await self._scheduler.acquire_turn(request.agent_id)
        if pcb.state in (AgentState.KILLED, AgentState.TERMINATED):
            response = SyscallResponse(SyscallResult.ERROR, error=f"agent is {pcb.state.value}")
            self._log(request, response)
            return response

        # Authoritative, race-free check: acquire_turn fully serializes
        # every concurrent caller for this same agent_id (Scheduler's
        # per-agent _turn_in_use gate), so by the time this line runs,
        # every prior same-agent call has already finished and applied its
        # own quota_used increment -- "admission-time quota check: exact
        # denial, no partial execution" now actually holds under
        # concurrency, not just for a single sequential caller.
        if pcb.quota_used >= pcb.quota_total:
            await self._kill_manager.terminate_quota_exhausted(pcb.agent_id)
            response = SyscallResponse(SyscallResult.DENY, error="quota exhausted")
            self._log(request, response)
            return response

        task = asyncio.ensure_future(self._dispatch(pcb, request))
        self._inflight[request.agent_id] = task
        try:
            value = await asyncio.wait_for(task, timeout=self._timeout)
            response = SyscallResponse(SyscallResult.OK, value=value)
        except TimeoutError:
            response = SyscallResponse(SyscallResult.ERROR, error="execution timed out; preempted")
        except asyncio.CancelledError:
            # Distinct from a timeout: this means interrupt() cancelled the
            # task directly, before the timeout ever elapsed. Contained
            # into a normal typed response here, same as every other
            # kernel decision -- a human interrupt is an observable
            # outcome, not a raw exception surprising the caller.
            response = SyscallResponse(SyscallResult.ERROR, error="interrupted (human/monitor signal)")
        except Exception as exc:  # noqa: BLE001 -- a single agent's failure must never crash the kernel
            response = SyscallResponse(SyscallResult.ERROR, error=str(exc))
        finally:
            token_result = self._last_token_result.pop(request.agent_id, None)
            if token_result is not None:
                # Real cost already charged inside _invoke_tool (it needed
                # the agent's *remaining* budget to set max_tokens before
                # the call even ran, so it owns the charge for this path).
                response.truncated = token_result.truncated
            else:
                pcb.quota_used += self._cost_fn(request)
            self._inflight.pop(request.agent_id, None)

        self._log(request, response)

        if pcb.state is AgentState.RUNNING:
            await self._scheduler.release_turn(request.agent_id, AgentState.READY)
        return response

    async def interrupt(self, agent_id: str, reason: str = "human_interrupt") -> bool:
        """Immediately cancel `agent_id`'s in-flight syscall (if any) and
        force-kill it, bypassing `execution_timeout` entirely. Returns True
        if there was an in-flight call to interrupt, False if the agent
        simply wasn't running anything at the moment this was called (the
        kill still happens either way, so a queued-but-not-yet-dispatched
        agent is also stopped). This is the primitive a human operator or
        an automated safety monitor uses to guarantee a bounded reaction
        time, independent of whatever timeout a given deployment configures
        for normal preemption."""
        task = self._inflight.get(agent_id)
        had_inflight_call = task is not None and not task.done()
        if had_inflight_call:
            task.cancel()
        await self._kill_manager.kill(agent_id, reason=reason)
        return had_inflight_call

    async def _dispatch(self, pcb: AgentPCB, request: SyscallRequest) -> Any:
        match request.syscall_type:
            case SyscallType.TOOL_CALL:
                return await self._handle_tool_call(pcb, request)
            case SyscallType.MEMORY_READ:
                return await self._handle_memory_read(pcb, request)
            case SyscallType.MEMORY_WRITE:
                return self._handle_memory_write(pcb, request)
            case SyscallType.SPAWN:
                return self._handle_spawn(pcb, request)
            case SyscallType.YIELD:
                return None
            case _:
                raise ValueError(f"unknown syscall type: {request.syscall_type}")

    async def _handle_tool_call(self, pcb: AgentPCB, request: SyscallRequest) -> Any:
        resource_key = request.arguments.get("resource_key")
        if resource_key:
            await self._resources.acquire(pcb.agent_id, resource_key)
            try:
                return await self._invoke_tool(pcb, request)
            finally:
                self._resources.release(pcb.agent_id, resource_key)
        return await self._invoke_tool(pcb, request)

    async def _invoke_tool(self, pcb: AgentPCB, request: SyscallRequest) -> Any:
        if "prompt" in request.arguments:
            prompt = request.arguments["prompt"]
            if self._token_aware_quota and hasattr(self._provider, "complete_metered"):
                remaining = max(1, pcb.quota_total - pcb.quota_used)
                result = await self._provider.complete_metered(prompt, max_tokens=remaining)
                pcb.quota_used += result.input_tokens + result.output_tokens
                self._last_token_result[pcb.agent_id] = result
                return result.text
            return await self._provider.complete(prompt)
        tool_fn = request.arguments.get("fn")
        if tool_fn is not None:
            return await tool_fn(**request.arguments.get("kwargs", {}))
        return None

    async def _handle_memory_read(self, pcb: AgentPCB, request: SyscallRequest) -> Any:
        if request.arguments.get("scope") == "kb":
            key = request.arguments["key"]
            want_verify = bool(request.arguments.get("verify")) and self._kb_verifier is not None
            if not want_verify:
                return self._memory.kb_read(key)

            entry = self._memory.kb_read_entry(key)
            if entry is None:
                return None
            result = self._kb_verifier(entry["value"], entry["evidence"])
            if inspect.isawaitable(result):
                result = await result
            if not result.verified and self._kb_verification_strict:
                from sentari.memory.hallucination import HallucinationSuspectedError

                raise HallucinationSuspectedError(
                    f"kb entry '{key}' failed verification: {result.reason}"
                )
            return {
                "value": entry["value"],
                "writer_agent_id": entry["writer_agent_id"],
                "verified": result.verified,
                "reason": result.reason,
            }
        return self._memory.read_context(pcb.agent_id, pcb.agent_id)

    def _handle_memory_write(self, pcb: AgentPCB, request: SyscallRequest) -> Any:
        if request.arguments.get("scope") == "kb":
            self._memory.kb_write(
                pcb.agent_id,
                request.arguments["key"],
                request.arguments["value"],
                evidence=request.arguments.get("evidence"),
            )
            return None
        self._memory.write_context(
            pcb.agent_id, pcb.agent_id, request.arguments["key"], request.arguments["value"]
        )
        return None

    def _handle_spawn(self, pcb: AgentPCB, request: SyscallRequest) -> str:
        child = self._spawn_fn(pcb.agent_id, request.arguments)
        return child.agent_id

    def _log(self, request: SyscallRequest, response: SyscallResponse) -> None:
        self._log_repo.log(
            request.agent_id, request.syscall_type.value, request.arguments, response.result.value
        )
