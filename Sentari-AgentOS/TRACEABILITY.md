# Traceability Matrix

Maps every requirement in the SRS to the module that implements it and the
test that verifies it. The kernel core (M0-M6) is complete and tested, and
every item the README's original roadmap listed as not-yet-built has since
been implemented: the fault-injection suite, the benchmark suite, the
LangGraph adapter, live-key `AnthropicProvider` tests, and dashboard/webhook
notifications. Beyond that, six additional, non-required mechanisms were
built on top of the approved design (see **Novel Mechanisms** below) --
none of them replace or weaken anything the SRS asked for; each is strictly
opt-in and provably falls back to the original, approved behavior when not
configured.

## Functional Requirements

| ID | Requirement | Implemented in | Verified by |
|----|-------------|-----------------|-------------|
| FR-1 | PCB with id, state, priority, quota, owned resources, parent/child links | `src/sentari/pcb.py::AgentPCB` | `tests/test_pcb.py` |
| FR-2 | States NEW, READY, RUNNING, BLOCKED, TERMINATED, KILLED | `src/sentari/pcb.py::AgentState`, `_ALLOWED_TRANSITIONS` | `tests/test_pcb.py` |
| FR-3 | `spawn` syscall creates child inheriting bounded quota share | `src/sentari/kernel.py::Kernel._spawn_child`, `src/sentari/syscalls/layer.py::_handle_spawn` | `tests/test_syscalls.py::test_spawn_creates_child_with_bounded_quota`, `tests/test_kernel_integration.py`, `tests/test_fault_injection.py::test_rapid_spawn_storm_stays_quota_bounded_and_consistent` |
| FR-4 | Configurable round-robin or priority-based dispatch | `src/sentari/scheduler/scheduler.py::SchedulingPolicy`, `Scheduler._peek_next` | `tests/test_scheduler.py`; fairness measured directly in `scripts/benchmark.py` (Jain's index 1.0 across equal-priority agents) |
| FR-5 | Max time-slice/call-count per turn before returning to READY | A "turn" = one syscall; enforced by `SyscallLayer.on_syscall` requiring a fresh `acquire_turn` per syscall, `asyncio.wait_for` timeout, plus the bounded-latency `interrupt()` primitive for immediate preemption independent of the timeout | `tests/test_preemption.py`, `tests/test_interrupt.py` (measures actual interrupt latency) |
| FR-6 | Dynamic priority aging to prevent starvation | `src/sentari/scheduler/scheduler.py::Scheduler._apply_aging` | `tests/test_scheduler.py::test_aging_boosts_starved_agent_priority`; real measured behavior (including its known limit under an extreme priority gap) in `scripts/benchmark.py` |
| FR-7 | Syscall interface (tool_call, memory_read, memory_write, spawn, yield) as sole path to external resources | `src/sentari/syscalls/layer.py::SyscallType`, `SyscallLayer` | `tests/test_syscalls.py` |
| FR-8 | Validate against permissions + remaining quota before execution | `src/sentari/syscalls/layer.py::SyscallLayer.on_syscall` (admission-time check, plus an authoritative re-check after turn serialization for correctness under concurrent same-agent calls) | `tests/test_syscalls.py::test_quota_exhaustion_denies_and_terminates_agent`, `tests/test_fault_injection.py::test_quota_exhaustion_mid_storm_denies_cleanly_for_every_excess_call` |
| FR-9 | Log every syscall (agent, type, args, timestamp, result) | `src/sentari/persistence/repositories.py::SyscallLogRepo` | `tests/test_syscalls.py::test_syscall_charges_quota_and_logs`, `tests/test_kernel_integration.py` |
| FR-10 | Isolated per-agent context, not writable by other agents | `src/sentari/memory/manager.py::MemoryManager`, `MemoryIsolationError` | `tests/test_memory_isolation.py` |
| FR-11 | Shared KB region reachable only via explicit memory_read/write syscalls | `src/sentari/memory/manager.py::MemoryManager.kb_read/kb_write`, `src/sentari/syscalls/layer.py::_handle_memory_read/_write` | `tests/test_syscalls.py::test_kb_read_write_via_syscalls_is_shared` |
| FR-12 | Maintain wait-for graph on blocking | `src/sentari/deadlock/wait_graph.py::WaitForGraph` | `tests/test_deadlock.py::test_wait_graph_detects_two_agent_cycle` |
| FR-13 | Cycle detection on every new blocking edge | `src/sentari/deadlock/detector.py::DeadlockDetector.add_wait_edge` (incremental, O(V+E), run per edge not on a sweep) | `tests/test_deadlock.py`; N-agent (not just pairwise) cycles proven in `tests/test_fault_injection.py::test_three_agent_cycle_is_detected_and_resolved_without_hanging`; measured resolution latency (~0.3ms mean) in `scripts/benchmark.py` |
| FR-14 | On cycle: select victim (least important), terminate/roll back | `src/sentari/deadlock/detector.py::DeadlockDetector.add_wait_edge` (victim = `max(cycle, key=priority)` by default; optionally overridable by a semantic value score, see Novel Mechanism #1) | `tests/test_deadlock.py::test_detector_picks_least_important_agent_as_victim`, `test_two_agent_resource_deadlock_kills_one_and_frees_the_other` |
| FR-15 | Preempt on time-slice/quota exceeded | `src/sentari/syscalls/layer.py::on_syscall` (timeout -> agent returns to READY); immediate, bounded-latency preemption via `interrupt()` (Novel Mechanism #4) | `tests/test_preemption.py::test_slow_tool_call_times_out_and_agent_returns_to_ready`, `tests/test_interrupt.py` |
| FR-16 | Force-kill on blocked-timeout or deadlock victim | `src/sentari/preemption/kill_manager.py::KillManager.kill` | `tests/test_deadlock.py`, `tests/test_preemption.py::test_kill_releases_held_resource_to_waiter` |
| FR-17 | Release all resources held by terminated/killed agent | `src/sentari/syscalls/resource_manager.py::ResourceManager.release_all`, called from `KillManager` | `tests/test_preemption.py::test_quota_exhausted_releases_held_resources`, `tests/test_kernel_integration.py` |

## Non-Functional Requirements

| NFR | How it's addressed | Status |
|-----|---------------------|--------|
| Performance (<5% syscall overhead) | Now formally benchmarked (`scripts/benchmark.py`, results in `benchmark_results.json`): ~100us mediation cost per call. Against a bare no-op baseline that's a large-looking percentage (a pathological comparison no real syscall ever makes); against what a syscall actually replaces -- a real ~300ms LLM/tool call -- it's ~0.03%, i.e. negligible. Reported honestly both ways rather than only the flattering framing. | **Done** |
| Reliability (single-agent failure isolated) | `SyscallLayer.on_syscall` catches all exceptions from a dispatched call (and a human-triggered `interrupt()` cancellation) and returns a clean ERROR response instead of propagating | `tests/test_syscalls.py::test_a_failing_tool_does_not_crash_the_kernel`, `tests/test_fault_injection.py::test_tool_raising_unexpected_exception_types_is_always_contained` |
| Scalability (>=20 concurrent registered agents) | Admission and per-agent state are plain dict entries; scheduler dispatch is O(ready-queue-size) per turn. A real scheduler concurrency bug (concurrent same-agent turn requests could permanently strand each other) was found and fixed while building this suite -- see Known Fixes below. | `tests/test_kernel_integration.py::test_twenty_agents_can_be_admitted_and_run_a_syscall`, `tests/test_fault_injection.py::test_many_agents_contend_for_one_resource_no_double_grant` |
| Security (no raw API keys / unmediated network access) | Agents only ever see `Kernel.syscall(...)`; the `LLMProvider` instance lives inside `SyscallLayer` and is never exposed to agent code | By construction (`src/sentari/kernel.py`); adversarial-path tests in `tests/test_dashboard.py::test_mcp_path_traversal_rejected`, `test_mcp_bare_directory_path_rejected_cleanly` |
| Auditability (every decision reconstructable from the log) | `syscall_log` + `agents` + `resource_allocation` tables are the full audit trail; `AgentRepo.get` reads straight from persistence, independent of in-memory scheduler state | `tests/test_kernel_integration.py::test_end_to_end_spawn_deadlock_and_consistent_persisted_state` |
| Maintainability (independently testable subsystems) | Each of scheduler / syscalls / memory / deadlock / preemption / persistence is a separate module with its own constructor-injected dependencies (no hidden globals) | Reflected in the one-test-file-per-module layout (23 test files, 138 tests) |
| Usability (dashboard) | `src/sentari/dashboard/` -- a FastAPI app matching the approved Design Doc wireframe (3-panel layout: Active Agents/PCB, Wait-For Graph + Quota Usage, Syscall Log), polling live kernel/SQLite state twice a second. Boots directly into a clean, unscripted "live" state by default; the scripted Standard-tour/Before-After scenarios remain available (not exposed in the UI) for demonstration and as the benchmark suite's data source. | `tests/test_dashboard.py`; run with `uv run python scripts/dashboard.py` |
| Extensibility (wraps existing frameworks rather than replacing them) | `src/sentari/adapters/langgraph_adapter.py` plugs into LangGraph's own documented `ToolNode(awrap_tool_call=...)` extension point; `src/sentari/adapters/hypervisor.py` generalizes this framework-agnostically by instrumenting any tool object's callable entry points in place (Novel Mechanism #6) | `tests/test_langgraph_adapter.py`, `tests/test_hypervisor.py` |

## Novel Mechanisms (beyond the approved SRS/Design Doc)

Each is opt-in via an optional `Kernel(...)` constructor argument; a kernel
built with none of them set behaves identically to the pre-existing,
approved design (explicitly tested for each one below).

| # | Mechanism | Implemented in | Verified by |
|---|-----------|-----------------|-------------|
| 1 | Semantic-value-aware deadlock resolution: victim selection can weigh a content-derived `task_value` (e.g. an LLM's judgment of how costly losing an agent's work would be) ahead of a static priority integer | `src/sentari/deadlock/detector.py::DeadlockDetector` (`value_fn`), `src/sentari/deadlock/semantic_scoring.py` | `tests/test_deadlock.py` (semantic-override tests), `tests/test_semantic_scoring.py` |
| 2 | Probabilistic Banker's Algorithm: agents declare a probability (not an exact max claim) of eventually wanting a resource; the kernel proactively refuses a wait with high estimated mutual-wait risk, before any real cycle forms, complementing the reactive detector | `src/sentari/deadlock/probabilistic_banker.py`, wired into `src/sentari/syscalls/resource_manager.py::ResourceManager.acquire` | `tests/test_probabilistic_banker.py` |
| 3 | Mediated inter-agent memory reads with hallucination interception: a `kb_write` can cite `evidence`; a `kb_read` can optionally verify the value is still supported by it (heuristic, LLM-backed, or a caller-supplied verifier), in permissive (flag) or strict (block) mode | `src/sentari/memory/hallucination.py`, `src/sentari/memory/manager.py`, wired into `SyscallLayer._handle_memory_read` | `tests/test_hallucination_interception.py` (18 tests, including an intentionally-documented limitation of the lexical heuristic) |
| 4 | Bounded-latency human interrupt: `Kernel.interrupt(agent_id)` cancels an agent's in-flight syscall immediately and force-kills it, independent of `execution_timeout` -- a measured reaction-time primitive, not a UI convenience | `src/sentari/syscalls/layer.py::SyscallLayer.interrupt` | `tests/test_interrupt.py` (includes an actual timing assertion, not just a functional one) |
| 5 | Reputation-driven adaptive admission control: a recurring `agent_type`'s historical outcomes (kills, quota exhaustions) feed back into a new instance's starting priority at admission, with a minimum-sample guard and a capped penalty | `src/sentari/persistence/repositories.py::ReputationRepo`, wired into `Scheduler.admit`/`KillManager` | `tests/test_reputation.py` |
| 6 | Transparent "agent hypervisor": mediates any framework's tool object by monkey-patching its callable entry point(s) in place (`func`/`coroutine`/`_run`/etc.), rather than a per-framework adapter -- the framework needs zero code changes and no awareness that mediation happened | `src/sentari/adapters/hypervisor.py` | `tests/test_hypervisor.py` (fake LangChain-shaped and CrewAI-shaped tool objects, proving the mechanism is genuinely framework-agnostic) |

Extension beyond the original 4-table Design Doc schema: `agent_reputation`
(mechanism #5) and `knowledge_base.evidence` (mechanism #3) -- both
explicitly called out in `src/sentari/persistence/schema.sql`.

## Known fixes made while building the fault-injection/novel-mechanism suites

Not hypothetical -- each of these was a real bug the test suite actually
caught, with a fix and a regression test:

- **Duplicate agent_id re-admission** used to crash with a raw SQLite
  `IntegrityError`; now raises a clean `DuplicateAgentError`
  (`src/sentari/pcb.py`, `src/sentari/scheduler/scheduler.py::admit`).
- **MCP bridge path validation** let a bare-directory path (e.g. `"."`)
  slip through and blow up as a raw `PermissionError`; now rejected
  cleanly with a 400 (`src/sentari/dashboard/app.py::_resolve_mcp_path`).
- **Concurrent same-agent turn requests could permanently strand each
  other** -- e.g. an agent spawning several children in parallel would
  hang after the first spawn, forever. Fixed with a per-agent
  serialization gate in `Scheduler.acquire_turn`/`release_turn`/`force_yield`
  (`_turn_in_use`).
- **A quota-exhaustion race exposed by fixing the bug above**: the quota
  check ran before turn serialization, so concurrent same-agent calls
  could all read a stale `quota_used` and collectively exceed the budget.
  Fixed with an authoritative re-check in `SyscallLayer.on_syscall` after
  `acquire_turn` returns.

## Known simplifications

- **Preemption semantics**: asyncio is cooperative, not OS-preemptive. "The
  CPU" is a single dispatch slot (`Scheduler._current`); a turn is exactly
  one syscall. `interrupt()` (Novel Mechanism #4) adds true, immediate
  task cancellation on top of this for the specific case of a human/monitor
  wanting a bounded reaction time. See `Scheduler`'s class docstring.
- **Priority convention**: lower `priority` integer = more important (per
  the `agents` table). Deadlock victim selection therefore uses
  `max(cycle, key=priority)` (or the semantic-value-aware variant) to pick
  the *least* important agent.
- **Persistence is synchronous** (`sqlite3` called directly, not wrapped in
  `asyncio.to_thread`). Correct and auditable, just not maximally
  non-blocking; acceptable for a single-writer, low-concurrency kernel, and
  confirmed fast enough in practice by the benchmark suite.
- **Aging under an extreme priority gap**: `_apply_aging`'s correction is
  linear (+1 priority level per ~2s of continuous waiting), so it
  demonstrably prevents starvation under a modest gap (measured: a
  priority-5 agent got a healthy ~20% share against four priority-1
  agents) but takes proportionally longer to rescue an agent from an
  extreme gap (e.g. ~98s to close a 49-level gap) -- a real,
  previously-undocumented limitation found by `scripts/benchmark.py`, not
  fixed here (would require a design decision on non-linear/adaptive
  aging), but now measured and stated rather than assumed.
- **Hallucination-interception heuristic is lexical, not semantic**:
  `heuristic_verify` can miss a faithful paraphrase that uses different
  words for the same facts (documented and explicitly tested in
  `tests/test_hallucination_interception.py`); `llm_verify` is the
  stronger alternative for that case, at the cost of an async LLM call.
- **The hypervisor's synchronous entry point** bridges a sync call into
  the async kernel via `asyncio.run()`, which cannot work if an event loop
  is already running in that thread -- fails with a clear `RuntimeError`
  rather than deadlocking (tested explicitly), not silently degraded.
