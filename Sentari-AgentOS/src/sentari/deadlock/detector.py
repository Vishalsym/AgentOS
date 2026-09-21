from __future__ import annotations

from collections.abc import Callable

from sentari.deadlock.wait_graph import WaitForGraph


class DeadlockDetector:
    """Incremental cycle detection on the wait-for graph, run on every new
    wait edge rather than as a periodic sweep -- O(V+E) per check, bounded
    by the number of currently-blocked agents (FR-12, FR-13, FR-14).

    Priority convention (matches the `agents` table): a lower `priority`
    value means a *more* important agent. Victim selection therefore picks
    the agent with the highest numeric priority value in the cycle -- the
    least important one -- which is what FR-14's "select victim e.g. lowest
    priority" means in practice.

    Semantic-value-aware resolution (novel mechanism #1): an optional
    `value_fn(agent_id) -> float | None` lets victim selection weigh a
    *content-derived* judgment of what's actually at stake -- e.g. "how
    costly would losing this agent's in-progress work be" -- ahead of a
    static priority integer assigned once at admission. When `value_fn` is
    None, or returns None for every agent in the cycle, this is byte-for-
    byte the original priority-only selection (max by priority) -- fully
    backward compatible. A higher value means "more important to spare";
    ties and unscored agents fall back to priority.
    """

    def __init__(
        self,
        priority_fn: Callable[[str], int],
        value_fn: Callable[[str], float | None] | None = None,
    ):
        self.graph = WaitForGraph()
        self._priority_fn = priority_fn
        self._value_fn = value_fn

    def _expendability(self, agent_id: str) -> tuple[float, int]:
        """Sort key for victim selection: higher = more expendable = more
        likely to be picked. Primary component is the semantic value (
        inverted, since a *higher* task_value means *less* expendable);
        secondary/fallback component is priority, matching the original
        max(cycle, key=priority_fn) exactly when no semantic score applies."""
        value = self._value_fn(agent_id) if self._value_fn else None
        semantic_expendability = 0.0 if value is None else -value
        return (semantic_expendability, self._priority_fn(agent_id))

    def add_wait_edge(self, waiter: str, holder: str) -> str | None:
        self.graph.add_edge(waiter, holder)
        if self.graph.has_cycle(waiter):
            cycle = self.graph.extract_cycle(waiter)
            victim = max(cycle, key=self._expendability)
            self.graph.remove_edges_for(victim)
            return victim
        return None
