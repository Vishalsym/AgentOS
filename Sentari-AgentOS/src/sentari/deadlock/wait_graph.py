from __future__ import annotations

from collections import defaultdict


class WaitForGraph:
    """Directed wait-for graph: an edge waiter -> holder means `waiter` is
    blocked on a resource held by `holder` (FR-12)."""

    def __init__(self) -> None:
        self._edges: dict[str, set[str]] = defaultdict(set)

    def add_edge(self, waiter: str, holder: str) -> None:
        self._edges[waiter].add(holder)

    def remove_edges_for(self, agent_id: str) -> None:
        self._edges.pop(agent_id, None)
        for neighbors in self._edges.values():
            neighbors.discard(agent_id)

    def edges(self, node: str) -> set[str]:
        return set(self._edges.get(node, ()))

    def all_edges(self) -> list[tuple[str, str]]:
        """All (waiter, holder) pairs currently on the graph -- for display
        (e.g. a dashboard), not used by the detection algorithm itself."""
        return [(waiter, holder) for waiter, holders in self._edges.items() for holder in holders]

    def has_cycle(self, start: str) -> bool:
        visited: set[str] = set()
        stack: set[str] = set()
        return self._dfs(start, visited, stack)

    def _dfs(self, node: str, visited: set[str], stack: set[str]) -> bool:
        visited.add(node)
        stack.add(node)
        for neighbor in self.edges(node):
            if neighbor in stack:
                return True  # back-edge = cycle
            if neighbor not in visited and self._dfs(neighbor, visited, stack):
                return True
        stack.discard(node)
        return False

    def extract_cycle(self, start: str) -> list[str]:
        """Return the agent_ids forming the cycle reachable from `start`.
        Only meaningful to call after has_cycle(start) is True."""
        path: list[str] = []
        visited: set[str] = set()

        def dfs(node: str) -> list[str] | None:
            path.append(node)
            visited.add(node)
            for neighbor in self.edges(node):
                if neighbor in path:
                    return path[path.index(neighbor):]
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:
                        return result
            path.pop()
            return None

        return dfs(start) or []
