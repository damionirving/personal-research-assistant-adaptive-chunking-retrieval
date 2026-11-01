from threading import Lock

from pra_core.graphstores.base import GraphEdge, GraphNode, GraphStore


class LocalGraphStore(GraphStore):
    """In-memory graph store for local development and tests."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._lock = Lock()

    def add_node(self, node: GraphNode) -> None:
        with self._lock:
            self._nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        with self._lock:
            self._edges.append(edge)

    def neighbors(self, node_id: str) -> list[GraphNode]:
        with self._lock:
            neighbor_ids = [
                edge.target_id for edge in self._edges if edge.source_id == node_id
            ]
            return [
                self._nodes[neighbor_id]
                for neighbor_id in neighbor_ids
                if neighbor_id in self._nodes
            ]

