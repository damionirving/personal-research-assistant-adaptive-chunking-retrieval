from pra_core.graphstores.base import GraphEdge, GraphNode, GraphStore


class NeptuneGraphStore(GraphStore):
    """Placeholder for future Amazon Neptune integration.

    Phase 5 defines the boundary only. No network client or production Neptune
    implementation is included.
    """

    def add_node(self, node: GraphNode) -> None:
        raise NotImplementedError("Neptune integration is deferred.")

    def add_edge(self, edge: GraphEdge) -> None:
        raise NotImplementedError("Neptune integration is deferred.")

    def neighbors(self, node_id: str) -> list[GraphNode]:
        raise NotImplementedError("Neptune integration is deferred.")

