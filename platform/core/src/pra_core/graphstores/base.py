from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    properties: dict[str, str]


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    relationship: str
    properties: dict[str, str]


class GraphStore(ABC):
    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None:
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, node_id: str) -> list[GraphNode]:
        raise NotImplementedError

