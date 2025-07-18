import pytest
import networkx as nx
from primalstep.utils.graph_helpers import validate_dag

class TestGraphHelpers:
    def test_validate_dag_valid(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        assert validate_dag(graph) is True

    def test_validate_dag_cyclic(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        with pytest.raises(ValueError, match="检测到循环依赖"):
            validate_dag(graph)

    def test_validate_dag_empty_graph(self):
        graph = nx.DiGraph()
        assert validate_dag(graph) is True

    def test_validate_dag_single_node(self):
        graph = nx.DiGraph()
        graph.add_node('A')
        assert validate_dag(graph) is True

    def test_validate_dag_disconnected_graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(['A', 'B', 'C'])
        graph.add_edge('A', 'B')
        assert validate_dag(graph) is True