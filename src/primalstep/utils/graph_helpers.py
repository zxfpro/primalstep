import networkx as nx

def validate_dag(graph: nx.DiGraph) -> bool:
    """
    验证给定的NetworkX图是否为有向无环图 (DAG)。
    如果检测到循环，抛出 ValueError。
    """
    try:
        # nx.is_directed_acyclic_graph() 如果图是DAG则返回True，否则返回False
        if not nx.is_directed_acyclic_graph(graph):
            # 如果不是DAG，尝试找到一个循环来提供更详细的错误信息
            # networkx.find_cycle 会找到一个循环并返回边列表
            cycle = nx.find_cycle(graph)
            raise ValueError(f"检测到循环依赖: {cycle}")
        return True
    except nx.NetworkXNoCycle:
        # 如果图是DAG，nx.find_cycle 会抛出 NetworkXNoCycle 异常
        return True
    except Exception as e:
        # 捕获其他可能的NetworkX异常
        raise ValueError(f"图验证失败: {e}")