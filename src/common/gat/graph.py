from typing import Tuple

import pydotplus
import numpy as np
import torch
from torch import Tensor


class Graph:
    n_feature = 4  # {compute_size, program_size, result_size, memory_size}
    node_feature_lower_bound = np.array([11724524452.0, 271128.0, 0, 13017736.0])  # 上下界是各个维度数据 q=75 和 q=25 的数据
    node_feature_upper_bound = np.array([368293445632.0, 789278.0, 70656.0, 39183392.0])
    node_feature_span = node_feature_upper_bound - node_feature_lower_bound

    def __init__(self):
        self.is_normalized = False
        """ 是否被归一化 """

        self.node_features = np.array((0, self.n_feature))
        """ 节点矩阵 (N, n_feature) 
        n_feature={compute_size, program_size, result_size, memory_size}
        """

        self.adj = None
        """ 邻接矩阵 (N, N) """

    def normalize(self):
        if not self.is_normalized:
            self.node_features = (self.node_features - self.node_feature_lower_bound) / self.node_feature_span - 0.5
            self.is_normalized = True

    def inverse_normalize(self):
        if self.is_normalized:
            self.node_features = (self.node_features + 0.5) * self.node_feature_span + self.node_feature_lower_bound
            self.is_normalized = False

    def to_tensor(self) -> 'Tuple[Tensor, Tensor]':
        self.normalize()
        x = torch.tensor(self.node_features, dtype=torch.float32)
        adj = torch.tensor(self.adj, dtype=torch.float32)
        return x, adj

    @staticmethod
    def from_tensor(x: "Tensor", adj: "Tensor") -> 'Graph':
        graph = Graph()
        graph.node_features = x.numpy()
        graph.adj = adj.numpy()
        graph.is_normalized = True
        graph.inverse_normalize()
        return graph

    @staticmethod
    def from_gv_file(file_path: str) -> 'Graph':
        graph = Graph()
        gv_document: pydotplus.Dot = pydotplus.graphviz.graph_from_dot_file(file_path)
        gv_nodes = gv_document.get_nodes()
        gv_edges = gv_document.get_edges()

        nodes = []
        for gv_node in gv_nodes:
            gv_attributes: dict = gv_node.get_attributes()

            compute_size = int(gv_attributes['compute_size'].strip('"'))
            program_size = int(gv_attributes['trans_size'].strip('"'))
            result_size = int(gv_attributes['result_size'].strip('"'))
            memory_size = int(gv_attributes['ram'].strip('"'))

            node = [compute_size, program_size, result_size, memory_size]
            nodes.append(node)

        N = len(gv_nodes)
        graph.node_features = np.array(nodes, dtype=np.float32)
        graph.adj = np.zeros((N, N), dtype=np.int32)

        for gv_edge in gv_edges:
            gv_src_node_name = int(gv_edge.get_source()) - 1
            gv_dst_node_name = int(gv_edge.get_destination()) - 1

            graph.adj[gv_src_node_name][gv_dst_node_name] = 1

        return graph
