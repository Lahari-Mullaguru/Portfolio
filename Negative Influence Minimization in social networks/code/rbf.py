"""
This file contains the implementation of the Rumor Aware Betweenness First (RBF) algorithm and the Conformity-Aware Degree Demand (CDD) algorithm
from the paper "Influence minimization with conformity-aware competitions" by Gong et al.
"""
import math
import os.path
import pickle

import util
import networkx as nx

def rbf(graph, m, k_t):
    """
    Get truth spreader nodes using Rumor Aware Betweenness First (RBF) algorithm
    :param graph: the networkx graph
    :param m: misinformation spreaders
    :param k_t: number of nodes to return
    :return: k truth spreader nodes
    """
    rbv_nodes = list(set(list(graph.nodes())) - set(m))
    rbv = dict.fromkeys(rbv_nodes, 0)
    for node in m:
        pred = dict.fromkeys(graph.nodes(), [])
        dist = dict.fromkeys(graph.nodes(), None)
        beta = dict.fromkeys(graph.nodes(), 0)
        dist[node] = 0
        beta[node] = 1
        queue = []
        stack = []
        queue.append(node)
        while queue:
            v = queue.pop(0)
            stack.append(v)
            # temp = all out-neighbours of v except m
            temp = list(set(list(graph.neighbors(v))) - set(m))
            for w in temp:
                if dist[w] is None:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    beta[w] += beta[v]
                    pred[w].append(v)

        delta = dict.fromkeys(graph.nodes(), 0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (1 + delta[w]) * (beta[v] / beta[w])
            if w != node:
                rbv[w] += delta[w]
    constant = 1 / (len(m) * (len(graph.nodes()) - len(m) - 1))
    rbv.update((x, y * constant) for x, y in rbv.items())
    # get the keys with the top k values
    rbv_sorted = sorted(rbv.items(), key=lambda x: x[1], reverse=True)
    # return the top k_t nodes
    rbv_top_k = [x[0] for x in rbv_sorted[:k_t]]
    return rbv_top_k


def cdd(graph, m, k):
    """
    Get truth spreader nodes using Conformity-Aware Degree Demand (CDD) algorithm
    :param graph: the networkx graph
    :param m: the misinformation spreaders
    :param k: number of nodes to return
    :return: k truth spreader nodes
    """
    alpha = 1
    T = []
    pe = dict.fromkeys(graph.nodes(), 0)  # protection effect
    n_out_nodes = []
    for node in m:
        n_out_nodes.extend(graph.neighbors(node))
    # remove duplicates from n_out_nodes
    n_out_nodes = list(set(n_out_nodes))
    # remove from q the nodes that are in m
    n_out_nodes = list(set(n_out_nodes).difference(set(m)))
    DM = dict()
    for i in n_out_nodes:
        predecessors = nx.predecessor(graph, i)
        number_of_predecessors = len(predecessors)
        n_not_in_m_list = [x for x in predecessors if x not in m]
        n_not_in_m = len(n_not_in_m_list)
        n_in_m = number_of_predecessors - n_not_in_m
        DM[i] = n_in_m / alpha
        if n_not_in_m >= DM[i]:
            for node in n_not_in_m_list:
                pe[node] += 1
    while len(T) < k:
        # get the node with the highest protection effect
        for node in m:
            pe[node] = -math.inf
        pe_sorted = sorted(pe.items(), key=lambda x: x[1], reverse=True)
        t = pe_sorted[0][0]
        T.append(t)
        pe[t] = -math.inf
        if t in n_out_nodes:
            n_not_in_m_list = [x for x in nx.predecessor(graph, t) if x not in m]
            for node in n_not_in_m_list:
                pe[node] -= 1

        overlap = list(set(nx.neighbors(graph, t)).intersection(set(n_out_nodes)))
        for v in overlap:
            DM[v] -= 1
            if DM[v] <= 0:
                n_not_in_m_list = [x for x in nx.predecessor(graph, v) if x not in m]
                for node in n_not_in_m_list:
                    pe[node] -= 1
    return T

