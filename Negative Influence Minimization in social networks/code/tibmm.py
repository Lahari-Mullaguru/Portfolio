"""
This file contains the implementation of the TIBMM algorithm from the paper
"Temporal rumor blocking in online social networks: A sampling-based approach." by Manouchehri et al.
"""
import math
import os.path
import pickle
from functools import partial

import numpy as np
import util
import networkx as nx
from tqdm import tqdm
import multiprocessing as multip

def generate_temporal_rr_set(graph, root, d, m, shortest_path_lengths, shortest_path_lengths_from_m):
    """
    Generates a temporal RR set for a given node u and a given distance d
    :param graph: the graph
    :param root: the node
    :param d: the distance
    :param m: the misinformation starter
    :param shortest_path_lengths: the shortest path lengths between every pair of nodes
    :param shortest_path_lengths_from_m: the shortest path lengths between the misinformation starter and every other node
    :return: the temporal RR set
    """
    t_rr_set = []
    checked = []
    # add u to the temporal RR set
    t_rr_set.append(root)
    # add u to the queue
    queue = dict()
    depth = dict()
    depth[root] = 0
    queue[depth[root]] = [root]
    # add u to the visited nodes
    checked.append(root)
    m_added = False
    current_depth = 0
    m_added_stop_depth = None
    while len(queue) >0:#<= d:

        if len(queue[current_depth]) == 0:
            current_depth += 1
        if m_added_stop_depth:
            if current_depth > m_added_stop_depth:
                break
        if current_depth > d:
            break
        if current_depth not in queue:
            break
        # get the first node in the queue
        v = queue[current_depth].pop(0)
        if v != root:
            if shortest_path_lengths[v][root] == d:
                continue
        # get the incident nodes of v
        neighbors = graph.predecessors(v)
        for u in neighbors:
            depth[u] = depth[v] + 1
            # if the neighbor is not in the temporal RR set
            if u not in checked:
                if shortest_path_lengths_from_m[root] < shortest_path_lengths[u][root] <= d:
                    flip = np.random.random()
                    if flip <= graph[u][v]['negative']:
                        t_rr_set.append(u)
                        # if m is inserted into the temporal RR set, terminate the algorithm after
                        # queue[depth[v]] is empty
                        if u not in m:
                            m_added_stop_depth = current_depth
                        # add the neighbor to the queue if m is not added yet
                        if not m_added:
                            if depth[u] not in queue:
                                queue[depth[u]] = []
                            queue[depth[u]].append(u)
                # add the neighbor to the visited nodes
                checked.append(u)
    return t_rr_set


def generate_temporal_rr_set_parallel(i, network, d, m, shortest_path_lengths, shortest_path_lengths_from_m):
    # generate a temporal RR set
    np.random.seed(i)
    random_node = np.random.choice(list(network.nodes()))
    while random_node not in m:
        random_node = np.random.choice(list(network.nodes()))
    return generate_temporal_rr_set(network, random_node, d, m, shortest_path_lengths,
                                               shortest_path_lengths_from_m)


def node_selection(graph, k, N, m, d, shortest_path_lengths, shortest_path_lengths_from_m):
    """
    Selects k savior nodes for the given graph with maximum marginal gain

    :param graph: the graph
    :param k: the number of savior nodes
    :param N: the number of iterations
    :param m: the misinformation starter
    :param d: the distance
    :param shortest_path_lengths_from_m:
    :param shortest_path_lengths:
    :return: the selected savior nodes
    """
    theta_r = []
    T = []

    if os.path.exists(f"../outputs/tibmm/theta_r_{N}_completed.pkl"):
        print("Loading theta_r from file")
        with open(f"../outputs/tibmm/theta_r_{N}.pkl", "rb") as f:
            theta_r = pickle.load(f)
    else:
        print("Generating theta_r")
        range_N = range(int(N))
        if os.path.exists(f"../outputs/tibmm/theta_r_{N}.pkl"):
            with open(f"../outputs/tibmm/theta_r_{N}.pkl", "rb") as f:
                theta_r = pickle.load(f)
            print(f"Loaded {len(theta_r)} temporal RR sets")
            range_N = range(int(N) - len(theta_r))
        pool = multip.Pool(multip.cpu_count())
        i=0
        partial_func = partial(generate_temporal_rr_set_parallel, network=graph, d=d, m=m,
                                                  shortest_path_lengths=shortest_path_lengths,
                                                  shortest_path_lengths_from_m=shortest_path_lengths_from_m)
        for temporal_rr_set in tqdm(pool.imap_unordered(partial_func, range_N), total=N):
            i+=1
            if len(set(temporal_rr_set).intersection(set(m))) > 0:
                theta_r.append(temporal_rr_set)
            if i%300 == 0:
                # write the temporal RR sets to a pickle file
                with open(f"../outputs/tibmm/theta_r_{N}.pkl", "wb") as f:
                    pickle.dump(theta_r, f)
        # write the temporal RR sets to a pickle file
        with open(f"../outputs/tibmm/theta_r_{N}.pkl", "wb") as f:
            pickle.dump(theta_r, f)
        with open(f"../outputs/tibmm/theta_r_{N}_completed.pkl", "wb") as f:
            pickle.dump("theta_r", f)
        pool.close()
        pool.join()

    X = np.zeros((len(theta_r), nx.number_of_nodes(graph)))
    for i, theta in enumerate(theta_r):
        X[i, list(theta)] = 1

    occurrence_count = np.sum(X, axis=0)
    max_count_nodes = np.argsort(occurrence_count)[::-1]
    for node in max_count_nodes:
        if node not in m:
            max_count_node = node
            break

    # add the node to the seed set
    T.append(max_count_node)
    R_count_set = set(list(range(len(theta_r))))
    R_removed = []
    for i in range(k - 1):
        prev_T = T[-1]
        print(f"S_old_{i} = {util.get_old_labels(graph, T)}")
        r_remained = list(R_count_set - set(R_removed))
        for r_count in tqdm(r_remained, total=len(r_remained)):
            RR = theta_r[r_count]
            if prev_T in RR:
                occurrence_count[list(RR)] -= 1
                R_removed.append(r_count)
        print("occurrence_count = ", occurrence_count)
        max_count_nodes = np.argsort(occurrence_count)[::-1]
        for node in max_count_nodes:
            if node not in m and node not in T:
                max_count_node = node
                break
        if occurrence_count[max_count_node] == 0:
            print(f"occurrence_count[{max_count_node}] = 0")
            print("T = ", T)
            print(f"S_old_labels =  {util.get_old_labels(graph, T)}")
            break
        T.append(max_count_node)
        print("max_occurrence_count = ", occurrence_count[max_count_node])
        print("S = ", T)
        print(f"S_old_labels = {util.get_old_labels(graph, T)}")
    T_old_labels = util.get_old_labels(graph, T)
    return T_old_labels



def compute_rho(X, n, m):
    """
    Computes the probability of a node being in the temporal RR set overlaps with m and the savior nodes
    :param X: the indicator vector X
    :param n: the number of nodes in the graph
    :param m: the misinformation starter
    :return: the probability of a node being in the temporal RR set overlaps with m and the savior nodes
    """
    return sum(X) / (n - len(m))


def compute_martingale(X, rho, i):
    """
    Computes the martingale
    :param X: the indicator vector X
    :param rho: the probability of a node being in the temporal RR set overlaps with m and the savior nodes
    :param i: the index of the node
    :return: the martingale
    """
    return sum([X[j] - rho for j in X[:i]])


def lower_bound_estimation(graph, k, m):
    """
    Estimates the lower bound of the maximum expected number of nodes that can be reached by the savior nodes
    :param graph: the graph
    :param k: the number of savior nodes
    :param m: the misinformation starter
    :return: the lower bound of the maximum expected number of nodes that can be reached by the savior nodes
    """
    q = []
    for node in m:
        q.extend(graph.neighbors(node))
    # remove duplicates from q
    q = list(set(q))
    # remove from q the nodes that are in m
    q = list(set(q).difference(set(m)))

    # compute the influence from m to each node in q
    score = dict()
    for v in q:
        score[v] = 1
        for node in m:
            if v in graph[node]:
                score[v] *= 1 - graph[node][v]['negative']
        score[v] = 1 - score[v]
    # sort the nodes in q in descending order of their influence from m
    q = sorted(q, key=lambda x: score[x], reverse=True)
    # select the first k nodes in q as the savior nodes
    T = q[:k]
    # compute the lower bound of the maximum expected number of nodes that can be reached by the savior nodes
    lb = sum(score[v] for v in T)
    return lb


def tibmm(network, misinfo_starter, k, epsilon = 0.2):
    """
    Implements the Temporal Influence Blocking Misinformation Mitigation (TIBMM) algorithm
    :param network: the network
    :param misinfo_starter: the set of nodes that start the misinformation
    :param k: number of truth spreading nodes
    :return: k truth spreading nodes
    """
    network = nx.convert_node_labels_to_integers(network, label_attribute="old_label")
    misinfo_starter = util.get_new_labels(network, misinfo_starter)
    print("Computing shortest path lengths...")
    # get the shortest distance between every pair of nodes
    shortest_path_lengths = dict(nx.shortest_path_length(network))
    print("Computing shortest path lengths between all pairs of nodes in the network...")
    shortest_path_lengths_from_m = util.compute_proximity(network, misinfo_starter)
    d = max(shortest_path_lengths_from_m.values())
    print(f"d={d}")

    n = network.number_of_nodes()
    l = 1
    k_prime = len(misinfo_starter)
    delta_sum_max = 1 / ((n - k_prime) ** l)
    delta_1 = 1 / (2 * (n - k_prime) ** l)
    delta_2 = 1 / (2 * (n - k_prime) ** l)
    alpha = math.sqrt(l * math.log(n - k_prime) + math.log(2))
    beta = math.sqrt(
        (1 - (1 / math.e)) * (math.log(math.comb(n - k_prime, k)) + l * math.log(n - k_prime) + math.log(2)))

    print("epsilon: ", epsilon)
    epsilon_1 = epsilon * alpha / ((1 - (1 / math.e)) * alpha + beta)
    print("parameter values: ", "n: ", n, "l: ", l, "k: ", k, "k_prime: ", k_prime, "delta_sum_max: ", delta_sum_max,
          "delta_1: ", delta_1, "delta_2: ", delta_2, "alpha: ", alpha, "beta: ", beta, "epsilon: ", epsilon,
          "epsilon_1: ", epsilon_1)
    print("Computing the lower bound ")
    opt = lower_bound_estimation(network, k, misinfo_starter)
    print("Lower bound : ", opt)
    # N1 = 2(n-k_prime)*math.log(1/delta_1)/(opt*epsilon_1**2)
    # N2 = 2(1-(1/math.e))*(n-k_prime)*math.log(math.comb((n-k_prime), k)/delta_2)/(opt*(epsilon-(1-(1/math.e))*epsilon_1)**2)
    # N = max(N1, N2)
    N = 2 * (n - k_prime) * ((1 - (1 / math.e)) * alpha + beta) ** 2 / (opt * epsilon ** 2)
    N = math.ceil(N)
    print("N: ", N)
    print("--------------------------------------------")
    T = node_selection(network, k, N, misinfo_starter, d, shortest_path_lengths, shortest_path_lengths_from_m)
    return T

