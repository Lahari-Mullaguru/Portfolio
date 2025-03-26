"""
This file contains the individual fairness baseline methods.
"""
from functools import partial
import networkx as nx
import numpy as np
import timeit
import multiprocessing
import diffusionmodels as dm
import util
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)





def prob_est_parallel_sim(g, M, T, num_simulations):
    """Estimate the probability of not being influenced for each node in the graph using
    Monte Carlo simulation
    :param g: networkx graph
    :param M: set of nodes
    :param T: set of nodes
    :param num_simulations: number of simulations
    :return: list of probabilities
    """
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    hits = np.zeros(g.number_of_nodes())
    prob_est_partial_func = partial(dm.coicm, g, M, T)
    for misinfo_impressed in tqdm(pool.imap_unordered(prob_est_partial_func, list(range(num_simulations))),
                                        total=num_simulations):
        for node in misinfo_impressed:
            hits[node] += 1

    pool.close()
    pool.join()
    mp = hits / num_simulations
    sp = 1 - mp
    return sp

def prob_est_parallel(g, M, T, num_simulations, t):
    """Estimate the probability of not being influenced for each node in the graph using
    Monte Carlo simulation
    :param g: networkx graph
    :param M: set of nodes
    :param T: set of nodes
    :param num_simulations: number of simulations
    :return: list of probabilities
    """
    temp_T = T[:]
    temp_T.append(t)
    hits = np.zeros(g.number_of_nodes())
    for i in range(num_simulations):
        misinfo_impressed = dm.coicm(g, M, temp_T, i)
        for node in misinfo_impressed:
            hits[node] += 1

    mp = hits / num_simulations
    sp = 1 - mp
    return t, sp




def greedy_maximin(g, M, k):
    """Greedy algorithm for finding k nodes with the minimum probability of being saved
    :param g: networkx graph
    :param M: set of misinformation nodes
    :param k: number of nodes to select
    :return: list of truth spreading nodes
    """
    g = nx.convert_node_labels_to_integers(g, label_attribute="old_label")
    M = util.get_new_labels(g, M)
    T = []
    proximity = util.compute_proximity(g, M)
    vulnerable = [node for node in g.nodes() if node in proximity and proximity[node] > 0]
    for _ in tqdm(range(k)):
        candidate_nodes = list(set(g.nodes()) - set(M) - set(T))
        next_min, min_counter, avg_prob = prob_est_computation(g, M, T, candidate_nodes, vulnerable)
        max_min_value = max(next_min[candidate_nodes])
        print(f"max_min_value: {max_min_value}")
        nodes_with_maximin = np.where(next_min[candidate_nodes] == max_min_value)[0]
        nodes_with_maximin = np.array(candidate_nodes)[nodes_with_maximin]
        # if there are multiple nodes with the same minimum probability,
        # choose the one with the smallest number of nodes with the minimum probability
        if len(nodes_with_maximin) > 1:
            print(f"nodes with the same minimum probability {max_min_value}: {len(nodes_with_maximin)}, {nodes_with_maximin}")
            min_min_count = min(min_counter[nodes_with_maximin])
            min_min_counter_nodes = np.where(min_counter[nodes_with_maximin] == min_min_count)[0]
            min_min_counter_nodes = nodes_with_maximin[min_min_counter_nodes]
            if len(min_min_counter_nodes) > 1:
                print(f"nodes with the same minimum probability and min counter {min_min_count}: {len(min_min_counter_nodes)}, {min_min_counter_nodes}")
                max_avg_prob = max(avg_prob[min_min_counter_nodes])
                max_avg_prob_nodes = np.where(avg_prob[min_min_counter_nodes] == max_avg_prob)[0]
                print(f"nodes with the same minimum probability, min counter, and max avg prob {max_avg_prob}: {len(max_avg_prob_nodes)}, {max_avg_prob_nodes}")
                max_avg_prob_nodes = nodes_with_maximin[max_avg_prob_nodes]
                if len(max_avg_prob_nodes) > 1:
                    #pick the node with max out degree
                    max_out_degree = max([g.out_degree(node) for node in max_avg_prob_nodes])
                    max_out_degree_nodes = [node for node in max_avg_prob_nodes if g.out_degree(node) == max_out_degree]
                    print(f"outdegree {max_out_degree}: {len(max_out_degree_nodes)}, {max_out_degree_nodes}")
                    v = max_out_degree_nodes[0]
                else:
                    v = max_avg_prob_nodes[0]
            else:
                v = min_min_counter_nodes[0]
        else:
            v = nodes_with_maximin[0]
        T.append(v)
        print("#####value of min is {}".format(max_min_value))
        print("#####Greedy Welfare NIM T is {}".format(util.get_old_labels(g, T)))
    T = util.get_old_labels(g, T)
    return T




def prob_est_computation(g, M, T, candidate_nodes, vulnerable):
    next_min = np.zeros(g.number_of_nodes())
    min_counter = np.zeros(g.number_of_nodes())
    avg_prob = np.zeros(g.number_of_nodes())
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    prob_est_partial_func = partial(prob_est_parallel, g, M, T, 100)
    for node, prob in tqdm(pool.imap_unordered(prob_est_partial_func, candidate_nodes),
                                  total=len(candidate_nodes)):
        next_min[node] = np.min(prob[vulnerable])
        min_counter[node] = np.count_nonzero(prob[vulnerable] == next_min[node])
        avg_prob[node] = np.mean(prob[vulnerable])
    pool.close()
    pool.join()
    return next_min, min_counter, avg_prob


def myopic_maximin(g, M, k):
    """Myopic algorithm for finding k nodes with the minimum probability of being influenced
    :param g: networkx graph
    :param M: set of misinformation nodes
    :param k: number of nodes to select
    :return: list of nodes to spread truth
    """
    g = nx.convert_node_labels_to_integers(g, label_attribute="old_label")
    M = util.get_new_labels(g, M)
    T = []
    start = timeit.default_timer()
    while len(T) < k:
        prob = prob_est_parallel_sim(g, M, T, 100)
        candidates = np.argsort(prob)
        # add to T the node with the smallest prob and not in M
        for v in candidates:
            if v not in M and v not in T:
                T.append(v)
                break
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    T = util.get_old_labels(g, T)
    return T
