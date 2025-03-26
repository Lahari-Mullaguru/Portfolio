"""
This file contains the implementation of the proximity based methods.
"""
import welfare_nim as wibm
import timeit
from tqdm import tqdm
import numpy as np
import networkx as nx
import util
import diffusionmodels as dm


def distance_computation(network, M):
    A = nx.to_numpy_array(network, weight=1, nodelist=list(range(len(network.nodes))))
    n = len(A)
    for i in range(n):
        for m in M:
            A[i][m] = 0
    D = np.zeros((n, n))
    A = np.array(A)
    np.fill_diagonal(A, 0)
    Aq = A[:]
    start = timeit.default_timer()
    nonseed_nodes = list(set(network.nodes()) - set(M))
    all_found = False
    i = 0
    while not all_found:
        if i != 0:
            Aq = np.matmul(Aq, A)
        np.fill_diagonal(Aq, 0)
        Pos = np.where(Aq > 0, 1, 0)
        D = np.where(D >= Pos, D, i + 1)
        del Pos
        D_M = np.sum(D[M, :], axis=0)
        i += 1
        if np.all(D_M[nonseed_nodes] != 0):
            all_found = True
    max_proximity = i
    # D[i,j] is the distance at which i infects j
    # D[i,j] = 0 if i cannot reach j
    D = np.where(D > 0, D, np.inf)
    np.fill_diagonal(D, 0)
    stop = timeit.default_timer()
    print(f"Time taken to compute distance matrix = {stop - start}")
    # find minimum distance from M to all other nodes
    D_m = D[M, :]
    D_m = np.min(D_m, axis=0)
    vulnerable = np.where(D_m != np.inf)[0]  # set of nodes that are vulnerable to the negative seeds
    vulnerable = list(set(vulnerable) - set(M))
    D_m[M] = 0
    C_m = np.sum(D[M] == D_m, axis=0)
    C_m = C_m[vulnerable]
    D_m = D_m[vulnerable]
    D_from_candidates = D[:, vulnerable]
    sp_mat = np.where(D_from_candidates <= D_m, 1, 0)
    return D_from_candidates, D_m, C_m, max_proximity, sp_mat, vulnerable



def prob_est(g, M, T, num_simulations=100):
    hits = np.zeros(g.number_of_nodes())
    for i in tqdm(range(num_simulations)):
        misinfo_impressed = dm.coicm(g, M, T, i)
        for node in misinfo_impressed:
            hits[node] += 1
    mp = hits / num_simulations
    sp = 1 - mp
    return sp



def myopic_sim_distance_method(network, M, k_p, epsilons = 0.05, num_simulations = 100):
    network = nx.convert_node_labels_to_integers(network, label_attribute="old_label")
    M = util.get_new_labels(network, M)
    D_from_candidates, D_m, _, max_proximity, sp_mat, vulnerable = distance_computation(network, M)
    T=[]
    D_from_T = np.ones(len(vulnerable)) * np.inf
    candidate_nodes = list(set(network.nodes()) - set(T) - set(M))
    remaining_k = k_p - len(T)
    for _ in tqdm(range(remaining_k)):
        look_for_more = False
        save_probability = wibm.prob_est_parallel_sim(network, M, T, num_simulations)
        save_probability = save_probability[vulnerable]
        min_sp = np.min(save_probability)
        nodes_with_min_save_prob = np.where(save_probability == min_sp)[0]
        D_from_T_candidates = D_from_candidates[:, nodes_with_min_save_prob]
        save_count = np.where(D_from_T_candidates <= D_m[nodes_with_min_save_prob], 1, 0)
        save_count = np.sum(save_count, axis=1)
        # pick the node that saves the maximum number of most vulnerable nodes
        max_save_count = np.max(save_count[candidate_nodes])
        nodes_with_max_save_count = np.where(save_count[candidate_nodes] == max_save_count)[0]
        new_candidates = np.array(candidate_nodes)[nodes_with_max_save_count]


        if max_save_count == len(nodes_with_min_save_prob):
           look_for_more = True

        if look_for_more:
            most_vulnerable = np.where(save_probability <= min_sp + epsilons)[0]
            D_from_T_candidates = D_from_candidates[:, most_vulnerable]
            save_count = np.where(D_from_T_candidates <= D_m[most_vulnerable], 1, 0)
            save_count = np.sum(save_count, axis=1)
            # pick the node that saves the maximum number of most vulnerable nodes
            max_save_count = np.max(save_count[new_candidates])
            nodes_with_max_save_count = np.where(save_count[new_candidates] == max_save_count)[0]
            new_candidates = np.array(new_candidates)[nodes_with_max_save_count]
        if len(new_candidates) > 1:
            # pick the nodes that save more in total to increase the number of nodes benefitted.
            D_from_T_candidates_copy = np.minimum(D_from_candidates, D_from_T)
            save_count = np.where(D_from_T_candidates_copy <= D_m, 1, 0)
            save_count = np.sum(save_count, axis=1)
            max_save_count = np.max(save_count[new_candidates])
            nodes_with_max_save_count = np.where(save_count[new_candidates] == max_save_count)[0]
            new_candidates = np.array(new_candidates)[nodes_with_max_save_count]
        if len(new_candidates) > 1:
            # pick the node with the maximum average utility
            D_from_T_candidates_for_avg = np.where(D_from_T_candidates_copy == np.inf, 0, D_from_T_candidates_copy)
            avg_distance = np.mean(D_from_T_candidates_for_avg, axis=1)
            nodes_with_min_avg_distance = np.where(avg_distance[new_candidates] == np.min(avg_distance[new_candidates]))[0]
            new_candidates = np.array(new_candidates)[nodes_with_min_avg_distance]
        if len(new_candidates) > 1:
            #pick the node with the highest degree
            degrees = np.array([network.degree[node] for node in new_candidates])
            max_degree = np.max(degrees)
            nodes_with_max_degree = np.where(degrees == max_degree)[0]
            new_candidates = np.array(new_candidates)[nodes_with_max_degree]
        t = new_candidates[0]
        T.append(t)
        print(f"min_sp = {min_sp}, T[{len(T)}] = {t} => {util.get_old_labels(network, [t])}")
        D_from_T = np.minimum(D_from_candidates[t], D_from_T)
        candidate_nodes.remove(t)

    T_old_labels = util.get_old_labels(network, T)
    return T_old_labels





def myopic_distance_count_method(network, M, k_p):
    network = nx.convert_node_labels_to_integers(network, label_attribute="old_label")
    M = util.get_new_labels(network, M)
    D_from_candidates, D_m, C_m, max_proximity, sp_mat, vulnerable = distance_computation(network, M)
    T = []
    epsilons = 0.05
    D_from_T = np.ones(len(vulnerable)) * np.inf
    saved_distance = -1
    current_vulnerable = []
    all_saved = False
    C_T = np.sum(D_from_candidates[T] <= D_m, axis=0)
    while len(T) < k_p:
        look_for_more = False
        candidate_nodes = list(set(network.nodes()) - set(T) - set(M))
        if len(current_vulnerable) == 0:
            saved_distance += 1
            print("saved_distance = ", saved_distance)
            current_vulnerable = np.where(D_m == saved_distance + 1)[0]
            # current_vulnerable has indices of vulnerable nodes at distance saved_distance + 1
            saved = np.where(C_T >= C_m)[0]
            current_vulnerable = [cv for cv in current_vulnerable if cv not in saved]
            if len(current_vulnerable) == 0:
                print("No more vulnerable nodes at distance = ", saved_distance + 1)
                if saved_distance + 1 == max_proximity:
                    print("No more vulnerable nodes")
                    all_saved = True
                    break
                continue
        current_vulnerable_C_M = C_m[current_vulnerable]
        current_vulnerablt_C_T = C_T[current_vulnerable]
        diff = (current_vulnerable_C_M - current_vulnerablt_C_T) / current_vulnerable_C_M
        max_diff = np.max(diff)
        mode = 2 # difference minimization
        if max_diff == 1:  # current_vulnerablt_C_T == 0 no T saves current_vulnerable
            mode = 1 # save
            max_diff_nodes = np.where(diff == max_diff)[0]
            diff = np.ones(len(current_vulnerable)) * np.inf * -1
            diff[max_diff_nodes] = current_vulnerable_C_M[max_diff_nodes]  # number of Ms influencing current_vulnerable
            max_diff = np.max(diff[max_diff_nodes])
        most_vulnerable_idx = np.where(diff == max_diff)[0]
        most_vulnerable = np.array(current_vulnerable)[most_vulnerable_idx]


        max_diff = max_diff
        nodes_with_max_diff = most_vulnerable
        print(np.argsort(diff)[:-10:-1])
        print(np.sort(diff)[:-10:-1])

        if len(nodes_with_max_diff) > 1:
            if mode == 2:
                save_count = np.where(D_from_candidates[:, nodes_with_max_diff] <= D_m[nodes_with_max_diff], 1, 0)
            else:
                D_from_T_candidates = np.minimum(D_from_candidates, D_from_T)
                D_from_T_candidates = D_from_T_candidates[:, nodes_with_max_diff]
                save_count = np.where(D_from_T_candidates <= D_m[nodes_with_max_diff], 1, 0)
            save_count = np.sum(save_count, axis=1)
            # pick the node that saves the maximum number of most vulnerable nodes
            max_save_count = np.max(save_count[candidate_nodes])
            nodes_with_max_save_count = np.where(save_count[candidate_nodes] == max_save_count)[0]
            new_candidates = np.array(candidate_nodes)[nodes_with_max_save_count]
            # if len(new_candidates) > 1:
            #     look_for_more = True
            if max_save_count == len(nodes_with_max_diff):
               look_for_more = True
        else:
            node_to_save = nodes_with_max_diff[0]
            saviours = np.where(sp_mat[:, node_to_save] == 1)[0]
            new_candidates = np.array([saviour for saviour in saviours if saviour in candidate_nodes])
            if len(new_candidates) > 1:
                look_for_more = True

        if look_for_more:
            if mode == 1: # current_vulnerablt_C_T == 0 no T saves current_vulnerable
                epsilon = 1
            else:
                epsilon = epsilons
            most_vulnerable = np.where(diff >= max_diff - epsilon)[0]
            most_vulnerable = np.array(current_vulnerable)[most_vulnerable]
            if mode == 2:
                save_count = np.where(D_from_candidates[:, most_vulnerable] <= D_m[most_vulnerable], 1, 0)
            else:
                D_from_T_candidates = np.minimum(D_from_candidates, D_from_T)
                D_from_T_candidates = D_from_T_candidates[:, most_vulnerable]
                save_count = np.where(D_from_T_candidates <= D_m[most_vulnerable], 1, 0)
            save_count = np.sum(save_count, axis=1)
            # pick the node that saves the maximum number of most vulnerable nodes
            max_save_count = np.max(save_count[new_candidates])
            nodes_with_max_save_count = np.where(save_count[new_candidates] == max_save_count)[0]
            new_candidates = np.array(new_candidates)[nodes_with_max_save_count]
        if len(new_candidates) > 1:
            # pick the nodes that save more in total
            #D_from_T_candidates_copy = np.minimum(D_from_candidates, D_from_T)
            save_count = np.where(D_from_candidates <= D_m, 1, 0)
            save_count = np.sum(save_count, axis=1)
            node_with_max_save_count = np.argmax(save_count[new_candidates])
            t_idx = new_candidates[node_with_max_save_count]
        else:
            t_idx = new_candidates[0]
        t = t_idx
        T.append(t)
        print(f"max_diff = {max_diff}, T[{len(T)}] = {t} => {util.get_old_labels(network, [t])}")
        D_from_T = np.minimum(D_from_candidates[t_idx], D_from_T)
        C_T = np.sum(D_from_candidates[T] <= D_m, axis=0)
        saved = np.where(C_T >= C_m)[0]
        current_vulnerable = [cv for cv in current_vulnerable if cv not in saved]
    if all_saved and len(T) < k_p:
        candidate_nodes = list(set(network.nodes()) - set(T) - set(M))
        save_count = np.where(D_from_candidates <= D_m, 1, 0)
        save_count = np.sum(save_count, axis=1)
        new_candidates = np.argsort(save_count[candidate_nodes])[::-1]
        remaining_budget = k_p - len(T)
        T.extend(new_candidates[:remaining_budget])
    T_old_labels = util.get_old_labels(network, T)
    return T_old_labels



