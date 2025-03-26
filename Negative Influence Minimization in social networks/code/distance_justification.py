# distance_justification.py
"""
This file contains the code for experiments for justification of distance based method
"""
import multiprocessing as multip
import os.path
import pickle
import timeit
from datetime import datetime
import matplotlib.pyplot as plt
from functools import partial

import proximity_based_methods as dist_m
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import util
import diffusionmodels as dm
import seed_selection

def get_mp(method, pos, num_simulations, neg_policy):
    i = len(pos)
    if os.path.exists(f"{folder}/mp_{i}_{num_simulations}_{neg_policy}.pkl"):
        with open(f"{folder}/mp_{i}_{num_simulations}_{neg_policy}.pkl", 'rb') as f:
            mp = pickle.load(f)
    else:
        print("Running simulations...", method, datetime.now())
        start = timeit.default_timer()
        pool = multip.Pool(multip.cpu_count())
        pos_seeds_list = [pos] * num_simulations
        pos_seeds_list = [(i, pos_seeds) for i, pos_seeds in enumerate(pos_seeds_list)]
        neg_impressed_counts = []
        hits = np.zeros(network_new_labels.number_of_nodes())
        inf_partial_func = partial(util.coicm, network_new_labels, neg_seeds_new_labels)
        for neg_impressed_nodes in pool.imap_unordered(inf_partial_func, pos_seeds_list):
            hits[neg_impressed_nodes] += 1
            neg_impressed_counts.append(len(neg_impressed_nodes))
        mp = hits / num_simulations
        pool.close()
        pool.join()
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        with open(f"{folder}/mp_{i}_{num_simulations}_{neg_policy}.pkl", 'wb') as f:
            pickle.dump(mp, f)
    return mp

# main method
if __name__ == '__main__':
    dataset = "email"
    k_n = 10
    k_p = 25
    imp_prob = 0.3
    neg_policy = "degree"
    weight = "value"
    print(f"dataset: {dataset}, k_n: {k_n}, k_p: {k_p}, imp_prob: {imp_prob}, neg_policy: {neg_policy}, weight: {weight}")
    
    # Initialize the graph and negative seeds
    network, neg_seeds, network_new_labels, neg_seeds_new_labels = util.init_settings(dataset, weight, imp_prob, neg_policy, k_n)

    folder = f"../outputs/{dataset}_{weight}_weight_{neg_policy}_neg_policy_{imp_prob}"
    welfare_output_folder = f"{folder}/figs/"

    print(dataset, " dataset loaded the graph successfully! There are ", network.number_of_nodes(), " nodes and ", network.number_of_edges(), " edges")
    print("Negative seed set is", neg_seeds)

    # Ensure the positive seeds are correctly selected using the new algorithms
    method = "ga"  # or "community" or any other method you want to test
    if os.path.exists(f"{folder}/pos_seeds_{method}.txt"):
        pos_seeds = util.read_pos_seeds(network, method, k_p, folder)
        pos_seeds = util.get_new_labels(network_new_labels, pos_seeds)
        print("Positive seed set is", pos_seeds)
    else:
        pos_seeds = seed_selection.select_seeds(network, method, k_p, neg_seeds, imp_prob)
        pos_seed_file = f"{folder}/pos_seeds_{method}.txt"
        with open(pos_seed_file, 'w') as f:
            for seed in pos_seeds:
                f.write("%s\n" % str(seed))

    num_simulations = 1000
    if os.path.exists(f"{folder}/vul_{num_simulations}_{neg_policy}.pkl"):
        with open(f"{folder}/vul_{num_simulations}_{neg_policy}.pkl", 'rb') as f:
            vul = pickle.load(f)
    else:
        pool = multip.Pool(multip.cpu_count())
        hits = np.zeros(network.number_of_nodes())
        inf_vuln_partial_func = partial(dm.icm, network_new_labels, neg_seeds_new_labels)
        for misinfo_impressed in tqdm(pool.imap_unordered(inf_vuln_partial_func, list(range(num_simulations))), total=num_simulations):
            for node in misinfo_impressed:
                hits[node] += 1
        pool.close()
        pool.join()
        vul = hits / num_simulations
        with open(f"{folder}/vul_{num_simulations}_{neg_policy}.pkl", 'wb') as f:
            pickle.dump(vul, f)
    
    initial_vul = vul
    mp = vul
    D_from_candidates, D_m, C_m, max_proximity, sp_mat, vulnerable = dist_m.distance_computation(network_new_labels, neg_seeds_new_labels)
    for i in range(1):
        vul = mp
        next_positive_seed = pos_seeds[i]
        closer_nodes = np.where(D_from_candidates[next_positive_seed] <= D_m)[0]
        closer_nodes = np.array(list(set(closer_nodes) - set(neg_seeds_new_labels)))
        vul_closer_nodes = vul[closer_nodes]
        vul_rank_closer_nodes = np.argsort(vul_closer_nodes)[::-1]
        mp = get_mp(method, [next_positive_seed], num_simulations, neg_policy)
        mp_closer_nodes = mp[closer_nodes]

        plt.figure(figsize=(8, 6))
        plt.rcParams['lines.markersize'] = 1
        plt.rcParams.update({'font.size': 16})
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        plt.plot(vul_closer_nodes[vul_rank_closer_nodes], 'r.')
        plt.plot(mp_closer_nodes[vul_rank_closer_nodes], 'b.')
        plt.xlabel("Node", fontsize=16)
        plt.ylabel("Probability", fontsize=16)
        plt.legend(["Vulnerability", "mp"], loc='upper center', bbox_to_anchor=(0, 1, 1, 0.1), ncols=2)
        plt.savefig(f"{welfare_output_folder}mp_vul_{method}_{i + 1}.pdf")

        corr = []
        avg_mp_closer_new = []
        avg_mp_closer_old = []
        total_avg_mp = []
        max_mp = []
        closer_nodes_count = []
        non_neg_seed_nodes = np.array(list(set(range(network_new_labels.number_of_nodes())) - set(neg_seeds_new_labels)))
        for k in [5, 10, 15, 20, 25]:
            proximity_from_T = util.compute_proximity_arr(network_new_labels, pos_seeds[:k])
            D_from_T = np.min(D_from_candidates[pos_seeds[:k]], axis=0)
            closer_nodes = np.where(D_from_T <= D_m)[0]
            closer_nodes = np.array(list(set(closer_nodes) - set(neg_seeds_new_labels)))
            closer_nodes_count.append(len(closer_nodes))
            proximity_from_T_closer_nodes = proximity_from_T[closer_nodes]
            if k == 5:
                avg_mp_closer_old.append(np.mean(initial_vul[closer_nodes]))
            else:
                avg_mp_closer_old.append(np.mean(mp[closer_nodes]))
            mp = get_mp(method, pos_seeds[:k], 100, neg_policy)
            avg_mp_closer_new.append(np.mean(mp[closer_nodes]))
            total_avg_mp.append(np.mean(mp[non_neg_seed_nodes]))
            max_mp.append(np.max(mp[non_neg_seed_nodes]))
            mp_closer_nodes = mp[closer_nodes]
            corr.append(pearsonr(proximity_from_T_closer_nodes, mp_closer_nodes))
            plt.figure()
            plt.rcParams['lines.markersize'] = 1
            plt.scatter(proximity_from_T_closer_nodes, mp_closer_nodes, c='r', marker='.')
            plt.xlabel("Proximity from T")
            plt.ylabel("mp")
            plt.savefig(f"{welfare_output_folder}mp_proximity_{method}_{k}.pdf")

        plt.figure()
        plt.rcParams['lines.markersize'] = 10
        plt.plot([5, 10, 15, 20, 25], [c[0] for c in corr], 'r+')
        plt.xlabel("k")
        plt.ylabel("Correlation")
        plt.savefig(f"{welfare_output_folder}mp_proximity_corr_{method}.pdf")

        plt.figure()
        plt.rcParams['lines.markersize'] = 10
        plt.plot([5, 10, 15, 20, 25], closer_nodes_count, 'b+')
        plt.xlabel("k")
        plt.ylabel("Closer nodes count")
        plt.savefig(f"{welfare_output_folder}closer_nodes_count_{method}.pdf")

        plt.figure()
        plt.rcParams['lines.markersize'] = 10
        plt.bar(np.array([5, 10, 15, 20, 25]) - 0.25, avg_mp_closer_old, color='r', width=0.25)
        plt.bar(np.array([5, 10, 15, 20, 25]), avg_mp_closer_new, color='b', width=0.25)
        plt.bar(np.array([5, 10, 15, 20, 25]) + 0.25, total_avg_mp, color='g', width=0.25)
        plt.bar(np.array([5, 10, 15, 20, 25]) + 0.50, max_mp, color='y', width=0.25)
        plt.xlabel("k")
        plt.ylabel("Probability")
        plt.legend(["Average mp of closer nodes", "Average mp of closer nodes after T update", "Average mp ", "Max mp "], loc='upper center', bbox_to_anchor=(0, 1, 1, 0.4), ncols=1)
        plt.savefig(f"{welfare_output_folder}avg_mp_{method}.pdf", bbox_inches="tight")