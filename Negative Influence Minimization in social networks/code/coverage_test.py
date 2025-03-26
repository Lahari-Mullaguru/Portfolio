"""
This file is used to test the coverage of the negative seeds in the network
This file is also used to compute network characteristics
"""
import multiprocessing as multip
from functools import partial
from collections import Counter
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import timeit
import util
import diffusionmodels as dm
import os
import sys
sys.path.append(os.getcwd())

# main method
if __name__ == '__main__':
    k_n = 10
    neg_policy = "degree"
    weight = "value"
    dataset = "email"
    if "facebook" in dataset or "fb" in dataset:
        network = util.load_bidirectional_graph(f"../data/{dataset}.txt")
    else:
        network = util.load_graph(f"../data/{dataset}.txt")
    print("dataset loaded the graph successfully! There are ",
          network.number_of_nodes(), " nodes and ", network.number_of_edges(), " edges")
    largest_component = max(nx.strongly_connected_components(network), key=len)
    network = network.subgraph(largest_component)
    print(f"largest_subgraph has {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
    print("Average degree :" + str(sum([network.out_degree(x) for x in network.nodes()]) / len(network.nodes())))
    print("clustering coefficient: " + str(nx.average_clustering(network, network.nodes())))
    print("diameter: " + str(nx.diameter(network)))
    print("average shortest path length: " + str(nx.average_shortest_path_length(network)))
    neg_seeds = util.get_neg_seeds(network, neg_policy, k_n)
    network_new_labels = nx.convert_node_labels_to_integers(network, label_attribute="old_label")
    neg_seeds_new_labels = util.get_new_labels(network_new_labels, neg_seeds)
    proximity = util.compute_proximity(network_new_labels,neg_seeds_new_labels)
    # # compute avg number of neighbours
    avg_neighbours = sum([network_new_labels.out_degree(x) for x in neg_seeds_new_labels]) / len(neg_seeds_new_labels)
    print(f"avg neighbours: {avg_neighbours}")
    proximity_count = dict(Counter(proximity.values()))
    print(f"proximity_count: {proximity_count}")
    #plot proximity_count dict
    plt.bar(sorted(proximity_count.keys()), [proximity_count[key] for key in sorted(proximity_count.keys())])
    plt.xlabel("shortest distance from M")
    plt.ylabel("number of nodes")
    plt.title(f"Proximity of nodes to M in {dataset}")
    plt.show()
    plt.close()
    degree_freq = nx.degree_histogram(network_new_labels)
    print(f"degree_freq: {degree_freq}")
    degrees = range(len(degree_freq))
    print(f"degrees: {degrees}")
    plt.figure(figsize=(8, 6))
    plt.loglog(degrees, degree_freq, 'b*')
    plt.xlabel("Degree", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"../outputs/figs/{dataset}_deg.pdf")
    print("Done")
    for min_prob in [0.1,0.2,0.3]:
        weight = "value"
        if weight == "value":
            network = util.set_weight_value(network, min_prob, "positive")
            network = util.set_weight_value(network, min_prob, "negative")
        elif weight == "degree":
            network = util.set_weight_degree(network, "positive")
            network = util.set_weight_degree(network, "negative")
        network_new_labels = nx.convert_node_labels_to_integers(network, label_attribute="old_label")
        neg_seeds_new_labels = util.get_new_labels(network_new_labels, neg_seeds)

        num_simulations = 100
        pool = multip.Pool(multip.cpu_count())
        neg_impressed_counts = []
        start = timeit.default_timer()

        partial_func = partial(dm.icm, network_new_labels, neg_seeds_new_labels)
        for neg_impressed in tqdm(pool.imap_unordered(partial_func, list(range(num_simulations))),
                                        total=num_simulations):
            neg_impressed_counts.append(len(neg_impressed))

        avg_neg_impressed = np.mean(neg_impressed_counts)
        coverage = avg_neg_impressed / network.number_of_nodes() * 100
        stop = timeit.default_timer()
        print("Time taken: ", stop - start)
        print("Average negative influence is", avg_neg_impressed)
        print("Coverage is", coverage)
        pool.close()
        pool.join()
