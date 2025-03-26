"""
This file contains the code for generating HICH-BA graphs
"""
import util
import os
import sys
import networkx as nx
from tqdm import tqdm
import random

sys.path.append(os.getcwd())

def hichoba(n, m, r, h, p_PA, p_T):
    """
    from https://github.com/cristinagub/master_project
    HIgh Clustering Homophily Barab´asi-Albert(HICH-BA) model
    :param n: number of nodes
    :param r: population of communities
    :param h: homofilic preference
    :param p_PA: probability of preferential attachment
    :param p_out:
    :param p_R:
    :param p_T: probability of triadic closure
    :return:
    """
    num_com = len(r)
    G = nx.Graph()
    nx.set_node_attributes(G, [], "community")
    G.add_nodes_from(range(num_com))
    nodes = len(G.nodes())

    choices_c = {c: [] for c in range(num_com)}

    c = 0
    for v in G.nodes():
        G.nodes[v]['community'] = c
        choices_c[c].append(v)
        c += 1

    print(G.nodes())
    print(G.edges())
    print([(x, G.nodes[x]['community']) for x in G.nodes()])

    iter = len(G.nodes()) + len(G.edges()) - len(G.nodes())
    pbar = tqdm(total=n+m/2, position=0, leave=True)
    pbar.update(len(G.nodes()))
    p_N = 2 * n / m

    while nodes <= n:
        add_nodes_flag = random.uniform(0, 1) <= p_N
        print("add_nodes_flag: ", add_nodes_flag, "nodes: ", nodes, "n: ", n)
        if add_nodes_flag:
            G.add_node(nodes - 1)
            source = nodes - 1
            nodes += 1
            c = random.choices(range(num_com), weights=r, k=1)[0]
            G.nodes[source]['community'] = c

            choices_c[c].append(source)

            choices = [x for x in choices_c[c] if x != source]

            if random.uniform(0, 1) <= p_PA:
                weights = [G.degree(v) + 1 for v in choices]
            # weights = [choices_weights_c[G.nodes[v]['community']][v] for v in choices]
            else:
                weights = [1 for v in choices]
        else:
            if random.uniform(0, 1) <= p_T:
                if random.uniform(0, 1) <= p_PA:
                    if len([x for x in G.nodes() if G.degree(x) >= 2]) == 0: continue
                    weights = [G.degree(x) + 1 for x in G.nodes() if G.degree(x) >= 2]

                else:
                    if len([x for x in G.nodes() if G.degree(x) >= 2]) == 0: continue
                    weights = [1 for x in G.nodes() if G.degree(x) >= 2]
                v = random.choices([x for x in G.nodes() if G.degree(x) >= 2], weights = weights, k = 1)[0]

                source = random.choice(list(G.neighbors(v)))
                options = [y for y in G.neighbors(v) if not G.has_edge(source, y) and y != source]
                if len(options) == 0:
                    continue
                intra_inter = random.uniform(0, 1)
                if intra_inter <= h:
                    choices = [x for x in options if G.nodes[v]['community'] == G.nodes[x]['community']]
                else:
                    choices = [x for x in options if G.nodes[v]['community'] != G.nodes[x]['community']]

                if random.uniform(0, 1) <= p_PA:
                    weights = [G.degree(w) + 1 for w in choices]
                else:
                    weights = [1 for w in choices]

                if len(choices) == 0: print("no ", intra_inter);continue
            else:
                if random.uniform(0, 1) <= p_PA:
                    weights = [G.degree(x) + 1 for x in G.nodes()]
                else:
                    weights = [1 for x in G.nodes()]
                source = random.choices([x for x in G.nodes()], weights=weights, k=1)[0]

                # neigh = list(G.neighbors(v))
                # options = [x for x in G.nodes() if x not in neigh]
                options = [x for x in G.nodes() if x != source]
                intra_inter = random.uniform(0, 1)
                if intra_inter <= h:
                    choices = [x for x in options if G.nodes[v]['community'] == G.nodes[x]['community']]
                else:
                    choices = [x for x in options if G.nodes[v]['community'] != G.nodes[x]['community']]

                if random.uniform(0, 1) <= p_PA:
                    weights = [G.degree(w) + 1 for w in choices]
                else:
                    weights = [1 for v in choices]

                if len(choices) == 0: continue

        target = random.choices(choices, weights=weights, k=1)[0]
        G.add_edge(source, target)
        pbar.update(1)
        iter += 1

    while G.number_of_edges() < (m/2):
        print(f"edges: {G.number_of_edges()}, m: {m}")
        if random.uniform(0, 1) <= p_T:
            if random.uniform(0, 1) <= p_PA:
                if len([x for x in G.nodes() if G.degree(x) >= 2]) == 0: continue
                weights = [G.degree(x) + 1 for x in G.nodes() if G.degree(x) >= 2]

            else:
                if len([x for x in G.nodes() if G.degree(x) >= 2]) == 0: continue
                weights = [1 for x in G.nodes() if G.degree(x) >= 2]
            v = random.choices([x for x in G.nodes() if G.degree(x) >= 2], weights = weights, k = 1)[0]

            source = random.choice(list(G.neighbors(v)))
            options = [y for y in G.neighbors(v) if not G.has_edge(source, y) and y!=source]
            if len(options) == 0:
                continue
            intra_inter = random.uniform(0, 1)
            if intra_inter <= h:
                choices = [x for x in options if G.nodes[v]['community'] == G.nodes[x]['community']]
            else:
                choices = [x for x in options if G.nodes[v]['community'] != G.nodes[x]['community']]

            if random.uniform(0, 1) <= p_PA:
                weights = [G.degree(w) + 1 for w in choices]
            else:
                weights = [1 for w in choices]

            if len(choices) == 0: print("no ", intra_inter);continue
        else:
            if random.uniform(0, 1) <= p_PA:
                weights = [G.degree(x) + 1 for x in G.nodes()]
            else:
                weights = [1 for x in G.nodes()]
            source = random.choices([x for x in G.nodes()], weights=weights, k=1)[0]

            # neigh = list(G.neighbors(v))
            # options = [x for x in G.nodes() if x not in neigh]
            options = [x for x in G.nodes() if x != source]
            intra_inter = random.uniform(0, 1)
            if intra_inter <= h:
                choices = [x for x in options if G.nodes[v]['community'] == G.nodes[x]['community']]
            else:
                choices = [x for x in options if G.nodes[v]['community'] != G.nodes[x]['community']]

            if random.uniform(0, 1) <= p_PA:
                weights = [G.degree(w) + 1 for w in choices]
            else:
                weights = [1 for v in choices]

            if len(choices) == 0: continue
        target = random.choices(choices, weights=weights, k=1)[0]
        G.add_edge(source, target)
        pbar.update(1)
        iter += 1
    pbar.close()
    return G

# main method
if __name__ == '__main__':
    n = 10000
    config_list = [(200000,0.9,0.9,0.9),
        (40000,0.9,0.9,0.9),
        (100000, 0.9, 0.9, 0.1),
        (100000,0.9,0.2,0.9),
        (100000,0.2,0.9,0.9)
    ]
    k_n = 10
    neg_policy = "degree"
    weight = "value"
    r = [0.54, 0.3, 0.15, 0.005, 0.005]

    for m, h, p_T, p_PA in config_list:
        dataset = f"HICHBA_{n}_{m}_{h}_{p_PA}_{p_T}"
        network = util.hichoba(n, m, r, h, p_PA, p_T)
        network = network.to_directed()
        # write networkx network edge list to file
        nx.write_edgelist(network, f"../data/{dataset}_el.txt", data=False)
        # write networkx network with community data to file
        nx.write_gexf(network, f"../data/{dataset}_gpickle")
        print(f"Dataset {dataset} written to file!")
