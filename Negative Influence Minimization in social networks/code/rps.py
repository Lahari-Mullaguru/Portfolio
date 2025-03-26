"""
This file contains the implementation of the RPS algorithm from the paper
"Scalable misinformation mitigation in social networks using reverse sampling" by Simpson et al.
"""
import math
from functools import partial

from tqdm import tqdm
import networkx as nx
import numpy as np
import util
import multiprocessing as mp

eps = 0.2


def bfs(graph, misinfo_starters):
    """Do a BFS to get the influence set in g
    :param graph: a possible world
    :param misinfo_starters: set of misinformation nodes
    :return: the influence set
    """
    # initialize a set of nodes
    influence_set = set()
    # initialize a queue of nodes
    queue = []
    # initialize a set of visited nodes
    visited = set()
    # for each misinformation node in M
    for u in misinfo_starters:
        # add the misinformation node to the queue
        queue.append(u)
        # add the misinformation node to the set of visited nodes
        visited.add(u)
        # add the misinformation node to the influence set
        influence_set.add(u)
    # while the queue is not empty
    while queue:
        # remove the first node in the queue
        u = queue.pop(0)
        # for each neighbor of the node
        for neighbor in graph.neighbors(u):
            # if the neighbor is not visited
            if neighbor not in visited:
                # add the neighbor to the queue
                queue.append(neighbor)
                # add the neighbor to the set of visited nodes
                visited.add(neighbor)
                # add the neighbor to the influence set
                influence_set.add(neighbor)
    return influence_set


def bfs1(graph, misinfo_starters):
    """Do a BFS to get the influence set in g
    :param graph: a possible world
    :param misinfo_starters: set of misinformation nodes
    :return: the influence set
    """
    # initialize a set of nodes
    influence_set = set()
    # initialize a queue of nodes
    queue = []
    # initialize a set of visited nodes
    visited = set()

    # for each misinformation node in M
    for u in misinfo_starters:
        # add the misinformation node to the queue
        queue.append(u)
        # add the misinformation node to the set of visited nodes
        visited.add(u)
        # add the misinformation node to the influence set
        influence_set.add(u)
    # while the queue is not empty
    while queue:
        # remove the first node in the queue
        u = queue.pop(0)
        # for each neighbor of the node
        for neighbor in graph.predecessors(u):
            if neighbor in visited:
                continue
            flip = np.random.random() / 2
            if flip > graph[u][neighbor]['negative']:
                continue
            # add the neighbor to the queue
            queue.append(neighbor)
            # add the neighbor to the set of visited nodes
            visited.add(neighbor)
            # add the neighbor to the influence set
            influence_set.add(neighbor)
    return influence_set


def compute_subwidth(graph, rrc_set):
    """compute the number of edges in graph that point to nodes in RRC_set
    as sum of indegree of nodes in RRC_set"
    :param graph: networkx graph
    :param rrc_set: set of nodes
    :return: the number of edges in graph that point to nodes in RRC_set

    """
    subwidth = 0
    for node in rrc_set:
        subwidth += graph.in_degree(node)
    return subwidth


def compute_k_r(graph, RRC_set, k):
    """compute k_r as (1-(1-subwidth(RRC_set)/m)^k)"
    :param graph: networkx graph
    :param RRC_set: set of nodes
    :param k: the number of nodes in the truth seed set
    :return: k_r
    """
    m = graph.number_of_edges()
    subwidth = compute_subwidth(graph, RRC_set)
    k_r = (1 - math.pow(1 - subwidth / m, k))
    return k_r


def kpt_estimation(graph, k, M, depth):
    """Estimate the number of nodes saved vy a seed set where the seeds are chosen proportional to outdegree
    :param graph: networkx graph
    :param k: the number of nodes in the truth seed set
    :param M: set of misinformation nodes
    :return: the number of nodes saved
    """
    n = graph.number_of_nodes()
    limit = math.log(n) / math.log(2)
    print(f"limit: {limit}")
    beta = update_beta(graph, M)
    for i in range(1, int(limit)):
        est_sum = 0.0
        c_i = (6 * math.log(n) + 6 * math.log(math.log(n) / math.log(2))) * pow(2, i)
        print(f"c_{i}: {c_i}")
        for j in tqdm(range(int(c_i))):
            continue_flag, RR = get_RRC_set(M, graph, depth, beta, j)
            if continue_flag:
                continue
            # compute k_r
            k_r = compute_k_r(graph, RR, k)
            est_sum += k_r

        if (est_sum / c_i) > (1 / math.pow(2, i)):
            return n * est_sum / (2 * c_i)
        else:
            print(f"Estimation failed,{(est_sum / c_i)} < {(1 / math.pow(2, i))} try again with a larger limit")
    return 1 / n


def update_beta(g, M):
    """update the beta value for each node in the graph to be the distance from the nearest misinformation node
    :param g: networkx graph
    :param M: set of misinformation nodes
    :return: None
    """
    proximity = {}
    queue = []
    for m in M:
        proximity[m] = 0
        queue.append(m)
    while len(queue) > 0:
        node = queue.pop(0)
        neighbors = g.neighbors(node)
        for neighbor in neighbors:
            if neighbor in proximity:
                oldproximity = proximity[neighbor]
                newproximity = proximity[node] + 1
                if newproximity < oldproximity:
                    proximity[neighbor] = newproximity
                    queue.append(neighbor)
            else:
                proximity[neighbor] = proximity[node] + 1
                queue.append(neighbor)
    beta = proximity
    return beta


def get_RRC_set(M, graph, depth, beta, j):
    """get a RR set from a possible world
    :param M: set of misinformation nodes
    :param graph: networkx graph
    :param j: the seed for the random number generator
    :return: a RR set
    """
    continue_flag = False
    np.random.seed(j)
    influence_set = bfs1(graph, M)

    # select a node in random from the influence set
    root = np.random.choice(list(graph.nodes()))
    while root in M:
        root = np.random.choice(list(graph.nodes()))
    while root not in influence_set:
        continue_flag = True
        return continue_flag, None
    # do RR to get the RR set
    max_d = beta[root]
    RR = generate_rrc(root, max_d, graph, depth, beta)
    return continue_flag, RR


def get_RRC_set_parallel(network, M, depth, beta, j):
    """get a RR set from a possible world
    :param j: the seed for the random number generator
    :return: a RR set
    """
    graph = network
    continue_flag = False
    np.random.seed(j + 2000)

    influence_set = bfs1(graph, M)
    # select a node in random from the influence set
    root = np.random.choice(list(graph.nodes()))
    while root in M:
        root = np.random.choice(list(graph.nodes()))
    while root not in influence_set:
        continue_flag = True
        return continue_flag, None
    # do RR to get the RR set
    max_d = beta[root]
    RR = generate_rrc(root, max_d, network, depth, beta)
    return continue_flag, RR


def refine_kpt(graph, k, M, kpt_star, depth):
    """Refine the estimate of kpt by improving the lower bound
    :param graph: networkx graph
    :param k: the number of nodes in the truth seed set
    :param M: set of misinformation nodes
    :param kpt_star: the lower bound of kpt
    :return: the refined estimate of kpt
    """
    n = graph.number_of_nodes()
    eps_prime = 5.0 * pow((eps * eps) / (k + 1), 1.0 / 3.0)
    theta_prime = (2.0 + eps_prime) * (n * math.log(n)) / (eps_prime * eps_prime * kpt_star)
    print("theta_prime: " + str(theta_prime))
    R_prime = []
    beta = update_beta(graph, M)
    for i in tqdm(range(int(theta_prime))):
        continue_flag, RR = get_RRC_set(M, graph, depth, beta, i + 1000)
        if continue_flag:
            continue
        R_prime.append(RR)
    S_prime, expected_prevention = node_selection(R_prime, k, n, theta_prime)
    kpt_plus = expected_prevention / (1 + eps_prime)
    return max(kpt_plus, kpt_star)


def node_selection(R_set, k, n, theta):
    """Select k optimal nodes using R_set
    :param R_set: set of RR sets
    :param k: the number of nodes in the truth seed set
    :param n: the number of nodes in the graph
    :param theta: the number of RR sets
    :return: the optimal seed set and the expected prevention
    """
    S = []
    preventions = []
    # get number of times each node appears in R_prime
    occurence_count = [0] * n
    occurence_count = get_occurrence_count(R_set, n, occurence_count)
    # get the node with the highest occurence count
    max_count_node = np.argmax(occurence_count)
    # add the node to the seed set
    S.append(max_count_node)
    preventions.append(occurence_count[max_count_node])
    for i in range(k - 1):
        for RR in R_set:
            if max_count_node in RR:
                for node in RR:
                    occurence_count[node] -= 1
        max_count_node = np.argmax(occurence_count)
        S.append(max_count_node)
    expected_prevention = 0.0
    for prevention in preventions:
        expected_prevention += prevention
    expected_prevention = (expected_prevention / theta) * n
    return S, expected_prevention


def node_selection_from_occurrence_count(occurrence_count, k, n, theta, R, network, M):
    """Select k optimal nodes using occurrence_count
    :param occurrence_count: the number of times each node appears in R_set
    :param k: the number of nodes in the truth seed set
    :param n: the number of nodes in the graph
    :param theta: the number of RR sets
    :return: the optimal seed set and the expected prevention
    """
    S = []
    preventions = []
    # get the node with the highest occurrence count
    max_count_nodes = np.argsort(occurrence_count)[::-1]
    for node in max_count_nodes:
        if node not in M:
            max_count_node = node
            break

    # add the node to the seed set
    S.append(max_count_node)
    preventions.append(occurrence_count[max_count_node])
    R_count_set = set(list(range(len(R))))
    R_removed = []
    for i in range(k - 1):
        prev_S = S[-1]
        print(f"S_old_{i} = {util.get_old_labels(network, S)}")
        r_remained = list(R_count_set - set(R_removed))
        for r_count in tqdm(r_remained, total=len(r_remained)):
            RR = R[r_count]
            if prev_S in RR:
                occurrence_count[list(RR)] -= 1
                R_removed.append(r_count)
        print("occurrence_count = ", occurrence_count)
        max_count_nodes = np.argsort(occurrence_count)[::-1]
        for node in max_count_nodes:
            if node not in M and node not in S:
                max_count_node = node
                break
        if occurrence_count[max_count_node] == 0:
            print(f"occurrence_count[{max_count_node}] = 0")
            print("S = ", S)
            print("preventions = ", preventions)
            print(f"S_old_labels =  {util.get_old_labels(network, S)}")
            break
        S.append(max_count_node)
        preventions.append(occurrence_count[max_count_node])
        print("max_occurrence_count = ", occurrence_count[max_count_node])
        print("S = ", S)
        print(f"S_old_labels = {util.get_old_labels(network, S)}")
    # write to file
    expected_prevention = 0.0
    for prevention in preventions:
        expected_prevention += prevention
    expected_prevention = (expected_prevention / theta) * n
    return S, expected_prevention


def get_occurrence_count(R_set, n, occurence_count):
    for i in range(n):
        for RR in R_set:
            if i in RR:
                occurence_count[i] += 1
    return occurence_count


def get_occurence_list(R_set, n):
    occurrence_list = np.zeros(n)
    for RR in tqdm(R_set):
        occurrence_list[list(RR)] += 1
    return occurrence_list


def logcnk(n, k):
    val = 0
    for i in range(n - k + 1, n + 1):
        val += math.log(i)
    for i in range(1, k + 1):
        val -= math.log(i)
    return val


def compute_theta(graph, k, kpt_plus):
    n = graph.number_of_nodes()
    return (8.0 + 2.0 * eps) * n * (math.log(n) + math.log(2) + logcnk(n, k)) / (eps * eps * kpt_plus)


def generate_rrc(root, max_d, graph, depth, beta):
    """Generate an RRC set for a given node in the graph G
    :param root: the root node
    :param max_d: the maximum distance between a node and the root node
    :param graph: networkx graph
    :return: a set of nodes
    """
    # initialize a set of nodes
    RRC = set()
    # initialize a queue of nodes
    Q = []
    # initialize a set of visited nodes
    visited = set()
    # add the root node to the queue
    Q.append(root)
    # add the root node to the set of visited nodes
    visited.add(root)
    # add the root node to the set of RRC nodes
    RRC.add(root)
    depth[root] = 0
    # while the queue is not empty
    while Q:
        # remove the first node in the queue
        node = Q.pop(0)
        RRC.add(node)
        if depth[node] == max_d:
            continue
        if beta[node] == 0:
            continue
        # for each neighbor of the node
        for neighbor in graph.predecessors(node):
            # if the neighbor is not visited
            if neighbor in visited:
                continue
            if beta[node] and beta[node] > 0 and beta[neighbor] and beta[neighbor] > 0:
                # if the node is closer to the M node than the neighbor is
                if beta[node] - 1 < beta[neighbor]:
                    beta[neighbor] = beta[node] - 1
            elif beta[node] and beta[node] > 0 and (beta[neighbor] is None):
                beta[neighbor] = beta[node] - 1
            depth[neighbor] = depth[node] + 1
            # add the neighbor to visited nodes
            visited.add(neighbor)
            # add the neighbor to the queue
            Q.append(neighbor)
    return RRC


def rps(network, M, k_p):
    network = nx.convert_node_labels_to_integers(network, label_attribute="old_label")
    M = util.get_new_labels(network, M)
    depth = np.zeros(network.number_of_nodes())
    kpt_star = kpt_estimation(network, k_p, M, depth)
    print("kpt_star = ", kpt_star)
    print("Estimating kpt_plus...")
    kpt_plus = refine_kpt(network, k_p, M, kpt_star, depth)
    print("kpt_plus = ", kpt_plus)
    print("Estimating theta...")
    theta = compute_theta(network, k_p, kpt_plus)
    print("theta = ", theta)
    n = network.number_of_nodes()
    N = int(theta)
    print("N = ", N)

    R = []
    it_counter = 0
    print("Generating R sets...")
    r_list = range(it_counter, N)
    pool = mp.Pool(mp.cpu_count())
    beta = update_beta(network, M)
    for continue_flag, RR in tqdm(pool.imap_unordered(partial(get_RRC_set_parallel, network, M, depth, beta), r_list),
                                  total=N):
        if continue_flag:
            continue
        # add the RR set to the list of RR sets
        R.append(RR)
        it_counter += 1
    print("R sets generated successfully!")
    print(f"Total number of R sets generated: {len(R)}")
    occurrence_count = get_occurence_list(R, n)
    print("occurrence_count = ", occurrence_count)
    print("Node selection...")
    print("k_p = ", k_p)
    S, expected_prevention = node_selection_from_occurrence_count(occurrence_count, k_p, n,
                                                                  theta, R, network, M)
    print("The expected prevention is ", expected_prevention)
    old_labels = [network.nodes[node]["old_label"] for node in S]
    print("old labels :", old_labels)
    return old_labels
