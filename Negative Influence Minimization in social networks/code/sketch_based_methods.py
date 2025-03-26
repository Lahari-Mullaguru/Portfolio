from decimal import Decimal
from tqdm import tqdm
import util
from collections import Counter, defaultdict

from tqdm import tqdm
import networkx as nx
import numpy as np
import timeit
import heapq as heap
import collections
import random
import math




def dijkstra_MIA(startingNode, G_path, G_deg, in_edges, min_prob, S_N, p_n):  # dijkstra's algorithm to compute the MIIA

    MIA = nx.DiGraph()
    MIA.add_node(startingNode)

    prob, probabilities = {startingNode: 1}, {startingNode: 1}
    steps, steps_final = {startingNode: 0}, {startingNode: 0}
    parentsMap = {}

    pq = []
    nodeCosts = defaultdict(lambda: float('inf'))
    nodeCosts[startingNode] = 0
    heap.heappush(pq, (0, startingNode))

    while pq:
        # go greedily by always extending the shorter cost nodes first
        _, node = heap.heappop(pq)
        if node in S_N: continue
        if not in_edges:
            neighbors = list(G_path.successors(node))
        else:
            neighbors = list(G_path.predecessors(node))
        for adjNode in neighbors:
            if not in_edges:
                weight = -np.log(G_deg[node][adjNode][p_n])
                prob_val = round(G_deg[node][adjNode][p_n] * prob[node], 3)
            else:
                weight = -np.log(G_deg[adjNode][node][p_n])
                prob_val = round(G_deg[adjNode][node][p_n] * prob[node], 3)
            newCost = nodeCosts[node] + weight
            if nodeCosts[adjNode] > newCost and prob_val >= min_prob:
                parentsMap[adjNode] = node
                nodeCosts[adjNode] = newCost
                prob[adjNode] = prob_val
                steps[adjNode] = steps[node] + 1
                heap.heappush(pq, (newCost, adjNode))

    for w in parentsMap.keys():
        if in_edges:
            MIA.add_edge(w, parentsMap[w])
        else:
            MIA.add_edge(parentsMap[w], w)
        probabilities[w] = prob[w]
        steps_final[w] = steps[w]

    return MIA, steps_final, probabilities  # we return the MIA and the parentMap in case we need to reconstruct the MIA


def get_p_value(v, pos, t, S_N, S_P):  # initializes the values of P_p_t, P_n_t, ap_n_t and ap_p_t in compute_ap_N
    if v in S_P:
        if pos:
            if t == 0:
                return 1, 1
            else:
                return 0, 1
        else:
            return 0, 0

    elif v in S_N:
        if pos:
            return 0, 0
        else:
            if t == 0:
                return 1, 1
            else:
                0, 1
    else:
        return 0, 0


def compute_ap_N(u, S_N, S_P, MIA_n, MIA_p,
                 G):  # probability of transmission is inversly proportional to the in-degree

    POS_t = [v for v in S_P if v in MIA_p.nodes()]
    NEG_t = [v for v in S_N if v in MIA_n.nodes()]
    POS_next = set()
    NEG_next = set()

    P_p_t = {}
    P_p_next = {}
    P_n_t = {}
    P_n_next = {}

    ap_n_t = {}
    ap_p_t = {}
    for node in list(MIA_n.nodes()) + list(MIA_p.nodes()):
        P_p_t[node], ap_p_t[node] = get_p_value(node, True, 0, S_N, S_P)
        P_n_t[node], ap_n_t[node] = get_p_value(node, False, 0, S_N, S_P)
    t = 0

    while NEG_t:

        ap_n_next = ap_n_t.copy()
        ap_p_next = ap_p_t.copy()

        temp_P = {k: 1 for k in list(MIA_n.nodes()) + list(MIA_p.nodes())}
        temp_N = {k: 1 for k in list(MIA_n.nodes()) + list(MIA_p.nodes())}

        for v in POS_t:  # positive influence has priority

            if not list(MIA_p.successors(v)):
                continue
            for w in list(MIA_p.successors(v)):
                POS_next.add(w)
                temp_P[w] *= (1 - P_p_t[v] * G[v][w]['positive'])

        for v in POS_next:
            P_p_next[v] = (1 - temp_P[v]) * (1 - ap_n_t[v]) * (1 - ap_p_t[v])
            ap_p_next[v] = ap_p_t[v] + P_p_next[v]

        for v in NEG_t:
            if not list(MIA_n.successors(v)):
                continue
            for w in list(MIA_n.successors(v)):
                NEG_next.add(w)
                temp_N[w] *= (1 - P_n_t[v] * G[v][w]['negative'])

        for v in NEG_next:
            P_n_next[v] = temp_P[v] * (1 - temp_N[v]) * (1 - ap_n_t[v]) * (1 - ap_p_t[v])
            ap_n_next[v] = ap_n_t[v] + P_n_next[v]

        if list(POS_next):
            ap_p_t = ap_p_next.copy()

        if list(NEG_next):
            ap_n_t = ap_n_next.copy()

        P_p_t, P_n_t, NEG_t, POS_t = P_p_next, P_n_next, NEG_next, POS_next
        POS_next = set()
        NEG_next = set()
        t += 1

    return ap_n_t[u]


def path_blocked_MIA(from_node,to_node, MIIA, S): # if a negative information needs to go through a node in S_P the path is blocked
    start=timeit.default_timer()
    current_node=from_node
    if current_node in S:
        return True
    while current_node != to_node and list(MIIA.successors(current_node)):
        if list(MIIA.successors(current_node))[0] in S:
            return True
        else:
            current_node=list(MIIA.successors(current_node))[0]
    return False


def path_blocked_MOA(from_node,to_node, MIOA, S): # if a negative information needs to go through a node in S_P the path is blocked
    current_node=to_node
    if current_node in S:
        return True
    while current_node != from_node:
        if list(MIOA.predecessors(current_node))[0] in S:
            return True
        else:
            current_node=list(MIOA.predecessors(current_node))[0]
    return False

def cost(S_P,funct,G,avg_deg=1):
    if funct=="uniform":
        return len(S_P)
    if funct=="degree_penalty":
        if len(S_P)==0: return 0
        return np.sum([G.out_degree(x) for x in S_P])/(avg_deg)


def dijkstra_MIOG(startingNode, G, min_prob, S_N):  # dijkstra's algorithm to compute the MIOG

    MIOG = nx.DiGraph()
    MIOG.add_node(startingNode)

    prob, probabilities = {startingNode: 1}, {startingNode: 1}
    steps, steps_final = {startingNode: 0}, {startingNode: 0}
    parentsMap = {}

    pq = []
    nodeCosts = defaultdict(lambda: float('inf'))
    nodeCosts[startingNode] = 0
    heap.heappush(pq, (0, startingNode))

    while pq:
        # go greedily by always extending the shorter cost nodes first
        _, node = heap.heappop(pq)
        if node in S_N: continue

        neighbors = list(G.successors(node))
        for adjNode in neighbors:

            weight = -np.log(G[node][adjNode]["negative"])
            prob_val = round(G[node][adjNode]["negative"] * prob[node], 3)

            newCost = nodeCosts[node] + weight
            if nodeCosts[adjNode] > newCost and prob_val >= min_prob:
                parentsMap[adjNode] = [node]
                nodeCosts[adjNode] = newCost
                prob[adjNode] = prob_val
                steps[adjNode] = steps[node] + 1
                heap.heappush(pq, (newCost, adjNode))
                continue
            if nodeCosts[adjNode] == newCost and prob_val >= min_prob and adjNode != startingNode and node not in \
                    parentsMap[adjNode]:
                parentsMap[adjNode] = parentsMap[adjNode] + [node]

    for w in parentsMap.keys():
        for v in parentsMap[w]:
            MIOG.add_edge(v, w)
        probabilities[w] = prob[w]
        steps_final[w] = steps[w]
    return MIOG


def compute_ap_N_MIOG(S_N, S_P, MIOG_N, MIOG_P,
                      G):  # probability of transmission is inversly proportional to the in-degree

    POS_t = [v for v in S_P if v in MIOG_P.nodes()]
    NEG_t = [v for v in S_N if v in MIOG_N.nodes()]
    POS_next = set()
    NEG_next = set()

    P_p_t = {}
    P_p_next = {}
    P_n_t = {}
    P_n_next = {}

    ap_n_t = {}
    ap_p_t = {}
    for node in list(MIOG_N.nodes()) + list(MIOG_P.nodes()):
        P_p_t[node], ap_p_t[node] = get_p_value(node, True, 0, S_N, S_P)
        P_n_t[node], ap_n_t[node] = get_p_value(node, False, 0, S_N, S_P)
    t = 0

    while NEG_t:

        ap_n_next = ap_n_t.copy()
        ap_p_next = ap_p_t.copy()

        temp_P = {k: 1 for k in list(MIOG_N.nodes()) + list(MIOG_P.nodes())}
        temp_N = {k: 1 for k in list(MIOG_N.nodes()) + list(MIOG_P.nodes())}

        for v in POS_t:  # positive influence has priority

            if not list(MIOG_P.successors(v)):
                continue
            for w in list(MIOG_P.successors(v)):
                POS_next.add(w)
                temp_P[w] *= (1 - P_p_t[v] * G[v][w]['positive'])

        for v in POS_next:
            P_p_next[v] = (1 - temp_P[v]) * (1 - ap_n_t[v]) * (1 - ap_p_t[v])
            ap_p_next[v] = ap_p_t[v] + P_p_next[v]

        for v in NEG_t:
            if not list(MIOG_N.successors(v)):
                continue
            for w in list(MIOG_N.successors(v)):
                NEG_next.add(w)
                temp_N[w] *= (1 - P_n_t[v] * G[v][w]['negative'])

        for v in NEG_next:
            P_n_next[v] = temp_P[v] * (1 - temp_N[v]) * (1 - ap_n_t[v]) * (1 - ap_p_t[v])
            ap_n_next[v] = ap_n_t[v] + P_n_next[v]

        if list(POS_next):
            ap_p_t = ap_p_next.copy()

        if list(NEG_next):
            ap_n_t = ap_n_next.copy()

        P_p_t, P_n_t, NEG_t, POS_t = P_p_next, P_n_next, NEG_next, POS_next
        POS_next = set()
        NEG_next = set()
        t += 1

    return ap_n_t


def compute_threat_levels(G, S_N):
    s1 = timeit.default_timer()
    G_source = G.copy()
    G_source.add_edges_from([("new", x) for x in S_N])
    G_source.remove_edges_from([e for e in G.edges() if e[1] in S_N])
    DAG = nx.DiGraph()
    sign = {}

    new_nodes = collections.deque()
    new_nodes.append(("new", 0))
    visited = set()

    DAG.add_edges_from(nx.dfs_edges(G_source, source="new", depth_limit=10))
    order = list(nx.dfs_preorder_nodes(DAG, source="new"))
    if False:
        while new_nodes:
            vertex, depth = new_nodes.pop()
            if vertex in visited: continue
            visited.add(vertex)
            order.append(vertex)
            for neighbor in list(G_source.successors(vertex)):
                if neighbor not in visited and neighbor not in DAG.nodes():
                    DAG.add_edge(vertex, neighbor)
                    new_nodes.append((neighbor, depth + 1))

    # print("DFS done")
    # print(list(DAG.edges()))
    # print(list(reversed(list(nx.topological_sort(DAG)))))
    # print([(e[0],e[1]) for e in list(DAG.edges()) if e[1]==18])
    # plt.figure()
    # pos=nx.kamada_kawai_layout(DAG)
    # nx.draw(DAG,pos,with_labels=1)
    # plt.show()
    sigma = {k: order.index(k) for k in order}
    sign = {}
    for u in order:
        # print("u"+str(u)+str(list(DAG.predecessors(u))))
        sign[u] = {}
        if u == "new": continue

        for p in DAG.predecessors(u):
            # print("p"+str(p))
            sign[u].update(sign[p])
            childreen = list(DAG.successors(p))
            if len(childreen) > 1:
                sign[u][p] = u

    # print(sign)
    # print(timeit.default_timer()-s1)
    for e in G.edges():
        # print(e,e[0] not in DAG.nodes(),e[1] not in DAG.nodes())
        if e[0] not in DAG.nodes() or e[1] not in DAG.nodes() or e in DAG.edges() or e[1] in S_N: continue
        if sigma[e[1]] >= sigma[e[0]]: continue
        set1 = set(sign[e[0]])
        set2 = set(sign[e[1]])
        if not set1.intersection(set2): continue
        # print("e"+str(e))
        # print(set1.intersection(set2),sign[e[0]])
        opts = {key: sign[e[1]][key] for key in set1.intersection(set2)}
        wu1 = sign[e[0]][max(opts, key=opts.get)]
        # print(wu1,sigma[e[1]],sigma[wu1],sigma[e[0]])
        if sigma[e[1]] < sigma[wu1] and sigma[wu1] <= sigma[e[0]]:
            DAG.add_edge(e[0], e[1])
            # print("yes")

    # print(timeit.default_timer()-s1)
    DAG.remove_nodes_from(["new"])
    print("DAG done")
    print(f"DAG has {nx.number_of_nodes(DAG)} nodes and {nx.number_of_edges(DAG)} edges")
    # plt.figure()
    # nx.draw_kamada_kawai(DAG,with_labels=1)
    # plt.show()
    reversed_topological_ordering = list(reversed(list(nx.topological_sort(DAG))))
    # print(list(DAG.edges()))
    threat_levels = {k: defaultdict(lambda: 0) for k in reversed_topological_ordering if k not in S_N}
    # print(threat_levels)
    for u in reversed_topological_ordering:
        if u in S_N: continue
        threat_levels[u][0] = 1
        q_u = {}
        for v in DAG.successors(u):
            q_u[v] = G[u][v]["negative"] * np.prod(
                [1 - G[parent][v]["negative"] for parent in list(G.predecessors(v)) if parent != u])
        t = 1

        while any(threat_levels[x][t - 1] != 0 for x in DAG.successors(u)):
            for v in DAG.successors(u):
                threat_levels[u][t] += q_u[v] * threat_levels[v][t - 1]
                # print(v,t-1,threat_levels[u][t],threat_levels[v][t-1])
            t += 1

    return threat_levels, set([k for k in reversed_topological_ordering if k not in S_N]), DAG


def Max_Coverage_TIB(RRs, k, G, S):
    start = timeit.default_timer()
    S_k = set()
    need_covering = [x for x in range(len(RRs))]
    for i in range(k):

        list_rr = []
        for i in need_covering:
            list_rr += RRs[i]

        options = Counter(list_rr)
        u = options.most_common(1)[0][0]

        need_covering = [x for x in need_covering if u not in RRs[x]]

        S_k.add(u)
        Cov_S_k = len(RRs) - len(need_covering)
    # print(timeit.default_timer()-start)
    return S_k, Cov_S_k * len(S) / len(RRs)


def ipsilon(e, d):
    return (2 + 2 / 3 * e) * np.log(1 / d) * (1 / (e ** 2))


def ipsilon_WRS(e,d):
    #print(type(e),type(d))
    return ((2+Decimal(2/3)*e)*((1/d).ln())*(1/(e**2)))


def generate_WRR(G, S, S_N, threat_levels, max_depth):
    s1 = timeit.default_timer()
    WRR = nx.DiGraph()
    A, m, s = [], defaultdict(lambda: 0), {}
    r = random.choice(list(S))
    A.append((r, 0))
    WRR.add_node(r)
    # print(r)
    original_node = {}
    original_node[r] = r
    copies = defaultdict(lambda: 0)
    ancestors_u = defaultdict(lambda: set())
    score = {}
    S_N_WRR = set()
    while A:

        v, l = A.pop(0)

        v_original = original_node[v]
        s[v] = l
        if v == r:
            m[v] = 1
        else:  # compute mv
            for c in WRR.predecessors(v):
                c_original = c
                if c in original_node: c_original = original_node[c]
                m[v] += G[v_original][c_original]["negative"] * m[c]

        if v_original not in S_N:
            if l >= max_depth: continue
            for u in G.predecessors(v_original):
                # print(v,u)
                if random.random() <= G[u][v_original]['negative']:
                    # print("yes")
                    if u in WRR.nodes():
                        if u in ancestors_u[v]:
                            # print("ups")
                            continue
                        copies[u] += 1
                        original_node[str(u) + "copy" + str(copies[u])] = u
                        WRR.add_edge(v, str(u) + "copy" + str(copies[u]))
                        A.append((str(u) + "copy" + str(copies[u]), s[v] + 1))
                        for node in nx.ancestors(WRR, str(u) + "copy" + str(copies[u])): ancestors_u[node].update(
                            [str(u) + "copy" + str(copies[u])])

                    else:
                        WRR.add_edge(v, u)
                        A.append((u, s[v] + 1))
                        for node in nx.ancestors(WRR, u): ancestors_u[node].update([u])
                        original_node[u] = u

                # print("no")
        else:

            S_N_WRR.add(v)
    if list(S_N_WRR):
        for u in WRR.nodes():
            if not str(u).isnumeric(): continue
            if u in S_N_WRR:
                score[u] = 0
                continue
            num_copies = copies[u]
            if num_copies == 0:
                beta = m[u] * np.prod(
                    [(1 - m[s_n]) for s_n in list(set(S_N_WRR) - set(ancestors_u[u])) if s[s_n] < s[u]])
                sum_threat = 0
                t = 0
                while threat_levels[r][t] != 0: sum_threat += threat_levels[r][t]; t += 1
                score[u] = beta * sum_threat
            else:
                score[u] = 0
                beta = defaultdict(lambda: 0)
                copies_u = [str(u) + "copy" + str(i) for i in range(1, num_copies + 1)] + [u]
                ancestors_total = set(x for copy in copies_u for x in ancestors_u[copy])
                steps_u = set(s[copy] for copy in copies_u)
                sum_threat = 0
                t = 0
                while threat_levels[r][t] != 0: sum_threat += threat_levels[r][t]; t += 1

                for t in steps_u:
                    beta[t] = (1 - np.prod([1 - m[u_c] for u_c in copies_u if s[u_c] == t]))
                    beta[t] *= np.prod(
                        [(1 - m[s_n]) for s_n in list(set(S_N_WRR) - set(ancestors_total)) if s[s_n] < t])
                    score[u] += beta[t] * sum_threat
        # print(timeit.default_timer()-s1)
        # print(list(WRR.nodes()),list(original_node))
        return True, r, WRR, score, original_node, ancestors_u, s, m, S_N_WRR, copies  # the returned WRR has to have S_N node


    else:
        # print(timeit.default_timer()-s1)
        # print(list(WRR.nodes()),list(original_node))
        return False, None, WRR, None, None, None, None, None, None, None  # the returned WRR has to have S_N node


def D_SSA_TIB(G, S, S_N, k, e, d, threat_levels, fair, max_depth):
    # print("k"+str(k))
    n = len(S)
    d = Decimal(d)
    e = Decimal(e)
    choose = (Decimal(math.factorial(n)) / (Decimal(math.factorial(n - k)) * Decimal(math.factorial(k))))
    d_prime = d / 6 / choose

    N_max = 8 * (Decimal(1 - 1 / math.e) / (2 + 2 * e / 3)) * (ipsilon_WRS(e, d_prime)) * Decimal(n / k)
    # print(N_max)
    t_max = math.ceil((2 * N_max / (ipsilon_WRS(e, d / 3))).log10() / Decimal(2).log10())
    t = 1
    Lambda = round(ipsilon_WRS(e, d / (3 * t_max)))
    Lambda_1 = 1 + (1 + e) * ipsilon_WRS(e, d / (3 * t_max))
    # print(Lambda,Lambda_1)
    keep, scores, WRRs, original_nodes, RRs, roots, S_N_u, s, m, S_N_WRR, copies = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for i in tqdm(range(Lambda * 2 ** (t - 1)), position=0, leave=True):

        # if fair:
        #     keep[i], roots[i], WRR, scores[i], original_nodes[i], S_N_u[i], s[i], m[i], S_N_WRR[i], copies[
        #         i] = get_WRR_contributions(G, list(set(G.nodes) - set(S_N)), S_N, max_depth)
        # else:
        #     keep[i], roots[i], WRR, scores[i], original_nodes[i], S_N_u[i], s[i], m[i], S_N_WRR[i], copies[
        #         i] = generate_WRR(G, S, S_N, threat_levels, max_depth)
        keep[i], roots[i], WRR, scores[i], original_nodes[i], S_N_u[i], s[i], m[i], S_N_WRR[i], copies[
                 i] = generate_WRR(G, S, S_N, threat_levels, max_depth)
        WRRs[i] = [e for e in WRR.edges()].copy()
        RRs[i] = [e for e in WRR.nodes()].copy()

    # print(len({k: v for k, v in roots.items() if v is not None}))
    # print(t,len(RRs))
    while len(RRs) < N_max:

        keep_t, scores_t, WRRs_t, original_nodes_t, RRs_t, roots_t, S_N_u_t, s_t, m_t, S_N_WRR_t, copies_t = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        for i in tqdm(range(Lambda * 2 ** (t - 1), Lambda * 2 ** t), position=0, leave=True):

            # if fair:
            #     keep_t[i], roots_t[i], WRR_t, scores_t[i], original_nodes_t[i], S_N_u_t[i], s_t[i], m_t[i], S_N_WRR_t[
            #         i], copies_t[i] = get_WRR_contributions(G, list(set(G.nodes) - set(S_N)), S_N, max_depth)
            # else:
            #     keep_t[i], roots_t[i], WRR_t, scores_t[i], original_nodes_t[i], S_N_u_t[i], s_t[i], m_t[i], S_N_WRR_t[
            #         i], copies_t[i] = generate_WRR(G, S, S_N, threat_levels, max_depth)
            keep_t[i], roots_t[i], WRR_t, scores_t[i], original_nodes_t[i], S_N_u_t[i], s_t[i], m_t[i], S_N_WRR_t[
                i], copies_t[i] = generate_WRR(G, S, S_N, threat_levels, max_depth)
            WRRs_t[i] = [e for e in WRR_t.edges()].copy()
            RRs_t[i] = [e for e in WRR_t.nodes()].copy()
        RRs_t_kept = [x for x in keep_t]
        S_k, I = Max_Coverage_TIB(RRs, k, G, S)
        # Cov_S_k_t = np.sum([1 if not list(S_k & set(RRs_t[i])) else 0  for i in range(Lambda*2**(t-1),Lambda*2**t)])
        Cov_S_k_t = np.sum(
            [1 if any([j in RRs_t[i] for j in S_k]) else 0 for i in range(Lambda * 2 ** (t - 1), Lambda * 2 ** t)])
        # print("cov"+str((Cov_S_k_t,len(RRs_t))))
        if Cov_S_k_t >= Lambda_1:
            I_t = Cov_S_k_t * n / len(RRs_t_kept)
            e1 = Decimal((I / I_t) - 1)
            e2 = e * np.sqrt(Decimal(n * (1 + e)) / Decimal(2 ** (t - 1) * I_t))
            e3 = e * np.sqrt(Decimal(n * (1 + e) * (Decimal(1 - 1 / math.e) - e)) / Decimal(
                (1 + e / 3) * 2 ** (t - 1) * Decimal(I_t)))
            e_t = (e1 + e2 + e1 * e2) * (Decimal(1 - 1 / math.e) - e) + Decimal(1 - 1 / math.e) * e3
            # print(e1,e2,e3)
            # print("e_t"+str(e_t))
            if e_t <= e: return keep, scores, WRRs, original_nodes, roots, RRs, S_N_u, s, m, S_N_WRR, copies

        t += 1
        scores.update(scores_t)
        WRRs.update(WRRs_t)
        original_nodes.update(original_nodes_t)
        RRs.update(RRs_t)

        roots.update(roots_t)
        S_N_u.update(S_N_u_t)
        s.update(s_t)
        m.update(m_t)
        copies.update(copies_t)
        keep.update(keep_t)
        S_N_WRR.update(S_N_WRR_t)
        # print(t,len(RRs),len({k: v for k, v in roots.items() if v is not None}))

    return keep, scores, WRRs, original_nodes, roots, RRs, S_N_u, s, m, S_N_WRR, copies


def CMIA_O(G, S_N, k, min_prob,
           cost_function="uniform"):  # we reduce the propagation to the maximum probability paths and choose node with biggest reduction in overall probability of infection
    start = timeit.default_timer()
    avg_deg = sum([G.out_degree(x) for x in G.nodes()]) / len(G.nodes())

    S_P = []
    Neg_range = set()  # nodes that could be reached by S_N with probability higer than min_prob
    DecInf = {}  # dictionary storing the contribution a node has if added to S_P
    MIA_prev_nodes = {}  # to keep track of the MIIAs without having to recompute them
    steps_v = {}
    proba_v = {}

    for u in tqdm(S_N, position=0, leave=True):
        MIOA, _, _ = dijkstra_MIA(u, G, G, False, min_prob, [], "negative")
        for v in list(set(MIOA.nodes()) - set(S_N)):
            Neg_range.add(v)  # contains the nodes that could get infected

    for u in tqdm(Neg_range, position=0,
                  leave=True):  # for each node that gets ifected  we compute how much the probability of infection value lowers if we add a node to the seed

        MIIA, steps, probs = dijkstra_MIA(u, G, G, True, min_prob, [],
                                          "negative")  # our candidates are the nodes that reach u with probability higher than min_prob
        MIA_prev_nodes[u] = [e for e in MIIA.edges()].copy()  # skeleton of MIIA
        steps_v[u] = {key: val for key, val in steps.items()}

        ap = compute_ap_N(u, S_N, [], MIIA, MIIA, G)  # probability of infection of  u if we dont intervene
        if ap == 0: continue
        longest = max([x for x in steps.items() if
                       x[0] in S_N and not path_blocked_MIA(x[0], u, MIIA, list(set(S_N) - set([x[0]])))],
                      key=lambda item: item[1])[0]
        for v in [x for x in list(MIIA.nodes()) if
                  steps[x] <= steps[longest] and not path_blocked_MIA(x, u, MIIA, S_N)]:
            ap_v = compute_ap_N(u, S_N, S_P + [v], MIIA, MIIA, G)  # probability of infection of u if v is added to S_P

            if v in DecInf:
                DecInf[v] += (ap - ap_v)  # compute total impact of the node as truth campaigner
            else:
                DecInf[v] = (ap - ap_v)
    while cost(S_P, cost_function, G, avg_deg) < k:
        print(cost(S_P, cost_function, G, avg_deg))
        # for i in tqdm(range(k), position=0, leave=True):

        candidates = {key: v for key, v in DecInf.items() if
                      key not in S_N + list(S_P) and cost(S_P + [key], cost_function, G, avg_deg) <= k}
        if len(candidates) == 0: break
        u = max(candidates, key=candidates.get)  # chose node with higest impact

        # we now proceed to update the contribution of each node
        MIOA, _, _ = dijkstra_MIA(u, G, G, False, min_prob, [], "negative")
        # print(u,list(MIOA.nodes()))
        for v in [x for x in MIOA.nodes() if x in Neg_range and not path_blocked_MOA(u, x, MIOA,
                                                                                     S_N + S_P)]:  # nodes that u could potentially save

            MIIA.clear()  # we recontruct the MIIA, which we already computed and stored
            MIIA.add_edges_from(MIA_prev_nodes[v])

            longest = max([x for x in steps_v[v].items() if x[0] in S_N and not path_blocked_MIA(x[0], v, MIIA, S_P)],
                          key=lambda item: item[1], default=(None, None))[0]
            if longest == None: continue

            ap = compute_ap_N(v, S_N, S_P, MIIA, MIIA, G)  # probability of infection of v with current S_P
            ap_u = compute_ap_N(v, S_N, S_P + [u], MIIA, MIIA,
                                G)  # probability of infection of v with current S_P which now includes u

            for w in [x for x in list(MIIA.nodes()) if
                      steps_v[v][x] <= steps_v[v][longest] and not path_blocked_MIA(x, v, MIIA,
                                                                                    S_N + S_P)]:  # nodes whose contribution now decreses since u helps
                ap_w = compute_ap_N(v, S_N, S_P + [w], MIIA, MIIA,
                                    G)  # probability of infection of v if w is added to S_P

                if w in DecInf:
                    DecInf[w] -= (ap - ap_w)  # we update the contribution of w to be the one after we add u
                if v != u:
                    ap_u_w = compute_ap_N(v, S_N, S_P + [u] + [w], MIIA, MIIA,
                                          G)  # probability of infection of v if w is added to S_P which now includes u

                    if w in DecInf:
                        DecInf[w] += (ap_u - ap_u_w)
        S_P.append(u)
        print("S_P", S_P)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return S_P


def BIOG(G, S_N, k, min_prob,
         cost_function="uniform"):  # we now consider the communities, and take as input the partition
    avg_deg = sum([G.out_degree(x) for x in G.nodes()]) / len(G.nodes())
    start = timeit.default_timer()
    S_P = []
    MIOGs_N, MIOGs_P = {}, {}
    MIOGs_ap = {}
    contribution = {}
    candidates_s = {}

    candidates = set()
    candidates_s = {}
    time, time1, time2 = 0, 0, 0
    for s in tqdm(S_N, position=0, leave=True):
        s1 = timeit.default_timer()
        MIOG = dijkstra_MIOG(s, G, min_prob, list(set(S_N) - set([s])))
        # print(len(MIOG.nodes()))
        time += timeit.default_timer() - s1
        MIOGs_N[s] = [e for e in MIOG.edges()].copy()
        MIOGs_P[s] = [e for e in MIOG.edges()].copy() + [(w, u) for u in MIOG.successors(s) for w in G.predecessors(u)]
        MIOG_P = nx.DiGraph()
        MIOG_P.add_edges_from(MIOGs_P[s])
        # print(len(MIOG_P.nodes()))

        steps1 = [x for x in MIOG.successors(s) if x not in S_N]
        steps2 = [x for u in steps1 for x in G.predecessors(u) if x not in S_N]
        candidates.update(steps1 + steps2)
        candidates_s[s] = set(steps1 + steps2)
        MIOGs_ap[s] = compute_ap_N_MIOG(S_N, [], MIOG, MIOG_P, G)

        for candidate in candidates_s[s]:
            if candidate not in contribution.keys(): contribution[candidate] = 0
            first, second = 0, 0
            if candidate in MIOG.nodes():
                for v in list(nx.descendants(MIOG, candidate)) + [candidate]:
                    contribution[candidate] += MIOGs_ap[s][v]
                    first += MIOGs_ap[s][v]
            common = list(set(MIOG_P.successors(candidate)) & set(set(MIOG.successors(s))))
            for (w, v) in [(w, v) for v in common if v != candidate for w in list(nx.descendants(MIOG, v)) + [v] if
                           w != candidate]:
                contribution[candidate] += MIOGs_ap[s][w] * G[candidate][v]["positive"]
                second += MIOGs_ap[s][w] * G[candidate][v]["positive"]

    pbar = tqdm(total=k, position=0, leave=True)
    while cost(S_P, cost_function, G, avg_deg) < k:
        options = {n: contribution[n] for n in candidates if cost(S_P + [n], cost_function, G,
                                                                  avg_deg) <= k}  # we store for each node the min percentage of saved nodes in a community if added to S_P

        if not options: break
        u = max(options,
                key=options.get)  # chose node maximin value that has the biggest contribution within the communities with min saved percentage
        S_P.append(u)
        pbar.update(1)
        candidates.remove(u)

        for s in S_N:  # updates
            MIOG = nx.DiGraph()
            MIOG.add_edges_from(MIOGs_N[s])
            MIOG_P = nx.DiGraph()
            MIOG_P.add_edges_from(MIOGs_P[s])
            if u not in MIOG_P.nodes(): continue
            ap_before = MIOGs_ap[s].copy()
            MIOGs_ap[s][u] = 0
            contribution[u] = 0
            N_up = []

            common = list(set(MIOG_P.successors(u)) & set(set(MIOG.successors(s))))
            for v in common:
                MIOGs_ap[s][v] *= (1 - G[u][v]["positive"])

            if u in MIOG.nodes(): common + [u]
            next_level_common = [v for w in common for v in MIOG.successors(w)]

            while next_level_common:
                for v in next_level_common:
                    p = 1 - np.prod([1 - MIOGs_ap[s][node] * G[node][v]["negative"] for node in MIOG.predecessors(v)])
                    MIOGs_ap[s][v] = p * (
                                1 - np.sum([G[node][v]["positive"] for node in MIOG.predecessors(v) if node in S_P]))
                    N_up += [x for x in nx.ancestors(MIOG_P, v) if x in candidates_s[s]]

                next_level_common = [v for w in next_level_common for v in MIOG.successors(w)]

            for candidate in set(N_up) - set(S_P):
                if candidate in MIOG.nodes():
                    for v in list(nx.descendants(MIOG, candidate)) + [candidate]:
                        contribution[candidate] += MIOGs_ap[s][v] - ap_before[v]

                common = list(set(MIOG_P.successors(candidate)) & set(set(MIOG.successors(s))))
                for (w, v) in [(w, v) for v in common if v != candidate for w in nx.descendants(MIOG, v) if
                               w != candidate]:
                    contribution[candidate] += MIOGs_ap[s][w] * G[candidate][v]["positive"] - ap_before[w] * \
                                               G[candidate][v]["positive"]

    pbar.close()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return S_P


def TIB_Solver(G, S_N, k, cost_function="uniform"):
    max_depth = max(util.compute_proximity(G, S_N).values())
    print("max_depth", max_depth)
    start = timeit.default_timer()
    S_P = []
    threat_levels, S, DAG = compute_threat_levels(G, S_N)
    print("threat_levels computed")
    if cost_function == "degree_penalty":
        k_orig = int(k / int(sum([G.out_degree(x) for x in G.nodes()]) / len(G.nodes())))
    else:
        k_orig = k
    # print(sorted(list(threat_levels.keys())))
    keep, scores, WRRs, original_nodes, roots, RRs, ancestors_u, s, m, S_N_WRR, copies = D_SSA_TIB(G, S, S_N, k_orig,
                                                                                                   0.1, 1 / len(S),
                                                                                                   threat_levels, False,
                                                                                                   max_depth)
    print("D_SSA_TIB computed")
    WRRs_to_keep = [i for i in range(len(roots)) if roots[i] is not None]
    # print(len(WRRs_to_keep),len({k: v for k, v in roots.items() if v is not None}))

    scores = {key: scores[key] for key in WRRs_to_keep}
    WRRs = {key: WRRs[key] for key in WRRs_to_keep}
    original_nodes = {key: original_nodes[key] for key in WRRs_to_keep}
    roots = {key: roots[key] for key in WRRs_to_keep}
    RRs = {key: RRs[key] for key in WRRs_to_keep}
    m = {key: m[key] for key in WRRs_to_keep}
    s = {key: s[key] for key in WRRs_to_keep}
    copies = {key: copies[key] for key in WRRs_to_keep}
    ancestors_u = {key: ancestors_u[key] for key in WRRs_to_keep}
    # print(len(WRRs_to_keep),len(roots))
    starting = timeit.default_timer()

    # print(sorted(total_scores.keys()))
    pbar = tqdm(total=k, position=0, leave=True)
    while cost(S_P, cost_function, G) < k:
        # print(len(WRRs_to_keep))
        s0 = timeit.default_timer()
        options = defaultdict(lambda: 0)
        best = (None, None, 0)
        for i in WRRs_to_keep:
            for key in scores[i].keys():
                if scores[i][key] <= best[2] or cost(S_P + [key], "uniform", G) > k: continue
                if original_nodes[i][key] not in S_N + S_P:
                    best = (key, i, scores[i][key])
            options[best[0]] = best[2]

        # options=defaultdict(lambda: 0)
        # for j in WRRs_to_keep:
        # for v in scores[j].keys():
        # if v not in S_N+S_P:
        # options[v]+=scores[j][v]
        # options={key: np.sum([scores[i][key]  for i in WRRs_to_keep if key in scores[i].keys() and key not in S_P]) for key in list(G.nodes())}
        # print([(i,options[i])for i in S_P])
        # print(options)
        # print(timeit.default_timer()-s0)
        # u,i,val = best

        # print(u,i,val)
        u = max(options, key=options.get)
        # print("u ------> "+str(u)+" "+str(options[u])+" "+str(timeit.default_timer()-starting))
        threat_levels_before_u = threat_levels.copy()
        S_P.append(u)
        pbar.update(1)
        starting = timeit.default_timer()
        sstep1 = timeit.default_timer()
        next_level = set()
        for v in DAG.predecessors(u):
            if v not in S_N:
                q_v_u = G[v][u]["negative"] * np.prod(
                    [1 - G[parent][u]["negative"] for parent in list(G.predecessors(u)) if parent != v])
                t = 1
                while threat_levels[v][t] != 0:
                    threat_levels[v][t] -= q_v_u * threat_levels[u][t - 1]
                    t += 1
                # print(v,dict(threat_levels[v]))
                next_level.update(list(DAG.predecessors(v)))
        # print("step 1  half time" +str(timeit.default_timer()-sstep1))
        while next_level:
            # print(list(next_level))
            future_level = set()
            for w in next_level:
                future_level.update(list(DAG.predecessors(w)))
                if w in S_N + S_P: continue
                q_u = {}
                for v in DAG.successors(w):
                    q_u[v] = G[w][v]["negative"] * np.prod(
                        [1 - G[parent][v]["negative"] for parent in list(G.predecessors(v)) if parent != w])
                t = 1
                threat_levels[w][t] = 0
                while any(threat_levels[x][t - 1] != 0 for x in DAG.successors(w)):
                    for v in DAG.successors(w): threat_levels[w][t] -= q_u[v] * threat_levels[v][t - 1]
                    t += 1
            next_level = future_level

        threat_levels[u] = defaultdict(lambda: 0)
        # print("step 1 time" +str(timeit.default_timer()-sstep1))

        # print("Step 2:"+str(len([j for j in WRRs_to_keep if roots[j] in nx.ancestors(DAG,u)])))

        sstep2 = timeit.default_timer()
        time_with_totals, rest = 0, 0
        parents_u = nx.ancestors(DAG, u)
        for j in WRRs_to_keep:
            s_totals = timeit.default_timer()
            if roots[j] not in parents_u: continue
            sum_previous_threat = 0
            t = 0
            while threat_levels_before_u[roots[j]][t] != 0: sum_previous_threat += threat_levels_before_u[roots[j]][
                t]; t += 1

            sum_threat = 0
            t = 0
            while threat_levels[roots[j]][t] != 0: sum_threat += threat_levels[roots[j]][t]; t += 1
            time_with_totals += (timeit.default_timer() - s_totals)
            for w in scores[j]:
                st = timeit.default_timer()
                if w == u:
                    scores[j][w] = 0
                    continue
                scores[j][w] *= (sum_threat / sum_previous_threat)
                # w_original= original_nodes[j][w]
                # num_copies=copies[j][w_original]
                # copies_w=[str(w)+"copy"+str(i) for i in range(1,num_copies+1)]+[w]
                # ancestors_total=set(x for copy in copies_w for x in ancestors_u[j][copy] )
                # steps_w=set(s[j][copy] for copy in copies_w)

                # scores[j][w]=0
                # beta=defaultdict(lambda: 0)
                # for t in steps_w:
                # beta[t]=(1-np.prod([1-m[j][w_c] for w_c in copies_w if s[j][w_c]==t]))
                # beta[t]*=np.prod([(1-m[j][s_n]) for s_n in list(set(S_N_WRR[j])-set(ancestors_total)) if s[j][s_n]<t])
                # scores[j][w]+=beta[t]*sum_threat
                rest += timeit.default_timer() - st
        # print("step 2 time" +str(timeit.default_timer()-sstep2))

        s_inter = timeit.default_timer()
        to_remove = [x for x in WRRs_to_keep if roots[x] == u]
        WRRs_to_keep = list(set(WRRs_to_keep) - set(to_remove))
        # print(timeit.default_timer()-s_inter)

        time1, time2, time3, time4 = 0, 0, 0, 0
        sstep3 = timeit.default_timer()
        # print("Step 3:"+str(len([j for j in WRRs_to_keep if u in RRs[j] and roots[j] not in nx.ancestors(DAG,u)])))
        for j in [j for j in WRRs_to_keep if u in RRs[j] and roots[j] not in list(nx.ancestors(DAG, u)) + [u]]:
            # print("j"+str(j))
            s4 = timeit.default_timer()

            S_P_T = []
            steps_S_P = {x: s[j][x] for x in RRs[j] if original_nodes[j][x] in S_P}
            # print(steps_S_P)
            for v, _ in sorted(steps_S_P.items(), key=lambda x: x[1], reverse=False):
                if list(set(S_P_T) & ancestors_u[j][v]): continue
                S_P_T.append(v)

            Z_S_N_u = set(S_N_WRR[j]) - set([x for v in S_P_T for x in ancestors_u[j][v]])
            steps_Z = [s[j][x] for x in S_P_T]
            beta_Z = defaultdict(lambda: 0)
            # print(Z_S_N_u,steps_S_P,steps_Z)

            for t in set(steps_Z):
                beta_Z[t] = (1 - np.prod([(1 - m[j][v]) for v in S_P_T if s[j][v] == t])) * np.prod(
                    [(1 - m[j][v]) for v in Z_S_N_u if s[j][v] < t])

            time4 += timeit.default_timer() - s4

            scores[j][u] = 0
            for w in scores[j]:
                # print("initially"+str(scores[j][w]))
                if w not in S_N_WRR[j] and w not in list(S_P):
                    # print(w)
                    s1 = timeit.default_timer()
                    original_w = original_nodes[j][w]

                    S_P_w_T = S_P_T.copy()
                    can_we_add_w = True
                    for v in [v for v in S_P_T if steps_S_P[v] < s[j][w]]:
                        if v in list(ancestors_u[j][w]):
                            can_we_add_w = False
                            break
                    if can_we_add_w:
                        S_P_w_T.append(w)
                        for v in [v for v in S_P_T if s[j][v] >= s[j][w]]:
                            if v in list(ancestors_u[j][w]):
                                S_P_w_T.remove(v)

                                # print("S_P"+str(S_P_T)+" "+str(S_P_w_T))
                    if S_P_T == S_P_w_T:
                        scores[j][w] = 0
                        continue

                    Z_w_S_N_u = set(S_N_WRR[j]) - set([x for v in S_P_w_T for x in ancestors_u[j][v]])
                    # print("Z"+str(Z_S_N_u)+" "+str(Z_w_S_N_u))

                    steps_Z_w = [s[j][x] for x in S_P_w_T]
                    # print(Z_w_S_N_u,steps_Z_w)

                    # print(Z_S_N_u,Z_w_S_N_u)
                    beta_Z_w = defaultdict(lambda: 0)

                    time1 += timeit.default_timer() - s1

                    s2 = timeit.default_timer()

                    for t in set(steps_Z_w):
                        beta_Z_w[t] = (1 - np.prod([(1 - m[j][v]) for v in S_P_w_T if s[j][v] == t])) * np.prod(
                            [(1 - m[j][v]) for v in Z_w_S_N_u if s[j][v] < t])
                    # print(beta_Z,beta_Z_w)
                    time2 += timeit.default_timer() - s2

                    s3 = timeit.default_timer()

                    scores[j][w] = 0
                    sum_threat = 0
                    t = 0
                    while threat_levels[roots[j]][t] != 0: sum_threat += threat_levels[roots[j]][t]; t += 1

                    t = 0
                    while (beta_Z[t] != 0 or beta_Z_w[t] != 0):
                        # print(t,beta_Z_w[t]-beta_Z[t])
                        scores[j][w] += (beta_Z_w[t] - beta_Z[t]) * sum_threat
                        t += 1
                    # print("updated"+str(scores[j][w]))
                    time3 += timeit.default_timer() - s3
        # print("step3 time"+str(timeit.default_timer()-sstep3))
    pbar.close()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return S_P
