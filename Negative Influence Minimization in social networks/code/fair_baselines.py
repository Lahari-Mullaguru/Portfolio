"""
This file contains the implementation of group fairness baselines taken from
https://github.com/cristinagub/master_project
"""
from decimal import Decimal
from collections import Counter, defaultdict
from tqdm import tqdm
import heapq as heap
from community import community_louvain
import networkx as nx
import numpy as np
import timeit
import random
import math
import util

def construct_WRR(G, S, p, S_N, paths):
    max_depth = float('inf')
    WRR = set()
    A, s, m, original_node, copies, ancestors_u, descendants_u, contribution = [], {}, {}, {}, defaultdict(
        lambda: 0), defaultdict(lambda: set()), defaultdict(lambda: set()), {}
    r = random.choices(list(S), weights=p, k=1)[0]
    S_N_WRR = set()
    WRR.add(r)
    A.append((r, 0))
    original_node[r] = r
    prev_v = defaultdict(lambda: [])
    while A:
        v, l = A.pop(0)
        s[v], m[v] = l, 0
        if v == r:
            m[v] = 1
        else:  # compute mv
            v_original = original_node[v]
            c = prev_v[v]
            m[v] += G[v_original][original_node[c]]["negative"] * m[c]

        if original_node[v] not in S_N:
            if l >= max_depth: continue

            parents = list(G.pred[original_node[v]].keys())

            connections = [u for u in parents if random.random() <= G[u][original_node[v]]["negative"]]

            for u in connections:
                if u in WRR:
                    if u in descendants_u[v]: continue
                    if not any([set(list(descendants_u[v]) + [v]).issubset(x) for x in paths[r]]) and copies[
                        u] > 9: print("ups");continue
                    # add a copy of the node
                    copies[u] += 1
                    original_node[str(u) + "copy" + str(copies[u])] = u
                    # WRR.add_edge(str(u)+"copy"+str(copies[u]),v)
                    descendants_u[str(u) + "copy" + str(copies[u])].update(list(descendants_u[v]) + [v])
                    A.append((str(u) + "copy" + str(copies[u]), s[v] + 1))
                    prev_v[str(u) + "copy" + str(copies[u])] = v
                else:
                    # WRR.add_edge(u,v)
                    WRR.add(u)
                    descendants_u[u].update(list(descendants_u[v]) + [v])
                    A.append((u, s[v] + 1))
                    original_node[u] = u
                    prev_v[u] = v
        else:
            S_N_WRR.add(v)
            max_depth = l

        for node in descendants_u[v]:
            ancestors_u[node].update([v])

    prob_inf = 1 - np.prod([1 - m[s_n] for s_n in S_N_WRR])
    if prob_inf != 0:
        # max_distance_S_N=max([s[s_n] for s_n in S_N_WRR])
        max_prob = max([m[s_n] for s_n in S_N_WRR])
        # prob_inf=compute_ap_N_WRR(r,list(S_N_WRR),[],WRR,WRR,G,original_node)
        for u in WRR:
            if not str(u).isnumeric() or m[u] < max_prob or u in S_N_WRR: continue
            if False:
                # if not str(u).isnumeric() or s[u]> max_distance_S_N or u in S_N_WRR: continue
                num_copies = copies[u]
                copies_u = [str(u) + "copy" + str(i) for i in range(1, num_copies + 1)] + [u]
                ancestors_total = set(x for copy in copies_u for x in ancestors_u[copy])
                S_N_u = list(set(S_N_WRR) - set(ancestors_total))

                # prob_inf_u=compute_ap_N_WRR(r,S_N_u,copies_u,WRR,WRR,G,original_node)
                prob_inf_u = (1 - np.prod([1 - m[s_n] for s_n in S_N_u])) * (np.prod([1 - m[u_c] for u_c in copies_u]))

                contribution[u] = (prob_inf - prob_inf_u) / prob_inf
            else:
                contribution[u] = 1
        return True, r, 1, WRR, contribution, original_node, ancestors_u, s, m, S_N_WRR, copies
    return False, r, 0, WRR, None, original_node, ancestors_u, s, m, None, None


def Max_Coverage_WRS(RRs, roots, k, G, S_N, S, keep, cost_function, avg_deg):
    start = timeit.default_timer()
    S_k = set()
    # need_covering=[x for x,r in enumerate(RRs) if keep[r] ]
    need_covering = [x for x, r in enumerate(RRs)]
    num_RRs = len(need_covering)
    cost_so_far = 0
    count_roots_total = Counter(roots.values())
    coverage = {v: 0 for v in G.nodes()}
    for j in RRs.keys():
        for v in RRs[j]:
            coverage[v] += 1 / count_roots_total[roots[j]]
    Cov_S_k = 0
    while len(S_k) < k:

        u = max(coverage, key=coverage.get)
        # cost_so_far+= cost([u],cost_function,G,avg_deg)
        # print(u,options[u],len(need_covering))
        for j in [j for j in need_covering if u in RRs[j]]:
            for v in set(RRs[j]):
                coverage[v] -= 1 / count_roots_total[roots[j]]
        need_covering = set(need_covering) - set([j for j in need_covering if u in RRs[j]])
        S_k.add(u)
        Cov_S_k += coverage[u]
    # print(Cov_S_k,len([x for x,r in enumerate(RRs) if keep[r]]),len(RRs))

    return S_k, Cov_S_k


def construct_WFR(G, S_N):
    A, S = [], set()
    r = "new"
    A.append(r)
    first = True
    original = {r: r}
    path = {r: []}
    while A:
        v = A.pop(0)
        childreen = iter(G[original[v]])
        connections = [u for u in childreen if random.random() <= G[original[v]][u]["negative"]]

        for u in connections:
            if (u not in S_N or first):
                if u not in S:
                    path[u] = path[v] + [v]
                    A.append(u)
                    S.add(u)
                    original[u] = u
        first = False
    return S, path


def WFRS(G, S_N):
    # f = Fraction(threshold).limit_denominator()
    # denom, num = f.denominator, f.numerator
    # print(num, denom)
    S = set()
    probs = defaultdict(lambda: 0)
    count = 0
    paths = defaultdict(lambda: [])
    for i in tqdm(range(1000), position=0, leave=True):
        S_new, path = construct_WFR(G, S_N)
        S.update(S_new)
        count += 1
        for s in S_new: probs[s] += 1; paths[s].append(path[s])
        if len(S) == len(G.nodes()): break
    for s in S: probs[s] *= 1 / count
    S = [s for s in S if probs[s] >= 0.01]
    return S, probs, paths


def ipsilon_WRS(e, d):
    # print(type(e),type(d))
    return ((2 + Decimal(2 / 3) * e) * ((1 / d).ln()) * (1 / (e ** 2)))


def D_SSA_WRS(G, S, S_N, k, e, d, probs, cost_function, avg_deg, dis, paths):
    exp_inf = np.ceil(np.sum([probs[x] for x in S]))
    print(len(S), exp_inf, exp_inf / len(S), k)
    S_orig = S.copy()
    n = len(S)
    d = Decimal(1 / n)
    e = Decimal(e)
    choose = (Decimal(math.factorial(n)) / (Decimal(math.factorial(n - k)) * Decimal(math.factorial(k))))
    d_prime = d / 6 / choose

    N_max = 8 * (Decimal(1 - 1 / math.e) / (2 + 2 * e / 3)) * (ipsilon_WRS(e, d_prime)) * Decimal(n / k)
    # print(N_max)
    t_max = math.ceil((2 * N_max / (ipsilon_WRS(e, d / 3))).log10() / Decimal(2).log10())
    t = 1
    Lambda = round(ipsilon_WRS(e, d / (3 * t_max)))
    Lambda_1 = 1 + (1 + e) * ipsilon_WRS(e, d / (3 * t_max))

    keep, prob_inf_r, contributions, WRRs, original_nodes, RRs, roots, S_N_u, s, m, S_N_WRR, copies = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    print(Lambda_1, t_max)

    # p=[1/len(S)]*len(S)
    p = [dis[x] for x in S]
    i = 0
    pbar = tqdm(total=Lambda * 2 ** (t - 1), position=0, leave=True)
    while len([x for x, v in keep.items()]) < Lambda * 2 ** (t - 1):
        # print(len([x for x,v in keep.items() if v]),Lambda*2**(t-1))
        keep[i], roots[i], prob_inf_r[i], WRR, contributions[i], original_nodes[i], S_N_u[i], s[i], m[i], S_N_WRR[i], \
        copies[i] = construct_WRR(G, S, p, S_N, paths)
        WRRs[i] = [e for e in WRR].copy()
        RRs[i] = [n for n in WRR].copy()
        # if keep[i]:pbar.update(1)
        pbar.update(1)
        i += 1
    pbar.close()
    RRs_kept = [x for x in keep]
    RRs_kept_true = [x for x in keep if keep[x]]
    # print(len(RRs),len(RRs_kept))

    while len(RRs_kept) < N_max:
        RRs_kept = [x for x in keep]
        RRs_kept_true = [x for x in keep if keep[x]]
        keep_t, prob_inf_r_t, contributions_t, WRRs_t, original_nodes_t, RRs_t, roots_t, S_N_u_t, s_t, m_t, S_N_WRR_t, copies_t = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        i = len(keep)
        pbar = tqdm(total=Lambda * 2 ** (t - 1), position=0, leave=True)
        while len([x for x, v in keep_t.items()]) < Lambda * 2 ** (t - 1):
            # print(len([x for x,v in keep_t.items() if v]),Lambda*2**(t))
            keep_t[i], roots_t[i], prob_inf_r_t[i], WRR_t, contributions_t[i], original_nodes_t[i], S_N_u_t[i], s_t[i], \
            m_t[i], S_N_WRR_t[i], copies_t[i] = construct_WRR(G, S, p, S_N, paths)

            WRRs_t[i] = [e for e in WRR_t].copy()
            RRs_t[i] = [n for n in WRR_t].copy()
            # if keep_t[i]:pbar.update(1)
            pbar.update(1)
            i += 1

        pbar.close()
        RRs_t_kept_true = [x for x in keep_t if keep_t[x]]
        RRs_t_kept = [x for x in keep_t]
        print(len(RRs_t), len(RRs_t_kept))
        S_k, I = Max_Coverage_WRS(RRs, roots, k, G, S_N, S, keep, cost_function, avg_deg)
        # I*=len([x for x in keep if x])/len(keep)
        # Cov_S_k_t =np.sum([1 if ( any([j in RRs_t[i] for j in S_k]) ) else 0 for i,v in keep_t.items() if v])

        count_roots_total_t = Counter(roots_t.values())
        count_roots_total = Counter(roots.values())
        Cov_S_k_t = np.sum(
            [1 / count_roots_total_t[roots_t[i]] if (any([j in RRs_t[i] for j in S_k])) else 0 for i, v in
             keep_t.items()])
        print("cov is ", Cov_S_k_t * len(RRs_t) / n)
        if Cov_S_k_t * len(RRs_t) / n >= Lambda_1:
            I_t = Cov_S_k_t
            e1 = Decimal((I / I_t) - 1)
            e2 = e * np.sqrt(Decimal(n * (1 + e)) / Decimal(2 ** (t - 1) * I_t))
            e3 = e * np.sqrt(Decimal(n * (1 + e) * (Decimal(1 - 1 / math.e) - e)) / Decimal(
                (1 + e / 3) * 2 ** (t - 1) * Decimal(I_t)))
            e_t = (e1 + e2 + e1 * e2) * (Decimal(1 - 1 / math.e) - e) + Decimal(1 - 1 / math.e) * e3
            print(e1, e2, e3, e_t)
            if e_t <= e:
                # Cov_S_k_t_s =np.sum([1 if ( any([j in RRs[i] for j in S_k]) ) else 0 for i in RRs_kept_true])
                Cov_S_k_s = np.sum(
                    [1 / count_roots_total[roots[i]] if (any([j in RRs[i] for j in S_N])) else 0 for i, v in
                     keep.items()])
                print(Cov_S_k_s, Lambda_1)
                print(Cov_S_k_s * len(RRs) / n)
                if Cov_S_k_s * len(RRs) / n >= Lambda_1:
                    return keep, contributions, WRRs, original_nodes, roots, RRs, S_N_u, s, m, S_N_WRR, copies, prob_inf_r, e_t
                Cov_S_t_k_s = np.sum(
                    [1 / count_roots_total_t[roots_t[i]] if (any([j in RRs_t[i] for j in S_N])) else 0 for i, v in
                     keep_t.items()])

                if Cov_S_k_s * len(RRs) / n + Cov_S_t_k_s * len(RRs_t) / n >= Lambda_1:
                    contributions.update(contributions_t)
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
                    prob_inf_r.update(prob_inf_r_t)
                    return keep, contributions, WRRs, original_nodes, roots, RRs, S_N_u, s, m, S_N_WRR, copies, prob_inf_r, e_t

        t += 1

        contributions.update(contributions_t)
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
        prob_inf_r.update(prob_inf_r_t)
    return keep, contributions, WRRs, original_nodes, roots, RRs, S_N_u, s, m, S_N_WRR, copies, prob_inf_r, e_t


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


def compute_ap_total(S_N, G, DAG):
    NEG_t = [v for v in S_N if v in DAG.nodes()]
    NEG_next = set()

    P_n_t = {}
    P_n_next = {}

    ap_n_t = {}
    for node in list(DAG.nodes()) + list(DAG.nodes()):
        P_n_t[node], ap_n_t[node] = get_p_value(node, False, 0, S_N, [])
    t = 0
    while NEG_t:
        ap_n_next = ap_n_t.copy()
        temp_N = {k: 1 for k in list(DAG.nodes())}

        for v in NEG_t:
            if not list(DAG.successors(v)):
                continue
            for w in list(DAG.successors(v)):
                NEG_next.add(w)
                temp_N[w] *= (1 - (P_n_t[v] * G[v][w]['negative']))

        for v in NEG_next:
            P_n_next[v] = (1 - temp_N[v]) * (1 - ap_n_t[v])
            ap_n_next[v] = ap_n_t[v] + P_n_next[v]

        if list(NEG_next):
            ap_n_t = ap_n_next.copy()
        P_n_t, NEG_t = P_n_next, NEG_next
        NEG_next = set()
        t += 1

    return ap_n_t

def get_communities(part,G): # returns the number of communities in G given the partition part
    communities = []
    for i in G.nodes():
        if part[i] not in communities:
            communities.append(part[i])
    return len(communities)



def cost(S_P,funct,G,avg_deg=1):
    if funct=="uniform":
        return len(S_P)
    if funct=="degree_penalty":
        if len(S_P)==0: return 0
        return np.sum([G.out_degree(x) for x in S_P])/(avg_deg)


def WRS_method(G, S_N, k, partition, cost_function="uniform"):
    start = timeit.default_timer()
    avg_deg = sum([G.out_degree(x) for x in G.nodes()]) / len(G.nodes())
    G_extended = G.copy()
    G_extended.add_edges_from([("new", x) for x in S_N])
    for x in S_N: G_extended["new"][x]["negative"] = 1
    S, probs, paths = WFRS(G_extended, S_N)
    paths_blocked = defaultdict(lambda: 0)
    S = list(set(S) - set(S_N + ["new"]))
    d = {x: probs[x] for x in S}
    paths = {x: paths[x] for x in S}

    S_P = []
    e = 0.1
    keep, contributions, WRRs, original_nodes, roots_all, RRs, ancestors_u, s, m, S_N_WRR, copies, prob_inf_r, e_new \
        = D_SSA_WRS(G, S, S_N, k, e, 1 / len(G), probs, cost_function, avg_deg, d, paths)
    del WRRs, ancestors_u, S_N_WRR, copies, prob_inf_r, e_new

    WRRs_to_keep = [i for i, b in enumerate(keep.values()) if b]
    WRRs_to_remove = [i for i, b in enumerate(keep.values()) if not b]
    del keep

    roots = {key: roots_all[key] for key in WRRs_to_keep}
    count_roots = Counter(roots.values())
    count_roots_total = Counter(roots_all.values())

    RRs_copies = RRs.copy()
    prob_infections = {x: count_roots[x] / count_roots_total[x] for x in count_roots.keys()}
    current_prob_infections = defaultdict(lambda: 0)
    inform_power = defaultdict(lambda: 0)

    initial_infections = {}

    for x in count_roots.keys():
        current_prob_infections[x] = count_roots[x] / count_roots_total[x]
        if partition[x] not in initial_infections.keys(): initial_infections[partition[x]] = 0
        initial_infections[partition[x]] += prob_infections[x]

    expected_inf = initial_infections.copy()
    communities_to_save = [x for x in initial_infections.keys() if initial_infections[x] > 0]
    Gcc = sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)
    G_sub = G.subgraph(Gcc[0])

    com_g_sub = set([partition[x] for x in G_sub.nodes()])

    min_help_c = communities_to_save.copy()
    candidates = list(G.nodes())
    maximin_value = 0
    first = True

    maximin_c = {c: (initial_infections[c] - expected_inf[c]) / initial_infections[c] for c in communities_to_save}
    total_inform = defaultdict(lambda: 0)
    while cost(S_P, cost_function, G, avg_deg) < k:
        max_c = [c for c in communities_to_save if expected_inf[c] == max(expected_inf.values())][0]
        s2 = timeit.default_timer()
        print("num candidates  ", len(candidates))
        no_help = [c for c in communities_to_save if
                   (initial_infections[c] - expected_inf[c]) / initial_infections[c] == 0]
        need_saving = [c for c in communities_to_save if
                       (initial_infections[c] - expected_inf[c]) / initial_infections[c] <= maximin_value * (
                                   1 + e) + len(no_help) / len(communities_to_save) * e]

        print(len(need_saving), len(communities_to_save), maximin_value)
        print([(initial_infections[c], expected_inf[c], maximin_c[c]) for c in communities_to_save])

        cost_u = {}
        if first:
            inf_u = {x: expected_inf.copy() for x in candidates}
            candidates_next = set()
            for j in WRRs_to_keep:
                for key in contributions[j].keys():
                    candidates_next.add(key)
                    inf_u[key][partition[roots[j]]] -= 1 / count_roots_total[roots[j]]
                    total_inform[key] += 1 / count_roots_total[roots[j]]

            for j in WRRs_to_remove:
                for key in RRs[j]:
                    if not str(key).isnumeric(): continue
                    candidates_next.add(key)
                    total_inform[key] += 1 / count_roots_total[roots_all[j]]
            print(sorted(paths_blocked.items(), key=lambda x: x[1], reverse=True))
            candidates = candidates_next

        maximin_c = {c: (initial_infections[c] - expected_inf[c]) / initial_infections[c] for c in communities_to_save}

        maximin_pev = min(
            [(initial_infections[c] - expected_inf[c]) / initial_infections[c] for c in communities_to_save])

        maximin_u = {
            x: min([(initial_infections[c] - inf_u[x][c]) / initial_infections[c] for c in communities_to_save]) for x
            in candidates}
        max_value = max([maximin_u[x] for x in maximin_u.keys()])
        u_maximin = max(maximin_u, key=maximin_u.get)
        print("maximin max:", max_value)
        maximins_u_m = {c: (initial_infections[c] - inf_u[u_maximin][c]) / initial_infections[c] for c in
                        communities_to_save}
        c_s = [k for k, x in maximins_u_m.items() if x == max_value]
        c = c_s[0]
        max_inf = max([expected_inf[c] for c in communities_to_save])
        c_s = [c for c in communities_to_save if expected_inf[c] == max_inf]
        c = c_s[0]
        exp_total = np.sum([expected_inf[c] for c in communities_to_save])
        print((c, initial_infections[c], exp_total))
        u_s = [key for key, value in maximin_u.items() if
               value >= max_value * (1 - e * (1 - expected_inf[c] / exp_total))]

        print("1: ", len(u_s))
        if len(u_s) > 1:

            avg_help = np.mean(
                [(initial_infections[c] - expected_inf[c]) / initial_infections[c] for c in communities_to_save])

            need_saving_2 = [c for c in communities_to_save if
                             (initial_infections[c] - expected_inf[c]) / initial_infections[c] <= avg_help * (1 + e)]
            tot = {x: np.sum([(expected_inf[c] - inf_u[x][c]) for c in communities_to_save]) for x in u_s}

            if len(no_help) > 0:
                inf_need_saving = {x: np.sum([(expected_inf[c] - inf_u[x][c]) for c in no_help]) for x in u_s}
                exp_no_help = np.sum([expected_inf[c] for c in no_help])
                u_s = [x for x in u_s if inf_need_saving[x] > 0 or exp_no_help < e * exp_total]
                dis_c = [x for x in no_help if x not in com_g_sub]
                if len(dis_c) > 0:
                    inf_need_saving = {x: np.sum([(expected_inf[c] - inf_u[x][c]) for c in dis_c]) for x in u_s}
                    u_s = [x for x in u_s if inf_need_saving[x] > 0]
            else:
                inf_need_saving = {x: 1 for x in u_s}

            inf_need_saving_2 = {
                x: np.sum([(expected_inf[c] - inf_u[x][c]) * (avg_help - maximin_c[c]) for c in need_saving_2]) for x in
                u_s}

            max_saved = max(inf_need_saving_2.values())
            exp_i = np.sum([expected_inf[c] for c in need_saving_2])
            init_i = np.sum([initial_infections[c] for c in need_saving_2])
            options_u = [key for key, value in inf_need_saving_2.items() if
                         value >= (1 - e) * max_saved - max(init_i - exp_i, 0) * avg_help]

            inf_need_saving = {x: np.sum([max(expected_inf[c] - inf_u[x][c], 0) for c in need_saving]) for x in
                               options_u}
            max_saved_tot = max([inf_need_saving[x] for x in options_u])
            options_u_2 = [key for key, value in inf_need_saving.items() if
                           value >= (1 - e) * max_saved_tot - e * max(init_i - exp_i, 0)]
            print("3: ", len(options_u_2))

            if max([tot[x] for x in options_u_2]) < max(tot.values()) * (1 - e):
                sav_tot = {x: np.sum([inf_u[x][c] for c in communities_to_save]) for x in options_u_2}
                opts = [x for x, v in sav_tot.items() if v <= min(sav_tot.values()) * (1 + e)]
                sav_u = {x: np.sum([current_prob_infections[v] for v in list(G.successors(x)) + [x]]) for x in opts}
            else:
                sav_u = {x: np.sum([current_prob_infections[v] for v in list(G.successors(x)) + [x]]) for x in
                         options_u_2}
            u = max(sav_u, key=sav_u.get)
            print(tot[u], total_inform[u], max(tot.values()))



        else:
            u = u_s[0]

        expected_inf = inf_u[u].copy()
        helps_u = {c: (initial_infections[c] - expected_inf[c]) / initial_infections[c] for c in communities_to_save}
        min_help = min(helps_u.values())
        maximin_value = min_help
        S_P.append(u)

        first = False
        candidates -= set([u])
        to_remove = set()
        for j in [j for j in WRRs_to_keep if u in contributions[j]]:
            current_prob_infections[roots[j]] -= 1 / count_roots_total[roots[j]]
            current_prob_infections[roots[j]] = max(0, current_prob_infections[roots[j]])
            for w in set(candidates) - set(contributions[j].keys()):
                inf_u[w][partition[roots[j]]] -= 1 / count_roots_total[roots[j]]
            for w in set(candidates) & set(contributions[j].keys()):
                total_inform[w] -= 1 / count_roots_total[roots[j]]
        WRRs_to_keep = list(set(WRRs_to_keep) - set([j for j in WRRs_to_keep if u in contributions[j]]))

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return S_P

def FWRR_single_partition(network, M, k):
    network = nx.convert_node_labels_to_integers(network, label_attribute="old_label")
    M = util.get_new_labels(network, M)
    partition = {}
    for v in list(network.nodes()):
        partition[v] = 1
    T = WRS_method(network, M, k, partition)
    T_old = util.get_old_labels(network, T)
    return T_old


def FWRRS(network, M, k):
    network = nx.convert_node_labels_to_integers(network, label_attribute="old_label")
    M = util.get_new_labels(network, M)
    partition = community_louvain.best_partition(network.to_undirected())
    for node in network.nodes():
        network.nodes[node]['community'] = partition[node]
    T = WRS_method(network, M, k, partition)
    T_old = util.get_old_labels(network, T)
    return T_old


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


def CMIA_O_fair(G, S_N, k, min_prob, cost_function="uniform"):  # we now consider the communities, and take as input the partition
    G = nx.convert_node_labels_to_integers(G, label_attribute="old_label")
    S_N = util.get_new_labels(G, S_N)
    partition = community_louvain.best_partition(G.to_undirected())
    for node in G.nodes():
        G.nodes[node]['community'] = partition[node]
    avg_deg = sum([G.out_degree(x) for x in G.nodes()]) / len(G.nodes())
    start = timeit.default_timer()
    S_P = []
    Neg_range = set()  # nodes that could be reached by S_N with probability higer than min_prob
    communities = get_communities(partition, G)
    MIA_edges_n, MIA_p_edges = {}, {}  # to keep track of the MIIAs without having to recompute them
    DecInf_per_comm = {}  # dictionary storing the contribution per community a node has if added to S_P
    E_inf_num = [0] * communities  # value of expected infected nodes per community
    saved_per_comm = [0] * communities  # expected number of saved nodes per community
    steps_n, steps_p = {}, {}
    proba_n, proba_p = {}, {}

    inf = []
    for u in S_N:
        MIOA, s, p = dijkstra_MIA(u, G, G, False, min_prob, [], "negative")
        for v in list(set(MIOA.nodes()) - set(S_N) - set(Neg_range)):
            Neg_range.add(v)  # contains the nodes that could get infected
            inf.append((v, u))
    print(len(Neg_range))
    time = 0
    time1 = 0
    for u in tqdm(Neg_range, position=0,
                  leave=True):  # for each node that gets ifected  we compute how much the probability of infection value lowers if we add a node to the seed
        s1 = timeit.default_timer()
        MIIA, steps, probs = dijkstra_MIA(u, G, G, True, min_prob, S_N,
                                          "negative")  # our candidates are the nodes that reach u with probability higher than min_prob
        MIA_edges_n[u] = [e for e in MIIA.edges()].copy()  # skeleton of MIIA
        steps_n[u] = {key: val for key, val in steps.items()}

        MIIA_p, steps_pos, probs_p = dijkstra_MIA(u, G, G, True, min_prob, S_N,
                                                  "positive")  # our candidates are the nodes that reach u with probability higher than min_prob
        MIA_p_edges[u] = [e for e in MIIA_p.edges()].copy()  # skeleton of MIIA
        steps_p[u] = {key: val for key, val in steps_pos.items()}
        time += timeit.default_timer() - s1
        s2 = timeit.default_timer()
        ap = compute_ap_N(u, S_N, S_P, MIIA, MIIA_p, G)  # probability of infection of u if we dont intervene
        if ap == 0:
            continue

        time1 += timeit.default_timer() - s2
        infectees = [x for x in MIIA.nodes() if x in S_N]

        longest = [x for _, x in sorted(zip([steps[x] for x in infectees], infectees), reverse=True)][0]

        # print(len(infectees))
        E_inf_num[
            partition[u]] += ap  # we add the propability of infection of u to our expected value within its community

        for v in [x for x in list(set(MIIA_p.nodes()) - set(S_N)) if steps_pos[x] <= steps[longest]]:
            s2 = timeit.default_timer()
            if len(infectees) == 1 and path_blocked_MIA(u, infectees, MIIA, [v]):
                ap_v = 0
            else:
                ap_v = compute_ap_N(u, S_N, S_P + [v], MIIA, MIIA_p,
                                    G)  # probability of infection of u if v is added to S_P
            time1 += timeit.default_timer() - s2
            if v in DecInf_per_comm:
                DecInf_per_comm[v][partition[u]] += (ap - ap_v)
            else:
                DecInf_per_comm[v] = [0] * communities
                DecInf_per_comm[v][partition[u]] = (
                            ap - ap_v)  # compute total impact of the node as truth campaigner per community

    E_inf_num = [round(x, 1) for x in E_inf_num]

    while cost(S_P, cost_function, G, avg_deg) < k:

        percentage_saved = [float(saved_per_comm[c] / E_inf_num[c]) if E_inf_num[c] != 0 else 1 for c in
                            range(communities)]
        need_saving = [c for c in range(communities) if percentage_saved[c] == min(
            percentage_saved)]  # communities with minimum percentage of saved nodes
        DecInf_needed = {v: sum([DecInf_per_comm[v][c] for c in need_saving]) for v in
                         DecInf_per_comm}  # contibution of each node within the communities that need_saving

        Saved_per_com_v = {}  # dictionary storing percentage of nodes we expect to save if we add v to S_P per community
        for key in DecInf_per_comm.keys():
            Saved_per_com_v[key] = [
                float((saved_per_comm[c] + DecInf_per_comm[key][c]) / E_inf_num[c]) if E_inf_num[c] != 0 else 1 for c in
                range(communities)]

        candidates = {n: min(v) for n, v in Saved_per_com_v.items() if
                      n not in S_N + list(S_P) and cost(S_P + [n], cost_function, G,
                                                        avg_deg) <= k}  # we store for each node the min percentage of saved nodes in a community if added to S_P
        candidates = dict(sorted(candidates.items(), key=lambda kv: DecInf_needed[kv[0]],
                                 reverse=True))  # some might tie, so we order them in descending order of DecInf_needed vaue
        if len(candidates) == 0: break

        u = max(candidates,
                key=candidates.get)  # chose node maximin value that has the biggest contribution within the communities with min saved percentage
        # print(u)
        saved_per_comm = [DecInf_per_comm[u][c] + saved_per_comm[c] for c in
                          range(communities)]  # we update the expected number of saved nodes
        # we now proceed to update the contribution of each node
        MIOA, _, _ = dijkstra_MIA(u, G, G, False, min_prob, S_N + S_P, "positive")

        for v in [x for x in MIOA.nodes() if x in Neg_range]:  # nodes that u could potentially save

            MIIA.clear()  # we recontruct the MIIA, which we already computed and stored

            MIIA.add_edges_from(MIA_edges_n[v])

            MIIA_p.clear()  # we recontruct the MIIA, which we already computed and stored
            MIIA_p.add_edges_from(MIA_p_edges[v])

            longest = max([x for x in steps_n[v].items() if x[0] in S_N and not path_blocked_MIA(x[0], v, MIIA, S_P)],
                          key=lambda item: item[1], default=(None, None))[0]
            if longest == None: continue

            ap = compute_ap_N(v, S_N, S_P, MIIA, MIIA_p, G)  # probability of infection of v with current S_P
            ap_u = compute_ap_N(v, S_N, S_P + [u], MIIA, MIIA_p,
                                G)  # probability of infection of v with current S_P which now includes u
            for w in [x for x in list(set(MIIA_p.nodes()) - set(S_N)) if
                      steps_p[v][x] <= steps_n[v][longest]]:  # nodes whose contribution now decreses since u helps
                # if  path_blocked_MIA(w,v, MIIA, S_N):continue  # if we have to go through a negative seed to reach v, we cant save him with w and we continue
                ap_w = compute_ap_N(v, S_N, S_P + [w], MIIA, MIIA_p,
                                    G)  # probability of infection of v if w is added to S_P

                DecInf_per_comm[w][partition[v]] -= (ap - ap_w)  # remove all influence of w when u was not in S_P
                if ap < min_prob: continue
                if v != u or ap_u == 0:
                    ap_u_w = compute_ap_N(v, S_N, S_P + [w] + [u], MIIA, MIIA_p, G)
                    DecInf_per_comm[w][partition[v]] += (ap_u - ap_u_w)

        S_P.append(u)

    saved_per_comm_final = [
        float((saved_per_comm[c] + DecInf_per_comm[key][c]) / E_inf_num[c]) if E_inf_num[c] != 0 else 1 for c in
        range(communities)]

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    S_P = util.get_old_labels(G, S_P)
    return S_P


def CMIA_O_syn_group_fair(G_name, S_N, k, min_prob, cost_function="uniform"):  # we now consider the communities, and take as input the partition
    G = nx.read_gexf(f"../data/{G_name}_gpickle", node_type=int)
    G = util.set_weight_value(G, min_prob, "positive")
    G = util.set_weight_value(G, min_prob, "negative")
    avg_deg = sum([G.out_degree(x) for x in G.nodes()]) / len(G.nodes())
    start = timeit.default_timer()
    S_P = []
    Neg_range = set()  # nodes that could be reached by S_N with probability higer than min_prob
    communities = [G.nodes[v]['community'] for v in G.nodes()]
    partition = {}
    d = dict([(y, x) for x, y in enumerate(sorted(set(communities)))])
    for v in list(G.nodes()):
        partition[v] = d[G.nodes[v]['community']]
    communities = len(np.unique(communities))
    MIA_edges_n, MIA_p_edges = {}, {}  # to keep track of the MIIAs without having to recompute them
    DecInf_per_comm = {}  # dictionary storing the contribution per community a node has if added to S_P
    E_inf_num = [0] * communities  # value of expected infected nodes per community
    saved_per_comm = [0] * communities  # expected number of saved nodes per community
    steps_n, steps_p = {}, {}
    proba_n, proba_p = {}, {}

    inf = []
    for u in S_N:
        MIOA, s, p = dijkstra_MIA(u, G, G, False, min_prob, [], "negative")
        for v in list(set(MIOA.nodes()) - set(S_N) - set(Neg_range)):
            Neg_range.add(v)  # contains the nodes that could get infected
            inf.append((v, u))
    print(len(Neg_range))
    time = 0
    time1 = 0
    for u in tqdm(Neg_range, position=0,
                  leave=True):  # for each node that gets ifected  we compute how much the probability of infection value lowers if we add a node to the seed
        s1 = timeit.default_timer()
        MIIA, steps, probs = dijkstra_MIA(u, G, G, True, min_prob, S_N,
                                          "negative")  # our candidates are the nodes that reach u with probability higher than min_prob
        MIA_edges_n[u] = [e for e in MIIA.edges()].copy()  # skeleton of MIIA
        steps_n[u] = {key: val for key, val in steps.items()}

        MIIA_p, steps_pos, probs_p = dijkstra_MIA(u, G, G, True, min_prob, S_N,
                                                  "positive")  # our candidates are the nodes that reach u with probability higher than min_prob
        MIA_p_edges[u] = [e for e in MIIA_p.edges()].copy()  # skeleton of MIIA
        steps_p[u] = {key: val for key, val in steps_pos.items()}
        time += timeit.default_timer() - s1
        s2 = timeit.default_timer()
        ap = compute_ap_N(u, S_N, S_P, MIIA, MIIA_p, G)  # probability of infection of u if we dont intervene
        if ap == 0:
            continue

        time1 += timeit.default_timer() - s2
        infectees = [x for x in MIIA.nodes() if x in S_N]

        longest = [x for _, x in sorted(zip([steps[x] for x in infectees], infectees), reverse=True)][0]

        # print(len(infectees))
        E_inf_num[
            partition[u]] += ap  # we add the propability of infection of u to our expected value within its community

        for v in [x for x in list(set(MIIA_p.nodes()) - set(S_N)) if steps_pos[x] <= steps[longest]]:
            s2 = timeit.default_timer()
            if len(infectees) == 1 and path_blocked_MIA(u, infectees, MIIA, [v]):
                ap_v = 0
            else:
                ap_v = compute_ap_N(u, S_N, S_P + [v], MIIA, MIIA_p,
                                    G)  # probability of infection of u if v is added to S_P
            time1 += timeit.default_timer() - s2
            if v in DecInf_per_comm:
                DecInf_per_comm[v][partition[u]] += (ap - ap_v)
            else:
                DecInf_per_comm[v] = [0] * communities
                DecInf_per_comm[v][partition[u]] = (
                            ap - ap_v)  # compute total impact of the node as truth campaigner per community

    E_inf_num = [round(x, 1) for x in E_inf_num]

    while cost(S_P, cost_function, G, avg_deg) < k:

        percentage_saved = [float(saved_per_comm[c] / E_inf_num[c]) if E_inf_num[c] != 0 else 1 for c in
                            range(communities)]
        need_saving = [c for c in range(communities) if percentage_saved[c] == min(
            percentage_saved)]  # communities with minimum percentage of saved nodes
        DecInf_needed = {v: sum([DecInf_per_comm[v][c] for c in need_saving]) for v in
                         DecInf_per_comm}  # contibution of each node within the communities that need_saving

        Saved_per_com_v = {}  # dictionary storing percentage of nodes we expect to save if we add v to S_P per community
        for key in DecInf_per_comm.keys():
            Saved_per_com_v[key] = [
                float((saved_per_comm[c] + DecInf_per_comm[key][c]) / E_inf_num[c]) if E_inf_num[c] != 0 else 1 for c in
                range(communities)]

        candidates = {n: min(v) for n, v in Saved_per_com_v.items() if
                      n not in S_N + list(S_P) and cost(S_P + [n], cost_function, G,
                                                        avg_deg) <= k}  # we store for each node the min percentage of saved nodes in a community if added to S_P
        candidates = dict(sorted(candidates.items(), key=lambda kv: DecInf_needed[kv[0]],
                                 reverse=True))  # some might tie, so we order them in descending order of DecInf_needed vaue
        if len(candidates) == 0: break

        u = max(candidates,
                key=candidates.get)  # chose node maximin value that has the biggest contribution within the communities with min saved percentage
        # print(u)
        saved_per_comm = [DecInf_per_comm[u][c] + saved_per_comm[c] for c in
                          range(communities)]  # we update the expected number of saved nodes
        # we now proceed to update the contribution of each node
        MIOA, _, _ = dijkstra_MIA(u, G, G, False, min_prob, S_N + S_P, "positive")

        for v in [x for x in MIOA.nodes() if x in Neg_range]:  # nodes that u could potentially save

            MIIA.clear()  # we recontruct the MIIA, which we already computed and stored

            MIIA.add_edges_from(MIA_edges_n[v])

            MIIA_p.clear()  # we recontruct the MIIA, which we already computed and stored
            MIIA_p.add_edges_from(MIA_p_edges[v])

            longest = max([x for x in steps_n[v].items() if x[0] in S_N and not path_blocked_MIA(x[0], v, MIIA, S_P)],
                          key=lambda item: item[1], default=(None, None))[0]
            if longest == None: continue

            ap = compute_ap_N(v, S_N, S_P, MIIA, MIIA_p, G)  # probability of infection of v with current S_P
            ap_u = compute_ap_N(v, S_N, S_P + [u], MIIA, MIIA_p,
                                G)  # probability of infection of v with current S_P which now includes u
            for w in [x for x in list(set(MIIA_p.nodes()) - set(S_N)) if
                      steps_p[v][x] <= steps_n[v][longest]]:  # nodes whose contribution now decreses since u helps
                # if  path_blocked_MIA(w,v, MIIA, S_N):continue  # if we have to go through a negative seed to reach v, we cant save him with w and we continue
                ap_w = compute_ap_N(v, S_N, S_P + [w], MIIA, MIIA_p,
                                    G)  # probability of infection of v if w is added to S_P

                DecInf_per_comm[w][partition[v]] -= (ap - ap_w)  # remove all influence of w when u was not in S_P
                if ap < min_prob: continue
                if v != u or ap_u == 0:
                    ap_u_w = compute_ap_N(v, S_N, S_P + [w] + [u], MIIA, MIIA_p, G)
                    DecInf_per_comm[w][partition[v]] += (ap_u - ap_u_w)

        S_P.append(u)

    saved_per_comm_final = [
        float((saved_per_comm[c] + DecInf_per_comm[key][c]) / E_inf_num[c]) if E_inf_num[c] != 0 else 1 for c in
        range(communities)]

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return S_P
