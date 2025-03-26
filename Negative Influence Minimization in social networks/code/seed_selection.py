
"""
This file contains the seed selection methods.
"""
import sketch_based_methods as sbm
import networkx as nx
import rbf
import welfare_nim
import tibmm
import fair_baselines
import rps
import proximity_based_methods as proximity_method
import numpy as np
import random
import util
import pandas as pd


# RL-FNIM Algorithm
class QLearningAgent:
    def __init__(self, graph, m, k_t, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.graph = graph
        self.m = m
        self.k_t = k_t
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.initialize_q_table()

    def initialize_q_table(self):
        for node in self.graph.nodes():
            self.q_table[node] = np.zeros(len(self.graph.nodes()))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.graph.nodes()))
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - predict)

    def select_nodes(self):
        state = random.choice(list(self.graph.nodes()))
        selected_nodes = []
        for _ in range(self.k_t):
            action = self.choose_action(state)
            selected_nodes.append(action)
            next_state = action
            reward = -self.graph.degree[action]
            self.learn(state, action, reward, next_state)
            state = next_state
        return selected_nodes

def rl_fnim(graph, m, k_t):
    agent = QLearningAgent(graph, m, k_t)
    return agent.select_nodes()

# HIFM Algorithm
def initial_selection(graph, k_t):
    nodes = list(graph.nodes())
    degrees = np.array([graph.degree(n) for n in nodes])
    topk_indices = np.argpartition(degrees, -k_t)[-k_t:]
    return [nodes[i] for i in topk_indices]

def fairness_adjustment(graph, selected_nodes, m):
    sp = {node: 1 - graph.degree[node]/len(graph.nodes()) for node in graph.nodes()}
    sp_selected = np.mean([sp[node] for node in selected_nodes])
    fair_nodes = sorted(graph.nodes(), key=lambda x: abs(sp[x] - sp_selected))
    return fair_nodes[:len(selected_nodes)]

def hifm(graph, m, k_t):
    initial_nodes = initial_selection(graph, k_t)
    final_nodes = fairness_adjustment(graph, initial_nodes, m)
    return final_nodes

def select_seeds(g, policy, seeds_number, neg_seeds, min_prob, dataset_name=None):
    # Seed selection
    if policy == 'degree':
        node_deg_list = sorted(g.out_degree, key=lambda x: x[1], reverse=True)
        seeds = [x for (x,_) in node_deg_list if x not in neg_seeds][:seeds_number]
        print("seeds from degree:", seeds)
    elif policy == 'closeness':
        g_rev = nx.DiGraph.reverse(g)
        node_clo_list = sorted(nx.closeness_centrality(g_rev).items(), key=lambda x: x[1], reverse=True)
        seeds = [x for (x,_) in node_clo_list if x not in neg_seeds][:seeds_number]
        print("seeds from closeness:", seeds)
    elif policy == 'betweenness':
        node_bet_list = sorted(nx.betweenness_centrality(g).items(), key=lambda x: x[1], reverse=True)
        seeds = [x for (x,_) in node_bet_list if x not in neg_seeds][:seeds_number]
        print("seeds from betweenness:", seeds)
    elif policy == "pagerank":
        node_pr_list = sorted(nx.pagerank(g.reverse()).items(), key=lambda x: x[1], reverse=True)
        seeds = [x for (x,_) in node_pr_list if x not in neg_seeds][:seeds_number]
        print("seeds from pagerank_reverse:", seeds)
    elif policy == 'cmia-o':
        seeds = sbm.CMIA_O(g, neg_seeds, seeds_number, min_prob)
        print("seeds from cmia-o:", seeds)
    elif policy == 'biog':
        seeds = sbm.BIOG(g, neg_seeds, seeds_number, min_prob)
        print("seeds from biog:", seeds)
    elif policy == 'tib':
        seeds = sbm.TIB_Solver(g, neg_seeds, seeds_number)
        print("seeds from tib:", seeds)
    elif policy == 'rbf':
        seeds = rbf.rbf(g, neg_seeds, seeds_number)
        print("seeds from rbf:", seeds)
    elif policy == 'cdd':
        seeds = rbf.cdd(g, neg_seeds, seeds_number)
        print("seeds from cdd:", seeds)
    elif policy == 'tibmm':
        seeds = tibmm.tibmm(g, neg_seeds, seeds_number)
        print("seeds from tibmm:", seeds)
    elif policy == 'rps':
        seeds = rps.rps(g, neg_seeds, seeds_number)
        print("seeds from tibmm:", seeds)
    elif policy == 'myopic_maximin':
        seeds = welfare_nim.myopic_maximin(g, neg_seeds, seeds_number)
        print("seeds from welfare_myopic_maximin:", seeds)
    elif policy == 'naive_protect':
        seeds = proximity_method.myopic_distance_count_method(g, neg_seeds, seeds_number)
        print("seeds from myopic_distance_count_method:", seeds)
    elif policy == 'protect':
        seeds = proximity_method.myopic_sim_distance_method(g, neg_seeds, seeds_number)
        print("seeds from sp_maximin_using_M_distance_v2:", seeds)
    elif policy == 'naive_myopic':
        seeds = welfare_nim.naive_myopic(g, neg_seeds, seeds_number)
        print("seeds from naive_myopic:", seeds)
    elif policy == 'greedy_maximin':
        seeds = welfare_nim.greedy_maximin(g, neg_seeds, seeds_number)
        print("seeds from greedy_maximin:", seeds)
    elif policy == 'fwrrs':
        seeds = fair_baselines.FWRRS(g, neg_seeds, seeds_number)
        print("seeds from fwrrs:", seeds)
    elif policy == 'fair-cmia-o':
        seeds = fair_baselines.CMIA_O_fair(g, neg_seeds, seeds_number, min_prob)
        print("seeds from fair-cmia-o:", seeds)
    elif policy == 'fair-cmia-o-sn':
        seeds = fair_baselines.CMIA_O_syn_group_fair(dataset_name, neg_seeds, seeds_number, min_prob)
        print("seeds from fair-cmia-o-sn:", seeds)
    elif policy == 'rl_fnim':
        seeds = rl_fnim(g, neg_seeds, seeds_number)
        print("seeds from rl_fnim:", seeds)
    elif policy == 'hifm':
        seeds = hifm(g, neg_seeds, seeds_number)
        print("seeds from hifm:", seeds)
    #elif policy == 'gnn':
        #labels = [0 if node in neg_seeds else 1 for node in g.nodes()]  # Dummy labels
        #model = train_gnn_model(g, labels)
        #seeds = gnn_seed_selection(g, model, seeds_number)
        #print("seeds from gnn:", seeds)
    elif policy == 'rf':
        # Extract features
        features = util.extract_features(g)
        labels = pd.Series({node: (1 if node in neg_seeds else 0) for node in g.nodes})
        
        # Train the ML model
        model, scaler = util.train_ml_model(features, labels)
        
        # Predict positive seeds
        seeds = util.predict_positive_seeds(model, scaler, features, seeds_number)
        print("seeds from rf:", seeds)
    elif policy == 'gb':
        # Extract features
        features = util.extract_features(g)
        labels = pd.Series({node: (1 if node in neg_seeds else 0) for node in g.nodes})
        
        # Train the Gradient Boosting model
        model, scaler = util.train_gb_model(features, labels)
        
        # Predict positive seeds
        seeds = util.predict_gb_positive_seeds(model, scaler, features, seeds_number)
        print("seeds from gb:", seeds)
    else:
        raise NameError("Unknown policy")
    print(f'Number of Seeds: {len(seeds)}')
    return seeds
