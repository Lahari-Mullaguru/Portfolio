import pandas as pd
import networkx as nx
import numpy as np
import diffusionmodels as dm
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def read_pos_seeds(network, policy, num_seeds, folder):
    if policy == 'random':
        pos_seeds = np.random.choice(network.nodes(), size=num_seeds, replace=False)
    else:
        pos_seeds = []
        pos_seed_file = f"{folder}/pos_seeds_{policy}.txt"
        with open(pos_seed_file, 'r') as f:
             for line in f:
                 pos_seeds.append(int(line[:-1]))

        if len(pos_seeds) < num_seeds:
            print("Not enough seeds in the file")
            return []
        pos_seeds = pos_seeds[:num_seeds]
    return pos_seeds

def compute_proximity(g, M):
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
    return proximity

def compute_proximity_arr(g, M):
    proximity = np.ones(g.number_of_nodes()) * np.inf
    queue = []
    for m in M:
        proximity[m] = 0
        queue.append(m)
    while len(queue) > 0:
        node = queue.pop(0)
        neighbors = g.neighbors(node)
        for neighbor in neighbors:
            if proximity[neighbor] != np.inf:
                old_proximity = proximity[neighbor]
                new_proximity = proximity[node] + 1
                if new_proximity < old_proximity:
                    proximity[neighbor] = new_proximity
                    queue.append(neighbor)
            else:
                proximity[neighbor] = proximity[node] + 1
                queue.append(neighbor)
    return proximity

def get_neg_seeds(network, policy, num_seeds):
    if policy == 'random':
        np.random.seed(100)
        neg_seeds = list(np.random.choice(network.nodes(), size=num_seeds, replace=False))
    elif policy == 'degree':
        node_deg_list = sorted(network.out_degree, key=lambda x: x[1], reverse=True)
        neg_seeds = [x for (x, _) in node_deg_list][:num_seeds]
    elif policy == 'page_rank_reverse':
        node_pr_list = sorted(nx.pagerank(network.reverse()).items(), key=lambda x: x[1], reverse=True)
        neg_seeds = [x for (x, _) in node_pr_list][:num_seeds]
    elif policy == 'betweenness':
        node_bet_list = sorted(nx.betweenness_centrality(network).items(), key=lambda x: x[1], reverse=True)
        neg_seeds = [x for (x, _) in node_bet_list][:num_seeds]
    return neg_seeds

def load_graph(file):
    graph_df = pd.read_csv(file, sep=" ", header=None)
    
    # Check if the data has the expected number of columns
    if graph_df.shape[1] != 2:
        raise ValueError(f"Expected 2 columns, but got {graph_df.shape[1]} columns")
    
    graph_df.columns = ['s', 't']

    edges = []

    for index, row in tqdm(graph_df.iterrows(), total=graph_df.shape[0]):
        if row.s == row.t:
            continue
        edge_cur = (row.s, row.t)
        edges.append(edge_cur)

    g = nx.DiGraph(edges)
    return g

def coicm(network, neg_seeds, pos_seeds):
    iter, pos_seeds = pos_seeds
    neg_impressed = dm.coicm(network, neg_seeds, pos_seeds, iter)
    return neg_impressed

def load_bidirectional_graph(file):
    graph_df = pd.read_csv(file, sep=" ", header=None)
    
    # Check if the data has the expected number of columns
    if graph_df.shape[1] != 2:
        raise ValueError(f"Expected 2 columns, but got {graph_df.shape[1]} columns")

    graph_df.columns = ['s', 't']

    edges = []

    for index, row in tqdm(graph_df.iterrows(), total=graph_df.shape[0]):
        if row.s == row.t:
            continue
        edge_cur = (row.s, row.t)
        edges.append(edge_cur)
        edge_cur = (row.t, row.s)
        edges.append(edge_cur)

    g = nx.DiGraph(edges)
    return g


def set_weight_value(G, val, p_n):
    for u, v in G.edges():
        G[u][v][p_n] = round(val, 3)
    return G

def set_weight_degree(G, p_n):
    for u, v in G.edges():
        G[u][v][p_n] = 1 / G.in_degree(v)
    return G

def get_old_labels(g, M):
    return [g.nodes[node]["old_label"] for node in M]

def compute_consistency(vul, sp):
    vul = vul / np.max(vul)
    vul_mat = np.abs(vul[:, None] - vul[None, :])
    similarity_mat = 1 - (vul_mat)
    sp_mat = np.abs(sp[:, None] - sp[None, :])
    pdt = np.multiply(similarity_mat, sp_mat)
    num = np.sum(pdt)
    den = np.sum(similarity_mat)
    consistency = 1 - (num / den)
    return consistency

def get_new_labels(g, M):
    new_to_old_labels = nx.get_node_attributes(g, "old_label")
    old_to_new_labels = {v: k for k, v in new_to_old_labels.items()}

    new_labels = []
    for node in M:
        if node in old_to_new_labels:
            new_labels.append(old_to_new_labels[node])
        else:
            print(f"Warning: Node {node} not found in old_to_new_labels and will be ignored.")
    return new_labels

def compute_gini_coefficient(sp):
    total = 0
    for i, xi in enumerate(sp[:-1], 1):
        total += np.sum(np.abs(xi - sp[i:]))
    return total / (len(sp) ** 2 * np.mean(sp))

def init_settings(dataset_file, edge_weight_policy, imp_prob, neg_policy="degree", k_m=10):
    if "facebook" in dataset_file or "fb" in dataset_file:
        g = load_bidirectional_graph(f"../data/{dataset_file}.txt")
    else:
        g = load_graph(f"../data/{dataset_file}.txt")
    strongly_connected = nx.is_strongly_connected(g)
    print("is strongly connected: " + str(strongly_connected))
    if not strongly_connected:
        largest_component = max(nx.strongly_connected_components(g), key=len)
        g = g.subgraph(largest_component)
        print(f"largest_subgraph has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
    neg_seeds = get_neg_seeds(g, neg_policy, k_m)
    if edge_weight_policy == "value":
        g = set_weight_value(g, imp_prob, "positive")
        g = set_weight_value(g, imp_prob, "negative")
    elif edge_weight_policy == "degree":
        g = set_weight_value(g, 1, "positive")
        g = set_weight_degree(g, "negative")
    elif edge_weight_policy == "degree_both":
        g = set_weight_degree(g, "positive")
        g = set_weight_degree(g, "negative")
    elif edge_weight_policy == "1_value":
        g = set_weight_value(g, 1, "positive")
        g = set_weight_value(g, imp_prob, "negative")
    network_new_labels = nx.DiGraph(g)
    network_new_labels = nx.convert_node_labels_to_integers(network_new_labels, label_attribute="old_label")
    neg_seeds_new_labels = get_new_labels(network_new_labels, neg_seeds)
    return g, neg_seeds, network_new_labels, neg_seeds_new_labels






def extract_features(network):
    features = pd.DataFrame(index=network.nodes)
    features['degree'] = [network.degree(n) for n in network.nodes]
    features['closeness'] = [nx.closeness_centrality(network, n) for n in network.nodes]
    features['betweenness'] = [nx.betweenness_centrality(network)[n] for n in network.nodes]
    features['clustering'] = [nx.clustering(network, n) for n in network.nodes]
    return features

def train_ml_model(features, labels):
    X = features.values
    y = labels.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    return model, scaler

def predict_positive_seeds(model, scaler, features, num_seeds):
    X = features.values
    X_scaled = scaler.transform(X)
    
    pos_seed_prob = model.predict_proba(X_scaled)[:, 1]
    predicted_pos_seeds = features.index[np.argsort(pos_seed_prob)[-num_seeds:]]
    
    return predicted_pos_seeds

def train_gb_model(features, labels):
    X = features.values
    y = labels.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"GB Accuracy: {accuracy:.2f}")
    print(f"GB Precision: {precision:.2f}")
    print(f"GB Recall: {recall:.2f}")
    print(f"GB F1 Score: {f1:.2f}")
    
    return model, scaler

def predict_gb_positive_seeds(model, scaler, features, num_seeds):
    X = features.values
    X_scaled = scaler.transform(X)
    
    pos_seed_prob = model.predict_proba(X_scaled)[:, 1]
    predicted_pos_seeds = features.index[np.argsort(pos_seed_prob)[-num_seeds:]]
    
    return predicted_pos_seeds

def plot_k_vs_df_columns(welfare_analysis_df, column_name, methods, y_label, filename, x_ticks):
    plt.figure(figsize=(8, 6))
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    
    # Flatten any lists or arrays in the "seed count" column to scalar values
    welfare_analysis_df["seed count"] = welfare_analysis_df["seed count"].apply(
        lambda x: x[0] if isinstance(x, (list, tuple)) else x
    ).astype(float)
    
    markers = ["D", "s", "o", "P", "h", "v", "<", ">", "^", "*", "p", "X", "d", ".", "H", "8", "1", "2", "3", "4", "|", "_", "+"]
    colors = ["hotpink", "dodgerblue", "limegreen", "orange", "mediumseagreen", "orangered", "green", "olive", "darkred", "darkblue",
              "purple", "mediumvioletred", "red", "blue", "steelblue", "gray", "darkgray", "peru", "crimson", "pink", "violet"]
    all_methods = ["pagerank", "cmia-o", "rbf", "protect", "naive_protect", "rl_fnim", "hifm", "rf", "gb"]

    addtl_counter = -1
    
    # Ensure that k_p is treated as the maximum value in x_ticks
    k_p = max(x_ticks)
    
    for policy in methods:
        if policy not in all_methods:
            addtl_counter += 1
            idx = len(all_methods) + addtl_counter
        else:
            idx = all_methods.index(policy)
        
        # Filter DataFrame based on policy and k_p
        df = welfare_analysis_df[welfare_analysis_df["policy"] == policy]
        
        # Apply the filtering based on k_p after ensuring "seed count" is scalar
        df = df[df["seed count"] <= k_p]
        
        # Skip plotting if there is no data left after filtering
        if df.empty:
            print(f"No data for policy: {policy} with seed count <= {k_p}")
            continue
        
        xs, ys = zip(*sorted(zip(df["seed count"], df[column_name])))
        plt.plot(xs, ys, label=policy, linestyle='dashed', marker=markers[idx], color=colors[idx], markersize=14)
    
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("k", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move the legend outside the plot
    plt.savefig(filename, bbox_inches='tight')  # Save the plot to file with tight bounding box
    plt.close()
def plot_bar_df(welfare_analysis_df, column_name, methods, y_label, filename, x_ticks):
    # Ensure k_p is the maximum value from x_ticks
    k_p = max(x_ticks)
    
    # Flatten any lists or arrays in the "seed count" column to scalar values
    welfare_analysis_df["seed count"] = welfare_analysis_df["seed count"].apply(
        lambda x: x[0] if isinstance(x, (list, tuple)) else x
    ).astype(float)
    
    # Filter the DataFrame by the scalar value k_p
    df = welfare_analysis_df[welfare_analysis_df["seed count"] <= k_p]
    
    # Ensure we only plot methods that exist in the methods list
    df = df[df["policy"].isin(methods)]
    
    # Pivot the table for plotting
    pivot_df = pd.pivot_table(df, index="policy", columns="seed count", values=column_name)
    
    # Plot the bar chart
    pivot_df.plot(kind="bar", rot=90, figsize=(8, 6))
    
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("Policy", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move the legend outside the plot
    plt.savefig(filename, bbox_inches='tight')  # Save the figure to file with tight bounding box
    plt.close()

def plot_fairness_bars(welfare_analysis_df, column_name, methods, y_label, filename, k_list=[5, 10, 25]):
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    plt.rcParams.update({'font.size': 30})
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(30, 15))

    for ax_id, k in enumerate(k_list):
        df = welfare_analysis_df[welfare_analysis_df["seed count"] == k][["policy", column_name]]
        df = df[df["policy"].isin(methods)]

        # Debugging print statement to check the DataFrame
        print(f"DataFrame for k={k}:\n{df.head()}")

        if df.empty:
            print(f"No data to plot for seed count = {k}")
            continue
        
        # Exploding the column and checking for numeric data
        try:
            df = df.explode(column_name).reset_index().rename(columns={'index': 'distance_from_M'})
            df['distance_from_M'] = df.groupby('distance_from_M').cumcount()
            df['distance'] = df['distance_from_M'] + 1

            # Convert the column to numeric if necessary
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

            if df[column_name].isnull().all():
                print(f"No numeric data to plot for {column_name} at seed count = {k}")
                continue

            df = df.pivot(index='policy', columns='distance', values=column_name)
            df.plot.bar(ax=axs[ax_id], rot=90, figsize=(10, 7), label='distance')
        except Exception as e:
            print(f"Error processing data for k={k}: {e}")
            continue

        axs[ax_id].set_xlabel("Method", fontsize=30)
        axs[ax_id].set_ylabel(y_label, fontsize=30)
        axs[ax_id].legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move the legend outside the plot

    plt.savefig(filename, bbox_inches='tight')  # Save the figure to file with tight bounding box
    plt.close()


'''def plot_k_vs_df_columns(welfare_analysis_df, column_name, methods, y_label, filename, k_p=25):
    plt.figure(figsize=(8, 6))
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    markers = ["D", "s", "o", "P", "h", "v",  "<", ">", "^", "*", "p", "X", "d", ".","H","8","1","2","3","4","|","_","+"]
    colors = ["hotpink", "dodgerblue", "limegreen", "orange", "mediumseagreen", "orangered", "green", "olive", "darkred", "darkblue",
              "purple", "mediumvioletred", "red", "blue", "steelblue", "gray", "darkgray", "peru", "crimson", "pink", "violet"]
    all_methods = [ "pagerank", "cmia-o", "rbf", "protect", "naive_protect", "rl_fnim", "hifm",'rf',"gb"]
    #["degree", "pagerank", "cmia-o", "biog", "rbf", "tib", "tibmm", "rps", "fwrrs", "fair-cmia-o", "greedy_maximin",
                   #"myopic_maximin", "protect", "naive_protect", "rl_fnim", "hifm", 'rf', 'gb']
    addtl_counter = -1
    for policy in methods:
        if policy not in all_methods:
            addtl_counter += 1
            idx = len(all_methods) + addtl_counter
        else:
            idx = all_methods.index(policy)
        df = welfare_analysis_df[welfare_analysis_df["policy"] == policy]
        df = df[df["seed count"] <= k_p]
        xs, ys = zip(*sorted(zip(df["seed count"], df[column_name])))
        plt.plot(xs, ys, label=policy, linestyle='dashed', marker=markers[idx], color=colors[idx], markersize=14)
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("k", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move the legend outside the plot
    plt.savefig(filename, bbox_inches='tight')  # Save the plot to file with tight bounding box
    plt.close()'''


'''def plot_fairness_bars(welfare_analysis_df, column_name, methods, y_label, filename, k_list=[5, 10, 25]):
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    plt.rcParams.update({'font.size': 30})
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(30, 15))

    for ax_id, k in enumerate(k_list):
        df = welfare_analysis_df[welfare_analysis_df["seed count"] == k][["policy", column_name]]
        df = df[df["policy"].isin(methods)]
        df = df.explode(column_name).reset_index().rename(columns={'index': 'distance_from_M'})
        df['distance_from_M'] = df.groupby('distance_from_M').cumcount()
        df['distance'] = df['distance_from_M'] + 1
        df = df.pivot(index='policy', columns='distance', values=column_name)
        df.plot.bar(ax=axs[ax_id], rot=90, figsize=(10, 7), label='distance')
        axs[ax_id].set_xlabel("Method", fontsize=30)
        axs[ax_id].set_ylabel(y_label, fontsize=30)
        axs[ax_id].legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move the legend outside the plot

    plt.savefig(filename, bbox_inches='tight')  # Save the figure to file with tight bounding box
    plt.close()'''

'''def plot_bar_df(welfare_analysis_df, column_name, methods, y_label, filename, k_p=25):
    df = welfare_analysis_df[welfare_analysis_df["seed count"] <= k_p]
    df = df[df["policy"].isin(methods)]
    pd.pivot_table(df, index="policy", columns="seed count", values=column_name).plot(kind="bar", rot=90, figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("k", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move the legend outside the plot
    plt.savefig(filename, bbox_inches='tight')  # Save the figure to file with tight bounding box
    plt.close()'''




def run_tradeoff_analysis(welfare_analysis_df, methods):
    results = []
    for method in methods:
        for k in [5, 10, 15, 20, 25]:
            row = welfare_analysis_df[(welfare_analysis_df['policy'] == method) & (welfare_analysis_df['seed count'] == k)]
            if not row.empty:
                fairness = row['consistency'].values[0]
                efficiency = row['avg_prob'].values[0]
                results.append((method, k, fairness, efficiency))
    return results
    
def plot_tradeoff_curve(results, filename):
    plt.figure(figsize=(10, 6))
    for method in set([r[0] for r in results]):
        method_results = [(r[2], r[3]) for r in results if r[0] == method]
        method_results = sorted(method_results)
        fairness_results = [r[0] for r in method_results]
        efficiency_results = [r[1] for r in method_results]
        plt.plot(fairness_results, efficiency_results, marker='o', label=method)
    plt.xlabel('Fairness (Consistency)')
    plt.ylabel('Efficiency (Average Save Probability)')
    plt.title('Fairness-Efficiency Trade-off Curve')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move the legend outside of the plot
    plt.savefig(filename, bbox_inches='tight')  # Save the figure to file with tight bounding box
    plt.close()



def identify_pareto_frontier(results):
    costs = np.array([(r[2], r[3]) for r in results])
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    pareto_efficient_points = costs[is_efficient]
    pareto_methods = [results[i][0] for i in range(len(results)) if is_efficient[i]]
    return pareto_efficient_points, pareto_methods

def plot_pareto_frontier(results, pareto_efficient_points, pareto_methods, filename):
    costs = np.array([(r[2], r[3]) for r in results])
    plt.scatter(costs[:, 0], costs[:, 1], label='All Points')
    plt.scatter(pareto_efficient_points[:, 0], pareto_efficient_points[:, 1], color='r', label='Pareto Frontier')
    for i, method in enumerate(pareto_methods):
        plt.annotate(method, (pareto_efficient_points[i][0], pareto_efficient_points[i][1]))
    plt.xlabel('Fairness (Consistency)')
    plt.ylabel('Efficiency (Average Save Probability)')
    plt.title('Pareto Frontier')
    plt.legend()
    plt.savefig(filename)  # Save the figure to file
    plt.close()
    

'''def plot_pareto_frontier(results, pareto_efficient_points, pareto_methods, filename):
    if not results or not pareto_efficient_points or not pareto_methods:
        print("Insufficient data to plot Pareto frontier.")
        return

    # Extract costs and check dimensions
    costs = np.array([point for point in results])

    if costs.ndim != 2 or costs.shape[1] != 2:
        print(f"Unexpected shape for costs: {costs.shape}. Expected a 2D array with shape (n_points, 2).")
        return

    # Plot all points
    plt.scatter(costs[:, 0], costs[:, 1], label='All Points')

    # Plot Pareto efficient points
    pareto_costs = np.array([costs[i] for i in pareto_efficient_points])
    plt.scatter(pareto_costs[:, 0], pareto_costs[:, 1], color='red', label='Pareto Efficient')

    # Annotate Pareto methods
    for i, method in enumerate(pareto_methods):
        plt.annotate(method, (pareto_costs[i, 0], pareto_costs[i, 1]))

    plt.xlabel('Cost')
    plt.ylabel('Benefit')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move the legend outside of the plot
    plt.title('Pareto Frontier')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()'''
