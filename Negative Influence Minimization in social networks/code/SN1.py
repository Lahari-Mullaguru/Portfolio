import networkx as nx
import numpy as np

def hich_ba_model(n, m, high_clustering=True, rewiring_probability=0.1):
    """
    Generate a synthetic network using a variant of the Barabási-Albert model 
    with optional high clustering and assortativity (HICH-BA model).
    
    Parameters:
    - n: Total number of nodes in the network.
    - m: Number of edges to attach from a new node to existing nodes.
    - high_clustering: Boolean, if True will attempt to increase clustering.
    - rewiring_probability: Probability of rewiring an edge to increase clustering.
    
    Returns:
    - G: A NetworkX graph object representing the generated network.
    """
    # Step 1: Start with a small fully connected network
    G = nx.complete_graph(m)
    
    # Step 2: Add nodes one by one
    for i in range(m, n):
        # Connect new node to m existing nodes with probability proportional to degree
        targets = _preferential_attachment_targets(G, m)
        G.add_edges_from((i, target) for target in targets)
        
        # Rewire edges to increase clustering if needed
        if high_clustering:
            G = _increase_clustering(G, i, rewiring_probability)
    
    return G

def _preferential_attachment_targets(G, m):
    """
    Choose m nodes from the graph G using preferential attachment.
    """
    node_degrees = np.array([G.degree(n) for n in G.nodes()])
    node_probs = node_degrees / node_degrees.sum()
    targets = np.random.choice(G.nodes(), size=m, replace=False, p=node_probs)
    return targets

def _increase_clustering(G, node, rewiring_probability):
    """
    Attempt to increase clustering in the network by rewiring edges.
    
    Parameters:
    - G: NetworkX graph object.
    - node: The new node recently added to the network.
    - rewiring_probability: The probability of rewiring an edge to increase clustering.
    
    Returns:
    - G: The modified graph with potentially higher clustering.
    """
    neighbors = list(G.neighbors(node))
    for neighbor in neighbors:
        if np.random.rand() < rewiring_probability:
            non_neighbors = set(G.nodes()) - set(neighbors) - {node}
            if non_neighbors:
                new_target = np.random.choice(list(non_neighbors))
                G.remove_edge(node, neighbor)
                G.add_edge(node, new_target)
    return G

# Parameters for the HICH-BA model
n = 1000  # Number of nodes
m = 5     # Number of edges to attach from a new node to existing nodes
rewiring_probability = 0.3  # Probability of rewiring to increase clustering

# Generate the synthetic network
G = hich_ba_model(n, m, high_clustering=True, rewiring_probability=rewiring_probability)

# Save the generated network to a text file
output_file = "synthetic_network_HICHBA.txt"
nx.write_edgelist(G, output_file, data=False)

print(f"Synthetic network generated and saved as {output_file}")

