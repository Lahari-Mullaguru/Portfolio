import networkx as nx
import numpy as np

# Parameters for the SBM
n = 1000  # Total number of nodes
sizes = [250, 250, 250, 250]  # Sizes of communities (blocks)
p_intra = 0.7  # Probability of intra-community edges
p_inter = 0.2  # Increased probability of inter-community edges

# Create the probability matrix
p = np.full((len(sizes), len(sizes)), p_inter)
np.fill_diagonal(p, p_intra)

# Generate the graph using SBM
G = nx.stochastic_block_model(sizes, p, seed=42)

# Convert the graph to a directed graph
G = G.to_directed()

# Find the largest strongly connected component
largest_scc = max(nx.strongly_connected_components(G), key=len)

if len(largest_scc) > 10:  # Arbitrary threshold to ensure the SCC is large enough
    G = G.subgraph(largest_scc).copy()
    output_file = "C:/Users/gopal/Downloads/master-project-master/master-project-master/data/SBM_SN1_1000_4_0.7_0.2.txt"
    nx.write_edgelist(G, output_file, data=False)
    print(f"Synthetic dataset SBM_SN1 generated and saved as {output_file}")
else:
    print("The largest SCC is too small. Consider adjusting the parameters.")
