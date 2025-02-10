def generate_fully_connected_subgraphs_with_links(N, k):
    subgraphs = []
    edges = set()  # To store unique edges

    num_subgraphs = N // k  # Number of subgraphs

    for s in range(num_subgraphs):
        start = s * k  # Start index for the subgraph
        subgraph = [(start + j, start + i + j + 1) for j in range(k - 1) for i in range(k - j - 1)]
        
        # Add inter-subgraph connections (previous subgraph to current)
        if s > 0:
            for i in range(k):  # Connect each node to its corresponding node in the previous subgraph
                prev_node = (s - 1) * k + i
                curr_node = start + i
                if prev_node < curr_node:  # Avoid duplicate edges
                    edges.add((prev_node, curr_node))

        # Add current subgraph edges to the set
        edges.update(subgraph)

    return sorted(edges)  # Return edges sorted to maintain order

# Example usage
N = 12  # Total nodes
k = 4   # Nodes per subgraph
edges = generate_fully_connected_subgraphs_with_links(N, k)

# Print results
print("Edges:", edges)
print("Edges:", len(edges))

# Example usage
N = 9  # Total nodes
k = 3   # Nodes per subgraph
edges = generate_fully_connected_subgraphs_with_links(N, k)

# Print results
print("Edges:", edges)
print("Edges:", len(edges))

def generate_edges_with_timesteps(N, k):
    subgraphs = []
    edges = set()  # To store unique edges
    timesteps = {}  # Dictionary to store edge timesteps

    num_subgraphs = N // k  # Number of subgraphs

    for s in range(num_subgraphs):
        start = s * k  # Start index for the subgraph
        subgraph = [(start + j, start + i + j + 1) for j in range(k - 1) for i in range(k - j - 1)]
        
        # Assign timestep for the current subgraph edges
        for edge in subgraph:
            edges.add(edge)
            timesteps[edge] = s + 1  # Time starts from 1

        # Add inter-subgraph connections (previous subgraph to current)
        if s > 0:
            for i in range(k):  # Connect each node to its corresponding node in the previous subgraph
                prev_node = (s - 1) * k + i
                curr_node = start + i
                if prev_node < curr_node:  # Avoid duplicate edges
                    edges.add((prev_node, curr_node))
                    timesteps[(prev_node, curr_node)] = s + 1  # Assign next subgraph's timestep

    sorted_edges = sorted(edges)  # Sort edges for consistency
    timestep_values = [timesteps[edge] for edge in sorted_edges]  # Extract timesteps in sorted order

    return sorted_edges, timestep_values

# Example usage
N = 12  # Total nodes
k = 4   # Nodes per subgraph
edges, timesteps = generate_edges_with_timesteps(N, k)

# Print results
print(edges)
print(timesteps)
print(len(edges))
print(len(timesteps))
# for edge, timestep in zip(edges, timesteps):
#     print(f"Edge: {edge}, Timestep: {timestep}")

import random

def sample_edges(edges, timesteps, s):
    """
    Randomly samples s edges and their corresponding timesteps.

    :param edges: List of edges (tuples)
    :param timesteps: List of timesteps corresponding to the edges
    :param s: Number of samples to draw
    :return: List of sampled edges and their corresponding timesteps
    """
    sampled_indices = random.sample(range(len(edges)), min(s, len(edges)))  # Ensure we don't sample more than available
    sampled_edges = [edges[i] for i in sampled_indices]
    sampled_timesteps = [timesteps[i] for i in sampled_indices]
    
    return sampled_edges, sampled_timesteps

# Example usage
N = 12  # Total nodes
k = 4   # Nodes per subgraph
s = 5   # Number of samples

edges, timesteps = generate_edges_with_timesteps(N, k)
sampled_edges, sampled_timesteps = sample_edges(edges, timesteps, s)

print(sampled_edges) 
print(sampled_timesteps) 

# # Print sampled results
# for edge, timestep in zip(sampled_edges, sampled_timesteps):
#     print(f"Sampled Edge: {edge}, Timestep: {timestep}")


def generate_edges_with_reset_timesteps(N, k, t):
    """
    Generate edges for fully connected subgraphs and inter-subgraph connections,
    with timesteps resetting after every t subgraphs.

    :param N: Total number of nodes
    :param k: Number of nodes per subgraph
    :param t: Reset interval for timesteps
    :return: Tuple (sorted_edges, timestep_values)
    """
    edges = set()  # To store unique edges
    timesteps = {}  # Dictionary to store edge timesteps

    num_subgraphs = N // k  # Number of subgraphs

    for s in range(num_subgraphs):
        start = s * k  # Start index for the subgraph
        current_timestep = (s % t) + 1  # Reset after every t subgraphs

        # Generate fully connected subgraph edges
        subgraph = [(start + j, start + i + j + 1) for j in range(k - 1) for i in range(k - j - 1)]
        
        # Assign timestep for the current subgraph edges
        for edge in subgraph:
            edges.add(edge)
            timesteps[edge] = current_timestep

        # Add inter-subgraph connections (previous subgraph to current)
        if s > 0:
            for i in range(k):  # Connect each node to its corresponding node in the previous subgraph
                prev_node = (s - 1) * k + i
                curr_node = start + i
                if prev_node < curr_node:  # Avoid duplicate edges
                    edges.add((prev_node, curr_node))
                    timesteps[(prev_node, curr_node)] = current_timestep

    sorted_edges = sorted(edges)  # Sort edges for consistency
    timestep_values = [timesteps[edge] for edge in sorted_edges]  # Extract timesteps in sorted order

    return sorted_edges, timestep_values

# Example usage
N = 12  # Total nodes
k = 4   # Nodes per subgraph
t = 2   # Timestep reset interval
edges, timesteps = generate_edges_with_reset_timesteps(N, k, t)

# Print results
for edge, timestep in zip(edges, timesteps):
    print(f"Edge: {edge}, Timestep: {timestep}")

def generate_edges_with_reset_timesteps_no_interlinks(N, k, t):
    """
    Generate edges for fully connected subgraphs with timesteps resetting after every t subgraphs.
    Removes inter-subgraph edges when the timestep resets.

    :param N: Total number of nodes
    :param k: Number of nodes per subgraph
    :param t: Reset interval for timesteps
    :return: Tuple (sorted_edges, timestep_values)
    """
    edges = set()  # To store unique edges
    timesteps = {}  # Dictionary to store edge timesteps

    num_subgraphs = N // k  # Number of subgraphs

    for s in range(num_subgraphs):
        start = s * k  # Start index for the subgraph
        current_timestep = (s % t) + 1  # Reset after every t subgraphs

        # Generate fully connected subgraph edges
        subgraph = [(start + j, start + i + j + 1) for j in range(k - 1) for i in range(k - j - 1)]
        
        # Assign timestep for the current subgraph edges
        for edge in subgraph:
            edges.add(edge)
            timesteps[edge] = current_timestep

        # Add inter-subgraph connections (previous subgraph to current) only if no reset occurs
        if s > 0 and s % t != 0:  # Avoid adding interlinks when timestep resets
            for i in range(k):  # Connect each node to its corresponding node in the previous subgraph
                prev_node = (s - 1) * k + i
                curr_node = start + i
                if prev_node < curr_node:  # Avoid duplicate edges
                    edges.add((prev_node, curr_node))
                    timesteps[(prev_node, curr_node)] = current_timestep

    sorted_edges = sorted(edges)  # Sort edges for consistency
    timestep_values = [timesteps[edge] for edge in sorted_edges]  # Extract timesteps in sorted order

    return sorted_edges, timestep_values

# Example usage
N = 12  # Total nodes
k = 4   # Nodes per subgraph
t = 2   # Timestep reset interval
edges, timesteps = generate_edges_with_reset_timesteps_no_interlinks(N, k, t)

print(edges)
print(timesteps) 

# Print results
for edge, timestep in zip(edges, timesteps):
    print(f"Edge: {edge}, Timestep: {timestep}")