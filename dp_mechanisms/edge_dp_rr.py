import torch
import numpy as np

from .config import split_epsilon


def edge_dp_randomized_response(data, epsilon: float):
    """
    Apply Edge Differential Privacy using Randomized Response for sparse graphs.

    Uses an asymmetric RR formulation designed for sparse adjacency matrices,
    where the vast majority of entries are 0. The symmetric formula
    p = e^(eps/2)/(e^(eps/2)+1) is unsuitable here: applied to ~3.66M
    non-edges it generates hundreds of thousands of false edges.

    Instead, we derive p0 and p1 such that:
      - The expected number of edges is preserved: E[|E'|] = |E_priv|
      - The DP ratio is exactly e^(epsilon2): p1 / (1-p0) = e^(epsilon2)

    where |E_priv| is the edge count ESTIMATED FROM THE PRIVATIZED DENSITY
    d_tilde (not the true edge count |E|). This is the key point: the
    asymmetric probabilities are calibrated against d_tilde, which is itself
    a DP quantity (Laplace mechanism with budget epsilon1), so the epsilon1
    budget is actually consumed and p0/p1 no longer depend on the exact
    (sensitive) number of original edges.

    Solving the system (with |E_priv| = d_tilde * |possible|):
      p1 = e^(eps2) * (1 - p0)
      p1 * |E_priv| + (1-p0) * |non-edges_priv| = |E_priv|

    Gives:
      (1 - p0) = |E_priv| / (e^(eps2) * |E_priv| + |non-edges_priv|)
      p1       = e^(eps2) * (1 - p0)

    The epsilon budget split is defined ONCE in dp_mechanisms/config.py
    (currently 1% for the density query, 99% for randomized response):
      - epsilon1 = EPSILON_DENSITY_FRACTION   * epsilon -> density (Laplace)
      - epsilon2 = EPSILON_STRUCTURE_FRACTION * epsilon -> randomized response
    By sequential composition the full mechanism is epsilon-edge-DP.

    Parameters
    ----------
    data : PyG Data object
    epsilon : float
        Total privacy budget.

    Returns
    -------
    edge_index_new : torch.Tensor
    graph_changes : dict
    """

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # 1. Build upper-triangular adjacency (undirected graph)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1
    adj = torch.triu(adj, diagonal=1)

    triu_idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
    adj_triu = adj[triu_idx[0], triu_idx[1]]

    # 2. Density stats
    total_possible = num_nodes * (num_nodes - 1) / 2
    num_edges = adj_triu.sum().item()
    non_edges = total_possible - num_edges
    d_g = num_edges / total_possible

    # Split epsilon (single source of truth: dp_mechanisms/config.py)
    epsilon1, epsilon2 = split_epsilon(epsilon)

    # 3. Perturb density via Laplace
    sensitivity_density = 1.0 / total_possible
    scale_density = sensitivity_density / epsilon1
    d_tilde = d_g + np.random.laplace(0, scale_density)
    d_tilde = float(np.clip(d_tilde, 1e-6, 1.0))

    # 4. Asymmetric RR probabilities (sparse-graph formulation)
    #    Calibrated against the PRIVATIZED density d_tilde (not the true
    #    edge count), so the epsilon1 budget spent above is actually used.
    #    Ensures E[|E'|] = |E_priv| and ratio p1/(1-p0) = e^epsilon2.
    est_edges = d_tilde * total_possible
    est_non_edges = total_possible - est_edges
    exp_eps2 = np.exp(epsilon2)
    one_minus_p0 = est_edges / (exp_eps2 * est_edges + est_non_edges)
    p0 = float(np.clip(1.0 - one_minus_p0, 0.0, 1.0))
    p1 = float(np.clip(exp_eps2 * one_minus_p0, 0.0, 1.0))

    # 5. Vectorized RR
    rand_vals = torch.tensor(
        np.random.uniform(0, 1, size=int(adj_triu.shape[0])),
        dtype=torch.float32
    )

    edge_mask     = adj_triu == 1
    non_edge_mask = adj_triu == 0

    # Existing edges kept with prob p1; removed if rand > p1
    removed_mask = edge_mask & (rand_vals > p1)
    # Non-edges flipped to edge with prob (1-p0); added if rand > p0
    added_mask   = non_edge_mask & (rand_vals > p0)

    new_adj_triu = adj_triu.clone()
    new_adj_triu[removed_mask] = 0
    new_adj_triu[added_mask]   = 1

    edges_removed   = int(removed_mask.sum().item())
    edges_added     = int(added_mask.sum().item())
    new_edges_count = int(new_adj_triu.sum().item())

    # 6. Reconstruct symmetric edge_index
    new_adj_full = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    new_adj_full[triu_idx[0], triu_idx[1]] = new_adj_triu
    new_adj_full = new_adj_full + new_adj_full.t()

    edge_index_new = new_adj_full.nonzero(as_tuple=False).t()

    graph_changes = {
        'original_edges': num_edges,
        'new_edges': new_edges_count,
        'edges_removed': edges_removed,
        'edges_added': edges_added,
        'original_density': d_g,
        'perturbed_density': d_tilde,
        'epsilon': epsilon,
        'epsilon1': epsilon1,
        'epsilon2': epsilon2,
        'p0': p0,
        'p1': p1,
    }

    return edge_index_new, graph_changes