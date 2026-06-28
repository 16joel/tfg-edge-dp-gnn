import torch
import numpy as np

from .config import split_epsilon


def laplace_noise(scale: float) -> float:
    """
    Generate Laplace noise with mean 0 and given scale.
    """
    return np.random.laplace(0, scale)


def edge_dp_laplace_mechanism(data, epsilon: float):
    """
    Apply Edge Differential Privacy using Laplace Mechanism.

    Adds Laplace noise to the upper-triangular adjacency matrix entries only,
    then binarizes by selecting the top-k noisy entries, where k is derived
    from a privatized density estimate.

    The epsilon budget split is defined ONCE in dp_mechanisms/config.py
    (currently 1% for the density query, 99% for the adjacency noise):
    - epsilon1 = EPSILON_DENSITY_FRACTION   * epsilon -> density -> d_tilde
    - epsilon2 = EPSILON_STRUCTURE_FRACTION * epsilon -> adjacency noise

    Sensitivity = 1 (changing one edge changes at most one entry by 1),
    so each upper-triangular entry receives Lap(1/epsilon2) noise.

    Privacy guarantee
    -----------------
    The output is a post-processing of two DP quantities:
      (a) adj_noisy_triu = adj_triu + Lap(1/epsilon2)  (epsilon2-DP, per entry,
          sensitivity 1), and
      (b) d_tilde, the Laplace-privatized density (epsilon1-DP).
    The binarization keeps the k = round(d_tilde * total_possible) entries
    with the largest noisy value. Both k (a function of d_tilde only) and the
    top-k selection (a function of adj_noisy_triu only) never re-access the
    real adjacency, so by sequential composition + post-processing the whole
    mechanism is (epsilon1 + epsilon2) = epsilon-DP.

    Density preservation
    --------------------
    Exactly k edges are emitted, so the perturbed density equals
    k / total_possible = d_tilde (up to integer rounding). The output density
    therefore tracks the private density target instead of depending on where
    the Laplace noise happens to fall.

    Parameters
    ----------
    data : PyG Data object
        Graph data containing edge_index and num_nodes
    epsilon : float
        Total privacy budget (smaller = more noise = more privacy).
        Split between density and structural noise according to
        dp_mechanisms/config.py.

    Returns
    -------
    edge_index_new : torch.Tensor
        New edge_index with DP applied via Laplace
    graph_changes : dict
        Statistics about the changes made
    """

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # 1. Build adjacency matrix (upper triangular only — undirected graph)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1
    adj = torch.triu(adj, diagonal=1)

    # Extract only the upper triangular indices (the meaningful entries)
    triu_idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
    adj_triu = adj[triu_idx[0], triu_idx[1]]  # shape: [N*(N-1)/2]

    # 2. Compute original density
    total_possible = num_nodes * (num_nodes - 1) / 2
    original_edges = adj_triu.sum().item()
    original_density = original_edges / total_possible

    # Split epsilon (single source of truth: dp_mechanisms/config.py)
    epsilon1, epsilon2 = split_epsilon(epsilon)

    # Perturb density using Laplace mechanism
    sensitivity_density = 1.0 / total_possible
    scale_density = sensitivity_density / epsilon1
    d_tilde = original_density + np.random.laplace(0, scale_density)
    d_tilde = np.clip(d_tilde, 1e-6, 1.0)

    # 3. Add Laplace noise ONLY to the upper triangular entries
    # Sensitivity = 1 (one edge flip changes one entry by exactly 1)
    sensitivity = 1.0
    scale = sensitivity / epsilon2

    noise_triu = torch.tensor(
        np.random.laplace(0, scale, size=adj_triu.shape),
        dtype=torch.float32
    )

    adj_noisy_triu = adj_triu + noise_triu

    # 4. Top-k selection driven by the PRIVATIZED density d_tilde.
    #
    # DP justification (this is the key correction):
    #   - k depends ONLY on d_tilde, which is itself a DP quantity
    #     (Laplace mechanism with budget epsilon1). No access to the
    #     real adjacency here.
    #   - The selection keeps the k entries with the largest value of
    #     adj_noisy_triu, i.e. it is a function ONLY of adj_noisy_triu
    #     (the Laplace mechanism output, budget epsilon2) and of k.
    #   Hence the final binary graph is a post-processing of the pair
    #   (adj_noisy_triu, d_tilde), which by sequential composition is
    #   (epsilon1 + epsilon2)-DP = epsilon-DP. Post-processing cannot
    #   weaken the guarantee.
    #
    # Density preservation: exactly k edges are produced, so the final
    # density equals k / total_possible = d_tilde (up to integer
    # rounding), guaranteeing the perturbed density matches the private
    # density target rather than depending on the noise distribution.
    k = int(round(d_tilde * total_possible))
    k = int(np.clip(k, 0, adj_noisy_triu.numel()))

    new_adj_triu = torch.zeros_like(adj_noisy_triu)
    if k > 0:
        # Indices of the k largest noisy entries (post-processing of the
        # Laplace output only).
        topk_idx = torch.topk(adj_noisy_triu, k).indices
        new_adj_triu[topk_idx] = 1.0
        # threshold reported for analysis: the smallest selected noisy value
        threshold = float(adj_noisy_triu[topk_idx].min().item())
    else:
        threshold = float('inf')

    # 5. Count changes (relative to the original triu)
    edges_removed = ((adj_triu == 1) & (new_adj_triu == 0)).sum().item()
    edges_added   = ((adj_triu == 0) & (new_adj_triu == 1)).sum().item()
    new_edges_count = new_adj_triu.sum().item()

    avg_noise = noise_triu.abs().mean().item()
    max_noise = noise_triu.abs().max().item()

    # 6. Reconstruct full symmetric adjacency and convert to edge_index
    new_adj_full = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    new_adj_full[triu_idx[0], triu_idx[1]] = new_adj_triu
    # Make symmetric for undirected graph
    new_adj_full = new_adj_full + new_adj_full.t()

    new_edges_coords = new_adj_full.nonzero(as_tuple=False).t()
    edge_index_new = new_edges_coords  # already symmetric (both directions)

    perturbed_density = float(new_edges_count / total_possible)

    graph_changes = {
        'original_edges': int(original_edges),
        'new_edges': int(new_edges_count),
        'edges_removed': int(edges_removed),
        'edges_added': int(edges_added),
        'total_changes': int(edges_removed + edges_added),
        'change_ratio': float((edges_removed + edges_added) / max(original_edges, 1)),
        'original_density': original_density,
        'perturbed_density': perturbed_density,
        'epsilon': epsilon,
        'epsilon1': epsilon1,
        'epsilon2': epsilon2,
        'scale_density': scale_density,
        'scale_adjacency': scale,
        'threshold': float(threshold),
        'avg_noise_magnitude': avg_noise,
        'max_noise_magnitude': max_noise,
    }

    return edge_index_new, graph_changes