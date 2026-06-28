"""
Main execution script for evaluating baseline and DP mechanisms on node label
prediction.

Pipeline:
1. Load graph dataset (Cora)
2. Create fixed train/validation/test split
3. Execute baseline (no DP): train, validate and evaluate on the ORIGINAL graph
4. Execute DP mechanisms (5 runs each). The DP mechanism perturbs the FULL
   edge set of the graph; each run is then evaluated under three scenarios:

     S1: train, validate and evaluate on the PERTURBED graph.
     S2: train and validate on the PERTURBED graph, evaluate on the ORIGINAL.
     S3: train on the PERTURBED graph, validate and evaluate on the ORIGINAL.

5. Compute statistics (mean, std) across runs
6. Report and persist results

The epsilon budget split between the density query and the structural
perturbation is defined once in dp_mechanisms/config.py and read from there
for every print and every saved artifact.
"""

import argparse
import torch
import numpy as np
import copy
from torch_geometric.datasets import Planetoid

# Conjunts de dades Planetoid admesos. Tots comparteixen exactament la mateixa
# API (mateix objecte Data, mateixes màscares), per la qual cosa tot el
# pipeline existent (split 60/20/20, mecanismes DP, escenaris) és reutilitzable
# sense cap canvi: només cal variar el nom passat a Planetoid.
SUPPORTED_DATASETS = ('Cora', 'Citeseer', 'Pubmed')

# Import models
from models import GCN

# Import DP mechanisms and budget-split configuration (single source of truth)
from dp_mechanisms import (
    edge_dp_randomized_response,
    edge_dp_laplace_mechanism,
    split_epsilon,
    budget_split_description,
    EPSILON_DENSITY_FRACTION,
    EPSILON_STRUCTURE_FRACTION,
)

# Import utils
from utils import train_and_evaluate, train_and_evaluate_scenarios
from utils.results_saver import create_results_directory, save_all_results


# Method keys produced by the experimental loop, in reporting order.
METHOD_KEYS = [
    'rr_dp_s1', 'rr_dp_s2', 'rr_dp_s3',
    'laplace_dp_s1', 'laplace_dp_s2', 'laplace_dp_s3',
]

SCENARIO_LEGEND = (
    "S1 = train/val/eval on perturbed graph | "
    "S2 = train/val on perturbed, eval on original | "
    "S3 = train on perturbed, val/eval on original"
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_custom_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Create custom train/validation/test split with specified ratios.
    
    Uses only the labeled nodes and creates new masks with the specified proportions.
    This ensures reproducibility by using a fixed seed.
    
    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data with existing masks
    train_ratio : float
        Proportion for training (default: 0.6)
    val_ratio : float
        Proportion for validation (default: 0.2)
    test_ratio : float
        Proportion for testing (default: 0.2)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    data : torch_geometric.data.Data
        Modified data with new train_mask, val_mask, test_mask
    """
    # Get all labeled nodes from the original masks
    labeled_mask = data.train_mask | data.val_mask | data.test_mask
    labeled_indices = torch.where(labeled_mask)[0]
    
    # Shuffle with seed for reproducibility
    rng = np.random.RandomState(seed)
    shuffled_indices = labeled_indices[rng.permutation(len(labeled_indices))]
    
    # Calculate split sizes
    total_labeled = len(shuffled_indices)
    train_size = int(total_labeled * train_ratio)
    val_size = int(total_labeled * val_ratio)
    
    # Create new masks
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]
    
    # Initialize new masks
    new_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    new_val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    new_test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    # Set masks
    new_train_mask[train_indices] = True
    new_val_mask[val_indices] = True
    new_test_mask[test_indices] = True
    
    # Update data
    data.train_mask = new_train_mask
    data.val_mask = new_val_mask
    data.test_mask = new_test_mask
    
    return data


def apply_dp_to_graph(data, dp_func, dp_params, seed=None):
    """
    Apply a DP mechanism to the FULL edge set of the graph.

    The mechanism perturbs every entry of the (upper-triangular) adjacency
    matrix, i.e. the entire topology. It does NOT restrict the perturbation
    to training edges: the published artifact is one single private graph,
    consistent with the edge-DP release model. What changes between
    scenarios (S1/S2/S3) is which graph — perturbed or original — is used
    for validation and final evaluation, not which edges get perturbed.

    Node features, labels and the train/val/test masks are never modified.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data with train_mask, val_mask, test_mask
    dp_func : callable
        DP mechanism function to apply
    dp_params : dict or tuple
        Parameters for the DP function
    seed : int, optional
        Random seed for reproducibility (sets torch and numpy seeds)
        
    Returns
    -------
    data_dp : torch_geometric.data.Data
        Copy of data whose edge_index is the DP-perturbed topology
    changes : dict
        Statistics about the changes made
    """
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    data_dp = copy.deepcopy(data)
    
    # Apply DP to edges
    if isinstance(dp_params, dict):
        dp_edge_index, changes = dp_func(data, **dp_params)
    else:
        dp_edge_index, changes = dp_func(data, *dp_params)
    
    data_dp.edge_index = dp_edge_index
    
    return data_dp, changes


def run_multiple_experiments(data, epsilon_values, num_runs=5, num_epochs=500, patience=20, seed_base=42):
    """
    Run baseline and DP experiments multiple times with different epsilon values.

    For each (mechanism, epsilon, run), the graph is perturbed once and the
    model is trained once; the three scenarios S1/S2/S3 are derived from that
    single training via train_and_evaluate_scenarios (see utils/metrics.py).

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data (original, unperturbed)
    epsilon_values : list or float
        List of epsilon values to test, or a single float
    num_runs : int
        Number of experimental runs per epsilon
    num_epochs : int
        Maximum number of training epochs per run
    patience : int
        Early stopping patience (epochs without val improvement)
    seed_base : int
        Base random seed
        
    Returns
    -------
    all_results : dict
        Results for each epsilon and method. For each epsilon key, the
        methods are those in METHOD_KEYS.
    """
    
    # Convert single epsilon to list
    if isinstance(epsilon_values, (int, float)):
        epsilon_values = [epsilon_values]

    # Monitor de rendiment (temps, RAM, overhead). Import local per no afegir
    # dependència dura del mòdul si no s'usa des d'altres punts d'entrada.
    from utils.perf_monitor import PerfMonitor
    perf = PerfMonitor()

    all_results = {}
    
    # ========== BASELINE (NO DP) ==========
    print("="*70)
    print(f"BASELINE EXPERIMENT ({num_runs} RUNS)")
    print("="*70 + "\n")
    
    baseline_results = []
    
    for run in range(1, num_runs + 1):
        print(f"RUN {run}/{num_runs}")
        print("-" * 70)
        torch.manual_seed(seed_base + run)
        np.random.seed(seed_base + run)
        
        baseline_model = GCN(data.num_features, 64, data.y.max().item() + 1)
        with perf.timer('baseline_train'):
            run_results, _ = train_and_evaluate(baseline_model, data, num_epochs=num_epochs, epochs_log=50, patience=patience)
        baseline_results.append(run_results)
    
    all_results['baseline'] = baseline_results
    
    # ========== DP EXPERIMENTS FOR EACH EPSILON ==========
    mechanisms = [
        ('rr_dp', 'EDGE DP RANDOMIZED RESPONSE', edge_dp_randomized_response),
        ('laplace_dp', 'EDGE DP LAPLACE MECHANISM', edge_dp_laplace_mechanism),
    ]

    for epsilon_idx, epsilon_dp in enumerate(epsilon_values, 1):
        print("\n" + "="*70)
        print(f"EPSILON = {epsilon_dp:.2f} (BATCH {epsilon_idx}/{len(epsilon_values)})")
        print("="*70 + "\n")
        
        epsilon_results = {'epsilon': epsilon_dp}
        for key in METHOD_KEYS:
            epsilon_results[key] = []

        eps1, eps2 = split_epsilon(epsilon_dp)

        for method_key, method_title, dp_func in mechanisms:
            print("="*70)
            print(f"{method_title} ({num_runs} RUNS)")
            print("="*70 + "\n")
            print(f"Privacy Parameter: epsilon={epsilon_dp} "
                  f"(split: {eps1:.4f} for density [{EPSILON_DENSITY_FRACTION:.0%}], "
                  f"{eps2:.4f} for structure [{EPSILON_STRUCTURE_FRACTION:.0%}])\n")

            for run in range(1, num_runs + 1):
                print(f"RUN {run}/{num_runs}")
                print("-" * 70)

                # Perturb the FULL graph once per (mechanism, epsilon, run)
                with perf.timer('dp_perturb', mechanism=method_key, epsilon=epsilon_dp):
                    data_dp, changes = apply_dp_to_graph(
                        data, dp_func, (epsilon_dp,), seed=seed_base + run
                    )

                print(f"Graph modifications:")
                print(f"  Original edges: {changes['original_edges']}")
                print(f"  New edges: {changes['new_edges']}")
                print(f"  Added: {changes['edges_added']}, Removed: {changes['edges_removed']}")
                if method_key == 'rr_dp':
                    print(f"  Probability keep edge (p1): {changes['p1']:.4f}")
                    print(f"  Probability keep non-edge (p0): {changes['p0']:.4f}\n")
                else:
                    print(f"  Original density: {changes['original_density']:.6f}")
                    print(f"  Perturbed density: {changes['perturbed_density']:.6f}")
                    print(f"  Avg noise magnitude: {changes['avg_noise_magnitude']:.4f}\n")

                # One training, three scenarios (S1, S2, S3)
                dp_model = GCN(data.num_features, 64, data.y.max().item() + 1)
                with perf.timer('dp_train', mechanism=method_key, epsilon=epsilon_dp):
                    scenario_results = train_and_evaluate_scenarios(
                        dp_model,
                        data_perturbed=data_dp,
                        data_original=data,
                        num_epochs=num_epochs,
                        epochs_log=50,
                        patience=patience,
                    )

                epsilon_results[f'{method_key}_s1'].append(scenario_results['s1'])
                epsilon_results[f'{method_key}_s2'].append(scenario_results['s2'])
                epsilon_results[f'{method_key}_s3'].append(scenario_results['s3'])
        
        # Store results for this epsilon
        all_results[f'epsilon_{epsilon_dp:.2f}'] = epsilon_results

    perf.snapshot_memory()
    return all_results, perf


def compute_statistics(all_results):
    """
    Compute mean and std of results across runs for all epsilons.
    
    Parameters
    ----------
    all_results : dict
        Results from multiple runs and epsilons
        
    Returns
    -------
    stats : dict
        Statistics for each epsilon and method
    """
    def _aggregate(method_results):
        return {
            'test_accuracy_mean': np.mean([r['test_accuracy'] for r in method_results]),
            'test_accuracy_std': np.std([r['test_accuracy'] for r in method_results], ddof=1),
            'test_f1_mean': np.mean([r['test_f1'] for r in method_results]),
            'test_f1_std': np.std([r['test_f1'] for r in method_results], ddof=1),
            'val_accuracy_mean': np.mean([r['val_accuracy'] for r in method_results]),
            'val_accuracy_std': np.std([r['val_accuracy'] for r in method_results], ddof=1),
            'val_f1_mean': np.mean([r['val_f1'] for r in method_results]),
            'val_f1_std': np.std([r['val_f1'] for r in method_results], ddof=1),
        }

    stats = {}
    
    # Handle baseline
    if 'baseline' in all_results:
        stats['baseline'] = _aggregate(all_results['baseline'])
    
    # Handle each epsilon
    for key, value in all_results.items():
        if key == 'baseline':
            continue
        
        if isinstance(value, dict) and 'epsilon' in value:
            stats[key] = {'epsilon': value['epsilon']}
            
            for method in METHOD_KEYS:
                if method not in value or len(value[method]) == 0:
                    continue
                stats[key][method] = _aggregate(value[method])
    
    return stats


def print_results_summary(stats):
    """
    Print formatted results summary for all epsilons.
    
    Parameters
    ----------
    stats : dict
        Statistics computed from multiple runs and epsilons
    """
    # Print baseline results
    print("\n" + "="*70)
    print("BASELINE RESULTS")
    print("="*70 + "\n")
    
    if 'baseline' in stats:
        baseline = stats['baseline']
        print("Test Set Performance:")
        print(f"  Accuracy: {baseline['test_accuracy_mean']:.4f} ± {baseline['test_accuracy_std']:.4f}")
        print(f"  F1 Score: {baseline['test_f1_mean']:.4f} ± {baseline['test_f1_std']:.4f}\n")
        
        print("Validation Set Performance:")
        print(f"  Accuracy: {baseline['val_accuracy_mean']:.4f} ± {baseline['val_accuracy_std']:.4f}")
        print(f"  F1 Score: {baseline['val_f1_mean']:.4f} ± {baseline['val_f1_std']:.4f}\n")
    
    # Print results for each epsilon
    for key, value in sorted(stats.items()):
        if key == 'baseline':
            continue
        
        epsilon = value['epsilon']
        
        print("\n" + "="*70)
        print(f"RESULTS FOR EPSILON = {epsilon:.2f}")
        print("="*70 + "\n")
        print(SCENARIO_LEGEND + "\n")

        print("Test Set Performance:")
        print(f"{'Method':<18} | {'Accuracy':<25} | {'F1 Score':<25}")
        print(f"{'-'*72}")

        for method in METHOD_KEYS:
            if method in value:
                m = value[method]
                print(f"{method:<18} | {m['test_accuracy_mean']:.4f} ± {m['test_accuracy_std']:.4f}     "
                      f"| {m['test_f1_mean']:.4f} ± {m['test_f1_std']:.4f}")

        print("\nValidation Set Performance (on each scenario's own validation graph):")
        print(f"{'Method':<18} | {'Accuracy':<25} | {'F1 Score':<25}")
        print(f"{'-'*72}")

        for method in METHOD_KEYS:
            if method in value:
                m = value[method]
                print(f"{method:<18} | {m['val_accuracy_mean']:.4f} ± {m['val_accuracy_std']:.4f}     "
                      f"| {m['val_f1_mean']:.4f} ± {m['val_f1_std']:.4f}")

        # Compute utility loss
        if 'baseline' in stats:
            baseline_acc = stats['baseline']['test_accuracy_mean']
            baseline_f1 = stats['baseline']['test_f1_mean']

            print("\nUtility Loss (% compared to Baseline):")
            print(f"{'Method':<18} | {'Accuracy Loss':<20} | {'F1 Loss':<20}")
            print(f"{'-'*62}")

            for method in METHOD_KEYS:
                if method in value:
                    acc_loss = ((baseline_acc - value[method]['test_accuracy_mean']) / baseline_acc) * 100
                    f1_loss = ((baseline_f1 - value[method]['test_f1_mean']) / baseline_f1) * 100
                    print(f"{method:<18} | {acc_loss:+7.2f}%         | {f1_loss:+7.2f}%")

    print("\n" + "="*70)
    print("OVERALL SUMMARY (Test Accuracy by scenario)")
    print("="*70 + "\n")
    print(SCENARIO_LEGEND + "\n")

    header = f"{'Epsilon':<10}"
    for method in METHOD_KEYS:
        header += f" | {method:<14}"
    print(header)
    print("-" * len(header))

    for key, value in sorted(stats.items()):
        if key == 'baseline':
            continue

        row = f"{value['epsilon']:<10.2f}"
        for method in METHOD_KEYS:
            acc = value[method]['test_accuracy_mean'] if method in value else float('nan')
            row += f" | {acc:<14.4f}"
        print(row)

    if 'baseline' in stats:
        print(f"\n{'Baseline':<10} | {stats['baseline']['test_accuracy_mean']:.4f} "
              f"(train/val/eval on original graph)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(dataset_name='Cora', epsilon_values=None, num_runs=5,
         num_epochs=500, patience=20, global_seed=42):
    """
    Executa el pipeline complet (baseline + DP per escenaris) sobre un
    conjunt Planetoid.

    Tots els paràmetres tenen valors per defecte idèntics al comportament
    original (Cora, epsilon=[0.01], 5 execucions), de manera que cridar
    main() sense arguments reprodueix exactament l'experiment previ. La
    parametrització només AFEGEIX la possibilitat d'executar Citeseer o
    Pubmed i d'escollir l'escombrat d'epsilon, sense duplicar cap lògica:
    create_custom_split, run_multiple_experiments, compute_statistics i
    print_results_summary es reutilitzen tal qual.

    Parameters
    ----------
    dataset_name : str
        Un de SUPPORTED_DATASETS ('Cora', 'Citeseer', 'Pubmed').
    epsilon_values : list[float] or None
        Escombrat d'epsilon. Si és None, s'usa [0.01] (comportament previ).
    num_runs : int
        Execucions per configuració.
    num_epochs, patience : int
        Hiperparàmetres d'entrenament (early stopping).
    global_seed : int
        Llavor global de reproductibilitat.

    Returns
    -------
    all_results, stats, data, dataset_name, perf
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' no admès. Opcions: {SUPPORTED_DATASETS}"
        )
    if epsilon_values is None:
        epsilon_values = [0.01]

    # ========== REPRODUCIBILITY: SET GLOBAL SEEDS ==========
    GLOBAL_SEED = global_seed
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # =======================================================
    print("="*70)
    print("DIFFERENTIAL PRIVACY ON GRAPH NEURAL NETWORKS")
    print("Task: Node Label Prediction")
    print("="*70 + "\n")
    print(f"Global Seed: {GLOBAL_SEED}")
    print(f"Dataset: {dataset_name}")
    print(f"Epsilon budget split: {budget_split_description()}")
    print(f"Scenarios: {SCENARIO_LEGEND}\n")

    # ========== 1. LOAD AND PREPARE DATA ==========
    print(f"1. Loading dataset ({dataset_name})...")
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
    data = dataset[0]
    
    print(f"   Original dataset info:")
    print(f"   - Total nodes: {data.num_nodes}")
    print(f"   - Edges: {data.num_edges}")
    print(f"   - Features: {data.num_features}")
    print(f"   - Classes: {data.y.max().item() + 1}")
    print(f"   - Original train: {data.train_mask.sum().item()}, val: {data.val_mask.sum().item()}, test: {data.test_mask.sum().item()}\n")
    
    # ========== CREATE CUSTOM 60/20/20 SPLIT ==========
    print("2. Creating custom 60/20/20 split...")
    data = create_custom_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=GLOBAL_SEED)
    
    print(f"   Custom split (60/20/20):")
    print(f"   - Nodes: {data.num_nodes}")
    print(f"   - Edges: {data.num_edges}")
    print(f"   - Features: {data.num_features}")
    print(f"   - Classes: {data.y.max().item() + 1}")
    print(f"   - Train nodes: {data.train_mask.sum().item()}")
    print(f"   - Val nodes: {data.val_mask.sum().item()}")
    print(f"   - Test nodes: {data.test_mask.sum().item()}\n")

    # ========== 2. RUN EXPERIMENTS WITH MULTIPLE EPSILONS ==========
    print(f"3. Running experiments with epsilon values: {epsilon_values}\n")
    all_results, perf = run_multiple_experiments(
        data, epsilon_values, num_runs=num_runs,
        num_epochs=num_epochs, patience=patience, seed_base=global_seed,
    )

    # ========== 3. COMPUTE STATISTICS ==========
    stats = compute_statistics(all_results)

    # ========== 4. PRINT RESULTS ==========
    print_results_summary(stats)

    return all_results, stats, data, dataset_name, perf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Edge-DP sobre GNN: baseline + mecanismes DP per escenari."
    )
    parser.add_argument(
        '--dataset', default='Cora', choices=SUPPORTED_DATASETS,
        help="Conjunt Planetoid a utilitzar (per defecte: Cora)."
    )
    parser.add_argument(
        '--epsilons', type=float, nargs='+', default=None,
        help="Llista d'epsilons (p. ex. --epsilons 0.1 1 3 5 7 9 11 13 15). "
             "Per defecte [0.01], idèntic al comportament original."
    )
    parser.add_argument(
        '--runs', type=int, default=5,
        help="Execucions per configuració (per defecte: 5)."
    )
    parser.add_argument(
        '--epochs', type=int, default=500,
        help="Èpoques màximes d'entrenament (per defecte: 500)."
    )
    parser.add_argument(
        '--patience', type=int, default=20,
        help="Paciència d'early stopping (per defecte: 20)."
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Llavor global (per defecte: 42)."
    )
    args = parser.parse_args()

    all_results, stats, data, dataset_name, perf = main(
        dataset_name=args.dataset,
        epsilon_values=args.epsilons,
        num_runs=args.runs,
        num_epochs=args.epochs,
        patience=args.patience,
        global_seed=args.seed,
    )

    # ========== 5. SAVE RESULTS ==========
    print("="*70)
    print("SAVING RESULTS")
    print("="*70)

    results_dir = create_results_directory(dataset_name=dataset_name, base_path="results")

    dataset_info = {
        'name': dataset_name,
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.num_features,
        'num_classes': data.y.max().item() + 1
    }

    # Extract epsilon values from results
    epsilon_values = []
    for key, value in all_results.items():
        if key != 'baseline' and isinstance(value, dict) and 'epsilon' in value:
            epsilon_values.append(value['epsilon'])

    privacy_params = {
        'epsilon_values': sorted(epsilon_values),
        # Read from dp_mechanisms/config.py so the saved description can
        # never diverge from the executed split.
        'budget_split': budget_split_description(),
        'scenarios': SCENARIO_LEGEND,
    }

    save_all_results(results_dir, all_results, stats, dataset_info, privacy_params)

    # Desar mètriques de rendiment (temps, RAM, overhead de privadesa)
    from utils.perf_monitor import save_performance
    save_performance(results_dir, perf, dataset_name=dataset_name)

    print("✓ Analysis complete!\n")
