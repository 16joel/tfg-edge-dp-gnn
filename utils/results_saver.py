"""
Module for saving experimental results in an organized and structured way.
Results are stored in directories with clear naming conventions.
"""

import os
import json
import csv
from datetime import datetime
import numpy as np


def create_results_directory(dataset_name="Cora", base_path="results"):
    """
    Create a timestamped results directory with clear structure.
    
    Directory naming format: {dataset}_{YYYY-MM-DD}_{HHMM}
    Example: Cora_2026-04-23_1630
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., "Cora")
    base_path : str
        Base path for results directory
        
    Returns
    -------
    results_dir : str
        Path to the newly created results directory
    """
    # Create timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H%M")
    
    # Create directory name
    dir_name = f"{dataset_name}_{timestamp}"
    results_dir = os.path.join(base_path, dir_name)
    
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    return results_dir


def save_summary(results_dir, stats, dataset_name="Cora", epsilon1=5.0, epsilon2=5.0, epsilon_laplace=1.0):
    """
    Save a human-readable summary of the results.
    
    Parameters
    ----------
    results_dir : str
        Path to results directory
    stats : dict
        Statistics dictionary from compute_statistics()
    dataset_name : str
        Name of dataset used
    epsilon1, epsilon2, epsilon_laplace : float
        Privacy parameters used
    """
    summary_file = os.path.join(results_dir, "summary.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DIFFERENTIAL PRIVACY ON GRAPH NEURAL NETWORKS - RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Metadata
        f.write("EXPERIMENT METADATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Dataset:              {dataset_name}\n")
        f.write(f"Task:                 Node Label Prediction\n")
        f.write(f"Number of Runs:       N/A\n")
        f.write(f"Timestamp:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nPrivacy Parameters:\n")
        f.write(f"  - RR DP (epsilon1):  {epsilon1}\n")
        f.write(f"  - RR DP (epsilon2):  {epsilon2}\n")
        f.write(f"  - Laplace DP:        {epsilon_laplace}\n\n")
        
        # Test Set Results
        f.write("TEST SET PERFORMANCE (Mean ± Std Dev)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<20} | {'Accuracy':<25} | {'F1 Score':<25}\n")
        f.write("-" * 80 + "\n")
        
        for method in ['baseline', 'rr_dp', 'laplace_dp']:
            acc_mean = stats[method]['test_accuracy_mean']
            acc_std = stats[method]['test_accuracy_std']
            f1_mean = stats[method]['test_f1_mean']
            f1_std = stats[method]['test_f1_std']
            
            method_name = "Baseline" if method == 'baseline' else ("RR DP" if method == 'rr_dp' else "Laplace DP")
            f.write(f"{method_name:<20} | {acc_mean:.4f} ± {acc_std:.4f}     | {f1_mean:.4f} ± {f1_std:.4f}\n")
        
        f.write("\n")
        
        # Validation Set Results
        f.write("VALIDATION SET PERFORMANCE (Mean ± Std Dev)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<20} | {'Accuracy':<25} | {'F1 Score':<25}\n")
        f.write("-" * 80 + "\n")
        
        for method in ['baseline', 'rr_dp', 'laplace_dp']:
            acc_mean = stats[method]['val_accuracy_mean']
            acc_std = stats[method]['val_accuracy_std']
            f1_mean = stats[method]['val_f1_mean']
            f1_std = stats[method]['val_f1_std']
            
            method_name = "Baseline" if method == 'baseline' else ("RR DP" if method == 'rr_dp' else "Laplace DP")
            f.write(f"{method_name:<20} | {acc_mean:.4f} ± {acc_std:.4f}     | {f1_mean:.4f} ± {f1_std:.4f}\n")
        
        f.write("\n")
        
        # Utility Loss
        f.write("UTILITY LOSS (% compared to Baseline)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<20} | {'Accuracy Loss':<20} | {'F1 Loss':<20}\n")
        f.write("-" * 80 + "\n")
        
        baseline_acc = stats['baseline']['test_accuracy_mean']
        baseline_f1 = stats['baseline']['test_f1_mean']
        
        for method, name in [('rr_dp', 'RR DP'), ('laplace_dp', 'Laplace DP')]:
            acc_loss = ((baseline_acc - stats[method]['test_accuracy_mean']) / baseline_acc) * 100
            f1_loss = ((baseline_f1 - stats[method]['test_f1_mean']) / baseline_f1) * 100
            
            f.write(f"{name:<20} | {acc_loss:+7.2f}%         | {f1_loss:+7.2f}%\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")


def save_detailed_results(results_dir, results):
    """
    Save detailed per-run results to CSV files.
    
    Creates separate CSV files for each method containing all 5 runs.
    
    Parameters
    ----------
    results_dir : str
        Path to results directory
    results : dict
        Results dictionary from run_multiple_experiments()
    """
    # Baseline results
    baseline_file = os.path.join(results_dir, "baseline_runs.csv")
    with open(baseline_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Test_Accuracy', 'Test_F1', 'Val_Accuracy', 'Val_F1'])
        for i, run_result in enumerate(results['baseline'], 1):
            writer.writerow([
                i,
                f"{run_result['test_accuracy']:.6f}",
                f"{run_result['test_f1']:.6f}",
                f"{run_result['val_accuracy']:.6f}",
                f"{run_result['val_f1']:.6f}"
            ])
    
    # RR DP results
    rr_dp_file = os.path.join(results_dir, "rr_dp_runs.csv")
    with open(rr_dp_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Test_Accuracy', 'Test_F1', 'Val_Accuracy', 'Val_F1'])
        for i, run_result in enumerate(results['rr_dp'], 1):
            writer.writerow([
                i,
                f"{run_result['test_accuracy']:.6f}",
                f"{run_result['test_f1']:.6f}",
                f"{run_result['val_accuracy']:.6f}",
                f"{run_result['val_f1']:.6f}"
            ])
    
    # Laplace DP results
    laplace_file = os.path.join(results_dir, "laplace_dp_runs.csv")
    with open(laplace_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Test_Accuracy', 'Test_F1', 'Val_Accuracy', 'Val_F1'])
        for i, run_result in enumerate(results['laplace_dp'], 1):
            writer.writerow([
                i,
                f"{run_result['test_accuracy']:.6f}",
                f"{run_result['test_f1']:.6f}",
                f"{run_result['val_accuracy']:.6f}",
                f"{run_result['val_f1']:.6f}"
            ])


def save_metadata(results_dir, dataset_name, num_nodes, num_edges, num_features, num_classes, num_runs=5,
                  budget_split=None):
    """
    Save metadata about the experiment.
    
    Parameters
    ----------
    results_dir : str
        Path to results directory
    dataset_name : str
        Name of dataset
    num_nodes : int
        Number of nodes
    num_edges : int
        Number of edges
    num_features : int
        Number of node features
    num_classes : int
        Number of classes
    num_runs : int
        Number of experimental runs
    """
    if budget_split is None:
        # Read from the single source of truth so the recorded split can
        # never diverge from the executed one.
        from dp_mechanisms import budget_split_description
        budget_split = budget_split_description()

    metadata_file = os.path.join(results_dir, "metadata.txt")
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("EXPERIMENT CONFIGURATION & DATASET INFORMATION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Name:            {dataset_name}\n")
        f.write(f"  Nodes:           {num_nodes}\n")
        f.write(f"  Edges:           {num_edges}\n")
        f.write(f"  Features:        {num_features}\n")
        f.write(f"  Classes:         {num_classes}\n\n")
        
        f.write("Experimental Configuration:\n")
        f.write(f"  Number of Runs:  {num_runs}\n")
        f.write(f"  Training Epochs: 500\n")
        f.write(f"  Model:           GCN (2 layers)\n")
        f.write(f"  Hidden Dims:     64\n")
        f.write(f"  Optimizer:       Adam (lr=0.01)\n\n")
        
        f.write("Methods Evaluated:\n")
        f.write(f"  1. Baseline (no DP)\n")
        f.write(f"  2. Edge DP Randomized Response\n")
        f.write(f"  3. Edge DP Laplace Mechanism\n\n")
        
        f.write("Privacy Configuration:\n")
        f.write(f"  Budget split:    {budget_split}\n\n")

        f.write("Evaluation Scenarios (DP runs):\n")
        f.write(f"  S1: train/val/eval on perturbed graph\n")
        f.write(f"  S2: train/val on perturbed graph, eval on original graph\n")
        f.write(f"  S3: train on perturbed graph, val/eval on original graph\n\n")

        f.write("Notes:\n")
        f.write(f"  - DP mechanisms perturb the FULL edge set of the graph\n")
        f.write(f"  - Node features, labels and masks are never perturbed\n")
        f.write(f"  - Fixed dataset split across all runs\n")
        f.write(f"  - Early stopping based on validation accuracy (dual\n")
        f.write(f"    trackers: perturbed-val for S1/S2, original-val for S3)\n")


def _infer_num_runs(all_results):
    """
    Infereix el nombre real d'execucions (runs) a partir de l'estructura
    all_results, comptant les files del baseline o, si no n'hi ha, del primer
    bloc d'epsilon/mètode disponible. Retorna 0 si no es pot determinar.
    """
    if isinstance(all_results.get('baseline'), list):
        return len(all_results['baseline'])
    for key, value in all_results.items():
        if key == 'baseline':
            continue
        if isinstance(value, dict):
            for mkey, mval in value.items():
                if isinstance(mval, list) and len(mval) > 0:
                    return len(mval)
    return 0


def save_all_results(results_dir, all_results, stats, dataset_info, privacy_params=None):
    """
    Master function to save all results to organized directory structure.
    Handles multiple epsilon values.
    
    Parameters
    ----------
    results_dir : str
        Path to results directory (from create_results_directory)
    all_results : dict
        Results from run_multiple_experiments() with multiple epsilons
    stats : dict
        Statistics from compute_statistics() with multiple epsilons
    dataset_info : dict
        Dataset information with keys: name, num_nodes, num_edges, num_features, num_classes
    privacy_params : dict, optional
        Privacy parameters with epsilon_values list
    """
    
    if privacy_params is None:
        privacy_params = {
            'epsilon_values': [0.5, 1.0, 2.0, 5.0, 10.0]
        }

    # Determinar el nombre REAL d'execucions a partir de les dades, en lloc
    # d'un valor fix. Es compta el nombre de runs del baseline (o, si no n'hi
    # ha, del primer mètode disponible). Així el resum mai divergeix del que
    # s'ha executat realment.
    num_runs = _infer_num_runs(all_results)

    # Save summary for all epsilons
    save_summary_multi_epsilon(results_dir, stats, dataset_info, privacy_params,
                               num_runs=num_runs)

    # Save detailed results for all epsilons
    save_detailed_results_multi_epsilon(results_dir, all_results)

    # Save metadata
    save_metadata(
        results_dir,
        dataset_name=dataset_info['name'],
        num_nodes=dataset_info['num_nodes'],
        num_edges=dataset_info['num_edges'],
        num_features=dataset_info['num_features'],
        num_classes=dataset_info['num_classes'],
        num_runs=num_runs,
        budget_split=privacy_params.get('budget_split')
    )
    
    print(f"\n✓ Results saved to: {results_dir}\n")
    print(f"Files created:")
    print(f"  - summary.txt         (Human-readable summary with all epsilons)")
    print(f"  - metadata.txt        (Experiment configuration)")
    print(f"  - detailed_results/   (CSV files for each epsilon)")
    print()


def save_summary_multi_epsilon(results_dir, stats, dataset_info, privacy_params, num_runs=None):
    """
    Save a human-readable summary for multiple epsilon values.
    """
    summary_file = os.path.join(results_dir, "summary.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DIFFERENTIAL PRIVACY ON GRAPH NEURAL NETWORKS - RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Metadata
        f.write("EXPERIMENT METADATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Dataset:              {dataset_info['name']}\n")
        f.write(f"Task:                 Node Label Prediction (60/20/20 split)\n")
        f.write(f"Number of Runs:       {num_runs if num_runs is not None else 'N/A'} per epsilon\n")
        f.write(f"Timestamp:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        budget_split = privacy_params.get('budget_split', 'see dp_mechanisms/config.py')
        scenarios = privacy_params.get('scenarios', None)
        f.write(f"Privacy Budget split: {budget_split}\n")
        if scenarios:
            f.write(f"Scenarios:            {scenarios}\n")
        f.write("\n")
        
        # Baseline results
        f.write(f"BASELINE RESULTS ({num_runs if num_runs is not None else 'N/A'} RUNS)\n")
        f.write("-" * 80 + "\n")
        
        if 'baseline' in stats:
            baseline = stats['baseline']
            f.write(f"Test Accuracy:   {baseline['test_accuracy_mean']:.4f} ± {baseline['test_accuracy_std']:.4f}\n")
            f.write(f"Test F1 Score:   {baseline['test_f1_mean']:.4f} ± {baseline['test_f1_std']:.4f}\n")
            f.write(f"Val Accuracy:    {baseline['val_accuracy_mean']:.4f} ± {baseline['val_accuracy_std']:.4f}\n")
            f.write(f"Val F1 Score:    {baseline['val_f1_mean']:.4f} ± {baseline['val_f1_std']:.4f}\n\n")
        
        # Results for each epsilon
        f.write("RESULTS FOR EACH EPSILON\n")
        f.write("-" * 80 + "\n\n")
        
        for key, value in sorted(stats.items()):
            if key == 'baseline':
                continue
            
            epsilon = value['epsilon']
            
            f.write(f"EPSILON = {epsilon:.2f}\n")
            f.write(f"{'Method':<15} | {'Test Acc':<12} | {'Test F1':<12} | {'Val Acc':<12} | {'Val F1':<12}\n")
            f.write("-" * 75 + "\n")
            
            method_keys = [k for k in value.keys() if k != 'epsilon']
            for method in method_keys:
                if method in value:
                    rr_data = value[method]
                    f.write(f"{method:<15} | {rr_data['test_accuracy_mean']:.4f}±{rr_data['test_accuracy_std']:.4f} | ")
                    f.write(f"{rr_data['test_f1_mean']:.4f}±{rr_data['test_f1_std']:.4f} | ")
                    f.write(f"{rr_data['val_accuracy_mean']:.4f}±{rr_data['val_accuracy_std']:.4f} | ")
                    f.write(f"{rr_data['val_f1_mean']:.4f}±{rr_data['val_f1_std']:.4f}\n")
            
            f.write("\n")


def save_detailed_results_multi_epsilon(results_dir, all_results):
    """
    Save detailed results (per run) for multiple epsilon values to CSV files.
    """
    # Create detailed results directory
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    # Save baseline
    if 'baseline' in all_results:
        baseline_file = os.path.join(detailed_dir, "baseline.csv")
        with open(baseline_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Run', 'Test Accuracy', 'Test F1', 'Val Accuracy', 'Val F1'])
            
            for i, result in enumerate(all_results['baseline'], 1):
                writer.writerow([
                    i,
                    f"{result['test_accuracy']:.4f}",
                    f"{result['test_f1']:.4f}",
                    f"{result['val_accuracy']:.4f}",
                    f"{result['val_f1']:.4f}"
                ])
    
    # Save for each epsilon
    for key, value in all_results.items():
        if key == 'baseline':
            continue
        
        if isinstance(value, dict) and 'epsilon' in value:
            epsilon = value['epsilon']
            epsilon_dir = os.path.join(detailed_dir, f"epsilon_{epsilon:.2f}")
            os.makedirs(epsilon_dir, exist_ok=True)
            
            # Save one CSV per method present for this epsilon
            # (e.g. rr_dp_s1, rr_dp_s2, rr_dp_s3, laplace_dp_s1, ...).
            # Iterating dynamically guarantees that every scenario produced
            # by the experimental loop is persisted.
            for method, runs in value.items():
                if method == 'epsilon' or not isinstance(runs, list):
                    continue
                method_file = os.path.join(epsilon_dir, f"{method}.csv")
                with open(method_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Run', 'Test Accuracy', 'Test F1', 'Val Accuracy', 'Val F1'])

                    for i, result in enumerate(runs, 1):
                        writer.writerow([
                            i,
                            f"{result['test_accuracy']:.4f}",
                            f"{result['test_f1']:.4f}",
                            f"{result['val_accuracy']:.4f}",
                            f"{result['val_f1']:.4f}"
                        ])


# ============================================================================
# INFERENCE ATTACK RESULT SAVING FUNCTIONS
# ============================================================================

# Ordre canònic dels scores (ha de coincidir amb SCORE_NAMES a
# inference_attack.py). Es defineix aquí per no acoblar el saver a l'import
# del mòdul d'atac, però l'ordre és el mateix.
_ATTACK_SCORE_NAMES = [
    'common_neighbors',
    'jaccard',
    'adamic_adar',
    'resource_allocation',
    'preferential_attachment',
    'embedding_cosine',
]


def save_inference_attack_summary(results_dir, stats, epsilon_values):
    """
    Save a human-readable summary of the UNSUPERVISED inference attack.

    Each score (Common Neighbors, Jaccard, ..., embedding cosine) is an
    independent attack; AUC-ROC and Average Precision are reported per score
    and per configuration. AP must be read against the prevalence printed for
    each configuration (a random attack achieves AP ~ prevalence).
    """
    summary_file = os.path.join(results_dir, "summary.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 88 + "\n")
        f.write("EDGE INFERENCE ATTACK AGAINST EDGE DIFFERENTIAL PRIVACY (UNSUPERVISED)\n")
        f.write("=" * 88 + "\n\n")

        f.write("EXPERIMENT METADATA\n")
        f.write("-" * 88 + "\n")
        f.write(f"Dataset:              Cora\n")
        f.write(f"Task:                 Edge Inference Attack (unsupervised, score-based)\n")
        f.write(f"Adversary Model:      Black-box, NO knowledge of any real edge\n")
        f.write(f"                      (observes only the published graph G' + GCN embeddings)\n")
        f.write(f"Timestamp:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Attack scores:        CN, Jaccard, Adamic-Adar, Resource Allocation,\n")
        f.write(f"                      Preferential Attachment, embedding cosine\n")
        f.write(f"Epsilon Values:       {epsilon_values}\n\n")

        def write_block(key, label):
            if key not in stats:
                return
            prevalence = stats[key].get('prevalence', float('nan'))
            f.write(f"{label}  (random-AP baseline ~ prevalence = {prevalence:.4f})\n")
            f.write(f"{'Score':<26} {'AUC-ROC':<20} {'Avg Precision':<20}\n")
            f.write("-" * 66 + "\n")
            for name in _ATTACK_SCORE_NAMES:
                sc = stats[key]['scores'][name]
                f.write(f"{name:<26} "
                        f"{sc['auc_roc']['mean']:.4f}±{sc['auc_roc']['std']:.3f}      "
                        f"{sc['avg_precision']['mean']:.4f}±{sc['avg_precision']['std']:.3f}\n")
            f.write("\n")

        f.write("BASELINE RESULTS (No DP)\n")
        f.write("-" * 88 + "\n")
        write_block('Baseline', 'Baseline (no DP)')

        f.write("RESULTS FOR EACH EPSILON (Mean +/- Std Dev across runs)\n")
        f.write("-" * 88 + "\n\n")
        for epsilon in sorted(epsilon_values):
            f.write(f"EPSILON = {epsilon:.2f}\n")
            for mech in ['RR', 'Laplace']:
                write_block(f'{mech}_eps_{epsilon:.2f}', f"  {mech}")
            f.write("\n")

        f.write("=" * 88 + "\n")
        f.write("INTERPRETATION\n")
        f.write("-" * 88 + "\n")
        f.write("AUC-ROC = 0.50  -> perfect privacy (attacker no better than random)\n")
        f.write("AUC-ROC = 1.00  -> complete edge-privacy breach\n")
        f.write("Average Precision must be compared to the per-configuration prevalence\n")
        f.write("(a random attack achieves AP ~ prevalence).\n")
        f.write("Each score is an INDEPENDENT attack; no classifier is trained, so the\n")
        f.write("adversary never accesses any ground-truth edge.\n")
        f.write("=" * 88 + "\n")


def save_inference_attack_detailed_results(results_dir, all_results, epsilon_values):
    """
    Save detailed per-run results to CSV files organized by epsilon.

    One CSV row per run; columns hold AUC and AP for every score.
    """
    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)

    header = ['Run', 'Prevalence']
    for name in _ATTACK_SCORE_NAMES:
        header.append(f'{name}_AUC')
        header.append(f'{name}_AP')

    def write_csv(filepath, runs):
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, result in enumerate(runs, 1):
                row = [i, f"{result['prevalence']:.6f}"]
                for name in _ATTACK_SCORE_NAMES:
                    sc = result['scores'][name]
                    row.append(f"{sc['auc_roc']:.4f}")
                    row.append(f"{sc['avg_precision']:.4f}")
                writer.writerow(row)

    if 'Baseline' in all_results:
        write_csv(os.path.join(detailed_dir, "baseline.csv"), all_results['Baseline'])

    for epsilon in sorted(epsilon_values):
        epsilon_dir = os.path.join(detailed_dir, f"epsilon_{epsilon:.2f}")
        os.makedirs(epsilon_dir, exist_ok=True)
        for mech in ['RR', 'Laplace']:
            key = f'{mech}_eps_{epsilon:.2f}'
            if key in all_results:
                write_csv(os.path.join(epsilon_dir, f"{mech.lower()}.csv"),
                          all_results[key])


def save_inference_attack_metadata(results_dir, num_nodes, num_edges, num_features,
                                   num_classes, num_runs, epsilon_values,
                                   dataset_name="Cora"):
    """
    Save metadata about the unsupervised inference attack experiment.
    """
    metadata_file = os.path.join(results_dir, "metadata.txt")

    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("EDGE INFERENCE ATTACK EXPERIMENT CONFIGURATION (UNSUPERVISED)\n")
        f.write("=" * 70 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  Name:            {dataset_name}\n")
        f.write(f"  Nodes:           {num_nodes}\n")
        f.write(f"  Edges:           {num_edges}\n")
        f.write(f"  Features:        {num_features}\n")
        f.write(f"  Classes:         {num_classes}\n\n")

        f.write("Experimental Configuration:\n")
        f.write(f"  Number of Runs per Config:  {num_runs}\n")
        f.write(f"  Epsilon Values:             {epsilon_values}\n")
        f.write(f"  Total Configurations:       1 baseline + {len(epsilon_values) * 2} DP\n")
        f.write(f"  GCN Training Epochs:        500\n")
        f.write(f"  GCN Model:                  2-layer GCN (hidden=64)\n\n")

        f.write("Adversary Model (threat model):\n")
        f.write(f"  - Observes ONLY the published graph G' = DP(G) and node features\n")
        f.write(f"  - Trains a GCN on G' and extracts embeddings from it\n")
        f.write(f"  - Knows NO real edge: no supervised classifier is trained\n")
        f.write(f"  - The original graph G is used ONLY by the evaluator to measure\n")
        f.write(f"    AUC-ROC / AP; the adversary never accesses it\n\n")

        f.write("Attack Pipeline:\n")
        f.write(f"  1. Perturb the full graph with edge-DP (or no DP for baseline)\n")
        f.write(f"  2. Train GCN on the published graph G'\n")
        f.write(f"  3. Extract GCN embeddings from G'\n")
        f.write(f"  4. Build evaluation universe of pairs (deduplicated negatives,\n")
        f.write(f"     sampled without replacement) + ground truth from G\n")
        f.write(f"  5. Compute link-prediction scores on G' (CN, Jaccard, AA, RA, PA,\n")
        f.write(f"     embedding cosine) -- one independent attack per score\n")
        f.write(f"  6. Measure AUC-ROC and Average Precision of each score vs ground truth\n\n")

        f.write("Scores (each is an independent unsupervised attack):\n")
        f.write(f"  - common_neighbors, jaccard, adamic_adar, resource_allocation,\n")
        f.write(f"    preferential_attachment   (structural heuristics on G')\n")
        f.write(f"  - embedding_cosine          (cosine similarity of GCN embeddings)\n\n")

        f.write("Notes:\n")
        f.write(f"  - DP mechanisms perturb the FULL edge set of the graph\n")
        f.write(f"  - Negatives are sampled WITHOUT replacement and deduplicated, so the\n")
        f.write(f"    evaluation universe contains no repeated pairs\n")
        f.write(f"  - No train/test split and no classifier: nothing to leak by construction\n")
        f.write(f"  - Average Precision must be read against the reported prevalence\n")


def save_inference_attack_results(results_dir, all_results, stats, data,
                                 epsilon_values, num_runs, dataset_name="Cora"):
    """
    Master function to save all unsupervised inference attack results.
    Mirrors save_all_results for the utility experiments.
    """
    save_inference_attack_summary(results_dir, stats, epsilon_values)
    save_inference_attack_detailed_results(results_dir, all_results, epsilon_values)
    save_inference_attack_metadata(
        results_dir,
        num_nodes=data.num_nodes,
        num_edges=data.num_edges,
        num_features=data.num_features,
        num_classes=data.y.max().item() + 1,
        num_runs=num_runs,
        epsilon_values=epsilon_values,
        dataset_name=dataset_name,
    )

    print(f"\n\u2713 Inference attack results saved to: {results_dir}\n")
    print(f"Files created:")
    print(f"  - summary.txt              (per-score AUC-ROC and AP, all epsilons)")
    print(f"  - metadata.txt             (threat model and attack pipeline)")
    print(f"  - detailed_results/        (CSV per configuration, one row per run)")
    print()