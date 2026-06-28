"""
Pipeline One-vs-Rest (OvR) per a Edge-Level Differential Privacy sobre GNN.

Converteix la classificació multiclasse de Cora (7 classes) a 7 tasques
binàries independents usant l'esquema One-vs-Rest. Per a cada tasca:
  - y_bin = 1 si y_original == c, y_bin = 0 altrament
  - S'entrena un GCN(out_channels=2) independent
  - S'avaluen mètriques binàries: accuracy, F1 binari, precision, recall, AUC-ROC

Raó per OvR i no per altres reduccions binàries:
  OvR maximitza la informació disponible (tota la mostra s'usa per a cada
  tasca), manté els hiperparàmetres idèntics entre classes i permet una
  interpretació directa per classe. Alternatives com One-vs-One (OvO) o
  binaritzar totes les classes alhora serien menys comparables amb main.py.

Raó per usar F1 binari i AUC-ROC i no accuracy:
  Cora sota OvR té un fort desequilibri de classes (entre ~10% i ~30% de
  positius per classe). Accuracy seria trivial: un classificador que prediu
  sempre la classe majoritària ("negatiu") obtindria 70-90% sense aprendre
  res. F1 binari i AUC-ROC mesuren directament la qualitat sobre la classe
  minoritaria d'interès (positiu = classe c).

Raó per la qual la DP no es veu afectada per la transformació d'etiquetes:
  El mecanisme DP actua sobre l'adjacència del graf (edge_index), que és
  completament independent de les etiquetes dels nodes. Per tant, la
  pertorbació es pot aplicar una sola vegada per (run, epsilon, mètode) i
  reutilitzar per a les 7 tasques OvR: la garantia ε-edge-DP és sobre
  les arestes, no sobre les etiquetes.
"""

import torch
import numpy as np
import copy
import csv
import os
from datetime import datetime
from torch_geometric.datasets import Planetoid
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)

from models import GCN
from dp_mechanisms import (
    edge_dp_randomized_response,
    edge_dp_laplace_mechanism,
    split_epsilon,
    budget_split_description,
)
from utils import train_and_evaluate
from utils.results_saver import create_results_directory, save_metadata


# ============================================================================
# HELPERS (duplicats de main.py per mantenir main2.py autocontingut)
# Fonts originals: main.create_custom_split, main.apply_dp_to_graph
# ============================================================================

def create_custom_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Crea un split train/val/test reproduïble. [Duplicat de main.py, sense canvis.]

    Parameters
    ----------
    data : torch_geometric.data.Data
    train_ratio, val_ratio, test_ratio : float
    seed : int

    Returns
    -------
    data : torch_geometric.data.Data
        Modificat amb noves train_mask, val_mask, test_mask.
    """
    labeled_mask = data.train_mask | data.val_mask | data.test_mask
    labeled_indices = torch.where(labeled_mask)[0]
    rng = np.random.RandomState(seed)
    shuffled_indices = labeled_indices[rng.permutation(len(labeled_indices))]
    total_labeled = len(shuffled_indices)
    train_size = int(total_labeled * train_ratio)
    val_size = int(total_labeled * val_ratio)
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]
    new_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    new_val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    new_test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    new_train_mask[train_indices] = True
    new_val_mask[val_indices] = True
    new_test_mask[test_indices] = True
    data.train_mask = new_train_mask
    data.val_mask = new_val_mask
    data.test_mask = new_test_mask
    return data


def apply_dp_to_graph(data, dp_func, dp_params, seed=None):
    """
    Aplica el mecanisme DP a TOT el conjunt d'arestes del graf.
    [Duplicat de main.apply_dp_to_graph per mantenir aquest mòdul autocontingut.]

    El mecanisme pertorba la topologia completa (totes les entrades del
    triangle superior de l'adjacència); les característiques dels nodes, les
    etiquetes i les màscares train/val/test no es modifiquen mai.

    Parameters
    ----------
    data : torch_geometric.data.Data
    dp_func : callable
    dp_params : dict or tuple
    seed : int, optional

    Returns
    -------
    data_dp : torch_geometric.data.Data
    changes : dict
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    data_dp = copy.deepcopy(data)
    if isinstance(dp_params, dict):
        dp_edge_index, changes = dp_func(data, **dp_params)
    else:
        dp_edge_index, changes = dp_func(data, *dp_params)
    data_dp.edge_index = dp_edge_index
    return data_dp, changes


# ============================================================================
# FUNCIONS AUXILIARS OvR
# ============================================================================

def make_binary_data(data, class_idx):
    """
    Crea una còpia del dataset amb etiquetes binàries per a un classificador OvR.

    y_bin = 1 si y_original == class_idx, y_bin = 0 altrament.
    L'estructura del graf (x, edge_index) i les màscares es mantenen intactes.

    Parameters
    ----------
    data : torch_geometric.data.Data
    class_idx : int
        Índex de la classe positiva per a aquesta tasca OvR.

    Returns
    -------
    data_binary : torch_geometric.data.Data
        Còpia amb data_binary.y ∈ {0, 1}.
    """
    data_binary = copy.deepcopy(data)
    data_binary.y = (data.y == class_idx).long()
    return data_binary


@torch.no_grad()
def evaluate_binary(model, data, mask):
    """
    Calcula mètriques binàries addicionals que train_and_evaluate no reporta.

    train_and_evaluate usa average='weighted' per al F1, adequat per a
    multiclasse però no per a classificació binària desequilibrada. Aquesta
    funció complementa el resultat amb average='binary' i AUC-ROC.

    Parameters
    ----------
    model : GCN
        Model entrenat amb out_channels=2.
    data : torch_geometric.data.Data
        Dataset amb y ∈ {0, 1}.
    mask : torch.BoolTensor

    Returns
    -------
    metrics : dict
        f1_binary, precision, recall, auc_roc.
    """
    model.eval()
    logits = model(data.x, data.edge_index)
    masked_logits = logits[mask]
    masked_labels = data.y[mask]

    preds = masked_logits.argmax(dim=1).cpu().numpy()
    labels = masked_labels.cpu().numpy()

    # Probabilitat de la classe positiva (índex 1) via softmax per a AUC-ROC
    probs = torch.softmax(masked_logits, dim=1)[:, 1].cpu().numpy()

    f1_bin = f1_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    prec = precision_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, average='binary', pos_label=1, zero_division=0)

    # AUC-ROC pot fallar si el conjunt avaluat conté només una classe
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float('nan')

    return {
        'f1_binary': float(f1_bin),
        'precision': float(prec),
        'recall': float(rec),
        'auc_roc': float(auc),
    }


# ============================================================================
# LOOP EXPERIMENTAL OvR
# ============================================================================

def _run_single_run_ovr(data, num_classes, seed, num_epochs=500, patience=20):
    """
    Executa un sol run del pipeline OvR sobre un dataset (net o pertorbat per DP).

    Per a cada classe c ∈ {0, …, num_classes-1}:
      1. Genera data_binary (y ∈ {0,1})
      2. Entrena GCN(out_channels=2) via train_and_evaluate (early stopping)
      3. Computa mètriques binàries addicionals via evaluate_binary

    La seed es fixa al principi de cada run: tots els models OvR d'un mateix
    run parteixen del mateix estat aleatori inicial, que és reproduïble entre
    condicions (baseline, RR, Laplace) per al mateix run ID.

    Parameters
    ----------
    data : torch_geometric.data.Data
    num_classes : int
    seed : int
    num_epochs, patience : int

    Returns
    -------
    run_data : dict
        Claus: int c → dict amb mètriques val/test per a la classe c.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_data = {}
    for c in range(num_classes):
        print(f"    [OvR classe {c}/{num_classes - 1}]")
        data_binary = make_binary_data(data, c)
        model = GCN(data.num_features, 64, 2)

        # Reutilitzem train_and_evaluate sense modificar-lo.
        # Usa average='weighted' per al F1 intern, acceptable per al criteri
        # d'early stopping (val_acc). Les mètriques binàries finals venen
        # de evaluate_binary.
        results, _ = train_and_evaluate(
            model, data_binary,
            num_epochs=num_epochs,
            epochs_log=100,  # menys verbose: 7 classes per run → molt output
            patience=patience,
        )

        bin_val = evaluate_binary(model, data_binary, data_binary.val_mask)
        bin_test = evaluate_binary(model, data_binary, data_binary.test_mask)

        run_data[c] = {
            'val_accuracy':    results['val_accuracy'],
            'val_f1_weighted': results['val_f1'],
            'test_accuracy':   results['test_accuracy'],
            'test_f1_weighted': results['test_f1'],
            'val_f1_binary':   bin_val['f1_binary'],
            'val_precision':   bin_val['precision'],
            'val_recall':      bin_val['recall'],
            'val_auc_roc':     bin_val['auc_roc'],
            'test_f1_binary':  bin_test['f1_binary'],
            'test_precision':  bin_test['precision'],
            'test_recall':     bin_test['recall'],
            'test_auc_roc':    bin_test['auc_roc'],
        }

    return run_data


def run_ovr_experiments(
    data,
    epsilon_values,
    num_classes,
    num_runs=5,
    num_epochs=500,
    patience=20,
    seed_base=42,
):
    """
    Executa tots els experiments OvR: baseline + DP (RR + Laplace) per a cada epsilon.

    Decisió — on s'aplica la pertorbació DP respecte al bucle OvR:
    La pertorbació DP s'aplica UNA VEGADA per (run, epsilon, mètode), FORA del
    bucle OvR sobre les 7 classes. La privadesa DP és sobre l'estructura del
    graf (edge_index), que és INDEPENDENT de les etiquetes dels nodes. Per tant,
    totes les 7 tasques OvR d'un mateix run comparteixen el mateix graf pertorbat.
    Això és equivalent a: "el propietari del graf publica una versió DP del graf;
    les 7 tasques de classificació usen aquest graf públic". La transformació
    d'etiquetes (OvR) no afecta la garantia de privadesa sobre les arestes.

    Parameters
    ----------
    data : torch_geometric.data.Data
    epsilon_values : list of float
    num_classes : int
    num_runs, num_epochs, patience, seed_base : int

    Returns
    -------
    all_results : dict
        Estructura:
          all_results['baseline']           → list[num_runs] de {c: metrics}
          all_results['epsilon_X.XX']       → dict amb claus:
            'epsilon': float
            'rr_dp'     → list[num_runs] de {c: metrics}
            'laplace_dp'→ list[num_runs] de {c: metrics}
    """
    if isinstance(epsilon_values, (int, float)):
        epsilon_values = [epsilon_values]

    all_results = {}

    # ========== BASELINE (sense DP) ==========
    print("=" * 70)
    print("BASELINE OvR (sense DP) — 5 runs × 7 classes OvR")
    print("=" * 70 + "\n")

    baseline_results = []
    for run in range(1, num_runs + 1):
        print(f"RUN {run}/{num_runs}")
        print("-" * 70)
        run_data = _run_single_run_ovr(
            data, num_classes, seed=seed_base + run,
            num_epochs=num_epochs, patience=patience,
        )
        baseline_results.append(run_data)
        print(f"  → Run {run} completat ({num_classes} models OvR)\n")

    all_results['baseline'] = baseline_results

    # ========== DP PER A CADA EPSILON ==========
    for epsilon_idx, epsilon_dp in enumerate(epsilon_values, 1):
        print("\n" + "=" * 70)
        print(f"EPSILON = {epsilon_dp:.2f}  (lote {epsilon_idx}/{len(epsilon_values)})")
        print("=" * 70 + "\n")

        epsilon_results = {
            'epsilon': epsilon_dp,
            'rr_dp': [],
            'laplace_dp': [],
        }

        # ---- Randomized Response ----
        print("EDGE DP — Randomized Response (5 runs × 7 classes)")
        print("-" * 70)
        eps1, eps2 = split_epsilon(epsilon_dp)
        print(f"Budget ({budget_split_description()}): "
              f"eps_density={eps1:.4f}, eps_RR={eps2:.4f}\n")

        for run in range(1, num_runs + 1):
            print(f"RUN {run}/{num_runs}")
            # Pertorbació DP aplicada UNA VEGADA per a totes les classes OvR
            data_dp, changes = apply_dp_to_graph(
                data, edge_dp_randomized_response, (epsilon_dp,), seed=seed_base + run
            )
            print(f"  Graf: {changes['original_edges']} → {changes['new_edges']} arestes "
                  f"(+{changes['edges_added']} / -{changes['edges_removed']}), "
                  f"p1={changes['p1']:.4f}, p0={changes['p0']:.4f}")

            run_data = _run_single_run_ovr(
                data_dp, num_classes, seed=seed_base + run,
                num_epochs=num_epochs, patience=patience,
            )
            epsilon_results['rr_dp'].append(run_data)
            print(f"  → Run {run} RR completat\n")

        # ---- Laplace Mechanism ----
        print("EDGE DP — Laplace Mechanism (5 runs × 7 classes)")
        print("-" * 70)
        eps1, eps2 = split_epsilon(epsilon_dp)
        print(f"Budget ({budget_split_description()}): "
              f"eps_density={eps1:.4f}, eps_noise={eps2:.4f}\n")

        for run in range(1, num_runs + 1):
            print(f"RUN {run}/{num_runs}")
            data_dp, changes = apply_dp_to_graph(
                data, edge_dp_laplace_mechanism, (epsilon_dp,), seed=seed_base + run
            )
            print(f"  Graf: {changes['original_edges']} → {changes['new_edges']} arestes "
                  f"(+{changes['edges_added']} / -{changes['edges_removed']}), "
                  f"soroll_mitja={changes['avg_noise_magnitude']:.4f}")

            run_data = _run_single_run_ovr(
                data_dp, num_classes, seed=seed_base + run,
                num_epochs=num_epochs, patience=patience,
            )
            epsilon_results['laplace_dp'].append(run_data)
            print(f"  → Run {run} Laplace completat\n")

        all_results[f'epsilon_{epsilon_dp:.2f}'] = epsilon_results

    return all_results


# ============================================================================
# ESTADÍSTIQUES
# ============================================================================

def _compute_class_stats(run_list, class_idx, metric_keys):
    """
    Calcula mean i std per a cada mètrica d'una classe OvR, a través dels runs.

    Els valors NaN (p.ex. AUC-ROC quan una partició no conté ambdues classes)
    s'exclouen del càlcul en lloc de propagar-se.

    Parameters
    ----------
    run_list : list of dict
        Un element per run. Cada element: {class_idx: {metric: value}}.
    class_idx : int
    metric_keys : list of str

    Returns
    -------
    result : dict
        {metric_mean: float, metric_std: float} per a cada metric a metric_keys.
    """
    result = {}
    for key in metric_keys:
        values = [
            r[class_idx][key]
            for r in run_list
            if not (
                r[class_idx][key] is None
                or (isinstance(r[class_idx][key], float) and np.isnan(r[class_idx][key]))
            )
        ]
        result[f'{key}_mean'] = float(np.mean(values)) if values else float('nan')
        result[f'{key}_std'] = float(np.std(values)) if values else float('nan')
    return result


def compute_ovr_statistics(all_results, num_classes, class_weights_test=None):
    """
    Calcula estadístiques (mean ± std) per classe i agregades (macro i micro).

    Agregat macro: mitjana no ponderada de les 7 classes OvR.
    Agregat micro: ponderada per la freqüència de positius de cada classe al
    conjunt de test (class_weights_test). Reflecteix el rendiment global
    quan les classes tenen freqüències desiguals.

    Parameters
    ----------
    all_results : dict
    num_classes : int
    class_weights_test : list of int or None
        Nombre de nodes positius per classe al test set. Si None, macro = micro.

    Returns
    -------
    stats : dict
        Estructura: stats[condition][class_idx] = {metric_mean, metric_std}
                    stats[condition]['aggregate'] = {macro_*, micro_*}
        On condition ∈ {'baseline', 'epsilon_X.XX'} i, per als epsilons,
        stats[condition]['rr_dp'] / stats[condition]['laplace_dp'] segueixen
        la mateixa estructura per classe + aggregate.
    """
    METRICS = [
        'test_accuracy',    'test_f1_binary',  'test_precision',
        'test_recall',      'test_auc_roc',
        'val_accuracy',     'val_f1_binary',   'val_precision',
        'val_recall',       'val_auc_roc',
    ]

    if class_weights_test is None:
        class_weights_test = [1.0] * num_classes

    def _aggregate(per_class):
        """Macro i micro a partir dels mean per classe."""
        agg = {}
        for key in METRICS:
            mean_key = f'{key}_mean'
            class_means = []
            class_wmeans = []
            total_w = 0.0
            for c in range(num_classes):
                v = per_class.get(c, {}).get(mean_key, float('nan'))
                if not (v is None or np.isnan(v)):
                    class_means.append(v)
                    class_wmeans.append(v * class_weights_test[c])
                    total_w += class_weights_test[c]
            agg[f'macro_{mean_key}'] = float(np.mean(class_means)) if class_means else float('nan')
            agg[f'micro_{mean_key}'] = float(sum(class_wmeans) / total_w) if total_w > 0 else float('nan')
        return agg

    stats = {}

    # ---- Baseline ----
    if 'baseline' in all_results:
        per_class = {c: _compute_class_stats(all_results['baseline'], c, METRICS)
                     for c in range(num_classes)}
        stats['baseline'] = per_class
        stats['baseline']['aggregate'] = _aggregate(per_class)

    # ---- Per epsilon ----
    for key, value in all_results.items():
        if key == 'baseline':
            continue
        if isinstance(value, dict) and 'epsilon' in value:
            stats[key] = {'epsilon': value['epsilon']}
            for method in ['rr_dp', 'laplace_dp']:
                if method in value:
                    per_class = {c: _compute_class_stats(value[method], c, METRICS)
                                 for c in range(num_classes)}
                    stats[key][method] = per_class
                    stats[key][method]['aggregate'] = _aggregate(per_class)

    return stats


# ============================================================================
# REPORTING
# ============================================================================

# Noms canònics de les 7 classes de Cora (índex 0–6)
_CORA_CLASS_NAMES = [
    'Case_Based', 'Genetic_Alg', 'Neural_Nets',
    'Probabilistic', 'Reinforce', 'Rule_Learn', 'Theory',
]


def print_ovr_results_summary(stats, num_classes, class_weights_test=None):
    """
    Imprimeix un resum jeràrquic dels resultats OvR.

    Per cada condició (baseline / mètode × epsilon) mostra:
      - Per classe: mean ± std de F1 binari i AUC-ROC (mètriques principals)
      - Agregats macro i micro

    Parameters
    ----------
    stats : dict
    num_classes : int
    class_weights_test : list of int or None
    """
    def _print_condition(label, per_class_stats):
        print(f"\n{label}")
        hdr = f"{'Classe':<16}|{'F1-bin test':^22}|{'AUC-ROC test':^22}|{'Precision':^12}|{'Recall':^12}"
        print(hdr)
        print("-" * len(hdr))
        for c in range(num_classes):
            cname = _CORA_CLASS_NAMES[c] if c < len(_CORA_CLASS_NAMES) else f'Classe_{c}'
            s = per_class_stats.get(c, {})
            f1_m = s.get('test_f1_binary_mean', float('nan'))
            f1_s = s.get('test_f1_binary_std', float('nan'))
            auc_m = s.get('test_auc_roc_mean', float('nan'))
            auc_s = s.get('test_auc_roc_std', float('nan'))
            prec_m = s.get('test_precision_mean', float('nan'))
            rec_m = s.get('test_recall_mean', float('nan'))
            n_str = f"(n={class_weights_test[c]})" if class_weights_test else ""
            print(f"  {cname:<14}{n_str:<4}| {f1_m:.4f} ± {f1_s:.4f}    | "
                  f"{auc_m:.4f} ± {auc_s:.4f}    | {prec_m:.4f}    | {rec_m:.4f}")
        agg = per_class_stats.get('aggregate', {})
        print("-" * len(hdr))
        macro_f1  = agg.get('macro_test_f1_binary_mean', float('nan'))
        micro_f1  = agg.get('micro_test_f1_binary_mean', float('nan'))
        macro_auc = agg.get('macro_test_auc_roc_mean', float('nan'))
        micro_auc = agg.get('micro_test_auc_roc_mean', float('nan'))
        print(f"  {'MACRO':<18}| {macro_f1:.4f}             | {macro_auc:.4f}")
        print(f"  {'MICRO/weighted':<18}| {micro_f1:.4f}             | {micro_auc:.4f}")

    print("\n" + "=" * 80)
    print("RESUM DE RESULTATS — CLASSIFICACIÓ BINÀRIA OvR")
    print("=" * 80)

    if 'baseline' in stats:
        _print_condition("BASELINE (sense DP)", stats['baseline'])

    for key, value in sorted(stats.items()):
        if key == 'baseline':
            continue
        eps = value.get('epsilon', '?')
        for method in ['rr_dp', 'laplace_dp']:
            if method in value:
                _print_condition(f"ε={eps:.2f} | {method.upper()}", value[method])

    # Taula resum final: F1 macro per epsilon
    print("\n" + "=" * 80)
    print("TAULA RESUM — F1 MACRO BINARI (test) per EPSILON")
    print("=" * 80)
    print(f"{'Epsilon':<12} | {'RR F1 macro':<22} | {'Laplace F1 macro':<22}")
    print("-" * 60)

    if 'baseline' in stats:
        bl = stats['baseline'].get('aggregate', {})
        bl_f1 = bl.get('macro_test_f1_binary_mean', float('nan'))
        print(f"{'Baseline':<12} | {bl_f1:.4f}               | (referència)")

    for key, value in sorted(stats.items()):
        if key == 'baseline':
            continue
        eps = value.get('epsilon', '?')
        rr_f1 = value.get('rr_dp', {}).get('aggregate', {}).get('macro_test_f1_binary_mean', float('nan'))
        lap_f1 = value.get('laplace_dp', {}).get('aggregate', {}).get('macro_test_f1_binary_mean', float('nan'))
        print(f"{eps:<12.2f} | {rr_f1:.4f}               | {lap_f1:.4f}")


# ============================================================================
# GUARDANT RESULTATS
# ============================================================================

def _fmt(v):
    """Formata un float per a CSV/TXT; gestiona NaN i None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'NaN'
    return f'{v:.6f}'


def _write_condition_block(f, label, per_class_stats, num_classes):
    """Escriu un bloc de resultats per a una condició al fitxer summary.txt."""
    cora_names = [
        'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
        'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory',
    ]
    f.write(f"\n{label}\n")
    f.write(f"{'Classe':<24}| {'F1-bin test':^22}| {'AUC-ROC test':^22}| {'Precision':^11}| {'Recall':^11}\n")
    f.write("-" * 95 + "\n")
    for c in range(num_classes):
        cname = cora_names[c] if c < len(cora_names) else f'Classe_{c}'
        s = per_class_stats.get(c, {})
        f1_m = s.get('test_f1_binary_mean', float('nan'))
        f1_s = s.get('test_f1_binary_std', float('nan'))
        auc_m = s.get('test_auc_roc_mean', float('nan'))
        auc_s = s.get('test_auc_roc_std', float('nan'))
        prec_m = s.get('test_precision_mean', float('nan'))
        rec_m = s.get('test_recall_mean', float('nan'))
        f.write(f"  {cname:<22}| {f1_m:.4f} ± {f1_s:.4f}    | "
                f"{auc_m:.4f} ± {auc_s:.4f}    | {prec_m:.4f}    | {rec_m:.4f}\n")
    agg = per_class_stats.get('aggregate', {})
    f.write("-" * 95 + "\n")
    f.write(f"  {'MACRO avg':<22}| {agg.get('macro_test_f1_binary_mean', float('nan')):.4f}              | "
            f"{agg.get('macro_test_auc_roc_mean', float('nan')):.4f}\n")
    f.write(f"  {'MICRO/weighted avg':<22}| {agg.get('micro_test_f1_binary_mean', float('nan')):.4f}              | "
            f"{agg.get('micro_test_auc_roc_mean', float('nan')):.4f}\n\n")


def save_ovr_summary(results_dir, stats, dataset_info, epsilon_values, num_classes):
    """
    Guarda un resum llegible de tots els resultats OvR.

    Parameters
    ----------
    results_dir : str
    stats : dict
    dataset_info : dict
    epsilon_values : list of float
    num_classes : int
    """
    summary_file = os.path.join(results_dir, "summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DP-GNN OvR BINARY CLASSIFICATION — RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset:    {dataset_info['name']}\n")
        f.write(f"Task:       Node Label Prediction — One-vs-Rest Binary ({num_classes} tasques)\n")
        f.write(f"Runs:       5 per (classe, mètode, epsilon)\n")
        f.write(f"Epsilons:   {epsilon_values}\n")
        f.write(f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Mètrica principal: F1 binari (average='binary', pos_label=1)\n")
        f.write("Raó: Cora sota OvR és desequilibrat (~10-30% positius); accuracy és trivial.\n\n")

        if 'baseline' in stats:
            _write_condition_block(f, "BASELINE (sense DP)", stats['baseline'], num_classes)

        for key, value in sorted(stats.items()):
            if key == 'baseline':
                continue
            eps = value.get('epsilon', '?')
            for method in ['rr_dp', 'laplace_dp']:
                if method in value:
                    _write_condition_block(f, f"ε={eps:.2f} | {method.upper()}", value[method], num_classes)

    print(f"  ✓ summary.txt guardat")


def save_ovr_detailed_csv(results_dir, all_results, num_classes):
    """
    Guarda resultats detallats (per run, per classe) en CSV.

    Estructura de fitxers generada:
      detailed_results/
        baseline/
          class_0.csv ... class_6.csv
        epsilon_X.XX/
          rr_dp/
            class_0.csv ... class_6.csv
          laplace_dp/
            class_0.csv ... class_6.csv

    Cada CSV té una fila per run i columnes per a totes les mètriques.

    Parameters
    ----------
    results_dir : str
    all_results : dict
    num_classes : int
    """
    METRICS = [
        'test_accuracy', 'test_f1_binary', 'test_precision', 'test_recall', 'test_auc_roc',
        'val_accuracy',  'val_f1_binary',  'val_precision',  'val_recall',  'val_auc_roc',
    ]

    detailed_dir = os.path.join(results_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)

    def _write_class_csv(directory, run_list, class_idx):
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"class_{class_idx}.csv")
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Run'] + METRICS)
            for i, run_data in enumerate(run_list, 1):
                row = [i] + [_fmt(run_data[class_idx].get(m)) for m in METRICS]
                writer.writerow(row)

    if 'baseline' in all_results:
        bl_dir = os.path.join(detailed_dir, 'baseline')
        for c in range(num_classes):
            _write_class_csv(bl_dir, all_results['baseline'], c)

    for key, value in all_results.items():
        if key == 'baseline':
            continue
        if isinstance(value, dict) and 'epsilon' in value:
            eps = value['epsilon']
            for method in ['rr_dp', 'laplace_dp']:
                if method in value:
                    method_dir = os.path.join(detailed_dir, f'epsilon_{eps:.2f}', method)
                    for c in range(num_classes):
                        _write_class_csv(method_dir, value[method], c)

    print(f"  ✓ detailed_results/ guardat")


def save_ovr_results(results_dir, all_results, stats, dataset_info, epsilon_values):
    """
    Funció mestra: guarda tots els resultats OvR al directori indicat.

    Fitxers generats:
      summary.txt        — resum llegible per a humans (per classe + macro/micro)
      metadata.txt       — configuració de l'experiment (via results_saver)
      detailed_results/  — CSV per classe, per mètode, per epsilon

    Decisió sobre results_saver.save_all_results:
    save_all_results espera una estructura plana (una mètrica per run), però
    aquí cada run conté 7 sub-resultats (un per classe OvR). En lloc d'encabir
    les dades en un format incompatible, s'usen funcions noves en aquest mòdul
    (save_ovr_summary, save_ovr_detailed_csv) i es reutilitza save_metadata
    de results_saver per a la informació bàsica de l'experiment.

    Parameters
    ----------
    results_dir : str
    all_results : dict
    stats : dict
    dataset_info : dict
        Claus: name, num_nodes, num_edges, num_features, num_classes.
    epsilon_values : list of float
    """
    num_classes = dataset_info.get('num_classes', 7)
    save_ovr_summary(results_dir, stats, dataset_info, epsilon_values, num_classes)
    save_ovr_detailed_csv(results_dir, all_results, num_classes)
    save_metadata(
        results_dir,
        dataset_name=f"{dataset_info['name']} (OvR Binary, {num_classes} tasques)",
        num_nodes=dataset_info['num_nodes'],
        num_edges=dataset_info['num_edges'],
        num_features=dataset_info['num_features'],
        num_classes=num_classes,
    )
    print(f"\n✓ Resultats OvR guardats a: {results_dir}\n")
    print(f"  Fitxers creats:")
    print(f"    summary.txt          (resum per classe + macro/micro)")
    print(f"    metadata.txt         (configuració de l'experiment)")
    print(f"    detailed_results/    (CSV per classe, mètode i epsilon)")


# ============================================================================
# EXECUCIÓ PRINCIPAL
# ============================================================================

def main():
    """
    Executa el pipeline complet OvR: càrrega, split, experiments, estadístiques.

    Returns
    -------
    all_results : dict
    stats : dict
    data : torch_geometric.data.Data
    num_classes : int
    class_sizes_test : list of int
    epsilon_values : list of float
    """
    # ---- Configuració experimental ----
    GLOBAL_SEED = 42
    NUM_RUNS = 5
    NUM_EPOCHS = 500
    PATIENCE = 20

    # Llista completa d'epsilons de la especificació: list(range(1, 21))
    # Valors reduïts per a proves ràpides (canvia per list(range(1, 21)) per a l'experiment complet):
    EPSILON_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # ---- Reproducibilitat global ----
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 70)
    print("DIFFERENTIAL PRIVACY ON GNN — CLASSIFICACIÓ BINÀRIA OvR")
    print("Tasca: Node Label Prediction (One-vs-Rest, Cora 7 classes)")
    print("=" * 70 + "\n")
    print(f"Global Seed:    {GLOBAL_SEED}")
    print(f"Epsilons:       {EPSILON_VALUES}")
    print(f"Runs per cond.: {NUM_RUNS}\n")

    # ---- 1. Carregar dataset ----
    print("1. Carregant dataset (Cora)...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    num_classes = int(data.y.max().item() + 1)

    print(f"   Nodes: {data.num_nodes}, Arestes: {data.num_edges}, "
          f"Features: {data.num_features}, Classes: {num_classes}\n")

    # ---- 2. Split 60/20/20 ----
    print("2. Creant split 60/20/20 (seed=42)...")
    data = create_custom_split(
        data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=GLOBAL_SEED
    )
    print(f"   Train: {data.train_mask.sum().item()}, "
          f"Val: {data.val_mask.sum().item()}, "
          f"Test: {data.test_mask.sum().item()}\n")

    # ---- 3. Distribució OvR (verificació) ----
    print("3. Distribució OvR — positius/negatius per classe al test set:")
    test_labels = data.y[data.test_mask]
    class_sizes_test = []
    for c in range(num_classes):
        n_pos = int((test_labels == c).sum().item())
        n_neg = int((test_labels != c).sum().item())
        total = n_pos + n_neg
        print(f"   Classe {c} ({_CORA_CLASS_NAMES[c] if c < len(_CORA_CLASS_NAMES) else '?'}): "
              f"{n_pos} positius / {n_neg} negatius  "
              f"(rati positiu = {n_pos / total * 100:.1f}%)")
        class_sizes_test.append(n_pos)
    print()

    # ---- 4. Experiments ----
    print("4. Executant experiments OvR...\n")
    all_results = run_ovr_experiments(
        data,
        EPSILON_VALUES,
        num_classes,
        num_runs=NUM_RUNS,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        seed_base=GLOBAL_SEED,
    )

    # ---- 5. Estadístiques ----
    stats = compute_ovr_statistics(
        all_results, num_classes, class_weights_test=class_sizes_test
    )

    # ---- 6. Resum ----
    print_ovr_results_summary(stats, num_classes, class_weights_test=class_sizes_test)

    return all_results, stats, data, num_classes, class_sizes_test, EPSILON_VALUES


if __name__ == "__main__":
    all_results, stats, data, num_classes, class_sizes_test, epsilon_values = main()

    # ---- 7. Guardar resultats ----
    print("\n" + "=" * 70)
    print("GUARDANT RESULTATS")
    print("=" * 70)

    results_dir = create_results_directory(
        dataset_name="Cora_binary", base_path="results"
    )

    dataset_info = {
        'name': 'Cora',
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.num_features,
        'num_classes': num_classes,
    }

    save_ovr_results(results_dir, all_results, stats, dataset_info, epsilon_values)
    print("✓ Anàlisi OvR completada!\n")
