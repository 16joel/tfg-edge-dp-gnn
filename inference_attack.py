"""
Edge Inference Attack against Edge Differential Privacy Mechanisms
(UNSUPERVISED / score-based formulation).

Avalua empíricament el grau de protecció que ofereixen els mecanismes
Edge-DP (Randomized Response i Laplace) davant d'un adversari que intenta
inferir les arestes originals del graf a partir ÚNICAMENT del graf publicat.

==========================================================================
MODEL D'ADVERSARI: Caixa negra SENSE coneixement d'arestes reals
==========================================================================
L'adversari observa ÚNICAMENT el graf publicat G' = DP(G):
  - L'estructura d'arestes de G' (edge_index pertorbat)
  - Les característiques de nodes (x), que es consideren públiques
  - Els embeddings d'un GCN entrenat sobre G'

L'adversari NO coneix, i mai accedeix a:
  - El graf original G
  - CAP aresta real (ni tan sols un subconjunt: no hi ha "shadow edges")
  - Els paràmetres interns del mecanisme DP

Conseqüència de disseny (clau): l'adversari NO entrena cap classificador
supervisat, perquè entrenar-lo exigiria conèixer la veritat-terra de quines
parelles són arestes reals — exactament la informació que el model d'amenaça
li nega. En lloc d'això, l'adversari calcula una PUNTUACIÓ DE VERSEMBLANÇA
D'ARESTA (link-prediction score) per a cada parella de nodes a partir només
de G', i ordena totes les parelles per aquesta puntuació. Una puntuació alta
significa "és probable que (u, v) fos una aresta del graf real".

Cada puntuació és, doncs, un ATAC INDEPENDENT (opció A: scores individuals,
sense agregació ni classificador). Es reporten:

  Heurístiques estructurals (calculades sobre G'):
    - Common Neighbors (CN)
    - Jaccard Coefficient
    - Adamic-Adar Index (AA)
    - Resource Allocation (RA)
    - Preferential Attachment (PA)
  Senyal d'embeddings (calculat sobre G'):
    - Cosine similarity entre h_u i h_v (similitud dels embeddings del GCN)

==========================================================================
PAPER DE L'AVALUADOR vs PAPER DE L'ADVERSARI
==========================================================================
És important distingir QUI sap QUÈ:
  - L'ADVERSARI (simulat): només toca G'. Produeix puntuacions.
  - L'AVALUADOR (nosaltres, com a investigadors): coneix G i l'utilitza
    NOMÉS per mesurar l'èxit de l'adversari (calcular AUC-ROC i Average
    Precision de cada puntuació contra la veritat-terra). L'adversari mai
    veu aquestes etiquetes.

Aquesta separació és el que fa el model d'amenaça consistent: el graf
original apareix exclusivament a la infraestructura de mesura, no a l'atac.

==========================================================================
MÈTRIQUES
==========================================================================
  - AUC-ROC: probabilitat que una aresta real rebi una puntuació superior
    a la d'una no-aresta. 0.5 = atzar (protecció màxima), 1.0 = filtració
    total. És invariant a la prevalença -> mètrica principal.
  - Average Precision (AP): àrea sota la corba precisió-record. S'ha
    d'interpretar SEMPRE contra la prevalença (proporció de positius); un
    classificador aleatori obté AP ≈ prevalença.

==========================================================================
REUTILITZACIÓ DE CODI (sense duplicació)
==========================================================================
Aquest mòdul NO reimplementa res que ja existeixi al projecte:
  - GCN i GCN.get_embeddings          <- models.gcn
  - Entrenament + early stopping       <- utils.metrics.train_and_evaluate
  - Pertorbació del graf sencer        <- main.apply_dp_to_graph
  - Divisió 60/20/20                    <- main.create_custom_split
  - Repartiment del pressupost ε        <- dp_mechanisms.config (via mecanismes)
  - Persistència de resultats           <- utils.results_saver

==========================================================================
REFERÈNCIA TEÒRICA
==========================================================================
- Liben-Nowell, D. & Kleinberg, J. (2007). "The link-prediction problem
  for social networks." JASIST.
- Fan Wu et al. (2022). "LinkTeller: Recovering Private Edges from Graph
  Neural Networks via Fine-Grained Explanations." IEEE S&P 2022.
- He, X. et al. (2021). "Node-Level Membership Inference Attacks Against
  Graph Neural Networks." arXiv:2102.05429
==========================================================================
"""

import argparse
import warnings
from collections import deque
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph
from sklearn.metrics import roc_auc_score, average_precision_score

# ----------------------------------------------------------------------------
# Reutilització de codi del projecte (cap funció es redefineix aquí)
# ----------------------------------------------------------------------------
from models.gcn import GCN
from utils.metrics import train_and_evaluate
from utils.results_saver import (
    create_results_directory,
    save_inference_attack_results,
)
from dp_mechanisms.edge_dp_rr import edge_dp_randomized_response
from dp_mechanisms.edge_dp_laplace import edge_dp_laplace_mechanism
# create_custom_split i apply_dp_to_graph viuen a main.py: es reutilitzen
# directament en lloc de duplicar-les. SUPPORTED_DATASETS també es reutilitza
# per garantir exactament el mateix conjunt de datasets admesos que main.py.
from main import create_custom_split, apply_dp_to_graph, SUPPORTED_DATASETS

warnings.filterwarnings('ignore')


# Noms dels scores (cada un és un atac independent). L'ordre fixa també
# l'ordre de columnes a tots els CSV i taules.
SCORE_NAMES = [
    'common_neighbors',
    'jaccard',
    'adamic_adar',
    'resource_allocation',
    'preferential_attachment',
    'embedding_cosine',
]


# Mida fixa del subconjunt de PubMed (bola BFS). PubMed té ~19.717 nodes;
# calcular les heurístiques estructurals i l'AUC sobre tot l'univers de
# parelles 1:100 seria computacionalment inviable, de manera que se'n pren
# una bola BFS connexa de ~PUBMED_SUBSET_NODES nodes. Cora i CiteSeer s'usen
# sencers.
PUBMED_SUBSET_NODES = 3000


# ============================================================================
# SUBMOSTREIG PER A PUBMED (bola BFS connexa)
# ============================================================================

def subsample_bfs_ball(data, max_nodes: int = PUBMED_SUBSET_NODES,
                       seed: int = 42, verbose: bool = True):
    """
    Extreu un subgraf connex de `data` mitjançant una exploració en amplada
    (BFS) a partir d'un node llavor aleatori, fins a reunir `max_nodes` nodes.

    Motivació
    ---------
    PubMed (~19.717 nodes) fa intractable l'atac complet: l'univers de
    parelles a ràtio 1:100 i el càlcul de les heurístiques estructurals
    escalen amb el nombre de nodes. Una bola BFS conserva l'estructura
    LOCAL del graf (veïnatges, triangles, comunitats) —exactament el que
    exploten Common Neighbors, Jaccard, Adamic-Adar, etc.— a diferència
    d'un mostreig de nodes uniforme, que trencaria la majoria d'arestes i
    deixaria un graf gairebé buit.

    El submostreig s'aplica UNA SOLA vegada sobre el graf original, ABANS
    de qualsevol pertorbació DP. Així, el graf publicat G' i la veritat-terra
    G comparteixen exactament el mateix conjunt de nodes (condició necessària
    perquè l'AUC-ROC tingui sentit).

    Reutilitza torch_geometric.utils.subgraph amb `relabel_nodes=True`, de
    manera que el subgraf resultant té índexs de node contigus [0, k) i és
    un objecte Data plenament compatible amb la resta del pipeline (split,
    DP, atac), sense cap canvi addicional.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graf original sencer (abans del split i de la DP).
    max_nodes : int
        Nombre objectiu de nodes del subgraf.
    seed : int
        Llavor per a l'elecció del node llavor (reproductibilitat).
    verbose : bool

    Returns
    -------
    data_sub : torch_geometric.data.Data
        Subgraf amb nodes reindexats a [0, k), on k <= max_nodes.
    """
    num_nodes = int(data.num_nodes)
    if num_nodes <= max_nodes:
        if verbose:
            print(f"   [SUBSET] {num_nodes} <= {max_nodes}: s'usa el graf sencer.")
        return data

    rng = np.random.RandomState(seed)

    # Llistes d'adjacència no dirigides del graf original.
    ei = data.edge_index.numpy()
    adjacency = [[] for _ in range(num_nodes)]
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u != v:
            adjacency[u].append(v)

    # BFS des d'un node llavor aleatori fins a reunir max_nodes nodes.
    seed_node = int(rng.randint(0, num_nodes))
    visited = {seed_node}
    order = [seed_node]
    queue = deque([seed_node])
    while queue and len(order) < max_nodes:
        node = queue.popleft()
        for nb in adjacency[node]:
            if nb not in visited:
                visited.add(nb)
                order.append(nb)
                queue.append(nb)
                if len(order) >= max_nodes:
                    break

    subset = torch.tensor(sorted(order), dtype=torch.long)

    # subgraph() filtra arestes i reindexa els nodes a [0, k) (relabel_nodes).
    new_edge_index, _ = subgraph(
        subset, data.edge_index, relabel_nodes=True, num_nodes=num_nodes
    )

    data_sub = data.__class__()
    data_sub.x = data.x[subset]
    data_sub.y = data.y[subset]
    data_sub.edge_index = new_edge_index
    data_sub.num_nodes = subset.numel()

    # Propaga les màscares originals de Planetoid (train/val/test_mask)
    # restringides als nodes seleccionats. create_custom_split() les
    # necessita per identificar els nodes etiquetats abans de construir
    # el split 60/20/20. Sense aquesta propagació, data_sub no tindria
    # train_mask i create_custom_split fallaria.
    for mask_name in ('train_mask', 'val_mask', 'test_mask'):
        if hasattr(data, mask_name) and getattr(data, mask_name) is not None:
            setattr(data_sub, mask_name, getattr(data, mask_name)[subset])

    if verbose:
        print(f"   [SUBSET] Bola BFS: {data_sub.num_nodes} nodes, "
              f"{data_sub.num_edges} arestes (llavor node {seed_node}).")

    return data_sub


# ============================================================================
# ENTRENAMENT DEL GCN (wrapper prim sobre utils.metrics.train_and_evaluate)
# ============================================================================

def train_gcn(model: GCN, data, num_epochs: int = 500, patience: int = 20) -> GCN:
    """
    Entrena el GCN sobre `data` reutilitzant train_and_evaluate() del
    projecte (mateixa lògica d'early stopping que els experiments d'utilitat).
    Retorna el model entrenat.
    """
    train_and_evaluate(
        model, data,
        num_epochs=num_epochs,
        epochs_log=99999,
        patience=patience,
    )
    return model


# ============================================================================
# CONSTRUCCIÓ DEL CONJUNT D'AVALUACIÓ (univers de parelles + veritat-terra)
# ============================================================================

def build_edge_pairs(
    data_original,
    neg_ratio: float = 100.0,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construeix el conjunt de parelles (u, v) sobre el qual es mesura l'atac,
    juntament amb la seva veritat-terra (1 = aresta real, 0 = no-aresta).

    Aquesta veritat-terra pertany a l'AVALUADOR, no a l'adversari: només
    s'utilitza per calcular AUC-ROC i AP a partir de les puntuacions que
    l'adversari deriva de G'.

    Negatius mostrejats SENSE REEMPLAÇAMENT i deduplicats: cada parella
    negativa apareix com a molt una vegada. (La versió anterior mostrejava
    amb reemplaçament i sense deduplicar, cosa que generava desenes de
    milers de parelles repetides al conjunt d'avaluació i distorsionava
    AUC i AP.)

    Parameters
    ----------
    data_original : torch_geometric.data.Data
        Graf original (veritat-terra, només per a mesura).
    neg_ratio : float
        Nombre de negatius per positiu (p. ex. 100 -> 1:100).
    seed : int
    verbose : bool

    Returns
    -------
    pairs : np.ndarray, shape [n_pos + n_neg, 2]
        Univers de parelles avaluades (positives seguides de negatives).
    labels : np.ndarray, shape [n_pos + n_neg]
        Veritat-terra (1 per a arestes reals, 0 per a no-arestes).
    """
    rng = np.random.RandomState(seed)
    num_nodes = int(data_original.num_nodes)

    # Arestes originals (triangle superior, sense duplicats dirigits).
    ei = data_original.edge_index.numpy()
    mask = ei[0] < ei[1]
    pos_pairs = np.unique(np.stack([ei[0][mask], ei[1][mask]], axis=1), axis=0)
    n_pos = len(pos_pairs)

    edge_set = {(int(u), int(v)) for u, v in pos_pairs}

    n_neg_target = int(n_pos * neg_ratio)
    max_possible_neg = num_nodes * (num_nodes - 1) // 2 - n_pos
    n_neg_target = min(n_neg_target, max_possible_neg)

    # Mostreig SENSE reemplaçament: es manté un conjunt de parelles ja
    # escollides i es descarta qualsevol repetició o aresta real.
    neg_set = set()
    attempts = 0
    max_attempts = n_neg_target * 50 + 1000

    while len(neg_set) < n_neg_target and attempts < max_attempts:
        u = int(rng.randint(0, num_nodes))
        v = int(rng.randint(0, num_nodes))
        attempts += 1
        if u == v:
            continue
        if u > v:
            u, v = v, u
        pair = (u, v)
        if pair in edge_set or pair in neg_set:
            continue
        neg_set.add(pair)

    neg_pairs = np.array(sorted(neg_set), dtype=np.int64) if neg_set \
        else np.empty((0, 2), dtype=np.int64)
    n_neg = len(neg_pairs)

    if verbose:
        prevalence = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
        print(f"  [PAIRS] Positius: {n_pos}, Negatius: {n_neg} "
              f"(prevalença {prevalence:.4f})", flush=True)
        if n_neg < int(n_neg_target * 0.95):
            print(f"    AVÍS: només s'han generat {n_neg}/{n_neg_target} negatius",
                  flush=True)

    pairs = np.vstack([pos_pairs, neg_pairs]).astype(np.int64)
    labels = np.concatenate([
        np.ones(n_pos, dtype=np.int32),
        np.zeros(n_neg, dtype=np.int32),
    ])

    return pairs, labels


# ============================================================================
# PUNTUACIONS DE VERSEMBLANÇA D'ARESTA (calculades NOMÉS sobre G')
# ============================================================================

def compute_link_scores(
    pairs: np.ndarray,
    data_perturbed,
    embeddings: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Calcula, per a cada parella (u, v), totes les puntuacions de
    versemblança d'aresta usant EXCLUSIVAMENT el graf publicat G'
    (estructura + embeddings del GCN entrenat sobre G').

    Cada puntuació retornada és un vector [n] que, ordenat de major a
    menor, defineix un atac independent.

    Heurístiques estructurals (Liben-Nowell & Kleinberg, 2007):
      - Common Neighbors:        |N(u) ∩ N(v)|
      - Jaccard:                 |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
      - Adamic-Adar:             Σ_{w ∈ N(u)∩N(v)} 1 / log(deg(w))
      - Resource Allocation:     Σ_{w ∈ N(u)∩N(v)} 1 / deg(w)
      - Preferential Attachment: deg(u) · deg(v)
    Senyal d'embeddings:
      - Cosine similarity:       <h_u, h_v> / (||h_u|| · ||h_v||)

    Parameters
    ----------
    pairs : np.ndarray, shape [n, 2]
    data_perturbed : torch_geometric.data.Data
        Graf publicat G' (l'únic que l'adversari pot observar).
    embeddings : np.ndarray, shape [num_nodes, hidden_dim]
        Embeddings del GCN entrenat sobre G'.

    Returns
    -------
    scores : dict[str, np.ndarray]
        Una entrada per nom de SCORE_NAMES, cadascuna shape [n].
    """
    num_nodes = int(data_perturbed.num_nodes)

    # Llistes d'adjacència i graus de G' (mateixa construcció que la versió
    # estructural prèvia; aquí és l'única topologia que veu l'adversari).
    ei = data_perturbed.edge_index.numpy()
    adjacency = [set() for _ in range(num_nodes)]
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u != v:
            adjacency[u].add(v)
            adjacency[v].add(u)
    degrees = np.array([len(adjacency[i]) for i in range(num_nodes)],
                       dtype=np.float64)

    n = len(pairs)
    cn = np.zeros(n, dtype=np.float64)
    jac = np.zeros(n, dtype=np.float64)
    aa = np.zeros(n, dtype=np.float64)
    ra = np.zeros(n, dtype=np.float64)
    pa = np.zeros(n, dtype=np.float64)

    for idx in range(n):
        u, v = int(pairs[idx, 0]), int(pairs[idx, 1])
        nu, nv = adjacency[u], adjacency[v]
        common = nu & nv
        union = nu | nv

        c = len(common)
        cn[idx] = c
        jac[idx] = c / len(union) if union else 0.0
        aa[idx] = sum(1.0 / np.log(degrees[w]) for w in common if degrees[w] > 1)
        ra[idx] = sum(1.0 / degrees[w] for w in common if degrees[w] > 0)
        pa[idx] = degrees[u] * degrees[v]

    # Cosine similarity dels embeddings (vectoritzat).
    hu = embeddings[pairs[:, 0]]
    hv = embeddings[pairs[:, 1]]
    dot = np.sum(hu * hv, axis=1)
    norm = np.linalg.norm(hu, axis=1) * np.linalg.norm(hv, axis=1)
    cosine = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)

    return {
        'common_neighbors': cn,
        'jaccard': jac,
        'adamic_adar': aa,
        'resource_allocation': ra,
        'preferential_attachment': pa,
        'embedding_cosine': cosine,
    }


# ============================================================================
# PIPELINE D'ATAC PER A UNA CONFIGURACIÓ
# ============================================================================

def run_single_attack(
    data_original,
    data_perturbed,
    mechanism_name: str,
    epsilon: Optional[float],
    seed: int = 42,
    gcn_epochs: int = 500,
    gcn_patience: int = 20,
    neg_ratio: float = 100.0,
    verbose: bool = True,
) -> Dict:
    """
    Executa l'atac no supervisat per a una configuració (Baseline, RR(ε)
    o Laplace(ε)) i retorna, per a cada puntuació, l'AUC-ROC i l'Average
    Precision contra la veritat-terra del graf original.

    NOTA METODOLÒGICA — asimetria del baseline:
    Quan mechanism_name == 'Baseline', data_perturbed és el graf original
    sense soroll, de manera que les puntuacions s'avaluen sobre G = G'.
    Això maximitza la filtració i representa el pitjor cas per a la
    privadesa (cap protecció); és la referència contra la qual es mesura
    la reducció d'AUC dels mecanismes DP.

    Parameters
    ----------
    data_original : torch_geometric.data.Data
        Graf original (NOMÉS per a la mesura: veritat-terra).
    data_perturbed : torch_geometric.data.Data
        Graf publicat G' (l'únic input de l'adversari).
    mechanism_name : str
    epsilon : float or None
    seed : int
    gcn_epochs, gcn_patience : int
    neg_ratio : float
    verbose : bool

    Returns
    -------
    attack_results : dict
        {
          'mechanism', 'epsilon', 'n_pairs', 'n_pos', 'n_neg', 'prevalence',
          'scores': { score_name: {'auc_roc':.., 'avg_precision':..}, ... }
        }
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    label = f"ε={epsilon:.2f}" if epsilon is not None else "No DP"
    if verbose:
        if mechanism_name == 'Baseline':
            print(f"  [GCN] Entrenant sobre graf original ({mechanism_name}, {label})...",
                  end=' ')
        else:
            print(f"  [GCN] Entrenant sobre graf publicat ({mechanism_name}, {label})...",
                  end=' ')

    # 1. Entrenar el GCN sobre el graf publicat (o original, per al baseline).
    #    L'adversari pot fer-ho: G' és públic.
    gcn = GCN(data_perturbed.num_features, 64, data_perturbed.y.max().item() + 1)
    gcn = train_gcn(gcn, data_perturbed, num_epochs=gcn_epochs, patience=gcn_patience)
    if verbose:
        print("fet.")

    # 2. Extreure embeddings de la capa oculta sobre G' (mètode natiu del GCN).
    gcn.eval()
    with torch.no_grad():
        embeddings = gcn.get_embeddings(
            data_perturbed.x, data_perturbed.edge_index
        ).numpy()

    # 3. Univers de parelles + veritat-terra (paper de l'avaluador).
    if verbose:
        print(f"  [ATAC] Construint univers de parelles (1:{int(neg_ratio)})...", end=' ')
    pairs, labels = build_edge_pairs(
        data_original, neg_ratio=neg_ratio, seed=seed, verbose=False
    )
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    prevalence = n_pos / len(labels) if len(labels) > 0 else 0.0
    if verbose:
        print(f"fet. ({n_pos} positius, {n_neg} negatius, prevalença {prevalence:.4f})")

    # 4. Puntuacions de versemblança (paper de l'adversari: només G').
    if verbose:
        print("  [ATAC] Calculant puntuacions sobre el graf publicat...", end=' ')
    scores = compute_link_scores(pairs, data_perturbed, embeddings)
    if verbose:
        print("fet.")

    # 5. Mesura: AUC-ROC i AP de cada puntuació contra la veritat-terra.
    #    Cada puntuació és un atac independent.
    score_results = {}
    for name in SCORE_NAMES:
        s = scores[name]
        # Si la puntuació és constant, AUC no està definida -> atzar (0.5).
        if np.all(s == s[0]):
            auc = 0.5
        else:
            auc = roc_auc_score(labels, s)
        ap = average_precision_score(labels, s)
        score_results[name] = {
            'auc_roc': float(auc),
            'avg_precision': float(ap),
        }

    return {
        'mechanism': mechanism_name,
        'epsilon': epsilon,
        'n_pairs': len(labels),
        'n_pos': n_pos,
        'n_neg': n_neg,
        'prevalence': prevalence,
        'scores': score_results,
    }


# ============================================================================
# PIPELINE PRINCIPAL: MÚLTIPLES EPSILONS I MECANISMES
# ============================================================================

def run_edge_inference_attack_experiment(
    data_original,
    epsilon_values: List[float],
    num_runs: int = 3,
    gcn_epochs: int = 500,
    gcn_patience: int = 20,
    neg_ratio: float = 100.0,
    seed_base: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Executa l'atac no supervisat per a múltiples epsilons i els dos
    mecanismes DP, amb `num_runs` execucions independents per configuració.

    La pertorbació del graf reutilitza main.apply_dp_to_graph (mateixa
    semàntica que els experiments d'utilitat: es pertorba tot el graf).

    Returns
    -------
    all_results : dict
        Claus: 'Baseline', 'RR_eps_{e:.2f}', 'Laplace_eps_{e:.2f}'.
        Cada valor és una llista de num_runs resultats de run_single_attack.
    """
    all_results = {}

    # -------- BASELINE (sense pertorbació) --------
    print("\n" + "=" * 70)
    print("ATAC D'INFERÈNCIA (no supervisat) — BASELINE (sense DP)")
    print("=" * 70)

    baseline_runs = []
    for run in range(num_runs):
        if verbose:
            print(f"\n  Run {run + 1}/{num_runs}")
        result = run_single_attack(
            data_original=data_original,
            data_perturbed=data_original,
            mechanism_name='Baseline',
            epsilon=None,
            seed=seed_base + run,
            gcn_epochs=gcn_epochs,
            gcn_patience=gcn_patience,
            neg_ratio=neg_ratio,
            verbose=verbose,
        )
        baseline_runs.append(result)
    all_results['Baseline'] = baseline_runs

    # -------- DP per a cada epsilon --------
    mechanisms = [
        ('RR', edge_dp_randomized_response),
        ('Laplace', edge_dp_laplace_mechanism),
    ]

    for eps_idx, epsilon in enumerate(epsilon_values, 1):
        print("\n" + "=" * 70)
        print(f"ATAC D'INFERÈNCIA (no supervisat) — ε = {epsilon:.2f} "
              f"({eps_idx}/{len(epsilon_values)})")
        print("=" * 70)

        for mech_name, dp_func in mechanisms:
            runs = []
            for run in range(num_runs):
                seed = seed_base + run
                if verbose:
                    print(f"\n  Run {run + 1}/{num_runs} — {mech_name} (ε={epsilon:.2f})")

                # Pertorbació del graf sencer (reutilitza main.apply_dp_to_graph)
                data_dp, _ = apply_dp_to_graph(
                    data_original, dp_func, (epsilon,), seed=seed
                )

                result = run_single_attack(
                    data_original=data_original,
                    data_perturbed=data_dp,
                    mechanism_name=mech_name,
                    epsilon=epsilon,
                    seed=seed,
                    gcn_epochs=gcn_epochs,
                    gcn_patience=gcn_patience,
                    neg_ratio=neg_ratio,
                    verbose=verbose,
                )
                runs.append(result)

            all_results[f'{mech_name}_eps_{epsilon:.2f}'] = runs

    return all_results


# ============================================================================
# AGREGACIÓ D'ESTADÍSTIQUES
# ============================================================================

def aggregate_results(all_results: Dict) -> Dict:
    """
    Mitjana ± desviació estàndard de l'AUC-ROC i l'AP de cada puntuació,
    a través de les execucions. La prevalença (constant entre runs amb la
    mateixa seed base) es conserva per a la interpretació de l'AP.

    Returns
    -------
    stats : dict
        stats[key]['prevalence'] : float
        stats[key]['scores'][score_name] = {
            'auc_roc': {'mean':.., 'std':..},
            'avg_precision': {'mean':.., 'std':..},
        }
    """
    stats = {}
    for key, runs in all_results.items():
        prevalence = float(np.mean([r['prevalence'] for r in runs]))
        score_stats = {}
        for name in SCORE_NAMES:
            aucs = [r['scores'][name]['auc_roc'] for r in runs]
            aps = [r['scores'][name]['avg_precision'] for r in runs]
            score_stats[name] = {
                'auc_roc': {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs, ddof=1))},
                'avg_precision': {'mean': float(np.mean(aps)), 'std': float(np.std(aps, ddof=1))},
            }
        stats[key] = {'prevalence': prevalence, 'scores': score_stats}
    return stats


# ============================================================================
# PRESENTACIÓ DE RESULTATS
# ============================================================================

def print_attack_results(stats: Dict, epsilon_values: List[float]):
    """
    Imprimeix els resultats de l'atac de forma estructurada.

    AUC-ROC és la mètrica principal:
      - AUC ≈ 0.50 -> protecció màxima (atzar)
      - AUC ≈ 0.75 -> filtració moderada
      - AUC ≈ 1.00 -> filtració severa
    L'AP s'ha d'interpretar contra la prevalença (un atac aleatori dona
    AP ≈ prevalença).
    """
    print("\n\n" + "=" * 88)
    print("RESULTATS DE L'ATAC D'INFERÈNCIA SOBRE ARESTES (no supervisat)")
    print("Mètrica principal: AUC-ROC (0.5 = protecció perfecta, 1.0 = filtració total)")
    print("Cada puntuació és un atac independent calculat només sobre el graf publicat")
    print("=" * 88)

    def _print_block(key, label):
        if key not in stats:
            return
        prevalence = stats[key]['prevalence']
        print(f"\n  {label}  (prevalença AP-aleatòria ≈ {prevalence:.4f})")
        print(f"  {'Score':<26} {'AUC-ROC':>16} {'Avg Precision':>18}")
        print(f"  {'─' * 62}")
        for name in SCORE_NAMES:
            s = stats[key]['scores'][name]
            print(f"  {name:<26} "
                  f"{s['auc_roc']['mean']:>8.4f}±{s['auc_roc']['std']:.3f} "
                  f"{s['avg_precision']['mean']:>10.4f}±{s['avg_precision']['std']:.3f}")

    _print_block('Baseline', 'Baseline (sense DP)')

    for epsilon in sorted(epsilon_values):
        for mech in ['RR', 'Laplace']:
            _print_block(f'{mech}_eps_{epsilon:.2f}', f"{mech} (ε={epsilon:.2f})")

    # Reducció d'AUC respecte al baseline, pel millor score de cada config.
    print("\n\n" + "=" * 88)
    print("MILLOR AUC-ROC PER CONFIGURACIÓ I REDUCCIÓ RESPECTE AL BASELINE")
    print("(El millor score és el que més filtra: pitjor cas per a la privadesa)")
    print("=" * 88)

    def _best_auc(key):
        if key not in stats:
            return float('nan'), '—'
        best_name, best_val = None, -1.0
        for name in SCORE_NAMES:
            v = stats[key]['scores'][name]['auc_roc']['mean']
            if v > best_val:
                best_val, best_name = v, name
        return best_val, best_name

    base_auc, base_name = _best_auc('Baseline')
    if not np.isnan(base_auc):
        print(f"\n  Baseline: millor AUC-ROC = {base_auc:.4f} ({base_name})\n")
        print(f"  {'Epsilon':<10} {'RR best AUC':>14} {'ΔAUC':>9}   "
              f"{'Laplace best AUC':>18} {'ΔAUC':>9}")
        print(f"  {'─' * 64}")
        for epsilon in sorted(epsilon_values):
            rr_auc, _ = _best_auc(f'RR_eps_{epsilon:.2f}')
            lap_auc, _ = _best_auc(f'Laplace_eps_{epsilon:.2f}')
            print(f"  ε={epsilon:<8.2f} {rr_auc:>14.4f} {base_auc - rr_auc:>+9.4f}   "
                  f"{lap_auc:>18.4f} {base_auc - lap_auc:>+9.4f}")
        print(f"\n  ΔAUC positiu = el mecanisme DP redueix la capacitat de l'adversari.")
        print(f"  ΔAUC ≈ {base_auc:.4f} - 0.50 = {base_auc - 0.5:.4f} seria protecció màxima.")


# ============================================================================
# PUNT D'ENTRADA PRINCIPAL
# ============================================================================

def main(dataset_name: str = 'Cora'):
    GLOBAL_SEED = 42
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 70)
    print("EXPERIMENT: ATAC D'INFERÈNCIA SOBRE ARESTES CONTRA EDGE-DP")
    print(f"Dataset: {dataset_name}  |  GCN  |  Adversari: caixa negra SENSE arestes reals")
    print("Atac no supervisat: puntuacions de versemblança (sense classificador)")
    print("Scores: CN, Jaccard, Adamic-Adar, Resource Allocation, PA, embedding cosine")
    print("=" * 70)

    print(f"\n1. Carregant dataset {dataset_name}...")
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
    data = dataset[0]

    # PubMed és massa gran per a l'atac complet: se'n pren una bola BFS de
    # ~PUBMED_SUBSET_NODES nodes ABANS del split i de la DP, de manera que
    # G' i G comparteixin el mateix conjunt de nodes. Cora i CiteSeer
    # s'usen sencers.
    if dataset_name == 'Pubmed':
        print(f"\n   Submostreig BFS de PubMed a ~{PUBMED_SUBSET_NODES} nodes...")
        data = subsample_bfs_ball(
            data, max_nodes=PUBMED_SUBSET_NODES, seed=GLOBAL_SEED, verbose=True
        )

    # Reutilitza el split de main.py (mateixa partició 60/20/20 i seed).
    data = create_custom_split(data, seed=GLOBAL_SEED)

    print(f"   Nodes: {data.num_nodes} | Arestes: {data.num_edges} | "
          f"Features: {data.num_features} | Classes: {data.y.max().item() + 1}")
    print(f"   Split: train={data.train_mask.sum().item()}, "
          f"val={data.val_mask.sum().item()}, "
          f"test={data.test_mask.sum().item()}")

    epsilon_values = [0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    num_runs = 10

    print(f"\n2. Valors d'epsilon: {epsilon_values}")
    print(f"   Execucions per configuració: {num_runs}")
    print(f"   Total configuracions: 1 baseline + {len(epsilon_values) * 2} DP")

    print("\n3. Creant directori de resultats...")
    results_dir = create_results_directory(dataset_name=f"InferenceAttack_{dataset_name}")
    print(f"   Resultats es guardaran a: {results_dir}\n")

    all_results = run_edge_inference_attack_experiment(
        data_original=data,
        epsilon_values=epsilon_values,
        num_runs=num_runs,
        gcn_epochs=500,
        gcn_patience=20,
        neg_ratio=100.0,
        seed_base=GLOBAL_SEED,
        verbose=True,
    )

    stats = aggregate_results(all_results)
    print_attack_results(stats, epsilon_values)

    print("\n4. Guardant resultats...")
    save_inference_attack_results(
        results_dir,
        all_results,
        stats,
        data,
        epsilon_values,
        num_runs,
        dataset_name=dataset_name,
    )

    return all_results, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Atac d'inferència sobre arestes contra Edge-DP "
                    "(no supervisat). Suporta Cora, CiteSeer i PubMed "
                    "(aquest darrer submostrejat per bola BFS)."
    )
    parser.add_argument(
        '--dataset', '-d', default='Cora', choices=SUPPORTED_DATASETS,
        help="Conjunt Planetoid a utilitzar (per defecte: Cora). "
             "PubMed s'submostreja a una bola BFS de ~3000 nodes."
    )
    args = parser.parse_args()
    main(dataset_name=args.dataset)