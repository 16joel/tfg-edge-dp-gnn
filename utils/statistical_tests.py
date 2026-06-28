"""
Tests estadístics aparellats sobre els resultats d'utilitat (Edge-DP / GNN).

MOTIVACIÓ
---------
Els experiments de main.py reporten mitjana ± desviació estàndard de la
precisió i la F1 al llarg de N execucions (seeds), però no proven si les
diferències observades entre condicions són estadísticament distingibles
del soroll. Aquest mòdul afegeix aquesta capa inferencial SENSE tornar a
executar res: consumeix directament l'estructura `all_results` que retorna
main.run_multiple_experiments (una llista de resultats per run per a cada
(epsilon, mètode)).

DISSENY APARELLAT
-----------------
La comparació clau és RR vs Laplace per a un mateix epsilon i escenari. Com
que tots dos mecanismes s'avaluen amb les MATEIXES seeds (seed_base + run),
els resultats estan aparellats run a run. Per a dades aparellades amb mostra
petita (típicament 5 seeds) i sense supòsit de normalitat, el test no
paramètric adequat és el de Wilcoxon de rangs amb signe sobre les diferències
per parella. Com a complement es reporta també la mida de l'efecte (mitjana
de les diferències i d de Cohen aparellada), perquè amb N petita un p-valor
no significatiu NO demostra equivalència: la mida de l'efecte ajuda a
distingir "sense efecte" de "sense potència".

INTERPRETACIÓ
-------------
- p alt (p. ex. > 0,05): no hi ha prou evidència per afirmar que els dos
  mecanismes difereixin en aquesta condició. Amb 5 seeds la potència és
  baixa, així que això és consistent amb la lectura "RR i Laplace són
  pràcticament equivalents", PERÒ s'ha de dir amb prudència (vegeu la mida
  de l'efecte).
- p baix: la diferència és sistemàtica entre seeds, no atribuïble al soroll.

REUTILITZACIÓ DE CODI
---------------------
No es redefineix cap lògica d'experiment. El mòdul només LLEGEIX
`all_results` (mateixa estructura que compute_statistics de main.py) i les
constants METHOD_KEYS via paràmetre. L'única dependència nova és
scipy.stats.wilcoxon, ja inclosa a l'stack científic del projecte.

REFERÈNCIA
----------
- Wilcoxon, F. (1945). "Individual Comparisons by Ranking Methods."
  Biometrics Bulletin, 1(6), 80-83.
- Demšar, J. (2006). "Statistical Comparisons of Classifiers over Multiple
  Data Sets." JMLR 7, 1-30. (justifica tests no paramètrics aparellats en
  comparació d'algorismes d'aprenentatge automàtic).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Aquest mòdul requereix scipy. Instal·la-la amb 'pip install scipy'."
    ) from exc


def _paired_vectors(
    all_results: Dict,
    epsilon_key: str,
    method_a: str,
    method_b: str,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extreu els vectors aparellats d'una mètrica per a dos mètodes sota un
    mateix epsilon.

    Parameters
    ----------
    all_results : dict
        Estructura retornada per run_multiple_experiments.
    epsilon_key : str
        Clau del tipus 'epsilon_3.00'.
    method_a, method_b : str
        Claus de METHOD_KEYS (p. ex. 'rr_dp_s2', 'laplace_dp_s2').
    metric : str
        'test_accuracy', 'test_f1', 'val_accuracy' o 'val_f1'.

    Returns
    -------
    a, b : np.ndarray
        Vectors aparellats (mateixa longitud = nombre de runs).
    """
    block = all_results[epsilon_key]
    runs_a = block[method_a]
    runs_b = block[method_b]
    if len(runs_a) != len(runs_b):
        raise ValueError(
            f"Nombre de runs diferent entre {method_a} ({len(runs_a)}) i "
            f"{method_b} ({len(runs_b)}); el test aparellat no és aplicable."
        )
    a = np.array([r[metric] for r in runs_a], dtype=np.float64)
    b = np.array([r[metric] for r in runs_b], dtype=np.float64)
    return a, b


def paired_wilcoxon(
    a: np.ndarray,
    b: np.ndarray,
) -> Dict[str, float]:
    """
    Test de Wilcoxon de rangs amb signe sobre diferències aparellades, més
    estadístics de mida d'efecte.

    Gestiona el cas degenerat en què totes les diferències són zero (Wilcoxon
    no està definit): es reporta p = 1.0 i efecte nul.

    Returns
    -------
    dict amb:
        n              : nombre de parelles
        n_nonzero      : parelles amb diferència no nul·la (les que conten)
        mean_diff      : mitjana de (a - b)
        std_diff       : desviació estàndard de (a - b), ddof=1
        cohen_dz       : mida d'efecte aparellada (±inf si efecte consistent
                         amb variància nul·la)
        statistic      : estadístic W de Wilcoxon (NaN si no aplicable)
        p_value        : p-valor de dues cues (1.0 si totes les dif. són 0)
        min_p_possible : p-valor mínim assolible donat n_nonzero (2/2^n)
        underpowered   : True si min_p_possible > 0,05 (cap resultat pot ser
                         significatiu en aquest règim, p. ex. n=5)
        significant    : p_value < 0.05
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    n = int(diff.size)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    # Mida d'efecte aparellada (Cohen dz = mitjana / desv. de les diferències).
    # Cas especial: si std_diff ≈ 0 però la mitjana no és 0, l'efecte és
    # perfectament consistent entre seeds (totes les diferències idèntiques);
    # dz divergeix matemàticament. Es marca com a infinit amb signe per no
    # reportar un valor numèric espuri (p. ex. 1e15 per error de coma flotant).
    std_is_zero = std_diff <= 1e-12
    if not std_is_zero:
        cohen_dz = float(mean_diff / std_diff)
    elif mean_diff == 0:
        cohen_dz = 0.0
    else:
        cohen_dz = float('inf') if mean_diff > 0 else float('-inf')

    # Wilcoxon no està definit si totes les diferències són exactament 0.
    if np.allclose(diff, 0.0):
        statistic, p_value = float('nan'), 1.0
    else:
        # zero_method='wilcox' descarta els zeros (comportament clàssic).
        # mode='auto' tria distribució exacta amb mostra petita.
        try:
            res = wilcoxon(diff, zero_method='wilcox', alternative='two-sided')
            statistic, p_value = float(res.statistic), float(res.pvalue)
        except ValueError:
            # p. ex. després de descartar zeros no queda cap diferència.
            statistic, p_value = float('nan'), 1.0

    # Límit de potència amb mostra petita: amb n parelles efectives, el
    # p-valor de dues cues de Wilcoxon no pot baixar de 2 / 2^n (= 0,0625 per
    # a n=5). És a dir, amb 5 seeds CAP comparació pot ser significativa a
    # 0,05 encara que l'efecte sigui perfectament consistent. Es reporta
    # aquesta cota perquè la interpretació no confongui "no significatiu" amb
    # "sense efecte" en aquest règim.
    n_nonzero = int(np.count_nonzero(~np.isclose(diff, 0.0)))
    min_p_possible = (2.0 / (2 ** n_nonzero)) if n_nonzero > 0 else 1.0
    underpowered = bool(min_p_possible > 0.05)

    return {
        'n': n,
        'n_nonzero': n_nonzero,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'cohen_dz': cohen_dz,
        'statistic': statistic,
        'p_value': p_value,
        'min_p_possible': min_p_possible,
        'underpowered': underpowered,
        'significant': bool(p_value < 0.05),
    }


def compare_mechanisms(
    all_results: Dict,
    epsilon_values: List[float],
    scenarios: Tuple[str, ...] = ('s1', 's2', 's3'),
    metric: str = 'test_accuracy',
    mech_a: str = 'rr_dp',
    mech_b: str = 'laplace_dp',
) -> List[Dict]:
    """
    Compara dos mecanismes (per defecte RR vs Laplace) per a cada epsilon i
    escenari amb un test de Wilcoxon aparellat sobre una mètrica.

    Returns
    -------
    list[dict]
        Una fila per (epsilon, escenari) amb les claus de paired_wilcoxon
        més 'epsilon' i 'scenario'.
    """
    rows = []
    for epsilon in sorted(epsilon_values):
        epsilon_key = f'epsilon_{epsilon:.2f}'
        if epsilon_key not in all_results:
            continue
        for sc in scenarios:
            method_a = f'{mech_a}_{sc}'
            method_b = f'{mech_b}_{sc}'
            block = all_results[epsilon_key]
            if method_a not in block or method_b not in block:
                continue
            if len(block[method_a]) == 0 or len(block[method_b]) == 0:
                continue
            a, b = _paired_vectors(
                all_results, epsilon_key, method_a, method_b, metric
            )
            result = paired_wilcoxon(a, b)
            result['epsilon'] = float(epsilon)
            result['scenario'] = sc.upper()
            rows.append(result)
    return rows


def print_comparison_table(
    rows: List[Dict],
    metric: str = 'test_accuracy',
    mech_a: str = 'RR',
    mech_b: str = 'Laplace',
):
    """
    Imprimeix una taula llegible dels resultats de compare_mechanisms.
    """
    print("\n" + "=" * 88)
    print(f"TEST DE WILCOXON APARELLAT — {mech_a} vs {mech_b}  (mètrica: {metric})")
    print("H0: no hi ha diferència sistemàtica entre mecanismes per a la mateixa seed")
    print("=" * 88)
    print(f"{'Epsilon':>8} {'Esc.':>5} {'n':>3} "
          f"{'mitjana(A-B)':>14} {'sd(dif)':>10} {'Cohen dz':>10} "
          f"{'W':>8} {'p-valor':>10} {'sig.':>6}")
    print("-" * 88)
    any_underpowered = False
    for r in rows:
        w = '—' if np.isnan(r['statistic']) else f"{r['statistic']:.1f}"
        sig = 'SÍ' if r['significant'] else 'no'
        if np.isinf(r['cohen_dz']):
            dz = '+inf' if r['cohen_dz'] > 0 else '-inf'
        else:
            dz = f"{r['cohen_dz']:+.3f}"
        if r.get('underpowered'):
            any_underpowered = True
            sig = 'no*'
        print(f"{r['epsilon']:>8.2f} {r['scenario']:>5} {r['n']:>3} "
              f"{r['mean_diff']:>+14.4f} {r['std_diff']:>10.4f} "
              f"{dz:>10} {w:>8} {r['p_value']:>10.4f} {sig:>6}")
    print("-" * 88)
    n_sig = sum(1 for r in rows if r['significant'])
    print(f"Comparacions significatives (p<0,05): {n_sig}/{len(rows)}")
    if any_underpowered:
        min_p = rows[0]['min_p_possible'] if rows else float('nan')
        print(f"* Amb n={rows[0]['n']} seeds, el p-valor mínim assolible és "
              f"{min_p:.4f} > 0,05: CAP comparació pot sortir significativa,")
        print("  encara que l'efecte sigui consistent. Per detectar diferències")
        print("  fines RR/Laplace amb significança caldrien més seeds (p. ex. "
              "n>=6 baixa el mínim a 0,03125).")
    print("NOTA: amb poques seeds la potència és baixa; un p alt és compatible")
    print("amb equivalència pràctica, però llegiu-lo junt amb la mida d'efecte")
    print("(Cohen dz): |dz| petit reforça l'equivalència; |dz| gran amb p alt")
    print("indica manca de potència, no absència d'efecte.")


def save_comparison_csv(rows: List[Dict], filepath: str, metric: str = 'test_accuracy'):
    """
    Desa els resultats de compare_mechanisms en un CSV (mateix estil
    minimalista que la resta de results_saver del projecte).
    """
    import csv
    fieldnames = ['epsilon', 'scenario', 'n', 'n_nonzero', 'mean_diff',
                  'std_diff', 'cohen_dz', 'statistic', 'p_value',
                  'min_p_possible', 'underpowered', 'significant']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})


def run_all_paired_tests(
    all_results: Dict,
    epsilon_values: List[float],
    metrics: Tuple[str, ...] = ('test_accuracy', 'test_f1'),
    output_dir: Optional[str] = None,
):
    """
    Funció de conveniència: executa la comparació RR vs Laplace per a diverses
    mètriques, imprimeix cada taula i (opcionalment) desa els CSV.

    Pensada per cridar-se DESPRÉS de main.main():

        from main import main
        from utils.statistical_tests import run_all_paired_tests
        all_results, stats, data, name = main(
            dataset_name='Cora',
            epsilon_values=[0.1, 1, 3, 5, 7, 9, 11, 13, 15],
            num_runs=5,
        )
        run_all_paired_tests(all_results, [0.1,1,3,5,7,9,11,13,15],
                             output_dir='results')
    """
    import os
    for metric in metrics:
        rows = compare_mechanisms(all_results, epsilon_values, metric=metric)
        print_comparison_table(rows, metric=metric)
        if output_dir is not None and rows:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f'wilcoxon_rr_vs_laplace_{metric}.csv')
            save_comparison_csv(rows, path, metric=metric)
            print(f"\n  CSV desat a: {path}")
