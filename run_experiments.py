"""
Orquestrador d'experiments: escombrat d'epsilon sobre un o més conjunts de
dades (Cora, Citeseer, Pubmed) + tests estadístics aparellats RR vs Laplace.

Aquest script NO duplica cap lògica: només encadena
  - main.main()                          (pipeline d'utilitat per escenaris)
  - utils.statistical_tests.run_all...   (Wilcoxon aparellat)
i desa els resultats amb el results_saver del projecte.

ÚS
--
  # Escombrat complet sobre Cora (reprodueix les taules de l'informe) + tests:
  python run_experiments.py --datasets Cora --epsilons 0.1 1 3 5 7 9 11 13 15 --runs 5

  # Afegir Citeseer (desactiva la crítica de "dataset únic"):
  python run_experiments.py --datasets Cora Citeseer \
      --epsilons 0.1 1 3 5 7 9 11 13 15 --runs 5

  # Prova ràpida (pocs epsilons, poques èpoques) per validar que tot corre:
  python run_experiments.py --datasets Cora --epsilons 1 3 --runs 3 --epochs 100

NOTA SOBRE EL TEMPS
-------------------
Cada (dataset, epsilon, mecanisme, run) entrena un GCN. L'escombrat complet
de 9 epsilons × 2 mecanismes × 5 runs × 3 escenaris (derivats d'un sol
entrenament) és assumible en CPU per a Cora i Citeseer; Pubmed és força més
gran (~19.700 nodes) i la implementació densa O(|V|^2) dels mecanismes el fa
molt costós: s'aconsella reservar Pubmed per a una execució a part.
"""

import argparse

from main import main, SUPPORTED_DATASETS
from utils.statistical_tests import run_all_paired_tests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Escombrat multi-dataset + tests aparellats RR vs Laplace."
    )
    parser.add_argument(
        '--datasets', nargs='+', default=['Cora'], choices=SUPPORTED_DATASETS,
        help="Conjunts a executar (per defecte: Cora)."
    )
    parser.add_argument(
        '--epsilons', type=float, nargs='+',
        default=[0.1, 1, 3, 5, 7, 9, 11, 13, 15],
        help="Escombrat d'epsilon (per defecte: el de l'informe)."
    )
    parser.add_argument('--runs', type=int, default=5,
                        help="Execucions per configuració (per defecte: 5).")
    parser.add_argument('--epochs', type=int, default=500,
                        help="Èpoques màximes (per defecte: 500).")
    parser.add_argument('--patience', type=int, default=20,
                        help="Paciència early stopping (per defecte: 20).")
    parser.add_argument('--seed', type=int, default=42,
                        help="Llavor global (per defecte: 42).")
    parser.add_argument('--no-tests', action='store_true',
                        help="Salta els tests estadístics (només utilitat).")
    parser.add_argument('--output-dir', default='results',
                        help="Directori on desar els CSV dels tests.")
    return parser.parse_args()


def main_cli():
    args = parse_args()

    for dataset_name in args.datasets:
        print("\n" + "#" * 88)
        print(f"# DATASET: {dataset_name}")
        print("#" * 88 + "\n")

        all_results, stats, data, name, perf = main(
            dataset_name=dataset_name,
            epsilon_values=args.epsilons,
            num_runs=args.runs,
            num_epochs=args.epochs,
            patience=args.patience,
            global_seed=args.seed,
        )

        # Persistència d'utilitat (reutilitza el results_saver via main.py
        # quan s'executa main.py directament; aquí ho fem explícit perquè
        # run_experiments.py crida main() com a funció).
        from utils.results_saver import create_results_directory, save_all_results
        from utils.perf_monitor import save_performance
        from dp_mechanisms import budget_split_description

        results_dir = create_results_directory(dataset_name=name, base_path=args.output_dir)
        dataset_info = {
            'name': name,
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges,
            'num_features': data.num_features,
            'num_classes': data.y.max().item() + 1,
        }
        privacy_params = {
            'epsilon_values': sorted(args.epsilons),
            'budget_split': budget_split_description(),
            'scenarios': 'S1/S2/S3 (vegeu main.py)',
        }
        save_all_results(results_dir, all_results, stats, dataset_info, privacy_params)
        save_performance(results_dir, perf, dataset_name=name)

        if not args.no_tests:
            run_all_paired_tests(
                all_results,
                args.epsilons,
                metrics=('test_accuracy', 'test_f1'),
                output_dir=results_dir,
            )

    print("\n✓ Tot completat.\n")


if __name__ == "__main__":
    main_cli()
