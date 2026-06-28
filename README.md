# Privadesa diferencial a nivell d'aresta en xarxes neuronals de grafs

> Compromís entre privadesa i utilitat d'aplicar **edge-level Differential
> Privacy (edge-DP)** sobre l'entrada d'una **Graph Convolutional Network
> (GCN)**, i protecció efectiva davant d'un **atac d'inferència d'arestes**.

Codi i resultats del Treball de Fi de Grau (Enginyeria de Dades, Escola
d'Enginyeria, Universitat Autònoma de Barcelona). S'implementen dos mecanismes
edge-DP —**Randomized Response asimètrica** i **Laplace amb selecció top-k**,
tots dos calibrats contra una estimació de densitat privada— i s'avaluen sobre
**Cora**, **Citeseer** i **Pubmed** distingint tres escenaris de desplegament
(S1/S2/S3) i un atac d'inferència d'arestes de caixa negra.

---

## Execució ràpida

```bash
# Experiments d'utilitat (baseline + RR + Laplace, escenaris S1/S2/S3)
python main.py --dataset Cora --epsilons 0.1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --runs 10

# Atac d'inferència d'arestes (adversari de caixa negra)
python inference_attack.py --dataset Cora
```

El detall complet d'arguments i opcions és a la secció **Manual d'ús**.

---

## Resultats principals

- El cost de la privadesa depèn **críticament de l'escenari de desplegament**.
  Quan el graf real és disponible en inferència (S2/S3), s'obté una utilitat a
  pocs punts del *baseline* fins i tot amb privadesa forta (ε ≤ 3); quan el
  model opera només sobre el graf perturbat (S1), la utilitat només es recupera
  per a ε gran.
- L'atac d'inferència confirma que els dos mecanismes redueixen l'AUC de
  l'adversari a nivells propers a l'atzar en privadesa forta, amb una protecció
  gairebé indistingible entre tots dos.
- Existeix una **finestra operativa** (ε ∈ [1, 3]) on protecció i utilitat (en
  inferència real) conviuen.

---

## Estructura del repositori

```
Codi/
├── main.py                  # Experiments d'utilitat (classificació de nodes, escenaris S1/S2/S3)
├── run_experiments.py       # Orquestrador multi-dataset + tests estadístics RR vs Laplace
├── inference_attack.py      # Atac d'inferència d'arestes (no supervisat, basat en scores)
├── main_binary.py           # [LLEGAT] Variant One-vs-Rest (no s'usa per als resultats principals)
├── generate_figures.py      # Generació de les figures de l'informe (a partir de results/)
├── requirements.txt         # Dependències
├── README.md                # Aquest fitxer
├── REPRODUCIBILITAT.md      # Mecanismes de reproductibilitat
├── CHANGELOG.md             # Llista de canvis del treball (evolució i decisions)
├── LICENSE                  # Llicència MIT
├── CITATION.cff             # Metadades de citació (botó "Cite this repository")
├── dp_mechanisms/
│   ├── config.py            # Repartiment del pressupost ε (FONT ÚNICA DE VERITAT)
│   ├── edge_dp_rr.py        # Randomized Response asimètrica
│   └── edge_dp_laplace.py   # Laplace amb selecció top-k
├── models/
│   └── gcn.py               # GCN de 2 capes (amb get_embeddings per a l'atac)
├── utils/
│   ├── metrics.py           # train, test, train_and_evaluate, train_and_evaluate_scenarios
│   ├── statistical_tests.py # Test de Wilcoxon aparellat RR vs Laplace
│   ├── perf_monitor.py      # Mesura de temps, RAM i overhead de privadesa
│   └── results_saver.py     # Persistència de resultats (utilitat i atac)
├── figures_generated/       # Figures de l'informe en PDF (fig2–fig5, figA, figB)
└── results/                 # Sortides dels experiments (versionades en aquest repo)
```

---

## Instal·lació

Es recomana **Python 3.10+** i un entorn virtual:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

El codi s'executa per defecte en **CPU**. Per a GPU, instal·leu la versió de
PyTorch corresponent a la vostra versió de CUDA des de
[pytorch.org](https://pytorch.org/get-started/locally/). Els conjunts de dades
(Cora, Citeseer, Pubmed) es descarreguen automàticament amb `Planetoid` la
primera vegada que s'executa.

---

## Manual d'ús

### 1. Experiments d'utilitat (`main.py`)

Executa el *baseline* sense privadesa i els dos mecanismes edge-DP, avaluant
els tres escenaris S1/S2/S3 per a cada ε.

```bash
# Execució mínima per defecte (Cora, ε=0.01, 5 runs) — només per validar que tot corre
python main.py

# Escombrat complet sobre Cora amb 10 execucions (reprodueix els experiments de l'informe)
python main.py --dataset Cora --epsilons 0.1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --runs 10

# Mateix experiment sobre Citeseer o Pubmed (només cal canviar --dataset)
python main.py --dataset Citeseer --epsilons 0.1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --runs 10
python main.py --dataset Pubmed   --epsilons 0.1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --runs 10
```

Arguments disponibles:

| Argument | Per defecte | Descripció |
|----------|-------------|------------|
| `--dataset` | `Cora` | Conjunt Planetoid: `Cora`, `Citeseer` o `Pubmed`. |
| `--epsilons` | `[0.01]` | Llista d'epsilons a avaluar. |
| `--runs` | `5` | Execucions independents per configuració. |
| `--epochs` | `500` | Èpoques màximes (amb *early stopping*). |
| `--patience` | `20` | Paciència de l'*early stopping*. |
| `--seed` | `42` | Llavor global de reproductibilitat. |

> Els resultats es desen a `results/<Dataset>_<timestamp>/`.
> Nota: l'informe es va generar amb `--runs 10`.

### 2. Escombrat multi-dataset + tests estadístics (`run_experiments.py`)

Encadena `main.py` sobre un o més conjunts i hi afegeix el test de Wilcoxon
aparellat RR vs Laplace. **És la via recomanada per reproduir l'informe.**

```bash
# Cora amb tests estadístics
python run_experiments.py --datasets Cora \
    --epsilons 0.1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --runs 10

# Diversos conjunts de cop
python run_experiments.py --datasets Cora Citeseer Pubmed \
    --epsilons 0.1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --runs 10
```

Opcions addicionals: `--no-tests` (salta el Wilcoxon), `--output-dir`.
Nota: Pubmed (~19.700 nodes) és molt més costós per la implementació densa
O(|V|²) dels mecanismes; convé reservar-lo per a una execució a part.

### 3. Atac d'inferència d'arestes (`inference_attack.py`)

Avalua la protecció efectiva amb un adversari de caixa negra **sense
coneixement de cap aresta real**. Calcula sis puntuacions de versemblança
d'aresta (Common Neighbors, Jaccard, Adamic-Adar, Resource Allocation,
Preferential Attachment i similitud cosinus d'embeddings) sobre el graf
publicat i en mesura l'AUC-ROC i l'Average Precision contra la veritat-terra.

```bash
python inference_attack.py --dataset Cora
python inference_attack.py --dataset Citeseer
python inference_attack.py --dataset Pubmed     # se submostreja a una bola BFS de ~3000 nodes
```

> L'escombrat d'epsilons `[0.01, 0.1, 1, …, 15]` i el nombre d'execucions (10)
> estan fixats dins de `inference_attack.main()`. Pubmed se submostreja amb una
> bola BFS connexa de ~3000 nodes (constant `PUBMED_SUBSET_NODES`), perquè
> l'atac complet sobre tot el graf seria computacionalment inviable.
> Els resultats es desen a `results/InferenceAttack_<Dataset>_<timestamp>/`.

### 4. `main_binary.py` (llegat — One-vs-Rest)

`main_binary.py` és una **variant antiga** que descompon la classificació
multiclasse de Cora en 7 tasques binàries (One-vs-Rest) i reporta F1 binari i
AUC-ROC per classe. **No s'utilitza per als resultats de l'informe** i es manté
només com a anàlisi complementària. Comparteix els mecanismes DP i el
repartiment de pressupost (`dp_mechanisms/config.py`) amb `main.py`, però no
implementa els escenaris S1/S2/S3 ni l'escombrat documentat a l'informe. Si es
vol executar:

```bash
python main_binary.py
```

No cal per reproduir cap resultat de la memòria; es pot ignorar amb seguretat.

---

## Escenaris d'avaluació (utilitat)

Tots tres **entrenen sempre sobre el graf perturbat**; difereixen en quin graf
s'usa per validar (selecció de model) i per avaluar:

| Escenari | Validació | Avaluació | Interpretació |
|----------|-----------|-----------|---------------|
| **S1** | perturbat | perturbat | Privadesa total (el graf real no s'usa mai). |
| **S2** | perturbat | real | Desenvolupament privat, inferència sobre el graf real (cas més habitual). |
| **S3** | real | real | Selecció de model amb el graf real (cota superior). |

---

## Sortides

Cada execució genera una carpeta amb marca de temps que conté:

- `summary.txt` — resum llegible dels resultats agregats (mitjana ± desv.).
- `metadata.txt` — configuració completa de l'experiment.
- `performance.txt` / `performance.csv` — temps, RAM i overhead de privadesa.
- `detailed_results/` — CSV per execució (utilitat).
- `wilcoxon_rr_vs_laplace_*.csv` — tests estadístics aparellats (utilitat).

---

## Reproductibilitat

Vegeu [REPRODUCIBILITAT.md](REPRODUCIBILITAT.md). En resum: llavor global 42,
llavors per execució (43, 44, …), partició 60/20/20 fixa i repartiment d'ε
centralitzat a `dp_mechanisms/config.py` garanteixen resultats idèntics a la
mateixa màquina i entorn.

> **Nota sobre les figures.** Les figures de l'informe (corbes d'utilitat,
> AUC de l'atac, mapa privadesa-utilitat, etc.) es generen amb
> `generate_figures.py` a partir dels CSV de `results/` i es desen a
> `figures_generated/`. Els valors numèrics complets que les sustenten són
> als fitxers `summary.txt` i `detailed_results/`.

---

## Citació i llicència

Codi distribuït sota llicència **MIT** (vegeu [LICENSE](LICENSE)). Per citar
aquest treball, vegeu [CITATION.cff](CITATION.cff).