# Reproductibilitat dels experiments

Aquest document descriu els mecanismes que garanteixen que els experiments
del projecte (utilitat i atac d'inferència) siguin reproductibles, i com
verificar-ho.

## Resum

Executant el codi tal qual a la mateixa màquina i entorn, els resultats
numèrics són idèntics entre execucions. En màquines o versions de
PyTorch/CUDA diferents, poden aparèixer diferències en els darrers decimals
per l'aritmètica de coma flotant, sense afectar les conclusions.

---

## 1. Llavor global

Tant `main.py` (experiments d'utilitat) com `inference_attack.py` (atac)
fixen una llavor global al començament de `main()`:

```python
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

- `torch.manual_seed` / `np.random.seed`: fixen els generadors de PyTorch i NumPy.
- `torch.cuda.manual_seed_all`: fixa la llavor de CUDA (si s'usa GPU).
- `cudnn.deterministic = True` i `cudnn.benchmark = False`: forcen algorismes
  deterministes a GPU.

## 2. Llavors per execució (run)

Cada run reinicialitza les llavors de forma determinista a partir d'una llavor
base (`seed_base = 42`):

- A `main.py`, el baseline fixa `torch.manual_seed(seed_base + run)` i
  `np.random.seed(seed_base + run)` a cada run; els experiments amb DP passen
  `seed=seed_base + run` a `apply_dp_to_graph`, que fixa les llavors abans de
  perturbar el graf.
- La seqüència de llavors és, doncs, 43, 44, …, 52 per als 10 runs.

**Disseny aparellat.** La mateixa seqüència de llavors es reutilitza entre
valors d'epsilon i entre mecanismes (RR i Laplace). Això és intencionat: les
comparacions entre configuracions queden aparellades (menys variància), a
canvi d'estar correlacionades. Cal tenir-ho present en interpretar les barres
d'error.

## 3. Conjunt de dades i partició fixos

Els experiments s'avaluen sobre tres conjunts de Planetoid —**Cora**,
**Citeseer** i **Pubmed**— seleccionables amb `--dataset`:

```python
dataset = Planetoid(root=f'/tmp/{name}', name=name)   # name ∈ {Cora, Citeseer, Pubmed}
data = dataset[0]
data = create_custom_split(data, train_ratio=0.6, val_ratio=0.2,
                           test_ratio=0.2, seed=GLOBAL_SEED)
```

Cada dataset de Planetoid és fix, i `create_custom_split` aplica la mateixa
partició 60/20/20 amb llavor controlada a tots tres, de manera que
`train_mask`, `val_mask` i `test_mask` són idèntiques en totes les execucions.

## 4. Configuració dels experiments

| Paràmetre | Valor | On |
|-----------|-------|-----|
| Llavor global | 42 | `main.py`, `inference_attack.py` |
| Nombre de runs (utilitat) | 10 | `run_multiple_experiments` |
| Nombre de runs (atac) | 10 | `inference_attack.main()` |
| Èpoques màximes | 500 | `run_multiple_experiments` |
| Paciència (early stopping) | 20 | `run_multiple_experiments` |
| Dimensió oculta GCN | 64 | construcció de `GCN` |
| Taxa d'aprenentatge (Adam) | 0,01 | `utils/metrics.py` |
| Repartiment d'epsilon | 1% densitat / 99% estructura | `dp_mechanisms/config.py` |
| Mostreig de l'atac (neg:pos) | 100:1 (prevalença ≈ 0,0099) | `inference_attack.py` |
| Epsilons (utilitat, per defecte) | [0.01] | `main.main()` |
| Epsilons (atac, per defecte) | [0.01] | `inference_attack.main()` |

> Nota: l'escombrat d'epsilon és un paràmetre editable a `main()` de cada
> script; el valor per defecte del repositori (`[0.01]`) és només un exemple
> mínim per a una execució ràpida. Els experiments de l'informe utilitzen
> l'escombrat complet `[0.1, 1, 2, 3, …, 15]` (epsilon = 0,1 i els enters
> d'1 a 15).

## 5. Repartiment del pressupost de privadesa

El repartiment d'epsilon entre la consulta de densitat (epsilon1) i la
perturbació estructural (epsilon2) està definit en un únic lloc,
`dp_mechanisms/config.py`:

```python
EPSILON_DENSITY_FRACTION = 0.01      # epsilon1
EPSILON_STRUCTURE_FRACTION = 0.99    # epsilon2
```

Tots els mòduls (mecanismes, orquestradors, persistència de resultats)
llegeixen aquest valor d'aquí, de manera que el repartiment que s'executa,
el que s'imprimeix i el que es desa als resultats no poden divergir.

## 6. Mostreig de l'atac d'inferència

L'atac (`inference_attack.py`) avalua les puntuacions sobre un conjunt de
parelles de nodes amb un desbalanceig que reflecteix el d'un graf dispers
real. El paràmetre `neg_ratio = 100.0` fixa una proporció de 100 no-arestes
per cada aresta (1:100), cosa que correspon a una prevalença d'arestes
positives ≈ 0,0099. Aquest mostreig es fa amb llavor controlada
(`seed_base + run`), de manera que el conjunt de parelles avaluat és idèntic
entre execucions amb la mateixa llavor. La separació train/test del conjunt
de parelles es fa abans del càlcul de característiques, i com que l'atac és
no supervisat (no s'entrena cap classificador; cada puntuació ordena les
parelles per versemblança d'aresta), no hi ha cap pas d'ajust amb
aleatorietat addicional més enllà del mostreig.

## 7. Scripts disponibles

- `main.py`: experiments d'utilitat (classificació multiclasse, escenaris
  S1/S2/S3). És el que genera els resultats reportats a l'informe.
- `inference_attack.py`: atac d'inferència d'arestes (no supervisat).
- `main_binary.py`: variant One-vs-Rest que descompon la classificació de 7
  classes en 7 tasques binàries. No s'utilitza per als resultats principals
  de l'informe; es manté com a anàlisi complementària. Comparteix la
  configuració de reproductibilitat de `main.py` (mateixes llavors, split i
  mecanismes).

---

## Verificació

**Execució doble.** Executar dues vegades i comparar la sortida:

```bash
python main.py > run1.txt
python main.py > run2.txt
diff run1.txt run2.txt    # hauria de no mostrar diferències numèriques
```

(Les úniques diferències possibles són els noms de carpeta de resultats, que
porten marca de temps.)

**Comparació de CSV.** Els resultats detallats es desen a
`results/<dataset>_<timestamp>/detailed_results/`. Els valors de dues
execucions a la mateixa màquina han de coincidir.

---

## Excepcions

Es poden obtenir diferències en els darrers decimals si es canvia de màquina
(CPU/GPU diferents fan operacions de coma flotant en ordres diferents), de
versió de PyTorch/CUDA, o de sistema operatiu. Per a una reproductibilitat
exacta, cal documentar les versions (`torch.__version__`,
`torch_geometric.__version__`) i el maquinari, i fixar l'entorn.