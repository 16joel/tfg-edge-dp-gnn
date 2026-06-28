# Llista de canvis del treball

Aquest document recull els **canvis més rellevants** que ha experimentat el
projecte al llarg del seu desenvolupament, amb la seva **justificació**. S'ha
elaborat contrastant l'**Informe Inicial**, els **Informes de Seguiment I i II**
i l'**informe final**, i reflecteix tant decisions d'abast (objectius que es
van afegir o retirar) com correccions metodològiques i refactoritzacions del
codi.

> Llegenda de procedència: **[Inicial]** = Informe Inicial · **[Seg. I]** =
> Informe de Seguiment I · **[Seg. II]** = Informe de Seguiment II · **[Final]**
> = informe final i codi lliurat.

---

## 1. Abast i objectius

- **Reducció de tres a dos mecanismes DP (es retira DP-SGD).**
  [Inicial] preveia tres mecanismes (Randomized Response, Laplace i **DP-SGD**
  adaptat a edge-level). A [Seg. I] es va ajornar DP-SGD i a [Seg. II] es va
  retirar definitivament.
  *Motiu:* DP-SGD actua sobre el **procés d'entrenament** (pertorbant els
  gradients), no sobre la **topologia** del graf. Barrejar-lo amb RR i Laplace
  compararia mecanismes de granularitats conceptualment diferents i confondria
  la lectura del compromís privadesa–utilitat, que és la pregunta central del
  treball. Queda apuntat com a treball futur.

- **Es retira l'extensió a altres arquitectures (GraphSAGE, GAT).**
  Prevista com a ampliació opcional [Inicial], ajornada [Seg. I] i retirada
  [Seg. II].
  *Motiu:* la GCN és l'arquitectura canònica del *message passing* i és la que
  millor aïlla l'efecte de la pertorbació topològica; GraphSAGE (mostreig de
  veïns) i GAT (atenció) hi introduirien factors addicionals que
  emmascararien l'efecte estudiat. Queda com a línia futura.

- **Es retira l'anàlisi estructural topològica detallada (OE4 original).**
  [Inicial] plantejava mesurar grau mitjà, distribució de graus, components
  connexes i coeficient de *clustering*. A [Seg. II] constava com a parcial. A
  [Final] els objectius es **reformulen de 6 a 5 OE** i l'anàlisi estructural
  detallada se substitueix per la **preservació de densitat** com a propietat
  estructural clau dels mecanismes; la resta de mètriques no s'hi inclou.

- **Canvi de tasca: de *link prediction* a *node classification*.**
  [Inicial]/[Seg. I] descrivien l'avaluació com a predicció d'enllaços amb
  mètriques AUC/AP; ja a [Seg. I] s'anota que es passarà a **predicció
  d'etiqueta per node**. [Final] consolida la classificació de nodes amb
  **Accuracy i F1 ponderada**.
  *Motiu:* la classificació de nodes mesura de manera més directa l'impacte de
  la pertorbació estructural sobre el *message passing*.

- **Reenfocament del títol i la pregunta de recerca.** D'"Anàlisi estructural i
  impacte en el *message passing*" [Inicial] a "**compromís entre privadesa i
  utilitat**" [Final], centrant el treball en el *trade-off* i la protecció
  efectiva.

## 2. Mecanismes Edge-DP

- **Repartiment del pressupost ε: de 50/50 a 1/99 (estructura/densitat).**
  [Seg. II] usava una divisió equitativa **ε/2 + ε/2** (50% densitat, 50%
  estructura). [Final] adopta **1% densitat / 99% estructura** després d'una
  exploració empírica (es van assajar repartiments com 50/50, 20/80 i 1/99).
  *Motiu:* la consulta de densitat té una sensibilitat minúscula
  (1/C(n,2) ≈ 2,7·10⁻⁷), de manera que una fracció mínima del pressupost ja
  dona una estimació de densitat prou precisa; dedicar el 99% restant a la
  perturbació estructural (sensibilitat 1, molt més exigent) maximitza el
  senyal topològic preservat per a un mateix ε.

- **Centralització del repartiment en una font única de veritat.**
  Es crea `dp_mechanisms/config.py` com a **únic lloc** on es defineix el
  repartiment d'ε.
  *Motiu:* durant el desenvolupament es va detectar que diferents punts del codi
  declaraven repartiments incoherents (el codi, els *prints*, els *docstrings* i
  els resums desats no coincidien). Centralitzar-ho garanteix que el que
  s'executa, el que s'imprimeix i el que es desa no puguin divergir.

- **Calibració contra la densitat privada `d̃` en lloc del recompte real
  d'arestes.** Correcció metodològica important: les probabilitats de RR
  (p₀, p₁) i el llindar del mecanisme de Laplace es calibren contra
  l'estimació de densitat **privada** `d̃` (quantitat DP, pressupost ε₁), no
  contra `|E|` real.
  *Motiu:* usar `|E|` real per calibrar trencaria la garantia de privadesa
  diferencial; calibrar contra `d̃` fa que el pressupost ε₁ es consumeixi
  efectivament i que els paràmetres no depenguin de cap valor sensible exacte.

- **Mecanisme de Laplace: de llindar adaptatiu per percentil a selecció top-k.**
  [Seg. II] binaritzava la matriu sorollosa amb un **llindar al percentil
  (1−d̃)·100**. [Final] conserva les **k entrades amb el valor sorollós més
  alt**, amb `k = round(d̃·C(n,2))`.
  *Motiu:* la selecció top-k garanteix que la densitat final coincideixi
  exactament amb l'objectiu privat `d̃` (en lloc de dependre de la cua de la
  distribució del soroll) i és igualment vàlida des del punt de vista de DP
  (postprocessament de la sortida de Laplace i de `d̃`).

- **Randomized Response asimètrica per a grafs dispersos** (es manté des de
  [Seg. I]/[Seg. II]). La formulació simètrica clàssica generaria centenars de
  milers de falses arestes en un graf dispers com Cora; la variant asimètrica
  (p₀, p₁) preserva el nombre esperat d'arestes i manté la garantia ε-edge-DP.

## 3. Disseny experimental

- **Introducció dels tres escenaris de desplegament S1/S2/S3.** [Final].
  És l'**aportació metodològica central** i no apareix als informes de
  seguiment. Tots tres entrenen sobre el graf perturbat; difereixen en quin
  graf (perturbat o real) s'usa per validar i avaluar. Permet aïllar el cost de
  *aprendre* sobre soroll del cost d'*avaluar* sobre soroll. S'implementen amb
  un únic bucle d'entrenament i dos rastrejadors d'*early stopping* (un terç del
  cost de tres entrenaments separats).

- **Abast de la perturbació: de "només arestes d'entrenament" a "tot el graf".**
  [Seg. II] aplicava la DP només sobre les arestes d'entrenament
  (`apply_dp_to_train_only`). [Final] perturba el **conjunt complet d'arestes**
  (`apply_dp_to_graph`).
  *Motiu:* publicar un únic graf privat sencer és coherent amb el model de
  *release* d'edge-DP; el que distingeix els escenaris és quin graf s'usa per
  validar/avaluar, no quines arestes es perturben.

- **Nombre d'execucions: de 5/3 a 10/10.** [Seg. II] usava **5 execucions**
  (utilitat) i **3** (atac). [Final] passa a **10 execucions** en tots dos
  casos.
  *Motiu:* amb 10 execucions el test de Wilcoxon aparellat té prou potència (el
  p-valor mínim assolible baixa a 0,0078), de manera que l'absència de
  significança en privadesa forta es pot interpretar com a equivalència real i
  no com a manca de potència.

- **Escombrat d'ε: de {1,…,20} enters a {0,1; 1; 2; …; 15} (+ 0,01).**
  [Seg. II] explorava ε ∈ {1,…,20}. [Final] usa ε ∈ {0,1; 1; 2; …; 15} per a la
  utilitat i {0,01; 0,1; 1; …; 15} per a l'atac, amb el punt extrem ε = 0,01
  discutit a les limitacions.
  *Motiu:* concentrar la resolució a la regió on passa la transició
  privadesa–utilitat i caracteritzar el règim de privadesa extrema.

- **Generalització multi-dataset: de només Cora a Cora + Citeseer + Pubmed.**
  [Inicial]–[Seg. II] treballaven només amb Cora (prevista com a possible
  ampliació "si el temps ho permet"). [Final] valida les conclusions sobre tres
  conjunts. Per a l'atac, **Pubmed se submostreja** amb una bola BFS connexa de
  ~3000 nodes, perquè l'atac complet seria computacionalment inviable.

## 4. Atac d'inferència d'arestes

- **Reformulació de supervisat a NO supervisat (basat en *scores*).**
  Canvi de fons. [Seg. II] entrenava **dos classificadors supervisats**
  (Logistic Regression i Random Forest) sobre característiques (heurístiques +
  operadors d'embeddings Hadamard/L1/concatenació, dim 4d=256) amb separació
  train/test i `StandardScaler`. [Final] **no entrena cap classificador**: cada
  puntuació de versemblança d'aresta (6 *scores*) és un **atac independent**.
  *Motiu:* entrenar un classificador supervisat exigeix conèixer la
  veritat-terra de quines parelles són arestes —exactament la informació que el
  model d'amenaça de **caixa negra sense cap aresta real** nega a l'adversari.
  La formulació no supervisada fa el model d'amenaça **consistent** i les
  mètriques reportades són una **cota inferior** de la fuga (un adversari amb
  accés a consultes al model podria fer-ho millor).

- **Conjunt de puntuacions.** [Seg. II] usava 6 heurístiques estructurals
  (incloent-hi **diferència de graus**) + embeddings combinats per a un
  classificador. [Final] reporta 5 heurístiques (Common Neighbors, Jaccard,
  Adamic-Adar, Resource Allocation, Preferential Attachment) + **similitud
  cosinus d'embeddings**, cadascuna com a atac independent.

- **Mostreig de negatius: de 1:1 amb reemplaçament a 1:100 sense reemplaçament
  i deduplicat.** [Seg. II] mostrejava 1:1; [Final] usa **1:100** (prevalença
  ≈ 0,0099), **sense reemplaçament i deduplicat**.
  *Motiu:* (i) un conjunt 1:1 sobreestimaria la capacitat de l'adversari en un
  escenari artificialment equilibrat, lluny del desbalanceig real d'un graf
  dispers; (ii) la versió anterior mostrejava amb reemplaçament i sense
  deduplicar, cosa que generava desenes de milers de parelles repetides al
  conjunt d'avaluació i distorsionava l'AUC i l'AP. L'AUC-ROC, invariant a la
  prevalença, passa a ser la mètrica principal.

- **AUC-ROC com a mètrica principal** (invariant a la prevalença), amb l'AP
  sempre interpretada contra la prevalença del conjunt.

## 5. Infraestructura i codi

- **`get_embeddings` integrat a la classe GCN** (abans injectat dinàmicament al
  mòdul d'atac via *monkey-patching*) [Seg. II → Final].
  *Motiu:* el *monkey-patch* desapareixia silenciosament si la classe es
  reimportava; integrar-lo a `models/gcn.py` elimina aquesta fragilitat.

- **Eliminació de `baseline.py`; el *baseline* s'integra a `main.py`.**
  L'estructura de [Seg. II] tenia un mòdul `baseline.py` independent. A [Final]
  el *baseline* (sense DP) forma part del pipeline de `main.py`.

- **Nous mòduls de suport a [Final]:**
  - `run_experiments.py` — orquestrador multi-dataset que encadena `main.py` i
    els tests estadístics.
  - `utils/statistical_tests.py` — **test de Wilcoxon aparellat** RR vs Laplace
    (amb mida d'efecte i avís de potència insuficient).
  - `utils/perf_monitor.py` — mesura de **temps, RAM i overhead de privadesa**
    (suport a la taula de cost computacional de l'informe).

- **Parametrització per línia d'ordres (`argparse`).** `main.py`,
  `run_experiments.py` i `inference_attack.py` exposen arguments (`--dataset`,
  `--epsilons`, `--runs`, …) en lloc d'editar el codi a mà.

- **`main_binary.py` (One-vs-Rest) queda com a variant de llegat**, no usada per
  als resultats principals; es manté com a anàlisi complementària i comparteix
  mecanismes i repartiment de pressupost amb `main.py`.

- **Persistència de resultats enriquida:** a més de `summary.txt`,
  `metadata.txt` i `detailed_results/`, s'afegeixen `performance.txt/.csv`
  (cost) i els CSV dels tests de Wilcoxon.

## 6. Resum cronològic

| Moment | Estat del treball |
|--------|-------------------|
| **Informe Inicial** | 6 objectius (OE1–OE6); 3 mecanismes previstos (RR, Laplace, DP-SGD); ampliació a GraphSAGE/GAT; tasca de *link prediction*; només Cora. |
| **Seguiment I** | OE1 i OE2 assolits; RR i Laplace implementats; DP-SGD i extensió d'arquitectures ajornats; anunci del pas a classificació de nodes. |
| **Seguiment II** | RR i Laplace consolidats (ε/2+ε/2); atac **supervisat** (LogReg + RF, 1:1); 5/3 execucions; ε ∈ {1,…,20}; només Cora; DP-SGD i GraphSAGE/GAT retirats definitivament. |
| **Informe final** | Repartiment **1/99**; calibració contra densitat privada; **escenaris S1/S2/S3**; atac **no supervisat** 1:100; **10 execucions**; ε ∈ {0,1;…;15}; **Cora + Citeseer + Pubmed**; tests de Wilcoxon i cost computacional. |