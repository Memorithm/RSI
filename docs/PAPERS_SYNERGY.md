# Synergie RSI × PAPERS-AGENT — étude

**Question posée** : que gagne-t-on à connecter/fusionner PAPERS-AGENT (moteur
d'analyse de papiers scientifiques + évolution de code) dans RSI, pour
« évoluer et apprendre à partir des papers » ?

**Réponse courte** : la synergie est réelle et forte — PAPERS fournit
exactement les deux entrées qui manquent à RSI (des **connaissances réelles**
pour la composante D, et des **objectifs directifs** pour la boucle DGM) —
mais la bonne fusion est **fonctionnelle** (sous-processus + orchestrateur),
**pas** une fusion de crates. Détail ci-dessous.

---

## 1. Ce que chaque système est

| | RSI | PAPERS-AGENT (papers_core) |
|---|---|---|
| Cœur | Modèle mathématique (surface SI, capacités D/M/R/A/C/V, substrat P_eff) + boucle DGM **empirique** sur dépôt réel | Pipeline papiers : extraction (PDF/arXiv/URL), parsing sémantique (sections, équations), embeddings + store vectoriel HNSW, analyse LLM multi-passes |
| Boucle d'évolution | `dgm.rs` : patch → copie isolée → **cargo build+test+bench** → élitisme + anti-bruit → promotion humaine. Éprouvée sur Jetson (5 améliorations promues, cf. `P1_DESIGN_SPIKE.md`) | `evolution/` : Researcher→Engineer→Analyzer, candidats Rust **libres** évalués en sandbox WASM + score structurel, échantillonnage UCB1/greedy |
| Dépendances | **Cœur std-only, zéro dépendance** ; backends réels derrière features | tokio, reqwest, wasmtime, **ORT/CUDA**, hnsw, scirust en `path = "/tmp/scirust/…"` |
| Sûreté | Gate autoritaire, DRY-RUN par défaut, le LLM ne contrôle rien | Sandbox WASM pour l'exécution des candidats |

**Recouvrement** : les deux ont une boucle d'évolution — mais de niches
différentes. PAPERS évolue du code *neuf* à partir d'idées (exploratoire, bon
marché, sandbox) ; RSI évolue un *dépôt existant* (empirique, coûteux, gate
cargo, promotion). Elles se composent en entonnoir plutôt qu'elles ne se
concurrencent.

## 2. Les trois canaux de synergie

### 2.1 Papers → D (connaissances) — **déjà câblé côté RSI**

`knowledge::PapersKnowledge` appelle `papers extract <source>` en
**sous-processus borné** (30 s / 8 Mio, dégradation propre si absent), extrait
les concepts et fait monter la composante **D** de l'agent — qui nourrit
`L(D)` dans la dynamique et donc SI. C'était conçu pour PAPERS dès §2bis ;
il ne manquait que le binaire. Avec PAPERS-AGENT compilé :

```bash
RSI_PAPERS_BIN=/chemin/papers   # puis RSIAgent::with_knowledge(PapersKnowledge…)
```

→ **Gain immédiat, zéro code** : l'agent apprend de vrais papiers au lieu
d'un corpus jouet.

### 2.2 Papers → objectifs DGM directifs — **le canal le plus puissant**

Leçon centrale de la campagne Jetson : un **objectif directif** (« déroule la
boucle de rondes par 8 ») débloque en un step ce qu'un objectif vague
(« accélère ») n'obtient pas en seize. Or les objectifs directifs, c'est
exactement ce qu'un papier contient : des techniques nommées.

Pipeline cible (« READ → PROPOSE → PROVE → KEEP ») :

```
papers analyze -s <arXiv/PDF> -o analysis.json      (techniques extraites)
      │
      ▼  conversion technique → objectif directif + fichiers --allow
rsi-dgm . --goal "<technique appliquée à <cible>>" --bench … --min-gain 0.05
      │
      ▼  le gate empirique ne garde que ce qui est PROUVÉ
promotion humaine après revue (doctrine inchangée)
```

C'est la définition opérationnelle d'« apprendre de la littérature pour
s'améliorer » : le corpus de papiers devient le **générateur d'idées**, le
gate cargo+bench reste le **filtre de vérité**. Aucun des deux ne peut mentir
à l'autre.

### 2.3 Papers → contexte du proposeur (RAG d'auto-amélioration)

`papers search -q "<sujet>" -k 3` (store vectoriel) → injecter l'extrait
pertinent dans le prompt DGM à côté du code. Le modèle local voit *la
description de la technique* et *le code à transformer* — au lieu de deviner.
Extension naturelle de `build_prompt` (un champ « contexte » optionnel).

### 2.4 (bonus) Entonnoir à deux étages

La boucle WASM de PAPERS pré-crible des idées **à bas coût** (pas de cargo
build) ; seuls les survivants passent au DGM réel. Utile plus tard si le
volume d'idées dépasse le budget de steps DGM (~1-3 min/step).

## 3. Fusionner ? — recommandation

**Non à la fusion de crates, oui à la fusion fonctionnelle.** Raisons :

1. **Doctrine RSI** : cœur std-only, zéro dépendance — c'est ce qui rend le
   gate digne de confiance et le build reproductible partout (dont CI).
   papers_core tire tokio/reqwest/wasmtime/**ORT-CUDA**/scirust ; en l'état
   son Cargo.toml référence même scirust en `path = "/tmp/scirust/…"` (non
   consommable en git-dep, comme soul-rsi l'était).
2. **Précédents éprouvés dans ce dépôt** : les intégrations lourdes passent
   par sous-processus borné (PapersKnowledge), feature optionnelle git
   (forge/octasoma/ccos/scirust), ou port std-only du noyau utile
   (soul-rsi → `dgm.rs`). Les trois patrons s'appliquent ici :
   - **sous-processus** : `papers extract/analyze/search` (canaux 2.1-2.3) ;
   - **orchestrateur mince** : un binaire `rsi-scholar` std-only qui enchaîne
     papers → objectifs → runs DGM → rapport (canal 2.2) ;
   - **port éventuel** : `paper_parser` (regex/sections, léger) si on veut un
     jour extraire sans binaire externe — à décider sur besoin réel.
3. **Couplage sain** : PAPERS évolue vite (V2, GPU, ONNX) ; RSI reste stable.
   Un lien par CLI+JSON survit aux refontes des deux côtés ; un lien par
   crate casserait à chaque bump scirust (cf. l'épisode CI forge).

## 4. Plan de construction proposé (ordre de valeur décroissante)

1. **`rsi-scholar`** (nouveau binaire std-only) : `rsi-scholar <ws>
   --paper <source> --allow <fichiers> --bench <…>` — appelle `papers analyze
   --no-llm -o analysis.json`, extrait les techniques (JSON), génère N
   objectifs directifs, lance la boucle DGM par objectif (dry-run), produit un
   rapport des ACCEPTÉ. Garde-fous DGM inchangés.
2. **Contexte RAG dans le prompt DGM** : champ optionnel alimenté par
   `papers search` (sous-processus, dégradation propre).
3. **Démo D réelle** : exemple `examples/knowledge_papers.rs` branchant
   `PapersKnowledge` sur un agent et traçant la montée de D/SI.
4. (plus tard) entonnoir WASM→DGM, port de `paper_parser`.

**Prérequis côté Jetson** : compiler papers_core (`cargo build --release`)
— il faut ses crates scirust sous `/tmp/scirust` ou corriger les `path` du
Cargo.toml ; le binaire `papers` doit être sur le PATH ou `RSI_PAPERS_BIN`.

## État — premier run complet (Jetson Thor)

Le pipeline **READ → PROPOSE → PROVE → KEEP** a tourné de bout en bout sur
arXiv 2006.06762 (Ansor) : analyse LLM de PAPERS (qwen3-coder:30b) → résumé
et contributions substantiels → 1 objectif directif portant le pseudocode du
papier → 6 steps DGM (5 candidats tout-verts) → **∅ honnête** (cible
`kernels::matmul` déjà à ×7,3, et Ansor est un papier *système* — cadre de
recherche, pas de micro-technique patchable). Trois corrections amont nées
des runs : filtrage des placeholders de PAPERS, `--paper-llm`/`--paper-model`
(l'heuristique `--no-llm` n'extrait pas les techniques), lecture du champ
`pseudo_code` (même en mode LLM, `algorithms` reste le placeholder — biais
amont). Leçon : la qualité du résultat dépend du **couple papier↔cible** —
viser un papier algorithmique dont la technique parle le langage de la cible
(p. ex. simdjson → `Json::parse`), pas un papier de framework.
