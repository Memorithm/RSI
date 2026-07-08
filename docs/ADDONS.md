# Addons RSI — contrat et catalogue

RSI est une **plateforme à addons** : le cœur (std-only, zéro dépendance,
gate empirique autoritaire) reste petit et digne de confiance ; les capacités
lourdes ou spécialisées se branchent comme **addons exclusifs** — c'est la
structure prévue pour la version commerciale (open-core : cœur stable,
addons à valeur ajoutée).

## Le contrat d'un addon

Un addon RSI = **un binaire externe** + **un adaptateur std-only** dans
`src/addons.rs`. Jamais une dépendance de crate. Quatre règles :

1. **Détection à l'exécution** (env dédiée, sinon PATH) — un addon n'est
   jamais *requis* : `PapersAddon::detect() -> Option<…>`.
2. **Sous-processus bornés** — timeout + plafond de sortie (8 Mio) : un addon
   bogué ou hostile ne bloque ni n'épuise le cœur (`addons::run_bounded`).
3. **Dégradation propre** — addon absent ⇒ fonctionnalité indisponible avec
   message clair, jamais de panique ni d'échec du cœur.
4. **Le gate reste autoritaire** — un addon *propose* (connaissances,
   objectifs, contexte…) ; l'évaluation empirique (cargo build+test+bench),
   l'élitisme anti-bruit et la promotion restent au moteur RSI. Un addon ne
   peut pas écrire l'arbre vivant.

Pourquoi pas des crates ? Les addons tirent ce qu'ils veulent (GPU/CUDA,
ORT, tokio, wasmtime…) sans contaminer le build du cœur ni sa CI — l'épisode
« forge » (une dépendance git réécrite = CI rouge) a montré le coût du
couplage par crate. Le lien CLI+JSON survit aux refontes des deux côtés.

`rsi::addons::registry()` recense les addons connus et leur disponibilité.

## Addon nᵒ 1 : papers-agent (PAPERS V2)

**Ce qu'il apporte** : apprendre de la littérature scientifique pour
s'améliorer (étude complète : `docs/PAPERS_SYNERGY.md`). Trois canaux :

| Canal | Mécanisme | Entrée de RSI |
|-------|-----------|---------------|
| Connaissances | `papers extract` → concepts | composante **D** (`knowledge::PapersKnowledge`) |
| **Objectifs DGM directifs** | `papers analyze` → techniques → objectifs | boucle **DGM** (`rsi-scholar`) |
| Contexte RAG | `papers search` | prompt du proposeur (`PapersAddon::search`) |

**Installation** (sur la machine d'exécution, p. ex. le Jetson) :

```bash
# PAPERS V2 (dépôt Memorithm/PAPERS-AGENT, crate papers_core)
cd PAPERS-AGENT/papers_core && cargo build --release
export RSI_PAPERS_BIN=$PWD/target/release/papers   # ou mettre sur le PATH
```

**La vitrine — `rsi-scholar`** (papier → objectifs → preuves) :

```bash
cargo build --release --features llm-ollama --bin rsi-scholar

rsi-scholar . \
  --paper 2401.00001 \                # PDF local, id arXiv ou URL
  --allow src/kernels.rs \
  --bench "run --release --example bench_kernel" \
  --max-goals 3 --steps 6 --min-gain 0.05
```

Déroulé : l'addon analyse le papier (`--no-llm`, heuristique déterministe),
RSI convertit chaque technique extraite en **objectif directif** (leçon de la
campagne Jetson : « déroule par 8 » débloque ce que « accélère » n'obtient
pas), lance une boucle DGM par objectif (LLM en connexion automatique), et
écrit un rapport Markdown (`.rsi_scholar/rapport.md`) listant les
améliorations **prouvées** — avec la commande `rsi-dgm --promote` à lancer
soi-même après revue du diff. **READ → PROPOSE → PROVE → KEEP** : le papier
propose, le gate dispose.

## Ajouter un addon

1. Écrire l'adaptateur dans `src/addons.rs` : `detect()`, méthodes métier via
   `run_bounded`, types de résultat parsés par `crate::json` (fonctions de
   parsing **pures et testées** sur fixtures — le schéma amont peut évoluer,
   les champs absents doivent dégrader en défauts, pas en erreurs).
2. L'enregistrer dans `registry()`.
3. Documenter ici : capacités, variable d'env du binaire, installation.
4. Si l'addon mérite une vitrine, un binaire `rsi-<nom>` (modèle :
   `src/bin/rsi_scholar.rs`).
