# Audit ciblé — stubs, placeholders, absences de liaison, code mort (RSI v0.10.0)

Audit demandé : recherche minutieuse de **stubs, TODOs, placeholders, absences
de liaison (code écrit mais jamais branché), code mort, valeurs codées en dur
faisant semblant de calculer, simulations déguisées en implémentations réelles**.

Périmètre : lecture intégrale des 55 fichiers source (`src/`, `src/bin/`),
exemples et tests. Vérifications compilateur : `cargo check`/`clippy` en
features par défaut **et** avec toutes les features (dont les dépendances git
`forge`/`octasoma`/`ccos`/`scirust`, qui compilent proprement) → **0 warning**.
Chaque item « mort » a été confirmé par `grep` (zéro appelant dans tout le
dépôt, bins/examples/tests inclus) avant d'être listé.

> Un `AUDIT.md` antérieur (v0.10.0, 2026-06-22) existe déjà ; il est
> partiellement **obsolète** (ne couvre pas `dgm`/`llm`/`wasm_domain`/`addons`/
> `criticality`/`omega_tasks`, et décrit un `vendor/scirust-rsi/` remplacé
> depuis par une dépendance git). Le présent document ne le remplace pas.

## Constat global

Aucun marqueur littéral (`TODO`/`FIXME`/`unimplemented!`/`todo!`/`placeholder`)
dans le code Rust. Les implémentations « maison » sont **réelles et complètes**
(SHA-256 FIPS validé NIST, parseur JSON récursif avec surrogates UTF-16,
xoshiro256\*\*, journal d'audit hash-chaîné, sonde matérielle lisant vraiment
`/proc` + `nvidia-smi`, sep-CMA-ES, moteur DGM lançant de **vrais**
`cargo build`+`test` en sandbox isolée). Les parties simulées (world-model,
objectifs synthétiques) sont **honnêtement documentées** et non autoritaires.
Les problèmes ci-dessous sont sémantiques (items `pub` non appelés — invisibles
au lint `dead_code` — et fonctionnalités écrites mais non câblées).

---

## Majeur

- **`criticality.rs:209` — garde-fou de sûreté inerte.** `RiskConfig::risk_delta`
  (doc : « hausse de Risk_global tolérée par pas (δ) ») n'est **lu nulle part**.
  Ses trois voisins de `RiskConfig` sont tous câblés dans `agent.rs`
  (`rpn_max`/`active_response` l.326, `kappa` l.442) ; celui-ci ne borne rien.
  Un utilisateur qui règle ce paramètre croit limiter la croissance du risque
  par pas — la garantie affichée n'est pas tenue.

## Moyen-élevé (bug latent sur un chemin livré)

- **`llm.rs:50-52` — backend Claude silencieusement dégradé.** `ClaudeClient`
  n'override pas `complete_raw` : il hérite du défaut
  `Ok(self.propose(prompt,1)?.join("\n"))`, or `ClaudeClient::propose`
  (l.858-862) supprime lignes vides **et** indentation
  (`.lines().map(trim).filter(!empty)`). Le moteur DGM appelle justement
  `complete_raw` « pour matcher le fichier au caractère près » (`dgm.rs:1129`),
  et `OllamaClient` prend soin de l'override (l.748-755) — Claude non.
  Conséquence : via `bin/rsi_dgm.rs:376` (`ClaudeClient::with_ureq`, feature
  `llm-claude-ureq`), les blocs `FIND` multi-lignes de DGM ne matchent plus.
  Chemin Ollama correct, chemin Claude dégradé.

## Moyen (absence de liaison)

- **`schedule.rs:51-77` — `MetaMeta::adapt` jamais invoquée.** La « boucle
  méta-méta » (révise les cadences selon la tendance) est implémentée et
  unit-testée mais aucun binaire, ni `RSIAgent::step`, ni `loop_ctrl::run_until*`
  ne l'appelle. `rg "\.adapt\("` → seulement ses propres tests. Les champs
  `min_meta`/`max_meta` ne pilotent rien. (`LoopSchedule`, lui, est câblé —
  `rsi_loopbench.rs:23`.)
- **`llm.rs:180` — `ascend_llm` + garde-fous `LlmGuard` non branchés.** Tout
  l'appareillage (budget d'appels, `max_wall_clock`, anti-overfitting, patience)
  n'est utilisé qu'en `#[cfg(test)]` avec `MockLlmClient`. Les chemins produit
  (DGM, commande MCP `propose` via `api.rs:879`) réimplémentent l'adoption
  élitiste ailleurs → ces garde-fous ne protègent aucune exécution réelle.
- **`bin/rsi_dgm.rs:237-240` — `--revise N` inerte sans `--prescreen-model`.**
  Le flag est affecté à `config.simulated_revisions` inconditionnellement, mais
  la boucle (`dgm.rs:1519`) ne se déclenche que si un prédicteur est attaché
  (bloc `if let Some(sim_model)=…` l.247). `rsi-dgm --revise 5` seul → 0
  révision, silencieusement, sans avertissement.

## Mineur — code mort (`pub`, zéro appelant, confirmé)

| Item | Emplacement |
|---|---|
| `state::capability_levels` | `state.rs:130` |
| `state::COMPONENTS` (const) | `state.rs:18` |
| `Matrix::identity` (test-only) | `linalg.rs:113` |
| `omega_tasks::standard_bottleneck` | `omega_tasks.rs:121` |
| `StaticKnowledge` (struct, ré-exportée) | `knowledge.rs:272` |
| `agent::with_substrate_interval` (builder) | `agent.rs:183` |
| `llm::with_base_url` / `with_max_tokens` (builders) | `llm.rs:807` / `803` |
| `knowledge::from_dir` / `with_args` | `knowledge.rs:43` / `154` |
| `rng::from_state` | `rng.rs:34` |
| `checkpoint::save` (I/O fichier ; le round-trip JSON, lui, est câblé) | `checkpoint.rs:151` |
| `with_tolerance` (×2) | `scirust_bridge.rs:71`, `synthesis.rs:404` |

## Mineur — divers

- **`llm.rs:737` & `:872` — paramètre `k` ignoré.** `OllamaClient::propose` et
  `ClaudeClient::propose` ignorent `k` (nombre de propositions), alors que le
  contrat du trait (l.42) le suppose honoré ; seul `MockLlmClient` le respecte.
  Le levier `k`/`guard.k` n'a aucun effet sur les backends réseau.
- **`criticality.rs:154/163` — redondance.** Occurrence `MEMORY_POISON` codée en
  dur `0.05`, puis `assess` réapplique `occ.max(self.memory_base)` avec
  `memory_base = 0.05` → `0.05.max(0.05)` strictement redondant.
- **`bin/rsi_mcp.rs` — outil listé mais toujours en erreur.** `rsi_dgm_start`
  est annoncé dans `tools/list` sans condition, mais `rsi-mcp` n'a pas de
  `required-features` : en build par défaut, la variante
  `#[cfg(not(feature="llm-ollama"))]` renvoie toujours « outil indisponible ».
- **`bin/rsi_connect.rs:123` — config morte.** `RSI_DEFAULT_OPTIMIZER=random`
  est injectée dans le descripteur MCP mais **jamais lue** (`rg` → 1 ligne).
- **`bin/rsi_scholar.rs:105` — `--paper-model` ignoré sans `--paper-llm`.**
- **`main.rs:57-60` — fallback silencieux.** Un optimiseur inconnu retombe sur
  `RSIAgent::demo` (random) sans erreur, alors que `api.rs:1079` renvoie une
  erreur explicite pour la même entrée.
- **`bin/rsi_dgm.rs` — flags cachés.** `--revise`/`--prescreen-num-predict`
  fonctionnels mais absents de `usage()` et du doc-comment d'en-tête.

## Info — simulations *déclarées* (non déguisées, listées pour traçabilité)

- `tuning.rs:105-118` — objectif gaussien synthétique à optimum caché codé en
  dur. Documenté comme stand-in, **mais** `describe()` (exposée au client MCP)
  ne restitue pas ce caractère synthétique → un utilisateur pourrait le prendre
  pour une optimisation réelle.
- `prompt.rs:73-93` — `quality` = comptage de cues à poids codés en dur.
  Explicitement étiqueté « stand-in synthétique, pas une vraie métrique ».
- `wasm_domain.rs:116` — `WasmSynthesis::refine` renvoie le candidat inchangé
  (documenté : seul le chemin LLM a un intérêt pour muter du WAT).
- `agent.rs`/`meta.rs` — le vocabulaire « auto-réécriture du logiciel O » /
  « auto-modification » désigne une perturbation d'un vecteur numérique (modèle
  de substrat), **pas** du vrai code. Track modélisé, à ne pas confondre avec
  l'auto-amélioration empirique réelle de `dgm.rs`.

---

*Aucune modification de code n'a été apportée par cet audit — document
d'analyse uniquement.*
