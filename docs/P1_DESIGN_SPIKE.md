# P1.0 — Design Spike : brancher un LLM sur le moteur RSI

> **Statut** : document de décision (à valider avant toute ligne de code P1).
> **Pré-requis** : P0 close (bugs A–H corrigés, invariants property-testés).
> **Objet** : trancher les trois décisions qui conditionnent la viabilité de
> toute la phase P1, *avant* d'écrire `LlmRefineTask`, les outils MCP, ou les
> domaines de démonstration.

Ce spike ne livre pas de code. Il fixe les contrats et les choix par défaut.
Chaque décision est présentée avec ses options, une recommandation, et la
raison du choix. Les questions encore ouvertes sont listées en fin de document.

---

## 0. Rappel du contrat de sûreté (non négociable)

Tout ce qui suit doit préserver l'invariant fondamental :

> **Le LLM propose, le moteur dispose.** L'agent IA génère des *candidats* ;
> le moteur les évalue en sandbox et les adopte **élitistement** (strictement
> meilleur) ou les rejette, sous garde-fous bornés (`Guard`) et audit
> hash-chaîné. Le LLM ne contrôle jamais `max_iters`, `target`, `patience`,
> ni le `line search`.

Le contrat actuel (`src/ascent.rs`) modélise déjà ça :

```rust
pub trait RefineTask {
    type Cand: Clone;
    fn score(&self, cand: &Self::Cand) -> f64;          // ÉVALUATEUR (moteur)
    fn refine(&mut self, cand: &Self::Cand, iter: usize) -> Self::Cand; // GÉNÉRATEUR
}
pub fn ascend<T: RefineTask>(task: &mut T, init: T::Cand, guard: &Guard) -> (T::Cand, Report);
```

Aujourd'hui `refine` est un générateur **déterministe interne**. P1 consiste à
remplacer ce générateur par un LLM — **sans toucher** ni à `ascend` (la boucle
élitiste), ni à `Guard` (les bornes), ni à `score` côté confiance.

---

## 1. Décision A — Qu'est-ce que le LLM améliore réellement ?

### Le problème

Le projet d'origine modélise l'auto-amélioration de l'**état cognitif `S`**
(D,M,R,A,C,V) — une *simulation* de dynamique. La roadmap P1 bascule vers « un
LLM optimise des **artefacts externes** (prompts, configs, code) ». Ce sont deux
produits différents, et il faut choisir lequel on construit. Confondre les deux
mène à du marketing trompeur (« auto-amélioration récursive » pour ce qui est en
fait de l'optimisation pilotée par LLM).

### Options

| Option | Description | Pour | Contre |
|---|---|---|---|
| **A1 — RSI-de-S** | Le LLM propose des modifications de la stratégie ℳ / de l'état cognitif simulé | fidèle au modèle mathématique ; « récursif » au sens propre | le résultat n'est qu'une trajectoire dans une simulation ; aucune valeur externe livrée |
| **A2 — Optim. d'artefacts** | Le LLM propose des révisions d'un artefact réel (prompt/config/code), évalué sur une tâche réelle | valeur concrète et mesurable ; démontrable sur benchmark | ce n'est pas du « self »-improvement : le LLM améliore un objet, pas lui-même |
| **A3 — Hybride en couches** | A2 comme produit ; le moteur RSI (S, criticité, ε) sert de **superviseur de sûreté** de la boucle A2 | valeur réelle **+** garde-fous formels réutilisés ; honnête | plus complexe ; nécessite de mapper l'état de la boucle A2 sur des signaux de criticité |

### Recommandation : **A3 (hybride en couches)**

- **Couche produit (A2)** : l'unité de valeur est un **artefact** (`Cand` =
  prompt structuré, JSON de config, ou AST de programme). C'est ce qui se vend
  et se benchmarke.
- **Couche sûreté (réutilise le cœur)** : la boucle d'ascension est surveillée
  par les garde-fous RSI existants — non-régression (`is_monotone`), disjoncteur
  de criticité, observateur HITL, audit hash-chaîné. Le « S » devient l'**état
  de santé de la boucle d'optimisation**, pas une simulation cognitive abstraite.

### Conséquence sur le vocabulaire

Assumer publiquement le glissement : le produit est un
**« moteur d'auto-amélioration d'artefacts piloté par LLM, sous garde-fous
formels »**. Le terme « RSI » reste justifié pour la *boucle* (propose → évalue
→ garde si meilleur → répète), pas pour une auto-modification du LLM lui-même.
À écrire tel quel dans le README et `docs/SAFETY.md`.

---

## 2. Décision B — Budget d'inférence LLM

### Le problème

Chaque proposition = **une inférence LLM** (coût + latence). La boucle actuelle
peut faire des milliers de pas × dizaines de candidats. Sans budget explicite,
le produit est inutilisable (coût prohibitif, latence de plusieurs heures).
La roadmap n'en parle pas — c'est l'angle mort le plus dangereux pour la
viabilité.

### Chiffrage d'ordre de grandeur

Pour fixer les idées (hypothèses à valider, modèle Claude via API) :

- 1 proposition ≈ 1 appel ≈ ~1–3 k tokens d'entrée (incumbent + historique) +
  ~0,5–1 k tokens de sortie.
- Une boucle « naïve » de `max_iters = 200` avec `λ = 8` candidats/pas =
  **1 600 appels**. À ~2–5 s/appel séquentiel ⇒ **50 min – 2 h** par run, et un
  coût non négligeable par run.

C'est rédhibitoire en l'état. Il faut des garde-fous de **coût**, au même titre
que les garde-fous de sûreté.

### Décisions

1. **`Guard` budgétaire** — étendre `Guard` (ou un `CostGuard` parallèle) avec :
   - `max_llm_calls` (borne dure d'appels — terminaison garantie côté coût) ;
   - `max_tokens_total` (plafond de tokens cumulés) ;
   - `max_wall_clock` (timeout mur).
   Atteindre l'un de ces plafonds = `StopReason::BudgetExhausted` (nouveau
   variant). **Le LLM ne fixe jamais ces bornes** (cf. contrat §0).

2. **Batching des propositions** — un seul appel LLM renvoie **k candidats**
   (« propose k variantes diverses »), pas k appels. Divise les appels par k.

3. **Cache de candidats** — clé = hash du candidat ; ne jamais ré-évaluer (ni
   re-proposer) un candidat déjà vu. Réutiliser le `si_cache` de
   `ForgeMetaSearch` comme modèle. L'évaluation (`score`) est souvent plus chère
   que stockée.

4. **Génération hiérarchique** — le LLM ne propose qu'aux pas « utiles » : sur
   plateau (détecté par `convergence.rs::Trend`), on régénère ; en progression
   régulière, on laisse la mutation locale bon marché (l'`evo` 1+λ existant)
   prendre le relais. Le LLM est la ressource rare, cadencé par `schedule.rs`.

### Recommandation

Adopter les 4. **Le coût est un garde-fou de première classe** : un run doit
afficher son budget consommé (appels, tokens, coût estimé, temps) dans le
`Report` et l'audit. Cible produit : un run de démonstration **< 100 appels LLM**
et **< 5 min**.

---

## 3. Décision C — Intégrité de l'évaluateur (anti-Goodhart)

### Le problème

« Garde si meilleur » ne vaut que ce que vaut `score()`. Si le LLM optimise un
prompt sur un jeu d'éval fixe, il va **surajuster ce jeu** (Goodhart /
contamination train-test). Le « +X % » devient alors un artefact, pas un gain
réel. La roadmap mentionne le wireheading mais pas ce mode d'échec, qui est le
plus probable en pratique.

### Décisions

1. **Séparation train / held-out stricte** — `score()` (qui pilote l'adoption
   élitiste) évalue sur un **jeu d'entraînement** ; un **jeu held-out**, jamais
   vu par la boucle ni par le LLM, mesure la qualité *rapportée*. **Seul le
   held-out** apparaît dans le benchmark public.

2. **Rotation / ré-échantillonnage** — à chaque adoption (ou tous les N pas),
   ré-échantillonner le sous-ensemble d'éval d'entraînement (validation croisée
   roulante). Surajuster un point fixe devient inutile.

3. **Détection d'overfitting comme signal de criticité** — l'écart
   `score_train − score_heldout` qui se creuse est un mode AMDEC (analogue à
   l'anti-wireheading `software_eff_gap` existant). Au-delà d'un seuil :
   mitigation (régénérer, élargir l'éval) ou disjoncteur.

4. **`safety_check` par domaine** — au-delà de la fitness, chaque domaine
   déclare ses interdits (cf. `LlmRefineTask::safety_check` ci-dessous) : un
   candidat peut être *meilleur* en score et pourtant **rejeté** s'il viole une
   contrainte de sûreté (ex. prompt qui exfiltre, code qui touche le réseau).

### Recommandation

Adopter les 4. **La fitness n'est jamais l'unique critère d'adoption** :
adoption ⟺ (`score` strictement meilleur) **ET** (`safety_check` OK) **ET**
(pas de divergence train/held-out au-delà du seuil). C'est l'extension naturelle
de l'élitisme borné existant.

---

## 4. Contrat proposé : `LlmRefineTask`

Esquisse ancrée sur `RefineTask` réel. **Ne pas implémenter avant validation
de ce spike.**

```rust
/// Domaine auto-améliorable piloté par LLM. Étend `RefineTask` : le moteur
/// garde la main sur `score` (évaluateur) et `ascend` (boucle élitiste) ; le
/// LLM n'intervient que via `propose`, jamais sur les bornes.
pub trait LlmRefineTask: RefineTask {
    /// Vue prompt-friendly de l'incumbent (ce que le LLM « voit »).
    fn describe(&self, incumbent: &Self::Cand) -> String;

    /// Intègre la sortie brute du LLM en k candidats typés (batching, §2).
    /// Le LLM ne produit jamais de Cand directement exécutable : on parse,
    /// on valide, on rejette si malformé.
    fn parse_proposals(&self, llm_output: &str) -> Vec<Self::Cand>;

    /// Évaluation held-out (anti-Goodhart, §3) — NE pilote PAS l'adoption,
    /// sert uniquement au reporting et à la détection d'overfitting.
    fn score_heldout(&self, cand: &Self::Cand) -> f64;

    /// Contraintes de sûreté spécifiques au domaine (§3.4). Un candidat qui
    /// échoue est rejeté quel que soit son score.
    fn safety_check(&self, cand: &Self::Cand) -> Result<(), SafetyViolation>;
}
```

La boucle `ascend` est complétée par un pilote `ascend_llm` qui : (1) appelle le
LLM via MCP avec `describe(incumbent)`, (2) `parse_proposals`, (3) pour chaque
candidat : `safety_check` puis `score`, (4) adopte si strictement meilleur, (5)
journalise score_train, score_heldout, budget, rationale dans l'audit. **Tout le
reste de la sûreté (Guard, disjoncteur, HITL) est inchangé.**

---

## 5. Trois domaines de démonstration (couvre texte → config → code)

| Domaine | `Cand` | `score` (train) | Sandbox | Risque |
|---|---|---|---|---|
| **Prompts** | chaîne structurée | qualité sur sous-ensemble de tâches | aucune exéc (texte) | nul |
| **Configs d'outil** | JSON validé par schéma | métrique sur benchmark fixe | JSON, aucune exéc | nul |
| **Programmes numériques** | AST étendu, puis WASM | tests + perf | interpréteur / WASM fuel | confiné |

**Ordre recommandé** : Prompts d'abord (zéro risque sandbox, valide toute la
chaîne MCP↔LLM↔boucle), puis Configs, puis Code (qui exige le sandbox WASM de
P1.2 et concentre tout le risque d'exécution).

---

## 6. Décisions transverses déjà prises (rappel P0 / audit)

- **`std::simd` est nightly** : pour P2, utiliser `wide` ou `std::arch`, pas
  `portable_simd`. (Corrige une affirmation erronée de la roadmap.)
- **Déterminisme vs parallélisme** : ✅ **tranché — déterminisme préservé.**
  `MetaOptimizer` évalue désormais les candidats en parallèle
  (`std::thread::scope`, sans dépendance) mais : génération RNG séquentielle,
  évaluation pure, **argmax à ordre d'indice fixe**. Résultat **bit-exact**
  identique au séquentiel (testé : `parallel_eval_is_bit_exact_vs_sequential`),
  donc `same_seed ⇒ same_audit_head` tient. **`CmaEsMeta` aussi** : `cma.rs`
  pré-échantillonne la population (RNG séquentiel) puis évalue les fitness en
  parallèle par index (`eval_population`), mise à jour CMA bit-exacte
  (`optimize_is_deterministic`). Les deux optimiseurs sont donc parallélisés
  sans rien céder sur le déterminisme.
- **Invariant `‖ΔS‖≤λ`** : garanti seulement dans le domaine `[0,1]ⁿ`
  (non-expansivité de la projection). Le moteur y reste toujours ; à documenter
  comme précondition dans `docs/SAFETY.md`.

---

## 7. Critères d'acceptation de P1

1. Un LLM pilote la boucle via MCP (`rsi_incumbent` → `rsi_propose` →
   `rsi_evaluate`) sans jamais accéder aux bornes.
2. Les trois domaines tournent, avec amélioration mesurée **sur held-out**
   (pas train).
3. Budget respecté : run de démo < 100 appels LLM, < 5 min, budget affiché.
4. `safety_check` rejette au moins un candidat « meilleur mais interdit » dans
   un test dédié.
5. Audit : chaque adoption porte `score_train`, `score_heldout`, `rationale`,
   coût. Trajectoire rejouable.
6. Aucune régression des invariants P0 (property tests toujours verts).

---

## 8. Questions ouvertes — arbitrages

| # | Question | Décision |
|---|---|---|
| 2 | Transport LLM | ✅ **RSI client du LLM** — le moteur orchestre la boucle, appelle le LLM comme une API. Cohérent avec le contrat §0 (le LLM ne pilote rien). |
| 3 | WASM en P1 ou P2 ? | ✅ **P2** — P1 = Prompts + Configs (zéro exécution). Le sandbox WASM et le domaine « code » arrivent en P2. |
| 5 | Licence | ✅ **Double licence** — appliquée : `Cargo.toml` déclare désormais `PolyForm-Noncommercial-1.0.0 OR LicenseRef-Commercial` (crate principal + vendor), cohérent avec `LICENSING.md` / README. |
| 1 | Fournisseur / modèle LLM | ✅ **Ollama (local) par défaut, Claude sélectionnable** — voir §8.1. |
| 4 | Politique held-out | ✅ **Défaut adopté** (70/30 gelé, modifiable) — voir §8.2. |

### 8.1 Décision — fournisseur / modèle (question 1)

**Backend LLM interchangeable, modèle local Ollama par défaut.** Chaque
utilisateur choisit son backend en configuration ; aucune dépendance dure à un
fournisseur cloud.

```rust
/// Abstraction du producteur de propositions. Le moteur ne connaît que ça ;
/// il ignore quel modèle tourne derrière. Aucun appel réseau dans le cœur.
pub trait LlmClient {
    /// Rend `k` propositions (texte brut) pour un prompt donné, sous budget.
    fn propose(&self, prompt: &str, k: usize) -> Result<Vec<String>, LlmError>;
}
```

Backends prévus (sélection par `RsiConfig`, cf. P3.3) :

| Backend | Statut | Usage |
|---|---|---|
| **Ollama (local)** | **défaut** | HTTP local `http://localhost:11434` ; modèle au choix (`llama3.x`, `qwen2.5`, `codestral`…). Zéro coût API, données locales, hors-ligne. |
| **Claude (API)** | sélectionnable | pour qui veut la capacité maximale ; clé API requise. Modèle réglable (Opus/Sonnet/Haiku). |
| **Mock déterministe** | dev/test | `LlmClient` factice (propositions scriptées) pour tester `ascend_llm` **hors-ligne et de façon reproductible** — c'est la première brique à écrire en P1.1. |

**Implication budget (§2)** : avec Ollama local, le coût *monétaire* tombe à ~0 ;
le `Guard` budgétaire borne alors surtout les **appels** et le **wall-clock**
(latence d'inférence locale), pas les dollars. Le plafond `max_tokens` reste
utile pour la fenêtre de contexte. Avec le backend Claude, le plafond
coût/tokens redevient pertinent. Les deux régimes passent par le même
`Guard` budgétaire.

> Le cœur reste **std-only** : les backends (Ollama HTTP, Claude API) vivent
> derrière une **feature optionnelle** (`llm-ollama`, `llm-claude`) et n'entrent
> jamais dans le cœur ni dans le `Mock`.

### 8.2 Défaut proposé — held-out (question 4)

- **Split** : 70 % train / 30 % held-out par domaine, **gelé pour tout le run**,
  séparé par graine déterministe (reproductible).
- **Source** : le held-out vient du **même corpus** que le train mais en
  partition disjointe (`tasks.rs` fournit déjà `builtin()`/`extended()` —
  partitionner ces ensembles).
- **Rotation** : seul le **sous-ensemble train** effectivement évalué par
  `score()` tourne (ré-échantillonné tous les N adoptions) ; le held-out, lui,
  ne tourne **jamais** (sinon il fuit dans la boucle).
- **Mesure** : held-out évalué aux checkpoints + en fin de run uniquement.

> À confirmer : ratio 70/30, et si certains domaines exigent un held-out
> *externe* (issu d'un autre corpus) pour une garantie anti-contamination plus
> forte.

---

## 9. Synthèse des recommandations

| Décision | Choix recommandé |
|---|---|
| **A** Quoi améliorer | A3 — optim. d'artefacts (produit) **sous** garde-fous RSI (sûreté) ; assumer le vocabulaire |
| **B** Budget LLM | `Guard` budgétaire + batching k-candidats + cache + génération cadencée ; coût = garde-fou de 1ʳᵉ classe |
| **C** Évaluateur | held-out strict + rotation + overfitting comme criticité + `safety_check` ; adoption ⟺ meilleur **ET** sûr **ET** non-surajusté |

**Le fil rouge** : étendre les garde-fous existants (élitisme, criticité, audit)
au coût et à l'intégrité d'éval, plutôt que d'ajouter une boucle LLM à côté.
Le LLM est une *source de propositions sous contrainte*, pas un pilote.

---

## 10. Avancement

- ✅ **P1.1 — squelette mécanique** (`src/llm.rs`, std-only, testé hors-ligne) :
  - `LlmClient` (backend interchangeable) + `MockLlmClient` déterministe ;
  - `LlmRefineTask` (describe / parse_proposals / score_heldout / safety_check) ;
  - `LlmGuard` (bornes + budget appels/temps + garde-fou overfitting) ;
  - `ascend_llm` (élitisme strict : adoption ⟺ *sûr* ET *strictement meilleur*),
    avec `LlmStop::{MaxIters, Patience, Target, BudgetExhausted, OverfitGuard}`.
  - 5 tests : convergence pilotée par le mock, plafond de budget, blocage de
    candidat interdit par `safety_check`, propositions vides, déterminisme.
- ✅ **Backend Ollama local** (`OllamaClient`, feature `llm-ollama`) : client
  HTTP/1.1 minimal sur `std::net::TcpStream` + `crate::json`, **zéro
  dépendance**. `/api/generate` non-streamé sur `127.0.0.1:11434` (réglable).
  Parties pures (`build_request` / `parse_response`) testées hors-ligne
  (4 tests) ; l'appel réseau réel reste à valider sur une machine avec Ollama.
- ✅ **Premier domaine réel câblé sur le LLM** : `SymbolicSynthesis`
  (synthèse symbolique) implémente `LlmRefineTask`. Ajouts : parseur
  `Expr::parse` (texte → AST, précédence, garde de profondeur), held-out
  réservé (`from_target_split`, ~30 %) pour `score_heldout`, `safety_check`
  bornant la taille d'AST. Sandbox inchangé (AST interprété, jamais exécuté).
  Tests hors-ligne : round-trip parseur, infixe naturel, rejet d'entrées
  hostiles, **synthèse pilotée par un mock LLM** (converge sur `x²+1`),
  rejet d'un AST surdimensionné par `safety_check`.
- ✅ **P1.4 — outils MCP** (`rsi_refine_new`, `rsi_incumbent`, `rsi_evaluate`,
  `rsi_propose`) : exposent la boucle pour qu'un **LLM externe** la pilote via le
  serveur MCP, le serveur restant **autoritaire** (il parse, applique
  `safety_check`, score, n'adopte qu'élitistement — strictement meilleur ET
  sûr ; le LLM ne contrôle aucun garde-fou). Domaine : synthèse symbolique,
  cibles `quadratic|linear|cubic`, held-out réservé. Vérifié end-to-end en
  JSON-RPC sur stdin. Les deux topologies coexistent désormais : `ascend_llm`
  (RSI client du LLM, autonome) et MCP (RSI serveur, LLM client, interactif).
- ✅ **Backend Claude** (`ClaudeClient`, feature `llm-claude`) : API Anthropic
  Messages. `std` n'ayant pas de TLS, le **transport HTTPS est injecté** par
  l'hôte (trait `ClaudeTransport`) ; toute la logique (construction de requête,
  parsing réponse + erreurs API) est **std-only et testée hors-ligne** via un
  transport mock (5 tests). Le branchement d'une pile TLS réelle (p. ex.
  `ureq`/`rustls`) côté hôte reste à faire dans un environnement avec réseau.
- ✅ **2ᵉ domaine — configuration** (`ConfigTuning`, `src/tuning.rs`) :
  optimisation d'hyperparamètres (objet JSON, `Cand` ≠ `Expr`) contre un
  objectif lisse à optimum caché, held-out décalé, `safety_check` = validation
  de bornes (schéma). Démontre la **généralité** de `LlmRefineTask` (texte →
  **config**). Testé hors-ligne via mock (réglage convergent, rejet de config
  hors bornes, filtrage des JSON malformés). Utilisable via `ascend_llm`.
- ✅ **Multi-domaine via MCP** : `RefineSession` généralisée derrière un trait
  object-safe `RefineDomain` ; `rsi_refine_new` accepte `domain` (`synthesis` |
  `tuning`). Les outils `rsi_incumbent`/`rsi_evaluate`/`rsi_propose` pilotent les
  deux domaines de façon uniforme, le serveur restant autoritaire. Vérifié
  end-to-end (tuning : config adoptée, config hors-bornes rejetée).
- ✅ **3ᵉ domaine — prompts** (`PromptOpt`, `src/prompt.rs`) : optimisation de
  prompt (texte), objectif synthétique par cues (raisonnement/exemple/format),
  held-out décalé, `safety_check` rejetant longueur excessive **et marqueurs
  d'injection**. Exposé via MCP (`domain: "prompt"`). Vérifié end-to-end
  (injection rejetée, bon prompt adopté).
- ✅ **4ᵉ domaine — WASM (P2, exécution réelle)** (`WasmSynthesis`,
  `src/wasm_domain.rs`, feature `wasm`) : le LLM propose du WebAssembly textuel,
  **exécuté en bac à sable `wasmi`** (interpréteur pur-Rust, déterministe) avec
  **fuel** (terminaison) et **zéro import host** (linker vide ⇒ capability-free).
  `safety_check` rejette WAT invalide / imports / export `run` manquant ;
  exposé via MCP (`domain: "wasm"`). **Le spectre texte → config → code est
  désormais entièrement couvert.** 137 tests avec `--features wasm`.
- ✅ **Backend Claude TLS turnkey** (`UreqTransport`, feature `llm-claude-ureq`).
- ✅ **Vrai moteur scirust** (git-dep amont validé : 131 tests `--features scirust`).
- ✅ **Docs** : `docs/LLM_INTEGRATION.md`, `docs/SAFETY.md`, `docs/WEB_ENV.md`.
- ✅ **Observabilité** : commande `metrics` au **format Prometheus** (texte,
  scrapeable, **sans dépendance**) + événements `tracing` derrière la feature
  `observability` (no-op sans elle, via `src/obs.rs` ; instrumente adoption/rejet
  de propositions). Le cœur reste std-only.
- ✅ **SIMD** (feature `simd`, OFF par défaut) : `dot`/`mean` vectorisés via
  `wide` (sûr, stable). Réduction par 4 voies + combinaison à ordre fixe ⇒
  déterministe à l'intérieur d'un build SIMD ; relâche assumé de la repro
  bit-exacte *cross-build* (documenté dans `docs/SAFETY.md`). Gain modéré : le
  coût dominant de `si_global` est l'évaluation Φ/g par tâche (trait objects),
  non la réduction finale. 132 tests verts sous `--features simd`.
- ✅ **P4 — packaging (artefacts, sans publication)** : `Dockerfile` multi-étage
  **musl statique** (étage `rust:alpine` → image `scratch` ; ne build que le
  CŒUR std-only, donc **0 dépendance ⇒ build hors-ligne** ; `rsi-full` sauté
  automatiquement faute de required-features) + `.dockerignore` (contexte
  minimal). Build musl **validé localement** : binaires `static-pie / statically
  linked` (`ldd` ⇒ « statically linked »), `rsi-demo` s'exécute. `flake.nix`
  reproductible (`nix build` / `nix run` → `rsi-mcp`, `nix develop` outillé) :
  build offline du cœur ; `doCheck=false` car un test (`/bin/echo` absolu) n'est
  pas hermétique en bac à sable Nix scellé (les tests se jouent en `nix develop`/CI).
  Métadonnées de manifeste complétées (repository/homepage/documentation/keywords/
  categories) ⇒ l'avertissement `cargo package` disparaît.
- ⚠️ **Publiabilité crates.io** : `cargo publish --dry-run` **échoue par design** —
  les 4 deps git privées (`forge-core`, `octasoma`, `ccos`, `scirust-rsi`) n'ont
  pas de version crates.io, et crates.io exige une version pour **toute** dép
  (même optionnelle, la source git étant retirée à la publication). Le cœur n'est
  donc **pas publiable tel quel** ; il faudrait d'abord publier ces 4 crates amont
  (ou les sortir du manifeste publié). La distribution se fait donc par **source +
  Docker/Nix** (validés), pas par registre. Aucune publication effectuée.
- ✅ **Intégration de `soul-rsi` (boucle Darwin–Gödel / STOP)** : port natif et
  **std-only** dans `src/dgm.rs` (le clone amont est un crate de workspace
  SoulSystem — deps `workspace = true` + `path = "../soul-automodify"` — donc non
  consommable en git-dep ; le porter est la seule voie correcte). Couvre tout le
  contenu du dépôt : `Patch`/`Fitness` (barrière lexicographique compile ▸ tests
  ▸ score), `Variant`/`Archive` (population ouverte, sélection qualité×nouveauté),
  `WorkspaceSnapshot`/`promote_to_live` (copie isolée jetable, arbre vivant intact),
  `CargoEvaluator`/`ClosureEvaluator`, `LlmProposer` (enveloppe stricte +
  liste blanche), `DgmEngine` (Reflexion). **Réutilise** les primitives RSI
  (`rng::Rng`, `sha256`, `json`, `obs`) au lieu des deps externes de l'amont
  (serde/uuid/chrono/tempfile/thiserror/tracing/soul_automodify). **Adaptateur**
  `LlmCodeModel` ⇒ pilotable par les backends RSI (Ollama/Claude). **Améliorations
  vs amont** : IDs de variantes déterministes (hash de lignée, ≠ UUID aléatoire)
  ⇒ archive bit-exacte reproductible ; horloge logique `seq` (≠ horodatage mur) ;
  sous-processus `cargo` **bornés** (timeout + sortie plafonnée). 23 tests + 1
  doctest + exemple `dgm_selfimprove`. Sûreté documentée (`SAFETY.md` §5bis + §8 :
  exécution de code réel, isolation = snapshot + sous-process borné, **pas** un
  bac à sable syscall). 155 tests au total en défaut, clippy 0 warning.
- ✅ **Binaire `rsi-dgm`** : CLI pilotant la boucle DGM/STOP sur un **dépôt
  réel** (`CargoEvaluator` build+test borné). **Sûr par défaut : DRY-RUN**
  (l'arbre vivant n'est jamais écrit) ; `--promote` applique le **seul** meilleur
  variant *tout-au-vert* avec sauvegarde réversible. Backend **Ollama local par
  défaut** (`required-features = ["llm-ollama"]`), Claude optionnel
  (`--features llm-claude-ureq`, `--backend claude`). Validé bout-en-bout
  (`--steps 0` : référence évaluée par vrai `cargo build`+`test`, arbre intact ;
  validation d'args). clippy 0 warning (llm-ollama, +llm-claude-ureq).
- ✅ **Ω concret + substrat mesuré SIMD** (extensions *additives* du moteur v9,
  sans duplication ; suite à l'analyse de la proposition « RSI v9 » qui réinventait
  un cœur déjà présent). (1) `src/omega_tasks.rs` : banc de 7 tâches réelles
  nommées (profil explicite `(D,M,R,A,C,V)` + demande + μ), branché sur les traits
  `CapabilityModel`/`CeilingModel` via `IntelligenceSurface::from_tasks` +
  `task_breakdown` ; `report()` montre par tâche si le goulot est `Φ_x` (cognitif)
  ou `g_x` (substrat). (2) `SimdMeasuredSubstrate` : `SubstrateImprover` ancrant
  `measured_software_eff` sur l'efficience **SIMD réelle** de l'hôte (réduction
  sérielle vs vectorisée), même *ratchet* monotone que `MeasuredSubstrate` ;
  `scirust-rsi` n'exposant pas de métrique matérielle, la mesure est en-process.
  Démo `examples/omega_real.rs`. 161 tests + doctests, clippy 0 warning (défaut +
  features publiques), CI verte (dépôt public).
- ✅ **VALIDATION MATÉRIELLE COMPLÈTE — Jetson Thor (aarch64, GPU Blackwell)** :
  la boucle DGM tourne **de bout en bout sur du réel** avec un **LLM local**
  (qwen3-coder:30b via Ollama) : propose → applique en copie isolée →
  `cargo build`+`test` (177 tests) → **bench de perf réel** (`--bench`,
  `RSI_BENCH_SCORE` = débit de `dot`) → élitisme. Bugs réels trouvés sur
  matériel et corrigés avec tests : *Transfer-Encoding chunked* d'Ollama,
  recomposition multi-ligne, fences ``` (fermées ou non), prompt incluant le
  contenu des fichiers (sinon le modèle hallucine du code), erreurs Ollama
  surfacées, **`complete_raw`** préservant les lignes vides (sinon `FIND` ne
  matche pas), raison de rejet affichée (`↳`). Un premier « ACCEPTÉ » à +0.25 %
  s'est révélé être du **bruit de mesure** → ajout de `min_score_gain`
  (`--min-gain`, anti-bruit : seuil relatif exigé pour les gains de score purs,
  gains structurels exemptés). Verdict final avec `--min-gain 0.03` : 8/8
  candidats compilent et passent les tests, aucun > 3 % — **`dot` est déjà
  optimal sur ce bench** (à n=2¹⁶ le kernel est borné par la bande passante
  mémoire, pas le calcul : les accumulateurs indépendants n'y peuvent rien).
  La boucle n'accepte que ce qu'elle **prouve** — comportement voulu. Également
  validés sur le Thor : `hw_probe` (nvidia-smi + VRAM), SIMD mesuré ×4.44,
  177 tests, features scirust/wasm/observability/simd/llm-*.
- ✅ **PREMIÈRE AMÉLIORATION AUTO-DÉCOUVERTE PROMUE — Jetson Thor** : la boucle
  DGM (qwen3-coder:30b local) a **découvert seule** le réordonnancement de
  boucles i,k,j de `kernels::matmul` (boucle interne contiguë sur B et C →
  auto-vectorisation NEON) : **7.70 → 54.22 matmuls/s à n=512, ×6.6 mesuré**,
  gate compile + 184 tests + bench + seuil anti-bruit 5 %, promu via
  `--promote` (sauvegarde `.rsi_backups`). Chemin durci pour y parvenir :
  parsing FIND/REPLACE **sans délimiteurs** (repli brut borné par les marqueurs
  de section), appariement de patch **tolérant aux espaces** (`locate_match` :
  exact d'abord, sinon ligne-à-ligne après `trim`, unicité exigée aux deux
  étages), `options.num_predict=4096` (les réponses tronquées ne parsaient
  pas), détail des échecs affiché (8 lignes de la queue cargo sous `↳`).
  Cible dédiée `src/kernels.rs` (≠ `matmul_naive`, qui est le *baseline de
  mesure* de `MeasuredSubstrate` — l'accélérer casserait l'instrument), bench
  `bench_kernel` à n=512 (3 Mo > L2 de 2 Mo/cœur : à n=160 tout tenait en L2
  et le naïf sans bounds-checks scorait déjà au niveau des variantes tuilées —
  aucun headroom). **Leçon majeure** : la revue humaine du diff promu a trouvé
  un bug de **contrat** invisible au gate — le patch accumulait dans `c` sans
  zéroter (`C += A·B` au lieu de `C = A·B`), tous les tests passant un `c`
  déjà nul. Correctif : zérotage de ligne (speedup préservé), trou de spec
  fermé par `matmul_overwrites_dirty_output` (c sale + idempotence), référence
  de test repassée en i,j,k (indépendante de l'implémentation). Le gate ne
  vérifie que ce que les tests **expriment** : une spec incomplète laisse
  passer des patchs plus rapides mais sémantiquement plus étroits — la boucle
  outillée trouve, la revue garde le contrat. 178 tests, clippy 0 warning.
- ✅ **Deuxième étage auto-découvert + deuxième trou de spec fermé** : la boucle
  a ensuite proposé un **tuilage i/k (TILE=64)** par-dessus l'ordre i,k,j —
  idée réelle (+27 % mesuré corrigé ; **7.70 → 56.01 sur Thor, ×7.3 cumulé**),
  mais le patch accepté zérotait `c` *dans* la boucle de blocs `kk` (chaque
  bloc écrasait les précédents — correct pour n ≤ 64 seulement). Le gate n'a
  rien vu : toutes les tailles de test étaient ≤ 33, et **le bench mesure la
  vitesse, jamais la correction** — le score promu (55.89) chronométrait un
  calcul faux. Revue : tailles 96/130 ajoutées à `matmul_matches_reference`
  (vérifié : le test durci échoue sur le patch tel quel), zérotage sorti avant
  `kk`. Doctrine issue des deux épisodes : chaque patch accepté est relu à la
  recherche du cas que les tests ne couvrent pas, et ce cas **devient un
  test** — le gate se durcit à chaque promotion.
- ✅ **CAMPAGNE « ON FAIT TOUT » — cinq cibles, cinq verdicts, tous honnêtes**
  (Jetson Thor, qwen3-coder:30b local) : (1) **sum ×2.6** (1149 → 2952 réd./s)
  — somme par blocs de 8 vectorisée, deux promotions successives, le gate
  Kahan a abattu 3 variantes fausses en direct ; (2) **sha256 +18 %** (213 →
  253 Mo/s) — déroulage par 8 de la boucle de rondes, obtenu avec un objectif
  *directif* après que l'objectif vague ne produisait que des no-ops ; gate =
  vecteurs officiels FIPS ; **dépendant de l'architecture** (−4 % mesuré sur
  x86 — le Thor est le matériel cible) ; (3) **json : pas de gain, et c'est
  le bon verdict** — un « ACCEPTÉ +6.5 % » s'est révélé être un *bruit de la
  RÉFÉRENCE* (le bench varie de ±5 % entre runs : le seuil anti-bruit protège
  contre une variante chanceuse, pas contre une référence malchanceuse —
  à stabiliser avant de retenter) ; (4) **matmul/transpose à min-gain 3 % :
  plateau confirmé** (5 variantes cassées rejetées par les tests durcis) ;
  (5) **conv2d écarté par sondage préalable** (×1.09 — l'auto-vectorisation
  fait déjà le travail, pas de gradient à offrir). Corrections d'outillage
  nées de la campagne : `options.num_ctx=16384` (le défaut serveur de 4096
  tronquait le prompt sur les fichiers réels — json passait de 0/8 à 8/8
  propositions applicables), MAX_FILE_CHARS 8k→24k, affichage live des steps
  (8 étapes muettes ≈ blocage apparent), compteurs de tokens Ollama en debug,
  **raison d'échec de parsing explicite** (l'aperçu tronqué à 2000 chars
  avait fait prendre des no-ops complets pour des troncatures serveur !),
  REPLACE borné par un FIND: suivant (double proposition concaténée), prompt
  anti-FIND-géant et anti-no-op. Leçon de méthode : quand le modèle tourne
  en rond, un **objectif directif** (« déroule par 8 ») débloque ce qu'un
  objectif vague (« accélère ») n'obtient pas.
- ⏭️ **Reste** : vérification formelle creusot/loom (P0.4) — outillage + temps
  expert ; non bloquant.
