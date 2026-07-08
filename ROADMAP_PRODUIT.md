# RSI : feuille de route produit — du prototype au moteur RSI industriel

Suite de l'audit v0.10.0. L'objectif : transformer ce prototype de recherche
en un **moteur RSI déployable** donnant à un modèle d'IA (LLM) la capacité
*d'auto-amélioration récursive sûre* — c'est-à-dire : proposer → évaluer →
garder si meilleur → répéter, le tout sous garde-fous vérifiables et
traçabilité forensique.

Cette feuille de route est organisée en 5 phases (P0 → P4), priorisées par
dépendance et valeur. Chaque phase est autonomiquement livrable.

---

## Diagnostic initial

Le projet a trois forces produit :

1. **Un cœur mathématique sain** : dynamique contrainte (λ, ε), criticité
   AMDEC, audit hash-chaîné — la « colonne vertébrale » de sûreté existe.
2. **Un socle Loop Engineering mature** : `run_until`, disjoncteur, checkpoint,
   swarm, observateur HITL — le pilotage opérationnel est en place.
3. **Un sandbox d'auto-amélioration** : `ascent`/`synthesis`/`scirust_bridge`
   avec AST interprété, élitisme borné, non-régression garantie par
   construction.

Et trois faiblesses produit :

1. **Aucune intégration LLM réelle** : le serveur MCP expose des commandes,
   mais l'agent IA ne *propose* rien — il appelle `rsi_run` et observe. Le
   « RSI » est simulé, pas piloté par le modèle.
2. **Capacités jouet** : la synthèse symbolique cible `x² + 1`. Aucune tâche
   utile n'est branchée. Le « moteur » ne produit rien de réutilisable.
3. **Garde-fous non formellement vérifiés** : bug B (line search contournable
   par ℳ), bug A (saturation mal définie), aucune preuve ni property-based
   testing à l'échelle industrielle.

La transformation en produit suppose de corriger ces trois faiblesses,
**dans cet ordre** (sûreté d'abord, capacités ensuite, intégration LLM
enfin).

---

## P0 — Stabilisation & sûreté du cœur (1–2 semaines)

> Sans cœur formellement correct, tout ce qui suit est du sable. P0 est
> non-négociable et doit précéder toute extension.

### P0.1 — Corriger les bugs hérités (A, B) et les nouveaux (F, G, H)

| Bug | Fichier | Correctif | Effort |
|---|---|---|---|
| **A** saturation | `dynamics.rs:85` | `mean(&state.capability_array())` (D,M,R,A,C,V moyens, déjà `[f64;6]` stack) — corrige perf (plus d'alloc par `eta`) ET sémantique | 1 ligne |
| **B** line search | `agent.rs:264-453` | Étendre le line search au pas combiné `meta + appr` : après `state_after_meta`, appliquer le même backtracking sur `SI_global(state_after_meta + delta_appr)` au lieu de `SI_global(state + delta_appr)`. Alternative : documenter que `delta_si` n'est pas garanti et ajuster le README. **Recommandé : étendre.** | ½ jour |
| **F** swarm panic | `swarm.rs:53` | `join().ok()` + marquage `SwarmMember { si_safe: NEG_INFINITY }` pour invalider le membre ; exclure de la sélection | 1 h |
| **G** sous-processus | `knowledge.rs:175` | `Command::stdout(Stdio::piped())` + lecture bornée (ex. 64 MB) + `wait_timeout` (30 s via thread + `join`). Rejeter si débordement. | ½ jour |
| **H** checkpoint | `checkpoint.rs:140` | Stocker `Dims` dans le checkpoint, valider contre la surface au `restore`, retourner `Result<_, String>` au lieu de panic | 2 h |

Bugs C, D, E : reporter à P3 (non-bloquants pour la sûreté).

### P0.2 — Property-based testing sur les invariants

La suite de tests actuelle (89 tests) couvre les cas nominaux mais pas les
*edge cases adversariaux*. Un RSI industriel doit prouver ses invariants sur
un espace d'entrées vaste.

- Ajouter [`proptest`](https://crates.io/crates/proptest) (dépendance dev-only,
  n'impacte pas le cœur std-only).
- Générateurs : `CognitiveState` aléatoire (dims 1–32, valeurs extrêmes),
  `Substrate` dégénéré (matrice nulle, H/O vide), `StabilityConfig` aux
  extrêmes (λ→0, ε→0, η₀→∞).
- Properties à vérifier sur 10 000 trajectoires de 200 pas :
  - `∀ t : ‖ΔS_t‖ ≤ λ` (garde-fou amplitude)
  - `∀ t : SI(t+1) ≥ SI(t) − ε_effectif` (garde-fou non-régression, **avec
    bug B corrigé**)
  - `∀ t : SI_global ∈ [0,1]` (cohérence Monte-Carlo)
  - `∀ t : P_eff ∈ (0,1)` (sigmoïdes bornées)
  - `∀ t : risk_global ∈ [0,1]`, `max_rpn ∈ [0,1]` (AMDEC bornée)
  - Déterminisme : `same_seed ⇒ same_audit_head` (déjà testé, généraliser)
- Cible : 0 échec sur 10 000 runs aléatoires. C'est le **seuil de confiance
  industriel** pour les garde-fous.

### P0.3 — Fuzzing du parseur JSON et du serveur MCP

Le parseur JSON maison (`json.rs`) et le serveur MCP (`rsi_mcp.rs`) traitent
des entrées non fiables. Ajouter :

- `cargo fuzz` sur `Json::parse` (cible : pas de panic sur entrées
  arbitraires, y compris UTF-8 invalide, profondeur limite, nombres extrêmes).
- Test MCP : envoyer une ligne JSON de 100 MB → doit échouer proprement, pas
  OOM. **Plafond stdin à 16 MB** dans `rsi_mcp.rs` avant `Json::parse`.

### P0.4 — Audit formel (optionnel, si budget)

Pour un produit commercial de sûreté IA, un audit formel du cœur est
envisageable :

- **`cargo-creusot`** (preuves Rust verifiées en Why3) sur `constrained_step`
  et `assess` (AMDEC) — prouver les invariants mathématiques.
- **`loom`** (model checker de concurrence) sur `swarm.rs` — exclure les
  data races.

Coût : 2–4 semaines d'expert. Valeur : différentiateur commercial
(crates.io « formally verified safety core »).

---

## P1 — Capacités réelles & branchement LLM (3–4 semaines)

> Le cœur est sûr. Maintenant il doit *faire* quelque chose d'utile, piloté
> par un LLM.

### P1.1 — Définir le contrat « tâche auto-améliorable par un LLM »

Le `RefineTask` actuel (`ascent.rs`) est trop abstrait pour un LLM. Définir
un trait **`LlmRefineTask`** qui :

```rust
pub trait LlmRefineTask: RefineTask {
    /// Le LLM propose une révision (texte/JSON/AST) — on l'injecte.
    fn propose(&mut self, incumbent: &Self::Cand, llm_output: &str)
        -> Result<Self::Cand, ProposalError>;
    /// Le LLM « voit » l'incumbent sous forme textuelle (prompt-friendly).
    fn describe(&self, incumbent: &Self::Cand) -> String;
    /// Critères de sûreté spécifiques au domaine (en plus de ℳ).
    fn safety_check(&self, cand: &Self::Cand) -> Result<(), SafetyViolation>;
}
```

Cela sépare proprement :
- **Le moteur** (`ascent`/`scirust_bridge`) — boucle élitiste bornée, non
  configurable par le LLM.
- **Le domaine** (`LlmRefineTask`) — ce que le LLM propose, comment on
  l'évalue, ce qui est interdit.
- **Le LLM** — producteur de propositions, jamais exécuté directement.

### P1.2 — Sandbox d'exécution réelle (pas seulement AST)

L'AST `Expr` est trop limité (add/sub/mul/neg). Pour des capacités utiles
(prompts, configurations, petits programmes), il faut un sandbox plus large :

- **Option A — AST étendu** : ajouter `If`, `Let`, `Call` (d'un ensemble
  fixe de fonctions sûres), `Compare`. Reste interprété, jamais compilé.
  Adapté à la synthèse de stratégies, pas à l'exécution de code arbitraire.
- **Option B — WASM sandbox** : compiler le candidat en WASM, l'exécuter
  dans [`wasmtime`](https://wasmtime.dev) avec fuel metering (limite
  d'instructions) + capabilities (pas de filesystem, pas de réseau).
  Permet au LLM de proposer du vrai code (Python/Rust→WASM) tout en
  garantissant terminaison et isolation.
- **Option C — Sous-processus jail** : `nsjail`/`firejail` + timeout +
  memory limit. Plus simple mais moins portable, moins garanties.

**Recommandation P1** : Option A (AST étendu) pour les capacités de
stratégie ; Option B (WASM) pour les capacités de code, en P2. L'Option C
est un anti-pattern pour un produit de sûreté.

### P1.3 — Trois domaines réels de démonstration

Pour prouver que le moteur RSI est utile, brancher trois domaines concrets :

1. **Optimisation de prompts** : le LLM propose des variantes d'un prompt
   (AST = chaîne structurée), évaluées sur une suite de tâches (p. ex.
   `mmlu`, `humaneval`). Non-régression = la qualité du prompt ne baisse
   pas. Sandbox : aucun risque (texte seulement).

2. **Configuration d'outils** : le LLM propose des configurations JSON pour
   un outil (p. ex. un retriever RAG : top-k, chunk size, embedding model).
   Évaluation : précision sur un benchmark fixe. Sandbox : JSON validé par
   schéma, aucune exécution.

3. **Petits programmes numériques** : le LLM propose des fonctions Rust
   compilées en WASM (Option B), évaluées sur des cas de test. Non-régression
   = tous les tests passent + performance non dégradée. Sandbox : WASM fuel.

Ces trois domaines couvrent le spectre « texte → config → code » et
démontrent que le moteur RSI est **générique**.

### P1.4 — Intégration MCP bidirectionnelle

Le serveur MCP actuel est unidirectionnel : l'agent IA appelle `rsi_run` et
observe. Pour du RSI réel, le LLM doit **produire** des propositions via MCP.

Ajouter les outils MCP :

| Outil | Rôle |
|---|---|
| `rsi_propose` | Le LLM soumet une proposition de révision (texte/JSON/AST) |
| `rsi_evaluate` | Le LLM demande l'évaluation d'un candidat (sans l'adopter) |
| `rsi_incumbent` | Renvoie l'incumbent courant + son `describe()` (prompt-friendly) |
| `rsi_history` | Renvoie l'historique des propositions + fitness (pour le raisonnement LLM) |
| `rsi_guard` | Renvoie les garde-fous actifs (max_iters, patience, target) — lecture seule |

Le LLM voit l'incumbent, propose, le moteur évalue en sandbox, adopte si
strictement meilleur, rejette sinon. **Le LLM ne contrôle jamais la boucle
elle-même** — il ne fait que proposer. C'est le contrat de sûreté fondamental.

### P1.5 — Observabilité & forensique LLM

Au-delà de l'audit hash-chaîné existant, ajouter :

- **Trace de raisonnement LLM** : chaque proposition porte le `rationale`
  (texte du LLM), stocké dans l'audit. Permet de *reconstruire* la chaîne de
  raisonnement ayant mené à une amélioration — essentiel pour la forensique
  d'un incident.
- **Métriques d'apprentissage** : taux d'acceptation, diversité des
  propositions, dérive du rationale (le LLM répète-t-il les mêmes idées ?).
- **Export structuré** : l'audit actuel est JSON ; ajouter export
  [OpenTelemetry](https://opentelemetry.io) traces pour intégration SIEM.

---

## P2 — Scalabilité & performance (2–3 semaines)

> Le moteur est sûr et utile. Maintenant il doit tenir la charge.

### P2.1 — Parallélisme de l'évaluation

L'évaluation `projected_si` est le goulot (chaque candidat ℳ déclenche un
`si_global` sur tout Ω). Pour `n_tasks=1024` et `candidates=48`, c'est 48 K
évaluations par pas — séquentiel aujourd'hui.

- **Vectoriser** `si_global` : SIMD via [`std::simd`](https://doc.rust-lang.org/std/simd/) (stable depuis Rust 1.0) ou
  [`wide`](https://crates.io/crates/wide). Gain attendu : 4–8× sur f64.
- **Paralléliser** l'évaluation des candidats ℳ : `rayon` (déjà une
  dépendance optionnelle via `forge`). `par_iter` sur les candidats.
- **Mémoïser** : `ForgeMetaSearch` a déjà un cache (`si_cache`), généraliser
  à `MetaOptimizer` et `CmaEsMeta`.

Cible : 10× de débit sur `MetaOptimizer::revise`, permettant
`n_tasks=10_000` + `candidates=200` en temps réel.

### P2.2 — Checkpoint incrémental & reprise

Le `Checkpoint` actuel sérialise tout l'état macro. Pour des trajectoires
longues (millions de pas) :

- **Checkpoint incrémental** : sérialiser seulement le delta depuis le dernier
  checkpoint (diff sur S, substrat, stratégie).
- **Compaction** : merger les anciens checkpoints (garde N derniers + 1 par
  décade).
- **Reprise diffable** : `restore(from, to)` pour rejouer un intervalle
  sans repartir de zéro.

### P2.3 — Streaming MCP

Le serveur MCP lit ligne par ligne, mais les réponses sont bloquantes (un
`rsi_run` de 10 000 pas bloque le canal). Ajouter :

- **Notifications JSON-RPC** : `notifications/progress` pendant un `run`
  long (toutes les N pas).
- **Annulation** : `cancel` interrompt un `run` en cours (grâce à
  `LoopObserver` qui peut vétoer).
- **Streaming d'audit** : publier les événements d'audit en temps réel via
  `notifications/audit_event` (abonnement opt-in).

### P2.4 — Backend GPU optionnel

Pour les domaines lourds (évaluation de modèles, WASM complexe) :

- **Feature `gpu`** : backend `wgpu` pour les kernels numériques (matmul,
  évaluation de surface sur GPU). Optionnel, préserve le cœur std-only.
- **Auto-détection** : fallback CPU si GPU indisponible.

---

## P3 — Qualité produit & robustesse API (2 semaines)

> Avant une release publique, tout doit être poli.

### P3.1 — Correction des bugs restants (C, D, E)

| Bug | Correctif |
|---|---|
| **C** encode/decode bornes | `decode` : `gain = GAIN_LO + (GAIN_HI-GAIN_LO)*sigmoid(theta[6])` → borner aussi `clamp(GAIN_LO, GAIN_HI)` avant retour ; test aux bornes `gain = GAIN_LO` et `gain = GAIN_HI` |
| **D** échappement `\u` | `write_escaped` : si `c as u32 > 0xFFFF`, émettre surrogate pairs (`\uD800\uDC00`-style). Tester avec `Json::Str("😀".into())` |
| **E** `as_u64` silencieux | `as_u64` : `if n.is_finite() && n >= 0.0 { Some(n as u64) } else { None }`. Tester avec `-1`, `NaN`, `Infinity` |

### P3.2 — API versionnée & stabilité

- **Versionning sémantique** : `0.10.0` → `0.11.0` pour P1 (breaking si
  nécessaire). `1.0.0` après P2 + validation terrain.
- **Stabilité de l'API publique** : `#[doc(hidden)]` sur les internals,
  `#[stable]` (via `#[cfg(feature = "stable-api")]`) sur les types publics
  garantis stables.
- **Déprécation** : `#[deprecated]` sur `RSIAgent::demo` (remplacer par
  `RSIAgent::builder()` plus explicite) avec guide de migration.

### P3.3 — Configuration déclarative

Remplacer les builders chaînés par un `RsiConfig` sérialisable (TOML/JSON) :

```toml
[agent]
seed = 2026
dim = 6
optimizer = "cma"  # ou "random", "scirust", "forge"

[dynamics]
lambda = 0.5
epsilon = 1e-3
adaptive_epsilon = true

[risk]
rpn_max = 0.3
kappa = 0.5
active_response = true

[loop]
max_steps = 1000
plateau_window = 12
breaker_rpn = 0.5

[capabilities]
substrate_improver = "measured"  # ou "forge", "none"
memory = "linear"  # ou "octasoma", "none"
knowledge = "corpus"  # ou "papers", "none"
audit = "hashchain"  # ou "ccos", "none"
```

Permet le déploiement reproductible (`rsi run --config agent.toml`) et la
différence de config entre environnements (dev/prod/research).

### P3.4 — Observabilité opérationnelle

- **Métriques Prometheus** : `rsi_steps_total`, `rsi_backtracks_total`,
  `rsi_circuit_breaker_trips_total`, `rsi_audit_chain_integrity` (gauge).
- **Health check** : `GET /health` sur le serveur MCP (si mode HTTP) →
  JSON `{ "ok": true, "audit_integrity": true, "t": 1234 }`.
- **Logs structurés** : `tracing` (déjà std-compatible) avec niveaux
  (INFO pour pas, DEBUG pour candidats, WARN pour backtracks, ERROR pour
  disjoncteur).

### P3.5 — Documentation produit

- **Guide de démarrage** : `docs/QUICKSTART.md` — 5 minutes du `cargo run`
  au premier RSI piloté par LLM.
- **Guide de sûreté** : `docs/SAFETY.md` — quels garde-fous sont garantis,
  lesquels ne le sont pas, comment les étendre, quels modes de défaillance
  sont surveillés.
- **Guide d'intégration LLM** : `docs/LLM_INTEGRATION.md` — schéma de
  boucle LLM ↔ MCP, exemples de prompts, anti-patterns (ne pas laisser le
  LLM contrôler `max_iters`).
- **Guide de forensique** : `docs/FORENSICS.md` — comment rejouer une
  trajectoire depuis l'audit, détecter une anomalie, exporter vers SIEM.
- **Mise à jour du README** : aligner sur la licence réelle (cf. P3.6).

### P3.6 — Cohérence licence & commercialisation

- **Décider** : `PolyForm-Noncommercial` seul (et retirer la mention « double
  licence » du README) OU double licence réelle (déclarer les deux dans
  `Cargo.toml` via `license = "PolyForm-Noncommercial-1.0.0 OR Commercial"`).
- **CLAUDE.md / CONTRIBUTING.md** : clarifier le régime de contribution
  (CLA nécessaire pour usage commercial ?).
- **Mentions de sûreté** : disclaimer produit — RSI est un outil de
  *recherche* et de *prototypage* d'auto-amélioration, pas une certification
  de sûreté. Les garde-fous sont des *hypothèses*, pas des garanties.

---

## P4 — Écosystème & distribution (ongoing)

> Le produit fonctionne. Maintenant il doit être adoptable.

### P4.1 — Packaging

- **crates.io** : publier `rsi` 0.11.0 après P0+P1+P3. Vérifier
  `cargo publish --dry-run`.
- **Homebrew** : `brew install rsi` (tap `Memorithm/tap`).
- **Docker** : `docker run -i rsi-mcp` (image multi-arch, ~20 MB via
  `cargo build --target x86_64-unknown-linux-musl`).
- **Nix** : `flake.nix` pour reproductibilité bit-identique.
- **APT/RPM** : paquets système pour déploiement serveur.

### P4.2 — SDK clients

Le serveur MCP est un backend. Pour l'adoption, fournir des clients :

- **`rsi-py`** : bindings Python via PyO3 (ou wrapper subprocess + JSON-RPC).
  Permet `import rsi; agent = rsi.Agent(); agent.step()`.
- **`rsi-ts`** : client TypeScript/Node pour intégration LLM
  (Vercel AI SDK, LangChain).
- **`rsi-cli`** : CLI enrichi (`rsi create`, `rsi run`, `rsi propose`,
  `rsi audit verify`) — au-delà du `rsi-demo` actuel.

### P4.3 — Benchmark public

Publier un benchmark reproductible de RSI sur des tâches standards :

- **MMLU** (prompt optimization) — combien d'itérations pour +X % ?
- **HumanEval** (code generation config) — taux d'amélioration.
- **GPQA** (reasoning) — robustesse sur tâches difficiles.

Comparaison vs baseline (pas de RSI) et vs optimisation manuelle. C'est la
**preuve de valeur** du produit.

### P4.4 — Communauté & gouvernance

- **RFC process** : `docs/rfc/` pour les changements majeurs (nouveau
  domaine, nouveau garde-fou, breaking API).
- **Security policy** : `SECURITY.md` — canal de signalement, SLA de
  réponse, politique de divulgation.
- **Advisory database** : `rustsec-advisory` pour les vulnérabilités du
  produit (distinct du code).

---

## Synthèse : priorisation et dépendances

```
P0 (sûreté du cœur) ──────► P1 (capacités + LLM) ──────► P2 (scalabilité)
  │                                                              │
  └── P3 (qualité produit) ◄──────────────────────────────────────┘
         │
         └── P4 (écosystème) [ongoing]
```

| Phase | Durée | Blocage | Valeur |
|---|---|---|---|
| **P0** Sûreté | 1–2 sem | aucun | sans elle, rien n'est fiable |
| **P1** Capacités + LLM | 3–4 sem | P0 | produit utilisable par un LLM |
| **P2** Scalabilité | 2–3 sem | P1 | tenir la charge en production |
| **P3** Qualité produit | 2 sem | P0+P1 | release publique 1.0 |
| **P4** Écosystème | ongoing | P3 | adoption |

**Total vers 1.0** : ~10–12 semaines (2–3 mois) pour un produit
industriel avec intégration LLM réelle, capacités utiles, scalabilité et
documentation produit.

---

## Critères de sortie « produit industriel »

Un produit RSI industriel doit satisfaire :

1. ✅ **Sûreté prouvée** : 0 échec sur 100 000 trajectoires property-tested,
   invariants documentés, audit hash-chaîné vérifiable.
2. ✅ **Intégration LLM** : un LLM peut piloter la boucle via MCP, proposer
   des révisions, observer l'incumbent — sans jamais contrôler la boucle.
3. ✅ **Capacités utiles** : au moins 3 domaines réels (prompts, config,
   code) avec amélioration mesurable sur benchmark public.
4. ✅ **Scalabilité** : 10× débit vs v0.10.0, support GPU optionnel,
   checkpoint incrémental.
5. ✅ **Observabilité** : métriques Prometheus, logs structurés, export
   SIEM, forensique reproductible.
6. ✅ **Documentation** : guides de démarrage, sûreté, intégration LLM,
   forensique — complets et à jour.
7. ✅ **Distribution** : crates.io, Docker, Homebrew, SDK Python/TS.
8. ✅ **Gouvernance** : RFC process, security policy, advisory database.

Le projet RSI v0.10.0 remplit déjà ~30 % de ces critères (cœur mathématique,
Loop Engineering, audit, sandbox AST). Les 70 % restants sont l'objet de
cette feuille de route.

---

## Risques & mitigeations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Le LLM propose du code malicieux (exfiltration, boucle infinie) | élevée | critique | Sandbox WASM fuel + capabilities (P1.2) ; `safety_check` par domaine (P1.1) ; aucun réseau/fs |
| Le LLM « wireheading » : apprend à tromper l'évaluateur | moyen | critique | Anti-wireheading existant (`software_eff_gap`) ; étendre à `safety_check` ; audit forensique |
| Régression de SI non détectée (bug B non corrigé) | moyen | élevé | P0.1 corrige formellement ; banc d'ablation surveille |
| Panique en production (bug F swarm, bug H checkpoint) | faible | moyen | P0.1 corrige ; `catch_unwind` en plus |
| Saturation perf (évaluation ℳ séquentielle) | élevé | moyen | P2.1 parallélisme + SIMD |
| Adoption faible (pas de cas d'usage clair) | moyen | élevé | P1.3 domaines réels + P4.3 benchmark public |
| Licence ambiguë bloque adoption commerciale | faible | moyen | P3.6 décide et aligne |

---

## Conclusion

RSI v0.10.0 est un **prototype de recherche mature** avec une colonne
vertébrale de sûreté solide. Le transformer en produit industriel suppose
trois sauts :

1. **Sûreté prouvée** (P0) — corriger les bugs hérités, property testing,
   plafonds de robustesse. Non-négociable.

2. **Utilité réelle** (P1) — brancher un LLM comme source de propositions,
   pas seulement comme observateur. Le RSI devient alors *réellement*
   récursif : l'agent IA s'améliore lui-même via le moteur, sous garde-fous.

3. **Maturité opérationnelle** (P2+P3) — scalabilité, observabilité,
   documentation, distribution. Le passage du « ça marche » au « c'est
   déployable en production ».

L'ordre importe : **sûreté d'abord, capacités ensuite, scalabilité enfin**.
Un RSI qui s'améliore vite mais dont les garde-fous ne sont pas prouvés est
un danger ; un RSI sûr mais qui ne fait rien d'utile est un jouet. P0+P1
donnent le premier produit démontre la valeur ; P2+P3+P4 en font un
produit industriel.

Le contrat de sûreté fondamental à traverser intact toutes les phases :

> **Le LLM propose, le moteur dispose.** L'agent IA ne contrôle jamais la
> boucle d'ascension — il génère des candidats, le moteur les évalue en
> sandbox, les adopte élitistement (strictement meilleur) ou les rejette,
> sous garde-fous bornés et audit hash-chaîné vérifiable.

Toute extension qui viole ce contrat (donner au LLM le contrôle de
`max_iters`, de `target`, ou pire du `line search`) est un recul de sûreté
et doit être refusée par design.

---

*Fin de la feuille de route produit.*