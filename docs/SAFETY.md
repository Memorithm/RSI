# Sûreté — garde-fous, garanties et limites

Ce document recense les garde-fous de RSI, **ce qu'ils garantissent réellement**
et **ce qu'ils ne garantissent pas**. Il est délibérément honnête : RSI est un
**laboratoire de recherche** sur l'auto-amélioration encadrée, pas une
certification de sûreté. Plusieurs garde-fous sont des *hypothèses de modèle*,
pas des preuves.

> **Disclaimer.** Les garanties ci-dessous portent sur le **modèle exécutable**
> (dynamique contrainte, élitisme, audit). Elles ne disent rien de la sûreté
> d'un système réel qui brancherait ce moteur sur des capacités réelles. Voir
> §8 « Ce qui n'est PAS garanti ».

---

## 1. Principes

1. **Sûreté d'abord, capacités ensuite.** Les garde-fous sont appliqués *après*
   chaque pas brut et conditionnent l'adoption.
2. **Le LLM propose, le moteur dispose.** Un LLM ne produit que du texte ; le
   moteur parse, valide, évalue en sandbox et adopte élitistement. Le LLM ne
   contrôle aucun garde-fou (cf. `docs/LLM_INTEGRATION.md`).
3. **Tout est borné et reproductible.** Bornes dures de ressources, terminaison
   garantie, déterminisme par graine, audit vérifiable.

---

## 2. Garde-fous de la dynamique (§4, `src/dynamics.rs`)

### 2.1 Amplitude `‖ΔS‖ ≤ λ`
Chaque pas est projeté radialement sous la borne `λ` (`StabilityConfig.lambda`,
défaut `0.5`).
- **Garanti** dans le **domaine opérationnel `[0,1]ⁿ`** — la projection de
  clipping y est non-expansive. Le moteur clippe l'état à chaque pas, donc il y
  reste toujours.
- **Précondition** : l'invariant ne tient PAS pour un état semé hors `[0,1]ⁿ`
  (découvert par property testing — cf. `guardrails_hold_under_random_extreme_configs`).

### 2.2 Non-régression `SI(t+1) ≥ SI(t) − ε`
Line search d'atténuation tant que `SI` régresse au-delà de `ε`
(`StabilityConfig.epsilon`, défaut `1e-3`).
- **Garanti pour l'étape d'apprentissage** (`constrained_step`).
- **Étendu au pas combiné ℳ + apprentissage** (`RSIAgent::step`, §4bis) : hors
  override de sûreté explicite, l'incumbent ne régresse pas au-delà de `ε`.
- **Exception assumée** : un override de sûreté (p. ex. `trust_floor`
  anti-wireheading, qui abaisse délibérément `P_eff`) peut faire régresser `SI`.
  C'est intentionnel — la sûreté prime. La portée exacte est documentée dans le
  README (§4) et testée (`combined_step_non_regression_when_no_override`).

### 2.3 ε adaptatif
Optionnel (`adaptive_epsilon`) : `ε_eff = ε + z·stderr(SI)` pour ne pas pénaliser
une variation sous le bruit Monte-Carlo. Évite les faux backtracks.

---

## 3. Criticité / AMDEC (§7, `src/criticality.rs`, `src/loop_ctrl.rs`)

- **Modèle de risque AMDEC** : `risk_global` et `max_rpn` ∈ [0,1] (bornés et
  testés), agrégés sur des modes (dérive de valeurs, effondrement de substrat,
  wireheading…).
- **Réponses ciblées** (`RSIAgent::step`) : `damp_gain`, `realign_V`,
  `trust_floor` selon le mode le plus critique.
- **Disjoncteur de criticité** (L4, `run_until`) : si `max_rpn` dépasse
  `breaker_rpn`, arrêt + **rollback** vers le dernier état sain
  (`circuit_breaker_trips_and_rolls_back`).
- **Observateur HITL** (L6) : `LoopObserver::on_step` peut **véto** la poursuite
  (`observer_veto_stops_loop`).
- **Arrêts motivés** (`run_until`) : budget, cible, plateau, divergence,
  timeout, disjoncteur, véto.

---

## 4. Élitisme & audit

- **Adoption élitiste** (`src/ascent.rs`, `ascend` / `ascend_llm`) : une révision
  n'est adoptée que si elle est **strictement meilleure** (`min_delta`).
  `Report::is_monotone()` : l'historique de l'incumbent est non décroissant.
- **Audit hash-chaîné** (§7bis, `src/audit.rs`) : trace reproductible et
  inviolable des pas ℳ. `audit_head()` est déterministe à graine fixe
  (`audit_head_is_deterministic`), `audit_verify()` détecte toute altération.

---

## 5. Sûreté de la boucle LLM (`src/llm.rs`)

- **`safety_check` par domaine** : un candidat qui viole les interdits du domaine
  est **rejeté quel que soit son score** (taille d'AST bornée pour la synthèse,
  bornes de schéma pour le tuning).
- **Sandbox** : le « code » candidat n'est **jamais exécuté**. Les expressions
  sont **interprétées** (`Expr::eval`), les configs **validées**. Aucun
  `eval`/sous-processus sur la sortie du modèle.
- **Adoption = strictement meilleur ET sûr.** Côté MCP, le **serveur** applique
  parse + `safety_check` + score + élitisme ; le client LLM ne contourne rien.
- **Anti-Goodhart (held-out)** : `score` (adoption) sur le train, `score_heldout`
  (reporting) sur un jeu jamais vu ; `LlmGuard.max_overfit_gap` stoppe si l'écart
  train/held-out explose.
- **Budget** : `LlmGuard.max_llm_calls` / `max_wall_clock` bornent la boucle
  (terminaison côté coût). `LlmStop::BudgetExhausted`.

### 5bis. Boucle d'auto-amélioration empirique — DGM/STOP (`src/dgm.rs`)

Contrairement aux domaines ci-dessus (qui n'exécutent jamais la sortie du
modèle), la boucle **Darwin–Gödel / STOP** modifie du **code source réel** et
peut le **construire et le tester** (`CargoEvaluator`). Ses garde-fous :

- **Liste blanche** (`LlmProposer.allowed_paths`) : une proposition ciblant un
  fichier hors liste est **écartée** avant toute application.
- **Patch exact & non ambigu** : `find` doit apparaître **exactement une fois**
  (motif absent ou multiple ⇒ rejet — jamais d'édition silencieuse de la
  mauvaise occurrence).
- **Isolation** : chaque candidat est évalué sur un [`WorkspaceSnapshot`]
  **jetable** (copie temp, `target/.git/node_modules` sautés), supprimé au
  `Drop`. L'évasion hors-racine (`..`, chemins absolus) est rejetée
  (`PathNotAllowed`, vérif après canonicalisation).
- **Barrière empirique** : adoption ssi `Fitness::is_better_than(parent)` sous
  l'ordre lexicographique **(compile ▸ non-régression de tests ▸ score)** ; un
  build cassé ne peut jamais dominer (`broken` = −∞), un score NaN non plus.
- **Arbre vivant intouché** : la boucle ne mute jamais le dépôt réel ; seule
  [`promote_to_live`] le fait, et l'appelant la garde sur une évaluation
  **tout-au-vert**. Chaque édition est sauvegardée (`.bak`) donc réversible.
- **Exécution non fiable fail-closed** : par défaut, `cargo build`/`test`/bench
  passent par Bubblewrap + `prlimit` avec réseau/IPC/PID/UTS/cgroup séparés,
  environnement nettoyé, montages runtime en lecture seule, workspace jetable
  seul en écriture et limites CPU/RAM/PIDs. Dépendances disponibles hors ligne
  requises. Le noyau doit autoriser les espaces de noms utilisateur non
  privilégiés nécessaires à Bubblewrap. L'absence du backend ou de cette
  capacité fait échouer l'évaluation ; elle ne déclenche jamais un repli
  silencieux sur l'hôte.
- **Mode fiable explicite** : `CargoExecutionPolicy::TrustedHost` (CLI
  `--trusted-host`) conserve l'exécution historique pour du code **et des
  dépendances** dont l'opérateur garantit la confiance.
- **Sous-processus bornés** : chaque commande a un **timeout** (défaut 300 s),
  une sortie plafonnée (défaut 4 MiB/flux), un drain commun borné et son propre
  groupe de processus ; un délai tue aussi les descendants.
- **Déterminisme** : IDs de variantes = hash SHA-256 de la lignée (≠ UUID
  aléatoire), horloge logique `seq` ⇒ archive **bit-exacte reproductible**.

---

## 6. Durcissement ressources (entrées non fiables)

| Surface | Garde-fou | Constante |
|---|---|---|
| Parseur JSON | profondeur d'imbrication bornée | `MAX_DEPTH = 128` (`json.rs`) |
| Parseur d'expressions | profondeur bornée (anti stack-overflow) | `MAX_EXPR_DEPTH = 256` (`synthesis.rs`) |
| Serveur MCP | plafond de taille par ligne stdin | `16 MiB` (`rsi_mcp.rs`) |
| API sessions | nombre max de sessions | `MAX_SESSIONS = 64` (`api.rs`) |
| API dimensions | |Ω|, dim, substrat, pas bornés | `MAX_TASKS=50_000`, `MAX_DIM=1_024`, `MAX_SUBSTRATE=256`, `MAX_STEPS=100_000` |
| Raffinement | points / propositions bornés | `MAX_REFINE_POINTS=4_096`, `MAX_PROPOSALS_PER_CALL=64` |
| Sous-processus `papers` | timeout + sortie bornée | `30 s` / `8 MiB` (`knowledge.rs`) |
| Sous-processus `cargo` (DGM) | Bubblewrap fail-closed + réseau coupé + env nettoyé + limites RAM/PIDs/CPU + timeout/groupe de processus + sortie bornée | `4 GiB` / `256` / `300 s` / `4 MiB` (`dgm.rs`) |
| Synthèse | taille d'AST adoptable bornée | `MAX_EXPR_SIZE = 25` (`synthesis.rs`) |
| Accès numériques JSON | rejet NaN/∞/négatifs | `as_u64`/`as_usize` (`json.rs`) |

---

## 7. Couverture de tests

- **155 tests** unitaires/intégration, déterministes (dont **23** pour la boucle
  DGM/STOP de `src/dgm.rs` : barrière de fitness, isolation du snapshot, rejet de
  patch ambigu, liste blanche, déterminisme bout-en-bout).
- **Property testing in-tree** (RNG déterministe, std-only, sans dépendance) :
  - garde-fous de la dynamique sur configs extrêmes
    (`guardrails_hold_under_random_extreme_configs`, 1 200 configs × 12 pas) ;
  - bornes de chaque `StepReport` sur 250 graines
    (`report_invariants_hold_over_many_seeds`) ;
  - robustesse du parseur JSON (`parse_never_panics_on_random_input`, 10 000
    entrées aléatoires + adversariales).
- Build + `clippy --all-targets` **0 warning** en défaut / `scirust` /
  `llm-ollama` / `llm-claude`.

---

## 8. Ce qui n'est PAS garanti (limites honnêtes)

- **Un modèle reste un modèle.** Les garanties portent sur la dynamique
  simulée, pas sur un système réel. `Φ`, `P_eff`, `SI` sont des constructions du
  modèle.
- **Les overrides de sûreté peuvent faire régresser `SI`** (par conception, §2.2).
  L'invariant de non-régression a une **portée**, pas une universalité.
- **Les objectifs des domaines de démo sont des stand-ins.** `ConfigTuning`
  optimise un objectif analytique synthétique ; ce n'est pas une vraie métrique
  de benchmark. Le held-out y mesure une généralisation *simulée*.
- **Pas de preuve formelle.** Les invariants sont *property-testés* sur un large
  échantillon (0 échec observé), pas prouvés (creusot/loom envisagés mais non
  faits — cf. spike P0.4).
- **Sandbox = portée du domaine.** La garantie « jamais d'exécution » vaut pour
  les domaines fournis sans exécution (AST interprété, config validée, prompt
  texte). Le **domaine WASM** (`src/wasm_domain.rs`, feature `wasm`) est le seul
  qui **exécute** du code candidat : il apporte son isolation explicite via
  `wasmi` — **zéro import host** (linker vide ⇒ aucun accès réseau/fs/syscall) et
  **fuel borné** (terminaison garantie). Tout module déclarant un import ou
  dépassant le fuel est rejeté/trappé. C'est la condition d'un domaine exécutant,
  désormais satisfaite.
- **La boucle DGM/STOP exécute du code source réel.** Le mode par défaut refuse
  de l'exécuter hors d'un sandbox Linux Bubblewrap borné et sans réseau. Ce
  confinement réduit fortement les effets hôte, mais ne constitue pas une
  preuve formelle de sûreté du noyau, de Bubblewrap ou de la toolchain montée en
  lecture seule. Les dépendances doivent être préparées hors ligne ; aucune
  récupération réseau n'est autorisée pendant l'évaluation. Le mode
  `TrustedHost` contourne volontairement le confinement et **ne doit servir
  qu'à du code et des dépendances de confiance**. Pour des modules autonomes,
  préférer encore le domaine WASM à imports vides et fuel borné.
- **Le déterminisme est bit-exact à configuration fixe.** L'évaluation
  parallèle (MetaOptimizer/CMA) préserve l'ordre d'index → bit-exacte. En
  revanche la feature **`simd`** (OFF par défaut) vectorise les réductions
  `dot`/`mean` : l'ordre de sommation change, donc les valeurs (et l'audit head)
  diffèrent d'un build scalaire — `same_seed ⇒ same_audit_head` ne tient plus
  qu'**à feature égale** (un build SIMD reste déterministe avec lui-même).
  Ne pas comparer des sorties entre un build scalaire et un build SIMD.

---

## 9. Étendre sans casser la sûreté

- Un nouveau domaine LLM **doit** implémenter `safety_check` (interdits) et
  **ne jamais exécuter** la sortie brute du modèle.
- Tout nouveau garde-fou de boucle se branche via `LoopObserver` / `LlmGuard` —
  **jamais** exposé au contrôle du LLM.
- Préserver les property tests : un changement de la dynamique doit re-passer
  les sweeps d'invariants (et re-baseliner les valeurs attendues si la numérique
  change).
- Toute extension qui violerait le contrat « le LLM propose, le moteur dispose »
  (donner au LLM `max_iters`, `target`, ou le critère d'adoption) est un **recul
  de sûreté** et doit être refusée par conception.

---

## 10. Références

- Dynamique & invariants : `src/dynamics.rs`, `src/agent.rs`, `README.md` (§4)
- Criticité & boucle : `src/criticality.rs`, `src/loop_ctrl.rs`, `src/convergence.rs`
- Audit : `src/audit.rs`
- Boucle LLM : `src/llm.rs`, `docs/LLM_INTEGRATION.md`
- Décisions de conception : `docs/P1_DESIGN_SPIKE.md`
