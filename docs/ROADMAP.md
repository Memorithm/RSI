# Feuille de route RSI — Objectifs

> État actuel (v0.10) : cœur `std`-only sans dépendance (surface Σ_I, état
> S=(D,M,R,A,C,V), substrat P_eff, dynamique contrainte, méta-révision ℳ,
> criticité §7, audit hash-chaîné), backends réels en features
> (Forge, OctaSoma, CCOS), corpus de tâches ancré, source `D` (PAPERS en
> sous-processus). 65/70 tests, 0 warning.

## Objectif phare — ⚙️ LOOP ENGINEERING (travail massif)

La boucle d'auto-amélioration est aujourd'hui un `agent.step()` à cadence fixe.
Pour passer d'un *modèle* à un véritable **moteur de boucle RSI**, on engage un
chantier transverse et prioritaire sur l'**ingénierie de la boucle** : sa
structure, son contrôle, sa sûreté, son observabilité et son pilotage.

### Pourquoi
Dans un système récursif, *la boucle EST le produit*. La qualité du RSI dépend
moins d'un pas isolé que de : quand s'arrêter, à quelle cadence méta-réviser,
comment détecter un plateau / une divergence, comment reprendre après incident,
comment imbriquer les échelles de temps, et comment un humain (ou un agent)
peut piloter, mettre en pause, rejouer et reconfigurer la boucle en sûreté.

### Chantiers (L1–L9)

**L1 — Pilote de boucle & critères d'arrêt.** ✅ `LoopConfig`/`StopReason`/
`LoopOutcome` + `RSIAgent::run_until(cfg)` (`loop_ctrl.rs`) : arrêt motivé sur
budget pas/temps, cible `SI`, plateau ou divergence. *Testé* (cible, plateau
avant budget, max_steps).

**L2 — Détection de convergence / divergence / attracteur.** ✅
`ConvergenceDetector` (`convergence.rs`) : pente par moindres carrés sur fenêtre
glissante, `Trend::{Improving, Plateau, Diverging}`. Détecte le plateau de
l'attracteur substrate-limited (utilisé par L1). *Testé.* À étendre : signaux de
divergence basés sur `risk`/oscillation.

**L3 — Boucles multi-échelles (nested loops).** ✅ `schedule.rs` :
`LoopSchedule` (cadences `meta_every`/`substrate_every`) +
`RSIAgent::with_schedule` + `substrate_interval` ; `MetaMeta` révise les
cadences selon la tendance (plateau→ralentit, progrès→accélère). *Testé.*

**L4 — Disjoncteurs de sûreté au niveau boucle.** ✅ `LoopConfig.breaker_rpn`
+ `rollback_on_breach` → `StopReason::CircuitBreaker` : un pic de `max_rpn`
déclenche rollback (au dernier état sain, via L5) puis halte. *Testé.* (Porte
human-in-the-loop : voir L6.)

**L5 — Checkpoint / reprise / replay.** ✅ `checkpoint.rs` : `Checkpoint`
(S, substrat, stratégie, t) sérialisable JSON ; `RSIAgent::snapshot/restore` ;
`save/load`. Reprise vérifiée ; replay bit-identique garanti par graine
(prouvé par le hash de tête d'audit). *Testé.*

**L6 — Plan de contrôle observable.** ✅ trait `LoopObserver`
(`on_step`/`on_stop`) + `run_until_observed` ; **veto human-in-the-loop**
(`StopReason::Vetoed`). *Testé.*

**L7 — Pilotage interactif & live-reconfig.** ✅ commande API/MCP `run_until`
(`rsi_run_until`) — budget/cible/plateau/disjoncteur ; pause/reprise naturelles
via sessions (`step`/`run`/`run_until` incrémentaux). *Testé.*

**L8 — Parallélisme & portefeuille de boucles.** ✅ `swarm.rs` :
`run_swarm`/`run_swarm_demo` (threads `std`, déterministe par graine) ;
sélection du meilleur par `SI_safe`. *Testé.*

**L9 — Banc d'essai de boucle.** ✅ `bin/rsi-loopbench` : effet des cadences
(L3) sur convergence/coût via `run_until`, + apport du portefeuille (L8).
Constat : cadence méta ÷ → ~2,4× moins de méta-évaluations à SI ≈ constant ;
swarm-8 ≈ +13 % vs agent moyen.

### Statut : epic **Loop Engineering — L1→L9 complets** ✅

### Phasage (réalisé)
1. **Socle** : ✅ L1 (pilote/arrêt) → ✅ L2 (convergence) → ✅ L5 (checkpoint/replay).
2. **Sûreté & échelles** : ✅ L4 (disjoncteurs) → ✅ L3 (multi-échelles).
3. **Pilotage & passage à l'échelle** : ✅ L6 (observabilité) → ✅ L7 (MCP) → ✅ L8 (swarm).
4. **Mesure** : L9 (banc d'essai + ablations).

### Invariants à préserver
Cœur `std`-only sans dépendance par défaut ; garde-fous `‖ΔS‖<λ` et
non-régression ; déterminisme par graine ; auditabilité (CCOS) de toute
transition de boucle.

---

## Autres objectifs

- **Validation empirique** : ✅ banc d'ablation `rsi-ablate` (cœur pur) +
  corpus élargi (Ω=40) — voir [`docs/VALIDATION.md`](VALIDATION.md). Constats :
  non-régression préservée partout, connaissances ↑ SI, substrat ↑ vitesse,
  ε adaptatif ↓ risque. À étendre : corpus issu d'un **benchmark public** réel
  (chargeable via `TaskCorpus::from_json`).
- **Connaissances `D`** : ✅ port + `CorpusKnowledge` + `PapersKnowledge`
  (sous-processus). Étendre : ingestion incrémentale, dédup sémantique.
- **Évaluation de code candidat** : ✅ mode non fiable fail-closed sous
  Bubblewrap et cgroup v2 délégué, sans réseau, avec environnement nettoyé,
  workspace jetable, budgets agrégés mémoire/PIDs/temps CPU et destruction de
  tout le groupe au dépassement. Le mode hôte historique reste un choix
  `TrustedHost` explicite réservé au code et aux dépendances de confiance.
- **Substrat** : ✅ `MeasuredSubstrate` natif + Forge. Étendre : domaines réels
  (GPU/SIMD via Forge côté toolchain), efficacité matérielle `H` mesurée.
- **Surface** : ✅ corpus ancré. Étendre : tâches *exécutées* (compétence =
  succès réel d'un solveur), import de jeux de tâches publics.
- **GPU / temps réel** : documenté ([`docs/VALIDATION.md`](VALIDATION.md)) — non
  bloquant par conception (seam `SubstrateImprover` + Forge SIMD/CUDA côté
  toolchain) ; `MeasuredSubstrate` (CPU) couvre l'environnement sans GPU.
