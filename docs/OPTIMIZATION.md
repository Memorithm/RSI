# Optimisation du moteur RSI pour ses intégrations + §7 criticité (AMDEC)

Ce document décrit comment le moteur a été rendu **optimal avec ses
intégrations** (OctaSoma, Forge, et le futur CCOS), et l'ajout d'une **analyse
des modes de défaillance et de criticité** (AMDEC / FMECA) au système
mathématique.

## Diagnostic initial

Les intégrations étaient *branchées mais passives/cloisonnées* :

- la mémoire OctaSoma était **écrite mais jamais relue** → composante `C`
  décorative ;
- le substrat Forge **écrasait** le facteur logiciel analytique → le
  `software_edit` de `ℳ` devenait un no-op (canaux qui se neutralisent) ;
- chaque révision Forge repartait **à froid** (campagne re-seedée, fitness
  recalculée) ;
- aucune **théorie des défaillances** : seulement deux garde-fous ponctuels
  (`‖ΔS‖<λ`, non-régression `ε`).

## A — Mémoire active (recall → ℳ)

`MetaSearch` reçoit une méthode `warm_start(&[MetaStrategy])` (no-op par
défaut). À chaque pas, l'agent **rappelle** les contextes proches
(`recall_similar`), **décode** les stratégies gagnantes mémorisées
(`decode_strategy_payload`) et les **réinjecte comme graines** de la révision.
Les trois optimiseurs (`MetaOptimizer`, `CmaEsMeta`, `ForgeMetaSearch`) les
exploitent (évaluation directe / centre d'exploration / warm-start de campagne).
OctaSoma passe ainsi de *journal* à *accélérateur*. *(`memory.rs`, `meta.rs`,
`agent.rs`)*

## B — Canal substrat unifié

`software_efficiency()` combine désormais l'analytique σ(OᵀB O) et l'efficience
mesurée par **maximum** au lieu d'un écrasement :
`software_efficiency = max(σ(OᵀB O), measured)`. Les deux leviers coopèrent :
améliorer `O` reste utile, et la mesure Forge agit comme **plancher**.
*(`substrate.rs`)*

## C — Campagnes Forge amorties

- `RSIAgent::meta_interval` : la révision `ℳ` n'est exécutée que tous les `k`
  pas (`with_meta_interval`).
- `RsiDomain.si_cache` : `SI_global` est **mémoïsé par id de candidat**
  (`Mutex<HashMap>`), évitant de recalculer la fitness des candidats réapparus.
- Warm-start implicite : la campagne est centrée sur la meilleure stratégie
  connue (courante ou graine mémoire). *(`agent.rs`, `forge_meta.rs`)*

## D — Routage par criticité

L'améliorateur de substrat (coûteux) n'est invoqué **que lorsque le substrat est
la contrainte qui bride réellement** : goulot substrat ≥ `route_threshold` **ou**
mode le plus critique = effondrement du substrat. Généralisation du routage par
goulot vers un routage piloté par l'AMDEC. *(`agent.rs`)*

## §7 — Modes de défaillance & criticité (AMDEC / FMECA)

Module cœur [`criticality.rs`](../src/criticality.rs), sans dépendance.

Pour chaque mode `f` : `RPN_f = Sévérité_f · Occurrence_f · Détection_f`
(facteurs ∈ [0,1], `Détection` = *difficulté* de détection). Modes couverts :

| Mode | Occurrence (dynamique) | Détecteur |
|------|------------------------|-----------|
| régression de compétence | `−ΔSI / ε` | garde-fou ε |
| instabilité / divergence | `‖ΔS‖ / λ` | garde-fou λ |
| dérive des valeurs | `A − V` | (AMDEC) |
| effondrement du substrat | `(1−P_eff)·%substrat` | bottleneck |
| Goodhart / sur-ajustement | `backtracks / 5` | line-search |
| empoisonnement mémoire | base si mémoire active | (CCOS à venir) |
| wireheading | `max(0, mesuré − analytique)` | Forge `verify` |

Agrégats : `Risk_global = moyenne_f RPN_f` et **intelligence ajustée au
risque** `SI_safe = SI_global − κ · Risk_global`.

**Garde-fou actif** (`RiskConfig { kappa, rpn_max, risk_delta, active_response }`)
— quand `max RPN > rpn_max`, l'agent applique une **réponse ciblée** selon le
mode le plus critique (champ `StepReport.mitigation`) :

| Mode critique | Réponse | Effet |
|---------------|---------|-------|
| (tous) | `damp_gain` | atténue le gain de `ℳ` (pas conservateur) |
| dérive des valeurs | `realign_V` | pousse `V` vers le niveau d'autonomie (corrige `A−V`) |
| wireheading | `trust_floor` | rabaisse l'efficience *mesurée* vers l'analytique |

Vraie boucle de contrôle : la réponse fait retomber le RPN, qui remonte, qui
redéclenche la réponse (dents de scie visibles dans la démo `rsi-full`).

Chaque `StepReport` expose `risk_global`, `max_rpn`, `most_critical`, `si_safe`
et `mitigation` (présents aussi dans l'export CSV/JSON et l'API/MCP). Les garde-fous
λ/ε de §4 deviennent des cas particuliers de la maîtrise du risque (modes
régression/instabilité).

### §7bis — Port d'audit & déterminisme (CCOS)

Implémenté : le module cœur [`audit.rs`](../src/audit.rs) fournit le trait
`AuditLog` et `HashChainLog`, un **journal hash-chaîné SHA-256** (SHA-256 en
Rust pur, [`sha256.rs`](../src/sha256.rs), validé contre les vecteurs NIST) du
**même schéma que l'`EventLog` de CCOS** (`TraceEvent { sequence_number,
prev_hash, hash, event_type, payload }`). Chaque pas de `ℳ` est enregistré :

- **traçabilité** : `record(AuditEvent)` chaîne SHA-256 chaque pas ;
- **intégrité** : `verify()` détecte toute altération (lien rompu / contenu
  falsifié) ;
- **déterminisme** : même trajectoire ⇒ même `head()` (hash de tête
  reproductible) ;
- **replay** : `replay(from, to)` rejoue une sous-séquence ;
- **pont CCOS** : `to_ccos_json()` exporte un flux ingestable par
  `ExternalMemory::ingest_source` de CCOS pour la forensique avancée.

Branchement : `RSIAgent::with_audit(Box::new(HashChainLog::new()))`, puis
`audit_head()`, `audit_verify()`, `audit_len()`.

> **Pourquoi pas une dépendance directe à CCOS ?** Le port d'audit reproduit
> nativement le schéma hash-chaîné de CCOS (zéro dépendance) et **exporte au
> format CCOS**.

### Adaptateur CCOS (feature `ccos`) — **activé**

[`src/ccos_audit.rs`](../src/ccos_audit.rs) fournit `CcosAudit`, qui implémente
`AuditLog` en déléguant à l'`EventLog` de CCOS (`EventType::AgentAction` +
`EventPayload::Custom`, `verify_integrity`, `replay_events`).

```bash
cargo build --features ccos      # tire Memorithm/CCOS (sans async/TLS)
```
```rust
let agent = RSIAgent::demo(0).with_audit(Box::new(rsi::CcosAudit::new("session")));
```

Le dépôt CCOS a été corrigé (sous-module git nettoyé + licence PolyForm
Noncommercial + `tokio`/`reqwest` rendus optionnels), donc la dépendance git se
résout et compile **sans tirer l'async/TLS**. Le port natif `HashChainLog`
reste disponible (zéro dépendance) pour qui ne veut pas la dépendance CCOS.

## Effet observé

Invariants préservés (‖ΔS‖<λ, non-régression ε) ; l'agent progresse toujours ;
`SI_safe ≤ SI_global` ; routage et warm-start actifs. Tests : 44 (défaut) /
48 (forge+octasoma), aucun warning.

## Configuration

```rust
let agent = RSIAgent::new(state, substrate, surface, cfg, meta)
    .with_memory(Box::new(LinearContextMemory::new()))   // §A (ou OctaSomaMemory)
    .with_meta_interval(3)                                // §C
    .with_route_threshold(0.5)                            // §D
    .with_risk_config(RiskConfig::default());             // §7
```

---

## De-stylisation & robustesse (v0.10)

Réponse aux limites identifiées au bilan produit — tout en cœur, sans dépendance.

### #1 — Ancrer Ω/Φ sur des tâches réelles (`tasks.rs`)
- `TaskCorpus` : espace de tâches **ancré sur des données** (profils d'exigences
  sur (D,M,R,A,C,V), difficulté, importance), avec un corpus intégré
  d'archétypes et un chargement **depuis JSON** (`TaskCorpus::from_json/load`).
- `GroundedCapability` : compétence par **loi de Liebig** — Φ = minimum des
  marges sur les composantes *requises* (le maillon faible plombe la tâche),
  plus fidèle qu'un produit scalaire lissé.
- `IntelligenceSurface::from_corpus(&corpus)` remplace l'échantillonnage
  synthétique. La démo `rsi-full` tourne désormais sur ce corpus.

### #2 — Bruit Monte-Carlo vs ε (`surface.rs`, `dynamics.rs`)
- `IntelligenceSurface::si_global_stats` renvoie `(SI, erreur-type)` via la
  variance pondérée et la taille effective d'échantillon de Kish.
- `StabilityConfig.adaptive_epsilon` : la tolérance de non-régression devient
  `ε + z·stderr`, donc on ne pénalise pas une variation **sous le bruit**.

### #4 — Substrat mesuré sans GPU/Forge (`measured_substrate.rs`)
- `MeasuredSubstrate` : `SubstrateImprover` **natif** qui chronomètre un vrai
  kernel CPU (matmul tuilé vs naïf), balaie une grille de tuilages et calibre
  l'efficience logicielle mesurée — portable partout, zéro dépendance. (Forge
  reste disponible pour l'évolution de kernels SIMD/CUDA quand la toolchain est
  présente.)

### #5 — Composante D depuis une vraie source (`knowledge.rs`)
- Port `KnowledgeSource` + `CorpusKnowledge` : ingère de vrais documents (en
  mémoire ou un répertoire), extrait des **concepts distincts** et fait tendre
  `D` vers un niveau saturant. `RSIAgent::with_knowledge`.
- **`PapersKnowledge`** : adaptateur **PAPERS en sous-processus** — pour chaque
  papier (PDF/arXiv/URL/chemin) il invoque `papers extract <source>`
  (configurable : `with_binary`, `with_subcommand`, `with_args`, p. ex.
  `analyze --no-llm`), capture la sortie standard et en extrait les concepts.
  **Zéro dépendance** (PAPERS n'est pas lié comme crate — il tire
  scirust/ORT/CUDA) ; **dégradation propre** : binaire absent ou en échec ⇒
  repli sur le descripteur de la source (`last_degraded()`), donc compilable et
  testable ici sans PAPERS installé. Binaire résolu via `RSI_PAPERS_BIN`.

### #3 — Backends privés, cœur autonome (`agent.rs`)
- `RSIAgent::active_backends()` : introspection des backends réels branchés. Le
  cœur reste pleinement fonctionnel sans aucune feature (mémoire linéaire,
  substrat mesuré natif, audit hash-chaîné, méta aléatoire/CMA-ES, corpus).
