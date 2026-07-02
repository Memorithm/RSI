# RSI — Recursive Self-Improvement (Rust)

[![CI](https://github.com/CHECKUPAUTO/RSI/actions/workflows/ci.yml/badge.svg)](https://github.com/CHECKUPAUTO/RSI/actions/workflows/ci.yml)

Implémentation **100 % Rust** (std-only, aucune dépendance externe) du
**système mathématique d'auto-amélioration récursive** — formulation
géométrique unifiée, v9.

Le système modélise un **agent cognitif** dont la *surface de compétence*
`Σ_I(t)` se déforme sous l'effet de l'apprentissage, du substrat
matériel/logiciel et d'une **méta-optimisation récursive**, le tout sous des
**garde-fous de stabilité**.

```
SI_global : 0.1385 → 0.6760   (+388 %)   en 120 pas
goulot : cognitif (0 %) ──────────────► substrat (100 %)
```

> **Feuille de route** : objectifs et chantiers à venir — dont l'**epic phare
> « Loop Engineering »** (moteur de boucle : pilote/arrêt, convergence,
> checkpoint/replay, disjoncteurs de sûreté, multi-échelles, pilotage MCP,
> swarm, banc d'essai) — dans [`docs/ROADMAP.md`](docs/ROADMAP.md).

## 🚀 Installation en une commande

Connecte RSI à ton agent IA (openclaw, hermes-agent, soullink, ou tout client
MCP via `RSI_CONNECT_TARGETS`) — aucune configuration manuelle :

```bash
git clone https://github.com/CHECKUPAUTO/RSI && cd RSI && ./install.sh
```

C'est tout. `install.sh` compile le serveur MCP et l'enregistre auprès de tes
agents. Au prochain démarrage, ils disposent des outils `rsi_*` (`rsi_create`,
`rsi_run`, `rsi_state`, …). Équivalent : `make install`.

> Pas d'agent sous la main ? Teste le moteur directement :
> `cargo run --release --bin rsi-demo`

### Conteneur / build reproductible

Le **cœur** (features par défaut) est *std-only* et **sans dépendance externe** :
il se package en binaires statiques sans accès réseau.

```bash
# Image Docker minimale (build musl statique → image `scratch`)
docker build -t rsi:latest .
docker run --rm -i rsi:latest          # serveur MCP (stdio)
docker run --rm --entrypoint /usr/local/bin/rsi-demo rsi:latest

# Build reproductible via Nix (flake)
nix build               # → ./result/bin/{rsi-mcp,rsi-demo,…}
nix run                 # lance rsi-mcp
nix develop             # shell outillé (cargo, clippy, rustfmt)
```

> **Distribution.** RSI dépend de 4 crates git privées (forge/octasoma/ccos/
> scirust) **optionnelles** : il n'est donc **pas publié sur crates.io** (le
> registre exige une version pour toute dépendance). La distribution se fait par
> **source + Docker/Nix** ci-dessus. `rsi-full` (features git) sort de ce
> périmètre hors-ligne.

## Correspondance équations ↔ code

| Section | Équation | Module |
|---------|----------|--------|
| **§1** Surface d'intelligence | `Σ_I(t)={(x,C_réel)}`, `C_réel=min(Φ_x(S),g_x(P_eff))`, `SI_global=∫_Ω C_réel dμ` | [`src/surface.rs`](src/surface.rs) |
| **§2** Vecteur d'état cognitif | `S=(D,M,R,A,C,V)` | [`src/state.rs`](src/state.rs) |
| **§3** Substrat physique/logiciel | `P_eff=σ(HᵀAH)·σ(OᵀBO)·σ(HᵀCO)` | [`src/substrate.rs`](src/substrate.rs) |
| **§4** Dynamique + stabilité | `dS/dt=η·[L+E+U]−P`, `‖ΔS‖<λ`, `SI(t+1)≥SI(t)−ε` | [`src/dynamics.rs`](src/dynamics.rs) |
| **§5** Boucle discrète + méta | `S_{t+1}=S_t+ℳ(…)+ΔS_appr`, `ℳ_{t+1}=argmax_ℳ SI_global(ℳ(S_t))` | [`src/meta.rs`](src/meta.rs) |
| **§6** Forme compacte (onde) | `Σ_I(t+1)=Σ_I(t)+η·ℳ−P` | [`src/agent.rs`](src/agent.rs) |

> **Portée de l'invariant de non-régression.** `SI(t+1) ≥ SI(t) − ε` est garanti
> par line search sur le **pas combiné** (méta ℳ + apprentissage ΔS_appr), *sauf*
> lorsqu'un **override de sûreté explicite** se déclenche (p. ex. `trust_floor`
> anti-wireheading, qui abaisse délibérément `P_eff`) : dans ce cas la régression
> est assumée au nom de la sûreté. Hors override, l'invariant est appliqué activement
> (cf. `RSIAgent::step`, §4bis), pas seulement émergent.

## Modèle

1. **Espace des tâches** `(Ω,𝒜,μ)` : Ω est échantillonné une fois par
   Monte-Carlo (profils de besoins via Dirichlet sur les 6 composantes),
   μ pondère l'importance des tâches. Le même échantillon sert à toutes les
   évaluations ⇒ `SI_global` comparable d'un pas à l'autre.

2. **Compétence réelle** `C_réel = min(Φ, g)` : minimum entre la compétence
   *cognitive* `Φ_x(S)` (ce que l'agent sait faire) et le plafond *physique*
   `g_x(P_eff)` (ce que le substrat permet). C'est un **goulot
   d'étranglement** : être limité par le cerveau OU par le matériel.

3. **Efficacité multiplicative** `P_eff` : un matériel puissant est inutile
   sans logiciel capable de l'exploiter — d'où le produit de trois sigmoïdes,
   dont un terme de **couplage** `σ(HᵀCO)`.

4. **Dynamique contrainte** : le pas d'apprentissage brut `η·[L+E+U]−P` est
   borné en amplitude (`‖ΔS‖<λ`, projection radiale) puis atténué par
   *line search* tant qu'il ferait régresser `SI_global` de plus de `ε`.

5. **Méta-révision récursive** : à chaque pas, l'agent explore un voisinage de
   stratégies d'auto-modification `ℳ` et retient celle qui **maximise le
   `SI_global` projeté** — y compris en réécrivant son propre logiciel `O`
   (auto-amélioration du substrat, cœur du RSI).

## Utilisation

```bash
# tests (unitaires + intégration + doctest)
cargo test

# simulation de démonstration : cargo run -- [n_pas] [graine]
cargo run --release --bin rsi-demo -- 120 2026
```

### Comme bibliothèque

```rust
use rsi::RSIAgent;

let mut agent = RSIAgent::demo(2026);
let start = agent.si_global();
let reports = agent.run(120);          // trajectoire complète
let end = reports.last().unwrap();
println!("SI_global : {start:.4} → {:.4}", end.si_global);
```

Pour assembler un agent sur mesure :

```rust
use rsi::*;

let mut rng = Rng::new(7);
let state     = CognitiveState::random(Dims::uniform(6), &mut rng, 0.08);
let substrate = Substrate::default_with(/*hw*/ 4, /*sw*/ 4, &mut rng);
let surface   = IntelligenceSurface::sample(1024, &mut rng);
let meta      = MetaOptimizer::new(/*candidats*/ 48, /*exploration*/ 0.12, 999);

let mut agent = RSIAgent::new(state, substrate, surface,
                              StabilityConfig::default(), meta);
agent.run(150);
```

## Interprétation des sorties

Chaque [`StepReport`](src/agent.rs) expose :

- `si_global` — volume sous Σ_I (intelligence globale, ∈ [0,1]) ;
- `delta_si` — variation par pas (garantie `≥ −ε`) ;
- `p_eff` — efficacité du substrat (monte si l'agent améliore son logiciel) ;
- `appr.delta_norm` — `‖ΔS‖` du pas d'apprentissage (garanti `< λ`) ;
- `frac_limited_by_substrate` — fraction des tâches bridées par le matériel
  plutôt que par le cognitif (le **basculement** vers 100 % est la signature
  d'un agent devenu substrate-limited) ;
- `capabilities` — niveaux `(D,M,R,A,C,V)`.

## Extensions

- **Modèles `Φ_x` / `g_x` configurables** via les traits `CapabilityModel` /
  `CeilingModel` (`surface.rs`) — branchez n'importe quelle loi de
  compétence/plafond sans toucher au reste.
- **Méta-optimiseur sep-CMA-ES** (`cma.rs`) en plus de la recherche aléatoire,
  derrière le trait `MetaSearch` ; sélectionnable par `optimizer: random|cma`.
- **Export CSV / JSON** de la trajectoire (`report.rs`, flags `--csv`/`--json`).
- **API** orientée commandes JSON (`api.rs`, `RsiApi`).
- **Serveur MCP** (`rsi-mcp`) pour piloter le système depuis un agent IA / LLM.
- **Moteur d'auto-amélioration piloté par LLM** (`llm.rs`, `ascent.rs`) — « le
  LLM propose, le moteur dispose » : le LLM ne produit que du texte, le moteur
  parse, valide (`safety_check`), évalue en sandbox et **adopte élitistement**
  (strictement meilleur ET sûr), sous garde-fous bornés (`LlmGuard` : budget,
  anti-overfitting). Backends interchangeables : **Ollama** local (défaut),
  **Claude** (transport injecté), mock déterministe. Trois domaines : synthèse
  symbolique, configuration (JSON), prompts (rejet d'injection). Pilotable en
  **autonome** (`ascend_llm`), via **MCP** (`rsi_refine_new`/`incumbent`/
  `evaluate`/`propose`, multi-domaine, checkpoint/resume) ou en **CLI**
  (`rsi-refine`). Voir [`docs/LLM_INTEGRATION.md`](docs/LLM_INTEGRATION.md) et
  [`docs/SAFETY.md`](docs/SAFETY.md).
- **Auto-connexion** (`rsi-connect` + `scripts/auto-connect.sh`) aux runtimes
  d'agents (openclaw, hermes-agent, soullink, ou n'importe quel autre via
  `RSI_CONNECT_TARGETS="nom=chemin,…"`) sans intervention humaine. Côté LLM,
  **connexion automatique** : `rsi-dgm` (défaut `--backend auto`) découvre les
  modèles Ollama installés (`/api/tags`) et choisit le meilleur modèle de code
  (`RSI_LLM`/`RSI_LLM_MODEL` pour préférer) ; la boucle DGM est aussi pilotable
  depuis un agent via MCP (`rsi_dgm_start`/`rsi_dgm_status`, dry-run strict —
  la promotion reste un acte humain en CLI).
- **Addons** (`addons.rs`, [`docs/ADDONS.md`](docs/ADDONS.md)) — capacités
  enfichables par binaire externe borné + adaptateur std-only (détection à
  l'exécution, dégradation propre, le gate reste autoritaire). Premier addon :
  **papers-agent** (PAPERS V2) — apprendre de la littérature pour s'améliorer :
  connaissances → composante D, techniques des papiers → **objectifs DGM
  directifs** (`rsi-scholar` : papier → objectifs → preuves empiriques →
  rapport), recherche sémantique → contexte du proposeur. Étude :
  [`docs/PAPERS_SYNERGY.md`](docs/PAPERS_SYNERGY.md).
- **Backends réels** (features optionnelles, cœur sans dépendance par défaut) —
  voir l'étude : [`docs/INTEGRATION_STUDY.md`](docs/INTEGRATION_STUDY.md) :
  - `forge` — `ℳ` réel (recherche évolutionnaire exécutée, fitness `SI_global`)
    **et** `P_eff` réel (efficience logicielle mesurée sur un vrai kernel) ;
  - `octasoma` — composante `C` réelle (mémoire vectorielle fractale, k-NN).

  ```bash
  cargo build --features forge              # ℳ + P_eff réels (Forge)
  cargo build --features octasoma           # mémoire C réelle (OctaSoma)
  cargo build --features "forge octasoma"   # tout
  ```
- **§7 — Modes de défaillance & criticité (AMDEC/FMECA)** : module
  `criticality.rs` (cœur, sans dépendance) — `RPN`, `Risk_global`, intelligence
  ajustée au risque `SI_safe = SI_global − κ·Risk_global`, garde-fou de
  criticité et **routage par criticité**. Chaque `StepReport` expose
  `risk_global`, `max_rpn`, `most_critical`, `si_safe`. Optimisations
  d'intégration (mémoire active, canal substrat unifié, campagnes amorties) :
  voir [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md).
- **§7bis — Audit & déterminisme** : `audit.rs` (cœur) — journal **hash-chaîné
  SHA-256** (SHA-256 Rust pur, schéma `EventLog` de CCOS) rendant chaque pas de
  `ℳ` **traçable, vérifiable et rejouable** (`with_audit`, `audit_head/verify`),
  avec export ingestable par CCOS. Feature `ccos` : délègue au vrai `EventLog`
  de CCOS (`cargo build --features ccos`, sans async/TLS).
- **Ancrage & robustesse (v0.10, cœur sans dépendance)** :
  - `tasks.rs` — **corpus de tâches réel** (Ω ancré, chargeable JSON) + compétence
    par **loi de Liebig** (`GroundedCapability`) ; `IntelligenceSurface::from_corpus`.
  - `knowledge.rs` — port `KnowledgeSource` qui **alimente `D`** depuis une vraie
    source : `CorpusKnowledge` (documents) et **`PapersKnowledge`** (ingestion de
    papiers via **PAPERS en sous-processus**, sans dépendance, dégradation propre) ;
    `RSIAgent::with_knowledge`.
  - `measured_substrate.rs` — `P_eff` **mesuré nativement** : `MeasuredSubstrate`
    (kernel CPU/tuilage cache chronométré) et **`SimdMeasuredSubstrate`**
    (efficience **SIMD/vectorisation** réelle de l'hôte) — sans Forge ni GPU.
  - `omega_tasks.rs` — **Ω concret** : banc de tâches réelles *nommées* (profil
    explicite `(D,M,R,A,C,V)`, demande, μ) branché sur les traits `Φ`/`g`
    existants ; rapport par tâche montrant le goulot (`Φ_x` cognitif vs `g_x`
    substrat). Démo : `cargo run --release --example omega_real`.
  - `hw_probe.rs` — **sonde matérielle réelle** (CPU/mémoire `/proc`, GPU sysfs
    Tegra ou `nvidia-smi`) qui ancre le vecteur `H` du substrat sur l'état
    machine. Dégradation propre (GPU absent → neutre). Idéal **Jetson** :
    `cargo run --release --example jetson_substrate`.
  - **ε adaptatif** au bruit Monte-Carlo (`si_global_stats`, `adaptive_epsilon`).
  - introspection : `RSIAgent::active_backends()`.
- **Validation empirique** : banc d'ablation `cargo run --release --bin rsi-ablate`
  (cœur pur) + corpus élargi — voir [`docs/VALIDATION.md`](docs/VALIDATION.md).
- **Loop engineering (L1–L9 complet)** : moteur de boucle RSI — pilote
  `run_until(LoopConfig)` avec arrêt motivé (budget/cible/**plateau**/divergence,
  `ConvergenceDetector`), **checkpoint/reprise** (`snapshot`/`restore`),
  **disjoncteurs de sûreté** (`breaker_rpn` + rollback), **cadences
  multi-échelles** (`LoopSchedule`/`MetaMeta`), **observateur + veto HITL**
  (`LoopObserver`), pilotage **MCP** (`rsi_run_until`), **swarm** parallèle
  (`run_swarm`), et banc d'essai `rsi-loopbench`. Voir [`docs/ROADMAP.md`](docs/ROADMAP.md).

### Démo « tout intégré »

Un agent réunissant les quatre backends + la criticité §7 :

```bash
cargo run --release --bin rsi-full --features "forge octasoma ccos" -- 45 2026
cargo run --release --bin rsi-full --features "forge octasoma ccos" -- compare 30 2026
```

Affiche la trajectoire (SI, SI_safe, P_eff, risque, mode critique, **réponse de
sûreté active**, goulot), vérifie l'intégrité du journal d'audit CCOS, rappelle
un contexte via OctaSoma et exporte la trajectoire (CSV/JSON + **graphe SVG**).
Le mode `compare` oppose un agent « nu » (cœur) à l'agent « tout intégré ».
Source : [`src/bin/rsi_full.rs`](src/bin/rsi_full.rs).

### Intégration agent IA / LLM

```bash
cargo build --release --bins
./scripts/auto-connect.sh            # build + enregistrement MCP automatique
./target/release/rsi-mcp             # serveur MCP (JSON-RPC 2.0 sur stdio)
```

Voir le guide complet : [`docs/INTEGRATION.md`](docs/INTEGRATION.md)
(API, outils MCP, schémas, boucle d'auto-amélioration côté LLM, auto-connexion).

**Brancher un LLM sur le moteur d'ascension** (Ollama local / Claude, boucle
autonome ou interactive via MCP, sous garde-fous) :
[`docs/LLM_INTEGRATION.md`](docs/LLM_INTEGRATION.md).

**Garde-fous, garanties et limites** (ce qui est garanti vs hypothèse de
modèle) : [`docs/SAFETY.md`](docs/SAFETY.md).

**Recréer l'environnement Claude Code on the web** (accès réseau « Trusted »,
variables, setup script, déblocage du transport TLS Claude) :
[`docs/WEB_ENV.md`](docs/WEB_ENV.md).

### Paper scientifique

Formalisation, implémentation et résultats expérimentaux :
[`paper/rsi.md`](paper/rsi.md) (Markdown) et [`paper/rsi.tex`](paper/rsi.tex)
(source LaTeX).

## Agent d'auto-amélioration (génère → évalue → améliore)

RSI peut **générer des algorithmes candidats puis les améliorer** en boucle
élitiste bornée — *sans jamais régresser*. Ce travail consomme le contrat
documenté de `scirust-rsi` (RefineTask / Guard / ascend / Report).

```bash
cargo run --release --example self_improve   # synthèse symbolique : reconstruit x²+1
```

Exemple de sortie : la fitness monte `-0.01 → 0.30 → 0.41 → 0.89` puis se
stabilise (patience), 100 % des cas de test réussis, `is_monotone() == true`.

- **Évaluateur** (`SymbolicSynthesis::score`) : fraction de cas de test réussis
  (|sortie − cible| ≤ tolérance) **moins** une pénalité de complexité (taille
  de l'AST).
- **Générateur** (`refine`) : meilleure de `λ` mutations déterministes (révision
  « critiquée », façon 1+λ).
- **Pilote** ([`ascent::ascend`] + [`Guard`]) : `max_iters`, `patience`,
  `target`, `min_delta`.

### Contrat de sûreté
- **Sandbox** : le « code » candidat est un **AST arithmétique évalué par mon
  propre interpréteur** (`Expr::eval`) — **jamais** compilé ni exécuté comme du
  code arbitraire, **aucun sous-processus**. L'évaluateur ne renvoie que des
  nombres ; le moteur ne voit jamais le code.
- **Non-régression (élitisme)** : une révision n'est adoptée que si sa fitness
  dépasse *strictement* l'incumbent ⇒ `Report::is_monotone()` vrai.
- **Bornage / terminaison** : au plus `max_iters` itérations.
- **Déterminisme** : entièrement piloté par une graine reproductible.

> **Statut d'intégration `scirust-rsi`.** Deux modes coexistent :
> - **Stand-in hors-ligne** (actif par défaut, testé) : [`src/ascent.rs`](src/ascent.rs)
>   reproduit fidèlement le contrat (`RefineTask`, `Guard`, `ascend`,
>   `Report::is_monotone`). C'est ce que lance `--example self_improve`.
> - **Moteur réel** (prêt, à activer) : [`src/scirust_bridge.rs`](src/scirust_bridge.rs)
>   implémente le **vrai** trait `scirust_rsi::refine::RefineTask`
>   (`type Solution`, `initial`/`score → Fitness`/`refine(&self, _, &mut StdRng)`)
>   et pilote la boucle avec `SelfRefiner::new(seed).run(&task, &Guard)`,
>   conformément à `scirust-rsi/INTEGRATION.md`.
>
> Le moteur réel est consommé en **dépendance git amont**
> (`scirust-rsi = { git = "https://github.com/CHECKUPAUTO/scirust" }`) — validé
> de bout en bout : `cargo test --features scirust` compile le vrai crate et
> passe les 131 tests sans modifier le bridge. Activation (réseau requis) :
> `cargo run --features scirust --release --example self_improve_real`. Détails :
> [`SCIRUST_ACTIVATION.md`](SCIRUST_ACTIVATION.md).

## Auto-amélioration **empirique** du code (Darwin–Gödel / STOP)

Là où l'agent ci-dessus améliore des objets abstraits (expressions, configs,
prompts) que le moteur *interprète*, le module [`src/dgm.rs`](src/dgm.rs) ajoute
la variante **empirique sur du code source réel** — port natif **std-only** du
crate [`soul-rsi`](https://github.com/CHECKUPAUTO/soul-rsi) :

> propose un patch `find → replace` ▸ l'évalue dans une **copie isolée** du dépôt
> (`cargo build` + `cargo test`) ▸ ne le **garde que s'il est prouvé meilleur**
> (compile ▸ non-régression de tests ▸ score) ▸ l'**archive** comme tremplin
> réutilisable ▸ recommence.

```bash
cargo run --release --example dgm_selfimprove    # boucle jouet, déterministe, hors-ligne

# Sur un dépôt RÉEL (build+test cargo), backend Ollama local, SÛR par défaut :
cargo run --release --features llm-ollama --bin rsi-dgm -- \
    /chemin/du/depot --goal "accélérer X sans casser les tests" \
    --allow src/foo.rs,src/bar.rs --steps 20
# DRY-RUN : l'arbre vivant n'est jamais écrit. Ajouter --promote pour appliquer
# le meilleur variant tout-au-vert (avec sauvegarde réversible). Claude :
# --features llm-claude-ureq puis --backend claude (ANTHROPIC_API_KEY).
```

- **STOP** (Zelikman 2023) : le proposeur est interchangeable
  ([`Proposer`]) ; [`LlmProposer`] le branche sur n'importe quel backend RSI via
  [`LlmCodeModel`] (Ollama, Claude…). La boucle ne fait **jamais** confiance à
  l'auto-évaluation du modèle.
- **Darwin Gödel Machine** (Zhang 2025) : [`Archive`] ouverte, sélection de
  parent qualité × nouveauté, acceptation **empirique**.
- **Reflexion** (Shinn 2023) : les rejets récents sont re-injectés au proposeur.

**Sûreté** (cf. [`docs/SAFETY.md`](docs/SAFETY.md) §5bis) : liste blanche de
fichiers éditables, patch exact **non ambigu** (motif unique), snapshot jetable,
sous-processus `cargo` **bornés** (timeout + sortie plafonnée), et l'arbre vivant
n'est mué que par [`promote_to_live`] (gardé tout-au-vert, sauvegarde réversible).
IDs de variantes déterministes (hash de lignée) ⇒ archive **reproductible**.
⚠️ `CargoEvaluator` exécute du code réel : à n'utiliser que sur du **code de
confiance** (ce n'est pas un bac à sable syscall ; pour du code non fiable,
préférer le domaine WASM).

## Architecture

```
src/
├── lib.rs          exports + table de correspondance équations/modules
├── rng.rs          PRNG xoshiro256** + Dirichlet/normale (std-only)
├── linalg.rs       vecteurs, matrices denses, formes quadratiques, σ
├── state.rs        §2  S = (D,M,R,A,C,V)
├── substrate.rs    §3  P_eff = σ(HᵀAH)·σ(OᵀBO)·σ(HᵀCO)
├── surface.rs      §1  Σ_I, C_réel = min(Φ,g), SI_global + traits Φ/g
├── omega_tasks.rs  §1  Ω concret : banc de tâches réelles nommées (Φ vs g par tâche)
├── hw_probe.rs     §3  sonde matérielle réelle (CPU/mém/GPU) → vecteur H du substrat
├── dynamics.rs     §4  dS/dt + contraintes ‖ΔS‖<λ et non-régression ε
├── meta.rs         §5  ℳ, trait MetaSearch, recherche aléatoire + CMA-ES
├── cma.rs          §5  sep-CMA-ES (covariance diagonale)
├── agent.rs        §5/§6  boucle discrète complète
├── dgm.rs          auto-amélioration empirique du code (Darwin–Gödel/STOP, port soul-rsi)
├── json.rs         (dé)sérialisation JSON std-only
├── report.rs       export CSV / JSON de la trajectoire
├── api.rs          façade RsiApi (commandes JSON in/out)
├── main.rs         binaire de démonstration (rsi-demo)
└── bin/
    ├── rsi_mcp.rs      serveur MCP (JSON-RPC 2.0 / stdio)
    ├── rsi_dgm.rs      CLI DGM/STOP sur dépôt réel (dry-run sûr, --promote opt-in)
    └── rsi_connect.rs  auto-enregistrement MCP (openclaw, hermes-agent, …)
scripts/
└── auto-connect.sh  build + connexion MCP automatique
docs/
└── INTEGRATION.md   guide API / MCP / LLM / auto-connexion
paper/
├── rsi.md           paper scientifique (anglais)
└── rsi.tex          source LaTeX
tests/
└── integration.rs   trajectoire stable & croissante, déterminisme, substrat
```

## Licence

**Double licence** (voir [LICENSING.md](LICENSING.md)) :

- **Non commercial** — particuliers, chercheurs, éducation, gouvernement :
  gratuit sous [PolyForm Noncommercial 1.0.0](LICENSE.md).
- **Commercial** — licence séparée : **contact@checkupauto.fr**.
