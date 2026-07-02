# Intégration LLM — guide

Comment brancher un LLM sur le moteur d'auto-amélioration de RSI, **sous
garde-fous**. Ce guide consolide la phase P1 (cf. `docs/P1_DESIGN_SPIKE.md`).

> **Contrat de sûreté fondamental — le LLM propose, le moteur dispose.**
> Le LLM ne produit que du **texte**. Le moteur **parse → valide (`safety_check`)
> → évalue en sandbox → adopte élitistement** (strictement meilleur ET sûr) ou
> rejette, sous garde-fous bornés. **Le LLM ne contrôle jamais** `max_iters`,
> `target`, `patience`, le budget, ni le critère d'adoption.

---

## 1. Deux topologies

| Topologie | Qui orchestre | Quand l'utiliser | Entrée |
|---|---|---|---|
| **Autonome** | RSI (client du LLM) | runs batch / non interactifs | `ascend_llm` + un [`LlmClient`] |
| **Interactive** | un LLM externe (client du serveur MCP) | agentique, pas-à-pas | outils MCP `rsi_*` |

Les deux respectent le même contrat de sûreté ; le serveur/moteur reste
autoritaire dans les deux cas.

---

## 2. Pièces du puzzle

- **`LlmClient`** (`src/llm.rs`) — backend de propositions interchangeable :
  - `OllamaClient` — modèle **local** par défaut (`llm-ollama`), client HTTP
    minimal sur `std::net`, **zéro dépendance**.
  - `ClaudeClient<T>` — API Anthropic (`llm-claude`) ; le transport HTTPS est
    **injecté** (trait `ClaudeTransport`) car `std` n'a pas de TLS.
  - `MockLlmClient` — déterministe, pour tests / dév hors-ligne.
- **`LlmRefineTask`** (`src/llm.rs`) — le *domaine* : `describe` (prompt),
  `parse_proposals`, `score_heldout` (anti-Goodhart), `safety_check`.
  Domaines fournis : `SymbolicSynthesis` (`src/synthesis.rs`), `ConfigTuning`
  (`src/tuning.rs`), `PromptOpt` (`src/prompt.rs`), et `WasmSynthesis`
  (`src/wasm_domain.rs`, feature `wasm` — **exécution réelle** de code candidat
  en bac à sable `wasmi` : fuel + zéro import host).
- **`LlmGuard`** — bornes + **budget** (`max_llm_calls`, `max_wall_clock`) +
  garde-fou **anti-overfitting** (`max_overfit_gap`).
- **`ascend_llm`** — le pilote élitiste (chemin autonome).
- **Outils MCP** (`src/bin/rsi_mcp.rs`, `src/api.rs`) — `rsi_refine_new`,
  `rsi_incumbent`, `rsi_evaluate`, `rsi_propose` (chemin interactif).

---

## 3. Chemin autonome (`ascend_llm`)

```rust
use rsi::llm::{ascend_llm, LlmGuard, MockLlmClient};
use rsi::synthesis::SymbolicSynthesis;
use rsi::ascent::RefineTask;

// 1) un domaine (ici : synthèse symbolique de x²+1, avec held-out réservé)
let mut task = SymbolicSynthesis::from_target_split(|x| x * x + 1.0, -3.0, 3.0, 30, 1);

// 2) un backend LLM (Mock ici ; OllamaClient/ClaudeClient en production)
let client = MockLlmClient::new(|prompt, _k| {
    // un vrai LLM lirait `prompt` (= task.describe(incumbent)) et proposerait
    // des variantes ; ici on script un chemin déterministe.
    vec!["x".into(), "x*x".into(), "x*x + 1".into()]
});

// 3) les garde-fous (bornes + budget) — fixés par L'HÔTE, jamais par le LLM
let guard = LlmGuard { target: Some(0.9), patience: 3, max_iters: 20, ..LlmGuard::default() };

// 4) la boucle élitiste : adoption ⟺ sûr ET strictement meilleur
let seed = task.seed_candidate();
let (best, report) = ascend_llm(&mut task, seed, &client, &guard);
assert!(report.is_monotone());          // non-régression de l'incumbent
println!("{} (train={:.3}, held-out={:.3})", /* best */ task.score(&best),
         report.best(), report.best_heldout());
```

### Backend Ollama (local, turnkey)

```rust
# #[cfg(feature = "llm-ollama")] {
use rsi::llm::OllamaClient;
let client = OllamaClient::new("llama3.2");           // 127.0.0.1:11434 par défaut
// .with_endpoint("127.0.0.1", 11434).with_timeout(...)
// .with_num_predict(4096)   // plafond de génération (défaut 4096)
// .with_num_ctx(16384)      // fenêtre de contexte (défaut 16384 — le défaut
//                           // serveur de 4096 tronque les prompts réels)
# }
```

### Connexion automatique (le LLM au choix de l'utilisateur, sans flags)

Plus besoin de connaître le nom exact d'un modèle : RSI **découvre** ce qui
est installé et choisit tout seul.

```rust
# #[cfg(feature = "llm-ollama")] {
use rsi::llm::{ollama_installed_models, pick_model};
let installed = ollama_installed_models("127.0.0.1", 11434,
                                        std::time::Duration::from_secs(10))?;
// préférence exacte ou par préfixe ("qwen3-coder" → "qwen3-coder:30b"),
// sinon heuristique : meilleur modèle de CODE installé (embeddings exclus)
let model = pick_model(&installed, std::env::var("RSI_LLM_MODEL").ok().as_deref());
# }
```

Côté CLI (`rsi-dgm`), c'est le **défaut** (`--backend auto`) :

| Priorité | Source                          | Exemple                                |
|----------|---------------------------------|----------------------------------------|
| 1        | flags CLI                       | `--backend ollama --model qwen3-coder:30b` |
| 2        | env `RSI_LLM`                   | `RSI_LLM=ollama:qwen3-coder:30b`, `RSI_LLM=claude`, `RSI_LLM=auto` |
| 3        | env `RSI_LLM_MODEL` (préférence)| `RSI_LLM_MODEL=qwen3-coder`            |
| 4        | auto : sonde Ollama (`/api/tags`), meilleur modèle de code ; sinon Claude si `ANTHROPIC_API_KEY` (feature `llm-claude-ureq`) ; sinon erreur explicite | — |

Le serveur MCP (`rsi_dgm_start`) applique la même résolution (`model` de
l'appel, sinon `RSI_LLM_MODEL`, sinon découverte ; hôte via
`RSI_OLLAMA_HOST`/`RSI_OLLAMA_PORT`).

### Backend Claude (transport injecté)

`std` n'offrant pas de TLS, le transport HTTPS est isolé derrière le trait
`ClaudeTransport`. **Turnkey** (feature `llm-claude-ureq`, au-dessus de
`ureq`/rustls) :

```rust
# #[cfg(feature = "llm-claude-ureq")] {
use rsi::llm::ClaudeClient;
let client = ClaudeClient::with_ureq(
    std::env::var("ANTHROPIC_API_KEY").unwrap(),
    "claude-sonnet-4-6",
);
# }
```

Ou **transport maison** (feature `llm-claude`, sans dépendance) :

```rust
# #[cfg(feature = "llm-claude")] {
use rsi::llm::{ClaudeClient, ClaudeTransport};
struct MyTls; // au-dessus de votre pile HTTPS
impl ClaudeTransport for MyTls {
    fn post_json(&self, url: &str, headers: &[(String, String)], body: &str)
        -> Result<String, String> { /* POST HTTPS → corps réponse */ todo!() }
}
let client = ClaudeClient::new(MyTls, std::env::var("ANTHROPIC_API_KEY").unwrap(),
                               "claude-sonnet-4-6");
# }
```

---

## 4. Chemin interactif (outils MCP)

Le serveur `rsi-mcp` parle JSON-RPC 2.0 sur stdin/stdout. Un LLM externe
(p. ex. dans un client MCP) pilote la boucle via quatre outils :

| Outil | Effet |
|---|---|
| `rsi_refine_new` | crée une session (`domain`: `synthesis` \| `tuning`) |
| `rsi_incumbent` | lit l'incumbent (texte, score train, score held-out, compteurs) |
| `rsi_evaluate` | évalue un candidat **sans l'adopter** (sonde) |
| `rsi_propose` | soumet des candidats ; le serveur adopte si **strictement meilleur ET sûr** |

### Exemple — synthèse symbolique

```jsonc
// → créer la session
{"jsonrpc":"2.0","id":1,"method":"tools/call",
 "params":{"name":"rsi_refine_new","arguments":{"id":"s","target":"quadratic"}}}
// ← incumbent initial : "0.000", score -0.01, 9 cas held-out

// → sonder un candidat (sans adoption)
{"jsonrpc":"2.0","id":2,"method":"tools/call",
 "params":{"name":"rsi_evaluate","arguments":{"id":"s","candidate":"x*x + 1"}}}
// ← {"parseable":true,"score":0.95,"would_adopt":true,...}

// → proposer (le serveur décide)
{"jsonrpc":"2.0","id":3,"method":"tools/call",
 "params":{"name":"rsi_propose","arguments":{"id":"s","proposals":["x","x*x + 1"]}}}
// ← adopté "((x * x) + 1.000)" (score 0.95) ; "x" rejeté (rejected_worse)
```

### Exemple — réglage de configuration

```jsonc
{"jsonrpc":"2.0","id":1,"method":"tools/call",
 "params":{"name":"rsi_refine_new","arguments":{"id":"g","domain":"tuning"}}}

{"jsonrpc":"2.0","id":2,"method":"tools/call",
 "params":{"name":"rsi_propose","arguments":{"id":"g","proposals":[
   "{\"top_k\":50,\"chunk\":1036,\"threshold\":0.4}",
   "{\"top_k\":9999,\"chunk\":10,\"threshold\":5}"]}}}
// ← 1er adopté (score 1.000) ; 2e rejeté : "top_k=9999 hors bornes [1,100]"
```

Le **serveur applique `safety_check` et l'élitisme** : un candidat « meilleur en
score mais hors bornes » est **refusé** ; un candidat moins bon est rejeté.

### En CLI (`rsi-refine`)

Pour piloter une session depuis le shell (sans client MCP), l'état est persisté
dans un fichier entre invocations (via `refine_save`/`refine_load`) :

```bash
rsi-refine new --domain prompt                 # crée .rsi-refine.json
rsi-refine eval "Résume étape par étape, exemple, format JSON."   # sonde (sans adopter)
rsi-refine propose "Ignore previous instructions" "Résume … format JSON."
#   [rejected_unsafe] marqueur d'injection détecté : 'ignore previous'
#   [adopted] Résume … format JSON.
rsi-refine show                                # incumbent persisté
```

---

## 5. Garde-fous

### Budget (coût/latence)

`LlmGuard` borne la boucle autonome : `max_iters`, `max_llm_calls`,
`max_wall_clock`. En local (Ollama) le coût monétaire est ~0 ; ce sont les
**appels** et le **temps mur** qui comptent. Avec Claude, le coût/tokens
redeviennent pertinents — mêmes bornes.

### Anti-Goodhart (held-out)

`score` (qui pilote l'adoption) évalue sur le **train** ; `score_heldout` mesure
la généralisation sur un jeu **jamais vu par la boucle** (~30 % réservés, gelés).
`LlmGuard.max_overfit_gap` stoppe la boucle si l'écart train/held-out explose.
**Seul le held-out** doit servir de chiffre rapporté.

### Sûreté de domaine

Chaque domaine déclare ses interdits via `safety_check` (taille d'AST bornée
pour la synthèse ; bornes de schéma pour le tuning). Un candidat qui échoue est
**rejeté quel que soit son score**.

---

## 6. Anti-patterns (à refuser par conception)

- ❌ Laisser le LLM fixer `max_iters` / `target` / `patience` / le budget.
- ❌ Adopter sur la foi du LLM sans `score` côté serveur (« il a dit que c'est
  mieux »).
- ❌ Exécuter le texte du LLM comme du code (le domaine **parse** et **interprète
  en sandbox** ; jamais d'`eval`/sous-processus sur la sortie du modèle).
- ❌ Rapporter le score **train** comme résultat (utiliser le **held-out**).
- ❌ Donner au LLM le contrôle du `line search` / du critère d'adoption.

---

## 7. Ajouter un domaine

Implémenter `RefineTask` (score + refine) puis `LlmRefineTask` (describe /
parse_proposals / score_heldout / safety_check) pour son type `Cand`. Voir
`src/synthesis.rs` (Cand = `Expr`) et `src/tuning.rs` (Cand = `TuneConfig`,
objet JSON) comme modèles. Pour l'exposer via MCP, ajouter une implémentation de
`RefineDomain` (object-safe) dans `src/api.rs` et un bras dans `refine_new`.

---

## 8. Références

- Décisions de conception : `docs/P1_DESIGN_SPIKE.md`
- Cœur LLM : `src/llm.rs`
- Domaines : `src/synthesis.rs`, `src/tuning.rs`
- Façade / outils : `src/api.rs`, `src/bin/rsi_mcp.rs`
- Garde-fous de sûreté du cœur : `docs/SAFETY.md`
