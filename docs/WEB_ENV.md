# Environnement Claude Code on the web — recréation & déblocage réseau

Ce dépôt se développe en sessions cloud (`claude.ai/code`). Cette page persiste
la configuration d'environnement requise et le plan des briques qui nécessitent
le réseau.

## Diagnostic

Si `cargo` ne peut récupérer aucune dépendance et que l'API Anthropic est
injoignable, l'environnement est sur **`None`** (aucun accès réseau sortant).

## Débloquer : passer en accès réseau « Trusted »

Le niveau **`Trusted`** (le défaut) autorise déjà tout ce dont RSI a besoin :

| Besoin | Domaine couvert par `Trusted` |
|---|---|
| backend Claude (`llm-claude`) | `api.anthropic.com` |
| dépendances cargo (`ureq`, `wide`, `wasmtime`, `tracing`…) | crates.io |
| git-deps (`Memorithm/scirust`, `forge`, `octasoma`, `CCOS`) | `github.com`, `codeload.github.com`, `api.github.com` |
| images Docker | Docker Hub / ghcr / gcr |

Niveaux : **None** (aucun) · **Trusted** (liste blanche) · **Full** (tout) ·
**Custom** (liste blanche perso, + option d'inclure les défauts).

### Étapes (interface web)
1. Icône **cloud** (nom de l'environnement) → sélecteur.
2. Survoler l'environnement → icône **réglages** (engrenage).
3. Sélecteur **Network access** → **Trusted** (ou **Custom** + cocher
   « Also include default list of common package managers »).
4. Renseigner les **Environment variables** (cf. `.env.example`) et le
   **Setup script** (cf. `scripts/web-setup.sh`).
5. Enregistrer. *(Changer le réseau ou le setup relance le cache au prochain
   démarrage — normal.)*

> Variables d'env : format `.env`, `KEY=value` sans guillemets. **Pas de coffre
> à secrets** — visibles par les éditeurs de l'environnement.

## Transport TLS pour Claude — IMPLÉMENTÉ ✅

Feature **`llm-claude-ureq`** : `UreqTransport` (au-dessus de `ureq`/rustls)
fournit un `ClaudeTransport` turnkey. Compilé et type-vérifié en environnement
réseau.

```rust
# #[cfg(feature = "llm-claude-ureq")] {
use rsi::llm::ClaudeClient;
let client = ClaudeClient::with_ureq(
    std::env::var("ANTHROPIC_API_KEY").unwrap(),
    "claude-sonnet-4-6",
);
// client implémente LlmClient → utilisable dans ascend_llm.
# }
```

Build : `cargo build --features llm-claude-ureq`. Pour un appel **réel**, définir
`ANTHROPIC_API_KEY` dans les variables d'environnement (cf. `.env.example`) —
non requis pour compiler/tester la logique (couverte hors-ligne par un transport
mock).

## Suite (ordre de valeur, une fois en Trusted)
1. Transport TLS Claude (ci-dessus).
2. Vraie git-dep `Memorithm/scirust` (remplacer la reconstruction vendorisée).
3. SIMD de `si_global` via la crate `wide` (sans `unsafe`).
4. Sandbox WASM (`wasmtime`) — 4ᵉ domaine (exécution réelle isolée).
5. Observabilité `tracing` + export Prometheus.

## Référence
Doc officielle : https://code.claude.com/docs/en/claude-code-on-the-web
