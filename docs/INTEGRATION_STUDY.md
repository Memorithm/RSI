# Étude de faisabilité — intégrer CCOS, OctaSoma, PAPERS-AGENT et Forge pour booster le moteur RSI

> Statut : **étude + Phase 1 implémentée**. Objectif : déterminer quoi
> réutiliser, comment le câbler, et dans quel ordre.
>
> ✅ **Phases 1–3 livrées** — backends réels derrière des features Cargo
> optionnelles ; le cœur RSI reste sans dépendance par défaut.
>
> - **Phase 1** `forge` — `ForgeMetaSearch` : la méta-révision `ℳ` devient une
>   recherche évolutionnaire *exécutée* (`forge-core`) dont la fitness est
>   `SI_global`. Démo : SI_global 0.138 → 0.443 (+220 %).
> - **Phase 2** `forge` — `ForgeSubstrate` (trait `SubstrateImprover`) : la
>   composante logicielle de `P_eff` est *mesurée* par une campagne Forge sur un
>   vrai kernel matriciel. Démo : efficience 0.594 → 0.742, P_eff monotone ↑.
> - **Phase 3** `octasoma` — `OctaSomaMemory` (trait `ContextMemory`) : la
>   composante `C` dispose d'un vrai magasin vectoriel (k-NN fractal) ;
>   l'agent y écrit son état à chaque pas et peut rappeler les contextes proches.
>
> ```bash
> cargo build --features forge              # ℳ + P_eff réels
> cargo build --features octasoma           # mémoire C réelle
> cargo test  --features "forge octasoma"   # 41 tests lib
> ```
> ```rust
> let agent = rsi::RSIAgent::new(state, substrate, surface, cfg,
>         Box::new(rsi::ForgeMetaSearch::new(8, 24, 0.15, 42)))   // Phase 1
>     .with_substrate_improver(Box::new(rsi::ForgeSubstrate::new(160, 2, 6, 7))) // Phase 2
>     .with_memory(Box::new(rsi::OctaSomaMemory::new(state_dim, 1)));            // Phase 3
> ```

## 1. Résumé exécutif

Les quatre dépôts sont des projets **Rust de la même organisation
(Memorithm)** et couvrent, à eux quatre, presque toutes les composantes que le
moteur RSI ne modélise aujourd'hui que de façon *stylisée*. Le constat clé :

- **Forge** peut rendre **réels** le méta-optimiseur `ℳ` et le substrat `P_eff`
  (il fait évoluer et *mesure* du vrai code de calcul). **Plus fort levier,
  meilleure licence, dépendances saines.**
- **OctaSoma** peut servir de **mémoire réelle** pour la composante `C` (et
  partiellement `D`). **Risque technique le plus faible** (une seule dépendance
  pure-Rust).
- **CCOS** apporte l'**auditabilité/déterminisme** de `C` et des pas de `ℳ`
  (journal hash-chaîné, replay). Bloqué par l'**absence de licence**.
- **PAPERS-AGENT** peut alimenter `D` (extraction/analyse de papiers) et fournit
  une **boucle d'évolution** concrète, mais c'est le plus lourd (deps GPU/ORT +
  crates locales `scirust` non publiées).

**Recommandation : intégration progressive, derrière des *features* Cargo
optionnelles, en gardant le cœur de RSI sans dépendance.** Commencer par Forge
(`ℳ`/`P_eff` réels), puis OctaSoma (`C` réelle).

## 2. Synthèse par dépôt

| Dépôt | Rôle | Licence | Édition | Deps lourdes | Fit RSI | Effort |
|-------|------|---------|---------|--------------|---------|--------|
| **Forge** (`forge-core`) | recherche évolutionnaire de kernels, LLM-guided, exécution réelle | **MIT OR Apache-2.0** ✅ | 2021 | `rand, rayon, serde, sled, bincode` ; `ureq` (feature `llm`) ; sandbox Unix ; CUDA/SIMD = `nvcc`/`cargo` au *runtime* | `ℳ` **réel** + `P_eff` **réel** (σ(OᵀBO), σ(HᵀCO)) | faible–moyen |
| **OctaSoma** (`octasoma`) | mémoire vectorielle fractale, k-NN exact, persistée | **PolyForm-NC 1.0.0** ⚠️ (non-commercial) | **2024** (Rust ≥1.85) | `lz4_flex` uniquement (pure Rust) ; `#![forbid(unsafe)]` | `C` (mémoire) **réelle**, partiellement `D` | faible |
| **CCOS** (`ccos`) | « MMU cognitive » causale, contexte borné, journal hash-chaîné, replay | **AUCUNE** ❌ (« add a license before any external use ») | 2021 | cœur sync ; `tokio`+`reqwest` confinés au module `llm`/`mcp` (droppables) | `C` **auditable** + audit des pas de `ℳ` | moyen |
| **PAPERS-AGENT** (`papers_core`) | papiers → analyse → évolution de code Rust → rapport | **MIT** (README, pas de fichier LICENSE) | 2021 | **bloquant** : `path=/tmp/scirust/*` non publiées, `ort`(CUDA), `wasmtime`, `pdf-extract`, `reqwest`+`tokio` | `D` (connaissances) + boucle `ℳ`/`ΔS_appr` | élevé (isolable : moyen) |

## 3. Correspondance avec les composantes RSI

```
        composante RSI            backend réel candidat
  ┌────────────────────────┬───────────────────────────────────────┐
  │ D  connaissances       │ PAPERS-AGENT (extraction/analyse)       │
  │                        │ OctaSoma (ShardedMemory par domaine)    │
  │ M  modèle/architecture │ Forge (évolue de vrais kernels)         │
  │ R  raisonnement        │ (reste interne au moteur RSI)           │
  │ A  autonomie           │ (reste interne)                         │
  │ C  mémoire contextuelle│ OctaSoma (recall k-NN) + CCOS (audit)   │
  │ V  valeurs/buts        │ (reste interne)                         │
  ├────────────────────────┼───────────────────────────────────────┤
  │ P_eff  substrat        │ Forge.measure() → σ(OᵀBO), σ(HᵀCO) réels│
  │ ℳ  méta-optimiseur     │ Forge Engine + PAPERS EvolutionLoop     │
  │ SI_global  surface Σ_I │ devient la *fitness* injectée dans ℳ    │
  └────────────────────────┴───────────────────────────────────────┘
```

Le moteur RSI possède déjà les **points d'extension par trait** nécessaires :
`MetaSearch` (pour `ℳ`), `CapabilityModel`/`CeilingModel` (pour Φ/g). Il suffit
d'ajouter un trait `ContextMemory` (pour `C`) et un trait `SubstrateEvaluator`
(pour `P_eff`), puis de fournir des implémentations adossées à ces dépôts
derrière des *features*.

## 4. Architecture d'intégration recommandée

Principe directeur : **le cœur de RSI reste `std`-only et sans dépendance** ; les
backends réels sont des *features* optionnelles, derrière des traits. Aucune
régression de build/sécurité pour qui n'active rien.

```toml
# Cargo.toml (esquisse)
[features]
default = []
forge    = ["dep:forge-core"]   # ℳ et P_eff réels
octasoma = ["dep:octasoma"]      # mémoire C réelle
ccos     = []                    # vendage du cœur sync (pas de licence → à régler)
papers   = []                    # via sous-processus CLI (évite scirust/ort)
```

Nouveaux seams de trait dans RSI :

```rust
// C réelle
pub trait ContextMemory {
    fn write(&mut self, key: &str, embedding: &[f32], payload: &[u8]);
    fn recall(&self, query: &[f32], k: usize) -> Vec<Vec<u8>>;
}
// P_eff réel
pub trait SubstrateEvaluator {
    /// renvoie (σ(OᵀBO), σ(HᵀCO)) mesurés sur du vrai code
    fn evaluate(&mut self, software: &[f64], hardware: &[f64]) -> (f64, f64);
}
```

- `MetaSearch` (déjà existant) → `ForgeMetaSearch` : un `Engine<RsiDomain>` dont
  `Domain::measure()` renvoie `SI_global`. La recherche `argmax_ℳ` devient une
  **évolution exécutée**, plus seulement aléatoire/CMA-ES.
- `SubstrateEvaluator` → `ForgeSubstrate` : une campagne Forge sur un kernel
  (GEMM/SIMD) fournit des facteurs d'efficacité **mesurés** qui remplacent les
  sigmoïdes stylisées de `substrate.rs`.
- `ContextMemory` → `OctaSomaMemory` (`FractalMemory3D::insert/nearest_embedding`)
  et/ou `CcosMemory` (`ExternalMemory::ingest_source/recall` + audit).

## 5. Plan par phases

| Phase | Contenu | Dépôt | Risque | Valeur |
|-------|---------|-------|--------|--------|
| **1** ✅ | `ForgeMetaSearch` : `forge-core` Domain dont la fitness = `SI_global` ; `ℳ` exécuté (**fait**) | Forge | faible–moyen | ★★★ |
| **2** ✅ | `ForgeSubstrate` + trait `SubstrateImprover` → `P_eff` mesuré par campagne Forge sur kernel matriciel (**fait**) | Forge | moyen | ★★★ |
| **3** ✅ | `OctaSomaMemory` + trait `ContextMemory` → mémoire `C` réelle (k-NN fractal) (**fait**) | OctaSoma | faible | ★★ |
| **4** | Audit déterministe de `C`/`ℳ` (journal hash-chaîné, replay) | CCOS | moyen (licence) | ★★ |
| **5** | Ingestion `D` depuis papiers (sous-processus CLI `papers`) | PAPERS | élevé | ★ |

## 6. Risques & points bloquants

1. **Licences (résolubles : tu es propriétaire des quatre dépôts).**
   - CCOS : **aucune licence** → ajouter une licence avant tout usage/vendage.
   - OctaSoma : **PolyForm Non-Commercial** → OK pour usage recherche/perso ;
     un produit commercial nécessiterait une licence commerciale (que tu peux
     t'accorder).
   - PAPERS : MIT annoncé mais **sans fichier LICENSE** → en ajouter un.
   - Forge : MIT/Apache déjà déclaré (manque le fichier LICENSE, hygiène).
2. **PAPERS-AGENT dépend de crates locales `scirust` non publiées** + `ort`
   (CUDA) → le crate entier **ne compile pas** hors de sa machine. Seul le
   sous-système `evolution/` est `scirust`-free et isolable ; sinon usage en
   **sous-processus**.
3. **Forge** : isolation **Unix-only** (`pre_exec`/rlimit) ; domaines CUDA/SIMD
   exigent `nvcc`/`cargo` au *runtime* (pas au build). `forge-bridge` **ne
   compile pas** en l'état (littéraux `Report`/`Config` périmés) → l'éviter,
   préférer `forge-core` en dépendance directe.
4. **Édition 2024 d'OctaSoma** → nécessite Rust ≥ 1.85 (OK : toolchain 1.94).
5. **Pureté du cœur RSI** : tout passe par des *features* optionnelles pour ne
   jamais imposer ces dépendances ni casser le build `std`-only par défaut.

## 7. Prochaine étape proposée

Implémenter la **Phase 1** : un adaptateur `ForgeMetaSearch` (feature `forge`)
qui transforme la méta-révision `ℳ` en recherche évolutionnaire *exécutée* dont
la fitness est `SI_global`. C'est le plus gros gain pour le moteur RSI, avec la
licence et les dépendances les plus saines, et sans toucher au cœur par défaut.
