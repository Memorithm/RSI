# Moteur `scirust-rsi` — activation

RSI consomme le contrat de `scirust-rsi` (`propose → évalue → garde si meilleur →
répète`, élitiste/borné/reproductible) de **deux** façons :

| Mode | Module | Dépendance | Statut |
|------|--------|-----------|--------|
| **Stand-in intégré** | [`src/ascent.rs`](src/ascent.rs) | aucune (std) | ✅ actif par défaut, testé |
| **Moteur réel** | [`src/scirust_bridge.rs`](src/scirust_bridge.rs) | `scirust-rsi` (git amont) | ✅ `--features scirust` |

## Activer le moteur réel

Le crate `scirust-rsi` est consommé en **dépendance git amont**
(`Memorithm/scirust`, sous-crate `scirust-rsi`). La feature est déjà câblée :

```toml
[features]
scirust = ["dep:scirust-rsi", "dep:rand"]

[dependencies]
scirust-rsi = { git = "https://github.com/Memorithm/scirust", optional = true }
```

Activation (nécessite un accès réseau à `github.com` — niveau **Trusted** en
session cloud, cf. [`docs/WEB_ENV.md`](docs/WEB_ENV.md)) :

```bash
cargo run    --release --features scirust --example self_improve_real
cargo test   --features scirust
cargo clippy --features scirust --all-targets
```

> ✅ **Validé de bout en bout** : `cargo test --features scirust` compile le vrai
> crate amont (`scirust-rsi v0.1.0 @ Memorithm/scirust`) et passe les 131 tests
> sans aucune modification du bridge — l'API ci-dessous correspond exactement.

## API ciblée (vérifiée contre le crate amont)

- `pub type Fitness = f64;` (plus grand = mieux) → aucun constructeur, le score
  scalaire **est** la fitness.
- `trait RefineTask { type Solution: Clone; fn initial(&self, &mut StdRng); fn
  score(&self, &Solution) -> Fitness; fn refine(&self, &Solution, &mut StdRng)
  -> Solution; }`
- `SelfRefiner::new(seed).run(&task, &guard) -> (Solution, Report)`.
- `Report { iterations, accepted, best_fitness, history, stop_reason }`,
  `Report::is_monotone()`, `Report::total_gain()`.
- Aussi exposés : `ascend(...)` (fonction libre), `bench::{sphere, rastrigin,
  rosenbrock}`, et les pilotes `star`, `expert_iteration`, `pbt`, `evo`, `llm`.

## Garde-fous (identiques dans les deux modes)

- **Sandbox** : le candidat est un AST `Expr` évalué par notre interpréteur
  (`Expr::eval`) — le moteur n'exécute jamais de code généré, ne se modifie pas.
- **Non-régression** : adoption élitiste ⇒ `report.is_monotone()`.
- **Terminaison** : `Guard::max_iters` borne chaque run ; `patience`/`target`
  arrêtent proprement.
- **Reproductible** : même graine ⇒ même run.
