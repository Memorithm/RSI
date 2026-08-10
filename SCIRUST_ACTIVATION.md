# Moteur `scirust-rsi` — activation

RSI conserve son moteur std-only [`src/ascent.rs`](src/ascent.rs) pour le mode par défaut,
mais **l'unique implémentation canonique de `scirust-rsi` vit désormais dans
`Memorithm/scirust/scirust-rsi`**. Le duplicat local a été retiré dans P1.2.

| Mode | Module | Dépendance | Statut |
|------|--------|-----------|--------|
| **Cœur RSI std-only** | [`src/ascent.rs`](src/ascent.rs) | aucune | ✅ actif par défaut, testé |
| **Moteur SciRust canonique** | [`src/scirust_bridge.rs`](src/scirust_bridge.rs) | `scirust-rsi` à révision git immuable | ✅ `--features scirust` |

## Révision canonique qualifiée

P1.1 a figé le contrat downstream de `scirust-rsi` dans SciRust PR #1138. RSI
consomme exactement le merge SciRust suivant :

```text
repository: https://github.com/Memorithm/scirust
revision:   8af0801b8bc0c69630797db82bb2dd3416cc8f0a
```

Le `Cargo.toml` utilise donc un `rev`, jamais une branche mouvante :

```toml
[features]
scirust = ["dep:scirust-rsi", "dep:rand"]

[dependencies]
scirust-rsi = {
  git = "https://github.com/Memorithm/scirust",
  rev = "8af0801b8bc0c69630797db82bb2dd3416cc8f0a",
  optional = true,
}
```

Une évolution de SciRust n'est **pas** absorbée implicitement. Elle doit produire
une nouvelle qualification, un nouveau `CompatibilitySet`, puis une PR RSI qui
met à jour explicitement cette révision.

## Activer le moteur SciRust

L'activation nécessite un accès réseau à `github.com` au moment où Cargo doit
récupérer la révision épinglée :

```bash
cargo run --release --features scirust --example self_improve_real
cargo test --locked --features scirust
cargo clippy --locked --features scirust --all-targets -- -D warnings
```

## API consommée

Le bridge utilise le contrat figé par SciRust P1.1 :

- `pub type Fitness = f64;` ;
- `refine::RefineTask` avec `StdRng` déterministe ;
- `SelfRefiner::new(seed).run(&task, &guard) -> (Solution, Report)` ;
- `Guard` pour borner `max_iters`, `patience`, `target` et `min_delta` ;
- `Report::history` comme historique **best-so-far** ;
- `Report::is_monotone()` pour vérifier la monotonie de l'incumbent conservé ;
- `Report::total_gain()` et les compteurs de convergence.

La sémantique importante de P1.1 est que le rejet d'un candidat moins bon ne
constitue pas une régression : l'historique enregistre l'incumbent conservé, pas
la fitness brute de chaque proposition rejetée.

## Garde-fous

- **Sandbox** : le candidat est un AST `Expr` évalué par notre interpréteur
  (`Expr::eval`) ; `scirust-rsi` n'exécute pas de code généré par ce bridge.
- **Non-régression** : l'incumbent best-so-far ne baisse jamais.
- **Terminaison** : `Guard::max_iters` borne chaque run ; `patience`/`target`
  arrêtent proprement.
- **Reproductibilité** : même graine + même révision + mêmes entrées ⇒ même run
  pour les contrats déterministes couverts par P1.1.

## Règle de propriété

Toute modification sémantique de `Guard`, `Report`, `SelfRefiner` ou
`RefineTask` doit être faite et qualifiée dans `Memorithm/scirust` d'abord. RSI
ne doit plus recréer ou maintenir une seconde implémentation du crate
`scirust-rsi`.
