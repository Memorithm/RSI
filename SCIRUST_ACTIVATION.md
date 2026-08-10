# Moteur `scirust-rsi` — activation

RSI consomme le contrat de `scirust-rsi` (`propose → évalue → garde si meilleur → répète`, élitiste/borné/reproductible) de **deux** façons :

| Mode | Module | Dépendance | Statut |
|------|--------|-----------|--------|
| **Stand-in intégré** | [`src/ascent.rs`](src/ascent.rs) | aucune (std) | ✅ actif par défaut, testé |
| **Moteur canonique SciRust** | [`src/scirust_bridge.rs`](src/scirust_bridge.rs) | `Memorithm/scirust/scirust-rsi` | ✅ `--features scirust` |

## Source canonique

Depuis P1.2 du programme RSI × COGNO-1 × SciRust, RSI **n'embarque plus de copie locale** du crate `scirust-rsi`. L'implémentation canonique vit exclusivement dans `Memorithm/scirust/scirust-rsi`.

La dépendance est épinglée sur la révision SciRust immuable qui a fusionné le contrat canonique P1.1 :

```text
8af0801b8bc0c69630797db82bb2dd3416cc8f0a
```

```toml
[features]
scirust = ["dep:scirust-rsi", "dep:rand"]

[dependencies]
scirust-rsi = { git = "https://github.com/Memorithm/scirust", rev = "8af0801b8bc0c69630797db82bb2dd3416cc8f0a", optional = true }
```

**Jamais de `branch = "master"` ou de révision flottante** : toute évolution du contrat doit d'abord être fusionnée et qualifiée dans SciRust, puis RSI avance son pin dans une PR dédiée avec CI verte et mise à jour du CompatibilitySet.

Le lockfile P1.2 conserve les versions registry existantes et ajoute seulement le graphe exigé par le crate canonique : `rand_distr 0.4.3` et l'activation `libm` de `num-traits`, en plus de `rand 0.8` et `serde` déjà présents. Ce détail est volontairement enregistré afin qu'une future avance du pin distingue une vraie évolution de dépendances d'un bruit de résolution Cargo.

## Activation

```bash
cargo run    --release --features scirust --example self_improve_real
cargo test   --locked --features scirust
cargo clippy --locked --features scirust --all-targets -- -D warnings
```

La CI RSI inclut la feature `scirust` dans les features publiques afin que le bridge soit compilé et testé contre le crate canonique épinglé.

## API ciblée

Le contrat canonique P1.1 verrouille notamment :

- `pub type Fitness = f64;` — plus grand = mieux ;
- `RefineTask { initial, score, refine }` piloté par un `StdRng` reproductible ;
- `SelfRefiner::new(seed).run(&task, &guard) -> (Solution, Report)` ;
- `Report::history` = incumbent best-so-far après chaque itération ;
- un candidat rejeté ne modifie pas l'incumbent et ne rend pas `is_monotone()` faux ;
- `Report::is_monotone()` et `Report::total_gain()` ;
- `Guard` pour les bornes d'itération, patience, cible, delta minimal et budget temporel du moteur canonique.

Le crate SciRust expose aussi Self-Refine, STaR, Expert Iteration, `(1+λ)`-ES, PBT et les pilotes LLM. Leur sémantique appartient à SciRust ; RSI ne doit pas en maintenir de fork local.

## Garde-fous

- **Sandbox** : le bridge actuel travaille sur un AST `Expr` évalué par l'interpréteur RSI ; `scirust-rsi` n'exécute pas de code arbitraire du DGM.
- **Non-régression** : adoption élitiste et historique best-so-far monotone.
- **Terminaison** : `Guard` borne chaque run.
- **Reproductibilité** : même graine + même tâche déterministe ⇒ même solution et même rapport.
- **Provenance** : la révision SciRust exacte fait partie de la compatibilité cross-repo ; un résultat ne peut pas être attribué à une branche mobile.
