# Meta-NeuroSymbolic — objectif RL avec validateurs symboliques

Module `src/meta_neuro_symbolic.rs` : implémente l'objectif d'apprentissage
**Meta-NeuroSymbolic** pour la politique neuro-symbolique `LLM_φ^{NS}`, avec
des validateurs formels injectables (`SymbolicValidator`), une perte calculée
**zéro allocation** et **0 bloc `unsafe`**.

## Objectif (à maximiser, converti en perte)

```
MetaNS(φ) = E_{x~D_RL} E_{y~LLM_φ(x)} [ RM_NS(x,y)
             − β_NS · log( LLM_φ(y|x) / LLM_SFT(y|x) ) ]
             + γ_NS · E_{x~D_pretrain} log LLM_φ(x)
```

| Terme | Rôle | Code |
|---|---|---|
| `RM_NS(x,y)` | Récompense neuro-symbolique par vérification formelle | `SymbolicValidator::validate` |
| `−β_NS · KL` | Régularise la politique contre la dérive du modèle SFT | `AgentExecutionTrace::kl` (clampé `[0, KL_MAX]`) |
| `+γ_NS · log LLM_φ(x)` | Anti-oubli catastrophique sur le corpus de pré-entraînement | `compute_meta_ns_loss` (paramètre `pretrain_logp_mean`) |

## Validateurs fournis

- **`IntegritySha256Validator`** — vérifie que l'artefact `y` porte le bon
  hachage SHA-256 (de `x ‖ sep ‖ payload`) ; violation = pénalité lourde
  (`REWARD_VIOLATION = -10`), conformité = récompense positive.
- **`NoUnsafeValidator`** — pénalise tout artefact contenant le bloc `unsafe`
  (politique « 0 bloc unsafe » des cœurs de calcul).
- **`SymbolicExprValidator`** — vérifie que `y` se parse en une
  [`Expr`](../src/synthesis.rs) valide (grammaire étendue) et est borné en
  complexité.
- **`WorkspaceConstraintsValidator`** — respect des contraintes d'espace de
  travail : préfixes autorisés (ex. `src/`), motifs interdits (ex. `../`,
  `target/`).

N'importe quel validateur peut être branché en implémentant
`SymbolicValidator` (trait injectable, `tag()` pour la traçabilité).

## Contraintes d'implémentation (HPC)

- **Zéro allocation sur chemins chauds** : `compute_meta_ns_loss` écrit dans
  des tampons `PerTraceBuffer` pré-alloués fournis par l'appelant, n'alloue
  rien dans ses boucles. L'attribut `#[deny(clippy::alloc_in_loop)]` est posé
  au niveau du module (avec `#[allow(unknown_lints)]` pour compatibilité avec
  les clippy qui n'ont pas ce lint) ; la garantie structurelle est verrouillée
  par le test `buffer_is_reused_without_allocations`.
- **0 bloc `unsafe`** : vérifiable par grep (`unsafe` n'apparaît que dans des
  commentaires et des artefacts de test).
- **Déterminisme absolu** : somme dans l'ordre exact des traces, `f64`.
- **Inlining agressif** : `#[inline(always)]` sur `validate`, `kl` — compatible
  AVX-512 / ARM Neon (types scalaires, aucune barrière).
- **Compatibilité vectorielle (feature `simd`)** : la somme du batch passe par
  `sum_f64` — réduction `wide::f64x4` (4 voies AVX/Neon) avec combinaison à
  ordre fixe, retombée scalaire sans la feature. Même convention que
  `rsi::linalg::dot`.

## Tests de propriété

Au-delà des tests unitaires, le module vérifie des **propriétés mathématiques**
sur des traces simulées (graines déterministes) :

- **Finitude & bornitude** : la perte reste finie et `|loss| < 1000` pour des
  traces aléatoires, quel que soit le batch (1..64) et `β_NS`/`γ_NS`
  (anti-explosion numérique) ;
- **Monotonie en récompense** : `loss` décroît quand `RM_NS` augmente (sur 20
  tirages) — minimiser la perte maximise bien la récompense ;
- **Monotonie en KL** : `loss` croît avec la divergence (β > 0) — la KL
  régularise la politique ;
- **Déterminisme** : même batch ⇒ même perte, appel répété.

Ces propriétés passent en **scalaire et en SIMD** (`--features simd`).

## Traçabilité

`AgentExecutionTrace` stocke, par pas d'optimisation :
`log_prob_policy`, `log_prob_sft`, `artifact (y)`, `context (x)`,
`symbolic_reward` (posée par la perte) et `validator_tag`.

## Exemple

```rust
use rsi::meta_neuro_symbolic::*;

let cfg = MetaNSConfig { beta_ns: 0.1, gamma_ns: 0.01 };
let mut traces = [AgentExecutionTrace::new(
    -0.5,           // log LLM_φ(y|x)
    -0.6,           // log LLM_SFT(y|x)
    b"fn main() { let x = 1; }",  // y (artefact)
    b"task",        // x (contexte)
)];
let mut state = MetaNSState::new();
let mut buf = PerTraceBuffer::new(16);
let loss = compute_meta_ns_loss(
    &cfg, &NoUnsafeValidator, &mut traces, /*pretrain_logp_mean=*/0.0,
    &mut state, &mut buf,
);
assert!(loss.is_finite());
assert_eq!(traces[0].validator_tag, "no_unsafe"); // traçabilité
```

## Références

- Équation RLHF/RL : objectif RL standard (récompense − β·KL) +
  régularisation de pré-entraînement (cf. littérature RLHF).
- `SHA-256` : [`src/sha256.rs`](../src/sha256.rs) (implémentation NIST pure).
- Grammaire étendue : [`src/synthesis.rs`](../src/synthesis.rs).
