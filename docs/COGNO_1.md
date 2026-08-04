# COGNO-1 — objectif Meta-NeuroSymbolique (SciRust backend)

Deux crates du workspace implémentent COGNO-1 :

- **`cogno-core`** (crates/cogno-core) : l'**oracle scalaire indépendant** —
  sécurité (gate d'admissibilité `F(x)`), objectif complet, perte COGNO-0.1,
  déterminisme. Aucune dépendance. Il est **l'autorité de sécurité** : les
  contraintes dures, autorisations, budgets, provenance et effets de bord y
  vivent.
- **`cogno-scirust`** (crates/cogno-scirust) : le **backend batch** qui doit
  correspondre exactement à l'oracle (cross-validation §14). Il n'est jamais
  l'autorité de sécurité.

## Équation complète (contrat §9)

```text
J_COGNO(φ,ψ) =
  E_{x~D_RL, y~π_φ, y∈F(x)} [ R̃_NS(x,y) − β_NS·(log π_φ(y|x) − log π_ref(y|x)) ]
  + η_pref·J_pref(φ)
  + η_sym·J_sym(φ)
  + η_mem·J_mem(φ,ψ)
  + γ_NS·E_{x~D_pretrain}[ log π_φ(x) ]
  − λ_cal·L_cal
  − λ_eff·L_resource

L_COGNO(φ,ψ) = −J_COGNO(φ,ψ)
```

## Ensemble admissible (contrat §2)

```text
F(x) = { y | H_h(x,y)=1 ∀h,  P_prov(x,y)=1,
             C_mem(x,y) ≤ B_mem,  C_lat(x,y) ≤ B_lat,  C_ctx(x,y) ≤ B_ctx }
```

Les contraintes de `F(x)` sont **dures** (rejet avant classement et adoption),
jamais des pénalités compensables. Implémentation : `AdmissibilityGate`
(`cogno-core::admissible`).

## Récompense neuro-symbolique décomposée (contrat §3)

```text
R̃_NS = R_formal + q_e·R_feedback + R_tests + R_heldout
       − P_regression − P_complexity − κ_u·U
```

Chaque composante est observable séparément (`RewardBreakdown`). Cas
analytique historique : `R̃_NS = 1.75` (documenté dans les tests).

## Termes (chaque terme a son API distincte)

| Terme | Équation | Module |
|---|---|---|
| Pairwise préférences | `J_pref = E[log σ(α·Δ)]`, Δ en log-espace (jamais de division) | `pref.rs` |
| Logique souple | `J_sym = E[Σ w_j log(ε+s_j)]` — t-norme produit `∧=ab`, `∨=a+b−ab`, `¬=1−a`, `⇒=1−a+ab` | `softlogic.rs` |
| Contraste mémoire | `J_mem = E[ log(exp(sim/τ) / Σ exp(sim/τ)) ]` (InfoNCE, cosinus) | `memory.rs` |
| Calibration | `L_cal = E[(p−z)²]` (Brier) + ECE, courbe, abstention | `calib.rs` |
| Ressources | `L_resource = E[ρ_m·C̄_mem + ρ_t·C̄_lat + ρ_c·C̄_ctx]` (unités explicites) | `resource.rs` |
| KL politique | `−β_NS·E[log π_φ − log π_ref]` (log-ratio en log-espace) | `objective.rs` |

## Perte COGNO-0.1 (contrat §10 — avant tout RL)

```text
L_0.1 = L_SFT + λ_pref·L_pairwise + λ_sym·L_softlogic
        + λ_mem·L_InfoNCE + λ_cal·L_Brier + λ_eff·L_resource
```

Ordre obligatoire des phases : SFT → pairwise → logique → mémoire → calibration
→ coût matériel → objectif complet hors ligne → rollouts contrôlés → PPO (seulement
après validation). `compute_cogno01_loss` l'implémente.

## Décomposition complète (jamais perdue)

```rust
pub struct CognoObjectiveBreakdown {
    pub admissible_reward: FiniteScalar,
    pub reference_log_ratio: FiniteScalar,
    pub preference_objective: FiniteScalar,
    pub symbolic_objective: FiniteScalar,
    pub memory_objective: FiniteScalar,
    pub pretraining_objective: FiniteScalar,
    pub calibration_loss: NonNegativeFinite,
    pub resource_loss: NonNegativeFinite,
    pub total_objective: FiniteScalar,
    pub total_loss: FiniteScalar,
}
```

## Cache KV (contrat §13)

`BoundedKvCache` : capacité explicite, préalloué, fallible, réutilisable,
nettoyable, mesurable en octets, décodage sans allocation par token,
protégé contre les dépassements.

## Contrat de déterminisme (contrat §16)

`DeterminismRecord` enregistre : graine, ordre des exemples, ordre des
réductions, configuration de l'objectif, versions (modèle/ref/encodeur),
dtype, backend, threads, mode, empreintes données/poids.

## Cross-validation (contrat §14)

`compare_oracle_and_backend` vérifie, pour chaque batch déterministe, que le
backend correspond à l'oracle composante par composante (tolérance
configurable). Cas couverts par les tests : batch vide, taille un, tous termes
actifs, log-probs très négatives, policy=ref, règle violée, mémoire mal
classée, surconfiance, coûts au budget, NaN, infini, mismatch de longueur,
préférence indifférente.

## Interdictions respectées (contrat §18)

- objectif jamais remplacé par une somme arbitraire ;
- termes jamais fusionnés sans observabilité (breakdown complet) ;
- contrainte dure jamais dans une récompense compensable ;
- ratio de politique jamais calculé par division ;
- InfoNCE jamais déclaré sans négatifs ;
- calibration jamais déclarée sans métriques ;
- coût mémoire jamais remplacé par une constante ;
- PPO non activé (seule la perte hors ligne COGNO-0.1 est implémentée) ;
- backend toujours validé contre l'oracle.
