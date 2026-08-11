//! **cogno-scirust** — backend batch (tensoriel) de COGNO-1, validé contre
//! l'oracle scalaire `cogno-core`.
//!
//! Fournit :
//! - une version batch de l'objectif complet et de la perte COGNO-0.1 ;
//! - l'apprentissage pairwise accepté/rejeté/édité ;
//! - la logique souple différentiable ;
//! - InfoNCE pour la mémoire ;
//! - la tête de calibration (Brier) ;
//! - la mesure des ressources ;
//! - AdamW avec clipping de gradient ;
//! - la **cross-validation systématique** contre `cogno-core` (mêmes entrées
//!   ⇒ mêmes composantes, même objectif, même perte).
//!
//! Ce backend n'est **jamais** l'autorité de sécurité : les contraintes dures,
//! autorisations, budgets, provenance et effets de bord restent contrôlés par
//! `cogno-core` (le gate d'admissibilité `F(x)` y vit).

pub mod adamw;
pub mod batch;
pub mod cross_validate;
pub mod train;

pub use adamw::{AdamW, AdamWConfig};
pub use batch::{CognoBatchOutput, compute_cogno01_loss_batch, compute_objective_batch};
pub use cross_validate::{
    BatchComparison, compare_after_optim_step, compare_oracle_and_backend, CrossValidationReport,
};
pub use train::{
    AllocationStats, ControlledRollout, GradientAccumulator, PpoConfig, PpoSample,
    Rollout, RolloutPolicy, compute_ppo_loss, validate_f32_path,
};
