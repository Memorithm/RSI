//! **cogno-core** — oracle scalaire indépendant de COGNO-1.
//!
//! Implémente l'objectif neuro-symbolique complet de COGNO-1, la perte de
//! démarrage COGNO-0.1, le gate d'admissibilité `F(x)`, la récompense
//! décomposée, les termes pairwise/logique floue/InfoNCE/Brier/ressources,
//! le cache KV borné et le contrat de déterminisme.
//!
//! Ce crate est **l'oracle numérique** : `cogno-scirust` doit correspondre
//! exactement à ces calculs (cross-validation). Il est **indépendant** de tout
//! backend tensoriel et **sans dépendance** (std uniquement).
//!
//! ## Contrat
//! - réductions dans un ordre défini (ordre du batch) ;
//! - rejette NaN et infinis non autorisés ([`FiniteScalar`], [`NonNegativeFinite`]) ;
//! - sommation compensée (Kahan) dans les accumulateurs critiques ;
//! - valide toutes les longueurs, shapes et masques ;
//! - arithmétique contrôlée pour les tailles (pas de multiplication non bornée) ;
//! - zéro allocation sur le chemin chaud après préparation ;
//! - décomposition complète de l'objectif ([`CognoObjectiveBreakdown`]) ;
//! - erreurs structurées ([`CognoError`]) ;
//! - testable sans backend tensoriel.

#![allow(unknown_lints)]
#![deny(clippy::alloc_in_loop)]

pub mod admissible;
pub mod budget;
pub mod calib;
pub mod determinism;
pub mod error;
pub mod kv_cache;
pub mod memory;
pub mod numeric;
pub mod objective;
pub mod pref;
pub mod reward;
pub mod resource;
pub mod softlogic;

pub use admissible::{AdmissibilityGate, AdmissibilityVerdict, HardValidator, ProvenanceValidator};
pub use budget::{NormalizedCosts, ResourceBudget, ResourceWeights};
pub use calib::{CalibrationMetrics, CalibrationPoint, compute_brier_calibration};
pub use determinism::{DeterminismRecord, ExecutionMode, fingerprint};
pub use error::{CognoError, CognoResult, checked_add, checked_mul};
pub use kv_cache::{BoundedKvCache, FixedKvCache, KvCacheConfig, KvCacheError, TensorView, TensorViewMut};
pub use memory::{MemoryMetrics, MemorySample, compute_memory_objective, cosine_similarity};
pub use numeric::{CompensatedSum, FiniteScalar, NonNegativeFinite};
pub use objective::{CognoObjectiveBreakdown, CognoObjectiveInput, CognoWeights, compute_cogno_objective, compute_cogno01_loss};
pub use pref::{PreferencePair, compute_preference_objective, sigmoid_stable};
pub use reward::{RewardBreakdown, compute_reward_breakdown};
pub use resource::{ResourceCosts, compute_resource_loss_batch};
pub use softlogic::{SoftLogic, SoftRule, compute_symbolic_objective};
