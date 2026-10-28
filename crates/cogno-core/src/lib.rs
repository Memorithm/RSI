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
pub mod engineering;
pub mod error;
pub mod kv_cache;
pub mod memory;
pub mod numeric;
pub mod objective;
pub mod pref;
pub mod resource;
pub mod reward;
pub mod seq;
pub mod softlogic;

pub use admissible::{AdmissibilityGate, AdmissibilityVerdict, HardValidator, ProvenanceValidator};
pub use budget::{NormalizedCosts, ResourceBudget, ResourceWeights};
pub use calib::{compute_brier_calibration, CalibrationMetrics, CalibrationPoint};
pub use determinism::{fingerprint, DeterminismRecord, ExecutionMode};
pub use engineering::{
    CheckStatus, EngineeringAdmissibility, EngineeringAdmissibilityVerdict, EngineeringCheck,
    EngineeringGateKind, EngineeringPolicyCheck,
};
pub use error::{checked_add, checked_mul, CognoError, CognoResult};
pub use kv_cache::{
    BoundedKvCache, FixedKvCache, KvCacheConfig, KvCacheError, TensorView, TensorViewMut,
};
pub use memory::{compute_memory_objective, cosine_similarity, MemoryMetrics, MemorySample};
pub use numeric::{CompensatedSum, FiniteScalar, NonNegativeFinite};
pub use objective::{
    compute_cogno01_loss, compute_cogno_objective, CognoObjectiveBreakdown, CognoObjectiveInput,
    CognoWeights,
};
pub use pref::{compute_preference_objective, sigmoid_stable, PreferencePair};
pub use resource::{compute_resource_loss_batch, ResourceCosts};
pub use reward::{compute_reward_breakdown, RewardBreakdown};
pub use seq::{Mask, Sequence, SequenceBatch};
pub use softlogic::{compute_symbolic_objective, SoftLogic, SoftRule};
