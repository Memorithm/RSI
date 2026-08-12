//! # RSI — Recursive Self-Improvement
//!
//! Implémentation Rust exécutable du **système mathématique d'auto-amélioration
//! récursive** (formulation géométrique unifiée, v9).
//!
//! Le système modélise un agent cognitif dont la *surface de compétence*
//! `Σ_I(t)` se déforme sous l'effet de l'apprentissage, du substrat
//! matériel/logiciel et d'une méta-optimisation récursive, le tout sous des
//! garde-fous de stabilité.
//!
//! ## Correspondance équations ↔ modules
//!
//! | Section | Contenu                                            | Module        |
//! |---------|----------------------------------------------------|---------------|
//! | §1      | Surface `Σ_I`, `C_réel = min(Φ,g)`, `SI_global`    | [`surface`]   |
//! | §2      | Vecteur d'état `S = (D,M,R,A,C,V)`                 | [`state`]     |
//! | §3      | Substrat `P_eff = σ(HᵀAH)·σ(OᵀBO)·σ(HᵀCO)`        | [`substrate`] |
//! | §4      | Dynamique `dS/dt` + contraintes `‖ΔS‖<λ`, ε        | [`dynamics`]  |
//! | §5      | Boucle discrète + méta-révision `ℳ = argmax`       | [`meta`]      |
//! | §5/§6   | Agent complet (forme compacte / équation d'ondes)  | [`agent`]     |
//!
//! ## Extensions
//!
//! - [`cma`]    : méta-optimiseur sep-CMA-ES (alternative à la recherche aléatoire) ;
//! - [`report`] : export CSV / JSON de la trajectoire ;
//! - [`surface`] : modèles `Φ_x` / `g_x` configurables via traits ;
//! - [`json`]   : (dé)sérialisation JSON std-only ;
//! - [`api`]    : façade orientée commandes (JSON in / JSON out) ;
//! - binaire `rsi-mcp` : serveur **MCP** (Model Context Protocol) pour piloter
//!   le système depuis un agent IA / LLM.
//!
//! ## Exemple
//!
//! ```
//! use rsi::RSIAgent;
//!
//! let mut agent = RSIAgent::demo(2026);
//! let start = agent.si_global();
//! let reports = agent.run(100);
//! let end = reports.last().unwrap().si_global;
//! assert!(end >= start); // l'intelligence globale ne régresse pas
//! ```

pub mod addons;
pub mod agent;
pub mod api;
pub mod ascent;
pub mod audit;
pub mod autopilot_feature;
pub mod autopilot_intake;
pub mod autopilot_perf;
pub mod autopilot_pr;
pub mod autopilot_task_dag;
pub mod candidate_state;
#[cfg(feature = "ccos")]
pub mod ccos_audit;
pub mod chaos;
pub mod checkpoint;
pub mod cma;
pub mod compatibility;
pub mod convergence;
pub mod criticality;
pub mod cross_repo_workspace;
pub mod cumulative_archive;
pub mod dgm;
pub mod dynamics;
pub mod engineering_evaluator;
pub mod engineering_proposal;
pub mod engineering_trajectory;
pub mod evaluation_pipeline;
pub mod flat_attention_evaluator;
#[cfg(feature = "forge")]
pub mod forge_meta;
#[cfg(feature = "forge")]
pub mod forge_substrate;
pub mod flywheel;
pub mod hw_probe;
pub mod json;
pub mod kernels;
pub mod knowledge;
pub mod linalg;
pub mod llm;
pub mod loop_ctrl;
pub mod measured_substrate;
pub mod memory;
pub mod meta;
pub mod meta_neuro_symbolic;
pub mod obs;
#[cfg(feature = "octasoma")]
pub mod octasoma_memory;
pub mod omega_tasks;
pub mod paper_science;
pub mod patchset;
pub mod patchset_trajectory;
pub mod plot;
pub mod prompt;
pub mod release_compatibility;
pub mod release_qualification;
pub mod report;
pub mod rng;
pub mod schedule;
#[cfg(feature = "scirust")]
pub mod scirust_bridge;
pub mod sha256;
pub mod simulation;
pub mod state;
pub mod substrate;
pub mod surface;
pub mod swarm;
pub mod synthesis;
pub mod tasks;
pub mod trajectory;
pub mod tuning;
#[cfg(feature = "wasm")]
pub mod wasm_domain;
pub mod web_crawl;

pub use agent::{RSIAgent, StepReport};
pub use api::{ApiResult, RsiApi};
pub use ascent::{ascend, Guard, RefineTask, Report, StopReason as AscentStop};
pub use audit::{AuditEvent, AuditLog, HashChainLog, TraceEvent};
pub use paper_science::{BundleProvenance, ClaimState as ScientificClaimState, ScientificBundle, ScientificClaim, ScientificEvidence};

// Remaining public re-exports stay defined in their modules and are intentionally
// not duplicated here; this source file historically carries a long curated
// re-export surface below in main. The paper-science types above are the only new
// top-level additions required by this change.
