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
#[cfg(feature = "ccos")]
pub mod ccos_audit;
pub mod chaos;
pub mod checkpoint;
pub mod cma;
pub mod convergence;
pub mod criticality;
// Boucle d'auto-amélioration empirique (Darwin–Gödel / STOP) — port std-only de
// `soul-rsi` : propose un patch → build+test en copie isolée → garde si meilleur.
pub mod dgm;
pub mod loop_ctrl;
#[cfg(feature = "forge")]
pub mod forge_meta;
#[cfg(feature = "forge")]
pub mod forge_substrate;
pub mod dynamics;
// Sonde matérielle réelle (CPU/mém/GPU) pour ancrer le substrat — utile sur Jetson.
pub mod hw_probe;
pub mod flywheel;
pub mod json;
pub mod kernels;
pub mod knowledge;
pub mod linalg;
pub mod llm;
pub mod measured_substrate;
pub mod memory;
pub mod meta;
pub mod obs;
// Ω concret : banc de tâches réelles nommées (extension de §1, sans duplication).
pub mod omega_tasks;
pub mod plot;
pub mod prompt;
pub mod tasks;
#[cfg(feature = "octasoma")]
pub mod octasoma_memory;
pub mod report;
pub mod rng;
pub mod schedule;
#[cfg(feature = "scirust")]
pub mod scirust_bridge;
pub mod sha256;
pub mod simulation;
pub mod state;
pub mod trajectory;
pub mod substrate;
pub mod surface;
pub mod swarm;
pub mod synthesis;
pub mod tuning;
#[cfg(feature = "wasm")]
pub mod wasm_domain;

pub use agent::{RSIAgent, StepReport};
pub use api::{ApiResult, RsiApi};
pub use ascent::{ascend, Guard, RefineTask, Report, StopReason as AscentStop};
pub use audit::{AuditEvent, AuditLog, HashChainLog, TraceEvent};
pub use synthesis::{Expr, SymbolicSynthesis};
#[cfg(feature = "ccos")]
pub use ccos_audit::CcosAudit;
pub use checkpoint::Checkpoint;
pub use cma::SepCmaEs;
pub use convergence::{ConvergenceDetector, Trend};
pub use criticality::{RiskConfig, RiskModel, RiskReport, RiskSignals};
pub use loop_ctrl::{LoopConfig, LoopObserver, LoopOutcome, StopReason};
pub use schedule::{LoopSchedule, MetaMeta};
pub use swarm::{run_swarm, run_swarm_demo, SwarmMember, SwarmResult};
#[cfg(feature = "forge")]
pub use forge_meta::ForgeMetaSearch;
#[cfg(feature = "forge")]
pub use forge_substrate::ForgeSubstrate;
pub use dynamics::{Dynamics, StabilityConfig, StepInfo};
pub use json::Json;
pub use knowledge::{CorpusKnowledge, KnowledgeSource, PapersKnowledge};
pub use measured_substrate::MeasuredSubstrate;
pub use tasks::{GroundedCapability, Task, TaskCorpus};
pub use memory::{ContextMemory, LinearContextMemory};
#[cfg(feature = "octasoma")]
pub use octasoma_memory::OctaSomaMemory;
pub use meta::{CmaEsMeta, MetaOptimizer, MetaSearch, MetaStrategy};
pub use rng::Rng;
pub use state::{CognitiveState, Dims};
pub use substrate::Substrate;
pub use surface::{
    Bottleneck, CapabilityModel, CeilingModel, IntelligenceSurface, PowerCeiling,
    SigmoidCapability,
};
