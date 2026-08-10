//! COGNO-aware engineering evaluation composition.
//!
//! The execution order is structural: collect complete hard-gate evidence,
//! resolve it through `cogno-core`, and invoke ranking only when COGNO says the
//! candidate is admissible. Ranking code therefore cannot compensate for or
//! even observe its own score after a hard-gate rejection.

use crate::dgm::{Evaluator, Fitness};
use cogno_core::{
    EngineeringAdmissibility, EngineeringAdmissibilityVerdict, EngineeringCheck,
    EngineeringPolicyCheck,
};
use std::fmt;
use std::path::Path;

/// Complete hard-gate evidence collection for one materialized candidate.
pub trait EngineeringEvidenceCollector {
    fn collect(&self, workspace: &Path) -> Result<EngineeringAdmissibility, EngineeringEvaluationError>;
}

/// Ranking measurement, deliberately separate from admissibility evidence.
pub trait EngineeringRanker {
    fn rank(&self, workspace: &Path) -> Result<RankingEvidence, EngineeringEvaluationError>;
}

/// A valid ranking result. Non-finite scores are rejected at construction.
#[derive(Debug, Clone, PartialEq)]
pub struct RankingEvidence {
    pub score: f64,
    pub notes: String,
}

impl RankingEvidence {
    pub fn new(score: f64, notes: impl Into<String>) -> Result<Self, EngineeringEvaluationError> {
        if !score.is_finite() {
            return Err(EngineeringEvaluationError::InvalidRankingScore);
        }
        Ok(Self {
            score,
            notes: notes.into(),
        })
    }
}

/// Full result preserving the hard-gate breakdown even when ranking is skipped.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringEvaluation {
    pub admissibility: EngineeringAdmissibility,
    pub verdict: EngineeringAdmissibilityVerdict,
    pub ranking: Option<RankingEvidence>,
}

/// Orchestrator enforcing evidence -> COGNO -> ranking ordering.
pub struct CognoEngineeringEvaluator<C, R> {
    collector: C,
    ranker: R,
}

impl<C, R> CognoEngineeringEvaluator<C, R>
where
    C: EngineeringEvidenceCollector,
    R: EngineeringRanker,
{
    pub fn new(collector: C, ranker: R) -> Self {
        Self { collector, ranker }
    }

    pub fn evaluate(&self, workspace: &Path) -> Result<EngineeringEvaluation, EngineeringEvaluationError> {
        let admissibility = self.collector.collect(workspace)?;
        let verdict = admissibility.verdict();
        if !verdict.admissible {
            return Ok(EngineeringEvaluation {
                admissibility,
                verdict,
                ranking: None,
            });
        }
        let ranking = self.ranker.rank(workspace)?;
        Ok(EngineeringEvaluation {
            admissibility,
            verdict,
            ranking: Some(ranking),
        })
    }
}

/// Build/test evidence produced from the existing DGM evaluator.
///
/// `Fitness::score` is intentionally ignored here: Cargo/build/test remains one
/// hard-evidence source, never the complete engineering policy or ranking gate.
#[derive(Debug, Clone, PartialEq)]
pub struct CargoGateEvidence {
    pub build: EngineeringCheck,
    pub required_tests: EngineeringCheck,
    pub measured_fitness: Fitness,
}

pub struct CargoEvidenceSource<E> {
    evaluator: E,
}

impl<E> CargoEvidenceSource<E>
where
    E: Evaluator,
{
    pub fn new(evaluator: E) -> Self {
        Self { evaluator }
    }

    pub fn collect(&self, workspace: &Path) -> Result<CargoGateEvidence, EngineeringEvaluationError> {
        let fitness = self
            .evaluator
            .evaluate(workspace)
            .map_err(|error| EngineeringEvaluationError::Evidence(error.to_string()))?;
        let build = if fitness.compiles {
            EngineeringCheck::pass("build succeeded")
        } else {
            EngineeringCheck::fail(if fitness.notes.is_empty() {
                "build failed".to_string()
            } else {
                fitness.notes.clone()
            })
        };
        let required_tests = if !fitness.compiles {
            EngineeringCheck::unknown("required tests unavailable because build failed")
        } else if fitness.tests_failed == 0 {
            EngineeringCheck::pass(format!(
                "required tests passed: {} passed, 0 failed",
                fitness.tests_passed
            ))
        } else {
            EngineeringCheck::fail(format!(
                "required tests failed: {} passed, {} failed",
                fitness.tests_passed, fitness.tests_failed
            ))
        };
        Ok(CargoGateEvidence {
            build,
            required_tests,
            measured_fitness: fitness,
        })
    }
}

/// Explicit assembler for the remaining independent hard evidence.
///
/// This keeps build/tests sourced from Cargo while parity, provenance,
/// determinism, resource ceilings and policy checks are supplied by their own
/// authorities. No default `Pass` exists for a missing required check.
#[derive(Debug, Clone)]
pub struct EngineeringEvidenceAssembler {
    pub numerical_parity: EngineeringCheck,
    pub provenance: EngineeringCheck,
    pub deterministic_contract: EngineeringCheck,
    pub resource_budget: EngineeringCheck,
    pub policy_checks: Vec<EngineeringPolicyCheck>,
}

impl EngineeringEvidenceAssembler {
    pub fn assemble(&self, cargo: CargoGateEvidence) -> EngineeringAdmissibility {
        EngineeringAdmissibility {
            build: cargo.build,
            required_tests: cargo.required_tests,
            numerical_parity: self.numerical_parity.clone(),
            provenance: self.provenance.clone(),
            deterministic_contract: self.deterministic_contract.clone(),
            resource_budget: self.resource_budget.clone(),
            policy_checks: self.policy_checks.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineeringEvaluationError {
    Evidence(String),
    Ranking(String),
    InvalidRankingScore,
}

impl fmt::Display for EngineeringEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evidence(message) => write!(f, "engineering evidence collection failed: {message}"),
            Self::Ranking(message) => write!(f, "engineering ranking failed: {message}"),
            Self::InvalidRankingScore => write!(f, "engineering ranking score must be finite"),
        }
    }
}

impl std::error::Error for EngineeringEvaluationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use cogno_core::{CheckStatus, EngineeringGateKind};
    use std::cell::Cell;

    #[derive(Clone)]
    struct FixedCollector(EngineeringAdmissibility);

    impl EngineeringEvidenceCollector for FixedCollector {
        fn collect(
            &self,
            _workspace: &Path,
        ) -> Result<EngineeringAdmissibility, EngineeringEvaluationError> {
            Ok(self.0.clone())
        }
    }

    struct CountingRanker<'a> {
        calls: &'a Cell<u32>,
        score: f64,
    }

    impl EngineeringRanker for CountingRanker<'_> {
        fn rank(&self, _workspace: &Path) -> Result<RankingEvidence, EngineeringEvaluationError> {
            self.calls.set(self.calls.get() + 1);
            RankingEvidence::new(self.score, "measured benchmark")
        }
    }

    fn pass() -> EngineeringCheck {
        EngineeringCheck::pass("ok")
    }

    fn admissible() -> EngineeringAdmissibility {
        EngineeringAdmissibility {
            build: pass(),
            required_tests: pass(),
            numerical_parity: pass(),
            provenance: pass(),
            deterministic_contract: pass(),
            resource_budget: pass(),
            policy_checks: vec![],
        }
    }

    #[test]
    fn ranker_is_never_called_after_hard_gate_failure() {
        let mut evidence = admissible();
        evidence.numerical_parity = EngineeringCheck::fail("oracle mismatch");
        let calls = Cell::new(0);
        let evaluator = CognoEngineeringEvaluator::new(
            FixedCollector(evidence),
            CountingRanker {
                calls: &calls,
                score: 1.0e30,
            },
        );
        let result = evaluator.evaluate(Path::new(".")).unwrap();
        assert!(!result.verdict.admissible);
        assert_eq!(
            result.verdict.first_violation,
            Some(EngineeringGateKind::NumericalParity)
        );
        assert_eq!(calls.get(), 0);
        assert!(result.ranking.is_none());
    }

    #[test]
    fn unknown_gate_skips_ranking_fail_closed() {
        let mut evidence = admissible();
        evidence.provenance = EngineeringCheck::unknown("missing attestation");
        let calls = Cell::new(0);
        let evaluator = CognoEngineeringEvaluator::new(
            FixedCollector(evidence),
            CountingRanker {
                calls: &calls,
                score: 5.0,
            },
        );
        let result = evaluator.evaluate(Path::new(".")).unwrap();
        assert!(!result.verdict.admissible);
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn ranker_runs_exactly_once_after_all_gates_pass() {
        let calls = Cell::new(0);
        let evaluator = CognoEngineeringEvaluator::new(
            FixedCollector(admissible()),
            CountingRanker {
                calls: &calls,
                score: 7.5,
            },
        );
        let result = evaluator.evaluate(Path::new(".")).unwrap();
        assert!(result.verdict.admissible);
        assert_eq!(calls.get(), 1);
        assert_eq!(result.ranking.unwrap().score, 7.5);
    }

    #[test]
    fn cargo_source_ignores_scalar_score_for_hard_decision() {
        let evaluator = crate::dgm::ClosureEvaluator::new(|_: &Path| Fitness {
            compiles: false,
            tests_passed: 0,
            tests_failed: 0,
            score: 1.0e300,
            notes: "compiler rejected candidate".to_string(),
        });
        let cargo = CargoEvidenceSource::new(evaluator)
            .collect(Path::new("."))
            .unwrap();
        assert_eq!(cargo.build.status, CheckStatus::Fail);
        assert_eq!(cargo.required_tests.status, CheckStatus::Unknown);
        assert_eq!(cargo.measured_fitness.score, 1.0e300);
    }

    #[test]
    fn cargo_test_failure_maps_to_required_test_failure() {
        let evaluator = crate::dgm::ClosureEvaluator::new(|_: &Path| Fitness {
            compiles: true,
            tests_passed: 10,
            tests_failed: 1,
            score: 99.0,
            notes: String::new(),
        });
        let cargo = CargoEvidenceSource::new(evaluator)
            .collect(Path::new("."))
            .unwrap();
        assert_eq!(cargo.build.status, CheckStatus::Pass);
        assert_eq!(cargo.required_tests.status, CheckStatus::Fail);
    }

    #[test]
    fn non_finite_ranking_is_rejected_after_admissibility() {
        let calls = Cell::new(0);
        let evaluator = CognoEngineeringEvaluator::new(
            FixedCollector(admissible()),
            CountingRanker {
                calls: &calls,
                score: f64::NAN,
            },
        );
        assert!(matches!(
            evaluator.evaluate(Path::new(".")),
            Err(EngineeringEvaluationError::InvalidRankingScore)
        ));
        assert_eq!(calls.get(), 1);
    }
}
