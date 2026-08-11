//! COGNO-gated FLAT-ATTENTION cross-repository evaluator for P6.4.
//!
//! Hard evidence is collected first through the bounded declarative pipeline.
//! The benchmark plan is executed only when every required COGNO gate passes.
//! Candidate-controlled shell commands are never introduced here: command
//! resolution remains delegated to the trusted [`EvaluationCommandHost`].

use crate::cross_repo_workspace::CrossRepoWorkspace;
use crate::evaluation_pipeline::{
    BoundedEvidencePipeline, EvaluationCommandHost, EvaluationEvidence, EvaluationPipelineError,
    EvaluationPlan, EvidenceKind, StepEvidence,
};
use cogno_core::{
    EngineeringAdmissibility, EngineeringAdmissibilityVerdict, EngineeringCheck,
    EngineeringPolicyCheck,
};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatAttentionEvaluation {
    pub hard_evidence: EvaluationEvidence,
    pub admissibility: EngineeringAdmissibility,
    pub verdict: EngineeringAdmissibilityVerdict,
    pub benchmark_evidence: Option<EvaluationEvidence>,
}

pub struct FlatAttentionEvaluator;

impl FlatAttentionEvaluator {
    pub fn evaluate<H: EvaluationCommandHost>(
        workspace: &CrossRepoWorkspace,
        hard_plan: &EvaluationPlan,
        benchmark_plan: &EvaluationPlan,
        host: &H,
    ) -> Result<FlatAttentionEvaluation, FlatAttentionEvaluationError> {
        validate_hard_plan(hard_plan)?;
        validate_benchmark_plan(benchmark_plan)?;

        let hard_evidence = BoundedEvidencePipeline::run(workspace, hard_plan, host)?;
        let admissibility = assemble_admissibility(&hard_evidence);
        let verdict = admissibility.verdict();
        if !verdict.admissible {
            return Ok(FlatAttentionEvaluation {
                hard_evidence,
                admissibility,
                verdict,
                benchmark_evidence: None,
            });
        }

        let benchmark_evidence = BoundedEvidencePipeline::run(workspace, benchmark_plan, host)?;
        if benchmark_evidence.steps.iter().any(|step| !step.success) {
            return Err(FlatAttentionEvaluationError::BenchmarkFailed);
        }

        Ok(FlatAttentionEvaluation {
            hard_evidence,
            admissibility,
            verdict,
            benchmark_evidence: Some(benchmark_evidence),
        })
    }
}

#[derive(Debug)]
pub enum FlatAttentionEvaluationError {
    InvalidPlan(String),
    Pipeline(EvaluationPipelineError),
    BenchmarkFailed,
}

impl fmt::Display for FlatAttentionEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(message) => write!(f, "invalid FLAT evaluation plan: {message}"),
            Self::Pipeline(error) => write!(f, "FLAT evidence pipeline failed: {error}"),
            Self::BenchmarkFailed => write!(f, "FLAT benchmark evidence contains a failed step"),
        }
    }
}

impl std::error::Error for FlatAttentionEvaluationError {}

impl From<EvaluationPipelineError> for FlatAttentionEvaluationError {
    fn from(error: EvaluationPipelineError) -> Self {
        Self::Pipeline(error)
    }
}

fn validate_hard_plan(plan: &EvaluationPlan) -> Result<(), FlatAttentionEvaluationError> {
    if plan
        .steps()
        .iter()
        .any(|step| matches!(step.evidence_kind, EvidenceKind::Benchmark))
    {
        return Err(FlatAttentionEvaluationError::InvalidPlan(
            "benchmark steps are forbidden in the hard-gate plan".into(),
        ));
    }
    Ok(())
}

fn validate_benchmark_plan(plan: &EvaluationPlan) -> Result<(), FlatAttentionEvaluationError> {
    if plan
        .steps()
        .iter()
        .any(|step| !matches!(step.evidence_kind, EvidenceKind::Benchmark))
    {
        return Err(FlatAttentionEvaluationError::InvalidPlan(
            "benchmark plan may contain only benchmark evidence".into(),
        ));
    }
    Ok(())
}

fn assemble_admissibility(evidence: &EvaluationEvidence) -> EngineeringAdmissibility {
    EngineeringAdmissibility {
        build: aggregate_required(evidence, EvidenceKind::Build, "build"),
        required_tests: aggregate_required(
            evidence,
            EvidenceKind::RequiredTests,
            "required tests",
        ),
        numerical_parity: aggregate_required(
            evidence,
            EvidenceKind::NumericalParity,
            "numerical parity",
        ),
        provenance: aggregate_required(evidence, EvidenceKind::Provenance, "provenance"),
        deterministic_contract: aggregate_required(
            evidence,
            EvidenceKind::Determinism,
            "determinism",
        ),
        resource_budget: aggregate_required(
            evidence,
            EvidenceKind::ResourceBudget,
            "resource budget",
        ),
        policy_checks: policy_checks(evidence),
    }
}

fn aggregate_required(
    evidence: &EvaluationEvidence,
    kind: EvidenceKind,
    label: &'static str,
) -> EngineeringCheck {
    let matching: Vec<&StepEvidence> = evidence
        .steps
        .iter()
        .filter(|step| step.evidence_kind == kind)
        .collect();
    if matching.is_empty() {
        return EngineeringCheck::unknown(format!("missing required {label} evidence"));
    }
    if let Some(failed) = matching.iter().find(|step| !step.success) {
        return EngineeringCheck::fail(format!(
            "{label} failed in {}:{} (exit={:?}, timeout={})",
            failed.repository_role, failed.command_kind, failed.exit_code, failed.timed_out
        ));
    }
    EngineeringCheck::pass(format!(
        "{label} evidence passed: {} step(s)",
        matching.len()
    ))
}

fn policy_checks(evidence: &EvaluationEvidence) -> Vec<EngineeringPolicyCheck> {
    let checks: Vec<EngineeringPolicyCheck> = evidence
        .steps
        .iter()
        .filter(|step| matches!(step.evidence_kind, EvidenceKind::Policy))
        .map(|step| {
            let check = if step.success {
                EngineeringCheck::pass(format!(
                    "{}:{} passed",
                    step.repository_role, step.command_kind
                ))
            } else {
                EngineeringCheck::fail(format!(
                    "{}:{} failed (exit={:?}, timeout={})",
                    step.repository_role, step.command_kind, step.exit_code, step.timed_out
                ))
            };
            EngineeringPolicyCheck::new(
                format!("{}:{}", step.repository_role, step.command_kind),
                check,
            )
        })
        .collect();

    if checks.is_empty() {
        vec![EngineeringPolicyCheck::new(
            "required-policy-evidence",
            EngineeringCheck::unknown("missing required policy evidence"),
        )]
    } else {
        checks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation_pipeline::{EvaluationPlanPolicy, EvaluationStep};
    use cogno_core::{CheckStatus, EngineeringGateKind};

    fn step(kind: EvidenceKind, command: &str, success: bool) -> StepEvidence {
        StepEvidence {
            repository_role: "target".into(),
            command_kind: command.into(),
            evidence_kind: kind,
            exit_code: Some(if success { 0 } else { 1 }),
            success,
            timed_out: false,
            stdout: vec![],
            stderr: vec![],
            output_truncated: false,
        }
    }

    fn all_pass_evidence() -> EvaluationEvidence {
        EvaluationEvidence {
            steps: vec![
                step(EvidenceKind::Build, "build", true),
                step(EvidenceKind::RequiredTests, "tests", true),
                step(EvidenceKind::NumericalParity, "parity", true),
                step(EvidenceKind::Provenance, "provenance", true),
                step(EvidenceKind::Determinism, "determinism", true),
                step(EvidenceKind::ResourceBudget, "resources", true),
                step(EvidenceKind::Policy, "allowlist", true),
            ],
        }
    }

    #[test]
    fn missing_required_gate_is_unknown_and_fail_closed() {
        let mut evidence = all_pass_evidence();
        evidence
            .steps
            .retain(|step| step.evidence_kind != EvidenceKind::NumericalParity);
        let admissibility = assemble_admissibility(&evidence);
        assert_eq!(admissibility.numerical_parity.status, CheckStatus::Unknown);
        assert_eq!(
            admissibility.verdict().first_violation,
            Some(EngineeringGateKind::NumericalParity)
        );
    }

    #[test]
    fn failed_hard_step_rejects_before_benchmark() {
        let mut evidence = all_pass_evidence();
        evidence
            .steps
            .iter_mut()
            .find(|step| step.evidence_kind == EvidenceKind::RequiredTests)
            .unwrap()
            .success = false;
        let admissibility = assemble_admissibility(&evidence);
        assert!(!admissibility.verdict().admissible);
        assert_eq!(
            admissibility.verdict().first_violation,
            Some(EngineeringGateKind::RequiredTests)
        );
    }

    #[test]
    fn missing_policy_evidence_is_fail_closed() {
        let mut evidence = all_pass_evidence();
        evidence
            .steps
            .retain(|step| step.evidence_kind != EvidenceKind::Policy);
        let admissibility = assemble_admissibility(&evidence);
        assert_eq!(admissibility.policy_checks.len(), 1);
        assert_eq!(admissibility.policy_checks[0].check.status, CheckStatus::Unknown);
        assert_eq!(
            admissibility.verdict().first_violation,
            Some(EngineeringGateKind::Policy)
        );
    }

    #[test]
    fn benchmark_steps_cannot_be_smuggled_into_hard_plan() {
        let policy = EvaluationPlanPolicy::new(2, 1_000, 1_024, 4, 128).unwrap();
        let benchmark = EvaluationStep::new(
            "scirust",
            "flat-benchmark",
            vec![],
            500,
            256,
            EvidenceKind::Benchmark,
        )
        .unwrap();
        let plan = EvaluationPlan::new(vec![benchmark], policy).unwrap();
        assert!(validate_hard_plan(&plan).is_err());
    }

    #[test]
    fn benchmark_plan_rejects_hard_gate_steps() {
        let policy = EvaluationPlanPolicy::new(2, 1_000, 1_024, 4, 128).unwrap();
        let build = EvaluationStep::new(
            "scirust",
            "build",
            vec![],
            500,
            256,
            EvidenceKind::Build,
        )
        .unwrap();
        let plan = EvaluationPlan::new(vec![build], policy).unwrap();
        assert!(validate_benchmark_plan(&plan).is_err());
    }
}
