//! P9.2 deterministic end-to-end release qualification.
//!
//! This module executes the repository-local parts of the release proof without
//! network access or live-tree mutation. The dedicated P9 GitHub workflow adds
//! the external exact-revision replay: SciAgent consumes the trajectory emitted
//! here and SciRust replays the qualified FLAT M15 integration on lavapipe.

use crate::candidate_state::CandidateStoragePolicy;
use crate::compatibility::{CompatibilitySet, RepositoryRevision};
use crate::cross_repo_workspace::{
    CrossRepoWorkspace, CrossRepoWorkspacePolicy, LocalRepositorySource,
};
use crate::cumulative_archive::CumulativeArchive;
use crate::dgm::{ClosureEvaluator, Fitness};
use crate::engineering_evaluator::{
    CognoEngineeringEvaluator, EngineeringEvidenceCollector, EngineeringEvaluationError,
    EngineeringRanker, RankingEvidence,
};
use crate::engineering_trajectory::{
    AdmissibilityBreakdown, BenchmarkRecord, EngineeringTrajectory, EngineeringVerdict, GateStatus,
    ProposerMetadata,
};
use crate::evaluation_pipeline::{
    EvaluationCommandHost, EvaluationPlan, EvaluationPlanPolicy, EvaluationStep, EvidenceKind,
    ResolvedCommand,
};
use crate::flat_attention_evaluator::FlatAttentionEvaluator;
use crate::json::Json;
use crate::patchset::{FileOperation, PatchSet};
use crate::release_compatibility::{ReleaseCompatibilityLock, current_release_compatibility_lock};
use cogno_core::{EngineeringAdmissibility, EngineeringCheck};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub const RELEASE_QUALIFICATION_SCHEMA_VERSION: u64 = 1;
static QUALIFICATION_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseQualificationReport {
    pub compatibility_fingerprint: String,
    pub cumulative_multi_file_lineage: bool,
    pub cogno_rejects_high_score_invalid: bool,
    pub cross_repo_flat_scirust_evaluation: bool,
    pub exact_revision_lock_replay: bool,
    pub trajectory_exported: bool,
    pub live_tree_unchanged: bool,
    pub external_sciagent_ingestion_required: bool,
    pub external_final_head_ci_required: bool,
}

impl ReleaseQualificationReport {
    pub fn local_contract_passed(&self) -> bool {
        self.cumulative_multi_file_lineage
            && self.cogno_rejects_high_score_invalid
            && self.cross_repo_flat_scirust_evaluation
            && self.exact_revision_lock_replay
            && self.trajectory_exported
            && self.live_tree_unchanged
            && self.external_sciagent_ingestion_required
            && self.external_final_head_ci_required
    }

    pub fn to_json_string(&self) -> String {
        let mut root = Json::obj();
        root.set(
            "cogno_rejects_high_score_invalid",
            Json::Bool(self.cogno_rejects_high_score_invalid),
        )
        .set(
            "compatibility_fingerprint",
            Json::Str(self.compatibility_fingerprint.clone()),
        )
        .set(
            "cross_repo_flat_scirust_evaluation",
            Json::Bool(self.cross_repo_flat_scirust_evaluation),
        )
        .set(
            "cumulative_multi_file_lineage",
            Json::Bool(self.cumulative_multi_file_lineage),
        )
        .set(
            "exact_revision_lock_replay",
            Json::Bool(self.exact_revision_lock_replay),
        )
        .set(
            "external_final_head_ci_required",
            Json::Bool(self.external_final_head_ci_required),
        )
        .set(
            "external_sciagent_ingestion_required",
            Json::Bool(self.external_sciagent_ingestion_required),
        )
        .set("live_tree_unchanged", Json::Bool(self.live_tree_unchanged))
        .set(
            "local_contract_passed",
            Json::Bool(self.local_contract_passed()),
        )
        .set(
            "schema_version",
            Json::Num(RELEASE_QUALIFICATION_SCHEMA_VERSION as f64),
        )
        .set("trajectory_exported", Json::Bool(self.trajectory_exported));
        root.to_string()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReleaseQualificationArtifacts {
    pub report: ReleaseQualificationReport,
    pub trajectory: EngineeringTrajectory,
}

impl ReleaseQualificationArtifacts {
    pub fn trajectory_json(&self) -> Result<String, ReleaseQualificationError> {
        self.trajectory
            .to_json_string()
            .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))
    }

    pub fn write_to(&self, root: &Path) -> Result<(), ReleaseQualificationError> {
        std::fs::create_dir_all(root)?;
        std::fs::write(root.join("qualification-report.json"), self.report.to_json_string())?;
        std::fs::write(root.join("engineering-trajectory.json"), self.trajectory_json()?)?;
        Ok(())
    }
}

pub fn run_local_release_qualification(
) -> Result<ReleaseQualificationArtifacts, ReleaseQualificationError> {
    let lock = current_release_compatibility_lock()
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let lock_round_trip = ReleaseCompatibilityLock::from_json_str(&lock.to_json_string())
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let exact_revision_lock_replay = lock_round_trip.fingerprint() == lock.fingerprint()
        && lock
            .compatibility()
            .revisions()
            .iter()
            .all(|revision| matches!(revision.revision.len(), 40 | 64));

    let fixture = QualificationFixture::new()?;
    let original_live = std::fs::read(fixture.live.join("state.txt"))?;
    let (cumulative_multi_file_lineage, parent_state_id, trajectory_patch) =
        prove_cumulative_lineage(&fixture.live)?;
    let live_tree_unchanged = std::fs::read(fixture.live.join("state.txt"))? == original_live
        && !fixture.live.join("a.txt").exists()
        && !fixture.live.join("b.txt").exists();

    let cogno_rejects_high_score_invalid = prove_cogno_rejection(&fixture.live)?;
    let cross_repo_flat_scirust_evaluation = prove_cross_repo_flat_scirust(&fixture)?;

    let trajectory = build_trajectory(&lock, parent_state_id, trajectory_patch)?;
    let trajectory_json = trajectory
        .to_json_string()
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let reparsed = EngineeringTrajectory::from_json_str(&trajectory_json)
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let trajectory_exported = reparsed == trajectory;

    let report = ReleaseQualificationReport {
        compatibility_fingerprint: lock.fingerprint(),
        cumulative_multi_file_lineage,
        cogno_rejects_high_score_invalid,
        cross_repo_flat_scirust_evaluation,
        exact_revision_lock_replay,
        trajectory_exported,
        live_tree_unchanged,
        external_sciagent_ingestion_required: true,
        external_final_head_ci_required: true,
    };
    if !report.local_contract_passed() {
        return Err(ReleaseQualificationError::Contract(
            "one or more local P9 release gates did not pass".to_string(),
        ));
    }
    Ok(ReleaseQualificationArtifacts { report, trajectory })
}

fn prove_cumulative_lineage(
    live: &Path,
) -> Result<(bool, String, PatchSet), ReleaseQualificationError> {
    let baseline = Fitness {
        compiles: true,
        tests_passed: 1,
        tests_failed: 0,
        score: 0.0,
        notes: "baseline".to_string(),
    };
    let policy = CandidateStoragePolicy::new(64, 1_000_000)
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let evaluator = ClosureEvaluator::new(|root: &Path| {
        let state = std::fs::read_to_string(root.join("state.txt")).unwrap_or_default();
        let score = if state.contains("phase-b") {
            2.0
        } else if state.contains("phase-a") {
            1.0
        } else {
            0.0
        };
        Fitness {
            compiles: true,
            tests_passed: 1,
            tests_failed: 0,
            score,
            notes: "deterministic fixture evaluator".to_string(),
        }
    });
    let mut archive = CumulativeArchive::new(live, baseline, policy)
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let baseline_id = archive.baseline_state_id().to_string();

    let phase_a = PatchSet::new(vec![
        FileOperation::modify_exact("state.txt", "base\n", "base\nphase-a\n"),
        FileOperation::create("a.txt", "ancestor\n"),
    ])
    .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let outcome_a = archive
        .evaluate_child(&baseline_id, phase_a, "P9 phase A", &evaluator, true)
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    if !outcome_a.accepted {
        return Err(ReleaseQualificationError::Contract(
            "cumulative phase A was not accepted".to_string(),
        ));
    }

    let phase_b = PatchSet::new(vec![
        FileOperation::modify_exact(
            "state.txt",
            "base\nphase-a\n",
            "base\nphase-a\nphase-b\n",
        ),
        FileOperation::create("b.txt", "descendant\n"),
    ])
    .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let outcome_b = archive
        .evaluate_child(
            &outcome_a.state_id,
            phase_b.clone(),
            "P9 phase B",
            &evaluator,
            true,
        )
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    if !outcome_b.accepted {
        return Err(ReleaseQualificationError::Contract(
            "cumulative phase B was not accepted".to_string(),
        ));
    }
    let root = archive
        .materialized_root(&outcome_b.state_id)
        .ok_or_else(|| ReleaseQualificationError::Contract("missing cumulative child".to_string()))?;
    let state = std::fs::read_to_string(root.join("state.txt"))?;
    let inherited = state == "base\nphase-a\nphase-b\n"
        && root.join("a.txt").is_file()
        && root.join("b.txt").is_file()
        && archive.len() == 3;
    Ok((inherited, outcome_a.state_id, phase_b))
}

#[derive(Clone)]
struct RejectingCollector;

impl EngineeringEvidenceCollector for RejectingCollector {
    fn collect(
        &self,
        _workspace: &Path,
    ) -> Result<EngineeringAdmissibility, EngineeringEvaluationError> {
        let pass = || EngineeringCheck::pass("P9 fixture pass");
        Ok(EngineeringAdmissibility {
            build: pass(),
            required_tests: pass(),
            numerical_parity: EngineeringCheck::fail("intentional P9 parity violation"),
            provenance: pass(),
            deterministic_contract: pass(),
            resource_budget: pass(),
            policy_checks: Vec::new(),
        })
    }
}

struct HighScoreRanker {
    calls: Arc<AtomicUsize>,
}

impl EngineeringRanker for HighScoreRanker {
    fn rank(&self, _workspace: &Path) -> Result<RankingEvidence, EngineeringEvaluationError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        RankingEvidence::new(1.0e300, "intentionally huge invalid-candidate score")
    }
}

fn prove_cogno_rejection(workspace: &Path) -> Result<bool, ReleaseQualificationError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let evaluator = CognoEngineeringEvaluator::new(
        RejectingCollector,
        HighScoreRanker {
            calls: Arc::clone(&calls),
        },
    );
    let result = evaluator
        .evaluate(workspace)
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    Ok(!result.verdict.admissible && result.ranking.is_none() && calls.load(Ordering::SeqCst) == 0)
}

struct QualificationHost;

impl EvaluationCommandHost for QualificationHost {
    fn resolve(
        &self,
        command_kind: &str,
        candidate_arguments: &[String],
        repository_root: &Path,
        _cargo_override_config: Option<&Path>,
    ) -> Result<ResolvedCommand, String> {
        if !candidate_arguments.is_empty() {
            return Err("P9 fixture commands do not accept candidate-controlled arguments".into());
        }
        if !repository_root.join("Cargo.toml").is_file()
            || !repository_root.join("src/lib.rs").is_file()
        {
            return Err("P9 fixture repository is missing its frozen Cargo source".into());
        }
        let arguments = match command_kind {
            "build" => vec!["check", "--quiet", "--offline"],
            "tests" => vec!["test", "--quiet", "--offline", "--all-targets"],
            "parity" => vec![
                "test",
                "--quiet",
                "--offline",
                "numerical_parity_contract",
            ],
            "provenance" => vec!["metadata", "--no-deps", "--format-version", "1", "--offline"],
            "determinism" => vec![
                "test",
                "--quiet",
                "--offline",
                "determinism_contract",
            ],
            "resources" => vec![
                "test",
                "--quiet",
                "--offline",
                "resource_budget_contract",
            ],
            "policy" => vec!["test", "--quiet", "--offline", "policy_contract"],
            "paired-e2e" => vec!["test", "--quiet", "--offline", "benchmark_contract"],
            other => return Err(format!("unknown P9 fixture command kind: {other}")),
        }
        .into_iter()
        .map(str::to_string)
        .collect();
        Ok(ResolvedCommand::new("cargo", arguments))
    }
}

fn prove_cross_repo_flat_scirust(
    fixture: &QualificationFixture,
) -> Result<bool, ReleaseQualificationError> {
    let (scirust_root, scirust_revision) = fixture.git_repository("scirust", "scirust-fixture")?;
    let (flat_root, flat_revision) = fixture.git_repository("flat", "flat-fixture")?;
    let compatibility = CompatibilitySet::new(
        vec![
            RepositoryRevision::new("Memorithm/scirust", &scirust_revision, "scirust")
                .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?,
            RepositoryRevision::new("Memorithm/FLAT-ATTENTION", &flat_revision, "flat")
                .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?,
        ],
        "P9 fixture toolchain",
        vec!["flat-attention:wgpu".to_string()],
    )
    .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let policy = CrossRepoWorkspacePolicy::new(
        2,
        2_000_000,
        CandidateStoragePolicy::new(128, 1_000_000)
            .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?,
    )
    .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let workspace = CrossRepoWorkspace::materialize(
        compatibility,
        vec![
            LocalRepositorySource::new(
                "Memorithm/scirust",
                "scirust",
                &scirust_revision,
                scirust_root,
            )
            .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?,
            LocalRepositorySource::new(
                "Memorithm/FLAT-ATTENTION",
                "flat",
                &flat_revision,
                flat_root,
            )
            .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?,
        ],
        Vec::new(),
        Vec::new(),
        policy,
    )
    .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;

    let plan_policy = EvaluationPlanPolicy::new(8, 30_000, 8_192, 1, 64)
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let hard_kinds = [
        ("scirust", "build", EvidenceKind::Build),
        ("flat", "tests", EvidenceKind::RequiredTests),
        ("scirust", "parity", EvidenceKind::NumericalParity),
        ("flat", "provenance", EvidenceKind::Provenance),
        ("scirust", "determinism", EvidenceKind::Determinism),
        ("flat", "resources", EvidenceKind::ResourceBudget),
        ("scirust", "policy", EvidenceKind::Policy),
    ];
    let hard_steps = hard_kinds
        .into_iter()
        .map(|(role, command, kind)| {
            EvaluationStep::new(role, command, Vec::new(), 20_000, 4_096, kind)
                .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let hard_plan = EvaluationPlan::new(hard_steps, plan_policy)
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let benchmark_plan = EvaluationPlan::new(
        vec![EvaluationStep::new(
            "flat",
            "paired-e2e",
            Vec::new(),
            20_000,
            4_096,
            EvidenceKind::Benchmark,
        )
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?],
        plan_policy,
    )
    .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    let result = FlatAttentionEvaluator::evaluate(
        &workspace,
        &hard_plan,
        &benchmark_plan,
        &QualificationHost,
    )
    .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    Ok(result.verdict.admissible
        && result.benchmark_evidence.is_some()
        && result
            .benchmark_evidence
            .as_ref()
            .is_some_and(|evidence| evidence.steps.iter().all(|step| step.success)))
}

fn build_trajectory(
    lock: &ReleaseCompatibilityLock,
    parent_state_id: String,
    patch_set: PatchSet,
) -> Result<EngineeringTrajectory, ReleaseQualificationError> {
    let trajectory = EngineeringTrajectory {
        task_spec_id: lock.fingerprint(),
        compatibility: lock.compatibility().clone(),
        parent_state_id,
        patch_set,
        proposer: ProposerMetadata::new("rsi-p9", "deterministic-harness", "qualified-v1")
            .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?,
        compiler_test_device_evidence: vec![
            "P9 cumulative lineage: pass".to_string(),
            "P9 COGNO high-score invalid rejection: pass".to_string(),
            "P9 cross-repo FLAT+SciRust hard-gate fixture commands: pass".to_string(),
            "P9 exact compatibility lock replay: pass".to_string(),
            "P9 live source tree mutation: none".to_string(),
        ],
        admissibility: AdmissibilityBreakdown {
            build: GateStatus::Pass,
            required_tests: GateStatus::Pass,
            numerical_parity: GateStatus::Pass,
            provenance: GateStatus::Pass,
            deterministic_contract: GateStatus::Pass,
            resource_budget: GateStatus::Pass,
            policy_checks: GateStatus::Pass,
        },
        benchmarks: vec![BenchmarkRecord::new(
            "p9-local-e2e-contract",
            "boolean",
            vec![1.0],
            1.0,
        )
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?],
        verdict: EngineeringVerdict::Accepted,
        verdict_reason: "all local P9.2 hard gates passed; external exact-revision CI remains mandatory"
            .to_string(),
        later_verdicts: Vec::new(),
    };
    trajectory
        .validate()
        .map_err(|error| ReleaseQualificationError::Contract(error.to_string()))?;
    Ok(trajectory)
}

struct QualificationFixture {
    root: PathBuf,
    live: PathBuf,
}

impl QualificationFixture {
    fn new() -> Result<Self, ReleaseQualificationError> {
        let sequence = QUALIFICATION_SEQ.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rsi-p9-release-qualification-{}-{sequence}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let live = root.join("live");
        std::fs::create_dir_all(&live)?;
        std::fs::write(live.join("state.txt"), "base\n")?;
        Ok(Self { root, live })
    }

    fn git_repository(
        &self,
        name: &str,
        package: &str,
    ) -> Result<(PathBuf, String), ReleaseQualificationError> {
        let root = self.root.join(format!("git-{name}"));
        std::fs::create_dir_all(root.join("src"))?;
        run_git(&root, &["init", "-q"])?;
        run_git(&root, &["config", "user.email", "rsi-p9@example.invalid"])?;
        run_git(&root, &["config", "user.name", "RSI P9"])?;
        std::fs::write(
            root.join("Cargo.toml"),
            format!(
                "[package]\nname = \"{package}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n"
            ),
        )?;
        let source = format!(
            r#"pub const COMPONENT: &str = "{name}";

pub fn deterministic_value(input: u64) -> u64 {{
    input.wrapping_mul(31).wrapping_add(7)
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn numerical_parity_contract() {{
        assert_eq!(deterministic_value(3), 100);
    }}

    #[test]
    fn determinism_contract() {{
        assert_eq!(deterministic_value(41), deterministic_value(41));
    }}

    #[test]
    fn resource_budget_contract() {{
        assert!(core::mem::size_of::<(u64, u64)>() <= 16);
    }}

    #[test]
    fn policy_contract() {{
        assert!(!COMPONENT.is_empty());
        assert!(COMPONENT.bytes().all(|byte| byte.is_ascii_lowercase()));
    }}

    #[test]
    fn benchmark_contract() {{
        let mut checksum = 0_u64;
        for index in 0..1024_u64 {{
            checksum = deterministic_value(checksum ^ index);
        }}
        assert_ne!(checksum, 0);
    }}
}}
"#
        );
        std::fs::write(root.join("src/lib.rs"), source)?;
        run_git(&root, &["add", "."])?;
        run_git(&root, &["commit", "-qm", "P9 fixture"])?;
        let revision = git_output(&root, &["rev-parse", "HEAD"])?;
        Ok((root, revision))
    }
}

impl Drop for QualificationFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseQualificationError {
    Io(String),
    Git(String),
    Contract(String),
}

impl fmt::Display for ReleaseQualificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "release qualification error: {self:?}")
    }
}

impl std::error::Error for ReleaseQualificationError {}

impl From<std::io::Error> for ReleaseQualificationError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

fn run_git(root: &Path, arguments: &[&str]) -> Result<(), ReleaseQualificationError> {
    let status = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(arguments)
        .status()
        .map_err(|error| ReleaseQualificationError::Git(error.to_string()))?;
    if status.success() {
        Ok(())
    } else {
        Err(ReleaseQualificationError::Git(format!(
            "git {} failed with {status}",
            arguments.join(" ")
        )))
    }
}

fn git_output(root: &Path, arguments: &[&str]) -> Result<String, ReleaseQualificationError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(arguments)
        .output()
        .map_err(|error| ReleaseQualificationError::Git(error.to_string()))?;
    if !output.status.success() {
        return Err(ReleaseQualificationError::Git(format!(
            "git {} failed with {}",
            arguments.join(" "),
            output.status
        )));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|error| ReleaseQualificationError::Git(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_p9_release_contract_passes_and_is_replayable() {
        let first = run_local_release_qualification().unwrap();
        let second = run_local_release_qualification().unwrap();
        assert!(first.report.local_contract_passed());
        assert!(second.report.local_contract_passed());
        assert_eq!(
            first.report.compatibility_fingerprint,
            second.report.compatibility_fingerprint
        );
        assert_eq!(first.trajectory_json().unwrap(), second.trajectory_json().unwrap());
    }

    #[test]
    fn emitted_trajectory_is_accepted_v3_and_bound_to_release_lock() {
        let artifacts = run_local_release_qualification().unwrap();
        let lock = current_release_compatibility_lock().unwrap();
        assert_eq!(artifacts.trajectory.task_spec_id, lock.fingerprint());
        assert_eq!(artifacts.trajectory.compatibility, *lock.compatibility());
        assert_eq!(artifacts.trajectory.verdict, EngineeringVerdict::Accepted);
        assert!(artifacts.trajectory.admissibility.is_admissible());
        let encoded = artifacts.trajectory_json().unwrap();
        assert_eq!(
            EngineeringTrajectory::from_json_str(&encoded).unwrap(),
            artifacts.trajectory
        );
    }

    #[test]
    fn artifact_writer_emits_report_and_trajectory_without_live_repo_mutation() {
        let artifacts = run_local_release_qualification().unwrap();
        let sequence = QUALIFICATION_SEQ.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rsi-p9-artifact-test-{}-{sequence}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        artifacts.write_to(&root).unwrap();
        assert!(root.join("qualification-report.json").is_file());
        assert!(root.join("engineering-trajectory.json").is_file());
        let report = std::fs::read_to_string(root.join("qualification-report.json")).unwrap();
        assert!(report.contains("\"local_contract_passed\":true"));
        let _ = std::fs::remove_dir_all(root);
    }
}
