//! P8.5 AUTOPILOT PR emission and human-review flywheel contract.
//!
//! The core remains network-free and cannot merge anything. It binds an
//! accepted engineering trajectory to one exact P8 task/repository role, emits
//! only a branch-creation plan followed by a pull-request plan, and accepts
//! exact-head CI/human-review evidence back into the same trajectory.

use crate::autopilot_intake::FrozenAutopilotSpec;
use crate::autopilot_perf::{
    FrozenPerfBenchmark, PerfCaseResult, PerfComparisonReport, PerfMeasurementBatch,
};
use crate::autopilot_task_dag::{AutopilotTask, AutopilotTaskDag, TaskOperation, TaskRegime};
use crate::compatibility::{CompatibilitySet, RepositoryRevision};
use crate::engineering_trajectory::{
    EngineeringTrajectory, EngineeringVerdict, GateStatus, LaterVerdict,
};
use crate::json::Json;
use crate::patchset::{FileOperation, PatchSet};
use crate::sha256::sha256;
use std::fmt;

pub const AUTOPILOT_PR_PLAN_SCHEMA_VERSION: u64 = 1;

/// The complete set of hosting mutations an AUTOPILOT PR plan can request.
/// There is deliberately no default-branch write and no merge action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostingAction {
    CreateBranch,
    OpenPullRequest,
}

impl HostingAction {
    fn as_str(self) -> &'static str {
        match self {
            Self::CreateBranch => "create_branch",
            Self::OpenPullRequest => "open_pull_request",
        }
    }
}

/// Proven P8.4 promotion evidence. The fields are private so callers cannot
/// manufacture a `promotable = true` report without re-running the frozen
/// profile through the P8.4 evaluator.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedPerfPromotion {
    task_id: String,
    profile_id: String,
    profile_sha256: String,
    environment_fingerprint: String,
    trajectory_sha256: String,
    evidence_sha256: String,
    report: PerfComparisonReport,
}

impl VerifiedPerfPromotion {
    pub fn evaluate(
        profile: &FrozenPerfBenchmark,
        trajectory: &EngineeringTrajectory,
        baseline: &[PerfMeasurementBatch],
        candidate: &[PerfMeasurementBatch],
    ) -> Result<Self, AutopilotPrError> {
        profile
            .verify()
            .map_err(|error| AutopilotPrError::InvalidPerfEvidence(error.to_string()))?;
        trajectory
            .validate()
            .map_err(|error| AutopilotPrError::InvalidTrajectory(error.to_string()))?;
        let report = PerfComparisonReport::evaluate(
            profile,
            &trajectory.admissibility,
            baseline,
            candidate,
        )
        .map_err(|error| AutopilotPrError::InvalidPerfEvidence(error.to_string()))?;
        if !report.hard_gates_passed || !report.promotable {
            return Err(AutopilotPrError::PerfCandidateNotPromotable);
        }
        let trajectory_sha256 = immutable_trajectory_identity(trajectory)?;
        let evidence_sha256 = perf_evidence_identity(profile, baseline, candidate, &report);
        Ok(Self {
            task_id: profile.task_id().to_string(),
            profile_id: profile.profile_id().to_string(),
            profile_sha256: profile.profile_sha256().to_string(),
            environment_fingerprint: profile.environment_fingerprint(),
            trajectory_sha256,
            evidence_sha256,
            report,
        })
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn profile_sha256(&self) -> &str {
        &self.profile_sha256
    }

    pub fn evidence_sha256(&self) -> &str {
        &self.evidence_sha256
    }

    pub fn report(&self) -> &PerfComparisonReport {
        &self.report
    }
}

/// Accepted engineering evidence bound to the exact P8 task that produced it.
/// PR emission no longer accepts an independent task ID/repository role, so a
/// trajectory cannot be reinterpreted under another overlapping task contract.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskBoundEngineeringTrajectory {
    spec_sha256: String,
    dag_sha256: String,
    task_id: String,
    repository_role: String,
    trajectory: EngineeringTrajectory,
    perf_promotion: Option<VerifiedPerfPromotion>,
    candidate_sha256: String,
}

impl TaskBoundEngineeringTrajectory {
    pub fn new(
        spec: &FrozenAutopilotSpec,
        dag: &AutopilotTaskDag,
        task_id: impl Into<String>,
        repository_role: impl Into<String>,
        trajectory: EngineeringTrajectory,
        perf_promotion: Option<VerifiedPerfPromotion>,
    ) -> Result<Self, AutopilotPrError> {
        spec.verify()
            .map_err(|error| AutopilotPrError::InvalidSpec(error.to_string()))?;
        dag.verify()
            .map_err(|error| AutopilotPrError::InvalidDag(error.to_string()))?;
        if dag.spec_sha256() != spec.spec_sha256() {
            return Err(AutopilotPrError::SpecDagMismatch);
        }
        trajectory
            .validate()
            .map_err(|error| AutopilotPrError::InvalidTrajectory(error.to_string()))?;
        if trajectory.task_spec_id != spec.spec_sha256() {
            return Err(AutopilotPrError::TrajectorySpecMismatch {
                expected: spec.spec_sha256().to_string(),
                actual: trajectory.task_spec_id.clone(),
            });
        }
        if trajectory.verdict != EngineeringVerdict::Accepted
            || !trajectory.admissibility.is_admissible()
        {
            return Err(AutopilotPrError::TrajectoryNotAccepted);
        }

        let task_id = task_id.into();
        let repository_role = repository_role.into();
        validate_identifier("task_id", &task_id)?;
        validate_identifier("repository_role", &repository_role)?;
        let task = dag
            .task(&task_id)
            .ok_or_else(|| AutopilotPrError::UnknownTask(task_id.clone()))?;
        if !task
            .repository_roles()
            .iter()
            .any(|role| role == &repository_role)
        {
            return Err(AutopilotPrError::RepositoryRoleOutsideTask(
                repository_role,
            ));
        }

        validate_compatibility_against_spec(spec, &trajectory.compatibility)?;
        unique_revision_for_role(&trajectory.compatibility, &repository_role)?;
        validate_patchset_for_target(task, &repository_role, &trajectory.patch_set)?;
        validate_perf_binding(task, &trajectory, perf_promotion.as_ref())?;

        let candidate_sha256 = bound_candidate_identity(
            &trajectory,
            &task_id,
            &repository_role,
            perf_promotion.as_ref(),
        )?;
        Ok(Self {
            spec_sha256: spec.spec_sha256().to_string(),
            dag_sha256: dag.dag_sha256().to_string(),
            task_id,
            repository_role,
            trajectory,
            perf_promotion,
            candidate_sha256,
        })
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn repository_role(&self) -> &str {
        &self.repository_role
    }

    pub fn candidate_sha256(&self) -> &str {
        &self.candidate_sha256
    }

    pub fn trajectory(&self) -> &EngineeringTrajectory {
        &self.trajectory
    }

    pub fn perf_promotion(&self) -> Option<&VerifiedPerfPromotion> {
        self.perf_promotion.as_ref()
    }

    pub fn into_trajectory(self) -> EngineeringTrajectory {
        self.trajectory
    }

    fn verify(
        &self,
        spec: &FrozenAutopilotSpec,
        dag: &AutopilotTaskDag,
    ) -> Result<(), AutopilotPrError> {
        spec.verify()
            .map_err(|error| AutopilotPrError::InvalidSpec(error.to_string()))?;
        dag.verify()
            .map_err(|error| AutopilotPrError::InvalidDag(error.to_string()))?;
        if self.spec_sha256 != spec.spec_sha256()
            || self.dag_sha256 != dag.dag_sha256()
            || dag.spec_sha256() != spec.spec_sha256()
        {
            return Err(AutopilotPrError::BoundCandidateContextMismatch);
        }
        self.trajectory
            .validate()
            .map_err(|error| AutopilotPrError::InvalidTrajectory(error.to_string()))?;
        let task = dag
            .task(&self.task_id)
            .ok_or_else(|| AutopilotPrError::UnknownTask(self.task_id.clone()))?;
        validate_compatibility_against_spec(spec, &self.trajectory.compatibility)?;
        validate_patchset_for_target(task, &self.repository_role, &self.trajectory.patch_set)?;
        validate_perf_binding(task, &self.trajectory, self.perf_promotion.as_ref())?;
        let actual = bound_candidate_identity(
            &self.trajectory,
            &self.task_id,
            &self.repository_role,
            self.perf_promotion.as_ref(),
        )?;
        if actual != self.candidate_sha256 {
            return Err(AutopilotPrError::BoundCandidateHashMismatch {
                expected: self.candidate_sha256.clone(),
                actual,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullRequestPlanDraft {
    pub default_branch: String,
    pub title: String,
}

impl PullRequestPlanDraft {
    pub fn new(default_branch: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            default_branch: default_branch.into(),
            title: title.into(),
        }
    }
}

/// Deterministic, network-free plan consumed by a trusted Git hosting adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutopilotPullRequestPlan {
    schema_version: u64,
    spec_sha256: String,
    dag_sha256: String,
    task_id: String,
    repository_role: String,
    repository: String,
    base_revision: String,
    default_branch: String,
    branch_name: String,
    title: String,
    body: String,
    compatibility_sha256: String,
    patch_set_sha256: String,
    candidate_sha256: String,
    plan_sha256: String,
}

impl AutopilotPullRequestPlan {
    pub fn new(
        spec: &FrozenAutopilotSpec,
        dag: &AutopilotTaskDag,
        candidate: &TaskBoundEngineeringTrajectory,
        draft: PullRequestPlanDraft,
    ) -> Result<Self, AutopilotPrError> {
        candidate.verify(spec, dag)?;
        validate_git_branch_name("default_branch", &draft.default_branch)?;
        validate_text("title", &draft.title)?;
        if draft.title.len() > 240 {
            return Err(AutopilotPrError::TitleTooLong(draft.title.len()));
        }

        let task = dag
            .task(candidate.task_id())
            .ok_or_else(|| AutopilotPrError::UnknownTask(candidate.task_id().to_string()))?;
        let trajectory = candidate.trajectory();
        let target = unique_revision_for_role(
            &trajectory.compatibility,
            candidate.repository_role(),
        )?;
        let patch_set_sha256 = trajectory
            .patch_set
            .identity()
            .map_err(|error| AutopilotPrError::InvalidPatchSet(error.to_string()))?;
        let compatibility_sha256 = trajectory.compatibility.fingerprint();
        let branch_name = format!(
            "autopilot/{}/{}",
            candidate.task_id(),
            &patch_set_sha256[..12]
        );
        validate_git_branch_name("generated_branch", &branch_name)?;
        if branch_name == draft.default_branch {
            return Err(AutopilotPrError::DefaultBranchMutationForbidden(
                draft.default_branch,
            ));
        }

        let body = render_pr_body(PrBodyContext {
            spec,
            dag,
            task,
            candidate,
            target: &target,
            compatibility_sha256: &compatibility_sha256,
            patch_set_sha256: &patch_set_sha256,
        });
        let mut plan = Self {
            schema_version: AUTOPILOT_PR_PLAN_SCHEMA_VERSION,
            spec_sha256: spec.spec_sha256().to_string(),
            dag_sha256: dag.dag_sha256().to_string(),
            task_id: candidate.task_id().to_string(),
            repository_role: candidate.repository_role().to_string(),
            repository: target.repository,
            base_revision: target.revision,
            default_branch: draft.default_branch,
            branch_name,
            title: draft.title,
            body,
            compatibility_sha256,
            patch_set_sha256,
            candidate_sha256: candidate.candidate_sha256().to_string(),
            plan_sha256: String::new(),
        };
        plan.plan_sha256 = plan.compute_sha256();
        Ok(plan)
    }

    pub fn hosting_actions(&self) -> [HostingAction; 2] {
        [HostingAction::CreateBranch, HostingAction::OpenPullRequest]
    }

    pub fn repository(&self) -> &str {
        &self.repository
    }

    pub fn repository_role(&self) -> &str {
        &self.repository_role
    }

    pub fn base_revision(&self) -> &str {
        &self.base_revision
    }

    pub fn default_branch(&self) -> &str {
        &self.default_branch
    }

    pub fn branch_name(&self) -> &str {
        &self.branch_name
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn body(&self) -> &str {
        &self.body
    }

    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }

    pub fn candidate_sha256(&self) -> &str {
        &self.candidate_sha256
    }

    pub fn verify(&self) -> Result<(), AutopilotPrError> {
        validate_git_branch_name("default_branch", &self.default_branch)?;
        validate_git_branch_name("generated_branch", &self.branch_name)?;
        if self.branch_name == self.default_branch {
            return Err(AutopilotPrError::DefaultBranchMutationForbidden(
                self.default_branch.clone(),
            ));
        }
        let actual = self.compute_sha256();
        if actual != self.plan_sha256 {
            return Err(AutopilotPrError::PlanHashMismatch {
                expected: self.plan_sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    pub fn to_json_string(&self) -> String {
        let mut root = self.unsigned_json();
        root.set("plan_sha256", Json::Str(self.plan_sha256.clone()));
        root.to_string()
    }

    fn compute_sha256(&self) -> String {
        hex_digest(&sha256(self.unsigned_json().to_string().as_bytes()))
    }

    fn unsigned_json(&self) -> Json {
        let mut root = Json::obj();
        root.set("base_revision", Json::Str(self.base_revision.clone()))
            .set("body", Json::Str(self.body.clone()))
            .set("branch_name", Json::Str(self.branch_name.clone()))
            .set(
                "candidate_sha256",
                Json::Str(self.candidate_sha256.clone()),
            )
            .set(
                "compatibility_sha256",
                Json::Str(self.compatibility_sha256.clone()),
            )
            .set("dag_sha256", Json::Str(self.dag_sha256.clone()))
            .set("default_branch", Json::Str(self.default_branch.clone()))
            .set(
                "hosting_actions",
                Json::Arr(
                    self.hosting_actions()
                        .iter()
                        .map(|action| Json::Str(action.as_str().to_string()))
                        .collect(),
                ),
            )
            .set(
                "patch_set_sha256",
                Json::Str(self.patch_set_sha256.clone()),
            )
            .set("repository", Json::Str(self.repository.clone()))
            .set(
                "repository_role",
                Json::Str(self.repository_role.clone()),
            )
            .set("schema_version", Json::Num(self.schema_version as f64))
            .set("spec_sha256", Json::Str(self.spec_sha256.clone()))
            .set("task_id", Json::Str(self.task_id.clone()))
            .set("title", Json::Str(self.title.clone()));
        root
    }
}

/// Immutable result returned by the trusted hosting adapter after branch/PR
/// creation. Authentication belongs to that adapter, not to model output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullRequestEmissionReceipt {
    plan_sha256: String,
    repository: String,
    pull_request_number: u64,
    branch_name: String,
    head_revision: String,
}

impl PullRequestEmissionReceipt {
    pub fn new(
        plan: &AutopilotPullRequestPlan,
        pull_request_number: u64,
        head_revision: impl Into<String>,
    ) -> Result<Self, AutopilotPrError> {
        plan.verify()?;
        if pull_request_number == 0 {
            return Err(AutopilotPrError::InvalidPullRequestNumber);
        }
        let head_revision = head_revision.into();
        validate_revision(&head_revision)?;
        Ok(Self {
            plan_sha256: plan.plan_sha256.clone(),
            repository: plan.repository.clone(),
            pull_request_number,
            branch_name: plan.branch_name.clone(),
            head_revision: head_revision.to_ascii_lowercase(),
        })
    }

    pub fn pull_request_number(&self) -> u64 {
        self.pull_request_number
    }

    pub fn head_revision(&self) -> &str {
        &self.head_revision
    }

    pub fn verify_against_plan(
        &self,
        plan: &AutopilotPullRequestPlan,
    ) -> Result<(), AutopilotPrError> {
        plan.verify()?;
        if self.plan_sha256 != plan.plan_sha256
            || self.repository != plan.repository
            || self.branch_name != plan.branch_name
        {
            return Err(AutopilotPrError::EmissionReceiptMismatch);
        }
        validate_revision(&self.head_revision)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalVerdictKind {
    Ci,
    HumanReview,
}

impl ExternalVerdictKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ci => "ci",
            Self::HumanReview => "human-review",
        }
    }
}

/// CI/review evidence tied to one exact pull-request head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalPrVerdict {
    kind: ExternalVerdictKind,
    plan_sha256: String,
    repository: String,
    pull_request_number: u64,
    head_revision: String,
    accepted: bool,
    reason: String,
    evidence_sha256: String,
}

impl ExternalPrVerdict {
    pub fn new(
        receipt: &PullRequestEmissionReceipt,
        kind: ExternalVerdictKind,
        accepted: bool,
        reason: impl Into<String>,
        evidence_sha256: impl Into<String>,
    ) -> Result<Self, AutopilotPrError> {
        let reason = reason.into();
        let evidence_sha256 = evidence_sha256.into();
        validate_text("external_verdict.reason", &reason)?;
        validate_digest("external_verdict.evidence_sha256", &evidence_sha256)?;
        Ok(Self {
            kind,
            plan_sha256: receipt.plan_sha256.clone(),
            repository: receipt.repository.clone(),
            pull_request_number: receipt.pull_request_number,
            head_revision: receipt.head_revision.clone(),
            accepted,
            reason,
            evidence_sha256: evidence_sha256.to_ascii_lowercase(),
        })
    }

    pub fn kind(&self) -> ExternalVerdictKind {
        self.kind
    }

    pub fn accepted(&self) -> bool {
        self.accepted
    }

    pub fn evidence_sha256(&self) -> &str {
        &self.evidence_sha256
    }
}

/// Append an external CI/review verdict to the same P7 engineering trajectory.
/// Rejections are retained as negative flywheel evidence. This function never
/// returns a merge decision.
pub fn append_external_pr_verdict(
    plan: &AutopilotPullRequestPlan,
    receipt: &PullRequestEmissionReceipt,
    verdict: &ExternalPrVerdict,
    candidate: &mut TaskBoundEngineeringTrajectory,
) -> Result<(), AutopilotPrError> {
    plan.verify()?;
    receipt.verify_against_plan(plan)?;
    let actual_candidate = bound_candidate_identity(
        &candidate.trajectory,
        &candidate.task_id,
        &candidate.repository_role,
        candidate.perf_promotion.as_ref(),
    )?;
    if actual_candidate != plan.candidate_sha256 || actual_candidate != candidate.candidate_sha256 {
        return Err(AutopilotPrError::TrajectoryCandidateMismatch {
            expected: plan.candidate_sha256.clone(),
            actual: actual_candidate,
        });
    }
    if verdict.plan_sha256 != plan.plan_sha256
        || verdict.repository != receipt.repository
        || verdict.pull_request_number != receipt.pull_request_number
        || verdict.head_revision != receipt.head_revision
    {
        return Err(AutopilotPrError::ExternalVerdictContextMismatch);
    }

    let source = format!(
        "github:{}:{}#{}@{}:{}",
        verdict.kind.as_str(),
        verdict.repository,
        verdict.pull_request_number,
        verdict.head_revision,
        verdict.evidence_sha256
    );
    if candidate
        .trajectory
        .later_verdicts
        .iter()
        .any(|existing| existing.source == source)
    {
        return Err(AutopilotPrError::DuplicateExternalVerdict(source));
    }
    let later = LaterVerdict::new(source, verdict.accepted, verdict.reason.clone())
        .map_err(|error| AutopilotPrError::InvalidTrajectory(error.to_string()))?;
    candidate.trajectory.later_verdicts.push(later);
    candidate
        .trajectory
        .validate()
        .map_err(|error| AutopilotPrError::InvalidTrajectory(error.to_string()))?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutopilotPrError {
    EmptyField(&'static str),
    InvalidText {
        field: &'static str,
        value: String,
    },
    InvalidIdentifier {
        field: &'static str,
        value: String,
    },
    InvalidBranch {
        field: &'static str,
        value: String,
    },
    InvalidRevision(String),
    InvalidDigest {
        field: &'static str,
        value: String,
    },
    TitleTooLong(usize),
    InvalidSpec(String),
    InvalidDag(String),
    SpecDagMismatch,
    InvalidTrajectory(String),
    TrajectorySpecMismatch {
        expected: String,
        actual: String,
    },
    TrajectoryNotAccepted,
    UnknownTask(String),
    RepositoryRoleOutsideTask(String),
    CompatibilityScopeMismatch {
        repository: String,
        role: String,
        revision: String,
    },
    MissingRepositoryRole(String),
    AmbiguousRepositoryRole(String),
    InvalidPatchSet(String),
    PatchOutsideTaskAllowance {
        repository_role: String,
        path: String,
        operation: TaskOperation,
    },
    RepositoryRoleReadOnly(String),
    InvalidPerfEvidence(String),
    PerfCandidateNotPromotable,
    MissingPerfPromotion {
        task_id: String,
        profile_id: String,
    },
    UnexpectedPerfPromotion(String),
    PerfPromotionTaskMismatch {
        expected: String,
        actual: String,
    },
    PerfPromotionProfileMismatch {
        expected: String,
        actual: String,
    },
    PerfPromotionTrajectoryMismatch,
    BoundCandidateContextMismatch,
    BoundCandidateHashMismatch {
        expected: String,
        actual: String,
    },
    DefaultBranchMutationForbidden(String),
    PlanHashMismatch {
        expected: String,
        actual: String,
    },
    InvalidPullRequestNumber,
    EmissionReceiptMismatch,
    ExternalVerdictContextMismatch,
    TrajectoryCandidateMismatch {
        expected: String,
        actual: String,
    },
    DuplicateExternalVerdict(String),
}

impl fmt::Display for AutopilotPrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AUTOPILOT PR contract error: {self:?}")
    }
}

impl std::error::Error for AutopilotPrError {}

struct PrBodyContext<'a> {
    spec: &'a FrozenAutopilotSpec,
    dag: &'a AutopilotTaskDag,
    task: &'a AutopilotTask,
    candidate: &'a TaskBoundEngineeringTrajectory,
    target: &'a RepositoryRevision,
    compatibility_sha256: &'a str,
    patch_set_sha256: &'a str,
}

fn validate_perf_binding(
    task: &AutopilotTask,
    trajectory: &EngineeringTrajectory,
    promotion: Option<&VerifiedPerfPromotion>,
) -> Result<(), AutopilotPrError> {
    match task.regime() {
        TaskRegime::Feature => {
            if promotion.is_some() {
                return Err(AutopilotPrError::UnexpectedPerfPromotion(
                    task.id().to_string(),
                ));
            }
        }
        TaskRegime::Perf {
            benchmark_profile_id,
        } => {
            let promotion = promotion.ok_or_else(|| AutopilotPrError::MissingPerfPromotion {
                task_id: task.id().to_string(),
                profile_id: benchmark_profile_id.clone(),
            })?;
            if promotion.task_id != task.id() {
                return Err(AutopilotPrError::PerfPromotionTaskMismatch {
                    expected: task.id().to_string(),
                    actual: promotion.task_id.clone(),
                });
            }
            if promotion.profile_id != *benchmark_profile_id {
                return Err(AutopilotPrError::PerfPromotionProfileMismatch {
                    expected: benchmark_profile_id.clone(),
                    actual: promotion.profile_id.clone(),
                });
            }
            let trajectory_sha256 = immutable_trajectory_identity(trajectory)?;
            if promotion.trajectory_sha256 != trajectory_sha256 {
                return Err(AutopilotPrError::PerfPromotionTrajectoryMismatch);
            }
            if !promotion.report.hard_gates_passed || !promotion.report.promotable {
                return Err(AutopilotPrError::PerfCandidateNotPromotable);
            }
        }
    }
    Ok(())
}

fn unique_revision_for_role(
    compatibility: &CompatibilitySet,
    role: &str,
) -> Result<RepositoryRevision, AutopilotPrError> {
    let matches: Vec<_> = compatibility
        .revisions()
        .iter()
        .filter(|revision| revision.role == role)
        .cloned()
        .collect();
    match matches.as_slice() {
        [only] => Ok(only.clone()),
        [] => Err(AutopilotPrError::MissingRepositoryRole(role.to_string())),
        _ => Err(AutopilotPrError::AmbiguousRepositoryRole(role.to_string())),
    }
}

fn validate_compatibility_against_spec(
    spec: &FrozenAutopilotSpec,
    compatibility: &CompatibilitySet,
) -> Result<(), AutopilotPrError> {
    for scope in spec.repository_scope() {
        let found = compatibility.revisions().iter().any(|revision| {
            revision.repository == scope.repository
                && revision.role == scope.role
                && revision.revision == scope.revision
        });
        if !found {
            return Err(AutopilotPrError::CompatibilityScopeMismatch {
                repository: scope.repository.clone(),
                role: scope.role.clone(),
                revision: scope.revision.clone(),
            });
        }
    }
    Ok(())
}

fn validate_patchset_for_target(
    task: &AutopilotTask,
    repository_role: &str,
    patch_set: &PatchSet,
) -> Result<(), AutopilotPrError> {
    patch_set
        .validate()
        .map_err(|error| AutopilotPrError::InvalidPatchSet(error.to_string()))?;
    let allowances: Vec<_> = task
        .edit_allowances()
        .iter()
        .filter(|allowance| allowance.repository_role() == repository_role)
        .collect();
    if allowances.is_empty() {
        return Err(AutopilotPrError::RepositoryRoleReadOnly(
            repository_role.to_string(),
        ));
    }
    for operation in patch_set.operations() {
        let required = patch_operation_kind(operation);
        let allowed = allowances.iter().any(|allowance| {
            allowance.operations().contains(&required)
                && allowance
                    .allowed_path_prefixes()
                    .iter()
                    .any(|prefix| path_is_within(operation.path(), prefix))
        });
        if !allowed {
            return Err(AutopilotPrError::PatchOutsideTaskAllowance {
                repository_role: repository_role.to_string(),
                path: operation.path().to_string(),
                operation: required,
            });
        }
    }
    Ok(())
}

fn patch_operation_kind(operation: &FileOperation) -> TaskOperation {
    match operation {
        FileOperation::Create { .. } => TaskOperation::Create,
        FileOperation::ModifyExact { .. } => TaskOperation::ModifyExact,
        FileOperation::Delete { .. } => TaskOperation::Delete,
    }
}

fn immutable_trajectory_identity(
    trajectory: &EngineeringTrajectory,
) -> Result<String, AutopilotPrError> {
    let mut frozen = trajectory.clone();
    frozen.later_verdicts.clear();
    frozen
        .validate()
        .map_err(|error| AutopilotPrError::InvalidTrajectory(error.to_string()))?;
    let json = frozen
        .to_json_string()
        .map_err(|error| AutopilotPrError::InvalidTrajectory(error.to_string()))?;
    Ok(hex_digest(&sha256(json.as_bytes())))
}

fn bound_candidate_identity(
    trajectory: &EngineeringTrajectory,
    task_id: &str,
    repository_role: &str,
    promotion: Option<&VerifiedPerfPromotion>,
) -> Result<String, AutopilotPrError> {
    let trajectory_sha256 = immutable_trajectory_identity(trajectory)?;
    let perf = promotion
        .map(|value| value.evidence_sha256.as_str())
        .unwrap_or("feature");
    let material = format!(
        "rsi-autopilot-bound-candidate-v1|{trajectory_sha256}|{task_id}|{repository_role}|{perf}"
    );
    Ok(hex_digest(&sha256(material.as_bytes())))
}

fn perf_evidence_identity(
    profile: &FrozenPerfBenchmark,
    baseline: &[PerfMeasurementBatch],
    candidate: &[PerfMeasurementBatch],
    report: &PerfComparisonReport,
) -> String {
    let mut material = String::new();
    material.push_str("rsi-autopilot-perf-evidence-v1|");
    material.push_str(profile.profile_sha256());
    append_measurements(&mut material, "baseline", baseline);
    append_measurements(&mut material, "candidate", candidate);
    for result in &report.cases {
        material.push('|');
        material.push_str(&result.case_id);
        material.push(':');
        material.push_str(if result.passed { "pass" } else { "fail" });
        material.push(':');
        material.push_str(&result.improvement_ppm.to_string());
        material.push(':');
        material.push_str(&result.max_relative_mad_ppm.to_string());
        material.push(':');
        material.push_str(&result.winning_batches.to_string());
    }
    hex_digest(&sha256(material.as_bytes()))
}

fn append_measurements(
    material: &mut String,
    label: &str,
    measurements: &[PerfMeasurementBatch],
) {
    let mut ordered: Vec<_> = measurements.iter().collect();
    ordered.sort_by(|left, right| {
        left.case_id
            .cmp(&right.case_id)
            .then_with(|| left.batch_id.cmp(&right.batch_id))
    });
    for batch in ordered {
        material.push('|');
        material.push_str(label);
        material.push(':');
        material.push_str(&batch.case_id);
        material.push(':');
        material.push_str(&batch.batch_id);
        material.push(':');
        material.push_str(&batch.environment_fingerprint);
        for sample in &batch.samples {
            material.push(':');
            material.push_str(&format!("{:016x}", sample.to_bits()));
        }
    }
}

fn render_pr_body(context: PrBodyContext<'_>) -> String {
    let PrBodyContext {
        spec,
        dag,
        task,
        candidate,
        target,
        compatibility_sha256,
        patch_set_sha256,
    } = context;
    let trajectory = candidate.trajectory();
    let mut body = String::new();
    body.push_str("## AUTOPILOT engineering candidate\n\n");
    body.push_str(&format!("- task: `{}`\n", task.id()));
    body.push_str(&format!("- frozen spec: `{}`\n", spec.spec_sha256()));
    body.push_str(&format!("- task DAG: `{}`\n", dag.dag_sha256()));
    body.push_str(&format!("- repository role: `{}`\n", target.role));
    body.push_str(&format!("- exact base revision: `{}`\n", target.revision));
    body.push_str(&format!("- parent state: `{}`\n", trajectory.parent_state_id));
    body.push_str(&format!("- PatchSet: `{patch_set_sha256}`\n"));
    body.push_str(&format!(
        "- candidate identity: `{}`\n",
        candidate.candidate_sha256()
    ));
    body.push_str(&format!(
        "- compatibility fingerprint: `{compatibility_sha256}`\n\n"
    ));

    body.push_str("## Compatibility set\n\n```json\n");
    body.push_str(&trajectory.compatibility.to_json_string());
    body.push_str("\n```\n\n## Hard-gate evidence\n\n");
    for (name, status) in admissibility_rows(trajectory) {
        body.push_str(&format!("- {name}: `{}`\n", gate_status(status)));
    }

    body.push_str("\n## Compiler / test / device evidence\n\n");
    for evidence in &trajectory.compiler_test_device_evidence {
        body.push_str("- ");
        body.push_str(evidence);
        body.push('\n');
    }

    body.push_str("\n## Benchmark evidence\n\n");
    if let Some(promotion) = candidate.perf_promotion() {
        body.push_str(&format!(
            "- frozen PERF profile: `{}`\n- environment: `{}`\n- evidence: `{}`\n",
            promotion.profile_sha256,
            promotion.environment_fingerprint,
            promotion.evidence_sha256
        ));
        for result in &promotion.report.cases {
            append_perf_case(&mut body, result);
        }
    } else {
        body.push_str("- no PERF ranking required by this accepted FEATURE trajectory\n");
    }

    body.push_str("\n## Review and merge policy\n\n");
    body.push_str(
        "This plan requests branch creation and pull-request creation only. It does not request or encode automatic merge. CI and human-review verdicts are external evidence appended to the engineering trajectory.\n",
    );
    body
}

fn append_perf_case(body: &mut String, result: &PerfCaseResult) {
    body.push_str(&format!(
        "- `{}`: improvement={}ppm, max_relative_mad={}ppm, winning_batches={}/{}, passed={}\n",
        result.case_id,
        result.improvement_ppm,
        result.max_relative_mad_ppm,
        result.winning_batches,
        result.required_winning_batches,
        result.passed
    ));
}

fn admissibility_rows(
    trajectory: &EngineeringTrajectory,
) -> [(&'static str, GateStatus); 7] {
    [
        ("build", trajectory.admissibility.build),
        ("required_tests", trajectory.admissibility.required_tests),
        ("numerical_parity", trajectory.admissibility.numerical_parity),
        ("provenance", trajectory.admissibility.provenance),
        (
            "deterministic_contract",
            trajectory.admissibility.deterministic_contract,
        ),
        ("resource_budget", trajectory.admissibility.resource_budget),
        ("policy_checks", trajectory.admissibility.policy_checks),
    ]
}

fn gate_status(status: GateStatus) -> &'static str {
    match status {
        GateStatus::Pass => "pass",
        GateStatus::Fail => "fail",
        GateStatus::Unknown => "unknown",
    }
}

fn path_is_within(path: &str, prefix: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn validate_text(field: &'static str, value: &str) -> Result<(), AutopilotPrError> {
    if value.is_empty() {
        return Err(AutopilotPrError::EmptyField(field));
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(AutopilotPrError::InvalidText {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), AutopilotPrError> {
    validate_text(field, value)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(AutopilotPrError::InvalidIdentifier {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

/// Implements the core `git check-ref-format --branch` constraints relevant to
/// generated/target branch names, including component-level `.lock`/dot rules.
fn validate_git_branch_name(
    field: &'static str,
    value: &str,
) -> Result<(), AutopilotPrError> {
    validate_text(field, value)?;
    let forbidden = value == "@"
        || value.starts_with('/')
        || value.ends_with('/')
        || value.ends_with('.')
        || value.contains("..")
        || value.contains("@{")
        || value.contains("//")
        || value.bytes().any(|byte| byte <= b' ' || byte == 0x7f)
        || value
            .chars()
            .any(|ch| matches!(ch, '~' | '^' | ':' | '?' | '*' | '[' | '\\'))
        || value.split('/').any(|component| {
            component.is_empty()
                || component.starts_with('.')
                || component.ends_with('.')
                || component.ends_with(".lock")
        });
    if forbidden {
        return Err(AutopilotPrError::InvalidBranch {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_revision(revision: &str) -> Result<(), AutopilotPrError> {
    if !matches!(revision.len(), 40 | 64)
        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(AutopilotPrError::InvalidRevision(revision.to_string()));
    }
    Ok(())
}

fn validate_digest(field: &'static str, value: &str) -> Result<(), AutopilotPrError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(AutopilotPrError::InvalidDigest {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn hex_digest(digest: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autopilot_intake::{
        AcceptanceCheck, AcceptanceCriterion, AutopilotSpecDraft, ExplorationObservation,
        ExplorationSource, ExploredObjective, RepositoryExploration, RepositoryScope, SpecBudget,
    };
    use crate::autopilot_perf::{
        AntiNoisePolicy, BenchmarkCase, BenchmarkCaseSpec, BenchmarkClass, BenchmarkEnvironment,
        FrozenBenchmarkArtifact, MetricDirection, PerfBenchmarkApproval, PerfBenchmarkDraft,
    };
    use crate::autopilot_task_dag::{
        AutopilotTask, AutopilotTaskDraft, HardGateProfile, TaskBudget, TaskDagPolicy,
        TaskEditAllowance,
    };
    use crate::engineering_trajectory::{AdmissibilityBreakdown, ProposerMetadata};

    fn digest(hex: char) -> String {
        hex.to_string().repeat(64)
    }

    fn revision(hex: char) -> String {
        hex.to_string().repeat(40)
    }

    fn spec() -> FrozenAutopilotSpec {
        let exploration = RepositoryExploration::new(
            "Memorithm/RSI",
            revision('a'),
            vec![
                ExplorationObservation::new(
                    "code",
                    ExplorationSource::repository_file("src/lib.rs", digest('b')).unwrap(),
                    "inspected the implementation and CI boundary",
                )
                .unwrap(),
            ],
        )
        .unwrap();
        ExploredObjective::new(
            "emit an auditable engineering pull request",
            "candidate already passed the frozen inner-loop contract",
            vec![exploration],
        )
        .unwrap()
        .questionnaire(Vec::new())
        .unwrap()
        .resolve(Vec::new())
        .unwrap()
        .freeze(AutopilotSpecDraft::new(
            vec![
                AcceptanceCriterion::new(
                    "tests-pass",
                    "all required tests pass",
                    AcceptanceCheck::command("rsi", "cargo_test", Vec::new()).unwrap(),
                )
                .unwrap(),
            ],
            vec!["automatic merge".to_string()],
            SpecBudget::new(20, 20_000, 20_000).unwrap(),
            vec![
                RepositoryScope::new(
                    "rsi",
                    "Memorithm/RSI",
                    revision('a'),
                    vec!["benches".to_string(), "src".to_string()],
                )
                .unwrap(),
            ],
        ))
        .unwrap()
    }

    fn task(id: &str, regime: TaskRegime, path: &str) -> AutopilotTask {
        AutopilotTask::new(AutopilotTaskDraft {
            id: id.to_string(),
            description: format!("execute {id}"),
            regime,
            repository_roles: vec!["rsi".to_string()],
            edit_allowances: vec![
                TaskEditAllowance::new(
                    "rsi",
                    vec![path.to_string()],
                    vec![TaskOperation::Create, TaskOperation::ModifyExact],
                )
                .unwrap(),
            ],
            hard_gate_profile: HardGateProfile::engineering_strict(),
            budget: TaskBudget::new(4, 8, 8_000, 8_000).unwrap(),
            dependencies: Vec::new(),
            done_criterion_id: "tests-pass".to_string(),
        })
        .unwrap()
    }

    fn dag(spec: &FrozenAutopilotSpec, tasks: Vec<AutopilotTask>) -> AutopilotTaskDag {
        AutopilotTaskDag::new(spec, tasks, TaskDagPolicy::new(8, 8, 16).unwrap()).unwrap()
    }

    fn gates() -> AdmissibilityBreakdown {
        AdmissibilityBreakdown {
            build: GateStatus::Pass,
            required_tests: GateStatus::Pass,
            numerical_parity: GateStatus::Pass,
            provenance: GateStatus::Pass,
            deterministic_contract: GateStatus::Pass,
            resource_budget: GateStatus::Pass,
            policy_checks: GateStatus::Pass,
        }
    }

    fn trajectory(spec: &FrozenAutopilotSpec) -> EngineeringTrajectory {
        EngineeringTrajectory {
            task_spec_id: spec.spec_sha256().to_string(),
            compatibility: CompatibilitySet::new(
                vec![
                    RepositoryRevision::new("Memorithm/RSI", revision('a'), "rsi").unwrap(),
                ],
                "stable",
                vec!["default".to_string()],
            )
            .unwrap(),
            parent_state_id: digest('c'),
            patch_set: PatchSet::new(vec![FileOperation::modify_exact(
                "src/lib.rs",
                "old",
                "new",
            )])
            .unwrap(),
            proposer: ProposerMetadata::new("sciagent", "engineering", "heldout-v1").unwrap(),
            compiler_test_device_evidence: vec!["cargo test: pass".to_string()],
            admissibility: gates(),
            benchmarks: Vec::new(),
            verdict: EngineeringVerdict::Accepted,
            verdict_reason: "frozen acceptance criteria satisfied".to_string(),
            later_verdicts: Vec::new(),
        }
    }

    fn bind_feature(
        spec: &FrozenAutopilotSpec,
        dag: &AutopilotTaskDag,
        task_id: &str,
        trajectory: EngineeringTrajectory,
    ) -> TaskBoundEngineeringTrajectory {
        TaskBoundEngineeringTrajectory::new(spec, dag, task_id, "rsi", trajectory, None).unwrap()
    }

    fn plan(
        spec: &FrozenAutopilotSpec,
        dag: &AutopilotTaskDag,
        candidate: &TaskBoundEngineeringTrajectory,
    ) -> AutopilotPullRequestPlan {
        AutopilotPullRequestPlan::new(
            spec,
            dag,
            candidate,
            PullRequestPlanDraft::new("main", "feat(autopilot): apply accepted candidate"),
        )
        .unwrap()
    }

    fn perf_case() -> BenchmarkCase {
        BenchmarkCase::new(BenchmarkCaseSpec {
            id: "e2e-latency".to_string(),
            repository_role: "rsi".to_string(),
            command_kind: "bench_e2e".to_string(),
            arguments: Vec::new(),
            metric: "latency".to_string(),
            unit: "ns".to_string(),
            direction: MetricDirection::Minimize,
            class: BenchmarkClass::EndToEnd,
            promotion_gate: true,
        })
        .unwrap()
    }

    fn perf_profile(
        spec: &FrozenAutopilotSpec,
        dag: &AutopilotTaskDag,
    ) -> FrozenPerfBenchmark {
        FrozenPerfBenchmark::freeze(
            spec,
            dag,
            "perf-task",
            PerfBenchmarkDraft {
                approval: PerfBenchmarkApproval::new("human-review", digest('d')).unwrap(),
                environment: BenchmarkEnvironment::new(digest('e'), digest('f')).unwrap(),
                policy: AntiNoisePolicy::new(5, 3, 20_000, 20_000, 3).unwrap(),
                cases: vec![perf_case()],
                artifacts: vec![
                    FrozenBenchmarkArtifact::new("rsi", "benches/perf.rs", digest('1')).unwrap(),
                ],
            },
        )
        .unwrap()
    }

    fn perf_batches(
        profile: &FrozenPerfBenchmark,
    ) -> (Vec<PerfMeasurementBatch>, Vec<PerfMeasurementBatch>) {
        let mut baseline = Vec::new();
        let mut candidate = Vec::new();
        for id in ["a", "b", "c"] {
            baseline.push(
                PerfMeasurementBatch::new(
                    "e2e-latency",
                    id,
                    profile.environment_fingerprint(),
                    vec![100.0, 100.5, 99.5, 100.2, 99.8],
                )
                .unwrap(),
            );
            candidate.push(
                PerfMeasurementBatch::new(
                    "e2e-latency",
                    id,
                    profile.environment_fingerprint(),
                    vec![90.0, 90.4, 89.6, 90.2, 89.8],
                )
                .unwrap(),
            );
        }
        (baseline, candidate)
    }

    #[test]
    fn plan_requests_branch_then_pr_and_never_merge() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task("implementation", TaskRegime::feature(), "src")],
        );
        let candidate = bind_feature(&spec, &dag, "implementation", trajectory(&spec));
        let plan = plan(&spec, &dag, &candidate);
        assert_eq!(
            plan.hosting_actions(),
            [HostingAction::CreateBranch, HostingAction::OpenPullRequest]
        );
        assert_ne!(plan.branch_name(), plan.default_branch());
        assert!(plan.branch_name().starts_with("autopilot/implementation/"));
        assert!(plan.body().contains("does not request or encode automatic merge"));
    }

    #[test]
    fn trajectory_binding_cannot_be_switched_at_pr_emission() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![
                task("task-a", TaskRegime::feature(), "src"),
                task("task-b", TaskRegime::feature(), "src"),
            ],
        );
        let candidate = bind_feature(&spec, &dag, "task-a", trajectory(&spec));
        let plan = plan(&spec, &dag, &candidate);
        assert_eq!(candidate.task_id(), "task-a");
        assert!(plan.body().contains("- task: `task-a`"));
        assert!(!plan.body().contains("- task: `task-b`"));
    }

    #[test]
    fn perf_candidate_requires_verified_frozen_profile_promotion() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task(
                "perf-task",
                TaskRegime::perf("decode-v1").unwrap(),
                "src",
            )],
        );
        let candidate_trajectory = trajectory(&spec);
        assert!(matches!(
            TaskBoundEngineeringTrajectory::new(
                &spec,
                &dag,
                "perf-task",
                "rsi",
                candidate_trajectory.clone(),
                None,
            ),
            Err(AutopilotPrError::MissingPerfPromotion { .. })
        ));

        let profile = perf_profile(&spec, &dag);
        let (baseline, candidate) = perf_batches(&profile);
        let promotion = VerifiedPerfPromotion::evaluate(
            &profile,
            &candidate_trajectory,
            &baseline,
            &candidate,
        )
        .unwrap();
        let bound = TaskBoundEngineeringTrajectory::new(
            &spec,
            &dag,
            "perf-task",
            "rsi",
            candidate_trajectory,
            Some(promotion),
        )
        .unwrap();
        let plan = plan(&spec, &dag, &bound);
        assert!(plan.body().contains(profile.profile_sha256()));
        assert!(plan.body().contains("e2e-latency"));
    }

    #[test]
    fn rejected_perf_comparison_cannot_create_promotion_evidence() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task(
                "perf-task",
                TaskRegime::perf("decode-v1").unwrap(),
                "src",
            )],
        );
        let profile = perf_profile(&spec, &dag);
        let candidate_trajectory = trajectory(&spec);
        let (baseline, mut candidate) = perf_batches(&profile);
        for batch in &mut candidate {
            batch.samples = vec![110.0, 110.5, 109.5, 110.2, 109.8];
        }
        assert!(matches!(
            VerifiedPerfPromotion::evaluate(
                &profile,
                &candidate_trajectory,
                &baseline,
                &candidate,
            ),
            Err(AutopilotPrError::PerfCandidateNotPromotable)
        ));
    }

    #[test]
    fn invalid_generated_git_branch_is_rejected() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task("bad.lock", TaskRegime::feature(), "src")],
        );
        let candidate = bind_feature(&spec, &dag, "bad.lock", trajectory(&spec));
        assert!(matches!(
            AutopilotPullRequestPlan::new(
                &spec,
                &dag,
                &candidate,
                PullRequestPlanDraft::new("main", "candidate")
            ),
            Err(AutopilotPrError::InvalidBranch {
                field: "generated_branch",
                ..
            })
        ));
    }

    #[test]
    fn body_contains_exact_compatibility_and_evidence() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task("implementation", TaskRegime::feature(), "src")],
        );
        let candidate = bind_feature(&spec, &dag, "implementation", trajectory(&spec));
        let plan = plan(&spec, &dag, &candidate);
        assert!(plan
            .body()
            .contains(&candidate.trajectory().compatibility.fingerprint()));
        assert!(plan
            .body()
            .contains(&candidate.trajectory().compatibility.to_json_string()));
        assert!(plan.body().contains("numerical_parity: `pass`"));
        assert!(plan.body().contains("cargo test: pass"));
        plan.verify().unwrap();
    }

    #[test]
    fn inadmissible_candidate_cannot_be_bound() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task("implementation", TaskRegime::feature(), "src")],
        );
        let mut candidate = trajectory(&spec);
        candidate.admissibility.policy_checks = GateStatus::Fail;
        candidate.verdict = EngineeringVerdict::Rejected;
        assert!(matches!(
            TaskBoundEngineeringTrajectory::new(
                &spec,
                &dag,
                "implementation",
                "rsi",
                candidate,
                None,
            ),
            Err(AutopilotPrError::TrajectoryNotAccepted)
                | Err(AutopilotPrError::InvalidTrajectory(_))
        ));
    }

    #[test]
    fn patchset_must_stay_inside_task_allowance() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task("implementation", TaskRegime::feature(), "src")],
        );
        let mut candidate = trajectory(&spec);
        candidate.patch_set =
            PatchSet::new(vec![FileOperation::create("docs/out.md", "x")]).unwrap();
        assert!(matches!(
            TaskBoundEngineeringTrajectory::new(
                &spec,
                &dag,
                "implementation",
                "rsi",
                candidate,
                None,
            ),
            Err(AutopilotPrError::PatchOutsideTaskAllowance { .. })
        ));
    }

    #[test]
    fn ci_and_human_review_are_appended_to_same_trajectory() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task("implementation", TaskRegime::feature(), "src")],
        );
        let mut candidate = bind_feature(&spec, &dag, "implementation", trajectory(&spec));
        let plan = plan(&spec, &dag, &candidate);
        let receipt = PullRequestEmissionReceipt::new(&plan, 42, revision('d')).unwrap();
        let ci = ExternalPrVerdict::new(
            &receipt,
            ExternalVerdictKind::Ci,
            true,
            "required final-head CI passed",
            digest('e'),
        )
        .unwrap();
        append_external_pr_verdict(&plan, &receipt, &ci, &mut candidate).unwrap();
        let review = ExternalPrVerdict::new(
            &receipt,
            ExternalVerdictKind::HumanReview,
            false,
            "review requested a safer implementation",
            digest('f'),
        )
        .unwrap();
        append_external_pr_verdict(&plan, &receipt, &review, &mut candidate).unwrap();

        assert_eq!(candidate.trajectory().later_verdicts.len(), 2);
        assert!(candidate.trajectory().later_verdicts[0].accepted);
        assert!(!candidate.trajectory().later_verdicts[1].accepted);
        assert!(candidate.trajectory().later_verdicts[1]
            .source
            .contains("github:human-review:Memorithm/RSI#42@"));
    }

    #[test]
    fn duplicate_external_verdict_is_rejected() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task("implementation", TaskRegime::feature(), "src")],
        );
        let mut candidate = bind_feature(&spec, &dag, "implementation", trajectory(&spec));
        let plan = plan(&spec, &dag, &candidate);
        let receipt = PullRequestEmissionReceipt::new(&plan, 42, revision('d')).unwrap();
        let ci = ExternalPrVerdict::new(
            &receipt,
            ExternalVerdictKind::Ci,
            true,
            "CI passed",
            digest('e'),
        )
        .unwrap();
        append_external_pr_verdict(&plan, &receipt, &ci, &mut candidate).unwrap();
        assert!(matches!(
            append_external_pr_verdict(&plan, &receipt, &ci, &mut candidate),
            Err(AutopilotPrError::DuplicateExternalVerdict(_))
        ));
    }

    #[test]
    fn verdict_for_other_head_is_rejected() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task("implementation", TaskRegime::feature(), "src")],
        );
        let mut candidate = bind_feature(&spec, &dag, "implementation", trajectory(&spec));
        let plan = plan(&spec, &dag, &candidate);
        let receipt = PullRequestEmissionReceipt::new(&plan, 42, revision('d')).unwrap();
        let mut verdict = ExternalPrVerdict::new(
            &receipt,
            ExternalVerdictKind::HumanReview,
            true,
            "approved",
            digest('e'),
        )
        .unwrap();
        verdict.head_revision = revision('f');
        assert!(matches!(
            append_external_pr_verdict(&plan, &receipt, &verdict, &mut candidate),
            Err(AutopilotPrError::ExternalVerdictContextMismatch)
        ));
    }

    #[test]
    fn post_plan_candidate_mutation_is_rejected() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task("implementation", TaskRegime::feature(), "src")],
        );
        let mut candidate = bind_feature(&spec, &dag, "implementation", trajectory(&spec));
        let plan = plan(&spec, &dag, &candidate);
        let receipt = PullRequestEmissionReceipt::new(&plan, 42, revision('d')).unwrap();
        let ci = ExternalPrVerdict::new(
            &receipt,
            ExternalVerdictKind::Ci,
            true,
            "CI passed",
            digest('e'),
        )
        .unwrap();
        candidate.trajectory.compiler_test_device_evidence[0] = "mutated evidence".to_string();
        assert!(matches!(
            append_external_pr_verdict(&plan, &receipt, &ci, &mut candidate),
            Err(AutopilotPrError::TrajectoryCandidateMismatch { .. })
        ));
    }

    #[test]
    fn compatibility_must_match_frozen_repository_revision() {
        let spec = spec();
        let dag = dag(
            &spec,
            vec![task("implementation", TaskRegime::feature(), "src")],
        );
        let mut candidate = trajectory(&spec);
        candidate.compatibility = CompatibilitySet::new(
            vec![RepositoryRevision::new("Memorithm/RSI", revision('9'), "rsi").unwrap()],
            "stable",
            Vec::new(),
        )
        .unwrap();
        assert!(matches!(
            TaskBoundEngineeringTrajectory::new(
                &spec,
                &dag,
                "implementation",
                "rsi",
                candidate,
                None,
            ),
            Err(AutopilotPrError::CompatibilityScopeMismatch { .. })
        ));
    }
}
