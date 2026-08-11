//! P8.5 AUTOPILOT PR emission and human-review flywheel contract.
//!
//! The core remains network-free and cannot merge anything. It emits exactly a
//! branch-creation plan followed by a pull-request plan. A hosting adapter may
//! execute those two actions and return an immutable receipt. CI and human
//! review verdicts are then appended to the existing [`EngineeringTrajectory`]
//! as [`LaterVerdict`] records tied to the exact PR head.

use crate::autopilot_intake::FrozenAutopilotSpec;
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullRequestPlanDraft {
    pub task_id: String,
    pub repository_role: String,
    pub default_branch: String,
    pub title: String,
}

impl PullRequestPlanDraft {
    pub fn new(
        task_id: impl Into<String>,
        repository_role: impl Into<String>,
        default_branch: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            repository_role: repository_role.into(),
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
        trajectory: &EngineeringTrajectory,
        draft: PullRequestPlanDraft,
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

        validate_identifier("task_id", &draft.task_id)?;
        validate_identifier("repository_role", &draft.repository_role)?;
        validate_branch_component("default_branch", &draft.default_branch)?;
        validate_text("title", &draft.title)?;
        if draft.title.len() > 240 {
            return Err(AutopilotPrError::TitleTooLong(draft.title.len()));
        }

        let task = dag
            .task(&draft.task_id)
            .ok_or_else(|| AutopilotPrError::UnknownTask(draft.task_id.clone()))?;
        if !task
            .repository_roles()
            .iter()
            .any(|role| role == &draft.repository_role)
        {
            return Err(AutopilotPrError::RepositoryRoleOutsideTask(
                draft.repository_role,
            ));
        }
        validate_compatibility_against_spec(spec, &trajectory.compatibility)?;
        let target = unique_revision_for_role(&trajectory.compatibility, &draft.repository_role)?;
        validate_patchset_for_target(task, &draft.repository_role, &trajectory.patch_set)?;
        if matches!(task.regime(), TaskRegime::Perf { .. }) && trajectory.benchmarks.is_empty() {
            return Err(AutopilotPrError::PerfTrajectoryMissingBenchmark);
        }

        let patch_set_sha256 = trajectory
            .patch_set
            .identity()
            .map_err(|error| AutopilotPrError::InvalidPatchSet(error.to_string()))?;
        let compatibility_sha256 = trajectory.compatibility.fingerprint();
        let candidate_sha256 = trajectory_candidate_identity(trajectory)?;
        let branch_name = format!(
            "autopilot/{}/{}",
            draft.task_id,
            &patch_set_sha256[..12]
        );
        if branch_name == draft.default_branch {
            return Err(AutopilotPrError::DefaultBranchMutationForbidden(
                draft.default_branch,
            ));
        }
        let body = render_pr_body(
            spec,
            dag,
            task,
            trajectory,
            &target,
            &compatibility_sha256,
            &patch_set_sha256,
            &candidate_sha256,
        )?;

        let mut plan = Self {
            schema_version: AUTOPILOT_PR_PLAN_SCHEMA_VERSION,
            spec_sha256: spec.spec_sha256().to_string(),
            dag_sha256: dag.dag_sha256().to_string(),
            task_id: task.id().to_string(),
            repository_role: draft.repository_role,
            repository: target.repository,
            base_revision: target.revision,
            default_branch: draft.default_branch,
            branch_name,
            title: draft.title,
            body,
            compatibility_sha256,
            patch_set_sha256,
            candidate_sha256,
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

/// Immutable result returned by the trusted hosting adapter after it has created
/// the branch and opened the PR described by a plan.
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

/// Authenticated evidence is collected by the hosting adapter and represented in
/// the core as an immutable hash plus the exact PR/head it describes.
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

/// Append one external CI/review verdict to the original engineering trajectory.
/// This never returns a merge decision. Rejections are retained as negative
/// flywheel evidence exactly like approvals.
pub fn append_external_pr_verdict(
    plan: &AutopilotPullRequestPlan,
    receipt: &PullRequestEmissionReceipt,
    verdict: &ExternalPrVerdict,
    trajectory: &mut EngineeringTrajectory,
) -> Result<(), AutopilotPrError> {
    plan.verify()?;
    receipt.verify_against_plan(plan)?;
    trajectory
        .validate()
        .map_err(|error| AutopilotPrError::InvalidTrajectory(error.to_string()))?;
    let actual_candidate = trajectory_candidate_identity(trajectory)?;
    if actual_candidate != plan.candidate_sha256 {
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
    if trajectory
        .later_verdicts
        .iter()
        .any(|existing| existing.source == source)
    {
        return Err(AutopilotPrError::DuplicateExternalVerdict(source));
    }
    let later = LaterVerdict::new(source, verdict.accepted, verdict.reason.clone())
        .map_err(|error| AutopilotPrError::InvalidTrajectory(error.to_string()))?;
    trajectory.later_verdicts.push(later);
    trajectory
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
    PerfTrajectoryMissingBenchmark,
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

fn trajectory_candidate_identity(
    trajectory: &EngineeringTrajectory,
) -> Result<String, AutopilotPrError> {
    let patch_set_sha256 = trajectory
        .patch_set
        .identity()
        .map_err(|error| AutopilotPrError::InvalidPatchSet(error.to_string()))?;
    let material = format!(
        "rsi-autopilot-candidate-v1|{}|{}|{}|{}|{}|{}|{}",
        trajectory.task_spec_id,
        trajectory.compatibility.fingerprint(),
        trajectory.parent_state_id,
        patch_set_sha256,
        trajectory.proposer.provider,
        trajectory.proposer.model,
        trajectory.proposer.configuration_id,
    );
    Ok(hex_digest(&sha256(material.as_bytes())))
}

fn render_pr_body(
    spec: &FrozenAutopilotSpec,
    dag: &AutopilotTaskDag,
    task: &AutopilotTask,
    trajectory: &EngineeringTrajectory,
    target: &RepositoryRevision,
    compatibility_sha256: &str,
    patch_set_sha256: &str,
    candidate_sha256: &str,
) -> Result<String, AutopilotPrError> {
    let compatibility_json = trajectory.compatibility.to_json_string();
    let mut body = String::new();
    body.push_str("## AUTOPILOT engineering candidate\n\n");
    body.push_str(&format!("- task: `{}`\n", task.id()));
    body.push_str(&format!("- frozen spec: `{}`\n", spec.spec_sha256()));
    body.push_str(&format!("- task DAG: `{}`\n", dag.dag_sha256()));
    body.push_str(&format!("- repository role: `{}`\n", target.role));
    body.push_str(&format!("- exact base revision: `{}`\n", target.revision));
    body.push_str(&format!("- parent state: `{}`\n", trajectory.parent_state_id));
    body.push_str(&format!("- PatchSet: `{patch_set_sha256}`\n"));
    body.push_str(&format!("- candidate identity: `{candidate_sha256}`\n"));
    body.push_str(&format!(
        "- compatibility fingerprint: `{compatibility_sha256}`\n\n"
    ));

    body.push_str("## Compatibility set\n\n```json\n");
    body.push_str(&compatibility_json);
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
    if trajectory.benchmarks.is_empty() {
        body.push_str("- no benchmark required by this accepted FEATURE trajectory\n");
    } else {
        for benchmark in &trajectory.benchmarks {
            body.push_str(&format!(
                "- `{}`: {} {} ({} samples)\n",
                benchmark.metric,
                benchmark.summary,
                benchmark.unit,
                benchmark.samples.len()
            ));
        }
    }
    body.push_str("\n## Review and merge policy\n\n");
    body.push_str(
        "This plan requests branch creation and pull-request creation only. It does not request or encode automatic merge. CI and human-review verdicts are external evidence appended to the engineering trajectory.\n",
    );
    Ok(body)
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

fn validate_branch_component(
    field: &'static str,
    value: &str,
) -> Result<(), AutopilotPrError> {
    validate_text(field, value)?;
    let invalid = value.starts_with('/')
        || value.ends_with('/')
        || value.contains("..")
        || value.contains("//")
        || value.contains(' ')
        || value.contains('~')
        || value.contains('^')
        || value.contains(':')
        || value.contains('?')
        || value.contains('*')
        || value.contains('[')
        || value.contains('\\');
    if invalid {
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
    use crate::autopilot_task_dag::{
        AutopilotTask, AutopilotTaskDraft, HardGateProfile, TaskBudget, TaskDagPolicy,
        TaskEditAllowance,
    };
    use crate::compatibility::RepositoryRevision;
    use crate::engineering_trajectory::{
        AdmissibilityBreakdown, BenchmarkRecord, ProposerMetadata,
    };

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
            SpecBudget::new(10, 10_000, 10_000).unwrap(),
            vec![
                RepositoryScope::new(
                    "rsi",
                    "Memorithm/RSI",
                    revision('a'),
                    vec!["src".to_string()],
                )
                .unwrap(),
            ],
        ))
        .unwrap()
    }

    fn task(regime: TaskRegime) -> AutopilotTask {
        AutopilotTask::new(AutopilotTaskDraft {
            id: "implementation".to_string(),
            description: "implement the accepted candidate".to_string(),
            regime,
            repository_roles: vec!["rsi".to_string()],
            edit_allowances: vec![
                TaskEditAllowance::new(
                    "rsi",
                    vec!["src".to_string()],
                    vec![TaskOperation::Create, TaskOperation::ModifyExact],
                )
                .unwrap(),
            ],
            hard_gate_profile: HardGateProfile::engineering_strict(),
            budget: TaskBudget::new(2, 4, 4_000, 4_000).unwrap(),
            dependencies: Vec::new(),
            done_criterion_id: "tests-pass".to_string(),
        })
        .unwrap()
    }

    fn dag(spec: &FrozenAutopilotSpec, regime: TaskRegime) -> AutopilotTaskDag {
        AutopilotTaskDag::new(
            spec,
            vec![task(regime)],
            TaskDagPolicy::new(4, 4, 4).unwrap(),
        )
        .unwrap()
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

    fn trajectory(spec: &FrozenAutopilotSpec, with_benchmark: bool) -> EngineeringTrajectory {
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
            benchmarks: if with_benchmark {
                vec![BenchmarkRecord::new("latency", "ns", vec![10.0, 9.0], 9.5).unwrap()]
            } else {
                Vec::new()
            },
            verdict: EngineeringVerdict::Accepted,
            verdict_reason: "frozen acceptance criteria satisfied".to_string(),
            later_verdicts: Vec::new(),
        }
    }

    fn plan(
        spec: &FrozenAutopilotSpec,
        dag: &AutopilotTaskDag,
        trajectory: &EngineeringTrajectory,
    ) -> AutopilotPullRequestPlan {
        AutopilotPullRequestPlan::new(
            spec,
            dag,
            trajectory,
            PullRequestPlanDraft::new(
                "implementation",
                "rsi",
                "main",
                "feat(autopilot): apply accepted candidate",
            ),
        )
        .unwrap()
    }

    #[test]
    fn plan_requests_branch_then_pr_and_never_merge() {
        let spec = spec();
        let dag = dag(&spec, TaskRegime::feature());
        let trajectory = trajectory(&spec, false);
        let plan = plan(&spec, &dag, &trajectory);
        assert_eq!(
            plan.hosting_actions(),
            [HostingAction::CreateBranch, HostingAction::OpenPullRequest]
        );
        assert_ne!(plan.branch_name(), plan.default_branch());
        assert!(plan.branch_name().starts_with("autopilot/implementation/"));
        assert!(plan.body().contains("does not request or encode automatic merge"));
    }

    #[test]
    fn body_contains_exact_compatibility_and_evidence() {
        let spec = spec();
        let dag = dag(&spec, TaskRegime::feature());
        let trajectory = trajectory(&spec, false);
        let plan = plan(&spec, &dag, &trajectory);
        assert!(plan.body().contains(&trajectory.compatibility.fingerprint()));
        assert!(plan.body().contains(&trajectory.compatibility.to_json_string()));
        assert!(plan.body().contains("numerical_parity: `pass`"));
        assert!(plan.body().contains("cargo test: pass"));
        plan.verify().unwrap();
    }

    #[test]
    fn inadmissible_or_rejected_candidate_cannot_emit_pr() {
        let spec = spec();
        let dag = dag(&spec, TaskRegime::feature());
        let mut trajectory = trajectory(&spec, false);
        trajectory.admissibility.policy_checks = GateStatus::Fail;
        trajectory.verdict = EngineeringVerdict::Rejected;
        assert!(matches!(
            AutopilotPullRequestPlan::new(
                &spec,
                &dag,
                &trajectory,
                PullRequestPlanDraft::new("implementation", "rsi", "main", "candidate")
            ),
            Err(AutopilotPrError::TrajectoryNotAccepted)
                | Err(AutopilotPrError::InvalidTrajectory(_))
        ));
    }

    #[test]
    fn perf_candidate_requires_recorded_benchmark() {
        let spec = spec();
        let dag = dag(&spec, TaskRegime::perf("perf-v1").unwrap());
        let trajectory = trajectory(&spec, false);
        assert!(matches!(
            AutopilotPullRequestPlan::new(
                &spec,
                &dag,
                &trajectory,
                PullRequestPlanDraft::new("implementation", "rsi", "main", "candidate")
            ),
            Err(AutopilotPrError::PerfTrajectoryMissingBenchmark)
        ));
        let measured = trajectory(&spec, true);
        plan(&spec, &dag, &measured).verify().unwrap();
    }

    #[test]
    fn patchset_must_stay_inside_task_allowance() {
        let spec = spec();
        let dag = dag(&spec, TaskRegime::feature());
        let mut trajectory = trajectory(&spec, false);
        trajectory.patch_set = PatchSet::new(vec![FileOperation::create("docs/out.md", "x")]).unwrap();
        assert!(matches!(
            AutopilotPullRequestPlan::new(
                &spec,
                &dag,
                &trajectory,
                PullRequestPlanDraft::new("implementation", "rsi", "main", "candidate")
            ),
            Err(AutopilotPrError::PatchOutsideTaskAllowance { .. })
        ));
    }

    #[test]
    fn ci_and_human_review_are_appended_to_same_trajectory() {
        let spec = spec();
        let dag = dag(&spec, TaskRegime::feature());
        let mut trajectory = trajectory(&spec, false);
        let plan = plan(&spec, &dag, &trajectory);
        let receipt = PullRequestEmissionReceipt::new(&plan, 42, revision('d')).unwrap();
        let ci = ExternalPrVerdict::new(
            &receipt,
            ExternalVerdictKind::Ci,
            true,
            "required final-head CI passed",
            digest('e'),
        )
        .unwrap();
        append_external_pr_verdict(&plan, &receipt, &ci, &mut trajectory).unwrap();
        let review = ExternalPrVerdict::new(
            &receipt,
            ExternalVerdictKind::HumanReview,
            false,
            "review requested a safer implementation",
            digest('f'),
        )
        .unwrap();
        append_external_pr_verdict(&plan, &receipt, &review, &mut trajectory).unwrap();

        assert_eq!(trajectory.later_verdicts.len(), 2);
        assert!(trajectory.later_verdicts[0].accepted);
        assert!(!trajectory.later_verdicts[1].accepted);
        assert!(trajectory.later_verdicts[1]
            .source
            .contains("github:human-review:Memorithm/RSI#42@"));
    }

    #[test]
    fn duplicate_external_verdict_is_rejected() {
        let spec = spec();
        let dag = dag(&spec, TaskRegime::feature());
        let mut trajectory = trajectory(&spec, false);
        let plan = plan(&spec, &dag, &trajectory);
        let receipt = PullRequestEmissionReceipt::new(&plan, 42, revision('d')).unwrap();
        let ci = ExternalPrVerdict::new(
            &receipt,
            ExternalVerdictKind::Ci,
            true,
            "CI passed",
            digest('e'),
        )
        .unwrap();
        append_external_pr_verdict(&plan, &receipt, &ci, &mut trajectory).unwrap();
        assert!(matches!(
            append_external_pr_verdict(&plan, &receipt, &ci, &mut trajectory),
            Err(AutopilotPrError::DuplicateExternalVerdict(_))
        ));
    }

    #[test]
    fn verdict_for_other_head_is_rejected() {
        let spec = spec();
        let dag = dag(&spec, TaskRegime::feature());
        let mut trajectory = trajectory(&spec, false);
        let plan = plan(&spec, &dag, &trajectory);
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
            append_external_pr_verdict(&plan, &receipt, &verdict, &mut trajectory),
            Err(AutopilotPrError::ExternalVerdictContextMismatch)
        ));
    }

    #[test]
    fn compatibility_must_match_frozen_repository_revision() {
        let spec = spec();
        let dag = dag(&spec, TaskRegime::feature());
        let mut trajectory = trajectory(&spec, false);
        trajectory.compatibility = CompatibilitySet::new(
            vec![RepositoryRevision::new("Memorithm/RSI", revision('9'), "rsi").unwrap()],
            "stable",
            Vec::new(),
        )
        .unwrap();
        assert!(matches!(
            AutopilotPullRequestPlan::new(
                &spec,
                &dag,
                &trajectory,
                PullRequestPlanDraft::new("implementation", "rsi", "main", "candidate")
            ),
            Err(AutopilotPrError::CompatibilityScopeMismatch { .. })
        ));
    }
}
