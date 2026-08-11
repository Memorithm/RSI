//! P8.3 AUTOPILOT FEATURE regime: tests first, then immutable implementation gate.
//!
//! A FEATURE task is split into two reviewed states. A dedicated test task
//! produces concrete test files whose contents are frozen only after approval
//! evidence is supplied. The implementation task must depend on that test task,
//! cannot have an edit allowance overlapping any frozen test, and every
//! candidate [`PatchSet`] is checked again before evaluation. The same manifest
//! can verify the test bytes in a materialized workspace, so changing a test
//! outside the candidate patch also fails closed.

use crate::autopilot_intake::FrozenAutopilotSpec;
use crate::autopilot_task_dag::{
    AutopilotTask, AutopilotTaskDag, TaskOperation, TaskRegime,
};
use crate::json::Json;
use crate::patchset::{FileOperation, PatchSet, PatchSetError};
use crate::sha256::sha256;
use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

pub const FEATURE_TEST_MANIFEST_SCHEMA_VERSION: u64 = 1;
pub const FEATURE_IMPLEMENTATION_CONTRACT_SCHEMA_VERSION: u64 = 1;

/// Immutable evidence that a human/trusted review step accepted the tests before
/// implementation. Authentication of the external review belongs to the P8.5
/// adapter; this core contract stores only its content hash and authority label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureTestApproval {
    pub authority: String,
    pub evidence_sha256: String,
}

impl FeatureTestApproval {
    pub fn new(
        authority: impl Into<String>,
        evidence_sha256: impl Into<String>,
    ) -> Result<Self, FeatureRegimeError> {
        let authority = authority.into();
        let evidence_sha256 = evidence_sha256.into();
        validate_text("approval.authority", &authority)?;
        validate_digest("approval.evidence_sha256", &evidence_sha256)?;
        Ok(Self {
            authority,
            evidence_sha256: evidence_sha256.to_ascii_lowercase(),
        })
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set("authority", Json::Str(self.authority.clone()))
            .set(
                "evidence_sha256",
                Json::Str(self.evidence_sha256.clone()),
            );
        out
    }
}

/// One approved test file at exact content bytes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FrozenTestArtifact {
    pub repository_role: String,
    pub path: String,
    pub sha256: String,
}

impl FrozenTestArtifact {
    pub fn new(
        repository_role: impl Into<String>,
        path: impl Into<String>,
        sha256: impl Into<String>,
    ) -> Result<Self, FeatureRegimeError> {
        let repository_role = repository_role.into();
        let path = path.into();
        let sha256 = sha256.into();
        validate_identifier("test.repository_role", &repository_role)?;
        validate_relative_path("test.path", &path)?;
        validate_digest("test.sha256", &sha256)?;
        Ok(Self {
            repository_role,
            path,
            sha256: sha256.to_ascii_lowercase(),
        })
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set("path", Json::Str(self.path.clone()))
            .set("repository_role", Json::Str(self.repository_role.clone()))
            .set("sha256", Json::Str(self.sha256.clone()));
        out
    }
}

/// Approved, byte-frozen test set produced by a dedicated FEATURE test task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenFeatureTests {
    schema_version: u64,
    spec_sha256: String,
    dag_sha256: String,
    test_task_id: String,
    approval: FeatureTestApproval,
    artifacts: Vec<FrozenTestArtifact>,
    manifest_sha256: String,
}

impl FrozenFeatureTests {
    pub fn freeze(
        spec: &FrozenAutopilotSpec,
        dag: &AutopilotTaskDag,
        test_task_id: impl Into<String>,
        approval: FeatureTestApproval,
        mut artifacts: Vec<FrozenTestArtifact>,
    ) -> Result<Self, FeatureRegimeError> {
        verify_spec_and_dag(spec, dag)?;
        let test_task_id = test_task_id.into();
        validate_identifier("test_task_id", &test_task_id)?;
        let task = dag
            .task(&test_task_id)
            .ok_or_else(|| FeatureRegimeError::UnknownTask(test_task_id.clone()))?;
        require_feature_task(task, "test")?;
        if artifacts.is_empty() {
            return Err(FeatureRegimeError::EmptyField("test_artifacts"));
        }

        artifacts.sort();
        for pair in artifacts.windows(2) {
            if pair[0].repository_role == pair[1].repository_role && pair[0].path == pair[1].path {
                return Err(FeatureRegimeError::DuplicateTestArtifact {
                    repository_role: pair[0].repository_role.clone(),
                    path: pair[0].path.clone(),
                });
            }
        }
        for artifact in &artifacts {
            if !task
                .repository_roles()
                .iter()
                .any(|role| role == &artifact.repository_role)
            {
                return Err(FeatureRegimeError::TestArtifactOutsideTaskRepositories {
                    repository_role: artifact.repository_role.clone(),
                    path: artifact.path.clone(),
                });
            }
            if !task_allows_path(task, &artifact.repository_role, &artifact.path) {
                return Err(FeatureRegimeError::TestArtifactOutsideEditScope {
                    repository_role: artifact.repository_role.clone(),
                    path: artifact.path.clone(),
                });
            }
        }

        let mut manifest = Self {
            schema_version: FEATURE_TEST_MANIFEST_SCHEMA_VERSION,
            spec_sha256: spec.spec_sha256().to_string(),
            dag_sha256: dag.dag_sha256().to_string(),
            test_task_id,
            approval,
            artifacts,
            manifest_sha256: String::new(),
        };
        manifest.manifest_sha256 = manifest.compute_sha256();
        Ok(manifest)
    }

    pub fn schema_version(&self) -> u64 {
        self.schema_version
    }

    pub fn spec_sha256(&self) -> &str {
        &self.spec_sha256
    }

    pub fn dag_sha256(&self) -> &str {
        &self.dag_sha256
    }

    pub fn test_task_id(&self) -> &str {
        &self.test_task_id
    }

    pub fn approval(&self) -> &FeatureTestApproval {
        &self.approval
    }

    pub fn artifacts(&self) -> &[FrozenTestArtifact] {
        &self.artifacts
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn verify(&self) -> Result<(), FeatureRegimeError> {
        let actual = self.compute_sha256();
        if actual != self.manifest_sha256 {
            return Err(FeatureRegimeError::FrozenManifestHashMismatch {
                expected: self.manifest_sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    /// Verify exact frozen test bytes in one materialized repository root.
    pub fn verify_workspace(
        &self,
        repository_role: &str,
        repository_root: &Path,
    ) -> Result<(), FeatureRegimeError> {
        self.verify()?;
        let mut matched = 0usize;
        for artifact in self
            .artifacts
            .iter()
            .filter(|artifact| artifact.repository_role == repository_role)
        {
            matched += 1;
            let bytes = std::fs::read(repository_root.join(&artifact.path)).map_err(|error| {
                FeatureRegimeError::FrozenTestRead {
                    repository_role: repository_role.to_string(),
                    path: artifact.path.clone(),
                    message: error.to_string(),
                }
            })?;
            let actual = hex_digest(&sha256(&bytes));
            if actual != artifact.sha256 {
                return Err(FeatureRegimeError::FrozenTestContentMismatch {
                    repository_role: repository_role.to_string(),
                    path: artifact.path.clone(),
                    expected: artifact.sha256.clone(),
                    actual,
                });
            }
        }
        if matched == 0 {
            return Err(FeatureRegimeError::UnknownFrozenTestRole(
                repository_role.to_string(),
            ));
        }
        Ok(())
    }

    pub fn to_json_string(&self) -> String {
        let mut root = self.unsigned_json();
        root.set(
            "manifest_sha256",
            Json::Str(self.manifest_sha256.clone()),
        );
        root.to_string()
    }

    fn compute_sha256(&self) -> String {
        hex_digest(&sha256(self.unsigned_json().to_string().as_bytes()))
    }

    fn unsigned_json(&self) -> Json {
        let mut root = Json::obj();
        root.set(
            "approval",
            self.approval.to_json(),
        )
        .set(
            "artifacts",
            Json::Arr(self.artifacts.iter().map(FrozenTestArtifact::to_json).collect()),
        )
        .set("dag_sha256", Json::Str(self.dag_sha256.clone()))
        .set("schema_version", Json::Num(self.schema_version as f64))
        .set("spec_sha256", Json::Str(self.spec_sha256.clone()))
        .set("test_task_id", Json::Str(self.test_task_id.clone()));
        root
    }
}

/// Immutable guard binding one implementation task to its already-approved test
/// task and test bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureImplementationContract {
    schema_version: u64,
    spec_sha256: String,
    dag_sha256: String,
    frozen_tests_sha256: String,
    test_task_id: String,
    implementation_task_id: String,
    contract_sha256: String,
}

impl FeatureImplementationContract {
    pub fn new(
        spec: &FrozenAutopilotSpec,
        dag: &AutopilotTaskDag,
        frozen_tests: &FrozenFeatureTests,
        implementation_task_id: impl Into<String>,
    ) -> Result<Self, FeatureRegimeError> {
        verify_spec_and_dag(spec, dag)?;
        frozen_tests.verify()?;
        if frozen_tests.spec_sha256() != spec.spec_sha256()
            || frozen_tests.dag_sha256() != dag.dag_sha256()
        {
            return Err(FeatureRegimeError::ManifestContextMismatch);
        }

        let implementation_task_id = implementation_task_id.into();
        validate_identifier("implementation_task_id", &implementation_task_id)?;
        let implementation = dag
            .task(&implementation_task_id)
            .ok_or_else(|| FeatureRegimeError::UnknownTask(implementation_task_id.clone()))?;
        require_feature_task(implementation, "implementation")?;
        if implementation.edit_allowances().is_empty() {
            return Err(FeatureRegimeError::ImplementationHasNoEditScope(
                implementation_task_id,
            ));
        }
        if !implementation
            .dependencies()
            .iter()
            .any(|dependency| dependency == frozen_tests.test_task_id())
        {
            return Err(FeatureRegimeError::ImplementationDoesNotDependOnTests {
                implementation_task_id: implementation.id().to_string(),
                test_task_id: frozen_tests.test_task_id().to_string(),
            });
        }

        for allowance in implementation.edit_allowances() {
            for prefix in allowance.allowed_path_prefixes() {
                for artifact in frozen_tests
                    .artifacts()
                    .iter()
                    .filter(|artifact| artifact.repository_role == allowance.repository_role())
                {
                    if path_is_within(&artifact.path, prefix) {
                        return Err(FeatureRegimeError::ImplementationAllowanceTouchesFrozenTest {
                            implementation_task_id: implementation.id().to_string(),
                            repository_role: artifact.repository_role.clone(),
                            frozen_path: artifact.path.clone(),
                            allowed_prefix: prefix.clone(),
                        });
                    }
                }
            }
        }

        let mut contract = Self {
            schema_version: FEATURE_IMPLEMENTATION_CONTRACT_SCHEMA_VERSION,
            spec_sha256: spec.spec_sha256().to_string(),
            dag_sha256: dag.dag_sha256().to_string(),
            frozen_tests_sha256: frozen_tests.manifest_sha256().to_string(),
            test_task_id: frozen_tests.test_task_id().to_string(),
            implementation_task_id: implementation.id().to_string(),
            contract_sha256: String::new(),
        };
        contract.contract_sha256 = contract.compute_sha256();
        Ok(contract)
    }

    pub fn implementation_task_id(&self) -> &str {
        &self.implementation_task_id
    }

    pub fn frozen_tests_sha256(&self) -> &str {
        &self.frozen_tests_sha256
    }

    pub fn contract_sha256(&self) -> &str {
        &self.contract_sha256
    }

    pub fn verify(&self) -> Result<(), FeatureRegimeError> {
        let actual = self.compute_sha256();
        if actual != self.contract_sha256 {
            return Err(FeatureRegimeError::FeatureContractHashMismatch {
                expected: self.contract_sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    /// Fail-closed candidate guard used before materialization/evaluation.
    pub fn validate_patchset(
        &self,
        dag: &AutopilotTaskDag,
        frozen_tests: &FrozenFeatureTests,
        repository_role: &str,
        patch_set: &PatchSet,
    ) -> Result<(), FeatureRegimeError> {
        self.verify()?;
        dag.verify()
            .map_err(|error| FeatureRegimeError::InvalidDag(error.to_string()))?;
        frozen_tests.verify()?;
        if dag.dag_sha256() != self.dag_sha256
            || frozen_tests.manifest_sha256() != self.frozen_tests_sha256
            || frozen_tests.test_task_id() != self.test_task_id
        {
            return Err(FeatureRegimeError::FeatureContractContextMismatch);
        }
        patch_set
            .validate()
            .map_err(FeatureRegimeError::PatchSet)?;
        let task = dag
            .task(&self.implementation_task_id)
            .ok_or_else(|| FeatureRegimeError::UnknownTask(self.implementation_task_id.clone()))?;
        if !task
            .repository_roles()
            .iter()
            .any(|role| role == repository_role)
        {
            return Err(FeatureRegimeError::RepositoryOutsideImplementationTask(
                repository_role.to_string(),
            ));
        }

        let allowances: Vec<_> = task
            .edit_allowances()
            .iter()
            .filter(|allowance| allowance.repository_role() == repository_role)
            .collect();
        if allowances.is_empty() {
            return Err(FeatureRegimeError::RepositoryIsReadOnly(
                repository_role.to_string(),
            ));
        }

        let frozen_paths: BTreeSet<_> = frozen_tests
            .artifacts()
            .iter()
            .filter(|artifact| artifact.repository_role == repository_role)
            .map(|artifact| artifact.path.as_str())
            .collect();

        for operation in patch_set.operations() {
            if frozen_paths.contains(operation.path()) {
                return Err(FeatureRegimeError::CandidateTouchesFrozenTest {
                    repository_role: repository_role.to_string(),
                    path: operation.path().to_string(),
                });
            }
            let required_operation = patch_operation_kind(operation);
            let allowed = allowances.iter().any(|allowance| {
                allowance
                    .allowed_path_prefixes()
                    .iter()
                    .any(|prefix| path_is_within(operation.path(), prefix))
                    && allowance.operations().contains(&required_operation)
            });
            if !allowed {
                return Err(FeatureRegimeError::CandidateOperationOutsideAllowance {
                    repository_role: repository_role.to_string(),
                    path: operation.path().to_string(),
                    operation: required_operation,
                });
            }
        }
        Ok(())
    }

    fn compute_sha256(&self) -> String {
        hex_digest(&sha256(self.unsigned_json().to_string().as_bytes()))
    }

    fn unsigned_json(&self) -> Json {
        let mut root = Json::obj();
        root.set("dag_sha256", Json::Str(self.dag_sha256.clone()))
            .set(
                "frozen_tests_sha256",
                Json::Str(self.frozen_tests_sha256.clone()),
            )
            .set(
                "implementation_task_id",
                Json::Str(self.implementation_task_id.clone()),
            )
            .set("schema_version", Json::Num(self.schema_version as f64))
            .set("spec_sha256", Json::Str(self.spec_sha256.clone()))
            .set("test_task_id", Json::Str(self.test_task_id.clone()));
        root
    }
}

#[derive(Debug)]
pub enum FeatureRegimeError {
    EmptyField(&'static str),
    InvalidText {
        field: &'static str,
        value: String,
    },
    InvalidIdentifier {
        field: &'static str,
        value: String,
    },
    InvalidDigest {
        field: &'static str,
        value: String,
    },
    InvalidPath {
        field: &'static str,
        value: String,
    },
    InvalidSpec(String),
    InvalidDag(String),
    UnknownTask(String),
    NonFeatureTask {
        role: &'static str,
        task_id: String,
    },
    DuplicateTestArtifact {
        repository_role: String,
        path: String,
    },
    TestArtifactOutsideTaskRepositories {
        repository_role: String,
        path: String,
    },
    TestArtifactOutsideEditScope {
        repository_role: String,
        path: String,
    },
    FrozenManifestHashMismatch {
        expected: String,
        actual: String,
    },
    ManifestContextMismatch,
    UnknownFrozenTestRole(String),
    FrozenTestRead {
        repository_role: String,
        path: String,
        message: String,
    },
    FrozenTestContentMismatch {
        repository_role: String,
        path: String,
        expected: String,
        actual: String,
    },
    ImplementationHasNoEditScope(String),
    ImplementationDoesNotDependOnTests {
        implementation_task_id: String,
        test_task_id: String,
    },
    ImplementationAllowanceTouchesFrozenTest {
        implementation_task_id: String,
        repository_role: String,
        frozen_path: String,
        allowed_prefix: String,
    },
    FeatureContractHashMismatch {
        expected: String,
        actual: String,
    },
    FeatureContractContextMismatch,
    PatchSet(PatchSetError),
    RepositoryOutsideImplementationTask(String),
    RepositoryIsReadOnly(String),
    CandidateTouchesFrozenTest {
        repository_role: String,
        path: String,
    },
    CandidateOperationOutsideAllowance {
        repository_role: String,
        path: String,
        operation: TaskOperation,
    },
}

impl fmt::Display for FeatureRegimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "FEATURE field '{field}' is empty"),
            Self::InvalidText { field, value } => {
                write!(f, "invalid FEATURE text '{field}': '{value}'")
            }
            Self::InvalidIdentifier { field, value } => {
                write!(f, "invalid FEATURE identifier '{field}': '{value}'")
            }
            Self::InvalidDigest { field, value } => {
                write!(f, "invalid FEATURE SHA-256 '{field}': '{value}'")
            }
            Self::InvalidPath { field, value } => {
                write!(f, "invalid FEATURE path '{field}': '{value}'")
            }
            Self::InvalidSpec(error) => write!(f, "invalid frozen spec: {error}"),
            Self::InvalidDag(error) => write!(f, "invalid frozen task DAG: {error}"),
            Self::UnknownTask(task) => write!(f, "unknown FEATURE task '{task}'"),
            Self::NonFeatureTask { role, task_id } => {
                write!(f, "{role} task '{task_id}' is not in FEATURE regime")
            }
            Self::DuplicateTestArtifact {
                repository_role,
                path,
            } => write!(f, "duplicate frozen test {repository_role}:{path}"),
            Self::TestArtifactOutsideTaskRepositories {
                repository_role,
                path,
            } => write!(
                f,
                "frozen test {repository_role}:{path} is outside the test task repository subset"
            ),
            Self::TestArtifactOutsideEditScope {
                repository_role,
                path,
            } => write!(
                f,
                "frozen test {repository_role}:{path} is outside the test task edit scope"
            ),
            Self::FrozenManifestHashMismatch { expected, actual } => write!(
                f,
                "frozen FEATURE test manifest hash mismatch: expected {expected}, actual {actual}"
            ),
            Self::ManifestContextMismatch => {
                write!(f, "frozen FEATURE tests belong to another spec/task DAG")
            }
            Self::UnknownFrozenTestRole(role) => {
                write!(f, "no frozen FEATURE tests for repository role '{role}'")
            }
            Self::FrozenTestRead {
                repository_role,
                path,
                message,
            } => write!(f, "cannot read frozen test {repository_role}:{path}: {message}"),
            Self::FrozenTestContentMismatch {
                repository_role,
                path,
                expected,
                actual,
            } => write!(
                f,
                "frozen test changed {repository_role}:{path}: expected {expected}, actual {actual}"
            ),
            Self::ImplementationHasNoEditScope(task) => {
                write!(f, "FEATURE implementation task '{task}' has no edit scope")
            }
            Self::ImplementationDoesNotDependOnTests {
                implementation_task_id,
                test_task_id,
            } => write!(
                f,
                "FEATURE implementation task '{implementation_task_id}' must depend on frozen test task '{test_task_id}'"
            ),
            Self::ImplementationAllowanceTouchesFrozenTest {
                implementation_task_id,
                repository_role,
                frozen_path,
                allowed_prefix,
            } => write!(
                f,
                "FEATURE implementation task '{implementation_task_id}' allowance {repository_role}:{allowed_prefix} covers frozen test {frozen_path}"
            ),
            Self::FeatureContractHashMismatch { expected, actual } => write!(
                f,
                "FEATURE implementation contract hash mismatch: expected {expected}, actual {actual}"
            ),
            Self::FeatureContractContextMismatch => {
                write!(f, "FEATURE implementation contract context mismatch")
            }
            Self::PatchSet(error) => write!(f, "invalid FEATURE candidate PatchSet: {error}"),
            Self::RepositoryOutsideImplementationTask(role) => write!(
                f,
                "repository role '{role}' is outside the FEATURE implementation task"
            ),
            Self::RepositoryIsReadOnly(role) => {
                write!(f, "repository role '{role}' is read-only for this FEATURE task")
            }
            Self::CandidateTouchesFrozenTest {
                repository_role,
                path,
            } => write!(f, "FEATURE candidate attempts to modify frozen test {repository_role}:{path}"),
            Self::CandidateOperationOutsideAllowance {
                repository_role,
                path,
                operation,
            } => write!(
                f,
                "FEATURE candidate operation {operation:?} on {repository_role}:{path} is outside the task allowlist"
            ),
        }
    }
}

impl std::error::Error for FeatureRegimeError {}

fn verify_spec_and_dag(
    spec: &FrozenAutopilotSpec,
    dag: &AutopilotTaskDag,
) -> Result<(), FeatureRegimeError> {
    spec.verify()
        .map_err(|error| FeatureRegimeError::InvalidSpec(error.to_string()))?;
    dag.verify()
        .map_err(|error| FeatureRegimeError::InvalidDag(error.to_string()))?;
    if dag.spec_sha256() != spec.spec_sha256() {
        return Err(FeatureRegimeError::ManifestContextMismatch);
    }
    Ok(())
}

fn require_feature_task(
    task: &AutopilotTask,
    role: &'static str,
) -> Result<(), FeatureRegimeError> {
    if !matches!(task.regime(), TaskRegime::Feature) {
        return Err(FeatureRegimeError::NonFeatureTask {
            role,
            task_id: task.id().to_string(),
        });
    }
    Ok(())
}

fn task_allows_path(task: &AutopilotTask, repository_role: &str, path: &str) -> bool {
    task.edit_allowances()
        .iter()
        .filter(|allowance| allowance.repository_role() == repository_role)
        .any(|allowance| {
            allowance
                .allowed_path_prefixes()
                .iter()
                .any(|prefix| path_is_within(path, prefix))
        })
}

fn patch_operation_kind(operation: &FileOperation) -> TaskOperation {
    match operation {
        FileOperation::Create { .. } => TaskOperation::Create,
        FileOperation::ModifyExact { .. } => TaskOperation::ModifyExact,
        FileOperation::Delete { .. } => TaskOperation::Delete,
    }
}

fn path_is_within(path: &str, prefix: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn validate_text(field: &'static str, value: &str) -> Result<(), FeatureRegimeError> {
    if value.is_empty() {
        return Err(FeatureRegimeError::EmptyField(field));
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(FeatureRegimeError::InvalidText {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), FeatureRegimeError> {
    validate_text(field, value)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(FeatureRegimeError::InvalidIdentifier {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_digest(field: &'static str, value: &str) -> Result<(), FeatureRegimeError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(FeatureRegimeError::InvalidDigest {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_relative_path(field: &'static str, value: &str) -> Result<(), FeatureRegimeError> {
    let invalid = value.is_empty()
        || value.starts_with('/')
        || value.ends_with('/')
        || value.contains('\\')
        || value.contains('\0')
        || value
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."));
    if invalid {
        return Err(FeatureRegimeError::InvalidPath {
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
        AutopilotTaskDag, AutopilotTaskDraft, HardGateProfile, TaskBudget, TaskDagPolicy,
        TaskEditAllowance,
    };

    fn revision(hex: char) -> String {
        hex.to_string().repeat(40)
    }

    fn digest(hex: char) -> String {
        hex.to_string().repeat(64)
    }

    fn spec() -> FrozenAutopilotSpec {
        let exploration = RepositoryExploration::new(
            "Memorithm/RSI",
            revision('a'),
            vec![
                ExplorationObservation::new(
                    "tests",
                    ExplorationSource::repository_file("tests/feature.rs", digest('b')).unwrap(),
                    "inspected the existing FEATURE test boundary",
                )
                .unwrap(),
            ],
        )
        .unwrap();
        ExploredObjective::new(
            "implement a reviewed FEATURE change",
            "tests are approved before implementation",
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
                    "frozen tests pass",
                    AcceptanceCheck::command("rsi", "cargo_test", Vec::new()).unwrap(),
                )
                .unwrap(),
            ],
            vec!["editing approved tests during implementation".to_string()],
            SpecBudget::new(20, 20_000, 20_000).unwrap(),
            vec![
                RepositoryScope::new(
                    "rsi",
                    "Memorithm/RSI",
                    revision('a'),
                    vec!["src".to_string(), "tests".to_string()],
                )
                .unwrap(),
            ],
        ))
        .unwrap()
    }

    fn task(
        id: &str,
        path: &str,
        operations: Vec<TaskOperation>,
        dependencies: Vec<String>,
    ) -> AutopilotTask {
        AutopilotTask::new(AutopilotTaskDraft {
            id: id.to_string(),
            description: format!("FEATURE task {id}"),
            regime: TaskRegime::feature(),
            repository_roles: vec!["rsi".to_string()],
            edit_allowances: vec![
                TaskEditAllowance::new("rsi", vec![path.to_string()], operations).unwrap(),
            ],
            hard_gate_profile: HardGateProfile::engineering_strict(),
            budget: TaskBudget::new(2, 4, 4_000, 4_000).unwrap(),
            dependencies,
            done_criterion_id: "tests-pass".to_string(),
        })
        .unwrap()
    }

    fn dag(implementation_path: &str) -> AutopilotTaskDag {
        AutopilotTaskDag::new(
            &spec(),
            vec![
                task(
                    "tests-only",
                    "tests",
                    vec![TaskOperation::Create, TaskOperation::ModifyExact],
                    Vec::new(),
                ),
                task(
                    "implementation",
                    implementation_path,
                    vec![TaskOperation::Create, TaskOperation::ModifyExact],
                    vec!["tests-only".to_string()],
                ),
            ],
            TaskDagPolicy::new(4, 4, 8).unwrap(),
        )
        .unwrap()
    }

    fn frozen_tests(spec: &FrozenAutopilotSpec, dag: &AutopilotTaskDag) -> FrozenFeatureTests {
        FrozenFeatureTests::freeze(
            spec,
            dag,
            "tests-only",
            FeatureTestApproval::new("human-review", digest('e')).unwrap(),
            vec![FrozenTestArtifact::new("rsi", "tests/feature.rs", digest('f')).unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn implementation_must_depend_on_approved_test_task() {
        let spec = spec();
        let dag = AutopilotTaskDag::new(
            &spec,
            vec![
                task(
                    "tests-only",
                    "tests",
                    vec![TaskOperation::Create],
                    Vec::new(),
                ),
                task(
                    "implementation",
                    "src",
                    vec![TaskOperation::ModifyExact],
                    Vec::new(),
                ),
            ],
            TaskDagPolicy::new(4, 4, 8).unwrap(),
        )
        .unwrap();
        let frozen = frozen_tests(&spec, &dag);
        assert!(matches!(
            FeatureImplementationContract::new(&spec, &dag, &frozen, "implementation"),
            Err(FeatureRegimeError::ImplementationDoesNotDependOnTests { .. })
        ));
    }

    #[test]
    fn implementation_allowlist_cannot_cover_frozen_test() {
        let spec = spec();
        let dag = dag("tests");
        let frozen = frozen_tests(&spec, &dag);
        assert!(matches!(
            FeatureImplementationContract::new(&spec, &dag, &frozen, "implementation"),
            Err(FeatureRegimeError::ImplementationAllowanceTouchesFrozenTest { .. })
        ));
    }

    #[test]
    fn candidate_patchset_cannot_touch_frozen_test_even_before_scope_check() {
        let spec = spec();
        let dag = dag("src");
        let frozen = frozen_tests(&spec, &dag);
        let contract =
            FeatureImplementationContract::new(&spec, &dag, &frozen, "implementation").unwrap();
        let patch = PatchSet::new(vec![FileOperation::modify_exact(
            "tests/feature.rs",
            "old",
            "new",
        )])
        .unwrap();
        assert!(matches!(
            contract.validate_patchset(&dag, &frozen, "rsi", &patch),
            Err(FeatureRegimeError::CandidateTouchesFrozenTest { .. })
        ));
    }

    #[test]
    fn candidate_operation_must_match_task_operation_allowlist() {
        let spec = spec();
        let dag = dag("src");
        let frozen = frozen_tests(&spec, &dag);
        let contract =
            FeatureImplementationContract::new(&spec, &dag, &frozen, "implementation").unwrap();
        let patch = PatchSet::new(vec![FileOperation::delete("src/lib.rs", digest('a'))]).unwrap();
        assert!(matches!(
            contract.validate_patchset(&dag, &frozen, "rsi", &patch),
            Err(FeatureRegimeError::CandidateOperationOutsideAllowance { .. })
        ));
    }

    #[test]
    fn valid_implementation_patch_passes_guard() {
        let spec = spec();
        let dag = dag("src");
        let frozen = frozen_tests(&spec, &dag);
        let contract =
            FeatureImplementationContract::new(&spec, &dag, &frozen, "implementation").unwrap();
        let patch = PatchSet::new(vec![FileOperation::modify_exact(
            "src/lib.rs",
            "old",
            "new",
        )])
        .unwrap();
        contract
            .validate_patchset(&dag, &frozen, "rsi", &patch)
            .unwrap();
        frozen.verify().unwrap();
        contract.verify().unwrap();
    }

    #[test]
    fn frozen_workspace_bytes_are_checked_exactly() {
        let spec = spec();
        let dag = dag("src");
        let bytes = b"#[test]\nfn approved() {}\n";
        let artifact = FrozenTestArtifact::new(
            "rsi",
            "tests/feature.rs",
            hex_digest(&sha256(bytes)),
        )
        .unwrap();
        let frozen = FrozenFeatureTests::freeze(
            &spec,
            &dag,
            "tests-only",
            FeatureTestApproval::new("human-review", digest('e')).unwrap(),
            vec![artifact],
        )
        .unwrap();

        let root = std::env::temp_dir().join(format!(
            "rsi-feature-frozen-tests-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("tests")).unwrap();
        std::fs::write(root.join("tests/feature.rs"), bytes).unwrap();
        frozen.verify_workspace("rsi", &root).unwrap();
        std::fs::write(root.join("tests/feature.rs"), b"changed").unwrap();
        assert!(matches!(
            frozen.verify_workspace("rsi", &root),
            Err(FeatureRegimeError::FrozenTestContentMismatch { .. })
        ));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn frozen_test_manifest_is_order_independent() {
        let spec = spec();
        let dag = dag("src");
        let approval = FeatureTestApproval::new("human-review", digest('e')).unwrap();
        let a = FrozenTestArtifact::new("rsi", "tests/a.rs", digest('a')).unwrap();
        let b = FrozenTestArtifact::new("rsi", "tests/b.rs", digest('b')).unwrap();
        let first = FrozenFeatureTests::freeze(
            &spec,
            &dag,
            "tests-only",
            approval.clone(),
            vec![b.clone(), a.clone()],
        )
        .unwrap();
        let second = FrozenFeatureTests::freeze(
            &spec,
            &dag,
            "tests-only",
            approval,
            vec![a, b],
        )
        .unwrap();
        assert_eq!(first.manifest_sha256(), second.manifest_sha256());
        assert_eq!(first.to_json_string(), second.to_json_string());
    }
}
