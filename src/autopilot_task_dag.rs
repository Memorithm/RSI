//! P8.2 AUTOPILOT task DAG contract.
//!
//! The DAG is derived from one frozen P8.1 specification. Tasks may narrow the
//! repository/path/operation scope but can never widen it. The base COGNO
//! engineering hard gates are structural and cannot be disabled by a task.
//! PERF tasks carry an explicit benchmark-profile identifier; FEATURE tasks do
//! not gain benchmark semantics until P8.3/P8.4 define their execution regimes.

use crate::autopilot_intake::{
    AcceptanceCheck, FrozenAutopilotSpec, RepositoryScope, SpecBudget,
};
use crate::json::Json;
use crate::sha256::sha256;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const AUTOPILOT_TASK_DAG_SCHEMA_VERSION: u64 = 1;

const BASE_HARD_GATES: [&str; 7] = [
    "build",
    "required_tests",
    "numerical_parity",
    "provenance",
    "deterministic_contract",
    "resource_budget",
    "policy_checks",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskOperation {
    Create,
    ModifyExact,
    Delete,
    Rename,
}

impl TaskOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::ModifyExact => "modify_exact",
            Self::Delete => "delete",
            Self::Rename => "rename",
        }
    }
}

/// Task-level edit scope. A repository can be present in the task's read/eval
/// subset without receiving an edit allowance.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaskEditAllowance {
    repository_role: String,
    allowed_path_prefixes: Vec<String>,
    operations: Vec<TaskOperation>,
}

impl TaskEditAllowance {
    pub fn new(
        repository_role: impl Into<String>,
        mut allowed_path_prefixes: Vec<String>,
        mut operations: Vec<TaskOperation>,
    ) -> Result<Self, AutopilotTaskDagError> {
        let repository_role = repository_role.into();
        validate_identifier("edit_allowance.repository_role", &repository_role)?;
        if allowed_path_prefixes.is_empty() {
            return Err(AutopilotTaskDagError::EmptyField(
                "edit_allowance.allowed_path_prefixes",
            ));
        }
        if operations.is_empty() {
            return Err(AutopilotTaskDagError::EmptyField(
                "edit_allowance.operations",
            ));
        }
        for path in &allowed_path_prefixes {
            validate_relative_path("edit_allowance.allowed_path_prefixes", path)?;
        }
        allowed_path_prefixes.sort();
        allowed_path_prefixes.dedup();
        operations.sort();
        operations.dedup();
        Ok(Self {
            repository_role,
            allowed_path_prefixes,
            operations,
        })
    }

    pub fn repository_role(&self) -> &str {
        &self.repository_role
    }

    pub fn allowed_path_prefixes(&self) -> &[String] {
        &self.allowed_path_prefixes
    }

    pub fn operations(&self) -> &[TaskOperation] {
        &self.operations
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set(
            "allowed_path_prefixes",
            Json::Arr(
                self.allowed_path_prefixes
                    .iter()
                    .cloned()
                    .map(Json::Str)
                    .collect(),
            ),
        )
        .set(
            "operations",
            Json::Arr(
                self.operations
                    .iter()
                    .map(|operation| Json::Str(operation.as_str().to_string()))
                    .collect(),
            ),
        )
        .set("repository_role", Json::Str(self.repository_role.clone()));
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskRegime {
    Feature,
    Perf { benchmark_profile_id: String },
}

impl TaskRegime {
    pub fn feature() -> Self {
        Self::Feature
    }

    pub fn perf(
        benchmark_profile_id: impl Into<String>,
    ) -> Result<Self, AutopilotTaskDagError> {
        let benchmark_profile_id = benchmark_profile_id.into();
        validate_identifier("benchmark_profile_id", &benchmark_profile_id)?;
        Ok(Self::Perf {
            benchmark_profile_id,
        })
    }

    fn validate(&self) -> Result<(), AutopilotTaskDagError> {
        if let Self::Perf {
            benchmark_profile_id,
        } = self
        {
            validate_identifier("benchmark_profile_id", benchmark_profile_id)?;
        }
        Ok(())
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        match self {
            Self::Feature => {
                out.set("kind", Json::Str("feature".to_string()));
            }
            Self::Perf {
                benchmark_profile_id,
            } => {
                out.set(
                    "benchmark_profile_id",
                    Json::Str(benchmark_profile_id.clone()),
                )
                .set("kind", Json::Str("perf".to_string()));
            }
        }
        out
    }
}

/// Base engineering gates are always required. Profiles can only add trusted
/// named gates; they cannot remove build/tests/parity/provenance/determinism/
/// resources/policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardGateProfile {
    additional_required_gates: Vec<String>,
}

impl HardGateProfile {
    pub fn engineering_strict() -> Self {
        Self {
            additional_required_gates: Vec::new(),
        }
    }

    pub fn engineering_strict_with(
        mut additional_required_gates: Vec<String>,
    ) -> Result<Self, AutopilotTaskDagError> {
        for gate in &additional_required_gates {
            validate_identifier("hard_gate_profile.additional_required_gates", gate)?;
            if BASE_HARD_GATES.contains(&gate.as_str()) {
                return Err(AutopilotTaskDagError::DuplicateHardGate(gate.clone()));
            }
        }
        additional_required_gates.sort();
        additional_required_gates.dedup();
        Ok(Self {
            additional_required_gates,
        })
    }

    pub fn required_gates(&self) -> Vec<String> {
        BASE_HARD_GATES
            .iter()
            .map(|gate| (*gate).to_string())
            .chain(self.additional_required_gates.iter().cloned())
            .collect()
    }

    fn to_json(&self) -> Json {
        Json::Arr(
            self.required_gates()
                .into_iter()
                .map(Json::Str)
                .collect(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskBudget {
    pub max_candidate_evaluations: u64,
    pub max_llm_steps: u64,
    pub max_wall_time_ms: u64,
    pub max_tokens: u64,
}

impl TaskBudget {
    pub fn new(
        max_candidate_evaluations: u64,
        max_llm_steps: u64,
        max_wall_time_ms: u64,
        max_tokens: u64,
    ) -> Result<Self, AutopilotTaskDagError> {
        if max_candidate_evaluations == 0
            || max_llm_steps == 0
            || max_wall_time_ms == 0
            || max_tokens == 0
        {
            return Err(AutopilotTaskDagError::InvalidTaskBudget);
        }
        Ok(Self {
            max_candidate_evaluations,
            max_llm_steps,
            max_wall_time_ms,
            max_tokens,
        })
    }

    fn to_json(self) -> Json {
        let mut out = Json::obj();
        out.set(
            "max_candidate_evaluations",
            Json::Num(self.max_candidate_evaluations as f64),
        )
        .set("max_llm_steps", Json::Num(self.max_llm_steps as f64))
        .set("max_tokens", Json::Num(self.max_tokens as f64))
        .set("max_wall_time_ms", Json::Num(self.max_wall_time_ms as f64));
        out
    }
}

/// Trusted host bounds that are not model-controlled spec content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskDagPolicy {
    pub max_tasks: usize,
    pub max_candidate_evaluations_per_task: u64,
    pub max_total_candidate_evaluations: u64,
}

impl TaskDagPolicy {
    pub fn new(
        max_tasks: usize,
        max_candidate_evaluations_per_task: u64,
        max_total_candidate_evaluations: u64,
    ) -> Result<Self, AutopilotTaskDagError> {
        if max_tasks == 0
            || max_candidate_evaluations_per_task == 0
            || max_total_candidate_evaluations == 0
        {
            return Err(AutopilotTaskDagError::InvalidDagPolicy);
        }
        Ok(Self {
            max_tasks,
            max_candidate_evaluations_per_task,
            max_total_candidate_evaluations,
        })
    }

    fn to_json(self) -> Json {
        let mut out = Json::obj();
        out.set("max_tasks", Json::Num(self.max_tasks as f64))
            .set(
                "max_candidate_evaluations_per_task",
                Json::Num(self.max_candidate_evaluations_per_task as f64),
            )
            .set(
                "max_total_candidate_evaluations",
                Json::Num(self.max_total_candidate_evaluations as f64),
            );
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutopilotTaskDraft {
    pub id: String,
    pub description: String,
    pub regime: TaskRegime,
    pub repository_roles: Vec<String>,
    pub edit_allowances: Vec<TaskEditAllowance>,
    pub hard_gate_profile: HardGateProfile,
    pub budget: TaskBudget,
    pub dependencies: Vec<String>,
    pub done_criterion_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutopilotTask {
    id: String,
    description: String,
    regime: TaskRegime,
    repository_roles: Vec<String>,
    edit_allowances: Vec<TaskEditAllowance>,
    hard_gate_profile: HardGateProfile,
    budget: TaskBudget,
    dependencies: Vec<String>,
    done_criterion_id: String,
}

impl AutopilotTask {
    pub fn new(mut draft: AutopilotTaskDraft) -> Result<Self, AutopilotTaskDagError> {
        validate_identifier("task.id", &draft.id)?;
        validate_text("task.description", &draft.description)?;
        draft.regime.validate()?;
        validate_identifier("task.done_criterion_id", &draft.done_criterion_id)?;
        if draft.repository_roles.is_empty() {
            return Err(AutopilotTaskDagError::EmptyField("task.repository_roles"));
        }
        for role in &draft.repository_roles {
            validate_identifier("task.repository_roles", role)?;
        }
        draft.repository_roles.sort();
        draft.repository_roles.dedup();

        draft.edit_allowances.sort();
        for pair in draft.edit_allowances.windows(2) {
            if pair[0].repository_role == pair[1].repository_role {
                return Err(AutopilotTaskDagError::DuplicateEditAllowance(
                    pair[0].repository_role.clone(),
                ));
            }
        }

        for dependency in &draft.dependencies {
            validate_identifier("task.dependencies", dependency)?;
            if dependency == &draft.id {
                return Err(AutopilotTaskDagError::SelfDependency(draft.id.clone()));
            }
        }
        draft.dependencies.sort();
        draft.dependencies.dedup();

        Ok(Self {
            id: draft.id,
            description: draft.description,
            regime: draft.regime,
            repository_roles: draft.repository_roles,
            edit_allowances: draft.edit_allowances,
            hard_gate_profile: draft.hard_gate_profile,
            budget: draft.budget,
            dependencies: draft.dependencies,
            done_criterion_id: draft.done_criterion_id,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn regime(&self) -> &TaskRegime {
        &self.regime
    }

    pub fn repository_roles(&self) -> &[String] {
        &self.repository_roles
    }

    pub fn edit_allowances(&self) -> &[TaskEditAllowance] {
        &self.edit_allowances
    }

    pub fn hard_gate_profile(&self) -> &HardGateProfile {
        &self.hard_gate_profile
    }

    pub fn budget(&self) -> TaskBudget {
        self.budget
    }

    pub fn dependencies(&self) -> &[String] {
        &self.dependencies
    }

    pub fn done_criterion_id(&self) -> &str {
        &self.done_criterion_id
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set("budget", self.budget.to_json())
            .set(
                "dependencies",
                Json::Arr(self.dependencies.iter().cloned().map(Json::Str).collect()),
            )
            .set("description", Json::Str(self.description.clone()))
            .set("done_criterion_id", Json::Str(self.done_criterion_id.clone()))
            .set(
                "edit_allowances",
                Json::Arr(
                    self.edit_allowances
                        .iter()
                        .map(TaskEditAllowance::to_json)
                        .collect(),
                ),
            )
            .set("hard_gates", self.hard_gate_profile.to_json())
            .set("id", Json::Str(self.id.clone()))
            .set("regime", self.regime.to_json())
            .set(
                "repository_roles",
                Json::Arr(
                    self.repository_roles
                        .iter()
                        .cloned()
                        .map(Json::Str)
                        .collect(),
                ),
            );
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutopilotTaskDag {
    schema_version: u64,
    spec_sha256: String,
    policy: TaskDagPolicy,
    tasks: Vec<AutopilotTask>,
    topological_order: Vec<String>,
    dag_sha256: String,
}

impl AutopilotTaskDag {
    pub fn new(
        spec: &FrozenAutopilotSpec,
        mut tasks: Vec<AutopilotTask>,
        policy: TaskDagPolicy,
    ) -> Result<Self, AutopilotTaskDagError> {
        spec.verify()
            .map_err(|error| AutopilotTaskDagError::InvalidFrozenSpec(error.to_string()))?;
        if tasks.is_empty() {
            return Err(AutopilotTaskDagError::EmptyField("tasks"));
        }
        if tasks.len() > policy.max_tasks {
            return Err(AutopilotTaskDagError::TaskLimitExceeded {
                actual: tasks.len(),
                maximum: policy.max_tasks,
            });
        }

        tasks.sort_by(|left, right| left.id.cmp(&right.id));
        for pair in tasks.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(AutopilotTaskDagError::DuplicateTask(pair[0].id.clone()));
            }
        }

        let spec_scope: BTreeMap<_, _> = spec
            .repository_scope()
            .iter()
            .map(|scope| (scope.role.as_str(), scope))
            .collect();
        let criteria: BTreeMap<_, _> = spec
            .acceptance_criteria()
            .iter()
            .map(|criterion| (criterion.id.as_str(), criterion))
            .collect();
        let task_ids: BTreeSet<_> = tasks.iter().map(|task| task.id.as_str()).collect();

        let mut total_candidate_evaluations = 0u64;
        let mut total_llm_steps = 0u64;
        let mut total_wall_time_ms = 0u64;
        let mut total_tokens = 0u64;

        for task in &tasks {
            validate_task_scope(task, &spec_scope, &criteria)?;
            for dependency in &task.dependencies {
                if !task_ids.contains(dependency.as_str()) {
                    return Err(AutopilotTaskDagError::UnknownDependency {
                        task_id: task.id.clone(),
                        dependency: dependency.clone(),
                    });
                }
            }
            if task.budget.max_candidate_evaluations > policy.max_candidate_evaluations_per_task {
                return Err(AutopilotTaskDagError::CandidateEvaluationLimitExceeded {
                    task_id: task.id.clone(),
                    actual: task.budget.max_candidate_evaluations,
                    maximum: policy.max_candidate_evaluations_per_task,
                });
            }
            total_candidate_evaluations = checked_budget_add(
                "candidate_evaluations",
                total_candidate_evaluations,
                task.budget.max_candidate_evaluations,
            )?;
            total_llm_steps = checked_budget_add(
                "llm_steps",
                total_llm_steps,
                task.budget.max_llm_steps,
            )?;
            total_wall_time_ms = checked_budget_add(
                "wall_time_ms",
                total_wall_time_ms,
                task.budget.max_wall_time_ms,
            )?;
            total_tokens = checked_budget_add("tokens", total_tokens, task.budget.max_tokens)?;
        }

        if total_candidate_evaluations > policy.max_total_candidate_evaluations {
            return Err(AutopilotTaskDagError::TotalCandidateEvaluationLimitExceeded {
                actual: total_candidate_evaluations,
                maximum: policy.max_total_candidate_evaluations,
            });
        }
        validate_objective_budget(
            spec.budget(),
            total_llm_steps,
            total_wall_time_ms,
            total_tokens,
        )?;

        let topological_order = topological_order(&tasks)?;
        let mut dag = Self {
            schema_version: AUTOPILOT_TASK_DAG_SCHEMA_VERSION,
            spec_sha256: spec.spec_sha256().to_string(),
            policy,
            tasks,
            topological_order,
            dag_sha256: String::new(),
        };
        dag.dag_sha256 = dag.compute_sha256();
        Ok(dag)
    }

    pub fn schema_version(&self) -> u64 {
        self.schema_version
    }

    pub fn spec_sha256(&self) -> &str {
        &self.spec_sha256
    }

    pub fn policy(&self) -> TaskDagPolicy {
        self.policy
    }

    pub fn tasks(&self) -> &[AutopilotTask] {
        &self.tasks
    }

    pub fn topological_order(&self) -> &[String] {
        &self.topological_order
    }

    pub fn dag_sha256(&self) -> &str {
        &self.dag_sha256
    }

    pub fn task(&self, id: &str) -> Option<&AutopilotTask> {
        self.tasks
            .binary_search_by(|task| task.id.as_str().cmp(id))
            .ok()
            .map(|index| &self.tasks[index])
    }

    pub fn to_json_string(&self) -> String {
        let mut root = self.unsigned_json();
        root.set("dag_sha256", Json::Str(self.dag_sha256.clone()));
        root.to_string()
    }

    pub fn verify(&self) -> Result<(), AutopilotTaskDagError> {
        let actual = self.compute_sha256();
        if actual != self.dag_sha256 {
            return Err(AutopilotTaskDagError::FrozenDagHashMismatch {
                expected: self.dag_sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    fn compute_sha256(&self) -> String {
        hex_digest(&sha256(self.unsigned_json().to_string().as_bytes()))
    }

    fn unsigned_json(&self) -> Json {
        let mut root = Json::obj();
        root.set("policy", self.policy.to_json())
            .set("schema_version", Json::Num(self.schema_version as f64))
            .set("spec_sha256", Json::Str(self.spec_sha256.clone()))
            .set(
                "tasks",
                Json::Arr(self.tasks.iter().map(AutopilotTask::to_json).collect()),
            )
            .set(
                "topological_order",
                Json::Arr(
                    self.topological_order
                        .iter()
                        .cloned()
                        .map(Json::Str)
                        .collect(),
                ),
            );
        root
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutopilotTaskDagError {
    EmptyField(&'static str),
    InvalidText {
        field: &'static str,
        value: String,
    },
    InvalidIdentifier {
        field: &'static str,
        value: String,
    },
    InvalidPath {
        field: &'static str,
        value: String,
    },
    InvalidTaskBudget,
    InvalidDagPolicy,
    InvalidFrozenSpec(String),
    DuplicateHardGate(String),
    DuplicateEditAllowance(String),
    DuplicateTask(String),
    SelfDependency(String),
    UnknownDependency {
        task_id: String,
        dependency: String,
    },
    UnknownRepositoryRole {
        task_id: String,
        repository_role: String,
    },
    EditRoleOutsideTaskSubset {
        task_id: String,
        repository_role: String,
    },
    PathOutsideFrozenScope {
        task_id: String,
        repository_role: String,
        path_prefix: String,
    },
    UnknownDoneCriterion {
        task_id: String,
        criterion_id: String,
    },
    DoneCriterionOutsideTaskSubset {
        task_id: String,
        criterion_id: String,
        repository_role: String,
    },
    TaskLimitExceeded {
        actual: usize,
        maximum: usize,
    },
    CandidateEvaluationLimitExceeded {
        task_id: String,
        actual: u64,
        maximum: u64,
    },
    TotalCandidateEvaluationLimitExceeded {
        actual: u64,
        maximum: u64,
    },
    ObjectiveBudgetExceeded {
        field: &'static str,
        actual: u64,
        maximum: u64,
    },
    BudgetOverflow(&'static str),
    CycleDetected(Vec<String>),
    FrozenDagHashMismatch {
        expected: String,
        actual: String,
    },
}

impl fmt::Display for AutopilotTaskDagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "AUTOPILOT DAG field '{field}' is empty"),
            Self::InvalidText { field, value } => {
                write!(f, "invalid AUTOPILOT DAG text '{field}': '{value}'")
            }
            Self::InvalidIdentifier { field, value } => {
                write!(f, "invalid AUTOPILOT DAG identifier '{field}': '{value}'")
            }
            Self::InvalidPath { field, value } => {
                write!(f, "invalid AUTOPILOT DAG path '{field}': '{value}'")
            }
            Self::InvalidTaskBudget => write!(f, "AUTOPILOT task budget values must be non-zero"),
            Self::InvalidDagPolicy => write!(f, "AUTOPILOT DAG policy limits must be non-zero"),
            Self::InvalidFrozenSpec(error) => write!(f, "invalid frozen AUTOPILOT spec: {error}"),
            Self::DuplicateHardGate(gate) => write!(f, "duplicate base hard gate '{gate}'"),
            Self::DuplicateEditAllowance(role) => {
                write!(f, "duplicate edit allowance for repository role '{role}'")
            }
            Self::DuplicateTask(id) => write!(f, "duplicate AUTOPILOT task '{id}'"),
            Self::SelfDependency(id) => write!(f, "AUTOPILOT task '{id}' depends on itself"),
            Self::UnknownDependency {
                task_id,
                dependency,
            } => write!(f, "task '{task_id}' depends on unknown task '{dependency}'"),
            Self::UnknownRepositoryRole {
                task_id,
                repository_role,
            } => write!(
                f,
                "task '{task_id}' references repository role '{repository_role}' outside the frozen spec"
            ),
            Self::EditRoleOutsideTaskSubset {
                task_id,
                repository_role,
            } => write!(
                f,
                "task '{task_id}' edits repository role '{repository_role}' outside its repository subset"
            ),
            Self::PathOutsideFrozenScope {
                task_id,
                repository_role,
                path_prefix,
            } => write!(
                f,
                "task '{task_id}' path '{path_prefix}' widens frozen scope for repository role '{repository_role}'"
            ),
            Self::UnknownDoneCriterion {
                task_id,
                criterion_id,
            } => write!(f, "task '{task_id}' references unknown done criterion '{criterion_id}'"),
            Self::DoneCriterionOutsideTaskSubset {
                task_id,
                criterion_id,
                repository_role,
            } => write!(
                f,
                "task '{task_id}' done criterion '{criterion_id}' targets repository role '{repository_role}' outside the task subset"
            ),
            Self::TaskLimitExceeded { actual, maximum } => {
                write!(f, "task count {actual} exceeds trusted maximum {maximum}")
            }
            Self::CandidateEvaluationLimitExceeded {
                task_id,
                actual,
                maximum,
            } => write!(
                f,
                "task '{task_id}' candidate-evaluation budget {actual} exceeds trusted maximum {maximum}"
            ),
            Self::TotalCandidateEvaluationLimitExceeded { actual, maximum } => write!(
                f,
                "total candidate-evaluation budget {actual} exceeds trusted maximum {maximum}"
            ),
            Self::ObjectiveBudgetExceeded {
                field,
                actual,
                maximum,
            } => write!(
                f,
                "task DAG {field} budget {actual} exceeds frozen objective maximum {maximum}"
            ),
            Self::BudgetOverflow(field) => write!(f, "task DAG budget overflow for '{field}'"),
            Self::CycleDetected(tasks) => write!(f, "AUTOPILOT task DAG contains a cycle: {tasks:?}"),
            Self::FrozenDagHashMismatch { expected, actual } => write!(
                f,
                "frozen AUTOPILOT DAG hash mismatch: expected {expected}, actual {actual}"
            ),
        }
    }
}

impl std::error::Error for AutopilotTaskDagError {}

fn validate_task_scope(
    task: &AutopilotTask,
    spec_scope: &BTreeMap<&str, &RepositoryScope>,
    criteria: &BTreeMap<&str, &crate::autopilot_intake::AcceptanceCriterion>,
) -> Result<(), AutopilotTaskDagError> {
    let task_roles: BTreeSet<_> = task.repository_roles.iter().map(String::as_str).collect();
    for role in &task.repository_roles {
        if !spec_scope.contains_key(role.as_str()) {
            return Err(AutopilotTaskDagError::UnknownRepositoryRole {
                task_id: task.id.clone(),
                repository_role: role.clone(),
            });
        }
    }

    for allowance in &task.edit_allowances {
        let role = allowance.repository_role();
        if !task_roles.contains(role) {
            return Err(AutopilotTaskDagError::EditRoleOutsideTaskSubset {
                task_id: task.id.clone(),
                repository_role: role.to_string(),
            });
        }
        let frozen = spec_scope
            .get(role)
            .ok_or_else(|| AutopilotTaskDagError::UnknownRepositoryRole {
                task_id: task.id.clone(),
                repository_role: role.to_string(),
            })?;
        for path in allowance.allowed_path_prefixes() {
            if !frozen
                .allowed_path_prefixes
                .iter()
                .any(|prefix| path_is_within(path, prefix))
            {
                return Err(AutopilotTaskDagError::PathOutsideFrozenScope {
                    task_id: task.id.clone(),
                    repository_role: role.to_string(),
                    path_prefix: path.clone(),
                });
            }
        }
    }

    let criterion = criteria.get(task.done_criterion_id.as_str()).ok_or_else(|| {
        AutopilotTaskDagError::UnknownDoneCriterion {
            task_id: task.id.clone(),
            criterion_id: task.done_criterion_id.clone(),
        }
    })?;
    let criterion_role = acceptance_repository_role(&criterion.check);
    if !task_roles.contains(criterion_role) {
        return Err(AutopilotTaskDagError::DoneCriterionOutsideTaskSubset {
            task_id: task.id.clone(),
            criterion_id: task.done_criterion_id.clone(),
            repository_role: criterion_role.to_string(),
        });
    }
    Ok(())
}

fn acceptance_repository_role(check: &AcceptanceCheck) -> &str {
    match check {
        AcceptanceCheck::Command {
            repository_role, ..
        }
        | AcceptanceCheck::FilePresent {
            repository_role, ..
        }
        | AcceptanceCheck::FileAbsent {
            repository_role, ..
        }
        | AcceptanceCheck::FileSha256 {
            repository_role, ..
        } => repository_role,
    }
}

fn path_is_within(path: &str, prefix: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn validate_objective_budget(
    budget: SpecBudget,
    llm_steps: u64,
    wall_time_ms: u64,
    tokens: u64,
) -> Result<(), AutopilotTaskDagError> {
    for (field, actual, maximum) in [
        ("llm_steps", llm_steps, budget.max_llm_steps),
        ("wall_time_ms", wall_time_ms, budget.max_wall_time_ms),
        ("tokens", tokens, budget.max_tokens),
    ] {
        if actual > maximum {
            return Err(AutopilotTaskDagError::ObjectiveBudgetExceeded {
                field,
                actual,
                maximum,
            });
        }
    }
    Ok(())
}

fn checked_budget_add(
    field: &'static str,
    left: u64,
    right: u64,
) -> Result<u64, AutopilotTaskDagError> {
    left.checked_add(right)
        .ok_or(AutopilotTaskDagError::BudgetOverflow(field))
}

fn topological_order(tasks: &[AutopilotTask]) -> Result<Vec<String>, AutopilotTaskDagError> {
    let mut indegree: BTreeMap<String, usize> = tasks
        .iter()
        .map(|task| (task.id.clone(), task.dependencies.len()))
        .collect();
    let mut dependents: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for task in tasks {
        for dependency in &task.dependencies {
            dependents
                .entry(dependency.clone())
                .or_default()
                .push(task.id.clone());
        }
    }
    for values in dependents.values_mut() {
        values.sort();
    }

    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut order = Vec::with_capacity(tasks.len());
    while let Some(id) = ready.iter().next().cloned() {
        ready.remove(&id);
        order.push(id.clone());
        if let Some(children) = dependents.get(&id) {
            for child in children {
                let degree = indegree.get_mut(child).expect("validated task ID");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(child.clone());
                }
            }
        }
    }

    if order.len() != tasks.len() {
        let cyclic = indegree
            .into_iter()
            .filter(|(_, degree)| *degree > 0)
            .map(|(id, _)| id)
            .collect();
        return Err(AutopilotTaskDagError::CycleDetected(cyclic));
    }
    Ok(order)
}

fn validate_text(field: &'static str, value: &str) -> Result<(), AutopilotTaskDagError> {
    if value.is_empty() {
        return Err(AutopilotTaskDagError::EmptyField(field));
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(AutopilotTaskDagError::InvalidText {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), AutopilotTaskDagError> {
    validate_text(field, value)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(AutopilotTaskDagError::InvalidIdentifier {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_relative_path(field: &'static str, value: &str) -> Result<(), AutopilotTaskDagError> {
    let invalid = value.is_empty()
        || value.starts_with('/')
        || value.ends_with('/')
        || value.contains('\\')
        || value.contains('\0')
        || value
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."));
    if invalid {
        return Err(AutopilotTaskDagError::InvalidPath {
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
        AcceptanceCriterion, AutopilotSpecDraft, ExplorationObservation, ExplorationSource,
        ExploredObjective, RepositoryExploration, RepositoryScope,
    };

    fn revision(hex: char) -> String {
        hex.to_string().repeat(40)
    }

    fn digest(hex: char) -> String {
        hex.to_string().repeat(64)
    }

    fn frozen_spec() -> FrozenAutopilotSpec {
        let explorations = vec![
            RepositoryExploration::new(
                "Memorithm/RSI",
                revision('a'),
                vec![
                    ExplorationObservation::new(
                        "code",
                        ExplorationSource::repository_file("src/lib.rs", digest('b')).unwrap(),
                        "inspected RSI public module boundary",
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
            RepositoryExploration::new(
                "Memorithm/scirust",
                revision('c'),
                vec![
                    ExplorationObservation::new(
                        "code",
                        ExplorationSource::repository_file(
                            "scirust-sciagent/src/lib.rs",
                            digest('d'),
                        )
                        .unwrap(),
                        "inspected SciAgent integration boundary",
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
        ];
        let intake = ExploredObjective::new(
            "deliver AUTOPILOT task DAG",
            "P8.1 froze exact repository scope before decomposition",
            explorations,
        )
        .unwrap()
        .questionnaire(Vec::new())
        .unwrap()
        .resolve(Vec::new())
        .unwrap();
        let criteria = vec![
            AcceptanceCriterion::new(
                "rsi-tests",
                "RSI tests pass",
                AcceptanceCheck::command("rsi", "cargo_test", Vec::new()).unwrap(),
            )
            .unwrap(),
            AcceptanceCriterion::new(
                "sciagent-check",
                "SciAgent compiles",
                AcceptanceCheck::command("scirust", "cargo_check", Vec::new()).unwrap(),
            )
            .unwrap(),
        ];
        intake
            .freeze(AutopilotSpecDraft::new(
                criteria,
                vec!["direct default-branch writes".to_string()],
                SpecBudget::new(20, 20_000, 20_000).unwrap(),
                vec![
                    RepositoryScope::new(
                        "rsi",
                        "Memorithm/RSI",
                        revision('a'),
                        vec!["src".to_string(), "tests".to_string()],
                    )
                    .unwrap(),
                    RepositoryScope::new(
                        "scirust",
                        "Memorithm/scirust",
                        revision('c'),
                        vec!["scirust-sciagent".to_string()],
                    )
                    .unwrap(),
                ],
            ))
            .unwrap()
    }

    fn allowance(role: &str, path: &str) -> TaskEditAllowance {
        TaskEditAllowance::new(
            role,
            vec![path.to_string()],
            vec![TaskOperation::Create, TaskOperation::ModifyExact],
        )
        .unwrap()
    }

    fn task(
        id: &str,
        dependencies: Vec<String>,
        role: &str,
        path: &str,
        criterion: &str,
    ) -> AutopilotTask {
        AutopilotTask::new(AutopilotTaskDraft {
            id: id.to_string(),
            description: format!("execute task {id}"),
            regime: TaskRegime::feature(),
            repository_roles: vec![role.to_string()],
            edit_allowances: vec![allowance(role, path)],
            hard_gate_profile: HardGateProfile::engineering_strict(),
            budget: TaskBudget::new(2, 4, 4_000, 4_000).unwrap(),
            dependencies,
            done_criterion_id: criterion.to_string(),
        })
        .unwrap()
    }

    fn policy() -> TaskDagPolicy {
        TaskDagPolicy::new(8, 4, 16).unwrap()
    }

    #[test]
    fn strict_hard_gate_profile_cannot_omit_base_gates() {
        let gates = HardGateProfile::engineering_strict().required_gates();
        assert_eq!(gates.len(), BASE_HARD_GATES.len());
        for gate in BASE_HARD_GATES {
            assert!(gates.iter().any(|item| item == gate));
        }
    }

    #[test]
    fn task_scope_can_narrow_but_not_widen_frozen_paths() {
        let spec = frozen_spec();
        let valid = task("a", Vec::new(), "rsi", "src/autopilot", "rsi-tests");
        AutopilotTaskDag::new(&spec, vec![valid], policy()).unwrap();

        let widened = task("a", Vec::new(), "rsi", "docs", "rsi-tests");
        assert!(matches!(
            AutopilotTaskDag::new(&spec, vec![widened], policy()),
            Err(AutopilotTaskDagError::PathOutsideFrozenScope { .. })
        ));
    }

    #[test]
    fn done_criterion_must_belong_to_task_repository_subset() {
        let spec = frozen_spec();
        let task = task("a", Vec::new(), "rsi", "src", "sciagent-check");
        assert!(matches!(
            AutopilotTaskDag::new(&spec, vec![task], policy()),
            Err(AutopilotTaskDagError::DoneCriterionOutsideTaskSubset { .. })
        ));
    }

    #[test]
    fn dag_rejects_unknown_dependencies_and_cycles() {
        let spec = frozen_spec();
        let unknown = task(
            "a",
            vec!["missing".to_string()],
            "rsi",
            "src",
            "rsi-tests",
        );
        assert!(matches!(
            AutopilotTaskDag::new(&spec, vec![unknown], policy()),
            Err(AutopilotTaskDagError::UnknownDependency { .. })
        ));

        let first = task(
            "a",
            vec!["b".to_string()],
            "rsi",
            "src",
            "rsi-tests",
        );
        let second = task(
            "b",
            vec!["a".to_string()],
            "rsi",
            "tests",
            "rsi-tests",
        );
        assert!(matches!(
            AutopilotTaskDag::new(&spec, vec![first, second], policy()),
            Err(AutopilotTaskDagError::CycleDetected(_))
        ));
    }

    #[test]
    fn aggregate_task_budget_cannot_exceed_frozen_objective_budget() {
        let spec = frozen_spec();
        let mut first = task("a", Vec::new(), "rsi", "src", "rsi-tests");
        let mut second = task(
            "b",
            vec!["a".to_string()],
            "rsi",
            "tests",
            "rsi-tests",
        );
        first.budget.max_tokens = 15_000;
        second.budget.max_tokens = 15_000;
        assert!(matches!(
            AutopilotTaskDag::new(&spec, vec![first, second], policy()),
            Err(AutopilotTaskDagError::ObjectiveBudgetExceeded {
                field: "tokens",
                ..
            })
        ));
    }

    #[test]
    fn topological_order_and_hash_are_deterministic() {
        let spec = frozen_spec();
        let a = task("a", Vec::new(), "rsi", "src", "rsi-tests");
        let b = task("b", Vec::new(), "scirust", "scirust-sciagent", "sciagent-check");
        let c = task(
            "c",
            vec!["a".to_string(), "b".to_string()],
            "rsi",
            "tests",
            "rsi-tests",
        );
        let first = AutopilotTaskDag::new(
            &spec,
            vec![c.clone(), b.clone(), a.clone()],
            policy(),
        )
        .unwrap();
        let second = AutopilotTaskDag::new(&spec, vec![a, c, b], policy()).unwrap();
        assert_eq!(first.topological_order(), &["a", "b", "c"]);
        assert_eq!(first.dag_sha256(), second.dag_sha256());
        assert_eq!(first.to_json_string(), second.to_json_string());
        first.verify().unwrap();
    }

    #[test]
    fn perf_regime_requires_nonempty_machine_identifier() {
        assert!(TaskRegime::perf("flat-decode-v1").is_ok());
        assert!(matches!(
            TaskRegime::perf(""),
            Err(AutopilotTaskDagError::EmptyField("benchmark_profile_id"))
        ));
    }

    #[test]
    fn read_only_repository_subset_is_allowed() {
        let spec = frozen_spec();
        let task = AutopilotTask::new(AutopilotTaskDraft {
            id: "cross-check".to_string(),
            description: "evaluate RSI while reading exact SciRust state".to_string(),
            regime: TaskRegime::feature(),
            repository_roles: vec!["rsi".to_string(), "scirust".to_string()],
            edit_allowances: vec![allowance("rsi", "src")],
            hard_gate_profile: HardGateProfile::engineering_strict(),
            budget: TaskBudget::new(1, 2, 2_000, 2_000).unwrap(),
            dependencies: Vec::new(),
            done_criterion_id: "rsi-tests".to_string(),
        })
        .unwrap();
        let dag = AutopilotTaskDag::new(&spec, vec![task], policy()).unwrap();
        assert_eq!(dag.tasks()[0].repository_roles(), &["rsi", "scirust"]);
    }
}
