//! P8.1 AUTOPILOT intake and frozen specification contract.
//!
//! This module is deliberately data-only and std-only. It makes the ordering
//! constraint structural: an objective must first become an [`ExploredObjective`],
//! then one grouped questionnaire is resolved, and only then can a frozen
//! [`FrozenAutopilotSpec`] be produced. Repository scope cannot silently expand:
//! every allowed repository must have been explored at the exact immutable
//! revision recorded by the spec.

use crate::json::Json;
use crate::sha256::sha256;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Version of the machine-readable AUTOPILOT specification contract.
pub const AUTOPILOT_SPEC_SCHEMA_VERSION: u64 = 1;

/// One immutable observation produced while exploring a repository.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExplorationSource {
    RepositoryFile {
        path: String,
        content_sha256: String,
    },
    Commit {
        revision: String,
    },
}

impl ExplorationSource {
    pub fn repository_file(
        path: impl Into<String>,
        content_sha256: impl Into<String>,
    ) -> Result<Self, AutopilotIntakeError> {
        let path = path.into();
        let content_sha256 = content_sha256.into();
        validate_relative_path("exploration.path", &path)?;
        validate_digest("exploration.content_sha256", &content_sha256)?;
        Ok(Self::RepositoryFile {
            path,
            content_sha256: content_sha256.to_ascii_lowercase(),
        })
    }

    pub fn commit(revision: impl Into<String>) -> Result<Self, AutopilotIntakeError> {
        let revision = revision.into();
        validate_revision(&revision)?;
        Ok(Self::Commit {
            revision: revision.to_ascii_lowercase(),
        })
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        match self {
            Self::RepositoryFile {
                path,
                content_sha256,
            } => {
                out.set("content_sha256", Json::Str(content_sha256.clone()))
                    .set("kind", Json::Str("repository_file".to_string()))
                    .set("path", Json::Str(path.clone()));
            }
            Self::Commit { revision } => {
                out.set("kind", Json::Str("commit".to_string()))
                    .set("revision", Json::Str(revision.clone()));
            }
        }
        out
    }
}

/// Evidence that the agent inspected code, documentation, tests, build metadata,
/// or history before asking questions about an objective.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExplorationObservation {
    pub category: String,
    pub source: ExplorationSource,
    pub summary: String,
}

impl ExplorationObservation {
    pub fn new(
        category: impl Into<String>,
        source: ExplorationSource,
        summary: impl Into<String>,
    ) -> Result<Self, AutopilotIntakeError> {
        let category = category.into();
        let summary = summary.into();
        validate_text("exploration.category", &category)?;
        validate_text("exploration.summary", &summary)?;
        Ok(Self {
            category,
            source,
            summary,
        })
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set("category", Json::Str(self.category.clone()))
            .set("source", self.source.to_json())
            .set("summary", Json::Str(self.summary.clone()));
        out
    }
}

/// Exploration record for one exact repository revision.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RepositoryExploration {
    pub repository: String,
    pub revision: String,
    pub observations: Vec<ExplorationObservation>,
}

impl RepositoryExploration {
    pub fn new(
        repository: impl Into<String>,
        revision: impl Into<String>,
        mut observations: Vec<ExplorationObservation>,
    ) -> Result<Self, AutopilotIntakeError> {
        let repository = repository.into();
        let revision = revision.into();
        validate_repository(&repository)?;
        validate_revision(&revision)?;
        if observations.is_empty() {
            return Err(AutopilotIntakeError::EmptyField(
                "exploration.observations",
            ));
        }
        observations.sort();
        observations.dedup();
        Ok(Self {
            repository,
            revision: revision.to_ascii_lowercase(),
            observations,
        })
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set(
            "observations",
            Json::Arr(
                self.observations
                    .iter()
                    .map(ExplorationObservation::to_json)
                    .collect(),
            ),
        )
        .set("repository", Json::Str(self.repository.clone()))
        .set("revision", Json::Str(self.revision.clone()));
        out
    }
}

/// Objective after repository exploration, before the single grouped
/// questionnaire is resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExploredObjective {
    objective: String,
    context: String,
    explorations: Vec<RepositoryExploration>,
}

impl ExploredObjective {
    pub fn new(
        objective: impl Into<String>,
        context: impl Into<String>,
        mut explorations: Vec<RepositoryExploration>,
    ) -> Result<Self, AutopilotIntakeError> {
        let objective = objective.into();
        let context = context.into();
        validate_text("objective", &objective)?;
        validate_text("context", &context)?;
        if explorations.is_empty() {
            return Err(AutopilotIntakeError::ExplorationRequired);
        }
        explorations.sort();
        for pair in explorations.windows(2) {
            if pair[0].repository == pair[1].repository {
                return Err(AutopilotIntakeError::DuplicateRepository(
                    pair[0].repository.clone(),
                ));
            }
        }
        Ok(Self {
            objective,
            context,
            explorations,
        })
    }

    /// Build the one grouped battery of questions. Even an objective with no
    /// unresolved ambiguity goes through this state with an empty vector, which
    /// keeps the exploration-before-spec ordering explicit.
    pub fn questionnaire(
        self,
        mut questions: Vec<IntakeQuestion>,
    ) -> Result<IntakeQuestionnaire, AutopilotIntakeError> {
        questions.sort_by(|left, right| left.id.cmp(&right.id));
        for pair in questions.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(AutopilotIntakeError::DuplicateQuestion(pair[0].id.clone()));
            }
        }
        Ok(IntakeQuestionnaire {
            explored: self,
            questions,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntakeQuestion {
    pub id: String,
    pub prompt: String,
}

impl IntakeQuestion {
    pub fn new(
        id: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<Self, AutopilotIntakeError> {
        let id = id.into();
        let prompt = prompt.into();
        validate_identifier("question.id", &id)?;
        validate_text("question.prompt", &prompt)?;
        Ok(Self { id, prompt })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuestionAnswer {
    pub question_id: String,
    pub answer: String,
}

impl QuestionAnswer {
    pub fn new(
        question_id: impl Into<String>,
        answer: impl Into<String>,
    ) -> Result<Self, AutopilotIntakeError> {
        let question_id = question_id.into();
        let answer = answer.into();
        validate_identifier("answer.question_id", &question_id)?;
        validate_text("answer.answer", &answer)?;
        Ok(Self {
            question_id,
            answer,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedQuestion {
    pub id: String,
    pub prompt: String,
    pub answer: String,
}

impl ResolvedQuestion {
    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set("answer", Json::Str(self.answer.clone()))
            .set("id", Json::Str(self.id.clone()))
            .set("prompt", Json::Str(self.prompt.clone()));
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntakeQuestionnaire {
    explored: ExploredObjective,
    questions: Vec<IntakeQuestion>,
}

impl IntakeQuestionnaire {
    pub fn resolve(
        self,
        answers: Vec<QuestionAnswer>,
    ) -> Result<ResolvedIntake, AutopilotIntakeError> {
        let mut answer_by_id = BTreeMap::new();
        for answer in answers {
            if answer_by_id
                .insert(answer.question_id.clone(), answer.answer)
                .is_some()
            {
                return Err(AutopilotIntakeError::DuplicateAnswer(answer.question_id));
            }
        }

        let question_ids: BTreeSet<_> = self.questions.iter().map(|q| q.id.as_str()).collect();
        for answer_id in answer_by_id.keys() {
            if !question_ids.contains(answer_id.as_str()) {
                return Err(AutopilotIntakeError::UnknownQuestion(answer_id.clone()));
            }
        }

        let mut decisions = Vec::with_capacity(self.questions.len());
        for question in self.questions {
            let answer = answer_by_id
                .remove(&question.id)
                .ok_or_else(|| AutopilotIntakeError::UnansweredQuestion(question.id.clone()))?;
            decisions.push(ResolvedQuestion {
                id: question.id,
                prompt: question.prompt,
                answer,
            });
        }

        Ok(ResolvedIntake {
            explored: self.explored,
            decisions,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcceptanceCheck {
    /// Declarative command kind resolved by the trusted P5.2 host. No raw
    /// executable or shell string is carried by the spec.
    Command {
        repository_role: String,
        command_kind: String,
        arguments: Vec<String>,
    },
    FilePresent {
        repository_role: String,
        path: String,
    },
    FileAbsent {
        repository_role: String,
        path: String,
    },
    FileSha256 {
        repository_role: String,
        path: String,
        sha256: String,
    },
}

impl AcceptanceCheck {
    pub fn command(
        repository_role: impl Into<String>,
        command_kind: impl Into<String>,
        arguments: Vec<String>,
    ) -> Result<Self, AutopilotIntakeError> {
        let repository_role = repository_role.into();
        let command_kind = command_kind.into();
        validate_identifier("acceptance.repository_role", &repository_role)?;
        validate_identifier("acceptance.command_kind", &command_kind)?;
        validate_arguments(&arguments)?;
        Ok(Self::Command {
            repository_role,
            command_kind,
            arguments,
        })
    }

    pub fn file_present(
        repository_role: impl Into<String>,
        path: impl Into<String>,
    ) -> Result<Self, AutopilotIntakeError> {
        let repository_role = repository_role.into();
        let path = path.into();
        validate_identifier("acceptance.repository_role", &repository_role)?;
        validate_relative_path("acceptance.path", &path)?;
        Ok(Self::FilePresent {
            repository_role,
            path,
        })
    }

    pub fn file_absent(
        repository_role: impl Into<String>,
        path: impl Into<String>,
    ) -> Result<Self, AutopilotIntakeError> {
        let repository_role = repository_role.into();
        let path = path.into();
        validate_identifier("acceptance.repository_role", &repository_role)?;
        validate_relative_path("acceptance.path", &path)?;
        Ok(Self::FileAbsent {
            repository_role,
            path,
        })
    }

    pub fn file_sha256(
        repository_role: impl Into<String>,
        path: impl Into<String>,
        sha256: impl Into<String>,
    ) -> Result<Self, AutopilotIntakeError> {
        let repository_role = repository_role.into();
        let path = path.into();
        let sha256 = sha256.into();
        validate_identifier("acceptance.repository_role", &repository_role)?;
        validate_relative_path("acceptance.path", &path)?;
        validate_digest("acceptance.sha256", &sha256)?;
        Ok(Self::FileSha256 {
            repository_role,
            path,
            sha256: sha256.to_ascii_lowercase(),
        })
    }

    fn repository_role(&self) -> &str {
        match self {
            Self::Command {
                repository_role, ..
            }
            | Self::FilePresent {
                repository_role, ..
            }
            | Self::FileAbsent {
                repository_role, ..
            }
            | Self::FileSha256 {
                repository_role, ..
            } => repository_role,
        }
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        match self {
            Self::Command {
                repository_role,
                command_kind,
                arguments,
            } => {
                out.set(
                    "arguments",
                    Json::Arr(arguments.iter().cloned().map(Json::Str).collect()),
                )
                .set("command_kind", Json::Str(command_kind.clone()))
                .set("kind", Json::Str("command".to_string()))
                .set("repository_role", Json::Str(repository_role.clone()));
            }
            Self::FilePresent {
                repository_role,
                path,
            } => {
                out.set("kind", Json::Str("file_present".to_string()))
                    .set("path", Json::Str(path.clone()))
                    .set("repository_role", Json::Str(repository_role.clone()));
            }
            Self::FileAbsent {
                repository_role,
                path,
            } => {
                out.set("kind", Json::Str("file_absent".to_string()))
                    .set("path", Json::Str(path.clone()))
                    .set("repository_role", Json::Str(repository_role.clone()));
            }
            Self::FileSha256 {
                repository_role,
                path,
                sha256,
            } => {
                out.set("kind", Json::Str("file_sha256".to_string()))
                    .set("path", Json::Str(path.clone()))
                    .set("repository_role", Json::Str(repository_role.clone()))
                    .set("sha256", Json::Str(sha256.clone()));
            }
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptanceCriterion {
    pub id: String,
    pub description: String,
    pub check: AcceptanceCheck,
}

impl AcceptanceCriterion {
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        check: AcceptanceCheck,
    ) -> Result<Self, AutopilotIntakeError> {
        let id = id.into();
        let description = description.into();
        validate_identifier("acceptance.id", &id)?;
        validate_text("acceptance.description", &description)?;
        Ok(Self {
            id,
            description,
            check,
        })
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set("check", self.check.to_json())
            .set("description", Json::Str(self.description.clone()))
            .set("id", Json::Str(self.id.clone()));
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecBudget {
    pub max_llm_steps: u64,
    pub max_wall_time_ms: u64,
    pub max_tokens: u64,
}

impl SpecBudget {
    pub fn new(
        max_llm_steps: u64,
        max_wall_time_ms: u64,
        max_tokens: u64,
    ) -> Result<Self, AutopilotIntakeError> {
        if max_llm_steps == 0 || max_wall_time_ms == 0 || max_tokens == 0 {
            return Err(AutopilotIntakeError::InvalidBudget);
        }
        Ok(Self {
            max_llm_steps,
            max_wall_time_ms,
            max_tokens,
        })
    }

    fn to_json(self) -> Json {
        let mut out = Json::obj();
        out.set("max_llm_steps", Json::Num(self.max_llm_steps as f64))
            .set("max_tokens", Json::Num(self.max_tokens as f64))
            .set("max_wall_time_ms", Json::Num(self.max_wall_time_ms as f64));
        out
    }
}

/// Exact repository and file-prefix allowlist frozen into one objective spec.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RepositoryScope {
    pub role: String,
    pub repository: String,
    pub revision: String,
    pub allowed_path_prefixes: Vec<String>,
}

impl RepositoryScope {
    pub fn new(
        role: impl Into<String>,
        repository: impl Into<String>,
        revision: impl Into<String>,
        mut allowed_path_prefixes: Vec<String>,
    ) -> Result<Self, AutopilotIntakeError> {
        let role = role.into();
        let repository = repository.into();
        let revision = revision.into();
        validate_identifier("scope.role", &role)?;
        validate_repository(&repository)?;
        validate_revision(&revision)?;
        if allowed_path_prefixes.is_empty() {
            return Err(AutopilotIntakeError::EmptyField(
                "scope.allowed_path_prefixes",
            ));
        }
        for path in &allowed_path_prefixes {
            validate_relative_path("scope.allowed_path_prefixes", path)?;
        }
        allowed_path_prefixes.sort();
        allowed_path_prefixes.dedup();
        Ok(Self {
            role,
            repository,
            revision: revision.to_ascii_lowercase(),
            allowed_path_prefixes,
        })
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
        .set("repository", Json::Str(self.repository.clone()))
        .set("revision", Json::Str(self.revision.clone()))
        .set("role", Json::Str(self.role.clone()));
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutopilotSpecDraft {
    pub acceptance_criteria: Vec<AcceptanceCriterion>,
    pub out_of_scope: Vec<String>,
    pub budget: SpecBudget,
    pub repository_scope: Vec<RepositoryScope>,
}

impl AutopilotSpecDraft {
    pub fn new(
        acceptance_criteria: Vec<AcceptanceCriterion>,
        out_of_scope: Vec<String>,
        budget: SpecBudget,
        repository_scope: Vec<RepositoryScope>,
    ) -> Self {
        Self {
            acceptance_criteria,
            out_of_scope,
            budget,
            repository_scope,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedIntake {
    explored: ExploredObjective,
    decisions: Vec<ResolvedQuestion>,
}

impl ResolvedIntake {
    pub fn freeze(
        self,
        mut draft: AutopilotSpecDraft,
    ) -> Result<FrozenAutopilotSpec, AutopilotIntakeError> {
        if draft.acceptance_criteria.is_empty() {
            return Err(AutopilotIntakeError::EmptyField("acceptance_criteria"));
        }
        if draft.repository_scope.is_empty() {
            return Err(AutopilotIntakeError::EmptyField("repository_scope"));
        }

        draft
            .acceptance_criteria
            .sort_by(|left, right| left.id.cmp(&right.id));
        for pair in draft.acceptance_criteria.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(AutopilotIntakeError::DuplicateCriterion(pair[0].id.clone()));
            }
        }

        for item in &draft.out_of_scope {
            validate_text("out_of_scope", item)?;
        }
        draft.out_of_scope.sort();
        draft.out_of_scope.dedup();

        draft.repository_scope.sort();
        let mut roles = BTreeSet::new();
        for scope in &draft.repository_scope {
            if !roles.insert(scope.role.as_str()) {
                return Err(AutopilotIntakeError::DuplicateRepositoryRole(
                    scope.role.clone(),
                ));
            }
            let explored = self.explored.explorations.iter().any(|item| {
                item.repository == scope.repository && item.revision == scope.revision
            });
            if !explored {
                return Err(AutopilotIntakeError::UnexploredRepositoryRevision {
                    repository: scope.repository.clone(),
                    revision: scope.revision.clone(),
                });
            }
        }

        for criterion in &draft.acceptance_criteria {
            if !roles.contains(criterion.check.repository_role()) {
                return Err(AutopilotIntakeError::CriterionOutsideScope {
                    criterion_id: criterion.id.clone(),
                    repository_role: criterion.check.repository_role().to_string(),
                });
            }
        }

        let mut spec = FrozenAutopilotSpec {
            schema_version: AUTOPILOT_SPEC_SCHEMA_VERSION,
            objective: self.explored.objective,
            context: self.explored.context,
            explorations: self.explored.explorations,
            decisions: self.decisions,
            acceptance_criteria: draft.acceptance_criteria,
            out_of_scope: draft.out_of_scope,
            budget: draft.budget,
            repository_scope: draft.repository_scope,
            spec_sha256: String::new(),
        };
        spec.spec_sha256 = spec.compute_sha256();
        Ok(spec)
    }
}

/// Immutable, deterministic specification used as the root contract for P8.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenAutopilotSpec {
    schema_version: u64,
    objective: String,
    context: String,
    explorations: Vec<RepositoryExploration>,
    decisions: Vec<ResolvedQuestion>,
    acceptance_criteria: Vec<AcceptanceCriterion>,
    out_of_scope: Vec<String>,
    budget: SpecBudget,
    repository_scope: Vec<RepositoryScope>,
    spec_sha256: String,
}

impl FrozenAutopilotSpec {
    pub fn schema_version(&self) -> u64 {
        self.schema_version
    }

    pub fn objective(&self) -> &str {
        &self.objective
    }

    pub fn context(&self) -> &str {
        &self.context
    }

    pub fn explorations(&self) -> &[RepositoryExploration] {
        &self.explorations
    }

    pub fn decisions(&self) -> &[ResolvedQuestion] {
        &self.decisions
    }

    pub fn acceptance_criteria(&self) -> &[AcceptanceCriterion] {
        &self.acceptance_criteria
    }

    pub fn out_of_scope(&self) -> &[String] {
        &self.out_of_scope
    }

    pub fn budget(&self) -> SpecBudget {
        self.budget
    }

    pub fn repository_scope(&self) -> &[RepositoryScope] {
        &self.repository_scope
    }

    pub fn spec_sha256(&self) -> &str {
        &self.spec_sha256
    }

    /// Canonical compact JSON. `Json::Obj` uses `BTreeMap`, so field ordering is
    /// stable and the arrays have already been canonicalized by constructors.
    pub fn to_json_string(&self) -> String {
        let mut root = self.unsigned_json();
        root.set("spec_sha256", Json::Str(self.spec_sha256.clone()));
        root.to_string()
    }

    pub fn verify(&self) -> Result<(), AutopilotIntakeError> {
        let actual = self.compute_sha256();
        if actual != self.spec_sha256 {
            return Err(AutopilotIntakeError::FrozenSpecHashMismatch {
                expected: self.spec_sha256.clone(),
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
        root.set(
            "acceptance_criteria",
            Json::Arr(
                self.acceptance_criteria
                    .iter()
                    .map(AcceptanceCriterion::to_json)
                    .collect(),
            ),
        )
        .set("budget", self.budget.to_json())
        .set("context", Json::Str(self.context.clone()))
        .set(
            "decisions",
            Json::Arr(self.decisions.iter().map(ResolvedQuestion::to_json).collect()),
        )
        .set(
            "explorations",
            Json::Arr(
                self.explorations
                    .iter()
                    .map(RepositoryExploration::to_json)
                    .collect(),
            ),
        )
        .set("objective", Json::Str(self.objective.clone()))
        .set(
            "out_of_scope",
            Json::Arr(self.out_of_scope.iter().cloned().map(Json::Str).collect()),
        )
        .set(
            "repository_scope",
            Json::Arr(
                self.repository_scope
                    .iter()
                    .map(RepositoryScope::to_json)
                    .collect(),
            ),
        )
        .set("schema_version", Json::Num(self.schema_version as f64));
        root
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutopilotIntakeError {
    EmptyField(&'static str),
    InvalidText {
        field: &'static str,
        value: String,
    },
    InvalidIdentifier {
        field: &'static str,
        value: String,
    },
    InvalidRepository(String),
    InvalidRevision(String),
    InvalidDigest {
        field: &'static str,
        value: String,
    },
    InvalidPath {
        field: &'static str,
        value: String,
    },
    InvalidArgument(String),
    InvalidBudget,
    ExplorationRequired,
    DuplicateRepository(String),
    DuplicateRepositoryRole(String),
    DuplicateQuestion(String),
    DuplicateAnswer(String),
    UnknownQuestion(String),
    UnansweredQuestion(String),
    DuplicateCriterion(String),
    UnexploredRepositoryRevision {
        repository: String,
        revision: String,
    },
    CriterionOutsideScope {
        criterion_id: String,
        repository_role: String,
    },
    FrozenSpecHashMismatch {
        expected: String,
        actual: String,
    },
}

impl fmt::Display for AutopilotIntakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "AUTOPILOT field '{field}' is empty"),
            Self::InvalidText { field, value } => {
                write!(f, "invalid AUTOPILOT text field '{field}': '{value}'")
            }
            Self::InvalidIdentifier { field, value } => {
                write!(f, "invalid AUTOPILOT identifier '{field}': '{value}'")
            }
            Self::InvalidRepository(repository) => {
                write!(f, "invalid AUTOPILOT repository '{repository}'")
            }
            Self::InvalidRevision(revision) => {
                write!(f, "invalid immutable git revision '{revision}'")
            }
            Self::InvalidDigest { field, value } => {
                write!(f, "invalid SHA-256 field '{field}': '{value}'")
            }
            Self::InvalidPath { field, value } => {
                write!(f, "invalid workspace-relative path '{field}': '{value}'")
            }
            Self::InvalidArgument(value) => write!(f, "invalid declarative argument '{value}'"),
            Self::InvalidBudget => write!(f, "AUTOPILOT budget values must be non-zero"),
            Self::ExplorationRequired => {
                write!(f, "repository exploration is required before AUTOPILOT intake")
            }
            Self::DuplicateRepository(repository) => {
                write!(f, "repository explored more than once: '{repository}'")
            }
            Self::DuplicateRepositoryRole(role) => write!(f, "duplicate repository role '{role}'"),
            Self::DuplicateQuestion(id) => write!(f, "duplicate intake question '{id}'"),
            Self::DuplicateAnswer(id) => write!(f, "duplicate intake answer '{id}'"),
            Self::UnknownQuestion(id) => write!(f, "answer references unknown question '{id}'"),
            Self::UnansweredQuestion(id) => write!(f, "intake question '{id}' is unanswered"),
            Self::DuplicateCriterion(id) => write!(f, "duplicate acceptance criterion '{id}'"),
            Self::UnexploredRepositoryRevision {
                repository,
                revision,
            } => write!(
                f,
                "spec scope references unexplored repository revision {repository}@{revision}"
            ),
            Self::CriterionOutsideScope {
                criterion_id,
                repository_role,
            } => write!(
                f,
                "criterion '{criterion_id}' references repository role '{repository_role}' outside the frozen scope"
            ),
            Self::FrozenSpecHashMismatch { expected, actual } => write!(
                f,
                "frozen AUTOPILOT spec hash mismatch: expected {expected}, actual {actual}"
            ),
        }
    }
}

impl std::error::Error for AutopilotIntakeError {}

fn validate_text(field: &'static str, value: &str) -> Result<(), AutopilotIntakeError> {
    if value.is_empty() {
        return Err(AutopilotIntakeError::EmptyField(field));
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(AutopilotIntakeError::InvalidText {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), AutopilotIntakeError> {
    validate_text(field, value)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(AutopilotIntakeError::InvalidIdentifier {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_repository(repository: &str) -> Result<(), AutopilotIntakeError> {
    if repository.trim() != repository || repository.chars().any(char::is_whitespace) {
        return Err(AutopilotIntakeError::InvalidRepository(
            repository.to_string(),
        ));
    }
    let mut parts = repository.split('/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    if owner.is_empty() || name.is_empty() || parts.next().is_some() {
        return Err(AutopilotIntakeError::InvalidRepository(
            repository.to_string(),
        ));
    }
    Ok(())
}

fn validate_revision(revision: &str) -> Result<(), AutopilotIntakeError> {
    if !matches!(revision.len(), 40 | 64)
        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(AutopilotIntakeError::InvalidRevision(revision.to_string()));
    }
    Ok(())
}

fn validate_digest(field: &'static str, value: &str) -> Result<(), AutopilotIntakeError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(AutopilotIntakeError::InvalidDigest {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_relative_path(field: &'static str, value: &str) -> Result<(), AutopilotIntakeError> {
    let invalid = value.is_empty()
        || value.starts_with('/')
        || value.ends_with('/')
        || value.contains('\\')
        || value.contains('\0')
        || value.split('/').any(|part| part.is_empty() || matches!(part, "." | ".."));
    if invalid {
        return Err(AutopilotIntakeError::InvalidPath {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_arguments(arguments: &[String]) -> Result<(), AutopilotIntakeError> {
    for argument in arguments {
        if argument.contains('\0') || argument.chars().any(char::is_control) {
            return Err(AutopilotIntakeError::InvalidArgument(argument.clone()));
        }
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

    fn digest(hex: char) -> String {
        hex.to_string().repeat(64)
    }

    fn revision(hex: char) -> String {
        hex.to_string().repeat(40)
    }

    fn exploration(repository: &str, rev: char, path: &str) -> RepositoryExploration {
        RepositoryExploration::new(
            repository,
            revision(rev),
            vec![
                ExplorationObservation::new(
                    "code",
                    ExplorationSource::repository_file(path, digest('a')).unwrap(),
                    "inspected the implementation boundary",
                )
                .unwrap(),
                ExplorationObservation::new(
                    "history",
                    ExplorationSource::commit(revision(rev)).unwrap(),
                    "inspected the immutable baseline commit",
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    fn resolved(explorations: Vec<RepositoryExploration>) -> ResolvedIntake {
        ExploredObjective::new(
            "implement bounded AUTOPILOT intake",
            "the repository already has cumulative PatchSets and COGNO hard gates",
            explorations,
        )
        .unwrap()
        .questionnaire(vec![
            IntakeQuestion::new("scope", "Should generated specs stay inside RSI?").unwrap(),
        ])
        .unwrap()
        .resolve(vec![QuestionAnswer::new("scope", "yes, RSI only for this slice").unwrap()])
        .unwrap()
    }

    fn draft(rev: char) -> AutopilotSpecDraft {
        let scope = RepositoryScope::new(
            "rsi",
            "Memorithm/RSI",
            revision(rev),
            vec!["src/autopilot_intake.rs".to_string(), "src/lib.rs".to_string()],
        )
        .unwrap();
        let criterion = AcceptanceCriterion::new(
            "unit-tests",
            "the default RSI test suite passes",
            AcceptanceCheck::command("rsi", "cargo_test", vec!["--lib".to_string()]).unwrap(),
        )
        .unwrap();
        AutopilotSpecDraft::new(
            vec![criterion],
            vec!["automatic merge".to_string(), "direct main writes".to_string()],
            SpecBudget::new(32, 3_600_000, 200_000).unwrap(),
            vec![scope],
        )
    }

    #[test]
    fn exploration_is_required_before_questionnaire_or_spec() {
        let error = ExploredObjective::new("objective", "context", Vec::new()).unwrap_err();
        assert_eq!(error, AutopilotIntakeError::ExplorationRequired);
    }

    #[test]
    fn grouped_questionnaire_fails_closed_on_missing_or_unknown_answers() {
        let questionnaire = ExploredObjective::new(
            "objective",
            "context",
            vec![exploration("Memorithm/RSI", 'a', "src/lib.rs")],
        )
        .unwrap()
        .questionnaire(vec![IntakeQuestion::new("q1", "Choose the scope").unwrap()])
        .unwrap();
        assert!(matches!(
            questionnaire.clone().resolve(Vec::new()),
            Err(AutopilotIntakeError::UnansweredQuestion(id)) if id == "q1"
        ));
        assert!(matches!(
            questionnaire.resolve(vec![QuestionAnswer::new("other", "answer").unwrap()]),
            Err(AutopilotIntakeError::UnknownQuestion(id)) if id == "other"
        ));
    }

    #[test]
    fn frozen_scope_must_have_been_explored_at_exact_revision() {
        let error = resolved(vec![exploration("Memorithm/RSI", 'a', "src/lib.rs")])
            .freeze(draft('b'))
            .unwrap_err();
        assert!(matches!(
            error,
            AutopilotIntakeError::UnexploredRepositoryRevision { .. }
        ));
    }

    #[test]
    fn acceptance_criterion_cannot_reference_unlisted_repository_role() {
        let mut value = draft('a');
        value.acceptance_criteria = vec![
            AcceptanceCriterion::new(
                "foreign",
                "must not silently expand scope",
                AcceptanceCheck::file_present("scirust", "Cargo.toml").unwrap(),
            )
            .unwrap(),
        ];
        let error = resolved(vec![exploration("Memorithm/RSI", 'a', "src/lib.rs")])
            .freeze(value)
            .unwrap_err();
        assert!(matches!(
            error,
            AutopilotIntakeError::CriterionOutsideScope { .. }
        ));
    }

    #[test]
    fn canonical_spec_identity_is_input_order_independent() {
        let explorations_a = vec![
            exploration("Memorithm/scirust", 'b', "scirust-sciagent/src/lib.rs"),
            exploration("Memorithm/RSI", 'a', "src/lib.rs"),
        ];
        let explorations_b = explorations_a.iter().cloned().rev().collect();

        let make_draft = || {
            let scopes = vec![
                RepositoryScope::new(
                    "scirust",
                    "Memorithm/scirust",
                    revision('b'),
                    vec!["scirust-sciagent/src/lib.rs".to_string()],
                )
                .unwrap(),
                RepositoryScope::new(
                    "rsi",
                    "Memorithm/RSI",
                    revision('a'),
                    vec!["src/lib.rs".to_string()],
                )
                .unwrap(),
            ];
            let criteria = vec![
                AcceptanceCriterion::new(
                    "b",
                    "SciRust compiles",
                    AcceptanceCheck::command("scirust", "cargo_check", Vec::new()).unwrap(),
                )
                .unwrap(),
                AcceptanceCriterion::new(
                    "a",
                    "RSI compiles",
                    AcceptanceCheck::command("rsi", "cargo_check", Vec::new()).unwrap(),
                )
                .unwrap(),
            ];
            AutopilotSpecDraft::new(
                criteria,
                vec!["scope creep".to_string(), "direct merge".to_string()],
                SpecBudget::new(16, 60_000, 50_000).unwrap(),
                scopes,
            )
        };

        let first = resolved(explorations_a).freeze(make_draft()).unwrap();
        let mut reversed = make_draft();
        reversed.acceptance_criteria.reverse();
        reversed.repository_scope.reverse();
        reversed.out_of_scope.reverse();
        let second = resolved(explorations_b).freeze(reversed).unwrap();

        assert_eq!(first.spec_sha256(), second.spec_sha256());
        assert_eq!(first.to_json_string(), second.to_json_string());
        first.verify().unwrap();
    }

    #[test]
    fn frozen_hash_detects_tampering() {
        let mut spec = resolved(vec![exploration("Memorithm/RSI", 'a', "src/lib.rs")])
            .freeze(draft('a'))
            .unwrap();
        spec.context.push_str(" modified after freeze");
        assert!(matches!(
            spec.verify(),
            Err(AutopilotIntakeError::FrozenSpecHashMismatch { .. })
        ));
    }

    #[test]
    fn traversal_and_noncanonical_paths_are_rejected() {
        for path in ["../src/lib.rs", "/tmp/file", "src//lib.rs", "./src/lib.rs"] {
            assert!(matches!(
                RepositoryScope::new("rsi", "Memorithm/RSI", revision('a'), vec![path.to_string()]),
                Err(AutopilotIntakeError::InvalidPath { .. })
            ));
        }
    }
}
