//! P8.4 AUTOPILOT PERF regime: immutable benchmark plus anti-noise promotion.
//!
//! Performance is considered only after the COGNO engineering hard gates pass.
//! The benchmark definition, target environment, benchmark source bytes and
//! anti-noise policy are frozen before candidate search. Supporting
//! microbenchmarks can produce evidence but can never be promotion gates; at
//! least one end-to-end case must independently prove the configured gain.

use crate::autopilot_intake::FrozenAutopilotSpec;
use crate::autopilot_task_dag::{AutopilotTask, AutopilotTaskDag, TaskOperation, TaskRegime};
use crate::engineering_trajectory::AdmissibilityBreakdown;
use crate::json::Json;
use crate::patchset::{FileOperation, PatchSet, PatchSetError};
use crate::sha256::sha256;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const PERF_BENCHMARK_SCHEMA_VERSION: u64 = 1;
const MAX_TOTAL_SAMPLES_PER_COMPARISON: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerfBenchmarkApproval {
    pub authority: String,
    pub evidence_sha256: String,
}

impl PerfBenchmarkApproval {
    pub fn new(
        authority: impl Into<String>,
        evidence_sha256: impl Into<String>,
    ) -> Result<Self, PerfRegimeError> {
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

/// Exact machine/software identity used for paired baseline/candidate evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkEnvironment {
    pub hardware_sha256: String,
    pub software_sha256: String,
}

impl BenchmarkEnvironment {
    pub fn new(
        hardware_sha256: impl Into<String>,
        software_sha256: impl Into<String>,
    ) -> Result<Self, PerfRegimeError> {
        let hardware_sha256 = hardware_sha256.into();
        let software_sha256 = software_sha256.into();
        validate_digest("environment.hardware_sha256", &hardware_sha256)?;
        validate_digest("environment.software_sha256", &software_sha256)?;
        Ok(Self {
            hardware_sha256: hardware_sha256.to_ascii_lowercase(),
            software_sha256: software_sha256.to_ascii_lowercase(),
        })
    }

    pub fn fingerprint(&self) -> String {
        let mut value = String::from("rsi-perf-environment-v1|");
        value.push_str(&self.hardware_sha256);
        value.push('|');
        value.push_str(&self.software_sha256);
        hex_digest(&sha256(value.as_bytes()))
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set(
            "hardware_sha256",
            Json::Str(self.hardware_sha256.clone()),
        )
        .set(
            "software_sha256",
            Json::Str(self.software_sha256.clone()),
        );
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FrozenBenchmarkArtifact {
    pub repository_role: String,
    pub path: String,
    pub sha256: String,
}

impl FrozenBenchmarkArtifact {
    pub fn new(
        repository_role: impl Into<String>,
        path: impl Into<String>,
        sha256: impl Into<String>,
    ) -> Result<Self, PerfRegimeError> {
        let repository_role = repository_role.into();
        let path = path.into();
        let sha256 = sha256.into();
        validate_identifier("benchmark_artifact.repository_role", &repository_role)?;
        validate_relative_path("benchmark_artifact.path", &path)?;
        validate_digest("benchmark_artifact.sha256", &sha256)?;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDirection {
    Minimize,
    Maximize,
}

impl MetricDirection {
    fn as_str(self) -> &'static str {
        match self {
            Self::Minimize => "minimize",
            Self::Maximize => "maximize",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkClass {
    EndToEnd,
    SupportingMicrobenchmark,
}

impl BenchmarkClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::EndToEnd => "end_to_end",
            Self::SupportingMicrobenchmark => "supporting_microbenchmark",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkCase {
    pub id: String,
    pub repository_role: String,
    pub command_kind: String,
    pub arguments: Vec<String>,
    pub metric: String,
    pub unit: String,
    pub direction: MetricDirection,
    pub class: BenchmarkClass,
    pub promotion_gate: bool,
}

impl BenchmarkCase {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        repository_role: impl Into<String>,
        command_kind: impl Into<String>,
        arguments: Vec<String>,
        metric: impl Into<String>,
        unit: impl Into<String>,
        direction: MetricDirection,
        class: BenchmarkClass,
        promotion_gate: bool,
    ) -> Result<Self, PerfRegimeError> {
        let value = Self {
            id: id.into(),
            repository_role: repository_role.into(),
            command_kind: command_kind.into(),
            arguments,
            metric: metric.into(),
            unit: unit.into(),
            direction,
            class,
            promotion_gate,
        };
        validate_identifier("benchmark_case.id", &value.id)?;
        validate_identifier("benchmark_case.repository_role", &value.repository_role)?;
        validate_identifier("benchmark_case.command_kind", &value.command_kind)?;
        validate_text("benchmark_case.metric", &value.metric)?;
        validate_text("benchmark_case.unit", &value.unit)?;
        validate_arguments(&value.arguments)?;
        if promotion_gate && class != BenchmarkClass::EndToEnd {
            return Err(PerfRegimeError::MicrobenchmarkCannotPromote(value.id));
        }
        Ok(value)
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set(
            "arguments",
            Json::Arr(self.arguments.iter().cloned().map(Json::Str).collect()),
        )
        .set("class", Json::Str(self.class.as_str().to_string()))
        .set("command_kind", Json::Str(self.command_kind.clone()))
        .set("direction", Json::Str(self.direction.as_str().to_string()))
        .set("id", Json::Str(self.id.clone()))
        .set("metric", Json::Str(self.metric.clone()))
        .set("promotion_gate", Json::Bool(self.promotion_gate))
        .set("repository_role", Json::Str(self.repository_role.clone()))
        .set("unit", Json::Str(self.unit.clone()));
        out
    }
}

/// Frozen repeated-measurement policy. Exact sample/batch counts prevent
/// cherry-picking extra runs after seeing candidate results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AntiNoisePolicy {
    pub samples_per_batch: usize,
    pub batches: usize,
    pub max_relative_mad_ppm: u64,
    pub min_improvement_ppm: u64,
    pub min_winning_batches: usize,
}

impl AntiNoisePolicy {
    pub fn new(
        samples_per_batch: usize,
        batches: usize,
        max_relative_mad_ppm: u64,
        min_improvement_ppm: u64,
        min_winning_batches: usize,
    ) -> Result<Self, PerfRegimeError> {
        if samples_per_batch < 3 || batches < 2 {
            return Err(PerfRegimeError::InsufficientRepetition);
        }
        if max_relative_mad_ppm == 0 || min_improvement_ppm == 0 {
            return Err(PerfRegimeError::InvalidAntiNoisePolicy);
        }
        if min_winning_batches == 0 || min_winning_batches > batches {
            return Err(PerfRegimeError::InvalidAntiNoisePolicy);
        }
        let total = samples_per_batch
            .checked_mul(batches)
            .ok_or(PerfRegimeError::SampleBudgetOverflow)?;
        if total > MAX_TOTAL_SAMPLES_PER_COMPARISON {
            return Err(PerfRegimeError::SampleBudgetExceeded {
                actual: total,
                maximum: MAX_TOTAL_SAMPLES_PER_COMPARISON,
            });
        }
        Ok(Self {
            samples_per_batch,
            batches,
            max_relative_mad_ppm,
            min_improvement_ppm,
            min_winning_batches,
        })
    }

    fn to_json(self) -> Json {
        let mut out = Json::obj();
        out.set("batches", Json::Num(self.batches as f64))
            .set(
                "max_relative_mad_ppm",
                Json::Num(self.max_relative_mad_ppm as f64),
            )
            .set(
                "min_improvement_ppm",
                Json::Num(self.min_improvement_ppm as f64),
            )
            .set(
                "min_winning_batches",
                Json::Num(self.min_winning_batches as f64),
            )
            .set(
                "samples_per_batch",
                Json::Num(self.samples_per_batch as f64),
            );
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenPerfBenchmark {
    schema_version: u64,
    spec_sha256: String,
    dag_sha256: String,
    task_id: String,
    profile_id: String,
    approval: PerfBenchmarkApproval,
    environment: BenchmarkEnvironment,
    policy: AntiNoisePolicy,
    cases: Vec<BenchmarkCase>,
    artifacts: Vec<FrozenBenchmarkArtifact>,
    profile_sha256: String,
}

impl FrozenPerfBenchmark {
    #[allow(clippy::too_many_arguments)]
    pub fn freeze(
        spec: &FrozenAutopilotSpec,
        dag: &AutopilotTaskDag,
        task_id: impl Into<String>,
        approval: PerfBenchmarkApproval,
        environment: BenchmarkEnvironment,
        policy: AntiNoisePolicy,
        mut cases: Vec<BenchmarkCase>,
        mut artifacts: Vec<FrozenBenchmarkArtifact>,
    ) -> Result<Self, PerfRegimeError> {
        verify_spec_and_dag(spec, dag)?;
        let task_id = task_id.into();
        validate_identifier("task_id", &task_id)?;
        let task = dag
            .task(&task_id)
            .ok_or_else(|| PerfRegimeError::UnknownTask(task_id.clone()))?;
        let profile_id = match task.regime() {
            TaskRegime::Perf {
                benchmark_profile_id,
            } => benchmark_profile_id.clone(),
            TaskRegime::Feature => return Err(PerfRegimeError::TaskIsNotPerf(task_id)),
        };
        if cases.is_empty() {
            return Err(PerfRegimeError::EmptyField("benchmark_cases"));
        }
        if artifacts.is_empty() {
            return Err(PerfRegimeError::EmptyField("benchmark_artifacts"));
        }

        cases.sort_by(|left, right| left.id.cmp(&right.id));
        for pair in cases.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(PerfRegimeError::DuplicateBenchmarkCase(pair[0].id.clone()));
            }
        }
        if !cases.iter().any(|case| case.promotion_gate) {
            return Err(PerfRegimeError::NoEndToEndPromotionGate);
        }
        for case in &cases {
            if !task
                .repository_roles()
                .iter()
                .any(|role| role == &case.repository_role)
            {
                return Err(PerfRegimeError::BenchmarkCaseOutsideTaskRepositories {
                    case_id: case.id.clone(),
                    repository_role: case.repository_role.clone(),
                });
            }
        }

        artifacts.sort();
        for pair in artifacts.windows(2) {
            if pair[0].repository_role == pair[1].repository_role && pair[0].path == pair[1].path {
                return Err(PerfRegimeError::DuplicateBenchmarkArtifact {
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
                return Err(PerfRegimeError::BenchmarkArtifactOutsideTaskRepositories {
                    repository_role: artifact.repository_role.clone(),
                    path: artifact.path.clone(),
                });
            }
            if !path_in_frozen_spec(spec, &artifact.repository_role, &artifact.path) {
                return Err(PerfRegimeError::BenchmarkArtifactOutsideFrozenSpec {
                    repository_role: artifact.repository_role.clone(),
                    path: artifact.path.clone(),
                });
            }
            for allowance in task
                .edit_allowances()
                .iter()
                .filter(|allowance| allowance.repository_role() == artifact.repository_role)
            {
                for prefix in allowance.allowed_path_prefixes() {
                    if path_is_within(&artifact.path, prefix) {
                        return Err(PerfRegimeError::TaskAllowanceTouchesFrozenBenchmark {
                            repository_role: artifact.repository_role.clone(),
                            frozen_path: artifact.path.clone(),
                            allowed_prefix: prefix.clone(),
                        });
                    }
                }
            }
        }

        let mut profile = Self {
            schema_version: PERF_BENCHMARK_SCHEMA_VERSION,
            spec_sha256: spec.spec_sha256().to_string(),
            dag_sha256: dag.dag_sha256().to_string(),
            task_id: task.id().to_string(),
            profile_id,
            approval,
            environment,
            policy,
            cases,
            artifacts,
            profile_sha256: String::new(),
        };
        profile.profile_sha256 = profile.compute_sha256();
        Ok(profile)
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn policy(&self) -> AntiNoisePolicy {
        self.policy
    }

    pub fn cases(&self) -> &[BenchmarkCase] {
        &self.cases
    }

    pub fn artifacts(&self) -> &[FrozenBenchmarkArtifact] {
        &self.artifacts
    }

    pub fn environment_fingerprint(&self) -> String {
        self.environment.fingerprint()
    }

    pub fn profile_sha256(&self) -> &str {
        &self.profile_sha256
    }

    pub fn verify(&self) -> Result<(), PerfRegimeError> {
        let actual = self.compute_sha256();
        if actual != self.profile_sha256 {
            return Err(PerfRegimeError::FrozenProfileHashMismatch {
                expected: self.profile_sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    /// Prevent candidate-controlled benchmark rewriting before evaluation.
    pub fn validate_patchset(
        &self,
        dag: &AutopilotTaskDag,
        repository_role: &str,
        patch_set: &PatchSet,
    ) -> Result<(), PerfRegimeError> {
        self.verify()?;
        dag.verify()
            .map_err(|error| PerfRegimeError::InvalidDag(error.to_string()))?;
        if dag.dag_sha256() != self.dag_sha256 {
            return Err(PerfRegimeError::ProfileContextMismatch);
        }
        patch_set.validate().map_err(PerfRegimeError::PatchSet)?;
        let task = dag
            .task(&self.task_id)
            .ok_or_else(|| PerfRegimeError::UnknownTask(self.task_id.clone()))?;
        if !task
            .repository_roles()
            .iter()
            .any(|role| role == repository_role)
        {
            return Err(PerfRegimeError::RepositoryOutsidePerfTask(
                repository_role.to_string(),
            ));
        }
        let allowances: Vec<_> = task
            .edit_allowances()
            .iter()
            .filter(|allowance| allowance.repository_role() == repository_role)
            .collect();
        if allowances.is_empty() {
            return Err(PerfRegimeError::RepositoryIsReadOnly(
                repository_role.to_string(),
            ));
        }
        let frozen_paths: BTreeSet<_> = self
            .artifacts
            .iter()
            .filter(|artifact| artifact.repository_role == repository_role)
            .map(|artifact| artifact.path.as_str())
            .collect();
        for operation in patch_set.operations() {
            if frozen_paths.contains(operation.path()) {
                return Err(PerfRegimeError::CandidateTouchesFrozenBenchmark {
                    repository_role: repository_role.to_string(),
                    path: operation.path().to_string(),
                });
            }
            let required = patch_operation_kind(operation);
            let allowed = allowances.iter().any(|allowance| {
                allowance.operations().contains(&required)
                    && allowance
                        .allowed_path_prefixes()
                        .iter()
                        .any(|prefix| path_is_within(operation.path(), prefix))
            });
            if !allowed {
                return Err(PerfRegimeError::CandidateOperationOutsideAllowance {
                    repository_role: repository_role.to_string(),
                    path: operation.path().to_string(),
                    operation: required,
                });
            }
        }
        Ok(())
    }

    pub fn to_json_string(&self) -> String {
        let mut root = self.unsigned_json();
        root.set("profile_sha256", Json::Str(self.profile_sha256.clone()));
        root.to_string()
    }

    fn compute_sha256(&self) -> String {
        hex_digest(&sha256(self.unsigned_json().to_string().as_bytes()))
    }

    fn unsigned_json(&self) -> Json {
        let mut root = Json::obj();
        root.set("approval", self.approval.to_json())
            .set(
                "artifacts",
                Json::Arr(
                    self.artifacts
                        .iter()
                        .map(FrozenBenchmarkArtifact::to_json)
                        .collect(),
                ),
            )
            .set(
                "cases",
                Json::Arr(self.cases.iter().map(BenchmarkCase::to_json).collect()),
            )
            .set("dag_sha256", Json::Str(self.dag_sha256.clone()))
            .set("environment", self.environment.to_json())
            .set("policy", self.policy.to_json())
            .set("profile_id", Json::Str(self.profile_id.clone()))
            .set("schema_version", Json::Num(self.schema_version as f64))
            .set("spec_sha256", Json::Str(self.spec_sha256.clone()))
            .set("task_id", Json::Str(self.task_id.clone()));
        root
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerfMeasurementBatch {
    pub case_id: String,
    pub batch_id: String,
    pub environment_fingerprint: String,
    pub samples: Vec<f64>,
}

impl PerfMeasurementBatch {
    pub fn new(
        case_id: impl Into<String>,
        batch_id: impl Into<String>,
        environment_fingerprint: impl Into<String>,
        samples: Vec<f64>,
    ) -> Result<Self, PerfRegimeError> {
        let value = Self {
            case_id: case_id.into(),
            batch_id: batch_id.into(),
            environment_fingerprint: environment_fingerprint.into(),
            samples,
        };
        validate_identifier("measurement.case_id", &value.case_id)?;
        validate_identifier("measurement.batch_id", &value.batch_id)?;
        validate_digest(
            "measurement.environment_fingerprint",
            &value.environment_fingerprint,
        )?;
        if value.samples.is_empty()
            || value
                .samples
                .iter()
                .any(|sample| !sample.is_finite() || *sample <= 0.0)
        {
            return Err(PerfRegimeError::InvalidSamples {
                case_id: value.case_id.clone(),
                batch_id: value.batch_id.clone(),
            });
        }
        Ok(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerfCaseResult {
    pub case_id: String,
    pub class: BenchmarkClass,
    pub promotion_gate: bool,
    pub baseline_median: f64,
    pub candidate_median: f64,
    pub improvement_ppm: i64,
    pub max_relative_mad_ppm: u64,
    pub winning_batches: usize,
    pub required_winning_batches: usize,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerfComparisonReport {
    pub hard_gates_passed: bool,
    pub promotable: bool,
    pub cases: Vec<PerfCaseResult>,
}

impl PerfComparisonReport {
    /// Compare exact paired evidence. Inadmissible candidates are never ranked:
    /// they return a negative report with no performance case results.
    pub fn evaluate(
        profile: &FrozenPerfBenchmark,
        admissibility: &AdmissibilityBreakdown,
        baseline: &[PerfMeasurementBatch],
        candidate: &[PerfMeasurementBatch],
    ) -> Result<Self, PerfRegimeError> {
        profile.verify()?;
        if !admissibility.is_admissible() {
            return Ok(Self {
                hard_gates_passed: false,
                promotable: false,
                cases: Vec::new(),
            });
        }

        let expected_environment = profile.environment_fingerprint();
        let mut results = Vec::with_capacity(profile.cases.len());
        for case in &profile.cases {
            let baseline_batches = batches_for_case(baseline, &case.id)?;
            let candidate_batches = batches_for_case(candidate, &case.id)?;
            if baseline_batches.len() != profile.policy.batches
                || candidate_batches.len() != profile.policy.batches
            {
                return Err(PerfRegimeError::BatchCountMismatch {
                    case_id: case.id.clone(),
                    expected: profile.policy.batches,
                    baseline: baseline_batches.len(),
                    candidate: candidate_batches.len(),
                });
            }
            let baseline_ids: BTreeSet<_> = baseline_batches.keys().copied().collect();
            let candidate_ids: BTreeSet<_> = candidate_batches.keys().copied().collect();
            if baseline_ids != candidate_ids {
                return Err(PerfRegimeError::UnpairedBatches(case.id.clone()));
            }

            let mut baseline_medians = Vec::with_capacity(profile.policy.batches);
            let mut candidate_medians = Vec::with_capacity(profile.policy.batches);
            let mut max_relative_mad_ppm = 0u64;
            let mut winning_batches = 0usize;

            for batch_id in baseline_ids {
                let base = baseline_batches[batch_id];
                let cand = candidate_batches[batch_id];
                validate_batch(profile, case, base, &expected_environment)?;
                validate_batch(profile, case, cand, &expected_environment)?;
                let base_median = median(&base.samples);
                let cand_median = median(&cand.samples);
                baseline_medians.push(base_median);
                candidate_medians.push(cand_median);
                max_relative_mad_ppm = max_relative_mad_ppm
                    .max(relative_mad_ppm(&base.samples)?)
                    .max(relative_mad_ppm(&cand.samples)?);
                if improvement_ppm(case.direction, base_median, cand_median)
                    >= profile.policy.min_improvement_ppm as i64
                {
                    winning_batches += 1;
                }
            }

            let baseline_median = median(&baseline_medians);
            let candidate_median = median(&candidate_medians);
            let improvement_ppm =
                improvement_ppm(case.direction, baseline_median, candidate_median);
            let passed = max_relative_mad_ppm <= profile.policy.max_relative_mad_ppm
                && improvement_ppm >= profile.policy.min_improvement_ppm as i64
                && winning_batches >= profile.policy.min_winning_batches;
            results.push(PerfCaseResult {
                case_id: case.id.clone(),
                class: case.class,
                promotion_gate: case.promotion_gate,
                baseline_median,
                candidate_median,
                improvement_ppm,
                max_relative_mad_ppm,
                winning_batches,
                required_winning_batches: profile.policy.min_winning_batches,
                passed,
            });
        }

        reject_unknown_case_evidence(profile, baseline)?;
        reject_unknown_case_evidence(profile, candidate)?;
        let promotable = results
            .iter()
            .filter(|result| result.promotion_gate)
            .all(|result| result.passed);
        Ok(Self {
            hard_gates_passed: true,
            promotable,
            cases: results,
        })
    }
}

#[derive(Debug)]
pub enum PerfRegimeError {
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
    InvalidArgument(String),
    InvalidSpec(String),
    InvalidDag(String),
    UnknownTask(String),
    TaskIsNotPerf(String),
    MicrobenchmarkCannotPromote(String),
    InsufficientRepetition,
    InvalidAntiNoisePolicy,
    SampleBudgetOverflow,
    SampleBudgetExceeded {
        actual: usize,
        maximum: usize,
    },
    DuplicateBenchmarkCase(String),
    NoEndToEndPromotionGate,
    BenchmarkCaseOutsideTaskRepositories {
        case_id: String,
        repository_role: String,
    },
    DuplicateBenchmarkArtifact {
        repository_role: String,
        path: String,
    },
    BenchmarkArtifactOutsideTaskRepositories {
        repository_role: String,
        path: String,
    },
    BenchmarkArtifactOutsideFrozenSpec {
        repository_role: String,
        path: String,
    },
    TaskAllowanceTouchesFrozenBenchmark {
        repository_role: String,
        frozen_path: String,
        allowed_prefix: String,
    },
    FrozenProfileHashMismatch {
        expected: String,
        actual: String,
    },
    ProfileContextMismatch,
    PatchSet(PatchSetError),
    RepositoryOutsidePerfTask(String),
    RepositoryIsReadOnly(String),
    CandidateTouchesFrozenBenchmark {
        repository_role: String,
        path: String,
    },
    CandidateOperationOutsideAllowance {
        repository_role: String,
        path: String,
        operation: TaskOperation,
    },
    InvalidSamples {
        case_id: String,
        batch_id: String,
    },
    DuplicateMeasurementBatch {
        case_id: String,
        batch_id: String,
    },
    BatchCountMismatch {
        case_id: String,
        expected: usize,
        baseline: usize,
        candidate: usize,
    },
    UnpairedBatches(String),
    WrongEnvironment {
        case_id: String,
        batch_id: String,
    },
    SampleCountMismatch {
        case_id: String,
        batch_id: String,
        expected: usize,
        actual: usize,
    },
    InvalidNoiseStatistic,
    UnknownEvidenceCase(String),
}

impl fmt::Display for PerfRegimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "PERF field '{field}' is empty"),
            Self::InvalidText { field, value } => write!(f, "invalid PERF text '{field}': '{value}'"),
            Self::InvalidIdentifier { field, value } => write!(f, "invalid PERF identifier '{field}': '{value}'"),
            Self::InvalidDigest { field, value } => write!(f, "invalid PERF SHA-256 '{field}': '{value}'"),
            Self::InvalidPath { field, value } => write!(f, "invalid PERF path '{field}': '{value}'"),
            Self::InvalidArgument(value) => write!(f, "invalid PERF argument '{value}'"),
            Self::InvalidSpec(error) => write!(f, "invalid frozen spec: {error}"),
            Self::InvalidDag(error) => write!(f, "invalid frozen task DAG: {error}"),
            Self::UnknownTask(task) => write!(f, "unknown PERF task '{task}'"),
            Self::TaskIsNotPerf(task) => write!(f, "task '{task}' is not in PERF regime"),
            Self::MicrobenchmarkCannotPromote(case) => write!(f, "supporting microbenchmark '{case}' cannot be a promotion gate"),
            Self::InsufficientRepetition => write!(f, "PERF requires at least 3 samples per batch and 2 independent batches"),
            Self::InvalidAntiNoisePolicy => write!(f, "invalid PERF anti-noise policy"),
            Self::SampleBudgetOverflow => write!(f, "PERF sample budget overflow"),
            Self::SampleBudgetExceeded { actual, maximum } => write!(f, "PERF sample budget {actual} exceeds maximum {maximum}"),
            Self::DuplicateBenchmarkCase(case) => write!(f, "duplicate benchmark case '{case}'"),
            Self::NoEndToEndPromotionGate => write!(f, "PERF profile requires at least one end-to-end promotion gate"),
            Self::BenchmarkCaseOutsideTaskRepositories { case_id, repository_role } => write!(f, "benchmark case '{case_id}' uses repository role '{repository_role}' outside the PERF task"),
            Self::DuplicateBenchmarkArtifact { repository_role, path } => write!(f, "duplicate frozen benchmark {repository_role}:{path}"),
            Self::BenchmarkArtifactOutsideTaskRepositories { repository_role, path } => write!(f, "benchmark artifact {repository_role}:{path} is outside the PERF task repositories"),
            Self::BenchmarkArtifactOutsideFrozenSpec { repository_role, path } => write!(f, "benchmark artifact {repository_role}:{path} is outside the frozen objective scope"),
            Self::TaskAllowanceTouchesFrozenBenchmark { repository_role, frozen_path, allowed_prefix } => write!(f, "PERF task allowance {repository_role}:{allowed_prefix} covers frozen benchmark {frozen_path}"),
            Self::FrozenProfileHashMismatch { expected, actual } => write!(f, "frozen PERF profile hash mismatch: expected {expected}, actual {actual}"),
            Self::ProfileContextMismatch => write!(f, "frozen PERF profile belongs to another task DAG"),
            Self::PatchSet(error) => write!(f, "invalid PERF candidate PatchSet: {error}"),
            Self::RepositoryOutsidePerfTask(role) => write!(f, "repository role '{role}' is outside the PERF task"),
            Self::RepositoryIsReadOnly(role) => write!(f, "repository role '{role}' is read-only for the PERF task"),
            Self::CandidateTouchesFrozenBenchmark { repository_role, path } => write!(f, "PERF candidate attempts to change frozen benchmark {repository_role}:{path}"),
            Self::CandidateOperationOutsideAllowance { repository_role, path, operation } => write!(f, "PERF candidate operation {operation:?} on {repository_role}:{path} is outside the task allowlist"),
            Self::InvalidSamples { case_id, batch_id } => write!(f, "invalid positive finite samples for {case_id}/{batch_id}"),
            Self::DuplicateMeasurementBatch { case_id, batch_id } => write!(f, "duplicate measurement batch {case_id}/{batch_id}"),
            Self::BatchCountMismatch { case_id, expected, baseline, candidate } => write!(f, "batch count mismatch for '{case_id}': expected {expected}, baseline {baseline}, candidate {candidate}"),
            Self::UnpairedBatches(case) => write!(f, "baseline/candidate batch IDs are not paired for '{case}'"),
            Self::WrongEnvironment { case_id, batch_id } => write!(f, "wrong measurement environment for {case_id}/{batch_id}"),
            Self::SampleCountMismatch { case_id, batch_id, expected, actual } => write!(f, "sample count mismatch for {case_id}/{batch_id}: expected {expected}, got {actual}"),
            Self::InvalidNoiseStatistic => write!(f, "invalid PERF noise statistic"),
            Self::UnknownEvidenceCase(case) => write!(f, "measurement evidence references unknown case '{case}'"),
        }
    }
}

impl std::error::Error for PerfRegimeError {}

fn verify_spec_and_dag(
    spec: &FrozenAutopilotSpec,
    dag: &AutopilotTaskDag,
) -> Result<(), PerfRegimeError> {
    spec.verify()
        .map_err(|error| PerfRegimeError::InvalidSpec(error.to_string()))?;
    dag.verify()
        .map_err(|error| PerfRegimeError::InvalidDag(error.to_string()))?;
    if dag.spec_sha256() != spec.spec_sha256() {
        return Err(PerfRegimeError::ProfileContextMismatch);
    }
    Ok(())
}

fn path_in_frozen_spec(spec: &FrozenAutopilotSpec, role: &str, path: &str) -> bool {
    spec.repository_scope()
        .iter()
        .find(|scope| scope.role == role)
        .is_some_and(|scope| {
            scope
                .allowed_path_prefixes
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

fn batches_for_case<'a>(
    evidence: &'a [PerfMeasurementBatch],
    case_id: &str,
) -> Result<BTreeMap<&'a str, &'a PerfMeasurementBatch>, PerfRegimeError> {
    let mut out = BTreeMap::new();
    for batch in evidence.iter().filter(|batch| batch.case_id == case_id) {
        if out.insert(batch.batch_id.as_str(), batch).is_some() {
            return Err(PerfRegimeError::DuplicateMeasurementBatch {
                case_id: case_id.to_string(),
                batch_id: batch.batch_id.clone(),
            });
        }
    }
    Ok(out)
}

fn reject_unknown_case_evidence(
    profile: &FrozenPerfBenchmark,
    evidence: &[PerfMeasurementBatch],
) -> Result<(), PerfRegimeError> {
    for batch in evidence {
        if !profile.cases.iter().any(|case| case.id == batch.case_id) {
            return Err(PerfRegimeError::UnknownEvidenceCase(batch.case_id.clone()));
        }
    }
    Ok(())
}

fn validate_batch(
    profile: &FrozenPerfBenchmark,
    case: &BenchmarkCase,
    batch: &PerfMeasurementBatch,
    expected_environment: &str,
) -> Result<(), PerfRegimeError> {
    if batch.environment_fingerprint != expected_environment {
        return Err(PerfRegimeError::WrongEnvironment {
            case_id: case.id.clone(),
            batch_id: batch.batch_id.clone(),
        });
    }
    if batch.samples.len() != profile.policy.samples_per_batch {
        return Err(PerfRegimeError::SampleCountMismatch {
            case_id: case.id.clone(),
            batch_id: batch.batch_id.clone(),
            expected: profile.policy.samples_per_batch,
            actual: batch.samples.len(),
        });
    }
    Ok(())
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn relative_mad_ppm(samples: &[f64]) -> Result<u64, PerfRegimeError> {
    let center = median(samples);
    if !center.is_finite() || center <= 0.0 {
        return Err(PerfRegimeError::InvalidNoiseStatistic);
    }
    let deviations: Vec<_> = samples.iter().map(|sample| (sample - center).abs()).collect();
    let mad = median(&deviations);
    let ppm = mad / center * 1_000_000.0;
    if !ppm.is_finite() || ppm < 0.0 || ppm > u64::MAX as f64 {
        return Err(PerfRegimeError::InvalidNoiseStatistic);
    }
    Ok(ppm.round() as u64)
}

fn improvement_ppm(direction: MetricDirection, baseline: f64, candidate: f64) -> i64 {
    let ratio = match direction {
        MetricDirection::Minimize => (baseline - candidate) / baseline,
        MetricDirection::Maximize => (candidate - baseline) / baseline,
    };
    let ppm = ratio * 1_000_000.0;
    if ppm >= i64::MAX as f64 {
        i64::MAX
    } else if ppm <= i64::MIN as f64 {
        i64::MIN
    } else {
        ppm.round() as i64
    }
}

fn path_is_within(path: &str, prefix: &str) -> bool {
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn validate_text(field: &'static str, value: &str) -> Result<(), PerfRegimeError> {
    if value.is_empty() {
        return Err(PerfRegimeError::EmptyField(field));
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(PerfRegimeError::InvalidText {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), PerfRegimeError> {
    validate_text(field, value)?;
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(PerfRegimeError::InvalidIdentifier {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_digest(field: &'static str, value: &str) -> Result<(), PerfRegimeError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PerfRegimeError::InvalidDigest {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_relative_path(field: &'static str, value: &str) -> Result<(), PerfRegimeError> {
    let invalid = value.is_empty()
        || value.starts_with('/')
        || value.ends_with('/')
        || value.contains('\\')
        || value.contains('\0')
        || value
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."));
    if invalid {
        return Err(PerfRegimeError::InvalidPath {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_arguments(arguments: &[String]) -> Result<(), PerfRegimeError> {
    if arguments.len() > 32 {
        return Err(PerfRegimeError::InvalidArgument(
            "more than 32 benchmark arguments".to_string(),
        ));
    }
    let mut bytes = 0usize;
    for argument in arguments {
        if argument.contains('\0') || argument.chars().any(char::is_control) {
            return Err(PerfRegimeError::InvalidArgument(argument.clone()));
        }
        bytes = bytes
            .checked_add(argument.len())
            .ok_or_else(|| PerfRegimeError::InvalidArgument("argument-byte overflow".to_string()))?;
    }
    if bytes > 4096 {
        return Err(PerfRegimeError::InvalidArgument(
            "benchmark arguments exceed 4096 bytes".to_string(),
        ));
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
    use crate::engineering_trajectory::GateStatus;

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
                    "benchmark",
                    ExplorationSource::repository_file("benches/perf.rs", digest('b')).unwrap(),
                    "inspected the end-to-end benchmark harness",
                )
                .unwrap(),
            ],
        )
        .unwrap();
        ExploredObjective::new(
            "improve measured end-to-end performance",
            "benchmark and anti-noise policy are frozen before candidate search",
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
                    "perf-gates",
                    "all hard gates pass before performance ranking",
                    AcceptanceCheck::command("rsi", "cargo_test", Vec::new()).unwrap(),
                )
                .unwrap(),
            ],
            vec!["editing the benchmark during candidate search".to_string()],
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

    fn perf_task(edit_prefix: &str) -> AutopilotTask {
        AutopilotTask::new(AutopilotTaskDraft {
            id: "perf-opt".to_string(),
            description: "optimize end-to-end latency".to_string(),
            regime: TaskRegime::perf("decode-v1").unwrap(),
            repository_roles: vec!["rsi".to_string()],
            edit_allowances: vec![
                TaskEditAllowance::new(
                    "rsi",
                    vec![edit_prefix.to_string()],
                    vec![TaskOperation::ModifyExact, TaskOperation::Create],
                )
                .unwrap(),
            ],
            hard_gate_profile: HardGateProfile::engineering_strict(),
            budget: TaskBudget::new(4, 10, 10_000, 10_000).unwrap(),
            dependencies: Vec::new(),
            done_criterion_id: "perf-gates".to_string(),
        })
        .unwrap()
    }

    fn dag(edit_prefix: &str) -> AutopilotTaskDag {
        let spec = spec();
        AutopilotTaskDag::new(
            &spec,
            vec![perf_task(edit_prefix)],
            TaskDagPolicy::new(4, 8, 8).unwrap(),
        )
        .unwrap()
    }

    fn profile() -> FrozenPerfBenchmark {
        let spec = spec();
        let dag = dag("src");
        FrozenPerfBenchmark::freeze(
            &spec,
            &dag,
            "perf-opt",
            PerfBenchmarkApproval::new("human-review", digest('c')).unwrap(),
            BenchmarkEnvironment::new(digest('d'), digest('e')).unwrap(),
            AntiNoisePolicy::new(5, 3, 20_000, 20_000, 3).unwrap(),
            vec![
                BenchmarkCase::new(
                    "e2e-latency",
                    "rsi",
                    "bench_e2e",
                    Vec::new(),
                    "latency",
                    "ns",
                    MetricDirection::Minimize,
                    BenchmarkClass::EndToEnd,
                    true,
                )
                .unwrap(),
                BenchmarkCase::new(
                    "micro-helper",
                    "rsi",
                    "bench_helper",
                    Vec::new(),
                    "throughput",
                    "items_s",
                    MetricDirection::Maximize,
                    BenchmarkClass::SupportingMicrobenchmark,
                    false,
                )
                .unwrap(),
            ],
            vec![FrozenBenchmarkArtifact::new("rsi", "benches/perf.rs", digest('f')).unwrap()],
        )
        .unwrap()
    }

    fn admissible() -> AdmissibilityBreakdown {
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

    fn batch(
        profile: &FrozenPerfBenchmark,
        case: &str,
        batch: &str,
        samples: &[f64],
    ) -> PerfMeasurementBatch {
        PerfMeasurementBatch::new(
            case,
            batch,
            profile.environment_fingerprint(),
            samples.to_vec(),
        )
        .unwrap()
    }

    fn paired_evidence(profile: &FrozenPerfBenchmark) -> (Vec<PerfMeasurementBatch>, Vec<PerfMeasurementBatch>) {
        let mut baseline = Vec::new();
        let mut candidate = Vec::new();
        for id in ["a", "b", "c"] {
            baseline.push(batch(profile, "e2e-latency", id, &[100.0, 100.5, 99.5, 100.2, 99.8]));
            candidate.push(batch(profile, "e2e-latency", id, &[90.0, 90.4, 89.6, 90.2, 89.8]));
            baseline.push(batch(profile, "micro-helper", id, &[1000.0, 1005.0, 995.0, 1002.0, 998.0]));
            candidate.push(batch(profile, "micro-helper", id, &[1200.0, 1205.0, 1195.0, 1202.0, 1198.0]));
        }
        (baseline, candidate)
    }

    #[test]
    fn robust_end_to_end_gain_is_promotable() {
        let profile = profile();
        let (baseline, candidate) = paired_evidence(&profile);
        let report = PerfComparisonReport::evaluate(
            &profile,
            &admissible(),
            &baseline,
            &candidate,
        )
        .unwrap();
        assert!(report.hard_gates_passed);
        assert!(report.promotable);
        assert!(report.cases.iter().all(|case| case.passed));
    }

    #[test]
    fn inadmissible_candidate_is_never_performance_ranked() {
        let profile = profile();
        let (baseline, candidate) = paired_evidence(&profile);
        let mut gates = admissible();
        gates.numerical_parity = GateStatus::Fail;
        let report = PerfComparisonReport::evaluate(&profile, &gates, &baseline, &candidate).unwrap();
        assert!(!report.hard_gates_passed);
        assert!(!report.promotable);
        assert!(report.cases.is_empty());
    }

    #[test]
    fn noisy_end_to_end_evidence_cannot_promote() {
        let profile = profile();
        let (baseline, mut candidate) = paired_evidence(&profile);
        for batch in candidate
            .iter_mut()
            .filter(|batch| batch.case_id == "e2e-latency")
        {
            batch.samples = vec![70.0, 110.0, 75.0, 108.0, 90.0];
        }
        let report = PerfComparisonReport::evaluate(
            &profile,
            &admissible(),
            &baseline,
            &candidate,
        )
        .unwrap();
        assert!(!report.promotable);
        assert!(report
            .cases
            .iter()
            .any(|case| case.promotion_gate && !case.passed));
    }

    #[test]
    fn microbenchmark_win_cannot_override_end_to_end_regression() {
        let profile = profile();
        let (baseline, mut candidate) = paired_evidence(&profile);
        for batch in candidate
            .iter_mut()
            .filter(|batch| batch.case_id == "e2e-latency")
        {
            batch.samples = vec![105.0, 105.2, 104.8, 105.1, 104.9];
        }
        let report = PerfComparisonReport::evaluate(
            &profile,
            &admissible(),
            &baseline,
            &candidate,
        )
        .unwrap();
        assert!(!report.promotable);
        assert!(report
            .cases
            .iter()
            .find(|case| case.case_id == "micro-helper")
            .unwrap()
            .passed);
    }

    #[test]
    fn perf_task_cannot_edit_frozen_benchmark_source() {
        let spec = spec();
        let dag = dag("benches");
        assert!(matches!(
            FrozenPerfBenchmark::freeze(
                &spec,
                &dag,
                "perf-opt",
                PerfBenchmarkApproval::new("human-review", digest('c')).unwrap(),
                BenchmarkEnvironment::new(digest('d'), digest('e')).unwrap(),
                AntiNoisePolicy::new(5, 3, 20_000, 20_000, 3).unwrap(),
                vec![BenchmarkCase::new(
                    "e2e",
                    "rsi",
                    "bench_e2e",
                    Vec::new(),
                    "latency",
                    "ns",
                    MetricDirection::Minimize,
                    BenchmarkClass::EndToEnd,
                    true,
                )
                .unwrap()],
                vec![FrozenBenchmarkArtifact::new("rsi", "benches/perf.rs", digest('f')).unwrap()],
            ),
            Err(PerfRegimeError::TaskAllowanceTouchesFrozenBenchmark { .. })
        ));
    }

    #[test]
    fn candidate_patch_cannot_touch_frozen_benchmark() {
        let profile = profile();
        let dag = dag("src");
        let patch = PatchSet::new(vec![FileOperation::modify_exact(
            "benches/perf.rs",
            "old",
            "new",
        )])
        .unwrap();
        assert!(matches!(
            profile.validate_patchset(&dag, "rsi", &patch),
            Err(PerfRegimeError::CandidateTouchesFrozenBenchmark { .. })
        ));
    }

    #[test]
    fn wrong_hardware_or_software_fingerprint_is_rejected() {
        let profile = profile();
        let (baseline, mut candidate) = paired_evidence(&profile);
        candidate[0].environment_fingerprint = digest('0');
        assert!(matches!(
            PerfComparisonReport::evaluate(&profile, &admissible(), &baseline, &candidate),
            Err(PerfRegimeError::WrongEnvironment { .. })
        ));
    }

    #[test]
    fn microbenchmark_cannot_be_declared_as_promotion_gate() {
        assert!(matches!(
            BenchmarkCase::new(
                "micro",
                "rsi",
                "bench_micro",
                Vec::new(),
                "latency",
                "ns",
                MetricDirection::Minimize,
                BenchmarkClass::SupportingMicrobenchmark,
                true,
            ),
            Err(PerfRegimeError::MicrobenchmarkCannotPromote(_))
        ));
    }
}
