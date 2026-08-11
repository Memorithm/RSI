//! Versioned deterministic engineering trajectories for the P7 flywheel.
//!
//! This schema records the complete evidence required to train and evaluate an
//! engineering-specialized SciAgent without reconstructing provenance from logs
//! or moving branch names. It is deliberately data-only: no network or process
//! execution occurs here.

use crate::compatibility::{CompatibilityError, CompatibilitySet};
use crate::json::Json;
use crate::patchset::{FileOperation, PatchSet, PatchSetError};
use std::fmt;

/// Current wire-format version for [`EngineeringTrajectory`].
pub const ENGINEERING_TRAJECTORY_SCHEMA_VERSION: u64 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateStatus {
    Pass,
    Fail,
    Unknown,
}

impl GateStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Unknown => "unknown",
        }
    }

    fn parse(value: &str) -> Result<Self, EngineeringTrajectoryError> {
        match value {
            "pass" => Ok(Self::Pass),
            "fail" => Ok(Self::Fail),
            "unknown" => Ok(Self::Unknown),
            other => Err(EngineeringTrajectoryError::InvalidValue {
                field: "gate_status",
                value: other.to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissibilityBreakdown {
    pub build: GateStatus,
    pub required_tests: GateStatus,
    pub numerical_parity: GateStatus,
    pub provenance: GateStatus,
    pub deterministic_contract: GateStatus,
    pub resource_budget: GateStatus,
    pub policy_checks: GateStatus,
}

impl AdmissibilityBreakdown {
    #[must_use]
    pub fn is_admissible(&self) -> bool {
        [
            self.build,
            self.required_tests,
            self.numerical_parity,
            self.provenance,
            self.deterministic_contract,
            self.resource_budget,
            self.policy_checks,
        ]
        .into_iter()
        .all(|status| status == GateStatus::Pass)
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set("build", Json::Str(self.build.as_str().to_string()))
            .set(
                "required_tests",
                Json::Str(self.required_tests.as_str().to_string()),
            )
            .set(
                "numerical_parity",
                Json::Str(self.numerical_parity.as_str().to_string()),
            )
            .set(
                "provenance",
                Json::Str(self.provenance.as_str().to_string()),
            )
            .set(
                "deterministic_contract",
                Json::Str(self.deterministic_contract.as_str().to_string()),
            )
            .set(
                "resource_budget",
                Json::Str(self.resource_budget.as_str().to_string()),
            )
            .set(
                "policy_checks",
                Json::Str(self.policy_checks.as_str().to_string()),
            );
        out
    }

    fn from_json(value: &Json) -> Result<Self, EngineeringTrajectoryError> {
        Ok(Self {
            build: gate_field(value, "build")?,
            required_tests: gate_field(value, "required_tests")?,
            numerical_parity: gate_field(value, "numerical_parity")?,
            provenance: gate_field(value, "provenance")?,
            deterministic_contract: gate_field(value, "deterministic_contract")?,
            resource_budget: gate_field(value, "resource_budget")?,
            policy_checks: gate_field(value, "policy_checks")?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposerMetadata {
    pub provider: String,
    pub model: String,
    pub configuration_id: String,
}

impl ProposerMetadata {
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        configuration_id: impl Into<String>,
    ) -> Result<Self, EngineeringTrajectoryError> {
        let value = Self {
            provider: provider.into(),
            model: model.into(),
            configuration_id: configuration_id.into(),
        };
        validate_text("proposer.provider", &value.provider)?;
        validate_text("proposer.model", &value.model)?;
        validate_text("proposer.configuration_id", &value.configuration_id)?;
        Ok(value)
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set("configuration_id", Json::Str(self.configuration_id.clone()))
            .set("model", Json::Str(self.model.clone()))
            .set("provider", Json::Str(self.provider.clone()));
        out
    }

    fn from_json(value: &Json) -> Result<Self, EngineeringTrajectoryError> {
        Self::new(
            required_string(value, "provider")?,
            required_string(value, "model")?,
            required_string(value, "configuration_id")?,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkRecord {
    pub metric: String,
    pub unit: String,
    pub samples: Vec<f64>,
    pub summary: f64,
}

impl BenchmarkRecord {
    pub fn new(
        metric: impl Into<String>,
        unit: impl Into<String>,
        samples: Vec<f64>,
        summary: f64,
    ) -> Result<Self, EngineeringTrajectoryError> {
        let value = Self {
            metric: metric.into(),
            unit: unit.into(),
            samples,
            summary,
        };
        validate_text("benchmark.metric", &value.metric)?;
        validate_text("benchmark.unit", &value.unit)?;
        if value.samples.is_empty() {
            return Err(EngineeringTrajectoryError::EmptyField("benchmark.samples"));
        }
        if value.samples.iter().any(|sample| !sample.is_finite()) || !value.summary.is_finite() {
            return Err(EngineeringTrajectoryError::InvalidNumber("benchmark"));
        }
        Ok(value)
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set("metric", Json::Str(self.metric.clone()))
            .set(
                "samples",
                Json::Arr(self.samples.iter().copied().map(Json::Num).collect()),
            )
            .set("summary", Json::Num(self.summary))
            .set("unit", Json::Str(self.unit.clone()));
        out
    }

    fn from_json(value: &Json) -> Result<Self, EngineeringTrajectoryError> {
        let samples = value
            .get("samples")
            .and_then(Json::as_array)
            .ok_or(EngineeringTrajectoryError::MissingField("samples"))?
            .iter()
            .map(|sample| {
                sample
                    .as_f64()
                    .ok_or(EngineeringTrajectoryError::InvalidNumber("benchmark.samples"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let summary = value
            .get("summary")
            .and_then(Json::as_f64)
            .ok_or(EngineeringTrajectoryError::InvalidNumber("benchmark.summary"))?;
        Self::new(
            required_string(value, "metric")?,
            required_string(value, "unit")?,
            samples,
            summary,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineeringVerdict {
    Accepted,
    Rejected,
}

impl EngineeringVerdict {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
        }
    }

    fn parse(value: &str) -> Result<Self, EngineeringTrajectoryError> {
        match value {
            "accepted" => Ok(Self::Accepted),
            "rejected" => Ok(Self::Rejected),
            other => Err(EngineeringTrajectoryError::InvalidValue {
                field: "verdict",
                value: other.to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaterVerdict {
    pub source: String,
    pub accepted: bool,
    pub reason: String,
}

impl LaterVerdict {
    pub fn new(
        source: impl Into<String>,
        accepted: bool,
        reason: impl Into<String>,
    ) -> Result<Self, EngineeringTrajectoryError> {
        let value = Self {
            source: source.into(),
            accepted,
            reason: reason.into(),
        };
        validate_text("later_verdict.source", &value.source)?;
        validate_text("later_verdict.reason", &value.reason)?;
        Ok(value)
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set("accepted", Json::Bool(self.accepted))
            .set("reason", Json::Str(self.reason.clone()))
            .set("source", Json::Str(self.source.clone()));
        out
    }

    fn from_json(value: &Json) -> Result<Self, EngineeringTrajectoryError> {
        Self::new(
            required_string(value, "source")?,
            value
                .get("accepted")
                .and_then(Json::as_bool)
                .ok_or(EngineeringTrajectoryError::MissingField("accepted"))?,
            required_string(value, "reason")?,
        )
    }
}

/// Deterministic v3 engineering trajectory used by the flywheel and future
/// SciAgent ingestion. Negative/rejected examples are first-class records.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringTrajectory {
    pub task_spec_id: String,
    pub compatibility: CompatibilitySet,
    pub parent_state_id: String,
    pub patch_set: PatchSet,
    pub proposer: ProposerMetadata,
    pub compiler_test_device_evidence: Vec<String>,
    pub admissibility: AdmissibilityBreakdown,
    pub benchmarks: Vec<BenchmarkRecord>,
    pub verdict: EngineeringVerdict,
    pub verdict_reason: String,
    pub later_verdicts: Vec<LaterVerdict>,
}

impl EngineeringTrajectory {
    pub fn validate(&self) -> Result<(), EngineeringTrajectoryError> {
        validate_identity("task_spec_id", &self.task_spec_id)?;
        validate_identity("parent_state_id", &self.parent_state_id)?;
        validate_text("verdict_reason", &self.verdict_reason)?;
        if self.compiler_test_device_evidence.is_empty() {
            return Err(EngineeringTrajectoryError::EmptyField(
                "compiler_test_device_evidence",
            ));
        }
        for evidence in &self.compiler_test_device_evidence {
            validate_text("compiler_test_device_evidence", evidence)?;
        }
        self.patch_set
            .identity()
            .map_err(EngineeringTrajectoryError::PatchSet)?;
        if self.verdict == EngineeringVerdict::Accepted && !self.admissibility.is_admissible() {
            return Err(EngineeringTrajectoryError::InadmissibleAcceptedVerdict);
        }
        Ok(())
    }

    pub fn to_json_string(&self) -> Result<String, EngineeringTrajectoryError> {
        self.validate()?;
        let mut root = Json::obj();
        root.set("admissibility", self.admissibility.to_json())
            .set(
                "benchmarks",
                Json::Arr(self.benchmarks.iter().map(BenchmarkRecord::to_json).collect()),
            )
            .set(
                "compatibility",
                Json::parse(&self.compatibility.to_json_string())
                    .map_err(EngineeringTrajectoryError::Json)?,
            )
            .set(
                "compiler_test_device_evidence",
                Json::Arr(
                    self.compiler_test_device_evidence
                        .iter()
                        .cloned()
                        .map(Json::Str)
                        .collect(),
                ),
            )
            .set(
                "later_verdicts",
                Json::Arr(self.later_verdicts.iter().map(LaterVerdict::to_json).collect()),
            )
            .set("parent_state_id", Json::Str(self.parent_state_id.clone()))
            .set("patch_set", patch_set_json(&self.patch_set)?)
            .set("proposer", self.proposer.to_json())
            .set(
                "schema_version",
                Json::Num(ENGINEERING_TRAJECTORY_SCHEMA_VERSION as f64),
            )
            .set("task_spec_id", Json::Str(self.task_spec_id.clone()))
            .set("verdict", Json::Str(self.verdict.as_str().to_string()))
            .set("verdict_reason", Json::Str(self.verdict_reason.clone()));
        Ok(root.to_string())
    }

    pub fn from_json_str(input: &str) -> Result<Self, EngineeringTrajectoryError> {
        let root = Json::parse(input).map_err(EngineeringTrajectoryError::Json)?;
        let version = root
            .get("schema_version")
            .and_then(Json::as_u64)
            .ok_or(EngineeringTrajectoryError::MissingField("schema_version"))?;
        if version != ENGINEERING_TRAJECTORY_SCHEMA_VERSION {
            return Err(EngineeringTrajectoryError::UnsupportedVersion(version));
        }

        let compatibility_json = root
            .get("compatibility")
            .ok_or(EngineeringTrajectoryError::MissingField("compatibility"))?
            .to_string();
        let compatibility = CompatibilitySet::from_json_str(&compatibility_json)
            .map_err(EngineeringTrajectoryError::Compatibility)?;
        let patch_set = patch_set_from_json(
            root.get("patch_set")
                .ok_or(EngineeringTrajectoryError::MissingField("patch_set"))?,
        )?;
        let evidence = root
            .get("compiler_test_device_evidence")
            .and_then(Json::as_array)
            .ok_or(EngineeringTrajectoryError::MissingField(
                "compiler_test_device_evidence",
            ))?
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or(EngineeringTrajectoryError::InvalidValue {
                        field: "compiler_test_device_evidence",
                        value: item.to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let benchmarks = root
            .get("benchmarks")
            .and_then(Json::as_array)
            .ok_or(EngineeringTrajectoryError::MissingField("benchmarks"))?
            .iter()
            .map(BenchmarkRecord::from_json)
            .collect::<Result<Vec<_>, _>>()?;
        let later_verdicts = root
            .get("later_verdicts")
            .and_then(Json::as_array)
            .ok_or(EngineeringTrajectoryError::MissingField("later_verdicts"))?
            .iter()
            .map(LaterVerdict::from_json)
            .collect::<Result<Vec<_>, _>>()?;

        let value = Self {
            task_spec_id: required_string(&root, "task_spec_id")?,
            compatibility,
            parent_state_id: required_string(&root, "parent_state_id")?,
            patch_set,
            proposer: ProposerMetadata::from_json(
                root.get("proposer")
                    .ok_or(EngineeringTrajectoryError::MissingField("proposer"))?,
            )?,
            compiler_test_device_evidence: evidence,
            admissibility: AdmissibilityBreakdown::from_json(
                root.get("admissibility")
                    .ok_or(EngineeringTrajectoryError::MissingField("admissibility"))?,
            )?,
            benchmarks,
            verdict: EngineeringVerdict::parse(&required_string(&root, "verdict")?)?,
            verdict_reason: required_string(&root, "verdict_reason")?,
            later_verdicts,
        };
        value.validate()?;
        Ok(value)
    }
}

#[derive(Debug)]
pub enum EngineeringTrajectoryError {
    Json(String),
    Compatibility(CompatibilityError),
    PatchSet(PatchSetError),
    MissingField(&'static str),
    EmptyField(&'static str),
    UnsupportedVersion(u64),
    InvalidIdentity { field: &'static str, value: String },
    InvalidValue { field: &'static str, value: String },
    InvalidNumber(&'static str),
    IdentityMismatch { expected: String, actual: String },
    InadmissibleAcceptedVerdict,
}

impl fmt::Display for EngineeringTrajectoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "engineering trajectory JSON error: {error}"),
            Self::Compatibility(error) => write!(f, "engineering trajectory compatibility: {error}"),
            Self::PatchSet(error) => write!(f, "engineering trajectory PatchSet: {error}"),
            Self::MissingField(field) => write!(f, "engineering trajectory field missing: {field}"),
            Self::EmptyField(field) => write!(f, "engineering trajectory field empty: {field}"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported engineering trajectory schema version: {version}")
            }
            Self::InvalidIdentity { field, value } => {
                write!(f, "invalid immutable identity for {field}: {value}")
            }
            Self::InvalidValue { field, value } => {
                write!(f, "invalid engineering trajectory value for {field}: {value}")
            }
            Self::InvalidNumber(field) => write!(f, "non-finite or missing number in {field}"),
            Self::IdentityMismatch { expected, actual } => {
                write!(f, "PatchSet identity mismatch: expected {expected}, got {actual}")
            }
            Self::InadmissibleAcceptedVerdict => {
                write!(f, "accepted trajectory must have every COGNO hard gate passing")
            }
        }
    }
}

impl std::error::Error for EngineeringTrajectoryError {}

fn validate_text(field: &'static str, value: &str) -> Result<(), EngineeringTrajectoryError> {
    if value.is_empty() {
        return Err(EngineeringTrajectoryError::EmptyField(field));
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(EngineeringTrajectoryError::InvalidValue {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_identity(field: &'static str, value: &str) -> Result<(), EngineeringTrajectoryError> {
    if !matches!(value.len(), 40 | 64) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(EngineeringTrajectoryError::InvalidIdentity {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn required_string(value: &Json, field: &'static str) -> Result<String, EngineeringTrajectoryError> {
    value
        .get(field)
        .and_then(Json::as_str)
        .map(str::to_string)
        .ok_or(EngineeringTrajectoryError::MissingField(field))
}

fn gate_field(value: &Json, field: &'static str) -> Result<GateStatus, EngineeringTrajectoryError> {
    let raw = value
        .get(field)
        .and_then(Json::as_str)
        .ok_or(EngineeringTrajectoryError::MissingField(field))?;
    GateStatus::parse(raw)
}

fn patch_set_json(patch_set: &PatchSet) -> Result<Json, EngineeringTrajectoryError> {
    let mut out = Json::obj();
    out.set(
        "identity",
        Json::Str(
            patch_set
                .identity()
                .map_err(EngineeringTrajectoryError::PatchSet)?,
        ),
    )
    .set(
        "operations",
        Json::Arr(
            patch_set
                .operations()
                .iter()
                .map(operation_json)
                .collect(),
        ),
    );
    Ok(out)
}

fn operation_json(operation: &FileOperation) -> Json {
    let mut out = Json::obj();
    match operation {
        FileOperation::ModifyExact {
            path,
            expected,
            replacement,
        } => {
            out.set("expected", Json::Str(expected.clone()))
                .set("kind", Json::Str("modify_exact".to_string()))
                .set("path", Json::Str(path.clone()))
                .set("replacement", Json::Str(replacement.clone()));
        }
        FileOperation::Create { path, content } => {
            out.set("content", Json::Str(content.clone()))
                .set("kind", Json::Str("create".to_string()))
                .set("path", Json::Str(path.clone()));
        }
        FileOperation::Delete {
            path,
            expected_sha256,
        } => {
            out.set("expected_sha256", Json::Str(expected_sha256.clone()))
                .set("kind", Json::Str("delete".to_string()))
                .set("path", Json::Str(path.clone()));
        }
    }
    out
}

fn patch_set_from_json(value: &Json) -> Result<PatchSet, EngineeringTrajectoryError> {
    let operations = value
        .get("operations")
        .and_then(Json::as_array)
        .ok_or(EngineeringTrajectoryError::MissingField("operations"))?
        .iter()
        .map(operation_from_json)
        .collect::<Result<Vec<_>, _>>()?;
    let patch_set = PatchSet::new(operations).map_err(EngineeringTrajectoryError::PatchSet)?;
    let expected = required_string(value, "identity")?;
    let actual = patch_set
        .identity()
        .map_err(EngineeringTrajectoryError::PatchSet)?;
    if expected != actual {
        return Err(EngineeringTrajectoryError::IdentityMismatch { expected, actual });
    }
    Ok(patch_set)
}

fn operation_from_json(value: &Json) -> Result<FileOperation, EngineeringTrajectoryError> {
    let kind = required_string(value, "kind")?;
    let path = required_string(value, "path")?;
    match kind.as_str() {
        "modify_exact" => Ok(FileOperation::modify_exact(
            path,
            required_string(value, "expected")?,
            required_string(value, "replacement")?,
        )),
        "create" => Ok(FileOperation::create(
            path,
            required_string(value, "content")?,
        )),
        "delete" => Ok(FileOperation::delete(
            path,
            required_string(value, "expected_sha256")?,
        )),
        other => Err(EngineeringTrajectoryError::InvalidValue {
            field: "operation.kind",
            value: other.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compatibility::RepositoryRevision;

    fn compatibility() -> CompatibilitySet {
        CompatibilitySet::new(
            vec![
                RepositoryRevision::new("Memorithm/RSI", "a".repeat(40), "rsi").unwrap(),
                RepositoryRevision::new("Memorithm/scirust", "b".repeat(40), "scirust").unwrap(),
                RepositoryRevision::new(
                    "Memorithm/FLAT-ATTENTION",
                    "c".repeat(40),
                    "flat",
                )
                .unwrap(),
            ],
            "rustc 1.89.0",
            vec!["flat-attention".into(), "wgpu".into()],
        )
        .unwrap()
    }

    fn trajectory(verdict: EngineeringVerdict) -> EngineeringTrajectory {
        let admitted = verdict == EngineeringVerdict::Accepted;
        EngineeringTrajectory {
            task_spec_id: "1".repeat(64),
            compatibility: compatibility(),
            parent_state_id: "2".repeat(64),
            patch_set: PatchSet::new(vec![
                FileOperation::modify_exact("src/a.rs", "old", "new"),
                FileOperation::create("src/new.rs", "content"),
            ])
            .unwrap(),
            proposer: ProposerMetadata::new("openai", "engineering-model", "cfg-v1").unwrap(),
            compiler_test_device_evidence: vec![
                "cargo check: pass".into(),
                "wgpu parity: pass".into(),
            ],
            admissibility: AdmissibilityBreakdown {
                build: GateStatus::Pass,
                required_tests: GateStatus::Pass,
                numerical_parity: GateStatus::Pass,
                provenance: GateStatus::Pass,
                deterministic_contract: GateStatus::Pass,
                resource_budget: GateStatus::Pass,
                policy_checks: if admitted {
                    GateStatus::Pass
                } else {
                    GateStatus::Fail
                },
            },
            benchmarks: vec![BenchmarkRecord::new(
                "decode_latency",
                "us",
                vec![12.0, 11.5, 11.75],
                11.75,
            )
            .unwrap()],
            verdict,
            verdict_reason: if admitted {
                "all frozen gates passed".into()
            } else {
                "policy gate rejected candidate".into()
            },
            later_verdicts: vec![LaterVerdict::new("ci", admitted, "final-head CI verdict").unwrap()],
        }
    }

    #[test]
    fn deterministic_round_trip_is_byte_stable() {
        let original = trajectory(EngineeringVerdict::Accepted);
        let encoded = original.to_json_string().unwrap();
        let decoded = EngineeringTrajectory::from_json_str(&encoded).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.to_json_string().unwrap(), encoded);
    }

    #[test]
    fn rejected_examples_are_preserved() {
        let original = trajectory(EngineeringVerdict::Rejected);
        let encoded = original.to_json_string().unwrap();
        let decoded = EngineeringTrajectory::from_json_str(&encoded).unwrap();
        assert_eq!(decoded.verdict, EngineeringVerdict::Rejected);
        assert_eq!(decoded.admissibility.policy_checks, GateStatus::Fail);
    }

    #[test]
    fn accepted_verdict_fails_closed_on_unknown_gate() {
        let mut value = trajectory(EngineeringVerdict::Accepted);
        value.admissibility.provenance = GateStatus::Unknown;
        assert!(matches!(
            value.to_json_string(),
            Err(EngineeringTrajectoryError::InadmissibleAcceptedVerdict)
        ));
    }

    #[test]
    fn patchset_identity_tampering_is_rejected() {
        let encoded = trajectory(EngineeringVerdict::Accepted)
            .to_json_string()
            .unwrap();
        let identity = trajectory(EngineeringVerdict::Accepted)
            .patch_set
            .identity()
            .unwrap();
        let tampered = encoded.replacen(&identity, &"f".repeat(64), 1);
        assert!(matches!(
            EngineeringTrajectory::from_json_str(&tampered),
            Err(EngineeringTrajectoryError::IdentityMismatch { .. })
        ));
    }

    #[test]
    fn moving_or_malformed_state_identity_is_rejected() {
        let mut value = trajectory(EngineeringVerdict::Rejected);
        value.parent_state_id = "main".into();
        assert!(matches!(
            value.to_json_string(),
            Err(EngineeringTrajectoryError::InvalidIdentity {
                field: "parent_state_id",
                ..
            })
        ));
    }

    #[test]
    fn non_finite_benchmark_sample_is_rejected() {
        assert!(matches!(
            BenchmarkRecord::new("latency", "us", vec![f64::NAN], 1.0),
            Err(EngineeringTrajectoryError::InvalidNumber("benchmark"))
        ));
    }
}
