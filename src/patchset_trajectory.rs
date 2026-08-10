//! Versioned trajectory records for multi-file engineering candidates.
//!
//! The historical [`crate::trajectory::Trajectory`] stays supported. P2.2 adds
//! an explicit PatchSet record so later flywheel ingestion does not have to
//! reconstruct multi-file edits from natural-language prompts.

use crate::json::Json;
use crate::patchset::{FileOperation, PatchSet, PatchSetError};
use crate::trajectory::Trajectory;
use std::fmt;

pub const PATCHSET_TRAJECTORY_SCHEMA_VERSION: u64 = 2;

#[derive(Debug, Clone, PartialEq)]
pub struct PatchSetTrajectory {
    pub patch_set: PatchSet,
    pub rationale: String,
    pub compiles: bool,
    pub tests_passed: u32,
    pub tests_failed: u32,
    pub score: f64,
    pub output: Option<String>,
}

impl PatchSetTrajectory {
    /// Convert a historical single-patch in-memory trajectory losslessly into
    /// the v2 PatchSet representation.
    pub fn from_legacy(legacy: &Trajectory) -> Self {
        let patch_set = PatchSet::new(vec![FileOperation::modify_exact(
            legacy.target.clone(),
            legacy.find.clone(),
            legacy.replace.clone(),
        )])
        .expect("legacy Trajectory contains one structurally valid operation");
        Self {
            patch_set,
            rationale: "legacy single-patch trajectory".to_string(),
            compiles: legacy.compiles,
            tests_passed: legacy.tests_passed,
            tests_failed: legacy.tests_failed,
            score: legacy.score,
            output: legacy.output.clone(),
        }
    }

    /// Stable, self-describing JSON line with the exact PatchSet identity and
    /// operations. This is intentionally richer than the legacy SFT-only
    /// `{prompt,completion,score}` record.
    pub fn to_jsonl(&self) -> Result<String, PatchSetTrajectoryError> {
        let mut root = Json::obj();
        root.set(
            "schema_version",
            Json::Num(PATCHSET_TRAJECTORY_SCHEMA_VERSION as f64),
        )
        .set(
            "patchset_id",
            Json::Str(self.patch_set.identity().map_err(PatchSetTrajectoryError::PatchSet)?),
        )
        .set("operations", operations_json(&self.patch_set))
        .set("rationale", Json::Str(self.rationale.clone()))
        .set("compiles", Json::Bool(self.compiles))
        .set("tests_passed", Json::Num(self.tests_passed as f64))
        .set("tests_failed", Json::Num(self.tests_failed as f64))
        .set(
            "score",
            if self.score.is_finite() {
                Json::Num(self.score)
            } else {
                Json::Null
            },
        )
        .set(
            "output",
            self.output.clone().map(Json::Str).unwrap_or(Json::Null),
        );
        Ok(root.to_string())
    }

    /// Decode the v2 record. For old live objects, use [`Self::from_legacy`].
    /// The old JSONL export intentionally omitted raw patch fields, so it cannot
    /// be reconstructed losslessly and is not guessed from prompt text.
    pub fn from_jsonl(line: &str) -> Result<Self, PatchSetTrajectoryError> {
        let json = Json::parse(line).map_err(PatchSetTrajectoryError::Json)?;
        let version = json
            .get("schema_version")
            .and_then(Json::as_u64)
            .ok_or(PatchSetTrajectoryError::Missing("schema_version"))?;
        if version != PATCHSET_TRAJECTORY_SCHEMA_VERSION {
            return Err(PatchSetTrajectoryError::UnsupportedVersion(version));
        }
        let ops = json
            .get("operations")
            .and_then(Json::as_array)
            .ok_or(PatchSetTrajectoryError::Missing("operations"))?;
        let mut operations = Vec::with_capacity(ops.len());
        for op in ops {
            operations.push(operation_from_json(op)?);
        }
        let patch_set = PatchSet::new(operations).map_err(PatchSetTrajectoryError::PatchSet)?;
        let expected_id = json
            .get("patchset_id")
            .and_then(Json::as_str)
            .ok_or(PatchSetTrajectoryError::Missing("patchset_id"))?;
        let actual_id = patch_set
            .identity()
            .map_err(PatchSetTrajectoryError::PatchSet)?;
        if expected_id != actual_id {
            return Err(PatchSetTrajectoryError::IdentityMismatch {
                expected: expected_id.to_string(),
                actual: actual_id,
            });
        }
        Ok(Self {
            patch_set,
            rationale: json
                .get("rationale")
                .and_then(Json::as_str)
                .unwrap_or("")
                .to_string(),
            compiles: json
                .get("compiles")
                .and_then(Json::as_bool)
                .ok_or(PatchSetTrajectoryError::Missing("compiles"))?,
            tests_passed: u32_checked(&json, "tests_passed")?,
            tests_failed: u32_checked(&json, "tests_failed")?,
            score: json.get("score").and_then(Json::as_f64).unwrap_or(f64::NEG_INFINITY),
            output: json
                .get("output")
                .and_then(Json::as_str)
                .map(str::to_string),
        })
    }
}

#[derive(Debug)]
pub enum PatchSetTrajectoryError {
    Json(String),
    PatchSet(PatchSetError),
    Missing(&'static str),
    UnsupportedVersion(u64),
    InvalidOperation(String),
    IntegerOverflow(&'static str),
    IdentityMismatch { expected: String, actual: String },
}

impl fmt::Display for PatchSetTrajectoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(e) => write!(f, "trajectory JSON error: {e}"),
            Self::PatchSet(e) => write!(f, "trajectory PatchSet error: {e}"),
            Self::Missing(field) => write!(f, "trajectory field missing: {field}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported trajectory schema version: {v}"),
            Self::InvalidOperation(msg) => write!(f, "invalid trajectory operation: {msg}"),
            Self::IntegerOverflow(field) => write!(f, "trajectory integer out of range: {field}"),
            Self::IdentityMismatch { expected, actual } => {
                write!(f, "PatchSet identity mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for PatchSetTrajectoryError {}

fn operations_json(patch_set: &PatchSet) -> Json {
    Json::Arr(
        patch_set
            .operations()
            .iter()
            .map(|op| {
                let mut json = Json::obj();
                match op {
                    FileOperation::ModifyExact {
                        path,
                        expected,
                        replacement,
                    } => {
                        json.set("kind", Json::Str("modify_exact".to_string()))
                            .set("path", Json::Str(path.clone()))
                            .set("expected", Json::Str(expected.clone()))
                            .set("replacement", Json::Str(replacement.clone()));
                    }
                    FileOperation::Create { path, content } => {
                        json.set("kind", Json::Str("create".to_string()))
                            .set("path", Json::Str(path.clone()))
                            .set("content", Json::Str(content.clone()));
                    }
                    FileOperation::Delete {
                        path,
                        expected_sha256,
                    } => {
                        json.set("kind", Json::Str("delete".to_string()))
                            .set("path", Json::Str(path.clone()))
                            .set("expected_sha256", Json::Str(expected_sha256.clone()));
                    }
                }
                json
            })
            .collect(),
    )
}

fn operation_from_json(json: &Json) -> Result<FileOperation, PatchSetTrajectoryError> {
    let kind = string_field(json, "kind")?;
    let path = string_field(json, "path")?;
    match kind {
        "modify_exact" => Ok(FileOperation::modify_exact(
            path,
            string_field(json, "expected")?,
            string_field(json, "replacement")?,
        )),
        "create" => Ok(FileOperation::create(path, string_field(json, "content")?)),
        "delete" => Ok(FileOperation::delete(
            path,
            string_field(json, "expected_sha256")?,
        )),
        other => Err(PatchSetTrajectoryError::InvalidOperation(format!(
            "unsupported kind {other}"
        ))),
    }
}

fn string_field<'a>(json: &'a Json, field: &'static str) -> Result<&'a str, PatchSetTrajectoryError> {
    json.get(field)
        .and_then(Json::as_str)
        .ok_or(PatchSetTrajectoryError::Missing(field))
}

fn u32_checked(json: &Json, field: &'static str) -> Result<u32, PatchSetTrajectoryError> {
    let n = json
        .get(field)
        .and_then(Json::as_u64)
        .ok_or(PatchSetTrajectoryError::Missing(field))?;
    u32::try_from(n).map_err(|_| PatchSetTrajectoryError::IntegerOverflow(field))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multi() -> PatchSetTrajectory {
        PatchSetTrajectory {
            patch_set: PatchSet::new(vec![
                FileOperation::modify_exact("src/a.rs", "a", "b"),
                FileOperation::create("src/new.rs", "new"),
                FileOperation::delete(
                    "old.txt",
                    "0000000000000000000000000000000000000000000000000000000000000000",
                ),
            ])
            .unwrap(),
            rationale: "multi".into(),
            compiles: true,
            tests_passed: 42,
            tests_failed: 0,
            score: 1.25,
            output: Some("all green".into()),
        }
    }

    #[test]
    fn v2_roundtrip_preserves_patchset_identity_and_verdict() {
        let original = multi();
        let line = original.to_jsonl().unwrap();
        assert_eq!(line.lines().count(), 1);
        let decoded = PatchSetTrajectory::from_jsonl(&line).unwrap();
        assert_eq!(decoded.patch_set, original.patch_set);
        assert_eq!(decoded.patch_set.identity().unwrap(), original.patch_set.identity().unwrap());
        assert_eq!(decoded.compiles, original.compiles);
        assert_eq!(decoded.tests_passed, original.tests_passed);
        assert_eq!(decoded.score, original.score);
    }

    #[test]
    fn tampered_identity_is_rejected() {
        let line = multi().to_jsonl().unwrap();
        let id = multi().patch_set.identity().unwrap();
        let tampered = line.replacen(&id, &"f".repeat(64), 1);
        assert!(matches!(
            PatchSetTrajectory::from_jsonl(&tampered),
            Err(PatchSetTrajectoryError::IdentityMismatch { .. })
        ));
    }

    #[test]
    fn legacy_single_patch_converts_losslessly_at_api_boundary() {
        let legacy = Trajectory {
            target: "src/a.rs".into(),
            find: "old".into(),
            replace: "new".into(),
            file_content: "old".into(),
            compiles: true,
            tests_passed: 3,
            tests_failed: 0,
            score: 2.0,
            output: None,
        };
        let upgraded = PatchSetTrajectory::from_legacy(&legacy);
        assert_eq!(upgraded.patch_set.len(), 1);
        assert_eq!(upgraded.compiles, legacy.compiles);
        assert_eq!(upgraded.tests_passed, legacy.tests_passed);
        assert_eq!(upgraded.score, legacy.score);
    }

    #[test]
    fn old_lossy_jsonl_is_not_guessed() {
        let legacy = Trajectory {
            target: "src/a.rs".into(),
            find: "old".into(),
            replace: "new".into(),
            file_content: "old".into(),
            compiles: true,
            tests_passed: 1,
            tests_failed: 0,
            score: 1.0,
            output: None,
        };
        let old_line = legacy.to_jsonl();
        assert!(matches!(
            PatchSetTrajectory::from_jsonl(&old_line),
            Err(PatchSetTrajectoryError::Missing("schema_version"))
        ));
    }
}
