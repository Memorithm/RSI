//! Cumulative archive and complete-state promotion for engineering DGM lineages.
//!
//! P3.2 builds on [`crate::candidate_state::CandidateState`]: accepted archive
//! nodes own a materialized immutable state, children are evaluated from the
//! exact selected parent, and promotion reproduces the complete accepted tree.
//! The live tree is never used as an implicit parent after archive creation.

use crate::candidate_state::{CandidateState, CandidateStateError, CandidateStoragePolicy};
use crate::dgm::{Evaluator, Fitness};
use crate::json::Json;
use crate::patchset::{FileOperation, PatchSet, PatchSetError};
use crate::rng::Rng;
use crate::sha256::sha256_hex;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

pub const CUMULATIVE_ARCHIVE_SCHEMA_VERSION: u64 = 1;
const SKIP_DIRS: &[&str] = &["target", ".git", "node_modules", ".rsi_backups"];

/// One accepted, replayable archive node.
#[derive(Debug, Clone, PartialEq)]
pub struct CumulativeRecord {
    pub state_id: String,
    pub parent_state_id: Option<String>,
    pub patch_set_id: Option<String>,
    pub patch_set: Option<PatchSet>,
    pub rationale: String,
    pub fitness: Fitness,
}

/// Result of evaluating one child against a materialized parent.
#[derive(Debug, Clone, PartialEq)]
pub struct CumulativeOutcome {
    pub state_id: String,
    pub parent_state_id: String,
    pub patch_set_id: String,
    pub accepted: bool,
    pub fitness: Fitness,
}

/// Durable evidence for a complete-state promotion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionReceipt {
    pub from_live_state_id: String,
    pub promoted_state_id: String,
    pub backup_path: PathBuf,
}

/// Accepted cumulative lineage plus its materialized immutable states.
///
/// `records` is deterministic/replayable metadata. `states` is runtime storage
/// rebuilt from the baseline plus the recorded PatchSets when deserializing.
pub struct CumulativeArchive {
    records: Vec<CumulativeRecord>,
    states: BTreeMap<String, CandidateState>,
    baseline_state_id: String,
    policy: CandidateStoragePolicy,
}

impl CumulativeArchive {
    /// Create an archive from an isolated snapshot of the supplied live/source tree.
    pub fn new(
        source_root: &Path,
        baseline_fitness: Fitness,
        policy: CandidateStoragePolicy,
    ) -> Result<Self, CumulativeArchiveError> {
        let baseline = CandidateState::baseline(source_root, policy)?;
        let baseline_state_id = baseline.state_id().to_string();
        let root = CumulativeRecord {
            state_id: baseline_state_id.clone(),
            parent_state_id: None,
            patch_set_id: None,
            patch_set: None,
            rationale: "baseline (materialized codebase)".to_string(),
            fitness: baseline_fitness,
        };
        let mut states = BTreeMap::new();
        states.insert(baseline_state_id.clone(), baseline);
        Ok(Self {
            records: vec![root],
            states,
            baseline_state_id,
            policy,
        })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn baseline_state_id(&self) -> &str {
        &self.baseline_state_id
    }

    pub fn records(&self) -> &[CumulativeRecord] {
        &self.records
    }

    pub fn get(&self, state_id: &str) -> Option<&CumulativeRecord> {
        self.records.iter().find(|record| record.state_id == state_id)
    }

    pub fn materialized_root(&self, state_id: &str) -> Option<&Path> {
        self.states.get(state_id).map(CandidateState::root)
    }

    pub fn best(&self) -> Option<&CumulativeRecord> {
        use std::cmp::Ordering;
        self.records.iter().max_by(|left, right| {
            if left.fitness.is_better_than(&right.fitness) {
                Ordering::Greater
            } else if right.fitness.is_better_than(&left.fitness) {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
    }

    /// Open-ended deterministic parent selection over materializable accepted states.
    pub fn select_parent_id(&self, rng: &mut Rng) -> Option<&str> {
        if self.records.is_empty() {
            return None;
        }
        let weights: Vec<f64> = self
            .records
            .iter()
            .map(|record| {
                let children = self
                    .records
                    .iter()
                    .filter(|child| child.parent_state_id.as_deref() == Some(record.state_id.as_str()))
                    .count() as f64;
                let quality = if record.fitness.compiles { 1.0 } else { 0.1 };
                quality * (1.0 / (1.0 + children)) + f64::EPSILON
            })
            .collect();
        let total: f64 = weights.iter().sum();
        let mut pick = rng.uniform() * total;
        for (record, weight) in self.records.iter().zip(weights.iter()) {
            pick -= weight;
            if pick <= 0.0 {
                return Some(record.state_id.as_str());
            }
        }
        self.records.last().map(|record| record.state_id.as_str())
    }

    /// Materialize and evaluate `patch_set` from the exact selected parent.
    /// Rejected descendants are dropped and therefore cannot mutate accepted ancestors.
    pub fn evaluate_child<E: Evaluator>(
        &mut self,
        parent_state_id: &str,
        patch_set: PatchSet,
        rationale: impl Into<String>,
        evaluator: &E,
        require_all_green: bool,
    ) -> Result<CumulativeOutcome, CumulativeArchiveError> {
        let parent_record = self
            .get(parent_state_id)
            .cloned()
            .ok_or_else(|| CumulativeArchiveError::UnknownParent(parent_state_id.to_string()))?;
        let parent = self
            .states
            .get(parent_state_id)
            .ok_or_else(|| CumulativeArchiveError::UnmaterializedParent(parent_state_id.to_string()))?;
        let patch_set_id = patch_set.identity()?;
        let child = parent.child(&patch_set)?;
        let state_id = child.state_id().to_string();
        let fitness = evaluator.evaluate(child.root()).map_err(|error| {
            CumulativeArchiveError::Evaluation(error.to_string())
        })?;
        let accepted = (!require_all_green || fitness.all_green())
            && fitness.is_better_than(&parent_record.fitness);

        if accepted {
            if self.states.contains_key(&state_id) {
                return Err(CumulativeArchiveError::DuplicateState(state_id));
            }
            let record = CumulativeRecord {
                state_id: state_id.clone(),
                parent_state_id: Some(parent_state_id.to_string()),
                patch_set_id: Some(patch_set_id.clone()),
                patch_set: Some(patch_set),
                rationale: rationale.into(),
                fitness: fitness.clone(),
            };
            self.states.insert(state_id.clone(), child);
            self.records.push(record);
        }

        Ok(CumulativeOutcome {
            state_id,
            parent_state_id: parent_state_id.to_string(),
            patch_set_id,
            accepted,
            fitness,
        })
    }

    /// Promote the complete accepted state into `live_root`.
    ///
    /// Promotion is fail-closed against stale live-tree changes: the current live
    /// tree must still equal `expected_live_state_id`. A complete backup is
    /// written before mutation; an I/O failure during replacement triggers a
    /// best-effort rollback from that backup and returns an error.
    pub fn promote_complete_state(
        &self,
        state_id: &str,
        live_root: &Path,
        expected_live_state_id: &str,
        backup_root: &Path,
    ) -> Result<PromotionReceipt, CumulativeArchiveError> {
        let accepted = self
            .states
            .get(state_id)
            .ok_or_else(|| CumulativeArchiveError::UnknownState(state_id.to_string()))?;

        let observed = CandidateState::baseline(live_root, self.policy)?;
        let observed_id = observed.state_id().to_string();
        if observed_id != expected_live_state_id {
            return Err(CumulativeArchiveError::StaleLiveTree {
                expected: expected_live_state_id.to_string(),
                observed: observed_id,
            });
        }

        std::fs::create_dir_all(backup_root).map_err(io_err)?;
        let backup_id = sha256_hex(&format!(
            "rsi-p3-promotion-v1|{expected_live_state_id}|{state_id}"
        ));
        let backup_path = backup_root.join(backup_id);
        if backup_path.exists() {
            return Err(CumulativeArchiveError::BackupExists(backup_path));
        }
        std::fs::create_dir_all(&backup_path).map_err(io_err)?;
        if let Err(error) = copy_mutable_tree(live_root, &backup_path) {
            let _ = std::fs::remove_dir_all(&backup_path);
            return Err(error);
        }

        if let Err(error) = replace_mutable_tree(live_root, accepted.root()) {
            let rollback = replace_mutable_tree(live_root, &backup_path);
            return match rollback {
                Ok(()) => Err(error),
                Err(rollback_error) => Err(CumulativeArchiveError::RollbackFailed {
                    promotion_error: error.to_string(),
                    rollback_error: rollback_error.to_string(),
                }),
            };
        }

        let promoted = CandidateState::baseline(live_root, self.policy)?;
        if promoted.state_id() != state_id {
            let rollback = replace_mutable_tree(live_root, &backup_path);
            return match rollback {
                Ok(()) => Err(CumulativeArchiveError::PromotionVerification {
                    expected: state_id.to_string(),
                    observed: promoted.state_id().to_string(),
                }),
                Err(rollback_error) => Err(CumulativeArchiveError::RollbackFailed {
                    promotion_error: format!(
                        "promotion verification mismatch: expected {state_id}, observed {}",
                        promoted.state_id()
                    ),
                    rollback_error: rollback_error.to_string(),
                }),
            };
        }

        Ok(PromotionReceipt {
            from_live_state_id: expected_live_state_id.to_string(),
            promoted_state_id: state_id.to_string(),
            backup_path,
        })
    }

    /// Deterministic replay metadata. Runtime temp paths are never serialized.
    pub fn to_json(&self) -> String {
        let mut root = Json::obj();
        root.set(
            "schema_version",
            Json::Num(CUMULATIVE_ARCHIVE_SCHEMA_VERSION as f64),
        )
        .set(
            "baseline_state_id",
            Json::Str(self.baseline_state_id.clone()),
        )
        .set("max_files", Json::Num(self.policy.max_files as f64))
        .set("max_bytes", Json::Num(self.policy.max_bytes as f64))
        .set(
            "records",
            Json::Arr(self.records.iter().map(record_to_json).collect()),
        );
        root.to_string()
    }

    /// Rebuild every accepted state from the supplied baseline and verify all
    /// serialized state/PatchSet identities while replaying.
    pub fn from_json(
        raw: &str,
        source_root: &Path,
    ) -> Result<Self, CumulativeArchiveError> {
        let json = Json::parse(raw).map_err(CumulativeArchiveError::Decode)?;
        let version = json
            .get("schema_version")
            .and_then(Json::as_u64)
            .ok_or_else(|| CumulativeArchiveError::Decode("missing schema_version".to_string()))?;
        if version != CUMULATIVE_ARCHIVE_SCHEMA_VERSION {
            return Err(CumulativeArchiveError::Decode(format!(
                "unsupported cumulative archive schema {version}"
            )));
        }
        let max_files = json
            .get("max_files")
            .and_then(Json::as_u64)
            .ok_or_else(|| CumulativeArchiveError::Decode("missing max_files".to_string()))?;
        let max_bytes = json
            .get("max_bytes")
            .and_then(Json::as_u64)
            .ok_or_else(|| CumulativeArchiveError::Decode("missing max_bytes".to_string()))?;
        let policy = CandidateStoragePolicy::new(max_files, max_bytes)?;
        let expected_baseline = required_str(&json, "baseline_state_id")?.to_string();
        let serialized = json
            .get("records")
            .and_then(Json::as_array)
            .ok_or_else(|| CumulativeArchiveError::Decode("missing records array".to_string()))?;
        if serialized.is_empty() {
            return Err(CumulativeArchiveError::Decode(
                "cumulative archive requires a root record".to_string(),
            ));
        }

        let baseline = CandidateState::baseline(source_root, policy)?;
        if baseline.state_id() != expected_baseline {
            return Err(CumulativeArchiveError::BaselineMismatch {
                expected: expected_baseline,
                observed: baseline.state_id().to_string(),
            });
        }

        let root_record = record_from_json(&serialized[0])?;
        if root_record.parent_state_id.is_some()
            || root_record.patch_set.is_some()
            || root_record.patch_set_id.is_some()
            || root_record.state_id != baseline.state_id()
        {
            return Err(CumulativeArchiveError::Decode(
                "invalid cumulative archive root record".to_string(),
            ));
        }

        let baseline_state_id = baseline.state_id().to_string();
        let mut states = BTreeMap::new();
        states.insert(baseline_state_id.clone(), baseline);
        let mut records = vec![root_record];

        for json_record in serialized.iter().skip(1) {
            let record = record_from_json(json_record)?;
            let parent_id = record.parent_state_id.as_deref().ok_or_else(|| {
                CumulativeArchiveError::Decode("non-root record missing parent".to_string())
            })?;
            let patch_set = record.patch_set.as_ref().ok_or_else(|| {
                CumulativeArchiveError::Decode("non-root record missing PatchSet".to_string())
            })?;
            let expected_patch_id = record.patch_set_id.as_deref().ok_or_else(|| {
                CumulativeArchiveError::Decode("non-root record missing PatchSet id".to_string())
            })?;
            let actual_patch_id = patch_set.identity()?;
            if actual_patch_id != expected_patch_id {
                return Err(CumulativeArchiveError::PatchIdentityMismatch {
                    expected: expected_patch_id.to_string(),
                    observed: actual_patch_id,
                });
            }
            let parent = states
                .get(parent_id)
                .ok_or_else(|| CumulativeArchiveError::UnknownParent(parent_id.to_string()))?;
            let child = parent.child(patch_set)?;
            if child.state_id() != record.state_id {
                return Err(CumulativeArchiveError::StateIdentityMismatch {
                    expected: record.state_id.clone(),
                    observed: child.state_id().to_string(),
                });
            }
            if states.insert(record.state_id.clone(), child).is_some() {
                return Err(CumulativeArchiveError::DuplicateState(record.state_id));
            }
            records.push(record);
        }

        Ok(Self {
            records,
            states,
            baseline_state_id,
            policy,
        })
    }
}

#[derive(Debug)]
pub enum CumulativeArchiveError {
    Candidate(CandidateStateError),
    PatchSet(PatchSetError),
    Evaluation(String),
    Decode(String),
    UnknownParent(String),
    UnmaterializedParent(String),
    UnknownState(String),
    DuplicateState(String),
    BaselineMismatch { expected: String, observed: String },
    PatchIdentityMismatch { expected: String, observed: String },
    StateIdentityMismatch { expected: String, observed: String },
    StaleLiveTree { expected: String, observed: String },
    BackupExists(PathBuf),
    PromotionVerification { expected: String, observed: String },
    RollbackFailed { promotion_error: String, rollback_error: String },
    Io(String),
}

impl fmt::Display for CumulativeArchiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Candidate(error) => write!(f, "candidate state error: {error}"),
            Self::PatchSet(error) => write!(f, "PatchSet error: {error}"),
            Self::Evaluation(error) => write!(f, "candidate evaluation failed: {error}"),
            Self::Decode(error) => write!(f, "cumulative archive decode error: {error}"),
            Self::UnknownParent(id) => write!(f, "unknown cumulative parent state: {id}"),
            Self::UnmaterializedParent(id) => write!(f, "parent state is not materialized: {id}"),
            Self::UnknownState(id) => write!(f, "unknown accepted state: {id}"),
            Self::DuplicateState(id) => write!(f, "duplicate accepted state identity: {id}"),
            Self::BaselineMismatch { expected, observed } => write!(
                f,
                "baseline identity mismatch: expected {expected}, observed {observed}"
            ),
            Self::PatchIdentityMismatch { expected, observed } => write!(
                f,
                "PatchSet identity mismatch: expected {expected}, observed {observed}"
            ),
            Self::StateIdentityMismatch { expected, observed } => write!(
                f,
                "state identity mismatch: expected {expected}, observed {observed}"
            ),
            Self::StaleLiveTree { expected, observed } => write!(
                f,
                "stale live tree: expected {expected}, observed {observed}"
            ),
            Self::BackupExists(path) => write!(f, "promotion backup already exists: {}", path.display()),
            Self::PromotionVerification { expected, observed } => write!(
                f,
                "promotion verification failed: expected {expected}, observed {observed}"
            ),
            Self::RollbackFailed {
                promotion_error,
                rollback_error,
            } => write!(
                f,
                "promotion failed ({promotion_error}) and rollback failed ({rollback_error})"
            ),
            Self::Io(error) => write!(f, "cumulative archive I/O error: {error}"),
        }
    }
}

impl std::error::Error for CumulativeArchiveError {}

impl From<CandidateStateError> for CumulativeArchiveError {
    fn from(value: CandidateStateError) -> Self {
        Self::Candidate(value)
    }
}

impl From<PatchSetError> for CumulativeArchiveError {
    fn from(value: PatchSetError) -> Self {
        Self::PatchSet(value)
    }
}

fn record_to_json(record: &CumulativeRecord) -> Json {
    let mut json = Json::obj();
    json.set("state_id", Json::Str(record.state_id.clone()))
        .set(
            "parent_state_id",
            record
                .parent_state_id
                .clone()
                .map(Json::Str)
                .unwrap_or(Json::Null),
        )
        .set(
            "patch_set_id",
            record
                .patch_set_id
                .clone()
                .map(Json::Str)
                .unwrap_or(Json::Null),
        )
        .set(
            "patch_set",
            record
                .patch_set
                .as_ref()
                .map(patch_set_to_json)
                .unwrap_or(Json::Null),
        )
        .set("rationale", Json::Str(record.rationale.clone()))
        .set("fitness", fitness_to_json(&record.fitness));
    json
}

fn record_from_json(json: &Json) -> Result<CumulativeRecord, CumulativeArchiveError> {
    Ok(CumulativeRecord {
        state_id: required_str(json, "state_id")?.to_string(),
        parent_state_id: json
            .get("parent_state_id")
            .and_then(Json::as_str)
            .map(str::to_string),
        patch_set_id: json
            .get("patch_set_id")
            .and_then(Json::as_str)
            .map(str::to_string),
        patch_set: match json.get("patch_set") {
            Some(Json::Null) | None => None,
            Some(value) => Some(patch_set_from_json(value)?),
        },
        rationale: json
            .get("rationale")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string(),
        fitness: fitness_from_json(
            json.get("fitness")
                .ok_or_else(|| CumulativeArchiveError::Decode("missing fitness".to_string()))?,
        )?,
    })
}

fn patch_set_to_json(patch_set: &PatchSet) -> Json {
    Json::Arr(
        patch_set
            .operations()
            .iter()
            .map(|operation| {
                let mut json = Json::obj();
                match operation {
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

fn patch_set_from_json(json: &Json) -> Result<PatchSet, CumulativeArchiveError> {
    let array = json
        .as_array()
        .ok_or_else(|| CumulativeArchiveError::Decode("PatchSet must be an array".to_string()))?;
    let mut operations = Vec::with_capacity(array.len());
    for operation in array {
        let kind = required_str(operation, "kind")?;
        let path = required_str(operation, "path")?;
        let parsed = match kind {
            "modify_exact" => FileOperation::modify_exact(
                path,
                required_str(operation, "expected")?,
                required_str(operation, "replacement")?,
            ),
            "create" => FileOperation::create(path, required_str(operation, "content")?),
            "delete" => FileOperation::delete(
                path,
                required_str(operation, "expected_sha256")?,
            ),
            other => {
                return Err(CumulativeArchiveError::Decode(format!(
                    "unsupported PatchSet operation: {other}"
                )))
            }
        };
        operations.push(parsed);
    }
    Ok(PatchSet::new(operations)?)
}

fn fitness_to_json(fitness: &Fitness) -> Json {
    let mut json = Json::obj();
    json.set("compiles", Json::Bool(fitness.compiles))
        .set("tests_passed", Json::Num(fitness.tests_passed as f64))
        .set("tests_failed", Json::Num(fitness.tests_failed as f64))
        .set(
            "score",
            if fitness.score.is_finite() {
                Json::Num(fitness.score)
            } else {
                Json::Null
            },
        )
        .set("notes", Json::Str(fitness.notes.clone()));
    json
}

fn fitness_from_json(json: &Json) -> Result<Fitness, CumulativeArchiveError> {
    let compiles = json
        .get("compiles")
        .and_then(Json::as_bool)
        .ok_or_else(|| CumulativeArchiveError::Decode("missing fitness.compiles".to_string()))?;
    let score = match json.get("score") {
        Some(Json::Num(value)) if value.is_finite() => *value,
        Some(Json::Null) if !compiles => f64::NEG_INFINITY,
        Some(Json::Null) => 0.0,
        _ => {
            return Err(CumulativeArchiveError::Decode(
                "invalid fitness.score".to_string(),
            ))
        }
    };
    Ok(Fitness {
        compiles,
        tests_passed: json
            .get("tests_passed")
            .and_then(Json::as_u64)
            .unwrap_or(0) as u32,
        tests_failed: json
            .get("tests_failed")
            .and_then(Json::as_u64)
            .unwrap_or(0) as u32,
        score,
        notes: json
            .get("notes")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string(),
    })
}

fn required_str<'a>(json: &'a Json, key: &str) -> Result<&'a str, CumulativeArchiveError> {
    json.get(key)
        .and_then(Json::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| CumulativeArchiveError::Decode(format!("missing or empty {key}")))
}

fn mutable_files(root: &Path) -> Result<BTreeMap<String, PathBuf>, CumulativeArchiveError> {
    let mut files = BTreeMap::new();
    collect_mutable_files(root, root, &mut files)?;
    Ok(files)
}

fn collect_mutable_files(
    base: &Path,
    current: &Path,
    files: &mut BTreeMap<String, PathBuf>,
) -> Result<(), CumulativeArchiveError> {
    for entry in std::fs::read_dir(current).map_err(io_err)? {
        let entry = entry.map_err(io_err)?;
        let file_type = entry.file_type().map_err(io_err)?;
        let name = entry.file_name();
        let name_lossy = name.to_string_lossy();
        if file_type.is_dir() {
            if SKIP_DIRS.contains(&name_lossy.as_ref()) {
                continue;
            }
            collect_mutable_files(base, &entry.path(), files)?;
        } else if file_type.is_file() {
            let relative = entry
                .path()
                .strip_prefix(base)
                .map_err(|error| CumulativeArchiveError::Io(error.to_string()))?
                .components()
                .map(|component| {
                    component.as_os_str().to_str().ok_or_else(|| {
                        CumulativeArchiveError::Io("non-UTF-8 promotion path".to_string())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
                .join("/");
            files.insert(relative, entry.path());
        }
    }
    Ok(())
}

fn copy_mutable_tree(source: &Path, destination: &Path) -> Result<(), CumulativeArchiveError> {
    for (relative, path) in mutable_files(source)? {
        let target = destination.join(relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(io_err)?;
        }
        std::fs::copy(path, target).map_err(io_err)?;
    }
    Ok(())
}

fn replace_mutable_tree(live_root: &Path, source_root: &Path) -> Result<(), CumulativeArchiveError> {
    let source = mutable_files(source_root)?;
    let live = mutable_files(live_root)?;
    let source_names: BTreeSet<&str> = source.keys().map(String::as_str).collect();

    for (relative, live_path) in live {
        if !source_names.contains(relative.as_str()) {
            std::fs::remove_file(live_path).map_err(io_err)?;
        }
    }
    for (relative, source_path) in source {
        let target = live_root.join(relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(io_err)?;
        }
        std::fs::copy(source_path, target).map_err(io_err)?;
    }
    Ok(())
}

fn io_err(error: std::io::Error) -> CumulativeArchiveError {
    CumulativeArchiveError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dgm::ClosureEvaluator;
    use crate::sha256::sha256;
    use std::fmt::Write as _;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQ: AtomicU64 = AtomicU64::new(0);

    fn fresh_root(tag: &str) -> PathBuf {
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rsi-p3-2-{tag}-{}-{seq}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/a.rs"), "a=0\n").unwrap();
        std::fs::write(root.join("src/b.rs"), "b=0\n").unwrap();
        root
    }

    fn policy() -> CandidateStoragePolicy {
        CandidateStoragePolicy::new(64, 64 * 1024).unwrap()
    }

    fn fit(score: f64) -> Fitness {
        Fitness {
            compiles: true,
            tests_passed: 1,
            tests_failed: 0,
            score,
            notes: String::new(),
        }
    }

    fn score_eval() -> ClosureEvaluator<impl Fn(&Path) -> Fitness> {
        ClosureEvaluator::new(|root: &Path| {
            let a = std::fs::read_to_string(root.join("src/a.rs")).unwrap();
            let b = std::fs::read_to_string(root.join("src/b.rs")).unwrap();
            let mut score = 0.0;
            if a.contains("a=1") {
                score += 1.0;
            }
            if b.contains("b=1") {
                score += 1.0;
            }
            fit(score)
        })
    }

    fn delete_hash(bytes: &[u8]) -> String {
        let mut out = String::new();
        for byte in sha256(bytes) {
            let _ = write!(out, "{byte:02x}");
        }
        out
    }

    #[test]
    fn child_is_evaluated_from_exact_materialized_parent() {
        let live = fresh_root("cumulative");
        let mut archive = CumulativeArchive::new(&live, fit(0.0), policy()).unwrap();
        let root_id = archive.baseline_state_id().to_string();
        let first = archive
            .evaluate_child(
                &root_id,
                PatchSet::new(vec![FileOperation::modify_exact("src/a.rs", "a=0", "a=1")])
                    .unwrap(),
                "A",
                &score_eval(),
                true,
            )
            .unwrap();
        assert!(first.accepted);
        let second = archive
            .evaluate_child(
                &first.state_id,
                PatchSet::new(vec![FileOperation::modify_exact("src/b.rs", "b=0", "b=1")])
                    .unwrap(),
                "B",
                &score_eval(),
                true,
            )
            .unwrap();
        assert!(second.accepted);
        let root = archive.materialized_root(&second.state_id).unwrap();
        assert_eq!(std::fs::read_to_string(root.join("src/a.rs")).unwrap(), "a=1\n");
        assert_eq!(std::fs::read_to_string(root.join("src/b.rs")).unwrap(), "b=1\n");
        let _ = std::fs::remove_dir_all(live);
    }

    #[test]
    fn rejected_descendant_does_not_mutate_or_enter_archive() {
        let live = fresh_root("reject");
        let mut archive = CumulativeArchive::new(&live, fit(0.0), policy()).unwrap();
        let root_id = archive.baseline_state_id().to_string();
        let evaluator = ClosureEvaluator::new(|_: &Path| fit(-1.0));
        let outcome = archive
            .evaluate_child(
                &root_id,
                PatchSet::new(vec![FileOperation::modify_exact("src/a.rs", "a=0", "a=1")])
                    .unwrap(),
                "regression",
                &evaluator,
                true,
            )
            .unwrap();
        assert!(!outcome.accepted);
        assert_eq!(archive.len(), 1);
        let root = archive.materialized_root(&root_id).unwrap();
        assert_eq!(std::fs::read_to_string(root.join("src/a.rs")).unwrap(), "a=0\n");
        let _ = std::fs::remove_dir_all(live);
    }

    #[test]
    fn serialization_replays_exact_lineage_and_identities() {
        let live = fresh_root("replay");
        let mut archive = CumulativeArchive::new(&live, fit(0.0), policy()).unwrap();
        let root_id = archive.baseline_state_id().to_string();
        let first = archive
            .evaluate_child(
                &root_id,
                PatchSet::new(vec![FileOperation::modify_exact("src/a.rs", "a=0", "a=1")])
                    .unwrap(),
                "A",
                &score_eval(),
                true,
            )
            .unwrap();
        let second = archive
            .evaluate_child(
                &first.state_id,
                PatchSet::new(vec![FileOperation::modify_exact("src/b.rs", "b=0", "b=1")])
                    .unwrap(),
                "B",
                &score_eval(),
                true,
            )
            .unwrap();
        let encoded = archive.to_json();
        let replayed = CumulativeArchive::from_json(&encoded, &live).unwrap();
        assert_eq!(replayed.to_json(), encoded);
        assert_eq!(replayed.records().last().unwrap().state_id, second.state_id);
        let replayed_root = replayed.materialized_root(&second.state_id).unwrap();
        assert_eq!(std::fs::read_to_string(replayed_root.join("src/a.rs")).unwrap(), "a=1\n");
        assert_eq!(std::fs::read_to_string(replayed_root.join("src/b.rs")).unwrap(), "b=1\n");
        let _ = std::fs::remove_dir_all(live);
    }

    #[test]
    fn complete_promotion_keeps_ancestor_edits_and_rejects_stale_live_tree() {
        let live = fresh_root("promote");
        let backups = std::env::temp_dir().join(format!(
            "rsi-p3-2-backups-{}-{}",
            std::process::id(),
            TEST_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let mut archive = CumulativeArchive::new(&live, fit(0.0), policy()).unwrap();
        let baseline_id = archive.baseline_state_id().to_string();
        let first = archive
            .evaluate_child(
                &baseline_id,
                PatchSet::new(vec![FileOperation::modify_exact("src/a.rs", "a=0", "a=1")])
                    .unwrap(),
                "A",
                &score_eval(),
                true,
            )
            .unwrap();
        let second = archive
            .evaluate_child(
                &first.state_id,
                PatchSet::new(vec![FileOperation::modify_exact("src/b.rs", "b=0", "b=1")])
                    .unwrap(),
                "B",
                &score_eval(),
                true,
            )
            .unwrap();

        let receipt = archive
            .promote_complete_state(&second.state_id, &live, &baseline_id, &backups)
            .unwrap();
        assert!(receipt.backup_path.exists());
        assert_eq!(std::fs::read_to_string(live.join("src/a.rs")).unwrap(), "a=1\n");
        assert_eq!(std::fs::read_to_string(live.join("src/b.rs")).unwrap(), "b=1\n");

        let stale_live = fresh_root("stale");
        let stale_backups = std::env::temp_dir().join(format!(
            "rsi-p3-2-stale-backups-{}-{}",
            std::process::id(),
            TEST_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(stale_live.join("src/a.rs"), "a=external\n").unwrap();
        let error = archive
            .promote_complete_state(&second.state_id, &stale_live, &baseline_id, &stale_backups)
            .unwrap_err();
        assert!(matches!(error, CumulativeArchiveError::StaleLiveTree { .. }));
        assert_eq!(
            std::fs::read_to_string(stale_live.join("src/a.rs")).unwrap(),
            "a=external\n"
        );

        let _ = std::fs::remove_dir_all(live);
        let _ = std::fs::remove_dir_all(backups);
        let _ = std::fs::remove_dir_all(stale_live);
        let _ = std::fs::remove_dir_all(stale_backups);
    }

    #[test]
    fn create_delete_roundtrip_is_replayable() {
        let live = fresh_root("ops");
        std::fs::write(live.join("src/delete.rs"), "delete me").unwrap();
        let mut archive = CumulativeArchive::new(&live, fit(0.0), policy()).unwrap();
        let root_id = archive.baseline_state_id().to_string();
        let evaluator = ClosureEvaluator::new(|root: &Path| {
            let good = root.join("src/new.rs").exists() && !root.join("src/delete.rs").exists();
            fit(if good { 1.0 } else { -1.0 })
        });
        let patch = PatchSet::new(vec![
            FileOperation::create("src/new.rs", "new\n"),
            FileOperation::delete("src/delete.rs", delete_hash(b"delete me")),
        ])
        .unwrap();
        let outcome = archive
            .evaluate_child(&root_id, patch, "create/delete", &evaluator, true)
            .unwrap();
        assert!(outcome.accepted);
        let encoded = archive.to_json();
        let replayed = CumulativeArchive::from_json(&encoded, &live).unwrap();
        assert_eq!(replayed.to_json(), encoded);
        let root = replayed.materialized_root(&outcome.state_id).unwrap();
        assert!(root.join("src/new.rs").exists());
        assert!(!root.join("src/delete.rs").exists());
        let _ = std::fs::remove_dir_all(live);
    }

    #[test]
    fn parent_selection_is_seed_deterministic() {
        let live = fresh_root("select");
        let archive = CumulativeArchive::new(&live, fit(0.0), policy()).unwrap();
        let mut left = Rng::new(42);
        let mut right = Rng::new(42);
        assert_eq!(
            archive.select_parent_id(&mut left),
            archive.select_parent_id(&mut right)
        );
        let _ = std::fs::remove_dir_all(live);
    }
}
