//! Bounded proposal envelope for multi-file engineering candidates.
//!
//! P2.2 keeps proposal safety independent from the eventual cumulative DGM
//! lineage semantics (P3). A proposal is accepted at this boundary only after
//! its PatchSet is structurally valid, allowlisted, and within explicit operation
//! and touched-byte budgets measured against the current workspace.

use crate::json::Json;
use crate::patchset::{FileOperation, PatchSet, PatchSetError};
use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

/// Hard limits applied before a multi-file proposal can enter evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProposalBudget {
    pub max_operations: usize,
    pub max_touched_bytes: u64,
}

impl ProposalBudget {
    pub fn new(max_operations: usize, max_touched_bytes: u64) -> Self {
        Self {
            max_operations,
            max_touched_bytes,
        }
    }

    pub fn validate(
        &self,
        patch_set: &PatchSet,
        workspace_root: &Path,
        allowed_paths: &[String],
    ) -> Result<ProposalCost, ProposalError> {
        if self.max_operations == 0 {
            return Err(ProposalError::InvalidBudget(
                "max_operations must be greater than zero".to_string(),
            ));
        }
        if self.max_touched_bytes == 0 {
            return Err(ProposalError::InvalidBudget(
                "max_touched_bytes must be greater than zero".to_string(),
            ));
        }
        patch_set.validate().map_err(ProposalError::PatchSet)?;
        if patch_set.len() > self.max_operations {
            return Err(ProposalError::OperationBudgetExceeded {
                actual: patch_set.len(),
                limit: self.max_operations,
            });
        }

        let allowlist: BTreeSet<&str> = allowed_paths.iter().map(String::as_str).collect();
        let mut touched = 0u64;
        for operation in patch_set.operations() {
            let path = operation.path();
            if !allowlist.contains(path) {
                return Err(ProposalError::PathNotAllowed(path.to_string()));
            }
            let cost = operation_cost(operation, workspace_root)?;
            touched = touched.checked_add(cost).ok_or(
                ProposalError::TouchedByteBudgetExceeded {
                    actual: u64::MAX,
                    limit: self.max_touched_bytes,
                },
            )?;
            if touched > self.max_touched_bytes {
                return Err(ProposalError::TouchedByteBudgetExceeded {
                    actual: touched,
                    limit: self.max_touched_bytes,
                });
            }
        }

        Ok(ProposalCost {
            operations: patch_set.len(),
            touched_bytes: touched,
        })
    }
}

/// Measured cost of an accepted proposal envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProposalCost {
    pub operations: usize,
    pub touched_bytes: u64,
}

/// A rationale plus a validated multi-file candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedProposal {
    pub patch_set: PatchSet,
    pub rationale: String,
    pub cost: ProposalCost,
}

impl BoundedProposal {
    pub fn new(
        patch_set: PatchSet,
        rationale: impl Into<String>,
        budget: ProposalBudget,
        workspace_root: &Path,
        allowed_paths: &[String],
    ) -> Result<Self, ProposalError> {
        let cost = budget.validate(&patch_set, workspace_root, allowed_paths)?;
        Ok(Self {
            patch_set,
            rationale: rationale.into(),
            cost,
        })
    }

    /// Parse a strict JSON proposal envelope and immediately apply all hard
    /// budgets. Unknown/malformed operation kinds fail closed.
    ///
    /// Schema:
    /// `{"operations":[{"kind":"modify_exact","path":"...","expected":"...","replacement":"..."}],"rationale":"..."}`
    pub fn from_json_envelope(
        raw: &str,
        budget: ProposalBudget,
        workspace_root: &Path,
        allowed_paths: &[String],
    ) -> Result<Self, ProposalError> {
        let json = Json::parse(raw).map_err(ProposalError::Envelope)?;
        let ops = json
            .get("operations")
            .and_then(Json::as_array)
            .ok_or_else(|| ProposalError::Envelope("missing operations array".to_string()))?;
        let mut operations = Vec::with_capacity(ops.len());
        for op in ops {
            let kind = required_str(op, "kind")?;
            let path = required_str(op, "path")?;
            let parsed = match kind {
                "modify_exact" => FileOperation::modify_exact(
                    path,
                    required_str(op, "expected")?,
                    required_str(op, "replacement")?,
                ),
                "create" => FileOperation::create(path, required_str(op, "content")?),
                "delete" => {
                    FileOperation::delete(path, required_str(op, "expected_sha256")?)
                }
                other => {
                    return Err(ProposalError::Envelope(format!(
                        "unsupported operation kind: {other}"
                    )))
                }
            };
            operations.push(parsed);
        }
        let patch_set = PatchSet::new(operations).map_err(ProposalError::PatchSet)?;
        let rationale = json
            .get("rationale")
            .and_then(Json::as_str)
            .unwrap_or("multi-file proposal");
        Self::new(patch_set, rationale, budget, workspace_root, allowed_paths)
    }

    /// Compatibility entry point for the historical one-file proposal API.
    pub fn from_legacy_patch(
        patch: &crate::dgm::Patch,
        rationale: impl Into<String>,
        budget: ProposalBudget,
        workspace_root: &Path,
        allowed_paths: &[String],
    ) -> Result<Self, ProposalError> {
        Self::new(
            PatchSet::from(patch),
            rationale,
            budget,
            workspace_root,
            allowed_paths,
        )
    }
}

#[derive(Debug)]
pub enum ProposalError {
    PatchSet(PatchSetError),
    Envelope(String),
    InvalidBudget(String),
    PathNotAllowed(String),
    OperationBudgetExceeded { actual: usize, limit: usize },
    TouchedByteBudgetExceeded { actual: u64, limit: u64 },
    Workspace(String),
}

impl fmt::Display for ProposalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PatchSet(e) => write!(f, "invalid PatchSet: {e}"),
            Self::Envelope(msg) => write!(f, "invalid proposal envelope: {msg}"),
            Self::InvalidBudget(msg) => write!(f, "invalid proposal budget: {msg}"),
            Self::PathNotAllowed(path) => write!(f, "proposal target outside allowlist: {path}"),
            Self::OperationBudgetExceeded { actual, limit } => {
                write!(f, "proposal has {actual} operations; limit is {limit}")
            }
            Self::TouchedByteBudgetExceeded { actual, limit } => {
                write!(f, "proposal touches {actual} bytes; limit is {limit}")
            }
            Self::Workspace(msg) => write!(f, "proposal workspace error: {msg}"),
        }
    }
}

impl std::error::Error for ProposalError {}

fn required_str<'a>(json: &'a Json, key: &str) -> Result<&'a str, ProposalError> {
    json.get(key)
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ProposalError::Envelope(format!("missing or empty {key}")))
}

fn operation_cost(operation: &FileOperation, root: &Path) -> Result<u64, ProposalError> {
    match operation {
        FileOperation::ModifyExact {
            path,
            expected,
            replacement,
        } => {
            let content = std::fs::read_to_string(root.join(path))
                .map_err(|e| ProposalError::Workspace(format!("read {path}: {e}")))?;
            let occurrences = content.matches(expected).count();
            if occurrences != 1 {
                return Err(ProposalError::Workspace(format!(
                    "ModifyExact expected text must occur once in {path}; got {occurrences}"
                )));
            }
            Ok(expected.len() as u64 + replacement.len() as u64)
        }
        FileOperation::Create { path, content } => {
            if root.join(path).exists() {
                return Err(ProposalError::Workspace(format!(
                    "Create refuses existing path {path}"
                )));
            }
            Ok(content.len() as u64)
        }
        FileOperation::Delete {
            path,
            expected_sha256: _,
        } => {
            let metadata = std::fs::metadata(root.join(path))
                .map_err(|e| ProposalError::Workspace(format!("metadata {path}: {e}")))?;
            if !metadata.is_file() {
                return Err(ProposalError::Workspace(format!(
                    "Delete target is not a file: {path}"
                )));
            }
            Ok(metadata.len())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::sha256;
    use std::fmt::Write as _;
    use std::path::PathBuf;

    fn hex(bytes: &[u8]) -> String {
        let mut s = String::new();
        for b in sha256(bytes) {
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    fn workspace() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "rsi-p2-2-proposal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/a.rs"), "let a = 1;\n").unwrap();
        std::fs::write(root.join("src/b.rs"), "remove-me").unwrap();
        root
    }

    #[test]
    fn accepts_bounded_multi_file_proposal() {
        let root = workspace();
        let set = PatchSet::new(vec![
            FileOperation::modify_exact("src/a.rs", "1", "2"),
            FileOperation::create("src/new.rs", "fn new() {}\n"),
            FileOperation::delete("src/b.rs", hex(b"remove-me")),
        ])
        .unwrap();
        let allowed = vec!["src/a.rs".into(), "src/new.rs".into(), "src/b.rs".into()];
        let proposal = BoundedProposal::new(
            set,
            "three-file migration",
            ProposalBudget::new(3, 64),
            &root,
            &allowed,
        )
        .unwrap();
        assert_eq!(proposal.cost.operations, 3);
        assert!(proposal.cost.touched_bytes > 0);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn strict_json_envelope_parses_multiple_operation_kinds() {
        let root = workspace();
        let delete_hash = hex(b"remove-me");
        let raw = format!(
            "{{\"operations\":[{{\"kind\":\"modify_exact\",\"path\":\"src/a.rs\",\"expected\":\"1\",\"replacement\":\"2\"}},{{\"kind\":\"create\",\"path\":\"src/new.rs\",\"content\":\"new\"}},{{\"kind\":\"delete\",\"path\":\"src/b.rs\",\"expected_sha256\":\"{delete_hash}\"}}],\"rationale\":\"bounded migration\"}}"
        );
        let allowed = vec!["src/a.rs".into(), "src/new.rs".into(), "src/b.rs".into()];
        let proposal = BoundedProposal::from_json_envelope(
            &raw,
            ProposalBudget::new(3, 64),
            &root,
            &allowed,
        )
        .unwrap();
        assert_eq!(proposal.patch_set.len(), 3);
        assert_eq!(proposal.rationale, "bounded migration");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_operation_and_byte_budget_overflow() {
        let root = workspace();
        let allowed = vec!["src/a.rs".into(), "src/new.rs".into()];
        let set = PatchSet::new(vec![
            FileOperation::modify_exact("src/a.rs", "1", "2222"),
            FileOperation::create("src/new.rs", "payload"),
        ])
        .unwrap();
        assert!(matches!(
            ProposalBudget::new(1, 1024).validate(&set, &root, &allowed),
            Err(ProposalError::OperationBudgetExceeded { .. })
        ));
        assert!(matches!(
            ProposalBudget::new(2, 4).validate(&set, &root, &allowed),
            Err(ProposalError::TouchedByteBudgetExceeded { .. })
        ));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_non_allowlisted_operation() {
        let root = workspace();
        let set = PatchSet::new(vec![FileOperation::modify_exact("src/a.rs", "1", "2")]).unwrap();
        let err = ProposalBudget::new(1, 128)
            .validate(&set, &root, &["src/other.rs".into()])
            .unwrap_err();
        assert!(matches!(err, ProposalError::PathNotAllowed(_)));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn delete_budget_charges_actual_file_size() {
        let root = workspace();
        let set = PatchSet::new(vec![FileOperation::delete("src/b.rs", hex(b"remove-me"))]).unwrap();
        let allowed = vec!["src/b.rs".into()];
        assert!(matches!(
            ProposalBudget::new(1, 3).validate(&set, &root, &allowed),
            Err(ProposalError::TouchedByteBudgetExceeded { .. })
        ));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_patch_enters_same_bounded_envelope() {
        let root = workspace();
        let old = crate::dgm::Patch::new("src/a.rs", "1", "2");
        let proposal = BoundedProposal::from_legacy_patch(
            &old,
            "legacy",
            ProposalBudget::new(1, 16),
            &root,
            &["src/a.rs".into()],
        )
        .unwrap();
        assert_eq!(proposal.patch_set.len(), 1);
        let _ = std::fs::remove_dir_all(root);
    }
}
