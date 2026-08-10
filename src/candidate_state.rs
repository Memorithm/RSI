//! Immutable cumulative engineering candidate states.
//!
//! P3.1 freezes the materialization contract only. Parent selection, archive
//! policy and live-tree promotion remain P3.2 responsibilities.

use crate::patchset::{PatchSet, PatchSetError, PatchSetSnapshot};
use crate::sha256::sha256;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const SKIP_DIRS: &[&str] = &["target", ".git", "node_modules", ".rsi_backups"];
static STATE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Hard storage limits applied to every materialized candidate tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateStoragePolicy {
    pub max_files: u64,
    pub max_bytes: u64,
}

impl CandidateStoragePolicy {
    pub fn new(max_files: u64, max_bytes: u64) -> Result<Self, CandidateStateError> {
        if max_files == 0 {
            return Err(CandidateStateError::InvalidPolicy(
                "max_files must be greater than zero".to_string(),
            ));
        }
        if max_bytes == 0 {
            return Err(CandidateStateError::InvalidPolicy(
                "max_bytes must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            max_files,
            max_bytes,
        })
    }

    fn enforce(&self, root: &Path) -> Result<TreeUsage, CandidateStateError> {
        let usage = measure_tree(root)?;
        if usage.files > self.max_files {
            return Err(CandidateStateError::StorageLimitExceeded {
                kind: "files",
                actual: usage.files,
                limit: self.max_files,
            });
        }
        if usage.bytes > self.max_bytes {
            return Err(CandidateStateError::StorageLimitExceeded {
                kind: "bytes",
                actual: usage.bytes,
                limit: self.max_bytes,
            });
        }
        Ok(usage)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeUsage {
    pub files: u64,
    pub bytes: u64,
}

/// One fully materialized, immutable engineering state.
///
/// `state_id` is the deterministic SHA-256 identity of the complete file tree,
/// independent of temporary paths, process IDs, wall-clock time, parent choice
/// or the sequence used to reach the tree. Lineage is retained separately via
/// `parent_state_id` and `patch_set_id`.
pub struct CandidateState {
    storage: CandidateStorage,
    state_id: String,
    parent_state_id: Option<String>,
    patch_set_id: Option<String>,
    usage: TreeUsage,
    policy: CandidateStoragePolicy,
}

enum CandidateStorage {
    Baseline { root: PathBuf, tmp: PathBuf },
    Derived(PatchSetSnapshot),
}

impl CandidateState {
    /// Snapshot a live/source tree into an isolated immutable baseline state.
    pub fn baseline(
        source_root: &Path,
        policy: CandidateStoragePolicy,
    ) -> Result<Self, CandidateStateError> {
        if !source_root.is_dir() {
            return Err(CandidateStateError::Io(format!(
                "baseline root is not a directory: {}",
                source_root.display()
            )));
        }
        policy.enforce(source_root)?;

        let tmp = unique_tmp_dir();
        let root = tmp.join("workspace");
        std::fs::create_dir_all(&root).map_err(io_err)?;
        if let Err(error) = copy_tree(source_root, &root) {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(error);
        }

        let usage = match policy.enforce(&root) {
            Ok(usage) => usage,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&tmp);
                return Err(error);
            }
        };
        let state_id = match tree_identity(&root) {
            Ok(identity) => identity,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&tmp);
                return Err(error);
            }
        };

        Ok(Self {
            storage: CandidateStorage::Baseline { root, tmp },
            state_id,
            parent_state_id: None,
            patch_set_id: None,
            usage,
            policy,
        })
    }

    /// Materialize `patch_set` on top of this exact state, never on the live
    /// baseline. The parent remains untouched regardless of success or failure.
    pub fn child(&self, patch_set: &PatchSet) -> Result<Self, CandidateStateError> {
        let patch_set_id = patch_set.identity().map_err(CandidateStateError::PatchSet)?;
        let snapshot = patch_set
            .materialize(self.root())
            .map_err(CandidateStateError::PatchSet)?;
        let usage = self.policy.enforce(snapshot.root())?;
        let state_id = tree_identity(snapshot.root())?;

        Ok(Self {
            storage: CandidateStorage::Derived(snapshot),
            state_id,
            parent_state_id: Some(self.state_id.clone()),
            patch_set_id: Some(patch_set_id),
            usage,
            policy: self.policy,
        })
    }

    pub fn root(&self) -> &Path {
        match &self.storage {
            CandidateStorage::Baseline { root, .. } => root,
            CandidateStorage::Derived(snapshot) => snapshot.root(),
        }
    }

    pub fn state_id(&self) -> &str {
        &self.state_id
    }

    pub fn parent_state_id(&self) -> Option<&str> {
        self.parent_state_id.as_deref()
    }

    pub fn patch_set_id(&self) -> Option<&str> {
        self.patch_set_id.as_deref()
    }

    pub fn usage(&self) -> TreeUsage {
        self.usage
    }

    pub fn policy(&self) -> CandidateStoragePolicy {
        self.policy
    }
}

impl Drop for CandidateStorage {
    fn drop(&mut self) {
        if let Self::Baseline { tmp, .. } = self {
            let _ = std::fs::remove_dir_all(tmp);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateStateError {
    InvalidPolicy(String),
    StorageLimitExceeded {
        kind: &'static str,
        actual: u64,
        limit: u64,
    },
    PatchSet(PatchSetError),
    Io(String),
}

impl fmt::Display for CandidateStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy(message) => write!(f, "invalid candidate storage policy: {message}"),
            Self::StorageLimitExceeded {
                kind,
                actual,
                limit,
            } => write!(
                f,
                "candidate storage {kind} limit exceeded: {actual} > {limit}"
            ),
            Self::PatchSet(error) => write!(f, "candidate PatchSet error: {error}"),
            Self::Io(message) => write!(f, "candidate state I/O error: {message}"),
        }
    }
}

impl std::error::Error for CandidateStateError {}

fn measure_tree(root: &Path) -> Result<TreeUsage, CandidateStateError> {
    let mut usage = TreeUsage { files: 0, bytes: 0 };
    measure_tree_into(root, &mut usage)?;
    Ok(usage)
}

fn measure_tree_into(root: &Path, usage: &mut TreeUsage) -> Result<(), CandidateStateError> {
    for entry in std::fs::read_dir(root).map_err(io_err)? {
        let entry = entry.map_err(io_err)?;
        let file_type = entry.file_type().map_err(io_err)?;
        let name = entry.file_name();
        let name_lossy = name.to_string_lossy();
        if file_type.is_dir() {
            if SKIP_DIRS.contains(&name_lossy.as_ref()) {
                continue;
            }
            measure_tree_into(&entry.path(), usage)?;
        } else if file_type.is_file() {
            let len = entry.metadata().map_err(io_err)?.len();
            usage.files = usage.files.checked_add(1).ok_or_else(|| {
                CandidateStateError::Io("file-count overflow while measuring tree".to_string())
            })?;
            usage.bytes = usage.bytes.checked_add(len).ok_or_else(|| {
                CandidateStateError::Io("byte-count overflow while measuring tree".to_string())
            })?;
        }
    }
    Ok(())
}

fn tree_identity(root: &Path) -> Result<String, CandidateStateError> {
    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));

    let mut material = Vec::new();
    material.extend_from_slice(b"rsi-candidate-tree-v1\0");
    append_bytes(&mut material, &(files.len() as u64).to_be_bytes());
    for (relative, path) in files {
        append_bytes(&mut material, relative.as_bytes());
        let bytes = std::fs::read(path).map_err(io_err)?;
        append_bytes(&mut material, &bytes);
    }
    Ok(hex_digest(sha256(&material)))
}

fn collect_files(
    base: &Path,
    current: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<(), CandidateStateError> {
    for entry in std::fs::read_dir(current).map_err(io_err)? {
        let entry = entry.map_err(io_err)?;
        let file_type = entry.file_type().map_err(io_err)?;
        let name = entry.file_name();
        let name_lossy = name.to_string_lossy();
        if file_type.is_dir() {
            if SKIP_DIRS.contains(&name_lossy.as_ref()) {
                continue;
            }
            collect_files(base, &entry.path(), files)?;
        } else if file_type.is_file() {
            let relative = entry
                .path()
                .strip_prefix(base)
                .map_err(|error| CandidateStateError::Io(error.to_string()))?
                .components()
                .map(|component| {
                    component.as_os_str().to_str().ok_or_else(|| {
                        CandidateStateError::Io("non-UTF-8 candidate path".to_string())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
                .join("/");
            files.push((relative, entry.path()));
        }
    }
    Ok(())
}

fn append_bytes(out: &mut Vec<u8>, value: &[u8]) {
    out.extend_from_slice(&(value.len() as u64).to_be_bytes());
    out.extend_from_slice(value);
}

fn hex_digest(digest: [u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), CandidateStateError> {
    for entry in std::fs::read_dir(source).map_err(io_err)? {
        let entry = entry.map_err(io_err)?;
        let file_type = entry.file_type().map_err(io_err)?;
        let name = entry.file_name();
        let name_lossy = name.to_string_lossy();
        if file_type.is_dir() {
            if SKIP_DIRS.contains(&name_lossy.as_ref()) {
                continue;
            }
            let target = destination.join(&name);
            std::fs::create_dir_all(&target).map_err(io_err)?;
            copy_tree(&entry.path(), &target)?;
        } else if file_type.is_file() {
            std::fs::copy(entry.path(), destination.join(&name)).map_err(io_err)?;
        }
    }
    Ok(())
}

fn unique_tmp_dir() -> PathBuf {
    let sequence = STATE_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rsi-candidate-state-{}-{sequence}",
        std::process::id()
    ))
}

fn io_err(error: std::io::Error) -> CandidateStateError {
    CandidateStateError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patchset::FileOperation;

    fn fixture() -> PathBuf {
        let root = unique_tmp_dir();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/a.rs"), "a=0\n").unwrap();
        std::fs::write(root.join("src/b.rs"), "b=0\n").unwrap();
        root
    }

    fn policy() -> CandidateStoragePolicy {
        CandidateStoragePolicy::new(32, 4096).unwrap()
    }

    #[test]
    fn baseline_identity_is_content_deterministic() {
        let source = fixture();
        let first = CandidateState::baseline(&source, policy()).unwrap();
        let second = CandidateState::baseline(&source, policy()).unwrap();
        assert_eq!(first.state_id(), second.state_id());
        assert_eq!(first.parent_state_id(), None);
        assert_eq!(first.patch_set_id(), None);
        let _ = std::fs::remove_dir_all(source);
    }

    #[test]
    fn grandchild_materializes_all_accepted_ancestor_changes() {
        let source = fixture();
        let baseline = CandidateState::baseline(&source, policy()).unwrap();
        let patch_a = PatchSet::new(vec![FileOperation::modify_exact("src/a.rs", "a=0", "a=1")])
            .unwrap();
        let patch_b = PatchSet::new(vec![FileOperation::modify_exact("src/b.rs", "b=0", "b=1")])
            .unwrap();

        let after_a = baseline.child(&patch_a).unwrap();
        let after_a_b = after_a.child(&patch_b).unwrap();
        let baseline_b = baseline.child(&patch_b).unwrap();

        assert_ne!(after_a_b.state_id(), baseline_b.state_id());
        assert_eq!(
            std::fs::read_to_string(after_a_b.root().join("src/a.rs")).unwrap(),
            "a=1\n"
        );
        assert_eq!(
            std::fs::read_to_string(after_a_b.root().join("src/b.rs")).unwrap(),
            "b=1\n"
        );
        assert_eq!(after_a_b.parent_state_id(), Some(after_a.state_id()));
        assert_eq!(
            after_a_b.patch_set_id(),
            Some(patch_b.identity().unwrap().as_str())
        );
        let _ = std::fs::remove_dir_all(source);
    }

    #[test]
    fn rejected_child_never_mutates_parent() {
        let source = fixture();
        let baseline = CandidateState::baseline(&source, policy()).unwrap();
        let invalid_for_parent = PatchSet::new(vec![FileOperation::modify_exact(
            "src/a.rs",
            "missing",
            "replacement",
        )])
        .unwrap();
        assert!(baseline.child(&invalid_for_parent).is_err());
        assert_eq!(
            std::fs::read_to_string(baseline.root().join("src/a.rs")).unwrap(),
            "a=0\n"
        );
        let _ = std::fs::remove_dir_all(source);
    }

    #[test]
    fn storage_policy_rejects_oversized_materialized_tree() {
        let source = fixture();
        let too_small = CandidateStoragePolicy::new(1, 4096).unwrap();
        assert!(matches!(
            CandidateState::baseline(&source, too_small),
            Err(CandidateStateError::StorageLimitExceeded { kind: "files", .. })
        ));
        let _ = std::fs::remove_dir_all(source);
    }
}
