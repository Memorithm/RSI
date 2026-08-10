//! Atomic multi-file patch representation for RSI engineering candidates.
//!
//! `PatchSet` is deliberately independent from the LLM proposer and DGM archive
//! wiring. P2.1 freezes the safe representation and candidate-materialization
//! semantics; P2.2 will teach proposers and trajectories to emit it.

use crate::sha256::{sha256, sha256_hex};
use std::collections::BTreeSet;
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const SKIP_DIRS: &[&str] = &["target", ".git", "node_modules", ".rsi_backups"];
static SNAPSHOT_SEQ: AtomicU64 = AtomicU64::new(0);

/// One deterministic filesystem operation in a [`PatchSet`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileOperation {
    /// Replace one and only one exact textual occurrence in an existing file.
    ModifyExact {
        path: String,
        expected: String,
        replacement: String,
    },
    /// Create a new UTF-8 file. Existing paths are never overwritten.
    Create { path: String, content: String },
    /// Delete an existing file only if its complete SHA-256 matches.
    Delete {
        path: String,
        expected_sha256: String,
    },
}

impl FileOperation {
    pub fn modify_exact(
        path: impl Into<String>,
        expected: impl Into<String>,
        replacement: impl Into<String>,
    ) -> Self {
        Self::ModifyExact {
            path: path.into(),
            expected: expected.into(),
            replacement: replacement.into(),
        }
    }

    pub fn create(path: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Create {
            path: path.into(),
            content: content.into(),
        }
    }

    pub fn delete(path: impl Into<String>, expected_sha256: impl Into<String>) -> Self {
        Self::Delete {
            path: path.into(),
            expected_sha256: expected_sha256.into(),
        }
    }

    pub fn path(&self) -> &str {
        match self {
            Self::ModifyExact { path, .. }
            | Self::Create { path, .. }
            | Self::Delete { path, .. } => path,
        }
    }
}

/// A bounded-by-caller, ordered group of file operations that is validated and
/// materialized as one candidate state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchSet {
    operations: Vec<FileOperation>,
}

impl PatchSet {
    pub fn new(operations: Vec<FileOperation>) -> Result<Self, PatchSetError> {
        let set = Self { operations };
        set.validate()?;
        Ok(set)
    }

    pub fn operations(&self) -> &[FileOperation] {
        &self.operations
    }

    pub fn len(&self) -> usize {
        self.operations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Validate path safety, operation-local invariants, and cross-operation
    /// conflicts without touching the filesystem.
    pub fn validate(&self) -> Result<(), PatchSetError> {
        if self.operations.is_empty() {
            return Err(PatchSetError::Empty);
        }

        let mut seen = BTreeSet::new();
        for op in &self.operations {
            let normalized = normalize_relative(op.path())?;
            if !seen.insert(normalized.clone()) {
                return Err(PatchSetError::Conflict(normalized));
            }

            match op {
                FileOperation::ModifyExact {
                    expected,
                    replacement,
                    ..
                } => {
                    if expected.is_empty() {
                        return Err(PatchSetError::InvalidOperation(
                            "ModifyExact expected text must not be empty".to_string(),
                        ));
                    }
                    if expected == replacement {
                        return Err(PatchSetError::InvalidOperation(
                            "ModifyExact must change content".to_string(),
                        ));
                    }
                }
                FileOperation::Create { .. } => {}
                FileOperation::Delete {
                    expected_sha256, ..
                } => {
                    if expected_sha256.len() != 64
                        || !expected_sha256
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    {
                        return Err(PatchSetError::InvalidOperation(
                            "Delete expected_sha256 must be 64 lowercase hex characters"
                                .to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Stable SHA-256 identity over operation order and length-delimited fields.
    pub fn identity(&self) -> Result<String, PatchSetError> {
        self.validate()?;
        let mut material = String::from("rsi-patchset-v1|");
        push_field(&mut material, &self.operations.len().to_string());
        for op in &self.operations {
            let path = normalize_relative(op.path())?;
            match op {
                FileOperation::ModifyExact {
                    expected,
                    replacement,
                    ..
                } => {
                    push_field(&mut material, "modify-exact");
                    push_field(&mut material, &path);
                    push_field(&mut material, expected);
                    push_field(&mut material, replacement);
                }
                FileOperation::Create { content, .. } => {
                    push_field(&mut material, "create");
                    push_field(&mut material, &path);
                    push_field(&mut material, content);
                }
                FileOperation::Delete {
                    expected_sha256, ..
                } => {
                    push_field(&mut material, "delete");
                    push_field(&mut material, &path);
                    push_field(&mut material, expected_sha256);
                }
            }
        }
        Ok(sha256_hex(&material))
    }

    /// Materialize the entire set into a fresh disposable candidate snapshot.
    ///
    /// The source tree is never mutated. All operations are preflighted against
    /// the copied candidate before any mutation occurs. If any operation fails,
    /// no candidate is returned and the temporary tree is removed, giving the
    /// DGM an atomic visible result: either the full PatchSet exists or none of
    /// it does.
    pub fn materialize(&self, source_root: &Path) -> Result<PatchSetSnapshot, PatchSetError> {
        self.validate()?;
        if !source_root.is_dir() {
            return Err(PatchSetError::Io(format!(
                "source root is not a directory: {}",
                source_root.display()
            )));
        }

        let tmp = unique_tmp_dir();
        let root = tmp.join("workspace");
        std::fs::create_dir_all(&root).map_err(io_err)?;
        if let Err(e) = copy_tree(source_root, &root) {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(e);
        }

        let paths = match self.preflight(&root) {
            Ok(paths) => paths,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&tmp);
                return Err(e);
            }
        };

        if let Err(e) = self.commit(&root, &paths) {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(e);
        }

        Ok(PatchSetSnapshot { root, tmp })
    }

    fn preflight(&self, root: &Path) -> Result<Vec<PathBuf>, PatchSetError> {
        let mut paths = Vec::with_capacity(self.operations.len());
        for op in &self.operations {
            let rel = normalize_relative(op.path())?;
            let target = root.join(&rel);
            match op {
                FileOperation::ModifyExact { expected, .. } => {
                    let content = std::fs::read_to_string(&target).map_err(|e| {
                        PatchSetError::Apply(format!("read {}: {e}", target.display()))
                    })?;
                    match content.matches(expected).count() {
                        1 => {}
                        0 => {
                            return Err(PatchSetError::Apply(format!(
                                "expected text not found in {rel}"
                            )))
                        }
                        n => {
                            return Err(PatchSetError::Apply(format!(
                                "expected text is ambiguous in {rel} ({n} occurrences)"
                            )))
                        }
                    }
                }
                FileOperation::Create { .. } => {
                    if target.exists() {
                        return Err(PatchSetError::Apply(format!(
                            "Create refuses to overwrite {rel}"
                        )));
                    }
                }
                FileOperation::Delete {
                    expected_sha256, ..
                } => {
                    let bytes = std::fs::read(&target).map_err(|e| {
                        PatchSetError::Apply(format!("read {}: {e}", target.display()))
                    })?;
                    let actual = bytes_sha256_hex(&bytes);
                    if &actual != expected_sha256 {
                        return Err(PatchSetError::Apply(format!(
                            "Delete hash mismatch for {rel}: expected {expected_sha256}, got {actual}"
                        )));
                    }
                }
            }
            paths.push(target);
        }
        Ok(paths)
    }

    fn commit(&self, root: &Path, paths: &[PathBuf]) -> Result<(), PatchSetError> {
        for (op, target) in self.operations.iter().zip(paths) {
            match op {
                FileOperation::ModifyExact {
                    expected,
                    replacement,
                    ..
                } => {
                    let content = std::fs::read_to_string(target).map_err(io_err)?;
                    let patched = content.replacen(expected, replacement, 1);
                    std::fs::write(target, patched).map_err(io_err)?;
                }
                FileOperation::Create { content, .. } => {
                    if let Some(parent) = target.parent() {
                        if parent != root {
                            std::fs::create_dir_all(parent).map_err(io_err)?;
                        }
                    }
                    std::fs::write(target, content).map_err(io_err)?;
                }
                FileOperation::Delete { .. } => {
                    std::fs::remove_file(target).map_err(io_err)?;
                }
            }
        }
        Ok(())
    }
}

/// Compatibility conversion for the historical one-file DGM patch API.
impl From<crate::dgm::Patch> for PatchSet {
    fn from(patch: crate::dgm::Patch) -> Self {
        Self {
            operations: vec![FileOperation::modify_exact(
                patch.target,
                patch.find,
                patch.replace,
            )],
        }
    }
}

impl From<&crate::dgm::Patch> for PatchSet {
    fn from(patch: &crate::dgm::Patch) -> Self {
        Self {
            operations: vec![FileOperation::modify_exact(
                patch.target.clone(),
                patch.find.clone(),
                patch.replace.clone(),
            )],
        }
    }
}

/// Disposable fully materialized PatchSet candidate.
pub struct PatchSetSnapshot {
    root: PathBuf,
    tmp: PathBuf,
}

impl PatchSetSnapshot {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn resolve(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }
}

impl Drop for PatchSetSnapshot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.tmp);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchSetError {
    Empty,
    UnsafePath(String),
    Conflict(String),
    InvalidOperation(String),
    Apply(String),
    Io(String),
}

impl fmt::Display for PatchSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "PatchSet must contain at least one operation"),
            Self::UnsafePath(p) => write!(f, "unsafe workspace-relative path: {p}"),
            Self::Conflict(p) => write!(f, "conflicting operations target the same path: {p}"),
            Self::InvalidOperation(msg) => write!(f, "invalid operation: {msg}"),
            Self::Apply(msg) => write!(f, "could not materialize PatchSet: {msg}"),
            Self::Io(msg) => write!(f, "PatchSet I/O error: {msg}"),
        }
    }
}

impl std::error::Error for PatchSetError {}

fn normalize_relative(input: &str) -> Result<String, PatchSetError> {
    if input.is_empty() {
        return Err(PatchSetError::UnsafePath(input.to_string()));
    }
    let path = Path::new(input);
    if path.is_absolute() {
        return Err(PatchSetError::UnsafePath(input.to_string()));
    }

    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(s) => {
                let s = s
                    .to_str()
                    .ok_or_else(|| PatchSetError::UnsafePath(input.to_string()))?;
                if s.is_empty() {
                    return Err(PatchSetError::UnsafePath(input.to_string()));
                }
                parts.push(s);
            }
            Component::CurDir | Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(PatchSetError::UnsafePath(input.to_string()));
            }
        }
    }
    if parts.is_empty() {
        return Err(PatchSetError::UnsafePath(input.to_string()));
    }
    Ok(parts.join("/"))
}

fn push_field(out: &mut String, value: &str) {
    out.push_str(&value.len().to_string());
    out.push(':');
    out.push_str(value);
    out.push('|');
}

fn bytes_sha256_hex(bytes: &[u8]) -> String {
    let digest = sha256(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn io_err(e: std::io::Error) -> PatchSetError {
    PatchSetError::Io(e.to_string())
}

fn unique_tmp_dir() -> PathBuf {
    let seq = SNAPSHOT_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("rsi-patchset-{pid}-{seq}-{nanos}"))
}

fn copy_tree(src: &Path, dst: &Path) -> Result<(), PatchSetError> {
    for entry in std::fs::read_dir(src).map_err(io_err)? {
        let entry = entry.map_err(io_err)?;
        let ty = entry.file_type().map_err(io_err)?;
        let name = entry.file_name();
        let name_lossy = name.to_string_lossy();
        if ty.is_dir() {
            if SKIP_DIRS.contains(&name_lossy.as_ref()) {
                continue;
            }
            let target = dst.join(&name);
            std::fs::create_dir_all(&target).map_err(io_err)?;
            copy_tree(&entry.path(), &target)?;
        } else if ty.is_file() {
            std::fs::copy(entry.path(), dst.join(&name)).map_err(io_err)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PathBuf {
        let root = unique_tmp_dir();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "alpha\nbeta\n").unwrap();
        std::fs::write(root.join("delete.txt"), "obsolete").unwrap();
        root
    }

    #[test]
    fn rejects_traversal_absolute_and_duplicate_targets() {
        assert!(PatchSet::new(vec![FileOperation::create("../escape", "x")]).is_err());
        assert!(PatchSet::new(vec![FileOperation::create("/tmp/escape", "x")]).is_err());
        assert!(PatchSet::new(vec![
            FileOperation::create("same.txt", "a"),
            FileOperation::delete(
                "same.txt",
                "0000000000000000000000000000000000000000000000000000000000000000",
            ),
        ])
        .is_err());
    }

    #[test]
    fn identity_is_deterministic_and_order_sensitive() {
        let a = PatchSet::new(vec![
            FileOperation::modify_exact("src/lib.rs", "alpha", "gamma"),
            FileOperation::create("new.txt", "new"),
        ])
        .unwrap();
        let b = a.clone();
        assert_eq!(a.identity().unwrap(), b.identity().unwrap());

        let reversed = PatchSet::new(vec![
            FileOperation::create("new.txt", "new"),
            FileOperation::modify_exact("src/lib.rs", "alpha", "gamma"),
        ])
        .unwrap();
        assert_ne!(a.identity().unwrap(), reversed.identity().unwrap());
    }

    #[test]
    fn materializes_modify_create_delete_as_one_candidate() {
        let source = fixture();
        let delete_hash = bytes_sha256_hex(b"obsolete");
        let set = PatchSet::new(vec![
            FileOperation::modify_exact("src/lib.rs", "alpha", "gamma"),
            FileOperation::create("nested/new.txt", "created"),
            FileOperation::delete("delete.txt", delete_hash),
        ])
        .unwrap();

        let candidate = set.materialize(&source).unwrap();
        assert_eq!(
            std::fs::read_to_string(candidate.resolve("src/lib.rs")).unwrap(),
            "gamma\nbeta\n"
        );
        assert_eq!(
            std::fs::read_to_string(candidate.resolve("nested/new.txt")).unwrap(),
            "created"
        );
        assert!(!candidate.resolve("delete.txt").exists());

        // The source is immutable: the candidate is an atomic separate state.
        assert_eq!(
            std::fs::read_to_string(source.join("src/lib.rs")).unwrap(),
            "alpha\nbeta\n"
        );
        assert!(source.join("delete.txt").exists());
        assert!(!source.join("nested/new.txt").exists());
        let _ = std::fs::remove_dir_all(source);
    }

    #[test]
    fn failed_operation_yields_no_partial_source_mutation() {
        let source = fixture();
        let set = PatchSet::new(vec![
            FileOperation::modify_exact("src/lib.rs", "alpha", "gamma"),
            FileOperation::create("src/lib.rs", "must not overwrite"),
        ]);
        assert!(set.is_err(), "same-path conflicts must fail before materialization");

        let set = PatchSet::new(vec![
            FileOperation::modify_exact("src/lib.rs", "alpha", "gamma"),
            FileOperation::delete(
                "delete.txt",
                "0000000000000000000000000000000000000000000000000000000000000000",
            ),
        ])
        .unwrap();
        assert!(set.materialize(&source).is_err());
        assert_eq!(
            std::fs::read_to_string(source.join("src/lib.rs")).unwrap(),
            "alpha\nbeta\n"
        );
        assert_eq!(
            std::fs::read_to_string(source.join("delete.txt")).unwrap(),
            "obsolete"
        );
        let _ = std::fs::remove_dir_all(source);
    }

    #[test]
    fn modify_exact_rejects_ambiguous_matches() {
        let source = fixture();
        std::fs::write(source.join("src/lib.rs"), "same\nsame\n").unwrap();
        let set = PatchSet::new(vec![FileOperation::modify_exact(
            "src/lib.rs",
            "same",
            "other",
        )])
        .unwrap();
        assert!(set.materialize(&source).is_err());
        assert_eq!(
            std::fs::read_to_string(source.join("src/lib.rs")).unwrap(),
            "same\nsame\n"
        );
        let _ = std::fs::remove_dir_all(source);
    }

    #[test]
    fn historical_patch_converts_without_semantic_loss() {
        let old = crate::dgm::Patch::new("src/lib.rs", "alpha", "gamma");
        let set = PatchSet::from(&old);
        assert_eq!(set.operations().len(), 1);
        assert_eq!(
            set.operations()[0],
            FileOperation::modify_exact("src/lib.rs", "alpha", "gamma")
        );
    }
}
