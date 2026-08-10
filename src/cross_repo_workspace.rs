//! Isolated multi-repository engineering workspaces for P5.1.
//!
//! Network resolution and arbitrary command execution stay outside this core.
//! Callers provide local Git checkouts; the workspace verifies immutable HEADs,
//! copies only tracked files into isolated roots, applies PatchSets only to
//! explicitly authorized repository roles, and records temporary Cargo patches.

use crate::candidate_state::{CandidateState, CandidateStateError, CandidateStoragePolicy};
use crate::compatibility::{CompatibilityError, CompatibilitySet, RepositoryRevision};
use crate::patchset::{PatchSet, PatchSetError};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static WORKSPACE_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalRepositorySource {
    pub repository: String,
    pub role: String,
    pub revision: String,
    pub root: PathBuf,
}

impl LocalRepositorySource {
    pub fn new(
        repository: impl Into<String>,
        role: impl Into<String>,
        revision: impl Into<String>,
        root: impl Into<PathBuf>,
    ) -> Result<Self, CrossRepoWorkspaceError> {
        let checked = RepositoryRevision::new(repository, revision, role)
            .map_err(CrossRepoWorkspaceError::Compatibility)?;
        Ok(Self {
            repository: checked.repository,
            role: checked.role,
            revision: checked.revision,
            root: root.into(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CargoPatchOverride {
    pub consumer_role: String,
    pub source: String,
    pub package: String,
    pub provider_role: String,
    pub provider_path: String,
}

impl CargoPatchOverride {
    pub fn new(
        consumer_role: impl Into<String>,
        source: impl Into<String>,
        package: impl Into<String>,
        provider_role: impl Into<String>,
        provider_path: impl Into<String>,
    ) -> Result<Self, CrossRepoWorkspaceError> {
        let consumer_role = checked_text("consumer_role", consumer_role.into())?;
        let source = checked_text("source", source.into())?;
        let package = checked_text("package", package.into())?;
        let provider_role = checked_text("provider_role", provider_role.into())?;
        let provider_path = normalize_provider_path(&provider_path.into())?;
        Ok(Self {
            consumer_role,
            source,
            package,
            provider_role,
            provider_path,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossRepoWorkspacePolicy {
    pub max_repositories: usize,
    pub max_total_bytes: u64,
    pub per_repository: CandidateStoragePolicy,
}

impl CrossRepoWorkspacePolicy {
    pub fn new(
        max_repositories: usize,
        max_total_bytes: u64,
        per_repository: CandidateStoragePolicy,
    ) -> Result<Self, CrossRepoWorkspaceError> {
        if max_repositories == 0 || max_total_bytes == 0 {
            return Err(CrossRepoWorkspaceError::InvalidPolicy(
                "repository and byte limits must be greater than zero".into(),
            ));
        }
        Ok(Self {
            max_repositories,
            max_total_bytes,
            per_repository,
        })
    }
}

struct MaterializedRepository {
    repository: String,
    role: String,
    revision: String,
    state: CandidateState,
    patch_set_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveRepositoryState {
    pub repository: String,
    pub role: String,
    pub base_revision: String,
    pub state_id: String,
    pub patch_set_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveCargoOverride {
    pub consumer_role: String,
    pub source: String,
    pub package: String,
    pub provider_role: String,
    pub provider_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossRepoWorkspaceResult {
    pub compatibility_fingerprint: String,
    pub repositories: Vec<EffectiveRepositoryState>,
    pub overrides: Vec<EffectiveCargoOverride>,
}

pub struct CrossRepoWorkspace {
    compatibility: CompatibilitySet,
    repositories: BTreeMap<String, MaterializedRepository>,
    allowed_patch_roles: BTreeSet<String>,
    overrides: Vec<CargoPatchOverride>,
    override_configs: BTreeMap<String, PathBuf>,
    control_root: PathBuf,
    policy: CrossRepoWorkspacePolicy,
}

impl CrossRepoWorkspace {
    pub fn materialize(
        compatibility: CompatibilitySet,
        sources: Vec<LocalRepositorySource>,
        allowed_patch_roles: Vec<String>,
        mut overrides: Vec<CargoPatchOverride>,
        policy: CrossRepoWorkspacePolicy,
    ) -> Result<Self, CrossRepoWorkspaceError> {
        if compatibility.revisions().len() > policy.max_repositories {
            return Err(CrossRepoWorkspaceError::RepositoryLimitExceeded {
                actual: compatibility.revisions().len(),
                limit: policy.max_repositories,
            });
        }
        if sources.len() != compatibility.revisions().len() {
            return Err(CrossRepoWorkspaceError::SourceSetMismatch(format!(
                "expected {} sources, got {}",
                compatibility.revisions().len(),
                sources.len()
            )));
        }

        let mut source_by_key = BTreeMap::new();
        for source in sources {
            let key = (source.repository.clone(), source.role.clone());
            if source_by_key.insert(key.clone(), source).is_some() {
                return Err(CrossRepoWorkspaceError::SourceSetMismatch(format!(
                    "duplicate source for {}/{}",
                    key.0, key.1
                )));
            }
        }

        let mut roles = BTreeSet::new();
        for revision in compatibility.revisions() {
            if !roles.insert(revision.role.clone()) {
                return Err(CrossRepoWorkspaceError::DuplicateRole(
                    revision.role.clone(),
                ));
            }
            let key = (revision.repository.clone(), revision.role.clone());
            let source = source_by_key.get(&key).ok_or_else(|| {
                CrossRepoWorkspaceError::SourceSetMismatch(format!(
                    "missing source for {}/{}",
                    revision.repository, revision.role
                ))
            })?;
            if source.revision != revision.revision {
                return Err(CrossRepoWorkspaceError::RevisionMismatch {
                    role: revision.role.clone(),
                    expected: revision.revision.clone(),
                    actual: source.revision.clone(),
                });
            }
        }

        let allowed_patch_roles: BTreeSet<_> = allowed_patch_roles.into_iter().collect();
        for role in &allowed_patch_roles {
            if !roles.contains(role) {
                return Err(CrossRepoWorkspaceError::UnknownRole(role.clone()));
            }
        }

        overrides.sort();
        for pair in overrides.windows(2) {
            if pair[0].consumer_role == pair[1].consumer_role
                && pair[0].source == pair[1].source
                && pair[0].package == pair[1].package
            {
                return Err(CrossRepoWorkspaceError::DuplicateOverride {
                    consumer_role: pair[0].consumer_role.clone(),
                    source: pair[0].source.clone(),
                    package: pair[0].package.clone(),
                });
            }
        }
        for rule in &overrides {
            if !roles.contains(&rule.consumer_role) || !roles.contains(&rule.provider_role) {
                return Err(CrossRepoWorkspaceError::UnknownRole(format!(
                    "{} or {}",
                    rule.consumer_role, rule.provider_role
                )));
            }
        }

        let control_root = unique_control_root();
        std::fs::create_dir_all(&control_root).map_err(io_error)?;
        let staged_root = control_root.join("staged");
        std::fs::create_dir_all(&staged_root).map_err(io_error)?;

        let build = (|| {
            let mut repositories = BTreeMap::new();
            let mut total_bytes = 0u64;
            for (index, revision) in compatibility.revisions().iter().enumerate() {
                let key = (revision.repository.clone(), revision.role.clone());
                let source = source_by_key.get(&key).expect("source set validated");
                verify_git_source(source, revision)?;
                let stage = staged_root.join(index.to_string());
                materialize_tracked_tree(&source.root, &stage)?;
                let state = CandidateState::baseline(&stage, policy.per_repository)
                    .map_err(CrossRepoWorkspaceError::CandidateState)?;
                let _ = std::fs::remove_dir_all(&stage);
                total_bytes = total_bytes.checked_add(state.usage().bytes).ok_or(
                    CrossRepoWorkspaceError::TotalByteLimitExceeded {
                        actual: u64::MAX,
                        limit: policy.max_total_bytes,
                    },
                )?;
                if total_bytes > policy.max_total_bytes {
                    return Err(CrossRepoWorkspaceError::TotalByteLimitExceeded {
                        actual: total_bytes,
                        limit: policy.max_total_bytes,
                    });
                }
                repositories.insert(
                    revision.role.clone(),
                    MaterializedRepository {
                        repository: revision.repository.clone(),
                        role: revision.role.clone(),
                        revision: revision.revision.clone(),
                        state,
                        patch_set_ids: Vec::new(),
                    },
                );
            }
            let mut workspace = Self {
                compatibility,
                repositories,
                allowed_patch_roles,
                overrides,
                override_configs: BTreeMap::new(),
                control_root: control_root.clone(),
                policy,
            };
            workspace.refresh_override_configs()?;
            Ok(workspace)
        })();

        if build.is_err() {
            let _ = std::fs::remove_dir_all(&control_root);
        }
        build
    }

    pub fn root_for_role(&self, role: &str) -> Option<&Path> {
        self.repositories.get(role).map(|repo| repo.state.root())
    }

    pub fn cargo_override_config(&self, consumer_role: &str) -> Option<&Path> {
        self.override_configs
            .get(consumer_role)
            .map(PathBuf::as_path)
    }

    pub fn apply_patch(
        &mut self,
        role: &str,
        patch_set: &PatchSet,
    ) -> Result<String, CrossRepoWorkspaceError> {
        if !self.allowed_patch_roles.contains(role) {
            return Err(CrossRepoWorkspaceError::PatchRoleNotAllowed(
                role.to_string(),
            ));
        }
        let current = self
            .repositories
            .get(role)
            .ok_or_else(|| CrossRepoWorkspaceError::UnknownRole(role.to_string()))?;
        let old_bytes = current.state.usage().bytes;
        let child = current
            .state
            .child(patch_set)
            .map_err(CrossRepoWorkspaceError::CandidateState)?;
        let patch_set_id = patch_set
            .identity()
            .map_err(CrossRepoWorkspaceError::PatchSet)?;
        let new_total = self
            .total_bytes()
            .saturating_sub(old_bytes)
            .checked_add(child.usage().bytes)
            .ok_or(CrossRepoWorkspaceError::TotalByteLimitExceeded {
                actual: u64::MAX,
                limit: self.policy.max_total_bytes,
            })?;
        if new_total > self.policy.max_total_bytes {
            return Err(CrossRepoWorkspaceError::TotalByteLimitExceeded {
                actual: new_total,
                limit: self.policy.max_total_bytes,
            });
        }
        let current = self.repositories.get_mut(role).expect("role checked");
        current.state = child;
        current.patch_set_ids.push(patch_set_id.clone());
        self.refresh_override_configs()?;
        Ok(patch_set_id)
    }

    pub fn result(&self) -> CrossRepoWorkspaceResult {
        CrossRepoWorkspaceResult {
            compatibility_fingerprint: self.compatibility.fingerprint(),
            repositories: self
                .repositories
                .values()
                .map(|repo| EffectiveRepositoryState {
                    repository: repo.repository.clone(),
                    role: repo.role.clone(),
                    base_revision: repo.revision.clone(),
                    state_id: repo.state.state_id().to_string(),
                    patch_set_ids: repo.patch_set_ids.clone(),
                })
                .collect(),
            overrides: self
                .overrides
                .iter()
                .map(|rule| EffectiveCargoOverride {
                    consumer_role: rule.consumer_role.clone(),
                    source: rule.source.clone(),
                    package: rule.package.clone(),
                    provider_role: rule.provider_role.clone(),
                    provider_path: rule.provider_path.clone(),
                })
                .collect(),
        }
    }

    fn total_bytes(&self) -> u64 {
        self.repositories
            .values()
            .map(|repo| repo.state.usage().bytes)
            .sum()
    }

    fn refresh_override_configs(&mut self) -> Result<(), CrossRepoWorkspaceError> {
        let dir = self.control_root.join("cargo-overrides");
        let next = self.control_root.join("cargo-overrides-next");
        let _ = std::fs::remove_dir_all(&next);
        std::fs::create_dir_all(&next).map_err(io_error)?;

        let mut grouped: BTreeMap<String, BTreeMap<String, Vec<&CargoPatchOverride>>> =
            BTreeMap::new();
        for rule in &self.overrides {
            grouped
                .entry(rule.consumer_role.clone())
                .or_default()
                .entry(rule.source.clone())
                .or_default()
                .push(rule);
        }

        let mut configs = BTreeMap::new();
        for (index, (consumer, sources)) in grouped.iter().enumerate() {
            let mut text = String::from("# generated by RSI CrossRepoWorkspace; do not commit\n");
            for (source, rules) in sources {
                text.push_str("\n[patch.");
                text.push_str(&toml_quote(source));
                text.push_str("]\n");
                for rule in rules {
                    let provider = self.repositories.get(&rule.provider_role).ok_or_else(|| {
                        CrossRepoWorkspaceError::UnknownRole(rule.provider_role.clone())
                    })?;
                    let provider_path = provider.state.root().join(&rule.provider_path);
                    if !provider_path.join("Cargo.toml").is_file() {
                        return Err(CrossRepoWorkspaceError::InvalidOverride(format!(
                            "provider {}/{} has no Cargo.toml",
                            rule.provider_role, rule.provider_path
                        )));
                    }
                    text.push_str(&toml_quote(&rule.package));
                    text.push_str(" = { path = ");
                    text.push_str(&toml_quote(&provider_path.to_string_lossy()));
                    text.push_str(" }\n");
                }
            }
            let staged = next.join(format!("{index}.toml"));
            std::fs::write(&staged, text).map_err(io_error)?;
            configs.insert(consumer.clone(), dir.join(format!("{index}.toml")));
        }

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::rename(&next, &dir).map_err(io_error)?;
        self.override_configs = configs;
        Ok(())
    }
}

impl Drop for CrossRepoWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.control_root);
    }
}

#[derive(Debug)]
pub enum CrossRepoWorkspaceError {
    Compatibility(CompatibilityError),
    CandidateState(CandidateStateError),
    PatchSet(PatchSetError),
    InvalidPolicy(String),
    RepositoryLimitExceeded { actual: usize, limit: usize },
    TotalByteLimitExceeded { actual: u64, limit: u64 },
    SourceSetMismatch(String),
    DuplicateRole(String),
    UnknownRole(String),
    RevisionMismatch { role: String, expected: String, actual: String },
    Git(String),
    UnsafeTrackedPath(String),
    UnsupportedTrackedEntry(String),
    PatchRoleNotAllowed(String),
    DuplicateOverride { consumer_role: String, source: String, package: String },
    InvalidOverride(String),
    Io(String),
}

impl fmt::Display for CrossRepoWorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compatibility(error) => write!(f, "compatibility error: {error}"),
            Self::CandidateState(error) => write!(f, "candidate-state error: {error}"),
            Self::PatchSet(error) => write!(f, "PatchSet error: {error}"),
            Self::InvalidPolicy(message) => write!(f, "invalid cross-repo policy: {message}"),
            Self::RepositoryLimitExceeded { actual, limit } => write!(f, "repository limit exceeded: {actual} > {limit}"),
            Self::TotalByteLimitExceeded { actual, limit } => write!(f, "workspace byte limit exceeded: {actual} > {limit}"),
            Self::SourceSetMismatch(message) => write!(f, "source set mismatch: {message}"),
            Self::DuplicateRole(role) => write!(f, "duplicate repository role: {role}"),
            Self::UnknownRole(role) => write!(f, "unknown repository role: {role}"),
            Self::RevisionMismatch { role, expected, actual } => write!(f, "revision mismatch for role {role}: expected {expected}, got {actual}"),
            Self::Git(message) => write!(f, "local git verification failed: {message}"),
            Self::UnsafeTrackedPath(path) => write!(f, "unsafe tracked path: {path}"),
            Self::UnsupportedTrackedEntry(path) => write!(f, "unsupported tracked filesystem entry: {path}"),
            Self::PatchRoleNotAllowed(role) => write!(f, "task does not allow patches to repository role {role}"),
            Self::DuplicateOverride { consumer_role, source, package } => write!(f, "duplicate Cargo override for {consumer_role}: {source} / {package}"),
            Self::InvalidOverride(message) => write!(f, "invalid Cargo override: {message}"),
            Self::Io(message) => write!(f, "cross-repo workspace I/O error: {message}"),
        }
    }
}

impl std::error::Error for CrossRepoWorkspaceError {}

fn verify_git_source(source: &LocalRepositorySource, expected: &RepositoryRevision) -> Result<(), CrossRepoWorkspaceError> {
    if !source.root.is_dir() {
        return Err(CrossRepoWorkspaceError::Git(format!("{} is not a directory", source.root.display())));
    }
    let head = git_output(&source.root, &["rev-parse", "HEAD"])?;
    if head.trim() != expected.revision {
        return Err(CrossRepoWorkspaceError::RevisionMismatch {
            role: expected.role.clone(),
            expected: expected.revision.clone(),
            actual: head.trim().to_string(),
        });
    }
    git_success(&source.root, &["diff", "--quiet", "--"])?;
    git_success(&source.root, &["diff", "--cached", "--quiet", "--"])?;
    Ok(())
}

fn materialize_tracked_tree(source: &Path, destination: &Path) -> Result<(), CrossRepoWorkspaceError> {
    std::fs::create_dir_all(destination).map_err(io_error)?;
    let output = Command::new("git")
        .arg("-C")
        .arg(source)
        .args(["ls-files", "-z"])
        .output()
        .map_err(|error| CrossRepoWorkspaceError::Git(error.to_string()))?;
    if !output.status.success() {
        return Err(CrossRepoWorkspaceError::Git(format!("git ls-files failed with {}", output.status)));
    }
    for raw in output.stdout.split(|byte| *byte == 0).filter(|part| !part.is_empty()) {
        let path = std::str::from_utf8(raw)
            .map_err(|_| CrossRepoWorkspaceError::UnsafeTrackedPath("non-UTF8 path".into()))?;
        let normalized = normalize_relative(path)?;
        let from = source.join(&normalized);
        let metadata = std::fs::symlink_metadata(&from).map_err(io_error)?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(CrossRepoWorkspaceError::UnsupportedTrackedEntry(normalized));
        }
        let to = destination.join(&normalized);
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent).map_err(io_error)?;
        }
        std::fs::copy(&from, &to).map_err(io_error)?;
    }
    Ok(())
}

fn git_output(root: &Path, args: &[&str]) -> Result<String, CrossRepoWorkspaceError> {
    let output = Command::new("git").arg("-C").arg(root).args(args).output()
        .map_err(|error| CrossRepoWorkspaceError::Git(error.to_string()))?;
    if !output.status.success() {
        return Err(CrossRepoWorkspaceError::Git(format!("git {} failed with {}", args.join(" "), output.status)));
    }
    String::from_utf8(output.stdout).map_err(|_| CrossRepoWorkspaceError::Git("git output is not UTF-8".into()))
}

fn git_success(root: &Path, args: &[&str]) -> Result<(), CrossRepoWorkspaceError> {
    let status = Command::new("git").arg("-C").arg(root).args(args).status()
        .map_err(|error| CrossRepoWorkspaceError::Git(error.to_string()))?;
    if status.success() {
        Ok(())
    } else {
        Err(CrossRepoWorkspaceError::Git(format!("git {} failed with {status}", args.join(" "))))
    }
}

fn normalize_provider_path(path: &str) -> Result<String, CrossRepoWorkspaceError> {
    if path == "." {
        return Ok(".".into());
    }
    normalize_relative(path)
}

fn normalize_relative(path: &str) -> Result<String, CrossRepoWorkspaceError> {
    if path.is_empty() {
        return Err(CrossRepoWorkspaceError::UnsafeTrackedPath(path.into()));
    }
    let parsed = Path::new(path);
    if parsed.is_absolute() {
        return Err(CrossRepoWorkspaceError::UnsafeTrackedPath(path.into()));
    }
    let mut parts = Vec::new();
    for component in parsed.components() {
        match component {
            Component::Normal(value) => parts.push(value.to_str().ok_or_else(|| CrossRepoWorkspaceError::UnsafeTrackedPath(path.into()))?),
            _ => return Err(CrossRepoWorkspaceError::UnsafeTrackedPath(path.into())),
        }
    }
    if parts.is_empty() {
        return Err(CrossRepoWorkspaceError::UnsafeTrackedPath(path.into()));
    }
    Ok(parts.join("/"))
}

fn checked_text(field: &'static str, value: String) -> Result<String, CrossRepoWorkspaceError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(CrossRepoWorkspaceError::InvalidOverride(format!("invalid {field}")));
    }
    Ok(value)
}

fn toml_quote(value: &str) -> String {
    let mut out = String::from("\"");
    for character in value.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

fn unique_control_root() -> PathBuf {
    let sequence = WORKSPACE_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("rsi-cross-repo-{}-{sequence}", std::process::id()))
}

fn io_error(error: std::io::Error) -> CrossRepoWorkspaceError {
    CrossRepoWorkspaceError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patchset::FileOperation;

    fn git(root: &Path, args: &[&str]) -> String {
        let output = Command::new("git").arg("-C").arg(root).args(args).output().unwrap();
        assert!(output.status.success(), "git {:?} failed", args);
        String::from_utf8(output.stdout).unwrap().trim().to_string()
    }

    fn repository(name: &str, package: &str) -> (PathBuf, String) {
        let root = std::env::temp_dir().join(format!(
            "rsi-cross-repo-test-{}-{}-{name}",
            std::process::id(),
            WORKSPACE_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join("src")).unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.email", "rsi-test@example.invalid"]);
        git(&root, &["config", "user.name", "RSI Test"]);
        std::fs::write(root.join("Cargo.toml"), format!("[package]\nname = \"{package}\"\nversion = \"0.1.0\"\n")).unwrap();
        std::fs::write(root.join("src/lib.rs"), format!("pub const NAME: &str = \"{name}\";\n")).unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-qm", "initial"]);
        let revision = git(&root, &["rev-parse", "HEAD"]);
        (root, revision)
    }

    fn policy() -> CrossRepoWorkspacePolicy {
        CrossRepoWorkspacePolicy::new(4, 1_000_000, CandidateStoragePolicy::new(100, 500_000).unwrap()).unwrap()
    }

    #[test]
    fn materializes_exact_revision_and_ignores_untracked_files() {
        let (a_root, a_rev) = repository("rsi", "rsi-fixture");
        let (b_root, b_rev) = repository("scirust", "scirust-fixture");
        std::fs::write(b_root.join("local-only.bin"), b"ignored").unwrap();
        let set = CompatibilitySet::new(
            vec![
                RepositoryRevision::new("Memorithm/RSI", &a_rev, "rsi").unwrap(),
                RepositoryRevision::new("Memorithm/scirust", &b_rev, "scirust").unwrap(),
            ],
            "rustc stable",
            vec![],
        ).unwrap();
        let workspace = CrossRepoWorkspace::materialize(
            set,
            vec![
                LocalRepositorySource::new("Memorithm/RSI", "rsi", &a_rev, &a_root).unwrap(),
                LocalRepositorySource::new("Memorithm/scirust", "scirust", &b_rev, &b_root).unwrap(),
            ],
            vec!["rsi".into()],
            vec![],
            policy(),
        ).unwrap();
        assert!(workspace.root_for_role("rsi").unwrap().join("Cargo.toml").is_file());
        assert!(!workspace.root_for_role("scirust").unwrap().join("local-only.bin").exists());
        let _ = std::fs::remove_dir_all(a_root);
        let _ = std::fs::remove_dir_all(b_root);
    }

    #[test]
    fn rejects_revision_mismatch_and_dirty_tracked_sources() {
        let (root, revision) = repository("rsi", "rsi-fixture");
        let set = CompatibilitySet::new(
            vec![RepositoryRevision::new("Memorithm/RSI", &revision, "rsi").unwrap()],
            "rustc stable",
            vec![],
        ).unwrap();
        let mismatch = CrossRepoWorkspace::materialize(
            set.clone(),
            vec![LocalRepositorySource::new("Memorithm/RSI", "rsi", "a".repeat(40), &root).unwrap()],
            vec![], vec![], policy(),
        ).err().unwrap();
        assert!(matches!(mismatch, CrossRepoWorkspaceError::RevisionMismatch { .. }));
        std::fs::write(root.join("src/lib.rs"), "pub const DIRTY: bool = true;\n").unwrap();
        let dirty = CrossRepoWorkspace::materialize(
            set,
            vec![LocalRepositorySource::new("Memorithm/RSI", "rsi", &revision, &root).unwrap()],
            vec![], vec![], policy(),
        ).err().unwrap();
        assert!(matches!(dirty, CrossRepoWorkspaceError::Git(_)));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn patch_permission_is_role_scoped_and_result_records_patchset() {
        let (root, revision) = repository("rsi", "rsi-fixture");
        let set = CompatibilitySet::new(
            vec![RepositoryRevision::new("Memorithm/RSI", &revision, "rsi").unwrap()],
            "rustc stable",
            vec![],
        ).unwrap();
        let source = LocalRepositorySource::new("Memorithm/RSI", "rsi", &revision, &root).unwrap();
        let patch = PatchSet::new(vec![FileOperation::modify_exact("src/lib.rs", "rsi", "rsi-patched")]).unwrap();
        let mut denied = CrossRepoWorkspace::materialize(set.clone(), vec![source.clone()], vec![], vec![], policy()).unwrap();
        assert!(matches!(denied.apply_patch("rsi", &patch), Err(CrossRepoWorkspaceError::PatchRoleNotAllowed(_))));
        let mut allowed = CrossRepoWorkspace::materialize(set, vec![source], vec!["rsi".into()], vec![], policy()).unwrap();
        let before = allowed.result().repositories[0].state_id.clone();
        let patch_id = allowed.apply_patch("rsi", &patch).unwrap();
        let result = allowed.result();
        assert_ne!(before, result.repositories[0].state_id);
        assert_eq!(result.repositories[0].base_revision, revision);
        assert_eq!(result.repositories[0].patch_set_ids, vec![patch_id]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cargo_override_lives_outside_tracked_repository_and_is_recorded() {
        let (consumer_root, consumer_rev) = repository("consumer", "consumer-fixture");
        let (provider_root, provider_rev) = repository("provider", "provider-fixture");
        let set = CompatibilitySet::new(
            vec![
                RepositoryRevision::new("Memorithm/RSI", &consumer_rev, "consumer").unwrap(),
                RepositoryRevision::new("Memorithm/scirust", &provider_rev, "provider").unwrap(),
            ],
            "rustc stable",
            vec![],
        ).unwrap();
        let rule = CargoPatchOverride::new(
            "consumer",
            "https://github.com/Memorithm/scirust",
            "provider-fixture",
            "provider",
            ".",
        ).unwrap();
        let workspace = CrossRepoWorkspace::materialize(
            set,
            vec![
                LocalRepositorySource::new("Memorithm/RSI", "consumer", &consumer_rev, &consumer_root).unwrap(),
                LocalRepositorySource::new("Memorithm/scirust", "provider", &provider_rev, &provider_root).unwrap(),
            ],
            vec!["provider".into()],
            vec![rule],
            policy(),
        ).unwrap();
        let config = workspace.cargo_override_config("consumer").unwrap();
        assert!(!config.starts_with(workspace.root_for_role("consumer").unwrap()));
        let text = std::fs::read_to_string(config).unwrap();
        assert!(text.contains("[patch.\"https://github.com/Memorithm/scirust\"]"));
        assert!(text.contains("provider-fixture"));
        assert_eq!(workspace.result().overrides.len(), 1);
        let _ = std::fs::remove_dir_all(consumer_root);
        let _ = std::fs::remove_dir_all(provider_root);
    }
}