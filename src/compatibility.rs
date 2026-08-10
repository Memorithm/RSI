//! Deterministic cross-repository compatibility metadata.
//!
//! A [`CompatibilitySet`] records the exact repository revisions and build
//! contract that produced an engineering result.  The type is deliberately
//! std-only and performs no network access: resolution of a moving branch name
//! is an orchestration concern, while this module accepts only immutable Git
//! object identifiers.

use crate::json::Json;
use crate::sha256::sha256;
use std::fmt;

/// Version of the JSON wire format emitted by [`CompatibilitySet`].
pub const COMPATIBILITY_SCHEMA_VERSION: u64 = 1;

/// Validation or decoding failure for compatibility metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatibilityError {
    EmptyField(&'static str),
    InvalidRepository(String),
    InvalidRevision(String),
    InvalidValue {
        field: &'static str,
        value: String,
    },
    DuplicateRepositoryRole {
        repository: String,
        role: String,
    },
    InvalidJson(String),
    UnsupportedSchema(u64),
    MissingField(&'static str),
}

impl fmt::Display for CompatibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "compatibility field '{field}' is empty"),
            Self::InvalidRepository(repository) => {
                write!(f, "invalid repository identifier '{repository}' (expected owner/name)")
            }
            Self::InvalidRevision(revision) => write!(
                f,
                "invalid immutable git revision '{revision}' (expected 40 or 64 hex characters)"
            ),
            Self::InvalidValue { field, value } => {
                write!(f, "invalid compatibility field '{field}': '{value}'")
            }
            Self::DuplicateRepositoryRole { repository, role } => write!(
                f,
                "duplicate compatibility entry for repository '{repository}' and role '{role}'"
            ),
            Self::InvalidJson(error) => write!(f, "invalid compatibility JSON: {error}"),
            Self::UnsupportedSchema(version) => {
                write!(f, "unsupported compatibility schema version {version}")
            }
            Self::MissingField(field) => write!(f, "missing compatibility field '{field}'"),
        }
    }
}

impl std::error::Error for CompatibilityError {}

/// One immutable repository participating in a qualified system state.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RepositoryRevision {
    /// Canonical repository locator in `owner/name` form.
    pub repository: String,
    /// Exact immutable Git object ID (SHA-1 today, SHA-256 compatible).
    pub revision: String,
    /// Semantic role in the compatibility set (`rsi`, `scirust`, `flat`, ...).
    pub role: String,
}

impl RepositoryRevision {
    pub fn new(
        repository: impl Into<String>,
        revision: impl Into<String>,
        role: impl Into<String>,
    ) -> Result<Self, CompatibilityError> {
        let repository = repository.into();
        let revision = revision.into();
        let role = role.into();

        validate_repository(&repository)?;
        validate_revision(&revision)?;
        validate_text("role", &role)?;

        Ok(Self {
            repository,
            revision: revision.to_ascii_lowercase(),
            role,
        })
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set("repository", Json::Str(self.repository.clone()))
            .set("revision", Json::Str(self.revision.clone()))
            .set("role", Json::Str(self.role.clone()));
        out
    }

    fn from_json(value: &Json) -> Result<Self, CompatibilityError> {
        let repository = required_string(value, "repository")?;
        let revision = required_string(value, "revision")?;
        let role = required_string(value, "role")?;
        Self::new(repository, revision, role)
    }
}

/// Exact, reproducible set of repositories and build features used together.
///
/// Construction canonicalizes repository and feature ordering so equivalent
/// inputs serialize to identical bytes and therefore to the same fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilitySet {
    revisions: Vec<RepositoryRevision>,
    toolchain: String,
    feature_contract: Vec<String>,
}

impl CompatibilitySet {
    pub fn new(
        mut revisions: Vec<RepositoryRevision>,
        toolchain: impl Into<String>,
        mut feature_contract: Vec<String>,
    ) -> Result<Self, CompatibilityError> {
        if revisions.is_empty() {
            return Err(CompatibilityError::EmptyField("revisions"));
        }

        let toolchain = toolchain.into();
        validate_text("toolchain", &toolchain)?;

        for feature in &feature_contract {
            validate_text("feature_contract", feature)?;
        }

        revisions.sort();
        for pair in revisions.windows(2) {
            if pair[0].repository == pair[1].repository && pair[0].role == pair[1].role {
                return Err(CompatibilityError::DuplicateRepositoryRole {
                    repository: pair[0].repository.clone(),
                    role: pair[0].role.clone(),
                });
            }
        }

        feature_contract.sort();
        feature_contract.dedup();

        Ok(Self {
            revisions,
            toolchain,
            feature_contract,
        })
    }

    pub fn revisions(&self) -> &[RepositoryRevision] {
        &self.revisions
    }

    pub fn toolchain(&self) -> &str {
        &self.toolchain
    }

    pub fn feature_contract(&self) -> &[String] {
        &self.feature_contract
    }

    /// Canonical compact JSON representation.
    pub fn to_json_string(&self) -> String {
        self.to_json().to_string()
    }

    /// Decode and re-canonicalize a compatibility set.
    pub fn from_json_str(input: &str) -> Result<Self, CompatibilityError> {
        let value = Json::parse(input).map_err(CompatibilityError::InvalidJson)?;
        let schema = value
            .get("schema")
            .and_then(Json::as_u64)
            .ok_or(CompatibilityError::MissingField("schema"))?;
        if schema != COMPATIBILITY_SCHEMA_VERSION {
            return Err(CompatibilityError::UnsupportedSchema(schema));
        }

        let revisions = value
            .get("revisions")
            .and_then(Json::as_array)
            .ok_or(CompatibilityError::MissingField("revisions"))?
            .iter()
            .map(RepositoryRevision::from_json)
            .collect::<Result<Vec<_>, _>>()?;

        let toolchain = required_string(&value, "toolchain")?;
        let feature_contract = value
            .get("feature_contract")
            .and_then(Json::as_array)
            .ok_or(CompatibilityError::MissingField("feature_contract"))?
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or(CompatibilityError::InvalidValue {
                        field: "feature_contract",
                        value: item.to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::new(revisions, toolchain, feature_contract)
    }

    /// SHA-256 of the canonical JSON bytes, suitable as a stable state key.
    pub fn fingerprint(&self) -> String {
        let digest = sha256(self.to_json_string().as_bytes());
        let mut out = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(out, "{byte:02x}");
        }
        out
    }

    fn to_json(&self) -> Json {
        let mut out = Json::obj();
        out.set(
            "feature_contract",
            Json::Arr(
                self.feature_contract
                    .iter()
                    .cloned()
                    .map(Json::Str)
                    .collect(),
            ),
        )
        .set(
            "revisions",
            Json::Arr(self.revisions.iter().map(RepositoryRevision::to_json).collect()),
        )
        .set("schema", Json::Num(COMPATIBILITY_SCHEMA_VERSION as f64))
        .set("toolchain", Json::Str(self.toolchain.clone()));
        out
    }
}

fn validate_repository(repository: &str) -> Result<(), CompatibilityError> {
    if repository.is_empty() {
        return Err(CompatibilityError::EmptyField("repository"));
    }
    if repository.trim() != repository || repository.chars().any(char::is_whitespace) {
        return Err(CompatibilityError::InvalidRepository(repository.to_string()));
    }
    let mut parts = repository.split('/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    if owner.is_empty() || name.is_empty() || parts.next().is_some() {
        return Err(CompatibilityError::InvalidRepository(repository.to_string()));
    }
    Ok(())
}

fn validate_revision(revision: &str) -> Result<(), CompatibilityError> {
    let valid_len = matches!(revision.len(), 40 | 64);
    if !valid_len || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CompatibilityError::InvalidRevision(revision.to_string()));
    }
    Ok(())
}

fn validate_text(field: &'static str, value: &str) -> Result<(), CompatibilityError> {
    if value.is_empty() {
        return Err(CompatibilityError::EmptyField(field));
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(CompatibilityError::InvalidValue {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn required_string(value: &Json, field: &'static str) -> Result<String, CompatibilityError> {
    value
        .get(field)
        .and_then(Json::as_str)
        .map(str::to_owned)
        .ok_or(CompatibilityError::MissingField(field))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rev(repository: &str, hex: char, role: &str) -> RepositoryRevision {
        RepositoryRevision::new(repository, hex.to_string().repeat(40), role).unwrap()
    }

    #[test]
    fn canonical_order_is_independent_of_input_order() {
        let a = CompatibilitySet::new(
            vec![
                rev("Memorithm/scirust", 'b', "scirust"),
                rev("Memorithm/RSI", 'a', "rsi"),
            ],
            "rustc 1.89.0",
            vec!["wgpu".into(), "flat-attention".into(), "wgpu".into()],
        )
        .unwrap();
        let b = CompatibilitySet::new(
            vec![
                rev("Memorithm/RSI", 'a', "rsi"),
                rev("Memorithm/scirust", 'b', "scirust"),
            ],
            "rustc 1.89.0",
            vec!["flat-attention".into(), "wgpu".into()],
        )
        .unwrap();

        assert_eq!(a, b);
        assert_eq!(a.to_json_string(), b.to_json_string());
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.feature_contract(), &["flat-attention", "wgpu"]);
    }

    #[test]
    fn json_round_trip_is_byte_stable() {
        let set = CompatibilitySet::new(
            vec![
                rev("Memorithm/FLAT-ATTENTION", 'c', "flat"),
                rev("Memorithm/RSI", 'a', "rsi"),
                rev("Memorithm/scirust", 'b', "scirust"),
            ],
            "nightly-2026-07-02",
            vec!["flat-attention".into(), "wgpu".into()],
        )
        .unwrap();

        let encoded = set.to_json_string();
        let decoded = CompatibilitySet::from_json_str(&encoded).unwrap();
        assert_eq!(decoded, set);
        assert_eq!(decoded.to_json_string(), encoded);
    }

    #[test]
    fn moving_branch_name_is_rejected_as_revision() {
        let error = RepositoryRevision::new("Memorithm/scirust", "master", "scirust")
            .unwrap_err();
        assert!(matches!(error, CompatibilityError::InvalidRevision(_)));
    }

    #[test]
    fn duplicate_repository_role_is_rejected() {
        let error = CompatibilitySet::new(
            vec![
                rev("Memorithm/scirust", 'a', "scirust"),
                rev("Memorithm/scirust", 'b', "scirust"),
            ],
            "rustc stable",
            vec![],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CompatibilityError::DuplicateRepositoryRole { .. }
        ));
    }

    #[test]
    fn invalid_repository_is_rejected() {
        let error = RepositoryRevision::new("scirust", "a".repeat(40), "scirust").unwrap_err();
        assert!(matches!(error, CompatibilityError::InvalidRepository(_)));
    }

    #[test]
    fn uppercase_revision_is_normalized() {
        let revision = RepositoryRevision::new(
            "Memorithm/RSI",
            "ABCDEF0123456789ABCDEF0123456789ABCDEF01",
            "rsi",
        )
        .unwrap();
        assert_eq!(revision.revision, "abcdef0123456789abcdef0123456789abcdef01");
    }
}