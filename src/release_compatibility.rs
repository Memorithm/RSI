//! P9.1 cross-repository release compatibility lock.
//!
//! The lock is deliberately network-free. It records only immutable repository
//! revisions that have already been qualified by the program and keeps moving
//! default-branch observations outside the release identity. Newer upstream
//! heads therefore cannot silently alter a replayable engineering release.

use crate::compatibility::{
    COMPATIBILITY_SCHEMA_VERSION, CompatibilityError, CompatibilitySet, RepositoryRevision,
};
use crate::json::Json;
use crate::sha256::sha256;
use std::fmt;

pub const RELEASE_COMPATIBILITY_LOCK_SCHEMA_VERSION: u64 = 1;

pub const CURRENT_RELEASE_COMPATIBILITY_LOCK_JSON: &str =
    include_str!("../compatibility/SCIRUST_RSI_COMPATIBILITY_LOCK.json");

const REQUIRED_REPOSITORY_ROLES: [(&str, &str); 4] = [
    ("rsi", "Memorithm/RSI"),
    ("scirust", "Memorithm/scirust"),
    ("scirust-rsi", "Memorithm/scirust"),
    ("flat", "Memorithm/FLAT-ATTENTION"),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseCompatibilityLock {
    compatibility: CompatibilitySet,
    cogno_contract_version: String,
    qualification_evidence: Vec<String>,
}

impl ReleaseCompatibilityLock {
    pub fn new(
        compatibility: CompatibilitySet,
        cogno_contract_version: impl Into<String>,
        mut qualification_evidence: Vec<String>,
    ) -> Result<Self, ReleaseCompatibilityError> {
        validate_required_repository_roles(&compatibility)?;

        let cogno_contract_version = cogno_contract_version.into();
        validate_text("cogno_contract_version", &cogno_contract_version)?;
        if !cogno_contract_version.starts_with("cogno-core@") {
            return Err(ReleaseCompatibilityError::InvalidCognoContractVersion(
                cogno_contract_version,
            ));
        }

        if qualification_evidence.is_empty() {
            return Err(ReleaseCompatibilityError::MissingQualificationEvidence);
        }
        for evidence in &qualification_evidence {
            validate_text("qualification_evidence", evidence)?;
        }
        qualification_evidence.sort();
        qualification_evidence.dedup();

        for prefix in ["flat:", "rsi:", "scirust:"] {
            if !qualification_evidence
                .iter()
                .any(|evidence| evidence.starts_with(prefix))
            {
                return Err(ReleaseCompatibilityError::MissingComponentEvidence(
                    prefix.trim_end_matches(':').to_string(),
                ));
            }
        }

        Ok(Self {
            compatibility,
            cogno_contract_version,
            qualification_evidence,
        })
    }

    pub fn compatibility(&self) -> &CompatibilitySet {
        &self.compatibility
    }

    pub fn cogno_contract_version(&self) -> &str {
        &self.cogno_contract_version
    }

    pub fn qualification_evidence(&self) -> &[String] {
        &self.qualification_evidence
    }

    pub fn locked_revision(&self, role: &str) -> Option<&RepositoryRevision> {
        self.compatibility
            .revisions()
            .iter()
            .find(|revision| revision.role == role)
    }

    pub fn to_json_string(&self) -> String {
        self.to_json().to_string()
    }

    pub fn from_json_str(input: &str) -> Result<Self, ReleaseCompatibilityError> {
        let root = Json::parse(input).map_err(ReleaseCompatibilityError::InvalidJson)?;
        let schema = root
            .get("schema")
            .and_then(Json::as_u64)
            .ok_or(ReleaseCompatibilityError::MissingField("schema"))?;
        if schema != RELEASE_COMPATIBILITY_LOCK_SCHEMA_VERSION {
            return Err(ReleaseCompatibilityError::UnsupportedSchema(schema));
        }

        let compatibility_json = root
            .get("compatibility")
            .ok_or(ReleaseCompatibilityError::MissingField("compatibility"))?
            .to_string();
        let compatibility = CompatibilitySet::from_json_str(&compatibility_json)
            .map_err(ReleaseCompatibilityError::Compatibility)?;
        let cogno_contract_version = required_string(&root, "cogno_contract_version")?;
        let qualification_evidence = root
            .get("qualification_evidence")
            .and_then(Json::as_array)
            .ok_or(ReleaseCompatibilityError::MissingField(
                "qualification_evidence",
            ))?
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| ReleaseCompatibilityError::InvalidValue {
                        field: "qualification_evidence",
                        value: item.to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::new(
            compatibility,
            cogno_contract_version,
            qualification_evidence,
        )
    }

    pub fn fingerprint(&self) -> String {
        hex_digest(&sha256(self.to_json_string().as_bytes()))
    }

    fn to_json(&self) -> Json {
        let mut root = Json::obj();
        root.set(
            "cogno_contract_version",
            Json::Str(self.cogno_contract_version.clone()),
        )
        .set("compatibility", compatibility_json(&self.compatibility))
        .set(
            "qualification_evidence",
            Json::Arr(
                self.qualification_evidence
                    .iter()
                    .cloned()
                    .map(Json::Str)
                    .collect(),
            ),
        )
        .set(
            "schema",
            Json::Num(RELEASE_COMPATIBILITY_LOCK_SCHEMA_VERSION as f64),
        );
        root
    }
}

pub fn current_release_compatibility_lock(
) -> Result<ReleaseCompatibilityLock, ReleaseCompatibilityError> {
    ReleaseCompatibilityLock::from_json_str(CURRENT_RELEASE_COMPATIBILITY_LOCK_JSON.trim())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseCompatibilityError {
    InvalidJson(String),
    UnsupportedSchema(u64),
    MissingField(&'static str),
    EmptyField(&'static str),
    InvalidValue {
        field: &'static str,
        value: String,
    },
    Compatibility(CompatibilityError),
    MissingRepositoryRole(String),
    AmbiguousRepositoryRole(String),
    RepositoryRoleMismatch {
        role: String,
        expected_repository: String,
        actual_repository: String,
    },
    InvalidCognoContractVersion(String),
    MissingQualificationEvidence,
    MissingComponentEvidence(String),
}

impl fmt::Display for ReleaseCompatibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "release compatibility lock error: {self:?}")
    }
}

impl std::error::Error for ReleaseCompatibilityError {}

fn validate_required_repository_roles(
    compatibility: &CompatibilitySet,
) -> Result<(), ReleaseCompatibilityError> {
    for (role, expected_repository) in REQUIRED_REPOSITORY_ROLES {
        let matches: Vec<_> = compatibility
            .revisions()
            .iter()
            .filter(|revision| revision.role == role)
            .collect();
        let revision = match matches.as_slice() {
            [only] => *only,
            [] => {
                return Err(ReleaseCompatibilityError::MissingRepositoryRole(
                    role.to_string(),
                ));
            }
            _ => {
                return Err(ReleaseCompatibilityError::AmbiguousRepositoryRole(
                    role.to_string(),
                ));
            }
        };
        if revision.repository != expected_repository {
            return Err(ReleaseCompatibilityError::RepositoryRoleMismatch {
                role: role.to_string(),
                expected_repository: expected_repository.to_string(),
                actual_repository: revision.repository.clone(),
            });
        }
    }
    Ok(())
}

fn compatibility_json(compatibility: &CompatibilitySet) -> Json {
    let mut root = Json::obj();
    root.set(
        "feature_contract",
        Json::Arr(
            compatibility
                .feature_contract()
                .iter()
                .cloned()
                .map(Json::Str)
                .collect(),
        ),
    )
    .set(
        "revisions",
        Json::Arr(
            compatibility
                .revisions()
                .iter()
                .map(|revision| {
                    let mut item = Json::obj();
                    item.set("repository", Json::Str(revision.repository.clone()))
                        .set("revision", Json::Str(revision.revision.clone()))
                        .set("role", Json::Str(revision.role.clone()));
                    item
                })
                .collect(),
        ),
    )
    .set("schema", Json::Num(COMPATIBILITY_SCHEMA_VERSION as f64))
    .set(
        "toolchain",
        Json::Str(compatibility.toolchain().to_string()),
    );
    root
}

fn validate_text(field: &'static str, value: &str) -> Result<(), ReleaseCompatibilityError> {
    if value.is_empty() {
        return Err(ReleaseCompatibilityError::EmptyField(field));
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(ReleaseCompatibilityError::InvalidValue {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn required_string(
    value: &Json,
    field: &'static str,
) -> Result<String, ReleaseCompatibilityError> {
    value
        .get(field)
        .and_then(Json::as_str)
        .map(str::to_owned)
        .ok_or(ReleaseCompatibilityError::MissingField(field))
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

    const RSI_P8_5: &str = "83009f1be2ddbdb42da840a3a3df6504aef57b5a";
    const SCIRUST_P7_3: &str = "07d3524bc241d59142c22515a009c8c2b64ef50e";
    const SCIRUST_RSI_CANONICAL: &str = "8af0801b8bc0c69630797db82bb2dd3416cc8f0a";
    const FLAT_M15: &str = "974ebbaf95f54917dba0dc3f394d9ad1a92e8349";

    fn revision(lock: &ReleaseCompatibilityLock, role: &str) -> &str {
        &lock.locked_revision(role).unwrap().revision
    }

    #[test]
    fn committed_lock_is_canonical_and_replayable() {
        let lock = current_release_compatibility_lock().unwrap();
        assert_eq!(revision(&lock, "rsi"), RSI_P8_5);
        assert_eq!(revision(&lock, "scirust"), SCIRUST_P7_3);
        assert_eq!(revision(&lock, "scirust-rsi"), SCIRUST_RSI_CANONICAL);
        assert_eq!(revision(&lock, "flat"), FLAT_M15);
        assert_eq!(lock.cogno_contract_version(), "cogno-core@0.1.0");
        assert_eq!(
            lock.to_json_string(),
            CURRENT_RELEASE_COMPATIBILITY_LOCK_JSON.trim()
        );
        assert_eq!(lock.fingerprint().len(), 64);
    }

    #[test]
    fn lock_uses_exact_features_and_toolchain_contract() {
        let lock = current_release_compatibility_lock().unwrap();
        assert_eq!(
            lock.compatibility().toolchain(),
            "rustc 1.97.1 (qualification); MSRV 1.89"
        );
        assert_eq!(
            lock.compatibility().feature_contract(),
            &[
                "cogno:hard-gates",
                "flat-attention:wgpu",
                "rsi:public-features",
                "rsi:scirust",
                "scirust-gpu:flat-attention",
                "scirust-sciagent:flat-attention",
                "tokenizer:canonical-parity",
            ]
        );
    }

    #[test]
    fn moving_branch_names_cannot_enter_the_lock() {
        let error = RepositoryRevision::new("Memorithm/scirust", "master", "scirust")
            .unwrap_err();
        assert!(matches!(error, CompatibilityError::InvalidRevision(_)));
    }

    #[test]
    fn missing_required_component_fails_closed() {
        let compatibility = CompatibilitySet::new(
            vec![
                RepositoryRevision::new("Memorithm/RSI", RSI_P8_5, "rsi").unwrap(),
                RepositoryRevision::new("Memorithm/scirust", SCIRUST_P7_3, "scirust").unwrap(),
                RepositoryRevision::new(
                    "Memorithm/scirust",
                    SCIRUST_RSI_CANONICAL,
                    "scirust-rsi",
                )
                .unwrap(),
            ],
            "rustc 1.97.1 (qualification); MSRV 1.89",
            vec!["cogno:hard-gates".to_string()],
        )
        .unwrap();
        let error = ReleaseCompatibilityLock::new(
            compatibility,
            "cogno-core@0.1.0",
            vec![
                "flat:qualified".to_string(),
                "rsi:qualified".to_string(),
                "scirust:qualified".to_string(),
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ReleaseCompatibilityError::MissingRepositoryRole("flat".to_string())
        );
    }

    #[test]
    fn qualification_evidence_is_canonicalized() {
        let lock = current_release_compatibility_lock().unwrap();
        let rebuilt = ReleaseCompatibilityLock::new(
            lock.compatibility().clone(),
            lock.cogno_contract_version(),
            lock.qualification_evidence().iter().rev().cloned().collect(),
        )
        .unwrap();
        assert_eq!(rebuilt.to_json_string(), lock.to_json_string());
        assert_eq!(rebuilt.fingerprint(), lock.fingerprint());
    }
}
