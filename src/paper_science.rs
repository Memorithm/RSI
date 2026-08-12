//! Typed, std-only consumer for PAPERS `memorithm.science/bundle-v1`.
//!
//! RSI deliberately consumes a stable JSON contract rather than linking the
//! heavyweight `papers_core` crate. The paper supplies hypotheses/methods;
//! RSI's empirical DGM evaluator remains authoritative for acceptance.

use crate::json::Json;

pub const SCIENTIFIC_BUNDLE_SCHEMA: &str = "memorithm.science/bundle-v1";
pub const SCIENTIFIC_CLAIM_SCHEMA: &str = "memorithm.science/claim-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimState {
    Reported,
    Inferred,
    Reproduced,
    PartiallyReproduced,
    Contradicted,
    NotApplicable,
}

impl ClaimState {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "reported" => Ok(Self::Reported),
            "inferred" => Ok(Self::Inferred),
            "reproduced" => Ok(Self::Reproduced),
            "partially_reproduced" => Ok(Self::PartiallyReproduced),
            "contradicted" => Ok(Self::Contradicted),
            "not_applicable" => Ok(Self::NotApplicable),
            other => Err(format!("état de claim scientifique inconnu: {other}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScientificEvidence {
    pub origin: String,
    pub locator: String,
    pub text_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScientificClaim {
    pub id: String,
    pub kind: String,
    pub statement: String,
    pub state: ClaimState,
    pub method: Option<String>,
    pub algorithm: Option<String>,
    pub evidence: Vec<ScientificEvidence>,
    pub confidence: Option<f64>,
}

impl ScientificClaim {
    /// A claim may direct code generation, but this predicate says nothing about
    /// empirical truth. `inferred` is allowed because many PAPERS methods are
    /// model-extracted; the DGM gate still has to prove any resulting patch.
    pub fn is_actionable_method(&self) -> bool {
        self.method.as_deref().is_some_and(|m| !m.trim().is_empty())
            && !matches!(self.state, ClaimState::Contradicted | ClaimState::NotApplicable)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleProvenance {
    pub paper_id: String,
    pub source: String,
    pub extracted_content_sha256: String,
    pub analysis_sha256: String,
    pub generator: String,
    pub generator_version: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScientificBundle {
    pub paper_id: String,
    pub title: String,
    pub claims: Vec<ScientificClaim>,
    pub provenance: BundleProvenance,
}

impl ScientificBundle {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let root = Json::parse(raw).map_err(|e| format!("bundle JSON invalide: {e}"))?;
        require_string(&root, "schema")
            .and_then(|schema| {
                if schema == SCIENTIFIC_BUNDLE_SCHEMA {
                    Ok(())
                } else {
                    Err(format!("schema de bundle non supporté: {schema}"))
                }
            })?;

        let paper = root.get("paper").ok_or("bundle.paper manquant")?;
        let paper_id = require_string(paper, "id")?.to_string();
        let title = require_string(paper, "title")?.to_string();

        let provenance_json = root.get("provenance").ok_or("bundle.provenance manquant")?;
        let provenance = BundleProvenance {
            paper_id: require_string(provenance_json, "paper_id")?.to_string(),
            source: require_string(provenance_json, "source")?.to_string(),
            extracted_content_sha256: require_sha256(provenance_json, "extracted_content_sha256")?,
            analysis_sha256: require_sha256(provenance_json, "analysis_sha256")?,
            generator: require_string(provenance_json, "generator")?.to_string(),
            generator_version: require_string(provenance_json, "generator_version")?.to_string(),
        };
        if provenance.paper_id != paper_id {
            return Err("bundle provenance paper_id incohérent".into());
        }

        let claim_values = root
            .get("claims")
            .and_then(Json::as_array)
            .ok_or("bundle.claims doit être un tableau")?;
        let mut claims = Vec::with_capacity(claim_values.len());
        for value in claim_values {
            let schema = require_string(value, "schema")?;
            if schema != SCIENTIFIC_CLAIM_SCHEMA {
                return Err(format!("schema de claim non supporté: {schema}"));
            }
            let claim_paper = require_string(value, "paper_id")?;
            if claim_paper != paper_id {
                return Err(format!("claim {} rattaché au mauvais papier", require_string(value, "id")?));
            }
            let confidence = match value.get("confidence") {
                None | Some(Json::Null) => None,
                Some(v) => {
                    let c = v.as_f64().ok_or("claim.confidence doit être numérique ou null")?;
                    if !c.is_finite() || !(0.0..=1.0).contains(&c) {
                        return Err("claim.confidence hors [0,1]".into());
                    }
                    Some(c)
                }
            };

            let evidence_values = value
                .get("evidence")
                .and_then(Json::as_array)
                .ok_or("claim.evidence doit être un tableau")?;
            let mut evidence = Vec::with_capacity(evidence_values.len());
            for item in evidence_values {
                let hash = optional_string(item, "text_sha256")?.map(str::to_string);
                if let Some(ref h) = hash {
                    validate_sha256(h)?;
                }
                evidence.push(ScientificEvidence {
                    origin: require_string(item, "origin")?.to_string(),
                    locator: require_string(item, "locator")?.to_string(),
                    text_sha256: hash,
                });
            }

            claims.push(ScientificClaim {
                id: require_string(value, "id")?.to_string(),
                kind: require_string(value, "kind")?.to_string(),
                statement: require_string(value, "statement")?.to_string(),
                state: ClaimState::parse(require_string(value, "state")?)?,
                method: optional_string(value, "method")?.map(str::to_string),
                algorithm: optional_string(value, "algorithm")?.map(str::to_string),
                evidence,
                confidence,
            });
        }

        Ok(Self {
            paper_id,
            title,
            claims,
            provenance,
        })
    }

    /// Convert typed method claims into direct DGM goals while retaining the
    /// claim id for traceability. No claim is described as "proven" here.
    pub fn directive_goals(&self, target_hint: &str, max_goals: usize) -> Vec<String> {
        self.claims
            .iter()
            .filter(|claim| claim.is_actionable_method())
            .take(max_goals.max(1))
            .map(|claim| {
                let mut goal = format!(
                    "évalue empiriquement la méthode « {} » (PAPERS claim {}) sur {target_hint}; conserve le comportement observable et ne promeus que si build+tests+benchmark démontrent le gain",
                    claim.method.as_deref().unwrap_or("méthode"),
                    claim.id
                );
                if let Some(algorithm) = claim.algorithm.as_deref().filter(|s| !s.trim().is_empty()) {
                    let short: String = algorithm.chars().take(500).collect();
                    goal.push_str(". Description/pseudocode non vérifié du papier: ");
                    goal.push_str(short.trim());
                }
                goal
            })
            .collect()
    }
}

fn require_string<'a>(json: &'a Json, key: &str) -> Result<&'a str, String> {
    json.get(key)
        .and_then(Json::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("champ chaîne requis manquant/invalide: {key}"))
}

fn optional_string<'a>(json: &'a Json, key: &str) -> Result<Option<&'a str>, String> {
    match json.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(Some)
            .ok_or_else(|| format!("champ {key} doit être une chaîne ou null")),
    }
}

fn require_sha256(json: &Json, key: &str) -> Result<String, String> {
    let value = require_string(json, key)?;
    validate_sha256(value)?;
    Ok(value.to_string())
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err("SHA-256 attendu en hexadécimal minuscule".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn fixture(state: &str) -> String {
        format!(
            r#"{{
              "schema":"memorithm.science/bundle-v1",
              "paper":{{"id":"P1","title":"Paper","authors":[],"publication_date":null,"source":"fixture","paper_url":null,"github_url":null}},
              "claims":[{{
                "schema":"memorithm.science/claim-v1",
                "id":"method-123",
                "paper_id":"P1",
                "kind":"method",
                "statement":"method X",
                "state":"{state}",
                "evidence":[{{"origin":"analysis_field","locator":"analysis.algorithms[0]","section":null,"page":null,"text":null,"text_sha256":"{HASH_A}"}}],
                "assumptions":[],"method":"X","algorithm":"for each tile...","baseline":null,"dataset":null,"metrics":[],"expected_effect":null,"reported_effect":null,"limitations":[],"falsification_criteria":[],"confidence":null,
                "provenance":{{"paper_id":"P1","source":"fixture","extracted_content_sha256":"{HASH_A}","extracted_content_scope":"abstract","analysis_sha256":"{HASH_B}","generator":"papers_core::scientific_contract","generator_version":"0.4.0","generated_at":"2026-08-12T00:00:00Z","model":null}}
              }}],
              "proposals":[],
              "provenance":{{"paper_id":"P1","source":"fixture","extracted_content_sha256":"{HASH_A}","extracted_content_scope":"abstract","analysis_sha256":"{HASH_B}","generator":"papers_core::scientific_contract","generator_version":"0.4.0","generated_at":"2026-08-12T00:00:00Z","model":null}}
            }}"#
        )
    }

    #[test]
    fn parses_v1_bundle_and_preserves_provenance() {
        let bundle = ScientificBundle::parse(&fixture("inferred")).unwrap();
        assert_eq!(bundle.paper_id, "P1");
        assert_eq!(bundle.claims.len(), 1);
        assert_eq!(bundle.claims[0].id, "method-123");
        assert_eq!(bundle.provenance.analysis_sha256, HASH_B);
    }

    #[test]
    fn directive_goal_names_claim_and_empirical_gate() {
        let bundle = ScientificBundle::parse(&fixture("inferred")).unwrap();
        let goals = bundle.directive_goals("src/kernel.rs", 3);
        assert_eq!(goals.len(), 1);
        assert!(goals[0].contains("method-123"));
        assert!(goals[0].contains("build+tests+benchmark"));
    }

    #[test]
    fn contradicted_claim_is_not_actionable() {
        let bundle = ScientificBundle::parse(&fixture("contradicted")).unwrap();
        assert!(bundle.directive_goals("src/kernel.rs", 3).is_empty());
    }

    #[test]
    fn rejects_unknown_schema() {
        let raw = fixture("inferred").replace("memorithm.science/bundle-v1", "future/v2");
        assert!(ScientificBundle::parse(&raw).is_err());
    }

    #[test]
    fn rejects_invalid_provenance_hash() {
        let raw = fixture("inferred").replace(HASH_B, "not-a-hash");
        assert!(ScientificBundle::parse(&raw).is_err());
    }
}
