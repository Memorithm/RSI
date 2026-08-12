//! Typed, std-only consumer for PAPERS `memorithm.science/bundle-v1`.
//!
//! RSI consumes a stable JSON/process contract rather than linking the
//! heavyweight `papers_core` crate. A paper supplies hypotheses and methods;
//! RSI's empirical DGM evaluator remains authoritative for acceptance.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::json::Json;

pub const SCIENTIFIC_BUNDLE_SCHEMA: &str = "memorithm.science/bundle-v1";
pub const SCIENTIFIC_CLAIM_SCHEMA: &str = "memorithm.science/claim-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaperAnalysisMode {
    Heuristic,
    Model { provider: String, model: String },
}

/// Process-isolated bridge to the PAPERS producer contract.
#[derive(Debug, Clone)]
pub struct ScientificPapersRunner {
    papers_bin: String,
    contract_bin: String,
    timeout: Duration,
}

impl ScientificPapersRunner {
    pub fn from_environment() -> Self {
        let papers_bin = std::env::var("RSI_PAPERS_BIN").unwrap_or_else(|_| "papers".into());
        let contract_bin = std::env::var("RSI_PAPERS_CONTRACT_BIN")
            .unwrap_or_else(|_| companion_contract_bin(&papers_bin));
        Self {
            papers_bin,
            contract_bin,
            timeout: Duration::from_secs(300),
        }
    }

    pub fn with_binaries(papers_bin: impl Into<String>, contract_bin: impl Into<String>) -> Self {
        Self {
            papers_bin: papers_bin.into(),
            contract_bin: contract_bin.into(),
            timeout: Duration::from_secs(300),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Analyze one paper and return a validated typed bundle.
    ///
    /// For model-backed analysis the requested provider is forced into the
    /// PAPERS child process through `PAPERS_LLM__PROVIDER`. PAPERS loads env
    /// after defaults/config.toml, so the provider recorded by `papers-contract`
    /// is the provider actually selected by the analysis process. The model is
    /// already forced by PAPERS' `--model` CLI argument.
    pub fn analyze_bundle(
        &self,
        source: &str,
        out_dir: &Path,
        mode: &PaperAnalysisMode,
    ) -> Result<ScientificBundle, String> {
        let model_identity = validate_mode(mode)?;
        std::fs::create_dir_all(out_dir)
            .map_err(|error| format!("création {}: {error}", out_dir.display()))?;

        let mut analyze_args = vec![
            "analyze".to_string(),
            "--source".to_string(),
            source.to_string(),
            "--output".to_string(),
            out_dir.display().to_string(),
        ];
        let mut analyze_env = Vec::new();
        match model_identity {
            None => analyze_args.push("--no-llm".into()),
            Some((provider, model)) => {
                analyze_args.push("--model".into());
                analyze_args.push(model.to_string());
                analyze_env.push(("PAPERS_LLM__PROVIDER", provider));
            }
        }
        run_bounded(
            &self.papers_bin,
            &analyze_args,
            &analyze_env,
            self.timeout,
        )?;

        let analysis_path = out_dir.join("analysis.json");
        if !analysis_path.is_file() {
            return Err(format!(
                "PAPERS n'a pas produit {}",
                analysis_path.display()
            ));
        }

        let bundle_path = out_dir.join("scientific_bundle.json");
        let mut contract_args = vec![
            "--input".to_string(),
            analysis_path.display().to_string(),
            "--output".to_string(),
            bundle_path.display().to_string(),
        ];
        if let Some((provider, model)) = model_identity {
            contract_args.extend([
                "--provider".into(),
                provider.to_string(),
                "--model".into(),
                model.to_string(),
            ]);
        }
        run_bounded(&self.contract_bin, &contract_args, &[], self.timeout)?;

        let raw = std::fs::read_to_string(&bundle_path)
            .map_err(|error| format!("lecture {}: {error}", bundle_path.display()))?;
        ScientificBundle::parse(&raw)
    }

    pub fn papers_bin(&self) -> &str {
        &self.papers_bin
    }

    pub fn contract_bin(&self) -> &str {
        &self.contract_bin
    }
}

fn validate_mode(mode: &PaperAnalysisMode) -> Result<Option<(&str, &str)>, String> {
    match mode {
        PaperAnalysisMode::Heuristic => Ok(None),
        PaperAnalysisMode::Model { provider, model } => {
            if provider.trim().is_empty() || model.trim().is_empty() {
                Err("provider et model doivent être non vides".into())
            } else {
                Ok(Some((provider.as_str(), model.as_str())))
            }
        }
    }
}

fn companion_contract_bin(papers_bin: &str) -> String {
    let path = Path::new(papers_bin);
    if path.is_absolute()
        || path
            .parent()
            .is_some_and(|parent| parent != Path::new(""))
    {
        let parent = path.parent().unwrap_or_else(|| Path::new(""));
        return parent.join("papers-contract").display().to_string();
    }
    "papers-contract".into()
}

fn run_bounded(
    bin: &str,
    args: &[String],
    env: &[(&str, &str)],
    timeout: Duration,
) -> Result<String, String> {
    const MAX_OUTPUT: u64 = 8 * 1024 * 1024;

    let mut command = Command::new(bin);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in env {
        command.env(key, value);
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("lancement de {bin}: {error}"))?;

    let stdout = child.stdout.take().ok_or("stdout PAPERS indisponible")?;
    let stderr = child.stderr.take().ok_or("stderr PAPERS indisponible")?;
    let (stdout_tx, stdout_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = stdout.take(MAX_OUTPUT);
        let mut buffer = String::new();
        let _ = reader.read_to_string(&mut buffer);
        let _ = stdout_tx.send(buffer);
    });
    let (stderr_tx, stderr_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = stderr.take(MAX_OUTPUT);
        let mut buffer = String::new();
        let _ = reader.read_to_string(&mut buffer);
        let _ = stderr_tx.send(buffer);
    });

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = stdout_rx
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap_or_default();
                let stderr = stderr_rx
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap_or_default();
                if status.success() {
                    return Ok(stdout);
                }
                return if stderr.trim().is_empty() {
                    Err(format!("{bin} a rendu {status}"))
                } else {
                    Err(format!("{bin} a rendu {status}: {}", stderr.trim()))
                };
            }
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{bin}: timeout ({}s)", timeout.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(error) => return Err(format!("attente de {bin}: {error}")),
        }
    }
}

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
    /// Actionable only means usable as a candidate-generation hint. It does not
    /// mean the scientific claim has been empirically proven in our environment.
    pub fn is_actionable_method(&self) -> bool {
        self.method
            .as_deref()
            .is_some_and(|method| !method.trim().is_empty())
            && !matches!(
                self.state,
                ClaimState::Contradicted | ClaimState::NotApplicable
            )
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
        let root = Json::parse(raw).map_err(|error| format!("bundle JSON invalide: {error}"))?;
        let schema = require_string(&root, "schema")?;
        if schema != SCIENTIFIC_BUNDLE_SCHEMA {
            return Err(format!("schema de bundle non supporté: {schema}"));
        }

        let paper = root.get("paper").ok_or("bundle.paper manquant")?;
        let paper_id = require_string(paper, "id")?.to_string();
        let title = require_string(paper, "title")?.to_string();

        let provenance_json = root
            .get("provenance")
            .ok_or("bundle.provenance manquant")?;
        let provenance = BundleProvenance {
            paper_id: require_string(provenance_json, "paper_id")?.to_string(),
            source: require_string(provenance_json, "source")?.to_string(),
            extracted_content_sha256: require_sha256(
                provenance_json,
                "extracted_content_sha256",
            )?,
            analysis_sha256: require_sha256(provenance_json, "analysis_sha256")?,
            generator: require_string(provenance_json, "generator")?.to_string(),
            generator_version: require_string(provenance_json, "generator_version")?.to_string(),
        };
        if provenance.paper_id != paper_id {
            return Err("bundle provenance paper_id incohérent".into());
        }

        let values = root
            .get("claims")
            .and_then(Json::as_array)
            .ok_or("bundle.claims doit être un tableau")?;
        let mut claims = Vec::with_capacity(values.len());
        for value in values {
            let claim_schema = require_string(value, "schema")?;
            if claim_schema != SCIENTIFIC_CLAIM_SCHEMA {
                return Err(format!("schema de claim non supporté: {claim_schema}"));
            }
            let claim_id = require_string(value, "id")?;
            if require_string(value, "paper_id")? != paper_id {
                return Err(format!("claim {claim_id} rattaché au mauvais papier"));
            }

            let confidence = match value.get("confidence") {
                None | Some(Json::Null) => None,
                Some(number) => {
                    let confidence = number
                        .as_f64()
                        .ok_or("claim.confidence doit être numérique ou null")?;
                    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                        return Err("claim.confidence hors [0,1]".into());
                    }
                    Some(confidence)
                }
            };

            let evidence_values = value
                .get("evidence")
                .and_then(Json::as_array)
                .ok_or("claim.evidence doit être un tableau")?;
            let mut evidence = Vec::with_capacity(evidence_values.len());
            for item in evidence_values {
                let text_sha256 = optional_string(item, "text_sha256")?.map(str::to_string);
                if let Some(hash) = &text_sha256 {
                    validate_sha256(hash)?;
                }
                evidence.push(ScientificEvidence {
                    origin: require_string(item, "origin")?.to_string(),
                    locator: require_string(item, "locator")?.to_string(),
                    text_sha256,
                });
            }

            claims.push(ScientificClaim {
                id: claim_id.to_string(),
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

    /// Convert method claims into DGM goals while retaining claim provenance.
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
                if let Some(algorithm) = claim
                    .algorithm
                    .as_deref()
                    .filter(|algorithm| !algorithm.trim().is_empty())
                {
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
        .filter(|value| !value.trim().is_empty())
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
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

    #[test]
    fn rejects_empty_model_identity_before_subprocess() {
        let mode = PaperAnalysisMode::Model {
            provider: "".into(),
            model: "m".into(),
        };
        assert!(validate_mode(&mode).is_err());
    }

    #[test]
    fn model_identity_is_preserved_for_subprocess_and_contract() {
        let mode = PaperAnalysisMode::Model {
            provider: "ollama".into(),
            model: "model-x".into(),
        };
        assert_eq!(validate_mode(&mode).unwrap(), Some(("ollama", "model-x")));
    }

    #[test]
    fn companion_contract_binary_follows_explicit_papers_path() {
        assert_eq!(
            companion_contract_bin("/opt/papers/bin/papers"),
            std::path::PathBuf::from("/opt/papers/bin/papers-contract")
                .display()
                .to_string()
        );
        assert_eq!(companion_contract_bin("papers"), "papers-contract");
    }

    #[test]
    fn runner_accepts_explicit_binaries_without_probe_side_effects() {
        let runner = ScientificPapersRunner::with_binaries("p", "c");
        assert_eq!(runner.papers_bin(), "p");
        assert_eq!(runner.contract_bin(), "c");
    }
}
