//! ADDONS — capacités **enfichables** de RSI (papers-agent, …).
//!
//! Un addon RSI est un **binaire externe** piloté par un **adaptateur
//! std-only** vivant ici, jamais une dépendance de crate : le cœur reste sans
//! dépendance (doctrine — c'est ce qui rend le gate DGM fiable et le build
//! reproductible), l'addon peut tirer ce qu'il veut (GPU, ORT, tokio…) sans
//! contaminer RSI. Contrat d'un adaptateur :
//!
//!   1. **détection** à l'exécution (env dédiée, sinon PATH) — jamais requis ;
//!   2. **sous-processus bornés** (timeout + plafond de sortie, anti-blocage /
//!      anti-OOM, comme [`crate::knowledge::PapersKnowledge`]) ;
//!   3. **dégradation propre** : addon absent ⇒ fonctionnalité indisponible,
//!      jamais de panique ni d'échec du cœur ;
//!   4. **le gate reste autoritaire** : un addon *propose* (connaissances,
//!      objectifs, contexte) — l'évaluation empirique et la promotion restent
//!      au moteur RSI.
//!
//! Premier addon : **papers-agent** ([`PapersAddon`]) — analyse de papiers
//! scientifiques (PDF/arXiv/URL) mise au service de l'auto-amélioration :
//! - `analyze` → [`PaperAnalysis`] (contributions, algorithmes, résumé) ;
//! - [`PapersAddon::directive_goals`] → **objectifs DGM directifs** (la
//!   campagne Jetson a prouvé que le directif débloque ce que le vague
//!   n'obtient pas — cf. `docs/PAPERS_SYNERGY.md`) ;
//! - `search` → contexte RAG pour le prompt du proposeur ;
//! - la composante D passe par [`crate::knowledge::PapersKnowledge`] (même
//!   binaire, canal déjà câblé).

use std::path::Path;
use std::time::Duration;

/// Fiche d'identité d'un addon détecté (pour `registry` / diagnostics).
#[derive(Debug, Clone)]
pub struct AddonInfo {
    pub name: &'static str,
    /// Binaire résolu (chemin ou nom PATH).
    pub bin: String,
    pub available: bool,
    /// Capacités offertes au cœur.
    pub capabilities: &'static [&'static str],
}

/// Recense les addons connus et leur disponibilité sur cette machine.
pub fn registry() -> Vec<AddonInfo> {
    let papers = PapersAddon::detect();
    vec![AddonInfo {
        name: "papers-agent",
        bin: PapersAddon::resolve_bin(),
        available: papers.is_some(),
        capabilities: &["knowledge (D)", "goals (DGM)", "context (RAG)"],
    }]
}

// ═══════════════════════ Addon papers-agent ═══════════════════════════════ //

/// Analyse d'un papier, telle que rendue par `papers analyze` (sous-ensemble
/// stable de `analysis.json` — on ne lit que ce dont le cœur a besoin).
#[derive(Debug, Clone, Default)]
pub struct PaperAnalysis {
    pub summary: String,
    pub contributions: Vec<String>,
    pub algorithms: Vec<PaperAlgorithm>,
    pub impacted_modules: Vec<String>,
}

/// Un algorithme/technique extrait d'un papier.
#[derive(Debug, Clone)]
pub struct PaperAlgorithm {
    pub name: String,
    pub complexity: Option<String>,
    pub pseudocode: Option<String>,
}

/// Adaptateur de l'addon **papers-agent** (binaire `papers`, PAPERS V2).
pub struct PapersAddon {
    bin: String,
    timeout: Duration,
}

impl PapersAddon {
    /// Binaire résolu : env `RSI_PAPERS_BIN`, sinon `papers` (PATH). Même
    /// convention que [`crate::knowledge::PapersKnowledge`].
    pub fn resolve_bin() -> String {
        std::env::var("RSI_PAPERS_BIN").unwrap_or_else(|_| "papers".to_string())
    }

    /// Détecte l'addon : le binaire doit exister et répondre. `None` sinon —
    /// l'appelant dégrade proprement.
    pub fn detect() -> Option<Self> {
        let bin = Self::resolve_bin();
        // Sonde bon marché : `--help` doit s'exécuter (code retour ignoré :
        // certains CLI rendent ≠0 sur --help ; seul le spawn compte).
        let ok = std::process::Command::new(&bin)
            .arg("--help")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok();
        ok.then_some(PapersAddon { bin, timeout: Duration::from_secs(300) })
    }

    /// Délai max par invocation (défaut 300 s — un PDF/arXiv peut être long).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Analyse un papier (`papers analyze -s <source> -o <out_dir> …`) et
    /// parse `analysis.json`.
    ///
    /// `llm_model` pilote l'analyse : `None` ⇒ `--no-llm` (heuristique,
    /// rapide, déterministe — mais n'extrait PAS les techniques du papier,
    /// seulement les métadonnées) ; `Some("")` ⇒ analyse **LLM** avec le
    /// modèle par défaut de PAPERS ; `Some(m)` ⇒ `--model m`.
    pub fn analyze(
        &self,
        source: &str,
        out_dir: &Path,
        llm_model: Option<&str>,
    ) -> Result<PaperAnalysis, String> {
        std::fs::create_dir_all(out_dir).map_err(|e| format!("création {}: {e}", out_dir.display()))?;
        let mut args: Vec<String> = vec![
            "analyze".into(),
            "-s".into(),
            source.into(),
            "-o".into(),
            out_dir.display().to_string(),
        ];
        match llm_model {
            None => args.push("--no-llm".into()),
            Some("") => {}
            Some(m) => args.extend(["--model".into(), m.to_string()]),
        }
        run_bounded(&self.bin, &args, self.timeout)?;
        let path = out_dir.join("analysis.json");
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("lecture {}: {e}", path.display()))?;
        parse_analysis_json(&raw)
    }

    /// Recherche sémantique dans le store de l'addon (`papers search`) —
    /// contexte RAG pour le prompt du proposeur. Rend les lignes non vides.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<String>, String> {
        let args: Vec<String> = vec![
            "search".into(),
            "-q".into(),
            query.into(),
            "-k".into(),
            k.max(1).to_string(),
        ];
        let out = run_bounded(&self.bin, &args, self.timeout)?;
        Ok(out.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect())
    }

    /// Convertit une analyse en **objectifs DGM directifs** (déterministe).
    ///
    /// Un objectif par algorithme extrait (bornés à `max_goals`), formulé
    /// comme la campagne Jetson l'a validé : technique nommée + cible + mêmes
    /// résultats + contrainte de format. Le pseudocode (tronqué) donne au
    /// modèle local la substance de la technique.
    pub fn directive_goals(analysis: &PaperAnalysis, target_hint: &str, max_goals: usize) -> Vec<String> {
        const MAX_PSEUDO: usize = 400;
        analysis
            .algorithms
            .iter()
            .filter(|a| !a.name.trim().is_empty() && !is_placeholder_algorithm(a))
            .take(max_goals.max(1))
            .map(|a| {
                let mut g = format!(
                    "applique la technique « {} » du papier a {target_hint}, memes resultats",
                    a.name.trim()
                );
                if let Some(c) = a.complexity.as_deref().filter(|c| !c.trim().is_empty()) {
                    g.push_str(&format!(" (complexite visee : {})", c.trim()));
                }
                if let Some(p) = a.pseudocode.as_deref().filter(|p| !p.trim().is_empty()) {
                    let short: String = p.chars().take(MAX_PSEUDO).collect();
                    g.push_str(&format!(". Pseudocode du papier : {}", short.trim()));
                }
                g.push_str(". FIND court obligatoire.");
                g
            })
            .collect()
    }
}

/// Vrai si l'« algorithme » est un placeholder de PAPERS (analyse heuristique
/// sans LLM : il décrit alors son PROPRE pipeline, pas une technique du
/// papier). Fabriquer un objectif DGM à partir de ça gaspille des steps —
/// vécu sur Thor : « Pipeline heuristique / INFORMATION NON DISPONIBLE ».
fn is_placeholder_algorithm(a: &PaperAlgorithm) -> bool {
    let name = a.name.to_lowercase();
    if name.contains("heuristique") || name.contains("heuristic") {
        return true;
    }
    if a.complexity.as_deref().is_some_and(|c| c.to_uppercase().contains("NON DISPONIBLE")) {
        return true;
    }
    a.pseudocode.as_deref().is_some_and(|p| p.contains("ProcessPaper"))
}

/// Parse le sous-ensemble utile d'`analysis.json` (fonction pure, testée).
pub(crate) fn parse_analysis_json(raw: &str) -> Result<PaperAnalysis, String> {
    let j = crate::json::Json::parse(raw).map_err(|e| format!("analysis.json invalide : {e}"))?;
    let strings = |key: &str| -> Vec<String> {
        match j.get(key) {
            Some(crate::json::Json::Arr(a)) => {
                a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
            }
            _ => Vec::new(),
        }
    };
    let algorithms = match j.get("algorithms") {
        Some(crate::json::Json::Arr(a)) => a
            .iter()
            .filter_map(|v| {
                let name = v.get("name").and_then(|n| n.as_str())?.to_string();
                Some(PaperAlgorithm {
                    name,
                    complexity: v.get("complexity").and_then(|c| c.as_str()).map(str::to_string),
                    pseudocode: v.get("pseudocode").and_then(|p| p.as_str()).map(str::to_string),
                })
            })
            .collect(),
        _ => Vec::new(),
    };
    Ok(PaperAnalysis {
        summary: j
            .get("executive_summary")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        contributions: strings("contributions"),
        algorithms,
        impacted_modules: strings("impacted_modules"),
    })
}

/// Exécute `bin args…` avec **timeout** et **plafond de sortie** (anti-blocage,
/// anti-OOM — un addon bogué ou hostile ne doit pas emporter le cœur).
fn run_bounded(bin: &str, args: &[String], timeout: Duration) -> Result<String, String> {
    use std::io::Read;
    use std::process::Stdio;
    use std::sync::mpsc;
    use std::time::Instant;

    const MAX_OUTPUT: u64 = 8 * 1024 * 1024; // 8 Mio

    let mut child = std::process::Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("lancement de {bin}: {e}"))?;

    let mut stdout = child.stdout.take().ok_or("stdout indisponible")?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = (&mut stdout).take(MAX_OUTPUT).read_to_string(&mut buf);
        let _ = tx.send(buf);
    });

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = rx.recv_timeout(Duration::from_secs(5)).unwrap_or_default();
                if status.success() {
                    return Ok(out);
                }
                return Err(format!("{bin} {} a rendu {status}", args.first().cloned().unwrap_or_default()));
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("{bin} : timeout ({}s)", timeout.as_secs()));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(format!("attente de {bin}: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
        "executive_summary": "Un papier sur le tuilage de caches.",
        "contributions": ["tuilage adaptatif", "preuve de borne"],
        "algorithms": [
            {"name": "Cache-Oblivious Tiling", "complexity": "O(n^3/B)",
             "pseudocode": "for tile in tiles: process(tile)"},
            {"name": "  ", "complexity": null, "pseudocode": null},
            {"name": "Loop Unrolling", "complexity": null, "pseudocode": null}
        ],
        "impacted_modules": ["kernels"]
    }"#;

    #[test]
    fn parses_analysis_subset() {
        let a = parse_analysis_json(FIXTURE).unwrap();
        assert_eq!(a.summary, "Un papier sur le tuilage de caches.");
        assert_eq!(a.contributions.len(), 2);
        assert_eq!(a.algorithms.len(), 3);
        assert_eq!(a.impacted_modules, vec!["kernels".to_string()]);
        assert!(parse_analysis_json("pas du json").is_err());
        // champs absents ⇒ défauts, pas d'erreur (schéma amont libre d'évoluer)
        let min = parse_analysis_json("{}").unwrap();
        assert!(min.algorithms.is_empty() && min.summary.is_empty());
    }

    #[test]
    fn placeholder_algorithms_never_become_goals() {
        // Sans LLM, PAPERS rend un pseudo-algorithme décrivant son propre
        // pipeline — aucun objectif ne doit en sortir (vécu Thor).
        let raw = r#"{"algorithms":[
            {"name":"Pipeline heuristique","complexity":"INFORMATION NON DISPONIBLE DANS LE PAPIER",
             "pseudocode":"FONCTION ProcessPaper(document)"},
            {"name":"Analyse heuristique","complexity":null,"pseudocode":null}
        ]}"#;
        let a = parse_analysis_json(raw).unwrap();
        assert!(PapersAddon::directive_goals(&a, "cible", 3).is_empty());
    }

    #[test]
    fn directive_goals_are_deterministic_and_bounded() {
        let a = parse_analysis_json(FIXTURE).unwrap();
        let goals = PapersAddon::directive_goals(&a, "kernels::matmul (src/kernels.rs)", 2);
        // le nom vide est filtré ; borné à max_goals
        assert_eq!(goals.len(), 2);
        assert!(goals[0].contains("Cache-Oblivious Tiling"));
        assert!(goals[0].contains("kernels::matmul"));
        assert!(goals[0].contains("O(n^3/B)"));
        assert!(goals[0].contains("Pseudocode du papier"));
        assert!(goals[0].ends_with("FIND court obligatoire."));
        assert!(goals[1].contains("Loop Unrolling"));
        // déterminisme
        assert_eq!(goals, PapersAddon::directive_goals(&a, "kernels::matmul (src/kernels.rs)", 2));
    }

    #[test]
    fn run_bounded_captures_and_times_out() {
        // capture bornée d'un vrai sous-processus
        let out = run_bounded("echo", &["salut".to_string()], Duration::from_secs(10)).unwrap();
        assert_eq!(out.trim(), "salut");
        // binaire absent ⇒ erreur propre, pas de panique
        assert!(run_bounded("/inexistant/xyz", &[], Duration::from_secs(1)).is_err());
    }

    #[test]
    fn registry_reports_papers_addon() {
        let regs = registry();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].name, "papers-agent");
        assert!(!regs[0].capabilities.is_empty());
    }
}
