//! TRAJECTORY — exportateur du **data flywheel** (axe 4, `docs/AGENTWORLD_STUDY.md`).
//!
//! Chaque évaluation RÉELLE du gate DGM est une trajectoire à **vérité terrain** :
//! un patch (`find`→`replace` sur un fichier) et le verdict `cargo build`+`test`
//! qu'il a réellement produit. Exportées en **JSONL** `{prompt, completion, score}` —
//! `prompt` = l'invite exacte du world model ([`crate::simulation::build_sim_prompt`]),
//! `completion` = une sortie `cargo` réaliste terminée par la ligne machine
//! `SIMCAL_VERDICT` — ces paires servent à **fine-tuner un world model
//! spécialisé du dépôt RSI** : RSI s'améliore, ses traces améliorent le
//! simulateur, qui accélère RSI (auto-amélioration à deux étages).
//!
//! Std-only, zéro dépendance : échappement JSON maison. Invariant testé :
//! `parse_sim_verdict(&traj.completion())` redonne exactement le verdict réel.

use crate::simulation::build_sim_prompt;

/// Une trajectoire à vérité terrain : un patch et le verdict RÉEL du gate.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    pub target: String,
    pub find: String,
    pub replace: String,
    /// Contenu (borné) du fichier cible AU MOMENT de l'évaluation.
    pub file_content: String,
    pub compiles: bool,
    pub tests_passed: u32,
    pub tests_failed: u32,
    pub score: f64,
    /// Sortie `cargo` RÉELLE (bornée) capturée au gate — erreur du compilateur
    /// ou détail des tests échoués. Rend la complétion beaucoup plus riche qu'un
    /// gabarit (le world model apprend *pourquoi* ça casse, pas seulement que ça
    /// casse). `None` sur succès : le gabarit « running N tests…ok » suffit et
    /// reste cargo-réaliste. La ligne `SIMCAL_VERDICT` reste toujours appendue.
    pub output: Option<String>,
}

impl Trajectory {
    /// `SIMCAL_VERDICT: compile=<oui|non>; tests=<pass|fail|na>`.
    pub fn verdict_line(&self) -> String {
        let compile = if self.compiles { "oui" } else { "non" };
        let tests = if !self.compiles {
            "na"
        } else if self.tests_failed == 0 {
            "pass"
        } else {
            "fail"
        };
        format!("SIMCAL_VERDICT: compile={compile}; tests={tests}")
    }

    /// Complétion cible : sortie `cargo` (réelle si capturée, sinon gabarit
    /// réaliste) **terminée par la ligne machine `SIMCAL_VERDICT`** — qui reste
    /// faisant autorité pour le parseur, donc l'invariant de round-trip tient
    /// quel que soit le corps.
    pub fn completion(&self) -> String {
        let body = self.output.clone().unwrap_or_else(|| self.template_body());
        format!("{body}\n{}", self.verdict_line())
    }

    /// Corps « cargo réaliste » synthétique — repli quand aucune sortie réelle
    /// n'a été capturée (typiquement les succès, où le détail n'apporte rien).
    fn template_body(&self) -> String {
        if !self.compiles {
            "error[E0308]: le patch ne compile pas (build échoué)".to_string()
        } else if self.tests_failed == 0 {
            format!(
                "running {n} tests\ntest result: ok. {n} passed; 0 failed; 0 ignored",
                n = self.tests_passed
            )
        } else {
            let total = self.tests_passed + self.tests_failed;
            format!(
                "running {total} tests\ntest result: FAILED. {} passed; {} failed; 0 ignored",
                self.tests_passed, self.tests_failed
            )
        }
    }

    /// Invite exacte du pré-crible pour cette trajectoire.
    pub fn prompt(&self) -> String {
        build_sim_prompt(&self.target, &self.file_content, &self.find, &self.replace)
    }

    /// Ligne JSONL `{"prompt": "...", "completion": "...", "score": <f64>}`.
    /// Le `score` (récompense mesurée du gate : perf réelle ou pass-rate) est
    /// exporté en champ à part — un pipeline SFT l'ignore, un world model qui
    /// prédit AUSSI la perf peut l'apprendre. Métadonnée additive et sûre.
    pub fn to_jsonl(&self) -> String {
        format!(
            "{{\"prompt\":\"{}\",\"completion\":\"{}\",\"score\":{}}}",
            json_escape(&self.prompt()),
            json_escape(&self.completion()),
            json_number(self.score),
        )
    }
}

/// Formate un `f64` en nombre JSON valide (non-fini ⇒ `null`, jamais `NaN`).
fn json_number(n: f64) -> String {
    if n.is_finite() {
        format!("{n}")
    } else {
        "null".to_string()
    }
}

/// Sérialise un lot de trajectoires en JSONL (une paire par ligne).
pub fn to_jsonl(trajectories: &[Trajectory]) -> String {
    let mut out = String::new();
    for t in trajectories {
        out.push_str(&t.to_jsonl());
        out.push('\n');
    }
    out
}

/// Échappe une chaîne pour un littéral JSON (std-only).
pub fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 16);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::parse_sim_verdict;

    fn traj(compiles: bool, passed: u32, failed: u32) -> Trajectory {
        Trajectory {
            target: "src/kernels.rs".into(),
            find: "let x = 0;".into(),
            replace: "let x = 1;".into(),
            file_content: "fn f() { let x = 0; }".into(),
            compiles,
            tests_passed: passed,
            tests_failed: failed,
            score: 1.0,
            output: None,
        }
    }

    #[test]
    fn verdict_line_covers_the_three_cases() {
        assert!(traj(true, 10, 0)
            .verdict_line()
            .contains("compile=oui; tests=pass"));
        assert!(traj(true, 8, 2)
            .verdict_line()
            .contains("compile=oui; tests=fail"));
        assert!(traj(false, 0, 0)
            .verdict_line()
            .contains("compile=non; tests=na"));
    }

    #[test]
    fn completion_roundtrips_through_the_prescreen_parser() {
        for t in [traj(true, 12, 0), traj(true, 9, 3), traj(false, 0, 0)] {
            let v = parse_sim_verdict(&t.completion());
            assert_eq!(v.compiles, Some(t.compiles), "compile: {}", t.completion());
            let want_tests = if !t.compiles {
                Some(false)
            } else {
                Some(t.tests_failed == 0)
            };
            assert_eq!(v.tests_pass, want_tests, "tests: {}", t.completion());
        }
    }

    #[test]
    fn real_output_enriches_completion_and_still_roundtrips() {
        // Échec de compilation avec vraie sortie cargo capturée.
        let mut t = traj(false, 0, 0);
        t.output = Some(
            "Compiling rsi v0.10.0\nerror[E0412]: cannot find type `Foo` in this scope".into(),
        );
        let c = t.completion();
        assert!(
            c.contains("error[E0412]"),
            "la vraie sortie doit apparaître : {c}"
        );
        // le verdict machine reste faisant autorité → round-trip intact
        let v = parse_sim_verdict(&c);
        assert_eq!(v.compiles, Some(false));
        assert_eq!(v.tests_pass, Some(false)); // na → non-pass

        // Test échoué avec détail réel.
        let mut t = traj(true, 40, 2);
        t.output =
            Some("test result: FAILED. 40 passed; 2 failed\n---- json::deep panicked ----".into());
        let c = t.completion();
        assert!(
            c.contains("panicked"),
            "le détail réel doit apparaître : {c}"
        );
        let v = parse_sim_verdict(&c);
        assert_eq!(v.compiles, Some(true));
        assert_eq!(v.tests_pass, Some(false));
    }

    #[test]
    fn jsonl_is_single_line_and_escaped() {
        let mut t = traj(true, 3, 0);
        t.file_content = "ligne1\n\"guillemets\"\tet tab".into();
        let line = t.to_jsonl();
        assert_eq!(line.matches('\n').count(), 0, "une trajectoire = une ligne");
        assert!(
            line.contains("\\n") && line.contains("\\\""),
            "échappement présent"
        );
        assert!(line.starts_with("{\"prompt\":\"") && line.ends_with('}'));
        // JSON valide de bout en bout, avec le champ score exporté.
        let j = crate::json::Json::parse(&line).expect("JSONL valide");
        assert!(j.get("prompt").is_some() && j.get("completion").is_some());
        assert_eq!(j.get("score").and_then(|v| v.as_f64()), Some(1.0));
    }

    #[test]
    fn json_escape_handles_control_chars() {
        assert_eq!(json_escape("a\"b\\c\nd\te"), "a\\\"b\\\\c\\nd\\te");
        assert_eq!(json_escape("\u{0}"), "\\u0000");
    }

    #[test]
    fn to_jsonl_one_line_per_trajectory() {
        let out = to_jsonl(&[traj(true, 1, 0), traj(false, 0, 0)]);
        assert_eq!(out.lines().count(), 2);
        assert!(out.ends_with('\n'));
    }
}
