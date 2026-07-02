//! SIMULATION — pré-crible du gate DGM par un **world model** (Qwen-AgentWorld).
//!
//! RSI possède un gate empirique **parfait mais coûteux** (`cargo build`+`test`
//! ≈ 1-3 min/candidat). Un *world model* — modèle entraîné à prédire l'état
//! suivant d'un environnement Terminal/SWE — offre l'inverse : un verdict
//! **rapide mais faillible**. La composition rend RSI plus rapide sans toucher
//! à sa doctrine : **le simulateur pré-crible, le réel tranche** (cf.
//! `docs/AGENTWORLD_STUDY.md`). Le simulateur n'a JAMAIS autorité — au pire un
//! faux positif coûte un vrai build (comme aujourd'hui), un faux négatif coûte
//! une idée (mesuré ici, pas supposé).
//!
//! Ce module est **std-only et sans dépendance** : constructeur de prompt,
//! parseur de verdict (heuristique + ligne structurée), et statistiques de
//! calibration (matrice de confusion). Le transport LLM vit dans le binaire
//! `rsi-simcal` (feature `llm-ollama`).

/// Verdict d'un patch : chaque axe est `None` si le simulateur n'a pas tranché.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimVerdict {
    /// Le workspace compile-t-il après le patch ?
    pub compiles: Option<bool>,
    /// La suite de tests passe-t-elle (compile ⇒ 0 échec) ?
    pub tests_pass: Option<bool>,
}

impl SimVerdict {
    /// Verdict de référence (vérité terrain) dérivé d'une [`crate::dgm::Fitness`]
    /// réelle : compile = `f.compiles`, tests_pass = compile ∧ 0 échec.
    pub fn from_real(compiles: bool, tests_failed: u32) -> Self {
        SimVerdict {
            compiles: Some(compiles),
            tests_pass: Some(compiles && tests_failed == 0),
        }
    }
}

/// Construit le prompt du world model : contenu du fichier, patch exact, et
/// **consigne de terminer par une ligne machine** que [`parse_sim_verdict`]
/// lit en priorité. On borne le fichier pour rester dans la fenêtre.
pub fn build_sim_prompt(file_label: &str, file_content: &str, find: &str, replace: &str) -> String {
    const MAX_FILE_CHARS: usize = 12_000;
    let shown: String = file_content.chars().take(MAX_FILE_CHARS).collect();
    let truncated = if file_content.len() > shown.len() { "\n// … (tronqué) …" } else { "" };
    format!(
        "Tu es un world model qui simule l'exécution d'un dépôt Rust. Voici le \
         fichier `{file_label}` :\n\
         ----- début -----\n{shown}{truncated}\n----- fin -----\n\n\
         On applique CE patch (remplace exactement le bloc FIND par REPLACE) :\n\
         FIND:\n<<<\n{find}\n>>>\n\
         REPLACE:\n<<<\n{replace}\n>>>\n\n\
         Simule `cargo build` puis `cargo test`. Raisonne, puis TERMINE ta \
         réponse par EXACTEMENT une ligne de ce format, rien après :\n\
         SIMCAL_VERDICT: compile=<oui|non>; tests=<pass|fail|na>\n\
         (compile=non si le patch ne compile pas ; tests=na si ça ne compile \
         pas ; tests=fail si au moins un test échoue ; tests=pass sinon.)"
    )
}

/// Parse le verdict simulé. Priorité à la **ligne structurée** `SIMCAL_VERDICT:`
/// ; à défaut, heuristiques sur la prose (sortie `cargo` simulée). Pur, testé
/// — y compris sur la vraie sortie d'AgentWorld.
pub fn parse_sim_verdict(text: &str) -> SimVerdict {
    if let Some(v) = parse_structured_line(text) {
        return v;
    }
    let low = text.to_lowercase();

    // A-t-on atteint l'exécution des tests ? (⇒ ça a compilé)
    let ran_tests = low.contains("test result:") || low.contains("running ") && low.contains(" test");
    // Marqueurs d'échec de compilation.
    let compile_fail = low.contains("error[e")
        || low.contains("cannot find")
        || low.contains("mismatched types")
        || low.contains("ne compile pas")
        || low.contains("erreur de compilation")
        || low.contains("does not compile")
        || (low.contains("error:") && !ran_tests);

    let compiles = if ran_tests {
        Some(true)
    } else if compile_fail {
        Some(false)
    } else {
        None
    };

    let tests_pass = if compiles == Some(false) {
        Some(false) // ne compile pas ⇒ tests ne passent pas
    } else if low.contains("test result: ok") || low.contains("0 failed") || low.contains("; 0 failed") {
        Some(true)
    } else if low.contains("test result: failed")
        || low.contains(" failed")
        || low.contains("panicked")
        || low.contains("assertion")
    {
        Some(false)
    } else {
        None
    };

    SimVerdict { compiles, tests_pass }
}

/// Lit la ligne `SIMCAL_VERDICT: compile=…; tests=…` si présente.
fn parse_structured_line(text: &str) -> Option<SimVerdict> {
    let line = text.lines().rev().find(|l| l.to_uppercase().contains("SIMCAL_VERDICT"))?;
    let low = line.to_lowercase();
    let after = low.split("simcal_verdict").nth(1)?;
    let field = |key: &str| -> Option<&str> {
        let seg = after.split(key).nth(1)?;
        Some(seg.trim_start_matches(['=', ' ']).split([';', ' ', ',']).next().unwrap_or("").trim())
    };
    let compiles = field("compile").and_then(|v| match v {
        "oui" | "yes" | "true" => Some(true),
        "non" | "no" | "false" => Some(false),
        _ => None,
    });
    let tests_pass = field("tests").and_then(|v| match v {
        "pass" | "ok" | "oui" => Some(true),
        "fail" | "failed" | "non" => Some(false),
        "na" | "n/a" => None,
        _ => None,
    });
    // Cohérence : ne pas compiler ⇒ tests non passants.
    let tests_pass = if compiles == Some(false) { Some(false) } else { tests_pass };
    Some(SimVerdict { compiles, tests_pass })
}

// ═══════════════════════════ Statistiques ════════════════════════════════ //

/// Matrice de confusion d'un axe binaire (prédit vs réel), + indécis.
#[derive(Debug, Clone, Default)]
pub struct Confusion {
    pub tp: u32,
    pub tn: u32,
    pub fp: u32,
    pub fn_: u32,
    /// Le simulateur n'a pas tranché (verdict `None`).
    pub undecided: u32,
}

impl Confusion {
    /// Ajoute un échantillon (`predicted`=None ⇒ indécis).
    pub fn add(&mut self, predicted: Option<bool>, actual: bool) {
        match predicted {
            None => self.undecided += 1,
            Some(true) if actual => self.tp += 1,
            Some(true) => self.fp += 1,
            Some(false) if !actual => self.tn += 1,
            Some(false) => self.fn_ += 1,
        }
    }

    /// Nombre de verdicts tranchés.
    pub fn decided(&self) -> u32 {
        self.tp + self.tn + self.fp + self.fn_
    }

    /// Exactitude sur les verdicts tranchés (`None` si aucun).
    pub fn accuracy(&self) -> Option<f64> {
        let d = self.decided();
        (d > 0).then(|| (self.tp + self.tn) as f64 / d as f64)
    }
}

/// Calibration cumulée : un axe « compile » et un axe « tests passent ».
///
/// Convention orientée **pré-crible** : la classe positive = « bon candidat »
/// (compile / tests passent). Un **faux négatif** (`fn_`) est le cas coûteux
/// pour un pré-crible — le simulateur écarte à tort une amélioration réelle
/// (idée perdue) ; un **faux positif** ne coûte qu'un vrai build.
#[derive(Debug, Clone, Default)]
pub struct CalibrationStats {
    pub compiles: Confusion,
    pub tests: Confusion,
    pub samples: u32,
}

impl CalibrationStats {
    pub fn record(&mut self, predicted: SimVerdict, actual: SimVerdict) {
        self.samples += 1;
        if let Some(a) = actual.compiles {
            self.compiles.add(predicted.compiles, a);
        }
        if let Some(a) = actual.tests_pass {
            self.tests.add(predicted.tests_pass, a);
        }
    }

    /// Rapport lisible (tableau texte).
    pub fn report(&self) -> String {
        let axis = |name: &str, c: &Confusion| {
            format!(
                "  {name:8} tp={} tn={} fp={} fn={} indécis={} · exactitude={}\n",
                c.tp,
                c.tn,
                c.fp,
                c.fn_,
                c.undecided,
                c.accuracy().map(|a| format!("{:.0}%", a * 100.0)).unwrap_or_else(|| "—".into()),
            )
        };
        format!(
            "calibration ({} échantillons)\n{}{}  \
             note : fn (faux négatif) = idée réelle écartée à tort — le coût d'un pré-crible.",
            self.samples,
            axis("compile", &self.compiles),
            axis("tests", &self.tests),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sortie RÉELLE d'AgentWorld sur la sonde `acc += x*2.0` (Jetson Thor) —
    /// le parseur DOIT en tirer « compile mais tests échouent ».
    const AGENTWORLD_PROBE: &str = "\
Voici la simulation du terminal :\n\
running 1 test\n\
test tests::test_sum ... FAILED\n\
failures:\n\
---- tests::test_sum stdout ----\n\
thread 'tests::test_sum' panicked at 'assertion failed: `(left == right)`\n\
  left: `200.0`,\n right: `100.0`', src/lib.rs:15:9\n\
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; finished in 0.00s\n\
error: test failed, to rerun pass `--lib`";

    #[test]
    fn parses_real_agentworld_probe() {
        let v = parse_sim_verdict(AGENTWORLD_PROBE);
        assert_eq!(v.compiles, Some(true), "les tests ont tourné ⇒ ça compile");
        assert_eq!(v.tests_pass, Some(false), "test result: FAILED");
    }

    #[test]
    fn structured_line_takes_priority() {
        let t = "blabla error[E0308] mais en fait…\nSIMCAL_VERDICT: compile=oui; tests=fail";
        let v = parse_sim_verdict(t);
        assert_eq!(v.compiles, Some(true));
        assert_eq!(v.tests_pass, Some(false));
    }

    #[test]
    fn compile_failure_forces_tests_false() {
        let v = parse_sim_verdict("error[E0308]: mismatched types\n--> src/x.rs:3");
        assert_eq!(v.compiles, Some(false));
        assert_eq!(v.tests_pass, Some(false));
        // ligne structurée cohérente aussi
        let v2 = parse_sim_verdict("SIMCAL_VERDICT: compile=non; tests=na");
        assert_eq!(v2.compiles, Some(false));
        assert_eq!(v2.tests_pass, Some(false));
    }

    #[test]
    fn all_green_detected() {
        let v = parse_sim_verdict(
            "running 178 tests\ntest result: ok. 178 passed; 0 failed; 0 ignored",
        );
        assert_eq!(v.compiles, Some(true));
        assert_eq!(v.tests_pass, Some(true));
    }

    #[test]
    fn undecided_when_silent() {
        let v = parse_sim_verdict("je ne sais pas trop quoi dire ici");
        assert_eq!(v, SimVerdict::default());
    }

    #[test]
    fn confusion_and_calibration_math() {
        let mut s = CalibrationStats::default();
        // prédit bon / réel bon (tp compile+tests)
        s.record(SimVerdict::from_real(true, 0), SimVerdict::from_real(true, 0));
        // prédit casse tests / réel casse tests (tn tests, tp compile)
        s.record(SimVerdict::from_real(true, 1), SimVerdict::from_real(true, 1));
        // prédit bon MAIS réel casse tests (fp sur l'axe tests)
        s.record(SimVerdict::from_real(true, 0), SimVerdict::from_real(true, 2));
        assert_eq!(s.samples, 3);
        assert_eq!(s.compiles.tp, 3); // 3× compile prédit vrai & réel vrai
        assert_eq!(s.tests.tp, 1);
        assert_eq!(s.tests.tn, 1);
        assert_eq!(s.tests.fp, 1);
        assert!((s.tests.accuracy().unwrap() - 2.0 / 3.0).abs() < 1e-9);
        assert!(s.report().contains("calibration (3 échantillons)"));
    }

    #[test]
    fn from_real_maps_gate_verdict() {
        assert_eq!(SimVerdict::from_real(false, 0).tests_pass, Some(false));
        assert_eq!(SimVerdict::from_real(true, 0).tests_pass, Some(true));
        assert_eq!(SimVerdict::from_real(true, 3).tests_pass, Some(false));
    }
}
