//! Domaine **WASM** : exécution RÉELLE de code candidat en **bac à sable**
//! (`wasmi`, interpréteur WebAssembly pur-Rust, déterministe). C'est le 4ᵉ point
//! du spectre texte → config → **code** du design spike (P2).
//!
//! ## Le seul domaine où du code candidat est EXÉCUTÉ
//! Contrairement à `synthesis` (AST interprété maison) ou `prompt`/`tuning`
//! (aucune exécution), ici le LLM propose du **WebAssembly textuel (WAT)** qu'on
//! assemble puis **exécute**. L'isolation est portée par `wasmi` :
//! - **Zéro import host** : on instancie avec un [`Linker`] vide ⇒ tout module
//!   qui déclare un import (réseau, fs, syscall…) échoue à l'instanciation et est
//!   rejeté. Le code candidat ne peut donc QUE calculer.
//! - **Fuel** : chaque appel reçoit un budget d'instructions borné ⇒ terminaison
//!   garantie (une boucle infinie épuise le fuel et trappe).
//! - **Déterministe** : l'interpréteur `wasmi` est déterministe ; même WAT ⇒
//!   mêmes sorties.
//!
//! `safety_check` rejette : WAT invalide, taille excessive, présence d'imports,
//! absence de l'export `run`. `score` exécute les modules valides sur des cas de
//! test (sortie attendue) et renvoie la fraction réussie moins une pénalité de
//! taille.

use crate::ascent::RefineTask;
use crate::llm::{LlmRefineTask, SafetyViolation};
use wasmi::{Config, Engine, Linker, Module, Store};

/// Taille maximale d'un candidat WAT (octets).
const MAX_WAT_BYTES: usize = 64 * 1024;
/// Budget d'instructions par appel (fuel) — borne la terminaison.
///
/// NOTE (audit a5, décision assumée) : le fuel borne le TEMPS de calcul mais
/// pas la mémoire statique déclarée par un module (`(memory N)`). wasmi 0.31
/// n'expose pas de limiteur pré-instantiation ; le risque est borné par la
/// taille max du WAT (64 Kio) et l'exécution locale mono-tenant.
const FUEL_PER_CALL: u64 = 5_000_000;

// ═══════════════ Évaluateur DGM bac à sable WASM (A6) ═══════════════ //

/// Évaluateur DGM **bac à sable WASM** : l'alternative à [`crate::dgm::
/// CargoEvaluator`] pour les workspaces dont la cible est un module
/// WebAssembly textuel. Au lieu de lancer `cargo build+test` sur du code
/// arbitraire (**réservé au code de confiance**), il lit le `.wat` candidat
/// du snapshot et l'exécute dans `wasmi` :
///
/// - **zéro import host** ⇒ le code ne peut que calculer ;
/// - **fuel borné** ⇒ terminaison garantie ;
/// - **déterministe** ⇒ fitness reproductible.
///
/// Barrières alignées sur [`crate::dgm::Fitness`] : WAT non compilable =
/// `Fitness::broken` ; chaque vecteur de test raté compte comme un test
/// échoué ; le score combine fraction réussie et pénalité de taille.
pub struct WasmEvaluator {
    /// Fichier `.wat` candidat, relatif à la racine du workspace snapshot.
    pub target: String,
    /// Vecteurs de test `(entrée, sortie attendue)` sur `run(i64) -> i64`.
    pub cases: Vec<(i64, i64)>,
    /// Pénalité de score par 100 octets de WAT (favorise la concision).
    pub size_penalty: f64,
}

impl WasmEvaluator {
    /// Évaluateur pour une fonction cible échantillonnée sur `inputs`,
    /// lisant `target` dans le snapshot.
    pub fn for_target(
        target: impl Into<String>,
        f: impl Fn(i64) -> i64,
        inputs: impl IntoIterator<Item = i64>,
    ) -> Self {
        WasmEvaluator {
            target: target.into(),
            cases: inputs.into_iter().map(|x| (x, f(x))).collect(),
            size_penalty: 0.0005,
        }
    }
}

impl crate::dgm::Evaluator for WasmEvaluator {
    fn evaluate(&self, workspace: &std::path::Path) -> crate::dgm::Result<crate::dgm::Fitness> {
        let path = workspace.join(&self.target);
        let wat = std::fs::read_to_string(&path).map_err(|e| {
            crate::dgm::DgmError::Evaluation(format!("lecture {} : {e}", path.display()))
        })?;

        // Barrière de compilation : WAT invalide = plancher absolu.
        let (engine, module) = match WasmSynthesis::compile(&wat) {
            Ok(em) => em,
            Err(msg) => return Ok(crate::dgm::Fitness::broken(msg)),
        };

        let mut passed = 0u32;
        let mut failed = 0u32;
        for (x, expected) in &self.cases {
            match WasmSynthesis::run_once(&engine, &module, *x) {
                Ok(got) if got == *expected => passed += 1,
                _ => failed += 1,
            }
        }
        let frac = passed as f64 / self.cases.len().max(1) as f64;
        Ok(crate::dgm::Fitness {
            compiles: true,
            tests_passed: passed,
            tests_failed: failed,
            score: frac - self.size_penalty * (wat.len() as f64 / 100.0),
            notes: format!(
                "{passed}/{} cas passés en bac à sable wasmi",
                passed + failed
            ),
        })
    }
}

/// Tâche de synthèse de code WASM : trouver un module exportant
/// `run: (i64) -> i64` qui calcule la fonction cible sur des entiers.
pub struct WasmSynthesis {
    /// cas de test (entrée, sortie attendue).
    cases: Vec<(i64, i64)>,
    /// pénalité par 100 octets de WAT (favorise la concision).
    size_penalty: f64,
}

impl WasmSynthesis {
    /// Construit la tâche depuis une fonction cible échantillonnée sur `inputs`.
    pub fn from_target(target: impl Fn(i64) -> i64, inputs: impl IntoIterator<Item = i64>) -> Self {
        let cases = inputs.into_iter().map(|x| (x, target(x))).collect();
        WasmSynthesis {
            cases,
            size_penalty: 0.0005,
        }
    }

    /// Candidat initial trivial : `run(x) = 0` (valide, mais médiocre).
    pub fn seed_candidate(&self) -> String {
        "(module (func (export \"run\") (param i64) (result i64) (i64.const 0)))".to_string()
    }

    /// Assemble le WAT en octets WASM, puis compile en module `wasmi` (moteur
    /// avec fuel). Renvoie `(engine, module)` ou une erreur lisible.
    fn compile(wat: &str) -> Result<(Engine, Module), String> {
        let bytes = wat::parse_str(wat).map_err(|e| format!("WAT invalide: {e}"))?;
        let mut config = Config::default();
        config.consume_fuel(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &bytes[..]).map_err(|e| format!("module WASM: {e}"))?;
        Ok((engine, module))
    }

    /// Plafonds mémoire/table par exécution (audit a5) : le fuel borne le
    /// TEMPS mais pas la mémoire statique déclarée ni les `memory.grow`.
    /// `StoreLimits` (wasmi 0.31) ferme ce trou : 16 Mio de mémoire linéaire
    /// max, 1 mémoire, 1 table de 10k éléments, 1 instance — le dépassement
    /// trappe au lieu de consommer l'hôte.
    const MAX_MEMORY_BYTES: usize = 16 * 1024 * 1024;

    /// Exécute `run(input)` dans un bac à sable neuf (linker vide ⇒ aucun import,
    /// fuel + limites mémoire bornés). `Err` si imports non résolus, trap,
    /// fuel épuisé, limite dépassée, ou export manquant.
    fn run_once(engine: &Engine, module: &Module, input: i64) -> Result<i64, String> {
        use wasmi::{StoreLimits, StoreLimitsBuilder};
        let linker = Linker::<StoreLimits>::new(engine);
        let limits = StoreLimitsBuilder::new()
            .memory_size(Self::MAX_MEMORY_BYTES)
            .memories(1)
            .table_elements(10_000)
            .tables(1)
            .instances(1)
            .trap_on_grow_failure(true)
            .build();
        let mut store = Store::new(engine, limits);
        // enregistre le limiteur (sinon StoreLimits n'est jamais consulté)
        store.limiter(|limits| limits);
        store.add_fuel(FUEL_PER_CALL).map_err(|e| e.to_string())?;
        let instance = linker
            .instantiate(&mut store, module)
            .map_err(|e| format!("instanciation (imports interdits): {e}"))?
            .start(&mut store)
            .map_err(|e| e.to_string())?;
        let run = instance
            .get_typed_func::<i64, i64>(&store, "run")
            .map_err(|e| format!("export 'run' (i64)->i64 absent: {e}"))?;
        run.call(&mut store, input)
            .map_err(|e| format!("exécution: {e}"))
    }

    /// Fraction de cas réussis (diagnostic).
    pub fn pass_fraction(&self, wat: &str) -> f64 {
        let Ok((engine, module)) = Self::compile(wat) else {
            return 0.0;
        };
        let passed = self
            .cases
            .iter()
            .filter(|(x, t)| Self::run_once(&engine, &module, *x) == Ok(*t))
            .count();
        passed as f64 / self.cases.len().max(1) as f64
    }
}

impl RefineTask for WasmSynthesis {
    type Cand = String;

    fn score(&self, cand: &String) -> f64 {
        let (engine, module) = match Self::compile(cand) {
            Ok(em) => em,
            Err(_) => return 0.0, // invalide ⇒ score plancher (rejeté en amont par safety_check)
        };
        let mut passed = 0usize;
        for (x, t) in &self.cases {
            if Self::run_once(&engine, &module, *x) == Ok(*t) {
                passed += 1;
            }
        }
        let frac = passed as f64 / self.cases.len().max(1) as f64;
        frac - self.size_penalty * (cand.len() as f64 / 100.0)
    }

    /// Générateur de repli (chemin non-LLM) : renvoie le candidat inchangé.
    /// Le domaine vise le chemin LLM ; muter du WAT déterministiquement n'a pas
    /// d'intérêt ici.
    fn refine(&mut self, cand: &String, _iter: usize) -> String {
        cand.clone()
    }
}

impl LlmRefineTask for WasmSynthesis {
    fn describe(&self, incumbent: &String) -> String {
        format!(
            "Tâche : écrire un module WebAssembly textuel (WAT) exportant une \
             fonction `run` de signature (param i64) (result i64) qui calcule la \
             fonction cible sur les entrées de test. Aucun import autorisé \
             (calcul pur). Module actuel :\n{incumbent}\n\
             Score (fraction de cas réussis) : {:.3}\n\
             Réponds avec un module WAT amélioré par ligne (échappe les sauts de \
             ligne, un module complet par ligne).",
            self.score(incumbent)
        )
    }

    fn parse_proposals(&self, raw: &[String]) -> Vec<String> {
        raw.iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Sûreté du bac à sable : le candidat doit s'assembler, rester borné en
    /// taille, **ne déclarer aucun import**, et exporter `run`. Un module avec
    /// import (tentative d'accès host) est rejeté.
    fn safety_check(&self, cand: &String) -> Result<(), SafetyViolation> {
        if cand.len() > MAX_WAT_BYTES {
            return Err(SafetyViolation(format!(
                "WAT trop long ({} > {MAX_WAT_BYTES} octets)",
                cand.len()
            )));
        }
        let (_engine, module) = Self::compile(cand).map_err(SafetyViolation)?;
        if module.imports().len() > 0 {
            return Err(SafetyViolation(
                "imports interdits (le code candidat doit être pur, sans accès host)".to_string(),
            ));
        }
        let has_run = module.exports().any(|e| e.name() == "run");
        if !has_run {
            return Err(SafetyViolation("export 'run' manquant".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ascend_llm, LlmGuard, LlmStop, MockLlmClient};

    /// WAT correct pour `run(x) = x*x + 1`.
    const SQUARE_PLUS_ONE: &str = "(module (func (export \"run\") (param i64) (result i64) \
        (i64.add (i64.mul (local.get 0) (local.get 0)) (i64.const 1))))";

    fn task() -> WasmSynthesis {
        WasmSynthesis::from_target(|x| x * x + 1, [-3, -1, 0, 2, 4, 5])
    }

    #[test]
    fn correct_module_passes_all_cases() {
        let t = task();
        assert_eq!(t.pass_fraction(SQUARE_PLUS_ONE), 1.0);
        // le score est élevé (fraction 1.0 moins une petite pénalité de taille)
        assert!(t.score(&SQUARE_PLUS_ONE.to_string()) > 0.9);
    }

    #[test]
    fn seed_is_valid_but_poor() {
        let t = task();
        // run(x)=0 ne réussit que le cas où la cible vaut 0 (aucun ici) ⇒ ~0
        assert!(t.score(&t.seed_candidate()) < 0.2);
        // mais il passe safety_check (valide, pur, exporte run)
        assert!(t.safety_check(&t.seed_candidate()).is_ok());
    }

    #[test]
    fn safety_rejects_imports_and_garbage_and_missing_export() {
        let t = task();
        // import d'une fonction host → rejeté
        let with_import = "(module (import \"env\" \"f\" (func)) \
            (func (export \"run\") (param i64) (result i64) (i64.const 0)))";
        assert!(t.safety_check(&with_import.to_string()).is_err());
        // WAT invalide → rejeté
        assert!(t.safety_check(&"pas du wat".to_string()).is_err());
        // pas d'export run → rejeté
        let no_run = "(module (func (export \"other\") (result i64) (i64.const 0)))";
        assert!(t.safety_check(&no_run.to_string()).is_err());
    }

    #[test]
    fn fuel_bounds_infinite_loop() {
        let t = task();
        // boucle infinie : doit épuiser le fuel (pas de hang) ⇒ aucun cas réussi
        let looper = "(module (func (export \"run\") (param i64) (result i64) \
            (loop (br 0)) (i64.const 0)))";
        // safety_check OK (valide, pur, exporte run), mais score 0 (trap fuel)
        assert!(t.safety_check(&looper.to_string()).is_ok());
        assert_eq!(t.pass_fraction(looper), 0.0);
    }

    #[test]
    fn llm_path_synthesizes_wasm_via_mock() {
        let mut t = task();
        let client = MockLlmClient::new(|_p, _k| {
            vec![
                // d'abord un module médiocre (constante), puis la solution exacte
                "(module (func (export \"run\") (param i64) (result i64) (i64.const 1)))"
                    .to_string(),
                SQUARE_PLUS_ONE.to_string(),
            ]
        });
        let guard = LlmGuard {
            target: Some(0.9),
            patience: 3,
            max_iters: 20,
            ..LlmGuard::default()
        };
        let seed = t.seed_candidate();
        let (best, report) = ascend_llm(&mut t, seed, &client, &guard);
        assert!(report.is_monotone());
        assert!(report.accepted > 0);
        assert_eq!(t.pass_fraction(&best), 1.0, "best ne calcule pas x²+1");
        assert_eq!(report.stop, LlmStop::Target);
    }

    /// Audit a5 : un module déclarant plus de mémoire que le plafond trappe
    /// à l'instanciation au lieu de consommer l'hôte.
    #[test]
    fn static_memory_over_limit_traps() {
        // 300 pages × 64 Kio = ~19 Mio > MAX_MEMORY_BYTES (16 Mio)
        let hog = "(module (memory 300) (func (export \"run\") (param i64) (result i64) (local.get 0)))";
        let (engine, module) = WasmSynthesis::compile(hog).unwrap();
        assert!(
            WasmSynthesis::run_once(&engine, &module, 1).is_err(),
            "la limite mémoire doit refuser l'instanciation"
        );
        // dans les clous : fonctionne toujours
        let ok = "(module (memory 1) (func (export \"run\") (param i64) (result i64) (local.get 0)))";
        let (engine, module) = WasmSynthesis::compile(ok).unwrap();
        assert_eq!(WasmSynthesis::run_once(&engine, &module, 7), Ok(7));
    }

    // ─── A6 — WasmEvaluator branché dans la boucle DGM ───────────────────── //

    use crate::dgm::{Archive, DgmConfig, DgmEngine, Patch, Proposal, Proposer};
    use crate::rng::Rng;
    use std::path::Path;

    fn write_ws(tag: &str, wat: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rsi-wasmeval-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("candidate.wat"), wat).unwrap();
        dir
    }

    /// Évaluateur WASM sur un workspace : module correct → tout-au-vert.
    #[test]
    fn wasm_evaluator_scores_workspace() {
        let ws = write_ws("good", SQUARE_PLUS_ONE);
        let ev = WasmEvaluator::for_target("candidate.wat", |x| x * x + 1, [-2, 0, 3]);
        let fit = crate::dgm::Evaluator::evaluate(&ev, &ws).unwrap();
        assert!(fit.compiles && fit.all_green());
        assert!(fit.score > 0.95);

        // WAT cassé → barrière de compilation
        let bad = write_ws("bad", "(module (func");
        let fit = crate::dgm::Evaluator::evaluate(&ev, &bad).unwrap();
        assert!(!fit.compiles);

        // boucle infinie → compile mais fuel épuisé ⇒ tests échoués
        let looper = write_ws(
            "loop",
            "(module (func (export \"run\") (param i64) (result i64) (loop (br 0)) (i64.const 0)))",
        );
        let fit = crate::dgm::Evaluator::evaluate(&ev, &looper).unwrap();
        assert!(fit.compiles && !fit.all_green() && fit.tests_failed == 3);

        for d in [ws, bad, looper] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    /// Proposeur déterministe qui améliore `candidate.wat` pas à pas
    /// (`run(x)=0` → `x²+1`).
    struct WatImprover;
    impl Proposer for WatImprover {
        fn propose(
            &self,
            ctx: &crate::dgm::ImprovementContext<'_>,
            _rng: &mut Rng,
        ) -> crate::dgm::Result<Option<Proposal>> {
            let cur = ctx.read("candidate.wat").unwrap_or_default();
            if cur.contains("(i64.const 0)))") {
                return Ok(Some(Proposal {
                    patch: Patch::new(
                        "candidate.wat",
                        "(i64.const 0)",
                        "(i64.add (i64.mul (local.get 0) (local.get 0)) (i64.const 1))",
                    ),
                    rationale: "compute x*x+1".into(),
                }));
            }
            Ok(None)
        }
    }

    /// Boucle DGM complète avec l'évaluateur WASM : le patch n'est archivé
    /// que s'il prouve son amélioration EN BAC À SABLE (aucun cargo lancé).
    #[test]
    fn dgm_loop_uses_wasm_sandbox_evaluator() {
        let seed_wat = "(module (func (export \"run\") (param i64) (result i64) (i64.const 0)))";
        let ws = write_ws("dgm", seed_wat);
        let baseline = crate::dgm::Fitness {
            compiles: true,
            tests_passed: 0,
            tests_failed: 6,
            score: 0.0,
            notes: "baseline run(x)=0".into(),
        };
        let eval = WasmEvaluator::for_target("candidate.wat", |x| x * x + 1, [-3, -1, 0, 2, 4, 5]);
        let mut eng = DgmEngine::new(
            Archive::with_root(baseline),
            WatImprover,
            eval,
            DgmConfig::new(&ws, "make run compute the target"),
            7,
        );
        eng.run(3).unwrap();
        let best = eng.best().unwrap().fitness.as_ref().unwrap();
        assert!(best.all_green(), "le candidat corrigé doit être tout-au-vert");
        assert_eq!(best.tests_failed, 0);
        assert_eq!(eng.archive().len(), 2, "seed + 1 variante acceptée");
        // l'arbre vivant n'a jamais été touché
        assert!(
            std::fs::read_to_string(ws.join("candidate.wat"))
                .unwrap()
                .contains("(i64.const 0)")
        );
        let _ = std::fs::remove_dir_all(&ws);
        let _ = Path::new(&ws).exists();
    }
}
