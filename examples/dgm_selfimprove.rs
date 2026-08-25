//! Exemple : la boucle d'auto-amélioration **empirique** (Darwin–Gödel / STOP)
//! du module [`rsi::dgm`] — port natif de `soul-rsi`.
//!
//! Lancement : `cargo run --release --example dgm_selfimprove`
//!
//! Le « code » ici est un fichier jouet `level.txt` contenant `level = N`. Le
//! proposeur incrémente N, l'évaluateur récompense un N plus grand. C'est une
//! démonstration **déterministe et hors-ligne** (ni LLM, ni `cargo`) du contrat :
//! propose → évalue en **copie isolée** → garde si **prouvé meilleur** → archive.
//!
//! En production, on remplace `Incrementer` par [`rsi::dgm::LlmProposer`] (avec
//! une liste blanche de fichiers) adossé à un backend RSI via
//! [`rsi::dgm::LlmCodeModel`], et `ClosureEvaluator` par
//! [`rsi::dgm::CargoEvaluator`] (build + tests réels, bornés). L'arbre vivant
//! n'est jamais touché tant qu'on n'appelle pas [`rsi::dgm::promote_to_live`].

use rsi::dgm::{
    Archive, ClosureEvaluator, DgmConfig, DgmEngine, Fitness, ImprovementContext, Patch, Proposal,
    Proposer, StepOutcome,
};
use rsi::rng::Rng;
use std::path::Path;

fn read_level(root: &Path) -> i64 {
    std::fs::read_to_string(root.join("level.txt"))
        .unwrap_or_default()
        .trim()
        .strip_prefix("level = ")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Proposeur jouet : « améliore » en montant le niveau d'une unité.
struct Incrementer;
impl Proposer for Incrementer {
    fn propose(
        &self,
        ctx: &ImprovementContext<'_>,
        _rng: &mut Rng,
    ) -> rsi::dgm::Result<Option<Proposal>> {
        let cur = read_level(ctx.workspace_root);
        let next = cur + 1;
        Ok(Some(Proposal {
            patch: Patch::new(
                "level.txt",
                format!("level = {cur}"),
                format!("level = {next}"),
            ),
            rationale: format!("raise level to {next}"),
        }))
    }
}

fn main() {
    // 1. Un workspace vivant jetable.
    let ws = std::env::temp_dir().join("rsi-dgm-example");
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(ws.join("level.txt"), "level = 0").unwrap();

    // 2. L'évaluateur empirique (ici une closure ; en prod : CargoEvaluator).
    let evaluator = ClosureEvaluator::new(|root: &Path| Fitness {
        compiles: true,
        tests_passed: 1,
        tests_failed: 0,
        score: read_level(root) as f64,
        notes: String::new(),
    });

    let baseline = Fitness {
        compiles: true,
        tests_passed: 1,
        tests_failed: 0,
        score: 0.0,
        notes: "baseline".to_string(),
    };

    let mut engine = DgmEngine::new(
        Archive::with_root(baseline),
        Incrementer,
        evaluator,
        DgmConfig::new(&ws, "raise the level as high as possible"),
        /* seed */ 42,
    );

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RSI · DGM — auto-amélioration empirique (propose→évalue→garde) ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let outcomes = engine.run(12).unwrap();
    for (i, o) in outcomes.iter().enumerate() {
        match o {
            StepOutcome::NoProposal { .. } => println!("  step {i:2} · pas de proposition"),
            StepOutcome::Evaluated {
                accepted,
                fitness,
                variant_id,
                ..
            } => println!(
                "  step {i:2} · {} · score={:>4} · variant={}",
                if *accepted { "ACCEPTÉ " } else { "rejeté  " },
                fitness.score,
                char_safe_short(variant_id),
            ),
        }
    }

    let best = engine.best().unwrap();
    println!(
        "\n  archive : {} variantes · meilleur score = {} (génération {})",
        engine.archive().len(),
        best.fitness.as_ref().unwrap().score,
        best.generation,
    );
    // L'arbre vivant n'a jamais été muté par la boucle.
    println!("  workspace vivant intact : level = {}", read_level(&ws));
    println!("  (promotion explicite via rsi::dgm::promote_to_live, non appelée ici)");

    let _ = std::fs::remove_dir_all(&ws);
}

/// Troncature alignée char-boundary (audit a21).
fn char_safe_short(s: &str) -> &str {
    let mut end = 8.min(s.len());
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
