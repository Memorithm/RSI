//! `rsi-science` — typed PAPERS scientific intake for RSI.
//!
//! This command intentionally stops before DGM execution. It converts a paper
//! (or an existing v1 bundle) into traceable directive goals. The existing
//! empirical `rsi-dgm` / GuardedDgm path remains the authority that may test and
//! promote code.

use std::path::{Path, PathBuf};
use std::process::exit;

use rsi::paper_science::{PaperAnalysisMode, ScientificBundle, ScientificPapersRunner};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 1 || has_flag(&args, "--help") || has_flag(&args, "-h") {
        usage();
        return;
    }

    let paper = flag_value(&args, "--paper");
    let bundle_path = flag_value(&args, "--bundle").map(PathBuf::from);
    if paper.is_some() == bundle_path.is_some() {
        fail("exactement un de --paper ou --bundle est requis");
    }

    let target = flag_value(&args, "--target").unwrap_or_else(|| "workspace".into());
    let max_goals = flag_value(&args, "--max-goals")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3)
        .clamp(1, 32);

    let provider = flag_value(&args, "--provider");
    let model = flag_value(&args, "--model");
    let mode = match (provider, model) {
        (None, None) => PaperAnalysisMode::Heuristic,
        (Some(provider), Some(model)) => PaperAnalysisMode::Model { provider, model },
        _ => fail("--provider et --model doivent être fournis ensemble"),
    };

    let bundle = if let Some(path) = bundle_path {
        if !matches!(mode, PaperAnalysisMode::Heuristic) {
            fail("--provider/--model ne s'appliquent pas à --bundle");
        }
        read_bundle(&path)
    } else {
        let source = paper.expect("paper checked above");
        let output = flag_value(&args, "--out")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".rsi_science"));
        let runner = ScientificPapersRunner::from_environment();
        match runner.analyze_bundle(&source, &output, &mode) {
            Ok(bundle) => bundle,
            Err(error) => fail(&format!("analyse PAPERS scientifique: {error}")),
        }
    };

    let goals = bundle.directive_goals(&target, max_goals);
    println!("schema={}", rsi::paper_science::SCIENTIFIC_BUNDLE_SCHEMA);
    println!("paper_id={}", bundle.paper_id);
    println!("analysis_sha256={}", bundle.provenance.analysis_sha256);
    println!("claims={}", bundle.claims.len());
    println!("actionable_goals={}", goals.len());

    if goals.is_empty() {
        println!("Aucun claim de méthode actionnable. Aucune amélioration n'est supposée ni promue.");
        return;
    }

    for (index, goal) in goals.iter().enumerate() {
        println!("\nGOAL {}\n{}", index + 1, goal);
    }
    println!(
        "\nCes GOAL sont des hypothèses. Utiliser le gate empirique RSI/GuardedDgm avant toute promotion."
    );
}

fn read_bundle(path: &Path) -> ScientificBundle {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|error| fail(&format!("lecture {}: {error}", path.display())));
    ScientificBundle::parse(&raw)
        .unwrap_or_else(|error| fail(&format!("bundle {} invalide: {error}", path.display())))
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].clone())
}

fn fail(message: &str) -> ! {
    eprintln!("erreur: {message}");
    exit(2)
}

fn usage() {
    println!(
        r#"rsi-science — PAPERS scientific bundle v1 → objectifs RSI traçables

Entrée existante:
  rsi-science --bundle scientific_bundle.json --target src/kernel.rs [--max-goals 3]

Analyse d'un paper:
  rsi-science --paper <PDF|arXiv|URL> --out .rsi_science --target src/kernel.rs
  rsi-science --paper <source> --provider ollama --model <model> --target src/kernel.rs

Variables:
  RSI_PAPERS_BIN           binaire `papers`
  RSI_PAPERS_CONTRACT_BIN  binaire `papers-contract`

La commande ne modifie pas le dépôt et ne promeut aucun patch. Les objectifs
produits doivent passer par l'évaluation empirique RSI/GuardedDgm."#
    );
}
