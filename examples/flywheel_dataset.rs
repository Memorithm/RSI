//! `flywheel_dataset` — assemble un dataset de fine-tune à partir des JSONL de
//! trajectoires exportés par `rsi-dgm --export-trajectories` (axe 4).
//!
//! Fusionne plusieurs runs/cibles, déduplique, mesure l'équilibre des classes,
//! split train/eval déterministe, et écrit les fichiers prêts pour le
//! fine-tuning. Std-only, auto-découvert par cargo (aucune entrée `Cargo.toml`).

use rsi::flywheel::{dedup, split, stats, to_chat_jsonl};

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value_flags = ["--out", "--eval-frac", "--seed"];
    let inputs: Vec<String> = {
        let mut v = Vec::new();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            if a.starts_with("--") {
                i += if value_flags.contains(&a.as_str()) { 2 } else { 1 };
            } else {
                v.push(a.clone());
                i += 1;
            }
        }
        v
    };
    if inputs.is_empty() {
        eprintln!(
            "usage : flywheel_dataset <run1.jsonl> [run2.jsonl …] \
             [--out PREFIX] [--eval-frac F] [--seed N] [--chat]"
        );
        std::process::exit(2);
    }
    let out = flag(&args, "--out").unwrap_or_else(|| "dataset".to_string());
    let eval_frac: f64 = flag(&args, "--eval-frac").and_then(|v| v.parse().ok()).unwrap_or(0.15);
    let seed: u64 = flag(&args, "--seed").and_then(|v| v.parse().ok()).unwrap_or(2026);
    let chat = args.iter().any(|a| a == "--chat");

    let mut all = Vec::new();
    for path in &inputs {
        match std::fs::read_to_string(path) {
            Ok(content) => all.extend(content.lines().map(|l| l.to_string())),
            Err(e) => {
                eprintln!("erreur : lecture de {path} : {e}");
                std::process::exit(1);
            }
        }
    }

    let (unique, removed) = dedup(all);
    let report = stats(&unique, removed);
    println!("{}", report.summary());

    let (train, eval) = split(&unique, eval_frac, seed);
    let render = |lines: &[String]| -> String {
        if chat { to_chat_jsonl(lines) } else { lines.join("\n") + if lines.is_empty() { "" } else { "\n" } }
    };
    let train_path = format!("{out}_train.jsonl");
    let eval_path = format!("{out}_eval.jsonl");
    if let Err(e) = std::fs::write(&train_path, render(&train)) {
        eprintln!("erreur : écriture de {train_path} : {e}");
        std::process::exit(1);
    }
    if let Err(e) = std::fs::write(&eval_path, render(&eval)) {
        eprintln!("erreur : écriture de {eval_path} : {e}");
        std::process::exit(1);
    }
    println!(
        "\n→ {} ({} paires) + {} ({} paires){}",
        train_path,
        train.len(),
        eval_path,
        eval.len(),
        if chat { " [format chat]" } else { " [format prompt/completion]" }
    );
    if !report.is_balanced(0.20) {
        eprintln!(
            "note : dataset déséquilibré — accumule des cibles variées (matmul casse \
             souvent, json passe souvent) pour un world model qui prédit bien les deux."
        );
    }
}
