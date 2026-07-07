//! `rsi-scholar` — **apprendre de la littérature pour s'améliorer** :
//! papier scientifique → objectifs DGM directifs → preuves empiriques.
//!
//! Vitrine de l'addon **papers-agent** (cf. `docs/PAPERS_SYNERGY.md`,
//! `docs/ADDONS.md`) : `papers analyze` extrait les techniques du papier, RSI
//! les convertit en objectifs **directifs** (la campagne Jetson a montré que
//! le directif débloque ce que le vague n'obtient pas) et lance une boucle
//! DGM par objectif. Le gate cargo build+test+bench ne garde que ce qui est
//! **prouvé** — le papier propose, le moteur dispose.
//!
//! **DRY-RUN STRICT** : rsi-scholar n'écrit jamais l'arbre vivant. Chaque
//! amélioration trouvée est rapportée avec la commande `rsi-dgm … --promote`
//! à lancer soi-même après revue du diff.
//!
//! ```text
//! rsi-scholar <workspace> --paper <PDF|arXiv|URL> --allow src/a.rs [options]
//!
//! Options :
//!   --paper SOURCE        papier à exploiter                      (requis)
//!   --allow LIST          fichiers éditables (',')                (requis)
//!   --target TEXT         description de la cible pour les objectifs
//!                         (défaut : la liste --allow)
//!   --bench "ARGS"        bench cargo imprimant RSI_BENCH_SCORE=<f64>
//!   --max-goals N         objectifs max tirés du papier           (défaut 3)
//!   --steps N             étapes DGM par objectif                 (défaut 6)
//!   --min-gain FRAC       seuil anti-bruit                        (défaut 0.05)
//!   --model NAME          préférence de modèle (sinon découverte auto)
//!   --seed N / --timeout SECS / --out RAPPORT.md
//! ```
//!
//! LLM : connexion automatique (mêmes règles que `rsi-dgm --backend auto`).
//! Addon : binaire `papers` via `RSI_PAPERS_BIN` ou le PATH.

use std::process::exit;
use std::time::Duration;

use rsi::addons::{PaperAnalysis, PapersAddon};
use rsi::dgm::{
    Archive, CargoEvaluator, DgmConfig, DgmEngine, Evaluator, LlmCodeModel, LlmProposer,
    StepOutcome, WorkspaceSnapshot,
};

const VALUE_FLAGS: &[&str] = &[
    "--paper", "--allow", "--target", "--bench", "--max-goals", "--steps", "--min-gain",
    "--model", "--seed", "--timeout", "--out", "--paper-model",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || matches!(args[1].as_str(), "-h" | "--help" | "help") {
        usage();
        exit(if args.len() < 2 { 2 } else { 0 });
    }
    let ws = match first_positional(&args) {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            eprintln!("erreur : racine du workspace manquante.\n");
            usage();
            exit(2);
        }
    };
    if !ws.is_dir() {
        eprintln!("erreur : '{}' n'est pas un répertoire.", ws.display());
        exit(2);
    }
    let paper = required(&args, "--paper");
    let allow_raw = required(&args, "--allow");
    let allowed: Vec<String> =
        allow_raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if allowed.is_empty() {
        eprintln!("erreur : --allow doit lister au moins un fichier.");
        exit(2);
    }
    let target = flag_value(&args, "--target").unwrap_or_else(|| allowed.join(", "));
    let max_goals: usize =
        flag_value(&args, "--max-goals").and_then(|v| v.parse().ok()).unwrap_or(3).clamp(1, 10);
    let steps: usize =
        flag_value(&args, "--steps").and_then(|v| v.parse().ok()).unwrap_or(6).clamp(1, 64);
    let min_gain: f64 = flag_value(&args, "--min-gain").and_then(|v| v.parse().ok()).unwrap_or(0.05);
    let seed: u64 = flag_value(&args, "--seed").and_then(|v| v.parse().ok()).unwrap_or(42);
    let timeout_secs: u64 =
        flag_value(&args, "--timeout").and_then(|v| v.parse().ok()).unwrap_or(300);
    let bench: Vec<String> = flag_value(&args, "--bench")
        .map(|s| s.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default();

    // --- 1. Addon papers (détection, dégradation claire). ------------------- //
    let papers = match PapersAddon::detect() {
        Some(p) => p,
        None => {
            eprintln!(
                "erreur : addon papers-agent introuvable.\n  Installez PAPERS V2 (papers_core) \
                 et exposez le binaire : PATH ou RSI_PAPERS_BIN=/chemin/papers.\n  \
                 Cf. docs/ADDONS.md."
            );
            exit(2);
        }
    };
    println!("• addon papers-agent : détecté");

    // --- 2. Analyse du papier → objectifs directifs. ------------------------ //
    // `--paper-llm` : analyse LLM multi-passes de PAPERS (extraction de VRAIES
    // techniques ; l'heuristique --no-llm ne rend qu'un placeholder — vécu).
    // `--paper-model` : modèle passé à PAPERS (défaut PAPERS : gemma4:e2b).
    let paper_llm = args.iter().any(|a| a == "--paper-llm");
    let paper_model = flag_value(&args, "--paper-model");
    if paper_model.is_some() && !paper_llm {
        eprintln!(
            "⚠️  --paper-model ignoré sans --paper-llm : l'analyse heuristique \
             (--no-llm) n'utilise aucun modèle."
        );
    }
    let llm_model: Option<String> = if paper_llm { Some(paper_model.unwrap_or_default()) } else { None };
    println!(
        "• analyse du papier « {paper} » (papers analyze{})…",
        if paper_llm { ", LLM — peut prendre plusieurs minutes" } else { " --no-llm" }
    );
    let out_dir = ws.join(".rsi_scholar");
    let analysis: PaperAnalysis = match papers.analyze(&paper, &out_dir, llm_model.as_deref()) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("erreur : analyse du papier : {e}");
            exit(1);
        }
    };
    if !analysis.summary.is_empty() {
        println!("  résumé : {}", one_line(&analysis.summary, 160));
    }
    for c in analysis.contributions.iter().take(3) {
        println!("  contribution : {}", one_line(c, 140));
    }
    let goals = PapersAddon::directive_goals(&analysis, &target, max_goals);
    if goals.is_empty() {
        println!(
            "\n  aucune technique exploitable extraite du papier — rien à tenter.\n  {}",
            if paper_llm {
                "(essayez un papier plus algorithmique, ou --paper-model <modèle plus fort>)"
            } else {
                "L'analyse heuristique (--no-llm) n'extrait pas les techniques : relancez \
                 avec --paper-llm (analyse LLM de PAPERS, modèle via --paper-model)."
            }
        );
        return;
    }
    println!("  {} objectif(s) directif(s) tiré(s) du papier\n", goals.len());

    // --- 3. LLM : connexion automatique. ------------------------------------ //
    let host = std::env::var("RSI_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 =
        std::env::var("RSI_OLLAMA_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(11434);
    let pref = flag_value(&args, "--model").or_else(|| std::env::var("RSI_LLM_MODEL").ok());
    let installed = match rsi::llm::ollama_installed_models(&host, port, Duration::from_secs(10)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("erreur : Ollama injoignable sur {host}:{port} ({e:?}).");
            exit(2);
        }
    };
    let model = match rsi::llm::pick_model(&installed, pref.as_deref()) {
        Some(m) => m,
        None => {
            eprintln!("erreur : aucun modèle utilisable sur {host}:{port}.");
            exit(2);
        }
    };
    println!("• backend ollama : modèle « {model} » ({} installé(s))", installed.len());

    // --- 4. Référence unique (l'arbre vivant ne change pas entre objectifs). - //
    let evaluator = || CargoEvaluator {
        bench_command: bench.clone(),
        timeout: Duration::from_secs(timeout_secs),
        score_from_passrate: true,
        ..Default::default()
    };
    eprintln!("• évaluation de la référence (arbre vivant, build+test)…");
    let baseline = match WorkspaceSnapshot::create(&ws).and_then(|s| evaluator().evaluate(s.root()))
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("erreur : référence : {e}");
            exit(1);
        }
    };
    println!(
        "  référence : compiles={} passed={} failed={} score={:.4}\n",
        baseline.compiles, baseline.tests_passed, baseline.tests_failed, baseline.score
    );

    // --- 5. Une boucle DGM par objectif (dry-run). --------------------------- //
    struct GoalResult {
        goal: String,
        accepted: usize,
        best_score: Option<f64>,
        best_variant: Option<String>,
    }
    let mut results: Vec<GoalResult> = Vec::new();
    for (gi, goal) in goals.iter().enumerate() {
        println!("═══ objectif {}/{} ═══", gi + 1, goals.len());
        println!("  {}\n", one_line(goal, 200));
        let client = rsi::llm::OllamaClient::new(model.clone())
            .with_endpoint(host.clone(), port)
            .with_timeout(Duration::from_secs(timeout_secs));
        let proposer = LlmProposer::new(LlmCodeModel::new(client), allowed.clone());
        let mut config = DgmConfig::new(&ws, goal);
        config.min_score_gain = min_gain;
        let mut engine = DgmEngine::new(
            Archive::with_root(baseline.clone()),
            proposer,
            evaluator(),
            config,
            seed + gi as u64,
        );
        let mut accepted = 0usize;
        for i in 0..steps {
            let o = match engine.step() {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("  step {i:2} · erreur : {e}");
                    break;
                }
            };
            match &o {
                StepOutcome::NoProposal { reason } => match reason {
                    Some(r) => println!("  step {i:2} · pas de proposition ({r})"),
                    None => println!("  step {i:2} · pas de proposition"),
                },
                StepOutcome::Evaluated { accepted: acc, fitness, variant_id, .. } => {
                    if *acc {
                        accepted += 1;
                    }
                    println!(
                        "  step {i:2} · {} · compiles={} passed={} failed={} score={:.4} · {}",
                        if *acc { "ACCEPTÉ " } else { "rejeté  " },
                        fitness.compiles,
                        fitness.tests_passed,
                        fitness.tests_failed,
                        fitness.score,
                        &variant_id[..8.min(variant_id.len())],
                    );
                }
            }
        }
        let best = engine.best().filter(|v| v.parent.is_some());
        results.push(GoalResult {
            goal: goal.clone(),
            accepted,
            best_score: best.and_then(|v| v.fitness.as_ref()).map(|f| f.score),
            best_variant: best.map(|v| v.id.clone()),
        });
        println!();
    }

    // --- 6. Rapport. ---------------------------------------------------------- //
    let mut md = String::new();
    md.push_str(&format!(
        "# rsi-scholar — rapport\n\n**Papier** : {paper}\n\n**Cible** : {target}\n\n\
         **Référence** : score {:.4}, {} tests verts\n\n",
        baseline.score, baseline.tests_passed
    ));
    if !analysis.summary.is_empty() {
        md.push_str(&format!("**Résumé du papier** : {}\n\n", analysis.summary));
    }
    md.push_str("| # | Objectif (technique du papier) | Acceptés | Meilleur score | Variant |\n");
    md.push_str("|---|-------------------------------|----------|----------------|--------|\n");
    let mut any = false;
    for (i, r) in results.iter().enumerate() {
        any |= r.accepted > 0;
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            i + 1,
            one_line(&r.goal, 90),
            r.accepted,
            r.best_score.map(|s| format!("{s:.4}")).unwrap_or_else(|| "—".into()),
            r.best_variant.as_deref().map(|v| &v[..8.min(v.len())]).unwrap_or("—"),
        ));
    }
    md.push_str(&format!(
        "\n**DRY-RUN** : rien n'a été appliqué. Pour promouvoir une amélioration : relancer\n\
         `rsi-dgm {} --goal \"<objectif>\" --allow {} {}--min-gain {} --promote`\n\
         puis **revoir le diff** avant de committer (doctrine : le gate prouve la vitesse,\n\
         la revue garde le contrat).\n",
        ws.display(),
        allowed.join(","),
        if bench.is_empty() { String::new() } else { format!("--bench \"{}\" ", bench.join(" ")) },
        min_gain,
    ));
    let out_path = flag_value(&args, "--out")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| out_dir.join("rapport.md"));
    if let Err(e) = std::fs::write(&out_path, &md) {
        eprintln!("avertissement : écriture du rapport {} : {e}", out_path.display());
    } else {
        println!("• rapport : {}", out_path.display());
    }
    if any {
        println!("✓ le papier a produit au moins une amélioration PROUVÉE (dry-run) — voir le rapport.");
    } else {
        println!("∅ aucune amélioration au-dessus du seuil — verdict honnête, le papier n'a rien apporté ici.");
    }
}

fn one_line(s: &str, max: usize) -> String {
    let flat: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        let cut: String = flat.chars().take(max).collect();
        format!("{cut}…")
    }
}

// ----------------------------- parsing args ------------------------------- //

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn required(args: &[String], flag: &str) -> String {
    match flag_value(args, flag) {
        Some(v) => v,
        None => {
            eprintln!("erreur : {flag} est requis.\n");
            usage();
            exit(2);
        }
    }
}

fn first_positional(args: &[String]) -> Option<String> {
    if !args[1].starts_with("--") {
        return Some(args[1].clone());
    }
    let mut i = 2;
    while i < args.len() {
        let a = &args[i];
        if a.starts_with("--") {
            if VALUE_FLAGS.contains(&a.as_str()) {
                i += 2;
            } else {
                i += 1;
            }
        } else {
            return Some(a.clone());
        }
    }
    None
}

fn usage() {
    eprintln!(
        "rsi-scholar — papier scientifique → objectifs DGM directifs → preuves empiriques\n\n\
         USAGE:\n  rsi-scholar <workspace> --paper <PDF|arXiv|URL> --allow src/a.rs [options]\n\n\
         DRY-RUN STRICT : l'arbre vivant n'est jamais écrit ; le rapport donne la\n\
         commande rsi-dgm --promote à lancer soi-même après revue du diff.\n\n\
         Options :\n  \
           --paper SOURCE      papier (requis)      --allow LIST  fichiers (requis)\n  \
           --target TEXT       cible des objectifs  (défaut : la liste --allow)\n  \
           --bench \"ARGS\"      score = RSI_BENCH_SCORE (perf réelle)\n  \
           --max-goals N       objectifs max (défaut 3)   --steps N  par objectif (défaut 6)\n  \
           --min-gain FRAC     anti-bruit (défaut 0.05)   --model NAME (sinon auto)\n  \
           --paper-llm         analyse LLM de PAPERS (extrait les VRAIES techniques ;\n                      sans lui, l'heuristique ne rend que des métadonnées)\n  \
           --paper-model NAME  modèle de l'analyse PAPERS (défaut : celui de PAPERS)\n  \
           --seed N  --timeout SECS  --out RAPPORT.md\n\n\
         Addon papers-agent requis : binaire `papers` sur le PATH ou RSI_PAPERS_BIN.\n\
         LLM : connexion automatique (Ollama local, modèle découvert)."
    );
}
