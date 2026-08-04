//! `rsi-simcal` — **calibration du pré-crible simulé** (Qwen-AgentWorld × DGM).
//!
//! Mesure, sur du VRAI code RSI et avec **vérité terrain**, la fidélité d'un
//! world model comme prédicteur du gate DGM. Boucle : le proposeur (modèle de
//! code, connexion automatique) génère de vrais patchs ; pour chacun, le
//! **world model** prédit le verdict (`cargo build`+`test`) ET le **gate réel**
//! le tranche. On accumule une matrice de confusion (cf. `docs/AGENTWORLD_STUDY.md`).
//!
//! C'est la décision go/no-go de l'axe « surrogate gate » : si la fidélité est
//! bonne, on peut pré-cribler les candidats (×3-10 explorés par build réel) ;
//! sinon on l'abandonne, chiffres à l'appui. **Aucune écriture de l'arbre vivant.**
//!
//! ```text
//! rsi-simcal <workspace> --goal "..." --allow src/a.rs [options]
//!   --sim-model NAME     world model Ollama       (défaut « agentworld »)
//!   --model NAME         proposeur de code        (sinon découverte auto)
//!   --steps N            patchs à calibrer        (défaut 10)
//!   --bench "ARGS" / --seed N / --timeout SECS / --out RAPPORT.md
//! ```

use std::process::exit;
use std::time::Duration;

use rsi::dgm::{
    CargoEvaluator, Evaluator, Fitness, ImprovementContext, LlmCodeModel, LlmProposer, Proposer,
    WorkspaceSnapshot,
};
use rsi::llm::{LlmClient, OllamaClient};
use rsi::rng::Rng;
use rsi::simulation::{build_sim_prompt, parse_sim_verdict, CalibrationStats, SimVerdict};

const VALUE_FLAGS: &[&str] = &[
    "--goal",
    "--allow",
    "--sim-model",
    "--model",
    "--steps",
    "--bench",
    "--seed",
    "--timeout",
    "--out",
    "--ollama-host",
    "--ollama-port",
    "--sim-num-predict",
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
    let goal = required(&args, "--goal");
    let allowed: Vec<String> = required(&args, "--allow")
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if allowed.is_empty() {
        eprintln!("erreur : --allow doit lister au moins un fichier.");
        exit(2);
    }
    let steps: usize = flag_value(&args, "--steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(10)
        .clamp(1, 100);
    let seed: u64 = flag_value(&args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);
    let timeout_secs: u64 = flag_value(&args, "--timeout")
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let host = flag_value(&args, "--ollama-host").unwrap_or_else(|| "127.0.0.1".to_string());
    let port: u16 = flag_value(&args, "--ollama-port")
        .and_then(|v| v.parse().ok())
        .unwrap_or(11434);
    let bench: Vec<String> = flag_value(&args, "--bench")
        .map(|s| s.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default();
    let sim_model = flag_value(&args, "--sim-model").unwrap_or_else(|| "agentworld".to_string());
    // AgentWorld raisonne EN LONG (chain-of-thought) : le défaut num_predict de
    // 4096 le tronque avant la ligne de verdict (constaté : 15/15 done_reason=
    // length). On lui laisse beaucoup de place + un contexte qui l'englobe.
    let sim_num_predict: u32 = flag_value(&args, "--sim-num-predict")
        .and_then(|v| v.parse().ok())
        .unwrap_or(12288);
    let sim_num_ctx: u32 = sim_num_predict + 8192; // prompt (~5k) + génération

    // --- Proposeur : connexion automatique (comme rsi-dgm). ----------------- //
    let pref = flag_value(&args, "--model").or_else(|| std::env::var("RSI_LLM_MODEL").ok());
    let installed = match rsi::llm::ollama_installed_models(&host, port, Duration::from_secs(10)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("erreur : Ollama injoignable sur {host}:{port} ({e:?}).");
            exit(2);
        }
    };
    let code_model = match rsi::llm::pick_model(&installed, pref.as_deref()) {
        Some(m) => m,
        None => {
            eprintln!("erreur : aucun modèle de code utilisable sur {host}:{port}.");
            exit(2);
        }
    };
    if !installed
        .iter()
        .any(|m| m == &sim_model || m.starts_with(&sim_model))
    {
        eprintln!(
            "erreur : world model « {sim_model} » introuvable dans Ollama.\n  \
             Modèles : {}.\n  Cf. docs/AGENTWORLD_STUDY.md pour l'installer.",
            installed.join(", ")
        );
        exit(2);
    }
    println!("• proposeur (code)  : {code_model}");
    println!(
        "• world model (sim) : {sim_model} (num_predict={sim_num_predict}, num_ctx={sim_num_ctx})"
    );

    let proposer = LlmProposer::new(
        LlmCodeModel::new(
            OllamaClient::new(code_model)
                .with_endpoint(host.clone(), port)
                .with_timeout(Duration::from_secs(timeout_secs)),
        ),
        allowed.clone(),
    );
    let sim_client = OllamaClient::new(sim_model.clone())
        .with_endpoint(host.clone(), port)
        .with_timeout(Duration::from_secs(timeout_secs))
        .with_num_predict(sim_num_predict)
        .with_num_ctx(sim_num_ctx);
    let evaluator = CargoEvaluator {
        bench_command: bench,
        timeout: Duration::from_secs(timeout_secs),
        score_from_passrate: true,
        ..Default::default()
    };

    // --- Référence (fitness du parent remise au proposeur). ----------------- //
    eprintln!("• évaluation de la référence (arbre vivant, build+test)…");
    let baseline = match WorkspaceSnapshot::create(&ws).and_then(|s| evaluator.evaluate(s.root())) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("erreur : référence : {e}");
            exit(1);
        }
    };
    println!(
        "  référence : compiles={} passed={} failed={}\n",
        baseline.compiles, baseline.tests_passed, baseline.tests_failed
    );

    // --- Boucle de calibration. --------------------------------------------- //
    let mut rng = Rng::new(seed);
    let mut rejections: Vec<String> = Vec::new();
    let mut stats = CalibrationStats::default();
    let mut rows: Vec<String> = Vec::new();
    let mut n = 0usize;

    while n < steps {
        let ctx = ImprovementContext {
            workspace_root: &ws,
            goal: &goal,
            parent_fitness: Some(&baseline),
            recent_rejections: &rejections,
            simulated_feedback: &[],
            external_context: &[],
        };
        let proposal = match proposer.propose(&ctx, &mut rng) {
            Ok(Some(p)) => p,
            Ok(None) => {
                println!("  · pas de proposition (on continue)");
                // évite une boucle infinie si le modèle boude
                n += 1;
                continue;
            }
            Err(e) => {
                eprintln!("  · erreur du proposeur : {e}");
                break;
            }
        };
        let patch = &proposal.patch;

        // 1) Vérité terrain : gate réel sur copie isolée.
        let real: Fitness =
            match WorkspaceSnapshot::create(&ws).map(|snap| match snap.apply(patch) {
                Ok(()) => evaluator
                    .evaluate(snap.root())
                    .unwrap_or_else(|e| Fitness::broken(e.to_string())),
                Err(e) => Fitness::broken(format!("patch did not apply: {e}")),
            }) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("  · snapshot : {e}");
                    break;
                }
            };
        let actual = SimVerdict::from_real(real.compiles, real.tests_failed);

        // 2) Prédiction du world model (sur le fichier vivant + patch).
        let file_content = ctx.read(&patch.target).unwrap_or_default();
        let prompt = build_sim_prompt(&patch.target, &file_content, &patch.find, &patch.replace);
        let predicted = match sim_client.complete_raw(&prompt) {
            Ok(text) => parse_sim_verdict(&text),
            Err(e) => {
                eprintln!("  · world model : {e:?} (verdict indécis)");
                SimVerdict::default()
            }
        };

        stats.record(predicted, actual);
        let mark = |p: Option<bool>, a: Option<bool>| match (p, a) {
            (Some(x), Some(y)) if x == y => "✓",
            (None, _) => "?",
            _ => "✗",
        };
        let line = format!(
            "  #{n:2} {} · compile sim={} réel={} {} · tests sim={} réel={} {} · {}",
            short(patch),
            b(predicted.compiles),
            b(actual.compiles),
            mark(predicted.compiles, actual.compiles),
            b(predicted.tests_pass),
            b(actual.tests_pass),
            mark(predicted.tests_pass, actual.tests_pass),
            one_line(&proposal.rationale, 40),
        );
        println!("{line}");
        rows.push(line);

        if !real.all_green() {
            rejections.push(proposal.rationale.clone());
            if rejections.len() > 8 {
                rejections.remove(0);
            }
        }
        n += 1;
    }

    // --- Rapport. ------------------------------------------------------------ //
    println!("\n{}", stats.report());
    let mut md = String::from("# rsi-simcal — calibration du pré-crible simulé\n\n");
    md.push_str(&format!(
        "world model : `{sim_model}`\n\n```\n{}\n```\n\n",
        stats.report()
    ));
    md.push_str("## Détail par patch\n\n```\n");
    for r in &rows {
        md.push_str(r.trim_start());
        md.push('\n');
    }
    md.push_str(
        "```\n\n**Lecture** : `fn` (faux négatif) = le simulateur a écarté une \
                 amélioration réelle (idée perdue — le seul coût dangereux d'un pré-crible). \
                 `fp` ne coûte qu'un vrai build. Décision surrogate-gate : viser une \
                 exactitude « tests » élevée ET peu de `fn`.\n",
    );
    let out_path = flag_value(&args, "--out")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| ws.join(".rsi_simcal_report.md"));
    match std::fs::write(&out_path, &md) {
        Ok(()) => println!("• rapport : {}", out_path.display()),
        Err(e) => eprintln!("avertissement : écriture {} : {e}", out_path.display()),
    }
}

fn b(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "oui",
        Some(false) => "non",
        None => " ? ",
    }
}

fn short(p: &rsi::dgm::Patch) -> String {
    let t = &p.target;
    t.rsplit('/').next().unwrap_or(t).to_string()
}

fn one_line(s: &str, max: usize) -> String {
    let flat: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        flat.chars().take(max).collect::<String>() + "…"
    }
}

// ----------------------------- parsing args ------------------------------- //

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
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
        "rsi-simcal — calibre un world model (Qwen-AgentWorld) comme pré-crible du gate DGM\n\n\
         USAGE:\n  rsi-simcal <workspace> --goal \"...\" --allow src/a.rs [options]\n\n\
         Chaque patch proposé est jugé DEUX fois : par le gate réel (vérité terrain)\n\
         et par le world model. Sortie : matrice de confusion. N'écrit jamais l'arbre vivant.\n\n\
         Options :\n  \
           --sim-model NAME    world model Ollama (défaut « agentworld »)\n  \
           --model NAME        proposeur de code (sinon découverte auto)\n  \
           --steps N           patchs à calibrer (défaut 10)   --bench \"ARGS\"\n  \
           --sim-num-predict N tokens de génération du world model (défaut 12288 —\n                      il raisonne long ; trop bas ⇒ verdict tronqué/indécis)\n  \
           --seed N  --timeout SECS  --out RAPPORT.md  --ollama-host/-port\n\n\
         Cf. docs/AGENTWORLD_STUDY.md."
    );
}
