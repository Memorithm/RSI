//! `rsi-dgm` — boucle d'auto-amélioration **empirique** (Darwin–Gödel / STOP)
//! sur un dépôt **réel**, depuis le shell. Pilote [`rsi::dgm`].
//!
//! **Sûr par défaut : DRY-RUN.** La boucle propose des patchs, les évalue dans
//! des **copies isolées** (`cargo build`+`test`) et archive les meilleurs, mais
//! **n'écrit JAMAIS l'arbre vivant** — sauf si l'on passe `--promote`, qui
//! applique le **seul meilleur** variant *tout-au-vert* au dépôt (avec
//! sauvegarde réversible).
//!
//! Backend LLM : **connexion automatique par défaut** (`--backend auto`) —
//! sonde Ollama local, découvre les modèles installés (`/api/tags`) et choisit
//! le meilleur modèle de code ([`rsi::llm::pick_model`]) ; retombe sur Claude
//! si `ANTHROPIC_API_KEY` est présent (feature `llm-claude-ureq`). Contrôlable
//! sans flags via `RSI_LLM` (« ollama:MODELE », « claude », « auto ») et
//! `RSI_LLM_MODEL` (préférence de modèle, exacte ou par préfixe).
//!
//! ```text
//! rsi-dgm <workspace> --goal "..." --allow src/a.rs,src/b.rs [options]
//!
//! Options :
//!   --goal TEXT           objectif remis au proposeur            (requis)
//!   --allow LIST          fichiers éditables (séparés par ',')    (requis)
//!   --steps N             nombre d'étapes                         (défaut 10)
//!   --seed N              graine déterministe                     (défaut 42)
//!   --package-subdir DIR  sous-répertoire de manifeste à builder  (défaut: racine)
//!   --test-args "ARGS"    args additionnels pour `cargo test`     (défaut: aucun)
//!   --backend auto|ollama|claude                                  (défaut auto)
//!   --model NAME          modèle LLM                    (défaut : auto-découvert)
//!   --ollama-host HOST                                            (défaut 127.0.0.1)
//!   --ollama-port PORT                                            (défaut 11434)
//!   --timeout SECS        délai max par invocation cargo          (défaut 300)
//!   --promote             applique le meilleur variant à l'arbre vivant
//!   --backups DIR         sauvegardes pour --promote        (défaut <ws>/.rsi_backups)
//! ```

use std::process::exit;
use std::time::Duration;

use rsi::dgm::{
    Archive, CargoEvaluator, CodeModel, DgmConfig, DgmEngine, DgmError, Evaluator, LlmCodeModel,
    LlmProposer, Prediction, StepOutcome, VerdictPredictor, WorkspaceSnapshot,
};

const VALUE_FLAGS: &[&str] = &[
    "--goal", "--allow", "--steps", "--seed", "--package-subdir", "--test-args", "--backend",
    "--model", "--ollama-host", "--ollama-port", "--timeout", "--backups", "--bench", "--min-gain",
    "--prescreen-model", "--prescreen-num-predict", "--proposer-num-predict", "--revise", "--export-trajectories",
    "--claude-max-tokens", "--claude-base-url", "--temperature", "--top-p",
];

/// Pré-crible par world model (Qwen-AgentWorld) : prédit le verdict d'un patch
/// pour éviter des `cargo build` sûrement perdus. Cf. `docs/AGENTWORLD_STUDY.md`.
/// N'écarte jamais une amélioration — ne rend `Some(false)` que sur un verdict
/// **conclu négatif** ; erreur réseau ⇒ `None` (indécis) ⇒ gate réel.
struct WorldModelPredictor {
    client: rsi::llm::OllamaClient,
}

impl VerdictPredictor for WorldModelPredictor {
    fn predict(&self, target: &str, content: &str, find: &str, replace: &str) -> Prediction {
        use rsi::llm::LlmClient;
        let prompt = rsi::simulation::build_sim_prompt(target, content, find, replace);
        let text = match self.client.complete_raw(&prompt) {
            Ok(t) => t,
            Err(_) => return Prediction::default(),
        };
        let v = rsi::simulation::parse_sim_verdict(&text);
        let (pass, reason) = if v.compiles == Some(false) {
            (Some(false), Some("le simulateur prédit que ce patch NE COMPILE PAS".to_string()))
        } else if v.tests_pass == Some(false) {
            (Some(false), Some("le simulateur prédit que ce patch CASSE des tests".to_string()))
        } else if v.tests_pass == Some(true) {
            (Some(true), None)
        } else {
            (None, None)
        };
        Prediction { pass, reason }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || matches!(args[1].as_str(), "-h" | "--help" | "help") {
        usage();
        exit(if args.len() < 2 { 2 } else { 0 });
    }

    // Premier positionnel (hors valeurs de drapeaux) = racine du workspace.
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
    let allow_raw = required(&args, "--allow");
    let allowed: Vec<String> = allow_raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if allowed.is_empty() {
        eprintln!("erreur : --allow doit lister au moins un fichier éditable.");
        exit(2);
    }

    let steps: usize = flag_value(&args, "--steps").and_then(|v| v.parse().ok()).unwrap_or(10);
    let seed: u64 = flag_value(&args, "--seed").and_then(|v| v.parse().ok()).unwrap_or(42);
    let timeout_secs: u64 = flag_value(&args, "--timeout").and_then(|v| v.parse().ok()).unwrap_or(300);
    let promote = args.iter().any(|a| a == "--promote");

    // --- Backend LLM : CONNEXION AUTOMATIQUE. -------------------------------- //
    // Résolution, du plus explicite au plus automatique :
    //   1. `--backend` / `--model` (CLI) ;
    //   2. env `RSI_LLM` (« ollama », « ollama:MODELE », « claude[:MODELE] »,
    //      « auto ») et `RSI_LLM_MODEL` (préférence de modèle) ;
    //   3. `auto` (défaut) : sonde Ollama local, découvre les modèles installés
    //      (`/api/tags`) et choisit le meilleur modèle de code ; sinon Claude si
    //      `ANTHROPIC_API_KEY` est présent (feature `llm-claude-ureq`) ; sinon
    //      erreur explicite. Zéro flag requis quand Ollama tourne.
    let env_llm = std::env::var("RSI_LLM").ok();
    let (env_backend, env_model) = match env_llm.as_deref() {
        Some(s) => match s.split_once(':') {
            Some((b, m)) => (Some(b.to_string()), Some(m.to_string())),
            None => (Some(s.to_string()), None),
        },
        None => (None, None),
    };
    let backend = flag_value(&args, "--backend")
        .or(env_backend)
        .unwrap_or_else(|| "auto".to_string());
    let model_pref = flag_value(&args, "--model")
        .or(env_model)
        .or_else(|| std::env::var("RSI_LLM_MODEL").ok());
    let host = flag_value(&args, "--ollama-host").unwrap_or_else(|| "127.0.0.1".to_string());
    let port: u16 = flag_value(&args, "--ollama-port").and_then(|v| v.parse().ok()).unwrap_or(11434);

    // Leviers d'exploration du proposeur (diversité des patchs proposés) — utile
    // quand un modèle décline/répète ; `None` ⇒ défaut du modèle.
    let temperature = flag_value(&args, "--temperature").and_then(|v| v.parse::<f64>().ok());
    let top_p = flag_value(&args, "--top-p").and_then(|v| v.parse::<f64>().ok());
    // Plafond de génération du PROPOSEUR : le défaut serveur (4096) tronque ses
    // patchs sur de gros fichiers → propositions coupées / no-ops. Le relever =
    // plus de propositions valides ET plus de trajectoires flywheel par run.
    let prop_np: Option<u32> =
        flag_value(&args, "--proposer-num-predict").and_then(|v| v.parse().ok());
    let make_ollama = |name: String| -> Box<dyn CodeModel> {
        let mut client = rsi::llm::OllamaClient::new(name)
            .with_endpoint(host.clone(), port)
            .with_timeout(Duration::from_secs(timeout_secs));
        if let Some(t) = temperature {
            client = client.with_temperature(t);
        }
        if let Some(p) = top_p {
            client = client.with_top_p(p);
        }
        if let Some(np) = prop_np {
            client = client.with_num_predict(np).with_num_ctx(np + 8192);
        }
        Box::new(LlmCodeModel::new(client))
    };
    // Découvre les modèles installés et applique la préférence éventuelle.
    let discover = |pref: Option<&str>| -> Result<String, String> {
        let installed =
            rsi::llm::ollama_installed_models(&host, port, Duration::from_secs(10))
                .map_err(|e| format!("Ollama injoignable sur {host}:{port} ({e:?})"))?;
        match rsi::llm::pick_model(&installed, pref) {
            Some(m) => {
                println!(
                    "• backend ollama : modèle « {m} » ({} installé(s){})",
                    installed.len(),
                    pref.map(|p| format!(", préférence « {p} »")).unwrap_or_default()
                );
                Ok(m)
            }
            None => Err(format!("aucun modèle utilisable sur {host}:{port} (ollama pull …)")),
        }
    };

    let (backend, model): (String, Box<dyn CodeModel>) = match backend.as_str() {
        "ollama" => match model_pref.clone() {
            // Modèle donné explicitement : utilisé tel quel (pas de sonde).
            Some(name) if flag_value(&args, "--model").is_some() => ("ollama".into(), make_ollama(name)),
            pref => match discover(pref.as_deref()) {
                Ok(m) => ("ollama".into(), make_ollama(m)),
                Err(e) => {
                    eprintln!("erreur : {e}");
                    exit(2);
                }
            },
        },
        "claude" => ("claude".into(), make_claude(&args)),
        "auto" => match discover(model_pref.as_deref()) {
            Ok(m) => ("ollama (auto)".into(), make_ollama(m)),
            Err(ollama_err) => {
                if std::env::var("ANTHROPIC_API_KEY").is_ok() && cfg!(feature = "llm-claude-ureq") {
                    println!("• backend auto : Ollama indisponible → Claude (ANTHROPIC_API_KEY)");
                    ("claude (auto)".into(), make_claude(&args))
                } else {
                    eprintln!(
                        "erreur : connexion automatique impossible.\n  - {ollama_err}\n  - pas \
                         d'ANTHROPIC_API_KEY (ou feature llm-claude-ureq absente).\n  Lancez \
                         Ollama, ou exportez RSI_LLM / ANTHROPIC_API_KEY, ou passez --backend."
                    );
                    exit(2);
                }
            }
        },
        other => {
            eprintln!("erreur : backend inconnu '{other}' (auto|ollama|claude).");
            exit(2);
        }
    };

    let proposer = LlmProposer::new(model, allowed.clone());

    // --- Évaluateur empirique réel (cargo build + test, borné). ------------- //
    let evaluator = CargoEvaluator {
        package_subdir: flag_value(&args, "--package-subdir").map(Into::into).unwrap_or_default(),
        test_args: flag_value(&args, "--test-args")
            .map(|s| s.split_whitespace().map(|t| t.to_string()).collect())
            .unwrap_or_default(),
        score_from_passrate: true,
        // Option B : `--bench "run --release --example bench_dot"` ⇒ score = perf
        // mesurée (RSI_BENCH_SCORE) au lieu du pass-rate ⇒ « optimise X » a un gradient.
        bench_command: flag_value(&args, "--bench")
            .map(|s| s.split_whitespace().map(|t| t.to_string()).collect())
            .unwrap_or_default(),
        timeout: Duration::from_secs(timeout_secs),
        ..Default::default()
    };

    // --- Référence : évaluer l'arbre vivant une fois (snapshot, intact). ---- //
    eprintln!("• évaluation de la référence (arbre vivant, build+test)…");
    let baseline = match WorkspaceSnapshot::create(&ws).and_then(|s| evaluator.evaluate(s.root())) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("erreur : impossible d'évaluer la référence : {e}");
            exit(1);
        }
    };
    println!(
        "  référence : compiles={} passed={} failed={} score={:.4}",
        baseline.compiles, baseline.tests_passed, baseline.tests_failed, baseline.score
    );

    // --- Boucle. ------------------------------------------------------------ //
    let min_gain: f64 = flag_value(&args, "--min-gain").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let revise: u32 = flag_value(&args, "--revise").and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut config = DgmConfig::new(&ws, &goal);
    config.min_score_gain = min_gain;
    config.simulated_revisions = revise;
    let mut engine = DgmEngine::new(Archive::with_root(baseline.clone()), proposer, evaluator, config, seed);

    // --- Pré-crible optionnel par world model (--prescreen-model TAG). ------- //
    // Évite le cargo build des patchs prédits cassés avec certitude ; ne peut
    // jamais écarter une amélioration (cf. docs/AGENTWORLD_STUDY.md).
    let mut prescreen_note = String::new();
    if let Some(sim_model) = flag_value(&args, "--prescreen-model") {
        let np: u32 =
            flag_value(&args, "--prescreen-num-predict").and_then(|v| v.parse().ok()).unwrap_or(12288);
        let client = rsi::llm::OllamaClient::new(sim_model.clone())
            .with_endpoint(host.clone(), port)
            .with_timeout(Duration::from_secs(timeout_secs))
            .with_num_predict(np)
            .with_num_ctx(np + 8192);
        engine = engine.with_predictor(Box::new(WorldModelPredictor { client }));
        prescreen_note = format!(", pré-crible={sim_model}");
        let mode = if revise > 0 {
            format!(" — révision simulée ×{revise} (le proposeur se corrige avant le gate)")
        } else {
            " — saute le build sur « cassé » sûr".to_string()
        };
        println!("• pré-crible world model : {sim_model} (num_predict={np}){mode}");
    } else if revise > 0 {
        // La révision simulée s'appuie sur le world model du pré-crible ; sans
        // --prescreen-model, la boucle de révision ne se déclenche jamais.
        eprintln!(
            "⚠️  --revise {revise} sans effet : la révision simulée requiert \
             --prescreen-model (cf. docs/AGENTWORLD_STUDY.md)."
        );
    }

    println!("• boucle DGM : {steps} étapes, backend={backend}{prescreen_note}, fichiers={allowed:?}");
    println!("  (chaque étape = proposition LLM + build+test+bench du snapshot : ~1-3 min)\n");
    // Étape par étape (et non `engine.run(steps)`) pour AFFICHER chaque
    // résultat au fil de l'eau — huit étapes muettes ressemblent à un blocage.
    // Une erreur du proposeur (timeout, complétion tronquée…) est transitoire :
    // l'étape est sautée au lieu d'avorter le run. Au-delà de
    // MAX_PROPOSER_FAILURES échecs consécutifs, le backend est réellement
    // mort : arrêt propre, en CONSERVANT tout ce qui a été collecté
    // (trajectoires flywheel, archive, meilleur variant).
    const MAX_PROPOSER_FAILURES: u32 = 3;
    let mut proposer_failures = 0u32;
    let mut loop_error: Option<String> = None;
    for i in 0..steps {
        let o = match engine.step() {
            Ok(o) => {
                proposer_failures = 0;
                o
            }
            Err(e @ DgmError::Proposer(_)) => {
                proposer_failures += 1;
                eprintln!(
                    "  step {i:2} · proposeur en erreur \
                     ({proposer_failures}/{MAX_PROPOSER_FAILURES} consécutives) : {e}"
                );
                if proposer_failures >= MAX_PROPOSER_FAILURES {
                    loop_error = Some(e.to_string());
                    break;
                }
                continue;
            }
            Err(e) => {
                loop_error = Some(e.to_string());
                break;
            }
        };
        match &o {
            StepOutcome::NoProposal { reason } => match reason {
                Some(r) => println!("  step {i:2} · pas de proposition ({r})"),
                None => println!("  step {i:2} · pas de proposition"),
            },
            StepOutcome::Evaluated { accepted, fitness, variant_id, .. } => {
                println!(
                    "  step {i:2} · {} · compiles={} passed={} failed={} score={:.4} · {}",
                    if *accepted { "ACCEPTÉ " } else { "rejeté  " },
                    fitness.compiles,
                    fitness.tests_passed,
                    fitness.tests_failed,
                    fitness.score,
                    &variant_id[..8.min(variant_id.len())],
                );
                // Raison du rejet : les notes portent le détail (queue de la
                // sortie cargo). On montre jusqu'à 8 lignes informatives —
                // « build failed: » seul ne permet pas de diagnostiquer à
                // distance ce que le modèle a cassé.
                if !*accepted {
                    let mut shown = 0;
                    for line in fitness.notes.lines() {
                        let line = line.trim_end();
                        if line.trim().is_empty() {
                            continue;
                        }
                        let capped: String = line.chars().take(160).collect();
                        println!("           ↳ {capped}");
                        shown += 1;
                        if shown >= 8 {
                            break;
                        }
                    }
                }
            }
        }
    }

    if let Some(e) = &loop_error {
        eprintln!("\n  ⚠ boucle interrompue : {e} — bilan et exports sur ce qui a été collecté.");
    }
    if engine.prescreen_skips() > 0 {
        println!(
            "\n  pré-crible : {} build(s) évité(s) (patchs prédits cassés par le world model)",
            engine.prescreen_skips()
        );
    }
    if engine.revisions() > 0 {
        println!("  révision simulée : {} correction(s) demandée(s) au proposeur", engine.revisions());
    }
    if let Some(path) = flag_value(&args, "--export-trajectories") {
        let traj = engine.trajectories();
        match std::fs::write(&path, rsi::trajectory::to_jsonl(traj)) {
            Ok(()) => println!(
                "  flywheel : {} trajectoire(s) à vérité terrain exportée(s) -> {path} (JSONL)",
                traj.len()
            ),
            Err(e) => eprintln!("  flywheel : échec d'écriture de {path} : {e}"),
        }
    }

    // --- Verdict / promotion. ---------------------------------------------- //
    let best = engine.best().cloned();
    let root_id = engine.archive().variants().first().map(|v| v.id.clone());
    println!("\n  archive : {} variantes gardées", engine.archive().len());

    let promotable = best.as_ref().filter(|b| {
        Some(&b.id) != root_id.as_ref()
            && b.fitness.as_ref().map(|f| f.all_green() && f.is_better_than(&baseline)).unwrap_or(false)
    });

    match promotable {
        None => {
            println!("  aucune amélioration promouvable trouvée (rien à appliquer).");
        }
        Some(b) => {
            let f = b.fitness.as_ref().unwrap();
            println!(
                "  meilleur variant promouvable : {} → {} (passed={} failed={} score={:.4})",
                b.patch.target,
                short(&b.id),
                f.tests_passed,
                f.tests_failed,
                f.score
            );
            if promote {
                let backups = flag_value(&args, "--backups")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| ws.join(".rsi_backups"));
                match rsi::dgm::promote_to_live(&ws, &b.patch, &backups) {
                    Ok(id) => println!(
                        "  ✓ PROMU vers l'arbre vivant (sauvegarde {id} dans {}).",
                        backups.display()
                    ),
                    Err(e) => {
                        eprintln!("  ✗ échec de la promotion : {e}");
                        exit(1);
                    }
                }
            } else {
                println!("  (DRY-RUN : arbre vivant intact. Relancer avec --promote pour appliquer.)");
                println!("  note : seul ce patch unique serait appliqué (variant = delta depuis la référence).");
            }
        }
    }

    // Code de sortie non-zéro si la boucle a été interrompue — les scripts
    // d'accumulation voient l'échec, mais les exports ci-dessus ont eu lieu.
    if loop_error.is_some() {
        exit(1);
    }
}

#[cfg(feature = "llm-claude-ureq")]
fn make_claude(args: &[String]) -> Box<dyn CodeModel> {
    let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    if key.is_empty() {
        eprintln!("erreur : backend claude requiert ANTHROPIC_API_KEY dans l'environnement.");
        exit(2);
    }
    let name = flag_value(args, "--model").unwrap_or_else(|| "claude-sonnet-4-6".to_string());
    let mut client = rsi::llm::ClaudeClient::with_ureq(key, name);
    // Réglages optionnels du backend Claude (symétrie avec les flags Ollama).
    if let Some(mt) = flag_value(args, "--claude-max-tokens").and_then(|v| v.parse::<u32>().ok()) {
        client = client.with_max_tokens(mt);
    }
    if let Some(url) = flag_value(args, "--claude-base-url") {
        client = client.with_base_url(url);
    }
    Box::new(LlmCodeModel::new(client))
}

#[cfg(not(feature = "llm-claude-ureq"))]
fn make_claude(_args: &[String]) -> Box<dyn CodeModel> {
    eprintln!("erreur : backend claude indisponible — recompiler avec --features llm-claude-ureq.");
    exit(2);
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

/// Premier argument positionnel (ni un drapeau, ni la valeur d'un drapeau).
fn first_positional(args: &[String]) -> Option<String> {
    let mut i = 2; // 0 = bin, 1 = (déjà géré : peut être le workspace ou un flag)
    // On considère aussi args[1] s'il n'est pas un flag.
    if !args[1].starts_with("--") {
        return Some(args[1].clone());
    }
    while i < args.len() {
        let a = &args[i];
        if a.starts_with("--") {
            // saute la valeur si ce drapeau en consomme une
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

fn short(id: &str) -> &str {
    &id[..8.min(id.len())]
}

fn usage() {
    eprintln!(
        "rsi-dgm — auto-amélioration empirique (Darwin–Gödel/STOP) sur un dépôt réel\n\n\
         USAGE:\n  \
           rsi-dgm <workspace> --goal \"...\" --allow src/a.rs,src/b.rs [options]\n\n\
         SÛR PAR DÉFAUT : DRY-RUN (l'arbre vivant n'est jamais écrit sans --promote).\n\n\
         Options principales :\n  \
           --goal TEXT           objectif (requis)\n  \
           --allow LIST          fichiers éditables, séparés par ',' (requis)\n  \
           --steps N             étapes (défaut 10)   --seed N (défaut 42)\n  \
           --backend auto|ollama|claude (défaut auto) --model NAME (sinon découvert)\n  \
           --package-subdir DIR  sous-crate à builder  --test-args \"ARGS\"\n  \
           --timeout SECS        borne par cargo (défaut 300)\n  \
           --temperature F       température du proposeur (exploration ; défaut modèle)\n  \
           --top-p F             nucleus sampling du proposeur ∈ [0,1]\n  \
           --bench \"ARGS\"        score = perf mesurée (RSI_BENCH_SCORE) au lieu\n                          du pass-rate — ex. \"run --release --example bench_dot\"\n  \
           --min-gain FRAC       gain relatif de score minimal (anti-bruit),\n                          ex. 0.02 = ≥ 2 %% (défaut 0 ; gains structurels exemptés)\n  \
           --prescreen-model TAG world model (Qwen-AgentWorld) qui saute le build\n                          des patchs prédits cassés (jamais une amélioration)\n  \
           --promote             applique le meilleur variant tout-au-vert\n  \
           --backups DIR         sauvegardes (défaut <ws>/.rsi_backups)\n\n\
         CONNEXION AUTOMATIQUE (défaut) : sonde Ollama local, découvre les modèles\n\
         installés (/api/tags) et choisit le meilleur modèle de code ; sinon Claude\n\
         si ANTHROPIC_API_KEY est présent (feature llm-claude-ureq). Overrides :\n\
         env RSI_LLM (« ollama:MODELE », « claude », « auto ») et RSI_LLM_MODEL."
    );
}
