//! `rsi-ascend` — **ascension auto-améliorante pilotée par LLM**, livrée en
//! binaire (câble [`rsi::llm::ascend_llm`] + [`rsi::llm::LlmGuard`] sur un
//! backend Ollama local, sans dépendance externe).
//!
//! À chaque itération : le modèle lit l'incumbent, propose `k` candidats ; le
//! moteur applique `safety_check` puis `score` et n'adopte qu'un candidat
//! **strictement meilleur** (élitisme ⇒ aucune régression). Les garde-fous
//! (bornes, budget d'appels/temps, anti-overfitting) sont tenus par le moteur ;
//! le LLM n'y a jamais accès.
//!
//! Domaines std-only : `synthesis` (régression symbolique), `prompt`
//! (optimisation d'invite), `tuning` (hyperparamètres).
//!
//! Requiert un serveur Ollama joignable (`ollama serve`) et un modèle chargé.

use std::str::FromStr;
use std::time::Duration;

use rsi::llm::{ascend_llm, LlmGuard, OllamaClient};
use rsi::prompt::PromptOpt;
use rsi::synthesis::SymbolicSynthesis;
use rsi::tuning::ConfigTuning;

const VALUE_FLAGS: &[&str] = &[
    "--model", "--ollama-host", "--ollama-port", "--iters", "--patience", "--k", "--target",
    "--min-delta", "--max-calls", "--max-seconds", "--max-overfit", "--seed", "--timeout",
    "--num-predict", "--temperature", "--top-p", "--fn", "--lo", "--hi", "--n",
];

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn parse_or<T: FromStr>(args: &[String], flag: &str, default: T) -> T {
    flag_value(args, flag).and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Fonction cible pour le domaine `synthesis` (même vocabulaire que l'API MCP).
fn target_fn(name: &str) -> Result<fn(f64) -> f64, String> {
    match name {
        "quadratic" => Ok(|x| x * x + 1.0),
        "linear" => Ok(|x| 2.0 * x - 1.0),
        "cubic" => Ok(|x| x * x * x - x),
        other => Err(format!("cible inconnue : '{other}' (quadratic|linear|cubic)")),
    }
}

fn usage() {
    eprintln!(
        "rsi-ascend — ascension auto-améliorante pilotée par LLM (Ollama local)\n\
         \n\
         USAGE : rsi-ascend <domaine> --model <NOM> [options]\n\
         domaines : synthesis | prompt | tuning\n\
         \n\
         Connexion Ollama :\n\
         \x20 --model NOM          modèle Ollama (requis ; ex. `ollama list`)\n\
         \x20 --ollama-host HÔTE   défaut 127.0.0.1\n\
         \x20 --ollama-port PORT   défaut 11434\n\
         \x20 --timeout SECS       timeout HTTP par appel (défaut 180)\n\
         \x20 --num-predict N      plafond de tokens générés (optionnel)\n\
         \n\
         Exploration (diversité des propositions) :\n\
         \x20 --temperature F      température d'échantillonnage (défaut modèle ;\n\
         \x20                      ↑ = plus de diversité, évite la stagnation)\n\
         \x20 --top-p F            nucleus sampling ∈ [0,1] (défaut modèle)\n\
         \n\
         Garde-fous (LlmGuard) :\n\
         \x20 --iters N            bornes d'itérations (défaut 20)\n\
         \x20 --k N                candidats demandés par itération (défaut 4)\n\
         \x20 --patience N         arrêt après N itérations sans gain (défaut 6)\n\
         \x20 --target F           arrêt si la fitness atteint F (optionnel)\n\
         \x20 --min-delta F        gain strict minimal pour adopter (défaut 1e-9)\n\
         \x20 --max-calls N        budget d'appels LLM (défaut 60)\n\
         \x20 --max-seconds F      budget temps mur (optionnel)\n\
         \x20 --max-overfit F      écart train−heldout toléré (optionnel)\n\
         \x20 --seed N             graine déterministe (défaut 2026)\n\
         \n\
         Domaine `synthesis` :\n\
         \x20 --fn NOM             cible : quadratic|linear|cubic (défaut quadratic)\n\
         \x20 --lo F --hi F        intervalle d'échantillonnage (défaut -3..3)\n\
         \x20 --n N                nombre de points (défaut 30)\n"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || has_flag(&args, "--help") || has_flag(&args, "-h") {
        usage();
        return;
    }

    // domaine = premier positionnel (hors valeurs de drapeaux).
    let mut domain: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a.starts_with("--") {
            i += if VALUE_FLAGS.contains(&a.as_str()) { 2 } else { 1 };
        } else {
            domain = Some(a.clone());
            break;
        }
    }
    let domain = match domain {
        Some(d) => d,
        None => {
            eprintln!("erreur : domaine manquant (synthesis|prompt|tuning).\n");
            usage();
            std::process::exit(2);
        }
    };

    let model = match flag_value(&args, "--model") {
        Some(m) => m,
        None => {
            eprintln!("erreur : --model requis (liste des modèles : `ollama list`).");
            std::process::exit(2);
        }
    };

    // --- Garde-fous (budget + bornes + anti-overfitting). ------------------ //
    let guard = LlmGuard {
        max_iters: parse_or(&args, "--iters", 20),
        patience: parse_or(&args, "--patience", 6),
        target: flag_value(&args, "--target").and_then(|v| v.parse().ok()),
        min_delta: parse_or(&args, "--min-delta", 1e-9),
        k: parse_or::<usize>(&args, "--k", 4).max(1),
        max_llm_calls: parse_or(&args, "--max-calls", 60),
        max_wall_clock: flag_value(&args, "--max-seconds")
            .and_then(|v| v.parse::<f64>().ok())
            .map(Duration::from_secs_f64),
        max_overfit_gap: flag_value(&args, "--max-overfit").and_then(|v| v.parse().ok()),
    };

    // --- Client Ollama (transport HTTP sur std::net, sans dépendance). ----- //
    let host = flag_value(&args, "--ollama-host").unwrap_or_else(|| "127.0.0.1".to_string());
    let port: u16 = parse_or(&args, "--ollama-port", 11434);
    let timeout: u64 = parse_or(&args, "--timeout", 180);
    let mut client = OllamaClient::new(model.clone())
        .with_endpoint(host.clone(), port)
        .with_timeout(Duration::from_secs(timeout));
    if let Some(np) = flag_value(&args, "--num-predict").and_then(|v| v.parse::<u32>().ok()) {
        client = client.with_num_predict(np).with_num_ctx(np + 8192);
    }
    // Leviers d'exploration (diversité des propositions LLM).
    let temperature = flag_value(&args, "--temperature").and_then(|v| v.parse::<f64>().ok());
    if let Some(t) = temperature {
        client = client.with_temperature(t);
    }
    if let Some(p) = flag_value(&args, "--top-p").and_then(|v| v.parse::<f64>().ok()) {
        client = client.with_top_p(p);
    }
    let seed: u64 = parse_or(&args, "--seed", 2026);

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   rsi-ascend — ascension auto-améliorante pilotée par LLM (Ollama)     ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("domaine = {domain}  |  modèle = {model}  |  endpoint = {host}:{port}");
    println!(
        "exploration : température = {}",
        temperature.map(|t| format!("{t}")).unwrap_or_else(|| "défaut modèle".to_string())
    );
    println!(
        "garde-fous : iters≤{}  k={}  patience={}  budget={} appels  cible={}  min_delta={}",
        guard.max_iters,
        guard.k,
        guard.patience,
        guard.max_llm_calls,
        guard.target.map(|t| format!("{t}")).unwrap_or_else(|| "—".to_string()),
        guard.min_delta,
    );
    println!("{}", "─".repeat(72));

    // --- Ascension sur le domaine choisi. ---------------------------------- //
    let (best_str, report) = match domain.as_str() {
        "synthesis" => {
            let fname = flag_value(&args, "--fn").unwrap_or_else(|| "quadratic".to_string());
            let f = match target_fn(&fname) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("erreur : {e}");
                    std::process::exit(2);
                }
            };
            let lo: f64 = parse_or(&args, "--lo", -3.0);
            let hi: f64 = parse_or(&args, "--hi", 3.0);
            let n: usize = parse_or::<usize>(&args, "--n", 30).max(4);
            if !lo.is_finite() || !hi.is_finite() || hi <= lo {
                eprintln!("erreur : intervalle invalide (exiger lo < hi finis).");
                std::process::exit(2);
            }
            println!("cible synthesis : {fname} sur [{lo}, {hi}] ({n} points, held-out réservé)\n");
            let mut task = SymbolicSynthesis::from_target_split(f, lo, hi, n, seed);
            let init = task.seed_candidate();
            let (best, report) = ascend_llm(&mut task, init, &client, &guard);
            (best.pretty(), report)
        }
        "prompt" => {
            println!("cible prompt : maximiser la qualité d'invite (indices structurants)\n");
            let mut task = PromptOpt::new();
            let init = task.seed_candidate();
            let (best, report) = ascend_llm(&mut task, init, &client, &guard);
            (best, report)
        }
        "tuning" => {
            println!("cible tuning : optimiser une configuration d'hyperparamètres\n");
            let mut task = ConfigTuning::new();
            let init = task.seed_candidate();
            let (best, report) = ascend_llm(&mut task, init, &client, &guard);
            (format!("{best:?}"), report)
        }
        other => {
            eprintln!("erreur : domaine inconnu '{other}' (synthesis|prompt|tuning).");
            std::process::exit(2);
        }
    };

    // --- Compte rendu. ----------------------------------------------------- //
    println!("{}", "─".repeat(72));
    println!("arrêt          : {:?}", report.stop);
    println!("itérations     : {}", report.iters);
    println!("appels LLM     : {} / {}", report.llm_calls, guard.max_llm_calls);
    println!(
        "adoptés        : {}  (rejetés : {} pires, {} non-sûrs)",
        report.accepted, report.rejected_worse, report.rejected_unsafe
    );
    println!("fitness train  : {:.4}", report.best());
    println!("fitness heldout: {:.4}", report.best_heldout());
    println!("monotone       : {}", report.is_monotone());
    println!("{}", "─".repeat(72));
    println!("MEILLEUR CANDIDAT :\n  {best_str}");

    if report.accepted == 0 {
        eprintln!(
            "\nNote : aucun candidat adopté. Vérifier qu'Ollama répond \
             (`curl http://{host}:{port}/api/tags`) et que le modèle « {model} » est chargé."
        );
    }
}
