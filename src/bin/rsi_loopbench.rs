//! `rsi-loopbench` — ⚙️ **L9 : banc d'essai de boucle** (cœur pur).
//!
//! Mesure l'effet des **cadences multi-échelles** (L3) sur la convergence et le
//! coût (nombre de méta-révisions), via le pilote `run_until` (L1/L2), puis
//! l'apport d'un **portefeuille** (L8 swarm).
//!
//! ```text
//! cargo run --release --bin rsi-loopbench -- [graines]
//! ```

use rsi::{
    run_swarm, CognitiveState, Dims, IntelligenceSurface, LoopConfig, LoopSchedule, MetaOptimizer,
    RSIAgent, Rng, StabilityConfig, StopReason, Substrate, TaskCorpus,
};

fn build(seed: u64, meta_every: usize, corpus: &TaskCorpus) -> RSIAgent {
    let mut rng = Rng::new(seed);
    let state = CognitiveState::random(Dims::uniform(6), &mut rng, 0.08);
    let substrate = Substrate::default_with(4, 4, &mut rng);
    let surface = IntelligenceSurface::from_corpus(corpus);
    let meta = Box::new(MetaOptimizer::new(48, 0.12, seed ^ 0xA));
    RSIAgent::new(state, substrate, surface, StabilityConfig::default(), meta)
        .with_schedule(LoopSchedule::new(meta_every, 1))
}

fn main() {
    let n_seeds: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let corpus = TaskCorpus::extended();

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║   RSI — BANC D'ESSAI DE BOUCLE (L9)   cadences × convergence × coût         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!(
        "corpus Ω={} | {n_seeds} graines | run_until(plateau)\n",
        corpus.len()
    );
    println!(
        "{:>10} │ {:>11} │ {:>9} │ {:>7} │ {:>7} │ {:>10}",
        "meta_every", "pas→arrêt", "raison", "SI_fin", "AUC", "méta-éval"
    );
    println!("{}", "─".repeat(74));

    let lcfg = LoopConfig {
        max_steps: 400,
        plateau_window: 15,
        plateau_eps: 1e-4,
        ..LoopConfig::default()
    };

    for meta_every in [1usize, 2, 4, 8] {
        let (mut steps, mut si, mut auc, mut meta_evals, mut plateaus) = (0.0, 0.0, 0.0, 0.0, 0);
        for s in 0..n_seeds {
            let mut agent = build(1000 + s, meta_every, &corpus);
            let out = agent.run_until(&lcfg);
            let last = out.reports.last().unwrap();
            steps += out.steps as f64;
            si += last.si_global;
            auc += out.reports.iter().map(|r| r.si_global).sum::<f64>() / out.reports.len() as f64;
            // nombre de méta-révisions réellement exécutées (cadence)
            meta_evals += (out.steps as f64 / meta_every as f64).ceil();
            if out.reason == StopReason::Plateau {
                plateaus += 1;
            }
        }
        let k = n_seeds as f64;
        let reason = if plateaus * 2 >= n_seeds as usize {
            "plateau"
        } else {
            "budget"
        };
        println!(
            "{:>10} │ {:>11.1} │ {:>9} │ {:>7.4} │ {:>7.4} │ {:>10.1}",
            meta_every,
            steps / k,
            reason,
            si / k,
            auc / k,
            meta_evals / k
        );
    }
    println!("{}", "─".repeat(74));
    println!("Lecture : une cadence méta plus lente réduit fortement les méta-évaluations");
    println!("(coût) pour un SI final proche → efficacité de calcul (méta-méta, L3).\n");

    // L8 — apport du portefeuille (swarm) vs agent unique moyen
    let steps_swarm = 60usize;
    let res = run_swarm(8, 2000, steps_swarm, |seed| build(seed, 1, &corpus));
    let mean: f64 = res.members.iter().map(|m| m.si_global).sum::<f64>() / res.members.len() as f64;
    let best = res.best();
    println!(
        "▸ Swarm (L8) : 8 boucles // {steps_swarm} pas — SI moyen {:.4}, MEILLEUR {:.4} (graine {}) → +{:.1} %",
        mean,
        best.si_global,
        best.seed,
        (best.si_global - mean) / mean * 100.0
    );
}
