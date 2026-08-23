//! `rsi-ablate` — **étude d'ablation** des garde-fous & intégrations (cœur pur,
//! aucune feature requise → reproductible partout).
//!
//! Pour chaque configuration (on/off d'un facteur), exécute l'agent sur le
//! **corpus de tâches élargi** (Ω ancré), sur plusieurs graines, et agrège :
//! SI final, SI_safe, risque moyen/max, vitesse de convergence (t@90 %), AUC,
//! nombre d'interventions de sûreté, et régressions notables.
//!
//! ```text
//! cargo run --release --bin rsi-ablate -- [pas] [graines]
//! ```

use rsi::{
    CognitiveState, CorpusKnowledge, Dims, IntelligenceSurface, LinearContextMemory,
    MeasuredSubstrate, MetaOptimizer, RSIAgent, RiskConfig, Rng, StabilityConfig, Substrate,
    TaskCorpus,
};

#[derive(Clone, Copy)]
struct Ablation {
    name: &'static str,
    memory: bool,
    substrate: bool,
    knowledge: bool,
    active_response: bool,
    adaptive_eps: bool,
}

const OFF: Ablation = Ablation {
    name: "",
    memory: false,
    substrate: false,
    knowledge: false,
    active_response: false,
    adaptive_eps: false,
};

fn configs() -> Vec<Ablation> {
    vec![
        Ablation {
            name: "baseline (nu)",
            ..OFF
        },
        Ablation {
            name: "+ mémoire",
            memory: true,
            ..OFF
        },
        Ablation {
            name: "+ substrat",
            substrate: true,
            ..OFF
        },
        Ablation {
            name: "+ connaissances",
            knowledge: true,
            ..OFF
        },
        Ablation {
            name: "+ réponse active",
            active_response: true,
            ..OFF
        },
        Ablation {
            name: "+ ε adaptatif",
            adaptive_eps: true,
            ..OFF
        },
        Ablation {
            name: "FULL (tout)",
            memory: true,
            substrate: true,
            knowledge: true,
            active_response: true,
            adaptive_eps: true,
        },
    ]
}

fn knowledge_docs() -> Vec<String> {
    (0..12)
        .map(|i| format!("domaine_{i} concept apprentissage raisonnement substrat memoire valeurs technique_{i}"))
        .collect()
}

fn build(a: &Ablation, seed: u64, corpus: &TaskCorpus) -> RSIAgent {
    let mut rng = Rng::new(seed);
    let state = CognitiveState::random(Dims::uniform(6), &mut rng, 0.08);
    let substrate = Substrate::default_with(4, 4, &mut rng);
    let surface = IntelligenceSurface::from_corpus(corpus);

    let cfg = StabilityConfig {
        adaptive_epsilon: a.adaptive_eps,
        ..StabilityConfig::default()
    };
    let meta = Box::new(MetaOptimizer::new(48, 0.12, seed ^ 0xA));
    let mut agent =
        RSIAgent::new(state, substrate, surface, cfg, meta).with_risk_config(RiskConfig {
            rpn_max: 0.3,
            active_response: a.active_response,
            ..RiskConfig::default()
        });
    if a.memory {
        agent = agent.with_memory(Box::new(LinearContextMemory::new()));
    }
    if a.substrate {
        agent = agent
            .with_substrate_improver(Box::new(MeasuredSubstrate::new(48)))
            .with_route_threshold(0.3);
    }
    if a.knowledge {
        agent = agent.with_knowledge(Box::new(
            CorpusKnowledge::from_texts(knowledge_docs()).with_scale(12.0),
        ));
    }
    agent
}

#[derive(Default, Clone, Copy)]
struct Metrics {
    si_end: f64,
    si_safe: f64,
    risk_mean: f64,
    risk_max: f64,
    t90: f64,
    auc: f64,
    interventions: f64,
    regressions: f64,
}

fn run_once(a: &Ablation, seed: u64, corpus: &TaskCorpus, steps: usize) -> Metrics {
    let mut agent = build(a, seed, corpus);
    let reports = agent.run(steps);
    let Some(last) = reports.last() else {
        // steps == 0 : métriques neutres (le main rejette déjà ce cas)
        return Metrics::default();
    };
    let n = reports.len().max(1) as f64;

    let si_end = last.si_global;
    let auc = reports.iter().map(|r| r.si_global).sum::<f64>() / n;
    let risk_mean = reports.iter().map(|r| r.risk_global).sum::<f64>() / n;
    let risk_max = reports.iter().map(|r| r.risk_global).fold(0.0, f64::max);
    let interventions = reports.iter().filter(|r| r.mitigation != "none").count() as f64;
    // régression notable : SI recule de plus de 0.05 sur le pas d'apprentissage
    let regressions = reports
        .iter()
        .filter(|r| r.appr.si_after < r.appr.si_before - 0.05)
        .count() as f64;
    let t90 = reports
        .iter()
        .find(|r| r.si_global >= 0.9 * si_end)
        .map(|r| r.t as f64)
        .unwrap_or(n);

    Metrics {
        si_end,
        si_safe: last.si_safe,
        risk_mean,
        risk_max,
        t90,
        auc,
        interventions,
        regressions,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(45);
    let n_seeds: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(6);
    if steps == 0 {
        eprintln!("erreur : steps=0 (aucune mesure possible) — fournir un nombre de pas ≥ 1");
        std::process::exit(2);
    }
    if n_seeds == 0 {
        eprintln!("erreur : n_seeds=0 (aucune moyenne possible) — fournir un nombre de graines ≥ 1");
        std::process::exit(2);
    }
    let corpus = TaskCorpus::extended();

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║   RSI — ÉTUDE D'ABLATION   (garde-fous & intégrations, cœur pur)                ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!(
        "corpus Ω = {} tâches (élargi)   |   {steps} pas   |   {n_seeds} graines (moyenne)\n",
        corpus.len()
    );
    println!(
        "{:<18} │ {:>6} │ {:>7} │ {:>8} │ {:>7} │ {:>5} │ {:>6} │ {:>5} │ {:>5}",
        "configuration", "SI", "SI_safe", "risk_moy", "risk_max", "t@90", "AUC", "interv", "régr"
    );
    println!("{}", "─".repeat(90));

    for a in configs() {
        let mut agg = Metrics::default();
        for seed in 0..n_seeds {
            let m = run_once(&a, 1000 + seed, &corpus, steps);
            agg.si_end += m.si_end;
            agg.si_safe += m.si_safe;
            agg.risk_mean += m.risk_mean;
            agg.risk_max += m.risk_max;
            agg.t90 += m.t90;
            agg.auc += m.auc;
            agg.interventions += m.interventions;
            agg.regressions += m.regressions;
        }
        let k = n_seeds as f64;
        println!(
            "{:<18} │ {:>6.4} │ {:>7.4} │ {:>8.4} │ {:>7.4} │ {:>5.1} │ {:>6.4} │ {:>6.1} │ {:>5.1}",
            a.name,
            agg.si_end / k,
            agg.si_safe / k,
            agg.risk_mean / k,
            agg.risk_max / k,
            agg.t90 / k,
            agg.auc / k,
            agg.interventions / k,
            agg.regressions / k,
        );
    }
    println!("{}", "─".repeat(90));
    println!("Lecture : 'interv' = pas où une réponse de sûreté s'est déclenchée ; 'régr' =");
    println!("régressions notables de SI (>0.05) — doit rester ~0 (non-régression préservée).");
    println!(
        "Comparer p. ex. '+ réponse active' vs 'baseline' : risk_max ↓ au prix d'interventions."
    );
}
