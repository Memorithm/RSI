//! Démo **complète** : agent RSI « tout intégré » (Forge ℳ + Forge P_eff +
//! OctaSoma C + CCOS audit + criticité §7), avec deux modes :
//!
//! - run simple : trajectoire, audit, rappel mémoire, export CSV/JSON + SVG ;
//! - `compare`  : agent « nu » (cœur, méta aléatoire) vs « tout intégré »,
//!   côte à côte, pour visualiser l'apport des intégrations.
//!
//! ```text
//! cargo run --release --bin rsi-full --features "forge octasoma ccos" -- [pas] [graine]
//! cargo run --release --bin rsi-full --features "forge octasoma ccos" -- compare [pas] [graine]
//! ```

use rsi::{
    report, CcosAudit, CognitiveState, Dims, ForgeMetaSearch, ForgeSubstrate, IntelligenceSurface,
    MetaOptimizer, RSIAgent, RiskConfig, Rng, StabilityConfig, StepReport, Substrate,
};

fn short(mode: &str) -> &str {
    match mode {
        "regression_competence" => "régression",
        "instabilite_divergence" => "instabilité",
        "derive_valeurs" => "dérive-V",
        "effondrement_substrat" => "substrat",
        "goodhart_surajustement" => "goodhart",
        "empoisonnement_memoire" => "mémoire",
        "wireheading" => "wirehead",
        other => other,
    }
}

/// Sous-systèmes partagés (mêmes état/substrat/surface pour une comparaison équitable).
/// La surface est **ancrée sur un corpus de tâches réel** (Ω = archétypes,
/// compétence par loi de Liebig), au lieu d'un échantillonnage synthétique.
fn build(seed: u64) -> (CognitiveState, Substrate, IntelligenceSurface) {
    let mut rng = Rng::new(seed);
    let state = CognitiveState::random(Dims::uniform(6), &mut rng, 0.08);
    let substrate = Substrate::default_with(4, 4, &mut rng);
    let surface = IntelligenceSurface::from_corpus(&rsi::TaskCorpus::builtin());
    (state, substrate, surface)
}

/// Petit corpus de documents pour alimenter la composante D (connaissances).
fn knowledge_docs() -> Vec<String> {
    (0..12)
        .map(|i| {
            format!(
                "domaine_{i} concept apprentissage raisonnement substrat memoire \
                 valeurs criticite surface intelligence recursive technique_{i} methode_{i}"
            )
        })
        .collect()
}

fn integrated(
    seed: u64,
    s: &CognitiveState,
    sub: &Substrate,
    surf: &IntelligenceSurface,
) -> RSIAgent {
    let dim = s.size();
    let mut agent = RSIAgent::new(
        s.clone(),
        sub.clone(),
        surf.clone(),
        StabilityConfig::default(),
        Box::new(ForgeMetaSearch::new(4, 12, 0.15, seed ^ 0xF0)),
    )
    .with_substrate_improver(Box::new(ForgeSubstrate::new(96, 2, 6, seed ^ 0x5B)))
    .with_memory(Box::new(rsi::OctaSomaMemory::new(dim, seed)))
    .with_audit(Box::new(CcosAudit::new(format!("rsi-{seed}"))))
    .with_knowledge(Box::new(
        rsi::CorpusKnowledge::from_texts(knowledge_docs()).with_scale(12.0),
    ))
    // seuil de criticité plus strict pour que les réponses actives s'engagent
    .with_risk_config(RiskConfig {
        rpn_max: 0.3,
        ..RiskConfig::default()
    });
    // ε adaptatif au bruit Monte-Carlo (corpus = petit Ω, donc estimateur bruité)
    agent.dynamics_cfg.adaptive_epsilon = true;
    agent
}

fn main() {
    // audit a22 : `--help` lançait un run complet (écritures incluses)
    if std::env::args().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "rsi-full — run intégré (agent + mémoire + substrat + knowledge)\n\n\
             USAGE:\n  rsi-full [n_steps] [seed] [optimizer]\n\n\
             Exports (écrasent) : rsi_full_trajectory.csv/.json/.svg dans le cwd.\n"
        );
        std::process::exit(0);
    }
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let compare = argv.iter().any(|a| a == "compare");
    let nums: Vec<usize> = argv.iter().filter_map(|a| a.parse().ok()).collect();
    let steps = nums.first().copied().unwrap_or(45);
    let seed = nums.get(1).copied().unwrap_or(2026) as u64;

    let (state, substrate, surface) = build(seed);

    if compare {
        run_compare(steps, seed, &state, &substrate, &surface);
    } else {
        run_single(steps, seed, &state, &substrate, &surface);
    }
}

fn run_single(
    steps: usize,
    seed: u64,
    s: &CognitiveState,
    sub: &Substrate,
    surf: &IntelligenceSurface,
) {
    let mut agent = integrated(seed, s, sub, surf);

    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║   RSI — AGENT TOUT INTÉGRÉ   (Forge ℳ + Forge P_eff + OctaSoma C + CCOS audit)║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝");
    println!(
        "seed={seed}  pas={steps}  |Ω|={}  dim(S)={}",
        agent.surface.tasks.len(),
        s.size()
    );
    println!();
    println!(
        "{:>3} │ {:>7} │ {:>7} │ {:>6} │ {:>6} │ {:>6} │ {:>10} │ {:>9} │ {:>4}",
        "t", "SI", "SI_safe", "P_eff", "risk", "maxRPN", "critique", "réponse", "%sub"
    );
    println!("{}", "─".repeat(82));

    let start_si = agent.si_global();
    let reports = agent.run(steps);
    let stride = (steps / 18).max(1);
    for (i, r) in reports.iter().enumerate() {
        // affiche : pas échantillonnés + dernier + TOUT pas où une réponse de
        // sûreté s'active (pour rendre le garde-fou actif visible)
        if i % stride != 0 && i + 1 != reports.len() && r.mitigation == "none" {
            continue;
        }
        println!(
            "{:>3} │ {:>7.4} │ {:>7.4} │ {:>6.4} │ {:>6.4} │ {:>6.4} │ {:>10} │ {:>9} │ {:>3.0}%",
            r.t,
            r.si_global,
            r.si_safe,
            r.p_eff,
            r.risk_global,
            r.max_rpn,
            short(r.most_critical),
            r.mitigation,
            r.frac_limited_by_substrate * 100.0,
        );
    }
    let Some(end) = reports.last() else {
        println!("Aucun pas exécuté (steps = 0). SI_global initial : {start_si:.4}");
        return;
    };
    println!("{}", "─".repeat(82));
    println!(
        "SI_global : {start_si:.4} → {:.4}  ({:+.1} %)   |   SI_safe final : {:.4}   |   P_eff : {:.4}",
        end.si_global, pct(end.si_global - start_si, start_si), end.si_safe, end.p_eff,
    );

    println!(
        "\n▸ Audit CCOS : {} événements — intégrité {}",
        agent.audit_len(),
        if agent.audit_verify() {
            "✓ valide"
        } else {
            "✗ ALTÉRÉE"
        }
    );
    if let Some(h) = agent.audit_head() {
        println!("  hash de tête : {}…", &h[..h.len().min(32)]);
    }

    println!("\n▸ Mémoire OctaSoma : rappel du contexte le plus proche de l'état final");
    if let Some(p) = agent.recall_similar(1).first() {
        if let Some((si, strat)) = rsi::meta::decode_strategy_payload(p) {
            println!(
                "  SI={si:.4}, gain ℳ={:.3}, focus={:?}",
                strat.gain,
                strat.focus.map(|x| (x * 100.0).round() / 100.0)
            );
        }
    }

    export(&reports);
}

fn run_compare(
    steps: usize,
    seed: u64,
    s: &CognitiveState,
    sub: &Substrate,
    surf: &IntelligenceSurface,
) {
    // agent « nu » : cœur seul, méta aléatoire, aucune intégration
    let mut naked = RSIAgent::new(
        s.clone(),
        sub.clone(),
        surf.clone(),
        StabilityConfig::default(),
        Box::new(MetaOptimizer::new(48, 0.12, seed ^ 1)),
    );
    let mut full = integrated(seed, s, sub, surf);

    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║   RSI — COMPARATIF   « nu » (cœur)   vs   « tout intégré »                    ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝");
    println!("seed={seed}  pas={steps}  (mêmes état/substrat/surface au départ)\n");

    let n0 = naked.si_global();
    let f0 = full.si_global();
    let rn = naked.run(steps);
    let rf = full.run(steps);

    println!("{:>3} │ {:^21} │ {:^29}", "", "NU (cœur)", "TOUT INTÉGRÉ");
    println!(
        "{:>3} │ {:>6} {:>6} {:>6} │ {:>6} {:>6} {:>6} {:>9}",
        "t", "SI", "P_eff", "risk", "SI", "P_eff", "risk", "réponse"
    );
    println!("{}", "─".repeat(78));
    let stride = (steps / 15).max(1);
    for i in (0..steps).step_by(stride) {
        let a = &rn[i];
        let b = &rf[i];
        println!(
            "{:>3} │ {:>6.4} {:>6.4} {:>6.4} │ {:>6.4} {:>6.4} {:>6.4} {:>9}",
            i + 1,
            a.si_global,
            a.p_eff,
            a.risk_global,
            b.si_global,
            b.p_eff,
            b.risk_global,
            b.mitigation,
        );
    }
    println!("{}", "─".repeat(78));
    let (Some(en), Some(ef)) = (rn.last(), rf.last()) else {
        println!("Aucun pas exécuté (steps = 0) — comparatif impossible.");
        return;
    };
    println!(
        "Final NU         : SI {:.4} (+{:.0}%)  P_eff {:.4}  risk {:.4}  SI_safe {:.4}",
        en.si_global,
        pct(en.si_global - n0, n0),
        en.p_eff,
        en.risk_global,
        en.si_safe
    );
    println!(
        "Final INTÉGRÉ     : SI {:.4} (+{:.0}%)  P_eff {:.4}  risk {:.4}  SI_safe {:.4}",
        ef.si_global,
        pct(ef.si_global - f0, f0),
        ef.p_eff,
        ef.risk_global,
        ef.si_safe
    );
    println!("\nApport des intégrations : P_eff {:.4} → {:.4} (substrat réel mesuré), audit CCOS {} événements,",
        en.p_eff, ef.p_eff, full.audit_len());
    println!("réponses de sûreté actives (réalignement V / plancher anti-wireheading) absentes du run nu.");

    export(&rf);
}

/// Pourcentage de variation sûr : 0.0 si la base est nulle (évite NaN/∞).
fn pct(delta: f64, base: f64) -> f64 {
    if base.abs() < f64::EPSILON {
        0.0
    } else {
        delta / base * 100.0
    }
}

fn export(reports: &[StepReport]) {
    println!("\n▸ Export de la trajectoire");
    for (label, res) in [
        (
            "CSV ",
            report::write_csv(reports, "rsi_full_trajectory.csv"),
        ),
        (
            "JSON",
            report::write_json(reports, "rsi_full_trajectory.json"),
        ),
    ] {
        match res {
            Ok(()) => println!(
                "  ✓ {label} : rsi_full_trajectory.{}",
                label.trim().to_lowercase()
            ),
            Err(e) => println!("  ✗ {label} : {e}"),
        }
    }
    match rsi::plot::write_svg(reports, "rsi_full_trajectory.svg") {
        Ok(()) => println!("  ✓ SVG  : rsi_full_trajectory.svg (ouvrable dans un navigateur)"),
        Err(e) => println!("  ✗ SVG  : {e}"),
    }
}
