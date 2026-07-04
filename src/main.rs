//! Démonstration de la boucle RSI : simule un agent auto-améliorant, affiche
//! la trajectoire de son intelligence globale `SI_global` et, en option,
//! exporte la trajectoire en CSV / JSON.
//!
//! Usage :
//! ```text
//! cargo run --release --bin rsi-demo -- [n_steps] [seed] [optimizer] \
//!     [--csv FICHIER] [--json FICHIER]
//!
//!   n_steps    : nombre de pas (défaut 120)
//!   seed       : graine reproductible (défaut 2026)
//!   optimizer  : 'random' (défaut) ou 'cma' (sep-CMA-ES)
//! ```

use rsi::{report, RSIAgent};

struct Args {
    n_steps: usize,
    seed: u64,
    optimizer: String,
    csv: Option<String>,
    json: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        n_steps: 120,
        seed: 2026,
        optimizer: "random".into(),
        csv: None,
        json: None,
    };
    let mut positional = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--csv" => a.csv = it.next(),
            "--json" => a.json = it.next(),
            _ => positional.push(arg),
        }
    }
    if let Some(v) = positional.first().and_then(|s| s.parse().ok()) {
        a.n_steps = v;
    }
    if let Some(v) = positional.get(1).and_then(|s| s.parse().ok()) {
        a.seed = v;
    }
    if let Some(v) = positional.get(2) {
        a.optimizer = v.clone();
    }
    a
}

fn main() {
    let args = parse_args();

    let mut agent = match args.optimizer.as_str() {
        "cma" | "cma-es" | "sep-cma-es" => RSIAgent::demo_cma(args.seed),
        "random" | "demo" | "" => RSIAgent::demo(args.seed),
        other => {
            eprintln!(
                "⚠️  optimiseur inconnu « {other} » — repli sur « random » \
                 (connus : random, cma)."
            );
            RSIAgent::demo(args.seed)
        }
    };

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   RSI — Auto-amélioration récursive (formulation géométrique v9)       ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!(
        "seed = {}  |  pas = {}  |  optimiseur = {}  |  tâches |Ω| = {}",
        args.seed,
        args.n_steps,
        args.optimizer,
        agent.surface.tasks.len()
    );
    println!();
    println!(
        "{:>4} │ {:>8} │ {:>8} │ {:>7} │ {:>7} │ {:>6} │ capacités (D M R A C V)",
        "t", "SI_glob", "ΔSI", "P_eff", "‖ΔS‖", "%subst"
    );
    println!("{}", "─".repeat(96));

    let start_si = agent.si_global();
    let reports = agent.run(args.n_steps);

    let stride = (args.n_steps / 24).max(1);
    for (i, r) in reports.iter().enumerate() {
        if i % stride != 0 && i + 1 != reports.len() {
            continue;
        }
        let caps = r
            .capabilities
            .iter()
            .map(|c| format!("{c:.2}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "{:>4} │ {:>8.4} │ {:>+8.4} │ {:>7.4} │ {:>7.4} │ {:>5.0}% │ {}",
            r.t,
            r.si_global,
            r.delta_si,
            r.p_eff,
            r.appr.delta_norm,
            r.frac_limited_by_substrate * 100.0,
            caps,
        );
    }

    let end = reports.last().unwrap();
    println!("{}", "─".repeat(96));
    println!(
        "SI_global : {start_si:.4} → {:.4}   (gain absolu {:+.4}, +{:.1} %)",
        end.si_global,
        end.si_global - start_si,
        if start_si > 0.0 {
            (end.si_global - start_si) / start_si * 100.0
        } else {
            0.0
        }
    );
    println!(
        "P_eff final : {:.4}   |   goulot substrat : {:.0} %   |   ‖S‖ : {:.3}",
        end.p_eff,
        end.frac_limited_by_substrate * 100.0,
        end.state_norm,
    );

    if let Some(path) = &args.csv {
        match report::write_csv(&reports, path) {
            Ok(()) => println!("✓ trajectoire CSV écrite : {path}"),
            Err(e) => eprintln!("✗ échec écriture CSV {path} : {e}"),
        }
    }
    if let Some(path) = &args.json {
        match report::write_json(&reports, path) {
            Ok(()) => println!("✓ trajectoire JSON écrite : {path}"),
            Err(e) => eprintln!("✗ échec écriture JSON {path} : {e}"),
        }
    }
}
