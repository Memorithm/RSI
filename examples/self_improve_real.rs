//! Démo de l'agent d'auto-amélioration sur le **moteur réel `scirust-rsi`**.
//!
//! Cet exemple ne compile/s'exécute qu'avec la feature `scirust` activée (donc
//! dans un environnement où `Memorithm/scirust` est joignable) :
//!
//! ```text
//! cargo run --release --features scirust --example self_improve_real
//! ```
//!
//! Sans la feature, il imprime simplement la marche à suivre (voir
//! `SCIRUST_ACTIVATION.md`). Ainsi le dépôt reste compilable hors-ligne.

#[cfg(feature = "scirust")]
fn main() {
    use rsi::scirust_bridge::{run_self_improve, SymbolicSynthesis};

    let (best, report) = run_self_improve(0);
    let task = SymbolicSynthesis::from_target(|x| x * x + 1.0, -2.0, 2.0, 21);

    println!("RSI — agent d'auto-amélioration (moteur réel scirust-rsi)");
    println!("meilleur candidat : {}", best.pretty());
    println!("fraction de tests réussis : {:.0}%", task.pass_fraction(&best) * 100.0);
    println!("fitness finale  : {:.4}", report.best_fitness);
    println!("gain total      : {:+.4}", report.total_gain());
    println!("itérations      : {}  (révisions gardées : {})", report.iterations, report.accepted);
    println!("non-régression (is_monotone) : {}", report.is_monotone());
}

#[cfg(not(feature = "scirust"))]
fn main() {
    eprintln!(
        "Cet exemple requiert le moteur réel `scirust-rsi`.\n\
         Active-le dans un environnement où Memorithm/scirust est autorisé :\n\
         \n\
           cargo run --release --features scirust --example self_improve_real\n\
         \n\
         Voir SCIRUST_ACTIVATION.md pour les 3 étapes (dépendance + feature + module).\n\
         En attendant, la version hors-ligne (stand-in) tourne via :\n\
         \n\
           cargo run --release --example self_improve"
    );
}
