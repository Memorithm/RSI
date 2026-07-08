//! Exemple : un agent qui **génère** des expressions candidates puis les
//! **améliore** en boucle élitiste bornée, sans jamais régresser.
//!
//! Lancement : `cargo run --release --example self_improve`
//!
//! NOTE : pilote local (`rsi::ascent`), miroir du contrat `scirust-rsi`, en
//! attendant l'accès au dépôt `Memorithm/scirust`. Le candidat est évalué
//! dans le sandbox d'AST de RSI (aucune exécution de code arbitraire).

use rsi::ascent::{ascend, Guard, RefineTask};
use rsi::synthesis::SymbolicSynthesis;

fn main() {
    // Cible à reconstruire : f(x) = x^2 + 1 sur [-2, 2] (21 cas de test).
    let mut task = SymbolicSynthesis::from_target(|x| x * x + 1.0, -2.0, 2.0, 21, 0)
        .with_lambda(24);

    let init = task.seed_candidate();
    let init_fit = task.score(&init);

    // Garde-fou explicite : borné, patient, avec cible.
    let guard = Guard::new()
        .max_iters(50)
        .patience(12)
        .target(0.99)
        .min_delta(0.0);

    let (best, report) = ascend(&mut task, init, &guard);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RSI — agent d'auto-amélioration (génère → évalue → améliore)  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("cible f(x) = x^2 + 1   |   fitness = %tests réussis − 0.01·taille\n");

    println!("fitness initiale : {:.4}", init_fit);
    println!("trajectoire de l'incumbent (1 valeur / itération) :");
    for (i, f) in report.history.iter().enumerate() {
        // n'affiche que les paliers (changements) + le dernier
        let changed = i == 0 || (report.history[i] - report.history[i - 1]).abs() > 1e-9;
        if changed || i + 1 == report.history.len() {
            let bar = "█".repeat(((f.max(0.0)) * 30.0) as usize);
            println!("  it {:>2} : {:>7.4} {}", i, f, bar);
        }
    }

    println!("\nRapport :");
    println!("  itérations      : {} (≤ max_iters = 50)", report.iters);
    println!("  révisions gardées: {}", report.accepted);
    println!("  arrêt           : {:?}", report.stop);
    println!("  monotone (non-régression) : {}", report.is_monotone());
    println!("  fitness finale  : {:.4}", report.best());
    println!("\nmeilleur candidat : {}", best.pretty());
    println!("  fraction de tests réussis : {:.0}%", task.pass_fraction(&best) * 100.0);

    println!("\nContrat de sûreté : boucle bornée (≤ {} it.), élitiste (aucune", 50);
    println!("régression adoptée ⇒ is_monotone), déterministe (graine), candidat");
    println!("évalué dans le sandbox d'AST de RSI — jamais exécuté comme du code.");
}
