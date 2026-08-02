//! Exemple : **Ω concret + substrat mesuré réel**.
//!
//! Lancement : `cargo run --release --example omega_real`
//!
//! Montre les deux extensions additives (sans duplication du cœur v9) :
//!   1. un **banc de tâches réelles nommées** ([`rsi::omega_tasks`]) branché sur
//!      la surface d'intelligence existante (`Φ_x` / `g_x` configurables) ;
//!   2. un **substrat à efficience mesurée** ([`rsi::measured_substrate`]) qui
//!      chronométre un vrai kernel SIMD pour ancrer `P_eff` sur la machine.
//!
//! On voit alors, tâche par tâche, **qui bride `C_réel`** : la compétence
//! cognitive `Φ_x` ou le plafond physique `g_x`.

use rsi::measured_substrate::SimdMeasuredSubstrate;
use rsi::omega_tasks::{report, standard_suite, Limiter};
use rsi::rng::Rng;
use rsi::state::{CognitiveState, Dims};
use rsi::substrate::{Substrate, SubstrateImprover};

fn main() {
    let mut rng = Rng::new(2026);
    let suite = standard_suite();

    // État cognitif modéré et substrat de base.
    let state = CognitiveState::random(Dims::uniform(6), &mut rng, 0.55);
    let base_sub = Substrate::default_with(6, 6, &mut rng);

    // Substrat ANCRÉ sur une mesure SIMD réelle de l'hôte.
    let mut simd = SimdMeasuredSubstrate::new(1 << 16);
    let measured_sub = simd.improve(&base_sub);

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  RSI v9 · Ω concret + substrat mesuré (Φ_x vs g_x par tâche)      ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!(
        "  P_eff  : analytique={:.4}  →  mesuré(SIMD)={:.4}   (speedup ×{:.2})",
        base_sub.effective_power(),
        measured_sub.effective_power(),
        simd.best_speedup(),
    );
    println!();
    println!(
        "  {:<20} {:>8} {:>8} {:>8}   goulot",
        "tâche", "Φ_x", "g_x", "C_réel"
    );
    println!("  {}", "─".repeat(62));

    for tr in report(&suite, &state, &measured_sub) {
        let tag = match tr.limiter {
            Limiter::Cognition => "cognitif (Φ<g)",
            Limiter::Substrate => "substrat (g<Φ)",
        };
        println!(
            "  {:<20} {:>8.4} {:>8.4} {:>8.4}   {}",
            tr.name, tr.phi, tr.g, tr.c_real, tag
        );
    }

    // SI_global sur le banc réel.
    let surf = rsi::omega_tasks::surface(&suite);
    let (si, stderr) = surf.si_global_stats(&state, &measured_sub);
    println!();
    println!("  SI_global = {si:.4}  (± {stderr:.4} Monte-Carlo)");

    let b = surf.bottleneck(&state, &measured_sub);
    println!(
        "  goulot agrégé : {:.0}% substrat / {:.0}% cognitif   (Φ̄={:.3}, ḡ={:.3})",
        100.0 * b.frac_limited_by_substrate,
        100.0 * b.frac_limited_by_cognition,
        b.mean_phi,
        b.mean_g,
    );
}
