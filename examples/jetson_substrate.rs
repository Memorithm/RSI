//! Exemple : **RSI sur matériel réel** (pensé pour Jetson Thor/Orin, mais
//! tourne partout — dégradation propre du GPU).
//!
//! Lancement : `cargo run --release --example jetson_substrate`
//!
//! Chaîne complète sur du réel :
//!   1. [`rsi::hw_probe`] mesure CPU/mémoire/GPU de l'hôte → vecteur matériel `H` ;
//!   2. [`rsi::measured_substrate::SimdMeasuredSubstrate`] ancre l'efficience
//!      logicielle sur un vrai speedup SIMD ;
//!   3. on évalue `SI_global` du banc [`rsi::omega_tasks`] et le goulot par tâche.
//!
//! Sur Jetson, `nvidia-smi` ou le sysfs Tegra fournit la charge GPU réelle ;
//! ailleurs, le GPU est marqué « absent » et `H[2]` prend une valeur neutre.

use rsi::hw_probe::{measured_hardware_substrate, HardwareSnapshot};
use rsi::measured_substrate::SimdMeasuredSubstrate;
use rsi::omega_tasks::{report, standard_suite, Limiter};
use rsi::rng::Rng;
use rsi::state::{CognitiveState, Dims};
use rsi::substrate::SubstrateImprover;

fn main() {
    let mut rng = Rng::new(2026);

    // 1. Mesure matérielle réelle.
    let snap = HardwareSnapshot::probe();
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  RSI · substrat ancré sur le matériel réel (Jetson-ready)        ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!(
        "  CPU   : {} cœurs, charge {:.0}%",
        snap.cpu_count,
        100.0 * snap.cpu_load_frac
    );
    println!("  Mémoire : {:.0}% utilisée", 100.0 * snap.mem_used_frac);
    match snap.gpu_load_frac {
        Some(g) => {
            let vram = snap
                .gpu_mem_used_frac
                .map(|m| format!(", VRAM {:.0}% utilisée", 100.0 * m))
                .unwrap_or_else(|| ", VRAM n/d (mémoire unifiée)".to_string());
            println!(
                "  GPU   : {:.0}% calcul{} ({})",
                100.0 * g,
                vram,
                snap.gpu_source
            );
        }
        None => println!(
            "  GPU   : absent/non lisible ({}) → H[2] neutre",
            snap.gpu_source
        ),
    }
    println!(
        "  vecteur matériel H (capacité dispo) = {:?}",
        snap.hardware_vector()
    );

    // 2. Substrat ancré matériel, puis efficience logicielle mesurée (SIMD).
    let mut sub = measured_hardware_substrate(&snap, 6, &mut rng);
    let p_hw = sub.effective_power();
    let mut simd = SimdMeasuredSubstrate::new(1 << 16);
    sub = simd.improve(&sub);
    println!();
    println!(
        "  P_eff : matériel={:.4}  →  +SIMD mesuré={:.4}  (speedup ×{:.2})",
        p_hw,
        sub.effective_power(),
        simd.best_speedup(),
    );

    // 3. SI_global sur le banc de tâches réelles + goulot par tâche.
    let suite = standard_suite();
    let state = CognitiveState::random(Dims::uniform(6), &mut rng, 0.55);
    println!();
    println!(
        "  {:<20} {:>8} {:>8} {:>8}   goulot",
        "tâche", "Φ_x", "g_x", "C_réel"
    );
    println!("  {}", "─".repeat(62));
    for tr in report(&suite, &state, &sub) {
        let tag = match tr.limiter {
            Limiter::Cognition => "cognitif (Φ<g)",
            Limiter::Substrate => "substrat (g<Φ)",
        };
        println!(
            "  {:<20} {:>8.4} {:>8.4} {:>8.4}   {}",
            tr.name, tr.phi, tr.g, tr.c_real, tag
        );
    }

    let surf = rsi::omega_tasks::surface(&suite);
    let (si, stderr) = surf.si_global_stats(&state, &sub);
    let b = surf.bottleneck(&state, &sub);
    println!();
    println!("  SI_global = {si:.4} (± {stderr:.4})");
    println!(
        "  goulot agrégé : {:.0}% substrat / {:.0}% cognitif",
        100.0 * b.frac_limited_by_substrate,
        100.0 * b.frac_limited_by_cognition,
    );
}
