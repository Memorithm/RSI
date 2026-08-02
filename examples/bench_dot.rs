//! Benchmark de `rsi::linalg::dot` pour la boucle DGM (Option B).
//!
//! Imprime `RSI_BENCH_SCORE=<débit>` (appels/seconde, **plus grand = plus
//! rapide**). C'est le score de fitness que consomme [`rsi::dgm::CargoEvaluator`]
//! quand `bench_command` est configuré : « optimise `dot` » a alors un vrai
//! gradient, et un patch qui accélère réellement `dot` sur la machine est
//! **accepté** (compile ▸ tests verts ▸ perf mesurée supérieure).
//!
//! Lancement direct : `cargo run --release --example bench_dot`
//! Via DGM (sur Jetson) :
//!   `rsi-dgm . --goal "accélère dot" --allow src/linalg.rs \
//!       --bench "run --release --example bench_dot" --steps 6`
//!
//! Médiane de plusieurs répétitions pour réduire le bruit de mesure (la perf
//! reste une grandeur non déterministe — cf. `docs/SAFETY.md`).

use rsi::linalg::dot;
use std::time::Instant;

fn main() {
    let n = 1 << 16;
    let a: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-3 - 30.0).collect();
    let b: Vec<f64> = (0..n).map(|i| (i as f64) * 2e-3 - 65.0).collect();

    // Échauffement (JIT du cache / fréquence CPU).
    let mut warm = 0.0f64;
    for _ in 0..100 {
        warm += dot(&a, &b);
    }
    std::hint::black_box(warm);

    // Médiane de `reps` mesures ; chaque mesure = `iters` appels à `dot`.
    let reps = 15;
    let iters = 4000u64;
    let mut times: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let mut s = 0.0f64;
        for _ in 0..iters {
            s += dot(&a, &b);
        }
        std::hint::black_box(s);
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = times[times.len() / 2].max(1e-12);

    // Débit = appels `dot` par seconde (plus grand = mieux).
    let score = iters as f64 / median;
    println!(
        "dot: n={n}, médiane {:.3} ms / {iters} appels",
        median * 1e3
    );
    println!("RSI_BENCH_SCORE={score}");
}
