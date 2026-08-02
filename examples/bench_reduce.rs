//! Benchmark de `rsi::kernels::sum` pour la boucle DGM.
//!
//! Imprime `RSI_BENCH_SCORE=<débit>` (réductions/seconde, plus grand = mieux).
//! Troisième cible DGM, d'un genre encore différent : la somme sérielle est
//! bornée par la **latence de la chaîne de dépendance** (`acc += x` — chaque
//! addition attend la précédente), ni par le cache ni par le calcul. Des
//! accumulateurs indépendants (que le compilateur vectorise) capturent un
//! headroom sondé à ×4.2.
//!
//! Lancement direct : `cargo run --release --example bench_reduce`
//! Via DGM :
//!   rsi-dgm . --goal "accelere kernels::sum par accumulateurs independants, meme somme" \
//!       --allow src/kernels.rs \
//!       --bench "run --release --example bench_reduce" --min-gain 0.05

use rsi::kernels::sum;
use std::time::Instant;

fn main() {
    // n=2^20 f64 = 8 Mo : déborde le L2 ; la chaîne de dépendance reste
    // pourtant le goulot (l'addition sérielle est bien plus lente que la
    // bande passante mémoire).
    let n = 1usize << 20;
    let v: Vec<f64> = (0..n)
        .map(|i| (((i * 2654435761) % 2001) as f64 - 1000.0) * 1e-3)
        .collect();

    // Échauffement.
    std::hint::black_box(sum(std::hint::black_box(&v)));

    // Médiane de `reps` mesures ; chaque mesure = `iters` réductions.
    let reps = 9;
    let iters = 16u64;
    let mut times: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(sum(std::hint::black_box(&v)));
        }
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = times[times.len() / 2].max(1e-12);

    let score = iters as f64 / median; // réductions/seconde
    println!(
        "kernels::sum: n=2^20, mediane {:.2} ms / {iters} appels",
        median * 1e3
    );
    println!("RSI_BENCH_SCORE={score}");
}
