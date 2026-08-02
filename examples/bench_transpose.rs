//! Benchmark de `rsi::kernels::transpose` pour la boucle DGM.
//!
//! Imprime `RSI_BENCH_SCORE=<débit>` (transpositions/seconde, plus grand =
//! mieux). Deuxième cible DGM (après `kernels::matmul`, ×7.3 auto-découvert) :
//! vérifie que la découverte **se généralise** à un kernel memory-bound où le
//! headroom vient de la localité des écritures (tuilage : ×1.4–1.5 sondé).
//!
//! n=2048 : 16 Mo par matrice — déborde largement L2, le naïf charge une ligne
//! de cache par élément écrit.
//!
//! Lancement direct : `cargo run --release --example bench_transpose`
//! Via DGM :
//!   rsi-dgm . --goal "accelere kernels::transpose par tuilage cache, memes resultats" \
//!       --allow src/kernels.rs \
//!       --bench "run --release --example bench_transpose" --min-gain 0.05

use rsi::kernels::transpose;
use std::time::Instant;

/// Matrice déterministe (PRNG linéaire simple).
fn matrix(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n * n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn main() {
    let n = 2048usize;
    let src = matrix(n, 0xC3);
    let mut dst = vec![0.0f32; n * n];

    // Échauffement.
    transpose(&src, &mut dst, n);
    std::hint::black_box(&dst);

    // Médiane de `reps` mesures ; chaque mesure = `iters` transpositions.
    let reps = 7;
    let iters = 4u64;
    let mut times: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for _ in 0..iters {
            transpose(&src, &mut dst, n);
            std::hint::black_box(&dst);
        }
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = times[times.len() / 2].max(1e-12);

    let score = iters as f64 / median; // transpositions/seconde
    println!(
        "kernels::transpose: n={n}, mediane {:.2} ms / {iters} appels",
        median * 1e3
    );
    println!("RSI_BENCH_SCORE={score}");
}
