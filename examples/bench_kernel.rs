//! Benchmark de `rsi::kernels::matmul` pour la boucle DGM.
//!
//! Imprime `RSI_BENCH_SCORE=<débit>` (matmuls/seconde, plus grand = mieux).
//! Contrairement à `bench_matmul` (qui vise `measured_substrate::matmul_naive`,
//! le baseline de mesure qu'il ne faut PAS accélérer), la cible ici est un
//! kernel **dédié et conservable** : un patch accepté est réellement promouvable.
//!
//! Un matmul N³ naïf en ordre i,j,k est borné par le CACHE → un tuilage/blocking
//! apporte un vrai gain au-dessus du bruit.
//!
//! Lancement direct : `cargo run --release --example bench_kernel`
//! Via DGM :
//!   rsi-dgm . --goal "accelere kernels::matmul par tuilage cache, memes resultats" \
//!       --allow src/kernels.rs \
//!       --bench "run --release --example bench_kernel" --min-gain 0.05

use rsi::kernels::matmul;
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
    // n=512 : ~1 Mo par matrice, 3 Mo au total — dépasse le L2 (2 Mo/cœur sur
    // Neoverse V3AE / Jetson Thor) ⇒ le kernel naïf est réellement borné par le
    // cache et le tuilage/réordonnancement a un vrai headroom. (À n=160 tout
    // tenait en L2 : les variantes tuilées scoraient au niveau du bruit.)
    let n = 512usize;
    let a = matrix(n, 0xA1);
    let b = matrix(n, 0xB2);
    let mut c = vec![0.0f32; n * n];

    // Échauffement.
    matmul(&a, &b, &mut c, n);
    std::hint::black_box(&c);

    // Médiane de `reps` mesures ; chaque mesure = `iters` matmuls.
    let reps = 7;
    let iters = 2u64;
    let mut times: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for _ in 0..iters {
            matmul(&a, &b, &mut c, n);
            std::hint::black_box(&c);
        }
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = times[times.len() / 2].max(1e-12);

    let score = iters as f64 / median; // matmuls/seconde, plus grand = mieux
    println!(
        "kernels::matmul: n={n}, mediane {:.2} ms / {iters} appels",
        median * 1e3
    );
    println!("RSI_BENCH_SCORE={score}");
}
