//! Benchmark de `matmul_naive` pour la boucle DGM.
//!
//! Imprime `RSI_BENCH_SCORE=<debit>` (multiplications de matrices par seconde,
//! plus grand = mieux). Contrairement a `dot` (borne par la bande passante
//! memoire a grande taille -> aucun headroom), un matmul N^3 naif en ordre
//! i,j,k est borne par le CACHE : un tuilage/blocking apporte un vrai gain
//! (cf. `MeasuredSubstrate`, speedups x2-4 mesures). C'est donc un sujet ou
//! la boucle DGM peut trouver une acceleration REELLE au-dessus du bruit.
//!
//! Lancement direct : `cargo run --release --example bench_matmul`
//! Via DGM :
//!   rsi-dgm . --goal "accelere matmul_naive par tuilage cache, memes resultats" \
//!       --allow src/measured_substrate.rs \
//!       --bench "run --release --example bench_matmul" --min-gain 0.05

use rsi::measured_substrate::matmul_naive;
use std::time::Instant;

/// Matrice deterministe (PRNG lineaire simple).
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
    // n=160 : ~100 Ko par matrice, N^3 = 4M MACs — assez gros pour que le
    // tuilage paie, assez petit pour un bench rapide (~quelques ms/appel).
    let n = 160usize;
    let a = matrix(n, 0xA1);
    let b = matrix(n, 0xB2);
    let mut c = vec![0.0f32; n * n];

    // Echauffement.
    matmul_naive(&a, &b, &mut c, n);
    std::hint::black_box(&c);

    // Mediane de `reps` mesures ; chaque mesure = `iters` matmuls.
    let reps = 9;
    let iters = 8u64;
    let mut times: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for _ in 0..iters {
            matmul_naive(&a, &b, &mut c, n);
            std::hint::black_box(&c);
        }
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = times[times.len() / 2].max(1e-12);

    let score = iters as f64 / median; // matmuls/seconde, plus grand = mieux
    println!(
        "matmul_naive: n={n}, mediane {:.2} ms / {iters} appels",
        median * 1e3
    );
    println!("RSI_BENCH_SCORE={score}");
}
