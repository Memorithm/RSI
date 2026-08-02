//! Benchmark de `rsi::sha256` pour la boucle DGM — deuxième cible **self**.
//!
//! Imprime `RSI_BENCH_SCORE=<débit>` (Mo hachés/seconde, plus grand = mieux).
//! Le gate est fort ici : `sha256::tests::known_vectors` (vecteurs officiels)
//! plus tous les usages internes (audit hash-chaîné, IDs de variantes DGM) —
//! le moindre bit faux casse la suite. Optimisations légitimes possibles :
//! déroulage de la boucle de rondes, réduction des copies du schedule.
//!
//! Lancement direct : `cargo run --release --example bench_sha256`
//! Via DGM :
//!   rsi-dgm . --goal "accelere sha256, memes empreintes" \
//!       --allow src/sha256.rs \
//!       --bench "run --release --example bench_sha256" --min-gain 0.05

use rsi::sha256::sha256;
use std::time::Instant;

fn main() {
    // 1 Mo de données déterministes.
    let data: Vec<u8> = (0..1usize << 20).map(|i| (i * 131 + 7) as u8).collect();
    let mb = data.len() as f64 / (1024.0 * 1024.0);

    // Échauffement + sanité (le digest doit être stable d'un appel à l'autre).
    let d0 = sha256(&data);
    assert_eq!(d0, sha256(&data));
    std::hint::black_box(&d0);

    // Médiane de `reps` mesures ; chaque mesure = `iters` hachages complets.
    let reps = 9;
    let iters = 8u64;
    let mut times: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for _ in 0..iters {
            let d = sha256(std::hint::black_box(&data));
            std::hint::black_box(&d);
        }
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = times[times.len() / 2].max(1e-12);

    let score = iters as f64 * mb / median; // Mo/s
    println!(
        "sha256: 1 Mo/hachage, mediane {:.2} ms / {iters} hachages",
        median * 1e3
    );
    println!("RSI_BENCH_SCORE={score}");
}
