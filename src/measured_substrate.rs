//! Substrat à efficience **mesurée nativement** (cœur, std-only) — §2/§4.
//!
//! `SubstrateImprover` qui calibre l'efficience logicielle de `P_eff` en
//! chronométrant un *vrai* kernel CPU (multiplication matricielle), **sans
//! dépendance** (ni Forge, ni GPU). Comparé à `forge_substrate::ForgeSubstrate`
//! (qui *fait évoluer* le tuilage par recherche évolutionnaire), celui-ci
//! balaie une petite grille déterministe de tuilages et retient le meilleur
//! speedup mesuré — portable partout, suffisant pour ancrer P_eff sur une
//! mesure réelle.
//!
//! L'efficience part de l'analytique (ancre) et capture une fraction
//! `1 − 1/speedup` du headroom ; **monotone** (ratchet).

use std::time::Instant;

use crate::substrate::{Substrate, SubstrateImprover};

const TILES: [usize; 4] = [8, 16, 32, 64];

/// Kernel de référence C = A·B (ordre i,j,k hostile au cache — la cible à
/// battre). `pub` pour servir de sujet aux benchs DGM (`examples/bench_matmul`).
pub fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    // ordre i,j,k (k interne) : hostile au cache → référence à battre
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f32;
            for k in 0..n {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

fn matmul_tiled(a: &[f32], b: &[f32], c: &mut [f32], n: usize, bm: usize, bn: usize, bk: usize) {
    for ci in c.iter_mut() {
        *ci = 0.0;
    }
    let mut i0 = 0;
    while i0 < n {
        let mut k0 = 0;
        while k0 < n {
            let mut j0 = 0;
            while j0 < n {
                for i in i0..(i0 + bm).min(n) {
                    for k in k0..(k0 + bk).min(n) {
                        let aik = a[i * n + k];
                        for j in j0..(j0 + bn).min(n) {
                            c[i * n + j] += aik * b[k * n + j];
                        }
                    }
                }
                j0 += bn;
            }
            k0 += bk;
        }
        i0 += bm;
    }
}

/// matrice déterministe (PRNG linéaire simple, sans dépendance).
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

/// Améliorateur de substrat à mesure native (kernel CPU chronométré).
pub struct MeasuredSubstrate {
    n: usize,
    reps: usize,
    best_speedup: f64,
    anchor: Option<f64>,
}

impl MeasuredSubstrate {
    /// `n` = taille du kernel N×N (défaut conseillé ≥ 96 pour que le tuilage paie).
    pub fn new(n: usize) -> Self {
        MeasuredSubstrate {
            n: n.max(16),
            reps: 3,
            best_speedup: 1.0,
            anchor: None,
        }
    }

    fn median_time(&self, run: impl Fn()) -> f64 {
        let mut ts: Vec<f64> = (0..self.reps)
            .map(|_| {
                let t0 = Instant::now();
                run();
                t0.elapsed().as_secs_f64()
            })
            .collect();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2]
    }

    fn efficiency(&self, anchor: f64) -> f64 {
        let s = self.best_speedup.max(1.0);
        anchor + (1.0 - anchor) * (1.0 - 1.0 / s)
    }
}

impl SubstrateImprover for MeasuredSubstrate {
    fn improve(&mut self, substrate: &Substrate) -> Substrate {
        let n = self.n;
        let a = matrix(n, 0xA1);
        let b = matrix(n, 0xB2);
        let mut c = vec![0.0f32; n * n];

        // baseline naïve
        let base_t = self.median_time(|| {
            let mut cc = vec![0.0f32; n * n];
            matmul_naive(&a, &b, &mut cc, n);
        });
        // référence de correction
        let mut reference = vec![0.0f32; n * n];
        matmul_naive(&a, &b, &mut reference, n);

        // balaye une petite grille de tuilages, garde le meilleur speedup correct
        for &bm in &TILES {
            for &bk in &TILES {
                matmul_tiled(&a, &b, &mut c, n, bm, bm, bk);
                let correct = c
                    .iter()
                    .zip(&reference)
                    .all(|(x, y)| (x - y).abs() <= 1e-2 * (1.0 + y.abs()));
                if !correct {
                    continue;
                }
                let t = self.median_time(|| {
                    let mut cc = vec![0.0f32; n * n];
                    matmul_tiled(&a, &b, &mut cc, n, bm, bm, bk);
                });
                if t > 0.0 {
                    let speedup = base_t / t;
                    if speedup > self.best_speedup {
                        self.best_speedup = speedup;
                    }
                }
            }
        }

        let anchor = *self
            .anchor
            .get_or_insert_with(|| substrate.software_efficiency());
        let mut out = substrate.clone();
        let measured = self.efficiency(anchor).max(out.software_efficiency());
        out.set_measured_software_eff(Some(measured));
        out
    }
}

// ===================================================================== //
// Variante : efficience **SIMD / vectorisation** mesurée
// ===================================================================== //

/// Réduction à **chaîne de dépendance** (anti-vectorisable) : chaque itération
/// dépend de la précédente ⇒ le CPU ne peut pas paralléliser → référence lente.
#[inline(never)]
fn reduce_chained(v: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for &x in v {
        // multiply-add sérielle : dépendance portée par `acc`
        acc = acc.mul_add(0.999_999_9, x);
    }
    acc
}

/// Réduction à **accumulateurs indépendants** : 8 voies sans dépendance croisée
/// ⇒ le compilateur l'auto-vectorise (SIMD) sur un CPU capable → rapide.
#[inline(never)]
fn reduce_independent(v: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let (chunks, rem) = v.as_chunks::<8>();
    for c in chunks {
        for (a, &x) in acc.iter_mut().zip(c) {
            *a = a.mul_add(0.999_999_9, x);
        }
    }
    let mut tail = 0.0f64;
    for &x in rem {
        tail = tail.mul_add(0.999_999_9, x);
    }
    acc.iter().sum::<f64>() + tail
}

/// Améliorateur de substrat mesurant l'efficience **SIMD/vectorisation** réelle
/// de l'hôte (machine + build), au lieu du tuilage cache de [`MeasuredSubstrate`].
///
/// Principe (même *ratchet* que [`MeasuredSubstrate`]) : on chronométre une
/// réduction sérielle (anti-vectorisable) vs une réduction à accumulateurs
/// indépendants (vectorisable). Le **speedup** mesuré reflète l'efficience SIMD
/// effective ; l'efficience logicielle part de l'analytique (ancre) et capture
/// une fraction `1 − 1/speedup` du *headroom*. **Monotone**.
///
/// > Note : `scirust-rsi` n'expose qu'un moteur de raffinement (pas de métrique
/// > matérielle) ; la mesure honnête de l'efficience SIMD est donc faite
/// > **en-process** ici. Comme toute mesure réelle, elle est *non déterministe*
/// > (canal `measured_software_eff`, opt-in) — cf. `docs/SAFETY.md`.
pub struct SimdMeasuredSubstrate {
    len: usize,
    reps: usize,
    best_speedup: f64,
    anchor: Option<f64>,
}

impl SimdMeasuredSubstrate {
    /// `len` = longueur du vecteur réduit (défaut conseillé ≥ 2¹⁶).
    pub fn new(len: usize) -> Self {
        SimdMeasuredSubstrate {
            len: len.max(4096),
            reps: 5,
            best_speedup: 1.0,
            anchor: None,
        }
    }

    /// Speedup SIMD mesuré (≥ 1.0) lors du dernier `improve`.
    pub fn best_speedup(&self) -> f64 {
        self.best_speedup
    }

    fn median_time(&self, run: impl Fn() -> f64) -> f64 {
        let mut ts: Vec<f64> = (0..self.reps)
            .map(|_| {
                let t0 = Instant::now();
                let out = run();
                // empêche l'élimination de code mort (le résultat doit "compter")
                std::hint::black_box(out);
                t0.elapsed().as_secs_f64()
            })
            .collect();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2]
    }

    fn efficiency(&self, anchor: f64) -> f64 {
        let s = self.best_speedup.max(1.0);
        anchor + (1.0 - anchor) * (1.0 - 1.0 / s)
    }
}

impl SubstrateImprover for SimdMeasuredSubstrate {
    fn improve(&mut self, substrate: &Substrate) -> Substrate {
        // données déterministes (suite de van der Corput-like, dans [-0.5, 0.5))
        let data: Vec<f64> = (0..self.len)
            .map(|i| (((i as f64) * 0.618_033_988_75) % 1.0) - 0.5)
            .collect();
        let d = std::hint::black_box(&data[..]);

        let t_serial = self.median_time(|| reduce_chained(d));
        let t_vector = self.median_time(|| reduce_independent(d));
        if t_vector > 0.0 {
            let speedup = t_serial / t_vector;
            if speedup > self.best_speedup {
                self.best_speedup = speedup;
            }
        }

        let anchor = *self
            .anchor
            .get_or_insert_with(|| substrate.software_efficiency());
        let mut out = substrate.clone();
        let measured = self.efficiency(anchor).max(out.software_efficiency());
        out.set_measured_software_eff(Some(measured));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn improves_or_keeps_p_eff_natively() {
        let mut rng = Rng::new(7);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let p0 = sub.effective_power();
        let mut imp = MeasuredSubstrate::new(96);
        let out = imp.improve(&sub);
        assert!(out.effective_power() >= p0 - 1e-12);
        assert!(out.measured_software_eff.is_some());
        assert!(imp.best_speedup >= 1.0);
    }

    #[test]
    fn simd_improver_is_monotone_and_measures() {
        let mut rng = Rng::new(13);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let p0 = sub.effective_power();
        let mut imp = SimdMeasuredSubstrate::new(1 << 15);
        let out = imp.improve(&sub);
        // ratchet : P_eff ne régresse jamais
        assert!(out.effective_power() >= p0 - 1e-12);
        assert!(out.measured_software_eff.is_some());
        // le speedup est borné par le bas par 1.0 (mesure réelle, ≥ 1)
        assert!(imp.best_speedup() >= 1.0);
    }

    #[test]
    fn simd_reductions_are_finite() {
        let v: Vec<f64> = (0..10_000).map(|i| (i as f64) * 1e-3 - 5.0).collect();
        assert!(reduce_chained(&v).is_finite());
        assert!(reduce_independent(&v).is_finite());
    }
}
