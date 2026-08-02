//! sep-CMA-ES — *Separable Covariance Matrix Adaptation Evolution Strategy*.
//!
//! Optimiseur stochastique sans dérivée (std-only). Variante diagonale de
//! CMA-ES (Ros & Hansen, 2008) : la matrice de covariance est restreinte à sa
//! diagonale, ce qui supprime le besoin d'une décomposition spectrale tout en
//! conservant l'adaptation d'échelle par coordonnée — idéal pour une
//! implémentation sans dépendance et un coût O(N·λ) par génération.
//!
//! Sert de méta-optimiseur alternatif à la recherche aléatoire pour la
//! méta-révision `ℳ_{t+1} = argmax_ℳ SI_global(ℳ(S_t))` (§5).
//
// Boucles indexées intentionnelles (algèbre vectorielle), lint stylistique désactivé.
#![allow(clippy::needless_range_loop)]

use crate::rng::Rng;

/// État et hyperparamètres d'une instance sep-CMA-ES en dimension `n`.
pub struct SepCmaEs {
    pub n: usize,
    pub lambda: usize, // taille de population
    mu: usize,         // nombre de parents sélectionnés
    weights: Vec<f64>, // poids de recombinaison
    mu_eff: f64,
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c_1: f64,
    c_mu: f64,
    chi_n: f64, // E‖N(0,I)‖
    rng: Rng,
}

impl SepCmaEs {
    /// Construit une instance pour un problème de dimension `n`.
    ///
    /// `lambda = 0` ⇒ valeur par défaut `4 + ⌊3·ln N⌋`.
    pub fn new(n: usize, lambda: usize, seed: u64) -> Self {
        let nf = n as f64;
        let lambda = if lambda == 0 {
            (4.0 + (3.0 * nf.ln()).floor()) as usize
        } else {
            lambda
        }
        .max(4);
        let mu = lambda / 2;

        // poids logarithmiques décroissants, normalisés à somme 1
        let mut weights: Vec<f64> = (0..mu)
            .map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())
            .collect();
        let sum_w: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= sum_w;
        }
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        let c_sigma = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
        let d_sigma = 1.0 + 2.0 * (((mu_eff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);

        let mut c_1 = 2.0 / ((nf + 1.3).powi(2) + mu_eff);
        let mut c_mu =
            (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0).powi(2) + mu_eff)).min(1.0 - c_1);
        // facteur d'accélération propre à sep-CMA-ES
        let sep = (nf + 2.0) / 3.0;
        c_1 = (c_1 * sep).min(1.0);
        c_mu = (c_mu * sep).min(1.0 - c_1);

        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

        SepCmaEs {
            n,
            lambda,
            mu,
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
            rng: Rng::new(seed),
        }
    }

    /// **Maximise** `f` à partir de `mean0` et d'un pas initial `sigma0`.
    ///
    /// Retourne `(meilleur_x, meilleure_valeur)` rencontrés sur l'ensemble des
    /// générations (élitisme externe : on conserve le meilleur vu).
    pub fn optimize(
        &mut self,
        mean0: &[f64],
        sigma0: f64,
        generations: usize,
        f: impl Fn(&[f64]) -> f64 + Sync,
    ) -> (Vec<f64>, f64) {
        let n = self.n;
        assert_eq!(mean0.len(), n, "dimension de mean0 incompatible");

        let mut mean = mean0.to_vec();
        let mut sigma = sigma0;
        let mut cov = vec![1.0f64; n]; // variances diagonales
        let mut p_sigma = vec![0.0f64; n];
        let mut p_c = vec![0.0f64; n];

        let mut best_x = mean.clone();
        let mut best_f = f(&mean);

        for gen in 0..generations {
            // --- échantillonnage de la population (RNG SÉQUENTIEL) ------- //
            // On échantillonne d'abord toute la population (ordre RNG préservé),
            // puis on évalue les fitness EN PARALLÈLE (f pure). La mise à jour du
            // best et la construction de `pop` se font ensuite à ORDRE D'INDEX
            // fixe : résultat bit-exact identique au séquentiel ⇒ déterminisme.
            let mut zs: Vec<Vec<f64>> = Vec::with_capacity(self.lambda);
            let mut xs: Vec<Vec<f64>> = Vec::with_capacity(self.lambda);
            for _ in 0..self.lambda {
                let z: Vec<f64> = (0..n).map(|_| self.rng.normal(0.0, 1.0)).collect();
                let x: Vec<f64> = (0..n)
                    .map(|j| mean[j] + sigma * cov[j].sqrt() * z[j])
                    .collect();
                zs.push(z);
                xs.push(x);
            }
            let fits = eval_population(&xs, &f);

            let mut pop: Vec<(f64, Vec<f64>)> = Vec::with_capacity(self.lambda);
            for (i, fit) in fits.into_iter().enumerate() {
                if fit > best_f {
                    best_f = fit;
                    best_x = xs[i].clone();
                }
                pop.push((fit, std::mem::take(&mut zs[i])));
            }
            // tri décroissant : meilleurs en tête (on maximise)
            pop.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // --- recombinaison ------------------------------------------ //
            let mut z_w = vec![0.0; n]; // moyenne pondérée des z
            let mut y_w = vec![0.0; n]; // moyenne pondérée des y = D·z
            for i in 0..self.mu {
                let wi = self.weights[i];
                let z = &pop[i].1;
                for j in 0..n {
                    z_w[j] += wi * z[j];
                    y_w[j] += wi * cov[j].sqrt() * z[j];
                }
            }

            // mise à jour de la moyenne (c_mean = 1)
            for j in 0..n {
                mean[j] += sigma * y_w[j];
            }

            // --- chemin d'évolution pour σ (sep : C^{-1/2}·y = z) -------- //
            let cs = self.c_sigma;
            let coef_ps = (cs * (2.0 - cs) * self.mu_eff).sqrt();
            for j in 0..n {
                p_sigma[j] = (1.0 - cs) * p_sigma[j] + coef_ps * z_w[j];
            }
            let ps_norm = p_sigma.iter().map(|x| x * x).sum::<f64>().sqrt();

            // indicateur h_σ (freine la mise à jour si σ croît trop vite)
            let g1 = (gen + 1) as f64;
            let denom = (1.0 - (1.0 - cs).powf(2.0 * g1)).sqrt();
            let h_sigma = if ps_norm / denom / self.chi_n < 1.4 + 2.0 / (n as f64 + 1.0) {
                1.0
            } else {
                0.0
            };

            // --- chemin d'évolution pour la covariance ------------------ //
            let cc = self.c_c;
            let coef_pc = (cc * (2.0 - cc) * self.mu_eff).sqrt();
            for j in 0..n {
                p_c[j] = (1.0 - cc) * p_c[j] + h_sigma * coef_pc * y_w[j];
            }

            // --- mise à jour diagonale de la covariance ----------------- //
            let (c1, cmu) = (self.c_1, self.c_mu);
            let delta_hs = (1.0 - h_sigma) * cc * (2.0 - cc);
            for j in 0..n {
                let mut rank_mu = 0.0;
                for i in 0..self.mu {
                    let yj = cov[j].sqrt() * pop[i].1[j];
                    rank_mu += self.weights[i] * yj * yj;
                }
                cov[j] = (1.0 - c1 - cmu) * cov[j]
                    + c1 * (p_c[j] * p_c[j] + delta_hs * cov[j])
                    + cmu * rank_mu;
                if cov[j] < 1e-20 {
                    cov[j] = 1e-20;
                }
            }

            // --- mise à jour de l'échelle σ ----------------------------- //
            sigma *= ((cs / self.d_sigma) * (ps_norm / self.chi_n - 1.0)).exp();
            if !sigma.is_finite() || sigma < 1e-20 {
                sigma = 1e-20;
            } else if sigma > 1e6 {
                sigma = 1e6;
            }
        }

        (best_x, best_f)
    }
}

/// Évalue une population en parallèle au-delà d'un seuil (`std::thread::scope`,
/// sans dépendance). Résultat **indexé** identique au séquentiel (`f` pure) — la
/// mise à jour CMA en aval reste donc bit-exacte et déterministe.
fn eval_population<F: Fn(&[f64]) -> f64 + Sync>(xs: &[Vec<f64>], f: &F) -> Vec<f64> {
    const PAR_MIN: usize = 8;
    let n = xs.len();
    if n == 0 {
        return Vec::new();
    }
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1);
    if threads <= 1 || n < PAR_MIN {
        return xs.iter().map(|x| f(x)).collect();
    }
    let chunk = n.div_ceil(threads);
    let mut out = vec![0.0_f64; n];
    std::thread::scope(|scope| {
        for (src, dst) in xs.chunks(chunk).zip(out.chunks_mut(chunk)) {
            scope.spawn(move || {
                for (i, x) in src.iter().enumerate() {
                    dst[i] = f(x);
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimize_is_deterministic() {
        // même graine ⇒ même résultat bit-exact (l'éval parallèle préserve l'ordre)
        let run = || {
            let mut cma = SepCmaEs::new(6, 0, 2024);
            let f = |x: &[f64]| -> f64 { -x.iter().map(|a| a * a).sum::<f64>() };
            let (_x, best_f) = cma.optimize(&[1.0; 6], 0.8, 60, f);
            best_f
        };
        assert_eq!(run().to_bits(), run().to_bits());
    }

    #[test]
    fn maximizes_negative_sphere() {
        // f(x) = −Σ (x_j − target_j)²  ; maximum en x = target, f = 0
        let target = [1.5, -2.0, 0.5, 3.0];
        let mut cma = SepCmaEs::new(4, 0, 12345);
        let f = |x: &[f64]| -> f64 {
            -x.iter()
                .zip(&target)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
        };
        let (best_x, best_f) = cma.optimize(&[0.0; 4], 1.0, 200, f);
        assert!(best_f > -1e-3, "best_f = {best_f}");
        for (a, b) in best_x.iter().zip(&target) {
            assert!((a - b).abs() < 0.05, "x={best_x:?}");
        }
    }
}
