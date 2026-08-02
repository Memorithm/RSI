//! Générateur pseudo-aléatoire (std-only, aucune dépendance externe).
//!
//! Implémente `xoshiro256**` + transformations utiles (uniforme, normale via
//! Box–Muller, échantillon de Dirichlet pour tirer les tâches de Ω ~ μ, §1).

/// PRNG xoshiro256** déterministe et reproductible (seed -> séquence).
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Crée un générateur à partir d'une graine (via splitmix64 pour l'état).
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut next = || {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
        }
    }

    /// État interne brut (pour sérialisation / reprise déterministe).
    pub fn state(&self) -> [u64; 4] {
        self.s
    }

    /// Reconstruit un RNG depuis un état brut.
    pub fn from_state(s: [u64; 4]) -> Self {
        Rng { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Réel uniforme dans [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        // 53 bits de mantisse
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Réel uniforme dans [lo, hi).
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Variable normale N(mean, std) via Box–Muller.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        let mag = std * (-2.0 * u1.ln()).sqrt();
        mean + mag * (std::f64::consts::TAU * u2).cos()
    }

    /// Échantillon Gamma(shape, 1) — méthode de Marsaglia & Tsang.
    fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let u = self.uniform().max(1e-12);
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal(0.0, 1.0);
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.uniform().max(1e-12);
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    /// Échantillon de Dirichlet(alpha) de dimension `alpha.len()`.
    /// Utilisé pour générer un profil de besoins de tâche sur (D,M,R,A,C,V).
    pub fn dirichlet(&mut self, alpha: &[f64]) -> Vec<f64> {
        let gammas: Vec<f64> = alpha.iter().map(|&a| self.gamma(a)).collect();
        let sum: f64 = gammas.iter().sum::<f64>().max(1e-12);
        gammas.iter().map(|g| g / sum).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn uniform_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn dirichlet_sums_to_one() {
        let mut r = Rng::new(1);
        let d = r.dirichlet(&[1.0; 6]);
        let s: f64 = d.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
        assert!(d.iter().all(|&x| x >= 0.0));
    }
}
