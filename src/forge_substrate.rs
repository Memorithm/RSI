//! Substrat RSI calibré par **Forge** (feature `forge`) — Phase 2.
//!
//! Rend l'efficience logicielle de `P_eff` **réellement mesurée** au lieu de la
//! forme analytique σ(OᵀB O). Une campagne `forge-core` fait évoluer la
//! configuration d'un *vrai* kernel de multiplication matricielle (tailles de
//! tuilage) ; `measure` exécute le kernel et chronomètre, `verify` contrôle la
//! correction (porte anti-triche). Le *speedup* mesuré contre une baseline
//! naïve est transformé en efficience ∈ (0,1) et injecté via
//! [`Substrate::set_measured_software_eff`].
//!
//! Propriétés clés :
//! - **monotone** : l'efficience mesurée ne fait que croître (ratchet), donc
//!   P_eff ne régresse jamais — compatible avec les garde-fous RSI ;
//! - **local** : aucune dépendance réseau ni GPU ; le kernel tourne in-process
//!   en Rust pur (les domaines CUDA/SIMD de Forge restent disponibles pour qui
//!   dispose de la toolchain, mais ne sont pas requis ici).

use std::time::Instant;

use forge_core::{
    fnv1a, Candidate, CandidateId, Config, Domain, Engine, ForgeError, Result as ForgeResult,
    Score, Trial,
};
use rand::rngs::StdRng;
use rand::Rng as _;

use crate::substrate::{Substrate, SubstrateImprover};

/// Tailles de tuile autorisées (indexées par le candidat).
const TILES: [usize; 4] = [8, 16, 32, 64];

/// Candidat = trois index de tuile (bm, bn, bk), encodés en réels.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct KernelCand {
    knobs: [f64; 3],
}

impl KernelCand {
    fn tiles(&self) -> (usize, usize, usize) {
        let idx = |x: f64| TILES[(x.round().max(0.0) as usize).min(TILES.len() - 1)];
        (idx(self.knobs[0]), idx(self.knobs[1]), idx(self.knobs[2]))
    }
}

impl Candidate for KernelCand {
    fn id(&self) -> CandidateId {
        fnv1a(self.repr())
    }
    fn repr(&self) -> String {
        let (bm, bn, bk) = self.tiles();
        format!("tile:{bm}x{bn}x{bk}")
    }
}

// --------------------------------------------------------------------- //
// Kernels réels (f32, row-major)
// --------------------------------------------------------------------- //

/// Baseline « manuel » : ordre i,j,k (k le plus interne). L'accès `b[k*n+j]`
/// saute de `n` à chaque itération → hostile au cache pour `n` grand. C'est la
/// référence que le tuilage doit battre.
fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
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

fn random_matrix(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::from_seed_u64(seed);
    (0..n * n).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

// petit pont : StdRng::from_seed via u64
trait FromSeedU64 {
    fn from_seed_u64(seed: u64) -> Self;
}
impl FromSeedU64 for StdRng {
    fn from_seed_u64(seed: u64) -> Self {
        use rand::SeedableRng;
        StdRng::seed_from_u64(seed)
    }
}

/// Domaine Forge : optimise le tuilage d'un matmul N×N pour minimiser le temps.
struct KernelDomain {
    n: usize,
    reps: usize,
    center: [f64; 3],
    explore: f64,
    counter: std::sync::atomic::AtomicUsize,
}

impl KernelDomain {
    fn time_tiled(&self, cand: &KernelCand, seed: u64) -> (f64, bool) {
        let (bm, bn, bk) = cand.tiles();
        let a = random_matrix(self.n, seed);
        let b = random_matrix(self.n, seed ^ 0xBEEF);
        let mut c = vec![0.0f32; self.n * self.n];
        // correction vs naïf
        let mut reference = vec![0.0f32; self.n * self.n];
        matmul_naive(&a, &b, &mut reference, self.n);
        matmul_tiled(&a, &b, &mut c, self.n, bm, bn, bk);
        let correct = c
            .iter()
            .zip(&reference)
            .all(|(x, y)| (x - y).abs() <= 1e-2 * (1.0 + y.abs()));
        // temps médian sur `reps` exécutions
        let mut times = Vec::with_capacity(self.reps);
        for _ in 0..self.reps {
            let t0 = Instant::now();
            matmul_tiled(&a, &b, &mut c, self.n, bm, bn, bk);
            times.push(t0.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (times[times.len() / 2], correct)
    }
}

impl Domain for KernelDomain {
    type Cand = KernelCand;

    fn name(&self) -> &str {
        "rsi-kernel"
    }

    fn seed(&self, rng: &mut StdRng) -> KernelCand {
        use std::sync::atomic::Ordering;
        let i = self.counter.fetch_add(1, Ordering::Relaxed);
        let knobs = if i == 0 {
            self.center
        } else {
            std::array::from_fn(|j| self.center[j] + rng.gen_range(-self.explore..=self.explore))
        };
        KernelCand { knobs }
    }

    fn mutate(&self, rng: &mut StdRng, parents: &[&KernelCand]) -> ForgeResult<KernelCand> {
        let base = parents.first().map(|p| p.knobs).unwrap_or(self.center);
        Ok(KernelCand {
            knobs: std::array::from_fn(|j| base[j] + rng.gen_range(-self.explore..=self.explore)),
        })
    }

    fn verify(&self, cand: &KernelCand, trial: &Trial) -> ForgeResult<bool> {
        Ok(self.time_tiled(cand, trial.seed).1)
    }

    fn measure(&self, cand: &KernelCand, trial: &Trial) -> ForgeResult<Vec<f64>> {
        let (t, correct) = self.time_tiled(cand, trial.seed);
        if !correct {
            return Err(ForgeError::Evaluation("kernel incorrect".into()));
        }
        Ok(vec![t]) // minimiser le temps
    }

    fn objective_names(&self) -> Vec<String> {
        vec!["seconds".into()]
    }

    fn baseline(&self, trial: &Trial) -> ForgeResult<Score> {
        let a = random_matrix(self.n, trial.seed);
        let b = random_matrix(self.n, trial.seed ^ 0xBEEF);
        let mut c = vec![0.0f32; self.n * self.n];
        let t0 = Instant::now();
        matmul_naive(&a, &b, &mut c, self.n);
        Ok(Score::valid(vec![t0.elapsed().as_secs_f64()]))
    }
}

/// Améliorateur de substrat : calibre l'efficience logicielle de P_eff par une
/// campagne Forge sur un vrai kernel matriciel. Conserve le meilleur tuilage et
/// le meilleur speedup d'un appel à l'autre (progression graduelle, monotone).
pub struct ForgeSubstrate {
    n: usize,
    reps: usize,
    generations: u64,
    population: usize,
    explore: f64,
    best_knobs: [f64; 3],
    best_speedup: f64,
    /// efficience logicielle analytique capturée au 1er appel (ancre).
    anchor: Option<f64>,
    seed: u64,
    counter: u64,
}

impl ForgeSubstrate {
    /// `n` = taille du kernel N×N ; budget par appel = `generations`×`population`.
    pub fn new(n: usize, generations: u64, population: usize, seed: u64) -> Self {
        ForgeSubstrate {
            n: n.max(8),
            reps: 3,
            generations: generations.max(1),
            population: population.max(4),
            explore: 1.2,
            best_knobs: [1.0, 1.0, 1.0], // tuile 16×16×16 au départ
            best_speedup: 1.0,
            anchor: None,
            seed,
            counter: 0,
        }
    }

    /// Efficience mesurée ∈ [ancre, 1) : part de l'efficience analytique et
    /// capture une fraction `1 − 1/speedup` du headroom restant. À speedup = 1,
    /// vaut l'ancre (aucun changement) ; croît vers 1 avec le speedup réel.
    fn efficiency(&self, anchor: f64) -> f64 {
        let s = self.best_speedup.max(1.0);
        anchor + (1.0 - anchor) * (1.0 - 1.0 / s)
    }
}

impl SubstrateImprover for ForgeSubstrate {
    fn improve(&mut self, substrate: &Substrate) -> Substrate {
        let base_seed = self.seed ^ self.counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.counter = self.counter.wrapping_add(1);

        let domain = KernelDomain {
            n: self.n,
            reps: self.reps,
            center: self.best_knobs,
            explore: self.explore,
            counter: std::sync::atomic::AtomicUsize::new(0),
        };
        let config = Config {
            generations: self.generations,
            population: self.population,
            survivors: (self.population / 3).max(2),
            base_seed,
            worker_addresses: None,
        };

        // ancre l'efficience analytique de départ au premier appel
        let anchor = *self
            .anchor
            .get_or_insert_with(|| substrate.software_efficiency());

        if let Ok(report) = Engine::new(domain, config).run() {
            if let (Some(best), Some(baseline)) = (report.best, report.final_baseline) {
                let best_t = best.score.objectives.first().copied().unwrap_or(f64::MAX);
                let base_t = baseline.objectives.first().copied().unwrap_or(best_t);
                if best_t > 0.0 {
                    let speedup = base_t / best_t;
                    if speedup > self.best_speedup {
                        self.best_speedup = speedup;
                        self.best_knobs = best.cand.knobs;
                    }
                }
            }
        }

        // efficience mesurée (≥ ancre), appliquée seulement si elle ne dégrade
        // pas le facteur logiciel courant (monotonie stricte de P_eff)
        let mut out = substrate.clone();
        let measured = self.efficiency(anchor).max(out.software_efficiency());
        out.set_measured_software_eff(Some(measured));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng as RsiRng;

    #[test]
    fn kernels_agree() {
        let n = 32;
        let a = random_matrix(n, 1);
        let b = random_matrix(n, 2);
        let (mut c1, mut c2) = (vec![0.0f32; n * n], vec![0.0f32; n * n]);
        matmul_naive(&a, &b, &mut c1, n);
        matmul_tiled(&a, &b, &mut c2, n, 16, 16, 8);
        assert!(c1
            .iter()
            .zip(&c2)
            .all(|(x, y)| (x - y).abs() <= 1e-2 * (1.0 + y.abs())));
    }

    #[test]
    fn improve_never_lowers_p_eff() {
        let mut rng = RsiRng::new(5);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let p_before = sub.effective_power();
        let mut imp = ForgeSubstrate::new(96, 2, 6, 123);
        let improved = imp.improve(&sub);
        // monotonie : P_eff ne baisse jamais
        assert!(improved.effective_power() >= p_before - 1e-12);
        // l'efficience mesurée a été posée et reste dans (0,1)
        let m = improved.measured_software_eff.unwrap();
        assert!(m > 0.0 && m < 1.0, "measured = {m}");
    }
}
