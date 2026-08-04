//! **scirust-rsi** — moteur d'ascension auto-améliorante.
//!
//! Remplacement local, sans dépendance (hors `rand`), du crate git
//! `scirust-rsi` de Memorithm. Implémente le contrat consommé par
//! `rsi/src/scirust_bridge.rs` et `examples/self_improve_real.rs` :
//!
//! - [`refine::RefineTask`] : le contrat de domaine (solution initiale, score,
//!   raffinement `1+λ` piloté par un `StdRng` reproductible) ;
//! - [`refine::SelfRefiner`] : la boucle d'ascension élitiste bornée ;
//! - [`Fitness`] : alias `f64` (plus grand = mieux) ;
//! - [`Guard`] : bornes de la boucle (`max_iters`, `patience`, `target`,
//!   `min_delta`) ;
//! - [`Report`] : rapport final (`best_fitness`, `total_gain()`, `iterations`,
//!   `accepted`, [`Report::is_monotone`]).
//!
//! La boucle est **élitiste** : une révision n'est adoptée que si sa fitness
//! dépasse strictement l'incumbent, donc `is_monotone()` est toujours vrai sur
//! une exécution réussie. Le déterminisme est assuré par graine (`StdRng`).

use rand::rngs::StdRng;

/// Fitness scalaire : plus grand = mieux.
pub type Fitness = f64;

/// Bornes de la boucle d'ascension.
#[derive(Debug, Clone)]
pub struct Guard {
    pub max_iters: usize,
    pub patience: usize,
    pub target: f64,
    pub min_delta: f64,
}

impl Default for Guard {
    fn default() -> Self {
        Guard {
            max_iters: 50,
            patience: 10,
            target: 0.99,
            min_delta: 1e-6,
        }
    }
}

impl Guard {
    pub fn new() -> Self {
        Guard::default()
    }

    pub fn max_iters(mut self, v: usize) -> Self {
        self.max_iters = v.max(1);
        self
    }

    pub fn patience(mut self, v: usize) -> Self {
        self.patience = v;
        self
    }

    pub fn target(mut self, v: f64) -> Self {
        self.target = v;
        self
    }

    pub fn min_delta(mut self, v: f64) -> Self {
        self.min_delta = v;
        self
    }
}

/// Rapport d'une ascension.
#[derive(Debug, Clone)]
pub struct Report {
    /// meilleure fitness atteinte.
    pub best_fitness: Fitness,
    /// fitness de départ.
    pub start_fitness: Fitness,
    /// nombre d'itérations effectuées.
    pub iterations: usize,
    /// nombre de révisions adoptées (strictement meilleures).
    pub accepted: usize,
    /// true si la fitness n'a jamais régressé (élitisme strict).
    pub monotone: bool,
}

impl Report {
    /// Non-régression : la meilleure fitness n'a jamais baissé.
    pub fn is_monotone(&self) -> bool {
        self.monotone
    }

    /// Gain total (fitness finale − fitness initiale).
    pub fn total_gain(&self) -> f64 {
        self.best_fitness - self.start_fitness
    }
}

/// Contrat de domaine : solution initiale + score + raffinement.
pub mod refine {
    use super::{Fitness, Guard, Report, StdRng};
    use rand::SeedableRng;

    /// Un domaine d'auto-amélioration : la solution à améliorer.
    pub trait RefineTask {
        type Solution: Clone;

        /// Solution initiale triviale (déterministe).
        fn initial(&self, rng: &mut StdRng) -> Self::Solution;

        /// Évalue une solution → fitness (plus grand = mieux).
        fn score(&self, s: &Self::Solution) -> Fitness;

        /// Produit une solution révisée (1+λ), pilotée par le RNG du moteur.
        fn refine(&self, s: &Self::Solution, rng: &mut StdRng) -> Self::Solution;
    }

    /// Moteur d'ascension élitiste borné.
    pub struct SelfRefiner {
        seed: u64,
    }

    impl SelfRefiner {
        pub fn new(seed: u64) -> Self {
            SelfRefiner { seed }
        }

        /// Exécute la boucle. Adopte une révision seulement si elle est
        /// strictement meilleure (élitisme) ; s'arrête sur `max_iters`,
        /// `patience` (pas d'amélioration) ou `target` atteint.
        ///
        /// Retourne `(meilleure_solution, rapport)`.
        pub fn run<T: RefineTask>(&self, task: &T, guard: &Guard) -> (T::Solution, Report) {
            let mut rng = StdRng::seed_from_u64(self.seed);
            let mut best = task.initial(&mut rng);
            let mut best_fit = task.score(&best);
            let start_fitness = best_fit;
            let mut monotone = true;
            let mut stalled = 0usize;
            let mut iterations = 0usize;
            let mut accepted = 0usize;

            while iterations < guard.max_iters {
                let cand = task.refine(&best, &mut rng);
                let fit = task.score(&cand);
                iterations += 1;

                if fit > best_fit + guard.min_delta {
                    best = cand;
                    best_fit = fit;
                    accepted += 1;
                    stalled = 0;
                } else {
                    stalled += 1;
                    if fit < best_fit {
                        // élitisme strict : on n'adopte jamais une régression
                        monotone = false;
                    }
                }

                if best_fit >= guard.target || stalled > guard.patience {
                    break;
                }
            }

            (
                best,
                Report {
                    best_fitness: best_fit,
                    start_fitness,
                    iterations,
                    accepted,
                    monotone,
                },
            )
        }
    }
}

pub use refine::{RefineTask, SelfRefiner};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refine::RefineTask;
    use rand::Rng;

    // Domaine jouet : maximiser x² en partant de 0 (x ∈ [0,1]).
    struct Square;
    impl RefineTask for Square {
        type Solution = f64;
        fn initial(&self, _rng: &mut StdRng) -> f64 {
            0.0
        }
        fn score(&self, s: &f64) -> Fitness {
            s * s
        }
        fn refine(&self, s: &f64, rng: &mut StdRng) -> f64 {
            let step = rng.gen_range(0.0..0.2);
            (s + step).min(1.0)
        }
    }

    #[test]
    fn refiner_improves_and_terminates() {
        let task = Square;
        let guard = Guard::new().max_iters(200).patience(20).target(0.99);
        let (_best, rep) = SelfRefiner::new(42).run(&task, &guard);
        assert!(rep.best_fitness > 0.9, "best={}", rep.best_fitness);
        assert!(rep.iterations <= 200);
        assert!(rep.total_gain() > 0.0);
        assert!(rep.is_monotone());
    }

    #[test]
    fn refiner_is_deterministic() {
        let task = Square;
        let guard = Guard::new().max_iters(50).patience(10).target(1.0);
        let (a, ra) = SelfRefiner::new(7).run(&task, &guard);
        let (b, rb) = SelfRefiner::new(7).run(&task, &guard);
        assert_eq!(a, b);
        assert_eq!(ra.best_fitness, rb.best_fitness);
        assert_eq!(ra.iterations, rb.iterations);
    }
}
