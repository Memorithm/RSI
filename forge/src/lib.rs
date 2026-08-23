//! **forge-core** — moteur d'optimisation évolutionnaire de RSI.
//!
//! Petit moteur évolutionnaire réel qui remplace la dépendance git privée
//! `forge-core` de Memorithm. Il implémente le contrat consommé par
//! `rsi/src/forge_meta.rs` (méta-optimisation ℳ) et
//! `rsi/src/forge_substrate.rs` (calibrage `P_eff` mesuré) :
//!
//! - [`Domain`] : l'espace de recherche (génération, mutation, mesure, vérif) ;
//! - [`Engine`] : exécute une campagne évolutionnaire par générations
//!   (sélection, mutation, élitisme), déterministe par graine ;
//! - [`Candidate`] / [`Trial`] / [`Score`] : les types manipulés par la boucle.
//!
//! Le trait [`Domain`] est générique sur son type de candidat. Un candidat est
//! identifié par un hash ([`fnv1a`]) et une représentation textuelle stable, ce
//! qui rend la campagne reproductible. Le moteur **minimise** les objectifs
//! (convention Forge) : renvoyer `−SI_global` maximise `SI_global`.
//!
//! Le générateur aléatoire est le `StdRng` du crate `rand` (alias `ChaCha12Rng`),
//! le même que celui utilisé par les domaines RSI — déterminisme partagé.

use std::fmt;

pub use rand::rngs::StdRng;
pub use rand::{Rng, SeedableRng};

/// FNV-1a 64 bits — hash déterministe des représentations de candidats.
/// Accepte tout type `AsRef<[u8]>` (`&str`, `&String`, `&[u8]`, …).
pub fn fnv1a(bytes: impl AsRef<[u8]>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes.as_ref() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Identifiant stable d'un candidat (dérivé de sa représentation textuelle).
pub type CandidateId = u64;

/// Erreurs du moteur Forge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeError {
    /// Candidat invalide (dimension, valeurs non finies, …).
    InvalidCandidate(String),
    /// Échec d'évaluation (kernel incorrect, mesure impossible, …).
    Evaluation(String),
    /// Configuration incohérente (population trop petite, …).
    Config(String),
}

impl fmt::Display for ForgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForgeError::InvalidCandidate(m) => write!(f, "candidat invalide : {m}"),
            ForgeError::Evaluation(m) => write!(f, "échec d'évaluation : {m}"),
            ForgeError::Config(m) => write!(f, "configuration invalide : {m}"),
        }
    }
}

impl std::error::Error for ForgeError {}

/// Résultat renvoyé par les opérations du domaine.
pub type Result<T> = std::result::Result<T, ForgeError>;

/// Résultat d'évaluation d'un candidat (vecteur d'objectifs, convention minimisation).
#[derive(Debug, Clone)]
pub struct Score {
    pub objectives: Vec<f64>,
}

impl Score {
    /// Construit un score valide (objectifs finis).
    pub fn valid(objectives: Vec<f64>) -> Self {
        Score { objectives }
    }
}

/// Un essai d'évaluation : couple candidat + graine déterministe.
#[derive(Debug, Clone)]
pub struct Trial {
    pub seed: u64,
}

impl Trial {
    pub fn new(seed: u64) -> Self {
        Trial { seed }
    }
}

/// Un candidat évolutionnaire (générique).
pub trait Candidate: Clone + Send + Sync + 'static {
    /// Identifiant stable (hash de la représentation).
    fn id(&self) -> CandidateId;
    /// Représentation textuelle stable (déterministe).
    fn repr(&self) -> String;
}

/// Domaine de recherche : comment générer, muter, vérifier et mesurer.
pub trait Domain: Send + Sync {
    type Cand: Candidate;

    fn name(&self) -> &str;
    /// Génère un candidat initial (le premier doit être un bon point de départ).
    fn seed(&self, rng: &mut StdRng) -> Self::Cand;
    /// Produit un candidat dérivé des parents (recombinaison/mutation).
    fn mutate(&self, rng: &mut StdRng, parents: &[&Self::Cand]) -> Result<Self::Cand>;
    /// Vérifie qu'un candidat est valide *avant* mesure (porte anti-triche).
    fn verify(&self, cand: &Self::Cand, trial: &Trial) -> Result<bool>;
    /// Mesure les objectifs d'un candidat (minimisation).
    fn measure(&self, cand: &Self::Cand, trial: &Trial) -> Result<Vec<f64>>;
    /// Noms des objectifs (diagnostic).
    fn objective_names(&self) -> Vec<String>;
    /// Score de référence (baseline) pour un essai.
    fn baseline(&self, trial: &Trial) -> Result<Score>;
}

/// Configuration d'une campagne Forge.
#[derive(Debug, Clone)]
pub struct Config {
    pub generations: u64,
    pub population: usize,
    pub survivors: usize,
    pub base_seed: u64,
    /// Adresses de workers distants (non utilisé localement — `None`).
    pub worker_addresses: Option<Vec<String>>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            generations: 8,
            population: 24,
            survivors: 8,
            base_seed: 42,
            worker_addresses: None,
        }
    }
}

/// Individu évalué de la population.
#[derive(Debug, Clone)]
pub struct Individual<C: Candidate> {
    pub cand: C,
    pub score: Score,
}

/// Rapport d'une campagne évolutionnaire.
#[derive(Debug, Clone)]
pub struct Report<C: Candidate> {
    pub best: Option<Individual<C>>,
    pub final_baseline: Option<Score>,
    pub generations: u64,
}

/// Moteur évolutionnaire Forge : sélection + mutation + élitisme.
pub struct Engine<D: Domain> {
    domain: D,
    config: Config,
}

impl<D: Domain> Engine<D> {
    pub fn new(domain: D, config: Config) -> Self {
        Engine { domain, config }
    }

    /// Clé scalaire de classement (minimisation) : somme **signée** des
    /// objectifs. Un objectif très bon (négatif) améliore la clé au lieu de la
    /// pénaliser ; une mesure malformée (NaN) vaut +∞ et ne peut jamais gagner.
    fn scalar_key(score: &Score) -> f64 {
        let s: f64 = score.objectives.iter().copied().sum();
        if s.is_nan() {
            f64::INFINITY
        } else {
            s
        }
    }

    fn sort_best(&self, pop: &mut [Individual<D::Cand>]) {
        // minimisation : meilleur = plus petite somme signée des objectifs
        // (agrégation déterministe en multi-objectif).
        pop.sort_by(|a, b| {
            let ka = Self::scalar_key(&a.score);
            let kb = Self::scalar_key(&b.score);
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn evaluate(&self, cand: &D::Cand, seed: u64) -> Result<Individual<D::Cand>> {
        let trial = Trial::new(seed);
        let valid = self.domain.verify(cand, &trial)?;
        if !valid {
            return Err(ForgeError::Evaluation(format!(
                "candidat {} refusé par verify",
                cand.repr()
            )));
        }
        let objectives = self.domain.measure(cand, &trial)?;
        Ok(Individual {
            cand: cand.clone(),
            score: Score::valid(objectives),
        })
    }

    /// Exécute la campagne. Évaluation séquentielle (déterministe bit-à-bit) ;
    /// la parallélisation bit-exacte est réalisée au niveau RSI pour les
    /// évaluations pures.
    pub fn run(&self) -> Result<Report<D::Cand>> {
        let gens = self.config.generations.max(1);
        let pop_size = self.config.population.max(2);
        let survivors = self.config.survivors.clamp(1, pop_size);

        let final_baseline = self
            .domain
            .baseline(&Trial::new(self.config.base_seed))
            .ok();

        let mut rng = StdRng::seed_from_u64(self.config.base_seed);
        let mut population: Vec<Individual<D::Cand>> = (0..pop_size)
            .map(|i| {
                let cand = self.domain.seed(&mut rng);
                let seed = self.config.base_seed ^ (i as u64).wrapping_mul(0x9E37_79B9);
                self.evaluate(&cand, seed)
            })
            .collect::<Result<Vec<_>>>()?;

        for gen in 0..gens {
            self.sort_best(&mut population);
            population.truncate(survivors);

            let parents: Vec<&D::Cand> = population.iter().map(|i| &i.cand).collect();
            let mut next: Vec<Individual<D::Cand>> = population.clone();
            while next.len() < pop_size {
                let cand = self.domain.mutate(&mut rng, &parents)?;
                let seed = self.config.base_seed
                    ^ (gen + 1).wrapping_mul(0x9E37_79B9)
                    ^ (next.len() as u64).wrapping_mul(0x85EB_CA6B);
                next.push(self.evaluate(&cand, seed)?);
            }
            population = next;
        }

        self.sort_best(&mut population);
        let best = population.into_iter().next();

        Ok(Report {
            best,
            final_baseline,
            generations: gens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Quad {
        x: f64,
    }

    impl Candidate for Quad {
        fn id(&self) -> CandidateId {
            fnv1a(self.repr().as_bytes())
        }
        fn repr(&self) -> String {
            format!("{:.6}", self.x)
        }
    }

    struct QuadDomain;
    impl Domain for QuadDomain {
        type Cand = Quad;
        fn name(&self) -> &str {
            "quad"
        }
        fn seed(&self, rng: &mut StdRng) -> Quad {
            Quad {
                x: rng.gen_range(-10.0..10.0),
            }
        }
        fn mutate(&self, rng: &mut StdRng, parents: &[&Quad]) -> Result<Quad> {
            let base = parents.first().map(|p| p.x).unwrap_or(0.0);
            Ok(Quad {
                x: base + rng.gen_range(-1.0..1.0),
            })
        }
        fn verify(&self, _cand: &Quad, _t: &Trial) -> Result<bool> {
            Ok(true)
        }
        fn measure(&self, cand: &Quad, _t: &Trial) -> Result<Vec<f64>> {
            Ok(vec![cand.x * cand.x])
        }
        fn objective_names(&self) -> Vec<String> {
            vec!["x2".into()]
        }
        fn baseline(&self, _t: &Trial) -> Result<Score> {
            Ok(Score::valid(vec![100.0]))
        }
    }

    #[test]
    fn engine_finds_better_than_baseline() {
        let cfg = Config {
            generations: 30,
            population: 20,
            survivors: 5,
            base_seed: 7,
            worker_addresses: None,
        };
        let rep = Engine::new(QuadDomain, cfg).run().unwrap();
        let best = rep.best.unwrap();
        assert!(best.score.objectives[0] < 1.0, "best={:?}", best.score);
        let base = rep.final_baseline.unwrap();
        assert!(best.score.objectives[0] < base.objectives[0]);
    }

    #[test]
    fn engine_is_deterministic() {
        let cfg = || Config {
            generations: 10,
            population: 12,
            survivors: 4,
            base_seed: 99,
            worker_addresses: None,
        };
        let a = Engine::new(QuadDomain, cfg()).run().unwrap();
        let b = Engine::new(QuadDomain, cfg()).run().unwrap();
        assert_eq!(a.best.unwrap().cand.repr(), b.best.unwrap().cand.repr());
    }

    /// Régression E4 : un objectif très bon (négatif) doit améliorer le
    /// classement, pas le pénaliser (l'ancienne clé sommait les |objectifs|).
    #[test]
    fn signed_objective_sorting() {
        let good = Individual {
            cand: Quad { x: 0.0 },
            score: Score::valid(vec![-10.0, 5.0]), // somme −5
        };
        let mediocre = Individual {
            cand: Quad { x: 1.0 },
            score: Score::valid(vec![1.0, 1.0]), // somme +2
        };
        let engine = Engine::new(QuadDomain, Config::default());
        let mut pop = vec![mediocre, good.clone()];
        engine.sort_best(&mut pop);
        assert_eq!(pop[0].score.objectives, vec![-10.0, 5.0]);
        // NaN ne gagne jamais
        let nan = Individual { cand: Quad { x: 2.0 }, score: Score::valid(vec![f64::NAN]) };
        let mut pop = vec![nan, good];
        engine.sort_best(&mut pop);
        assert!(pop[0].score.objectives[0].is_finite());
    }
}
