//! Méta-optimiseur RSI adossé à **Forge** (feature `forge`).
//!
//! Transforme la méta-révision `ℳ_{t+1} = argmax_ℳ SI_global(ℳ(S_t))` (§5) en
//! une **recherche évolutionnaire réellement exécutée** par le moteur
//! `forge-core` (sélection de Pareto + parallélisme rayon), au lieu d'une
//! simple recherche aléatoire ou CMA-ES interne.
//!
//! Principe : un `Domain` Forge encode une [`MetaStrategy`] dans un vecteur
//! non contraint `θ` (via `MetaStrategy::encode`/`decode`), et sa fonction de
//! mesure renvoie `−SI_global` de la stratégie *projetée* (Forge **minimise**
//! ses objectifs, on maximise donc SI_global en négativant). Le meilleur
//! individu de la campagne est renvoyé — jamais pire que la stratégie courante,
//! ce qui préserve la monotonie de la méta-révision.
//!
//! Le cœur de RSI reste sans dépendance : ce module n'est compilé que si la
//! feature `forge` est activée.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use forge_core::{
    fnv1a, Candidate, CandidateId, Config, Domain, Engine, ForgeError, Result as ForgeResult,
    Score, Trial,
};
use rand::rngs::StdRng;
use rand::Rng;

use crate::meta::{MetaSearch, MetaStrategy};
use crate::state::CognitiveState;
use crate::substrate::Substrate;
use crate::surface::IntelligenceSurface;

/// Candidat Forge : un encodage `θ` d'une [`MetaStrategy`].
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct StrategyCand {
    theta: Vec<f64>,
}

impl Candidate for StrategyCand {
    fn id(&self) -> CandidateId {
        fnv1a(self.repr())
    }
    fn repr(&self) -> String {
        // représentation textuelle stable (6 décimales) → hash déterministe
        let mut s = String::with_capacity(self.theta.len() * 10);
        for v in &self.theta {
            s.push_str(&format!("{v:.6};"));
        }
        s
    }
}

/// Domaine Forge dont la fitness est `−SI_global` de la stratégie projetée.
///
/// Porte une copie immuable du contexte RSI (état, substrat, surface) pour
/// pouvoir évaluer une stratégie de façon déterministe et thread-safe.
struct RsiDomain {
    state: CognitiveState,
    substrate: Substrate,
    surface: IntelligenceSurface,
    center: Vec<f64>, // θ de la stratégie courante (centre d'exploration)
    n_software: usize,
    explore: f64,
    seed_counter: AtomicUsize,
    /// (§C) cache des SI_global par candidat (id → SI), évite de recalculer la
    /// fitness des candidats réapparus au fil des générations.
    si_cache: Mutex<HashMap<u64, f64>>,
}

impl RsiDomain {
    /// SI_global de la stratégie encodée par `theta` (mémoïsé par id de candidat).
    fn si_cached(&self, cand: &StrategyCand) -> f64 {
        let id = cand.id();
        if let Some(v) = self.si_cache.lock().unwrap().get(&id) {
            return *v;
        }
        let si = MetaStrategy::decode(&cand.theta, self.n_software).projected_si(
            &self.state,
            &self.substrate,
            &self.surface,
        );
        self.si_cache.lock().unwrap().insert(id, si);
        si
    }

    /// SI_global d'un θ brut (non mémoïsé — pour le centre/baseline).
    fn si_of(&self, theta: &[f64]) -> f64 {
        MetaStrategy::decode(theta, self.n_software).projected_si(
            &self.state,
            &self.substrate,
            &self.surface,
        )
    }

    fn perturb(&self, rng: &mut StdRng, base: &[f64]) -> Vec<f64> {
        base.iter()
            .map(|&x| x + rng.gen_range(-self.explore..=self.explore))
            .collect()
    }
}

impl Domain for RsiDomain {
    type Cand = StrategyCand;

    fn name(&self) -> &str {
        "rsi-meta"
    }

    fn seed(&self, rng: &mut StdRng) -> StrategyCand {
        // le tout premier individu est la stratégie courante exacte ; les
        // suivants sont des perturbations autour d'elle
        let i = self.seed_counter.fetch_add(1, Ordering::Relaxed);
        let theta = if i == 0 {
            self.center.clone()
        } else {
            self.perturb(rng, &self.center)
        };
        StrategyCand { theta }
    }

    fn mutate(&self, rng: &mut StdRng, parents: &[&StrategyCand]) -> ForgeResult<StrategyCand> {
        let base = parents
            .first()
            .map(|p| p.theta.as_slice())
            .unwrap_or(&self.center);
        Ok(StrategyCand {
            theta: self.perturb(rng, base),
        })
    }

    fn verify(&self, cand: &StrategyCand, _trial: &Trial) -> ForgeResult<bool> {
        // candidat valide ssi son θ est fini et de la bonne dimension
        if cand.theta.len() != 7 + self.n_software {
            return Err(ForgeError::InvalidCandidate("dimension θ".into()));
        }
        Ok(cand.theta.iter().all(|x| x.is_finite()))
    }

    fn measure(&self, cand: &StrategyCand, _trial: &Trial) -> ForgeResult<Vec<f64>> {
        // Forge minimise → on renvoie −SI_global (maximiser SI_global) ; §C cache
        Ok(vec![-self.si_cached(cand)])
    }

    fn objective_names(&self) -> Vec<String> {
        vec!["neg_si_global".into()]
    }

    fn baseline(&self, _trial: &Trial) -> ForgeResult<Score> {
        Ok(Score::valid(vec![-self.si_of(&self.center)]))
    }
}

/// Méta-optimiseur `MetaSearch` propulsé par le moteur évolutionnaire Forge.
///
/// ```ignore
/// let meta = ForgeMetaSearch::new(/*generations*/ 8, /*population*/ 24, 0.15, 42);
/// let agent = RSIAgent::new(state, substrate, surface, cfg, Box::new(meta));
/// ```
pub struct ForgeMetaSearch {
    pub generations: u64,
    pub population: usize,
    pub explore: f64,
    seed: u64,
    counter: u64,
    /// graines mémoire (§A) réinjectées au prochain `revise`.
    seeds: Vec<MetaStrategy>,
}

impl ForgeMetaSearch {
    pub fn new(generations: u64, population: usize, explore: f64, seed: u64) -> Self {
        ForgeMetaSearch {
            generations: generations.max(1),
            population: population.max(4),
            explore: explore.max(1e-4),
            seed,
            counter: 0,
            seeds: Vec::new(),
        }
    }
}

impl MetaSearch for ForgeMetaSearch {
    fn revise(
        &mut self,
        current: &MetaStrategy,
        state: &CognitiveState,
        substrate: &Substrate,
        surface: &IntelligenceSurface,
    ) -> (MetaStrategy, f64) {
        let n_software = current.software_edit.len();

        // §A — centre l'exploration sur la meilleure graine mémoire si elle bat
        // la stratégie courante (warm-start de la campagne Forge).
        let cur_si = current.projected_si(state, substrate, surface);
        let mut center_strat = current.clone();
        let mut center_si = cur_si;
        for s in std::mem::take(&mut self.seeds) {
            if s.software_edit.len() == n_software {
                let si = s.projected_si(state, substrate, surface);
                if si > center_si {
                    center_strat = s;
                    center_si = si;
                }
            }
        }
        let center = center_strat.encode();

        let base_seed = self.seed ^ self.counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.counter = self.counter.wrapping_add(1);

        let domain = RsiDomain {
            state: state.clone(),
            substrate: substrate.clone(),
            surface: surface.clone(),
            center,
            n_software,
            explore: self.explore,
            seed_counter: AtomicUsize::new(0),
            si_cache: Mutex::new(HashMap::new()),
        };

        let config = Config {
            generations: self.generations,
            population: self.population,
            survivors: (self.population / 3).max(2),
            base_seed,
        };

        // baseline = meilleure entre stratégie courante et graine mémoire retenue
        let baseline_si = center_si;
        let baseline = center_strat;

        // exécute la campagne évolutionnaire
        match Engine::new(domain, config).run() {
            Ok(report) => match report.best {
                Some(ind) => {
                    // objectives[0] = −SI_global
                    let si = ind
                        .score
                        .objectives
                        .first()
                        .map(|o| -o)
                        .unwrap_or(baseline_si);
                    if si >= baseline_si {
                        // audit m10 : préserve les hyperparams DGM (cf. meta.rs)
                        let mut revised =
                            MetaStrategy::decode(&ind.cand.theta, n_software);
                        revised.dgm = baseline.dgm.clone();
                        (revised, si)
                    } else {
                        (baseline, baseline_si)
                    }
                }
                None => (baseline, baseline_si),
            },
            Err(_) => (baseline, baseline_si), // dégradation gracieuse
        }
    }

    fn warm_start(&mut self, seeds: &[MetaStrategy]) {
        self.seeds = seeds.to_vec();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng as RsiRng;
    use crate::state::Dims;

    #[test]
    fn forge_revision_never_worse_than_current() {
        let mut rng = RsiRng::new(4);
        let surf = IntelligenceSurface::sample(256, &mut rng);
        let state = CognitiveState::random(Dims::uniform(4), &mut rng, 0.3);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let strat = MetaStrategy::neutral(sub.o.len());
        let base = strat.projected_si(&state, &sub, &surf);

        let mut meta = ForgeMetaSearch::new(6, 16, 0.15, 777);
        let (_best, si) = meta.revise(&strat, &state, &sub, &surf);
        assert!(si >= base - 1e-9, "si={si} base={base}");
    }
}
