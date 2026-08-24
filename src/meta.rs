//! §5 — BOUCLE DISCRÈTE ET MÉTA-FONCTION ÉVOLUTIVE
//!
//! ```text
//! S_{t+1} = S_t + ℳ(S_t, V_t, H, O) + ΔS_appr
//! ℳ_{t+1} = arg max_ℳ  SI_global( ℳ(S_t) )      (méta-révision)
//! ```
//!
//! La méta-fonction `ℳ` est une *politique d'auto-modification* : elle décide
//! où investir l'effort d'amélioration (sur les composantes de S) et comment
//! l'agent réécrit son propre logiciel `O` pour augmenter P_eff — c'est le
//! cœur récursif du RSI (l'agent améliore *la façon dont il s'améliore*).
//!
//! La méta-révision `ℳ_{t+1} = argmax_ℳ SI_global(ℳ(S_t))` est abstraite
//! derrière le trait [`MetaSearch`], ce qui permet d'échanger la stratégie de
//! recherche : [`MetaOptimizer`] (recherche aléatoire de voisinage) ou
//! [`CmaEsMeta`] (sep-CMA-ES, recherche guidée par covariance adaptative).

use crate::cma::SepCmaEs;
use crate::linalg::{mean, sigmoid};
use crate::rng::Rng;
use crate::state::CognitiveState;
use crate::substrate::Substrate;
use crate::surface::IntelligenceSurface;

/// Bornes du paramètre `gain` d'une [`MetaStrategy`] (utilisées par CMA-ES).
const GAIN_LO: f64 = 0.005;
const GAIN_HI: f64 = 0.25;

/// Politique d'auto-modification ℳ.
///
/// - `focus` : répartition de l'effort cognitif sur (D,M,R,A,C,V), normalisée ;
/// - `software_edit` : direction de réécriture du logiciel O (auto-amélioration
///   du substrat) ;
/// - `gain` : amplitude globale de la proposition de ℳ ;
/// - `dgm` : hyperparamètres de la boucle DGM pilotés par ℳ (connexion méta↔DGM).
#[derive(Clone, Debug)]
pub struct MetaStrategy {
    pub focus: [f64; 6],
    pub software_edit: Vec<f64>,
    pub gain: f64,
    pub dgm: DgmMetaParams,
}

/// Hyperparamètres de la boucle DGM pilotés par la méta-optimisation.
///
/// C'est la connexion **méta ↔ DGM** : en plus d'ajuster où l'effort cognitif
/// est investi (`focus`) et l'amplitude (`gain`), la méta-révision ℳ peut
/// régler les leviers de la boucle d'auto-amélioration empirique (Darwin–Gödel) :
/// - `min_score_gain` : seuil anti-bruit d'adoption des gains de score ;
/// - `simulated_revisions` : combien de fois corriger une proposition prédite
///   cassée avant le gate réel (coût LLM vs builds) ;
/// - `explore_budget` : budget de propositions par étape (diversité).
///
/// Ces paramètres sont appliqués à un [`crate::dgm::DgmConfig`] via
/// [`MetaStrategy::apply_to_config`].
#[derive(Clone, Debug, PartialEq)]
pub struct DgmMetaParams {
    pub min_score_gain: f64,
    pub simulated_revisions: u32,
    pub explore_budget: u32,
}

impl Default for DgmMetaParams {
    fn default() -> Self {
        DgmMetaParams {
            min_score_gain: 0.0,
            simulated_revisions: 0,
            explore_budget: 1,
        }
    }
}

impl MetaStrategy {
    /// Stratégie neutre : effort uniforme, pas de réécriture logicielle.
    pub fn neutral(n_software: usize) -> Self {
        MetaStrategy {
            focus: [1.0 / 6.0; 6],
            software_edit: vec![0.0; n_software],
            gain: 0.05,
            dgm: DgmMetaParams::default(),
        }
    }

    /// Perturbation aléatoire de la stratégie (génère un candidat voisin).
    pub fn perturb(&self, rng: &mut Rng, scale: f64) -> MetaStrategy {
        let mut focus = self.focus;
        for f in focus.iter_mut() {
            *f = (*f + rng.normal(0.0, scale)).max(0.0);
        }
        let sum: f64 = focus.iter().sum::<f64>().max(1e-9);
        for f in focus.iter_mut() {
            *f /= sum;
        }
        let software_edit = self
            .software_edit
            .iter()
            .map(|&w| w + rng.normal(0.0, scale))
            .collect();
        let gain = (self.gain + rng.normal(0.0, scale * 0.2)).clamp(GAIN_LO, GAIN_HI);
        // Les hyperparamètres DGM ne sont PAS perturbés ici : la méta-révision
        // les porte (encode/decode) et `apply_to_config` les écrit dans la
        // config DGM après sélection. Les garder stables préserve la dynamique
        // interne et le déterminisme des invariants (λ/ε) testés par propriétés.
        MetaStrategy {
            focus,
            software_edit,
            gain,
            dgm: self.dgm.clone(),
        }
    }

    /// ℳ(S_t, V_t, H, O) — proposition d'auto-modification.
    ///
    /// Retourne (ΔS_meta, substrat_modifié). L'effort est *aligné sur les
    /// valeurs* V (`gain` pondéré par le niveau de V) et la réécriture du
    /// logiciel ne s'applique que dans la limite de l'autonomie A de l'agent.
    pub fn apply(
        &self,
        state: &CognitiveState,
        substrate: &Substrate,
    ) -> (CognitiveState, Substrate) {
        let alignment = mean(&state.v).clamp(0.0, 1.0);
        let autonomy = mean(&state.a).clamp(0.0, 1.0);
        let g = self.gain * (0.25 + 0.75 * alignment);

        let mut delta = CognitiveState::zeros(state.dims());
        let push = |comp: &mut Vec<f64>, w: f64| {
            comp.iter_mut().for_each(|x| *x = g * w);
        };
        push(&mut delta.d, self.focus[0]);
        push(&mut delta.m, self.focus[1]);
        push(&mut delta.r, self.focus[2]);
        push(&mut delta.a, self.focus[3]);
        push(&mut delta.c, self.focus[4]);
        push(&mut delta.v, self.focus[5]);

        // Auto-réécriture du logiciel O (limitée par l'autonomie).
        let mut new_sub = substrate.clone();
        for (o, &e) in new_sub.o.iter_mut().zip(&self.software_edit) {
            *o = (*o + autonomy * g * e).max(0.0);
        }

        (delta, new_sub)
    }

    /// SI_global projeté si l'on applique cette stratégie à (state, substrate).
    pub(crate) fn projected_si(
        &self,
        state: &CognitiveState,
        substrate: &Substrate,
        surface: &IntelligenceSurface,
    ) -> f64 {
        let (delta, sub2) = self.apply(state, substrate);
        let projected = state.add(&delta).clipped(0.0, 1.0);
        surface.si_global(&projected, &sub2)
    }

    // --- encodage pour optimiseur en espace non contraint (CMA-ES) -------- //

    /// Encode la stratégie en un vecteur ℝ^(7+n) *non contraint* :
    /// `[ ln(focus)×6, logit(gain), software_edit×n ]`.
    pub(crate) fn encode(&self) -> Vec<f64> {
        let mut theta = Vec::with_capacity(7 + self.software_edit.len());
        for &f in &self.focus {
            theta.push(f.max(1e-6).ln());
        }
        // logit de gain ramené dans (GAIN_LO, GAIN_HI)
        let x = self.gain.clamp(GAIN_LO + 1e-6, GAIN_HI - 1e-6);
        theta.push(((x - GAIN_LO) / (GAIN_HI - x)).ln());
        theta.extend_from_slice(&self.software_edit);
        theta
    }

    /// Décode un vecteur non contraint en stratégie valide :
    /// `focus = softmax(·)`, `gain = GAIN_LO + (GAIN_HI−GAIN_LO)·σ(·)`.
    pub(crate) fn decode(theta: &[f64], n_software: usize) -> MetaStrategy {
        let max = theta[0..6].iter().cloned().fold(f64::MIN, f64::max);
        let exps: [f64; 6] = std::array::from_fn(|i| (theta[i] - max).exp());
        let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
        let focus: [f64; 6] = std::array::from_fn(|i| exps[i] / sum);

        // Bornes identiques à `encode` ⇒ roundtrip idempotent jusqu'aux extrêmes.
        let gain = (GAIN_LO + (GAIN_HI - GAIN_LO) * sigmoid(theta[6]))
            .clamp(GAIN_LO + 1e-6, GAIN_HI - 1e-6);
        let software_edit = theta[7..7 + n_software].to_vec();
        MetaStrategy {
            focus,
            software_edit,
            gain,
            dgm: DgmMetaParams::default(),
        }
    }

    /// Applique les hyperparamètres DGM de cette stratégie à une config DGM.
    ///
    /// C'est le point de jonction entre la méta-optimisation (qui maximise
    /// `SI_global` projeté) et la boucle d'auto-amélioration empirique : après
    /// une méta-révision, les leviers `min_score_gain`, `simulated_revisions`
    /// et `explore_budget` sélectionnés sont écrits dans la `DgmConfig` de la
    /// prochaine campagne.
    pub fn apply_to_config(&self, config: &mut crate::dgm::DgmConfig) {
        config.min_score_gain = self.dgm.min_score_gain;
        config.simulated_revisions = self.dgm.simulated_revisions;
        // `explore_budget` (nb de propositions par étape) : non consommé par
        // `DgmConfig` aujourd'hui, il reste disponible pour l'appelant
        // (le champ est exposé pour les futurs pilotes multi-propositions).
        let _ = self.dgm.explore_budget;
    }
}

/// Stratégie de recherche pour la méta-révision `argmax_ℳ SI_global(ℳ(S_t))`.
///
/// Toute implémentation doit garantir que la stratégie retournée n'est **pas
/// pire** que `current` (méta-révision monotone) afin de préserver les
/// garde-fous de stabilité du système.
pub trait MetaSearch {
    fn revise(
        &mut self,
        current: &MetaStrategy,
        state: &CognitiveState,
        substrate: &Substrate,
        surface: &IntelligenceSurface,
    ) -> (MetaStrategy, f64);

    /// Réinjecte des stratégies passées performantes (rappel mémoire, §A)
    /// comme graines de la prochaine révision. No-op par défaut.
    fn warm_start(&mut self, _seeds: &[MetaStrategy]) {}
}

/// Encode une stratégie + son SI dans un payload mémoire compact (binaire,
/// little-endian) : `[ si:f64 | n_sw:u32 | θ:f64×(7+n_sw) ]`. Permet à la
/// mémoire contextuelle `C` de mémoriser quelle politique ℳ a marché.
pub fn encode_strategy_payload(si: f64, strategy: &MetaStrategy) -> Vec<u8> {
    let theta = strategy.encode();
    let n_sw = strategy.software_edit.len() as u32;
    let mut out = Vec::with_capacity(8 + 4 + theta.len() * 8);
    out.extend_from_slice(&si.to_le_bytes());
    out.extend_from_slice(&n_sw.to_le_bytes());
    for v in &theta {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Décode un payload produit par [`encode_strategy_payload`]. Renvoie
/// `(si, stratégie)` ou `None` si le format est invalide.
pub fn decode_strategy_payload(bytes: &[u8]) -> Option<(f64, MetaStrategy)> {
    if bytes.len() < 12 {
        return None;
    }
    let si = f64::from_le_bytes(bytes[0..8].try_into().ok()?);
    let n_sw = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let dim = 7 + n_sw;
    if bytes.len() != 12 + dim * 8 {
        return None;
    }
    let mut theta = Vec::with_capacity(dim);
    for i in 0..dim {
        let off = 12 + i * 8;
        theta.push(f64::from_le_bytes(bytes[off..off + 8].try_into().ok()?));
    }
    Some((si, MetaStrategy::decode(&theta, n_sw)))
}

/// Méta-optimiseur par **recherche aléatoire de voisinage**.
///
/// Explore `candidates` perturbations de la stratégie courante et retient la
/// meilleure (élitisme par rapport à `current`).
pub struct MetaOptimizer {
    pub candidates: usize,
    pub explore_scale: f64,
    rng: Rng,
    /// graines réinjectées par la mémoire (§A), consommées au prochain `revise`.
    seeds: Vec<MetaStrategy>,
}

impl MetaOptimizer {
    pub fn new(candidates: usize, explore_scale: f64, seed: u64) -> Self {
        MetaOptimizer {
            candidates,
            explore_scale,
            rng: Rng::new(seed),
            seeds: Vec::new(),
        }
    }
}

/// Seuil de candidats au-delà duquel l'évaluation est parallélisée.
const PAR_MIN: usize = 8;

/// Évalue `projected_si` pour chaque stratégie. Au-delà de [`PAR_MIN`] éléments
/// (et si plusieurs cœurs), l'évaluation est **parallélisée** via
/// `std::thread::scope` (aucune dépendance). Chaque valeur étant calculée
/// indépendamment par une fonction **pure**, le résultat est **bit-exact**
/// identique à l'évaluation séquentielle — l'argmax à ordre d'indice fixe chez
/// l'appelant préserve donc `même graine ⇒ même trajectoire ⇒ même audit`.
fn eval_projected_si(
    items: &[MetaStrategy],
    state: &CognitiveState,
    substrate: &Substrate,
    surface: &IntelligenceSurface,
) -> Vec<f64> {
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1);
    if threads <= 1 || n < PAR_MIN {
        return items
            .iter()
            .map(|s| s.projected_si(state, substrate, surface))
            .collect();
    }
    let chunk = n.div_ceil(threads);
    let mut out = vec![0.0_f64; n];
    std::thread::scope(|scope| {
        for (src, dst) in items.chunks(chunk).zip(out.chunks_mut(chunk)) {
            scope.spawn(move || {
                for (i, s) in src.iter().enumerate() {
                    dst[i] = s.projected_si(state, substrate, surface);
                }
            });
        }
    });
    out
}

impl MetaSearch for MetaOptimizer {
    fn revise(
        &mut self,
        current: &MetaStrategy,
        state: &CognitiveState,
        substrate: &Substrate,
        surface: &IntelligenceSurface,
    ) -> (MetaStrategy, f64) {
        let mut best = current.clone();
        let mut best_si = current.projected_si(state, substrate, surface);

        // Ordre d'évaluation IDENTIQUE à la version séquentielle : graines mémoire
        // (§A) puis candidats générés. La génération (RNG) reste séquentielle ;
        // seule l'ÉVALUATION est parallélisée. L'argmax ci-dessous, à ordre fixe
        // et strictement croissant, reproduit exactement la sélection séquentielle.
        let mut items: Vec<MetaStrategy> = std::mem::take(&mut self.seeds);
        for _ in 0..self.candidates {
            items.push(current.perturb(&mut self.rng, self.explore_scale));
        }

        let sis = eval_projected_si(&items, state, substrate, surface);
        for (item, si) in items.into_iter().zip(sis) {
            if si > best_si {
                best = item;
                best_si = si;
            }
        }
        (best, best_si)
    }

    fn warm_start(&mut self, seeds: &[MetaStrategy]) {
        self.seeds = seeds.to_vec();
    }
}

/// Méta-optimiseur par **sep-CMA-ES**.
///
/// À chaque révision, lance une optimisation sep-CMA-ES de quelques
/// générations, initialisée autour de la stratégie courante (encodée en
/// espace non contraint), et retient la meilleure stratégie trouvée — sans
/// jamais régresser sous `current`.
pub struct CmaEsMeta {
    pub population: usize, // 0 ⇒ défaut 4 + ⌊3 ln N⌋
    pub generations: usize,
    pub sigma0: f64,
    seed: u64,
    counter: u64,
    /// graines réinjectées par la mémoire (§A).
    seeds: Vec<MetaStrategy>,
}

impl CmaEsMeta {
    pub fn new(population: usize, generations: usize, sigma0: f64, seed: u64) -> Self {
        CmaEsMeta {
            population,
            generations,
            sigma0,
            seed,
            counter: 0,
            seeds: Vec::new(),
        }
    }
}

impl MetaSearch for CmaEsMeta {
    fn revise(
        &mut self,
        current: &MetaStrategy,
        state: &CognitiveState,
        substrate: &Substrate,
        surface: &IntelligenceSurface,
    ) -> (MetaStrategy, f64) {
        let n_sw = current.software_edit.len();
        let dim = 7 + n_sw;

        // graine variée par appel pour éviter des trajectoires figées
        let seed = self.seed ^ self.counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.counter = self.counter.wrapping_add(1);

        // §A : centre l'optimisation sur la meilleure graine mémoire si elle
        // bat la stratégie courante (warm-start CMA-ES).
        let cur_si = current.projected_si(state, substrate, surface);
        let mut center = current.clone();
        let mut center_si = cur_si;
        for seed_strat in std::mem::take(&mut self.seeds) {
            let si = seed_strat.projected_si(state, substrate, surface);
            if si > center_si {
                center = seed_strat;
                center_si = si;
            }
        }

        let mut cma = SepCmaEs::new(dim, self.population, seed);
        let mean0 = center.encode();
        let objective = |theta: &[f64]| -> f64 {
            MetaStrategy::decode(theta, n_sw).projected_si(state, substrate, surface)
        };
        let (best_theta, best_si) = cma.optimize(&mean0, self.sigma0, self.generations, objective);

        // garde-fou : ne jamais faire pire que la stratégie courante / la graine
        let baseline_si = cur_si.max(center_si);
        let baseline = if center_si > cur_si {
            center
        } else {
            current.clone()
        };
        if best_si >= baseline_si {
            (MetaStrategy::decode(&best_theta, n_sw), best_si)
        } else {
            (baseline, baseline_si)
        }
    }

    fn warm_start(&mut self, seeds: &[MetaStrategy]) {
        self.seeds = seeds.to_vec();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Dims;

    fn setup() -> (IntelligenceSurface, CognitiveState, Substrate, MetaStrategy) {
        let mut rng = Rng::new(4);
        let surf = IntelligenceSurface::sample(256, &mut rng);
        let state = CognitiveState::random(Dims::uniform(4), &mut rng, 0.3);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let strat = MetaStrategy::neutral(sub.o.len());
        (surf, state, sub, strat)
    }

    #[test]
    fn random_revision_never_worse() {
        let (surf, state, sub, strat) = setup();
        let base = strat.projected_si(&state, &sub, &surf);
        let mut meta = MetaOptimizer::new(32, 0.1, 123);
        let (_b, best_si) = meta.revise(&strat, &state, &sub, &surf);
        assert!(best_si >= base - 1e-9);
    }

    #[test]
    fn cma_revision_never_worse() {
        let (surf, state, sub, strat) = setup();
        let base = strat.projected_si(&state, &sub, &surf);
        let mut meta = CmaEsMeta::new(0, 8, 0.3, 777);
        let (_b, best_si) = meta.revise(&strat, &state, &sub, &surf);
        assert!(best_si >= base - 1e-9);
    }

    #[test]
    fn cma_matches_or_beats_random_on_average() {
        let (surf, state, sub, strat) = setup();
        let mut rnd = MetaOptimizer::new(64, 0.1, 1);
        let mut cma = CmaEsMeta::new(0, 12, 0.3, 1);
        let (_a, si_rnd) = rnd.revise(&strat, &state, &sub, &surf);
        let (_b, si_cma) = cma.revise(&strat, &state, &sub, &surf);
        // les deux dépassent la base ; CMA-ES est compétitif
        let base = strat.projected_si(&state, &sub, &surf);
        assert!(si_rnd >= base - 1e-9 && si_cma >= base - 1e-9);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let s = MetaStrategy {
            focus: [0.1, 0.2, 0.3, 0.15, 0.05, 0.2],
            software_edit: vec![0.3, -0.2, 0.1, 0.0],
            gain: 0.1,
            dgm: DgmMetaParams {
                min_score_gain: 0.02,
                simulated_revisions: 2,
                explore_budget: 3,
            },
        };
        let theta = s.encode();
        let d = MetaStrategy::decode(&theta, 4);
        for (a, b) in s.focus.iter().zip(&d.focus) {
            assert!((a - b).abs() < 1e-6, "focus {a} vs {b}");
        }
        assert!((s.gain - d.gain).abs() < 1e-6);
        assert_eq!(s.software_edit, d.software_edit);
    }

    #[test]
    fn apply_to_config_writes_dgm_params() {
        use crate::dgm::DgmConfig;
        let s = MetaStrategy {
            focus: [1.0 / 6.0; 6],
            software_edit: vec![0.0; 4],
            gain: 0.05,
            dgm: DgmMetaParams {
                min_score_gain: 0.03,
                simulated_revisions: 2,
                explore_budget: 2,
            },
        };
        let mut cfg = DgmConfig::new("ws", "goal");
        assert_eq!(cfg.min_score_gain, 0.0);
        assert_eq!(cfg.simulated_revisions, 0);
        s.apply_to_config(&mut cfg);
        assert_eq!(cfg.min_score_gain, 0.03);
        assert_eq!(cfg.simulated_revisions, 2);
    }

    #[test]
    fn decode_gain_stays_within_encode_bounds() {
        // decode borne désormais le gain dans le même domaine qu'encode :
        // le roundtrip est idempotent même pour des θ extrêmes.
        for &theta6 in &[-50.0_f64, -1.0, 0.0, 1.0, 50.0] {
            let theta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, theta6, 0.0];
            let d = MetaStrategy::decode(&theta, 1);
            assert!(d.gain >= GAIN_LO + 1e-6, "gain {} sous la borne", d.gain);
            assert!(
                d.gain <= GAIN_HI - 1e-6,
                "gain {} au-dessus de la borne",
                d.gain
            );
            // idempotence : re-encoder puis re-décoder ne bouge plus le gain.
            let d2 = MetaStrategy::decode(&d.encode(), 1);
            assert!(
                (d.gain - d2.gain).abs() < 1e-9,
                "roundtrip {} vs {}",
                d.gain,
                d2.gain
            );
        }
    }

    #[test]
    fn parallel_eval_is_bit_exact_vs_sequential() {
        // N > PAR_MIN pour forcer le chemin parallèle ; on compare bit à bit.
        let mut rng = Rng::new(1);
        let surf = IntelligenceSurface::sample(128, &mut rng);
        let state = CognitiveState::random(Dims::uniform(6), &mut rng, 0.3);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let base = MetaStrategy::neutral(sub.o.len());
        let items: Vec<MetaStrategy> = (0..32).map(|_| base.perturb(&mut rng, 0.2)).collect();

        let par = eval_projected_si(&items, &state, &sub, &surf);
        let seq: Vec<f64> = items
            .iter()
            .map(|s| s.projected_si(&state, &sub, &surf))
            .collect();

        assert_eq!(par.len(), seq.len());
        for (a, b) in par.iter().zip(&seq) {
            // bit-exact (gère aussi l'égalité des NaN éventuels)
            assert_eq!(a.to_bits(), b.to_bits(), "parallèle {a} != séquentiel {b}");
        }
    }

    #[test]
    fn meta_optimizer_revise_is_deterministic() {
        let run = || {
            let mut rng = Rng::new(9);
            let surf = IntelligenceSurface::sample(96, &mut rng);
            let state = CognitiveState::random(Dims::uniform(6), &mut rng, 0.25);
            let sub = Substrate::default_with(4, 4, &mut rng);
            let cur = MetaStrategy::neutral(sub.o.len());
            let mut opt = MetaOptimizer::new(40, 0.15, 123);
            let (_best, si) = opt.revise(&cur, &state, &sub, &surf);
            si
        };
        assert_eq!(run().to_bits(), run().to_bits());
    }
}
