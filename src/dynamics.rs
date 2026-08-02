//! §4 — DYNAMIQUE CONTINUE AVEC CONTRAINTE DE STABILITÉ
//!
//! ```text
//! dS/dt = η(S,H,O) · [ L(D) + E(A,V) + U(H,O) ] − P(S)
//!
//! Contraintes (au pas discret) :
//!     ‖ΔS‖ < λ
//!     SI_global(t+1) ≥ SI_global(t) − ε
//! ```
//!
//! - `η(S,H,O)` : taux d'apprentissage effectif (modulé par substrat et état) ;
//! - `L(D)`     : apprentissage tiré des connaissances ;
//! - `E(A,V)`   : exploration dirigée par autonomie A, alignée sur valeurs V ;
//! - `U(H,O)`   : uplift apporté par le substrat ;
//! - `P(S)`     : pénalité dissipative (oubli, coût d'entretien).
//!
//! Les deux contraintes sont des garde-fous appliqués *après* le pas brut :
//! on borne l'amplitude (`‖ΔS‖ < λ`) puis on atténue (line search) tout pas
//! qui ferait régresser SI_global au-delà de ε.

use crate::linalg::mean;
use crate::state::{delta_norm, CognitiveState};
use crate::substrate::Substrate;
use crate::surface::IntelligenceSurface;

/// Garde-fous et hyperparamètres de la dynamique (§4).
#[derive(Clone, Copy, Debug)]
pub struct StabilityConfig {
    /// λ : borne maximale sur ‖ΔS‖.
    pub lambda: f64,
    /// ε : régression tolérée sur SI_global.
    pub epsilon: f64,
    /// η₀ : taux d'apprentissage de base.
    pub eta0: f64,
    /// Coefficient de P(S) (dissipation / oubli).
    pub forgetting: f64,
    /// ε adaptatif : si vrai, la tolérance de non-régression devient
    /// `ε + epsilon_z · stderr(SI_global)`, pour ne pas pénaliser une variation
    /// sous le bruit d'échantillonnage Monte-Carlo (§2). `false` par défaut.
    pub adaptive_epsilon: bool,
    /// multiplicateur z de l'erreur-type dans l'ε adaptatif.
    pub epsilon_z: f64,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        StabilityConfig {
            lambda: 0.5,
            epsilon: 1e-3,
            eta0: 0.15,
            forgetting: 0.02,
            adaptive_epsilon: false,
            epsilon_z: 2.0,
        }
    }
}

/// Rapport sur l'application d'un pas contraint.
#[derive(Clone, Copy, Debug)]
pub struct StepInfo {
    pub si_before: f64,
    pub si_after: f64,
    pub delta_norm: f64,
    pub clamped_to_lambda: bool,
    pub backtracks: u32,
    pub step_factor: f64,
}

/// Opérateur de dynamique : calcule dS/dt et applique le pas sous contraintes.
pub struct Dynamics<'a> {
    pub surface: &'a IntelligenceSurface,
    pub config: StabilityConfig,
}

impl<'a> Dynamics<'a> {
    pub fn new(surface: &'a IntelligenceSurface, config: StabilityConfig) -> Self {
        Dynamics { surface, config }
    }

    /// η(S,H,O) ∈ (0, η₀] — meilleur substrat & mémoire ⇒ apprentissage plus
    /// rapide ; la saturation des compétences le freine.
    pub fn eta(&self, state: &CognitiveState, substrate: &Substrate) -> f64 {
        let p_eff = substrate.effective_power();
        let memory = mean(&state.c);
        // Saturation des *compétences* (D, M, R) uniquement — ni A/V (autonomie,
        // valeurs, qui ne sont pas des compétences saturantes) ni C (mémoire, déjà
        // prise en compte ci-dessus). Tableau sur la pile : aucune allocation
        // (η est appelé jusqu'à 20× par pas dans la line search).
        let competence = mean(&[mean(&state.d), mean(&state.m), mean(&state.r)]);
        let saturation = (1.0 - competence).clamp(0.0, 1.0);
        self.config.eta0 * p_eff * (0.5 + 0.5 * memory) * saturation
    }

    /// L(D) — apprentissage tiré des connaissances ; irrigue D, M, R.
    pub fn term_l(&self, state: &CognitiveState) -> CognitiveState {
        let drive = mean(&state.d) + 0.1;
        let mut out = CognitiveState::zeros(state.dims());
        out.d.iter_mut().for_each(|x| *x = drive);
        out.m.iter_mut().for_each(|x| *x = 0.7 * drive);
        out.r.iter_mut().for_each(|x| *x = 0.8 * drive);
        out
    }

    /// E(A,V) — exploration dirigée par l'autonomie, alignée sur les valeurs.
    /// Le produit A·V garantit qu'autonomie sans valeurs (ou l'inverse) ne
    /// produit pas d'exploration utile.
    pub fn term_e(&self, state: &CognitiveState) -> CognitiveState {
        let autonomy = mean(&state.a);
        let alignment = mean(&state.v);
        let drive = autonomy * alignment;
        let mut out = CognitiveState::zeros(state.dims());
        out.r.iter_mut().for_each(|x| *x = drive);
        out.a.iter_mut().for_each(|x| *x = 0.5 * autonomy);
        out.c.iter_mut().for_each(|x| *x = 0.6 * drive);
        out
    }

    /// U(H,O) — uplift apporté par le substrat ; irrigue M et D.
    pub fn term_u(&self, state: &CognitiveState, substrate: &Substrate) -> CognitiveState {
        let p_eff = substrate.effective_power();
        let mut out = CognitiveState::zeros(state.dims());
        out.m.iter_mut().for_each(|x| *x = p_eff);
        out.d.iter_mut().for_each(|x| *x = 0.4 * p_eff);
        out
    }

    /// P(S) — pénalité dissipative proportionnelle à l'état (empêche la
    /// divergence et force l'amélioration continue).
    pub fn term_p(&self, state: &CognitiveState) -> CognitiveState {
        state.scaled(self.config.forgetting)
    }

    /// Champ de vitesse dS/dt = η · [ L + E + U ] − P.
    pub fn velocity(&self, state: &CognitiveState, substrate: &Substrate) -> CognitiveState {
        let eta = self.eta(state, substrate);
        let growth = self
            .term_l(state)
            .add(&self.term_e(state))
            .add(&self.term_u(state, substrate));
        growth.scaled(eta).sub(&self.term_p(state))
    }

    /// Pas discret sous garde-fous ‖ΔS‖ < λ et non-régression de SI_global.
    pub fn constrained_step(
        &self,
        state: &CognitiveState,
        substrate: &Substrate,
        dt: f64,
    ) -> (CognitiveState, StepInfo) {
        let cfg = self.config;
        // ε effectif : adaptatif au bruit Monte-Carlo si demandé (§2).
        let (si_before, eps) = if cfg.adaptive_epsilon {
            let (si, se) = self.surface.si_global_stats(state, substrate);
            (si, cfg.epsilon + cfg.epsilon_z * se)
        } else {
            (self.surface.si_global(state, substrate), cfg.epsilon)
        };

        // 1) pas brut issu de la dynamique continue
        let mut delta = self.velocity(state, substrate).scaled(dt);

        // 2) contrainte d'amplitude : ‖ΔS‖ < λ (projection radiale)
        let dn = delta.norm();
        let mut clamped = false;
        if dn >= cfg.lambda {
            delta = delta.scaled((cfg.lambda * 0.999) / (dn + 1e-12));
            clamped = true;
        }

        let mut candidate = state.add(&delta).clipped(0.0, 1.0);

        // 3) garde-fou de non-régression : SI(t+1) ≥ SI(t) − ε  (line search)
        let mut si_after = self.surface.si_global(&candidate, substrate);
        let mut backtracks = 0u32;
        let mut factor = 1.0;
        while si_after < si_before - eps && backtracks < 20 {
            factor *= 0.5;
            candidate = state.add(&delta.scaled(factor)).clipped(0.0, 1.0);
            si_after = self.surface.si_global(&candidate, substrate);
            backtracks += 1;
        }

        // sécurité : si même un pas infinitésimal régresse, on reste sur place
        if si_after < si_before - eps {
            candidate = state.clone();
            si_after = si_before;
            factor = 0.0;
        }

        let info = StepInfo {
            si_before,
            si_after,
            delta_norm: delta_norm(state, &candidate),
            clamped_to_lambda: clamped,
            backtracks,
            step_factor: factor,
        };
        (candidate, info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::state::Dims;

    #[test]
    fn step_respects_lambda() {
        let mut rng = Rng::new(2);
        let surf = IntelligenceSurface::sample(256, &mut rng);
        let cfg = StabilityConfig::default();
        let dyn_ = Dynamics::new(&surf, cfg);
        let state = CognitiveState::random(Dims::uniform(4), &mut rng, 0.3);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let (_next, info) = dyn_.constrained_step(&state, &sub, 1.0);
        assert!(
            info.delta_norm <= cfg.lambda + 1e-9,
            "‖ΔS‖={}",
            info.delta_norm
        );
    }

    #[test]
    fn step_never_regresses_beyond_epsilon() {
        let mut rng = Rng::new(11);
        let surf = IntelligenceSurface::sample(256, &mut rng);
        let cfg = StabilityConfig::default();
        let dyn_ = Dynamics::new(&surf, cfg);
        let mut state = CognitiveState::random(Dims::uniform(4), &mut rng, 0.3);
        let sub = Substrate::default_with(4, 4, &mut rng);
        for _ in 0..50 {
            let (next, info) = dyn_.constrained_step(&state, &sub, 1.0);
            assert!(info.si_after >= info.si_before - cfg.epsilon - 1e-9);
            state = next;
        }
    }

    /// Property-based test *in-tree* (RNG déterministe, std-only) : les deux
    /// garde-fous de `constrained_step` — amplitude `‖ΔS‖ ≤ λ` et non-régression
    /// `SI(t+1) ≥ SI(t) − ε` — tiennent sur un large espace de surfaces, états,
    /// substrats et CONFIGS EXTRÊMES (λ→0, ε→0, η₀ grand). Cible : 0 échec.
    #[test]
    fn guardrails_hold_under_random_extreme_configs() {
        let mut rng = Rng::new(0xDEAD_BEEF);
        for _ in 0..1_200 {
            // config extrême (ε non adaptatif ⇒ tolérance exacte = epsilon)
            let cfg = StabilityConfig {
                lambda: rng.uniform_range(1e-3, 1.0),
                epsilon: rng.uniform_range(0.0, 0.05),
                eta0: rng.uniform_range(0.01, 3.0),
                forgetting: rng.uniform_range(0.0, 0.1),
                adaptive_epsilon: false,
                epsilon_z: 2.0,
            };
            let n = 1 + (rng.uniform_range(0.0, 8.0) as usize); // dims 1..=8
            let surf = IntelligenceSurface::sample(48, &mut rng);
            let dyn_ = Dynamics::new(&surf, cfg);
            // État clippé dans [0,1]ⁿ : domaine opérationnel réel du moteur (qui
            // clippe à chaque pas). L'invariant d'amplitude repose sur la
            // non-expansivité de la projection, valable seulement dans ce domaine.
            let mut state =
                CognitiveState::random(Dims::uniform(n), &mut rng, 0.5).clipped(0.0, 1.0);
            let sub = Substrate::default_with(n, n, &mut rng);

            for _ in 0..12 {
                let (next, info) = dyn_.constrained_step(&state, &sub, 1.0);
                assert!(
                    info.delta_norm <= cfg.lambda + 1e-9,
                    "‖ΔS‖={} > λ={}",
                    info.delta_norm,
                    cfg.lambda
                );
                assert!(
                    info.si_after >= info.si_before - cfg.epsilon - 1e-9,
                    "régression {} < {} − ε({})",
                    info.si_after,
                    info.si_before,
                    cfg.epsilon
                );
                // bornes de cohérence Monte-Carlo
                let si = surf.si_global(&next, &sub);
                assert!((0.0..=1.0).contains(&si), "SI_global={si} hors [0,1]");
                let p = sub.effective_power();
                assert!(p > 0.0 && p < 1.0, "P_eff={p} hors (0,1)");
                state = next;
            }
        }
    }
}
