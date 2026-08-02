//! §1 — MORPHOLOGIE DE LA SURFACE D'INTELLIGENCE
//!
//! ```text
//! (Ω, 𝒜, μ)                        espace probabilisé des tâches/contextes
//! Σ_I(t) = { (x, C_réel(x,t)) | x ∈ Ω }     surface (graphe dans Ω×[0,1])
//! C_réel(x,t) = min( Φ_x(S(t)), g_x(P_eff) )
//! SI_global(t) = ∫_Ω C_réel(x,t) dμ(x)      volume sous la surface
//! ```
//!
//! - `x ∈ Ω` : tâche/contexte (profil de besoins sur D,M,R,A,C,V) ;
//! - `μ` : mesure de probabilité (importance/fréquence des tâches) ;
//! - `Φ_x(S)` : compétence *cognitive* de l'agent sur x ;
//! - `g_x(P_eff)` : plafond *physique* imposé par le substrat ;
//! - `C_réel = min(Φ, g)` : goulot d'étranglement cognitif OU matériel ;
//! - `SI_global` : volume sous la surface, estimé par Monte-Carlo sur μ.
//!
//! Les fonctions `Φ_x` et `g_x` sont **configurables** via les traits
//! [`CapabilityModel`] et [`CeilingModel`] : on peut brancher n'importe quelle
//! loi de compétence/plafond sans toucher au reste du système.

use std::fmt::Debug;

use crate::linalg::dot;
use crate::rng::Rng;
use crate::state::CognitiveState;
use crate::substrate::Substrate;

// ----------------------------------------------------------------------- //
// Traits configurables Φ_x et g_x
// ----------------------------------------------------------------------- //

/// Modèle de compétence cognitive `Φ_x(S) ∈ [0, 1]`.
///
/// Reçoit le profil de besoins d'une tâche (`task`, 6 poids sur D,M,R,A,C,V)
/// et les niveaux de capacité courants de l'agent (`caps`).
///
/// `Send + Sync` est requis pour permettre l'évaluation parallèle de la
/// surface (p. ex. par un méta-optimiseur externe comme Forge).
pub trait CapabilityModel: Debug + Send + Sync {
    fn phi(&self, task: &[f64; 6], caps: &[f64; 6]) -> f64;
    /// Support du `Clone` pour les objets-traits (pattern clone_box).
    fn clone_box(&self) -> Box<dyn CapabilityModel>;
}

/// Modèle de plafond physique `g_x(P_eff) ∈ [0, 1]`.
///
/// Reçoit l'efficacité du substrat `p_eff` et l'exigence calculatoire
/// normalisée `demand ∈ [0,1]` de la tâche.
pub trait CeilingModel: Debug + Send + Sync {
    fn g(&self, p_eff: f64, demand: f64) -> f64;
    fn clone_box(&self) -> Box<dyn CeilingModel>;
}

impl Clone for Box<dyn CapabilityModel> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
impl Clone for Box<dyn CeilingModel> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Compétence par sigmoïde décalée : `Φ = σ( (⟨task,caps⟩ − bias) · slope )`.
///
/// C'est le modèle par défaut : la compétence est faible tant que les
/// capacités ne couvrent pas les besoins de la tâche, puis sature vers 1.
#[derive(Clone, Debug)]
pub struct SigmoidCapability {
    pub slope: f64,
    pub bias: f64,
}

impl Default for SigmoidCapability {
    fn default() -> Self {
        SigmoidCapability {
            slope: 4.0,
            bias: 0.5,
        }
    }
}

impl CapabilityModel for SigmoidCapability {
    fn phi(&self, task: &[f64; 6], caps: &[f64; 6]) -> f64 {
        let raw = dot(task, caps);
        1.0 / (1.0 + (-(raw - self.bias) * self.slope).exp())
    }
    fn clone_box(&self) -> Box<dyn CapabilityModel> {
        Box::new(self.clone())
    }
}

/// Plafond en loi de puissance : `g = P_eff^demand`.
///
/// Une tâche lourde (demand → 1) est davantage bridée par un substrat faible ;
/// une tâche légère (demand → 0) atteint un plafond proche de 1.
#[derive(Clone, Debug, Default)]
pub struct PowerCeiling;

impl CeilingModel for PowerCeiling {
    fn g(&self, p_eff: f64, demand: f64) -> f64 {
        p_eff.powf(demand)
    }
    fn clone_box(&self) -> Box<dyn CeilingModel> {
        Box::new(self.clone())
    }
}

// ----------------------------------------------------------------------- //
// Surface d'intelligence
// ----------------------------------------------------------------------- //

/// Surface d'intelligence Σ_I et fonctionnelle SI_global.
///
/// Ω est échantillonné une seule fois (Monte-Carlo selon μ) ; le même
/// échantillon sert à toutes les évaluations, ce qui rend SI_global
/// comparable d'un pas à l'autre (estimateur cohérent — essentiel pour le
/// garde-fou de non-régression §4 et le `argmax` méta §5).
#[derive(Clone, Debug)]
pub struct IntelligenceSurface {
    /// Profils de tâches, chacun un vecteur de 6 poids (D,M,R,A,C,V).
    pub tasks: Vec<[f64; 6]>,
    /// Exigence calculatoire normalisée de chaque tâche ∈ [0, 1].
    pub demand: Vec<f64>,
    /// Poids de μ, normalisés (∑ = 1).
    pub weights: Vec<f64>,
    /// Modèle de compétence cognitive Φ_x (configurable).
    pub capability: Box<dyn CapabilityModel>,
    /// Modèle de plafond physique g_x (configurable).
    pub ceiling: Box<dyn CeilingModel>,
}

impl IntelligenceSurface {
    /// Tire Ω ~ μ avec les modèles par défaut (sigmoïde + loi de puissance).
    pub fn sample(n_tasks: usize, rng: &mut Rng) -> Self {
        Self::sample_with(
            n_tasks,
            rng,
            Box::new(SigmoidCapability::default()),
            Box::new(PowerCeiling),
        )
    }

    /// Tire Ω ~ μ avec des modèles Φ_x / g_x personnalisés.
    pub fn sample_with(
        n_tasks: usize,
        rng: &mut Rng,
        capability: Box<dyn CapabilityModel>,
        ceiling: Box<dyn CeilingModel>,
    ) -> Self {
        let mut tasks = Vec::with_capacity(n_tasks);
        let mut weights = Vec::with_capacity(n_tasks);
        let mut raw_demand = Vec::with_capacity(n_tasks);

        let alpha = [1.0; 6];
        for _ in 0..n_tasks {
            let d = rng.dirichlet(&alpha);
            tasks.push([d[0], d[1], d[2], d[3], d[4], d[5]]);
            raw_demand.push(rng.uniform_range(0.5, 2.0));
            weights.push(rng.uniform_range(0.2, 1.0));
        }

        // normalise la demande dans [0,1]
        let max_d = raw_demand
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max)
            .max(1e-9);
        let demand: Vec<f64> = raw_demand.iter().map(|r| r / max_d).collect();

        // normalise μ
        let sum_w: f64 = weights.iter().sum::<f64>().max(1e-12);
        for w in weights.iter_mut() {
            *w /= sum_w;
        }

        IntelligenceSurface {
            tasks,
            demand,
            weights,
            capability,
            ceiling,
        }
    }

    /// Construit une surface à partir de tâches **explicites** (Ω *curé*, pas
    /// échantillonné aléatoirement). Les `demand` sont clampés dans `[0,1]`
    /// (valeurs absolues respectées, pas de renormalisation) ; les `weights`
    /// sont normalisés (∑ = 1) pour que `SI_global` reste une moyenne pondérée.
    ///
    /// Panique si les longueurs diffèrent ou si `tasks` est vide.
    pub fn from_tasks(
        tasks: Vec<[f64; 6]>,
        demand: Vec<f64>,
        weights: Vec<f64>,
        capability: Box<dyn CapabilityModel>,
        ceiling: Box<dyn CeilingModel>,
    ) -> Self {
        assert!(!tasks.is_empty(), "Ω ne peut pas être vide");
        assert_eq!(tasks.len(), demand.len(), "demand: longueur ≠ tâches");
        assert_eq!(tasks.len(), weights.len(), "weights: longueur ≠ tâches");

        let demand: Vec<f64> = demand.iter().map(|d| d.clamp(0.0, 1.0)).collect();
        let sum_w: f64 = weights.iter().sum::<f64>().max(1e-12);
        let weights: Vec<f64> = weights.iter().map(|w| w / sum_w).collect();

        IntelligenceSurface {
            tasks,
            demand,
            weights,
            capability,
            ceiling,
        }
    }

    /// Décomposition **par tâche** : `(Φ_x, g_x, C_réel = min(Φ,g))`.
    ///
    /// Permet de voir, tâche par tâche, si le goulot est **cognitif** (`Φ < g`)
    /// ou **substrat** (`g < Φ`) — c.-à-d. comment `Φ_x` et `g_x` s'opposent.
    pub fn task_breakdown(
        &self,
        state: &CognitiveState,
        substrate: &Substrate,
    ) -> Vec<(f64, f64, f64)> {
        let p_eff = substrate.effective_power();
        let caps = state.capability_array();
        self.tasks
            .iter()
            .zip(&self.demand)
            .map(|(task, &dem)| {
                let phi = self.capability.phi(task, &caps);
                let g = self.ceiling.g(p_eff, dem);
                (phi, g, phi.min(g))
            })
            .collect()
    }

    /// C_réel(x,t) = min( Φ_x(S), g_x(P_eff) ) pour chaque tâche de Ω.
    pub fn real_capability(&self, state: &CognitiveState, substrate: &Substrate) -> Vec<f64> {
        let p_eff = substrate.effective_power();
        let caps = state.capability_array();
        self.tasks
            .iter()
            .zip(&self.demand)
            .map(|(task, &dem)| {
                self.capability
                    .phi(task, &caps)
                    .min(self.ceiling.g(p_eff, dem))
            })
            .collect()
    }

    /// SI_global(t) = ∫_Ω C_réel dμ ≈ Σ_i w_i · C_réel(x_i).
    pub fn si_global(&self, state: &CognitiveState, substrate: &Substrate) -> f64 {
        let c = self.real_capability(state, substrate);
        dot(&self.weights, &c)
    }

    /// SI_global **et son erreur-type de Monte-Carlo** (mean, stderr).
    ///
    /// `SI_global` est une moyenne pondérée sur un échantillon fini de Ω : son
    /// estimation comporte un bruit d'échantillonnage. On l'estime par la
    /// variance pondérée rapportée à la **taille effective d'échantillon de
    /// Kish** `n_eff = 1 / Σ w_i²` :
    ///   `stderr = sqrt( (Σ w_i (c_i − SI)²) · (Σ w_i²) )`.
    /// Sert à rendre le garde-fou ε **adaptatif** (§4) : on ne pénalise pas une
    /// variation inférieure au bruit d'échantillonnage.
    pub fn si_global_stats(&self, state: &CognitiveState, substrate: &Substrate) -> (f64, f64) {
        let c = self.real_capability(state, substrate);
        let mean = dot(&self.weights, &c);
        let mut wvar = 0.0;
        let mut sum_w2 = 0.0;
        for (&w, &ci) in self.weights.iter().zip(&c) {
            let d = ci - mean;
            wvar += w * d * d;
            sum_w2 += w * w;
        }
        let stderr = (wvar * sum_w2).max(0.0).sqrt();
        (mean, stderr)
    }

    /// Diagnostic : la compétence est-elle bridée par le cognitif (Φ) ou par
    /// le substrat (g) ?
    pub fn bottleneck(&self, state: &CognitiveState, substrate: &Substrate) -> Bottleneck {
        let p_eff = substrate.effective_power();
        let caps = state.capability_array();
        let mut frac_substrate = 0.0;
        let mut mean_phi = 0.0;
        let mut mean_g = 0.0;
        for ((task, &dem), &w) in self.tasks.iter().zip(&self.demand).zip(&self.weights) {
            let p = self.capability.phi(task, &caps);
            let gg = self.ceiling.g(p_eff, dem);
            if gg < p {
                frac_substrate += w;
            }
            mean_phi += w * p;
            mean_g += w * gg;
        }
        Bottleneck {
            frac_limited_by_substrate: frac_substrate,
            frac_limited_by_cognition: 1.0 - frac_substrate,
            mean_phi,
            mean_g,
        }
    }
}

/// Diagnostic de goulot d'étranglement (cognitif vs substrat).
#[derive(Clone, Copy, Debug)]
pub struct Bottleneck {
    pub frac_limited_by_substrate: f64,
    pub frac_limited_by_cognition: f64,
    pub mean_phi: f64,
    pub mean_g: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Dims;

    #[test]
    fn si_in_unit_interval() {
        let mut rng = Rng::new(5);
        let surf = IntelligenceSurface::sample(256, &mut rng);
        let state = CognitiveState::random(Dims::uniform(4), &mut rng, 0.3);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let si = surf.si_global(&state, &sub);
        assert!((0.0..=1.0).contains(&si), "SI = {si}");
    }

    #[test]
    fn higher_state_higher_si() {
        let mut rng = Rng::new(9);
        let surf = IntelligenceSurface::sample(512, &mut rng);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let low = CognitiveState::from_vector(&[0.1; 24], Dims::uniform(4));
        let high = CognitiveState::from_vector(&[0.9; 24], Dims::uniform(4));
        assert!(surf.si_global(&high, &sub) >= surf.si_global(&low, &sub));
    }

    #[test]
    fn custom_models_are_used() {
        // plafond nul ⇒ C_réel ≡ 0 ⇒ SI_global = 0, quel que soit l'état
        #[derive(Debug, Clone)]
        struct ZeroCeiling;
        impl CeilingModel for ZeroCeiling {
            fn g(&self, _p: f64, _d: f64) -> f64 {
                0.0
            }
            fn clone_box(&self) -> Box<dyn CeilingModel> {
                Box::new(self.clone())
            }
        }
        let mut rng = Rng::new(1);
        let surf = IntelligenceSurface::sample_with(
            128,
            &mut rng,
            Box::new(SigmoidCapability::default()),
            Box::new(ZeroCeiling),
        );
        let state = CognitiveState::from_vector(&[0.9; 24], Dims::uniform(4));
        let sub = Substrate::default_with(4, 4, &mut rng);
        assert!(surf.si_global(&state, &sub).abs() < 1e-12);
    }
}
