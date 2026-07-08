//! §1 (extension) — **Ω concret** : un jeu de tâches *réelles et nommées*.
//!
//! Là où [`crate::surface::IntelligenceSurface::sample`] tire Ω ~ μ de façon
//! synthétique (profils Dirichlet aléatoires), ce module fournit un **banc de
//! tâches curé** : des contextes interprétables (analyse de données, synthèse
//! de code, raisonnement long-contexte…), chacun avec un profil explicite de
//! besoins sur `S = (D, M, R, A, C, V)`, une exigence calculatoire `demand` et
//! un poids `μ`.
//!
//! Il **ne duplique rien** : il se branche sur les traits existants
//! [`CapabilityModel`] / [`CeilingModel`] via
//! [`IntelligenceSurface::from_tasks`]. Son intérêt est de **voir `Φ_x` et
//! `g_x` s'opposer** sur des tâches concrètes (cf. [`TaskReport`]).

use crate::state::CognitiveState;
use crate::substrate::Substrate;
use crate::surface::{
    CapabilityModel, CeilingModel, IntelligenceSurface, PowerCeiling, SigmoidCapability,
};

/// Une tâche **concrète** de Ω : profil de besoins + exigence + importance.
#[derive(Clone, Copy, Debug)]
pub struct NamedTask {
    /// Nom interprétable de la tâche.
    pub name: &'static str,
    /// Profil de besoins normalisé sur `(D, M, R, A, C, V)` (∑ ≈ 1).
    pub profile: [f64; 6],
    /// Exigence calculatoire ∈ `[0, 1]` (combien la tâche sollicite le substrat).
    pub demand: f64,
    /// Importance/fréquence relative `μ` (normalisée à la construction de Ω).
    pub weight: f64,
}

/// Profil brut → normalisé (∑ = 1) pour rester comparable aux tirages Dirichlet.
const fn raw(name: &'static str, p: [f64; 6], demand: f64, weight: f64) -> NamedTask {
    NamedTask { name, profile: p, demand, weight }
}

/// Banc de tâches standard : sept contextes typiques d'un agent (style
/// Memorithm). Les profils mettent en avant les composantes dominantes ;
/// `demand` reflète l'intensité calculatoire (long-contexte/perception = lourd).
///
/// Ordre des composantes du profil : `[D, M, R, A, C, V]`.
pub fn standard_suite() -> Vec<NamedTask> {
    vec![
        // D    M    R    A    C    V        demand weight
        raw("analyse_donnees",      [0.30, 0.10, 0.30, 0.05, 0.20, 0.05], 0.70, 1.0),
        raw("synthese_code",        [0.10, 0.30, 0.35, 0.15, 0.05, 0.05], 0.85, 1.0),
        raw("raisonnement_long",    [0.10, 0.10, 0.35, 0.05, 0.35, 0.05], 0.95, 0.9),
        raw("planification",        [0.05, 0.10, 0.35, 0.25, 0.10, 0.15], 0.55, 0.8),
        raw("perception_ingest",    [0.35, 0.30, 0.10, 0.05, 0.15, 0.05], 0.90, 0.7),
        raw("usage_outils",         [0.10, 0.10, 0.25, 0.35, 0.05, 0.15], 0.45, 0.6),
        raw("revue_alignement",     [0.10, 0.05, 0.30, 0.10, 0.05, 0.40], 0.25, 0.5),
    ]
}

/// Construit une [`IntelligenceSurface`] à partir d'un banc de tâches, avec les
/// modèles `Φ`/`g` par défaut (sigmoïde + loi de puissance).
pub fn surface(tasks: &[NamedTask]) -> IntelligenceSurface {
    surface_with(
        tasks,
        Box::new(SigmoidCapability::default()),
        Box::new(PowerCeiling),
    )
}

/// Idem avec des modèles `Φ`/`g` personnalisés.
pub fn surface_with(
    tasks: &[NamedTask],
    capability: Box<dyn CapabilityModel>,
    ceiling: Box<dyn CeilingModel>,
) -> IntelligenceSurface {
    let profiles = tasks.iter().map(|t| t.profile).collect();
    let demand = tasks.iter().map(|t| t.demand).collect();
    let weights = tasks.iter().map(|t| t.weight).collect();
    IntelligenceSurface::from_tasks(profiles, demand, weights, capability, ceiling)
}

/// Diagnostic d'une tâche : qui bride `C_réel` — le cognitif ou le substrat ?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limiter {
    /// `Φ_x < g_x` : c'est la **compétence cognitive** qui bride.
    Cognition,
    /// `g_x < Φ_x` : c'est le **substrat physique** qui bride.
    Substrate,
}

/// Rapport par tâche : `Φ_x`, `g_x`, `C_réel` et le facteur limitant.
#[derive(Clone, Copy, Debug)]
pub struct TaskReport {
    pub name: &'static str,
    pub phi: f64,
    pub g: f64,
    pub c_real: f64,
    pub limiter: Limiter,
}

/// Évalue un banc de tâches contre un état et un substrat donnés, et renvoie un
/// rapport par tâche. Réutilise [`IntelligenceSurface::task_breakdown`].
pub fn report(
    tasks: &[NamedTask],
    state: &CognitiveState,
    substrate: &Substrate,
) -> Vec<TaskReport> {
    let surf = surface(tasks);
    let breakdown = surf.task_breakdown(state, substrate);
    tasks
        .iter()
        .zip(breakdown)
        .map(|(t, (phi, g, c_real))| TaskReport {
            name: t.name,
            phi,
            g,
            c_real,
            limiter: if g < phi { Limiter::Substrate } else { Limiter::Cognition },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::state::Dims;

    #[test]
    fn profiles_are_normalized() {
        for t in standard_suite() {
            let s: f64 = t.profile.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "{} profil somme {}", t.name, s);
            assert!((0.0..=1.0).contains(&t.demand));
        }
    }

    #[test]
    fn si_global_in_unit_interval() {
        let mut rng = Rng::new(11);
        let surf = surface(&standard_suite());
        let state = CognitiveState::random(Dims::uniform(4), &mut rng, 0.4);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let si = surf.si_global(&state, &sub);
        assert!((0.0..=1.0).contains(&si), "SI = {si}");
    }

    #[test]
    fn report_covers_every_task() {
        let mut rng = Rng::new(3);
        let suite = standard_suite();
        let state = CognitiveState::from_vector(&[0.6; 24], Dims::uniform(4));
        let sub = Substrate::default_with(4, 4, &mut rng);
        let r = report(&suite, &state, &sub);
        assert_eq!(r.len(), suite.len());
        for tr in &r {
            assert!((tr.c_real - tr.phi.min(tr.g)).abs() < 1e-12);
            assert!(tr.c_real <= tr.phi + 1e-12 && tr.c_real <= tr.g + 1e-12);
        }
    }

    #[test]
    fn strong_cognition_shifts_bottleneck_to_substrate() {
        // État cognitif très élevé + substrat faible ⇒ au moins une tâche doit
        // être bridée par le substrat (g < Φ).
        let mut rng = Rng::new(5);
        let suite = standard_suite();
        let strong = CognitiveState::from_vector(&[0.98; 24], Dims::uniform(4));
        let mut sub = Substrate::default_with(4, 4, &mut rng);
        // force un substrat faible en effaçant toute mesure et en réduisant O
        sub.set_measured_software_eff(None);
        for o in sub.o.iter_mut() {
            *o *= 0.05;
        }
        let r = report(&suite, &strong, &sub);
        assert!(
            r.iter().any(|tr| tr.limiter == Limiter::Substrate),
            "un substrat faible doit brider au moins une tâche"
        );
    }
}
