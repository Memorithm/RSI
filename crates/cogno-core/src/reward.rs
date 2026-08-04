//! Récompense neuro-symbolique **décomposée** `R̃_NS(x,y)` (contrat §3).
//!
//! ```text
//! R̃_NS = R_formal + q_e·R_feedback + R_tests + R_heldout
//!         − P_regression − P_complexity − κ_u·U
//! ```
//!
//! Toutes les composantes sont **observables séparément** dans les rapports —
//! le système n'expose pas seulement la somme finale (interdiction §18).

use crate::error::CognoResult;
use crate::numeric::{CompensatedSum, FiniteScalar};

/// Récompense neuro-symbolique décomposée.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardBreakdown {
    pub formal: f64,
    pub feedback: f64,
    pub feedback_quality: f64, // q_e ∈ [0,1]
    pub tests: f64,
    pub heldout: f64,
    pub regression_penalty: f64,
    pub complexity_penalty: f64,
    pub uncertainty: f64, // U ∈ [0,1]
    pub uncertainty_weight: f64, // κ_u ≥ 0
    /// Somme finale (compensée, dans l'ordre des termes).
    pub total: f64,
}

impl RewardBreakdown {
    /// Calcule la récompense décomposée.
    ///
    /// # Validations
    /// - `feedback_quality ∈ [0,1]`, `uncertainty ∈ [0,1]`, `uncertainty_weight ≥ 0` ;
    /// - les pénalités sont soustraites (le signe de `regression_penalty` et
    ///   `complexity_penalty` est positif, on les soustrait) ;
    /// - toutes les entrées doivent être finies.
    // Les 9 composantes de l'équation §3 sont exposées en paramètres nommés
    // (contrat d'API) — `too_many_arguments` est volontairement accepté.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        formal: f64,
        feedback: f64,
        feedback_quality: f64,
        tests: f64,
        heldout: f64,
        regression_penalty: f64,
        complexity_penalty: f64,
        uncertainty: f64,
        uncertainty_weight: f64,
    ) -> CognoResult<RewardBreakdown> {
        for (name, v) in [
            ("formal", formal),
            ("feedback", feedback),
            ("tests", tests),
            ("heldout", heldout),
            ("regression_penalty", regression_penalty),
            ("complexity_penalty", complexity_penalty),
        ] {
            FiniteScalar::try_new(v).map_err(|_| crate::error::CognoError::NonFinite(name))?;
        }
        if !(0.0..=1.0).contains(&feedback_quality) {
            return Err(crate::error::CognoError::InvalidInput("feedback_quality ∈ [0,1]"));
        }
        if !(0.0..=1.0).contains(&uncertainty) {
            return Err(crate::error::CognoError::InvalidInput("uncertainty ∈ [0,1]"));
        }
        if uncertainty_weight < 0.0 {
            return Err(crate::error::CognoError::InvalidInput("κ_u ≥ 0"));
        }

        // somme compensée, ordre exact des termes de l'équation
        let mut s = CompensatedSum::new();
        s.add(formal);
        s.add(feedback_quality * feedback);
        s.add(tests);
        s.add(heldout);
        s.add(-regression_penalty);
        s.add(-complexity_penalty);
        s.add(-uncertainty_weight * uncertainty);

        Ok(RewardBreakdown {
            formal,
            feedback,
            feedback_quality,
            tests,
            heldout,
            regression_penalty,
            complexity_penalty,
            uncertainty,
            uncertainty_weight,
            total: s.finish(),
        })
    }

    /// Somme compensée (ré-exposée pour l'oracle — ordre identique).
    pub fn total_compensated(&self) -> f64 {
        self.total
    }
}

/// Alias de `RewardBreakdown::compute` — fonction libre pour l'export racine.
// Les 9 composantes de l'équation §3 sont exposées en paramètres nommés
// (contrat d'API) — `too_many_arguments` est volontairement accepté.
#[allow(clippy::too_many_arguments)]
pub fn compute_reward_breakdown(
    formal: f64,
    feedback: f64,
    feedback_quality: f64,
    tests: f64,
    heldout: f64,
    regression_penalty: f64,
    complexity_penalty: f64,
    uncertainty: f64,
    uncertainty_weight: f64,
) -> CognoResult<RewardBreakdown> {
    RewardBreakdown::compute(
        formal,
        feedback,
        feedback_quality,
        tests,
        heldout,
        regression_penalty,
        complexity_penalty,
        uncertainty,
        uncertainty_weight,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decomposition_is_exact_and_ordered() {
        let r = RewardBreakdown::compute(
            1.0, // formal
            0.5, // feedback
            1.0, // q_e
            0.25, // tests
            0.25, // heldout
            0.1, // régression
            0.05, // complexité
            0.5, // U
            0.2, // κ_u
        )
        .unwrap();
        // total = 1.0 + 0.5 + 0.25 + 0.25 − 0.1 − 0.05 − 0.1 = 1.75
        assert!((r.total - 1.75).abs() < 1e-12, "total={}", r.total);
        // chaque composante observable
        assert_eq!(r.formal, 1.0);
        assert_eq!(r.feedback_quality, 1.0);
        assert_eq!(r.regression_penalty, 0.1);
        assert_eq!(r.uncertainty_weight, 0.2);
    }

    #[test]
    fn rejects_invalid_domains() {
        assert!(RewardBreakdown::compute(1.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).is_err());
        assert!(RewardBreakdown::compute(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0).is_err());
        assert!(RewardBreakdown::compute(f64::NAN, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).is_err());
    }
}
