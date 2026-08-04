//! Coût des ressources (contrat §8).
//!
//! ```text
//! L_resource = E[ ρ_m·C̄_mem + ρ_t·C̄_lat + ρ_c·C̄_ctx ]
//! ```
//!
//! - coûts normalisés `C̄ = C/B` (unités explicites) ;
//! - `ρ_m, ρ_t, ρ_c ≥ 0` ;
//! - aucun coût inventé ou remplacé par une constante (interdiction §18) ;
//! - favorise les solutions efficaces **parmi les admissibles** — ne remplace
//!   pas les limites dures de `cogno-core`.

use crate::budget::{NormalizedCosts, ResourceBudget};
use crate::error::{CognoError, CognoResult};
use crate::numeric::{CompensatedSum, NonNegativeFinite};

pub use crate::budget::ResourceWeights;

/// Coûts de ressources mesurés d'une sortie (unités explicites).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceCosts {
    /// Coût mémoire, en octets.
    pub mem_bytes: usize,
    /// Coût latence, en millisecondes.
    pub lat_ms: usize,
    /// Coût contexte, en tokens.
    pub ctx_tokens: usize,
}

impl ResourceCosts {
    pub fn new(mem_bytes: usize, lat_ms: usize, ctx_tokens: usize) -> Self {
        ResourceCosts {
            mem_bytes,
            lat_ms,
            ctx_tokens,
        }
    }

    /// Calcule la perte de ressources pour une sortie (coûts normalisés).
    ///
    /// Retourne `(perte, coûts normalisés)` — les coûts normalisés sont
    /// exposés pour le rapport (observabilité).
    pub fn compute_loss(
        &self,
        weights: &ResourceWeights,
        budget: &ResourceBudget,
    ) -> CognoResult<(NonNegativeFinite, NormalizedCosts)> {
        let norm = NormalizedCosts::try_new(self.mem_bytes, self.lat_ms, self.ctx_tokens, budget)?;
        let mut s = CompensatedSum::new();
        s.add(weights.mem.value() * norm.mem);
        s.add(weights.lat.value() * norm.lat);
        s.add(weights.ctx.value() * norm.ctx);
        let loss = s.finish();
        if !loss.is_finite() {
            return Err(CognoError::NonFinite("resource loss"));
        }
        Ok((NonNegativeFinite::try_new(loss)?, norm))
    }
}

/// Perte de ressources moyenne sur un batch de coûts.
pub fn compute_resource_loss_batch(
    costs: &[ResourceCosts],
    weights: &ResourceWeights,
    budget: &ResourceBudget,
) -> CognoResult<NonNegativeFinite> {
    let mut sum = CompensatedSum::new();
    for c in costs {
        let (loss, _) = c.compute_loss(weights, budget)?;
        sum.add(loss.value());
    }
    let n = costs.len().max(1) as f64;
    NonNegativeFinite::try_new(sum.finish() / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cas analytique ressource : coûts nuls → perte 0.
    #[test]
    fn zero_costs_give_zero_loss() {
        let b = ResourceBudget::new(100, 100, 100);
        let w = ResourceWeights::try_new(0.5, 0.5, 0.5).unwrap();
        let c = ResourceCosts::new(0, 0, 0);
        let (loss, norm) = c.compute_loss(&w, &b).unwrap();
        assert_eq!(loss.value(), 0.0);
        assert_eq!(norm.mem, 0.0);
    }

    /// Cas analytique : coûts à mi-budget, poids unitaires → perte 1.5
    /// (0.5 + 0.5 + 0.5).
    #[test]
    fn half_budget_unit_weights() {
        let b = ResourceBudget::new(100, 100, 100);
        let w = ResourceWeights::try_new(1.0, 1.0, 1.0).unwrap();
        let c = ResourceCosts::new(50, 50, 50);
        let (loss, _) = c.compute_loss(&w, &b).unwrap();
        assert!((loss.value() - 1.5).abs() < 1e-12);
    }

    /// Cas analytique : coût au budget exact → perte = somme des poids.
    #[test]
    fn at_budget_is_sum_of_weights() {
        let b = ResourceBudget::new(100, 200, 300);
        let w = ResourceWeights::try_new(0.1, 0.2, 0.3).unwrap();
        let c = ResourceCosts::new(100, 200, 300);
        let (loss, norm) = c.compute_loss(&w, &b).unwrap();
        assert!((norm.mem - 1.0).abs() < 1e-12);
        assert!((loss.value() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn zero_budget_is_rejected() {
        let b = ResourceBudget::new(0, 100, 100);
        let w = ResourceWeights::default();
        let c = ResourceCosts::new(0, 0, 0);
        assert!(c.compute_loss(&w, &b).is_err());
    }
}
