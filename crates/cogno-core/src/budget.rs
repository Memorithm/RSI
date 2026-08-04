//! Budgets durs des ressources (contrat §2 : `B_mem`, `B_lat`, `B_ctx`).
//!
//! Les budgets sont des **bornes dures** appliquées par `cogno-core` dans le
//! gate d'admissibilité `F(x)`. Ils ne sont jamais des pénalités compensables.

use crate::error::{CognoError, CognoResult};
use crate::numeric::NonNegativeFinite;

/// Budgets durs de ressources (unités explicites).
///
/// Chaque budget a une **unité documentée** :
/// - `mem` : octets ;
/// - `lat` : millisecondes ;
/// - `ctx` : tokens de contexte.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceBudget {
    /// Budget mémoire dur, en octets.
    pub mem_bytes: usize,
    /// Budget latence dur, en millisecondes.
    pub lat_ms: usize,
    /// Budget contexte dur, en tokens.
    pub ctx_tokens: usize,
}

impl ResourceBudget {
    pub fn new(mem_bytes: usize, lat_ms: usize, ctx_tokens: usize) -> Self {
        ResourceBudget {
            mem_bytes,
            lat_ms,
            ctx_tokens,
        }
    }

    /// Vérifie qu'une sortie reste dans tous les budgets durs.
    #[inline]
    pub fn permits(&self, mem_bytes: usize, lat_ms: usize, ctx_tokens: usize) -> bool {
        mem_bytes <= self.mem_bytes && lat_ms <= self.lat_ms && ctx_tokens <= self.ctx_tokens
    }
}

impl Default for ResourceBudget {
    fn default() -> Self {
        ResourceBudget {
            mem_bytes: 8 * 1024 * 1024,
            lat_ms: 500,
            ctx_tokens: 4096,
        }
    }
}

/// Coûts normalisés `C̄_mem, C̄_lat, C̄_ctx` (contrat §8).
///
/// Chaque coût normalisé = `coût / budget`, dans `[0, 1]` quand la sortie est
/// admissible (les dépassements sont rejetés par le gate avant le calcul de
/// perte ; si appelé sur un dépassement, la valeur > 1 est retournée telle
/// quelle pour le diagnostic, mais ne participe jamais à l'adoption).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedCosts {
    pub mem: f64,
    pub lat: f64,
    pub ctx: f64,
}

impl NormalizedCosts {
    /// Calcule les coûts normalisés. Division par zéro → erreur (budget nul).
    pub fn try_new(mem_bytes: usize, lat_ms: usize, ctx_tokens: usize, b: &ResourceBudget) -> CognoResult<Self> {
        if b.mem_bytes == 0 || b.lat_ms == 0 || b.ctx_tokens == 0 {
            return Err(CognoError::InvalidInput("budget nul"));
        }
        Ok(NormalizedCosts {
            mem: mem_bytes as f64 / b.mem_bytes as f64,
            lat: lat_ms as f64 / b.lat_ms as f64,
            ctx: ctx_tokens as f64 / b.ctx_tokens as f64,
        })
    }
}

/// Poids de la perte de ressources `ρ_m, ρ_t, ρ_c` (contrat §8, non négatifs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceWeights {
    pub mem: NonNegativeFinite,
    pub lat: NonNegativeFinite,
    pub ctx: NonNegativeFinite,
}

impl ResourceWeights {
    pub fn try_new(mem: f64, lat: f64, ctx: f64) -> CognoResult<Self> {
        Ok(ResourceWeights {
            mem: NonNegativeFinite::try_new(mem)?,
            lat: NonNegativeFinite::try_new(lat)?,
            ctx: NonNegativeFinite::try_new(ctx)?,
        })
    }
}

impl Default for ResourceWeights {
    fn default() -> Self {
        ResourceWeights {
            mem: NonNegativeFinite::try_new(0.1).expect("0.1 ≥ 0"),
            lat: NonNegativeFinite::try_new(0.1).expect("0.1 ≥ 0"),
            ctx: NonNegativeFinite::try_new(0.1).expect("0.1 ≥ 0"),
        }
    }
}
