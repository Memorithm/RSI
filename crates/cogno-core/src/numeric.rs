//! Types numériques **validés** et sommation compensée (Kahan).
//!
//! Contrat §9 : tous les coefficients et valeurs de l'objectif sont représentés
//! par des types validés — fini, pas de NaN, pas d'infini, signe conforme,
//! bornes explicites.

use crate::error::{CognoError, CognoResult};

/// Scalaire **fini** : n'admet ni NaN ni ±infini.
///
/// Construit via [`FiniteScalar::try_new`] (fallible) ou [`FiniteScalar::ZERO`]
/// pour la valeur nulle. Toute opération qui produirait un non-fini échoue.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FiniteScalar(f64);

impl FiniteScalar {
    pub const ZERO: FiniteScalar = FiniteScalar(0.0);
    pub const ONE: FiniteScalar = FiniteScalar(1.0);

    /// Construit un scalaire fini. Rejette NaN et ±infini.
    pub fn try_new(v: f64) -> CognoResult<Self> {
        if v.is_finite() {
            Ok(FiniteScalar(v))
        } else {
            Err(CognoError::NonFinite("FiniteScalar"))
        }
    }

    #[inline]
    pub fn value(self) -> f64 {
        self.0
    }

    /// Addition contrôlée : rejette le débordement vers non-fini.
    pub fn checked_add(self, o: FiniteScalar) -> CognoResult<FiniteScalar> {
        let r = self.0 + o.0;
        FiniteScalar::try_new(r)
    }

    pub fn checked_sub(self, o: FiniteScalar) -> CognoResult<FiniteScalar> {
        FiniteScalar::try_new(self.0 - o.0)
    }

    pub fn checked_mul(self, o: FiniteScalar) -> CognoResult<FiniteScalar> {
        FiniteScalar::try_new(self.0 * o.0)
    }
}

impl std::ops::Neg for FiniteScalar {
    type Output = FiniteScalar;
    fn neg(self) -> FiniteScalar {
        FiniteScalar(-self.0)
    }
}

/// Scalaire **non négatif et fini** : `0 ≤ v < +∞`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NonNegativeFinite(f64);

impl NonNegativeFinite {
    pub const ZERO: NonNegativeFinite = NonNegativeFinite(0.0);

    /// Construit un scalaire non négatif fini. Rejette NaN, infini, négatif.
    pub fn try_new(v: f64) -> CognoResult<Self> {
        if v.is_finite() && v >= 0.0 {
            Ok(NonNegativeFinite(v))
        } else {
            Err(CognoError::NonNegativeViolation)
        }
    }

    #[inline]
    pub fn value(self) -> f64 {
        self.0
    }
}

/// Sommation compensée (Kahan) : réduit l'erreur d'arrondi sur de longues
/// sommes. Ordre de parcours = ordre du slice (déterminisme).
#[derive(Debug, Clone, Copy)]
pub struct CompensatedSum {
    sum: f64,
    comp: f64,
}

impl Default for CompensatedSum {
    fn default() -> Self {
        CompensatedSum {
            sum: 0.0,
            comp: 0.0,
        }
    }
}

impl CompensatedSum {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ajoute une valeur à la somme compensée.
    #[inline]
    pub fn add(&mut self, v: f64) {
        let y = v - self.comp;
        let t = self.sum + y;
        self.comp = (t - self.sum) - y;
        self.sum = t;
    }

    /// Termine la somme (compensation finale). Non-fini possible si entrée
    /// non finie — l'appelant valide via `FiniteScalar::try_new`.
    #[inline]
    pub fn finish(&self) -> f64 {
        self.sum
    }

    /// Somme compensée d'un slice, dans l'ordre du slice.
    pub fn of_slice(v: &[f64]) -> f64 {
        let mut s = CompensatedSum::new();
        for &x in v {
            s.add(x);
        }
        s.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_scalar_rejects_nan_and_inf() {
        assert!(FiniteScalar::try_new(f64::NAN).is_err());
        assert!(FiniteScalar::try_new(f64::INFINITY).is_err());
        assert!(FiniteScalar::try_new(f64::NEG_INFINITY).is_err());
        assert!(FiniteScalar::try_new(1.25).is_ok());
    }

    #[test]
    fn non_negative_rejects_negatives() {
        assert!(NonNegativeFinite::try_new(-0.1).is_err());
        assert!(NonNegativeFinite::try_new(0.0).is_ok());
        assert!(NonNegativeFinite::try_new(3.0).is_ok());
    }

    #[test]
    fn compensated_sum_matches_plain() {
        let v = [0.1, 0.2, 0.3, 0.4, 1e-17];
        let s = CompensatedSum::of_slice(&v);
        assert!((s - v.iter().sum::<f64>()).abs() < 1e-12);
    }
}
