//! Terme de **satisfaction symbolique différentiable** (contrat §5).
//!
//! ```text
//! J_sym(φ) = E[ Σ_{j=1..m} w_j · log( ε + s_j(x,y) ) ]
//! ```
//!
//! - `s_j ∈ [0,1]` : satisfaction de la règle souple `j` ;
//! - `w_j ≥ 0` : poids de la règle ;
//! - `ε > 0` : constante de stabilité.
//!
//! Les règles **dures** ne sont pas introduites ici — elles restent dans
//! `F(x)` (voir [`crate::admissible`]).
//!
//! ## Logique floue différentiable
//!
//! Sémantique initiale (déclarée et testée) :
//! ```text
//! a ∧ b = ab
//! a ∨ b = a + b − ab
//! ¬a    = 1 − a
//! a ⇒ b = 1 − a + ab
//! ```
//! Toute autre t-norme doit être ajoutée explicitement (nouveau variant de
//! [`SoftLogic`]) et documentée.

use crate::error::{CognoError, CognoResult};
use crate::numeric::{CompensatedSum, FiniteScalar};

/// Opérateur de logique floue différentiable, avec sémantique déclarée.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftLogic {
    /// `a ∧ b = a·b` (t-norme produit).
    And,
    /// `a ∨ b = a + b − a·b` (t-conorme probabiliste).
    Or,
    /// `¬a = 1 − a`.
    Not,
    /// `a ⇒ b = 1 − a + a·b` (implication de Łukasiewicz-produit).
    Implies,
}

impl SoftLogic {
    /// Applique l'opérateur. Tous les opérandes doivent être dans `[0,1]`.
    pub fn apply(&self, a: f64, b: Option<f64>) -> CognoResult<f64> {
        match self {
            SoftLogic::And => {
                let b = b.ok_or(CognoError::InvalidInput("And requiert 2 opérandes"))?;
                check_unit(a)?;
                check_unit(b)?;
                Ok(a * b)
            }
            SoftLogic::Or => {
                let b = b.ok_or(CognoError::InvalidInput("Or requiert 2 opérandes"))?;
                check_unit(a)?;
                check_unit(b)?;
                Ok(a + b - a * b)
            }
            SoftLogic::Not => {
                check_unit(a)?;
                Ok(1.0 - a)
            }
            SoftLogic::Implies => {
                let b = b.ok_or(CognoError::InvalidInput("Implies requiert 2 opérandes"))?;
                check_unit(a)?;
                check_unit(b)?;
                Ok(1.0 - a + a * b)
            }
        }
    }
}

#[inline]
fn check_unit(v: f64) -> CognoResult<()> {
    if (0.0..=1.0).contains(&v) {
        Ok(())
    } else {
        Err(CognoError::InvalidInput("opérande hors [0,1]"))
    }
}

/// Règle souple : un degré de satisfaction `s_j ∈ [0,1]` et un poids `w_j ≥ 0`.
#[derive(Debug, Clone, Copy)]
pub struct SoftRule {
    pub satisfaction: f64,
    pub weight: f64,
    /// Nom de la règle (traçabilité).
    pub name: &'static str,
}

impl SoftRule {
    pub fn new(satisfaction: f64, weight: f64, name: &'static str) -> CognoResult<Self> {
        check_unit(satisfaction)?;
        if weight < 0.0 {
            return Err(CognoError::InvalidInput("w_j ≥ 0"));
        }
        Ok(SoftRule {
            satisfaction,
            weight,
            name,
        })
    }
}

/// Calcule `J_sym` sur un ensemble de règles souples.
///
/// `ε > 0` (constante de stabilité) ; par défaut `1e-6`. Contribution par
/// règle : `w_j · log(ε + s_j)` — une règle totalement violée (`s=0`) donne
/// `w·log ε` (fini, très négatif), jamais `−∞`.
pub fn compute_symbolic_objective(
    rules: &[SoftRule],
    epsilon: f64,
) -> CognoResult<FiniteScalar> {
    if epsilon <= 0.0 || !epsilon.is_finite() {
        return Err(CognoError::InvalidInput("ε > 0"));
    }
    let mut sum = CompensatedSum::new();
    for r in rules {
        check_unit(r.satisfaction)?;
        if r.weight < 0.0 {
            return Err(CognoError::InvalidInput("w_j ≥ 0"));
        }
        let term = r.weight * (epsilon + r.satisfaction).ln();
        FiniteScalar::try_new(term).map_err(|_| CognoError::NonFinite("softlogic term"))?;
        sum.add(term);
    }
    FiniteScalar::try_new(sum.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softlogic_semantics_are_declared_and_tested() {
        // a ∧ b = ab
        assert!((SoftLogic::And.apply(0.5, Some(0.5)).unwrap() - 0.25).abs() < 1e-12);
        // a ∨ b = a + b − ab
        assert!((SoftLogic::Or.apply(0.5, Some(0.5)).unwrap() - 0.75).abs() < 1e-12);
        // ¬a = 1 − a
        assert!((SoftLogic::Not.apply(0.25, None).unwrap() - 0.75).abs() < 1e-12);
        // a ⇒ b = 1 − a + ab
        assert!((SoftLogic::Implies.apply(0.5, Some(0.5)).unwrap() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn fully_satisfied_rule_gives_weight_log_eps_plus_one() {
        let rule = SoftRule::new(1.0, 2.0, "r").unwrap();
        let j = compute_symbolic_objective(&[rule], 1e-6).unwrap();
        let expected = 2.0 * (1.0 + 1e-6_f64).ln();
        assert!((j.value() - expected).abs() < 1e-9);
    }

    #[test]
    fn fully_violated_rule_is_finite() {
        let rule = SoftRule::new(0.0, 1.0, "r").unwrap();
        let j = compute_symbolic_objective(&[rule], 1e-6).unwrap();
        assert!(j.value().is_finite());
        assert!((j.value() - (1e-6_f64).ln()).abs() < 1e-9);
    }

    #[test]
    fn rejects_out_of_unit_satisfaction() {
        assert!(SoftRule::new(1.5, 1.0, "r").is_err());
        assert!(SoftRule::new(0.5, -1.0, "r").is_err());
    }
}
