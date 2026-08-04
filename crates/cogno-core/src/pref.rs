//! Terme d'apprentissage **pairwise des préférences** (contrat §4).
//!
//! ```text
//! Δ_φ,ref = [log π_φ(y+|x) − log π_φ(y−|x)]
//!           − [log π_ref(y+|x) − log π_ref(y−|x)]
//!
//! J_pref(φ) = E_{(x,y+,y−)~D_pref} [ log σ( α·Δ_φ,ref ) ]
//! ```
//!
//! Le calcul utilise les **log-probabilités** — il est interdit de reconstruire
//! le ratio par division directe de probabilités (instabilité numérique,
//! interdiction §18).

use crate::error::{CognoError, CognoResult};
use crate::numeric::{CompensatedSum, FiniteScalar};

/// Paire de préférence `(x, y+, y−)` avec log-probs du modèle et de référence.
#[derive(Debug, Clone, Copy)]
pub struct PreferencePair {
    /// contexte `x`.
    pub context: &'static [u8],
    /// log π_φ(y+ | x) — sortie préférée/acceptée/corrigée.
    pub log_prob_policy_positive: f64,
    /// log π_φ(y− | x) — sortie rejetée/originale.
    pub log_prob_policy_negative: f64,
    /// log π_ref(y+ | x) — modèle de référence figé.
    pub log_prob_ref_positive: f64,
    /// log π_ref(y− | x).
    pub log_prob_ref_negative: f64,
}

/// Sigmoïde **stable** : `σ(z) = 1/(1+e^{−z})`, sans débordement pour `|z|`
/// grand (retourne 0/1 aux extrêmes).
#[inline]
pub fn sigmoid_stable(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Calcule `J_pref` sur un batch de paires, avec l'intensité `α > 0`.
///
/// Le log-ratio `Δ` est calculé **en log-espace** : les log-probs sont
/// soustraites directement, jamais converties en probabilités.
pub fn compute_preference_objective(
    pairs: &[PreferencePair],
    alpha: f64,
) -> CognoResult<FiniteScalar> {
    if alpha <= 0.0 || !alpha.is_finite() {
        return Err(CognoError::InvalidInput("α > 0"));
    }
    let mut sum = CompensatedSum::new();
    for p in pairs {
        for (name, v) in [
            ("log_prob_policy_positive", p.log_prob_policy_positive),
            ("log_prob_policy_negative", p.log_prob_policy_negative),
            ("log_prob_ref_positive", p.log_prob_ref_positive),
            ("log_prob_ref_negative", p.log_prob_ref_negative),
        ] {
            FiniteScalar::try_new(v).map_err(|_| CognoError::NonFinite(name))?;
        }
        // Δ en log-espace (pas de division)
        let delta = (p.log_prob_policy_positive - p.log_prob_policy_negative)
            - (p.log_prob_ref_positive - p.log_prob_ref_negative);
        let term = sigmoid_stable(alpha * delta).ln();
        FiniteScalar::try_new(term).map_err(|_| CognoError::NonFinite("sigmoid log"))?;
        sum.add(term);
    }
    let n = pairs.len() as f64;
    let mean = if n > 0.0 { sum.finish() / n } else { 0.0 };
    FiniteScalar::try_new(mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cas analytique : préférence clairement favorable.
    ///
    /// Modèle : `log π_φ(y+|x) = ln 4`, `log π_φ(y−|x) = 0` (ratio 4 en
    /// faveur de y+). Référence **indifférente** : `ref_pos = ref_neg = 0`.
    /// `Δ = (ln4 − 0) − (0 − 0) = ln 4 ≈ 1.3863`. Avec `α = 1` :
    /// `log σ(Δ) = log(4/5) = ln 4 − ln 5 ≈ −0.2231`.
    #[test]
    fn analytic_pairwise_log_sigmoid() {
        let p = PreferencePair {
            context: b"x",
            log_prob_policy_positive: 4.0_f64.ln(),
            log_prob_policy_negative: 0.0,
            log_prob_ref_positive: 0.0, // référence indifférente
            log_prob_ref_negative: 0.0,
        };
        let j = compute_preference_objective(&[p], 1.0).unwrap();
        let expected = (4.0_f64 / 5.0).ln();
        assert!((j.value() - expected).abs() < 1e-12, "j={} expected={}", j.value(), expected);
    }

    #[test]
    fn delta_is_logspace_not_division() {
        // ratio énorme (e^40) — la division directe déborderait ; en
        // log-espace, σ(40) ≈ 1 → log σ ≈ 0 sans infini
        let p = PreferencePair {
            context: b"x",
            log_prob_policy_positive: 40.0,
            log_prob_policy_negative: 0.0,
            log_prob_ref_positive: 0.0, // référence indifférente
            log_prob_ref_negative: 0.0,
        };
        let j = compute_preference_objective(&[p], 1.0).unwrap();
        assert!(j.value().is_finite());
        assert!((j.value() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn indifferent_preference_gives_log_half() {
        // Δ = 0 → σ(0) = 0.5 → log σ = ln 0.5 ≈ −0.6931
        let p = PreferencePair {
            context: b"x",
            log_prob_policy_positive: -1.0,
            log_prob_policy_negative: -1.0,
            log_prob_ref_positive: -1.0,
            log_prob_ref_negative: -1.0,
        };
        let j = compute_preference_objective(&[p], 1.0).unwrap();
        assert!((j.value() - 0.5_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn rejects_non_positive_alpha() {
        let p = PreferencePair {
            context: b"x",
            log_prob_policy_positive: 0.0,
            log_prob_policy_negative: 0.0,
            log_prob_ref_positive: 0.0,
            log_prob_ref_negative: 0.0,
        };
        assert!(compute_preference_objective(&[p], 0.0).is_err());
    }
}
