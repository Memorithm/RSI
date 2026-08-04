//! Calibration de la confiance (contrat §7).
//!
//! Perte initiale : **Brier** `L_cal = E[(p_φ(z=1|x,y) − z)²]`.
//!
//! Métriques produites séparément :
//! - Brier score ;
//! - Expected Calibration Error (ECE) ;
//! - courbe fiabilité/confiance (bins) ;
//! - accuracy par intervalle de confiance ;
//! - taux d'abstention ;
//! - qualité après abstention.

use crate::error::{CognoError, CognoResult};
use crate::numeric::{CompensatedSum, NonNegativeFinite};

/// Paire (confiance prédite, vérité observée) pour la calibration.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationPoint {
    /// `p_φ(z=1|x,y) ∈ [0,1]`.
    pub predicted: f64,
    /// vérité `z ∈ {0,1}`.
    pub observed: u8,
}

/// Métriques de calibration complètes (jamais seulement le Brier).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CalibrationMetrics {
    pub brier: f64,
    pub ece: f64,
    pub abstention_rate: f64,
    pub accuracy_after_abstention: f64,
    /// (borne inf du bin, confiance moyenne, accuracy du bin) — courbe.
    pub reliability_curve: Vec<(f64, f64, f64)>,
    /// nb de points.
    pub n: usize,
}

/// Calcule la perte de Brier et toutes les métriques de calibration.
///
/// - `n_bins` : nombre de bins de la courbe fiabilité/confiance (défaut 10) ;
/// - `abstain_below` : seuil de confiance sous lequel on s'abstient (pour le
///   taux d'abstention et la qualité après abstention) ;
/// - les `predicted` doivent être dans `[0,1]`, les `observed` dans `{0,1}`.
pub fn compute_brier_calibration(
    points: &[CalibrationPoint],
    n_bins: usize,
    abstain_below: f64,
) -> CognoResult<(NonNegativeFinite, CalibrationMetrics)> {
    if n_bins == 0 {
        return Err(CognoError::InvalidInput("n_bins > 0"));
    }
    if !(0.0..1.0).contains(&abstain_below) {
        return Err(CognoError::InvalidInput("abstain_below ∈ [0,1)"));
    }
    let mut brier_sum = CompensatedSum::new();
    let mut abstained = 0usize;
    let mut kept = 0usize;
    let mut kept_correct = 0usize;
    // bins pour ECE + courbe
    let mut bin_conf: Vec<f64> = vec![0.0; n_bins];
    let mut bin_acc: Vec<f64> = vec![0.0; n_bins];
    let mut bin_n: Vec<usize> = vec![0; n_bins];

    for p in points {
        if !(0.0..=1.0).contains(&p.predicted) {
            return Err(CognoError::InvalidInput("predicted ∈ [0,1]"));
        }
        if p.observed > 1 {
            return Err(CognoError::InvalidInput("observed ∈ {0,1}"));
        }
        let err = p.predicted - p.observed as f64;
        brier_sum.add(err * err);
        let bin = ((p.predicted * n_bins as f64) as usize).min(n_bins - 1);
        bin_conf[bin] += p.predicted;
        bin_acc[bin] += p.observed as f64;
        bin_n[bin] += 1;
        if p.predicted < abstain_below {
            abstained += 1;
        } else {
            kept += 1;
            if p.observed == 1 {
                kept_correct += 1;
            }
        }
    }

    let n = points.len().max(1) as f64;
    let brier = brier_sum.finish() / n;

    // ECE : somme pondérée |conf moyenne − acc| par bin
    let mut ece_sum = CompensatedSum::new();
    let mut curve = Vec::new();
    for b in 0..n_bins {
        if bin_n[b] > 0 {
            let conf = bin_conf[b] / bin_n[b] as f64;
            let acc = bin_acc[b] / bin_n[b] as f64;
            let w = bin_n[b] as f64 / n;
            ece_sum.add(w * (conf - acc).abs());
            curve.push((b as f64 / n_bins as f64, conf, acc));
        }
    }

    Ok((
        NonNegativeFinite::try_new(brier)?,
        CalibrationMetrics {
            brier,
            ece: ece_sum.finish(),
            abstention_rate: abstained as f64 / n,
            accuracy_after_abstention: if kept > 0 {
                kept_correct as f64 / kept as f64
            } else {
                f64::NAN
            },
            reliability_curve: curve,
            n: points.len(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cas analytique Brier : parfaitement calibré, deux points.
    /// `(p=1,z=1)` → err 0 ; `(p=0,z=0)` → err 0 ⇒ Brier = 0.
    #[test]
    fn analytic_brier_perfect() {
        let pts = [
            CalibrationPoint { predicted: 1.0, observed: 1 },
            CalibrationPoint { predicted: 0.0, observed: 0 },
        ];
        let (loss, m) = compute_brier_calibration(&pts, 5, 0.5).unwrap();
        assert_eq!(loss.value(), 0.0);
        assert_eq!(m.brier, 0.0);
        assert_eq!(m.ece, 0.0);
    }

    /// Cas analytique Brier : surconfiance. `(p=0.75, z=0)` → err 0.5625.
    #[test]
    fn analytic_brier_overconfident() {
        let pts = [CalibrationPoint { predicted: 0.75, observed: 0 }];
        let (loss, m) = compute_brier_calibration(&pts, 5, 0.5).unwrap();
        let expected = 0.75 * 0.75;
        assert!((loss.value() - expected).abs() < 1e-12);
        // ECE sur un bin : |conf − acc| = |0.75 − 0| = 0.75 (poids 1)
        assert!((m.ece - 0.75).abs() < 1e-12);
    }

    #[test]
    fn abstention_metrics() {
        let pts = [
            CalibrationPoint { predicted: 0.2, observed: 1 }, // s'abstient
            CalibrationPoint { predicted: 0.9, observed: 1 }, // gardé, correct
            CalibrationPoint { predicted: 0.8, observed: 0 }, // gardé, faux
        ];
        let (_l, m) = compute_brier_calibration(&pts, 5, 0.5).unwrap();
        assert!((m.abstention_rate - 1.0 / 3.0).abs() < 1e-12);
        assert!((m.accuracy_after_abstention - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(compute_brier_calibration(
            &[CalibrationPoint { predicted: 1.2, observed: 1 }],
            5, 0.5,
        ).is_err());
        assert!(compute_brier_calibration(
            &[CalibrationPoint { predicted: 0.5, observed: 2 }],
            5, 0.5,
        ).is_err());
    }
}
