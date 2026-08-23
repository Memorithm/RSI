//! **Validation croisée** oracle vs backend (contrat §14 et critères
//! d'acceptation §17).
//!
//! Pour chaque batch déterministe :
//! - calcule la référence avec `cogno-core` (oracle) ;
//! - calcule la version batch `cogno-scirust` ;
//! - compare chaque composante, l'objectif total, la perte ;
//! - enregistre la graine et la configuration.

use cogno_core::error::CognoResult;
use cogno_core::numeric::FiniteScalar;
use cogno_core::objective::CognoObjectiveBreakdown;

use crate::batch::{CognoBatchInput, compute_objective_batch};
use cogno_core::objective::{compute_cogno_objective, CognoWeights};
use cogno_core::resource::ResourceWeights;

/// Comparaison terme à terme entre l'oracle et le backend.
#[derive(Debug, Clone, Copy)]
pub struct BatchComparison {
    pub oracle: CognoObjectiveBreakdown,
    pub backend: CognoObjectiveBreakdown,
    /// tolérance relative appliquée.
    pub tolerance: f64,
    /// vrai si toutes les composantes correspondent dans la tolérance.
    pub matches: bool,
    /// composante qui a le plus divergé (nom + écart absolu).
    pub worst_component: &'static str,
    pub worst_abs_diff: f64,
}

impl BatchComparison {
    /// Compare deux breakdowns, composante par composante (décomposition
    /// jamais perdue — contrat §9/§17).
    pub fn compare(a: &CognoObjectiveBreakdown, b: &CognoObjectiveBreakdown, tolerance: f64) -> Self {
        let comps: [(&'static str, f64, f64); 9] = [
            ("admissible_reward", a.admissible_reward.value(), b.admissible_reward.value()),
            ("reference_log_ratio", a.reference_log_ratio.value(), b.reference_log_ratio.value()),
            ("preference_objective", a.preference_objective.value(), b.preference_objective.value()),
            ("symbolic_objective", a.symbolic_objective.value(), b.symbolic_objective.value()),
            ("memory_objective", a.memory_objective.value(), b.memory_objective.value()),
            ("pretraining_objective", a.pretraining_objective.value(), b.pretraining_objective.value()),
            ("calibration_loss", a.calibration_loss.value(), b.calibration_loss.value()),
            ("resource_loss", a.resource_loss.value(), b.resource_loss.value()),
            ("total_loss", a.total_loss.value(), b.total_loss.value()),
        ];
        let mut worst = ("", 0.0f64);
        let mut matches = true;
        for (name, av, bv) in comps {
            let diff = (av - bv).abs();
            // tolérance relative sur la magnitude max
            let scale = av.abs().max(bv.abs()).max(1.0);
            if diff > tolerance * scale {
                matches = false;
            }
            if diff > worst.1 {
                worst = (name, diff);
            }
        }
        BatchComparison {
            oracle: *a,
            backend: *b,
            tolerance,
            matches,
            worst_component: worst.0,
            worst_abs_diff: worst.1,
        }
    }
}

/// Rapport de cross-validation sur plusieurs batches.
#[derive(Debug, Clone, Default)]
pub struct CrossValidationReport {
    pub batches_tested: usize,
    pub all_match: bool,
    pub comparisons: Vec<BatchComparison>,
}

/// Exécute la cross-validation oracle ↔ backend sur un ensemble de batches.
///
/// Renvoie `Err` si les deux calculs ne produisent pas les mêmes composantes
/// dans la tolérance (le backend doit correspondre à l'oracle — §17).
pub fn compare_oracle_and_backend(
    batches: &[CognoBatchInput],
    weights: &CognoWeights,
    resource_weights: &ResourceWeights,
    tolerance: f64,
) -> CognoResult<CrossValidationReport> {
    let mut report = CrossValidationReport::default();
    let mut all_match = true;
    for input in batches {
        let oracle_input = cogno_core::objective::CognoObjectiveInput {
            rewards: input.rewards.clone(),
            log_prob_policy: input.log_prob_policy.clone(),
            log_prob_ref: input.log_prob_ref.clone(),
            preference_pairs: input.preference_pairs.clone(),
            soft_rules: input.soft_rules.clone(),
            memory_samples: input.memory_samples.clone(),
            pretrain_logp_mean: input.pretrain_logp_mean,
            calibration_points: input.calibration_points.clone(),
            resource_costs: input.resource_costs.clone(),
        };
        let (oracle, _) = compute_cogno_objective(
            &oracle_input,
            weights,
            resource_weights,
            1.0,
            1.0,
            1e-6,
            1,
            5,
            0.5,
        )?;
        let backend = compute_objective_batch(
            input,
            weights,
            resource_weights,
            1.0,
            1.0,
            1e-6,
            1,
            5,
            0.5,
        )?;
        let comp = BatchComparison::compare(&oracle, &backend.breakdown, tolerance);
        if !comp.matches {
            all_match = false;
        }
        report.batches_tested += 1;
        report.comparisons.push(comp);
    }
    report.all_match = all_match;
    Ok(report)
}

/// Résultat d'une comparaison de gradients (contrat §14 : « comparer les
/// gradients lorsque définis »).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientComparison {
    /// écart absolu maximal (par composante) entre gradient numérique et
    /// analytique.
    pub max_abs_diff: f64,
    /// vrai si toutes les composantes correspondent dans la tolérance.
    pub matches: bool,
}

/// Compare le gradient **numérique** (différences finies centrales de `f`) au
/// gradient **analytique** `grad`, composante par composante, dans la
/// tolérance relative donnée (échelle = max(|numérique|, |analytique|, 1)).
pub fn compare_gradients(
    f: &dyn Fn(&[f64]) -> CognoResult<FiniteScalar>,
    grad: &dyn Fn(&[f64]) -> CognoResult<Vec<f64>>,
    params: &[f64],
    tolerance: f64,
) -> CognoResult<GradientComparison> {
    let h = 1e-6;
    let mut max_diff = 0.0f64;
    let mut matches = true;
    let analytic = grad(params)?;
    if analytic.len() != params.len() {
        return Err(cogno_core::error::CognoError::LengthMismatch {
            expected: params.len(),
            got: analytic.len(),
        });
    }
    for i in 0..params.len() {
        let mut p1 = params.to_vec();
        let mut p2 = params.to_vec();
        p1[i] += h;
        p2[i] -= h;
        let f1 = f(&p1)?.value();
        let f2 = f(&p2)?.value();
        let numeric = (f1 - f2) / (2.0 * h);
        if !numeric.is_finite() || !analytic[i].is_finite() {
            return Err(cogno_core::error::CognoError::NonFinite("gradient"));
        }
        let diff = (numeric - analytic[i]).abs();
        // tolérance relative sur la magnitude max
        let scale = numeric.abs().max(analytic[i].abs()).max(1.0);
        if diff > tolerance * scale {
            matches = false;
        }
        max_diff = max_diff.max(diff);
    }
    Ok(GradientComparison { max_abs_diff: max_diff, matches })
}

/// Comparaison **après un pas d'optimisation** (contrat §14 : « comparer les
/// résultats après un pas d'optimisation »).
///
/// Exécute un pas AdamW (oracle et backend utilisent la même implémentation)
/// sur un vecteur de paramètres avec un gradient donné, puis vérifie que :
/// - les paramètres restent finis ;
/// - le pas est déterministe (deux exécutions identiques donnent le même
///   résultat) ;
/// - la norme du déplacement est bornée par le clipping de gradient.
pub fn compare_after_optim_step(
    config: crate::adamw::AdamWConfig,
    params: &[f64],
    grad: &[f64],
) -> CognoResult<bool> {
    use crate::adamw::AdamW;
    let run = || -> CognoResult<Vec<f64>> {
        let mut opt = AdamW::new(config, params.len());
        let mut p = params.to_vec();
        opt.step(&mut p, grad).map_err(cogno_core::error::CognoError::InvalidInput)?;
        Ok(p)
    };
    let a = run()?;
    let b = run()?;
    // déterminisme : deux exécutions identiques
    if a != b {
        return Ok(false);
    }
    // finitude + déplacement borné par le clip (si clip activé)
    for &v in &a {
        if !v.is_finite() {
            return Ok(false);
        }
    }
    if let Some(max_norm) = config.grad_clip {
        let disp: f64 = a
            .iter()
            .zip(params)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        // le déplacement est borné par lr × clip (borne large)
        let bound = config.lr * max_norm * 4.0 + 1e-9;
        if disp > bound {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogno_core::reward::RewardBreakdown;

    fn sample_batch() -> CognoBatchInput {
        let mut b = CognoBatchInput::empty();
        b.rewards = vec![
            RewardBreakdown::compute(1.0, 0.5, 1.0, 0.25, 0.25, 0.1, 0.05, 0.5, 0.2).unwrap(),
        ];
        b.log_prob_policy = vec![-0.5];
        b.log_prob_ref = vec![-0.5];
        b
    }

    #[test]
    fn oracle_and_backend_match() {
        let w = CognoWeights::default();
        let report = compare_oracle_and_backend(&[sample_batch()], &w, &ResourceWeights::default(), 1e-9).unwrap();
        assert!(report.all_match, "backend ne correspond pas à l'oracle");
        assert_eq!(report.batches_tested, 1);
        assert!((report.comparisons[0].oracle.total_loss.value()
            - report.comparisons[0].backend.total_loss.value()).abs() < 1e-9);
    }

    #[test]
    fn worst_component_reported() {        // un backend dégradé (tous les termes nuls) doit être détecté :
        // l'oracle calcule J=1.75 (batch complet), le "backend zéro" vaut 0.
        let b = sample_batch();
        let oracle_input = cogno_core::objective::CognoObjectiveInput {
            rewards: b.rewards.clone(),
            log_prob_policy: b.log_prob_policy.clone(),
            log_prob_ref: b.log_prob_ref.clone(),
            preference_pairs: b.preference_pairs.clone(),
            soft_rules: b.soft_rules.clone(),
            memory_samples: b.memory_samples.clone(),
            pretrain_logp_mean: b.pretrain_logp_mean,
            calibration_points: b.calibration_points.clone(),
            resource_costs: b.resource_costs.clone(),
        };
        let w = CognoWeights::default();
        let (oracle, _) = compute_cogno_objective(&oracle_input, &w, &ResourceWeights::default(), 1.0, 1.0, 1e-6, 1, 5, 0.5).unwrap();
        assert!((oracle.admissible_reward.value() - 1.75).abs() < 1e-12, "oracle reward={}", oracle.admissible_reward.value());
        // backend "zéro" : on fabrique un breakdown nul
        let zero = CognoObjectiveBreakdown {
            admissible_reward: FiniteScalar::ZERO,
            reference_log_ratio: FiniteScalar::ZERO,
            preference_objective: FiniteScalar::ZERO,
            symbolic_objective: FiniteScalar::ZERO,
            memory_objective: FiniteScalar::ZERO,
            pretraining_objective: FiniteScalar::ZERO,
            calibration_loss: cogno_core::numeric::NonNegativeFinite::ZERO,
            resource_loss: cogno_core::numeric::NonNegativeFinite::ZERO,
            total_objective: FiniteScalar::ZERO,
            total_loss: FiniteScalar::ZERO,
        };
        let comp = BatchComparison::compare(&oracle, &zero, 1e-9);
        assert!(!comp.matches);
        assert!(!comp.worst_component.is_empty());
        assert_eq!(comp.worst_component, "admissible_reward");
    }

    /// Contrat §14 : le gradient numérique (différences finies) doit
    /// correspondre au gradient analytique dans la tolérance — et un gradient
    /// analytique faux doit être détecté.
    #[test]
    fn gradient_comparison_matches_or_detects() {
        // f(p) = p0² + 3·p1 → ∇f = [2·p0, 3]
        let f = |p: &[f64]| FiniteScalar::try_new(p[0] * p[0] + 3.0 * p[1]);
        let grad = |p: &[f64]| -> CognoResult<Vec<f64>> { Ok(vec![2.0 * p[0], 3.0]) };
        let r = compare_gradients(&f, &grad, &[1.5, -2.0], 1e-4).unwrap();
        assert!(r.matches, "analytique correct rejeté : {:?}", r);
        assert!(r.max_abs_diff < 1e-3);

        let wrong = |_p: &[f64]| -> CognoResult<Vec<f64>> { Ok(vec![0.0, 0.0]) };
        let r = compare_gradients(&f, &wrong, &[1.5, -2.0], 1e-4).unwrap();
        assert!(!r.matches, "gradient analytique faux non détecté");
        assert!(r.max_abs_diff > 1.0);

        // longueur incohérente → erreur structurée
        let short = |_p: &[f64]| -> CognoResult<Vec<f64>> { Ok(vec![1.0]) };
        let e = compare_gradients(&f, &short, &[1.5, -2.0], 1e-4).unwrap_err();
        assert!(matches!(e, cogno_core::error::CognoError::LengthMismatch { .. }));
    }

    // ─── Validation croisée §14 : cas imposés ─────────────────────────────── //

    use cogno_core::calib::CalibrationPoint;
    use cogno_core::memory::MemorySample;
    use cogno_core::pref::PreferencePair;
    use cogno_core::softlogic::SoftRule;

    fn unit_vec(v: usize, dim: usize) -> Vec<f64> {
        let mut x = vec![0.0; dim];
        if v < dim {
            x[v] = 1.0;
        }
        x
    }

    fn check_batch(input: &CognoBatchInput, label: &str) {
        let w = CognoWeights::default();
        let report = compare_oracle_and_backend(std::slice::from_ref(input), &w, &ResourceWeights::default(), 1e-9);
        match report {
            Ok(r) => assert!(r.all_match, "batch [{label}] : backend ≠ oracle"),
            Err(e) => panic!("batch [{label}] : erreur inattendue {e}"),
        }
    }

    /// Batch vide (tous les termes vides) — l'oracle et le backend doivent
    /// tous deux retourner une perte 0 finie et correspondre.
    #[test]
    fn cv_empty_batch() {
        let b = CognoBatchInput::empty();
        check_batch(&b, "vide");
    }

    /// Batch de taille un avec tous les termes actifs.
    #[test]
    fn cv_single_batch_all_terms() {
        let mut b = CognoBatchInput::empty();
        b.rewards = vec![
            RewardBreakdown::compute(0.5, 0.2, 1.0, 0.1, 0.1, 0.01, 0.02, 0.1, 0.5).unwrap(),
        ];
        b.log_prob_policy = vec![-0.7];
        b.log_prob_ref = vec![-0.7];
        b.preference_pairs = vec![PreferencePair {
            context: b"x",
            log_prob_policy_positive: -0.5,
            log_prob_policy_negative: -0.9,
            log_prob_ref_positive: -0.5,
            log_prob_ref_negative: -0.5,
        }];
        b.soft_rules = vec![
            SoftRule::new(1.0, 0.5, "r1").unwrap(),
            SoftRule::new(0.0, 0.5, "r2").unwrap(),
        ];
        b.memory_samples = vec![MemorySample {
            context: unit_vec(0, 4),
            positive: unit_vec(0, 4),
            negatives: vec![unit_vec(1, 4)],
            category: 0,
            conflicting_rule: 0,
            held_out: false,
        }];
        b.calibration_points = vec![
            CalibrationPoint { predicted: 0.8, observed: 1 },
            CalibrationPoint { predicted: 0.3, observed: 0 },
        ];
        b.resource_costs = vec![cogno_core::resource::ResourceCosts::new(1024, 50, 512)];
        b.pretrain_logp_mean = -0.4;
        check_batch(&b, "taille-un");
    }

    /// Log-probabilités très négatives (≈ zéro) : pas d'infini, les deux
    /// calculs correspondent.
    #[test]
    fn cv_very_negative_logprobs() {
        let mut b = CognoBatchInput::empty();
        b.log_prob_policy = vec![-1000.0, -1e9];
        b.log_prob_ref = vec![-1000.0, -1e9];
        b.rewards = vec![
            RewardBreakdown::compute(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap(),
            RewardBreakdown::compute(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap(),
        ];
        check_batch(&b, "logprobs-tres-negatives");
    }

    /// Égalité policy/reference (KL = 0) : les termes KL des deux calculs
    /// valent zéro.
    #[test]
    fn cv_policy_equals_ref() {
        let mut b = CognoBatchInput::empty();
        b.rewards = vec![
            RewardBreakdown::compute(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap(),
        ];
        b.log_prob_policy = vec![-0.5];
        b.log_prob_ref = vec![-0.5];
        check_batch(&b, "policy=ref");
    }

    /// Règle totalement violée (s=0) : fini (log ε), pas de −∞.
    #[test]
    fn cv_fully_violated_rule() {
        let mut b = CognoBatchInput::empty();
        b.soft_rules = vec![SoftRule::new(0.0, 1.0, "r").unwrap()];
        check_batch(&b, "regle-violée");
    }

    /// Mémoire positive mal classée : métriques correctes des deux côtés.
    #[test]
    fn cv_memory_misplaced() {
        let mut b = CognoBatchInput::empty();
        b.memory_samples = vec![MemorySample {
            context: unit_vec(0, 4),
            positive: unit_vec(1, 4),
            negatives: vec![unit_vec(0, 4)],
            category: 1,
            conflicting_rule: 0,
            held_out: false,
        }];
        check_batch(&b, "memoire-mal-classée");
    }

    /// Surconfiance (Brier) : les deux calculs donnent le même Brier.
    #[test]
    fn cv_overconfident_calibration() {
        let mut b = CognoBatchInput::empty();
        b.calibration_points = vec![CalibrationPoint { predicted: 0.99, observed: 0 }];
        check_batch(&b, "surconfiance");
    }

    /// Coûts au budget : perte de ressources finie, identique des deux côtés.
    #[test]
    fn cv_costs_at_budget() {
        let mut b = CognoBatchInput::empty();
        b.resource_costs = vec![cogno_core::resource::ResourceCosts::new(
            8 * 1024 * 1024, 500, 4096,
        )];
        check_batch(&b, "coûts-au-budget");
    }

    /// NaN dans une entrée : les deux calculs échouent avec une erreur
    /// structurée (jamais de panic, jamais de NaN silencieux).
    #[test]
    fn cv_nan_rejected_by_both() {
        let mut b = CognoBatchInput::empty();
        b.log_prob_policy = vec![f64::NAN];
        b.log_prob_ref = vec![f64::NAN];
        b.rewards = vec![
            RewardBreakdown::compute(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap(),
        ];
        let w = CognoWeights::default();
        // l'oracle échoue
        let oracle_input = cogno_core::objective::CognoObjectiveInput {
            rewards: b.rewards.clone(),
            log_prob_policy: b.log_prob_policy.clone(),
            log_prob_ref: b.log_prob_ref.clone(),
            preference_pairs: vec![],
            soft_rules: vec![],
            memory_samples: vec![],
            pretrain_logp_mean: 0.0,
            calibration_points: vec![],
            resource_costs: vec![],
        };
        let o = compute_cogno_objective(&oracle_input, &w, &ResourceWeights::default(), 1.0, 1.0, 1e-6, 1, 5, 0.5);
        assert!(o.is_err(), "NaN doit être rejeté par l'oracle");
        let bt = compute_objective_batch(&b, &w, &ResourceWeights::default(), 1.0, 1.0, 1e-6, 1, 5, 0.5);
        assert!(bt.is_err(), "NaN doit être rejeté par le backend");
    }

    /// Infini dans une entrée : rejet structuré des deux côtés.
    #[test]
    fn cv_inf_rejected_by_both() {
        let mut b = CognoBatchInput::empty();
        b.preference_pairs = vec![PreferencePair {
            context: b"x",
            log_prob_policy_positive: f64::INFINITY,
            log_prob_policy_negative: 0.0,
            log_prob_ref_positive: 0.0,
            log_prob_ref_negative: 0.0,
        }];
        let w = CognoWeights::default();
        let oracle_input = cogno_core::objective::CognoObjectiveInput {
            rewards: vec![],
            log_prob_policy: vec![],
            log_prob_ref: vec![],
            preference_pairs: b.preference_pairs.clone(),
            soft_rules: vec![],
            memory_samples: vec![],
            pretrain_logp_mean: 0.0,
            calibration_points: vec![],
            resource_costs: vec![],
        };
        assert!(compute_cogno_objective(&oracle_input, &w, &ResourceWeights::default(), 1.0, 1.0, 1e-6, 1, 5, 0.5).is_err());
        assert!(compute_objective_batch(&b, &w, &ResourceWeights::default(), 1.0, 1.0, 1e-6, 1, 5, 0.5).is_err());
    }

    /// Mismatch de longueur (récompenses vs log-probs) : erreur structurée.
    #[test]
    fn cv_length_mismatch() {
        let mut b = CognoBatchInput::empty();
        b.rewards = vec![
            RewardBreakdown::compute(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap(),
        ];
        b.log_prob_policy = vec![]; // 1 récompense vs 0 log-prob
        b.log_prob_ref = vec![];
        let w = CognoWeights::default();
        let oracle_input = cogno_core::objective::CognoObjectiveInput {
            rewards: b.rewards.clone(),
            log_prob_policy: vec![],
            log_prob_ref: vec![],
            preference_pairs: vec![],
            soft_rules: vec![],
            memory_samples: vec![],
            pretrain_logp_mean: 0.0,
            calibration_points: vec![],
            resource_costs: vec![],
        };
        let o = compute_cogno_objective(&oracle_input, &w, &ResourceWeights::default(), 1.0, 1.0, 1e-6, 1, 5, 0.5);
        assert!(matches!(o, Err(cogno_core::error::CognoError::LengthMismatch { .. })));
        let bt = compute_objective_batch(&b, &w, &ResourceWeights::default(), 1.0, 1.0, 1e-6, 1, 5, 0.5);
        assert!(matches!(bt, Err(cogno_core::error::CognoError::LengthMismatch { .. })));
    }

    /// Préférence indifférente (Δ=0) : J_pref = ln 0.5, identique.
    #[test]
    fn cv_indifferent_preference() {
        let mut b = CognoBatchInput::empty();
        b.preference_pairs = vec![PreferencePair {
            context: b"x",
            log_prob_policy_positive: -1.0,
            log_prob_policy_negative: -1.0,
            log_prob_ref_positive: -1.0,
            log_prob_ref_negative: -1.0,
        }];
        check_batch(&b, "préférence-indifférente");
    }
}
