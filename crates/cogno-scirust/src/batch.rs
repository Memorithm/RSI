//! Version **batch** de l'objectif COGNO-1 et de la perte COGNO-0.1
//! (contrat §12 : « version tensorielle différentiable de l'objectif complet »
//! et « version tensorielle de la perte COGNO-0.1 »).
//!
//! Le backend calcule en batch avec les mêmes équations que l'oracle
//! `cogno-core` (pas de convention mathématique inventée) ; la cross-validation
//! garantit la correspondance (voir [`crate::cross_validate`]).

use cogno_core::calib::{compute_brier_calibration, CalibrationPoint, CalibrationMetrics};
use cogno_core::error::{CognoError, CognoResult};
use cogno_core::memory::{compute_memory_objective, MemoryMetrics, MemorySample};
use cogno_core::numeric::{CompensatedSum, FiniteScalar, NonNegativeFinite};
use cogno_core::objective::{
    CognoObjectiveBreakdown, CognoWeights,
};
use cogno_core::pref::{compute_preference_objective, PreferencePair};
use cogno_core::resource::{compute_resource_loss_batch, ResourceCosts, ResourceWeights};
use cogno_core::reward::RewardBreakdown;
use cogno_core::softlogic::{compute_symbolic_objective, SoftRule};

/// Sortie batch de l'objectif complet (même décomposition que l'oracle).
#[derive(Debug, Clone, Copy)]
pub struct CognoBatchOutput {
    pub breakdown: CognoObjectiveBreakdown,
}

/// Entrées batch (identiques au contrat de `cogno-core`).
#[derive(Debug, Clone)]
pub struct CognoBatchInput {
    pub rewards: Vec<RewardBreakdown>,
    pub log_prob_policy: Vec<f64>,
    pub log_prob_ref: Vec<f64>,
    pub preference_pairs: Vec<PreferencePair>,
    pub soft_rules: Vec<SoftRule>,
    pub memory_samples: Vec<MemorySample>,
    pub pretrain_logp_mean: f64,
    pub calibration_points: Vec<CalibrationPoint>,
    pub resource_costs: Vec<ResourceCosts>,
}

impl CognoBatchInput {
    /// Prépare un batch vide (tous les termes à zéro) — pour les tests.
    pub fn empty() -> Self {
        CognoBatchInput {
            rewards: vec![],
            log_prob_policy: vec![],
            log_prob_ref: vec![],
            preference_pairs: vec![],
            soft_rules: vec![],
            memory_samples: vec![],
            pretrain_logp_mean: 0.0,
            calibration_points: vec![],
            resource_costs: vec![],
        }
    }
}

/// Version batch de l'objectif complet `J_COGNO` (contrat §9).
///
/// Appelle les mêmes fonctions d'évaluation que l'oracle, terme par terme —
/// chaque terme a son API distincte et est observable dans le breakdown.
#[allow(clippy::too_many_arguments)]
pub fn compute_objective_batch(
    input: &CognoBatchInput,
    weights: &CognoWeights,
    resource_weights: &ResourceWeights,
    alpha: f64,
    tau: f64,
    symbolic_epsilon: f64,
    k_recall: usize,
    n_calib_bins: usize,
    abstain_below: f64,
) -> CognoResult<CognoBatchOutput> {
    if input.rewards.len() != input.log_prob_policy.len()
        || input.rewards.len() != input.log_prob_ref.len()
    {
        return Err(CognoError::LengthMismatch {
            expected: input.rewards.len(),
            got: input.log_prob_policy.len().max(input.log_prob_ref.len()),
        });
    }

    // 1. récompense admissible
    let mut rew = CompensatedSum::new();
    for r in &input.rewards {
        rew.add(r.total);
    }
    let n_rl = input.rewards.len().max(1) as f64;
    let admissible_reward = FiniteScalar::try_new(rew.finish() / n_rl)?;

    // 2. KL (log-espace)
    let mut kl = CompensatedSum::new();
    for i in 0..input.rewards.len() {
        let d = input.log_prob_policy[i] - input.log_prob_ref[i];
        FiniteScalar::try_new(d).map_err(|_| CognoError::NonFinite("log_ratio"))?;
        kl.add(d);
    }
    let ref_log_ratio = FiniteScalar::try_new(-weights.beta_ns.value() * (kl.finish() / n_rl))?;

    // 3-8. termes (mêmes fonctions que l'oracle)
    let j_pref = compute_preference_objective(&input.preference_pairs, alpha)?;
    let j_sym = compute_symbolic_objective(&input.soft_rules, symbolic_epsilon)?;
    let (j_mem, _mem) = compute_memory_objective(&input.memory_samples, tau, k_recall)?;
    FiniteScalar::try_new(input.pretrain_logp_mean)?;
    let pretrain_obj =
        FiniteScalar::try_new(weights.gamma_ns.value() * input.pretrain_logp_mean)?;
    let (brier, _cal) =
        compute_brier_calibration(&input.calibration_points, n_calib_bins, abstain_below)?;
    let cal_loss = NonNegativeFinite::try_new(weights.lambda_cal.value() * brier.value())?;
    let res_loss = compute_resource_loss_batch(&input.resource_costs, resource_weights, &Default::default())?;
    let res_term = NonNegativeFinite::try_new(weights.lambda_eff.value() * res_loss.value())?;

    // somme finale (ordre exact de l'équation)
    let mut total = CompensatedSum::new();
    total.add(admissible_reward.value());
    total.add(ref_log_ratio.value());
    total.add(weights.eta_pref.value() * j_pref.value());
    total.add(weights.eta_sym.value() * j_sym.value());
    total.add(weights.eta_mem.value() * j_mem.value());
    total.add(pretrain_obj.value());
    total.add(-cal_loss.value());
    total.add(-res_term.value());
    let total_objective = FiniteScalar::try_new(total.finish())?;
    let total_loss = FiniteScalar::try_new(-total_objective.value())?;

    Ok(CognoBatchOutput {
        breakdown: CognoObjectiveBreakdown {
            admissible_reward,
            reference_log_ratio: ref_log_ratio,
            preference_objective: FiniteScalar::try_new(weights.eta_pref.value() * j_pref.value())?,
            symbolic_objective: FiniteScalar::try_new(weights.eta_sym.value() * j_sym.value())?,
            memory_objective: FiniteScalar::try_new(weights.eta_mem.value() * j_mem.value())?,
            pretraining_objective: pretrain_obj,
            calibration_loss: cal_loss,
            resource_loss: res_term,
            total_objective,
            total_loss,
        },
    })
}

/// Version batch de la perte COGNO-0.1 (contrat §10).
#[allow(clippy::too_many_arguments)]
pub fn compute_cogno01_loss_batch(
    log_prob_policy: &[f64],
    preference_pairs: &[PreferencePair],
    soft_rules: &[SoftRule],
    memory_samples: &[MemorySample],
    calibration_points: &[CalibrationPoint],
    resource_costs: &[ResourceCosts],
    weights: &CognoWeights,
    resource_weights: &ResourceWeights,
    alpha: f64,
    tau: f64,
    symbolic_epsilon: f64,
    n_calib_bins: usize,
    abstain_below: f64,
) -> CognoResult<(FiniteScalar, MemoryMetrics, CalibrationMetrics)> {
    // L_SFT
    let mut sft = CompensatedSum::new();
    for &lp in log_prob_policy {
        FiniteScalar::try_new(lp).map_err(|_| CognoError::NonFinite("log_prob"))?;
        sft.add(-lp);
    }
    let n_sft = log_prob_policy.len().max(1) as f64;
    let l_sft = sft.finish() / n_sft;

    let j_pref = compute_preference_objective(preference_pairs, alpha)?;
    let l_pairwise = -j_pref.value();
    let j_sym = compute_symbolic_objective(soft_rules, symbolic_epsilon)?;
    let l_softlogic = -j_sym.value();
    let (j_mem, mem) = compute_memory_objective(memory_samples, tau, 1)?;
    let l_infonce = -j_mem.value();
    let (brier, cal) = compute_brier_calibration(calibration_points, n_calib_bins, abstain_below)?;
    let res = compute_resource_loss_batch(resource_costs, resource_weights, &Default::default())?;

    let mut total = CompensatedSum::new();
    total.add(l_sft);
    total.add(weights.eta_pref.value() * l_pairwise);
    total.add(weights.eta_sym.value() * l_softlogic);
    total.add(weights.eta_mem.value() * l_infonce);
    total.add(weights.lambda_cal.value() * brier.value());
    total.add(weights.lambda_eff.value() * res.value());
    let loss = FiniteScalar::try_new(total.finish())?;
    Ok((loss, mem, cal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogno_core::reward::RewardBreakdown;

    /// Le backend batch correspond à l'oracle sur le cas analytique historique
    /// (J = 1.75, L = −1.75).
    #[test]
    fn batch_matches_oracle_historical() {
        let mut input = CognoBatchInput::empty();
        input.rewards = vec![
            RewardBreakdown::compute(1.0, 0.5, 1.0, 0.25, 0.25, 0.1, 0.05, 0.5, 0.2).unwrap(),
        ];
        input.log_prob_policy = vec![-0.5];
        input.log_prob_ref = vec![-0.5];
        let w = CognoWeights::try_new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
        let out = compute_objective_batch(
            &input,
            &w,
            &ResourceWeights::default(),
            1.0,
            1.0,
            1e-6,
            1,
            5,
            0.5,
        )
        .unwrap();
        assert!((out.breakdown.total_objective.value() - 1.75).abs() < 1e-12);
        assert!((out.breakdown.total_loss.value() - (-1.75)).abs() < 1e-12);
    }
}
