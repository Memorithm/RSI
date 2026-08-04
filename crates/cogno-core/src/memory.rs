//! Terme **contrastif de mémoire** InfoNCE (contrat §6) + métriques séparées.
//!
//! ```text
//! J_mem(φ,ψ) = E[ log( exp(sim(h_φ(x), e_ψ(m+))/τ)
//!                      / Σ_{m∈M_x} exp(sim(h_φ(x), e_ψ(m))/τ) ) ]
//! ```
//!
//! - `sim` : similarité **cosinus** (documentée, initiale) ;
//! - `τ > 0` : température ;
//! - les autres mémoires du batch sont des **négatifs internes au batch**.
//!
//! Métriques mesurées séparément (jamais seulement la somme) : Recall@1,
//! Recall@K, MRR, perte InfoNCE, rappel par catégorie, rappel des règles
//! conflictuelles, rappel held-out.

use crate::error::{CognoError, CognoResult};
use crate::numeric::{CompensatedSum, FiniteScalar};

/// Échantillon de mémoire : contexte `x`, mémoire pertinente `m+`, négatifs.
#[derive(Debug, Clone)]
pub struct MemorySample {
    pub context: Vec<f64>,
    pub positive: Vec<f64>,
    pub negatives: Vec<Vec<f64>>,
    /// catégorie de la mémoire (pour le rappel par catégorie).
    pub category: u32,
    /// règle conflictuelle associée (0 = aucune).
    pub conflicting_rule: u32,
    /// échantillon held-out (jamais vu en entraînement).
    pub held_out: bool,
}

/// Similarité **cosinus** : `a·b / (‖a‖·‖b‖)`. Vecteurs vides → erreur.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> CognoResult<f64> {
    if a.len() != b.len() {
        return Err(CognoError::LengthMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(CognoError::InvalidInput("vecteur vide"));
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        return Err(CognoError::InvalidInput("norme nulle"));
    }
    Ok((dot / denom).clamp(-1.0, 1.0))
}

/// Métriques de mémoire, mesurées séparément (contrat §6).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemoryMetrics {
    pub info_nce_loss_sum: f64,
    pub samples: usize,
    pub recall_at_1: f64,
    pub recall_at_k: f64,
    pub mrr: f64,
    pub recall_by_category: Vec<(u32, f64)>,
    pub recall_conflicting: f64,
    pub recall_held_out: f64,
    pub conflicting_count: usize,
    pub held_out_count: usize,
}

/// Calcule `J_mem` (moyenne InfoNCE) et les métriques séparées.
///
/// Le calcul est **déterministe** : les négatifs sont traités dans l'ordre du
/// batch. Utilise la somme compensée pour la stabilité.
pub fn compute_memory_objective(
    samples: &[MemorySample],
    tau: f64,
    k_recall: usize,
) -> CognoResult<(FiniteScalar, MemoryMetrics)> {
    if tau <= 0.0 || !tau.is_finite() {
        return Err(CognoError::InvalidInput("τ > 0"));
    }
    let mut total = CompensatedSum::new();
    let mut recall1 = 0usize;
    let mut recallk = 0usize;
    let mut mrr_sum = 0.0;
    let mut conflicting_hits = 0usize;
    let mut conflicting_total = 0usize;
    let mut heldout_hits = 0usize;
    let mut heldout_total = 0usize;
    let mut cat_hits: std::collections::HashMap<u32, (usize, usize)> = Default::default();

    for s in samples {
        // similarités (ordre : positif puis négatifs du batch)
        let sim_pos = cosine_similarity(&s.context, &s.positive)?;
        let mut sims = vec![sim_pos];
        for n in &s.negatives {
            sims.push(cosine_similarity(&s.context, n)?);
        }
        // InfoNCE stable : logsumexp sur les exp(sim/τ)
        let scaled: Vec<f64> = sims.iter().map(|v| v / tau).collect();
        let lse = logsumexp(&scaled)?;
        let nce = sim_pos / tau - lse;
        total.add(nce);
        let _ = FiniteScalar::try_new(nce).map_err(|_| CognoError::NonFinite("nce"))?;

        // classement du positif parmi [positif + négatifs]
        let rank = rank_of_positive(&sims);
        if rank == 0 {
            recall1 += 1;
        }
        if rank >= 0 && (rank as usize) < k_recall {
            recallk += 1;
        }
        if rank >= 0 {
            mrr_sum += 1.0 / (rank as f64 + 1.0);
        }

        // rappel par catégorie
        let e = cat_hits.entry(s.category).or_insert((0, 0));
        e.0 += 1;
        if rank == 0 {
            e.1 += 1;
        }
        // règles conflictuelles
        if s.conflicting_rule != 0 {
            conflicting_total += 1;
            if rank == 0 {
                conflicting_hits += 1;
            }
        }
        // held-out
        if s.held_out {
            heldout_total += 1;
            if rank == 0 {
                heldout_hits += 1;
            }
        }
    }

    let n = samples.len().max(1) as f64;
    let mut metrics = MemoryMetrics {
        info_nce_loss_sum: total.finish(),
        samples: samples.len(),
        recall_at_1: recall1 as f64 / n,
        recall_at_k: recallk as f64 / n,
        mrr: mrr_sum / n,
        recall_conflicting: if conflicting_total > 0 {
            conflicting_hits as f64 / conflicting_total as f64
        } else {
            f64::NAN
        },
        recall_held_out: if heldout_total > 0 {
            heldout_hits as f64 / heldout_total as f64
        } else {
            f64::NAN
        },
        conflicting_count: conflicting_total,
        held_out_count: heldout_total,
        ..Default::default()
    };
    let mut cats: Vec<(u32, f64)> = cat_hits
        .iter()
        .map(|(&c, &(total_c, hits))| (c, hits as f64 / total_c as f64))
        .collect();
    cats.sort_by_key(|&(c, _)| c);
    metrics.recall_by_category = cats;

    let mean = if samples.is_empty() {
        0.0
    } else {
        total.finish() / n
    };
    Ok((FiniteScalar::try_new(mean)?, metrics))
}

/// LogSumExp stable (max-shift) : évite les débordements de `exp`.
fn logsumexp(v: &[f64]) -> CognoResult<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return Err(CognoError::NonFinite("logsumexp max"));
    }
    let mut s = CompensatedSum::new();
    for &x in v {
        s.add((x - max).exp());
    }
    Ok(max + s.finish().ln())
}

/// Rang (0-based) de l'élément positif dans `sims` (index 0). Retourne −1 si
/// impossible (batch vide — ne devrait pas arriver).
fn rank_of_positive(sims: &[f64]) -> isize {
    if sims.is_empty() {
        return -1;
    }
    let pos = sims[0];
    sims.iter().filter(|&&s| s > pos).count() as isize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: usize, dim: usize) -> Vec<f64> {
        let mut x = vec![0.0; dim];
        if v < dim {
            x[v] = 1.0;
        }
        x
    }

    /// Cas analytique InfoNCE : contexte aligné avec le positif, un négatif
    /// orthogonal. `sim_pos = 1`, `sim_neg = 0`, `τ = 1` :
    /// `log(e^1/(e^1+e^0)) = log(e/(e+1)) = 1 − log(1+e) ≈ −0.3133`.
    #[test]
    fn analytic_infonce() {
        let sample = MemorySample {
            context: unit(0, 4),
            positive: unit(0, 4),
            negatives: vec![unit(1, 4)],
            category: 0,
            conflicting_rule: 0,
            held_out: false,
        };
        let (j, m) = compute_memory_objective(&[sample], 1.0, 1).unwrap();
        let expected = 1.0 - (1.0_f64 + std::f64::consts::E).ln();
        assert!((j.value() - expected).abs() < 1e-9, "j={}", j.value());
        assert_eq!(m.recall_at_1, 1.0);
        assert_eq!(m.mrr, 1.0);
        assert_eq!(m.samples, 1);
    }

    #[test]
    fn misplaced_positive_drops_metrics() {
        // le positif est orthogonal au contexte, le négatif aligné → mal classé
        let sample = MemorySample {
            context: unit(0, 4),
            positive: unit(1, 4),
            negatives: vec![unit(0, 4)],
            category: 1,
            conflicting_rule: 0,
            held_out: false,
        };
        let (_j, m) = compute_memory_objective(&[sample], 1.0, 1).unwrap();
        assert_eq!(m.recall_at_1, 0.0);
        assert_eq!(m.mrr, 0.5); // rang 1 → 1/2
    }

    #[test]
    fn metrics_are_separate_by_category_and_heldout() {
        let mk = |cat: u32, conf: u32, held: bool, aligned: bool| MemorySample {
            context: unit(0, 4),
            positive: if aligned { unit(0, 4) } else { unit(1, 4) },
            // négatif : pour un échantillon mal classé, le négatif est le
            // contexte lui-même (similarité 1) → le positif (sim 0) est battu
            negatives: if aligned { vec![unit(1, 4)] } else { vec![unit(0, 4)] },
            category: cat,
            conflicting_rule: conf,
            held_out: held,
        };
        let samples = vec![
            mk(0, 0, false, true), // cat 0, aligné
            mk(1, 0, false, false), // cat 1, mal classé
            mk(2, 7, true, true), // cat 2, conflictuelle + held-out, aligné
        ];
        let (_j, m) = compute_memory_objective(&samples, 1.0, 1).unwrap();
        assert_eq!(m.recall_by_category.len(), 3);
        assert_eq!(m.recall_by_category[0].1, 1.0);
        assert_eq!(m.recall_by_category[1].1, 0.0);
        assert_eq!(m.recall_conflicting, 1.0);
        assert_eq!(m.recall_held_out, 1.0);
        assert_eq!(m.conflicting_count, 1);
        assert_eq!(m.held_out_count, 1);
    }
}
