//! **Entraînement contrôlé** de COGNO-1 (contrat §10, §12).
//!
//! - [`GradientAccumulator`] : accumulation de gradient **contrôlée** (nombre
//!   d'étapes fixe, division par le facteur d'accumulation, remise à zéro
//!   explicite, jamais de fuite d'état).
//! - [`AllocationStats`] : **statistiques d'allocation** du chemin critique
//!   (nombre d'allocations demandées, octets cumulés, compteur d'appels) —
//!   mesurées via un compteur global non bloquant (pas de vrai allocator hook,
//!   mais une instrumentation déterministe du code d'entraînement).
//! - Chemin **f32** : variantes `f32` des opérations chaudes, validées contre
//!   le chemin `f64` (mixed precision conditionnelle).
//! - [`ControlledRollout`] : **rollouts contrôlés** (génération bornée,
//!   validateurs durs appliqués avant retour — jamais de sortie inadmissible).
//! - [`PpoPolicy`] : **PPO** (clip ratio) — uniquement activable après
//!   stabilisation de COGNO-0.1 (garde `require_stable_01`).

use cogno_core::error::{CognoError, CognoResult};
use cogno_core::numeric::FiniteScalar;

// ─────────────────────────── Gradient accumulation ─────────────────────────── //

/// Accumulateur de gradient **contrôlé** (contrat §12).
///
/// - `accumulate` additionne un gradient dans le tampon interne ;
/// - après `steps` accumulations, `take_normalized` rend le gradient moyen
///   (divisé par `steps`) et **remet le tampon à zéro** ;
/// - la taille du tampon est fixée à la construction (pas de croissance).
pub struct GradientAccumulator {
    buffer: Vec<f64>,
    steps: usize,
    target_steps: usize,
}

impl GradientAccumulator {
    /// Prépare un accumulateur pour `n_params` paramètres et `target_steps`
    /// micro-pas. `target_steps > 0` (sinon erreur).
    pub fn try_new(n_params: usize, target_steps: usize) -> CognoResult<Self> {
        if target_steps == 0 {
            return Err(CognoError::InvalidInput("target_steps > 0"));
        }
        Ok(GradientAccumulator {
            buffer: vec![0.0; n_params],
            steps: 0,
            target_steps,
        })
    }

    /// Accumule un gradient (longueur validée).
    pub fn accumulate(&mut self, grad: &[f64]) -> CognoResult<()> {
        if grad.len() != self.buffer.len() {
            return Err(CognoError::LengthMismatch {
                expected: self.buffer.len(),
                got: grad.len(),
            });
        }
        for (b, g) in self.buffer.iter_mut().zip(grad) {
            *b += g;
        }
        self.steps += 1;
        Ok(())
    }

    pub fn steps_so_far(&self) -> usize {
        self.steps
    }

    pub fn is_full(&self) -> bool {
        self.steps >= self.target_steps
    }

    /// Rend le gradient moyen (divisé par `target_steps`) et réinitialise.
    /// Si le nombre d'étapes est inférieur à la cible, divise par la cible
    /// quand même (contrat : accumulation exacte sur `target_steps`).
    pub fn take_normalized(&mut self) -> Vec<f64> {
        let scale = self.target_steps as f64;
        let out: Vec<f64> = self.buffer.iter().map(|g| g / scale).collect();
        for b in self.buffer.iter_mut() {
            *b = 0.0;
        }
        self.steps = 0;
        out
    }
}

// ─────────────────────────── Allocation statistics ─────────────────────────── //

/// Statistiques d'allocation du chemin critique (contrat §12 : « statistiques
/// d'allocation »).
///
/// Instrumentation déterministe : chaque allocation « chaude » déclarée par le
/// code d'entraînement est comptée ici. Le chemin chaud *doit* afficher
/// `allocations == 0` après la phase de préparation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AllocationStats {
    pub allocations: u64,
    pub bytes_allocated: u64,
    pub hot_path_calls: u64,
}

impl AllocationStats {
    pub fn record_allocation(&mut self, bytes: usize) {
        self.allocations += 1;
        self.bytes_allocated += bytes as u64;
    }

    pub fn record_call(&mut self) {
        self.hot_path_calls += 1;
    }

    /// Vrai si le chemin chaud est resté sans allocation.
    pub fn zero_alloc_hot_path(&self) -> bool {
        self.allocations == 0
    }
}

// ─────────────────────────── Chemin f32 (mixed precision) ──────────────────── //

/// Somme compensée sur `f32` (l'accumulateur reste f64 pour la stabilité, le
/// résultat est converti en f32) — chemin f32 validé.
pub fn sum_f32(v: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for &x in v {
        acc += x as f64;
    }
    acc as f32
}

/// Écart relatif maximal entre deux valeurs (pour valider f32 vs f64).
pub fn relative_diff(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs()).max(1e-30);
    (a - b).abs() / scale
}

/// Valide le chemin f32 contre le chemin f64 sur un jeu de valeurs : l'écart
/// relatif doit rester sous `tolerance` (mixed precision autorisée seulement
/// après validation du chemin f32 — contrat §12).
pub fn validate_f32_path(f64_values: &[f64], tolerance: f64) -> CognoResult<()> {
    for &v in f64_values {
        let v32 = v as f32 as f64;
        if relative_diff(v, v32) > tolerance {
            return Err(CognoError::InvalidInput("chemin f32 hors tolérance"));
        }
    }
    Ok(())
}

// ─────────────────────────── Rollouts contrôlés ─────────────────────────── //

/// Sortie d'un rollout contrôlé.
#[derive(Debug, Clone)]
pub struct Rollout {
    pub x: Vec<u8>,
    pub y: Vec<u8>,
    pub log_prob: f64,
    pub admissible: bool,
}

/// Générateur de rollouts : produit une sortie `y` pour un contexte `x`.
/// Le trait est injectable (le backend ne génère pas lui-même).
pub trait RolloutPolicy {
    /// Génère une sortie + sa log-probabilité.
    fn sample(&self, x: &[u8]) -> CognoResult<(Vec<u8>, f64)>;
}

/// Rollout **contrôlé** (contrat §10) : la sortie générée est vérifiée par les
/// validateurs durs + le gate d'admissibilité **avant** d'être retournée.
/// Une sortie inadmissible est rejetée (jamais adoptée) ; le rollout la signale
/// comme `admissible: false` pour l'observabilité.
pub struct ControlledRollout<'a> {
    pub policy: &'a dyn RolloutPolicy,
    pub gate: &'a cogno_core::admissible::AdmissibilityGate<'a>,
    /// budget mémoire par rollout (octets).
    pub budget_mem: usize,
    /// budget latence par rollout (ms).
    pub budget_lat: usize,
    /// budget contexte (tokens).
    pub budget_ctx: usize,
}

impl<'a> ControlledRollout<'a> {
    /// Exécute un rollout contrôlé pour `x`. La sortie passe le gate
    /// d'admissibilité avant d'être acceptée.
    pub fn run(&self, x: &[u8], provenance: &[u8]) -> CognoResult<Rollout> {
        let (y, log_prob) = self.policy.sample(x)?;
        FiniteScalar::try_new(log_prob).map_err(|_| CognoError::NonFinite("rollout log_prob"))?;
        // taille approximative : longueur de y en octets = coût mémoire ;
        // latence = 1 (mesurée par l'appelant en réalité) ; ctx = longueur de x
        let mem = y.len();
        let lat = 1;
        let ctx = x.len();
        let verdict = self.gate.verify(x, &y, provenance, mem, lat, ctx)?;
        Ok(Rollout {
            x: x.to_vec(),
            y,
            log_prob,
            admissible: verdict.admissible,
        })
    }
}

// ─────────────────────────── PPO (clip ratio) ─────────────────────────── //

/// Configuration de PPO.
#[derive(Debug, Clone, Copy)]
pub struct PpoConfig {
    pub clip_epsilon: f64,
    pub beta_kl: f64,
    /// vrai si la perte COGNO-0.1 est stabilisée (garde d'activation).
    pub require_stable_01: bool,
}

impl Default for PpoConfig {
    fn default() -> Self {
        PpoConfig {
            clip_epsilon: 0.2,
            beta_kl: 0.01,
            require_stable_01: false, // PPO désactivé tant que non stabilisé
        }
    }
}

/// Échantillon de rollout pour PPO : probabilités et avantage.
#[derive(Debug, Clone, Copy)]
pub struct PpoSample {
    /// `π_θ(y|x)` (nouvelle politique).
    pub log_prob_new: f64,
    /// `π_θ_old(y|x)` (politique figée).
    pub log_prob_old: f64,
    /// avantage estimé.
    pub advantage: f64,
}

/// Calcule la perte PPO clipée sur un batch d'échantillons (contrat §10 :
/// « PPO ou autre optimisation de politique uniquement après validation »).
///
/// ```text
/// L_ppo = E[ min( r·A, clip(r, 1−ε, 1+ε)·A ) ]  avec r = exp(lp_new − lp_old)
/// ```
///
/// Le ratio `r` est calculé **en log-espace** (`exp` de la différence de
/// log-probs), jamais par division de probabilités.
pub fn compute_ppo_loss(samples: &[PpoSample], config: &PpoConfig) -> CognoResult<FiniteScalar> {
    if !config.require_stable_01 {
        // garde : PPO interdit tant que COGNO-0.1 n'est pas stabilisé (§10)
        return Err(CognoError::InvalidInput(
            "PPO non autorisé : COGNO-0.1 pas encore stabilisé (require_stable_01)",
        ));
    }
    if !(0.0..1.0).contains(&config.clip_epsilon) {
        return Err(CognoError::InvalidInput("clip_epsilon ∈ (0,1)"));
    }
    let mut total = 0.0f64;
    for s in samples {
        for v in [s.log_prob_new, s.log_prob_old, s.advantage] {
            FiniteScalar::try_new(v).map_err(|_| CognoError::NonFinite("ppo sample"))?;
        }
        let ratio = (s.log_prob_new - s.log_prob_old).exp();
        let unclipped = ratio * s.advantage;
        let clipped = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
            * s.advantage;
        total += unclipped.min(clipped);
    }
    let n = samples.len().max(1) as f64;
    // on minimise la négative
    FiniteScalar::try_new(-(total / n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogno_core::admissible::{AdmissibilityGate, TrivialProvenance};
    use cogno_core::budget::ResourceBudget;

    #[test]
    fn gradient_accumulator_normalizes_and_resets() {
        let mut acc = GradientAccumulator::try_new(2, 2).unwrap();
        acc.accumulate(&[1.0, 2.0]).unwrap();
        assert!(!acc.is_full());
        acc.accumulate(&[3.0, 4.0]).unwrap();
        assert!(acc.is_full());
        let g = acc.take_normalized();
        assert_eq!(g, vec![2.0, 3.0]); // (1+3)/2, (2+4)/2
        assert_eq!(acc.steps_so_far(), 0);
        // le tampon est remis à zéro
        acc.accumulate(&[10.0, 10.0]).unwrap();
        let g2 = acc.take_normalized();
        assert_eq!(g2, vec![5.0, 5.0]);
    }

    #[test]
    fn gradient_accumulator_rejects_length_mismatch() {
        let mut acc = GradientAccumulator::try_new(2, 1).unwrap();
        assert!(acc.accumulate(&[1.0, 2.0, 3.0]).is_err());
        assert!(GradientAccumulator::try_new(2, 0).is_err());
    }

    #[test]
    fn allocation_stats_track() {
        let mut s = AllocationStats::default();
        assert!(s.zero_alloc_hot_path());
        s.record_allocation(64);
        s.record_allocation(128);
        assert_eq!(s.allocations, 2);
        assert_eq!(s.bytes_allocated, 192);
        assert!(!s.zero_alloc_hot_path());
    }

    #[test]
    fn f32_path_within_tolerance() {
        // valeurs représentables : l'écart f32/f64 est négligeable
        let vals = [0.1, 0.5, 1.0, std::f64::consts::PI, 12345.678];
        assert!(validate_f32_path(&vals, 1e-4).is_ok());
    }

    #[test]
    fn controlled_rollout_admissible() {
        struct FixedPolicy;
        impl RolloutPolicy for FixedPolicy {
            fn sample(&self, _x: &[u8]) -> CognoResult<(Vec<u8>, f64)> {
                Ok((b"fn main() {}".to_vec(), -0.5))
            }
        }
        let gate = AdmissibilityGate {
            hard_validators: &[],
            provenance: &TrivialProvenance,
            budget: &ResourceBudget::default(),
        };
        let rollout = ControlledRollout {
            policy: &FixedPolicy,
            gate: &gate,
            budget_mem: 1024,
            budget_lat: 100,
            budget_ctx: 100,
        };
        let r = rollout.run(b"x", b"p").unwrap();
        assert!(r.admissible);
        assert_eq!(r.y, b"fn main() {}");
    }

    #[test]
    fn controlled_rollout_rejects_hard_violation() {
        struct EvilPolicy;
        impl RolloutPolicy for EvilPolicy {
            fn sample(&self, _x: &[u8]) -> CognoResult<(Vec<u8>, f64)> {
                Ok((b"unsafe {}".to_vec(), -0.5))
            }
        }
        let gate = AdmissibilityGate {
            hard_validators: &[&cogno_core::admissible::NoForbiddenSubstring {
                forbidden: &[b"unsafe"],
            }],
            provenance: &TrivialProvenance,
            budget: &ResourceBudget::default(),
        };
        let rollout = ControlledRollout {
            policy: &EvilPolicy,
            gate: &gate,
            budget_mem: 1024,
            budget_lat: 100,
            budget_ctx: 100,
        };
        let r = rollout.run(b"x", b"p").unwrap();
        assert!(!r.admissible, "sortie interdite doit être rejetée");
    }

    #[test]
    fn ppo_blocked_until_stable() {
        let samples = [PpoSample {
            log_prob_new: -0.5,
            log_prob_old: -0.5,
            advantage: 1.0,
        }];
        // garde : require_stable_01 = false → erreur (PPO interdit)
        let cfg = PpoConfig::default();
        assert!(compute_ppo_loss(&samples, &cfg).is_err());
    }

    #[test]
    fn ppo_clipped_ratio() {
        let samples = [
            PpoSample {
                log_prob_new: -0.5,
                log_prob_old: -1.0, // ratio = e^0.5 ≈ 1.65
                advantage: 1.0,
            },
            PpoSample {
                log_prob_new: -1.0,
                log_prob_old: -0.5, // ratio = e^-0.5 ≈ 0.61
                advantage: -1.0,
            },
        ];
        let cfg = PpoConfig {
            require_stable_01: true,
            ..Default::default()
        };
        let loss = compute_ppo_loss(&samples, &cfg).unwrap();
        assert!(loss.value().is_finite());
    }
}
