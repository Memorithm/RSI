//! ⚙️ Loop Engineering — **L1 : pilote de boucle & critères d'arrêt**.
//!
//! Au-dessus de `RSIAgent::step`, un pilote configurable qui exécute la boucle
//! RSI jusqu'à un critère d'arrêt **motivé** : budget de pas, budget de temps,
//! cible de `SI_global` atteinte, **plateau** (convergence) ou **divergence**
//! (via [`crate::convergence`]). Premier maillon du chantier « Loop
//! Engineering » (cf. `docs/ROADMAP.md`).

use std::time::Instant;

use crate::agent::{RSIAgent, StepReport};
use crate::convergence::{ConvergenceDetector, Trend};
use crate::schedule::{LoopSchedule, MetaMeta};

/// Critères d'arrêt du pilote de boucle.
#[derive(Clone, Copy, Debug)]
pub struct LoopConfig {
    /// budget maximal de pas.
    pub max_steps: usize,
    /// arrêt si `SI_global ≥ target` (si défini).
    pub target_si: Option<f64>,
    /// fenêtre d'analyse de tendance (plateau/divergence).
    pub plateau_window: usize,
    /// seuil de pente |slope| sous lequel on déclare un **plateau** (par pas).
    pub plateau_eps: f64,
    /// arrête sur divergence (pente fortement négative) si vrai.
    pub stop_on_divergence: bool,
    /// budget de temps optionnel (secondes).
    pub max_seconds: Option<f64>,
    /// **disjoncteur de criticité (L4)** : si `max_rpn` d'un pas dépasse ce
    /// seuil, la boucle s'arrête (`CircuitBreaker`). `None` = désactivé.
    pub breaker_rpn: Option<f64>,
    /// en cas de déclenchement du disjoncteur, restaure le dernier état sain
    /// (rollback) avant d'arrêter.
    pub rollback_on_breach: bool,
    /// **boucle méta-méta (L3)** : si défini, la boucle *adapte les cadences*
    /// (`meta_interval`) selon la tendance au lieu de s'arrêter au plateau —
    /// ralentit sur plateau (économie), accélère en progression/divergence.
    /// `None` = comportement d'origine (arrêt motivé).
    pub meta_meta: Option<MetaMeta>,
}

impl Default for LoopConfig {
    fn default() -> Self {
        LoopConfig {
            max_steps: 1000,
            target_si: None,
            plateau_window: 12,
            plateau_eps: 1e-4,
            stop_on_divergence: true,
            max_seconds: None,
            breaker_rpn: None,
            rollback_on_breach: false,
            meta_meta: None,
        }
    }
}

/// Raison de l'arrêt de la boucle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    MaxSteps,
    TargetReached,
    Plateau,
    Diverged,
    Timeout,
    /// disjoncteur de criticité déclenché (L4).
    CircuitBreaker,
    /// arrêt demandé par l'observateur (veto human-in-the-loop, L6).
    Vetoed,
}

/// ⚙️ **L6 — plan de contrôle observable**. Observateur recevant chaque
/// transition de boucle ; peut **véto**er la poursuite (human-in-the-loop).
pub trait LoopObserver {
    /// Appelé après chaque pas. Retourne `false` pour **arrêter** la boucle.
    fn on_step(&mut self, _report: &StepReport) -> bool {
        true
    }
    /// Appelé une fois la boucle terminée.
    fn on_stop(&mut self, _reason: StopReason, _steps: usize) {}
}

/// Observateur neutre (no-op).
impl LoopObserver for () {}

/// Résultat d'un run piloté.
#[derive(Clone, Debug)]
pub struct LoopOutcome {
    pub reports: Vec<StepReport>,
    pub reason: StopReason,
    pub steps: usize,
    /// pente finale de `SI_global` sur la fenêtre.
    pub final_slope: f64,
}

impl RSIAgent {
    /// Exécute la boucle jusqu'à un critère d'arrêt (L1). Retourne la
    /// trajectoire et la **raison** motivée de l'arrêt.
    pub fn run_until(&mut self, cfg: &LoopConfig) -> LoopOutcome {
        self.run_until_observed(cfg, &mut ())
    }

    /// Variante observée (L6) : un [`LoopObserver`] reçoit chaque pas et peut
    /// vétoer la poursuite (human-in-the-loop).
    pub fn run_until_observed<O: LoopObserver>(
        &mut self,
        cfg: &LoopConfig,
        observer: &mut O,
    ) -> LoopOutcome {
        let start = Instant::now();
        let mut det = ConvergenceDetector::new(cfg.plateau_window);
        let mut reports = Vec::new();
        let mut reason = StopReason::MaxSteps;
        // dernier état sain (pour rollback du disjoncteur L4)
        let mut last_good = self.snapshot();

        for _ in 0..cfg.max_steps {
            let r = self.step();
            det.push(r.si_global);
            let si = r.si_global;
            let max_rpn = r.max_rpn;
            reports.push(r);

            // §L6 — observateur : peut vétoer la poursuite (HITL)
            if !observer.on_step(reports.last().unwrap()) {
                reason = StopReason::Vetoed;
                break;
            }

            // §L4 — disjoncteur de criticité
            if let Some(thr) = cfg.breaker_rpn {
                if max_rpn > thr {
                    if cfg.rollback_on_breach {
                        self.restore(&last_good); // retour au dernier état sain
                    }
                    reason = StopReason::CircuitBreaker;
                    break;
                }
                // mémorise l'état comme « sain » tant qu'on est loin du seuil
                if max_rpn < 0.5 * thr {
                    last_good = self.snapshot();
                }
            }

            if let Some(t) = cfg.target_si {
                if si >= t {
                    reason = StopReason::TargetReached;
                    break;
                }
            }
            if let Some(secs) = cfg.max_seconds {
                if start.elapsed().as_secs_f64() >= secs {
                    reason = StopReason::Timeout;
                    break;
                }
            }
            if det.filled() {
                let trend = det.trend(cfg.plateau_eps);
                if let Some(mm) = cfg.meta_meta {
                    // §L3 — méta-méta : au lieu de s'arrêter, on révise les
                    // cadences (ralentit sur plateau, accélère sinon) et on
                    // poursuit. Divergence dure → arrêt reste possible.
                    let sched = LoopSchedule::new(self.meta_interval, self.substrate_interval);
                    mm.adapt(sched, trend).apply(self);
                    if matches!(trend, Trend::Diverging) && cfg.stop_on_divergence {
                        reason = StopReason::Diverged;
                        break;
                    }
                } else {
                    match trend {
                        Trend::Plateau => {
                            reason = StopReason::Plateau;
                            break;
                        }
                        Trend::Diverging if cfg.stop_on_divergence => {
                            reason = StopReason::Diverged;
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        observer.on_stop(reason, reports.len());
        LoopOutcome {
            steps: reports.len(),
            final_slope: det.slope(),
            reports,
            reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_on_target() {
        let mut agent = RSIAgent::demo(2026);
        let start = agent.si_global();
        let cfg = LoopConfig {
            max_steps: 500,
            target_si: Some(start + 0.05),
            ..LoopConfig::default()
        };
        let out = agent.run_until(&cfg);
        assert_eq!(out.reason, StopReason::TargetReached);
        assert!(out.reports.last().unwrap().si_global >= start + 0.05);
        assert!(out.steps < 500);
    }

    #[test]
    fn stops_on_plateau_before_max() {
        // l'agent converge vers l'attracteur → plateau détecté avant max_steps
        let mut agent = RSIAgent::demo(7);
        let cfg = LoopConfig {
            max_steps: 2000,
            plateau_window: 15,
            plateau_eps: 1e-4,
            ..LoopConfig::default()
        };
        let out = agent.run_until(&cfg);
        assert_eq!(out.reason, StopReason::Plateau);
        assert!(out.steps < 2000, "doit s'arrêter au plateau, pas au budget");
        assert!(out.final_slope.abs() < 1e-3);
    }

    #[test]
    fn circuit_breaker_trips_and_rolls_back() {
        let mut agent = RSIAgent::demo(2026);
        let cfg = LoopConfig {
            max_steps: 400,
            breaker_rpn: Some(0.1), // seuil bas → déclenchement garanti
            rollback_on_breach: true,
            plateau_window: 9999, // neutralise l'arrêt plateau
            ..LoopConfig::default()
        };
        let out = agent.run_until(&cfg);
        assert_eq!(out.reason, StopReason::CircuitBreaker);
        // rollback : l'état de l'agent est revenu en deçà du dernier pas exécuté
        assert!(agent.t < out.steps, "rollback doit ramener t en arrière");
    }

    #[test]
    fn observer_veto_stops_loop() {
        // observateur qui coupe après 5 pas (human-in-the-loop)
        struct Veto {
            seen: usize,
        }
        impl LoopObserver for Veto {
            fn on_step(&mut self, _r: &StepReport) -> bool {
                self.seen += 1;
                self.seen < 5
            }
        }
        let mut agent = RSIAgent::demo(1);
        let mut obs = Veto { seen: 0 };
        let cfg = LoopConfig {
            max_steps: 1000,
            plateau_window: 9999,
            ..LoopConfig::default()
        };
        let out = agent.run_until_observed(&cfg, &mut obs);
        assert_eq!(out.reason, StopReason::Vetoed);
        assert_eq!(out.steps, 5);
        assert_eq!(obs.seen, 5);
    }

    #[test]
    fn meta_meta_adapts_cadence_instead_of_stopping() {
        // Référence : sans méta-méta, l'agent convergent s'arrête au plateau.
        let mut base = RSIAgent::demo(7);
        let base_cfg = LoopConfig {
            max_steps: 2000,
            plateau_window: 15,
            ..LoopConfig::default()
        };
        assert_eq!(base.run_until(&base_cfg).reason, StopReason::Plateau);

        // Avec méta-méta : même scénario, mais on ADAPTE les cadences au lieu de
        // s'arrêter → la méta ralentit sur plateau (interval croît) et la boucle
        // poursuit jusqu'au budget.
        let mut agent = RSIAgent::demo(7);
        assert_eq!(agent.meta_interval, 1);
        let cfg = LoopConfig {
            max_steps: 800,
            plateau_window: 15,
            meta_meta: Some(MetaMeta::default()),
            ..LoopConfig::default()
        };
        let out = agent.run_until(&cfg);
        assert_ne!(
            out.reason,
            StopReason::Plateau,
            "méta-méta ne doit pas s'arrêter au plateau"
        );
        assert!(
            agent.meta_interval > 1,
            "la cadence méta doit être ralentie sur plateau (interval={})",
            agent.meta_interval
        );
    }

    #[test]
    fn respects_max_steps() {
        let mut agent = RSIAgent::demo(1);
        let cfg = LoopConfig {
            max_steps: 10,
            plateau_window: 50, // jamais rempli → pas d'arrêt plateau
            target_si: None,
            ..LoopConfig::default()
        };
        let out = agent.run_until(&cfg);
        assert_eq!(out.reason, StopReason::MaxSteps);
        assert_eq!(out.steps, 10);
    }

    #[test]
    fn test_deficit_geometry_interceptor_continue() {
        let mut interceptor = DeficitGeometryInterceptor::new(5.0);
        // Déficit faible : phi=0.8, g=0.7 => écart 0.1
        let dec = interceptor.intercept_token(0.8, 0.7);
        assert_eq!(dec, InterceptorDecision::Continue);
        assert!((interceptor.cumulative_deficit - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_deficit_geometry_interceptor_pruning() {
        let mut interceptor = DeficitGeometryInterceptor::new(1.0);
        // Écart initial faible
        let dec1 = interceptor.intercept_token(0.5, 0.4); // écart 0.1
        assert_eq!(dec1, InterceptorDecision::Continue);

        // Écart fort, divergence majeure par rapport au substrat : phi=2.0, g=0.5 => écart 1.5
        let dec2 = interceptor.intercept_token(2.0, 0.5);
        assert_eq!(dec2, InterceptorDecision::Prune);
        assert!(interceptor.cumulative_deficit > 1.0);
    }
}

// ===================== GÉOMÉTRIE DU DÉFICIT EN TEMPS RÉEL ================= //

/// Décision prise par l'intercepteur en temps réel pour chaque token ou pas de génération.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterceptorDecision {
    /// Continuer la génération sur la branche courante.
    Continue,
    /// Éliger (prune) immédiatement la branche pour cause de divergence géométrique.
    Prune,
}

/// Intercepteur de géométrie du déficit s'injectant en temps réel dans le flux de génération de tokens.
/// Remplace avantageusement les rollbacks post-génération en stoppant immédiatement les branches divergentes.
pub struct DeficitGeometryInterceptor {
    /// Seuil maximal de déficit cumulatif autorisé sur une branche.
    pub max_cumulative_deficit: f64,
    /// Somme cumulative du déficit mesuré sur la trajectoire courante.
    pub cumulative_deficit: f64,
    /// Dernière valeur de la capacité enregistrée pour analyse de transition.
    pub last_val: f64,
}

impl DeficitGeometryInterceptor {
    /// Crée un nouvel intercepteur avec un seuil de déficit maximum fixé.
    pub fn new(max_cumulative_deficit: f64) -> Self {
        DeficitGeometryInterceptor {
            max_cumulative_deficit,
            cumulative_deficit: 0.0,
            last_val: 0.0,
        }
    }

    /// S'injecte dans la boucle de génération d'un token. Calcule en temps réel le déficit instantané
    /// entre la performance cognitive théorique `phi` (ex. charge du réseau d'attention) et la
    /// bande passante réelle autorisée par le substrat matériel `g`.
    ///
    /// Renvoie `InterceptorDecision::Prune` pour arrêter immédiatement la branche si la trajectoire
    /// s'écarte trop des contraintes mathématiques fixées par `max_cumulative_deficit`.
    pub fn intercept_token(&mut self, phi: f64, g: f64) -> InterceptorDecision {
        // Calcul du déficit géométrique (Law of the Minimum) : écart entre cognition attendue et substrat effectif.
        let real_capability = phi.min(g);
        let instant_deficit = (phi - real_capability).abs();

        self.cumulative_deficit += instant_deficit;
        self.last_val = real_capability;

        if self.cumulative_deficit > self.max_cumulative_deficit {
            InterceptorDecision::Prune
        } else {
            InterceptorDecision::Continue
        }
    }
}
