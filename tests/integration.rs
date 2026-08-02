//! Tests d'intégration de bout en bout du système RSI.

use rsi::{CognitiveState, Dims, IntelligenceSurface, RSIAgent, Rng, StabilityConfig, Substrate};

/// La trajectoire complète d'un agent de démo respecte tous les garde-fous
/// (§4) à chaque pas et améliore l'intelligence globale (§1).
#[test]
fn full_trajectory_is_stable_and_improving() {
    let mut agent = RSIAgent::demo(2026);
    let cfg = agent.dynamics_cfg;
    let start = agent.si_global();

    let reports = agent.run(150);

    for r in &reports {
        // ‖ΔS‖ < λ
        assert!(
            r.appr.delta_norm <= cfg.lambda + 1e-9,
            "violation de ‖ΔS‖ < λ au pas {}",
            r.t
        );
        // non-régression : SI(t+1) ≥ SI(t) − ε  (sur l'étape d'apprentissage)
        assert!(
            r.appr.si_after >= r.appr.si_before - cfg.epsilon - 1e-9,
            "régression de SI au-delà de ε au pas {}",
            r.t
        );
        // SI_global ∈ [0, 1]
        assert!((0.0..=1.0).contains(&r.si_global));
        // P_eff ∈ (0, 1)
        assert!(r.p_eff > 0.0 && r.p_eff < 1.0);
    }

    let end = reports.last().unwrap().si_global;
    assert!(end > start, "l'agent doit progresser : {start} → {end}");
}

/// Un meilleur substrat de départ ⇒ intelligence globale finale plus élevée
/// (l'efficacité multiplicative P_eff lève le plafond physique g_x).
#[test]
fn better_substrate_yields_higher_ceiling() {
    let run_with = |hardware_scale: f64| -> f64 {
        let mut rng = Rng::new(123);
        let dims = Dims::uniform(6);
        let state = CognitiveState::random(dims, &mut rng, 0.08);
        let mut substrate = Substrate::default_with(4, 4, &mut rng);
        for h in substrate.h.iter_mut() {
            *h *= hardware_scale;
        }
        let surface = IntelligenceSurface::sample(1024, &mut rng);
        let meta = Box::new(rsi::MetaOptimizer::new(48, 0.12, 999));
        let mut agent = RSIAgent::new(state, substrate, surface, StabilityConfig::default(), meta);
        agent.run(150);
        agent.si_global()
    };

    let weak = run_with(1.0);
    let strong = run_with(2.0);
    assert!(
        strong >= weak,
        "substrat plus fort devrait atteindre un SI ≥ : faible={weak}, fort={strong}"
    );
}

/// La méta-révision est reproductible (même graine ⇒ même trajectoire).
#[test]
fn deterministic_given_seed() {
    let a = RSIAgent::demo(42).run(40).last().unwrap().si_global;
    let b = RSIAgent::demo(42).run(40).last().unwrap().si_global;
    assert_eq!(a, b);
}
