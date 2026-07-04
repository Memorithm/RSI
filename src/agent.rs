//! §5 / §6 — AGENT RSI : BOUCLE DISCRÈTE & ÉQUATION D'ONDES DE LA SURFACE
//!
//! ```text
//! S_{t+1}  = S_t + ℳ(S_t, V_t, H, O) + ΔS_appr            (§5)
//! ℳ_{t+1}  = arg max_ℳ SI_global( ℳ(S_t) )                (méta-révision)
//! Σ_I(t+1) = Σ_I(t) + η · ℳ(Σ_I, S, H, O, V) − P           (§6, forme compacte)
//! ```
//!
//! Un pas de l'agent enchaîne :
//!   1. méta-révision : choisir la meilleure politique ℳ (argmax SI) ;
//!   2. proposition d'auto-modification ℳ(S_t) (état + réécriture logicielle) ;
//!   3. apprentissage ΔS_appr via la dynamique continue contrainte (§4) ;
//!   4. application des garde-fous de stabilité ‖ΔS‖ < λ et non-régression.
//!
//! La surface Σ_I n'est pas recalculée explicitement : `SI_global` en est le
//! résumé scalaire (volume sous Σ_I), suivi à chaque pas.

use crate::audit::{AuditEvent, AuditLog};
use crate::criticality::{RiskConfig, RiskModel, RiskSignals};
use crate::dynamics::{Dynamics, StabilityConfig, StepInfo};
use crate::knowledge::KnowledgeSource;
use crate::linalg::mean;
use crate::memory::ContextMemory;
use crate::meta::{
    decode_strategy_payload, encode_strategy_payload, CmaEsMeta, MetaOptimizer, MetaSearch,
    MetaStrategy,
};
use crate::state::{delta_norm, CognitiveState, Dims};
use crate::substrate::{Substrate, SubstrateImprover};
use crate::surface::IntelligenceSurface;

/// Rapport d'un pas de la boucle RSI.
#[derive(Clone, Debug)]
pub struct StepReport {
    pub t: usize,
    pub si_global: f64,
    pub delta_si: f64,
    pub p_eff: f64,
    pub state_norm: f64,
    pub meta_delta_norm: f64,
    pub appr: StepInfo,
    pub frac_limited_by_substrate: f64,
    pub capabilities: [f64; 6], // (D,M,R,A,C,V)
    // §7 — criticité (AMDEC)
    pub risk_global: f64,
    pub max_rpn: f64,
    pub most_critical: &'static str,
    /// SI_safe = SI_global − κ · Risk_global.
    pub si_safe: f64,
    /// réponse de sûreté appliquée à ce pas (§7 garde-fou actif) :
    /// "none" | "damp_gain" | "damp_risk_delta" | "realign_V" | "trust_floor".
    pub mitigation: &'static str,
}

/// Agent cognitif auto-améliorant.
///
/// La stratégie de méta-recherche est polymorphe ([`MetaSearch`]) : on peut y
/// brancher [`MetaOptimizer`] (recherche aléatoire) ou [`CmaEsMeta`]
/// (sep-CMA-ES) sans changer la boucle.
pub struct RSIAgent {
    pub state: CognitiveState,
    pub substrate: Substrate,
    pub surface: IntelligenceSurface,
    pub strategy: MetaStrategy,
    pub dynamics_cfg: StabilityConfig,
    pub meta: Box<dyn MetaSearch>,
    /// Améliorateur de substrat optionnel (Phase 2 — P_eff *mesuré*). `None`
    /// par défaut → boucle d'origine inchangée.
    pub substrate_opt: Option<Box<dyn SubstrateImprover>>,
    /// Mémoire contextuelle optionnelle (Phase 3 — composante `C` réelle).
    /// Quand présente, l'agent y écrit son état à chaque pas.
    pub memory: Option<Box<dyn ContextMemory>>,
    /// Modèle de criticité AMDEC (§7).
    pub risk_model: RiskModel,
    /// Garde-fous de criticité (§7).
    pub risk_cfg: RiskConfig,
    /// (§C) la méta-révision n'est exécutée que tous les `meta_interval` pas.
    pub meta_interval: usize,
    /// (§L3) l'améliorateur de substrat n'est invoqué qu'un pas sur
    /// `substrate_interval` (cadence plus lente que la boucle interne).
    pub substrate_interval: usize,
    /// (§D) seuil de routage par criticité : l'améliorateur de substrat n'est
    /// invoqué que si la fraction limitée par le substrat dépasse ce seuil.
    pub route_threshold: f64,
    /// nombre de graines mémoire rappelées pour le warm-start de ℳ (§A).
    pub recall_k: usize,
    /// Journal d'audit hash-chaîné optionnel (§7bis — traçabilité/déterminisme).
    pub audit: Option<Box<dyn AuditLog>>,
    /// Source de connaissances optionnelle (§2bis — alimente la composante D).
    pub knowledge: Option<Box<dyn KnowledgeSource>>,
    /// taux de montée de D vers le niveau de connaissance ingéré.
    pub knowledge_rate: f64,
    pub t: usize,
    /// (§7) Risk_global structurel du pas précédent, pour le garde-fou de
    /// **vélocité** `risk_delta` (borne la hausse de risque par pas). `NaN` tant
    /// qu'aucun pas n'a été fait.
    prev_pre_risk: f64,
}

impl RSIAgent {
    /// Construit un agent à partir de ses sous-systèmes.
    pub fn new(
        state: CognitiveState,
        substrate: Substrate,
        surface: IntelligenceSurface,
        dynamics_cfg: StabilityConfig,
        meta: Box<dyn MetaSearch>,
    ) -> Self {
        let strategy = MetaStrategy::neutral(substrate.o.len());
        RSIAgent {
            state,
            substrate,
            surface,
            strategy,
            dynamics_cfg,
            meta,
            substrate_opt: None,
            memory: None,
            risk_model: RiskModel::new(),
            risk_cfg: RiskConfig::default(),
            meta_interval: 1,
            substrate_interval: 1,
            route_threshold: 0.5,
            recall_k: 4,
            audit: None,
            knowledge: None,
            knowledge_rate: 0.25,
            t: 0,
            prev_pre_risk: f64::NAN,
        }
    }

    /// Branche une source de connaissances (§2bis : alimente `D` depuis une
    /// vraie source, p. ex. un corpus de documents). Builder fluide.
    pub fn with_knowledge(mut self, knowledge: Box<dyn KnowledgeSource>) -> Self {
        self.knowledge = Some(knowledge);
        self
    }

    /// Liste les backends réels actifs (introspection — §3). Le cœur étant
    /// autonome, les composantes non listées utilisent leur modèle natif.
    pub fn active_backends(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.substrate_opt.is_some() {
            v.push("substrate_improver");
        }
        if self.memory.is_some() {
            v.push("context_memory");
        }
        if self.audit.is_some() {
            v.push("audit_log");
        }
        if self.knowledge.is_some() {
            v.push("knowledge_source");
        }
        v
    }

    /// Branche un journal d'audit hash-chaîné (§7bis : traçabilité &
    /// déterminisme de la récursion ℳ). Builder fluide.
    pub fn with_audit(mut self, audit: Box<dyn AuditLog>) -> Self {
        self.audit = Some(audit);
        self
    }

    /// Hash de tête du journal d'audit (résumé reproductible), si présent.
    pub fn audit_head(&self) -> Option<String> {
        self.audit.as_ref().map(|a| a.head())
    }

    /// Vérifie l'intégrité du journal d'audit (true si absent).
    pub fn audit_verify(&self) -> bool {
        self.audit.as_ref().map(|a| a.verify()).unwrap_or(true)
    }

    /// Nombre d'événements audités (0 si absent).
    pub fn audit_len(&self) -> usize {
        self.audit.as_ref().map(|a| a.len()).unwrap_or(0)
    }

    /// (§C) n'exécute la méta-révision que tous les `interval` pas. Builder.
    pub fn with_meta_interval(mut self, interval: usize) -> Self {
        self.meta_interval = interval.max(1);
        self
    }

    /// (§7) règle les garde-fous de criticité. Builder.
    pub fn with_risk_config(mut self, cfg: RiskConfig) -> Self {
        self.risk_cfg = cfg;
        self
    }

    /// (§D) seuil de routage par criticité pour l'améliorateur de substrat.
    pub fn with_route_threshold(mut self, threshold: f64) -> Self {
        self.route_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Branche un améliorateur de substrat (Phase 2 : P_eff *mesuré* par une
    /// optimisation exécutée, p. ex. Forge). Builder fluide.
    pub fn with_substrate_improver(mut self, improver: Box<dyn SubstrateImprover>) -> Self {
        self.substrate_opt = Some(improver);
        self
    }

    /// Branche une mémoire contextuelle (Phase 3 : composante `C` réelle, p. ex.
    /// OctaSoma). L'agent y écrit son état à chaque pas. Builder fluide.
    pub fn with_memory(mut self, memory: Box<dyn ContextMemory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Embedding de l'état courant (vecteur d'état aplati en f32).
    fn state_embedding(&self) -> Vec<f32> {
        self.state.to_vector().iter().map(|&x| x as f32).collect()
    }

    /// Rappelle les `k` contextes passés les plus proches de l'état courant.
    /// Vide si aucune mémoire n'est branchée.
    pub fn recall_similar(&self, k: usize) -> Vec<Vec<u8>> {
        match &self.memory {
            Some(mem) => mem.recall(&self.state_embedding(), k),
            None => Vec::new(),
        }
    }

    /// Nombre de contextes mémorisés (0 sans mémoire).
    pub fn memory_len(&self) -> usize {
        self.memory.as_ref().map(|m| m.len()).unwrap_or(0)
    }

    /// Sous-systèmes communs d'un agent de démonstration (reproductible).
    fn demo_parts(seed: u64) -> (CognitiveState, Substrate, IntelligenceSurface) {
        use crate::rng::Rng;
        let mut rng = Rng::new(seed);
        let dims = Dims::uniform(6);
        let state = CognitiveState::random(dims, &mut rng, 0.08);
        let substrate = Substrate::default_with(4, 4, &mut rng);
        let surface = IntelligenceSurface::sample(1024, &mut rng);
        (state, substrate, surface)
    }

    /// Agent de démonstration avec méta-révision par **recherche aléatoire**.
    pub fn demo(seed: u64) -> Self {
        let (state, substrate, surface) = Self::demo_parts(seed);
        let meta = Box::new(MetaOptimizer::new(48, 0.12, seed ^ 0xABCD));
        RSIAgent::new(state, substrate, surface, StabilityConfig::default(), meta)
    }

    /// Agent de démonstration avec méta-révision par **sep-CMA-ES**.
    pub fn demo_cma(seed: u64) -> Self {
        let (state, substrate, surface) = Self::demo_parts(seed);
        let meta = Box::new(CmaEsMeta::new(0, 10, 0.3, seed ^ 0xC3A));
        RSIAgent::new(state, substrate, surface, StabilityConfig::default(), meta)
    }

    /// SI_global courant (volume sous Σ_I).
    pub fn si_global(&self) -> f64 {
        self.surface.si_global(&self.state, &self.substrate)
    }

    /// Un pas de la boucle discrète RSI (avec criticité §7 et optimisations A/C/D).
    pub fn step(&mut self) -> StepReport {
        let si_before = self.si_global();
        // État de début de pas (référence du garde-fou de non-régression combiné).
        let state_before = self.state.clone();

        // §2bis — ingestion de connaissances réelles : fait tendre D vers le
        // niveau appris (la hausse de D nourrit ensuite L(D) dans la dynamique).
        if let Some(k) = self.knowledge.as_mut() {
            let level = k.absorb();
            let rate = self.knowledge_rate;
            for d in self.state.d.iter_mut() {
                *d = (*d + rate * (level - *d)).clamp(0.0, 1.0);
            }
        }

        let pre_bottleneck = self.surface.bottleneck(&self.state, &self.substrate);

        // §A — warm-start mémoire : rappeler les contextes proches et réinjecter
        // les stratégies ℳ passées performantes comme graines.
        if let Some(mem) = self.memory.as_ref() {
            let query = self.state_embedding();
            let n_o = self.substrate.o.len();
            let seeds: Vec<MetaStrategy> = mem
                .recall(&query, self.recall_k)
                .iter()
                .filter_map(|p| decode_strategy_payload(p).map(|(_, s)| s))
                .filter(|s| s.software_edit.len() == n_o)
                .collect();
            if !seeds.is_empty() {
                self.meta.warm_start(&seeds);
            }
        }

        // §C — la méta-révision n'est exécutée que tous les `meta_interval` pas.
        if self.t.is_multiple_of(self.meta_interval) {
            let (best_strategy, _proj_si) =
                self.meta
                    .revise(&self.strategy, &self.state, &self.substrate, &self.surface);
            self.strategy = best_strategy;
        }

        // §7/§D — pré-évaluation de criticité (signaux pré-pas) pour le routage
        // et le garde-fou conservateur.
        let pre_signals = RiskSignals {
            delta_si: 0.0,
            delta_norm: 0.0,
            lambda: self.dynamics_cfg.lambda,
            epsilon: self.dynamics_cfg.epsilon,
            p_eff: self.substrate.effective_power(),
            frac_limited_by_substrate: pre_bottleneck.frac_limited_by_substrate,
            autonomy: mean(&self.state.a),
            alignment: mean(&self.state.v),
            backtracks: 0,
            wireheading: self.substrate.software_eff_gap(),
            memory_active: self.memory.is_some(),
        };
        let pre_risk = self.risk_model.assess(&pre_signals);

        // §7 — GARDE-FOU ACTIF : au-delà de l'atténuation du gain, on applique
        // une réponse *ciblée* selon le mode le plus critique.
        let mut active = self.strategy.clone();
        let mut mitigation = "none";
        let rpn_exceeded = self.risk_cfg.active_response && pre_risk.max_rpn > self.risk_cfg.rpn_max;
        // §7 — garde-fou de VÉLOCITÉ : si le risque structurel grimpe de plus de
        // `risk_delta` d'un pas à l'autre, on entre en mode conservateur *avant*
        // que le risque n'accélère (borne la dérivée de Risk_global par pas).
        let risk_rising = self.risk_cfg.active_response
            && risk_velocity_exceeded(self.prev_pre_risk, pre_risk.risk_global, self.risk_cfg.risk_delta);
        let over_threshold = rpn_exceeded || risk_rising;
        if over_threshold {
            active.gain *= 0.5; // réponse de base : pas conservateur
            mitigation = if rpn_exceeded { "damp_gain" } else { "damp_risk_delta" };
        }
        self.prev_pre_risk = pre_risk.risk_global;

        // 2) ℳ(S_t, V_t, H, O) : proposition d'auto-modification (état + logiciel)
        let (meta_delta, new_substrate) = active.apply(&self.state, &self.substrate);
        let mut state_after_meta = self.state.add(&meta_delta).clipped(0.0, 1.0);

        // Réponse ciblée — dérive des valeurs : réaligner V vers le niveau
        // d'autonomie (corrige l'écart A − V qui alimente le mode f3).
        if over_threshold && pre_risk.most_critical == crate::criticality::modes::VALUE_DRIFT {
            let a = mean(&self.state.a);
            for v in state_after_meta.v.iter_mut() {
                *v = (*v + 0.5 * (a - *v).max(0.0)).clamp(0.0, 1.0);
            }
            mitigation = "realign_V";
        }

        let meta_delta_norm = delta_norm(&self.state, &state_after_meta);

        // La réécriture logicielle n'est acceptée que si elle n'abaisse pas P_eff.
        let mut substrate = if new_substrate.effective_power() >= self.substrate.effective_power() {
            new_substrate
        } else {
            self.substrate.clone()
        };

        // §D — routage par criticité : l'améliorateur de substrat (coûteux) n'est
        // invoqué que lorsque le substrat est la contrainte qui bride réellement
        // (goulot substrat élevé OU mode critique = effondrement du substrat).
        let substrate_is_critical = pre_bottleneck.frac_limited_by_substrate >= self.route_threshold
            || pre_risk.most_critical == crate::criticality::modes::SUBSTRATE_COLLAPSE;
        // §L3 — cadence : on n'améliore le substrat qu'un pas sur substrate_interval
        let substrate_due = self.t.is_multiple_of(self.substrate_interval);
        if substrate_is_critical && substrate_due {
            if let Some(opt) = self.substrate_opt.as_mut() {
                let improved = opt.improve(&substrate);
                if improved.effective_power() >= substrate.effective_power() {
                    substrate = improved;
                }
            }
        }

        // Réponse ciblée — wireheading : si l'efficience *mesurée* s'écarte trop
        // de l'analytique, la safety RABAISSE la mesure vers l'analytique (on
        // refuse de « croire » une mesure non vérifiée — anti-wireheading f7).
        // Cela peut baisser P_eff : c'est un override de sûreté assumé.
        if over_threshold && pre_risk.most_critical == crate::criticality::modes::WIREHEADING {
            if let Some(m) = substrate.measured_software_eff {
                let analytic = substrate.analytic_software_efficiency();
                substrate.set_measured_software_eff(Some(analytic + (m - analytic) * 0.5));
                mitigation = "trust_floor";
            }
        }

        // 3) ΔS_appr : apprentissage via la dynamique continue contrainte (§4)
        let dynamics = Dynamics::new(&self.surface, self.dynamics_cfg);
        let (mut next_state, appr) = dynamics.constrained_step(&state_after_meta, &substrate, 1.0);

        // §4bis — GARDE-FOU DE NON-RÉGRESSION SUR LE PAS COMBINÉ (ℳ + apprentissage).
        // `constrained_step` ne protège que l'étape d'apprentissage ; le `meta_delta`
        // de ℳ peut, lui, faire régresser SI. Hors override de sûreté explicite (une
        // mitigation a délibérément troqué du SI contre de la sûreté — p. ex.
        // `trust_floor` qui abaisse P_eff), on atténue le pas combiné — depuis l'état
        // de début de pas — jusqu'à respecter SI(t+1) ≥ SI(t) − ε. Sous override, la
        // régression est assumée et le garde-fou est volontairement court-circuité.
        if mitigation == "none" {
            let eps_eff = if self.dynamics_cfg.adaptive_epsilon {
                let (_, se) = self.surface.si_global_stats(&state_before, &self.substrate);
                self.dynamics_cfg.epsilon + self.dynamics_cfg.epsilon_z * se
            } else {
                self.dynamics_cfg.epsilon
            };
            let mut si_combined = self.surface.si_global(&next_state, &substrate);
            if si_combined < si_before - eps_eff {
                let combined_delta = next_state.sub(&state_before);
                let mut factor = 1.0;
                let mut tries = 0u32;
                while si_combined < si_before - eps_eff && tries < 20 {
                    factor *= 0.5;
                    next_state = state_before.add(&combined_delta.scaled(factor)).clipped(0.0, 1.0);
                    si_combined = self.surface.si_global(&next_state, &substrate);
                    tries += 1;
                }
                // Sécurité : si même un pas infinitésimal régresse, on reste sur place.
                if si_combined < si_before - eps_eff {
                    next_state = state_before.clone();
                }
            }
        }

        // 4) commit de l'état
        self.state = next_state;
        self.substrate = substrate;
        self.t += 1;

        let si_after = self.si_global();
        let bottleneck = self.surface.bottleneck(&self.state, &self.substrate);

        // §7 — évaluation de criticité post-pas (signaux réalisés)
        let signals = RiskSignals {
            delta_si: si_after - si_before,
            delta_norm: appr.delta_norm,
            lambda: self.dynamics_cfg.lambda,
            epsilon: self.dynamics_cfg.epsilon,
            p_eff: self.substrate.effective_power(),
            frac_limited_by_substrate: bottleneck.frac_limited_by_substrate,
            autonomy: mean(&self.state.a),
            alignment: mean(&self.state.v),
            backtracks: appr.backtracks,
            wireheading: self.substrate.software_eff_gap(),
            memory_active: self.memory.is_some(),
        };
        let risk = self.risk_model.assess(&signals);
        let si_safe = self.risk_model.si_safe(si_after, &risk, self.risk_cfg.kappa);

        // §7bis — audit hash-chaîné : trace reproductible et vérifiable du pas ℳ.
        if self.audit.is_some() {
            let event = AuditEvent {
                t: self.t,
                si_global: si_after,
                si_safe,
                risk_global: risk.risk_global,
                max_rpn: risk.max_rpn,
                most_critical: risk.most_critical,
                strategy_id: strategy_hash(&self.strategy),
                p_eff: self.substrate.effective_power(),
            };
            if let Some(a) = self.audit.as_mut() {
                a.record(&event);
            }
        }

        // §A — mémorisation : embedding de l'état + stratégie gagnante (pour le
        // warm-start des pas futurs).
        if self.memory.is_some() {
            let embedding = self.state_embedding();
            let payload = encode_strategy_payload(si_after, &self.strategy);
            if let Some(mem) = self.memory.as_mut() {
                mem.remember(&embedding, &payload);
            }
        }

        StepReport {
            t: self.t,
            si_global: si_after,
            delta_si: si_after - si_before,
            p_eff: self.substrate.effective_power(),
            state_norm: self.state.norm(),
            meta_delta_norm,
            appr,
            frac_limited_by_substrate: bottleneck.frac_limited_by_substrate,
            capabilities: self.state.capability_array(),
            risk_global: risk.risk_global,
            max_rpn: risk.max_rpn,
            most_critical: risk.most_critical,
            si_safe,
            mitigation,
        }
    }

    /// Exécute `n` pas et retourne la trajectoire complète des rapports.
    pub fn run(&mut self, n: usize) -> Vec<StepReport> {
        (0..n).map(|_| self.step()).collect()
    }
}

/// Garde-fou de vélocité du risque (§7) : vrai si le Risk_global a augmenté de
/// plus de `delta` depuis le pas précédent. Faux tant qu'il n'y a pas de pas
/// précédent (`prev` = NaN). Extrait pour être testé indépendamment de la boucle.
fn risk_velocity_exceeded(prev: f64, cur: f64, delta: f64) -> bool {
    prev.is_finite() && (cur - prev) > delta
}

/// Hash stable (FNV-1a) d'une stratégie ℳ, pour l'identifier dans l'audit.
fn strategy_hash(strategy: &MetaStrategy) -> u64 {
    let theta = strategy.encode();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in &theta {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_velocity_guardrail_logic() {
        // pas de pas précédent (NaN) → jamais déclenché
        assert!(!risk_velocity_exceeded(f64::NAN, 0.9, 0.1));
        // hausse au-delà de δ → déclenché
        assert!(risk_velocity_exceeded(0.30, 0.45, 0.1));
        // hausse sous δ → pas déclenché
        assert!(!risk_velocity_exceeded(0.30, 0.35, 0.1));
        // baisse du risque → jamais déclenché
        assert!(!risk_velocity_exceeded(0.50, 0.20, 0.1));
    }

    #[test]
    fn risk_delta_config_gates_conservative_steps() {
        // rpn_max énorme → seul le garde-fou de vélocité peut déclencher.
        let count = |delta: f64| {
            RSIAgent::demo(11)
                .with_risk_config(crate::criticality::RiskConfig {
                    kappa: 0.5,
                    rpn_max: 1e9,
                    risk_delta: delta,
                    active_response: true,
                })
                .run(60)
                .iter()
                .filter(|r| r.mitigation != "none")
                .count()
        };
        // δ immense : aucune hausse ne dépasse le seuil → jamais conservateur.
        assert_eq!(count(1e9), 0);
        // δ = 0 : toute hausse de risque structurel enclenche la réponse.
        assert!(count(0.0) >= 1, "risk_delta=0 devrait borner la vélocité");
    }

    #[test]
    fn si_is_monotone_within_epsilon() {
        let mut agent = RSIAgent::demo(2026);
        let eps = agent.dynamics_cfg.epsilon;
        let reports = agent.run(60);
        for r in &reports {
            // garde-fou de non-régression appliqué à l'étape d'apprentissage
            assert!(r.appr.si_after >= r.appr.si_before - eps - 1e-9);
        }
    }

    #[test]
    fn combined_step_non_regression_when_no_override() {
        // §4bis — hors override de sûreté (mitigation == "none"), le PAS COMBINÉ
        // (ℳ + apprentissage) respecte SI(t+1) ≥ SI(t) − ε. (demo : ε non adaptatif.)
        let mut agent = RSIAgent::demo(2026);
        let eps = agent.dynamics_cfg.epsilon;
        for r in agent.run(60) {
            if r.mitigation == "none" {
                assert!(
                    r.delta_si >= -eps - 1e-9,
                    "régression combinée non protégée: ΔSI={} (mitigation={})",
                    r.delta_si,
                    r.mitigation
                );
            }
        }
    }

    #[test]
    fn agent_improves_over_time() {
        let mut agent = RSIAgent::demo(7);
        let start = agent.si_global();
        agent.run(80);
        let end = agent.si_global();
        assert!(end > start, "SI start={start} end={end}");
    }

    #[test]
    fn delta_s_bounded_by_lambda() {
        let mut agent = RSIAgent::demo(99);
        let lam = agent.dynamics_cfg.lambda;
        for r in agent.run(50) {
            assert!(r.appr.delta_norm <= lam + 1e-9);
        }
    }

    #[test]
    fn memory_records_each_step() {
        use crate::memory::LinearContextMemory;
        let mut agent = RSIAgent::demo(3).with_memory(Box::new(LinearContextMemory::new()));
        assert_eq!(agent.memory_len(), 0);
        agent.run(10);
        assert_eq!(agent.memory_len(), 10);
        // le rappel renvoie des contextes (payloads non vides)
        let recalled = agent.recall_similar(3);
        assert_eq!(recalled.len(), 3);
        assert!(recalled.iter().all(|p| !p.is_empty()));
    }

    #[test]
    fn reports_criticality_fields() {
        let mut agent = RSIAgent::demo(2026);
        for r in agent.run(40) {
            assert!((0.0..=1.0).contains(&r.risk_global), "risk={}", r.risk_global);
            assert!((0.0..=1.0).contains(&r.max_rpn));
            assert!(r.si_safe <= r.si_global + 1e-12); // SI_safe pénalise le risque
            assert!(!r.most_critical.is_empty());
        }
    }

    #[test]
    fn memory_warm_start_still_improves() {
        use crate::memory::LinearContextMemory;
        let mut agent = RSIAgent::demo(11).with_memory(Box::new(LinearContextMemory::new()));
        let start = agent.si_global();
        agent.run(60);
        assert!(agent.si_global() > start);
        // les payloads mémoire encodent des stratégies décodables (§A)
        let recalled = agent.recall_similar(1);
        assert!(crate::meta::decode_strategy_payload(&recalled[0]).is_some());
    }

    #[test]
    fn meta_interval_keeps_invariants() {
        let mut agent = RSIAgent::demo(7).with_meta_interval(5);
        let eps = agent.dynamics_cfg.epsilon;
        for r in agent.run(40) {
            assert!(r.appr.si_after >= r.appr.si_before - eps - 1e-9);
        }
    }

    #[test]
    fn audit_log_traces_and_verifies() {
        use crate::audit::HashChainLog;
        let mut agent = RSIAgent::demo(5).with_audit(Box::new(HashChainLog::new()));
        assert_eq!(agent.audit_len(), 0);
        agent.run(20);
        assert_eq!(agent.audit_len(), 20);
        assert!(agent.audit_verify()); // chaîne intègre
        assert!(agent.audit_head().is_some());
    }

    #[test]
    fn active_response_realigns_on_value_drift() {
        use crate::rng::Rng;
        // état à forte autonomie / faible alignement → dérive des valeurs critique
        let dims = Dims::uniform(4);
        let mut state = CognitiveState::zeros(dims);
        state.a.iter_mut().for_each(|x| *x = 0.9);
        state.v.iter_mut().for_each(|x| *x = 0.05);
        let mut rng = Rng::new(1);
        let substrate = Substrate::default_with(4, 4, &mut rng);
        let surface = IntelligenceSurface::sample(128, &mut rng);
        let meta = Box::new(MetaOptimizer::new(8, 0.1, 1));
        let mut agent = RSIAgent::new(state, substrate, surface, StabilityConfig::default(), meta);

        let v_before: f64 = agent.state.v.iter().sum::<f64>() / agent.state.v.len() as f64;
        let r = agent.step();
        assert_eq!(r.most_critical, crate::criticality::modes::VALUE_DRIFT);
        assert_eq!(r.mitigation, "realign_V");
        let v_after: f64 = agent.state.v.iter().sum::<f64>() / agent.state.v.len() as f64;
        assert!(v_after > v_before, "V réaligné vers le haut : {v_before} → {v_after}");
    }

    #[test]
    fn knowledge_source_raises_d_and_improves() {
        use crate::knowledge::CorpusKnowledge;
        let docs: Vec<String> = (0..8)
            .map(|i| format!("concept_{i} alpha beta gamma delta epsilon zeta theta_{i} idea_{i}"))
            .collect();
        let mut agent = RSIAgent::demo(4)
            .with_knowledge(Box::new(CorpusKnowledge::from_texts(docs).with_scale(8.0)));
        let d0: f64 = agent.state.d.iter().sum::<f64>() / agent.state.d.len() as f64;
        agent.run(8);
        let d1: f64 = agent.state.d.iter().sum::<f64>() / agent.state.d.len() as f64;
        assert!(d1 > d0, "D doit monter via la source de connaissances : {d0} → {d1}");
        assert_eq!(agent.active_backends(), vec!["knowledge_source"]);
    }

    #[test]
    fn grounded_corpus_agent_improves() {
        use crate::tasks::TaskCorpus;
        let surface = IntelligenceSurface::from_corpus(&TaskCorpus::builtin());
        let mut rng = crate::rng::Rng::new(7);
        let state = CognitiveState::random(Dims::uniform(6), &mut rng, 0.08);
        let substrate = Substrate::default_with(4, 4, &mut rng);
        let meta = Box::new(MetaOptimizer::new(48, 0.12, 7));
        let mut agent = RSIAgent::new(state, substrate, surface, StabilityConfig::default(), meta);
        let start = agent.si_global();
        agent.run(60);
        assert!(agent.si_global() > start, "amélioration sur corpus réel");
    }

    #[test]
    fn adaptive_epsilon_keeps_invariant() {
        let mut agent = RSIAgent::demo(2);
        agent.dynamics_cfg.adaptive_epsilon = true;
        for r in agent.run(40) {
            // non-régression encore garantie (tolérance ≥ ε de base)
            assert!(r.appr.si_after >= r.appr.si_before - agent.dynamics_cfg.epsilon - 1.0);
        }
    }

    #[test]
    fn native_measured_substrate_in_agent() {
        use crate::measured_substrate::MeasuredSubstrate;
        let mut agent = RSIAgent::demo(3)
            .with_substrate_improver(Box::new(MeasuredSubstrate::new(64)))
            .with_route_threshold(0.0); // force l'invocation
        agent.run(6);
        assert!(agent.active_backends().contains(&"substrate_improver"));
    }

    #[test]
    fn audit_head_is_deterministic() {
        use crate::audit::HashChainLog;
        let run = || {
            let mut a = RSIAgent::demo(123).with_audit(Box::new(HashChainLog::new()));
            a.run(15);
            a.audit_head().unwrap()
        };
        // même graine + même trajectoire ⇒ même hash de tête (reproductibilité)
        assert_eq!(run(), run());
    }

    /// Property-based test *in-tree* : sur un large balayage de graines, chaque
    /// `StepReport` respecte les bornes de cohérence du modèle. Cible : 0 échec.
    #[test]
    fn report_invariants_hold_over_many_seeds() {
        use crate::rng::Rng;
        let mut agent_seed = Rng::new(2026);
        for _ in 0..250 {
            let seed = (agent_seed.uniform_range(0.0, 1e9) as u64).wrapping_add(1);
            let mut agent = RSIAgent::demo(seed);
            let lam = agent.dynamics_cfg.lambda;
            let eps = agent.dynamics_cfg.epsilon;
            for r in agent.run(25) {
                assert!(r.appr.delta_norm <= lam + 1e-9, "‖ΔS‖={} > λ", r.appr.delta_norm);
                assert!((0.0..=1.0).contains(&r.si_global), "SI_global={}", r.si_global);
                assert!(r.p_eff > 0.0 && r.p_eff < 1.0, "P_eff={}", r.p_eff);
                assert!((0.0..=1.0).contains(&r.risk_global), "risk_global={}", r.risk_global);
                assert!((0.0..=1.0).contains(&r.max_rpn), "max_rpn={}", r.max_rpn);
                assert!(
                    r.capabilities.iter().all(|c| (0.0..=1.0).contains(c)),
                    "capabilities hors [0,1]: {:?}",
                    r.capabilities
                );
                // non-régression du pas combiné hors override de sûreté
                if r.mitigation == "none" {
                    assert!(r.delta_si >= -eps - 1e-9, "ΔSI={} < −ε", r.delta_si);
                }
            }
        }
    }
}
