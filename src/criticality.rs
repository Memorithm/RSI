//! §7 — MODES DE DÉFAILLANCE & CRITICITÉ (AMDEC / FMECA)
//!
//! L'auto-amélioration *récursive* amplifie les défaillances : un système RSI a
//! besoin, au-delà des garde-fous ponctuels (‖ΔS‖<λ, non-régression ε), d'une
//! théorie explicite des modes de défaillance, de leur **criticité**, et d'un
//! objectif **ajusté au risque**.
//!
//! Pour chaque mode de défaillance `f`, on définit (échelle [0,1]) :
//!   - `severity`   S_f  : gravité de l'effet ;
//!   - `occurrence` O_f  : probabilité d'occurrence (dépend de l'état vivant) ;
//!   - `detection`  D_f  : *difficulté* de détection (0 = trivial, 1 = indétectable).
//!
//! Indice de priorité du risque (Risk Priority Number) :
//!   `RPN_f = S_f · O_f · D_f`
//!
//! Risque global et intelligence ajustée au risque :
//!   `Risk_global = moyenne_f RPN_f`
//!   `SI_safe = SI_global − κ · Risk_global`
//!
//! Les garde-fous λ/ε de §4 deviennent des cas particuliers de la maîtrise du
//! risque (modes f1/f2). Le mode le plus critique (`most_critical`) pilote le
//! **routage par criticité** (généralisation du routage par goulot).

/// Noms canoniques des modes de défaillance du RSI.
pub mod modes {
    pub const REGRESSION: &str = "regression_competence"; // f1
    pub const INSTABILITY: &str = "instabilite_divergence"; // f2
    pub const VALUE_DRIFT: &str = "derive_valeurs"; // f3
    pub const SUBSTRATE_COLLAPSE: &str = "effondrement_substrat"; // f4
    pub const GOODHART: &str = "goodhart_surajustement"; // f5
    pub const MEMORY_POISON: &str = "empoisonnement_memoire"; // f6
    pub const WIREHEADING: &str = "wireheading"; // f7
}

/// Signaux vivants extraits d'un pas RSI, qui pilotent les occurrences.
#[derive(Clone, Copy, Debug)]
pub struct RiskSignals {
    pub delta_si: f64,
    pub delta_norm: f64,
    pub lambda: f64,
    pub epsilon: f64,
    pub p_eff: f64,
    pub frac_limited_by_substrate: f64,
    pub autonomy: f64,  // moyenne de A
    pub alignment: f64, // moyenne de V
    pub backtracks: u32,
    /// écart |mesuré − analytique| de l'efficience logicielle (proxy wireheading).
    pub wireheading: f64,
    /// une mémoire contextuelle est-elle active (active le risque f6) ?
    pub memory_active: bool,
}

impl Default for RiskSignals {
    fn default() -> Self {
        RiskSignals {
            delta_si: 0.0,
            delta_norm: 0.0,
            lambda: 0.5,
            epsilon: 1e-3,
            p_eff: 0.5,
            frac_limited_by_substrate: 0.0,
            autonomy: 0.0,
            alignment: 0.0,
            backtracks: 0,
            wireheading: 0.0,
            memory_active: false,
        }
    }
}

/// Rapport AMDEC pour un mode.
#[derive(Clone, Debug)]
pub struct ModeReport {
    pub name: &'static str,
    pub severity: f64,
    pub occurrence: f64,
    pub detection: f64,
    pub rpn: f64,
}

/// Rapport de criticité agrégé d'un pas.
#[derive(Clone, Debug)]
pub struct RiskReport {
    pub modes: Vec<ModeReport>,
    pub risk_global: f64,
    pub max_rpn: f64,
    pub most_critical: &'static str,
}

/// Définition statique (sévérité, détection) d'un mode ; l'occurrence est
/// calculée dynamiquement à partir des signaux.
struct ModeDef {
    name: &'static str,
    severity: f64,
    detection: f64,
    occurrence: fn(&RiskSignals) -> f64,
}

#[inline]
fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// Modèle de risque AMDEC du RSI (catalogue de modes + objectif ajusté).
pub struct RiskModel {
    defs: Vec<ModeDef>,
    /// occurrence de base de l'empoisonnement mémoire quand une mémoire est active.
    memory_base: f64,
}

impl Default for RiskModel {
    fn default() -> Self {
        RiskModel::new()
    }
}

impl RiskModel {
    pub fn new() -> Self {
        let defs = vec![
            ModeDef {
                name: modes::REGRESSION,
                severity: 0.8,
                detection: 0.1, // détecté par le garde-fou ε
                occurrence: |s| clamp01(-s.delta_si / s.epsilon.max(1e-12)),
            },
            ModeDef {
                name: modes::INSTABILITY,
                severity: 0.9,
                detection: 0.1, // détecté par le garde-fou λ
                occurrence: |s| clamp01(s.delta_norm / s.lambda.max(1e-12)),
            },
            ModeDef {
                name: modes::VALUE_DRIFT,
                severity: 1.0,
                detection: 0.7, // difficile à détecter
                occurrence: |s| clamp01(s.autonomy - s.alignment),
            },
            ModeDef {
                name: modes::SUBSTRATE_COLLAPSE,
                severity: 0.6,
                detection: 0.3,
                occurrence: |s| clamp01((1.0 - s.p_eff) * s.frac_limited_by_substrate),
            },
            ModeDef {
                name: modes::GOODHART,
                severity: 0.8,
                detection: 0.8, // sournois
                occurrence: |s| clamp01(s.backtracks as f64 / 5.0),
            },
            ModeDef {
                name: modes::MEMORY_POISON,
                // Occurrence de base appliquée via `memory_base` dans `assess`
                // (unique source de vérité, évite le doublon 0.05.max(0.05)).
                severity: 0.6,
                detection: 0.7,
                occurrence: |_s| 0.0,
            },
            ModeDef {
                name: modes::WIREHEADING,
                severity: 1.0,
                detection: 0.9, // le plus difficile à détecter
                occurrence: |s| clamp01(s.wireheading),
            },
        ];
        RiskModel {
            defs,
            memory_base: 0.05,
        }
    }

    /// Évalue la criticité pour les signaux donnés.
    pub fn assess(&self, s: &RiskSignals) -> RiskReport {
        let mut modes = Vec::with_capacity(self.defs.len());
        let mut sum = 0.0;
        let mut max_rpn = 0.0;
        let mut most_critical = self.defs[0].name;
        for d in &self.defs {
            let mut occ = (d.occurrence)(s);
            if d.name == modes::MEMORY_POISON && s.memory_active {
                occ = occ.max(self.memory_base);
            }
            let rpn = d.severity * occ * d.detection;
            sum += rpn;
            if rpn > max_rpn {
                max_rpn = rpn;
                most_critical = d.name;
            }
            modes.push(ModeReport {
                name: d.name,
                severity: d.severity,
                occurrence: occ,
                detection: d.detection,
                rpn,
            });
        }
        let risk_global = sum / self.defs.len() as f64;
        RiskReport {
            modes,
            risk_global,
            max_rpn,
            most_critical,
        }
    }

    /// SI_safe = SI_global − κ · Risk_global.
    pub fn si_safe(&self, si_global: f64, report: &RiskReport, kappa: f64) -> f64 {
        si_global - kappa * report.risk_global
    }
}

/// Configuration du garde-fou de criticité (§7), analogue à `StabilityConfig`.
#[derive(Clone, Copy, Debug)]
pub struct RiskConfig {
    /// poids κ du risque dans SI_safe.
    pub kappa: f64,
    /// RPN maximal toléré : au-delà, l'agent adopte un pas conservateur.
    pub rpn_max: f64,
    /// hausse de Risk_global tolérée par pas (δ).
    pub risk_delta: f64,
    /// active les réponses ciblées par mode (réalignement V, plancher
    /// anti-wireheading, atténuation du gain). `false` = simple mesure du risque.
    pub active_response: bool,
}

impl Default for RiskConfig {
    fn default() -> Self {
        RiskConfig {
            kappa: 0.5,
            rpn_max: 0.4,
            risk_delta: 0.1,
            active_response: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpn_in_unit_interval_and_argmax() {
        let model = RiskModel::new();
        // ‖ΔS‖ = λ → instabilité max
        let s = RiskSignals {
            delta_norm: 0.5,
            lambda: 0.5,
            ..Default::default()
        };
        let r = model.assess(&s);
        assert!(r.modes.iter().all(|m| (0.0..=1.0).contains(&m.rpn)));
        assert!((0.0..=1.0).contains(&r.risk_global));
        // l'instabilité est sévérité 0.9 × occ 1.0 × det 0.1 = 0.09 ;
        // la dérive de valeurs domine si autonomy>alignment, sinon instabilité.
        assert!(r.max_rpn > 0.0);
    }

    #[test]
    fn value_drift_dominates_when_unaligned() {
        let model = RiskModel::new();
        // forte autonomie, faible alignement
        let s = RiskSignals {
            autonomy: 0.9,
            alignment: 0.1,
            ..Default::default()
        };
        let r = model.assess(&s);
        assert_eq!(r.most_critical, modes::VALUE_DRIFT);
    }

    #[test]
    fn si_safe_penalizes_risk() {
        let model = RiskModel::new();
        let s = RiskSignals {
            autonomy: 0.9,
            alignment: 0.0,
            ..Default::default()
        };
        let r = model.assess(&s);
        let safe = model.si_safe(0.5, &r, 1.0);
        assert!(safe < 0.5, "SI_safe doit être pénalisé : {safe}");
    }

    #[test]
    fn substrate_collapse_when_low_peff_and_bound() {
        let model = RiskModel::new();
        let s = RiskSignals {
            p_eff: 0.05,
            frac_limited_by_substrate: 1.0,
            ..Default::default()
        };
        let r = model.assess(&s);
        let sub = r
            .modes
            .iter()
            .find(|m| m.name == modes::SUBSTRATE_COLLAPSE)
            .unwrap();
        assert!(sub.occurrence > 0.9);
    }
}
