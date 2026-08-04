//! Gate d'admissibilité `F(x)` (contrat §2).
//!
//! ```text
//! F(x) = { y | H_h(x,y)=1 ∀h, P_prov(x,y)=1,
//!             C_mem(x,y) ≤ B_mem, C_lat(x,y) ≤ B_lat, C_ctx(x,y) ≤ B_ctx }
//! ```
//!
//! Les contraintes de `F(x)` sont **dures** : une violation entraîne un rejet
//! **avant classement et avant adoption**. Elles ne deviennent jamais des
//! pénalités compensables de la récompense (interdiction §18).

use crate::budget::ResourceBudget;
use crate::error::CognoResult;

/// Validateur symbolique **dur** : retourne `true` si la contrainte `h` est
/// satisfaite pour `(x, y)`. Un seul échec rend la sortie inadmissible.
///
/// L'implémentation est injectable (validateurs formels, structure, absence de
/// dérive syntaxique, …). Les règles **dures** restent ici, jamais dans le
/// terme symbolique souple (contrat §5).
pub trait HardValidator {
    /// Évalue la contrainte dure. `true` = satisfaite.
    fn validate(&self, x: &[u8], y: &[u8]) -> bool;

    /// Nom de la contrainte (traçabilité).
    fn name(&self) -> &'static str;
}

/// Validation de **provenance** : la sortie doit avoir une provenance valide
/// et vérifiée (ex. traçable jusqu'à un artefact autorisé).
pub trait ProvenanceValidator {
    fn valid(&self, x: &[u8], y: &[u8], provenance: &[u8]) -> bool;
}

/// Résultat du gate d'admissibilité.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmissibilityVerdict {
    pub admissible: bool,
    /// Nom de la première contrainte dure violée (diagnostic).
    pub first_violation: Option<&'static str>,
    /// Coûts mesurés (toujours remplis, pour le rapport).
    pub mem_bytes: usize,
    pub lat_ms: usize,
    pub ctx_tokens: usize,
}

/// Gate d'admissibilité `F(x)` : valideurs durs + provenance + budgets.
pub struct AdmissibilityGate<'a> {
    pub hard_validators: &'a [&'a dyn HardValidator],
    pub provenance: &'a dyn ProvenanceValidator,
    pub budget: &'a ResourceBudget,
}

impl<'a> AdmissibilityGate<'a> {
    /// Évalue si `(x, y, provenance)` est admissible. Rejette la sortie dès la
    /// première contrainte dure violée (ordre : validateurs, provenance,
    /// budgets). N'applique **jamais** de pénalité — c'est un gate binaire.
    pub fn verify(
        &self,
        x: &[u8],
        y: &[u8],
        provenance: &[u8],
        mem_bytes: usize,
        lat_ms: usize,
        ctx_tokens: usize,
    ) -> CognoResult<AdmissibilityVerdict> {
        for h in self.hard_validators {
            if !h.validate(x, y) {
                return Ok(AdmissibilityVerdict {
                    admissible: false,
                    first_violation: Some(h.name()),
                    mem_bytes,
                    lat_ms,
                    ctx_tokens,
                });
            }
        }
        if !self.provenance.valid(x, y, provenance) {
            return Ok(AdmissibilityVerdict {
                admissible: false,
                first_violation: Some("provenance"),
                mem_bytes,
                lat_ms,
                ctx_tokens,
            });
        }
        if !self
            .budget
            .permits(mem_bytes, lat_ms, ctx_tokens)
        {
            // identifie le premier budget dépassé (diagnostic)
            let viol = if mem_bytes > self.budget.mem_bytes {
                Some("budget_mem")
            } else if lat_ms > self.budget.lat_ms {
                Some("budget_lat")
            } else {
                Some("budget_ctx")
            };
            return Ok(AdmissibilityVerdict {
                admissible: false,
                first_violation: viol,
                mem_bytes,
                lat_ms,
                ctx_tokens,
            });
        }
        Ok(AdmissibilityVerdict {
            admissible: true,
            first_violation: None,
            mem_bytes,
            lat_ms,
            ctx_tokens,
        })
    }
}

/// Validateur dur trivial (accepte tout) — pour les tests et la composition.
pub struct TrivialHardValidator;
impl HardValidator for TrivialHardValidator {
    fn validate(&self, _x: &[u8], _y: &[u8]) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "trivial"
    }
}

/// Validateur de provenance trivial (accepte tout) — pour les tests.
pub struct TrivialProvenance;
impl ProvenanceValidator for TrivialProvenance {
    fn valid(&self, _x: &[u8], _y: &[u8], _p: &[u8]) -> bool {
        true
    }
}

/// Validateur dur : rejette tout artefact contenant un sous-ensemble interdit
/// (ex. `unsafe`) — exemple concret de contrainte dure injectable.
pub struct NoForbiddenSubstring {
    pub forbidden: &'static [&'static [u8]],
}

impl HardValidator for NoForbiddenSubstring {
    fn validate(&self, _x: &[u8], y: &[u8]) -> bool {
        !self.forbidden.iter().any(|f| contains_sub(y, f))
    }
    fn name(&self) -> &'static str {
        "no_forbidden_substring"
    }
}

fn contains_sub(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_rejects_on_hard_violation_before_ranking() {
        let gate = AdmissibilityGate {
            hard_validators: &[&NoForbiddenSubstring {
                forbidden: &[b"unsafe"],
            }],
            provenance: &TrivialProvenance,
            budget: &ResourceBudget::default(),
        };
        // violation dure → inadmissible (même avec budgets OK)
        let v = gate.verify(b"x", b"fn main() { unsafe {} }", b"p", 10, 1, 1).unwrap();
        assert!(!v.admissible);
        assert_eq!(v.first_violation, Some("no_forbidden_substring"));
        // conforme → admissible
        let v2 = gate.verify(b"x", b"fn main() {}", b"p", 10, 1, 1).unwrap();
        assert!(v2.admissible);
    }

    #[test]
    fn gate_rejects_on_budget_overflow() {
        let gate = AdmissibilityGate {
            hard_validators: &[],
            provenance: &TrivialProvenance,
            budget: &ResourceBudget::new(100, 100, 100),
        };
        let v = gate.verify(b"x", b"y", b"p", 101, 1, 1).unwrap();
        assert!(!v.admissible);
        assert_eq!(v.first_violation, Some("budget_mem"));
    }

    #[test]
    fn hard_constraint_never_compensable() {
        // même avec tous les budgets à zéro coût et provenance OK, une
        // violation dure reste un rejet — jamais une pénalité
        let gate = AdmissibilityGate {
            hard_validators: &[&NoForbiddenSubstring {
                forbidden: &[b"evil"],
            }],
            provenance: &TrivialProvenance,
            budget: &ResourceBudget::new(usize::MAX, usize::MAX, usize::MAX),
        };
        let v = gate.verify(b"x", b"evil payload", b"p", 0, 0, 0).unwrap();
        assert!(!v.admissible);
    }
}
