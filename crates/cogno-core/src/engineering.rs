//! Typed hard-gate contract for engineering candidates.
//!
//! This module is deliberately separate from ranking. Build/test/parity/
//! provenance/determinism/resource/policy evidence is resolved into a binary
//! admissibility verdict first. A failed or unknown required check is never a
//! scalar penalty and can never be compensated by benchmark performance.

/// State of one required engineering hard check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    Pass,
    Fail,
    Unknown,
}

impl CheckStatus {
    pub fn is_pass(self) -> bool {
        matches!(self, Self::Pass)
    }
}

/// Canonical hard-gate category used for diagnostics and audit trails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineeringGateKind {
    Build,
    RequiredTests,
    NumericalParity,
    Provenance,
    DeterministicContract,
    ResourceBudget,
    Policy,
}

/// Evidence attached to one hard gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineeringCheck {
    pub status: CheckStatus,
    pub evidence: String,
}

impl EngineeringCheck {
    pub fn pass(evidence: impl Into<String>) -> Self {
        Self {
            status: CheckStatus::Pass,
            evidence: evidence.into(),
        }
    }

    pub fn fail(evidence: impl Into<String>) -> Self {
        Self {
            status: CheckStatus::Fail,
            evidence: evidence.into(),
        }
    }

    pub fn unknown(evidence: impl Into<String>) -> Self {
        Self {
            status: CheckStatus::Unknown,
            evidence: evidence.into(),
        }
    }
}

/// Named policy check. Every entry is required: `Unknown` is fail-closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineeringPolicyCheck {
    pub name: String,
    pub check: EngineeringCheck,
}

impl EngineeringPolicyCheck {
    pub fn new(name: impl Into<String>, check: EngineeringCheck) -> Self {
        Self {
            name: name.into(),
            check,
        }
    }
}

/// Complete hard-gate evidence for one engineering candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineeringAdmissibility {
    pub build: EngineeringCheck,
    pub required_tests: EngineeringCheck,
    pub numerical_parity: EngineeringCheck,
    pub provenance: EngineeringCheck,
    pub deterministic_contract: EngineeringCheck,
    pub resource_budget: EngineeringCheck,
    pub policy_checks: Vec<EngineeringPolicyCheck>,
}

/// Binary COGNO decision plus the first failing/unknown gate for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineeringAdmissibilityVerdict {
    pub admissible: bool,
    pub first_violation: Option<EngineeringGateKind>,
    pub first_policy_violation: Option<String>,
}

impl EngineeringAdmissibility {
    /// Resolve hard evidence before any ranking or benchmark comparison.
    pub fn verdict(&self) -> EngineeringAdmissibilityVerdict {
        if !self.build.status.is_pass() {
            return rejected(EngineeringGateKind::Build);
        }
        if !self.required_tests.status.is_pass() {
            return rejected(EngineeringGateKind::RequiredTests);
        }
        if !self.numerical_parity.status.is_pass() {
            return rejected(EngineeringGateKind::NumericalParity);
        }
        if !self.provenance.status.is_pass() {
            return rejected(EngineeringGateKind::Provenance);
        }
        if !self.deterministic_contract.status.is_pass() {
            return rejected(EngineeringGateKind::DeterministicContract);
        }
        if !self.resource_budget.status.is_pass() {
            return rejected(EngineeringGateKind::ResourceBudget);
        }
        for policy in &self.policy_checks {
            if !policy.check.status.is_pass() {
                return EngineeringAdmissibilityVerdict {
                    admissible: false,
                    first_violation: Some(EngineeringGateKind::Policy),
                    first_policy_violation: Some(policy.name.clone()),
                };
            }
        }
        EngineeringAdmissibilityVerdict {
            admissible: true,
            first_violation: None,
            first_policy_violation: None,
        }
    }

    /// Explicit name for the ordering contract used by P4.2: ranking may run
    /// only after this returns true.
    pub fn permits_ranking(&self) -> bool {
        self.verdict().admissible
    }
}

fn rejected(kind: EngineeringGateKind) -> EngineeringAdmissibilityVerdict {
    EngineeringAdmissibilityVerdict {
        admissible: false,
        first_violation: Some(kind),
        first_policy_violation: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_pass() -> EngineeringAdmissibility {
        EngineeringAdmissibility {
            build: EngineeringCheck::pass("cargo build exit=0"),
            required_tests: EngineeringCheck::pass("required suite 42/42"),
            numerical_parity: EngineeringCheck::pass("max_abs=0"),
            provenance: EngineeringCheck::pass("attestation verified"),
            deterministic_contract: EngineeringCheck::pass("replay bit-exact"),
            resource_budget: EngineeringCheck::pass("within frozen ceilings"),
            policy_checks: vec![EngineeringPolicyCheck::new(
                "frozen-tests-unchanged",
                EngineeringCheck::pass("hash match"),
            )],
        }
    }

    #[test]
    fn all_required_checks_must_pass_before_ranking() {
        let evidence = all_pass();
        let verdict = evidence.verdict();
        assert!(verdict.admissible);
        assert!(evidence.permits_ranking());
        assert_eq!(verdict.first_violation, None);
    }

    #[test]
    fn failed_hard_gate_is_never_compensable() {
        let mut evidence = all_pass();
        evidence.numerical_parity = EngineeringCheck::fail("oracle mismatch");
        let verdict = evidence.verdict();
        assert!(!verdict.admissible);
        assert!(!evidence.permits_ranking());
        assert_eq!(
            verdict.first_violation,
            Some(EngineeringGateKind::NumericalParity)
        );
    }

    #[test]
    fn unknown_required_gate_fails_closed() {
        let mut evidence = all_pass();
        evidence.provenance = EngineeringCheck::unknown("attestation unavailable");
        let verdict = evidence.verdict();
        assert!(!verdict.admissible);
        assert_eq!(
            verdict.first_violation,
            Some(EngineeringGateKind::Provenance)
        );
    }

    #[test]
    fn resource_unknown_is_not_treated_as_zero_cost() {
        let mut evidence = all_pass();
        evidence.resource_budget = EngineeringCheck::unknown("device telemetry missing");
        let verdict = evidence.verdict();
        assert!(!verdict.admissible);
        assert_eq!(
            verdict.first_violation,
            Some(EngineeringGateKind::ResourceBudget)
        );
    }

    #[test]
    fn policy_breakdown_retains_exact_first_failure() {
        let mut evidence = all_pass();
        evidence.policy_checks.push(EngineeringPolicyCheck::new(
            "allowlist",
            EngineeringCheck::fail("candidate touched .github/workflows/ci.yml"),
        ));
        let verdict = evidence.verdict();
        assert!(!verdict.admissible);
        assert_eq!(verdict.first_violation, Some(EngineeringGateKind::Policy));
        assert_eq!(
            verdict.first_policy_violation.as_deref(),
            Some("allowlist")
        );
    }

    #[test]
    fn check_order_is_stable_for_diagnostics() {
        let mut evidence = all_pass();
        evidence.build = EngineeringCheck::fail("compile error");
        evidence.required_tests = EngineeringCheck::fail("tests not runnable");
        let verdict = evidence.verdict();
        assert_eq!(verdict.first_violation, Some(EngineeringGateKind::Build));
    }
}
