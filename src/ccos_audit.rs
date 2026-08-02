//! Adaptateur d'audit **CCOS** (feature `ccos`) — §7bis.
//!
//! Branche l'`EventLog` hash-chaîné de CCOS derrière le trait [`AuditLog`] :
//! chaque pas de `ℳ` est enregistré comme un `TraceEvent` CCOS
//! (`EventType::AgentAction` + `EventPayload::Custom`), avec la vérification
//! d'intégrité et le *replay* natifs de CCOS (forensique avancée).
//!
//! Le port natif [`crate::audit::HashChainLog`] reproduit le même schéma sans
//! dépendance ; cet adaptateur délègue au vrai moteur CCOS quand on veut sa
//! couche de forensique / MMU cognitive complète.
//!
//! Activé via la feature `ccos` (dépendance git sur Memorithm/CCOS, qui
//! n'impose pas l'async/TLS). Usage :
//! `RSIAgent::demo(0).with_audit(Box::new(CcosAudit::new("session")))`.

use ccos::event_log::{EventLog, EventPayload, EventType};

use crate::audit::{AuditEvent, AuditLog};

/// Journal d'audit RSI adossé à l'`EventLog` de CCOS.
pub struct CcosAudit {
    log: EventLog,
}

impl CcosAudit {
    /// Crée un journal CCOS pour la session donnée.
    pub fn new(session_id: impl Into<String>) -> Self {
        CcosAudit {
            log: EventLog::new(session_id.into()),
        }
    }

    /// Accès en lecture à l'`EventLog` CCOS sous-jacent (replay, export, …).
    pub fn event_log(&self) -> &EventLog {
        &self.log
    }
}

impl AuditLog for CcosAudit {
    fn record(&mut self, event: &AuditEvent) -> String {
        self.log.append(
            EventType::AgentAction,
            EventPayload::Custom {
                key: "rsi_step".to_string(),
                value: event.payload(),
            },
        )
    }

    fn len(&self) -> usize {
        self.log.event_count()
    }

    fn head(&self) -> String {
        self.log.chain_head()
    }

    fn verify(&self) -> bool {
        self.log.verify_integrity().valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RSIAgent;

    #[test]
    fn ccos_audit_traces_and_verifies() {
        let mut agent = RSIAgent::demo(5).with_audit(Box::new(CcosAudit::new("rsi-session")));
        agent.run(15);
        assert_eq!(agent.audit_len(), 15);
        assert!(agent.audit_verify()); // chaîne CCOS intègre
        assert!(agent.audit_head() != Some("GENESIS".to_string()));
    }
}
