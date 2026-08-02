//! ⚙️ Loop Engineering — **L3 : boucles multi-échelles (ordonnanceur)**.
//!
//! Le RSI imbrique des échelles de temps : boucle **interne** (apprentissage
//! ΔS, chaque pas), boucle **méta** (révision ℳ, tous les `meta_every`), boucle
//! **substrat** (amélioration de P_eff, plus lente, tous les `substrate_every`),
//! et une boucle **méta-méta** qui *révise les cadences elles-mêmes* selon la
//! tendance de convergence.
//!
//! [`LoopSchedule`] applique ces cadences à un [`RSIAgent`] ; [`MetaMeta`]
//! adapte le planning : ralentir la méta sur plateau (économie), l'accélérer
//! quand l'agent progresse encore.

use crate::agent::RSIAgent;
use crate::convergence::Trend;

/// Cadences des boucles imbriquées.
#[derive(Clone, Copy, Debug)]
pub struct LoopSchedule {
    /// méta-révision ℳ un pas sur `meta_every`.
    pub meta_every: usize,
    /// amélioration du substrat un pas sur `substrate_every`.
    pub substrate_every: usize,
}

impl Default for LoopSchedule {
    fn default() -> Self {
        LoopSchedule {
            meta_every: 1,
            substrate_every: 1,
        }
    }
}

impl LoopSchedule {
    pub fn new(meta_every: usize, substrate_every: usize) -> Self {
        LoopSchedule {
            meta_every: meta_every.max(1),
            substrate_every: substrate_every.max(1),
        }
    }

    /// Applique les cadences à l'agent.
    pub fn apply(&self, agent: &mut RSIAgent) {
        agent.meta_interval = self.meta_every.max(1);
        agent.substrate_interval = self.substrate_every.max(1);
    }
}

impl RSIAgent {
    /// Configure les cadences multi-échelles (L3). Builder.
    pub fn with_schedule(mut self, schedule: LoopSchedule) -> Self {
        schedule.apply(&mut self);
        self
    }
}

/// Boucle **méta-méta** : révise les cadences selon la tendance observée.
#[derive(Clone, Copy, Debug)]
pub struct MetaMeta {
    pub min_meta: usize,
    pub max_meta: usize,
}

impl Default for MetaMeta {
    fn default() -> Self {
        MetaMeta {
            min_meta: 1,
            max_meta: 16,
        }
    }
}

impl MetaMeta {
    /// Adapte le planning : sur **plateau**, on ralentit la méta (cadence ×2,
    /// économie de calcul une fois l'attracteur atteint) ; en **progression**,
    /// on l'accélère (cadence ÷2) ; en **divergence**, on accélère au maximum
    /// pour reprendre le contrôle au plus vite.
    pub fn adapt(&self, schedule: LoopSchedule, trend: Trend) -> LoopSchedule {
        let meta_every = match trend {
            Trend::Plateau => (schedule.meta_every * 2).min(self.max_meta),
            Trend::Improving => (schedule.meta_every / 2).max(self.min_meta),
            Trend::Diverging => self.min_meta,
        };
        LoopSchedule {
            meta_every,
            substrate_every: schedule.substrate_every,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_applies_cadences() {
        let agent = RSIAgent::demo(1).with_schedule(LoopSchedule::new(3, 5));
        assert_eq!(agent.meta_interval, 3);
        assert_eq!(agent.substrate_interval, 5);
    }

    #[test]
    fn meta_meta_adapts() {
        let mm = MetaMeta::default();
        let s = LoopSchedule::new(4, 2);
        assert!(mm.adapt(s, Trend::Plateau).meta_every >= 8); // ralentit
        assert!(mm.adapt(s, Trend::Improving).meta_every <= 2); // accélère
        assert_eq!(mm.adapt(s, Trend::Diverging).meta_every, 1); // max réactivité
    }

    #[test]
    fn slower_meta_still_improves() {
        // une méta plus lente (économe) progresse toujours sur l'horizon
        let mut agent = RSIAgent::demo(7).with_schedule(LoopSchedule::new(4, 2));
        let start = agent.si_global();
        agent.run(80);
        assert!(agent.si_global() > start);
    }
}
