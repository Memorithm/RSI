//! Pilote d'**ascension élitiste bornée** — *stand-in local* du contrat
//! `scirust-rsi`, en attendant l'autorisation d'accès au dépôt
//! `Memorithm/scirust` dans la session.
//!
//! Reproduit fidèlement le contrat documenté de `scirust-rsi` :
//! - trait [`RefineTask`] : `score` (évaluateur → *fitness*) + `refine`
//!   (générateur → révision critiquée) ;
//! - [`Guard`] : `max_iters`, `patience`, `target`, `min_delta` ;
//! - [`ascend`] : boucle « propose → évalue → **garde si STRICTEMENT meilleur**
//!   → répète » ;
//! - [`Report`] avec [`Report::is_monotone`].
//!
//! **Contrat de sûreté** (garanti par construction) :
//! - *élitisme* : une révision n'est adoptée que si sa fitness dépasse
//!   STRICTEMENT l'incumbent (par `min_delta` ≥ 0) ⇒ **aucune régression**.
//! - *bornage / terminaison* : au plus `max_iters` itérations.
//! - *déterminisme* : entièrement piloté par la graine de la tâche.
//!
//! Pour basculer sur le vrai moteur : remplacer `crate::ascent::{RefineTask,
//! Guard, ascend, Report}` par `scirust_rsi::{…}` (mêmes noms / sémantique).

/// Tâche d'auto-amélioration : un évaluateur (`score`) et un générateur
/// (`refine`). `Cand` est la représentation du candidat (algorithme/expression).
pub trait RefineTask {
    type Cand: Clone;
    /// ÉVALUATEUR : fitness du candidat (plus grand = meilleur).
    fn score(&self, cand: &Self::Cand) -> f64;
    /// GÉNÉRATEUR : produit une révision critiquée du candidat (déterministe
    /// pour une `iter` donnée).
    fn refine(&mut self, cand: &Self::Cand, iter: usize) -> Self::Cand;
}

/// Garde-fou de boucle (bornes + arrêt anticipé).
#[derive(Clone, Copy, Debug)]
pub struct Guard {
    max_iters: usize,
    patience: usize,
    target: Option<f64>,
    min_delta: f64,
}

impl Default for Guard {
    fn default() -> Self {
        Guard { max_iters: 50, patience: 0, target: None, min_delta: 0.0 }
    }
}

impl Guard {
    pub fn new() -> Self {
        Guard::default()
    }
    /// Borne dure du nombre d'itérations (terminaison garantie).
    pub fn max_iters(mut self, n: usize) -> Self {
        self.max_iters = n;
        self
    }
    /// Arrêt après `n` itérations consécutives sans amélioration (0 = désactivé).
    pub fn patience(mut self, n: usize) -> Self {
        self.patience = n;
        self
    }
    /// Arrêt dès que la fitness atteint cette cible.
    pub fn target(mut self, t: f64) -> Self {
        self.target = Some(t);
        self
    }
    /// Amélioration minimale (strictement) requise pour adopter une révision.
    pub fn min_delta(mut self, d: f64) -> Self {
        self.min_delta = d.max(0.0);
        self
    }
}

/// Raison d'arrêt de l'ascension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    MaxIters,
    Patience,
    Target,
}

/// Compte rendu d'une ascension.
#[derive(Clone, Debug)]
pub struct Report {
    /// fitness de l'**incumbent** après chaque itération (index 0 = initial).
    pub history: Vec<f64>,
    pub iters: usize,
    /// nombre de révisions strictement meilleures adoptées.
    pub accepted: usize,
    pub stop: StopReason,
}

impl Report {
    /// Meilleure fitness atteinte.
    pub fn best(&self) -> f64 {
        self.history.last().copied().unwrap_or(f64::NEG_INFINITY)
    }

    /// **Non-régression** : l'historique de l'incumbent est non décroissant.
    pub fn is_monotone(&self) -> bool {
        self.history.windows(2).all(|w| w[1] >= w[0] - 1e-12)
    }
}

/// Boucle d'ascension élitiste bornée. Garde l'incumbent ; n'adopte une
/// révision que si sa fitness est **strictement** supérieure (par `min_delta`).
pub fn ascend<T: RefineTask>(task: &mut T, init: T::Cand, guard: &Guard) -> (T::Cand, Report) {
    let mut best = init;
    let mut best_fit = task.score(&best);
    let mut history = vec![best_fit];
    let mut accepted = 0usize;
    let mut stale = 0usize;
    let mut iters = 0usize;
    let mut stop = StopReason::MaxIters;

    for i in 0..guard.max_iters {
        iters = i + 1;
        let cand = task.refine(&best, i);
        let fit = task.score(&cand);
        if fit > best_fit + guard.min_delta {
            best = cand; // adoption seulement si STRICTEMENT meilleur
            best_fit = fit;
            accepted += 1;
            stale = 0;
        } else {
            stale += 1;
        }
        history.push(best_fit); // incumbent (monotone non décroissant)

        if let Some(t) = guard.target {
            if best_fit >= t {
                stop = StopReason::Target;
                break;
            }
        }
        if guard.patience > 0 && stale >= guard.patience {
            stop = StopReason::Patience;
            break;
        }
    }

    (best, Report { history, iters, accepted, stop })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tâche jouet : un entier qu'on rapproche d'une cible ; `refine` propose
    /// parfois pire (pour prouver que l'élitisme rejette les régressions).
    struct Toy {
        target: i64,
    }
    impl RefineTask for Toy {
        type Cand = i64;
        fn score(&self, c: &i64) -> f64 {
            -((self.target - *c).abs() as f64) // 0 au mieux
        }
        fn refine(&mut self, c: &i64, iter: usize) -> i64 {
            // alterne propositions meilleures / pires
            if iter.is_multiple_of(2) {
                c + 1
            } else {
                c - 3
            }
        }
    }

    #[test]
    fn elitist_is_monotone_and_bounded() {
        let mut task = Toy { target: 20 };
        let guard = Guard::new().max_iters(50);
        let (best, report) = ascend(&mut task, 0, &guard);
        assert!(report.is_monotone(), "aucune régression ne doit être adoptée");
        assert!(report.iters <= 50, "terminaison bornée");
        assert_eq!(report.history.len(), report.iters + 1);
        // les propositions « pires » (-3) n'ont jamais été adoptées
        assert!(best <= 20);
        assert!(report.best() >= task.score(&0));
    }

    #[test]
    fn stops_on_target() {
        let mut task = Toy { target: 5 };
        let guard = Guard::new().max_iters(100).target(0.0);
        let (_b, report) = ascend(&mut task, 0, &guard);
        assert_eq!(report.stop, StopReason::Target);
        assert!(report.best() >= 0.0);
        assert!(report.iters < 100);
    }

    #[test]
    fn stops_on_patience() {
        // cible atteignable en montant, mais refine ne propose que du pire
        struct OnlyWorse;
        impl RefineTask for OnlyWorse {
            type Cand = i64;
            fn score(&self, c: &i64) -> f64 {
                -(c.abs() as f64)
            }
            fn refine(&mut self, c: &i64, _i: usize) -> i64 {
                c + 5 // toujours pire (s'éloigne de 0)
            }
        }
        let guard = Guard::new().max_iters(100).patience(3);
        let (_b, report) = ascend(&mut OnlyWorse, 0, &guard);
        assert_eq!(report.stop, StopReason::Patience);
        assert_eq!(report.accepted, 0);
        assert!(report.is_monotone());
    }
}
