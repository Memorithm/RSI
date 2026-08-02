//! ⚙️ Loop Engineering — **L2 : détection de convergence / divergence**.
//!
//! Estimateur **en ligne** de la tendance de `SI_global` (ou `SI_safe`) sur une
//! fenêtre glissante, par régression linéaire des moindres carrés. Sert au
//! pilote de boucle (`loop_ctrl`) pour décider d'un arrêt sur **plateau**
//! (attracteur substrate-limited) ou d'un signal de **divergence**.

use std::collections::VecDeque;

/// Tendance détectée sur la fenêtre courante.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trend {
    /// pente nettement positive — progression.
    Improving,
    /// pente quasi nulle — plateau (convergence).
    Plateau,
    /// pente nettement négative — régression / divergence.
    Diverging,
}

/// Détecteur de convergence à fenêtre glissante.
#[derive(Clone, Debug)]
pub struct ConvergenceDetector {
    window: usize,
    buf: VecDeque<f64>,
}

impl ConvergenceDetector {
    pub fn new(window: usize) -> Self {
        ConvergenceDetector {
            window: window.max(2),
            buf: VecDeque::new(),
        }
    }

    /// Ajoute une observation (FIFO sur la fenêtre).
    pub fn push(&mut self, value: f64) {
        if self.buf.len() == self.window {
            self.buf.pop_front();
        }
        self.buf.push_back(value);
    }

    /// La fenêtre est-elle pleine (assez de points pour estimer) ?
    pub fn filled(&self) -> bool {
        self.buf.len() >= self.window
    }

    /// Pente par moindres carrés sur la fenêtre (x = 0..n-1). 0 si < 2 points.
    pub fn slope(&self) -> f64 {
        let n = self.buf.len();
        if n < 2 {
            return 0.0;
        }
        let nf = n as f64;
        let mean_x = (nf - 1.0) / 2.0;
        let mean_y = self.buf.iter().sum::<f64>() / nf;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &y) in self.buf.iter().enumerate() {
            let dx = i as f64 - mean_x;
            num += dx * (y - mean_y);
            den += dx * dx;
        }
        if den == 0.0 {
            0.0
        } else {
            num / den
        }
    }

    /// Classe la tendance selon un seuil de pente `eps` (par pas).
    pub fn trend(&self, eps: f64) -> Trend {
        let s = self.slope();
        if s > eps {
            Trend::Improving
        } else if s < -eps {
            Trend::Diverging
        } else {
            Trend::Plateau
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(d: &mut ConvergenceDetector, vals: &[f64]) {
        for &v in vals {
            d.push(v);
        }
    }

    #[test]
    fn detects_improving() {
        let mut d = ConvergenceDetector::new(5);
        feed(&mut d, &[0.1, 0.2, 0.3, 0.4, 0.5]);
        assert!(d.filled());
        assert!(d.slope() > 0.05);
        assert_eq!(d.trend(1e-3), Trend::Improving);
    }

    #[test]
    fn detects_plateau() {
        let mut d = ConvergenceDetector::new(5);
        feed(&mut d, &[0.50, 0.501, 0.4995, 0.5005, 0.50]);
        assert_eq!(d.trend(1e-2), Trend::Plateau);
    }

    #[test]
    fn detects_diverging() {
        let mut d = ConvergenceDetector::new(4);
        feed(&mut d, &[0.6, 0.5, 0.4, 0.3]);
        assert_eq!(d.trend(1e-3), Trend::Diverging);
    }

    #[test]
    fn window_slides() {
        let mut d = ConvergenceDetector::new(3);
        feed(&mut d, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(d.slope().round(), 1.0); // dernière fenêtre [3,4,5] pente=1
    }
}
