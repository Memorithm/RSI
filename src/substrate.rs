//! §3 — SUBSTRAT PHYSIQUE & LOGICIEL (EFFICACITÉ MULTIPLICATIVE)
//!
//! ```text
//! H ∈ ℝⁿᴴ (matériel),  O ∈ ℝⁿᴼ (logiciel)
//! P_eff = σ(HᵀA H) · σ(OᵀB O) · σ(HᵀC O)
//! σ(x) = 1 / (1 + e⁻ˣ)
//! ```
//!
//! L'efficacité est *multiplicative* : un matériel puissant est inutile sans
//! un logiciel capable de l'exploiter, et le terme de couplage σ(HᵀC O)
//! capture la synergie hardware ↔ software.

use crate::linalg::{sigmoid, Matrix};
use crate::rng::Rng;

/// Substrat (H, O) et ses matrices d'efficience A, B et de couplage C.
#[derive(Clone, Debug)]
pub struct Substrate {
    pub h: Vec<f64>, // vecteur matériel ∈ ℝⁿᴴ
    pub o: Vec<f64>, // vecteur logiciel ∈ ℝⁿᴼ
    pub a: Matrix,   // nᴴ × nᴴ — efficience interne matérielle
    pub b: Matrix,   // nᴼ × nᴼ — efficience interne logicielle
    pub c: Matrix,   // nᴴ × nᴼ — couplage hardware ↔ software
    /// Efficience logicielle **mesurée** ∈ (0,1), si disponible (Phase 2 :
    /// calibrée par une campagne Forge sur un vrai kernel). Quand `Some`, elle
    /// remplace la forme analytique σ(OᵀB O) dans `software_efficiency`.
    /// `None` par défaut → comportement d'origine inchangé.
    pub measured_software_eff: Option<f64>,
}

impl Substrate {
    pub fn new(h: Vec<f64>, o: Vec<f64>, a: Matrix, b: Matrix, c: Matrix) -> Self {
        let (nh, no) = (h.len(), o.len());
        assert_eq!((a.rows, a.cols), (nh, nh), "A doit être nᴴ×nᴴ");
        assert_eq!((b.rows, b.cols), (no, no), "B doit être nᴼ×nᴼ");
        assert_eq!((c.rows, c.cols), (nh, no), "C doit être nᴴ×nᴼ");
        Substrate {
            h,
            o,
            a,
            b,
            c,
            measured_software_eff: None,
        }
    }

    /// Fixe (ou efface) l'efficience logicielle mesurée. Bornée dans (0,1).
    pub fn set_measured_software_eff(&mut self, value: Option<f64>) {
        self.measured_software_eff = value.map(|v| v.clamp(1e-6, 1.0 - 1e-9));
    }

    /// Substrat raisonnable : matrices d'efficience symétriques définies
    /// positives, couplage modéré, vecteurs HW/SW positifs.
    pub fn default_with(n_hardware: usize, n_software: usize, rng: &mut Rng) -> Self {
        // M·Mᵀ/n + 0.1·I  => symétrique définie positive
        let spd = |n: usize, scale: f64, rng: &mut Rng| -> Matrix {
            let mut raw = Matrix::zeros(n, n);
            for k in 0..n * n {
                raw.data[k] = rng.normal(0.0, scale);
            }
            let mut out = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let mut acc = 0.0;
                    for k in 0..n {
                        acc += raw.get(i, k) * raw.get(j, k);
                    }
                    out.set(i, j, acc / n as f64 + if i == j { 0.1 } else { 0.0 });
                }
            }
            out
        };

        let h: Vec<f64> = (0..n_hardware)
            .map(|_| rng.normal(0.5, 0.2).abs())
            .collect();
        let o: Vec<f64> = (0..n_software)
            .map(|_| rng.normal(0.5, 0.2).abs())
            .collect();
        let a = spd(n_hardware, 0.3, rng);
        let b = spd(n_software, 0.3, rng);
        let mut c = Matrix::zeros(n_hardware, n_software);
        for k in 0..n_hardware * n_software {
            c.data[k] = rng.normal(0.0, 0.2);
        }
        Substrate::new(h, o, a, b, c)
    }

    /// σ(HᵀA H) — efficience interne matérielle ∈ (0, 1).
    pub fn hardware_efficiency(&self) -> f64 {
        sigmoid(self.a.quadratic(&self.h))
    }

    /// Efficience logicielle **analytique** seule σ(OᵀB O) (sans la mesure).
    pub fn analytic_software_efficiency(&self) -> f64 {
        sigmoid(self.b.quadratic(&self.o))
    }

    /// Écart (≥0) entre l'efficience effective et l'analytique : proxy de
    /// *wireheading* (§7, f7) — l'agent gonfle-t-il la *mesure* au-delà de ce
    /// que `O` justifie analytiquement ?
    pub fn software_eff_gap(&self) -> f64 {
        (self.software_efficiency() - self.analytic_software_efficiency()).max(0.0)
    }

    /// Efficience logicielle ∈ (0, 1).
    ///
    /// **Canal unifié** : on combine la forme analytique σ(OᵀB O) — pilotée par
    /// le `software_edit` de ℳ — et l'efficience *mesurée* (Phase 2, campagne
    /// Forge) par un **maximum**. Les deux leviers d'auto-amélioration
    /// logicielle coopèrent au lieu de se neutraliser : améliorer `O` reste
    /// utile même après une mesure, et la mesure agit comme un plancher.
    pub fn software_efficiency(&self) -> f64 {
        let analytic = sigmoid(self.b.quadratic(&self.o));
        match self.measured_software_eff {
            Some(v) => analytic.max(v),
            None => analytic,
        }
    }

    /// σ(HᵀC O) — efficience du couplage hardware ↔ software ∈ (0, 1).
    pub fn coupling_efficiency(&self) -> f64 {
        sigmoid(self.c.bilinear(&self.h, &self.o))
    }

    /// P_eff = σ(HᵀA H) · σ(OᵀB O) · σ(HᵀC O) ∈ (0, 1).
    pub fn effective_power(&self) -> f64 {
        self.hardware_efficiency() * self.software_efficiency() * self.coupling_efficiency()
    }
}

/// Améliorateur de substrat (Phase 2).
///
/// Implémenté par un backend d'optimisation *exécutée* (p. ex. une campagne
/// Forge sur un vrai kernel) qui calibre l'efficience logicielle mesurée et
/// renvoie un substrat amélioré. **Contrat de monotonie** : ne doit jamais
/// renvoyer un substrat de `effective_power()` inférieur à l'entrée (l'agent
/// applique de toute façon un garde-fou, mais l'implémentation doit l'assurer).
pub trait SubstrateImprover {
    fn improve(&mut self, substrate: &Substrate) -> Substrate;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_efficiency_combines_by_max() {
        let mut rng = Rng::new(3);
        let mut s = Substrate::default_with(4, 4, &mut rng);
        let analytic = s.software_efficiency();
        // mesuré supérieur → l'emporte (plancher)
        s.set_measured_software_eff(Some(0.95));
        assert!((s.software_efficiency() - 0.95).abs() < 1e-12);
        // mesuré inférieur → l'analytique (piloté par O) reste utilisé
        s.set_measured_software_eff(Some(analytic * 0.5));
        assert!((s.software_efficiency() - analytic).abs() < 1e-12);
        // effacé → analytique
        s.set_measured_software_eff(None);
        assert!((s.software_efficiency() - analytic).abs() < 1e-12);
    }

    #[test]
    fn p_eff_in_unit_interval() {
        let mut rng = Rng::new(0);
        let s = Substrate::default_with(4, 4, &mut rng);
        let p = s.effective_power();
        assert!(p > 0.0 && p < 1.0, "P_eff = {p}");
    }

    #[test]
    fn scaling_hardware_increases_efficiency() {
        let mut rng = Rng::new(1);
        let mut s = Substrate::default_with(4, 4, &mut rng);
        let before = s.hardware_efficiency();
        for x in s.h.iter_mut() {
            *x *= 2.0;
        }
        assert!(s.hardware_efficiency() >= before);
    }
}
