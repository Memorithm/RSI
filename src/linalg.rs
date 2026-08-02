//! Petites primitives d'algèbre linéaire (std-only) : vecteurs, matrices
//! denses, formes quadratiques xᵀMx, et la sigmoïde σ utilisée partout.
//
// Boucles indexées intentionnelles (produits matrice/vecteur), lint désactivé.
#![allow(clippy::needless_range_loop)]

/// σ(x) = 1 / (1 + e⁻ˣ), implémentation numériquement stable.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Produit scalaire de deux vecteurs de même longueur (scalaire, ordre G→D).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Produit scalaire **vectorisé** (`wide::f64x4`, feature `simd`) : réduction par
/// 4 voies puis combinaison à ordre fixe — déterministe dans un build SIMD, mais
/// numériquement distinct du scalaire (l'ordre de sommation diffère).
#[cfg(feature = "simd")]
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    use wide::f64x4;
    let n = a.len().min(b.len());
    let lanes = n / 4;
    let mut acc = f64x4::splat(0.0);
    for i in 0..lanes {
        let base = i * 4;
        let va = f64x4::from([a[base], a[base + 1], a[base + 2], a[base + 3]]);
        let vb = f64x4::from([b[base], b[base + 1], b[base + 2], b[base + 3]]);
        acc += va * vb;
    }
    let [s0, s1, s2, s3] = acc.to_array();
    let mut sum = s0 + s1 + s2 + s3;
    for i in (lanes * 4)..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Norme euclidienne ‖v‖.
#[inline]
pub fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Moyenne des éléments (0.0 si vide) — scalaire.
#[cfg(not(feature = "simd"))]
#[inline]
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Moyenne des éléments **vectorisée** (`wide::f64x4`, feature `simd`).
#[cfg(feature = "simd")]
#[inline]
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    use wide::f64x4;
    let n = v.len();
    let lanes = n / 4;
    let mut acc = f64x4::splat(0.0);
    for i in 0..lanes {
        let base = i * 4;
        acc += f64x4::from([v[base], v[base + 1], v[base + 2], v[base + 3]]);
    }
    let [s0, s1, s2, s3] = acc.to_array();
    let mut sum = s0 + s1 + s2 + s3;
    for i in (lanes * 4)..n {
        sum += v[i];
    }
    sum / n as f64
}

/// Matrice dense en ligne-major.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "dimensions incompatibles");
        Matrix { rows, cols, data }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    /// Produit matrice-vecteur M·v (v de taille `cols`, sortie de taille `rows`).
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.cols, "matvec: dimension du vecteur");
        let mut out = vec![0.0; self.rows];
        for i in 0..self.rows {
            let mut acc = 0.0;
            let base = i * self.cols;
            for j in 0..self.cols {
                acc += self.data[base + j] * v[j];
            }
            out[i] = acc;
        }
        out
    }

    /// Forme bilinéaire aᵀ · M · b.
    pub fn bilinear(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), self.rows);
        let mb = self.matvec(b);
        dot(a, &mb)
    }

    /// Forme quadratique xᵀ·M·x.
    pub fn quadratic(&self, x: &[f64]) -> f64 {
        self.bilinear(x, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(40.0) > 0.999);
        assert!(sigmoid(-40.0) < 0.001);
    }

    #[test]
    fn quadratic_identity() {
        let m = Matrix::identity(3);
        let x = [1.0, 2.0, 2.0];
        assert!((m.quadratic(&x) - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_neon_matmul_correctness() {
        let n = 8;
        let a = vec![1.0f32; n * n];
        let b = vec![2.0f32; n * n];
        let mut c = vec![0.0f32; n * n];
        unsafe {
            neon_matmul(&a, &b, &mut c, n);
        }
        // Pour n=8, chaque cellule de C doit valoir 8 * 1.0 * 2.0 = 16.0
        for &val in &c {
            assert!((val - 16.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_neon_matmul_non_multiple_of_4() {
        let n = 7; // Non multiple de 4, pour valider la boucle de reste
        let a = vec![1.5f32; n * n];
        let b = vec![3.0f32; n * n];
        let mut c = vec![0.0f32; n * n];
        unsafe {
            neon_matmul(&a, &b, &mut c, n);
        }
        // Pour n=7, chaque cellule de C doit valoir 7 * 1.5 * 3.0 = 31.5
        for &val in &c {
            assert!((val - 31.5).abs() < 1e-5);
        }
    }
}

// ===================== OPTIMISATION SCIRUST (ARM Neon) =================== //

/// Multiplication matricielle de poids cognitifs optimisée par les instructions intrinsèques ARM Neon.
/// Calcule C = A·B (row-major, matrices carrées N×N).
///
/// # Safety
/// - Les tranches `a`, `b` et `c` doivent posséder au moins `n * n` éléments de type `f32`.
/// - `c` ne doit pas chevaucher `a` ni `b` en mémoire.
/// - Si compilé pour `aarch64`, s'exécute avec les intrinsèques matériels Neon.
#[cfg(target_arch = "aarch64")]
pub unsafe fn neon_matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    use core::arch::aarch64::*;

    // Invariants de Sûreté :
    // - Les pointeurs d'adresses de tranches doivent être valides, non nuls et alignés.
    // - L'index j doit être incrémenté par pas de 4 (largeur du registre vectoriel float32x4_t).
    // - La boucle de reste scalaire assure la correction pour toutes les dimensions de matrice non multiples de 4.

    // Écrasement initial : c est vidé
    c[..n * n].fill(0.0f32);

    for i in 0..n {
        for k in 0..n {
            let aik = *a.get_unchecked(i * n + k);
            let va = vdupq_n_f32(aik);

            let mut j = 0;
            while j + 3 < n {
                let offset = k * n + j;
                let out_offset = i * n + j;

                let vb = vld1q_f32(b.as_ptr().add(offset));
                let vc = vld1q_f32(c.as_ptr().add(out_offset));
                let vr = vfmaq_f32(vc, va, vb);
                vst1q_f32(c.as_mut_ptr().add(out_offset), vr);

                j += 4;
            }

            while j < n {
                let out_offset = i * n + j;
                *c.get_unchecked_mut(out_offset) += aik * *b.get_unchecked(k * n + j);
                j += 1;
            }
        }
    }
}

/// Fallback scalaire portable pour les architectures non-ARM64 (comme x86_64 de développement).
///
/// # Safety
/// - Les tranches `a`, `b` et `c` doivent posséder au moins `n * n` éléments.
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn neon_matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    // Invariants de Sûreté :
    // - Les tranches doivent être valides pour au moins n * n éléments.
    // - L'utilisation de get_unchecked/get_unchecked_mut évite la vérification des bornes
    //   et maximise la bande passante CPU sur l'architecture de repli.
    c[..n * n].fill(0.0f32);

    for i in 0..n {
        for k in 0..n {
            let aik = *a.get_unchecked(i * n + k);
            for j in 0..n {
                let out_offset = i * n + j;
                *c.get_unchecked_mut(out_offset) += aik * *b.get_unchecked(k * n + j);
            }
        }
    }
}
