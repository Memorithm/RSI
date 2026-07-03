//! Kernels numériques — **cible d'optimisation dédiée** pour la boucle DGM.
//!
//! Contrairement à [`crate::measured_substrate::matmul_naive`], qui sert de
//! *baseline lent* à `MeasuredSubstrate` (l'accélérer casserait la mesure du
//! speedup), les kernels d'ici sont des sujets d'optimisation **légitimes et
//! conservables** : rien d'autre ne dépend de leur lenteur. DGM peut donc les
//! réécrire, et un patch accepté est réellement promouvable dans l'arbre vivant.
//!
//! Chaque kernel vient avec un test de correction (barrière du gate) et un
//! `examples/bench_*` qui imprime `RSI_BENCH_SCORE=<débit>` pour donner un
//! gradient de perf réel. **std-only, sans dépendance.**

/// Produit matrice-matrice C = A·B (row-major, N×N), ordre i,j,k.
///
/// Implémentation naïve délibérément *hostile au cache* (k interne balaie une
/// colonne de B à grands pas) : il existe un vrai *headroom* qu'un tuilage
/// (blocking) capture. C'est la cible que la boucle DGM cherche à accélérer
/// **à résultat identique** (cf. `examples/bench_kernel`).
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), n * n);
    assert_eq!(c.len(), n * n);
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f32;
            for k in 0..n {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PRNG linéaire déterministe (sans dépendance) pour des matrices reproductibles.
    fn matrix(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n * n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    /// Référence indépendante (triple boucle, ordre différent) pour vérifier que
    /// toute réécriture de `matmul` calcule bien le même produit.
    fn matmul_reference(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; n * n];
        for i in 0..n {
            for k in 0..n {
                let aik = a[i * n + k];
                for j in 0..n {
                    c[i * n + j] += aik * b[k * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn matmul_matches_reference() {
        for &n in &[1usize, 2, 7, 16, 33] {
            let a = matrix(n, 0xA1);
            let b = matrix(n, 0xB2);
            let mut c = vec![0.0f32; n * n];
            matmul(&a, &b, &mut c, n);
            let expect = matmul_reference(&a, &b, n);
            for (x, y) in c.iter().zip(&expect) {
                assert!(
                    (x - y).abs() <= 1e-3 * (1.0 + y.abs()),
                    "n={n}: {x} vs {y}"
                );
            }
        }
    }

    #[test]
    fn matmul_identity_is_passthrough() {
        let n = 8;
        let a = matrix(n, 0x7);
        let mut id = vec![0.0f32; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        let mut c = vec![0.0f32; n * n];
        matmul(&a, &id, &mut c, n);
        for (x, y) in c.iter().zip(&a) {
            assert!((x - y).abs() <= 1e-6, "{x} vs {y}");
        }
    }
}
