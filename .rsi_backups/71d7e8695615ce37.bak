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

/// Produit matrice-matrice C = A·B (row-major, N×N). `c` est **écrasé**
/// (pas accumulé) : le contenu entrant de `c` est ignoré.
///
/// Deux étages d'optimisation, tous deux **découverts par la boucle DGM**
/// (qwen3-coder:30b local, Jetson Thor, cf. `examples/bench_kernel`), chacun
/// corrigé en revue humaine sur un point de contrat invisible au gate :
///
/// 1. **Ordre i,k,j** (×6,6 à n=512) : la boucle interne balaie B et C
///    **contigûment** (auto-vectorisable), contrairement au naïf i,j,k dont le
///    k interne saute de ligne en ligne dans B. Revue : le patch accumulait
///    dans `c` sans zéroter (`C += A·B`) — les tests ne passaient que des `c`
///    nuls ; trou fermé par `matmul_overwrites_dirty_output`.
/// 2. **Tuilage i/k `TILE`=64** (+quart environ en sus) : les blocs de A et B
///    restent chauds en L1/L2 entre itérations. Revue : le patch zérotait `c`
///    *à l'intérieur* de la boucle de blocs `kk`, écrasant les contributions
///    des blocs précédents — correct pour n ≤ 64 seulement, or les tests
///    n'allaient que jusqu'à n=33 ; trou fermé par des tailles 96 et 130 dans
///    `matmul_matches_reference`, le zérotage est sorti avant la boucle `kk`.
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), n * n);
    assert_eq!(c.len(), n * n);
    const TILE: usize = 64;
    for ii in (0..n).step_by(TILE) {
        let i_end = (ii + TILE).min(n);
        for i in ii..i_end {
            c[i * n..i * n + n].fill(0.0);
        }
        for kk in (0..n).step_by(TILE) {
            let k_end = (kk + TILE).min(n);
            for i in ii..i_end {
                for k in kk..k_end {
                    let aik = a[i * n + k];
                    for j in 0..n {
                        c[i * n + j] += aik * b[k * n + j];
                    }
                }
            }
        }
    }
}

/// Transposition hors-place `dst = srcᵀ` (row-major, N×N). `dst` est
/// **écrasé** : son contenu entrant est ignoré. `src` et `dst` sont distincts
/// (transposition hors-place — l'emprunteur l'impose déjà : `&`/`&mut`).
///
/// Tuilage 64×64 **découvert par la boucle DGM** (qwen3-coder:30b local,
/// Jetson Thor, +23 % mesuré à n=2048 vs le naïf ligne-à-ligne dont chaque
/// écriture chargeait une ligne de cache entière) — et, contrairement aux deux
/// étages de [`matmul`], **correct du premier coup** : le gate durci en amont
/// (spec directe sur sortie sale, tailles 96/130 chevauchant la tuile,
/// involution) ne laissait plus d'angle mort, et les tuiles partitionnent
/// [0,n)² donc chaque élément de `dst` est écrit exactement une fois.
pub fn transpose(src: &[f32], dst: &mut [f32], n: usize) {
    assert_eq!(src.len(), n * n);
    assert_eq!(dst.len(), n * n);
    const TILE: usize = 64;
    for ii in (0..n).step_by(TILE) {
        let i_end = (ii + TILE).min(n);
        for jj in (0..n).step_by(TILE) {
            let j_end = (jj + TILE).min(n);
            for i in ii..i_end {
                for j in jj..j_end {
                    dst[j * n + i] = src[i * n + j];
                }
            }
        }
    }
}

/// Somme de tous les éléments (f64).
///
/// Implémentation naïve délibérément **sérielle** : chaque itération dépend de
/// la précédente (`acc += x`), le CPU ne peut ni pipeliner ni vectoriser — la
/// latence de l'addition flottante borne le débit, loin de la bande passante
/// mémoire. Des accumulateurs indépendants capturent un vrai headroom (×4.2
/// sondé à n=2²⁰). Troisième cible DGM, d'un genre encore différent
/// (latence de dépendance, ni cache ni calcul — cf. `examples/bench_reduce`).
///
/// Contrat : toute réassociation est acceptable tant que le résultat reste à
/// `1e-12 × Σ|x|` de la somme exacte (référence Kahan dans les tests) et que
/// **tous** les éléments comptent (tailles non multiples de tout largeur de
/// bloc plausible couvertes — un reste oublié échoue).
pub fn sum(v: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for &x in v {
        acc += x;
    }
    acc
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

    /// Référence indépendante (ordre i,j,k, différent de l'implémentation) pour
    /// vérifier que toute réécriture de `matmul` calcule bien le même produit.
    fn matmul_reference(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f32;
                for k in 0..n {
                    s += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    #[test]
    fn matmul_matches_reference() {
        // Tailles de part et d'autre de toute taille de tuile plausible (64,
        // 128…) : un tuilage dont l'accumulation est cassée entre blocs (p. ex.
        // zérotage à l'intérieur de la boucle de blocs k — patch DGM réellement
        // proposé, correct pour n <= tuile seulement) DOIT échouer ici.
        for &n in &[1usize, 2, 7, 16, 33, 96, 130] {
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
    fn matmul_overwrites_dirty_output() {
        // Contrat : C = A·B, le contenu ENTRANT de c est ignoré. Le patch DGM
        // accepté (réordonnancement i,k,j) accumulait dans c sans le zéroter —
        // invisible pour les tests d'alors qui passaient toujours un c nul.
        // Ce test ferme ce trou de spec : c sale ⇒ même résultat.
        let n = 16;
        let a = matrix(n, 0xA1);
        let b = matrix(n, 0xB2);
        let expect = matmul_reference(&a, &b, n);
        let mut c = vec![123.456f32; n * n]; // sortie « sale »
        matmul(&a, &b, &mut c, n);
        for (x, y) in c.iter().zip(&expect) {
            assert!((x - y).abs() <= 1e-3 * (1.0 + y.abs()), "{x} vs {y}");
        }
        // Idempotence : un second appel rend exactement le même C.
        let first = c.clone();
        matmul(&a, &b, &mut c, n);
        assert_eq!(c, first);
    }

    /// Référence : somme de Kahan (compensée) — quasi-exacte, indépendante de
    /// toute stratégie d'accumulation qu'une réécriture pourrait choisir.
    fn kahan_sum(v: &[f64]) -> f64 {
        let (mut s, mut c) = (0.0f64, 0.0f64);
        for &x in v {
            let y = x - c;
            let t = s + y;
            c = (t - s) - y;
            s = t;
        }
        s
    }

    #[test]
    fn sum_matches_kahan_reference() {
        // Tailles de part et d'autre de toute largeur de bloc plausible
        // (8/16/64…) et non multiples : un reste de chunk oublié ÉCHOUE ici.
        // Valeurs mêlées (positives et négatives) : l'annulation partielle
        // punit aussi les réassociations dégénérées.
        for &n in &[0usize, 1, 7, 8, 9, 63, 64, 65, 127, 130, 1001] {
            let v: Vec<f64> = (0..n)
                .map(|i| (((i * 2654435761) % 2001) as f64 - 1000.0) * 1e-3)
                .collect();
            let expect = kahan_sum(&v);
            let got = sum(&v);
            let mass: f64 = v.iter().map(|x| x.abs()).sum::<f64>().max(1.0);
            assert!(
                (got - expect).abs() <= 1e-12 * mass,
                "n={n}: {got} vs {expect} (masse {mass})"
            );
        }
    }

    #[test]
    fn sum_of_ones_is_exact() {
        // n sommable exactement en f64 : aucune tolérance ici — chaque élément
        // doit compter exactement une fois (ni doublon ni omission).
        for &n in &[1usize, 100, 1000, 4097] {
            let v = vec![1.0f64; n];
            assert_eq!(sum(&v), n as f64, "n={n}");
        }
    }

    #[test]
    fn transpose_matches_spec() {
        // Vérification DIRECTE de la spec (dst[j,i] == src[i,j]), indépendante
        // de toute implémentation. Tailles de part et d'autre de toute tuile
        // plausible (leçon matmul : un tuilage cassé entre blocs n'est visible
        // qu'à n > tuile) ; 130 n'est multiple d'aucune tuile usuelle.
        for &n in &[1usize, 2, 7, 16, 33, 96, 130] {
            let src = matrix(n, 0xC3);
            let mut dst = vec![123.456f32; n * n]; // sortie « sale » d'emblée
            transpose(&src, &mut dst, n);
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(
                        dst[j * n + i],
                        src[i * n + j],
                        "n={n}, dst[{j},{i}] != src[{i},{j}]"
                    );
                }
            }
        }
    }

    #[test]
    fn transpose_twice_is_identity() {
        let n = 96;
        let src = matrix(n, 0xD4);
        let mut once = vec![0.0f32; n * n];
        let mut twice = vec![0.0f32; n * n];
        transpose(&src, &mut once, n);
        transpose(&once, &mut twice, n);
        assert_eq!(twice, src);
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
