//! §2 — VECTEUR D'ÉTAT COGNITIF ÉTENDU
//!
//! ```text
//! S = (D, M, R, A, C, V)
//!   D : connaissances          M : modèle (paramètres + architecture)
//!   R : raisonnement           A : autonomie
//!   C : mémoire contextuelle   V : valeurs / buts
//! ```
//!
//! Chaque composante est un vecteur réel. L'état complet peut être aplati en
//! un unique vecteur (`to_vector`) pour la dynamique (§4) et reconstruit
//! (`from_vector`).

use crate::linalg::{mean, norm};
use crate::rng::Rng;

/// Dimensions de chaque composante de S.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub d: usize,
    pub m: usize,
    pub r: usize,
    pub a: usize,
    pub c: usize,
    pub v: usize,
}

impl Dims {
    pub fn uniform(n: usize) -> Self {
        Dims { d: n, m: n, r: n, a: n, c: n, v: n }
    }

    pub fn total(&self) -> usize {
        self.d + self.m + self.r + self.a + self.c + self.v
    }
}

/// État cognitif S = (D, M, R, A, C, V).
#[derive(Clone, Debug)]
pub struct CognitiveState {
    pub d: Vec<f64>,
    pub m: Vec<f64>,
    pub r: Vec<f64>,
    pub a: Vec<f64>,
    pub c: Vec<f64>,
    pub v: Vec<f64>,
}

impl CognitiveState {
    /// État nul aux dimensions données.
    pub fn zeros(dims: Dims) -> Self {
        CognitiveState {
            d: vec![0.0; dims.d],
            m: vec![0.0; dims.m],
            r: vec![0.0; dims.r],
            a: vec![0.0; dims.a],
            c: vec![0.0; dims.c],
            v: vec![0.0; dims.v],
        }
    }

    /// État initial aléatoire (petites valeurs positives).
    pub fn random(dims: Dims, rng: &mut Rng, scale: f64) -> Self {
        let gen = |n: usize, rng: &mut Rng| -> Vec<f64> {
            (0..n).map(|_| rng.normal(0.0, scale).abs()).collect()
        };
        CognitiveState {
            d: gen(dims.d, rng),
            m: gen(dims.m, rng),
            r: gen(dims.r, rng),
            a: gen(dims.a, rng),
            c: gen(dims.c, rng),
            v: gen(dims.v, rng),
        }
    }

    pub fn dims(&self) -> Dims {
        Dims {
            d: self.d.len(),
            m: self.m.len(),
            r: self.r.len(),
            a: self.a.len(),
            c: self.c.len(),
            v: self.v.len(),
        }
    }

    /// Accès aux composantes dans l'ordre canonique.
    pub fn components(&self) -> [&Vec<f64>; 6] {
        [&self.d, &self.m, &self.r, &self.a, &self.c, &self.v]
    }

    /// Aplatit S en un unique vecteur (ordre D,M,R,A,C,V).
    pub fn to_vector(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.dims().total());
        for comp in self.components() {
            out.extend_from_slice(comp);
        }
        out
    }

    /// Reconstruit S depuis un vecteur plat et un schéma de dimensions.
    pub fn from_vector(vector: &[f64], dims: Dims) -> Self {
        assert_eq!(vector.len(), dims.total(), "taille de vecteur incompatible");
        let mut off = 0;
        let mut take = |n: usize| -> Vec<f64> {
            let slice = vector[off..off + n].to_vec();
            off += n;
            slice
        };
        CognitiveState {
            d: take(dims.d),
            m: take(dims.m),
            r: take(dims.r),
            a: take(dims.a),
            c: take(dims.c),
            v: take(dims.v),
        }
    }

    /// Taille totale du vecteur d'état.
    pub fn size(&self) -> usize {
        self.dims().total()
    }

    /// Niveaux sous forme de tableau ordonné (D,M,R,A,C,V).
    pub fn capability_array(&self) -> [f64; 6] {
        [
            mean(&self.d),
            mean(&self.m),
            mean(&self.r),
            mean(&self.a),
            mean(&self.c),
            mean(&self.v),
        ]
    }

    pub fn norm(&self) -> f64 {
        norm(&self.to_vector())
    }

    // --- algèbre (ΔS, contraintes, etc.) --------------------------------- //

    pub fn add(&self, other: &CognitiveState) -> CognitiveState {
        self.zip_with(other, |x, y| x + y)
    }

    pub fn sub(&self, other: &CognitiveState) -> CognitiveState {
        self.zip_with(other, |x, y| x - y)
    }

    pub fn scaled(&self, factor: f64) -> CognitiveState {
        self.map(|x| x * factor)
    }

    /// Borne chaque composante dans [lo, hi] (compétences normalisées).
    pub fn clipped(&self, lo: f64, hi: f64) -> CognitiveState {
        self.map(|x| x.clamp(lo, hi))
    }

    fn map(&self, f: impl Fn(f64) -> f64) -> CognitiveState {
        CognitiveState {
            d: self.d.iter().map(|&x| f(x)).collect(),
            m: self.m.iter().map(|&x| f(x)).collect(),
            r: self.r.iter().map(|&x| f(x)).collect(),
            a: self.a.iter().map(|&x| f(x)).collect(),
            c: self.c.iter().map(|&x| f(x)).collect(),
            v: self.v.iter().map(|&x| f(x)).collect(),
        }
    }

    fn zip_with(&self, other: &CognitiveState, f: impl Fn(f64, f64) -> f64) -> CognitiveState {
        let z = |a: &[f64], b: &[f64]| -> Vec<f64> {
            a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
        };
        CognitiveState {
            d: z(&self.d, &other.d),
            m: z(&self.m, &other.m),
            r: z(&self.r, &other.r),
            a: z(&self.a, &other.a),
            c: z(&self.c, &other.c),
            v: z(&self.v, &other.v),
        }
    }
}

/// ‖ΔS‖ — norme euclidienne du pas (contrainte de stabilité §4).
pub fn delta_norm(before: &CognitiveState, after: &CognitiveState) -> f64 {
    norm(&after.sub(before).to_vector())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_roundtrip() {
        let mut rng = Rng::new(3);
        let s = CognitiveState::random(Dims::uniform(4), &mut rng, 0.2);
        let v = s.to_vector();
        let s2 = CognitiveState::from_vector(&v, s.dims());
        assert_eq!(s.to_vector(), s2.to_vector());
        assert_eq!(v.len(), 24);
    }

    #[test]
    fn algebra() {
        let dims = Dims::uniform(2);
        let a = CognitiveState::from_vector(&[1.0; 12], dims);
        let b = a.scaled(2.0);
        let diff = b.sub(&a);
        assert!((delta_norm(&a, &b) - diff.norm()).abs() < 1e-12);
    }
}
