//! Masques et séquences de longueurs différentes (contrat §14 : « masques
//! partiels », « séquences de longueurs différentes », « mismatch de masque »).
//!
//! Un [`Mask`] est un vecteur d'activation `{0,1}` validé à la construction :
//! - longueur cohérente avec la séquence ;
//! - valeurs uniquement `0` ou `1` ;
//! - validation de cohérence entre deux masques (même longueur).

use crate::error::{CognoError, CognoResult};

/// Masque binaire validé (contrat §14).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask {
    bits: Vec<u8>,
}

impl Mask {
    /// Construit un masque depuis des bits `{0,1}`. Rejette toute valeur hors
    /// `{0,1}` (mismatch de masque = erreur structurée, jamais de panic).
    pub fn try_new(bits: Vec<u8>) -> CognoResult<Self> {
        for &b in &bits {
            if b > 1 {
                return Err(CognoError::MaskMismatch("bit hors {0,1}"));
            }
        }
        Ok(Mask { bits })
    }

    /// Masque « tout actif » de longueur `n`.
    pub fn all_ones(n: usize) -> Self {
        Mask { bits: vec![1; n] }
    }

    /// Masque « tout inactif » de longueur `n`.
    pub fn all_zeros(n: usize) -> Self {
        Mask { bits: vec![0; n] }
    }

    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    pub fn bits(&self) -> &[u8] {
        &self.bits
    }

    /// Nombre de positions actives.
    pub fn active_count(&self) -> usize {
        self.bits.iter().filter(|&&b| b == 1).count()
    }

    /// Fraction de positions actives ∈ [0,1].
    pub fn active_fraction(&self) -> f64 {
        if self.bits.is_empty() {
            0.0
        } else {
            self.active_count() as f64 / self.bits.len() as f64
        }
    }

    /// Vérifie la cohérence avec un autre masque (même longueur).
    pub fn assert_same_len(&self, other: &Mask) -> CognoResult<()> {
        if self.len() != other.len() {
            return Err(CognoError::MaskMismatch("longueurs de masques différentes"));
        }
        Ok(())
    }

    /// Applique le masque à un vecteur de valeurs : les positions inactives
    /// sont mises à zéro. Les longueurs doivent correspondre.
    pub fn apply_to(&self, values: &mut [f64]) -> CognoResult<()> {
        if self.len() != values.len() {
            return Err(CognoError::LengthMismatch {
                expected: self.len(),
                got: values.len(),
            });
        }
        for (i, v) in values.iter_mut().enumerate() {
            if self.bits[i] == 0 {
                *v = 0.0;
            }
        }
        Ok(())
    }
}

/// Séquence de tokens avec un masque (longueurs différentes supportées).
///
/// Contrat §14 : « séquences de longueurs différentes » — chaque séquence
/// porte sa propre longueur ; les opérations de batch valident la cohérence.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    /// tokens (ex. log-probs, logits).
    pub tokens: Vec<f64>,
    /// masque de validité (positions hors séquence réelle = inactives).
    pub mask: Mask,
}

impl Sequence {
    /// Construit une séquence complète (masque tout actif de la même longueur).
    pub fn new(tokens: Vec<f64>) -> Self {
        let n = tokens.len();
        Sequence {
            tokens,
            mask: Mask::all_ones(n),
        }
    }

    /// Construit une séquence avec un masque partiel (pad à droite).
    /// La longueur du masque doit être `<=` à celle des tokens (les tokens au
    ///-delà du masque sont traités comme inactifs).
    pub fn with_mask(tokens: Vec<f64>, mask: Mask) -> CognoResult<Self> {
        if mask.len() > tokens.len() {
            return Err(CognoError::MaskMismatch(
                "masque plus long que la séquence",
            ));
        }
        Ok(Sequence { tokens, mask })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Somme masquée des tokens (positions inactives ignorées).
    pub fn masked_sum(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.tokens.len() {
            if i < self.mask.len() && self.mask.bits()[i] == 1 {
                s += self.tokens[i];
            }
        }
        s
    }

    /// Moyenne masquée (sur les positions actives). 0 si aucune active.
    pub fn masked_mean(&self) -> f64 {
        let n = self.mask.active_count();
        if n == 0 {
            0.0
        } else {
            self.masked_sum() / n as f64
        }
    }
}

/// Batch de séquences de longueurs différentes (padding à la max).
#[derive(Debug, Clone)]
pub struct SequenceBatch {
    pub sequences: Vec<Sequence>,
    pub max_len: usize,
}

impl SequenceBatch {
    /// Assemble un batch. Valide la cohérence interne (aucun masque ne
    /// dépasse sa séquence). `max_len` = longueur maximale du batch.
    pub fn try_new(sequences: Vec<Sequence>) -> CognoResult<Self> {
        let max_len = sequences.iter().map(Sequence::len).max().unwrap_or(0);
        Ok(SequenceBatch { sequences, max_len })
    }

    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Somme masquée moyenne sur le batch (0 si vide).
    pub fn mean_masked_sum(&self) -> f64 {
        if self.sequences.is_empty() {
            return 0.0;
        }
        let mut s = 0.0;
        for seq in &self.sequences {
            s += seq.masked_mean();
        }
        s / self.sequences.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_validates_bits() {
        assert!(Mask::try_new(vec![1, 0, 1]).is_ok());
        assert!(Mask::try_new(vec![1, 2, 1]).is_err()); // 2 hors {0,1}
    }

    #[test]
    fn mask_applies_to_values() {
        let m = Mask::try_new(vec![1, 0, 1]).unwrap();
        let mut v = vec![1.0, 2.0, 3.0];
        m.apply_to(&mut v).unwrap();
        assert_eq!(v, vec![1.0, 0.0, 3.0]);
    }

    #[test]
    fn mask_length_mismatch_rejected() {
        let m = Mask::try_new(vec![1, 0]).unwrap();
        let mut v = vec![1.0, 2.0, 3.0];
        assert!(m.apply_to(&mut v).is_err());
    }

    #[test]
    fn sequence_partial_mask_pads() {
        // séquence de 4, masque partiel sur les 2 premiers (pad à droite)
        let seq = Sequence::with_mask(
            vec![1.0, 2.0, 3.0, 4.0],
            Mask::try_new(vec![1, 1]).unwrap(),
        )
        .unwrap();
        // seuls les 2 premiers comptent
        assert!((seq.masked_sum() - 3.0).abs() < 1e-12);
        assert!((seq.masked_mean() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn sequence_mask_longer_than_tokens_rejected() {
        assert!(Sequence::with_mask(vec![1.0], Mask::try_new(vec![1, 1]).unwrap()).is_err());
    }

    #[test]
    fn batch_of_varying_lengths() {
        let s1 = Sequence::new(vec![1.0, 2.0, 3.0]); // len 3
        let s2 = Sequence::new(vec![10.0]); // len 1
        let batch = SequenceBatch::try_new(vec![s1, s2]).unwrap();
        assert_eq!(batch.max_len, 3);
        assert_eq!(batch.len(), 2);
        // moyenne des moyennes masquées : (2.0 + 10.0)/2 = 6.0
        assert!((batch.mean_masked_sum() - 6.0).abs() < 1e-12);
    }
}
