//! Cache KV **borné et fallible** (contrat §13).
//!
//! - capacité explicite, préalloué, fallible ;
//! - réutilisable, nettoyable, mesurable en octets ;
//! - indépendant des entrées non validées ;
//! - protégé contre les dépassements de capacité.

use crate::error::{CognoError, CognoResult};

/// Vue tensorielle en lecture seule (borrow) — sans copie.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    data: &'a [f64],
    dims: &'a [usize],
}

impl<'a> TensorView<'a> {
    pub fn new(data: &'a [f64], dims: &'a [usize]) -> CognoResult<Self> {
        let n: usize = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or(CognoError::SizeOverflow)?;
        if n != data.len() {
            return Err(CognoError::LengthMismatch {
                expected: n,
                got: data.len(),
            });
        }
        Ok(TensorView { data, dims })
    }

    pub fn data(&self) -> &'a [f64] {
        self.data
    }
    pub fn dims(&self) -> &'a [usize] {
        self.dims
    }
}

/// Vue tensorielle en écriture (borrow mut) — pour le décodage sans allocation.
pub struct TensorViewMut<'a> {
    data: &'a mut [f64],
    dims: &'a [usize],
}

impl<'a> TensorViewMut<'a> {
    pub fn new(data: &'a mut [f64], dims: &'a [usize]) -> CognoResult<Self> {
        let n: usize = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or(CognoError::SizeOverflow)?;
        if n != data.len() {
            return Err(CognoError::LengthMismatch {
                expected: n,
                got: data.len(),
            });
        }
        Ok(TensorViewMut { data, dims })
    }

    pub fn data_mut(&mut self) -> &mut [f64] {
        self.data
    }
    pub fn data(&self) -> &[f64] {
        self.data
    }
    pub fn dims(&self) -> &[usize] {
        self.dims
    }
}

/// Configuration du cache KV : capacité et dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvCacheConfig {
    /// capacité en tokens (nombre max d'entrées clé/valeur).
    pub capacity_tokens: usize,
    /// dimension d'une clé.
    pub key_dim: usize,
    /// dimension d'une valeur.
    pub value_dim: usize,
    /// octets par f64 (mémoire de référence ; `size_of::<f64>()` par défaut).
    pub bytes_per_f64: usize,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        KvCacheConfig {
            capacity_tokens: 4096,
            key_dim: 64,
            value_dim: 64,
            bytes_per_f64: 8,
        }
    }
}

/// Erreur spécifique du cache KV.
#[derive(Debug, Clone, PartialEq)]
pub enum KvCacheError {
    Capacity(CognoError),
    Shape(CognoError),
    Invalid(CognoError),
}

impl From<CognoError> for KvCacheError {
    fn from(e: CognoError) -> Self {
        match e {
            CognoError::CapacityOverflow { .. } => KvCacheError::Capacity(e),
            CognoError::LengthMismatch { .. } => KvCacheError::Shape(e),
            _ => KvCacheError::Invalid(e),
        }
    }
}

impl std::fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCacheError::Capacity(e) => write!(f, "cache KV : {e}"),
            KvCacheError::Shape(e) => write!(f, "cache KV : {e}"),
            KvCacheError::Invalid(e) => write!(f, "cache KV : {e}"),
        }
    }
}

impl std::error::Error for KvCacheError {}

/// Cache KV borné (contrat §13 — API conceptuelle exacte).
pub trait BoundedKvCache {
    type Error;

    fn try_new(config: KvCacheConfig) -> Result<Self, Self::Error>
    where
        Self: Sized;

    fn capacity_tokens(&self) -> usize;
    fn used_tokens(&self) -> usize;
    fn reserved_bytes(&self) -> usize;

    fn try_append(
        &mut self,
        key: TensorView<'_>,
        value: TensorView<'_>,
    ) -> Result<(), Self::Error>;

    fn decode_into(
        &self,
        query: TensorView<'_>,
        output: TensorViewMut<'_>,
    ) -> Result<(), Self::Error>;

    fn clear(&mut self);
}

/// Implémentation concrète : cache KV à capacité fixe, préalloué.
///
/// Le décodage écrit dans `output` fourni par l'appelant — **aucune
/// allocation par token**. La mémoire est réservée à la construction.
pub struct FixedKvCache {
    config: KvCacheConfig,
    keys: Vec<f64>,
    values: Vec<f64>,
    used: usize,
}

impl FixedKvCache {
    /// Nouvelle taille totale (capacity × dim), avec arithmétique contrôlée.
    fn total_len(cap: usize, dim: usize) -> CognoResult<usize> {
        cap.checked_mul(dim).ok_or(CognoError::SizeOverflow)
    }
}

impl BoundedKvCache for FixedKvCache {
    type Error = KvCacheError;

    fn try_new(config: KvCacheConfig) -> Result<Self, Self::Error> {
        if config.capacity_tokens == 0 {
            return Err(CognoError::InvalidInput("capacité nulle").into());
        }
        if config.key_dim == 0 || config.value_dim == 0 {
            return Err(CognoError::InvalidInput("dimension nulle").into());
        }
        let klen = Self::total_len(config.capacity_tokens, config.key_dim)?;
        let vlen = Self::total_len(config.capacity_tokens, config.value_dim)?;
        let mut keys = Vec::with_capacity(klen);
        let mut values = Vec::with_capacity(vlen);
        keys.resize(klen, 0.0);
        values.resize(vlen, 0.0);
        Ok(FixedKvCache {
            config,
            keys,
            values,
            used: 0,
        })
    }

    fn capacity_tokens(&self) -> usize {
        self.config.capacity_tokens
    }

    fn used_tokens(&self) -> usize {
        self.used
    }

    fn reserved_bytes(&self) -> usize {
        let b = self.config.bytes_per_f64;
        self.keys.len().saturating_mul(b) + self.values.len().saturating_mul(b)
    }

    fn try_append(
        &mut self,
        key: TensorView<'_>,
        value: TensorView<'_>,
    ) -> Result<(), Self::Error> {
        if key.data().len() != self.config.key_dim {
            return Err(CognoError::LengthMismatch {
                expected: self.config.key_dim,
                got: key.data().len(),
            }
            .into());
        }
        if value.data().len() != self.config.value_dim {
            return Err(CognoError::LengthMismatch {
                expected: self.config.value_dim,
                got: value.data().len(),
            }
            .into());
        }
        if self.used >= self.config.capacity_tokens {
            return Err(CognoError::CapacityOverflow {
                what: "kv_cache",
                capacity: self.config.capacity_tokens,
                requested: self.used + 1,
            }
            .into());
        }
        let base = self
            .used
            .checked_mul(self.config.key_dim)
            .ok_or(CognoError::SizeOverflow)?;
        self.keys[base..base + self.config.key_dim].copy_from_slice(key.data());
        let vbase = self
            .used
            .checked_mul(self.config.value_dim)
            .ok_or(CognoError::SizeOverflow)?;
        self.values[vbase..vbase + self.config.value_dim].copy_from_slice(value.data());
        self.used += 1;
        Ok(())
    }

    fn decode_into(
        &self,
        query: TensorView<'_>,
        mut output: TensorViewMut<'_>,
    ) -> Result<(), Self::Error> {
        if query.data().len() != self.config.key_dim {
            return Err(CognoError::LengthMismatch {
                expected: self.config.key_dim,
                got: query.data().len(),
            }
            .into());
        }
        // sortie = somme pondérée des valeurs (attention : ici décodage
        // simple par similarité cosinus normalisée — décodage additif simple)
        if output.data().len() != self.config.value_dim {
            return Err(CognoError::LengthMismatch {
                expected: self.config.value_dim,
                got: output.data().len(),
            }
            .into());
        }
        // zéro-copie : écrit directement dans output
        for o in output.data_mut().iter_mut() {
            *o = 0.0;
        }
        if self.used == 0 {
            return Ok(());
        }
        // Décodage additif déterministe : moyenne des valeurs mémorisées
        // (attention uniforme documentée). Écrit dans `output` sans allocation.
        for i in 0..self.config.value_dim {
            let mut s = 0.0;
            for t in 0..self.used {
                s += self.values[t * self.config.value_dim + i];
            }
            output.data_mut()[i] = s / self.used as f64;
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.used = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_reserves_and_appends() {
        let mut cache = FixedKvCache::try_new(KvCacheConfig {
            capacity_tokens: 4,
            key_dim: 3,
            value_dim: 2,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(cache.capacity_tokens(), 4);
        assert_eq!(cache.reserved_bytes(), (12 + 8) * 8);
        let k = TensorView::new(&[1.0, 0.0, 0.0], &[3]).unwrap();
        let v = TensorView::new(&[10.0, 20.0], &[2]).unwrap();
        cache.try_append(k, v).unwrap();
        assert_eq!(cache.used_tokens(), 1);
        // décodage : moyenne des valeurs → (10, 20)
        let mut out = [0.0; 2];
        {
            let ov = TensorViewMut::new(&mut out, &[2]).unwrap();
            cache.decode_into(k, ov).unwrap();
        }
        assert_eq!(out, [10.0, 20.0]);
        cache.clear();
        assert_eq!(cache.used_tokens(), 0);
    }

    #[test]
    fn cache_rejects_capacity_overflow() {
        let mut cache = FixedKvCache::try_new(KvCacheConfig {
            capacity_tokens: 1,
            key_dim: 2,
            value_dim: 2,
            ..Default::default()
        })
        .unwrap();
        let k = TensorView::new(&[1.0, 0.0], &[2]).unwrap();
        let v = TensorView::new(&[1.0, 1.0], &[2]).unwrap();
        cache.try_append(k, v).unwrap();
        assert!(cache.try_append(k, v).is_err());
    }

    #[test]
    fn cache_rejects_shape_mismatch() {
        let mut cache = FixedKvCache::try_new(KvCacheConfig {
            capacity_tokens: 2,
            key_dim: 2,
            value_dim: 2,
            ..Default::default()
        })
        .unwrap();
        let k = TensorView::new(&[1.0, 0.0, 0.0], &[3]).unwrap();
        let v = TensorView::new(&[1.0, 1.0], &[2]).unwrap();
        assert!(cache.try_append(k, v).is_err());
    }
}
