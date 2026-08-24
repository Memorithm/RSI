//! Mémoire contextuelle `C` adossée à **OctaSoma** (feature `octasoma`).
//!
//! Le stockage canonique est [`octasoma::HybridMemory`] : le tier SimHash +
//! rerank cosine exact fournit le rappel précis, tandis que la projection 3-D
//! reste une lentille explicable/visualisable sur les mêmes souvenirs.
//!
//! OctaSoma reste optionnel : sans la feature `octasoma`, le cœur RSI ne dépend
//! pas du moteur de mémoire externe.

use octasoma::{Explanation, HybridMemory, QueryStrategy};

use crate::memory::ContextMemory;

const SIMHASH_BITS: usize = 256;
const DEFAULT_SHORTLIST: usize = 256;

/// Mémoire contextuelle RSI indexée par l'OctaSoma canonique.
pub struct OctaSomaMemory {
    mem: HybridMemory,
}

impl OctaSomaMemory {
    /// `high_dim` est la dimension exacte des embeddings RSI.
    pub fn new(high_dim: usize, seed: u64) -> Self {
        Self {
            mem: HybridMemory::new(high_dim, seed, SIMHASH_BITS)
                .with_shortlist(DEFAULT_SHORTLIST),
        }
    }

    /// Lentille spatiale explicable sur les mêmes items que le tier de précision.
    /// Cette vue n'est jamais utilisée comme autorité de rappel par `ContextMemory`.
    pub fn explain(&self, query: &[f32], k: usize) -> Option<Explanation> {
        self.mem.explain(query, k)
    }
}

impl ContextMemory for OctaSomaMemory {
    fn remember(&mut self, embedding: &[f32], payload: &[u8]) {
        // `HybridMemory::insert` est atomique entre la lentille 3-D et le tier
        // précis : une entrée invalide n'est stockée dans aucun des deux.
        let _ = self.mem.insert(embedding, payload);
    }

    fn recall(&self, query: &[f32], k: usize) -> Vec<Vec<u8>> {
        self.mem
            .query(query, QueryStrategy::PrecisionSketch, k)
            .into_iter()
            .map(|(payload, _score)| payload.to_vec())
            .collect()
    }

    fn len(&self) -> usize {
        self.mem.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_recall_uses_precision_tier() {
        let mut m = OctaSomaMemory::new(8, 42);
        let e1 = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let e2 = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        m.remember(&e1, b"a");
        m.remember(&e2, b"b");
        assert_eq!(m.len(), 2);

        let r = m.recall(&e1, 1);
        assert_eq!(r.first().map(Vec::as_slice), Some(b"a".as_slice()));
    }

    #[test]
    fn spatial_lens_explains_the_same_store() {
        let mut m = OctaSomaMemory::new(8, 42);
        let e = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        m.remember(&e, b"experience");
        assert!(m.explain(&e, 1).is_some());
    }

    /// Garde de régression **M1** (résolu amont octasoma v0.5, rev `145761a`) :
    /// le contrat d'insertion est honnête — une entrée invalide (dimension
    /// fausse ou embedding non fini) n'est stockée dans **aucun** des deux tiers
    /// et n'enfle pas la mémoire ; la lentille 3-D reste cohérente.
    #[test]
    fn m1_honest_insert_rejects_invalid_input() {
        let mut m = OctaSomaMemory::new(8, 42);
        let good = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Dimension fausse / embedding non fini → rien n'est stocké.
        m.remember(&good[..4], b"dimension fausse");
        m.remember(&[f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], b"nan");
        m.remember(&[f32::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], b"inf");
        assert_eq!(m.len(), 0, "une entrée invalide ne doit pas être stockée");
        assert!(m.recall(&good, 3).is_empty());
        // La lentille 3-D reste vide aussi : aucune explication ne peut
        // invoquer un voisin inexistant.
        let lens = m.explain(&good, 1).expect("requête valide → Some");
        assert!(lens.neighbors.is_empty());
        let total: usize = lens.zoom_path.iter().map(|r| r.count).sum();
        assert_eq!(total, 0, "aucun souvenir dans les régions zoomées");

        // Sémantique doublons assumée amont (`duplicate_points_all_retained`) :
        // chaque `remember` compte — aucun dédoublonnage implicite.
        m.remember(&good, b"a");
        m.remember(&good, b"b");
        assert_eq!(m.len(), 2);
        assert_eq!(m.recall(&good, 2).len(), 2);
    }
}
