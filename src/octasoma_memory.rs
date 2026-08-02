//! Mémoire contextuelle `C` adossée à **OctaSoma** (feature `octasoma`) — Phase 3.
//!
//! Branche le moteur de mémoire fractale `octasoma::FractalMemory3D` (projection
//! 3-D + octree PR, k-NN exact) derrière le trait [`ContextMemory`], donnant à
//! la composante `C` un véritable magasin vectoriel indexé et persistable, à la
//! place d'un simple vecteur abstrait.

use octasoma::FractalMemory3D;

use crate::memory::ContextMemory;

/// Mémoire contextuelle indexée par OctaSoma.
pub struct OctaSomaMemory {
    mem: FractalMemory3D,
    count: usize,
}

impl OctaSomaMemory {
    /// `high_dim` = dimension des embeddings (p. ex. la taille du vecteur d'état).
    pub fn new(high_dim: usize, seed: u64) -> Self {
        OctaSomaMemory {
            mem: FractalMemory3D::new(high_dim, seed),
            count: 0,
        }
    }
}

impl ContextMemory for OctaSomaMemory {
    fn remember(&mut self, embedding: &[f32], payload: &[u8]) {
        if self.mem.insert(embedding, Some(payload)).is_some() {
            self.count += 1;
        }
    }

    fn recall(&self, query: &[f32], k: usize) -> Vec<Vec<u8>> {
        self.mem
            .query_k(query, k)
            .into_iter()
            .map(|p| p.to_vec())
            .collect()
    }

    fn len(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_recall() {
        let mut m = OctaSomaMemory::new(8, 42);
        let e1 = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let e2 = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        m.remember(&e1, b"a");
        m.remember(&e2, b"b");
        assert_eq!(m.len(), 2);
        let r = m.recall(&e1, 1);
        assert!(!r.is_empty());
    }
}
