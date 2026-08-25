//! Mémoire contextuelle réelle pour la composante `C` (Phase 3).
//!
//! Le trait [`ContextMemory`] abstrait un magasin épisodique : l'agent y écrit
//! un *embedding* de son état à chaque pas et peut **rappeler** les contextes
//! passés les plus proches. Le cœur fournit une implémentation `std`-only
//! ([`LinearContextMemory`], k-NN exact par balayage) ; la feature `octasoma`
//! fournit un backend fractal indexé (cf. `octasoma_memory`).
//!
//! La mémoire est *attachée* à l'agent sans entrer dans la dynamique de
//! `SI_global` : elle enrichit `C` (épisodique, interrogeable) sans toucher aux
//! garde-fous de stabilité (§4).

/// Magasin de mémoire contextuelle : écriture d'embeddings + rappel k-NN.
pub trait ContextMemory {
    /// Mémorise un embedding et sa charge utile (payload sérialisé).
    fn remember(&mut self, embedding: &[f32], payload: &[u8]);
    /// Rappelle les `k` payloads dont l'embedding est le plus proche de `query`.
    fn recall(&self, query: &[f32], k: usize) -> Vec<Vec<u8>>;
    /// Nombre d'éléments mémorisés.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Mémoire contextuelle `std`-only : k-NN exact par balayage linéaire
/// (distance euclidienne). Simple, sans dépendance ; convient comme défaut et
/// comme référence face au backend OctaSoma.
///
/// Contrat durci (audit m15, miroir de la garde M1 côté OctaSoma) :
/// - la dimension est **verrouillée au premier** `remember` ; un embedding de
///   dimension différente ou non fini est ignoré (comme le backend canonique) ;
/// - capacité plafonnée (`with_capacity`) : au-delà, éviction du plus ancien
///   — les runs DGM longs ne croissent plus sans borne.
#[derive(Debug)]
pub struct LinearContextMemory {
    items: std::collections::VecDeque<(Vec<f32>, Vec<u8>)>,
    dim: Option<usize>,
    capacity: usize,
}

/// Capacité par défaut (éléments) : bornes la croissance mémoire des runs
/// longs tout en couvrant largement les campagnes usuelles.
pub const DEFAULT_LINEAR_MEMORY_CAPACITY: usize = 10_000;

impl Default for LinearContextMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearContextMemory {
    pub fn new() -> Self {
        LinearContextMemory::with_capacity(DEFAULT_LINEAR_MEMORY_CAPACITY)
    }

    /// Mémoire plafonnée à `capacity` items (éviction FIFO au-delà).
    pub fn with_capacity(capacity: usize) -> Self {
        LinearContextMemory {
            items: std::collections::VecDeque::new(),
            dim: None,
            capacity: capacity.max(1),
        }
    }
}

impl ContextMemory for LinearContextMemory {
    fn remember(&mut self, embedding: &[f32], payload: &[u8]) {
        // entrée invalide = ignorée, exactement comme `HybridMemory` amont
        if embedding.iter().any(|x| !x.is_finite()) {
            return;
        }
        match self.dim {
            Some(d) if d != embedding.len() => return,
            None => self.dim = Some(embedding.len()),
            _ => {}
        }
        if self.items.len() >= self.capacity {
            self.items.pop_front();
        }
        self.items.push_back((embedding.to_vec(), payload.to_vec()));
    }

    fn recall(&self, query: &[f32], k: usize) -> Vec<Vec<u8>> {
        let dist2 = |v: &[f32]| -> f32 {
            v.iter()
                .zip(query)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
        };
        let mut scored: Vec<(f32, &Vec<u8>)> = self
            .items
            .iter()
            .map(|(e, p)| (dist2(e), p))
            .filter(|(d, _)| d.is_finite())
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, p)| p.clone()).collect()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Audit m15 : entrée non finie ignorée ; dimension verrouillée ;
    /// capacité plafonnée avec éviction FIFO.
    #[test]
    fn linear_memory_validates_and_bounds() {
        let mut m = LinearContextMemory::with_capacity(2);
        let bad = [f32::NAN, 0.0];
        m.remember(&bad, b"nan");
        assert_eq!(m.len(), 0, "embedding NaN refusé");

        m.remember(&[1.0, 0.0], b"a");
        m.remember(&[3.0, 0.0, 1.0], b"c"); // dimension différente : ignoré
        assert_eq!(m.len(), 1, "dimension verrouillée au premier insert");

        m.remember(&[0.5, 0.0], b"b");
        m.remember(&[0.9, 0.0], b"d"); // capacité 2 atteinte → "a" évincé
        assert_eq!(m.len(), 2);
        let r = m.recall(&[0.0, 0.0], 10);
        assert_eq!(r, vec![b"b".to_vec(), b"d".to_vec()]);
        assert!(!r.iter().any(|p| p == b"a"), "le plus ancien a été évincé");
    }

    #[test]
    fn recall_returns_nearest() {
        let mut m = LinearContextMemory::new();
        m.remember(&[0.0, 0.0], b"origin");
        m.remember(&[10.0, 10.0], b"far");
        m.remember(&[1.0, 1.0], b"near");
        assert_eq!(m.len(), 3);
        let r = m.recall(&[0.9, 0.9], 1);
        assert_eq!(r[0], b"near");
        let r2 = m.recall(&[0.0, 0.0], 2);
        assert_eq!(r2[0], b"origin");
    }
}
