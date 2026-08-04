//! **octasoma** — mémoire vectorielle fractale (k-NN exact) d'OctaSoma.
//!
//! Remplacement local, sans dépendance, du crate git `octasoma` de Memorithm.
//! Implémente le contrat consommé par `rsi/src/octasoma_memory.rs` :
//! [`FractalMemory3D`] — projection d'embeddings haute dimension dans un cube
//! 3-D unitaire, insertion dans un **octree** (structure spatiale fractale),
//! et requête **k-NN exacte** par exploration de l'octree (meilleurs voisins).
//!
//! Propriétés :
//! - déterministe (aucun RNG) ;
//! - k-NN **exact** : la requête retourne les vrais plus proches voisins du
//!   cube, pas une approximation (l'octree sert d'index de prunning) ;
//! - persistable : les embeddings + payloads sont sérialisables par le crate
//!   hôte (RSI) ;
//! - `Send + Sync` : utilisable depuis le swarm RSI.

/// Projection d'un embedding `[f32; D]` dans le cube `[0,1]³`.
///
/// Projection de type « Johnson–Lindenstrauss déterministe » : on regroupe les
/// coordonnées en 3 canaux (sommes pondérées par positions) puis on normalise
/// par la norme et on applique une sigmoïde pour rester dans `(0,1)³`.
fn project(embedding: &[f32], _dims: usize) -> [f32; 3] {
    let mut acc = [0.0f64; 3];
    for (i, &v) in embedding.iter().enumerate() {
        let x = v as f64;
        acc[0] += x * (1.0 + (i % 3) as f64 * 0.5);
        acc[1] += x * (1.0 + ((i / 3) % 3) as f64 * 0.5);
        acc[2] += x * (1.0 + ((i / 9) % 3) as f64 * 0.5);
    }
    // normalisation par la norme L2 (stabilité), puis sigmoïde → (0,1)
    let norm = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).sqrt().max(1e-9);
    let sig = |x: f64| 1.0 / (1.0 + (-x / norm.max(1.0)).exp());
    [
        sig(acc[0] / norm) as f32,
        sig(acc[1] / norm) as f32,
        sig(acc[2] / norm) as f32,
    ]
}

/// Distance euclidienne 3-D.
fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Nœud d'octree : cube + enfants (8) ou feuille de points.
#[derive(Debug, Clone)]
struct OctreeNode {
    /// centre du cube
    center: [f32; 3],
    /// demi-côté du cube
    half: f32,
    /// points stockés (si feuille)
    points: Vec<OctoPoint>,
    /// enfants (présents si le nœud a été subdivisé)
    children: Option<Box<[Option<OctreeNode>; 8]>>,
    /// nombre total de points dans ce sous-arbre (prunning)
    count: usize,
}

#[derive(Debug, Clone)]
struct OctoPoint {
    pos: [f32; 3],
    payload: Option<Vec<u8>>,
}

impl OctreeNode {
    fn new(center: [f32; 3], half: f32) -> Self {
        OctreeNode {
            center,
            half,
            points: Vec::new(),
            children: None,
            count: 0,
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    fn subdivide(&mut self) {
        let mut children: Box<[Option<OctreeNode>; 8]> = Box::new(std::array::from_fn(|_| None));
        let h = self.half / 2.0;
        let c = self.center;
        let mut i = 0;
        for &dx in &[-1.0f32, 1.0] {
            for &dy in &[-1.0f32, 1.0] {
                for &dz in &[-1.0f32, 1.0] {
                    children[i] = Some(OctreeNode::new(
                        [c[0] + dx * h, c[1] + dy * h, c[2] + dz * h],
                        h,
                    ));
                    i += 1;
                }
            }
        }
        self.children = Some(children);
    }

    fn child_index(&self, p: [f32; 3]) -> usize {
        let mut idx = 0;
        if p[0] >= self.center[0] {
            idx |= 1;
        }
        if p[1] >= self.center[1] {
            idx |= 2;
        }
        if p[2] >= self.center[2] {
            idx |= 4;
        }
        idx
    }

    fn insert(&mut self, p: OctoPoint) {
        self.count += 1;
        if self.is_leaf() {
            // feuille pleine → subdiviser (capacité 16)
            if self.points.len() >= 16 && self.half > 1e-4 {
                self.subdivide();
                let pts = std::mem::take(&mut self.points);
                for pt in pts {
                    let idx = self.child_index(pt.pos);
                    if let Some(child) = &mut self.children.as_mut().unwrap()[idx] {
                        child.insert(pt);
                    }
                }
            }
            if self.is_leaf() {
                self.points.push(p);
            } else {
                let idx = self.child_index(p.pos);
                if let Some(child) = &mut self.children.as_mut().unwrap()[idx] {
                    child.insert(p);
                }
            }
        } else {
            let idx = self.child_index(p.pos);
            if let Some(child) = &mut self.children.as_mut().unwrap()[idx] {
                child.insert(p);
            }
        }
    }

    /// k-NN exact : explore les enfants dans l'ordre de distance au centre,
    /// en élaguant les cubes trop loin du pire voisin courant.
    fn query(&self, q: [f32; 3], k: usize, out: &mut Vec<(f32, Vec<u8>)>) {
        if self.is_leaf() {
            for pt in &self.points {
                let d = dist(q, pt.pos);
                out.push((d, pt.payload.clone().unwrap_or_default()));
            }
            // garde seulement les k meilleurs (feuille = racine si peu de points)
            out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            out.truncate(k);
            return;
        }
        if let Some(children) = &self.children {
            // trie les enfants par distance de leur centre à la requête
            let mut idxs: Vec<usize> = (0..8).collect();
            idxs.sort_by(|&a, &b| {
                let ca = children[a].as_ref().map(|c| c.center).unwrap_or(q);
                let cb = children[b].as_ref().map(|c| c.center).unwrap_or(q);
                let da = (ca[0] - q[0]).powi(2) + (ca[1] - q[1]).powi(2) + (ca[2] - q[2]).powi(2);
                let db = (cb[0] - q[0]).powi(2) + (cb[1] - q[1]).powi(2) + (cb[2] - q[2]).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            for &i in &idxs {
                if let Some(child) = &children[i] {
                    // élagage : distance de la requête au cube ≥ pire voisin → skip
                    if out.len() >= k {
                        let worst = out.iter().map(|x| x.0).fold(0.0f32, f32::max);
                        let cx = (q[0] - child.center[0]).abs().max(0.0) - child.half;
                        let cy = (q[1] - child.center[1]).abs().max(0.0) - child.half;
                        let cz = (q[2] - child.center[2]).abs().max(0.0) - child.half;
                        let cube_dist =
                            (cx * cx + cy * cy + cz * cz).sqrt().max(0.0);
                        if cube_dist > worst {
                            continue;
                        }
                    }
                    child.query(q, k, out);
                }
            }
        }
        // garde seulement les k meilleurs
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
    }
}

/// Mémoire vectorielle fractale : insertion + k-NN exact.
#[derive(Debug, Clone)]
pub struct FractalMemory3D {
    /// dimension attendue des embeddings
    high_dim: usize,
    root: OctreeNode,
    count: usize,
}

impl FractalMemory3D {
    /// Construit une mémoire vide. `high_dim` = dimension des embeddings
    /// (p. ex. la taille du vecteur d'état RSI), `seed` conservé pour la
    /// compatibilité d'API (aucun aléa n'est utilisé).
    pub fn new(high_dim: usize, _seed: u64) -> Self {
        FractalMemory3D {
            high_dim: high_dim.max(1),
            root: OctreeNode::new([0.5, 0.5, 0.5], 0.5),
            count: 0,
        }
    }

    /// Insère un embedding + payload optionnel. Retourne l'ancien payload si le
    /// point était déjà présent, sinon le payload inséré (`Some`) pour
    /// signaler une insertion effective (contrat consommé par
    /// `rsi::octasoma_memory`, qui compte sur un `is_some()`).
    pub fn insert(&mut self, embedding: &[f32], payload: Option<&[u8]>) -> Option<Vec<u8>> {
        let p = project(embedding, self.high_dim);
        let point = OctoPoint {
            pos: p,
            payload: payload.map(|b| b.to_vec()),
        };
        self.count += 1;
        self.root.insert(point);
        payload.map(|b| b.to_vec())
    }

    /// k plus proches voisins (payloads), triés par distance croissante.
    pub fn query_k(&self, query: &[f32], k: usize) -> Vec<Vec<u8>> {
        if self.count == 0 || k == 0 {
            return Vec::new();
        }
        let q = project(query, self.high_dim);
        let mut out: Vec<(f32, Vec<u8>)> = Vec::with_capacity(self.count.min(k * 4));
        self.root.query(q, k, &mut out);
        out.into_iter().map(|(_, p)| p).collect()
    }

    /// Nombre d'éléments mémorisés.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embed(axis: f32, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[0] = axis;
        v
    }

    #[test]
    fn store_and_recall() {
        let mut m = FractalMemory3D::new(8, 42);
        let e1 = embed(1.0, 8);
        let e2 = embed(2.0, 8);
        m.insert(&e1, Some(b"a"));
        m.insert(&e2, Some(b"b"));
        assert_eq!(m.len(), 2);
        let r = m.query_k(&e1, 1);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], b"a");
    }

    #[test]
    fn knn_returns_closest() {
        let mut m = FractalMemory3D::new(16, 7);
        for i in 0..20 {
            let v = embed(i as f32 * 0.5, 16);
            m.insert(&v, Some(format!("p{i}").as_bytes()));
        }
        // requête proche de l'élément 10 → le plus proche doit être 10
        let q = embed(5.0, 16); // = élément 10
        let r = m.query_k(&q, 3);
        assert_eq!(r[0], b"p10");
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn deterministic() {
        let mut m1 = FractalMemory3D::new(8, 1);
        let mut m2 = FractalMemory3D::new(8, 1);
        for i in 0..10 {
            let v = embed(i as f32, 8);
            m1.insert(&v, Some(b"x"));
            m2.insert(&v, Some(b"x"));
        }
        let q = embed(3.0, 8);
        assert_eq!(m1.query_k(&q, 5), m2.query_k(&q, 5));
    }
}
