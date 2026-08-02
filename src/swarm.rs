//! ⚙️ Loop Engineering — **L8 : parallélisme & portefeuille de boucles**.
//!
//! Exécute un **essaim** d'agents indépendants (graines distinctes) en
//! **parallèle** (threads `std`, sans dépendance) et sélectionne le meilleur du
//! portefeuille (par `SI_safe`). Chaque agent est construit *dans* son thread
//! par une closure `build(seed)`, donc l'agent ne traverse jamais les threads —
//! seuls les résultats scalaires reviennent. Déterministe par graine.

#![allow(clippy::items_after_test_module)]

use crate::agent::RSIAgent;

/// Résultat d'un membre de l'essaim.
#[derive(Clone, Copy, Debug)]
pub struct SwarmMember {
    pub seed: u64,
    pub si_global: f64,
    pub si_safe: f64,
}

/// Résultat agrégé d'un essaim.
#[derive(Clone, Debug)]
pub struct SwarmResult {
    pub members: Vec<SwarmMember>,
    pub best_index: usize,
}

impl SwarmResult {
    pub fn best(&self) -> SwarmMember {
        self.members[self.best_index]
    }
}

/// Exécute `size` boucles en parallèle ; `build(seed)` construit chaque agent
/// (graines `base_seed..base_seed+size`), chacune avancée de `steps` pas. Le
/// meilleur membre est sélectionné par `SI_safe`.
pub fn run_swarm<F>(size: usize, base_seed: u64, steps: usize, build: F) -> SwarmResult
where
    F: Fn(u64) -> RSIAgent + Sync,
{
    let size = size.max(1);
    // Membre marqué invalide (jamais sélectionné, `SI_safe = -∞`) : sert de
    // repli quand un membre panique ou ne produit aucun rapport.
    let invalid = |seed: u64| SwarmMember {
        seed,
        si_global: 0.0,
        si_safe: f64::NEG_INFINITY,
    };
    let members: Vec<SwarmMember> = std::thread::scope(|scope| {
        let handles: Vec<(u64, _)> = (0..size)
            .map(|i| {
                let seed = base_seed + i as u64;
                let build = &build;
                let handle = scope.spawn(move || {
                    let mut agent = build(seed);
                    let reports = agent.run(steps);
                    match reports.last() {
                        Some(last) => SwarmMember {
                            seed,
                            si_global: last.si_global,
                            si_safe: last.si_safe,
                        },
                        // run(0) ⇒ aucun rapport : membre invalide plutôt que panic.
                        None => invalid(seed),
                    }
                });
                (seed, handle)
            })
            .collect();
        // Un membre qui panique est *isolé* (marqué invalide) au lieu de faire
        // s'effondrer tout l'essaim via `join().unwrap()`.
        handles
            .into_iter()
            .map(|(seed, h)| h.join().unwrap_or_else(|_| invalid(seed)))
            .collect()
    });

    let best_index = members
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.si_safe.partial_cmp(&b.si_safe).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    SwarmResult {
        members,
        best_index,
    }
}

/// Essaim de démonstration (agents `RSIAgent::demo`) — pratique pour benchmark.
pub fn run_swarm_demo(size: usize, base_seed: u64, steps: usize) -> SwarmResult {
    run_swarm(size, base_seed, steps, RSIAgent::demo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swarm_runs_and_selects_best() {
        let res = run_swarm_demo(6, 100, 30);
        assert_eq!(res.members.len(), 6);
        let best = res.best();
        // le meilleur a bien le SI_safe maximal
        assert!(res
            .members
            .iter()
            .all(|m| m.si_safe <= best.si_safe + 1e-12));
        assert!(best.si_global > 0.0);
    }

    #[test]
    fn swarm_is_deterministic() {
        let a = run_swarm_demo(4, 42, 20);
        let b = run_swarm_demo(4, 42, 20);
        assert_eq!(a.best_index, b.best_index);
        for (x, y) in a.members.iter().zip(&b.members) {
            assert_eq!(x.seed, y.seed);
            assert!((x.si_global - y.si_global).abs() < 1e-12);
        }
    }

    #[test]
    fn swarm_with_zero_steps_does_not_panic() {
        // run(0) ⇒ aucun rapport : les membres sont marqués invalides
        // (SI_safe = -∞) au lieu de faire paniquer l'essaim via `unwrap()`.
        let res = run_swarm_demo(3, 7, 0);
        assert_eq!(res.members.len(), 3);
        assert!(res.members.iter().all(|m| m.si_safe == f64::NEG_INFINITY));
    }

    #[test]
    fn test_swarm_mesh_6_lock_free() {
        let mesh = std::sync::Arc::new(SwarmMesh6::new());
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(6));
        let mut handles = Vec::new();

        // On lance 6 threads représentant les 6 agents cognitifs
        for i in 0..6 {
            let mesh_clone = mesh.clone();
            let barrier_clone = barrier.clone();
            let handle = std::thread::spawn(move || {
                // Initialisation du nœud
                mesh_clone.update_node_state(i, 0.5 + i as f64, 0.9, 10, true);

                // Envoi de messages lock-free vers tous les autres 5 nœuds
                for j in 0..6 {
                    if i != j {
                        let msg = (i as u64 * 100) + j as u64;
                        mesh_clone.send_message(i, j, msg).unwrap();
                    }
                }

                // Barrière de synchronisation pour s'assurer que tous les envois sont complétés
                barrier_clone.wait();

                // Lecture de ses propres boîtes aux lettres
                let msgs = mesh_clone.read_mailboxes(i);
                for (idx, &msg) in msgs.iter().enumerate() {
                    let expected_sender = if idx < i { idx } else { idx + 1 };
                    let expected_val = (expected_sender as u64 * 100) + i as u64;
                    assert_eq!(msg, expected_val);
                }
            });
            handles.push(handle);
        }

        for h in handles {
            h.join().unwrap();
        }

        // Vérification des états finaux
        for i in 0..6 {
            let (si_glob, si_saf, steps, active) = mesh.get_node_state(i);
            assert_eq!(si_glob, 0.5 + i as f64);
            assert_eq!(si_saf, 0.9);
            assert_eq!(steps, 10);
            assert!(active);
        }
    }
}

// ===================== SWARM LOCK-FREE & CACHE-LINE ====================== //

/// Un nœud d'agent cognitif aligné sur la ligne de cache (128 octets) pour éviter tout false sharing.
#[repr(align(128))]
pub struct CognitiveAgentNode {
    /// Score global (transmuté f64 -> u64 pour manipulation atomique).
    pub si_global: std::sync::atomic::AtomicU64,
    /// Score de sûreté (transmuté f64 -> u64 pour manipulation atomique).
    pub si_safe: std::sync::atomic::AtomicU64,
    /// Nombre de pas effectués par ce nœud.
    pub steps: std::sync::atomic::AtomicUsize,
    /// Indicateur si le nœud est toujours actif.
    pub active: std::sync::atomic::AtomicBool,
    /// Boîtes aux lettres de réception de messages (une pour chacun des 5 autres agents du maillage de 6).
    /// Chaque boîte stocke la dernière information d'état transmutée de manière lock-free.
    pub mailboxes: [std::sync::atomic::AtomicU64; 5],
}

impl Default for CognitiveAgentNode {
    fn default() -> Self {
        CognitiveAgentNode {
            si_global: std::sync::atomic::AtomicU64::new(0),
            si_safe: std::sync::atomic::AtomicU64::new(0),
            steps: std::sync::atomic::AtomicUsize::new(0),
            active: std::sync::atomic::AtomicBool::new(false),
            mailboxes: [
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
                std::sync::atomic::AtomicU64::new(0),
            ],
        }
    }
}

/// Structure de mémoire partagée pour un maillage de 6 agents cognitifs (topologie de 6 nœuds).
/// Architecture lock-free, zero-allocation, sans aucun Mutex ni RwLock, immunisée contre le false sharing.
pub struct SwarmMesh6 {
    pub nodes: [CognitiveAgentNode; 6],
}

impl Default for SwarmMesh6 {
    fn default() -> Self {
        Self::new()
    }
}

impl SwarmMesh6 {
    /// Initialise un nouveau maillage lock-free de 6 agents.
    pub fn new() -> Self {
        SwarmMesh6 {
            nodes: [
                CognitiveAgentNode::default(),
                CognitiveAgentNode::default(),
                CognitiveAgentNode::default(),
                CognitiveAgentNode::default(),
                CognitiveAgentNode::default(),
                CognitiveAgentNode::default(),
            ],
        }
    }

    /// Met à jour de manière atomique l'état d'un nœud spécifique.
    pub fn update_node_state(
        &self,
        node_id: usize,
        si_global: f64,
        si_safe: f64,
        steps: usize,
        active: bool,
    ) {
        assert!(node_id < 6, "ID de nœud hors limites");
        let node = &self.nodes[node_id];
        node.si_global
            .store(si_global.to_bits(), std::sync::atomic::Ordering::Release);
        node.si_safe
            .store(si_safe.to_bits(), std::sync::atomic::Ordering::Release);
        node.steps
            .store(steps, std::sync::atomic::Ordering::Release);
        node.active
            .store(active, std::sync::atomic::Ordering::Release);
    }

    /// Récupère l'état courant d'un nœud spécifique.
    pub fn get_node_state(&self, node_id: usize) -> (f64, f64, usize, bool) {
        assert!(node_id < 6, "ID de nœud hors limites");
        let node = &self.nodes[node_id];
        let si_global = f64::from_bits(node.si_global.load(std::sync::atomic::Ordering::Acquire));
        let si_safe = f64::from_bits(node.si_safe.load(std::sync::atomic::Ordering::Acquire));
        let steps = node.steps.load(std::sync::atomic::Ordering::Acquire);
        let active = node.active.load(std::sync::atomic::Ordering::Acquire);
        (si_global, si_safe, steps, active)
    }

    /// Envoie un message de manière atomique d'un nœud vers un autre (maillage complet).
    pub fn send_message(&self, from_id: usize, to_id: usize, msg: u64) -> Result<(), &'static str> {
        if from_id >= 6 || to_id >= 6 {
            return Err("Nœud source ou destination hors limites");
        }
        if from_id == to_id {
            return Err("Un nœud ne peut s'envoyer de message à lui-même");
        }

        // Calcule l'index de boîte aux lettres cible pour le nœud récepteur
        let mailbox_idx = if from_id < to_id {
            from_id
        } else {
            from_id - 1
        };

        self.nodes[to_id].mailboxes[mailbox_idx].store(msg, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Lit et consomme les messages dans les boîtes aux lettres d'un nœud.
    pub fn read_mailboxes(&self, node_id: usize) -> [u64; 5] {
        assert!(node_id < 6, "ID de nœud hors limites");
        let node = &self.nodes[node_id];
        let mut msgs = [0u64; 5];
        for (i, msg) in msgs.iter_mut().enumerate() {
            *msg = node.mailboxes[i].load(std::sync::atomic::Ordering::Acquire);
        }
        msgs
    }
}
