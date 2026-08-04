//! Contrat de **déterminisme** (contrat §16).
//!
//! Une exécution déterministe enregistre : graine, ordre des exemples, ordre
//! des réductions, configuration de l'objectif, versions (modèle, référence,
//! encodeur mémoire), dtype, backend, nombre de threads, mode d'exécution,
//! empreintes (données, poids).

/// Empreinte (hash) courte d'un artefact — ici SHA-256 tronqué en hex, via
/// une fonction locale sans dépendance (FNV-1a 64 pour la légèreté ; le
/// contrat exige une *empreinte reproductible*, pas nécessairement crypto).
pub fn fingerprint(bytes: &[u8]) -> String {
    // FNV-1a 64 — déterministe, sans dépendance
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// Mode d'exécution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Oracle scalaire (cogno-core).
    ScalarOracle,
    /// Backend batch/tensoriel (cogno-scirust).
    Batched,
    /// Mode contrôlé (rollouts supervisés).
    Controlled,
}

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionMode::ScalarOracle => write!(f, "scalar-oracle"),
            ExecutionMode::Batched => write!(f, "batched"),
            ExecutionMode::Controlled => write!(f, "controlled"),
        }
    }
}

/// Enregistrement complet de déterminisme (contrat §16).
#[derive(Debug, Clone)]
pub struct DeterminismRecord {
    pub seed: u64,
    /// ordre des exemples (description courte, ex. "batch-order").
    pub example_order: String,
    /// ordre des réductions ("compensated-in-batch-order").
    pub reduction_order: String,
    /// configuration de l'objectif (empreinte).
    pub objective_config_fingerprint: String,
    pub model_version: String,
    pub ref_model_version: String,
    pub memory_encoder_version: String,
    pub dtype: String,
    pub backend: String,
    pub num_threads: usize,
    pub mode: ExecutionMode,
    pub data_fingerprint: String,
    pub weights_fingerprint: String,
}

impl DeterminismRecord {
    /// Empreinte canonique de l'enregistrement — deux exécutions qui ont la
    /// même empreinte ont les mêmes ingrédients de déterminisme.
    pub fn fingerprint(&self) -> String {
        let mut s = String::new();
        s.push_str(&self.seed.to_string());
        s.push('|');
        s.push_str(&self.example_order);
        s.push('|');
        s.push_str(&self.reduction_order);
        s.push('|');
        s.push_str(&self.objective_config_fingerprint);
        s.push('|');
        s.push_str(&self.model_version);
        s.push('|');
        s.push_str(&self.ref_model_version);
        s.push('|');
        s.push_str(&self.memory_encoder_version);
        s.push('|');
        s.push_str(&self.dtype);
        s.push('|');
        s.push_str(&self.backend);
        s.push('|');
        s.push_str(&self.num_threads.to_string());
        s.push('|');
        s.push_str(&self.mode.to_string());
        s.push('|');
        s.push_str(&self.data_fingerprint);
        s.push('|');
        s.push_str(&self.weights_fingerprint);
        fingerprint(s.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_record_is_reproducible() {
        let rec = || DeterminismRecord {
            seed: 42,
            example_order: "batch-order".into(),
            reduction_order: "compensated-in-batch-order".into(),
            objective_config_fingerprint: fingerprint(b"cfg-v1"),
            model_version: "cogno-0.1".into(),
            ref_model_version: "sft-1".into(),
            memory_encoder_version: "enc-1".into(),
            dtype: "f64".into(),
            backend: "oracle".into(),
            num_threads: 1,
            mode: ExecutionMode::ScalarOracle,
            data_fingerprint: fingerprint(b"data"),
            weights_fingerprint: fingerprint(b"weights"),
        };
        assert_eq!(rec().fingerprint(), rec().fingerprint());
        assert_ne!(rec().fingerprint(), {
            let mut r = rec();
            r.seed = 43;
            r.fingerprint()
        });
    }
}
