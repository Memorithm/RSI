//! §2bis — SOURCE DE CONNAISSANCES (ancrage de la composante D)
//!
//! La composante `D` (connaissances) n'est plus une abstraction : le port
//! [`KnowledgeSource`] l'alimente depuis une **vraie source** (documents). À
//! chaque absorption, la source ingère du contenu réel, en extrait des
//! *concepts* distincts, et renvoie un **niveau de connaissance** ∈ [0,1]
//! (saturant) vers lequel l'agent fait tendre `D`.
//!
//! [`CorpusKnowledge`] lit des textes en mémoire ou un répertoire de fichiers.
//! Une source lourde (p. ex. PAPERS) se brancherait via un adaptateur en
//! sous-processus implémentant le même trait, sans alourdir le cœur.

use std::collections::HashSet;

/// Source de connaissances : chaque `absorb` ingère un lot et renvoie le niveau
/// cumulé de connaissance ∈ [0,1] (monotone croissant).
pub trait KnowledgeSource {
    fn absorb(&mut self) -> f64;
    /// Niveau courant sans nouvelle ingestion.
    fn level(&self) -> f64;
}

/// Source de connaissances adossée à un corpus de documents réels.
///
/// Ingère un document par appel ; le niveau sature avec le nombre de **concepts
/// distincts** appris : `level = 1 − exp(−|concepts| / scale)`.
pub struct CorpusKnowledge {
    documents: Vec<String>,
    cursor: usize,
    concepts: HashSet<String>,
    scale: f64,
}

impl CorpusKnowledge {
    /// Construit depuis des textes en mémoire.
    pub fn from_texts(documents: Vec<String>) -> Self {
        CorpusKnowledge { documents, cursor: 0, concepts: HashSet::new(), scale: 64.0 }
    }

    /// Règle l'échelle de saturation (nombre de concepts pour ~63 % du niveau).
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale.max(1.0);
        self
    }

    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    fn compute_level(&self) -> f64 {
        saturating_level(self.concepts.len(), self.scale)
    }
}

/// Extrait les *concepts* (jetons alphabétiques significatifs) d'un texte.
pub(crate) fn extract_concepts(text: &str, into: &mut HashSet<String>) {
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        let tok = raw.to_lowercase();
        if tok.len() >= 4 && tok.chars().any(|c| c.is_alphabetic()) {
            into.insert(tok);
        }
    }
}

/// Niveau saturant ∈ [0,1) : `1 − exp(−n / scale)`.
pub(crate) fn saturating_level(n: usize, scale: f64) -> f64 {
    1.0 - (-(n as f64) / scale.max(1.0)).exp()
}

impl KnowledgeSource for CorpusKnowledge {
    fn absorb(&mut self) -> f64 {
        if self.cursor < self.documents.len() {
            let doc = self.documents[self.cursor].clone();
            extract_concepts(&doc, &mut self.concepts);
            self.cursor += 1;
        }
        self.compute_level()
    }

    fn level(&self) -> f64 {
        self.compute_level()
    }
}

/// Source de connaissances adossée à **PAPERS** via **sous-processus**.
///
/// Pour chaque source (PDF / arXiv / URL / chemin), invoque le binaire `papers`
/// (par défaut `papers extract <source>`, léger et sans LLM), capture sa sortie
/// standard, en extrait les concepts et fait monter le niveau de `D`.
///
/// **Aucune dépendance** : PAPERS n'est PAS lié comme crate (il tire
/// scirust/ORT/CUDA) — on l'appelle en processus externe. **Dégradation
/// propre** : si le binaire est absent ou échoue, on retombe sur le texte de la
/// source elle-même, de sorte que l'ingestion reste fonctionnelle (et testable)
/// sans PAPERS installé.
///
/// Binaire résolu via `--bin`, sinon l'env `RSI_PAPERS_BIN`, sinon `papers`.
pub struct PapersKnowledge {
    bin: String,
    subcommand: String,
    extra_args: Vec<String>,
    sources: Vec<String>,
    cursor: usize,
    concepts: HashSet<String>,
    scale: f64,
    last_degraded: bool,
}

impl PapersKnowledge {
    /// Construit l'adaptateur pour une liste de sources (papiers).
    pub fn new(sources: Vec<String>) -> Self {
        let bin = std::env::var("RSI_PAPERS_BIN").unwrap_or_else(|_| "papers".to_string());
        PapersKnowledge {
            bin,
            subcommand: "extract".to_string(),
            extra_args: Vec::new(),
            sources,
            cursor: 0,
            concepts: HashSet::new(),
            scale: 96.0,
            last_degraded: false,
        }
    }

    /// Chemin explicite du binaire `papers`.
    pub fn with_binary(mut self, path: impl Into<String>) -> Self {
        self.bin = path.into();
        self
    }

    /// Sous-commande PAPERS (défaut `extract` ; p. ex. `analyze`).
    pub fn with_subcommand(mut self, sub: impl Into<String>) -> Self {
        self.subcommand = sub.into();
        self
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale.max(1.0);
        self
    }

    /// `true` si la dernière absorption a dû dégrader (PAPERS indisponible).
    pub fn last_degraded(&self) -> bool {
        self.last_degraded
    }

    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// Exécute `papers <subcommand> <source> <extra…>` ; renvoie sa sortie
    /// standard si le processus réussit avec une sortie non triviale.
    fn run_papers(&self, source: &str) -> Option<String> {
        use std::io::Read;
        use std::process::Stdio;
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        // Garde-fous : un binaire `papers` hostile ou bogué ne doit ni bloquer
        // indéfiniment ni épuiser la RAM en remplissant stdout.
        const TIMEOUT: Duration = Duration::from_secs(30);
        const MAX_OUTPUT: u64 = 8 * 1024 * 1024; // 8 Mio

        let mut cmd = std::process::Command::new(&self.bin);
        if !self.subcommand.is_empty() {
            cmd.arg(&self.subcommand);
        }
        cmd.arg(source);
        for a in &self.extra_args {
            cmd.arg(a);
        }
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::null());

        let mut child = cmd.spawn().ok()?;

        // Lecture *bornée* de stdout dans un thread dédié : évite un blocage si le
        // process remplit le tampon du pipe, et plafonne la mémoire capturée.
        let stdout = child.stdout.take()?;
        let (tx, rx) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout.take(MAX_OUTPUT).read_to_end(&mut buf);
            let _ = tx.send(buf);
        });

        // Attente *bornée* de la fin du process (sondage léger, std-only).
        let deadline = Instant::now() + TIMEOUT;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        break None; // timeout ⇒ dégradation propre
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
            }
        };

        let buf = rx.recv_timeout(Duration::from_secs(1)).unwrap_or_default();
        let _ = reader.join();

        if !status.map(|s| s.success()).unwrap_or(false) {
            return None;
        }
        let text = String::from_utf8_lossy(&buf).into_owned();
        if text.trim().len() >= 8 {
            Some(text)
        } else {
            None
        }
    }
}

impl KnowledgeSource for PapersKnowledge {
    fn absorb(&mut self) -> f64 {
        if self.cursor < self.sources.len() {
            let source = self.sources[self.cursor].clone();
            self.cursor += 1;
            match self.run_papers(&source) {
                Some(text) => {
                    self.last_degraded = false;
                    extract_concepts(&text, &mut self.concepts);
                }
                None => {
                    // dégradation : ingère au moins le descripteur de la source
                    self.last_degraded = true;
                    extract_concepts(&source, &mut self.concepts);
                }
            }
        }
        saturating_level(self.concepts.len(), self.scale)
    }

    fn level(&self) -> f64 {
        saturating_level(self.concepts.len(), self.scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_grows_with_documents() {
        let docs = vec![
            "recursive self improvement geometry surface intelligence".to_string(),
            "substrate hardware software coupling efficiency multiplicative".to_string(),
            "criticality failure modes risk priority number wireheading".to_string(),
        ];
        let mut k = CorpusKnowledge::from_texts(docs).with_scale(16.0);
        let l0 = k.level();
        let l1 = k.absorb();
        let l2 = k.absorb();
        let l3 = k.absorb();
        assert!(l0 == 0.0);
        assert!(l1 > l0 && l2 > l1 && l3 > l2, "{l0} {l1} {l2} {l3}");
        assert!(l3 < 1.0 && l3 > 0.0);
        // au-delà du corpus, le niveau se stabilise
        let l4 = k.absorb();
        assert!((l4 - l3).abs() < 1e-12);
    }

    #[test]
    fn papers_subprocess_path_via_echo() {
        // simule PAPERS avec /bin/echo : la sortie standard contient les concepts
        let mut p = PapersKnowledge::new(vec![
            "alpha beta gamma delta epsilon".to_string(),
            "substrat memoire criticite raisonnement".to_string(),
        ])
        .with_binary("/bin/echo")
        .with_subcommand("paper")
        .with_scale(8.0);
        let l1 = p.absorb();
        assert!(!p.last_degraded(), "echo doit réussir (pas de dégradation)");
        assert!(l1 > 0.0);
        let l2 = p.absorb();
        assert!(l2 > l1);
        assert!(p.concept_count() >= 8);
    }

    #[test]
    fn papers_degrades_gracefully_when_absent() {
        // binaire inexistant → dégradation : on ingère le descripteur de source
        let mut p = PapersKnowledge::new(vec![
            "transformer attention architecture scaling".to_string(),
        ])
        .with_binary("definitely_not_a_real_binary_xyz_42")
        .with_scale(8.0);
        let l = p.absorb();
        assert!(p.last_degraded(), "doit dégrader si le binaire est absent");
        assert!(l > 0.0, "le niveau monte quand même via le texte source");
        assert!(p.concept_count() >= 3);
    }
}
