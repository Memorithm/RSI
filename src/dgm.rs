//! # Boucle d'auto-amélioration empirique (Darwin–Gödel / STOP)
//!
//! Port natif, **std-only et sans dépendance**, du contenu de `soul-rsi`
//! (`github.com/CHECKUPAUTO/soul-rsi`) dans le moteur RSI. Là où le reste de
//! RSI fait **proposer du texte** au LLM pour des domaines abstraits (synthèse
//! d'expressions, configs, prompts, WASM ; cf. [`crate::llm`]), ce module fait
//! de l'auto-amélioration **empirique de code source réel** :
//!
//! 1. on **propose** une édition (`find → replace` exacte sur un fichier) ;
//! 2. on l'**évalue** dans une **copie isolée** du dépôt (`cargo build`+`test`) ;
//! 3. on ne la **garde que si elle prouve une amélioration** (la compilation
//!    domine, puis l'absence de régression de tests, puis le score) ;
//! 4. on **archive** la variante survivante comme tremplin réutilisable ;
//! 5. on **recommence**, en branchant depuis n'importe quelle variante gardée.
//!
//! ## Correspondance avec la littérature
//!
//! - **STOP — Self-Taught Optimizer** (Zelikman et al., 2023) : le *proposeur*
//!   est une fonction interchangeable ([`Proposer`]). [`LlmProposer`] en est la
//!   version LLM ; la boucle ne fait **jamais** confiance à l'auto-évaluation
//!   du modèle.
//! - **Darwin Gödel Machine** (Zhang et al., 2025) : chaque changement validé
//!   est gardé comme tremplin dans une [`Archive`] ouverte ; l'acceptation est
//!   décidée **empiriquement** (build + tests), jamais sur la prétention du
//!   modèle.
//! - **Gödel Machine** (Schmidhuber, 2007) : une amélioration n'est appliquée
//!   qu'après une vérification de bénéfice — ici la barrière build+test de
//!   [`Fitness`], où un changement qui casse la compilation ne peut jamais
//!   dominer un qui compile.
//! - **Reflexion** (Shinn et al., 2023) : les justifications rejetées sont
//!   re-injectées au proposeur (renforcement verbal bon marché).
//! - **Open-endedness** (Lehman & Stanley, 2015) : la sélection de parent mêle
//!   qualité et bonus de nouveauté pour les lignées sous-explorées.
//!
//! ## Sûreté
//!
//! L'évaluation tourne toujours sur un [`WorkspaceSnapshot`] jetable ; l'arbre
//! **vivant** n'est touché que par l'appel explicite [`promote_to_live`], gardé
//! par une évaluation tout-au-vert. Le proposeur LLM est borné par une
//! **liste blanche** de fichiers éditables. Les sous-processus `cargo` sont
//! **bornés** (timeout + sortie plafonnée), dans l'esprit de [`crate::knowledge`].
//!
//! ## Améliorations vs `soul-rsi`
//!
//! - **IDs déterministes** : l'identité d'une variante est un hash SHA-256 de sa
//!   lignée et de son patch (et non un UUID aléatoire) ⇒ l'archive est
//!   **bit-exacte reproductible** à graine fixe, cohérent avec le reste de RSI.
//! - **Horloge logique** : un index de création `seq` remplace l'horodatage mur
//!   (non déterministe).
//! - **Sous-processus bornés** : [`CargoEvaluator`] impose timeout + plafond de
//!   sortie (l'original lançait `cargo` sans borne).
//!
//! ## Exemple (boucle déterministe, sans LLM ni `cargo`)
//!
//! ```
//! use rsi::dgm::{Archive, DgmConfig, DgmEngine, ClosureEvaluator, Fitness,
//!                ImprovementContext, Patch, Proposal, Proposer};
//! use rsi::rng::Rng;
//! # use rsi::dgm::Result;
//! use std::path::Path;
//!
//! // Proposeur jouet : incrémente `level = N` dans un fichier texte.
//! struct Inc;
//! impl Proposer for Inc {
//!     fn propose(&self, ctx: &ImprovementContext<'_>, _rng: &mut Rng)
//!         -> Result<Option<Proposal>> {
//!         let txt = std::fs::read_to_string(ctx.resolve("level.txt")).unwrap_or_default();
//!         let cur: i64 = txt.trim().strip_prefix("level = ").and_then(|s| s.parse().ok()).unwrap_or(0);
//!         Ok(Some(Proposal {
//!             patch: Patch::new("level.txt", format!("level = {cur}"), format!("level = {}", cur + 1)),
//!             rationale: format!("raise to {}", cur + 1),
//!         }))
//!     }
//! }
//! # fn demo() -> Result<()> {
//! let dir = std::env::temp_dir().join("rsi-dgm-doctest");
//! let _ = std::fs::remove_dir_all(&dir);
//! std::fs::create_dir_all(&dir).unwrap();
//! std::fs::write(dir.join("level.txt"), "level = 0").unwrap();
//!
//! let eval = ClosureEvaluator::new(|root: &Path| {
//!     let txt = std::fs::read_to_string(root.join("level.txt")).unwrap_or_default();
//!     let n: f64 = txt.trim().strip_prefix("level = ").and_then(|s| s.parse().ok()).unwrap_or(0.0);
//!     Fitness { compiles: true, tests_passed: 1, tests_failed: 0, score: n, notes: String::new() }
//! });
//! let baseline = Fitness { compiles: true, tests_passed: 1, tests_failed: 0, score: 0.0, notes: String::new() };
//! let mut eng = DgmEngine::new(Archive::with_root(baseline), Inc, eval,
//!     DgmConfig::new(&dir, "raise the level"), 42);
//! eng.run(5)?;
//! assert!(eng.best().unwrap().fitness.as_ref().unwrap().score >= 1.0);
//! let _ = std::fs::remove_dir_all(&dir);
//! # Ok(())
//! # }
//! # demo().unwrap();
//! ```

use crate::json::Json;
use crate::rng::Rng;
use crate::sha256::sha256_hex;
use crate::trajectory::Trajectory;
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

// ════════════════════════════════ Erreurs ════════════════════════════════ //

/// Erreur de la boucle d'auto-amélioration.
#[derive(Debug)]
pub enum DgmError {
    Io(String),
    /// La cible du patch est hors des chemins autorisés.
    PathNotAllowed(String),
    /// Le patch n'a pas pu être appliqué (motif absent / non unique / no-op).
    Apply(String),
    /// L'évaluation a échoué (impossible de lancer le build/test).
    Evaluation(String),
    /// Le proposeur a échoué.
    Proposer(String),
}

impl fmt::Display for DgmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DgmError::Io(e) => write!(f, "io error: {e}"),
            DgmError::PathNotAllowed(p) => write!(f, "patch target {p} is outside the allowed paths"),
            DgmError::Apply(e) => write!(f, "could not apply patch: {e}"),
            DgmError::Evaluation(e) => write!(f, "evaluation failed: {e}"),
            DgmError::Proposer(e) => write!(f, "proposer failed: {e}"),
        }
    }
}

impl std::error::Error for DgmError {}

impl From<std::io::Error> for DgmError {
    fn from(e: std::io::Error) -> Self {
        DgmError::Io(e.to_string())
    }
}

/// Résultat de la boucle.
pub type Result<T> = std::result::Result<T, DgmError>;

// ════════════════════════════════ Patch ══════════════════════════════════ //

/// Édition concrète et réversible d'un fichier source, exprimée comme une
/// substitution exacte `find → replace`. Le motif `find` doit apparaître
/// **exactement une fois** dans le fichier (sinon le patch est rejeté).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Patch {
    /// Chemin du fichier, relatif à la racine du workspace.
    pub target: String,
    /// Texte exact à remplacer (doit apparaître une seule fois).
    pub find: String,
    /// Texte de remplacement.
    pub replace: String,
}

impl Patch {
    pub fn new(target: impl Into<String>, find: impl Into<String>, replace: impl Into<String>) -> Self {
        Self { target: target.into(), find: find.into(), replace: replace.into() }
    }

    /// Un patch est un no-op s'il ne change rien.
    pub fn is_noop(&self) -> bool {
        self.find == self.replace
    }

    fn to_json(&self) -> Json {
        let mut o = Json::obj();
        o.set("target", Json::Str(self.target.clone()))
            .set("find", Json::Str(self.find.clone()))
            .set("replace", Json::Str(self.replace.clone()));
        o
    }

    fn from_json(j: &Json) -> Option<Patch> {
        Some(Patch::new(
            j.get("target")?.as_str()?,
            j.get("find")?.as_str()?,
            j.get("replace")?.as_str()?,
        ))
    }
}

// ════════════════════════════════ Fitness ════════════════════════════════ //

/// Qualité **mesurée empiriquement** d'une variante.
///
/// L'ordre est lexicographique et reflète ce qu'un ingénieur prudent
/// accepterait : un changement doit (1) compiler, (2) ne pas régresser les
/// tests, et seulement alors (3) son `score` scalaire est comparé. Encoder la
/// barrière ainsi est ce qui rend la boucle **sûre** : contrairement à un
/// objectif scalaire pur, une variante qui casse le build ne peut jamais
/// dominer une qui compile.
#[derive(Debug, Clone, PartialEq)]
pub struct Fitness {
    /// Le workspace de la variante a-t-il compilé ?
    pub compiles: bool,
    /// Nombre de tests passés.
    pub tests_passed: u32,
    /// Nombre de tests échoués.
    pub tests_failed: u32,
    /// Récompense scalaire spécifique au domaine (plus = mieux). Comparée
    /// seulement quand les barrières compile/tests sont à égalité.
    pub score: f64,
    /// Notes libres (erreurs du compilateur, remarques de l'évaluateur).
    pub notes: String,
}

impl Fitness {
    /// Fitness d'une variante qui n'a pas compilé — le plancher absolu.
    pub fn broken(notes: impl Into<String>) -> Self {
        Self {
            compiles: false,
            tests_passed: 0,
            tests_failed: 0,
            score: f64::NEG_INFINITY,
            notes: notes.into(),
        }
    }

    /// Vrai si tout test exécuté a passé (et qu'au moins le build a réussi).
    pub fn all_green(&self) -> bool {
        self.compiles && self.tests_failed == 0
    }

    /// Clé de comparaison lexicographique : (compiles, -tests_failed, tests_passed, score).
    fn key(&self) -> (bool, i64, i64, f64) {
        (
            self.compiles,
            -(self.tests_failed as i64),
            self.tests_passed as i64,
            self.score,
        )
    }

    /// Strictement meilleur que `other` sous l'ordre de barrière lexicographique.
    ///
    /// Les scores NaN sont traités comme le plancher : une mesure malformée ne
    /// peut jamais être jugée comme une amélioration.
    pub fn is_better_than(&self, other: &Fitness) -> bool {
        use std::cmp::Ordering;
        let (ac, af, ap, asc) = self.key();
        let (bc, bf, bp, bsc) = other.key();
        match (ac.cmp(&bc), af.cmp(&bf), ap.cmp(&bp)) {
            (Ordering::Greater, _, _) => true,
            (Ordering::Less, _, _) => false,
            (_, Ordering::Greater, _) => true,
            (_, Ordering::Less, _) => false,
            (_, _, Ordering::Greater) => true,
            (_, _, Ordering::Less) => false,
            _ => asc.partial_cmp(&bsc) == Some(Ordering::Greater),
        }
    }

    fn to_json(&self) -> Json {
        let mut o = Json::obj();
        o.set("compiles", Json::Bool(self.compiles))
            .set("tests_passed", Json::Num(self.tests_passed as f64))
            .set("tests_failed", Json::Num(self.tests_failed as f64))
            // Les scores non finis (p. ex. `broken` = -∞) ne sont pas du JSON
            // valide : on les sérialise en `null` et on les reconstruit à la
            // lecture (− ∞ si !compiles, sinon 0.0).
            .set(
                "score",
                if self.score.is_finite() { Json::Num(self.score) } else { Json::Null },
            )
            .set("notes", Json::Str(self.notes.clone()));
        o
    }

    fn from_json(j: &Json) -> Option<Fitness> {
        let compiles = j.get("compiles")?.as_bool()?;
        let score = match j.get("score") {
            Some(Json::Num(n)) => *n,
            _ if !compiles => f64::NEG_INFINITY,
            _ => 0.0,
        };
        Some(Fitness {
            compiles,
            tests_passed: j.get("tests_passed").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            tests_failed: j.get("tests_failed").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            score,
            notes: j.get("notes").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        })
    }
}

// ════════════════════════════════ Statut ═════════════════════════════════ //

/// Statut d'acceptation d'une variante après évaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Proposée mais pas encore évaluée.
    Pending,
    /// Évaluée et gardée dans l'archive (elle améliore son parent).
    Accepted,
    /// Évaluée et écartée (régression, casse de build, ou no-op).
    Rejected,
}

impl Status {
    fn as_str(&self) -> &'static str {
        match self {
            Status::Pending => "pending",
            Status::Accepted => "accepted",
            Status::Rejected => "rejected",
        }
    }
    fn from_str(s: &str) -> Status {
        match s {
            "accepted" => Status::Accepted,
            "rejected" => Status::Rejected,
            _ => Status::Pending,
        }
    }
}

// ════════════════════════════════ Variant ════════════════════════════════ //

/// Une candidate auto-modification, avec sa lignée et sa fitness mesurée.
/// C'est l'objet « tremplin » de la Darwin Gödel Machine.
#[derive(Debug, Clone)]
pub struct Variant {
    /// Identité **déterministe** (hash de lignée + patch + seq).
    pub id: String,
    /// Id du parent dont la variante dérive (`None` pour la racine).
    pub parent: Option<String>,
    /// 0 pour la graine, incrémenté le long de chaque lignée.
    pub generation: u64,
    /// Index de création logique (horloge déterministe, remplace l'horodatage).
    pub seq: u64,
    pub patch: Patch,
    /// Pourquoi le proposeur croit que ce changement aide.
    pub rationale: String,
    pub status: Status,
    pub fitness: Option<Fitness>,
}

impl Variant {
    /// Racine immuable de l'archive : le code non modifié, par définition la
    /// référence que toute proposition doit battre.
    pub fn root(baseline: Fitness) -> Self {
        let patch = Patch::new("", "", "");
        Self {
            id: variant_id(None, 0, 0, &patch),
            parent: None,
            generation: 0,
            seq: 0,
            patch,
            rationale: "baseline (unmodified codebase)".to_string(),
            status: Status::Accepted,
            fitness: Some(baseline),
        }
    }

    /// Un enfant frais, pas encore évalué, de `parent`. `seq` est l'index de
    /// création attribué par le moteur (horloge logique déterministe).
    pub fn child(parent: &Variant, patch: Patch, rationale: impl Into<String>, seq: u64) -> Self {
        let generation = parent.generation + 1;
        Self {
            id: variant_id(Some(&parent.id), generation, seq, &patch),
            parent: Some(parent.id.clone()),
            generation,
            seq,
            patch,
            rationale: rationale.into(),
            status: Status::Pending,
            fitness: None,
        }
    }

    fn to_json(&self) -> Json {
        let mut o = Json::obj();
        o.set("id", Json::Str(self.id.clone()))
            .set(
                "parent",
                self.parent.clone().map(Json::Str).unwrap_or(Json::Null),
            )
            .set("generation", Json::Num(self.generation as f64))
            .set("seq", Json::Num(self.seq as f64))
            .set("patch", self.patch.to_json())
            .set("rationale", Json::Str(self.rationale.clone()))
            .set("status", Json::Str(self.status.as_str().to_string()))
            .set(
                "fitness",
                self.fitness.as_ref().map(|f| f.to_json()).unwrap_or(Json::Null),
            );
        o
    }

    fn from_json(j: &Json) -> Option<Variant> {
        Some(Variant {
            id: j.get("id")?.as_str()?.to_string(),
            parent: j.get("parent").and_then(|v| v.as_str()).map(|s| s.to_string()),
            generation: j.get("generation").and_then(|v| v.as_u64()).unwrap_or(0),
            seq: j.get("seq").and_then(|v| v.as_u64()).unwrap_or(0),
            patch: Patch::from_json(j.get("patch")?)?,
            rationale: j.get("rationale").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            status: Status::from_str(j.get("status").and_then(|v| v.as_str()).unwrap_or("pending")),
            fitness: j.get("fitness").and_then(Fitness::from_json),
        })
    }
}

/// Identité déterministe d'une variante : 16 hex de SHA-256 sur sa lignée et
/// son patch. Reproductible à graine fixe (≠ UUID aléatoire de `soul-rsi`).
fn variant_id(parent: Option<&str>, generation: u64, seq: u64, patch: &Patch) -> String {
    let material = format!(
        "{}|{}|{}|{}|{}|{}",
        parent.unwrap_or(""),
        generation,
        seq,
        patch.target,
        patch.find,
        patch.replace
    );
    sha256_hex(&material)[..16].to_string()
}

// ════════════════════════════════ Archive ════════════════════════════════ //

/// Collection ouverte (append-mostly) des variantes acceptées.
///
/// Contrairement à un grimpeur de colline qui ne garde que la meilleure
/// variante, la Darwin Gödel Machine garde **chaque** tremplin validé et peut
/// brancher depuis n'importe lequel — l'open-endedness de Lehman & Stanley.
#[derive(Debug, Clone, Default)]
pub struct Archive {
    variants: Vec<Variant>,
}

impl Archive {
    pub fn new() -> Self {
        Self::default()
    }

    /// Amorce l'archive avec la référence (le code non modifié).
    pub fn with_root(baseline: Fitness) -> Self {
        let mut a = Self::new();
        a.variants.push(Variant::root(baseline));
        a
    }

    pub fn len(&self) -> usize {
        self.variants.len()
    }

    pub fn is_empty(&self) -> bool {
        self.variants.is_empty()
    }

    pub fn variants(&self) -> &[Variant] {
        &self.variants
    }

    /// Insère une variante acceptée.
    pub fn insert(&mut self, variant: Variant) {
        self.variants.push(variant);
    }

    /// Recherche une variante par id.
    pub fn get(&self, id: &str) -> Option<&Variant> {
        self.variants.iter().find(|v| v.id == id)
    }

    /// La meilleure variante par fitness (la candidate à promouvoir).
    pub fn best(&self) -> Option<&Variant> {
        use std::cmp::Ordering;
        self.variants.iter().max_by(|a, b| match (&a.fitness, &b.fitness) {
            (Some(fa), Some(fb)) => {
                if fa.is_better_than(fb) {
                    Ordering::Greater
                } else if fb.is_better_than(fa) {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            }
            (Some(_), None) => Ordering::Greater,
            (None, Some(_)) => Ordering::Less,
            (None, None) => Ordering::Equal,
        })
    }

    /// Sélectionne un parent dont brancher.
    ///
    /// La sélection est ouverte : chaque variante a une chance non nulle, mais
    /// les meilleures variantes et les lignées **sous-explorées** (peu d'enfants
    /// jusqu'ici) sont favorisées — une pondération qualité-diversité légère.
    /// Déterministe pour un état de RNG donné.
    pub fn select_parent(&self, rng: &mut Rng) -> Option<&Variant> {
        if self.variants.is_empty() {
            return None;
        }
        let weights: Vec<f64> = self
            .variants
            .iter()
            .map(|v| {
                let children = self
                    .variants
                    .iter()
                    .filter(|c| c.parent.as_deref() == Some(v.id.as_str()))
                    .count() as f64;
                let quality = v
                    .fitness
                    .as_ref()
                    .map(|f| if f.compiles { 1.0 } else { 0.1 })
                    .unwrap_or(0.1);
                let novelty = 1.0 / (1.0 + children);
                quality * novelty + f64::EPSILON
            })
            .collect();

        let total: f64 = weights.iter().sum();
        let mut pick = rng.uniform() * total;
        for (v, w) in self.variants.iter().zip(weights.iter()) {
            pick -= w;
            if pick <= 0.0 {
                return Some(v);
            }
        }
        self.variants.last()
    }

    /// Sérialise l'archive en JSON (std-only, via [`crate::json`]).
    pub fn to_json(&self) -> String {
        let arr = Json::Arr(self.variants.iter().map(|v| v.to_json()).collect());
        let mut o = Json::obj();
        o.set("variants", arr);
        o.to_string()
    }

    /// Reconstruit une archive depuis le JSON produit par [`Archive::to_json`].
    pub fn from_json(s: &str) -> Result<Self> {
        let j = Json::parse(s).map_err(DgmError::Apply)?;
        let arr = j
            .get("variants")
            .and_then(|v| v.as_array())
            .ok_or_else(|| DgmError::Apply("missing 'variants' array".to_string()))?;
        let variants = arr
            .iter()
            .filter_map(Variant::from_json)
            .collect::<Vec<_>>();
        Ok(Archive { variants })
    }
}

// ════════════════════════════════ Rôles ══════════════════════════════════ //

/// Modèle de complétion de texte minimal. Un adaptateur LLM l'implémente, mais
/// aussi un stub déterministe pour les tests. Le cœur ne dépend d'aucun crate
/// LLM concret — l'« improver » est « juste une fonction » (STOP).
pub trait CodeModel {
    fn complete(&self, prompt: &str) -> Result<String>;
}

/// Permet de choisir un backend à l'exécution (`Box<dyn CodeModel>`).
impl CodeModel for Box<dyn CodeModel> {
    fn complete(&self, prompt: &str) -> Result<String> {
        (**self).complete(prompt)
    }
}

/// Tout ce dont un proposeur a besoin pour raisonner sur le prochain changement.
pub struct ImprovementContext<'a> {
    /// Racine du workspace (vivant, lecture seule) que le proposeur peut inspecter.
    pub workspace_root: &'a Path,
    /// Objectif de haut niveau que l'amélioration doit servir.
    pub goal: &'a str,
    /// Fitness du parent dont le changement va brancher.
    pub parent_fitness: Option<&'a Fitness>,
    /// Justifications des tentatives récemment rejetées, pour éviter de répéter
    /// les impasses (« renforcement verbal » bon marché, cf. Reflexion).
    pub recent_rejections: &'a [String],
    /// Verdicts d'un simulateur sur les tentatives de CE step (axe 3).
    pub simulated_feedback: &'a [String],
}

impl ImprovementContext<'_> {
    /// Lit un fichier source relatif à la racine du workspace.
    pub fn read(&self, rel: &str) -> std::io::Result<String> {
        std::fs::read_to_string(self.resolve(rel))
    }

    pub fn resolve(&self, rel: &str) -> PathBuf {
        self.workspace_root.join(rel)
    }
}

/// Une auto-modification proposée avec son raisonnement.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub patch: Patch,
    pub rationale: String,
}

/// Génère la prochaine édition candidate. Rendre `None` signifie « pas de
/// changement utile cette étape » — un résultat légitime et courant.
pub trait Proposer {
    fn propose(&self, ctx: &ImprovementContext<'_>, rng: &mut Rng) -> Result<Option<Proposal>>;
}

/// **Pré-crible** optionnel : un *world model* prédit le verdict d'un patch
/// avant le coûteux `cargo build`+`test`. Rendu :
/// - `Some(false)` : prédit **cassé** (compile ou tests) — l'engine PEUT sauter
///   le build réel (économie) ;
/// - `Some(true)` : prédit bon ; `None` : indécis.
///
/// **Sûreté (cf. `docs/AGENTWORLD_STUDY.md`)** : l'engine ne saute le gate réel
/// **que** sur `Some(false)` — un `None` ou un `Some(true)` passe toujours au
/// vrai `cargo`. Le simulateur n'a donc que le pouvoir d'*économiser un build
/// quand il est certain que ça casse*, jamais celui d'écarter une amélioration
/// (calibration : zéro faux négatif sur les verdicts conclus). Std-only : le
/// transport LLM vit chez l'appelant (binaire, feature réseau).
/// Verdict prédit par un world model, avec raison optionnelle.
#[derive(Debug, Clone, Default)]
pub struct Prediction {
    /// `None` = indécis ; `Some(true)` = prédit bon ; `Some(false)` = prédit cassé.
    pub pass: Option<bool>,
    /// Cause d'une casse prédite, réinjectée au proposeur (axe 3).
    pub reason: Option<String>,
}

pub trait VerdictPredictor {
    /// Prédit si le patch (diff `find`→`replace` sur `target`, dont le contenu
    /// actuel est `file_content`) laissera le workspace tout-au-vert.
    fn predict(&self, target: &str, file_content: &str, find: &str, replace: &str) -> Prediction;
}

/// Mesure empiriquement un workspace candidat. Le contrat : ne **jamais** faire
/// confiance à l'auto-évaluation d'une proposition — la construire et lancer les
/// tests. C'est la barrière de validation non négociable de la DGM.
pub trait Evaluator {
    /// Évalue le workspace enraciné à `workspace` (une copie isolée avec le
    /// patch candidat déjà appliqué). Ne doit jamais toucher l'arbre vivant.
    fn evaluate(&self, workspace: &Path) -> Result<Fitness>;
}

// ═════════════════════════════ Snapshot isolé ════════════════════════════ //

/// Répertoires jamais utiles à copier dans un snapshot (gros et régénérés).
const SKIP_DIRS: &[&str] = &["target", ".git", "node_modules"];

static SNAP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Copie jetable d'une racine de workspace, avec (optionnellement) un patch
/// appliqué. Le répertoire temporaire sous-jacent est supprimé au `Drop`.
pub struct WorkspaceSnapshot {
    root: PathBuf,
    /// Répertoire temporaire parent à nettoyer.
    tmp: PathBuf,
}

impl WorkspaceSnapshot {
    /// Copie récursivement `src_root` dans un répertoire temporaire frais.
    pub fn create(src_root: &Path) -> Result<Self> {
        let tmp = unique_tmp_dir();
        let root = tmp.join("workspace");
        std::fs::create_dir_all(&root)?;
        copy_tree(src_root, &root)?;
        Ok(Self { root, tmp })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn resolve(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }

    /// Applique un patch au snapshot (édition sauvegardée dans `.rsi_backups`,
    /// exactement comme une édition vivante le serait).
    pub fn apply(&self, patch: &Patch) -> Result<()> {
        if patch.is_noop() {
            return Err(DgmError::Apply("patch is a no-op".to_string()));
        }
        let target = self.resolve(&patch.target);
        // Empêche l'évasion hors de la racine (`..`, chemins absolus…).
        let canon_root = self.root.canonicalize().unwrap_or_else(|_| self.root.clone());
        let canon_target = target
            .canonicalize()
            .unwrap_or_else(|_| target.clone());
        if !canon_target.starts_with(&canon_root) {
            return Err(DgmError::PathNotAllowed(patch.target.clone()));
        }
        let backups = self.root.join(".rsi_backups");
        patch_file_with_backup(&target, &patch.find, &patch.replace, &backups)?;
        Ok(())
    }
}

impl Drop for WorkspaceSnapshot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.tmp);
    }
}

/// Applique un patch accepté à l'arbre **vivant**, avec sauvegarde, et rend l'id
/// de sauvegarde. C'est la **seule** fonction qui mute le vrai code source ; les
/// appelants la gardent par une évaluation passante et tout-au-vert.
pub fn promote_to_live(live_root: &Path, patch: &Patch, backup_dir: &Path) -> Result<String> {
    if patch.is_noop() {
        return Err(DgmError::Apply("patch is a no-op".to_string()));
    }
    let target = live_root.join(&patch.target);
    patch_file_with_backup(&target, &patch.find, &patch.replace, backup_dir)
}

/// Substitution exacte `find → replace` avec sauvegarde de l'original.
///
/// `find` doit apparaître **exactement une fois** (motif absent ou ambigu ⇒
/// rejet : un patch ambigu ne doit jamais éditer silencieusement la mauvaise
/// occurrence). Rend l'id de sauvegarde (16 hex SHA-256 du contenu original).
fn patch_file_with_backup(target: &Path, find: &str, replace: &str, backup_dir: &Path) -> Result<String> {
    let content = std::fs::read_to_string(target)
        .map_err(|e| DgmError::Apply(format!("read {}: {e}", target.display())))?;
    let (start, end) = locate_match(&content, find)
        .map_err(|msg| DgmError::Apply(format!("{msg} in {}", target.display())))?;
    std::fs::create_dir_all(backup_dir)?;
    let id = sha256_hex(&format!("{}|{}", target.display(), content))[..16].to_string();
    std::fs::write(backup_dir.join(format!("{id}.bak")), &content)?;
    let mut patched = String::with_capacity(content.len() - (end - start) + replace.len());
    patched.push_str(&content[..start]);
    patched.push_str(replace);
    patched.push_str(&content[end..]);
    std::fs::write(target, patched)
        .map_err(|e| DgmError::Apply(format!("write {}: {e}", target.display())))?;
    Ok(id)
}

/// Localise le span d'octets `[start, end)` de `content` que `find` doit
/// remplacer. Deux passes, de la plus stricte à la plus tolérante :
///
///   1. **exacte** : `find` doit apparaître *exactement une fois* (0 ⇒ on tente
///      le repli ; >1 ⇒ ambigu, rejet immédiat) ;
///   2. **ligne-à-ligne insensible aux espaces** : le modèle local recopie
///      souvent le bloc avec une indentation légèrement différente ⇒ chaque
///      ligne est comparée après `trim`, les lignes vides de tête/queue sont
///      ignorées, et l'appariement doit rester **unique** (sinon rejet). Le
///      span rendu couvre des lignes entières (sans le `\n` final), si bien que
///      les sauts de ligne autour du bloc sont préservés.
///
/// La barrière compile+tests+bench et la contrainte d'unicité garantissent que
/// ce repli ne peut ni corrompre l'arbre vivant ni éditer la mauvaise
/// occurrence : au pire, un candidat mal aligné ne compile pas → rejeté.
fn locate_match(content: &str, find: &str) -> std::result::Result<(usize, usize), String> {
    match content.matches(find).count() {
        1 => {
            let start = content.find(find).unwrap();
            return Ok((start, start + find.len()));
        }
        0 => {} // absent tel quel → tente le repli tolérant aux espaces
        n => return Err(format!("pattern is not unique ({n} occurrences)")),
    }

    let file_lines = line_spans(content);
    let needle: Vec<&str> = find.lines().map(str::trim).collect();
    let first = needle.iter().position(|l| !l.is_empty());
    let last = needle.iter().rposition(|l| !l.is_empty());
    let needle = match (first, last) {
        (Some(a), Some(b)) => &needle[a..=b],
        _ => return Err("pattern not found".to_string()),
    };
    let w = needle.len();
    if w == 0 || file_lines.len() < w {
        return Err("pattern not found".to_string());
    }

    let mut hits: Vec<usize> = Vec::new();
    for start in 0..=(file_lines.len() - w) {
        let matches = (0..w).all(|k| {
            let (s, e) = file_lines[start + k];
            content[s..e].trim() == needle[k]
        });
        if matches {
            hits.push(start);
        }
    }
    match hits.as_slice() {
        [start] => {
            let (s, _) = file_lines[*start];
            let (_, e) = file_lines[*start + w - 1];
            Ok((s, e))
        }
        [] => Err("pattern not found".to_string()),
        many => Err(format!("pattern is not unique ({} fuzzy occurrences)", many.len())),
    }
}

/// Spans d'octets `[début, fin)` de chaque ligne de `s`, **sans** le `\n` final.
fn line_spans(s: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        if ch == '\n' {
            spans.push((start, i));
            start = i + 1;
        }
    }
    if start < s.len() || spans.is_empty() {
        spans.push((start, s.len()));
    }
    spans
}

fn unique_tmp_dir() -> PathBuf {
    let n = SNAP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("rsi-dgm-{pid}-{n}-{nanos}"))
}

fn copy_tree(src: &Path, dst: &Path) -> Result<()> {
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let ty = entry.file_type()?;
        let to = dst.join(&name);
        if ty.is_dir() {
            if SKIP_DIRS.contains(&name_str.as_ref()) {
                continue;
            }
            std::fs::create_dir_all(&to)?;
            copy_tree(&entry.path(), &to)?;
        } else if ty.is_file() {
            std::fs::copy(entry.path(), &to)?;
        }
        // liens symboliques et autres types de nœuds intentionnellement ignorés.
    }
    Ok(())
}

// ═══════════════════════════════ Évaluateurs ═════════════════════════════ //

/// Construit et teste un workspace candidat avec `cargo`. **Sous-processus
/// borné** (timeout + sortie plafonnée), conformément à la doctrine de sûreté
/// de RSI (cf. [`crate::knowledge`]).
pub struct CargoEvaluator {
    /// Répertoire de manifeste à construire, relatif à la racine du snapshot.
    pub package_subdir: PathBuf,
    /// Args additionnels passés à `cargo test` (p. ex. `-p some_crate`).
    pub test_args: Vec<String>,
    /// Récompense scalaire = taux de réussite des tests (sinon 0.0).
    pub score_from_passrate: bool,
    /// **Option B — score par BENCHMARK** : si non vide, ce sont les arguments
    /// `cargo` d'une commande exécutée *après* le passage des barrières
    /// compile+tests, dont la sortie doit contenir `RSI_BENCH_SCORE=<f64>`
    /// (plus grand = mieux). Le score de fitness devient alors cette **perf
    /// réelle mesurée** — « optimise X » a enfin un gradient. Ex. :
    /// `["run", "--release", "--example", "bench_dot"]`. Vide ⇒ pass-rate.
    pub bench_command: Vec<String>,
    /// Délai max par invocation `cargo` (anti-blocage).
    pub timeout: Duration,
    /// Plafond d'octets capturés par flux (anti-OOM).
    pub max_output: u64,
}

impl Default for CargoEvaluator {
    fn default() -> Self {
        Self {
            package_subdir: PathBuf::new(),
            test_args: Vec::new(),
            score_from_passrate: true,
            bench_command: Vec::new(),
            timeout: Duration::from_secs(300),
            max_output: 4 * 1024 * 1024,
        }
    }
}

impl Evaluator for CargoEvaluator {
    fn evaluate(&self, workspace: &Path) -> Result<Fitness> {
        let dir = workspace.join(&self.package_subdir);

        // 1. Barrière de compilation.
        let mut build = Command::new("cargo");
        build.arg("build").arg("--quiet").current_dir(&dir);
        let (build_ok, build_out) = run_bounded(build, self.timeout, self.max_output)
            .map_err(|e| DgmError::Evaluation(format!("cargo build: {e}")))?;
        if !build_ok {
            return Ok(Fitness::broken(format!("build failed:\n{}", tail(&build_out, 1500))));
        }

        // 2. Barrière de tests.
        let mut test = Command::new("cargo");
        test.arg("test").arg("--quiet").current_dir(&dir);
        for a in &self.test_args {
            test.arg(a);
        }
        let (test_ok, out) = run_bounded(test, self.timeout, self.max_output)
            .map_err(|e| DgmError::Evaluation(format!("cargo test: {e}")))?;
        let (passed, failed) = parse_test_counts(&out);
        let passrate = {
            let total = passed + failed;
            if total == 0 { 0.0 } else { passed as f64 / total as f64 }
        };

        // 3. Score. Option B : si un benchmark est configuré ET que les tests
        // sont tout-au-vert (inutile de mesurer la perf d'un code cassé, et la
        // barrière compile/tests domine de toute façon l'ordre de fitness), on
        // exécute le bench et on prend `RSI_BENCH_SCORE` comme score (perf
        // réelle, plus grand = mieux). Sinon, pass-rate (comportement d'origine).
        let mut notes = if test_ok {
            "all tests passed".to_string()
        } else {
            format!("tests failed:\n{}", tail(&out, 1500))
        };
        let all_green = failed == 0;
        let score = if !self.bench_command.is_empty() && all_green {
            // Le bench tourne PLUSIEURS fois, on garde le MAX : le bruit
            // machine est unilatéral (il ne fait que ralentir), donc le max
            // approche la vraie capacité. Surtout, la RÉFÉRENCE reçoit le même
            // traitement — une référence mesurée pendant une interférence
            // soutenue fabriquait de faux ACCEPTÉ (+6.5 % « gagnés » sur json
            // en campagne parce que la référence seule était sortie lente).
            const BENCH_TRIES: usize = 3;
            let mut best: Option<f64> = None;
            for _ in 0..BENCH_TRIES {
                let mut bench = Command::new("cargo");
                bench.current_dir(&dir);
                for a in &self.bench_command {
                    bench.arg(a);
                }
                let (bench_ok, bench_out) = run_bounded(bench, self.timeout, self.max_output)
                    .map_err(|e| DgmError::Evaluation(format!("cargo bench cmd: {e}")))?;
                if let Some(s) = parse_bench_score(&bench_out).filter(|_| bench_ok) {
                    best = Some(best.map_or(s, |b: f64| b.max(s)));
                }
            }
            match best {
                Some(s) => {
                    notes = format!("all green; RSI_BENCH_SCORE={s} (max de {BENCH_TRIES} runs)");
                    s
                }
                // bench échoué / score absent ⇒ neutre (pass-rate), pas d'accept
                // sur une mesure manquante.
                None => {
                    notes = "all green; bench sans score → pass-rate".to_string();
                    passrate
                }
            }
        } else if self.score_from_passrate {
            passrate
        } else {
            0.0
        };

        Ok(Fitness { compiles: true, tests_passed: passed, tests_failed: failed, score, notes })
    }
}

/// Anti-bruit : quand l'amélioration ne porte QUE sur le `score` (barrières
/// compile/tests à égalité), exige un gain relatif ≥ `min_gain` (rapporté à
/// `|score parent|`). Les améliorations **structurelles** (compile, tests) sont
/// toujours acceptées. `min_gain <= 0` ⇒ pas de seuil.
fn meets_min_gain(cand: &Fitness, parent: &Fitness, min_gain: f64) -> bool {
    if min_gain <= 0.0 {
        return true;
    }
    // Gain structurel (compile / tests) : toujours accepté.
    if cand.compiles != parent.compiles
        || cand.tests_failed != parent.tests_failed
        || cand.tests_passed != parent.tests_passed
    {
        return true;
    }
    // Gain purement de score : exiger le seuil relatif.
    let base = parent.score.abs().max(1e-12);
    cand.score >= parent.score + base * min_gain
}

/// Extrait la dernière valeur `RSI_BENCH_SCORE=<f64>` d'une sortie de bench
/// (fonction pure, testable). `None` si absente ou non finie.
fn parse_bench_score(output: &str) -> Option<f64> {
    output.lines().rev().find_map(|l| {
        l.trim()
            .strip_prefix("RSI_BENCH_SCORE=")
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|s| s.is_finite())
    })
}

/// Lance une commande avec **timeout** et **sortie bornée** (stdout+stderr
/// fusionnés). Rend `(success, output)`. Un dépassement de délai ⇒ kill et
/// `success = false`. std-only (sondage `try_wait`, lecture en threads).
fn run_bounded(mut cmd: Command, timeout: Duration, max_output: u64) -> std::io::Result<(bool, String)> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let per_stream = (max_output / 2).max(1);

    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let (tx_o, rx_o) = mpsc::channel();
    let (tx_e, rx_e) = mpsc::channel();
    if let Some(o) = stdout.take() {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = o.take(per_stream).read_to_end(&mut buf);
            let _ = tx_o.send(buf);
        });
    } else {
        let _ = tx_o.send(Vec::new());
    }
    if let Some(e) = stderr.take() {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = e.take(per_stream).read_to_end(&mut buf);
            let _ = tx_e.send(buf);
        });
    } else {
        let _ = tx_e.send(Vec::new());
    }

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None; // timeout
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };

    let ob = rx_o.recv_timeout(Duration::from_secs(2)).unwrap_or_default();
    let eb = rx_e.recv_timeout(Duration::from_secs(2)).unwrap_or_default();
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&ob),
        String::from_utf8_lossy(&eb)
    );
    Ok((status.map(|s| s.success()).unwrap_or(false), combined))
}

/// Somme les lignes `test result: ok. N passed; M failed` que `cargo` émet par
/// binaire de test.
fn parse_test_counts(output: &str) -> (u32, u32) {
    let mut passed = 0u32;
    let mut failed = 0u32;
    for line in output.lines() {
        let line = line.trim();
        if !line.starts_with("test result:") {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        for pair in tokens.windows(2) {
            let kind = pair[1].trim_end_matches(';');
            match kind {
                "passed" => passed += pair[0].parse().unwrap_or(0),
                "failed" => failed += pair[0].parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    (passed, failed)
}

fn tail(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let start = s.len() - max;
        let start = (start..s.len()).find(|i| s.is_char_boundary(*i)).unwrap_or(s.len());
        format!("…{}", &s[start..])
    }
}

/// Évaluateur adossé à une closure arbitraire — pratique pour les tests et les
/// récompenses de domaine qui n'ont pas besoin d'un build complet.
pub struct ClosureEvaluator<F>
where
    F: Fn(&Path) -> Fitness,
{
    f: F,
}

impl<F> ClosureEvaluator<F>
where
    F: Fn(&Path) -> Fitness,
{
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F> Evaluator for ClosureEvaluator<F>
where
    F: Fn(&Path) -> Fitness,
{
    fn evaluate(&self, workspace: &Path) -> Result<Fitness> {
        Ok((self.f)(workspace))
    }
}

// ════════════════════════════ Proposeur LLM ══════════════════════════════ //

/// Adapte n'importe quel [`crate::llm::LlmClient`] de RSI (Ollama, Claude…) en
/// [`CodeModel`], pour piloter la boucle DGM avec les backends existants.
pub struct LlmCodeModel<C: crate::llm::LlmClient> {
    client: C,
}

impl<C: crate::llm::LlmClient> LlmCodeModel<C> {
    pub fn new(client: C) -> Self {
        Self { client }
    }
}

impl<C: crate::llm::LlmClient> CodeModel for LlmCodeModel<C> {
    fn complete(&self, prompt: &str) -> Result<String> {
        // DGM a besoin de la complétion **brute entière** (lignes vides
        // préservées) : le `FIND` de l'enveloppe TARGET/FIND/REPLACE doit matcher
        // le fichier au caractère près. `complete_raw` évite le découpage par
        // ligne (qui filtrait les lignes vides et faisait échouer l'application
        // du patch — « pattern not found »).
        let raw = self
            .client
            .complete_raw(prompt)
            .map_err(|e| DgmError::Proposer(format!("{e:?}")))?;
        if raw.trim().is_empty() {
            return Err(DgmError::Proposer("backend returned no completion".to_string()));
        }
        Ok(raw)
    }
}

/// Enveloppe un [`CodeModel`] pour qu'il pilote la boucle d'auto-amélioration —
/// c'est l'« improver » de STOP.
pub struct LlmProposer<M: CodeModel> {
    model: M,
    /// Fichiers que le modèle a le droit de toucher, relatifs à la racine. Un
    /// changement sur autre chose est écarté — le garde-fou principal.
    allowed_paths: Vec<String>,
}

impl<M: CodeModel> LlmProposer<M> {
    pub fn new(model: M, allowed_paths: Vec<String>) -> Self {
        Self { model, allowed_paths }
    }

    fn build_prompt(&self, ctx: &ImprovementContext<'_>) -> String {
        let fitness = ctx
            .parent_fitness
            .map(|f| {
                format!(
                    "compiles={} tests_passed={} tests_failed={} score={:.4}",
                    f.compiles, f.tests_passed, f.tests_failed, f.score
                )
            })
            .unwrap_or_else(|| "unknown".to_string());

        let mut lessons = String::new();
        if !ctx.recent_rejections.is_empty() {
            lessons.push_str("\nRecently rejected ideas (do not repeat these):\n");
            for r in ctx.recent_rejections {
                lessons.push_str(&format!("- {r}\n"));
            }
        }
        if !ctx.simulated_feedback.is_empty() {
            lessons.push_str(
                "\nA fast simulator predicted your PREVIOUS attempt this step would FAIL. \
                 Fix the specific problem — do NOT resubmit the same edit:\n",
            );
            for f in ctx.simulated_feedback {
                lessons.push_str(&format!("- {f}\n"));
            }
        }

        // Contenu ACTUEL des fichiers éditables : sans lui, le modèle invente du
        // code inexistant et son `FIND` ne matche jamais. On borne chaque fichier
        // pour garder le prompt raisonnable (~7k tokens — le client Ollama fixe
        // `num_ctx=16384`, cf. `llm::OllamaClient` : le défaut serveur de 4096
        // tronquait le prompt sur les fichiers réels comme src/json.rs).
        const MAX_FILE_CHARS: usize = 24_000;
        let mut sources = String::new();
        for path in &self.allowed_paths {
            match ctx.read(path) {
                Ok(content) => {
                    let shown: String = content.chars().take(MAX_FILE_CHARS).collect();
                    let truncated = if content.len() > shown.len() {
                        "\n// … (tronqué) …"
                    } else {
                        ""
                    };
                    sources.push_str(&format!("\n----- {path} -----\n{shown}{truncated}\n"));
                }
                Err(_) => sources.push_str(&format!("\n----- {path} (illisible) -----\n")),
            }
        }

        format!(
            "You are improving a Rust codebase. Goal: {goal}\n\
             Parent fitness: {fitness}\n\
             You may ONLY edit these files: {allowed}\n\
             Here is their CURRENT content — your FIND text MUST be copied \
             verbatim from it (exact, occurring once):\n{sources}\n{lessons}\n\
             Propose ONE small, safe, compiling change. The FIND block must be an \
             EXACT substring of the file above, and it must be SHORT: the smallest \
             unique fragment enclosing your edit (at most ~25 lines). NEVER quote a \
             whole function — long FIND blocks get truncated and are rejected. \
             The REPLACE block MUST DIFFER from FIND: replaying the same code \
             unchanged is a no-op and is rejected. If you see no real improvement, \
             answer exactly: NO PROPOSAL. \
             Respond EXACTLY in this format and nothing else (close every code fence):\n\
             TARGET: <relative/path.rs>\n\
             FIND:\n<<<\n<exact existing text, occurring once>\n>>>\n\
             REPLACE:\n<<<\n<the replacement text>\n>>>\n\
             RATIONALE: <one short line>\n",
            goal = ctx.goal,
            fitness = fitness,
            allowed = self.allowed_paths.join(", "),
            sources = sources,
            lessons = lessons,
        )
    }
}

impl<M: CodeModel> Proposer for LlmProposer<M> {
    fn propose(&self, ctx: &ImprovementContext<'_>, _rng: &mut Rng) -> Result<Option<Proposal>> {
        let prompt = self.build_prompt(ctx);
        let raw = self.model.complete(&prompt)?;
        let proposal = match parse_proposal(&raw) {
            Some(p) => p,
            None => {
                // Diagnostic : `RSI_DGM_DEBUG=1` affiche la RAISON de l'échec
                // et la réponse brute EN ENTIER. L'ancien aperçu de 2000 chars
                // masquait la fin — des réponses complètes mais no-op (FIND ==
                // REPLACE) ont été prises pour des troncatures serveur.
                if std::env::var("RSI_DGM_DEBUG").is_ok() {
                    let full: String = raw.chars().take(20_000).collect();
                    eprintln!(
                        "[dgm] réponse LLM non parsée ({} chars) — {} :\n{full}\n--- fin ---",
                        raw.len(),
                        explain_parse_failure(&raw)
                    );
                }
                return Ok(None);
            }
        };
        // Garde-fou : ne jamais laisser le modèle s'échapper de la liste blanche.
        if !self.allowed_paths.iter().any(|a| a == &proposal.patch.target) {
            crate::obs::warn(
                "dgm.proposal_outside_allowlist",
                &format!("target={}", proposal.patch.target),
            );
            return Ok(None);
        }
        Ok(Some(proposal))
    }
}

/// Parse l'enveloppe. Rend `None` si le modèle est hors-format — la boucle
/// traite cela comme « pas de proposition », jamais comme un crash.
fn parse_proposal(raw: &str) -> Option<Proposal> {
    let target = line_value(raw, "TARGET:")?;
    // FIND est borné par `REPLACE:` ; REPLACE par `RATIONALE:` (ou la fin).
    let find = extract_block(raw, "FIND:", &["REPLACE:"])?;
    // REPLACE aussi borné par `FIND:` : quand le modèle concatène DEUX
    // propositions, le bloc REPLACE de la première ne doit pas avaler
    // l'enveloppe de la seconde (vécu : littéral « REPLACE: » injecté dans le
    // fichier, attrapé par le gate mais évitable ici).
    let replace = extract_block(raw, "REPLACE:", &["RATIONALE:", "TARGET:", "FIND:", "REPLACE:"])?;
    let rationale = line_value(raw, "RATIONALE:").unwrap_or_else(|| "llm proposal".to_string());
    if find.is_empty() || find == replace {
        return None;
    }
    Some(Proposal { patch: Patch::new(target, find, replace), rationale })
}

/// Explique pourquoi [`parse_proposal`] a rendu `None` (diagnostic debug) :
/// distingue enveloppe absente, bloc tronqué et **no-op** (FIND == REPLACE) —
/// trois échecs aux remèdes opposés (prompt, num_predict, ou modèle à court
/// d'idées qui recopie le bloc à l'identique).
fn explain_parse_failure(raw: &str) -> &'static str {
    if line_value(raw, "TARGET:").is_none() {
        return "TARGET absent";
    }
    let find = extract_block(raw, "FIND:", &["REPLACE:"]);
    if find.is_none() {
        return "bloc FIND absent ou vide";
    }
    let replace = extract_block(raw, "REPLACE:", &["RATIONALE:", "TARGET:", "FIND:", "REPLACE:"]);
    if replace.is_none() {
        return "bloc REPLACE absent ou vide (réponse tronquée ?)";
    }
    if find == replace {
        return "FIND == REPLACE (no-op : le modèle recopie le bloc sans le changer)";
    }
    "raison inconnue"
}

fn line_value(raw: &str, key: &str) -> Option<String> {
    raw.lines()
        .find_map(|l| l.trim().strip_prefix(key).map(|v| v.trim().to_string()))
        .filter(|s| !s.is_empty())
}

/// Extrait le bloc qui suit `key`, borné par le prochain marqueur de section
/// `ends` (ou la fin). Accepte **trois cadrages**, du plus strict au plus
/// permissif (les modèles varient d'une réponse à l'autre) :
///   1. balises strictes `<<<` … `>>>` ;
///   2. bloc clôturé par ``` ``` ``` ``` (étiquette de langage ```` ```rust ````
///      ignorée ; tolère une fence non fermée) ;
///   3. **brut** : le texte entre la ligne `key` et le prochain marqueur — cas
///      fréquent où le modèle colle le code directement après `FIND:`/`REPLACE:`
///      sans délimiteur.
fn extract_block(raw: &str, key: &str, ends: &[&str]) -> Option<String> {
    let key_pos = raw.find(key)?;
    let after = &raw[key_pos + key.len()..];
    // borne : jusqu'au 1ᵉʳ marqueur de section suivant (isole les sections).
    let bound = ends.iter().filter_map(|m| after.find(m)).min().unwrap_or(after.len());
    let region = &after[..bound];

    // 1) <<< … >>>
    if let Some(open) = region.find("<<<") {
        let rest = &region[open + 3..];
        if let Some(close) = rest.find(">>>") {
            return Some(rest[..close].trim_matches('\n').to_string());
        }
    }
    // 2) ``` … ```  (fence, éventuellement non fermée dans la région)
    if let Some(open) = region.find("```") {
        let rest = &region[open + 3..];
        let body = match rest.find('\n') {
            Some(nl) => &rest[nl + 1..],
            None => rest,
        };
        let end = body.find("```").unwrap_or(body.len());
        let block = body[..end].trim_matches('\n');
        if !block.is_empty() {
            return Some(block.to_string());
        }
    }
    // 3) brut : région après la 1ʳᵉ ligne (reste de la ligne `key`, souvent vide)
    let body = match region.find('\n') {
        Some(nl) => &region[nl + 1..],
        None => region,
    };
    let block = body.trim_matches('\n');
    if block.is_empty() {
        None
    } else {
        Some(block.to_string())
    }
}

/// Un verdict prédit est-il un cassé conclu (`pass == Some(false)`) ?
fn pred_is_broken(pred: &Option<Prediction>) -> bool {
    matches!(pred, Some(p) if p.pass == Some(false))
}

// ════════════════════════════════ Moteur ═════════════════════════════════ //

/// Réglages de la boucle.
#[derive(Debug, Clone)]
pub struct DgmConfig {
    /// Racine du workspace vivant, snapshotée à chaque évaluation.
    pub workspace_root: PathBuf,
    /// Objectif de haut niveau remis au proposeur.
    pub goal: String,
    /// Si vrai, une variante n'est acceptée que si toute sa suite de tests est
    /// verte (compile + zéro échec). Fortement recommandé en mode non surveillé.
    pub accept_requires_all_green: bool,
    /// Combien de justifications de rejet récentes re-fournir au proposeur.
    pub rejection_memory: usize,
    /// **Gain minimal relatif** exigé quand l'amélioration porte *uniquement*
    /// sur le `score` (barrières compile/tests à égalité). Anti-bruit : une
    /// mesure de perf a une variance run-to-run ; sans seuil, la boucle
    /// « accepte » le moindre écart, y compris du bruit. `0.0` = comportement
    /// d'origine (tout `>` strict accepté). Ex. `0.02` = exiger ≥ 2 % de gain.
    /// Sans effet sur les gains *structurels* (compile / tests) qui restent
    /// toujours acceptés.
    pub min_score_gain: f64,
    /// Révision simulée (axe 3) : nb max de renvois d'une proposition prédite
    /// cassée au proposeur (avec raison) avant le gate réel. `0` = pré-crible pur.
    pub simulated_revisions: u32,
}

impl DgmConfig {
    pub fn new(workspace_root: impl Into<PathBuf>, goal: impl Into<String>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            goal: goal.into(),
            accept_requires_all_green: true,
            rejection_memory: 8,
            min_score_gain: 0.0,
            simulated_revisions: 0,
        }
    }
}

/// Ce qui s'est passé en une étape — l'unité auditée de progrès.
#[derive(Debug, Clone)]
pub enum StepOutcome {
    /// Le proposeur a décliné de suggérer un changement.
    NoProposal,
    /// Une candidate a été évaluée. `accepted` est vrai ssi elle entre dans
    /// l'archive.
    Evaluated {
        variant_id: String,
        parent_id: Option<String>,
        accepted: bool,
        fitness: Fitness,
    },
}

impl StepOutcome {
    pub fn accepted(&self) -> bool {
        matches!(self, StepOutcome::Evaluated { accepted: true, .. })
    }
}

/// Le moteur d'auto-amélioration. Générique sur le proposeur et l'évaluateur :
/// la même boucle pilote les tests unitaires (stubs déterministes) et la
/// production (LLM + `cargo`).
pub struct DgmEngine<P: Proposer, E: Evaluator> {
    archive: Archive,
    proposer: P,
    evaluator: E,
    config: DgmConfig,
    rng: Rng,
    recent_rejections: Vec<String>,
    history: Vec<StepOutcome>,
    next_seq: u64,
    /// Pré-crible optionnel (world model) : évite des `cargo build` sur les
    /// patchs prédits cassés avec certitude. Jamais autoritaire — cf.
    /// [`VerdictPredictor`].
    predictor: Option<Box<dyn VerdictPredictor>>,
    /// Nombre de builds économisés par le pré-crible (reporting).
    prescreen_skips: u64,
    /// Nombre de révisions simulées effectuées (axe 3, reporting).
    revisions: u64,
    /// Trajectoires à vérité terrain (axe 4, flywheel). Cf. [`crate::trajectory`].
    trajectories: Vec<Trajectory>,
}

impl<P: Proposer, E: Evaluator> DgmEngine<P, E> {
    pub fn new(archive: Archive, proposer: P, evaluator: E, config: DgmConfig, seed: u64) -> Self {
        let next_seq = archive.len() as u64;
        Self {
            archive,
            proposer,
            evaluator,
            config,
            rng: Rng::new(seed),
            recent_rejections: Vec::new(),
            history: Vec::new(),
            next_seq,
            predictor: None,
            prescreen_skips: 0,
            revisions: 0,
            trajectories: Vec::new(),
        }
    }

    /// Attache un pré-crible (world model) — cf. [`VerdictPredictor`].
    pub fn with_predictor(mut self, predictor: Box<dyn VerdictPredictor>) -> Self {
        self.predictor = Some(predictor);
        self
    }

    /// Nombre de `cargo build` évités par le pré-crible depuis le début.
    pub fn prescreen_skips(&self) -> u64 {
        self.prescreen_skips
    }

    /// Nombre de révisions simulées (axe 3) effectuées depuis le début.
    pub fn revisions(&self) -> u64 {
        self.revisions
    }

    /// Trajectoires à vérité terrain (axe 4). Sérialiser via [`crate::trajectory::to_jsonl`].
    pub fn trajectories(&self) -> &[Trajectory] {
        &self.trajectories
    }

    pub fn archive(&self) -> &Archive {
        &self.archive
    }

    pub fn history(&self) -> &[StepOutcome] {
        &self.history
    }

    pub fn best(&self) -> Option<&Variant> {
        self.archive.best()
    }

    /// Lance une étape propose → évalue → sélectionne.
    pub fn step(&mut self) -> Result<StepOutcome> {
        // 1. Choisir un parent dont brancher (sélection ouverte).
        let parent = match self.archive.select_parent(&mut self.rng) {
            Some(v) => v.clone(),
            None => return Ok(self.record(StepOutcome::NoProposal)),
        };

        // 2. Proposeur (premier essai : aucun feedback simulé).
        let rejections = self.recent_rejections.clone();
        let mut sim_feedback: Vec<String> = Vec::new();
        let mut proposal = match self.propose_with(&parent, &rejections, &sim_feedback)? {
            Some(p) if !p.patch.is_noop() => p,
            _ => return Ok(self.record(StepOutcome::NoProposal)),
        };

        // 2b. Pré-crible + RÉVISION SIMULÉE (axe 3) sur la proposition finale.
        let mut final_pred = self.predict_patch(&proposal.patch);
        let mut rev = 0;
        while rev < self.config.simulated_revisions && pred_is_broken(&final_pred) {
            let reason = final_pred
                .as_ref()
                .and_then(|p| p.reason.clone())
                .unwrap_or_else(|| "un simulateur prédit que ce patch casserait".to_string());
            sim_feedback.push(reason);
            self.revisions += 1;
            rev += 1;
            match self.propose_with(&parent, &rejections, &sim_feedback)? {
                Some(p) if !p.patch.is_noop() => proposal = p,
                _ => break,
            }
            final_pred = self.predict_patch(&proposal.patch);
        }

        // 3. Évaluer. Prédit cassé ⇒ build sauté ; sinon gate réel autoritaire.
        let fitness = if pred_is_broken(&final_pred) {
            self.prescreen_skips += 1;
            Fitness::broken("prescreen: prédit cassé (build évité)")
        } else {
            match self.evaluate_candidate(&proposal.patch) {
                Ok(f) => {
                    self.capture_trajectory(&proposal.patch, &f);
                    f
                }
                Err(e) => Fitness::broken(format!("could not evaluate: {e}")),
            }
        };

        // 4. N'accepter que si elle bat le parent sous l'ordre de barrière.
        let parent_fit = parent
            .fitness
            .clone()
            .unwrap_or_else(|| Fitness::broken("parent had no fitness"));
        let gate_ok = !self.config.accept_requires_all_green || fitness.all_green();
        let accepted = gate_ok
            && fitness.is_better_than(&parent_fit)
            && meets_min_gain(&fitness, &parent_fit, self.config.min_score_gain);

        let seq = self.next_seq;
        self.next_seq += 1;
        let mut child = Variant::child(&parent, proposal.patch, proposal.rationale, seq);
        child.fitness = Some(fitness.clone());
        child.status = if accepted { Status::Accepted } else { Status::Rejected };

        if accepted {
            crate::obs::info(
                "dgm.accepted",
                &format!(
                    "variant={} generation={} score={}",
                    child.id, child.generation, fitness.score
                ),
            );
            self.archive.insert(child.clone());
        } else {
            self.remember_rejection(&child.rationale);
        }

        Ok(self.record(StepOutcome::Evaluated {
            variant_id: child.id,
            parent_id: child.parent,
            accepted,
            fitness,
        }))
    }

    /// Lance jusqu'à `max_steps`, en rendant le résultat de chaque étape.
    pub fn run(&mut self, max_steps: usize) -> Result<Vec<StepOutcome>> {
        let mut out = Vec::with_capacity(max_steps);
        for _ in 0..max_steps {
            out.push(self.step()?);
        }
        Ok(out)
    }

    fn record(&mut self, o: StepOutcome) -> StepOutcome {
        self.history.push(o.clone());
        o
    }

    /// Demande une proposition (rejets passés + feedback simulé du step).
    fn propose_with(&mut self, parent: &Variant, rejections: &[String], sim_feedback: &[String]) -> Result<Option<Proposal>> {
        let ctx = ImprovementContext {
            workspace_root: &self.config.workspace_root,
            goal: &self.config.goal,
            parent_fitness: parent.fitness.as_ref(),
            recent_rejections: rejections,
            simulated_feedback: sim_feedback,
        };
        self.proposer.propose(&ctx, &mut self.rng)
    }

    /// Interroge le prédicteur. `None` si pas de prédicteur / fichier illisible.
    fn predict_patch(&self, patch: &Patch) -> Option<Prediction> {
        let pred = self.predictor.as_ref()?;
        let content =
            std::fs::read_to_string(self.config.workspace_root.join(&patch.target)).ok()?;
        Some(pred.predict(&patch.target, &content, &patch.find, &patch.replace))
    }

    /// Enregistre une trajectoire à vérité terrain (axe 4). Contenu borné (~12k).
    fn capture_trajectory(&mut self, patch: &Patch, fitness: &Fitness) {
        const MAX_FILE_CHARS: usize = 12_000;
        let content = std::fs::read_to_string(self.config.workspace_root.join(&patch.target))
            .unwrap_or_default();
        let file_content: String = content.chars().take(MAX_FILE_CHARS).collect();
        // Sur échec (build cassé / tests rouges), `notes` porte la vraie sortie
        // cargo bornée → complétion riche. Sur succès, on laisse le gabarit
        // « running N tests…ok » (plus cargo-réaliste que le résumé de `notes`).
        let output = if !fitness.compiles || fitness.tests_failed > 0 {
            Some(fitness.notes.clone())
        } else {
            None
        };
        self.trajectories.push(Trajectory {
            target: patch.target.clone(),
            find: patch.find.clone(),
            replace: patch.replace.clone(),
            file_content,
            compiles: fitness.compiles,
            tests_passed: fitness.tests_passed,
            tests_failed: fitness.tests_failed,
            score: fitness.score,
            output,
        });
    }

    fn evaluate_candidate(&self, patch: &Patch) -> Result<Fitness> {
        let snap = WorkspaceSnapshot::create(&self.config.workspace_root)?;
        if let Err(e) = snap.apply(patch) {
            // Un patch qui ne s'applique pas (motif manquant) est une candidate
            // ratée, pas un crash de la boucle.
            return Ok(Fitness::broken(format!("patch did not apply: {e}")));
        }
        self.evaluator.evaluate(snap.root())
    }

    fn remember_rejection(&mut self, rationale: &str) {
        self.recent_rejections.push(rationale.to_string());
        let cap = self.config.rejection_memory;
        if self.recent_rejections.len() > cap {
            let overflow = self.recent_rejections.len() - cap;
            self.recent_rejections.drain(0..overflow);
        }
    }
}

// ════════════════════════════════ Tests ══════════════════════════════════ //

#[cfg(test)]
mod tests {
    use super::*;

    fn fit(compiles: bool, passed: u32, failed: u32, score: f64) -> Fitness {
        Fitness { compiles, tests_passed: passed, tests_failed: failed, score, notes: String::new() }
    }

    // ---- Fitness : barrière lexicographique ---- //

    #[test]
    fn compile_gate_dominates_score() {
        let broken = fit(false, 0, 0, 1_000.0);
        let working = fit(true, 1, 0, -1.0);
        assert!(working.is_better_than(&broken));
        assert!(!broken.is_better_than(&working));
    }

    #[test]
    fn test_regression_beats_score() {
        let regressed = fit(true, 10, 1, 100.0);
        let clean = fit(true, 5, 0, 0.0);
        assert!(clean.is_better_than(&regressed));
    }

    #[test]
    fn score_breaks_ties() {
        let lo = fit(true, 5, 0, 1.0);
        let hi = fit(true, 5, 0, 2.0);
        assert!(hi.is_better_than(&lo));
        assert!(!lo.is_better_than(&hi));
    }

    #[test]
    fn nan_score_is_never_an_improvement() {
        let nan = fit(true, 5, 0, f64::NAN);
        let ok = fit(true, 5, 0, 0.0);
        assert!(!nan.is_better_than(&ok));
    }

    #[test]
    fn noop_patch_detected() {
        assert!(Patch::new("a.rs", "x", "x").is_noop());
        assert!(!Patch::new("a.rs", "x", "y").is_noop());
    }

    // ---- IDs déterministes (amélioration vs uuid) ---- //

    #[test]
    fn variant_ids_are_deterministic() {
        let base = fit(true, 1, 0, 0.0);
        let r1 = Variant::root(base.clone());
        let r2 = Variant::root(base);
        assert_eq!(r1.id, r2.id, "root id must be reproducible");
        let p = Patch::new("a.rs", "x", "y");
        let c1 = Variant::child(&r1, p.clone(), "improve", 1);
        let c2 = Variant::child(&r2, p, "improve", 1);
        assert_eq!(c1.id, c2.id, "child id must be reproducible");
        assert_ne!(c1.id, r1.id);
    }

    // ---- Archive ---- //

    fn afit(score: f64) -> Fitness {
        fit(true, 1, 0, score)
    }

    #[test]
    fn root_archive_has_one_entry() {
        let a = Archive::with_root(afit(0.0));
        assert_eq!(a.len(), 1);
        assert!(!a.is_empty());
    }

    #[test]
    fn best_tracks_highest_fitness() {
        let mut a = Archive::with_root(afit(0.0));
        let root = a.variants()[0].clone();
        let mut better = Variant::child(&root, Patch::new("a", "x", "y"), "improve", 1);
        better.fitness = Some(afit(5.0));
        a.insert(better.clone());
        assert_eq!(a.best().unwrap().id, better.id);
    }

    #[test]
    fn parent_selection_is_seed_deterministic() {
        let a = Archive::with_root(afit(0.0));
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        let p1 = a.select_parent(&mut r1).unwrap().id.clone();
        let p2 = a.select_parent(&mut r2).unwrap().id.clone();
        assert_eq!(p1, p2);
    }

    #[test]
    fn json_round_trip_archive() {
        let mut a = Archive::with_root(afit(1.5));
        let root = a.variants()[0].clone();
        // une variante normale…
        let mut child = Variant::child(&root, Patch::new("src/x.rs", "a", "b"), "tweak", 1);
        child.fitness = Some(afit(2.0));
        child.status = Status::Accepted;
        a.insert(child);
        // …et une variante cassée (-∞, doit round-tripper via null).
        let mut broken = Variant::child(&root, Patch::new("src/y.rs", "c", "d"), "oops", 2);
        broken.fitness = Some(Fitness::broken("syntax error"));
        broken.status = Status::Rejected;
        a.insert(broken);

        let s = a.to_json();
        let b = Archive::from_json(&s).unwrap();
        assert_eq!(b.len(), a.len());
        // l'id, le statut et la fitness survivent au tour.
        let rebuilt_broken = b.variants().iter().find(|v| v.patch.target == "src/y.rs").unwrap();
        assert_eq!(rebuilt_broken.status, Status::Rejected);
        let f = rebuilt_broken.fitness.as_ref().unwrap();
        assert!(!f.compiles && f.score == f64::NEG_INFINITY);
    }

    // ---- Snapshot : isolation ---- //

    fn write(root: &Path, rel: &str, content: &str) {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, content).unwrap();
    }

    fn fresh_dir(tag: &str) -> PathBuf {
        let d = unique_tmp_dir().join(tag);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn snapshot_copies_and_isolates() {
        let live = fresh_dir("live");
        write(&live, "src/lib.rs", "fn main() { let x = 0; }");
        write(&live, "target/junk.o", "binary"); // doit être sauté

        let snap = WorkspaceSnapshot::create(&live).unwrap();
        assert!(snap.resolve("src/lib.rs").exists());
        assert!(!snap.resolve("target/junk.o").exists());

        snap.apply(&Patch::new("src/lib.rs", "let x = 0;", "let x = 1;")).unwrap();
        let live_src = std::fs::read_to_string(live.join("src/lib.rs")).unwrap();
        assert!(live_src.contains("let x = 0;"), "live tree was mutated");
        let snap_src = std::fs::read_to_string(snap.resolve("src/lib.rs")).unwrap();
        assert!(snap_src.contains("let x = 1;"));
        let _ = std::fs::remove_dir_all(&live);
    }

    #[test]
    fn noop_patch_is_rejected_by_snapshot() {
        let live = fresh_dir("noop");
        write(&live, "a.rs", "x");
        let snap = WorkspaceSnapshot::create(&live).unwrap();
        assert!(snap.apply(&Patch::new("a.rs", "x", "x")).is_err());
        let _ = std::fs::remove_dir_all(&live);
    }

    #[test]
    fn ambiguous_patch_is_rejected() {
        // Un motif présent deux fois ⇒ rejet (sûreté : pas d'édition ambiguë).
        let live = fresh_dir("ambig");
        write(&live, "a.rs", "let v = 1; let v = 1;");
        let snap = WorkspaceSnapshot::create(&live).unwrap();
        let err = snap.apply(&Patch::new("a.rs", "let v = 1;", "let v = 2;"));
        assert!(err.is_err(), "non-unique pattern must be rejected");
        let _ = std::fs::remove_dir_all(&live);
    }

    #[test]
    fn fuzzy_match_tolerates_indentation() {
        // Le modèle recopie le bloc avec une indentation différente (2 espaces
        // au lieu de 4) : l'appariement exact échoue, le repli ligne-à-ligne le
        // retrouve et édite les BONNES lignes en préservant les alentours.
        let content = "fn f() {\n    let x = 0;\n    g();\n}\n";
        let (s, e) = locate_match(content, "  let x = 0;\n  g();").unwrap();
        assert_eq!(&content[s..e], "    let x = 0;\n    g();");
    }

    #[test]
    fn fuzzy_match_stays_unique() {
        // Deux blocs identiques après trim ⇒ le repli refuse (jamais d'édition
        // ambiguë, même en tolérant les espaces).
        let content = "  a();\n  a();\n";
        assert!(locate_match(content, "a();").is_err());
    }

    #[test]
    fn exact_match_preferred_over_fuzzy() {
        // Présent tel quel exactement une fois → chemin exact (span = substring).
        let content = "let value = 1;\n";
        let (s, e) = locate_match(content, "let value = 1;").unwrap();
        assert_eq!(&content[s..e], "let value = 1;");
    }

    #[test]
    fn fuzzy_patch_applies_through_snapshot() {
        // Bout en bout : un patch dont le FIND diffère par l'indentation est
        // néanmoins appliqué dans le snapshot.
        let live = fresh_dir("fuzzy");
        write(&live, "a.rs", "fn f() {\n    let x = 0;\n}\n");
        let snap = WorkspaceSnapshot::create(&live).unwrap();
        // FIND sur-indenté (8 espaces) : PAS un sous-chaîne exacte de la ligne à
        // 4 espaces ⇒ force le repli ligne-à-ligne.
        snap.apply(&Patch::new("a.rs", "        let x = 0;", "    let x = 42;")).unwrap();
        let src = std::fs::read_to_string(snap.root().join("a.rs")).unwrap();
        assert_eq!(src, "fn f() {\n    let x = 42;\n}\n");
        let _ = std::fs::remove_dir_all(&live);
    }

    #[test]
    fn promote_writes_live_tree() {
        let live = fresh_dir("promote");
        write(&live, "a.rs", "value = 1");
        let backups = fresh_dir("promote-bak");
        promote_to_live(&live, &Patch::new("a.rs", "value = 1", "value = 2"), &backups).unwrap();
        let after = std::fs::read_to_string(live.join("a.rs")).unwrap();
        assert_eq!(after, "value = 2");
        let _ = std::fs::remove_dir_all(&live);
        let _ = std::fs::remove_dir_all(&backups);
    }

    // ---- Évaluateur : parsing de sortie cargo ---- //

    #[test]
    fn parses_multi_binary_test_output() {
        let out = "\
running 3 tests
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
running 2 tests
test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
";
        assert_eq!(parse_test_counts(out), (4, 1));
    }

    #[test]
    fn no_test_lines_is_zero() {
        assert_eq!(parse_test_counts("nothing here"), (0, 0));
    }

    #[test]
    fn min_gain_rejects_noise_keeps_real_and_structural() {
        let base = fit(true, 176, 0, 19640.0);
        let noise = fit(true, 176, 0, 19689.0); // +0.25 % → bruit
        let real = fit(true, 176, 0, 20500.0); // +4.4 % → réel
        let more_tests = fit(true, 177, 0, 19000.0); // moins de perf mais +1 test
        // sans seuil : tout `>` accepté
        assert!(meets_min_gain(&noise, &base, 0.0));
        // seuil 2 % : le bruit est rejeté, le vrai gain passe
        assert!(!meets_min_gain(&noise, &base, 0.02));
        assert!(meets_min_gain(&real, &base, 0.02));
        // amélioration STRUCTURELLE (tests) toujours acceptée, même seuil actif
        assert!(meets_min_gain(&more_tests, &base, 0.02));
    }

    #[test]
    fn parses_bench_score_last_finite() {
        let out = "Compiling…\nRunning…\nRSI_BENCH_SCORE=1234.5\nnoise\n";
        assert_eq!(parse_bench_score(out), Some(1234.5));
        // dernière valeur retenue
        assert_eq!(
            parse_bench_score("RSI_BENCH_SCORE=1\nRSI_BENCH_SCORE=2\n"),
            Some(2.0)
        );
        // absente ou non finie ⇒ None
        assert_eq!(parse_bench_score("pas de score"), None);
        assert_eq!(parse_bench_score("RSI_BENCH_SCORE=inf"), None);
        assert_eq!(parse_bench_score("RSI_BENCH_SCORE=abc"), None);
    }

    #[test]
    fn closure_evaluator_runs() {
        let e = ClosureEvaluator::new(|_p: &Path| afit(9.0));
        let tmp = fresh_dir("clos");
        let f = e.evaluate(&tmp).unwrap();
        assert!(f.all_green());
        assert_eq!(f.score, 9.0);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ---- Proposeur LLM : parsing + liste blanche ---- //

    struct FixedModel(String);
    impl CodeModel for FixedModel {
        fn complete(&self, _prompt: &str) -> Result<String> {
            Ok(self.0.clone())
        }
    }

    const WELL_FORMED: &str = "\
TARGET: src/lib.rs
FIND:
<<<
let x = 0;
>>>
REPLACE:
<<<
let x = 1;
>>>
RATIONALE: bump the constant
";

    #[test]
    fn parses_well_formed_proposal() {
        let p = parse_proposal(WELL_FORMED).unwrap();
        assert_eq!(p.patch.target, "src/lib.rs");
        assert_eq!(p.patch.find, "let x = 0;");
        assert_eq!(p.patch.replace, "let x = 1;");
        assert_eq!(p.rationale, "bump the constant");
    }

    #[test]
    fn parses_fenced_proposal() {
        // Un modèle de code rend souvent des blocs ``` au lieu de <<< >>>.
        let raw = "TARGET: src/lib.rs\n\
                   FIND:\n```rust\nlet x = 0;\n```\n\
                   REPLACE:\n```rust\nlet x = 1;\n```\n\
                   RATIONALE: bump\n";
        let p = parse_proposal(raw).unwrap();
        assert_eq!(p.patch.target, "src/lib.rs");
        assert_eq!(p.patch.find, "let x = 0;");
        assert_eq!(p.patch.replace, "let x = 1;");
    }

    #[test]
    fn parses_unclosed_replace_fence() {
        // Cas réel Jetson : le modèle oublie de fermer la fence du REPLACE et
        // enchaîne sur RATIONALE. On doit quand même extraire le bloc.
        let raw = "TARGET: a.rs\n\
                   FIND:\n```\nlet x = 0;\n```\n\
                   REPLACE:\n```\nlet x = 1;\n}\n\
                   RATIONALE: fix\n";
        let p = parse_proposal(raw).unwrap();
        assert_eq!(p.patch.find, "let x = 0;");
        assert!(p.patch.replace.contains("let x = 1;"));
        assert!(!p.patch.replace.contains("RATIONALE"));
    }

    #[test]
    fn parses_undelimited_proposal() {
        // Cas réel Jetson : le modèle colle le code brut après FIND:/REPLACE:
        // sans <<< ni ``` — borné par le marqueur de section suivant.
        let raw = "TARGET: src/m.rs\n\
                   FIND:\n\
                   fn f() {\n    let x = 0;\n}\n\
                   REPLACE:\n\
                   fn f() {\n    let x = 1;\n}\n\
                   RATIONALE: bump\n";
        let p = parse_proposal(raw).unwrap();
        assert_eq!(p.patch.target, "src/m.rs");
        assert_eq!(p.patch.find, "fn f() {\n    let x = 0;\n}");
        assert_eq!(p.patch.replace, "fn f() {\n    let x = 1;\n}");
        assert!(!p.patch.replace.contains("RATIONALE"));
    }

    #[test]
    fn off_format_yields_none() {
        assert!(parse_proposal("I think you should change something.").is_none());
    }

    #[test]
    fn two_concatenated_proposals_keep_first_replace_clean() {
        // Cas réel Jetson/sha256 : le modèle émet DEUX propositions à la
        // suite. Le REPLACE de la première s'arrête au `FIND:` suivant — sans
        // quoi l'enveloppe de la seconde (« REPLACE: » littéral…) était
        // injectée dans le fichier (E-compile au gate, mais évitable ici).
        let raw = "TARGET: a.rs\nFIND:\n<<<\nlet x = 0;\n>>>\nREPLACE:\n<<<\nlet x = 1;\n>>>\n\
                   FIND:\n<<<\nlet y = 0;\n>>>\nREPLACE:\n<<<\nlet y = 1;\n>>>\nRATIONALE: deux\n";
        let p = parse_proposal(raw).unwrap();
        assert_eq!(p.patch.replace, "let x = 1;");
        assert!(!p.patch.replace.contains("REPLACE:"));
    }

    #[test]
    fn repeated_replace_label_does_not_leak_into_file() {
        let raw = "TARGET: src/kernels.rs\n\
                   FIND:\n\
                   for i in 0..n {\n    c[i] = 0.0;\n}\n\
                   REPLACE:\n\
                   for i in 0..n {\n    c[i] = 0.0f32;\n}\n\
                   REPLACE:\n\
                   for i in 0..n {\n    c[i] = 0.0f32;\n}\n\
                   RATIONALE: retype\n";
        let p = parse_proposal(raw).unwrap();
        assert_eq!(p.patch.replace, "for i in 0..n {\n    c[i] = 0.0f32;\n}");
        assert!(!p.patch.replace.contains("REPLACE:"));
    }

    #[test]
    fn explains_each_parse_failure() {
        // Cas réel Jetson/sha256 : réponse COMPLÈTE mais FIND == REPLACE — le
        // diagnostic doit nommer le no-op, pas laisser croire à une troncature.
        let noop = "TARGET: a.rs\nFIND:\n<<<\nlet x = 0;\n>>>\nREPLACE:\n<<<\nlet x = 0;\n>>>\nRATIONALE: r\n";
        assert!(parse_proposal(noop).is_none());
        assert!(explain_parse_failure(noop).contains("no-op"));

        assert_eq!(explain_parse_failure("blabla"), "TARGET absent");
        // Réponse coupée net juste après l'entête REPLACE: (rien derrière).
        let cut = "TARGET: a.rs\nFIND:\n<<<\nlet x = 0;\n>>>\nREPLACE:\n";
        assert!(explain_parse_failure(cut).contains("REPLACE"));
    }

    #[test]
    fn llm_code_model_rejoins_split_lines() {
        // Un LlmClient (style Ollama) rend une chaîne PAR LIGNE ; le modèle de
        // code DGM doit recomposer la complétion multi-ligne entière, pas se
        // limiter à la 1ʳᵉ ligne (régression du bug Jetson « raw = 21 chars »).
        struct LineClient;
        impl crate::llm::LlmClient for LineClient {
            fn propose(
                &self,
                _p: &str,
                _k: usize,
            ) -> std::result::Result<Vec<String>, crate::llm::LlmError> {
                Ok(vec![
                    "TARGET: a.rs".into(),
                    "FIND:".into(),
                    "<<<".into(),
                    "let x = 0;".into(),
                    ">>>".into(),
                    "REPLACE:".into(),
                    "<<<".into(),
                    "let x = 1;".into(),
                    ">>>".into(),
                    "RATIONALE: bump".into(),
                ])
            }
        }
        let model = LlmCodeModel::new(LineClient);
        let raw = model.complete("prompt").unwrap();
        let p = parse_proposal(&raw).expect("doit parser après recomposition");
        assert_eq!(p.patch.target, "a.rs");
        assert_eq!(p.patch.find, "let x = 0;");
        assert_eq!(p.patch.replace, "let x = 1;");
    }

    #[test]
    fn proposer_enforces_allow_list() {
        let mut rng = Rng::new(0);
        let root = std::path::Path::new("/tmp");
        let ctx = ImprovementContext {
            workspace_root: root,
            goal: "g",
            parent_fitness: None,
            recent_rejections: &[],
            simulated_feedback: &[],
        };
        let ok = LlmProposer::new(FixedModel(WELL_FORMED.to_string()), vec!["src/lib.rs".into()]);
        assert!(ok.propose(&ctx, &mut rng).unwrap().is_some());
        let blocked = LlmProposer::new(FixedModel(WELL_FORMED.to_string()), vec!["other.rs".into()]);
        assert!(blocked.propose(&ctx, &mut rng).unwrap().is_none());
    }

    // ---- Boucle complète (jouet, déterministe, sans cargo) ---- //

    fn toy_workspace(tag: &str) -> PathBuf {
        let d = fresh_dir(tag);
        write(&d, "src/level.txt", "level = 0");
        d
    }

    fn read_level(root: &Path) -> i64 {
        std::fs::read_to_string(root.join("src/level.txt"))
            .unwrap()
            .trim()
            .strip_prefix("level = ")
            .and_then(|s| s.parse().ok())
            .unwrap_or(-1)
    }

    struct Incrementer;
    impl Proposer for Incrementer {
        fn propose(&self, ctx: &ImprovementContext<'_>, _rng: &mut Rng) -> Result<Option<Proposal>> {
            let cur = read_level(ctx.workspace_root);
            let next = cur + 1;
            Ok(Some(Proposal {
                patch: Patch::new(
                    "src/level.txt",
                    format!("level = {cur}"),
                    format!("level = {next}"),
                ),
                rationale: format!("raise level to {next}"),
            }))
        }
    }

    fn level_evaluator() -> ClosureEvaluator<impl Fn(&Path) -> Fitness> {
        ClosureEvaluator::new(|root: &Path| fit(true, 1, 0, read_level(root) as f64))
    }

    fn engine(ws: &Path) -> DgmEngine<Incrementer, ClosureEvaluator<impl Fn(&Path) -> Fitness>> {
        let archive = Archive::with_root(fit(true, 1, 0, 0.0));
        let config = DgmConfig::new(ws, "raise the level");
        DgmEngine::new(archive, Incrementer, level_evaluator(), config, 1)
    }

    #[test]
    fn loop_accumulates_only_real_improvements() {
        let ws = toy_workspace("acc");
        let mut eng = engine(&ws);
        eng.run(5).unwrap();
        assert!(eng.archive().len() >= 2, "no improvement was archived");
        let best = eng.best().unwrap();
        assert!(best.fitness.as_ref().unwrap().score >= 1.0);
        // L'arbre vivant n'est jamais muté par la boucle.
        assert_eq!(read_level(&ws), 0);
        let _ = std::fs::remove_dir_all(&ws);
    }

    struct AlwaysBreaks;
    impl VerdictPredictor for AlwaysBreaks {
        fn predict(&self, _t: &str, _c: &str, _f: &str, _r: &str) -> Prediction {
            Prediction { pass: Some(false), reason: Some("test: casse".into()) }
        }
    }

    struct AlwaysUndecided;
    impl VerdictPredictor for AlwaysUndecided {
        fn predict(&self, _t: &str, _c: &str, _f: &str, _r: &str) -> Prediction {
            Prediction::default()
        }
    }

    struct BreaksWhileBad;
    impl VerdictPredictor for BreaksWhileBad {
        fn predict(&self, _t: &str, _c: &str, _f: &str, replace: &str) -> Prediction {
            if replace.contains("BAD") {
                Prediction { pass: Some(false), reason: Some("contient BAD".into()) }
            } else {
                Prediction { pass: Some(true), reason: None }
            }
        }
    }

    struct FixesOnFeedback;
    impl Proposer for FixesOnFeedback {
        fn propose(&self, ctx: &ImprovementContext<'_>, _rng: &mut Rng) -> Result<Option<Proposal>> {
            let replace = if ctx.simulated_feedback.is_empty() { "level = 0 BAD" } else { "level = 1" };
            Ok(Some(Proposal {
                patch: Patch::new("src/level.txt", "level = 0", replace),
                rationale: format!("feedback={}", ctx.simulated_feedback.len()),
            }))
        }
    }

    #[test]
    fn prescreen_skips_build_on_confident_break() {
        let ws = toy_workspace("prescreen-skip");
        // évaluateur qui PANIQUE s'il tourne — prouve que le build est évité
        let evaluator = ClosureEvaluator::new(|_r: &Path| -> Fitness {
            panic!("le gate réel ne doit PAS tourner quand le pré-crible dit « cassé »")
        });
        let mut eng = DgmEngine::new(
            Archive::with_root(fit(true, 1, 0, 0.0)),
            Incrementer,
            evaluator,
            DgmConfig::new(&ws, "raise"),
            1,
        )
        .with_predictor(Box::new(AlwaysBreaks));
        let outcomes = eng.run(3).unwrap();
        assert!(outcomes.iter().all(|o| !o.accepted()), "prédit cassé ⇒ rien accepté");
        assert_eq!(eng.prescreen_skips(), 3, "3 builds évités");
        assert_eq!(eng.archive().len(), 1, "rien archivé, arbre = racine");
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn prescreen_undecided_falls_through_to_real_gate() {
        // Indécis ⇒ le gate réel tranche ⇒ les vraies améliorations passent
        // (le simulateur n'écarte jamais une idée qu'il n'a pas condamnée).
        let ws = toy_workspace("prescreen-none");
        let mut eng = DgmEngine::new(
            Archive::with_root(fit(true, 1, 0, 0.0)),
            Incrementer,
            level_evaluator(),
            DgmConfig::new(&ws, "raise"),
            1,
        )
        .with_predictor(Box::new(AlwaysUndecided));
        eng.run(5).unwrap();
        assert!(eng.archive().len() >= 2, "indécis ⇒ gate réel ⇒ améliorations archivées");
        assert_eq!(eng.prescreen_skips(), 0);
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn simulated_revision_fixes_a_predicted_break_in_one_step() {
        let ws = toy_workspace("sim-revise");
        let mut cfg = DgmConfig::new(&ws, "raise");
        cfg.simulated_revisions = 2;
        let mut eng = DgmEngine::new(Archive::with_root(fit(true, 1, 0, 0.0)), FixesOnFeedback, level_evaluator(), cfg, 1)
            .with_predictor(Box::new(BreaksWhileBad));
        eng.run(1).unwrap();
        assert_eq!(eng.revisions(), 1);
        assert_eq!(eng.prescreen_skips(), 0);
        assert!(eng.archive().len() >= 2);
        assert!(eng.best().unwrap().fitness.as_ref().unwrap().score >= 1.0);
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn without_revisions_a_predicted_break_is_skipped() {
        let ws = toy_workspace("sim-norevise");
        let mut eng = DgmEngine::new(Archive::with_root(fit(true, 1, 0, 0.0)), FixesOnFeedback, level_evaluator(), DgmConfig::new(&ws, "raise"), 1)
            .with_predictor(Box::new(BreaksWhileBad));
        eng.run(3).unwrap();
        assert_eq!(eng.revisions(), 0);
        assert_eq!(eng.prescreen_skips(), 3);
        assert_eq!(eng.archive().len(), 1);
        let _ = std::fs::remove_dir_all(&ws);
    }

    struct Saboteur;
    impl Proposer for Saboteur {
        fn propose(&self, _ctx: &ImprovementContext<'_>, _rng: &mut Rng) -> Result<Option<Proposal>> {
            Ok(Some(Proposal {
                patch: Patch::new("src/level.txt", "level = 0", "level = 0 BROKEN"),
                rationale: "sabotage".to_string(),
            }))
        }
    }

    #[test]
    fn regressions_are_rejected() {
        let ws = toy_workspace("sab");
        let evaluator = ClosureEvaluator::new(|root: &Path| {
            let txt = std::fs::read_to_string(root.join("src/level.txt")).unwrap_or_default();
            if txt.contains("BROKEN") {
                Fitness::broken("syntax error")
            } else {
                fit(true, 1, 0, 0.0)
            }
        });
        let mut eng = DgmEngine::new(
            Archive::with_root(fit(true, 1, 0, 0.0)),
            Saboteur,
            evaluator,
            DgmConfig::new(&ws, "break things"),
            1,
        );
        eng.run(4).unwrap();
        assert_eq!(eng.archive().len(), 1, "nothing should be accepted");
        assert!(eng.history().iter().all(|o| !o.accepted()));
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn run_is_seed_deterministic() {
        let ws1 = toy_workspace("det1");
        let ws2 = toy_workspace("det2");
        let mut a = engine(&ws1);
        let mut b = engine(&ws2);
        a.run(5).unwrap();
        b.run(5).unwrap();
        assert_eq!(a.archive().len(), b.archive().len());
        assert_eq!(
            a.best().unwrap().fitness.as_ref().unwrap().score,
            b.best().unwrap().fitness.as_ref().unwrap().score
        );
        // IDs déterministes : la meilleure variante a la même identité.
        assert_eq!(a.best().unwrap().id, b.best().unwrap().id);
        let _ = std::fs::remove_dir_all(&ws1);
        let _ = std::fs::remove_dir_all(&ws2);
    }
}
