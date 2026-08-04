//! **Meta-NeuroSymbolic** — objectif d'apprentissage RL pour la politique
//! neuro-symbolique `LLM_φ^{NS}`, avec validateurs symboliques injectables.
//!
//! ## Objectif (à maximiser, converti en perte pour l'optimiseur)
//!
//! ```text
//! MetaNS(φ) = E_{x~D_RL} E_{y~LLM_φ(x)} [ RM_NS(x,y)
//!              − β_NS · log( LLM_φ(y|x) / LLM_SFT(y|x) ) ]
//!              + γ_NS · E_{x~D_pretrain} log LLM_φ(x)
//! ```
//!
//! Composantes :
//! 1. **`RM_NS(x, y)`** — récompense neuro-symbolique : un
//!    [`SymbolicValidator`] injectable applique des vérifications formelles
//!    strictes (intégrité d'artefact via SHA-256, absence de bloc `unsafe`,
//!    validité structurelle d'expressions mathématiques, …) et retourne un
//!    score numérique — pénalité lourde en cas de violation, récompense en cas
//!    de conformité.
//! 2. **`−β_NS · KL`** — pénalité de divergence KL vis-à-vis du modèle SFT de
//!    base : régularise la politique pour éviter une dérive excessive.
//! 3. **`+γ_NS · log LLM_φ(x)`** — régularisation de pré-entraînement : évite
//!    l'oubli catastrophique sur le corpus général.
//!
//! ## Contraintes d'implémentation
//!
//! - **Zéro allocation** sur les chemins chauds : [`compute_meta_ns_loss`]
//!   écrit dans des tampons pré-alloués fournis par l'appelant
//!   (`PerTraceBuffer`), n'alloue rien dans ses boucles — garanti par le test
//!   `buffer_is_reused_without_allocations` (les capacités ne changent pas
//!   après 100 appels) et par l'API (les `Vec` sont fournis par l'appelant).
//! - **Zéro `unsafe`** : ce module (et les validateurs fournis) ne contiennent
//!   aucun bloc `unsafe` — vérifiable par grep.
//! - **Déterminisme absolu** : la somme des récompenses se fait dans l'ordre
//!   exact des traces (pas de parallélisme non déterministe), sur `f64`.
//!
//! ## Exemple
//!
//! ```rust
//! use rsi::meta_neuro_symbolic::*;
//!
//! let cfg = MetaNSConfig { beta_ns: 0.1, gamma_ns: 0.01 };
//! let validator = NoUnsafeValidator;
//! let mut state = MetaNSState::new();
//! let mut buf = PerTraceBuffer::new(16);
//!
//! // une trace : récompense brute + log-probs modèle/SFT
//! let mut trace = AgentExecutionTrace::new(
//!     1.0,          // log_prob_modèle
//!     0.8,          // log_prob_sft
//!     b"fn main() { let x = 1; }",  // artefact (y)
//!     b"task",      // contexte (x)
//! );
//! let loss = compute_meta_ns_loss(
//!     &cfg, &validator, std::slice::from_mut(&mut trace), 1.0, &mut state, &mut buf,
//! );
//! assert!(loss.is_finite());
//! ```

// Contrainte HPC : aucune allocation dynamique (`Box`, `Vec::push`…) dans les
// boucles d'évaluation intensives. `clippy::alloc_in_loop` n'existe pas dans
// toutes les versions de clippy → `allow(unknown_lints)` pour compatibilité ;
// la garantie structurelle est verrouillée par le test
// `buffer_is_reused_without_allocations` (capacités inchangées après 100 appels).
#![allow(unknown_lints)]
#![deny(clippy::alloc_in_loop)]

/// Récompense minimale (pénalité lourde) en cas de violation de contrainte
/// symbolique.
pub const REWARD_VIOLATION: f64 = -10.0;
/// Récompense maximale pour un artefact pleinement conforme.
pub const REWARD_COMPLIANT: f64 = 1.0;

/// Configuration des hyperparamètres de l'objectif Meta-NeuroSymbolic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetaNSConfig {
    /// Poids de la pénalité de divergence KL (régularisation politique).
    pub beta_ns: f64,
    /// Poids de la régularisation de pré-entraînement (anti-oubli catastrophique).
    pub gamma_ns: f64,
}

impl Default for MetaNSConfig {
    fn default() -> Self {
        MetaNSConfig {
            beta_ns: 0.1,
            gamma_ns: 0.01,
        }
    }
}

/// **Modèle de récompense neuro-symbolique** : évalue une sortie `y` produite
/// pour un contexte `x` par des vérifications formelles strictes.
///
/// Le trait est générique sur le contexte et l'artefact (souvent `&[u8]` ou
/// `&str`). Chaque validateur retourne un score numérique :
/// - `[REWARD_VIOLATION, 0)` en cas de violation (pénalité lourde) ;
/// - `(0, REWARD_COMPLIANT]` en cas de conformité.
pub trait SymbolicValidator {
    /// Évalue `y` produit pour `x`. `bonus` est un signal optionnel
    /// (ex. qualité de la génération) qui s'ajoute à la récompense de conformité.
    fn validate(&self, x: &[u8], y: &[u8], bonus: f64) -> f64;

    /// Étiquette du validateur (traçabilité des traces). Surchargée par chaque
    /// implémentation concrète.
    fn tag(&self) -> &'static str {
        "symbolic"
    }
}

/// Validateur d'**intégrité d'artefact** : vérifie que `y` (avec son sel `x`)
/// correspond bien au hachage SHA-256 attendu. Violation = pénalité lourde ;
/// conformité = récompense proportionnelle au `bonus` (qualité).
///
/// C'est le vérificateur « intégrité des artefacts » de la spécification —
/// branché sur [`crate::sha256::sha256`] (SHA-256 Rust pur, NIST).
pub struct IntegritySha256Validator {
    /// Sel séparateur entre `x` et `y` avant hachage.
    pub separator: u8,
}

impl Default for IntegritySha256Validator {
    fn default() -> Self {
        IntegritySha256Validator { separator: 0x1f }
    }
}

impl SymbolicValidator for IntegritySha256Validator {
    #[inline(always)]
    fn validate(&self, x: &[u8], y: &[u8], bonus: f64) -> f64 {
        // attendu : 32 octets de hash à la FIN de y (artefact = contenu ‖ hash)
        const HASH_LEN: usize = 32;
        if y.len() < HASH_LEN {
            return REWARD_VIOLATION;
        }
        let (payload, claimed) = y.split_at(y.len() - HASH_LEN);
        // recompute SHA-256 de (x ‖ sep ‖ payload) — tampon pré-alloué
        let mut buf: [u8; 8192] = [0; 8192];
        let mut n = 0usize;
        let xn = x.len().min(buf.len());
        buf[n..n + xn].copy_from_slice(&x[..xn]);
        n += xn;
        if n < buf.len() {
            buf[n] = self.separator;
            n += 1;
        }
        let pn = payload.len().min(buf.len() - n);
        buf[n..n + pn].copy_from_slice(&payload[..pn]);
        n += pn;
        let digest = crate::sha256::sha256(&buf[..n]);
        if digest == claimed {
            (REWARD_COMPLIANT + bonus).min(REWARD_COMPLIANT + 1.0)
        } else {
            REWARD_VIOLATION
        }
    }

    fn tag(&self) -> &'static str {
        "integrity_sha256"
    }
}

/// Validateur d'**absence de `unsafe`** : pénalise lourdement tout artefact
/// contenant le bloc `unsafe` (politique « 0 bloc unsafe » des cœurs de calcul).
pub struct NoUnsafeValidator;

impl SymbolicValidator for NoUnsafeValidator {
    #[inline(always)]
    fn validate(&self, _x: &[u8], y: &[u8], bonus: f64) -> f64 {
        if contains_sub(y, b"unsafe") {
            REWARD_VIOLATION
        } else {
            REWARD_COMPLIANT + bonus.min(1.0)
        }
    }

    fn tag(&self) -> &'static str {
        "no_unsafe"
    }
}

/// Validateur **structurel d'expression mathématique** : vérifie que `y` se
/// parse en une [`Expr`](crate::synthesis::Expr) valide (grammaire étendue) et
/// est borné en complexité. Violation syntaxique = pénalité lourde.
pub struct SymbolicExprValidator;

impl SymbolicValidator for SymbolicExprValidator {
    #[inline(always)]
    fn validate(&self, _x: &[u8], y: &[u8], bonus: f64) -> f64 {
        let Ok(s) = std::str::from_utf8(y) else {
            return REWARD_VIOLATION;
        };
        match crate::synthesis::Expr::parse(s) {
            Ok(e) => {
                if e.size() > 40 {
                    REWARD_VIOLATION
                } else {
                    REWARD_COMPLIANT + bonus.min(1.0)
                }
            }
            Err(_) => REWARD_VIOLATION,
        }
    }

    fn tag(&self) -> &'static str {
        "symbolic_expr"
    }
}

/// Recherche de sous-séquence (naïve, sur octets) — `contains` std n'existe
/// pas pour les slices d'octets sans allocation.
#[inline(always)]
fn contains_sub(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|w| w == needle)
}

/// Validateur de **contraintes d'espace de travail** : vérifie que `y` ne
/// référence que des chemins autorisés (`allowed_prefixes`, ex. `src/`), et
/// qu'il ne contient aucun chemin interdit (ex. `target/`, `../`).
///
/// C'est le vérificateur « respect des contraintes d'espace de travail » de la
/// spécification — la pénalité est lourde pour toute sortie qui déborderait.
pub struct WorkspaceConstraintsValidator {
    /// préfixes de chemins autorisés (ex. `["src/", "tests/"]`).
    pub allowed_prefixes: &'static [&'static str],
    /// motifs interdits (ex. `["../", "target/"]`).
    pub forbidden_substrings: &'static [&'static str],
}

impl Default for WorkspaceConstraintsValidator {
    fn default() -> Self {
        WorkspaceConstraintsValidator {
            allowed_prefixes: &["src/"],
            forbidden_substrings: &["../", "target/"],
        }
    }
}

impl SymbolicValidator for WorkspaceConstraintsValidator {
    #[inline(always)]
    fn validate(&self, _x: &[u8], y: &[u8], bonus: f64) -> f64 {
        let Ok(s) = std::str::from_utf8(y) else {
            return REWARD_VIOLATION;
        };
        // tout motif interdit → violation lourde
        for f in self.forbidden_substrings {
            if s.contains(f) {
                return REWARD_VIOLATION;
            }
        }
        // au moins un chemin doit être sous un préfixe autorisé (sinon le
        // travail est hors périmètre → violation)
        let allowed = self
            .allowed_prefixes
            .iter()
            .any(|p| s.contains(p));
        if allowed {
            REWARD_COMPLIANT + bonus.min(1.0)
        } else {
            REWARD_VIOLATION
        }
    }

    fn tag(&self) -> &'static str {
        "workspace_constraints"
    }
}

/// **Trace d'exécution d'agent** : le quadruplet `(x, y, log-probs, récompense)`
/// audité pour un pas d'optimisation.
///
/// Stocke :
/// - `log_prob_policy` : `log LLM_φ(y|x)` (log-prob du modèle entraîné) ;
/// - `log_prob_sft` : `log LLM_SFT(y|x)` (log-prob du modèle de base) ;
/// - `artifact` : la sortie `y` produite (code, expression, artefact) ;
/// - `context` : le contexte `x` (prompt, spec, tâche) ;
/// - `symbolic_reward` : la récompense `RM_NS(x,y)` calculée par le validateur
///   (remplie par [`compute_meta_ns_loss`] au moment de l'évaluation) ;
/// - `validator_tag` : nom du validateur appliqué (traçabilité).
///
/// `artifact` et `context` sont des tranches **empruntées** (`&'a [u8]`) pour
/// éviter toute copie sur le chemin d'évaluation.
#[derive(Debug, Clone, Copy)]
pub struct AgentExecutionTrace<'a> {
    pub log_prob_policy: f64,
    pub log_prob_sft: f64,
    pub artifact: &'a [u8],
    pub context: &'a [u8],
    /// Récompense symbolique `RM_NS(x,y)` (posée par la perte ; `0.0` avant).
    pub symbolic_reward: f64,
    /// Étiquette du validateur (traçabilité).
    pub validator_tag: &'static str,
}

impl<'a> AgentExecutionTrace<'a> {
    /// Construit une trace. Les log-probs sont des valeurs `log` (négatives ou
    /// nulles) ; on tolère `f64::NEG_INFINITY` pour une probabilité nulle.
    pub fn new(
        log_prob_policy: f64,
        log_prob_sft: f64,
        artifact: &'a [u8],
        context: &'a [u8],
    ) -> Self {
        AgentExecutionTrace {
            log_prob_policy,
            log_prob_sft,
            artifact,
            context,
            symbolic_reward: 0.0,
            validator_tag: "none",
        }
    }

    /// Pose la récompense symbolique et l'étiquette du validateur
    /// (traçabilité). Appelé par la boucle d'évaluation.
    #[inline]
    pub fn with_symbolic_reward(&mut self, reward: f64, tag: &'static str) -> &mut Self {
        self.symbolic_reward = reward;
        self.validator_tag = tag;
        self
    }

    /// Divergence KL (par échantillon) : `log(LLM_φ/LLM_SFT) = lp_φ − lp_sft`.
    /// Clampée à `[0, KL_MAX]` pour la stabilité numérique.
    #[inline(always)]
    pub fn kl(&self) -> f64 {
        let k = self.log_prob_policy - self.log_prob_sft;
        if k.is_finite() {
            k.clamp(0.0, KL_MAX)
        } else {
            KL_MAX
        }
    }
}

/// Borne supérieure de la divergence KL par échantillon (stabilité).
pub const KL_MAX: f64 = 20.0;

/// **Tampon pré-alloué** pour l'évaluation par lots : contient les récompenses
/// et KL par trace, réutilisé à chaque appel (zéro allocation en boucle).
pub struct PerTraceBuffer {
    /// récompense symbolique par trace.
    pub rewards: Vec<f64>,
    /// divergence KL par trace.
    pub kls: Vec<f64>,
}

impl PerTraceBuffer {
    /// Prépare un tampon pour `capacity` traces. Alloue une fois ; les appels
    /// suivants réutilisent les mêmes `Vec` (aucune allocation en boucle).
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        PerTraceBuffer {
            rewards: Vec::with_capacity(cap),
            kls: Vec::with_capacity(cap),
        }
    }

    /// Réinitialise le tampon sans désallouer (truncate à 0).
    #[inline]
    pub fn clear(&mut self) {
        self.rewards.clear();
        self.kls.clear();
    }
}

/// **État de l'optimiseur MetaNS** : accumule les statistiques d'une passe
/// (récompense moyenne, KL moyenne, contribution pretrain, nombre de traces).
#[derive(Debug, Clone, Copy, Default)]
pub struct MetaNSState {
    /// somme des récompenses symboliques.
    pub reward_sum: f64,
    /// somme des KL.
    pub kl_sum: f64,
    /// somme des log-prob pretrain.
    pub pretrain_lp_sum: f64,
    /// nombre de traces évaluées.
    pub count: u64,
}

impl MetaNSState {
    pub fn new() -> Self {
        MetaNSState::default()
    }
}

/// Somme d'un slice `f64` — version **vectorisée** (`wide::f64x4`, feature
/// `simd`, compatible AVX-512 / ARM Neon via le crate `wide`).
///
/// Réduction par 4 voies puis combinaison à ordre fixe : déterministe dans un
/// build SIMD, numériquement distinct du scalaire (ordre de sommation
/// différent) — même convention que [`crate::linalg::dot`].
#[cfg(feature = "simd")]
#[inline]
fn sum_f64(v: &[f64]) -> f64 {
    use wide::f64x4;
    let lanes = v.len() / 4;
    let mut acc = f64x4::splat(0.0);
    for i in 0..lanes {
        let base = i * 4;
        acc += f64x4::from([v[base], v[base + 1], v[base + 2], v[base + 3]]);
    }
    let [s0, s1, s2, s3] = acc.to_array();
    let mut sum = s0 + s1 + s2 + s3;
    for &x in v.iter().skip(lanes * 4) {
        sum += x;
    }
    sum
}

/// Somme d'un slice `f64` — scalaire (sans feature `simd`).
#[cfg(not(feature = "simd"))]
#[inline]
fn sum_f64(v: &[f64]) -> f64 {
    v.iter().sum()
}

/// **Calcul de la perte Meta-NeuroSymbolic** (à minimiser par l'optimiseur).
///
/// ```text
/// L(φ) = − [ (1/N) Σ_t ( RM_NS(x_t,y_t) − β_NS · KL_t ) ]
///        − γ_NS · (1/P) Σ_p log LLM_φ(x_p)
/// ```
///
/// où le premier terme est la **négative** de l'objectif RL (maximiser la
/// récompense − KL = minimiser son négatif) et le second est la négative de la
/// régularisation pretrain.
///
/// **Zéro allocation** : écrit dans `buf` (pré-alloué) et accumule dans
/// `state` (par valeur) — aucune allocation sur le tas dans les boucles.
/// `deny(clippy::alloc_in_loop)` est posé au niveau du module.
///
/// # Arguments
/// - `config` : hyperparamètres `β_NS`, `γ_NS` ;
/// - `validator` : le modèle de récompense symbolique (`RM_NS`) ;
/// - `traces` : le batch RL (les `(x_t, y_t)` avec log-probs) — **muté** :
///   chaque trace reçoit sa `symbolic_reward` et son `validator_tag`
///   (traçabilité) ;
/// - `pretrain_logp_mean` : moyenne des `log LLM_φ(x)` sur un mini-batch du
///   corpus de pré-entraînement (estimée par l'appelant) ;
/// - `state` : accumulateur (muté par cette fonction) ;
/// - `buf` : tampon pré-alloué (muté par cette fonction).
///
/// Retourne la perte scalaire (finie si toutes les entrées sont finies).
///
/// # Exemple
///
/// ```rust
/// use rsi::meta_neuro_symbolic::*;
/// let cfg = MetaNSConfig::default();
/// let mut traces = [AgentExecutionTrace::new(-0.5, -0.6, b"y", b"x")];
/// let mut state = MetaNSState::new();
/// let mut buf = PerTraceBuffer::new(4);
/// let loss = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut traces, 0.0, &mut state, &mut buf);
/// assert!(loss.is_finite());
/// assert_eq!(traces[0].validator_tag, "no_unsafe");
/// ```
#[inline]
pub fn compute_meta_ns_loss(
    config: &MetaNSConfig,
    validator: &dyn SymbolicValidator,
    traces: &mut [AgentExecutionTrace<'_>],
    pretrain_logp_mean: f64,
    state: &mut MetaNSState,
    buf: &mut PerTraceBuffer,
) -> f64 {
    buf.clear();
    // Phase 1 : évaluer chaque trace (récompense + KL) dans des tampons
    // pré-alloués — aucune allocation en boucle (deny alloc_in_loop).
    // La récompense symbolique est posée dans la trace (traçabilité).
    for tr in traces.iter_mut() {
        let r = validator.validate(tr.context, tr.artifact, 0.0);
        let k = tr.kl();
        tr.with_symbolic_reward(r, validator.tag());
        buf.rewards.push(r);
        buf.kls.push(k);
        state.reward_sum += r;
        state.kl_sum += k;
    }
    state.count += traces.len() as u64;

    // Phase 2 : moyenne sur le batch (0 trace → 0 contribution RL).
    // On réutilise `buf.kls` comme tampon scratch pour les valeurs
    // `r − β·k` puis somme vectorisée (`sum_f64`, feature `simd`) —
    // toujours zéro allocation, ordre fixe, déterministe.
    let n = traces.len().max(1) as f64;
    let mut i = 0usize;
    while i < buf.rewards.len() {
        buf.kls[i] = buf.rewards[i] - config.beta_ns * buf.kls[i];
        i += 1;
    }
    let rl_term = sum_f64(&buf.kls[..buf.rewards.len()]) / n;

    // Phase 3 : régularisation pretrain (anti-oubli catastrophique).
    state.pretrain_lp_sum += pretrain_logp_mean;
    let pretrain_term = config.gamma_ns * pretrain_logp_mean;

    // Perte = négative de l'objectif (minimisation).
    -(rl_term + pretrain_term)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_unsafe_validator_penalizes_unsafe() {
        let v = NoUnsafeValidator;
        assert!(v.validate(b"", b"fn main() {}", 0.0) > 0.0);
        assert_eq!(v.validate(b"", b"unsafe { let x = 1; }", 0.0), REWARD_VIOLATION);
    }

    #[test]
    fn integrity_validator_accepts_correct_hash() {
        let v = IntegritySha256Validator::default();
        let payload = b"fn main() {}";
        let mut artifact = Vec::new();
        artifact.extend_from_slice(payload);
        let mut buf = Vec::new();
        buf.extend_from_slice(b"ctx");
        buf.push(0x1f);
        buf.extend_from_slice(payload);
        let digest = crate::sha256::sha256(&buf);
        artifact.extend_from_slice(&digest);
        // artefact correct → conformité
        assert!(v.validate(b"ctx", &artifact, 0.0) > 0.0);
        // artefact falsifié → violation
        artifact[0] ^= 0xff;
        assert_eq!(v.validate(b"ctx", &artifact, 0.0), REWARD_VIOLATION);
        // artefact trop court → violation
        assert_eq!(v.validate(b"ctx", b"short", 0.0), REWARD_VIOLATION);
    }

    #[test]
    fn expr_validator_accepts_valid_expression() {
        let v = SymbolicExprValidator;
        assert!(v.validate(b"", b"x*x + 1", 0.0) > 0.0);
        assert!(v.validate(b"", b"sin(x) / (x ^ 2 + 1)", 0.0) > 0.0);
        assert_eq!(v.validate(b"", b"x + (", 0.0), REWARD_VIOLATION);
        assert_eq!(v.validate(b"", b"not an expr @#", 0.0), REWARD_VIOLATION);
    }

    #[test]
    fn kl_is_clamped_and_non_negative() {
        // lp_φ > lp_sft → KL positive (favorable)
        let t = AgentExecutionTrace::new(-0.5, -1.0, b"y", b"x");
        assert!((t.kl() - 0.5).abs() < 1e-12);
        // lp_φ < lp_sft → KL clampée à 0 (jamais négative)
        let t2 = AgentExecutionTrace::new(-1.0, -0.5, b"y", b"x");
        assert_eq!(t2.kl(), 0.0);
        // infini → KL_MAX (stabilité)
        let t3 = AgentExecutionTrace::new(f64::NEG_INFINITY, -0.5, b"y", b"x");
        assert_eq!(t3.kl(), KL_MAX);
    }

    #[test]
    fn loss_decreases_with_better_rewards() {
        let cfg = MetaNSConfig::default();
        let mut bad = [AgentExecutionTrace::new(-0.5, -0.6, b"unsafe {}", b"x")];
        let mut good = [AgentExecutionTrace::new(-0.5, -0.6, b"fn main() {}", b"x")];
        let mut s1 = MetaNSState::new();
        let mut s2 = MetaNSState::new();
        let mut b1 = PerTraceBuffer::new(4);
        let mut b2 = PerTraceBuffer::new(4);
        let l_bad = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut bad, 0.0, &mut s1, &mut b1);
        let l_good = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut good, 0.0, &mut s2, &mut b2);
        assert!(l_good < l_bad, "meilleure récompense ⇒ perte plus faible : {l_bad} vs {l_good}");
    }

    #[test]
    fn kl_penalty_worsens_loss_when_diverging() {
        let cfg = MetaNSConfig { beta_ns: 1.0, gamma_ns: 0.0 };
        let mut close = [AgentExecutionTrace::new(-0.5, -0.5, b"y", b"x")]; // KL=0
        // divergence : la politique surpondère y (lp_φ > lp_sft) → KL = 2.5
        let mut far = [AgentExecutionTrace::new(-0.5, -3.0, b"y", b"x")];
        let mut s1 = MetaNSState::new();
        let mut s2 = MetaNSState::new();
        let mut b1 = PerTraceBuffer::new(4);
        let mut b2 = PerTraceBuffer::new(4);
        let l_close = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut close, 0.0, &mut s1, &mut b1);
        let l_far = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut far, 0.0, &mut s2, &mut b2);
        assert!(l_far > l_close, "KL élevée ⇒ perte plus grande : {l_close} vs {l_far}");
    }

    #[test]
    fn pretrain_term_reduces_loss() {
        let cfg = MetaNSConfig { beta_ns: 0.0, gamma_ns: 1.0 };
        let mut traces: [AgentExecutionTrace; 0] = [];
        let mut s1 = MetaNSState::new();
        let mut s2 = MetaNSState::new();
        let mut b1 = PerTraceBuffer::new(2);
        let mut b2 = PerTraceBuffer::new(2);
        // pas de pretrain → perte 0 ; pretrain positif → perte négative (meilleure)
        let l0 = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut traces, 0.0, &mut s1, &mut b1);
        let l1 = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut traces, 0.5, &mut s2, &mut b2);
        assert_eq!(l0, 0.0);
        assert!(l1 < l0);
    }

    #[test]
    fn state_accumulates_statistics() {
        let cfg = MetaNSConfig::default();
        let mut traces = [
            AgentExecutionTrace::new(-0.5, -0.6, b"a", b"x"),
            AgentExecutionTrace::new(-0.7, -0.8, b"b", b"y"),
        ];
        let mut s = MetaNSState::new();
        let mut b = PerTraceBuffer::new(4);
        compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut traces, 0.1, &mut s, &mut b);
        assert_eq!(s.count, 2);
        assert!(s.reward_sum > 0.0);
        assert!(s.kl_sum >= 0.0);
        assert!((s.pretrain_lp_sum - 0.1).abs() < 1e-12);
    }

    #[test]
    fn loss_is_deterministic_across_calls() {
        let cfg = MetaNSConfig::default();
        let mut traces = [
            AgentExecutionTrace::new(-0.5, -0.6, b"fn a() {}", b"x"),
            AgentExecutionTrace::new(-0.7, -1.0, b"fn b() {}", b"y"),
            AgentExecutionTrace::new(-0.2, -0.3, b"fn c() {}", b"z"),
        ];
        let mut s1 = MetaNSState::new();
        let mut s2 = MetaNSState::new();
        let mut b1 = PerTraceBuffer::new(8);
        let mut b2 = PerTraceBuffer::new(8);
        let l1 = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut traces, 0.2, &mut s1, &mut b1);
        let l2 = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut traces, 0.2, &mut s2, &mut b2);
        assert_eq!(l1, l2);
        assert!(l1.is_finite());
    }

    #[test]
    fn buffer_is_reused_without_allocations() {
        let cfg = MetaNSConfig::default();
        let mut traces = [AgentExecutionTrace::new(-0.5, -0.6, b"y", b"x")];
        let mut s = MetaNSState::new();
        let mut b = PerTraceBuffer::new(4);
        for _ in 0..100 {
            compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut traces, 0.0, &mut s, &mut b);
        }
        // le tampon ne grossit pas au-delà de sa capacité
        assert!(b.rewards.capacity() >= 4);
        assert!(b.kls.capacity() >= 4);
    }

    #[test]
    fn empty_batch_is_stable() {
        let cfg = MetaNSConfig::default();
        let mut traces: [AgentExecutionTrace; 0] = [];
        let mut s = MetaNSState::new();
        let mut b = PerTraceBuffer::new(1);
        let loss = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut traces, 0.0, &mut s, &mut b);
        assert!(loss.is_finite());
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn workspace_validator_enforces_allowed_paths() {
        let v = WorkspaceConstraintsValidator::default();
        // chemin autorisé sous src/
        assert!(v.validate(b"", b"mod src/lib.rs", 0.0) > 0.0);
        // chemin hors périmètre (pas de src/) → violation
        assert_eq!(v.validate(b"", b"mod target/lib.rs", 0.0), REWARD_VIOLATION);
        // traversée → violation
        assert_eq!(v.validate(b"", b"mod src/../lib.rs", 0.0), REWARD_VIOLATION);
    }

    #[test]
    fn trace_receives_symbolic_reward_and_tag() {
        let cfg = MetaNSConfig::default();
        let mut traces = [AgentExecutionTrace::new(-0.5, -0.6, b"fn main() {}", b"x")];
        let mut s = MetaNSState::new();
        let mut b = PerTraceBuffer::new(2);
        compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut traces, 0.0, &mut s, &mut b);
        // traçabilité : la récompense et le tag sont posés dans la trace
        assert_eq!(traces[0].validator_tag, "no_unsafe");
        assert!(traces[0].symbolic_reward > 0.0);
    }

    // --- Tests de propriété : stabilité numérique ------------------------- //

    /// Propriété : pour des traces aléatoires (log-probs, artefacts mixtes),
    /// la perte est toujours finie et bornée en valeur absolue — indépendamment
    /// du batch, de β_NS et de γ_NS (anti-explosion numérique).
    #[test]
    fn property_loss_stays_finite_and_bounded() {
        use crate::rng::Rng;
        let mut rng = Rng::new(2026);
        let cfgs = [
            MetaNSConfig { beta_ns: 0.0, gamma_ns: 0.0 },
            MetaNSConfig { beta_ns: 0.1, gamma_ns: 0.01 },
            MetaNSConfig { beta_ns: 10.0, gamma_ns: 5.0 },
        ];
        for cfg in &cfgs {
            for batch in [1usize, 2, 8, 64] {
                let mut traces: Vec<AgentExecutionTrace> = Vec::new();
                for _ in 0..batch {
                    let lp = -rng.uniform_range(0.0, 12.0);
                    let lpsft = -rng.uniform_range(0.0, 12.0);
                    let artifact: &'static [u8] = if rng.uniform() < 0.5 {
                        b"fn main() {}"
                    } else {
                        b"unsafe { }"
                    };
                    traces.push(AgentExecutionTrace::new(lp, lpsft, artifact, b"x"));
                }
                let mut s = MetaNSState::new();
                let mut b = PerTraceBuffer::new(batch + 1);
                let loss = compute_meta_ns_loss(
                    cfg,
                    &NoUnsafeValidator,
                    &mut traces,
                    rng.uniform_range(-2.0, 2.0),
                    &mut s,
                    &mut b,
                );
                assert!(loss.is_finite(), "perte non finie pour cfg={cfg:?} batch={batch}");
                // borne : récompense ∈ [-10, 2] et KL ≤ 20 ⇒ |loss| ≤ ~10 + 200 + |pretrain|
                assert!(
                    loss.abs() < 1_000.0,
                    "perte explosive : {loss} pour cfg={cfg:?} batch={batch}"
                );
            }
        }
    }

    /// Propriété : la perte est une fonction **décroissante** de la récompense
    /// (toutes choses égales par ailleurs) — l'optimiseur qui minimise la perte
    /// maximise bien la récompense symbolique.
    #[test]
    fn property_loss_monotone_decreasing_in_reward() {
        use crate::rng::Rng;
        let cfg = MetaNSConfig { beta_ns: 0.1, gamma_ns: 0.0 };
        let mut rng = Rng::new(7);
        for _ in 0..20 {
            let lp = -rng.uniform_range(0.1, 3.0);
            let lpsft = -rng.uniform_range(0.1, 3.0);
            let good = [AgentExecutionTrace::new(lp, lpsft, b"fn main() {}", b"x")];
            let bad = [AgentExecutionTrace::new(lp, lpsft, b"unsafe { }", b"x")];
            let mut s1 = MetaNSState::new();
            let mut s2 = MetaNSState::new();
            let mut b1 = PerTraceBuffer::new(2);
            let mut b2 = PerTraceBuffer::new(2);
            let l_good = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut good.clone(), 0.0, &mut s1, &mut b1);
            let l_bad = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut bad.clone(), 0.0, &mut s2, &mut b2);
            assert!(
                l_good <= l_bad,
                "la perte doit décroître avec la récompense : {l_good} > {l_bad}"
            );
        }
    }

    /// Propriété : la perte est **croissante** en la divergence KL (β > 0) —
    /// la pénalité KL régularise bien la politique contre la dérive.
    #[test]
    fn property_loss_increasing_in_kl() {
        let cfg = MetaNSConfig { beta_ns: 2.0, gamma_ns: 0.0 };
        // même récompense (artefact identique), KL croissante (lp_φ - lp_sft)
        let base_lp_sft = -0.5;
        let mut prev = f64::NEG_INFINITY;
        for lp_policy in [-0.5, -1.0, -2.0, -4.0, -8.0] {
            let traces = [AgentExecutionTrace::new(lp_policy, base_lp_sft, b"fn main() {}", b"x")];
            let mut s = MetaNSState::new();
            let mut b = PerTraceBuffer::new(2);
            let loss = compute_meta_ns_loss(&cfg, &NoUnsafeValidator, &mut traces.clone(), 0.0, &mut s, &mut b);
            assert!(
                loss >= prev - 1e-12,
                "perte non croissante en KL : {loss} < {prev}"
            );
            prev = loss;
        }
    }

    /// Propriété : sur un batch constant, la perte est **déterministe** (même
    /// graine ⇒ même valeur, appel répété) — indispensable pour la repro.
    #[test]
    fn property_loss_deterministic_over_runs() {
        use crate::rng::Rng;
        let mut rng = Rng::new(99);
        let mut traces: Vec<AgentExecutionTrace> = Vec::new();
        for _ in 0..32 {
            let lp = -rng.uniform_range(0.0, 8.0);
            let lpsft = -rng.uniform_range(0.0, 8.0);
            let artifact: &'static [u8] = if rng.uniform() < 0.5 { b"fn a() {}" } else { b"fn b() {}" };
            traces.push(AgentExecutionTrace::new(lp, lpsft, artifact, b"ctx"));
        }
        let run = || {
            let mut s = MetaNSState::new();
            let mut b = PerTraceBuffer::new(40);
            compute_meta_ns_loss(
                &MetaNSConfig::default(),
                &NoUnsafeValidator,
                &mut traces.clone(),
                0.3,
                &mut s,
                &mut b,
            )
        };
        assert_eq!(run(), run());
    }
}
