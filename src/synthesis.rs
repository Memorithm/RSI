//! Domaine de **synthèse symbolique** : l'agent génère des *expressions*
//! candidates puis les améliore via [`crate::ascent`] (contrat scirust-rsi).
//!
//! ## Sandbox (garde-fou clé)
//! Le « code » candidat est un **AST arithmétique** que J'ÉVALUE dans mon
//! propre interpréteur ([`Expr::eval`]) — **jamais** compilé ni exécuté comme
//! du code arbitraire, aucun sous-processus. C'est le sandbox que contrôle RSI :
//! l'évaluateur lit une fitness, le moteur d'ascension ne voit que des nombres.
//!
//! - ÉVALUATEUR (`score`) : fraction de cas de test réussis (|sortie − cible| ≤
//!   tolérance) **moins** une pénalité de complexité (taille de l'AST).
//! - GÉNÉRATEUR (`refine`) : produit la meilleure de `lambda` mutations
//!   déterministes (révision « critiquée », façon 1+λ).
//!
//! Entièrement déterministe (graine) et borné.

use crate::ascent::RefineTask;
use crate::llm::{LlmRefineTask, SafetyViolation};
use crate::rng::Rng;

/// Taille maximale d'AST acceptée par le chemin LLM (garde-fou de sûreté :
/// borne la complexité des candidats proposés par un modèle).
const MAX_EXPR_SIZE: usize = 25;

/// Expression arithmétique sur une variable `x` (AST candidat).
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    X,
    Const(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
}

impl Expr {
    /// ÉVALUATION en sandbox (interpréteur maison ; aucune exécution externe).
    pub fn eval(&self, x: f64) -> f64 {
        match self {
            Expr::X => x,
            Expr::Const(c) => *c,
            Expr::Add(a, b) => a.eval(x) + b.eval(x),
            Expr::Sub(a, b) => a.eval(x) - b.eval(x),
            Expr::Mul(a, b) => a.eval(x) * b.eval(x),
            Expr::Neg(a) => -a.eval(x),
        }
    }

    /// Nombre de nœuds (complexité).
    pub fn size(&self) -> usize {
        match self {
            Expr::X | Expr::Const(_) => 1,
            Expr::Neg(a) => 1 + a.size(),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => 1 + a.size() + b.size(),
        }
    }

    /// Représentation lisible (pour le log).
    pub fn pretty(&self) -> String {
        match self {
            Expr::X => "x".into(),
            Expr::Const(c) => format!("{c:.3}"),
            Expr::Add(a, b) => format!("({} + {})", a.pretty(), b.pretty()),
            Expr::Sub(a, b) => format!("({} - {})", a.pretty(), b.pretty()),
            Expr::Mul(a, b) => format!("({} * {})", a.pretty(), b.pretty()),
            Expr::Neg(a) => format!("(-{})", a.pretty()),
        }
    }

    /// Renvoie une copie du sous-arbre d'indice `idx` (préordre).
    fn subtree_at(&self, idx: usize, cur: &mut usize) -> Option<Expr> {
        let here = *cur;
        *cur += 1;
        if here == idx {
            return Some(self.clone());
        }
        match self {
            Expr::X | Expr::Const(_) => None,
            Expr::Neg(a) => a.subtree_at(idx, cur),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
                a.subtree_at(idx, cur).or_else(|| b.subtree_at(idx, cur))
            }
        }
    }

    /// Remplace le nœud d'indice `idx` (préordre) par `repl`.
    fn replace_at(&self, idx: usize, repl: &Expr, cur: &mut usize) -> Expr {
        let here = *cur;
        *cur += 1;
        if here == idx {
            return repl.clone();
        }
        match self {
            Expr::X | Expr::Const(_) => self.clone(),
            Expr::Neg(a) => Expr::Neg(Box::new(a.replace_at(idx, repl, cur))),
            Expr::Add(a, b) => Expr::Add(
                Box::new(a.replace_at(idx, repl, cur)),
                Box::new(b.replace_at(idx, repl, cur)),
            ),
            Expr::Sub(a, b) => Expr::Sub(
                Box::new(a.replace_at(idx, repl, cur)),
                Box::new(b.replace_at(idx, repl, cur)),
            ),
            Expr::Mul(a, b) => Expr::Mul(
                Box::new(a.replace_at(idx, repl, cur)),
                Box::new(b.replace_at(idx, repl, cur)),
            ),
        }
    }
}

fn random_terminal(rng: &mut Rng) -> Expr {
    if rng.uniform() < 0.5 {
        Expr::X
    } else {
        // constante dans [-2, 2], arrondie au quart (favorise les entiers simples)
        let c = (rng.uniform_range(-2.0, 2.0) * 4.0).round() / 4.0;
        Expr::Const(c)
    }
}

fn random_expr(rng: &mut Rng, depth: usize) -> Expr {
    if depth == 0 || rng.uniform() < 0.3 {
        return random_terminal(rng);
    }
    let a = Box::new(random_expr(rng, depth - 1));
    let b = Box::new(random_expr(rng, depth - 1));
    match (rng.uniform() * 3.0) as u32 {
        0 => Expr::Add(a, b),
        1 => Expr::Sub(a, b),
        _ => Expr::Mul(a, b),
    }
}

// --- Parseur d'expressions (texte → AST) -------------------------------- //
//
// Nécessaire pour le chemin LLM : les propositions arrivent en texte et doivent
// être parsées en `Expr` avant évaluation en sandbox. Accepte la sortie de
// `pretty()` comme les formes infixes naturelles (`x*x + 1`). Borne la
// profondeur (anti stack-overflow sur entrée hostile, comme le parseur JSON).

/// Profondeur d'imbrication maximale tolérée par le parseur d'expressions.
const MAX_EXPR_DEPTH: usize = 256;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    LParen,
    RParen,
    Plus,
    Minus,
    Star,
    X,
    Num(f64),
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\r' | '\n' => i += 1,
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                out.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                out.push(Tok::Star);
                i += 1;
            }
            'x' | 'X' => {
                out.push(Tok::X);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let lit: String = chars[start..i].iter().collect();
                let n: f64 = lit.parse().map_err(|_| format!("nombre invalide '{lit}'"))?;
                out.push(Tok::Num(n));
            }
            other => return Err(format!("caractère inattendu '{other}'")),
        }
    }
    Ok(out)
}

struct ExprParser<'a> {
    toks: &'a [Tok],
    pos: usize,
    depth: usize,
}

impl<'a> ExprParser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn enter(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_EXPR_DEPTH {
            return Err(format!("expression trop profonde (> {MAX_EXPR_DEPTH})"));
        }
        Ok(())
    }

    // expr := term (('+' | '-') term)*
    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.enter()?;
        let mut lhs = self.parse_term()?;
        while let Some(t) = self.peek() {
            let op = match t {
                Tok::Plus => 0,
                Tok::Minus => 1,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_term()?;
            lhs = if op == 0 {
                Expr::Add(Box::new(lhs), Box::new(rhs))
            } else {
                Expr::Sub(Box::new(lhs), Box::new(rhs))
            };
        }
        self.depth -= 1;
        Ok(lhs)
    }

    // term := factor ('*' factor)*
    fn parse_term(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_factor()?;
        while let Some(Tok::Star) = self.peek() {
            self.pos += 1;
            let rhs = self.parse_factor()?;
            lhs = Expr::Mul(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    // factor := '-' factor | primary
    fn parse_factor(&mut self) -> Result<Expr, String> {
        if let Some(Tok::Minus) = self.peek() {
            self.enter()?;
            self.pos += 1;
            let inner = self.parse_factor()?;
            self.depth -= 1;
            return Ok(Expr::Neg(Box::new(inner)));
        }
        self.parse_primary()
    }

    // primary := 'x' | number | '(' expr ')'
    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some(Tok::X) => {
                self.pos += 1;
                Ok(Expr::X)
            }
            Some(Tok::Num(n)) => {
                let n = *n;
                self.pos += 1;
                Ok(Expr::Const(n))
            }
            Some(Tok::LParen) => {
                self.pos += 1;
                let e = self.parse_expr()?;
                match self.peek() {
                    Some(Tok::RParen) => {
                        self.pos += 1;
                        Ok(e)
                    }
                    _ => Err("parenthèse fermante manquante".to_string()),
                }
            }
            other => Err(format!("primaire attendu, trouvé {other:?}")),
        }
    }
}

impl Expr {
    /// Parse une expression infixe (`x`, constantes, `+ - *`, négation unaire,
    /// parenthèses). Round-trip avec [`Expr::pretty`] ; accepte aussi l'infixe
    /// naturel. Borne la profondeur (anti stack-overflow).
    pub fn parse(s: &str) -> Result<Expr, String> {
        let toks = tokenize(s)?;
        if toks.is_empty() {
            return Err("expression vide".to_string());
        }
        let mut p = ExprParser {
            toks: &toks,
            pos: 0,
            depth: 0,
        };
        let e = p.parse_expr()?;
        if p.pos != p.toks.len() {
            return Err(format!("jetons superflus à partir de la position {}", p.pos));
        }
        Ok(e)
    }
}

/// Tâche de régression symbolique : ajuster une fonction cible sur des cas de
/// test, sous pénalité de complexité. Implémente [`RefineTask`] et
/// [`crate::llm::LlmRefineTask`].
pub struct SymbolicSynthesis {
    /// cas de test (x, cible) — la « suite de tests » d'entraînement du candidat.
    cases: Vec<(f64, f64)>,
    /// cas held-out (anti-Goodhart) : jamais vus par `score`, servent à mesurer
    /// la généralisation rapportée. Vide par défaut (cf. `from_target_split`).
    heldout: Vec<(f64, f64)>,
    /// tolérance d'acceptation par cas.
    tol: f64,
    /// pénalité par nœud d'AST (favorise la simplicité).
    complexity_penalty: f64,
    /// nombre de mutations évaluées par `refine` (1+λ).
    lambda: usize,
    rng: Rng,
}

impl SymbolicSynthesis {
    /// Construit la tâche à partir d'une fonction cible échantillonnée sur
    /// `n` points de `[lo, hi]`.
    pub fn from_target(
        target: impl Fn(f64) -> f64,
        lo: f64,
        hi: f64,
        n: usize,
        seed: u64,
    ) -> Self {
        let n = n.max(2);
        let cases = (0..n)
            .map(|i| {
                let x = lo + (hi - lo) * i as f64 / (n - 1) as f64;
                (x, target(x))
            })
            .collect();
        SymbolicSynthesis {
            cases,
            heldout: Vec::new(),
            tol: 0.25,
            complexity_penalty: 0.01,
            lambda: 16,
            rng: Rng::new(seed),
        }
    }

    /// Comme [`Self::from_target`] mais réserve ~30 % des points en **held-out**
    /// (entrelacés pour la couverture), jamais vus par `score` — base de la
    /// détection d'overfitting du chemin LLM (§3 du design spike).
    pub fn from_target_split(
        target: impl Fn(f64) -> f64,
        lo: f64,
        hi: f64,
        n: usize,
        seed: u64,
    ) -> Self {
        let n = n.max(4);
        let mut cases = Vec::new();
        let mut heldout = Vec::new();
        for i in 0..n {
            let x = lo + (hi - lo) * i as f64 / (n - 1) as f64;
            let pt = (x, target(x));
            // entrelacement déterministe : 3 points sur 10 en held-out (~30 %).
            if i % 10 < 3 {
                heldout.push(pt);
            } else {
                cases.push(pt);
            }
        }
        SymbolicSynthesis {
            cases,
            heldout,
            tol: 0.25,
            complexity_penalty: 0.01,
            lambda: 16,
            rng: Rng::new(seed),
        }
    }

    /// Nombre de cas held-out.
    pub fn heldout_len(&self) -> usize {
        self.heldout.len()
    }

    pub fn with_lambda(mut self, lambda: usize) -> Self {
        self.lambda = lambda.max(1);
        self
    }

    /// Candidat initial trivial.
    pub fn seed_candidate(&self) -> Expr {
        Expr::Const(0.0)
    }

    /// Fraction de cas réussis (sans pénalité) — utile pour le log/diagnostic.
    pub fn pass_fraction(&self, e: &Expr) -> f64 {
        let passed = self
            .cases
            .iter()
            .filter(|(x, t)| (e.eval(*x) - t).abs() <= self.tol)
            .count();
        passed as f64 / self.cases.len() as f64
    }

    /// Une mutation déterministe du candidat. Trois opérateurs :
    /// - **remplacement** d'un sous-arbre par un petit arbre aléatoire ;
    /// - **grow** : enrober le sous-arbre choisi dans un opérateur binaire avec
    ///   un terminal (construit de la structure, ex. `x → x*x`) ;
    /// - **perturbation** de constante.
    fn mutate(&mut self, e: &Expr) -> Expr {
        let n = e.size();
        let idx = (self.rng.uniform() * n as f64) as usize % n;
        let r = self.rng.uniform();
        let repl = if r < 0.45 {
            // grow : enrober le sous-arbre existant
            let mut cur = 0;
            let sub = e.subtree_at(idx, &mut cur).unwrap_or(Expr::X);
            let term = random_terminal(&mut self.rng);
            let (a, b) = (Box::new(sub), Box::new(term));
            match (self.rng.uniform() * 3.0) as u32 {
                0 => Expr::Add(a, b),
                1 => Expr::Sub(a, b),
                _ => Expr::Mul(a, b),
            }
        } else if r < 0.8 {
            random_expr(&mut self.rng, 2)
        } else {
            let c = (self.rng.uniform_range(-3.0, 3.0) * 4.0).round() / 4.0;
            Expr::Const(c)
        };
        let mut cur = 0;
        e.replace_at(idx, &repl, &mut cur)
    }
}

impl RefineTask for SymbolicSynthesis {
    type Cand = Expr;

    /// ÉVALUATEUR : fraction de tests réussis − pénalité de complexité.
    fn score(&self, cand: &Expr) -> f64 {
        let mut passed = 0usize;
        for (x, t) in &self.cases {
            let y = cand.eval(*x);
            if y.is_finite() && (y - t).abs() <= self.tol {
                passed += 1;
            }
        }
        let frac = passed as f64 / self.cases.len() as f64;
        frac - self.complexity_penalty * cand.size() as f64
    }

    /// GÉNÉRATEUR : meilleure de `lambda` mutations (révision critiquée, 1+λ).
    fn refine(&mut self, cand: &Expr, _iter: usize) -> Expr {
        let mut best = self.mutate(cand);
        let mut best_fit = self.score(&best);
        for _ in 1..self.lambda {
            let m = self.mutate(cand);
            let f = self.score(&m);
            if f > best_fit {
                best = m;
                best_fit = f;
            }
        }
        best
    }
}

impl LlmRefineTask for SymbolicSynthesis {
    /// Prompt : montre l'incumbent et son score, demande des variantes (une
    /// expression par ligne). C'est tout ce que le LLM « voit ».
    fn describe(&self, incumbent: &Expr) -> String {
        // Échantillon ÉTALÉ des points d'entraînement (x, f(x)) : donne au modèle
        // la *forme* de la cible (indispensable pour deviner le bon degré — sans
        // ça il tâtonne à l'aveugle et stagne sur un optimum local). Le held-out
        // n'est jamais montré (anti-Goodhart).
        let step = (self.cases.len() / 10).max(1);
        let points: String = self
            .cases
            .iter()
            .step_by(step)
            .take(10)
            .map(|(x, y)| format!("  f({x:.2}) = {y:.3}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "Trouve une expression arithmétique de la variable x qui reproduit la fonction cible f.\n\
             Points cibles (x, f(x)) :\n{points}\n\
             Grammaire STRICTE : uniquement `x`, des constantes numériques, les opérateurs + - * et des parenthèses. \
             Essaie des STRUCTURES VARIÉES et différents degrés (x ; x*x ; x*x*x ; combinaisons).\n\
             Meilleure expression actuelle : {inc}  (score {score:.3} ; plus haut = mieux).\n\
             RÉPONDS UNIQUEMENT par des expressions candidates, UNE PAR LIGNE — aucune prose, \
             aucun commentaire, aucune numérotation. Format attendu (exemples de SYNTAXE) :\n\
             x*x - 2\n\
             (x + 1) * x\n\
             3*x",
            inc = incumbent.pretty(),
            score = self.pass_fraction(incumbent),
        )
    }

    /// Parse chaque ligne en `Expr` ; ignore silencieusement les malformées.
    fn parse_proposals(&self, raw: &[String]) -> Vec<Expr> {
        raw.iter().filter_map(|s| Expr::parse(s).ok()).collect()
    }

    /// Évaluation held-out (généralisation rapportée, NE pilote PAS l'adoption).
    /// Retombe sur `score` si aucun held-out n'a été réservé.
    fn score_heldout(&self, cand: &Expr) -> f64 {
        if self.heldout.is_empty() {
            return self.score(cand);
        }
        let passed = self
            .heldout
            .iter()
            .filter(|(x, t)| {
                let y = cand.eval(*x);
                y.is_finite() && (y - t).abs() <= self.tol
            })
            .count();
        passed as f64 / self.heldout.len() as f64 - self.complexity_penalty * cand.size() as f64
    }

    /// Sûreté du domaine : rejette les AST trop complexes (un LLM pourrait
    /// proposer une expression qui explose en taille).
    fn safety_check(&self, cand: &Expr) -> Result<(), SafetyViolation> {
        if cand.size() > MAX_EXPR_SIZE {
            return Err(SafetyViolation(format!(
                "expression trop complexe ({} > {MAX_EXPR_SIZE} nœuds)",
                cand.size()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascent::{ascend, Guard};

    #[test]
    fn eval_and_size() {
        // x*x + 1
        let e = Expr::Add(
            Box::new(Expr::Mul(Box::new(Expr::X), Box::new(Expr::X))),
            Box::new(Expr::Const(1.0)),
        );
        assert!((e.eval(3.0) - 10.0).abs() < 1e-12);
        assert_eq!(e.size(), 5);
    }

    #[test]
    fn synthesis_improves_monotonically_and_terminates() {
        // cible : x^2 + 1
        let mut task = SymbolicSynthesis::from_target(|x| x * x + 1.0, -2.0, 2.0, 21, 42);
        let init = task.seed_candidate();
        let init_fit = task.score(&init);
        let guard = Guard::new().max_iters(60).patience(15).target(0.99).min_delta(0.0);
        let (best, report) = ascend(&mut task, init, &guard);

        // Contrat (garanti) : non-régression + terminaison bornée + amélioration.
        assert!(report.is_monotone(), "non-régression (élitisme)");
        assert!(report.iters <= 60, "terminaison bornée");
        assert!(report.best() > init_fit, "la fitness s'améliore");
        assert!(report.accepted >= 1, "au moins une révision adoptée");
        // amélioration substantielle de la couverture de tests vs l'initial
        let init = task.seed_candidate();
        assert!(
            task.pass_fraction(&best) > task.pass_fraction(&init),
            "couverture: {} ({})",
            task.pass_fraction(&best),
            best.pretty()
        );
    }

    #[test]
    fn deterministic_given_seed() {
        let run = || {
            let mut t = SymbolicSynthesis::from_target(|x| 2.0 * x - 1.0, -3.0, 3.0, 15, 7);
            let g = Guard::new().max_iters(40).target(0.99);
            let c = t.seed_candidate();
            let (_b, r) = ascend(&mut t, c, &g);
            r.best()
        };
        assert_eq!(run(), run()); // même graine ⇒ même résultat
    }

    // --- Parseur d'expressions ------------------------------------------- //

    #[test]
    fn expr_parse_roundtrips_pretty() {
        let e = Expr::Add(
            Box::new(Expr::Mul(Box::new(Expr::X), Box::new(Expr::X))),
            Box::new(Expr::Const(1.0)),
        );
        // pretty() doit se reparser à l'identique
        let reparsed = Expr::parse(&e.pretty()).unwrap();
        assert_eq!(reparsed, e);
    }

    #[test]
    fn expr_parse_accepts_natural_infix_with_precedence() {
        // x*x + 1 : '*' lie plus fort que '+'
        let e = Expr::parse("x*x + 1").unwrap();
        for x in [-2.0, 0.0, 3.5] {
            assert!((e.eval(x) - (x * x + 1.0)).abs() < 1e-9);
        }
        // négation unaire
        let n = Expr::parse("-(x + 2)").unwrap();
        assert!((n.eval(1.0) - (-(1.0 + 2.0))).abs() < 1e-9);
    }

    #[test]
    fn expr_parse_rejects_garbage_and_deep_nesting() {
        assert!(Expr::parse("").is_err());
        assert!(Expr::parse("x +").is_err());
        assert!(Expr::parse("(x + 1").is_err()); // parenthèse non fermée
        assert!(Expr::parse("@%$").is_err());
        // imbrication hostile bornée (pas de stack-overflow)
        let deep = "(".repeat(5_000);
        assert!(Expr::parse(&deep).is_err());
    }

    // --- Chemin LLM (LlmRefineTask) sur un vrai domaine ------------------ //

    #[test]
    fn llm_path_synthesizes_via_mock() {
        use crate::llm::{ascend_llm, LlmGuard, LlmStop, MockLlmClient};

        // cible x² + 1, avec held-out réservé
        let mut task = SymbolicSynthesis::from_target_split(|x| x * x + 1.0, -3.0, 3.0, 30, 1);
        assert!(task.heldout_len() > 0);

        // mock : un LLM scripté qui propose un chemin d'amélioration en texte
        let client = MockLlmClient::new(|_prompt, _k| {
            vec![
                "x".to_string(),
                "x*x".to_string(),
                "x*x + 1".to_string(), // solution exacte
            ]
        });
        let guard = LlmGuard {
            target: Some(0.9),
            patience: 3,
            max_iters: 20,
            ..LlmGuard::default()
        };
        let seed = task.seed_candidate();
        let (best, report) = ascend_llm(&mut task, seed, &client, &guard);

        assert!(report.is_monotone(), "incumbent train non monotone");
        assert!(report.accepted > 0);
        // la solution exacte passe tous les cas (train ET held-out)
        assert_eq!(task.pass_fraction(&best), 1.0, "best={}", best.pretty());
        assert!(report.best_heldout() > 0.9, "held-out faible: {}", report.best_heldout());
        assert_eq!(report.stop, LlmStop::Target);
    }

    #[test]
    fn llm_safety_check_rejects_oversized_expr() {
        use crate::llm::{ascend_llm, LlmGuard, MockLlmClient};

        let mut task = SymbolicSynthesis::from_target_split(|x| x * x + 1.0, -3.0, 3.0, 30, 2);
        // mock qui propose une bonne solution ET une expression géante (interdite)
        let huge = (0..40).map(|_| "x").collect::<Vec<_>>().join(" + "); // 40 termes
        let client = MockLlmClient::new(move |_p, _k| {
            vec!["x*x + 1".to_string(), huge.clone()]
        });
        let guard = LlmGuard { max_iters: 5, patience: 2, ..LlmGuard::default() };
        let seed = task.seed_candidate();
        let (best, report) = ascend_llm(&mut task, seed, &client, &guard);

        assert!(report.rejected_unsafe > 0, "l'expression géante aurait dû être rejetée");
        assert!(best.size() <= MAX_EXPR_SIZE, "un AST trop grand a été adopté");
    }
}
