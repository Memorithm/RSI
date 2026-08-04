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
///
/// Grammaire étendue (au-delà de `+ - *`) pour la **découverte
/// mathématique** : division, puissance entière, exponentielle, logarithme,
/// sinus, cosinus. Chaque fonction est évaluée dans le sandbox (interpréteur
/// maison, `f64` std) — jamais exécutée comme code arbitraire.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    X,
    Const(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    /// Division (protection : dénominateur nul → f64::NAN, refusé par la fitness).
    Div(Box<Expr>, Box<Expr>),
    /// Puissance entière `a^b` avec `b` constant (exposant petit, borné).
    Pow(Box<Expr>, u32),
    /// Exponentielle `e^x`.
    Exp(Box<Expr>),
    /// Logarithme naturel `ln(x)` (domaine x > 0).
    Ln(Box<Expr>),
    /// Sinus.
    Sin(Box<Expr>),
    /// Cosinus.
    Cos(Box<Expr>),
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
            Expr::Div(a, b) => {
                let d = b.eval(x);
                if d.abs() < 1e-12 {
                    f64::NAN
                } else {
                    a.eval(x) / d
                }
            }
            Expr::Pow(a, n) => a.eval(x).powi(*n as i32),
            Expr::Exp(a) => a.eval(x).exp(),
            Expr::Ln(a) => {
                let v = a.eval(x);
                if v > 0.0 {
                    v.ln()
                } else {
                    f64::NAN
                }
            }
            Expr::Sin(a) => a.eval(x).sin(),
            Expr::Cos(a) => a.eval(x).cos(),
        }
    }

    /// Nombre de nœuds (complexité).
    pub fn size(&self) -> usize {
        match self {
            Expr::X | Expr::Const(_) => 1,
            Expr::Neg(a) | Expr::Exp(a) | Expr::Ln(a) | Expr::Sin(a) | Expr::Cos(a) => 1 + a.size(),
            Expr::Pow(a, _) => 1 + a.size(),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                1 + a.size() + b.size()
            }
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
            Expr::Div(a, b) => format!("({} / {})", a.pretty(), b.pretty()),
            Expr::Pow(a, n) => format!("({} ^ {n})", a.pretty()),
            Expr::Exp(a) => format!("exp({})", a.pretty()),
            Expr::Ln(a) => format!("ln({})", a.pretty()),
            Expr::Sin(a) => format!("sin({})", a.pretty()),
            Expr::Cos(a) => format!("cos({})", a.pretty()),
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
            Expr::Neg(a) | Expr::Exp(a) | Expr::Ln(a) | Expr::Sin(a) | Expr::Cos(a) => {
                a.subtree_at(idx, cur)
            }
            Expr::Pow(a, _) => a.subtree_at(idx, cur),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
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
            Expr::Div(a, b) => Expr::Div(
                Box::new(a.replace_at(idx, repl, cur)),
                Box::new(b.replace_at(idx, repl, cur)),
            ),
            Expr::Pow(a, n) => Expr::Pow(Box::new(a.replace_at(idx, repl, cur)), *n),
            Expr::Exp(a) => Expr::Exp(Box::new(a.replace_at(idx, repl, cur))),
            Expr::Ln(a) => Expr::Ln(Box::new(a.replace_at(idx, repl, cur))),
            Expr::Sin(a) => Expr::Sin(Box::new(a.replace_at(idx, repl, cur))),
            Expr::Cos(a) => Expr::Cos(Box::new(a.replace_at(idx, repl, cur))),
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
    // opérateurs étendus : + - * (fréquents), / ^ (moins), fonctions (rare)
    match (rng.uniform() * 7.0) as u32 {
        0 => Expr::Add(a, b),
        1 => Expr::Sub(a, b),
        2 => Expr::Mul(a, b),
        3 => Expr::Div(a, b),
        4 => {
            let n = 2 + (rng.uniform() * 3.0) as u32; // exposant 2..4
            Expr::Pow(a, n)
        }
        5 => Expr::Exp(a),
        _ => Expr::Ln(a),
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
    Slash,
    Caret,
    X,
    Num(f64),
    Exp,
    Ln,
    Sin,
    Cos,
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
            '/' => {
                out.push(Tok::Slash);
                i += 1;
            }
            '^' => {
                out.push(Tok::Caret);
                i += 1;
            }
            'x' | 'X' => {
                out.push(Tok::X);
                i += 1;
            }
            c if c.is_ascii_alphabetic() => {
                let start = i;
                while i < chars.len() && chars[i].is_ascii_alphabetic() {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                match word.as_str() {
                    "exp" => out.push(Tok::Exp),
                    "ln" => out.push(Tok::Ln),
                    "sin" => out.push(Tok::Sin),
                    "cos" => out.push(Tok::Cos),
                    "e" => out.push(Tok::Num(std::f64::consts::E)),
                    "pi" => out.push(Tok::Num(std::f64::consts::PI)),
                    _ => return Err(format!("fonction inconnue '{word}'")),
                }
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let lit: String = chars[start..i].iter().collect();
                let n: f64 = lit
                    .parse()
                    .map_err(|_| format!("nombre invalide '{lit}'"))?;
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

    // term := factor (('*' | '/') factor)*
    fn parse_term(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_factor()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => 0,
                Some(Tok::Slash) => 1,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_factor()?;
            lhs = if op == 0 {
                Expr::Mul(Box::new(lhs), Box::new(rhs))
            } else {
                Expr::Div(Box::new(lhs), Box::new(rhs))
            };
        }
        Ok(lhs)
    }

    // factor := '-' factor | primary ('^' NUMBER)?
    fn parse_factor(&mut self) -> Result<Expr, String> {
        if let Some(Tok::Minus) = self.peek() {
            self.enter()?;
            self.pos += 1;
            let inner = self.parse_factor()?;
            self.depth -= 1;
            return Ok(Expr::Neg(Box::new(inner)));
        }
        let base = self.parse_primary()?;
        // puissance : base ^ n (n constant, borné)
        if let Some(Tok::Caret) = self.peek() {
            self.enter()?;
            self.pos += 1;
            match self.parse_primary()? {
                Expr::Const(n) if (0.0..=8.0).contains(&n) && n.fract() == 0.0 => {
                    self.depth -= 1;
                    Ok(Expr::Pow(Box::new(base), n as u32))
                }
                other => {
                    self.depth -= 1;
                    Err(format!(
                        "exposant de puissance invalide (attendu entier 0..8, trouvé {})",
                        other.pretty()
                    ))
                }
            }
        } else {
            Ok(base)
        }
    }

    // primary := 'x' | number | 'e' | 'pi' | '(' expr ')' | func '(' expr ')'
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
            Some(Tok::Exp) | Some(Tok::Ln) | Some(Tok::Sin) | Some(Tok::Cos) => {
                let f = self.peek().unwrap().clone();
                self.pos += 1;
                // attend '('
                match self.peek() {
                    Some(Tok::LParen) => {
                        self.pos += 1;
                        let inner = self.parse_expr()?;
                        match self.peek() {
                            Some(Tok::RParen) => {
                                self.pos += 1;
                                Ok(match f {
                                    Tok::Exp => Expr::Exp(Box::new(inner)),
                                    Tok::Ln => Expr::Ln(Box::new(inner)),
                                    Tok::Sin => Expr::Sin(Box::new(inner)),
                                    _ => Expr::Cos(Box::new(inner)),
                                })
                            }
                            _ => Err("parenthèse fermante manquante après fonction".to_string()),
                        }
                    }
                    _ => Err("une fonction nécessite une parenthèse : exp(x), ln(x), sin(x), cos(x)"
                        .to_string()),
                }
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
            return Err(format!(
                "jetons superflus à partir de la position {}",
                p.pos
            ));
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
    pub fn from_target(target: impl Fn(f64) -> f64, lo: f64, hi: f64, n: usize, seed: u64) -> Self {
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
            match (self.rng.uniform() * 5.0) as u32 {
                0 => Expr::Add(a, b),
                1 => Expr::Sub(a, b),
                2 => Expr::Mul(a, b),
                3 => Expr::Div(a, b),
                _ => {
                    let n = 2 + (self.rng.uniform() * 3.0) as u32;
                    Expr::Pow(a, n)
                }
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

// ═══════════════════════════ Découverte mathématique ═══════════════════════ //

/// Vérificateur d'égalité **symbolique** entre deux expressions.
///
/// Contrairement à une vérification purement numérique (échantillonner des
/// points), ce vérificateur combine :
/// 1. **réécriture algébrique** : `a*1 → a`, `a+0 → a`, `a-a → 0`,
///    `(a+b)-b → a`, `a^1 → a`, `a/a → 1` (récursivement) ;
/// 2. **vérification polynomiale** : si après simplification les deux
///    expressions sont des polynômes en `x` (pas de fonction transcendante),
///    on compare leurs évaluations sur `deg+1` points distincts — exact pour
///    les polynômes de degré ≤ n (l'interpolation est unique) ;
/// 3. **échantillonnage dense** : sinon, comparaison sur un grand nombre de
///    points avec tolérance relative (heuristique pour les fonctions
///    transcendantes).
///
/// Renvoie `true` si les deux expressions sont (très probablement) égales.
pub fn symbolic_equal(a: &Expr, b: &Expr, samples: usize) -> bool {
    // réécriture : simplifie les deux côtés
    let sa = rewrite_simplify(a);
    let sb = rewrite_simplify(b);
    if structurally_eq(&sa, &sb) {
        return true;
    }

    // cas polynomial : exact
    if is_polynomial(&sa) && is_polynomial(&sb) {
        let deg = sa.size().max(sb.size()) + 1;
        for i in 0..=deg {
            let x = -2.0 + 4.0 * i as f64 / deg as f64;
            if (sa.eval(x) - sb.eval(x)).abs() > 1e-6 * (1.0 + sa.eval(x).abs() + sb.eval(x).abs()) {
                return false;
            }
        }
        return true;
    }

    // cas général : échantillonnage dense
    for i in 0..samples {
        let x = -5.0 + 10.0 * i as f64 / samples as f64;
        let va = sa.eval(x);
        let vb = sb.eval(x);
        if va.is_nan() || vb.is_nan() {
            // domaine invalide d'un côté seulement → pas égales (sauf si les
            // deux sont NaN au même point, ce qui est peu informatif)
            if va.is_nan() != vb.is_nan() {
                return false;
            }
            continue;
        }
        if (va - vb).abs() > 1e-4 * (1.0 + va.abs() + vb.abs()) {
            return false;
        }
    }
    true
}

/// Simplification par réécriture (idempotent, borné).
fn rewrite_simplify(e: &Expr) -> Expr {
    let e = rewrite_once(e);
    if e.size() < 2 {
        return e;
    }
    // itère la réécriture jusqu'à stabilité (au plus 8 passes)
    let mut cur = e;
    for _ in 0..8 {
        let next = rewrite_once(&cur);
        if structurally_eq(&next, &cur) {
            break;
        }
        cur = next;
    }
    cur
}

/// Une passe de réécriture : applique les règles de bas niveau, récursivement.
fn rewrite_once(e: &Expr) -> Expr {
    use Expr::*;
    let r = |x: &Expr| rewrite_once(x);
    match e {
        X | Const(_) => e.clone(),
        Neg(a) => Neg(Box::new(r(a))),
        Exp(a) => Exp(Box::new(r(a))),
        Ln(a) => Ln(Box::new(r(a))),
        Sin(a) => Sin(Box::new(r(a))),
        Cos(a) => Cos(Box::new(r(a))),
        Pow(a, n) => {
            let a = r(a);
            if *n == 1 {
                a
            } else if *n == 0 {
                Const(1.0)
            } else {
                Pow(Box::new(a), *n)
            }
        }
        Add(a, b) => {
            let (a, b) = (r(a), r(b));
            match (&a, &b) {
                (Const(0.0), _) => b,
                (_, Const(0.0)) => a,
                _ => Add(Box::new(a), Box::new(b)),
            }
        }
        Sub(a, b) => {
            let (a, b) = (r(a), r(b));
            match (&a, &b) {
                (_, Const(0.0)) => a,
                (x, y) if structurally_eq(x, y) => Const(0.0),
                _ => Sub(Box::new(a), Box::new(b)),
            }
        }
        Mul(a, b) => {
            let (a, b) = (r(a), r(b));
            match (&a, &b) {
                (Const(0.0), _) | (_, Const(0.0)) => Const(0.0),
                (Const(1.0), _) => b,
                (_, Const(1.0)) => a,
                _ => Mul(Box::new(a), Box::new(b)),
            }
        }
        Div(a, b) => {
            let (a, b) = (r(a), r(b));
            match (&a, &b) {
                (Const(0.0), _) => Const(0.0),
                (x, y) if structurally_eq(x, y) => Const(1.0),
                _ => Div(Box::new(a), Box::new(b)),
            }
        }
    }
}

/// Égalité structurelle (AST identique).
fn structurally_eq(a: &Expr, b: &Expr) -> bool {
    a == b
}

/// Une expression est-elle un polynôme en `x` (pas de fonction transcendante) ?
fn is_polynomial(e: &Expr) -> bool {
    match e {
        Expr::X | Expr::Const(_) => true,
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            is_polynomial(a) && is_polynomial(b)
        }
        Expr::Neg(a) => is_polynomial(a),
        Expr::Pow(a, _) => is_polynomial(a),
        Expr::Div(_, _) | Expr::Exp(_) | Expr::Ln(_) | Expr::Sin(_) | Expr::Cos(_) => false,
    }
}

/// Une **conjecture** : deux expressions candidate + estimation de confiance.
#[derive(Debug, Clone, PartialEq)]
pub struct Conjecture {
    pub left: Expr,
    pub right: Expr,
    /// identité normalisée `left = right`.
    pub statement: String,
    /// confiance ∈ (0, 1] : fraction des points échantillonnés qui vérifient
    /// l'égalité (avec tolérance). 1.0 = tous les points passent.
    pub confidence: f64,
    /// vrai si le vérificateur symbolique confirme (preuve « algébrique »).
    pub proven: bool,
}

/// Générateur de **conjectures** : à partir de deux « briques » (opérateurs,
/// fonctions) et d'une recherche par mutation, découvre des identités
/// plausibles `left = right` et les évalue numériquement.
///
/// Algorithme (borné, déterministe) :
/// 1. génère `n` expressions aléatoires de profondeur ≤ `depth` sur la
///    grammaire étendue ;
/// 2. pour chaque paire `(a, b)` rencontrée (y compris `a` vs des variantes
///    simplifiées de `b`), teste l'égalité sur `samples` points ;
/// 3. si tous les points passent (`confidence == 1`), tente le vérificateur
///    symbolique ; retient la conjecture (avec complexité minimale).
pub struct ConjectureGenerator {
    seed: u64,
}

impl Default for ConjectureGenerator {
    fn default() -> Self {
        ConjectureGenerator { seed: 42 }
    }
}

impl ConjectureGenerator {
    pub fn new(seed: u64) -> Self {
        ConjectureGenerator { seed }
    }

    /// Découvre des identités en partant de `bricks` (expressions de base).
    /// Retourne les conjectures (triées par complexité, les plus simples
    /// d'abord) qui passent la vérification.
    pub fn discover(
        &self,
        bricks: &[Expr],
        n_pairs: usize,
        depth: usize,
        samples: usize,
    ) -> Vec<Conjecture> {
        let mut rng = Rng::new(self.seed);
        let mut pool: Vec<Expr> = bricks.to_vec();
        // Enrichit le pool avec des combinaisons UTILES : puissances, sommes,
        // produits, et expressions aléatoires de la grammaire étendue.
        for i in 0..n_pairs {
            let a = bricks[(rng.uniform() * bricks.len() as f64) as usize % bricks.len()].clone();
            let b = bricks[(rng.uniform() * bricks.len() as f64) as usize % bricks.len()].clone();
            // constructions ciblées (déterministes à partir des briques)
            let pow2 = Expr::Pow(Box::new(a.clone()), 2);
            let pow3 = Expr::Pow(Box::new(a.clone()), 3);
            let sum = Expr::Add(Box::new(a.clone()), Box::new(b.clone()));
            let diff = Expr::Sub(Box::new(a.clone()), Box::new(b.clone()));
            let prod = Expr::Mul(Box::new(a.clone()), Box::new(b.clone()));
            let quot = Expr::Div(Box::new(a.clone()), Box::new(b.clone()));
            pool.push(pow2);
            pool.push(pow3);
            pool.push(sum);
            pool.push(diff);
            pool.push(prod);
            pool.push(quot);
            // combinaisons de puissances (sin²+cos²), sinon(x)+cos(x), …
            let a2 = Expr::Pow(Box::new(a.clone()), 2);
            let b2 = Expr::Pow(Box::new(b.clone()), 2);
            pool.push(Expr::Add(Box::new(a2.clone()), Box::new(b2.clone())));
            pool.push(Expr::Sub(Box::new(a2), Box::new(b2)));
            // expressions aléatoires de la grammaire étendue
            pool.push(random_expr(&mut rng, depth));
            let _ = i;
        }

        let mut found: Vec<Conjecture> = Vec::new();
        for i in 0..pool.len().min(200) {
            for j in 0..pool.len().min(200) {
                if i == j {
                    continue;
                }
                let (a, b) = (&pool[i], &pool[j]);
                // filtre : pas trop complexes, pas trivialement identiques,
                // pas de constantes pures triviales (0, 1, -0, …)
                if a.size() > 20 || b.size() > 20 || a == b {
                    continue;
                }
                if is_trivial_conjecture(a, b) {
                    continue;
                }
                // échantillonne
                let mut ok = 0usize;
                for k in 0..samples {
                    let x = -3.0 + 6.0 * k as f64 / samples as f64;
                    let va = a.eval(x);
                    let vb = b.eval(x);
                    if va.is_nan() || vb.is_nan() {
                        if va.is_nan() && vb.is_nan() {
                            ok += 1;
                        }
                    } else if (va - vb).abs() <= 1e-4 * (1.0 + va.abs() + vb.abs()) {
                        ok += 1;
                    }
                }
                let confidence = ok as f64 / samples as f64;
                if confidence >= 0.99 {
                    let proven = symbolic_equal(a, b, samples);
                    found.push(Conjecture {
                        left: a.clone(),
                        right: b.clone(),
                        statement: format!("{} = {}", a.pretty(), b.pretty()),
                        confidence,
                        proven,
                    });
                }
            }
        }

        // tri : les plus simples (et prouvées) d'abord
        found.sort_by(|x, y| {
            let cx = x.left.size() + x.right.size() + if x.proven { 0 } else { 100 };
            let cy = y.left.size() + y.right.size() + if y.proven { 0 } else { 100 };
            cx.cmp(&cy)
        });
        found.truncate(20);
        found
    }
}

/// Vrai si la conjecture est triviale (multiplication par 1, `x-x`, `0*…`,
/// constante pure, `x = c·x`, `c^n = c^m`), à écarter de la découverte.
fn is_trivial_conjecture(a: &Expr, b: &Expr) -> bool {
    // constante pure d'un côté (0, 1, -0, 2, …) et de l'autre une expression
    // qui se réduit à une constante → peu intéressant
    let const_of = |e: &Expr| match e {
        Expr::Const(c) => Some(*c),
        Expr::Pow(x, n) => match x.as_ref() {
            Expr::Const(c) => Some(c.powi(*n as i32)),
            _ => None,
        },
        _ => None,
    };
    match (const_of(a), const_of(b)) {
        (Some(_), Some(_)) => return true, // deux constantes (dont c^n = c^m)
        (Some(_), None) | (None, Some(_)) => {
            // un côté constant, l'autre doit être une vraie fonction de x
            let other = if const_of(a).is_some() { b } else { a };
            if !contains_x(other) {
                return true; // les deux sont des constantes → trivial
            }
        }
        (None, None) => {}
    }
    // formes triviales : mul(0|1, e), div(e, 1), sub(e, e), div(e, e)
    let is_const = |e: &Expr, v: f64| matches!(e, Expr::Const(c) if (c - v).abs() <= 1e-9);
    let is_triv = |e: &Expr| match e {
        Expr::Mul(x, y) => {
            is_const(x, 0.0) || is_const(y, 0.0) || is_const(x, 1.0) || is_const(y, 1.0)
        }
        Expr::Div(x, y) => is_const(y, 1.0) || structurally_eq(x, y),
        Expr::Sub(x, y) => structurally_eq(x, y),
        _ => false,
    };
    if is_triv(a) || is_triv(b) {
        return true;
    }
    // un côté est `x` (ou une variable nue) et l'autre une forme `c·x` / `x/c`
    let is_bare_x = |e: &Expr| matches!(e, Expr::X);
    let is_scale_of_x = |e: &Expr| match e {
        Expr::Mul(x, y) => (matches!(x.as_ref(), Expr::X) && matches!(y.as_ref(), Expr::Const(_)))
            || (matches!(y.as_ref(), Expr::X) && matches!(x.as_ref(), Expr::Const(_))),
        Expr::Div(x, y) => matches!(x.as_ref(), Expr::X) && matches!(y.as_ref(), Expr::Const(_)),
        _ => false,
    };
    if (is_bare_x(a) && is_scale_of_x(b)) || (is_bare_x(b) && is_scale_of_x(a)) {
        return true;
    }
    false
}

/// Une expression contient-elle la variable `x` ?
fn contains_x(e: &Expr) -> bool {
    match e {
        Expr::X => true,
        Expr::Const(_) => false,
        Expr::Neg(a) | Expr::Exp(a) | Expr::Ln(a) | Expr::Sin(a) | Expr::Cos(a) => contains_x(a),
        Expr::Pow(a, _) => contains_x(a),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
            contains_x(a) || contains_x(b)
        }
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
            "Trouve une expression mathématique de la variable x qui reproduit la fonction cible f.\n\
             Points cibles (x, f(x)) :\n{points}\n\
             Grammaire STRICTE : `x`, des constantes numériques, les opérateurs + - * / et ^ (exposant \
             entier 0..8), les fonctions exp(x), ln(x), sin(x), cos(x), les constantes e et pi, des \
             parenthèses. Essaie des STRUCTURES VARIÉES : polynômes, fractions, exponentielles, \
             trigonométriques, compositions.\n\
             Meilleure expression actuelle : {inc}  (score {score:.3} ; plus haut = mieux).\n\
             RÉPONDS UNIQUEMENT par des expressions candidates, UNE PAR LIGNE — aucune prose, \
             aucun commentaire, aucune numérotation. Format attendu (exemples de SYNTAXE) :\n\
             x*x - 2\n\
             (x + 1) * x\n\
             3*x\n\
             sin(x) / (x ^ 2 + 1)\n\
             exp(x) - 1",
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
        let guard = Guard::new()
            .max_iters(60)
            .patience(15)
            .target(0.99)
            .min_delta(0.0);
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
        assert!(
            report.best_heldout() > 0.9,
            "held-out faible: {}",
            report.best_heldout()
        );
        assert_eq!(report.stop, LlmStop::Target);
    }

    #[test]
    fn llm_safety_check_rejects_oversized_expr() {
        use crate::llm::{ascend_llm, LlmGuard, MockLlmClient};

        let mut task = SymbolicSynthesis::from_target_split(|x| x * x + 1.0, -3.0, 3.0, 30, 2);
        // mock qui propose une bonne solution ET une expression géante (interdite)
        let huge = (0..40).map(|_| "x").collect::<Vec<_>>().join(" + "); // 40 termes
        let client = MockLlmClient::new(move |_p, _k| vec!["x*x + 1".to_string(), huge.clone()]);
        let guard = LlmGuard {
            max_iters: 5,
            patience: 2,
            ..LlmGuard::default()
        };
        let seed = task.seed_candidate();
        let (best, report) = ascend_llm(&mut task, seed, &client, &guard);

        assert!(
            report.rejected_unsafe > 0,
            "l'expression géante aurait dû être rejetée"
        );
        assert!(
            best.size() <= MAX_EXPR_SIZE,
            "un AST trop grand a été adopté"
        );
    }

    // --- Grammaire étendue ------------------------------------------------ //

    #[test]
    fn eval_extended_grammar() {
        // division
        let d = Expr::Div(Box::new(Expr::X), Box::new(Expr::Const(2.0)));
        assert!((d.eval(10.0) - 5.0).abs() < 1e-12);
        let z = Expr::Div(Box::new(Expr::Const(1.0)), Box::new(Expr::Const(0.0)));
        assert!(z.eval(0.0).is_nan()); // division par zéro → NaN
        // puissance
        let p = Expr::Pow(Box::new(Expr::X), 3);
        assert!((p.eval(3.0) - 27.0).abs() < 1e-12);
        // exp / ln
        let e = Expr::Exp(Box::new(Expr::X));
        assert!((e.eval(0.0) - 1.0).abs() < 1e-12);
        let l = Expr::Ln(Box::new(Expr::X));
        assert!((l.eval(std::f64::consts::E) - 1.0).abs() < 1e-12);
        assert!(l.eval(-1.0).is_nan()); // ln de négatif → NaN
        // sin / cos
        let s = Expr::Sin(Box::new(Expr::X));
        assert!((s.eval(0.0) - 0.0).abs() < 1e-12);
        let c = Expr::Cos(Box::new(Expr::X));
        assert!((c.eval(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn parse_extended_grammar() {
        // division et précédence
        let d = Expr::parse("x / 2 + 1").unwrap();
        assert!((d.eval(4.0) - 3.0).abs() < 1e-9);
        // puissance
        let p = Expr::parse("x ^ 2").unwrap();
        assert!((p.eval(5.0) - 25.0).abs() < 1e-9);
        // fonctions
        let f = Expr::parse("exp(x) + ln(x) + sin(x) + cos(x)").unwrap();
        for x in [0.5, 1.0, 2.0] {
            assert!((f.eval(x) - (x.exp() + x.ln() + x.sin() + x.cos())).abs() < 1e-9);
        }
        // constantes e et pi
        let e = Expr::parse("e").unwrap();
        assert!((e.eval(0.0) - std::f64::consts::E).abs() < 1e-9);
        let pi = Expr::parse("pi").unwrap();
        assert!((pi.eval(0.0) - std::f64::consts::PI).abs() < 1e-9);
        // round-trip pretty
        let orig = Expr::parse("sin(x) / (x ^ 2 + 1)").unwrap();
        let reparsed = Expr::parse(&orig.pretty()).unwrap();
        assert_eq!(reparsed, orig);
        // rejets
        assert!(Expr::parse("x ^ 3.5").is_err()); // exposant non entier
        assert!(Expr::parse("x ^ 9").is_err()); // exposant > 8
        assert!(Expr::parse("tan(x)").is_err()); // fonction inconnue
    }

    #[test]
    fn extended_synthesis_still_improves() {
        // cible trigonométrique : la grammaire étendue doit la retrouver
        let mut task =
            SymbolicSynthesis::from_target(|x| x.sin(), -1.0, 1.0, 21, 5);
        let init = task.seed_candidate();
        let init_fit = task.score(&init);
        let guard = Guard::new()
            .max_iters(60)
            .patience(15)
            .target(0.95)
            .min_delta(0.0);
        let (best, report) = ascend(&mut task, init, &guard);
        assert!(report.is_monotone());
        assert!(report.best() > init_fit);
        assert!(task.pass_fraction(&best) > 0.5, "best={}", best.pretty());
    }

    // --- Vérificateur d'égalité symbolique -------------------------------- //

    #[test]
    fn symbolic_equal_detects_identities() {
        // (x+1)^2 = x^2 + 2x + 1  (polynomiale, exacte)
        let a = Expr::parse("(x + 1) ^ 2").unwrap();
        let b = Expr::parse("x * x + 2 * x + 1").unwrap();
        assert!(symbolic_equal(&a, &b, 50), "identité polynomiale non détectée");

        // x/x = 1 (réécriture)
        let c = Expr::parse("x / x").unwrap();
        let one = Expr::parse("1").unwrap();
        assert!(symbolic_equal(&c, &one, 20));

        // sin(x)^2 + cos(x)^2 = 1 (transcendant : échantillonnage dense)
        let d = Expr::parse("sin(x) ^ 2 + cos(x) ^ 2").unwrap();
        assert!(symbolic_equal(&d, &one, 100), "identité trigonométrique");

        // négatif : x^2 ≠ x^3
        let e = Expr::parse("x ^ 2").unwrap();
        let f = Expr::parse("x ^ 3").unwrap();
        assert!(!symbolic_equal(&e, &f, 50));

        // négatif : x ≠ x+1
        let g = Expr::parse("x").unwrap();
        let h = Expr::parse("x + 1").unwrap();
        assert!(!symbolic_equal(&g, &h, 50));
    }

    // --- Générateur de conjectures ---------------------------------------- //

    #[test]
    fn conjecture_generator_discovers_trig_identity() {
        let gen = ConjectureGenerator::new(42);
        let bricks = vec![
            Expr::parse("sin(x)").unwrap(),
            Expr::parse("cos(x)").unwrap(),
            Expr::X,
            Expr::Const(1.0),
        ];
        let found = gen.discover(&bricks, 60, 2, 80);
        // l'identité sin²+cos² = 1 doit être découverte (ou une équivalente)
        let trig = found
            .iter()
            .find(|c| c.statement.contains("sin") && c.statement.contains("cos"));
        assert!(trig.is_some(), "identité trigonométrique non découverte : {found:?}");
        assert!(trig.unwrap().confidence >= 0.99);
    }

    #[test]
    fn conjecture_generator_is_deterministic() {
        let bricks = vec![Expr::X, Expr::Const(1.0)];
        let a = ConjectureGenerator::new(7).discover(&bricks, 20, 2, 30);
        let b = ConjectureGenerator::new(7).discover(&bricks, 20, 2, 30);
        assert_eq!(a, b);
    }
}
