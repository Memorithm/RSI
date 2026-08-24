//! Pont **RSI → moteur canonique `scirust-rsi`**.
//!
//! Ce module implémente le contrat qualifié de `scirust-rsi`
//! (`scirust_rsi::refine::{RefineTask, SelfRefiner}`, `scirust_rsi::{Fitness,
//! Guard}`) pour le domaine de synthèse symbolique de RSI.
//!
//! La dépendance est épinglée dans `Cargo.toml` sur la révision SciRust exacte
//! qualifiée par P1.1. RSI ne maintient plus de seconde implémentation locale du
//! crate `scirust-rsi`.
//!
//! ## Sandbox
//! Le candidat reste un AST [`crate::synthesis::Expr`] **évalué par notre propre
//! interpréteur** ([`Expr::eval`]) : le moteur `scirust-rsi` n'appelle que nos
//! `score`/`refine` et ne voit que des nombres. Il n'exécute jamais de code
//! généré et ne se modifie pas. Le contrat de sûreté (terminaison, non-régression
//! best-so-far, déterminisme par graine) est porté par le moteur canonique.
use crate::synthesis::Expr;
use rand::rngs::StdRng;
use rand::Rng as _;
use scirust_rsi::refine::{RefineTask, SelfRefiner};
use scirust_rsi::{Fitness, Guard};

/// Domaine de régression symbolique adossé au moteur canonique `scirust-rsi`.
pub struct SymbolicSynthesis {
    cases: Vec<(f64, f64)>,
    tol: f64,
    complexity_penalty: f64,
    lambda: usize,
}

impl SymbolicSynthesis {
    /// Échantillonne la fonction cible sur `n` points de `[lo, hi]`.
    pub fn from_target(target: impl Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> Self {
        let n = n.max(2);
        let cases = (0..n)
            .map(|i| {
                let x = lo + (hi - lo) * i as f64 / (n - 1) as f64;
                (x, target(x))
            })
            .collect();
        SymbolicSynthesis {
            cases,
            tol: 0.25,
            complexity_penalty: 0.01,
            lambda: 16,
        }
    }

    pub fn with_lambda(mut self, lambda: usize) -> Self {
        self.lambda = lambda.max(1);
        self
    }

    /// Score scalaire = fraction de cas réussis − pénalité de complexité.
    fn raw_score(&self, e: &Expr) -> f64 {
        let mut passed = 0usize;
        for (x, t) in &self.cases {
            let y = e.eval(*x);
            if y.is_finite() && (y - t).abs() <= self.tol {
                passed += 1;
            }
        }
        let frac = passed as f64 / self.cases.len() as f64;
        frac - self.complexity_penalty * e.size() as f64
    }

    /// Fraction de cas réussis (diagnostic / log).
    pub fn pass_fraction(&self, e: &Expr) -> f64 {
        let passed = self
            .cases
            .iter()
            .filter(|(x, t)| (e.eval(*x) - t).abs() <= self.tol)
            .count();
        passed as f64 / self.cases.len() as f64
    }
}

fn random_terminal(rng: &mut StdRng) -> Expr {
    if rng.gen::<f64>() < 0.5 {
        Expr::X
    } else {
        let c = (rng.gen_range(-2.0..2.0) * 4.0_f64).round() / 4.0;
        Expr::Const(c)
    }
}

fn random_expr(rng: &mut StdRng, depth: usize) -> Expr {
    if depth == 0 || rng.gen::<f64>() < 0.3 {
        return random_terminal(rng);
    }
    let a = Box::new(random_expr(rng, depth - 1));
    let b = Box::new(random_expr(rng, depth - 1));
    match rng.gen_range(0..3) {
        0 => Expr::Add(a, b),
        1 => Expr::Sub(a, b),
        _ => Expr::Mul(a, b),
    }
}

fn subtree_at(e: &Expr, idx: usize, cur: &mut usize) -> Option<Expr> {
    let here = *cur;
    *cur += 1;
    if here == idx {
        return Some(e.clone());
    }
    match e {
        Expr::X | Expr::Const(_) => None,
        Expr::Neg(a) | Expr::Exp(a) | Expr::Ln(a) | Expr::Sin(a) | Expr::Cos(a) => {
            subtree_at(a, idx, cur)
        }
        Expr::Pow(a, _) => subtree_at(a, idx, cur),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
            subtree_at(a, idx, cur).or_else(|| subtree_at(b, idx, cur))
        }
    }
}

fn replace_at(e: &Expr, idx: usize, repl: &Expr, cur: &mut usize) -> Expr {
    let here = *cur;
    *cur += 1;
    if here == idx {
        return repl.clone();
    }
    match e {
        Expr::X | Expr::Const(_) => e.clone(),
        Expr::Neg(a) => Expr::Neg(Box::new(replace_at(a, idx, repl, cur))),
        Expr::Add(a, b) => Expr::Add(
            Box::new(replace_at(a, idx, repl, cur)),
            Box::new(replace_at(b, idx, repl, cur)),
        ),
        Expr::Sub(a, b) => Expr::Sub(
            Box::new(replace_at(a, idx, repl, cur)),
            Box::new(replace_at(b, idx, repl, cur)),
        ),
        Expr::Mul(a, b) => Expr::Mul(
            Box::new(replace_at(a, idx, repl, cur)),
            Box::new(replace_at(b, idx, repl, cur)),
        ),
        Expr::Div(a, b) => Expr::Div(
            Box::new(replace_at(a, idx, repl, cur)),
            Box::new(replace_at(b, idx, repl, cur)),
        ),
        Expr::Pow(a, n) => Expr::Pow(Box::new(replace_at(a, idx, repl, cur)), *n),
        Expr::Exp(a) => Expr::Exp(Box::new(replace_at(a, idx, repl, cur))),
        Expr::Ln(a) => Expr::Ln(Box::new(replace_at(a, idx, repl, cur))),
        Expr::Sin(a) => Expr::Sin(Box::new(replace_at(a, idx, repl, cur))),
        Expr::Cos(a) => Expr::Cos(Box::new(replace_at(a, idx, repl, cur))),
    }
}

fn mutate(e: &Expr, rng: &mut StdRng) -> Expr {
    let n = e.size();
    let idx = rng.gen_range(0..n);
    let r = rng.gen::<f64>();
    let repl = if r < 0.45 {
        let mut cur = 0;
        let sub = subtree_at(e, idx, &mut cur).unwrap_or(Expr::X);
        let term = random_terminal(rng);
        let (a, b) = (Box::new(sub), Box::new(term));
        match rng.gen_range(0..3) {
            0 => Expr::Add(a, b),
            1 => Expr::Sub(a, b),
            _ => Expr::Mul(a, b),
        }
    } else if r < 0.8 {
        random_expr(rng, 2)
    } else {
        let c = (rng.gen_range(-3.0..3.0) * 4.0_f64).round() / 4.0;
        Expr::Const(c)
    };
    let mut cur = 0;
    replace_at(e, idx, &repl, &mut cur)
}

impl RefineTask for SymbolicSynthesis {
    type Solution = Expr;

    fn initial(&self, _rng: &mut StdRng) -> Expr {
        Expr::Const(0.0)
    }

    fn score(&self, a: &Expr) -> Fitness {
        self.raw_score(a)
    }

    fn refine(&self, a: &Expr, rng: &mut StdRng) -> Expr {
        let mut best = mutate(a, rng);
        let mut best_fit = self.raw_score(&best);
        for _ in 1..self.lambda {
            let m = mutate(a, rng);
            let f = self.raw_score(&m);
            if f > best_fit {
                best = m;
                best_fit = f;
            }
        }
        best
    }
}

/// Lance l'ascension sur la cible `x² + 1` via le moteur canonique.
pub fn run_self_improve(seed: u64) -> (Expr, scirust_rsi::Report) {
    let task = SymbolicSynthesis::from_target(|x| x * x + 1.0, -2.0, 2.0, 21).with_lambda(24);
    let guard = Guard::new().max_iters(50).patience(12).target(0.99);
    SelfRefiner::new(seed).run(&task, &guard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_report_is_deterministic_best_so_far() {
        let (_, a) = run_self_improve(0x5C1_2026);
        let (_, b) = run_self_improve(0x5C1_2026);

        assert_eq!(a.iterations, b.iterations);
        assert_eq!(a.accepted, b.accepted);
        assert_eq!(a.best_fitness, b.best_fitness);
        assert_eq!(a.history, b.history);
        assert_eq!(a.history.len(), a.iterations);
        assert!(a.is_monotone());
        assert!(a.history.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(a.history.last().copied(), Some(a.best_fitness));
    }
}
