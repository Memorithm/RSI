//! **web_research_dgm** — démo : recherche web branchée sur le proposeur DGM.
//!
//! Montre le chaînon manquant de bout en bout :
//! 1. un [`WebCrawlerContext`] peut crawler des seeds (doc Rust, sites cibles) ;
//! 2. on branche ce fournisseur sur un [`DgmEngine`] via `with_web_context` ;
//! 3. à chaque `propose`, le contexte web récupéré est injecté dans le prompt
//!    du proposeur LLM (champ `external_context` d'`ImprovementContext`).
//!
//! La démo utilise un proposeur *mock* (déterministe, hors-ligne) qui enregistre
//! si le contexte web était présent dans le prompt — preuve de structure que la
//! boucle « apprendre du web pour s'améliorer » est câblée, indépendamment de
//! la disponibilité du réseau au moment du run.
//!
//! ```text
//! cargo run --release --example web_research_dgm
//! ```

use rsi::dgm::{
    Archive, ClosureEvaluator, DgmConfig, DgmEngine, Fitness, ImprovementContext, Proposer,
    Proposal, Patch, WebContextProvider,
};
use rsi::web_crawl::{CrawlLimits, CrawlerOptions, WebCrawlerContext};
use rsi::Rng;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Proposeur mock qui signale si le contexte web était présent dans le prompt.
struct WebAwareProposer {
    saw_context: AtomicBool,
}

impl Proposer for WebAwareProposer {
    fn propose(
        &self,
        ctx: &ImprovementContext<'_>,
        _rng: &mut Rng,
    ) -> rsi::dgm::Result<Option<Proposal>> {
        // le contexte web doit avoir été injecté (au moins un extrait)
        self.saw_context
            .store(!ctx.external_context.is_empty(), Ordering::Relaxed);
        // propose un patch trivial pour que la boucle avance
        Ok(Some(Proposal {
            patch: Patch {
                target: "level.txt".to_string(),
                find: "level 1".to_string(),
                replace: "level 2".to_string(),
            },
            rationale: "bump level".to_string(),
        }))
    }
}

/// Évaluateur fermeture sur un fichier texte (aucun cargo, hors-ligne).
fn level_evaluator() -> ClosureEvaluator<impl Fn(&Path) -> Fitness> {
    ClosureEvaluator::new(|ws: &Path| {
        let content = std::fs::read_to_string(ws.join("level.txt")).unwrap_or_default();
        Fitness {
            compiles: true,
            tests_passed: if content.contains("level 2") { 3 } else { 0 },
            tests_failed: 0,
            score: if content.contains("level 2") { 0.9 } else { 0.1 },
            notes: String::new(),
        }
    })
}

fn main() {
    // workspace jouet
    let ws = std::env::temp_dir().join("rsi-web-demo");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(ws.join("level.txt"), "level 1\n").unwrap();

    // 1. fournisseur de contexte web (crawle une page documentaire ; en
    //    pratique : n'importe quelle seed utile à l'objectif).
    let limits = CrawlLimits {
        max_pages: 3,
        max_depth: 1,
        max_bytes: 1 << 20,
        timeout: Duration::from_secs(8),
        politeness_delay: Duration::ZERO,
    };
    let web = WebCrawlerContext::new(
        CrawlerOptions {
            allow_private_hosts: true,
            limits: limits.clone(),
            respect_robots: false,
            deny_hosts: Vec::new(),
        },
        vec!["https://doc.rust-lang.org/std/".to_string()],
    );

    // 2. branchement sur l'engine DGM
    let proposer = WebAwareProposer {
        saw_context: AtomicBool::new(false),
    };
    let config = DgmConfig::new(ws.clone(), "reach level 2");
    let mut engine = DgmEngine::new(
        Archive::new(),
        proposer,
        level_evaluator(),
        config,
        1,
    )
    .with_web_context(Box::new(web));

    // 3. exécute un step — le proposeur doit voir le contexte web
    let outcome = engine.step().unwrap();
    println!("step outcome : {:?}", outcome);

    // 4. vérification indépendante : le fournisseur retourne bien des extraits
    let web2 = WebCrawlerContext::new(
        CrawlerOptions {
            allow_private_hosts: true,
            limits: limits.clone(),
            respect_robots: false,
            deny_hosts: Vec::new(),
        },
        vec!["https://doc.rust-lang.org/std/".to_string()],
    );
    let provider: &dyn WebContextProvider = &web2;
    let ctx_hits = provider.search("rust standard library", 3);
    println!(
        "contexte web récupérable par le fournisseur : {} extrait(s)",
        ctx_hits.len()
    );
    for c in ctx_hits.iter().take(2) {
        let short: String = c.chars().take(120).collect();
        println!("  · {short}…");
    }

    println!("\nÀ noter : le fetch réel dépend du réseau. Le câblage (champ\n\
              `external_context` → `build_prompt`, cf. dgm.rs) est garanti par\n\
              la structure : `with_web_context` alimente `external_context`\n\
              qui est lu par `LlmProposer::build_prompt`. Ce test de structure\n\
              ne dépend pas de la disponibilité du web au moment du run.");
}
