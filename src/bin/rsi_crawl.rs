//! **rsi-crawl** — crawler web & recherche (esprit spider-rs), CLI.
//!
//! Trois sous-commandes :
//! - `crawl <url>…` : crawl BFS depuis les seeds, construit un index, exporte
//!   les pages (texte extrait) + résultats de recherche ;
//! - `fetch <url>` : récupère et extrait une page unique (surf web) ;
//! - `search <index.jsonl> <query>` : recherche dans un index construit par
//!   `crawl` (export).
//!
//! Sûreté : limites par défaut (max 50 pages, profondeur 2, 2 Mo/page, timeout
//! 10 s), robots.txt respecté, délai de politesse entre requêtes.
//!
//! Usage :
//! ```text
//! cargo run --release --bin rsi-crawl -- crawl https://example.com --max-pages 20
//! cargo run --release --bin rsi-crawl -- fetch https://example.com/doc
//! cargo run --release --bin rsi-crawl -- search out.jsonl "rust ownership"
//! ```

use rsi::web_crawl::{CrawlLimits, CrawlerOptions, TextIndex, WebCrawler};
use std::path::PathBuf;
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage : rsi-crawl <crawl|fetch|search> …");
        std::process::exit(2);
    }
    match args[1].as_str() {
        "crawl" => cmd_crawl(&args[2..]),
        "fetch" => cmd_fetch(&args[2..]),
        "search" => cmd_search(&args[2..]),
        other => {
            eprintln!("commande inconnue : {other}");
            std::process::exit(2);
        }
    }
}

/// Parse `--max-pages N`, `--depth N`, `--max-bytes N`, `--timeout S`,
/// `--delay MS`, `--no-robots`, `--out PATH`.
struct CliOpts {
    max_pages: usize,
    depth: usize,
    max_bytes: usize,
    timeout_secs: u64,
    delay_ms: u64,
    respect_robots: bool,
    allow_private: bool,
    out: Option<PathBuf>,
    query: Option<String>,
    seeds: Vec<String>,
}

fn parse_opts(args: &[String]) -> CliOpts {
    let mut o = CliOpts {
        max_pages: 50,
        depth: 2,
        max_bytes: 2 * 1024 * 1024,
        timeout_secs: 10,
        delay_ms: 200,
        respect_robots: true,
        allow_private: false,
        out: None,
        query: None,
        seeds: Vec::new(),
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--max-pages" => {
                o.max_pages = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(50);
                i += 2;
            }
            "--depth" => {
                o.depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2);
                i += 2;
            }
            "--max-bytes" => {
                o.max_bytes = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2 * 1024 * 1024);
                i += 2;
            }
            "--timeout" => {
                o.timeout_secs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(10);
                i += 2;
            }
            "--delay" => {
                o.delay_ms = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(200);
                i += 2;
            }
            "--no-robots" => {
                o.respect_robots = false;
                i += 1;
            }
            "--allow-private" => {
                // opt-out explicite de l'anti-SSRF (crawl localhost/réseau interne)
                o.allow_private = true;
                i += 1;
            }
            "--out" => {
                o.out = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--query" => {
                o.query = args.get(i + 1).cloned();
                i += 2;
            }
            s if s.starts_with('-') => {
                eprintln!("option inconnue : {s}");
                std::process::exit(2);
            }
            s => {
                o.seeds.push(s.to_string());
                i += 1;
            }
        }
    }
    o
}

fn cmd_crawl(args: &[String]) {
    let o = parse_opts(args);
    if o.seeds.is_empty() {
        eprintln!("rsi-crawl crawl : au moins une URL seed requise");
        std::process::exit(2);
    }
    let limits = CrawlLimits {
        max_pages: o.max_pages,
        max_depth: o.depth,
        max_bytes: o.max_bytes,
        timeout: Duration::from_secs(o.timeout_secs),
        politeness_delay: Duration::from_millis(o.delay_ms),
    };
    let options = CrawlerOptions {
        limits,
        respect_robots: o.respect_robots,
        deny_hosts: Vec::new(),
        // anti-SSRF par défaut ; --allow-private pour crawler localhost
        allow_private_hosts: o.allow_private,
    };
    let crawler = WebCrawler::new(options);
    println!(
        "crawl : {} seed(s), max {} pages, profondeur {}, robots {}",
        o.seeds.len(),
        o.max_pages,
        o.depth,
        if o.respect_robots { "oui" } else { "non" }
    );
    let report = crawler.crawl(&o.seeds);
    println!(
        "→ {} pages indexées, {} erreurs, {} ignorées",
        report.visited, report.errors, report.skipped
    );

    // export JSONL des pages — sérialisation via `crate::json` (échappement
    // complet : \n, \t, contrôles <0x20, unicode) ; l'ancienne esc() maison
    // produisait du JSONL invalide dès qu'une page contenait un retour ligne.
    if let Some(out) = &o.out {
        let mut s = String::new();
        for p in &report.pages {
            let mut o = rsi::json::Json::obj();
            o.set("url", rsi::json::Json::Str(p.url.clone()))
                .set("title", rsi::json::Json::Str(p.title.clone()))
                .set("text", rsi::json::Json::Str(p.text.clone()));
            s.push_str(&o.to_string());
            s.push('\n');
        }
        if let Err(e) = std::fs::write(out, s) {
            eprintln!("écriture {out:?} impossible : {e}");
        } else {
            println!("export : {out:?} ({} pages)", report.pages.len());
        }
    }

    // recherche éventuelle
    if let Some(q) = &o.query {
        let results = crawler.search(q, 5);
        println!("\nrésultats pour « {q} » :");
        for r in &results {
            println!("  [{:.3}] {} — {}", r.score, r.title, r.url);
            println!("         {}", r.snippet);
        }
        if results.is_empty() {
            println!("  (aucun résultat)");
        }
    }
}

fn cmd_fetch(args: &[String]) {
    let o = parse_opts(args);
    let Some(url) = o.seeds.first() else {
        eprintln!("rsi-crawl fetch : URL requise");
        std::process::exit(2);
    };
    let limits = CrawlLimits {
        max_pages: 1,
        max_depth: 0,
        max_bytes: o.max_bytes,
        timeout: Duration::from_secs(o.timeout_secs),
        politeness_delay: Duration::ZERO,
    };
    match rsi::web_crawl::fetch_page_text(url, &limits, o.allow_private) {
        Ok(page) => {
            println!("URL     : {}", page.url);
            println!("Titre   : {}", page.title);
            println!("Liens   : {}", page.links.len());
            println!("--- texte ({} caractères) ---", page.text.len());
            println!("{}", truncate(&page.text, 4000));
        }
        Err(e) => {
            eprintln!("échec : {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_search(args: &[String]) {
    // rsi-crawl search <index.jsonl> <query>
    let Some(index_path) = args.first() else {
        eprintln!("rsi-crawl search : fichier d'index requis");
        std::process::exit(2);
    };
    let query = args.get(1).cloned().unwrap_or_default();
    let data = match std::fs::read_to_string(index_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("lecture {index_path} impossible : {e}");
            std::process::exit(1);
        }
    };
    let mut idx = TextIndex::new();
    let mut count = 0usize;
    for line in data.lines() {
        if line.trim().is_empty() {
            continue;
        }
        // format JSONL {url,title,text} — parsé par le vrai parseur JSON
        // (l'extracteur naïf maison cassait sur `\"` échappé et sur l'ordre
        // de déséchappement).
        let Ok(j) = rsi::json::Json::parse(line) else {
            eprintln!("ligne ignorée (JSON invalide) : {:.60}", line);
            continue;
        };
        let get = |k: &str| j.get(k).and_then(|v| v.as_str()).map(str::to_string);
        if let (Some(u), Some(t), Some(x)) = (get("url"), get("title"), get("text")) {
            idx.add(&u, &t, &x);
            count += 1;
        }
    }
    println!("index : {count} documents");
    let results = idx.search(&query, 5);
    for r in &results {
        println!("  [{:.3}] {} — {}", r.score, r.title, r.url);
        println!("         {}", r.snippet);
    }
    if results.is_empty() {
        println!("  (aucun résultat)");
    }
}

fn truncate(s: &str, n: usize) -> String {
    let mut out: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        out.push('…');
    }
    out
}
