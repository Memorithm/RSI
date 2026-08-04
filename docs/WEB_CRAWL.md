# Recherche web & crawl — `web_crawl.rs`

Module std-only de **crawl, surf et recherche web**, inspiré de
[spider-rs](https://github.com/spider-rs/spider) (crawler Rust conçu pour les
agents IA/LLM) et incorporé dans l'architecture RSI : aucun code de spider-rs
n'est vendu tel quel — l'*esprit* (concurrency-first, streaming, limites
configurables) est réimplémenté en Rust std-only, cohérent avec le dogme
« cœur sans dépendance » du projet.

## Pourquoi

L'audit a montré que RSI n'avait **aucune capacité réseau** hors transport LLM
(Ollama local, API Anthropic). Pour renforcer l'auto-évolution, le proposeur
DGM doit pouvoir *apprendre du web* (docs, littérature, code) au lieu de
deviner. `web_crawl` fournit :

- un client **HTTP/1.1 minimal** sur `std::net::TcpStream` (zéro dépendance),
  gérant `Transfer-Encoding: chunked`, bornes de taille et timeouts ;
- un **crawler BFS concurrency-first** (pool de workers `std::thread`, files
  partagées) avec politesse : `robots.txt`, délai entre requêtes vers un même
  hôte, bornes (pages max, profondeur max, taille max, liste noire d'hôtes) ;
- un **parseur HTML maison** : texte visible (scripts/styles/commentaires
  retirés), entités décodées, liens `href`/`src` résolus en absolus, titre ;
- un **index plein-texte local** (score TF-IDF simplifié) pour répondre à des
  requêtes sur le contenu crawlée ;
- un **adaptateur `WebCrawlerContext`** qui implémente
  `rsi::dgm::WebContextProvider` : branché sur `DgmEngine::with_web_context`,
  il crawle des seeds puis injecte les extraits pertinents dans le prompt du
  proposeur LLM (champ `external_context` d'`ImprovementContext`).

## Sûreté

- Accès réseau **borné** : timeout par requête, taille de réponse plafonnée
  (2 Mo par défaut), nombre de pages max (50), profondeur max (2), liste noire
  de schémas (hors http/https) et d'hôtes ;
- `robots.txt` respecté (User-Agent `RSI-Bot/0.10`) ;
- délai minimum configurable entre deux requêtes vers le même hôte (politesse) ;
- déterministe par graine pour la file d'attente (pas d'aléa) ;
- échec propre : un domaine qui bloque (DNS, timeout, 403) est compté comme
  erreur, le crawl continue sur les autres seeds.

## API

```rust
use rsi::web_crawl::{CrawlLimits, CrawlerOptions, WebCrawler, WebCrawlerContext};
use rsi::dgm::WebContextProvider;
use std::time::Duration;

// 1. crawl direct
let limits = CrawlLimits {
    max_pages: 50, max_depth: 2, max_bytes: 2 << 20,
    timeout: Duration::from_secs(10), politeness_delay: Duration::from_millis(200),
};
let crawler = WebCrawler::new(CrawlerOptions { limits, ..Default::default() });
let report = crawler.crawl(&["https://doc.rust-lang.org/std/".into()]);
let hits = crawler.search("ownership borrowing", 5);

// 2. surf web (une URL → texte)
let page = rsi::web_crawl::fetch_page_text("https://example.com", &limits)?;

// 3. RAG dans la boucle DGM
let web = WebCrawlerContext::new(CrawlerOptions::default(), seeds);
let engine = DgmEngine::new(archive, proposer, evaluator, config, seed)
    .with_web_context(Box::new(web));   // → injecté dans le prompt du proposeur
```

## CLI — `rsi-crawl`

```bash
cargo run --release --bin rsi-crawl -- crawl https://example.com \
    --max-pages 20 --depth 2 --query "rust" --out index.jsonl

cargo run --release --bin rsi-crawl -- fetch https://example.com/doc

cargo run --release --bin rsi-crawl -- search index.jsonl "rust ownership"
```

- `crawl` : BFS depuis les seeds, indexe, exporte les pages (JSONL), recherche
  optionnelle (`--query`) ;
- `fetch` : récupère et affiche le texte d'une URL unique ;
- `search` : recherche dans un index exporté (hors-ligne).

## Feature `web`

Le module compile **toujours** (client std-only). La feature optionnelle `web`
(`cargo build --features web`) ajoute le client `reqwest` (TLS complet, gzip,
redirections) pour les environnements où une vraie pile HTTP est souhaitée.
Sans elle, le client minimal suffit pour `http://` et les pages statiques.

## Démo

```bash
cargo run --release --example web_research_dgm
```

Montre le câblage de bout en bout : `WebCrawlerContext` → `external_context`
→ `LlmProposer::build_prompt`. Le proposeur mock vérifie la structure
indépendamment de la disponibilité du réseau.

## Moteur de recherche externe — DuckDuckGo

`DuckDuckGoSearch` interroge `https://html.duckduckgo.com/html/?q=…` et parse
les résultats (titre, URL, extrait). Il implémente `WebContextProvider` :
branché sur `DgmEngine::with_web_context`, le proposeur DGM reçoit les **vrais
résultats de recherche web** pour son objectif à chaque étape.

- **Feature `web` requise** pour les endpoints HTTPS réels (le client HTTP/1.1
  minimal std-only ne fait pas de TLS) ; `reqwest` suit les redirections et
  gère gzip/TLS.
- **User-Agent navigateur** : DuckDuckGo refuse les bots (`RSI-Bot` → 400).
  Le client envoie un UA Firefox par défaut, tout en restant poli (robots.txt,
  délais, bornes).
- CLI : `rsi-dgm --web [--web-prefix "rust"]` sur la boucle DGM.

```rust
use rsi::web_crawl::{CrawlLimits, DuckDuckGoSearch};
use rsi::dgm::WebContextProvider;

let ddg = DuckDuckGoSearch::new(CrawlLimits::default()).with_prefix("rust");
let hits = ddg.search("ownership borrowing", 3); // Vec<String> d'extraits
```

## Découverte mathématique (grammaire étendue + conjectures)

Le module `synthesis.rs` a été étendu pour la **découverte mathématique** :

- **Grammaire** : `+ - * / ^` (exposant entier 0..8), `exp(x)`, `ln(x)`,
  `sin(x)`, `cos(x)`, constantes `e`/`pi` — le tout évalué en **sandbox**
  (interpréteur maison, jamais de code exécuté) ;
- **`symbolic_equal`** : vérificateur d'égalité symbolique — réécriture
  algébrique (`x/x → 1`, `a+0 → a`, `(a+b)-b → a`), vérification polynomiale
  exacte (interpolation sur deg+1 points), et échantillonnage dense pour les
  fonctions transcendantes ;
- **`ConjectureGenerator`** : découvre des identités `left = right` à partir
  de briques (ex. `sin(x)² + cos(x)² = 1`), avec confiance numérique et preuve
  symbolique, en filtrant les trivialités (`x = 1·x`, `0 = 0·f(x)`, …) ;
- la boucle 1+λ et le chemin LLM (`LlmRefineTask`) peuvent désormais retrouver
  des fonctions trigonométriques/exponentielles — le prompt `describe` décrit
  la grammaire étendue.

```rust
use rsi::synthesis::{ConjectureGenerator, Expr, symbolic_equal};

let a = Expr::parse("(x + 1) ^ 2").unwrap();
let b = Expr::parse("x * x + 2 * x + 1").unwrap();
assert!(symbolic_equal(&a, &b, 50));

let gen = ConjectureGenerator::new(42);
let identites = gen.discover(&[Expr::parse("sin(x)").unwrap(),
                              Expr::parse("cos(x)").unwrap(),
                              Expr::X, Expr::parse("1").unwrap()],
                             60, 2, 80);
```

## Étapes suivantes (pistes d'amélioration)

- persister l'index (`TextIndex` sérialisable) pour réutilisation inter-campagnes ;
- recherche arXiv / GitHub code search comme sources supplémentaires du proposeur ;
- intégrer `fetch_page_text` dans `rsi-scholar` (papiers) en plus de `rsi-dgm`.
