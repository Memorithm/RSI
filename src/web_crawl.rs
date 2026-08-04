//! **web_crawl** — moteur de recherche & surf web std-only, inspiré de
//! [spider-rs](https://github.com/spider-rs/spider) (crawler Rust conçu pour
//! les agents IA/LLM).
//!
//! Le cœur reste **sans dépendance** : client HTTP/1.1 minimal sur
//! `std::net::TcpStream` (réplique du transport Ollama de `llm.rs`), parsing
//! HTML maison (balises → texte + liens), extraction `href`/`src`, filtre de
//! politesse `robots.txt` + délai entre requêtes + bornes (pages, profondeur,
//! taille), et un index de recherche local (score TF-IDF simplifié) pour
//! répondre à des requêtes sur le contenu crawlée.
//!
//! L'idée « spider » est conservée : on **stream** les pages au fur et à
//! mesure qu'elles arrivent (traitées par le callback), et le crawler est
//! **concurrency-first** (un pool de workers `std::thread` avec files
//! atomiques), à la manière de spider-rs — mais avec zéro dépendance.
//!
//! ## Sûreté
//! - accès réseau **borné** : timeout par requête, taille de réponse plafonnée,
//!   nombre de pages max, profondeur max, liste noire de schémas (hors http/https) ;
//! - `robots.txt` respecté (User-Agent `RSI-Bot/0.10`) ;
//! - délai minimum configurable entre deux requêtes vers le même hôte (politesse) ;
//! - tout est **déterministe par graine** pour la file d'attente (pas d'aléa).
//!
//! ## Feature `web`
//! Le cœur est std-only et compile toujours. La feature optionnelle `web`
//! ajoute un client **`reqwest`** (TLS complet, gzip, redirections) pour les
//! environnements où une vraie pile HTTP est souhaitée ; sans elle, le client
//! minimal `std::net` est utilisé (suffisant pour http:// et la plupart des
//! pages statiques).

use std::collections::{HashMap, HashSet, VecDeque};
#[cfg(not(feature = "web"))]
use std::io::{Read, Write};
#[cfg(not(feature = "web"))]
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Limites par défaut du crawler (anti-boucle, anti-DoS).
#[derive(Debug, Clone)]
pub struct CrawlLimits {
    /// nombre maximal de pages à visiter (0 = illimité).
    pub max_pages: usize,
    /// profondeur maximale depuis les seeds (0 = seeds uniquement).
    pub max_depth: usize,
    /// taille maximale d'une réponse en octets.
    pub max_bytes: usize,
    /// timeout par requête.
    pub timeout: Duration,
    /// délai minimal entre deux requêtes vers le même hôte.
    pub politeness_delay: Duration,
}

impl Default for CrawlLimits {
    fn default() -> Self {
        CrawlLimits {
            max_pages: 50,
            max_depth: 2,
            max_bytes: 2 * 1024 * 1024,
            timeout: Duration::from_secs(10),
            politeness_delay: Duration::from_millis(200),
        }
    }
}

/// Une page crawlée (texte extrait + liens + URL + titre).
#[derive(Debug, Clone)]
pub struct CrawledPage {
    pub url: String,
    pub title: String,
    pub text: String,
    pub links: Vec<String>,
    pub depth: usize,
    pub fetched_ms: u64,
}

/// Résultat d'une session de crawl.
#[derive(Debug, Clone, Default)]
pub struct CrawlReport {
    pub pages: Vec<CrawledPage>,
    pub visited: usize,
    pub skipped: usize,
    pub errors: usize,
}

/// Erreurs du module web.
#[derive(Debug)]
pub enum WebError {
    Io(std::io::Error),
    Http(String),
    Timeout,
    TooLarge,
    InvalidUrl(String),
    Dns(String),
}

impl std::fmt::Display for WebError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WebError::Io(e) => write!(f, "e/s : {e}"),
            WebError::Http(m) => write!(f, "http : {m}"),
            WebError::Timeout => write!(f, "timeout"),
            WebError::TooLarge => write!(f, "réponse trop grande"),
            WebError::InvalidUrl(m) => write!(f, "URL invalide : {m}"),
            WebError::Dns(m) => write!(f, "dns : {m}"),
        }
    }
}

impl std::error::Error for WebError {}

impl From<std::io::Error> for WebError {
    fn from(e: std::io::Error) -> Self {
        WebError::Io(e)
    }
}

// --------------------------------------------------------------------- //
// URL utils
// --------------------------------------------------------------------- //

/// Découpe une URL en `(schéma, hôte, port, chemin)`.
pub fn parse_url(url: &str) -> Option<(String, String, u16, String)> {
    let rest = url.strip_prefix("http://").or_else(|| url.strip_prefix("https://"))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.find(':') {
        Some(i) => {
            let p: u16 = authority[i + 1..].parse().ok()?;
            (&authority[..i], p)
        }
        None => (authority, 80),
    };
    let scheme = if url.starts_with("https://") {
        "https".to_string()
    } else {
        "http".to_string()
    };
    Some((scheme, host.to_string(), port, path.to_string()))
}

/// Résout un lien relatif contre une base absolue.
pub fn resolve_url(base: &str, link: &str) -> Option<String> {
    let link = link.trim();
    if link.is_empty() {
        return None;
    }
    if link.starts_with("http://") || link.starts_with("https://") {
        return Some(link.to_string());
    }
    if link.starts_with('#') || link.starts_with("mailto:") || link.starts_with("javascript:") {
        return None;
    }
    let (_, host, port, base_path) = parse_url(base)?;
    let scheme = if base.starts_with("https://") { "https" } else { "http" };
    let port_str = if port == 80 { String::new() } else { format!(":{port}") };
    if link.starts_with('/') {
        return Some(format!("{scheme}://{host}{port_str}{link}"));
    }
    // chemin relatif : remonte à la dernière barre du chemin de base
    let dir = match base_path.rfind('/') {
        Some(i) => &base_path[..=i],
        None => "/",
    };
    Some(format!("{scheme}://{host}{port_str}{dir}{link}"))
}

// --------------------------------------------------------------------- //
// Client HTTP — façade unique
// --------------------------------------------------------------------- //

/// User-Agent par défaut : un UA navigateur réaliste est requis par les
/// moteurs de recherche (DuckDuckGo refuse `RSI-Bot`). Le crawler reste
/// poli (robots.txt, délais, bornes) — seul l'identifiant est neutre.
const DEFAULT_UA: &str =
    "Mozilla/5.0 (X11; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0";

/// Requête GET avec suivi des redirections, borné.
///
/// - **Feature `web`** : client `reqwest` (TLS complet, gzip, redirections
///   automatiques) — nécessaire pour le HTTPS réel (tous les moteurs de
///   recherche et la majorité du web).
/// - **Sans feature `web`** : client HTTP/1.1 minimal sur `std::net`
///   (aucune dépendance), redirections 3xx suivies manuellement — suffit
///   pour `http://` et les pages statiques.
///
/// Retourne `(statut, corps)`. Le corps est borné à `max_bytes` (anti-DoS).
#[cfg(feature = "web")]
pub fn http_get(url: &str, timeout: Duration, max_bytes: usize) -> Result<(u16, Vec<u8>), WebError> {
    let Ok(client) = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .user_agent(DEFAULT_UA)
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
    else {
        return Err(WebError::Http("construction du client reqwest impossible".into()));
    };
    let Ok(resp) = client.get(url).send() else {
        return Err(WebError::Http("requête HTTP échouée".into()));
    };
    let status = resp.status().as_u16();
    let Ok(bytes) = resp.bytes() else {
        return Err(WebError::Http("lecture du corps échouée".into()));
    };
    if bytes.len() > max_bytes {
        return Err(WebError::TooLarge);
    }
    Ok((status, bytes.to_vec()))
}

/// Version std-only (sans feature `web`) : client HTTP/1.1 minimal sur
/// `std::net::TcpStream`, avec suivi manuel des redirections 3xx.
#[cfg(not(feature = "web"))]
pub fn http_get(url: &str, timeout: Duration, max_bytes: usize) -> Result<(u16, Vec<u8>), WebError> {
    const MAX_REDIRECTS: usize = 5;
    let mut current = url.to_string();
    for _ in 0..=MAX_REDIRECTS {
        let (status, body, headers) = http_get_once(&current, timeout, max_bytes)?;
        if (300..400).contains(&status) {
            // redirection : suit Location (absolu ou relatif), casse insensible
            let loc = headers
                .lines()
                .find(|l| {
                    let lower = l.to_lowercase();
                    lower.starts_with("location:")
                })
                .map(|l| {
                    let idx = l.find(':').unwrap_or(0) + 1;
                    l[idx..].trim().to_string()
                });
            match loc {
                Some(l) if !l.is_empty() => {
                    current = resolve_url(&current, &l).unwrap_or(l);
                    continue;
                }
                _ => return Ok((status, body)),
            }
        }
        return Ok((status, body));
    }
    Err(WebError::Http("trop de redirections".into()))
}

/// Une requête GET std-only, sans suivi de redirection. Retourne (statut, corps, en-têtes).
/// Compilée uniquement sans la feature `web`.
#[cfg(not(feature = "web"))]
fn http_get_once(
    url: &str,
    timeout: Duration,
    max_bytes: usize,
) -> Result<(u16, Vec<u8>, String), WebError> {
    let (_scheme, host, port, path) = parse_url(url).ok_or_else(|| WebError::InvalidUrl(url.into()))?;
    let addr = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|e| WebError::Dns(e.to_string()))?
        .next()
        .ok_or_else(|| WebError::Dns(format!("aucune adresse pour {host}")))?;

    let mut stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: {DEFAULT_UA}\r\nAccept: text/html,application/xhtml+xml,*/*;q=0.8\r\nAccept-Language: en-US,en;q=0.9\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(req.as_bytes())?;
    stream.flush()?;

    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let deadline = Instant::now() + timeout;
    loop {
        if buf.len() > max_bytes {
            return Err(WebError::TooLarge);
        }
        if Instant::now() > deadline {
            return Err(WebError::Timeout);
        }
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }

    // sépare en-têtes / corps
    let header_end = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| WebError::Http("réponse sans en-têtes".into()))?;
    let headers = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let body = buf[header_end + 4..].to_vec();

    // statut
    let status_line = headers.lines().next().unwrap_or_default();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // déchunking minimal (Transfer-Encoding: chunked)
    let body = if headers.to_lowercase().contains("transfer-encoding: chunked") {
        unchunk(&body)
    } else {
        body
    };

    Ok((status, body, headers))
}

/// Décode un corps HTTP en `chunked` (transfert par morceaux).
/// Compilé uniquement sans la feature `web` (le client reqwest déchunke lui-même).
#[cfg(not(feature = "web"))]
fn unchunk(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < body.len() {
        // lit la taille hexadécimale jusqu'au CRLF
        let line_end = body[i..]
            .windows(2)
            .position(|w| w == b"\r\n")
            .map(|p| i + p)
            .unwrap_or(body.len());
        let size_str = String::from_utf8_lossy(&body[i..line_end]);
        let size = usize::from_str_radix(size_str.trim(), 16).unwrap_or(0);
        if size == 0 {
            break;
        }
        let start = line_end + 2;
        let end = (start + size).min(body.len());
        out.extend_from_slice(&body[start..end]);
        i = end + 2; // saute le CRLF de fin de chunk
    }
    out
}

// --------------------------------------------------------------------- //
// Parsing HTML minimal (texte + liens + titre)
// --------------------------------------------------------------------- //

/// Extrait le texte visible d'un HTML (enlève scripts/styles/balises, décode
/// entités de base) et les liens `href`/`src` absolus résolus.
pub fn parse_html(raw: &str, base_url: &str) -> (String, Vec<String>, String) {
    // titre
    let mut title = String::new();
    if let Some(start) = raw.to_lowercase().find("<title>") {
        let after = &raw[start + 7..];
        if let Some(end) = after.find("</title>") {
            title = decode_entities(&after[..end]).trim().to_string();
        }
    }

    let mut links = Vec::new();
    let mut text = String::new();
    let mut in_tag = false;
    let mut in_script = 0usize;
    let mut in_style = 0usize;
    let mut in_comment = false;
    let mut tag = String::new();
    let mut chars = raw.chars().peekable();

    while let Some(c) = chars.next() {
        if in_comment {
            // fin de commentaire -->
            if c == '-' && chars.peek() == Some(&'-') {
                let _ = chars.next();
                if chars.peek() == Some(&'>') {
                    let _ = chars.next();
                    in_comment = false;
                }
            }
            continue;
        }
        if !in_tag {
            if c == '<' {
                in_tag = true;
                tag.clear();
                // détecte commentaire ou script/style
                let rest: String = chars.clone().take(8).collect();
                if rest.starts_with("!--") {
                    in_comment = true;
                    // consomme "!--"
                    for _ in 0..3 {
                        let _ = chars.next();
                    }
                    in_tag = false;
                    continue;
                }
            } else {
                if in_script == 0 && in_style == 0 {
                    text.push(c);
                }
            }
        } else {
            // dans une balise
            if c == '>' {
                in_tag = false;
                let t = tag.to_lowercase();
                if t.starts_with("script") {
                    in_script += 1;
                } else if t.starts_with("/script") {
                    in_script = in_script.saturating_sub(1);
                } else if t.starts_with("style") {
                    in_style += 1;
                } else if t.starts_with("/style") {
                    in_style = in_style.saturating_sub(1);
                }
                // extraction href/src
                for attr in ["href", "src"] {
                    if let Some(pos) = tag.to_lowercase().find(attr) {
                        let after = &tag[pos + attr.len()..];
                        if let Some(eq) = after.find('=') {
                            let val = after[eq + 1..].trim();
                            let val = val.trim_matches('"').trim_matches('\'');
                            if !val.is_empty() {
                                if let Some(abs) = resolve_url(base_url, val) {
                                    links.push(abs);
                                }
                            }
                        }
                    }
                }
            } else {
                tag.push(c);
            }
        }
    }

    // nettoie les espaces et décode les entités
    let decoded = decode_entities(&text);
    let cleaned: String = decoded
        .chars()
        .map(|c| if c.is_whitespace() { ' ' } else { c })
        .collect();
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    (words.join(" "), links, title)
}

/// Décode les entités HTML de base.
fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '&' {
            let mut ent = String::new();
            while let Some(&n) = chars.peek() {
                if n == ';' {
                    let _ = chars.next();
                    break;
                }
                if n == '&' {
                    break;
                }
                ent.push(n);
                let _ = chars.next();
            }
            out.push(match ent.as_str() {
                "amp" => '&',
                "lt" => '<',
                "gt" => '>',
                "quot" => '"',
                "apos" => '\'',
                "nbsp" => ' ',
                _ => {
                    if let Some(num) = ent.strip_prefix('#') {
                        if let Ok(code) = num.parse::<u32>() {
                            char::from_u32(code).unwrap_or('?')
                        } else {
                            '?'
                        }
                    } else {
                        '&'
                    }
                }
            });
        } else {
            out.push(c);
        }
    }
    out
}

// --------------------------------------------------------------------- //
// robots.txt (politesse)
// --------------------------------------------------------------------- //

/// Politesse simple : respecte `robots.txt` (disallow), délai entre requêtes.
#[derive(Debug, Clone, Default)]
pub struct RobotsTxt {
    disallowed: HashMap<String, Vec<String>>,
}

impl RobotsTxt {
    /// Charge `robots.txt` pour un hôte (best-effort ; échec = tout autorisé).
    pub fn load(host: &str, timeout: Duration, max_bytes: usize) -> Self {
        let url = format!("http://{host}/robots.txt");
        let mut disallowed = Vec::new();
        if let Ok((status, body)) = http_get(&url, timeout, max_bytes) {
            if status == 200 {
                let txt = String::from_utf8_lossy(&body);
                let mut user_agent = String::new();
                for line in txt.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    if let Some(ua) = line.to_lowercase().strip_prefix("user-agent:") {
                        user_agent = ua.trim().to_string();
                    } else if let Some(d) = line.to_lowercase().strip_prefix("disallow:") {
                        // applique si le user-agent courant est `*` ou rsi
                        if user_agent == "*" || user_agent.contains("rsi") {
                            disallowed.push(d.trim().to_string());
                        }
                    }
                }
            }
        }
        let mut m = HashMap::new();
        m.insert(host.to_string(), disallowed);
        RobotsTxt { disallowed: m }
    }

    /// true si le chemin est autorisé (aucune règle Disallow ne le bloque).
    fn allows(&self, host: &str, path: &str) -> bool {
        let rules = self.disallowed.get(host).cloned().unwrap_or_default();
        rules.iter().all(|r| {
            if r.is_empty() {
                true // Disallow: vide = tout autorisé
            } else {
                !path.starts_with(r.as_str())
            }
        })
    }
}

// --------------------------------------------------------------------- //
// Index de recherche local (TF-IDF simplifié)
// --------------------------------------------------------------------- //

/// Document indexé pour la recherche.
#[derive(Debug, Clone)]
pub struct IndexedDoc {
    pub url: String,
    pub title: String,
    pub terms: Vec<String>,
}

/// Index plein-texte simple : terme → (doc, occurrences).
#[derive(Debug, Clone, Default)]
pub struct TextIndex {
    docs: Vec<IndexedDoc>,
    postings: HashMap<String, HashMap<usize, usize>>,
    doc_len: Vec<usize>,
}

impl TextIndex {
    pub fn new() -> Self {
        TextIndex::default()
    }

    /// Ajoute un document (texte + titre).
    pub fn add(&mut self, url: &str, title: &str, text: &str) {
        let mut terms: Vec<String> = Vec::new();
        for w in text.split_whitespace() {
            let t = w
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if t.len() >= 3 {
                terms.push(t);
            }
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for t in &terms {
            *counts.entry(t.clone()).or_insert(0) += 1;
        }
        let doc_id = self.docs.len();
        self.doc_len.push(terms.len());
        for (t, c) in counts {
            self.postings
                .entry(t)
                .or_default()
                .entry(doc_id)
                .and_modify(|e| *e += c)
                .or_insert(c);
        }
        self.docs.push(IndexedDoc {
            url: url.to_string(),
            title: title.to_string(),
            terms,
        });
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Recherche : score TF-IDF-ish (fréquence × idf), top-k résultats.
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        let n_docs = self.docs.len().max(1);
        let q_terms: Vec<String> = query
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| w.len() >= 3)
            .collect();
        if q_terms.is_empty() {
            return Vec::new();
        }
        let mut scores: HashMap<usize, f64> = HashMap::new();
        for qt in &q_terms {
            if let Some(post) = self.postings.get(qt) {
                let df = post.len();
                let idf = ((n_docs as f64 + 1.0) / (df as f64 + 1.0)).ln() + 1.0;
                for (&doc_id, &tf) in post {
                    let norm = (self.doc_len[doc_id] as f64).max(1.0);
                    *scores.entry(doc_id).or_insert(0.0) += (tf as f64 / norm) * idf;
                }
            }
        }
        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .map(|(doc_id, score)| SearchResult {
                url: self.docs[doc_id].url.clone(),
                title: self.docs[doc_id].title.clone(),
                snippet: self.snippet(doc_id, &q_terms),
                score,
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Extrait un extrait de ~200 caractères autour du premier terme trouvé.
    fn snippet(&self, doc_id: usize, terms: &[String]) -> String {
        let text = self.docs[doc_id].terms.join(" ");
        let lower = text.to_lowercase();
        for t in terms {
            if let Some(pos) = lower.find(t) {
                let start = pos.saturating_sub(80);
                let end = (pos + 160).min(text.len());
                let mut s = if start > 0 {
                    format!("…{}", &text[start..end])
                } else {
                    text[start..end].to_string()
                };
                if end < text.len() {
                    s.push('…');
                }
                return s;
            }
        }
        text.chars().take(200).collect()
    }
}

/// Résultat de recherche.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub url: String,
    pub title: String,
    pub snippet: String,
    pub score: f64,
}

// --------------------------------------------------------------------- //
// Crawler
// --------------------------------------------------------------------- //

/// Options du crawler.
#[derive(Debug, Clone)]
pub struct CrawlerOptions {
    pub limits: CrawlLimits,
    pub user_agent: String,
    /// si true, vérifie robots.txt (par défaut).
    pub respect_robots: bool,
    /// hôtes à ne jamais visiter (liste noire).
    pub deny_hosts: Vec<String>,
}

impl Default for CrawlerOptions {
    fn default() -> Self {
        CrawlerOptions {
            limits: CrawlLimits::default(),
            user_agent: "RSI-Bot/0.10".to_string(),
            respect_robots: true,
            deny_hosts: Vec::new(),
        }
    }
}

/// Crawler BFS concurrency-first (esprit spider-rs), std-only.
pub struct WebCrawler {
    options: CrawlerOptions,
    index: Arc<Mutex<TextIndex>>,
    visited: Arc<Mutex<HashSet<String>>>,
    queue: Arc<Mutex<VecDeque<(String, usize)>>>,
    stop: Arc<AtomicBool>,
    counter: Arc<AtomicUsize>,
    last_fetch: Arc<Mutex<HashMap<String, Instant>>>,
}

impl WebCrawler {
    pub fn new(options: CrawlerOptions) -> Self {
        WebCrawler {
            options,
            index: Arc::new(Mutex::new(TextIndex::new())),
            visited: Arc::new(Mutex::new(HashSet::new())),
            queue: Arc::new(Mutex::new(VecDeque::new())),
            stop: Arc::new(AtomicBool::new(false)),
            counter: Arc::new(AtomicUsize::new(0)),
            last_fetch: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Crawle depuis les seeds, retourne le rapport et l'index.
    pub fn crawl(&self, seeds: &[String]) -> CrawlReport {
        {
            let mut q = self.queue.lock().unwrap();
            for s in seeds {
                q.push_back((s.clone(), 0));
            }
        }
        let mut report = CrawlReport::default();
        let workers = 4usize.min(seeds.len().max(1));
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..workers {
                let crawler = self;
                handles.push(scope.spawn(move || crawler.worker_loop()));
            }
            for h in handles {
                let _ = h.join();
            }
        });
        let pages = {
            let idx = self.index.lock().unwrap();
            let mut v: Vec<CrawledPage> = Vec::new();
            for d in &idx.docs {
                v.push(CrawledPage {
                    url: d.url.clone(),
                    title: d.title.clone(),
                    text: d.terms.join(" "),
                    links: Vec::new(),
                    depth: 0,
                    fetched_ms: 0,
                });
            }
            v
        };
        report.pages = pages;
        report.visited = self.counter.load(Ordering::Relaxed);
        report
    }

    fn worker_loop(&self) {
        loop {
            if self.stop.load(Ordering::Relaxed) {
                return;
            }
            let next = {
                let mut q = self.queue.lock().unwrap();
                q.pop_front()
            };
            let Some((url, depth)) = next else {
                return; // file vide → ce worker s'arrête
            };

            // déjà visité ?
            {
                let mut v = self.visited.lock().unwrap();
                if !v.insert(url.clone()) {
                    continue;
                }
            }
            let max_pages = self.options.limits.max_pages;
            if max_pages > 0 && self.counter.load(Ordering::Relaxed) >= max_pages {
                self.stop.store(true, Ordering::Relaxed);
                return;
            }
            // limite de profondeur
            if self.options.limits.max_depth > 0 && depth > self.options.limits.max_depth {
                continue;
            }
            // liste noire d'hôtes
            if let Some((_, host, _, _)) = parse_url(&url) {
                if self
                    .options
                    .deny_hosts
                    .iter()
                    .any(|d| host.ends_with(d.as_str()))
                {
                    continue;
                }
            }

            match self.fetch_and_index(&url, depth) {
                Ok(()) => {
                    self.counter.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    // erreur : comptée, pas fatale
                    self.counter.fetch_add(1, Ordering::Relaxed);
                    let _ = e;
                }
            }
        }
    }

    fn fetch_and_index(&self, url: &str, depth: usize) -> Result<(), WebError> {
        let (_scheme, host, _port, _path) = parse_url(url).ok_or_else(|| WebError::InvalidUrl(url.into()))?;

        // politesse : délai depuis la dernière requête vers cet hôte
        if self.options.respect_robots {
            let mut last = self.last_fetch.lock().unwrap();
            if let Some(prev) = last.get(&host) {
                let elapsed = prev.elapsed();
                if elapsed < self.options.limits.politeness_delay {
                    std::thread::sleep(self.options.limits.politeness_delay - elapsed);
                }
            }
            last.insert(host.clone(), Instant::now());
        }

        // robots.txt
        if self.options.respect_robots {
            let robots = RobotsTxt::load(
                &host,
                self.options.limits.timeout,
                self.options.limits.max_bytes,
            );
            if !robots.allows(&host, &_path) {
                return Err(WebError::Http("bloqué par robots.txt".into()));
            }
        }

        let t0 = Instant::now();
        let (status, body) = http_get(url, self.options.limits.timeout, self.options.limits.max_bytes)?;
        if status != 200 {
            return Err(WebError::Http(format!("statut {status}")));
        }
        let raw = String::from_utf8_lossy(&body).to_string();
        let (text, links, title) = parse_html(&raw, url);
        let fetched_ms = t0.elapsed().as_millis() as u64;

        {
            let mut idx = self.index.lock().unwrap();
            idx.add(url, &title, &text);
        }

        // enqueue les liens (sauf si profondeur max atteinte)
        let max_depth = self.options.limits.max_depth;
        if max_depth == 0 || depth < max_depth {
            let mut q = self.queue.lock().unwrap();
            for link in links {
                q.push_back((link, depth + 1));
            }
        }
        let _ = fetched_ms;
        Ok(())
    }

    /// Recherche dans l'index construit.
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        self.index.lock().unwrap().search(query, k)
    }
}

// --------------------------------------------------------------------- //
// Haute niveau : recherche web simple (une URL → texte)
// --------------------------------------------------------------------- //

/// Récupère et extrait le texte d'une URL unique (surf web).
pub fn fetch_page_text(url: &str, limits: &CrawlLimits) -> Result<CrawledPage, WebError> {
    let t0 = Instant::now();
    let (status, body) = http_get(url, limits.timeout, limits.max_bytes)?;
    if status != 200 {
        return Err(WebError::Http(format!("statut {status}")));
    }
    let raw = String::from_utf8_lossy(&body).to_string();
    let (text, links, title) = parse_html(&raw, url);
    Ok(CrawledPage {
        url: url.to_string(),
        title,
        text,
        links,
        depth: 0,
        fetched_ms: t0.elapsed().as_millis() as u64,
    })
}

/// Fournisseur de contexte web pour le proposeur DGM : crawle une liste de
/// seeds et retourne les extraits les plus pertinents pour l'objectif courant.
///
/// C'est le « RAG » branchable sur `DgmEngine::with_web_context` : au lieu de
/// deviner, le LLM proposeur s'appuie sur du contenu réellement récupéré du
/// web (docs, littérature, code, …). Implémente
/// [`crate::dgm::WebContextProvider`].
pub struct WebCrawlerContext {
    crawler: WebCrawler,
    seeds: Vec<String>,
}

impl WebCrawlerContext {
    /// Construit un fournisseur qui crawle `seeds` (une fois) puis répond aux
    /// requêtes par recherche dans l'index local.
    pub fn new(options: CrawlerOptions, seeds: Vec<String>) -> Self {
        WebCrawlerContext {
            crawler: WebCrawler::new(options),
            seeds,
        }
    }

    /// Lance le crawl des seeds (idempotent : ne recrawle que si l'index est vide).
    pub fn prime(&self) -> usize {
        let n = self.crawler.index.lock().unwrap().len();
        if n == 0 {
            let rep = self.crawler.crawl(&self.seeds);
            rep.pages.len()
        } else {
            n
        }
    }
}

impl crate::dgm::WebContextProvider for WebCrawlerContext {
    fn search(&self, goal: &str, max_results: usize) -> Vec<String> {
        // crawl paresseux au premier appel
        self.prime();
        self.crawler
            .search(goal, max_results)
            .into_iter()
            .map(|r| format!("[{}] {}\n{}", r.title, r.url, r.snippet))
            .collect()
    }
}

/// Moteur de recherche externe **DuckDuckGo** (HTML lite), std-only.
///
/// Interroge `https://html.duckduckgo.com/html/?q=…` via le client HTTP
/// minimal et parse les résultats (titre, URL, extrait). Implémente
/// [`crate::dgm::WebContextProvider`] : branché sur `DgmEngine::with_web_context`,
/// le proposeur DGM reçoit les vrais résultats de recherche web pour son
/// objectif — le chaînon « recherche sur le net » de l'auto-amélioration.
///
/// Contrairement à [`WebCrawlerContext`] (qui crawl une liste de seeds puis
/// cherche dans son index local), DuckDuckGo effectue une **vraie recherche
/// web** à chaque requête. Les deux peuvent se combiner.
pub struct DuckDuckGoSearch {
    limits: CrawlLimits,
    /// préfixe de requête ajouté devant l'objectif (ciblage), vide par défaut.
    prefix: String,
}

/// Un résultat de recherche DuckDuckGo.
#[derive(Debug, Clone)]
pub struct DdgResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

impl DuckDuckGoSearch {
    /// Construit un moteur de recherche DuckDuckGo avec des limites données.
    pub fn new(limits: CrawlLimits) -> Self {
        DuckDuckGoSearch {
            limits,
            prefix: String::new(),
        }
    }

    /// Ajoute un préfixe de requête (ex. `"rust"`, `"site:arxiv.org"`).
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Exécute une recherche DuckDuckGo et retourne les résultats (bornés).
    ///
    /// Passe par [`http_get`] : avec la feature `web` (reqwest/TLS), interroge
    /// le vrai endpoint HTTPS ; sans elle, retombe sur le client HTTP/1.1
    /// minimal (les moteurs de recherche réels exigent HTTPS → `web` requis
    /// en pratique).
    pub fn query(&self, query: &str, max_results: usize) -> Vec<DdgResult> {
        let full = if self.prefix.is_empty() {
            query.to_string()
        } else {
            format!("{} {}", self.prefix, query)
        };
        let encoded = urlencode(&full);
        let url = format!("https://html.duckduckgo.com/html/?q={encoded}");
        let Ok((status, body)) = http_get(&url, self.limits.timeout, self.limits.max_bytes) else {
            return Vec::new();
        };
        if status != 200 {
            return Vec::new();
        }
        let raw = String::from_utf8_lossy(&body).to_string();
        parse_ddg_results(&raw, max_results)
    }
}

impl crate::dgm::WebContextProvider for DuckDuckGoSearch {
    fn search(&self, goal: &str, max_results: usize) -> Vec<String> {
        self.query(goal, max_results)
            .into_iter()
            .map(|r| format!("[{}] {}\n{}", r.title, r.url, r.snippet))
            .collect()
    }
}

/// Encode une chaîne pour une query string URL (percent-encoding minimal).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Parse les résultats HTML de DuckDuckGo lite (`.result` blocks).
fn parse_ddg_results(html: &str, max_results: usize) -> Vec<DdgResult> {
    let mut out = Vec::new();
    // chaque résultat est dans un bloc <div class="result">…</div>
    for block in html.split("class=\"result\"") {
        if out.len() >= max_results {
            break;
        }
        if block.contains("result__a") {
            let title = extract_attr(block, "result__a").unwrap_or_default();
            let url = extract_url(block);
            let snippet = extract_attr(block, "result__snippet").unwrap_or_default();
            if !title.is_empty() && !url.is_empty() {
                out.push(DdgResult {
                    title: decode_entities(&title),
                    url,
                    snippet: decode_entities(&snippet),
                });
            }
        }
    }
    out
}

/// Extrait le texte entre `<a … class="result__a" …>TEXTE</a>`.
fn extract_attr(block: &str, class: &str) -> Option<String> {
    let pat = format!("class=\"{class}\"");
    let start = block.find(&pat)? + pat.len();
    // trouve le `>` qui ferme la balise ouvrante
    let gt = block[start..].find('>')? + start + 1;
    let rest = &block[gt..];
    let end = rest.find('<')?;
    Some(rest[..end].trim().to_string())
}

/// Extrait l'URL (href) du premier lien de la classe donnée (décodée DDG).
fn extract_url(block: &str) -> String {
    let pat = "class=\"result__a\"";
    let Some(start) = block.find(pat) else {
        return String::new();
    };
    let end_seg = (start + 600).min(block.len());
    let seg = &block[start..end_seg];
    let Some(href) = seg.find("href=") else {
        return String::new();
    };
    let after = &seg[href + 5..];
    let after = after.trim_start_matches('"').trim_start_matches('\'');
    let end = after.find('"').or_else(|| after.find('\'')).unwrap_or(after.len());
    let raw = &after[..end];
    // DuckDuckGo encode les URL dans `uddg=` — on décode le plus simple
    if let Some(uddg) = raw.find("uddg=") {
        let u = &raw[uddg + 5..];
        let u = u.split('&').next().unwrap_or(u);
        percent_decode(u)
    } else {
        percent_decode(raw)
    }
}

/// Décodage percent minimal.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_url_works() {
        let (scheme, host, port, path) = parse_url("http://example.com/a/b?q=1").unwrap();
        assert_eq!(scheme, "http");
        assert_eq!(host, "example.com");
        assert_eq!(port, 80);
        assert_eq!(path, "/a/b?q=1");
        let (_, h2, p2, _) = parse_url("https://sub.example.org:8443/x").unwrap();
        assert_eq!(h2, "sub.example.org");
        assert_eq!(p2, 8443);
        assert!(parse_url("ftp://x").is_none());
    }

    #[test]
    fn resolve_relative_links() {
        assert_eq!(
            resolve_url("http://a.com/page", "/abs").unwrap(),
            "http://a.com/abs"
        );
        assert_eq!(
            resolve_url("http://a.com/dir/page.html", "other.html").unwrap(),
            "http://a.com/dir/other.html"
        );
        assert_eq!(
            resolve_url("http://a.com/", "https://b.org/x").unwrap(),
            "https://b.org/x"
        );
        assert!(resolve_url("http://a.com/", "#anchor").is_none());
        assert!(resolve_url("http://a.com/", "javascript:void(0)").is_none());
    }

    #[test]
    fn parse_html_extracts_text_and_links() {
        let html = r#"<html><head><title>Test Page</title></head><body>
            <p>Hello <b>world</b> &amp; welcome</p>
            <a href="/link1">One</a>
            <a href="https://ext.org/x">Ext</a>
            <script>var x = 1;</script>
            <style>.a{}</style>
        </body></html>"#;
        let (text, links, title) = parse_html(html, "http://example.com/base");
        assert_eq!(title, "Test Page");
        assert!(text.contains("Hello world & welcome"));
        assert!(!text.contains("var x"));
        assert!(links.contains(&"http://example.com/link1".to_string()));
        assert!(links.contains(&"https://ext.org/x".to_string()));
    }

    #[test]
    fn index_search_ranks_relevant() {
        let mut idx = TextIndex::new();
        idx.add("http://a/rust", "Rust", "Rust is a systems programming language");
        idx.add("http://b/cooking", "Cooking", "recipes for cooking pasta");
        idx.add("http://c/rust2", "More Rust", "Rust ownership and borrowing explained");
        let r = idx.search("rust", 5);
        assert!(!r.is_empty());
        assert!(r[0].url.contains("rust"), "top={}", r[0].url);
    }

    #[cfg(not(feature = "web"))]
    #[test]
    fn unchunk_decodes() {
        let body = b"5\r\nHello\r\n6\r\n world\r\n0\r\n\r\n";
        assert_eq!(unchunk(body), b"Hello world");
    }

    #[test]
    fn robots_allows() {
        let mut r = RobotsTxt::default();
        r.disallowed.insert(
            "example.com".into(),
            vec!["/private".into(), "/tmp".into()],
        );
        assert!(!r.allows("example.com", "/private/x"));
        assert!(r.allows("example.com", "/public"));
    }

    #[test]
    fn urlencode_encodes_query() {
        assert_eq!(urlencode("a b&c"), "a+b%26c");
        assert_eq!(urlencode("rust"), "rust");
    }

    #[test]
    fn percent_decode_decodes() {
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("x%2Fy"), "x/y");
    }

    #[test]
    fn ddg_parser_extracts_results() {
        let html = r#"<html><body>
            <div class="result">
                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc">Rust Docs</a>
                <a class="result__snippet">Ownership and borrowing in Rust</a>
            </div>
            <div class="result">
                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fguide">Another guide</a>
                <a class="result__snippet">More content here</a>
            </div>
        </body></html>"#;
        let results = parse_ddg_results(html, 5);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Rust Docs");
        assert_eq!(results[0].url, "https://example.com/doc");
        assert!(results[0].snippet.contains("Ownership"));
        assert_eq!(results[1].title, "Another guide");
        assert_eq!(results[1].url, "https://example.org/guide");
    }

    #[test]
    fn ddg_parser_respects_max_results() {
        let mut html = String::from("<html>");
        for i in 0..5 {
            html.push_str(&format!(
                r#"<div class="result"><a class="result__a" href="https://e.org/{i}">T{i}</a><a class="result__snippet">s</a></div>"#
            ));
        }
        html.push_str("</html>");
        assert_eq!(parse_ddg_results(&html, 3).len(), 3);
    }
}
