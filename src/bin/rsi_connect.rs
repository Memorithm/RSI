//! `rsi-connect` — **auto-enregistrement** du serveur MCP RSI auprès des
//! runtimes d'agents IA (openclaw, hermes-agent, …) *sans intervention
//! humaine*.
//!
//! Le connecteur :
//!   1. localise le binaire `rsi-mcp` (env `RSI_MCP_BIN`, sinon `target/`,
//!      sinon le PATH) ;
//!   2. génère le descripteur MCP standard (`mcpServers`) ;
//!   3. **fusionne** ce descripteur dans les fichiers de configuration des
//!      runtimes cibles (en préservant les serveurs déjà déclarés) ;
//!   4. est *idempotent* : ré-exécutable à volonté (mise à jour en place).
//!
//! Cibles résolues par variables d'environnement (avec valeurs par défaut),
//! ce qui permet de lancer le connecteur automatiquement au démarrage d'un
//! conteneur / d'une session, sans configuration manuelle :
//!
//! | Runtime       | Variable d'env             | Défaut                                   |
//! |---------------|----------------------------|------------------------------------------|
//! | openclaw      | `OPENCLAW_CONFIG`          | `~/.openclaw/mcp.json`                   |
//! | hermes-agent  | `HERMES_AGENT_CONFIG`      | `~/.config/hermes-agent/mcp.json`        |
//! | soullink      | `SOULLINK_CONFIG`          | `~/.soullink/mcp.json`                   |
//! | SoulSystem    | `SOULSYSTEM_CONFIG`        | `~/.soulsystem/mcp.json`                 |
//! | générique MCP | `MCP_CONFIG`               | `~/.config/mcp/servers.json`             |
//! | *n'importe quel autre* | `RSI_CONNECT_TARGETS="nom=chemin,…"` | (aucun)              |
//!
//! Usage :
//! ```text
//! rsi-connect                 # enregistre auprès de toutes les cibles
//! rsi-connect --print         # affiche seulement le descripteur (pas d'écriture)
//! rsi-connect --name rsi      # nom de la clé serveur (défaut: rsi)
//! rsi-connect --bin /chemin/rsi-mcp   # force le chemin du binaire MCP
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rsi::json::Json;

struct Options {
    print_only: bool,
    server_name: String,
    bin_override: Option<String>,
}

fn parse_options() -> Options {
    let mut opt = Options {
        print_only: false,
        server_name: "rsi".to_string(),
        bin_override: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--print" => opt.print_only = true,
            "--name" => {
                if let Some(v) = it.next() {
                    opt.server_name = v;
                }
            }
            "--bin" => opt.bin_override = it.next(),
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
    }
    opt
}

fn print_help() {
    println!(
        "rsi-connect — auto-enregistre le serveur MCP RSI auprès des runtimes d'agents.\n\n\
Options:\n  --print        affiche le descripteur MCP sans rien écrire\n  \
--name NAME    clé du serveur dans la config (défaut: rsi)\n  \
--bin CHEMIN   chemin explicite du binaire rsi-mcp\n  -h, --help     cette aide"
    );
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Localise le binaire `rsi-mcp` de façon robuste.
fn locate_mcp_binary(override_path: &Option<String>) -> String {
    if let Some(p) = override_path {
        return p.clone();
    }
    if let Ok(p) = std::env::var("RSI_MCP_BIN") {
        return p;
    }
    // cherche relativement au répertoire courant / à l'exécutable
    let candidates = [
        PathBuf::from("target/release/rsi-mcp"),
        PathBuf::from("target/debug/rsi-mcp"),
    ];
    for c in &candidates {
        if c.exists() {
            if let Ok(abs) = std::fs::canonicalize(c) {
                return abs.to_string_lossy().into_owned();
            }
        }
    }
    // à côté de l'exécutable rsi-connect lui-même
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join("rsi-mcp");
            if sibling.exists() {
                return sibling.to_string_lossy().into_owned();
            }
        }
    }
    // dernier recours : suppose qu'il est sur le PATH
    "rsi-mcp".to_string()
}

/// Construit l'entrée serveur MCP standard pour RSI.
fn server_entry(bin: &str) -> Json {
    let mut entry = Json::obj();
    entry
        .set("command", Json::Str(bin.into()))
        .set("args", Json::Arr(vec![]))
        .set("env", Json::obj())
        .set(
            "description",
            Json::Str(
                "RSI — agent cognitif auto-améliorant (surface d'intelligence, \
substrat, méta-optimisation récursive sous garde-fous de stabilité)."
                    .into(),
            ),
        )
        .set("transport", Json::Str("stdio".into()));
    entry
}

/// Descripteur complet `{ "mcpServers": { "<name>": { … } } }`.
fn descriptor(name: &str, bin: &str) -> Json {
    let mut servers = Json::obj();
    servers.set(name, server_entry(bin));
    let mut root = Json::obj();
    root.set("mcpServers", servers);
    root
}

/// Fusionne le descripteur RSI dans un fichier de config existant (ou crée).
///
/// Préserve tout `mcpServers` préexistant et n'écrase que la clé `name`.
///
/// Sûretés :
/// - une config existante **non parsable** n'est jamais écrasée : sauvegarde
///   `.bak` puis refus (l'utilisateur doit corriger ou déplacer le fichier) —
///   l'ancien comportement remplaçait silencieusement le contenu ;
/// - écriture **atomique** (tmpfile dans le même répertoire + rename) : une
///   coupure ne laisse jamais une config tronquée.
fn register_into(path: &Path, name: &str, bin: &str) -> Result<bool, String> {
    // charge l'existant ; non-JSON ⇒ sauvegarde .bak et refus d'écraser
    let mut root = match std::fs::read_to_string(path) {
        Ok(content) if !content.trim().is_empty() => match Json::parse(&content) {
            Ok(j) => j,
            Err(e) => {
                let bak = path.with_extension("json.bak");
                std::fs::copy(path, &bak)
                    .map_err(|e| format!("sauvegarde {} : {e}", bak.display()))?;
                return Err(format!(
                    "{} n'est pas du JSON valide ({e}) ; original copié en {} — \
                     corrigez ou déplacez le fichier puis relancez",
                    path.display(),
                    bak.display()
                ));
            }
        },
        _ => Json::obj(),
    };

    // assure root.mcpServers (objet)
    let existing_servers = root.get("mcpServers").cloned().unwrap_or_else(Json::obj);
    let mut servers_map: BTreeMap<String, Json> = match existing_servers {
        Json::Obj(m) => m,
        _ => BTreeMap::new(),
    };
    servers_map.insert(name.to_string(), server_entry(bin));

    if !matches!(root, Json::Obj(_)) && root != Json::obj() {
        // racine non-objet déjà couverte ci-dessus si non parsable ; ici une
        // racine JSON valide mais non objet (tableau, scalaire…) : on refuse
        // plutôt que de perdre le contenu.
        return Err(format!(
            "{} a une racine JSON non-objet ; fusion refusée",
            path.display()
        ));
    }
    if let Json::Obj(root_map) = &mut root {
        root_map.insert("mcpServers".to_string(), Json::Obj(servers_map));
    }

    // crée les répertoires parents si besoin
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("création de {} : {e}", parent.display()))?;
        }
    }

    // sérialisation lisible (indentée 2 espaces), écrite ATOMIQUEMENT :
    // tmpfile voisin + rename (une coupure ne laisse pas de config tronquée)
    let pretty = pretty_print(&root, 0);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, pretty + "\n")
        .map_err(|e| format!("écriture de {} : {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename vers {} : {e}", path.display())
    })?;

    // sécurité : restreint l'accès au fichier (lecture/écriture propriétaire)
    restrict_permissions(path)?;
    Ok(true)
}

/// Restreint les permissions du fichier de config à `0600` (Unix).
#[cfg(unix)]
fn restrict_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let perms = std::fs::Permissions::from_mode(0o600);
    std::fs::set_permissions(path, perms)
        .map_err(|e| format!("set_permissions {} : {e}", path.display()))
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) -> Result<(), String> {
    Ok(())
}

/// Sérialiseur JSON indenté (lisibilité des fichiers de config).
fn pretty_print(v: &Json, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    let pad1 = "  ".repeat(indent + 1);
    match v {
        Json::Obj(m) if !m.is_empty() => {
            let mut s = String::from("{\n");
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 {
                    s.push_str(",\n");
                }
                s.push_str(&format!(
                    "{pad1}{}: {}",
                    Json::Str(k.clone()).to_string(),
                    pretty_print(val, indent + 1)
                ));
            }
            s.push_str(&format!("\n{pad}}}"));
            s
        }
        Json::Arr(a) if !a.is_empty() => {
            let mut s = String::from("[\n");
            for (i, val) in a.iter().enumerate() {
                if i > 0 {
                    s.push_str(",\n");
                }
                s.push_str(&format!("{pad1}{}", pretty_print(val, indent + 1)));
            }
            s.push_str(&format!("\n{pad}]"));
            s
        }
        other => other.to_string(),
    }
}

fn target_path(env_var: &str, default_rel: &[&str]) -> PathBuf {
    if let Ok(p) = std::env::var(env_var) {
        return PathBuf::from(p);
    }
    let mut p = home();
    for seg in default_rel {
        p.push(seg);
    }
    p
}

fn main() {
    let opt = parse_options();
    let bin = locate_mcp_binary(&opt.bin_override);

    if opt.print_only {
        println!("{}", pretty_print(&descriptor(&opt.server_name, &bin), 0));
        return;
    }

    println!(
        "rsi-connect : enregistrement du serveur MCP « {} »",
        opt.server_name
    );
    println!("  binaire MCP : {bin}");
    if bin == "rsi-mcp" {
        eprintln!(
            "  ⚠ binaire introuvable dans target/ ; on suppose qu'il est sur le PATH.\n\
             \x20   (compilez avec `cargo build --release` ou passez --bin /chemin/rsi-mcp)"
        );
    }

    let mut targets: Vec<(String, PathBuf)> = vec![
        (
            "openclaw".into(),
            target_path("OPENCLAW_CONFIG", &[".openclaw", "mcp.json"]),
        ),
        (
            "hermes-agent".into(),
            target_path(
                "HERMES_AGENT_CONFIG",
                &[".config", "hermes-agent", "mcp.json"],
            ),
        ),
        (
            "soullink".into(),
            target_path("SOULLINK_CONFIG", &[".soullink", "mcp.json"]),
        ),
        (
            "SoulSystem".into(),
            target_path("SOULSYSTEM_CONFIG", &[".soulsystem", "mcp.json"]),
        ),
        (
            "mcp (générique)".into(),
            target_path("MCP_CONFIG", &[".config", "mcp", "servers.json"]),
        ),
    ];
    // Cibles arbitraires : RSI_CONNECT_TARGETS="nom=chemin,nom2=chemin2" —
    // n'importe quel runtime à config `mcpServers` s'ajoute sans recompiler.
    if let Ok(extra) = std::env::var("RSI_CONNECT_TARGETS") {
        for pair in extra.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match pair.split_once('=') {
                Some((name, path)) if !name.is_empty() && !path.is_empty() => {
                    targets.push((name.trim().to_string(), PathBuf::from(path.trim())));
                }
                _ => eprintln!(
                    "  ⚠ RSI_CONNECT_TARGETS : entrée ignorée « {pair} » (attendu nom=chemin)"
                ),
            }
        }
    }

    let mut ok = 0usize;
    for (runtime, path) in &targets {
        match register_into(path, &opt.server_name, &bin) {
            Ok(_) => {
                println!("  ✓ {runtime:<16} → {}", path.display());
                ok += 1;
            }
            Err(e) => eprintln!("  ✗ {runtime:<16} → {e}"),
        }
    }

    println!(
        "\nTerminé : {ok}/{} cible(s) enregistrée(s). Les runtimes chargeront\n\
le serveur RSI au prochain démarrage — aucune action manuelle requise.",
        targets.len()
    );

    // code de sortie exploitable par les scripts appelants (install.sh, hooks) :
    // échec si AUCUNE cible n'a pu être enregistrée
    if ok == 0 && !targets.is_empty() {
        std::process::exit(1);
    }
}
