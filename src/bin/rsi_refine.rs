//! `rsi-refine` — CLI pour piloter une **session de raffinement** (boucle
//! d'ascension pilotée par LLM) depuis le shell, sans client MCP.
//!
//! L'état de la session est **persisté dans un fichier JSON** entre invocations
//! (réutilise `refine_save` / `refine_load` de [`rsi::api::RsiApi`]). Le contrat
//! de sûreté est inchangé : l'utilisateur (ou un script LLM) *propose* du texte,
//! le serveur parse, valide (`safety_check`), score et n'adopte qu'élitistement.
//!
//! ```text
//! rsi-refine new [--domain synthesis|tuning|prompt] [--target T] [--state F]
//! rsi-refine show [--state F]
//! rsi-refine eval  [--state F] "<candidat>"
//! rsi-refine propose [--state F] "<candidat>" ["<candidat>" ...]
//! ```

use std::process::exit;

use rsi::api::RsiApi;
use rsi::json::Json;

const DEFAULT_STATE: &str = ".rsi-refine.json";
const SESSION_ID: &str = "cli";
/// Drapeaux qui consomment la valeur suivante (pour isoler les positionnels).
const VALUE_FLAGS: &[&str] = &[
    "--state", "--domain", "--target", "--n", "--lo", "--hi", "--seed",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        exit(2);
    }
    let state_path = flag_value(&args, "--state").unwrap_or_else(|| DEFAULT_STATE.to_string());

    match args[1].as_str() {
        "new" => cmd_new(&args, &state_path),
        "show" | "incumbent" => cmd_show(&state_path),
        "eval" | "evaluate" => cmd_eval(&args, &state_path),
        "propose" => cmd_propose(&args, &state_path),
        "-h" | "--help" | "help" => usage(),
        other => {
            eprintln!("commande inconnue : '{other}'\n");
            usage();
            exit(2);
        }
    }
}

fn usage() {
    eprintln!(
        "rsi-refine — raffinement piloté par LLM en CLI\n\n\
         USAGE:\n  \
           rsi-refine new [--domain synthesis|tuning|prompt] [--target quadratic|linear|cubic] [--state F]\n  \
           rsi-refine show [--state F]\n  \
           rsi-refine eval  [--state F] \"<candidat>\"\n  \
           rsi-refine propose [--state F] \"<candidat>\" [\"<candidat>\" ...]\n\n\
         L'état est persisté dans {DEFAULT_STATE} (ou --state F).\n\
         Le serveur reste autoritaire : il parse, valide la sûreté, score et\n\
         n'adopte qu'élitistement (strictement meilleur ET sûr)."
    );
}

// --- commandes ---------------------------------------------------------- //

fn cmd_new(args: &[String], state_path: &str) {
    let mut params = Json::obj();
    params.set("id", Json::Str(SESSION_ID.into()));
    if let Some(d) = flag_value(args, "--domain") {
        params.set("domain", Json::Str(d));
    }
    if let Some(t) = flag_value(args, "--target") {
        params.set("target", Json::Str(t));
    }
    for (flag, key) in [
        ("--n", "n"),
        ("--lo", "lo"),
        ("--hi", "hi"),
        ("--seed", "seed"),
    ] {
        if let Some(v) = flag_value(args, flag).and_then(|s| s.parse::<f64>().ok()) {
            params.set(key, Json::Num(v));
        }
    }

    let mut api = RsiApi::new();
    let res = unwrap_or_die(api.handle("refine_new", &params));
    save_state(&mut api, state_path);
    if let Some(inc) = res.get("incumbent") {
        print_incumbent(inc);
    }
    eprintln!("(session écrite dans {state_path})");
}

fn cmd_show(state_path: &str) {
    let mut api = load_state(state_path);
    let res = unwrap_or_die(api.handle("incumbent", &id_params()));
    print_incumbent(&res);
}

fn cmd_eval(args: &[String], state_path: &str) {
    let cand = positionals(args).join(" ");
    if cand.trim().is_empty() {
        eprintln!("eval : fournir un candidat, p. ex. : rsi-refine eval \"x*x + 1\"");
        exit(2);
    }
    let mut api = load_state(state_path);
    let mut p = id_params();
    p.set("candidate", Json::Str(cand));
    let res = unwrap_or_die(api.handle("evaluate", &p));
    println!("{}", res.to_string());
}

fn cmd_propose(args: &[String], state_path: &str) {
    let cands = positionals(args);
    if cands.is_empty() {
        eprintln!("propose : fournir au moins un candidat");
        exit(2);
    }
    let mut api = load_state(state_path);
    let mut p = id_params();
    p.set(
        "proposals",
        Json::Arr(cands.into_iter().map(Json::Str).collect()),
    );
    let res = unwrap_or_die(api.handle("propose", &p));
    save_state(&mut api, state_path);

    let adopted = res
        .get("adopted")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if let Some(items) = res.get("results").and_then(|v| v.as_array()) {
        for it in items {
            let status = it.get("status").and_then(|v| v.as_str()).unwrap_or("?");
            let info = it
                .get("reason")
                .or_else(|| it.get("error"))
                .or_else(|| it.get("pretty"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            println!("  [{status}] {info}");
        }
    }
    println!("adopté : {adopted}");
    if let Some(inc) = res.get("incumbent") {
        print_incumbent(inc);
    }
}

// --- persistance (refine_save / refine_load) ---------------------------- //

fn load_state(path: &str) -> RsiApi {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|_| {
        eprintln!("état introuvable ({path}) ; lancez d'abord : rsi-refine new");
        exit(1);
    });
    let state = Json::parse(&raw).unwrap_or_else(|e| {
        eprintln!("état corrompu ({path}) : {e}");
        exit(1);
    });
    let mut api = RsiApi::new();
    let mut p = id_params();
    p.set("state", state);
    unwrap_or_die(api.handle("refine_load", &p));
    api
}

fn save_state(api: &mut RsiApi, path: &str) {
    let res = unwrap_or_die(api.handle("refine_save", &id_params()));
    let state = res.get("state").cloned().unwrap_or_else(Json::obj);
    if let Err(e) = std::fs::write(path, state.to_string()) {
        eprintln!("écriture de l'état impossible ({path}) : {e}");
        exit(1);
    }
}

// --- utilitaires -------------------------------------------------------- //

fn id_params() -> Json {
    let mut p = Json::obj();
    p.set("id", Json::Str(SESSION_ID.into()));
    p
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Arguments positionnels (hors sous-commande, drapeaux et leurs valeurs).
fn positionals(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 2; // saute le binaire + la sous-commande
    while i < args.len() {
        let a = &args[i];
        if VALUE_FLAGS.contains(&a.as_str()) {
            i += 2; // saute le drapeau et sa valeur
            continue;
        }
        if a.starts_with("--") {
            i += 1;
            continue;
        }
        out.push(a.clone());
        i += 1;
    }
    out
}

fn print_incumbent(inc: &Json) {
    let pretty = inc.get("pretty").and_then(|v| v.as_str()).unwrap_or("?");
    let score = inc
        .get("score")
        .and_then(|v| v.as_f64())
        .unwrap_or(f64::NAN);
    let heldout = inc
        .get("heldout")
        .and_then(|v| v.as_f64())
        .unwrap_or(f64::NAN);
    let domain = inc.get("domain").and_then(|v| v.as_str()).unwrap_or("?");
    println!("incumbent [{domain}] : {pretty}");
    println!("  score(train)={score:.3}  score(held-out)={heldout:.3}");
}

fn unwrap_or_die(res: Result<Json, String>) -> Json {
    res.unwrap_or_else(|e| {
        eprintln!("erreur : {e}");
        exit(1);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn flag_value_reads_value_after_flag() {
        let a = argv(&[
            "rsi-refine",
            "new",
            "--domain",
            "prompt",
            "--state",
            "s.json",
        ]);
        assert_eq!(flag_value(&a, "--domain"), Some("prompt".into()));
        assert_eq!(flag_value(&a, "--state"), Some("s.json".into()));
        assert_eq!(flag_value(&a, "--target"), None);
    }

    #[test]
    fn positionals_skip_flags_and_their_values() {
        // sous-commande + 2 drapeaux à valeur + 2 candidats positionnels
        let a = argv(&[
            "rsi-refine",
            "propose",
            "--state",
            "s.json",
            "x*x + 1",
            "--domain",
            "synthesis",
            "x + 2",
        ]);
        assert_eq!(
            positionals(&a),
            vec!["x*x + 1".to_string(), "x + 2".to_string()]
        );
    }
}
