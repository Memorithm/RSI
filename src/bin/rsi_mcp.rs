//! Serveur **MCP** (Model Context Protocol) pour le système RSI.
//!
//! Transport stdio, messages JSON-RPC 2.0 délimités par des sauts de ligne.
//! Expose la [`RsiApi`] sous forme d'outils MCP, permettant à un agent IA / LLM
//! de créer un agent auto-améliorant, de l'avancer, d'inspecter son état et
//! d'exporter sa trajectoire.
//!
//! Méthodes JSON-RPC gérées :
//! - `initialize`               → handshake (capacités + infos serveur)
//! - `notifications/initialized`→ notification (ignorée)
//! - `ping`                     → `{}`
//! - `tools/list`               → catalogue d'outils (JSON Schema)
//! - `tools/call`               → exécute un outil et renvoie son résultat
//!
//! Lancement : `cargo run --release --bin rsi-mcp` puis dialogue JSON-RPC
//! ligne par ligne sur stdin/stdout.

use std::io::{self, BufRead, Read, Write};

use rsi::api::RsiApi;
use rsi::json::Json;

const PROTOCOL_VERSION: &str = "2024-11-05";

/// Plafond de taille d'une requête (une ligne JSON-RPC). Un client hostile
/// pourrait sinon envoyer une ligne de plusieurs Go et provoquer un OOM avant
/// même le parsing.
const MAX_LINE_BYTES: usize = 16 * 1024 * 1024; // 16 Mio

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut reader = stdin.lock();
    let mut api = RsiApi::new();
    let dgm = dgm_job::shared();

    let mut buf = Vec::with_capacity(4096);
    loop {
        buf.clear();
        // Lecture *bornée* d'une ligne : `take` plafonne CETTE lecture à
        // MAX_LINE_BYTES octets (anti-OOM sur ligne géante).
        let n = match (&mut reader).take(MAX_LINE_BYTES as u64).read_until(b'\n', &mut buf) {
            Ok(0) => break, // EOF
            Ok(n) => n,
            Err(_) => break,
        };
        // Limite atteinte sans fin de ligne ⇒ requête trop grande : on refuse et
        // on coupe proprement (drainer le reste risquerait l'OOM qu'on évite).
        if n >= MAX_LINE_BYTES && buf.last() != Some(&b'\n') {
            let resp = error_response(&Json::Null, -32600, "request line exceeds 16 MiB limit");
            let _ = writeln!(out, "{}", resp.to_string());
            let _ = out.flush();
            break;
        }

        let line = String::from_utf8_lossy(&buf);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let response = match Json::parse(line) {
            Ok(req) => handle_request(&mut api, &dgm, &req),
            Err(e) => Some(error_response(&Json::Null, -32700, &format!("parse error: {e}"))),
        };

        if let Some(resp) = response {
            let _ = writeln!(out, "{}", resp.to_string());
            let _ = out.flush();
        }
    }
}

/// Dispatche une requête JSON-RPC. Renvoie `None` pour les notifications.
fn handle_request(api: &mut RsiApi, dgm: &dgm_job::Shared, req: &Json) -> Option<Json> {
    let id = req.get("id").cloned().unwrap_or(Json::Null);
    let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let is_notification = req.get("id").is_none();

    let result: Result<Json, (i64, String)> = match method {
        "initialize" => Ok(initialize_result()),
        "notifications/initialized" | "initialized" => return None,
        "ping" => Ok(Json::obj()),
        "tools/list" => Ok(tools_list()),
        "tools/call" => tools_call(api, dgm, req.get("params")),
        other => Err((-32601, format!("méthode inconnue : '{other}'"))),
    };

    if is_notification {
        return None;
    }

    Some(match result {
        Ok(value) => success_response(&id, value),
        Err((code, msg)) => error_response(&id, code, &msg),
    })
}

fn initialize_result() -> Json {
    let mut caps = Json::obj();
    caps.set("tools", Json::obj());

    let mut server = Json::obj();
    server
        .set("name", Json::Str("rsi-mcp".into()))
        .set("version", Json::Str(env!("CARGO_PKG_VERSION").into()));

    let mut out = Json::obj();
    out.set("protocolVersion", Json::Str(PROTOCOL_VERSION.into()))
        .set("capabilities", caps)
        .set("serverInfo", server)
        .set(
            "instructions",
            Json::Str(
                "Système d'auto-amélioration récursive (RSI). Utilisez \
'rsi_create' pour instancier un agent, 'rsi_run'/'rsi_step' pour le faire \
évoluer, 'rsi_state' pour l'inspecter, 'rsi_export' pour récupérer la \
trajectoire. 'rsi_describe' documente le modèle mathématique."
                    .into(),
            ),
        );
    out
}

/// Description d'un outil MCP (name, description, inputSchema JSON Schema).
fn tool(name: &str, description: &str, properties: Json, required: &[&str]) -> Json {
    let mut schema = Json::obj();
    schema.set("type", Json::Str("object".into())).set("properties", properties);
    if !required.is_empty() {
        let req: Vec<Json> = required.iter().map(|s| Json::Str((*s).into())).collect();
        schema.set("required", Json::Arr(req));
    }
    let mut t = Json::obj();
    t.set("name", Json::Str(name.into()))
        .set("description", Json::Str(description.into()))
        .set("inputSchema", schema);
    t
}

fn prop(ty: &str, desc: &str) -> Json {
    let mut p = Json::obj();
    p.set("type", Json::Str(ty.into())).set("description", Json::Str(desc.into()));
    p
}

fn props(pairs: &[(&str, Json)]) -> Json {
    let mut o = Json::obj();
    for (k, v) in pairs {
        o.set(k, v.clone());
    }
    o
}

fn tools_list() -> Json {
    let id = || ("id", prop("string", "Identifiant de session (défaut: 'default')."));

    // Les outils DGM (rsi_dgm_start/status) ne sont listés que si la feature
    // `llm-ollama` est compilée — sinon ils échoueraient systématiquement.
    #[cfg_attr(not(feature = "llm-ollama"), allow(unused_mut))]
    let mut tools = vec![
        tool(
            "rsi_describe",
            "Décrit le système RSI (modèle mathématique) et le catalogue de commandes.",
            Json::obj(),
            &[],
        ),
        tool(
            "rsi_create",
            "Crée (ou remplace) une session d'agent RSI auto-améliorant.",
            props(&[
                id(),
                ("seed", prop("integer", "Graine reproductible (défaut 2026).")),
                ("optimizer", prop("string", "Méta-optimiseur: 'random' ou 'cma' (sep-CMA-ES).")),
                ("dim", prop("integer", "Dimension de chaque composante de S (défaut 6).")),
                ("n_tasks", prop("integer", "Taille de l'échantillon de tâches |Ω| (défaut 1024).")),
                ("n_hardware", prop("integer", "Dimension du vecteur matériel H (défaut 4).")),
                ("n_software", prop("integer", "Dimension du vecteur logiciel O (défaut 4).")),
                ("lambda", prop("number", "Borne de stabilité ‖ΔS‖<λ (défaut 0.5).")),
                ("epsilon", prop("number", "Régression tolérée sur SI_global (défaut 1e-3).")),
            ]),
            &[],
        ),
        tool(
            "rsi_step",
            "Avance la boucle RSI d'un pas et renvoie le rapport du pas.",
            props(&[id()]),
            &[],
        ),
        tool(
            "rsi_run",
            "Avance la boucle RSI de plusieurs pas et renvoie un résumé (gain de SI_global).",
            props(&[id(), ("steps", prop("integer", "Nombre de pas (défaut 100)."))]),
            &[],
        ),
        tool(
            "rsi_run_until",
            "Pilote la boucle RSI jusqu'à un critère d'arrêt motivé (budget, cible, plateau, disjoncteur de criticité).",
            props(&[
                id(),
                ("max_steps", prop("integer", "Budget de pas (défaut 500).")),
                ("target_si", prop("number", "Arrêt si SI_global ≥ cette valeur.")),
                ("plateau_window", prop("integer", "Fenêtre de détection de plateau (défaut 12).")),
                ("plateau_eps", prop("number", "Seuil de pente sous lequel = plateau.")),
                ("breaker_rpn", prop("number", "Disjoncteur : arrêt/rollback si max_rpn dépasse ce seuil.")),
                ("max_seconds", prop("number", "Budget de temps (s).")),
            ]),
            &[],
        ),
        tool(
            "rsi_state",
            "Renvoie un instantané: SI_global, P_eff, capacités (D,M,R,A,C,V), goulot d'étranglement.",
            props(&[id()]),
            &[],
        ),
        tool(
            "rsi_export",
            "Exporte la trajectoire accumulée de la session au format csv ou json.",
            props(&[id(), ("format", prop("string", "'csv' ou 'json' (défaut json)."))]),
            &[],
        ),
        tool("rsi_reset", "Réinitialise la session à partir de sa configuration.", props(&[id()]), &[]),
        tool("rsi_list_sessions", "Liste les sessions actives.", Json::obj(), &[]),
        tool(
            "rsi_health",
            "Santé du serveur : nombre de sessions, pas cumulés, intégrité de l'audit \
             hash-chaîné (audit_intact), activité de raffinement.",
            Json::obj(),
            &[],
        ),
        tool(
            "rsi_metrics",
            "Métriques au format d'exposition Prometheus (texte, scrapeable) : sessions, \
             pas cumulés, intégrité de l'audit, propositions vues/adoptées.",
            Json::obj(),
            &[],
        ),
        tool(
            "rsi_refine_new",
            "Crée une session de raffinement piloté par LLM. Le LLM propose des candidats (texte) ; \
             le serveur les évalue en sandbox et n'adopte que les strictement meilleurs et sûrs — \
             le LLM ne contrôle aucun garde-fou. Deux domaines : 'synthesis' (expressions \
             arithmétiques), 'tuning' (hyperparamètres JSON) et 'prompt' (prompt texte).",
            props(&[
                id(),
                ("domain", prop("string", "Domaine: 'synthesis' (défaut), 'tuning' ou 'prompt'.")),
                ("target", prop("string", "[synthesis] cible: 'quadratic' (x²+1), 'linear' (2x-1), 'cubic' (x³-x).")),
                ("lo", prop("number", "[synthesis] borne basse d'échantillonnage (défaut -3).")),
                ("hi", prop("number", "[synthesis] borne haute (défaut 3).")),
                ("n", prop("integer", "[synthesis] nombre de points (défaut 30 ; ~30% en held-out).")),
                ("seed", prop("integer", "[synthesis] graine reproductible (défaut 2026).")),
            ]),
            &[],
        ),
        tool(
            "rsi_incumbent",
            "Renvoie l'incumbent courant d'une session de raffinement : expression, score \
             (fraction de cas réussis), score held-out (généralisation), taille, compteurs.",
            props(&[id()]),
            &[],
        ),
        tool(
            "rsi_evaluate",
            "Évalue un candidat SANS l'adopter (sonde pour le raisonnement). Renvoie score, \
             held-out, sûreté et s'il serait adopté. N'altère pas l'incumbent.",
            props(&[
                id(),
                ("candidate", prop("string", "Candidat texte. synthesis: expression sur x (ex: 'x*x + 1'). tuning: config JSON (ex: '{\"top_k\":50,\"chunk\":1024,\"threshold\":0.4}').")),
            ]),
            &["candidate"],
        ),
        tool(
            "rsi_propose",
            "Soumet une ou plusieurs propositions d'expressions. Le serveur parse, vérifie la \
             sûreté (taille bornée), score, et n'adopte qu'élitistement (strictement meilleur ET \
             sûr). Renvoie le détail par proposition et le nouvel incumbent.",
            props(&[
                id(),
                ("proposals", prop("array", "Tableau de candidats (chaînes) : expressions (synthesis) ou configs JSON (tuning).")),
                ("candidate", prop("string", "Alternative: un ou plusieurs candidats, un par ligne.")),
            ]),
            &[],
        ),
        tool(
            "rsi_refine_save",
            "Sérialise l'état d'une session de raffinement (config + incumbent + compteurs) \
             pour reprise ultérieure. Renvoie un objet 'state' réutilisable par rsi_refine_load.",
            props(&[id()]),
            &[],
        ),
        tool(
            "rsi_refine_load",
            "Reprend une session de raffinement depuis un 'state' produit par rsi_refine_save \
             (checkpoint/resume d'une optimisation pilotée par LLM).",
            props(&[
                id(),
                ("state", prop("object", "État sérialisé renvoyé par rsi_refine_save.")),
            ]),
            &["state"],
        ),
    ];

    // Outils DGM — listés uniquement sous la feature `llm-ollama` (sinon
    // `dgm_job::start` renvoie systématiquement « outil indisponible »).
    #[cfg(feature = "llm-ollama")]
    {
        tools.push(tool(
            "rsi_dgm_start",
            "Lance en ARRIÈRE-PLAN une boucle d'auto-amélioration DGM/STOP sur un dépôt réel : \
             le LLM (connexion automatique : modèle Ollama découvert, ou 'model') propose des \
             patchs, chacun évalué en copie isolée (cargo build+test, bench optionnel), archivé \
             seulement s'il est strictement meilleur. DRY-RUN STRICT : rien n'est jamais écrit \
             dans l'arbre vivant depuis MCP — la promotion reste un acte humain en CLI \
             (rsi-dgm --promote) après revue du diff. Suivre via rsi_dgm_status. \
             Un seul job à la fois.",
            props(&[
                ("workspace", prop("string", "Racine du dépôt (répertoire).")),
                ("goal", prop("string", "Objectif d'amélioration remis au proposeur (directif = plus efficace).")),
                ("allow", prop("string", "Fichiers éditables, séparés par des virgules.")),
                ("steps", prop("integer", "Étapes (défaut 8, max 64) — ~1-3 min/étape.")),
                ("bench", prop("string", "Args cargo d'un bench imprimant RSI_BENCH_SCORE=<f64> (score = perf réelle).")),
                ("min_gain", prop("number", "Gain relatif minimal anti-bruit (défaut 0.05).")),
                ("model", prop("string", "Préférence de modèle (sinon découverte automatique /api/tags).")),
                ("seed", prop("integer", "Graine déterministe (défaut 42).")),
                ("timeout_secs", prop("integer", "Borne par invocation cargo (défaut 300, max 1800).")),
            ]),
            &["workspace", "goal", "allow"],
        ));
        tools.push(tool(
            "rsi_dgm_status",
            "État du job DGM d'arrière-plan : phase courante, résultats par étape (accepté/rejeté, \
             fitness, raison), meilleur variant promouvable (dry-run), erreur éventuelle.",
            Json::obj(),
            &[],
        ));
    }

    let mut out = Json::obj();
    out.set("tools", Json::Arr(tools));
    out
}

/// Mappe `tools/call` vers une commande de l'API (ou le job DGM).
fn tools_call(
    api: &mut RsiApi,
    dgm: &dgm_job::Shared,
    params: Option<&Json>,
) -> Result<Json, (i64, String)> {
    let params = params.ok_or((-32602, "params manquants".into()))?;
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "nom d'outil manquant".into()))?;
    let empty = Json::obj();
    let args = params.get("arguments").unwrap_or(&empty);

    // Outils du job DGM (gérés hors RsiApi : thread d'arrière-plan).
    match name {
        "rsi_dgm_start" => {
            return Ok(match dgm_job::start(dgm, args) {
                Ok(v) => tool_text_result(&v, false),
                Err(e) => tool_text_result(&Json::Str(e), true),
            })
        }
        "rsi_dgm_status" => return Ok(tool_text_result(&dgm_job::status(dgm), false)),
        _ => {}
    }

    // outils préfixés 'rsi_' → commandes de l'API
    let command = name.strip_prefix("rsi_").unwrap_or(name);

    match api.handle(command, args) {
        Ok(value) => Ok(tool_text_result(&value, false)),
        Err(e) => Ok(tool_text_result(&Json::Str(e), true)),
    }
}

/// Emballe un résultat dans le format `content` attendu par MCP.
fn tool_text_result(value: &Json, is_error: bool) -> Json {
    let mut text = Json::obj();
    text.set("type", Json::Str("text".into()))
        .set("text", Json::Str(value.to_string()));
    let mut out = Json::obj();
    out.set("content", Json::Arr(vec![text]));
    if is_error {
        out.set("isError", Json::Bool(true));
    }
    out
}

// ═══════════════ Job DGM d'arrière-plan (auto-amélioration) ═══════════════ //
//
// La boucle DGM (build+test+bench par étape) dure des minutes : un appel
// d'outil MCP bloquant serait inutilisable. `rsi_dgm_start` lance donc UN
// job en arrière-plan (thread), `rsi_dgm_status` s'interroge à volonté.
//
// **Sûreté : DRY-RUN UNIQUEMENT.** Aucun outil MCP ne promeut de patch vers
// l'arbre vivant — la promotion (`rsi-dgm --promote`) reste un acte humain
// délibéré en CLI, après revue du diff. Un agent-framework peut chercher des
// améliorations, jamais les appliquer.
mod dgm_job {
    use rsi::json::Json;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    pub struct JobState {
        pub running: bool,
        pub goal: String,
        pub model: String,
        pub phase: String,
        pub outcomes: Vec<Json>,
        pub best: Option<Json>,
        pub error: Option<String>,
    }

    pub type Shared = Arc<Mutex<JobState>>;

    pub fn shared() -> Shared {
        Arc::new(Mutex::new(JobState::default()))
    }

    pub fn status(shared: &Shared) -> Json {
        let st = shared.lock().unwrap();
        let mut out = Json::obj();
        out.set("running", Json::Bool(st.running))
            .set("goal", Json::Str(st.goal.clone()))
            .set("model", Json::Str(st.model.clone()))
            .set("phase", Json::Str(st.phase.clone()))
            .set("steps_done", Json::Num(st.outcomes.len() as f64))
            .set("outcomes", Json::Arr(st.outcomes.clone()));
        if let Some(b) = &st.best {
            out.set("best", b.clone());
        }
        if let Some(e) = &st.error {
            out.set("error", Json::Str(e.clone()));
        }
        out
    }

    #[cfg(not(feature = "llm-ollama"))]
    pub fn start(_shared: &Shared, _args: &Json) -> Result<Json, String> {
        Err("outil indisponible : recompiler rsi-mcp avec --features llm-ollama".to_string())
    }

    #[cfg(feature = "llm-ollama")]
    pub fn start(shared: &Shared, args: &Json) -> Result<Json, String> {
        use rsi::dgm::{
            Archive, CargoEvaluator, DgmConfig, DgmEngine, Evaluator, LlmCodeModel, LlmProposer,
            StepOutcome, WorkspaceSnapshot,
        };
        use std::time::Duration;

        {
            let st = shared.lock().unwrap();
            if st.running {
                return Err("un job DGM tourne déjà — interroger rsi_dgm_status".to_string());
            }
        }

        // --- Paramètres (bornés : entrée non fiable). ----------------------- //
        let workspace = args
            .get("workspace")
            .and_then(|v| v.as_str())
            .ok_or("paramètre 'workspace' requis (racine du dépôt)")?
            .to_string();
        let goal = args
            .get("goal")
            .and_then(|v| v.as_str())
            .ok_or("paramètre 'goal' requis")?
            .to_string();
        let allowed: Vec<String> = args
            .get("allow")
            .and_then(|v| v.as_str())
            .ok_or("paramètre 'allow' requis (fichiers éditables, séparés par ',')")?
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if allowed.is_empty() {
            return Err("'allow' doit lister au moins un fichier".to_string());
        }
        let steps = args.get("steps").and_then(|v| v.as_usize()).unwrap_or(8).clamp(1, 64);
        let seed = args.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);
        let min_gain = args.get("min_gain").and_then(|v| v.as_f64()).unwrap_or(0.05);
        let timeout_secs =
            args.get("timeout_secs").and_then(|v| v.as_u64()).unwrap_or(300).clamp(30, 1800);
        let bench: Vec<String> = args
            .get("bench")
            .and_then(|v| v.as_str())
            .map(|s| s.split_whitespace().map(str::to_string).collect())
            .unwrap_or_default();
        let ws = std::path::PathBuf::from(&workspace);
        if !ws.is_dir() {
            return Err(format!("'{workspace}' n'est pas un répertoire"));
        }

        // --- Connexion LLM automatique (échoue TÔT, avant le thread). ------- //
        let host = std::env::var("RSI_OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port: u16 = std::env::var("RSI_OLLAMA_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(11434);
        let pref = args
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .or_else(|| std::env::var("RSI_LLM_MODEL").ok());
        let installed = rsi::llm::ollama_installed_models(&host, port, Duration::from_secs(10))
            .map_err(|e| format!("Ollama injoignable sur {host}:{port} ({e:?})"))?;
        let model = rsi::llm::pick_model(&installed, pref.as_deref())
            .ok_or_else(|| format!("aucun modèle utilisable sur {host}:{port}"))?;

        {
            let mut st = shared.lock().unwrap();
            *st = JobState {
                running: true,
                goal: goal.clone(),
                model: model.clone(),
                phase: "référence (build+test de l'arbre vivant)".to_string(),
                ..JobState::default()
            };
        }

        let state = Arc::clone(shared);
        let model_name = model.clone();
        std::thread::spawn(move || {
            let fail = |state: &Shared, msg: String| {
                let mut st = state.lock().unwrap();
                st.error = Some(msg);
                st.running = false;
                st.phase = "échec".to_string();
            };

            let client = rsi::llm::OllamaClient::new(model_name)
                .with_endpoint(host, port)
                .with_timeout(Duration::from_secs(timeout_secs));
            let proposer = LlmProposer::new(LlmCodeModel::new(client), allowed);
            let evaluator = CargoEvaluator {
                bench_command: bench,
                timeout: Duration::from_secs(timeout_secs),
                score_from_passrate: true,
                ..Default::default()
            };

            let baseline =
                match WorkspaceSnapshot::create(&ws).and_then(|s| evaluator.evaluate(s.root())) {
                    Ok(f) => f,
                    Err(e) => return fail(&state, format!("évaluation de référence : {e}")),
                };
            let mut config = DgmConfig::new(&ws, &goal);
            config.min_score_gain = min_gain;
            let mut engine =
                DgmEngine::new(Archive::with_root(baseline), proposer, evaluator, config, seed);

            for i in 0..steps {
                {
                    let mut st = state.lock().unwrap();
                    st.phase = format!("étape {}/{steps}", i + 1);
                }
                let outcome = match engine.step() {
                    Ok(o) => o,
                    Err(e) => return fail(&state, format!("étape {i} : {e}")),
                };
                let mut j = Json::obj();
                j.set("step", Json::Num(i as f64));
                match &outcome {
                    StepOutcome::NoProposal { reason } => {
                        j.set("kind", Json::Str("no_proposal".into()));
                        if let Some(r) = reason {
                            j.set("reason", Json::Str(r.clone()));
                        }
                    }
                    StepOutcome::Evaluated { accepted, fitness, variant_id, .. } => {
                        j.set("kind", Json::Str("evaluated".into()))
                            .set("accepted", Json::Bool(*accepted))
                            .set("compiles", Json::Bool(fitness.compiles))
                            .set("tests_passed", Json::Num(fitness.tests_passed as f64))
                            .set("tests_failed", Json::Num(fitness.tests_failed as f64))
                            .set("score", Json::Num(fitness.score))
                            .set("variant", Json::Str(variant_id.clone()))
                            .set(
                                "reason",
                                Json::Str(
                                    fitness.notes.lines().next().unwrap_or("").trim().to_string(),
                                ),
                            );
                    }
                }
                state.lock().unwrap().outcomes.push(j);
            }

            let best = engine.best().map(|v| {
                let mut b = Json::obj();
                b.set("variant", Json::Str(v.id.clone()))
                    .set("target", Json::Str(v.patch.target.clone()))
                    .set("rationale", Json::Str(v.rationale.clone()));
                if let Some(f) = &v.fitness {
                    b.set("score", Json::Num(f.score))
                        .set("tests_passed", Json::Num(f.tests_passed as f64));
                }
                b.set(
                    "note",
                    Json::Str(
                        "DRY-RUN : rien n'a été appliqué. Promotion = acte humain : \
                         rsi-dgm <ws> … --promote, après revue du diff."
                            .into(),
                    ),
                );
                b
            });
            let mut st = state.lock().unwrap();
            st.best = best;
            st.running = false;
            st.phase = "terminé".to_string();
        });

        let mut out = Json::obj();
        out.set("started", Json::Bool(true))
            .set("model", Json::Str(model))
            .set("steps", Json::Num(steps as f64))
            .set("dry_run", Json::Bool(true));
        Ok(out)
    }
}

fn success_response(id: &Json, result: Json) -> Json {
    let mut out = Json::obj();
    out.set("jsonrpc", Json::Str("2.0".into()))
        .set("id", id.clone())
        .set("result", result);
    out
}

fn error_response(id: &Json, code: i64, message: &str) -> Json {
    let mut err = Json::obj();
    err.set("code", Json::Num(code as f64)).set("message", Json::Str(message.into()));
    let mut out = Json::obj();
    out.set("jsonrpc", Json::Str("2.0".into()))
        .set("id", id.clone())
        .set("error", err);
    out
}
