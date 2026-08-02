//! FLYWHEEL — assemblage du dataset d'entraînement (axe 4 → fine-tune).
//!
//! Le module [`crate::trajectory`] exporte, à chaque run DGM, des paires
//! `{prompt, completion}` à **vérité terrain** (une par évaluation réelle du
//! gate). Ce module **assemble le dataset** à travers plusieurs runs/cibles :
//! fusion, **déduplication**, mesure de l'**équilibre des classes** (verdict),
//! **split train/eval** déterministe, et **conversion au format chat**
//! (`{messages:[…]}`) directement consommable par unsloth / llama-factory pour
//! spécialiser un world model du dépôt RSI (calibration v2).
//!
//! Std-only, sans dépendance. Opère sur les **lignes JSONL brutes** : le tag
//! `SIMCAL_VERDICT` (sans caractère à échapper) survit tel quel dans la ligne,
//! donc la classification et l'équilibrage se font par sous-chaîne, sans
//! re-parser le JSON. La conversion chat ré-emballe les valeurs **déjà
//! échappées** (extraction respectant les `\"`), sans re-échapper.

use crate::rng::Rng;

/// Classe de verdict d'une paire (lue dans la ligne `SIMCAL_VERDICT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// compile + tests verts.
    CompilePass,
    /// compile mais tests rouges.
    TestsFail,
    /// ne compile pas.
    CompileFail,
    /// aucun tag reconnu.
    Unknown,
}

/// Classe une ligne JSONL par son tag `SIMCAL_VERDICT`.
pub fn classify_line(line: &str) -> Verdict {
    if line.contains("compile=oui; tests=pass") {
        Verdict::CompilePass
    } else if line.contains("compile=oui; tests=fail") {
        Verdict::TestsFail
    } else if line.contains("compile=non") {
        Verdict::CompileFail
    } else {
        Verdict::Unknown
    }
}

/// Bilan d'un dataset (comptes par classe + doublons retirés).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DatasetStats {
    pub total: usize,
    pub compile_pass: usize,
    pub tests_fail: usize,
    pub compile_fail: usize,
    pub unknown: usize,
    pub duplicates_removed: usize,
}

impl DatasetStats {
    /// La classe minoritaire pèse-t-elle au moins `frac` du total ? Un world
    /// model entraîné sur un dataset déséquilibré apprend surtout la majorité.
    pub fn is_balanced(&self, frac: f64) -> bool {
        if self.total == 0 {
            return false;
        }
        let min = self
            .compile_pass
            .min(self.tests_fail)
            .min(self.compile_fail);
        (min as f64 / self.total as f64) >= frac
    }

    pub fn summary(&self) -> String {
        format!(
            "dataset : {} paires ({} doublon(s) retiré(s))\n  \
             compile+pass : {}\n  compile+tests_fail : {}\n  ne compile pas : {}\n  \
             inconnu : {}\n  équilibre (classe min ≥ 20 %) : {}",
            self.total,
            self.duplicates_removed,
            self.compile_pass,
            self.tests_fail,
            self.compile_fail,
            self.unknown,
            if self.is_balanced(0.20) {
                "oui"
            } else {
                "NON (accumuler des cibles variées)"
            },
        )
    }
}

/// Bilan des classes sur des lignes déjà dédupliquées.
pub fn stats(lines: &[String], duplicates_removed: usize) -> DatasetStats {
    let mut s = DatasetStats {
        total: lines.len(),
        duplicates_removed,
        ..Default::default()
    };
    for l in lines {
        match classify_line(l) {
            Verdict::CompilePass => s.compile_pass += 1,
            Verdict::TestsFail => s.tests_fail += 1,
            Verdict::CompileFail => s.compile_fail += 1,
            Verdict::Unknown => s.unknown += 1,
        }
    }
    s
}

/// Déduplique des lignes exactes en **préservant l'ordre**. Rend
/// `(uniques, nb_doublons_retirés)`. Les lignes vides sont ignorées.
pub fn dedup(lines: Vec<String>) -> (Vec<String>, usize) {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(lines.len());
    let mut removed = 0;
    for l in lines {
        if l.trim().is_empty() {
            continue;
        }
        if seen.insert(l.clone()) {
            out.push(l);
        } else {
            removed += 1;
        }
    }
    (out, removed)
}

/// Mélange Fisher–Yates déterministe (graine) puis coupe : rend `(train, eval)`
/// avec `eval_frac` (borné [0,1]) de lignes en éval. Déterministe pour une même
/// graine (reproductibilité des splits).
pub fn split(lines: &[String], eval_frac: f64, seed: u64) -> (Vec<String>, Vec<String>) {
    let mut v = lines.to_vec();
    let mut rng = Rng::new(seed);
    for i in (1..v.len()).rev() {
        let j = (rng.uniform() * (i as f64 + 1.0)) as usize;
        let j = j.min(i);
        v.swap(i, j);
    }
    let frac = eval_frac.clamp(0.0, 1.0);
    let n_eval = ((v.len() as f64) * frac).round() as usize;
    let eval = v[..n_eval].to_vec();
    let train = v[n_eval..].to_vec();
    (train, eval)
}

/// Extrait la valeur (brute, encore échappée) de la clé JSON `key` dans une
/// ligne `{"key":"…"}`, en respectant les `\"`. Rend `None` si absente.
fn json_str_value(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let start = line.find(&pat)? + pat.len();
    let bytes = line.as_bytes();
    let mut i = start;
    let mut esc = false;
    while i < bytes.len() {
        let c = bytes[i];
        if esc {
            esc = false;
        } else if c == b'\\' {
            esc = true;
        } else if c == b'"' {
            return Some(line[start..i].to_string());
        }
        i += 1;
    }
    None
}

/// Convertit une ligne `{"prompt":"…","completion":"…"}` en ligne **chat**
/// `{"messages":[{"role":"user",…},{"role":"assistant",…}]}` (format SFT
/// unsloth/llama-factory). Les valeurs déjà échappées sont ré-emballées telles
/// quelles. Rend `None` si la ligne n'a pas les deux clés.
pub fn to_chat_line(line: &str) -> Option<String> {
    let p = json_str_value(line, "prompt")?;
    let c = json_str_value(line, "completion")?;
    Some(format!(
        "{{\"messages\":[{{\"role\":\"user\",\"content\":\"{p}\"}},\
         {{\"role\":\"assistant\",\"content\":\"{c}\"}}]}}"
    ))
}

/// Convertit un lot de lignes au format chat (ignore les lignes mal formées).
pub fn to_chat_jsonl(lines: &[String]) -> String {
    let mut out = String::new();
    for l in lines {
        if let Some(chat) = to_chat_line(l) {
            out.push_str(&chat);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(verdict_tag: &str) -> String {
        format!("{{\"prompt\":\"p\",\"completion\":\"running 1 tests\\n{verdict_tag}\"}}")
    }

    #[test]
    fn classify_covers_all_verdicts() {
        assert_eq!(
            classify_line(&pair("SIMCAL_VERDICT: compile=oui; tests=pass")),
            Verdict::CompilePass
        );
        assert_eq!(
            classify_line(&pair("SIMCAL_VERDICT: compile=oui; tests=fail")),
            Verdict::TestsFail
        );
        assert_eq!(
            classify_line(&pair("SIMCAL_VERDICT: compile=non; tests=na")),
            Verdict::CompileFail
        );
        assert_eq!(classify_line("{\"prompt\":\"x\"}"), Verdict::Unknown);
    }

    #[test]
    fn dedup_preserves_order_and_counts() {
        let (uniq, removed) = dedup(vec![
            "a".into(),
            "b".into(),
            "a".into(),
            "".into(),
            "c".into(),
        ]);
        assert_eq!(uniq, vec!["a", "b", "c"]);
        assert_eq!(removed, 1);
    }

    #[test]
    fn stats_tallies_classes_and_balance() {
        let lines = vec![
            pair("compile=oui; tests=pass"),
            pair("compile=oui; tests=pass"),
            pair("compile=oui; tests=fail"),
            pair("compile=non; tests=na"),
        ];
        let s = stats(&lines, 2);
        assert_eq!(s.total, 4);
        assert_eq!(s.compile_pass, 2);
        assert_eq!(s.tests_fail, 1);
        assert_eq!(s.compile_fail, 1);
        assert_eq!(s.duplicates_removed, 2);
        assert!(!s.is_balanced(0.30));
        assert!(s.is_balanced(0.20));
    }

    #[test]
    fn split_is_deterministic_and_partitions() {
        let lines: Vec<String> = (0..100).map(|i| format!("l{i}")).collect();
        let (tr1, ev1) = split(&lines, 0.2, 7);
        let (tr2, ev2) = split(&lines, 0.2, 7);
        assert_eq!(ev1, ev2);
        assert_eq!(tr1, tr2);
        assert_eq!(ev1.len(), 20);
        assert_eq!(tr1.len(), 80);
        let mut all: Vec<String> = tr1.iter().chain(ev1.iter()).cloned().collect();
        all.sort();
        let mut orig = lines.clone();
        orig.sort();
        assert_eq!(all, orig);
    }

    #[test]
    fn chat_conversion_respects_escapes() {
        let line = "{\"prompt\":\"dis \\\"bonjour\\\"\",\"completion\":\"ok\\nSIMCAL_VERDICT: compile=oui; tests=pass\"}";
        let chat = to_chat_line(line).unwrap();
        assert!(chat
            .starts_with("{\"messages\":[{\"role\":\"user\",\"content\":\"dis \\\"bonjour\\\"\"}"));
        assert!(chat.contains("\"role\":\"assistant\""));
        assert!(chat.contains("SIMCAL_VERDICT: compile=oui; tests=pass"));
        assert!(chat.ends_with("}]}"));
    }

    #[test]
    fn to_chat_jsonl_skips_malformed() {
        let lines = vec![
            "{\"prompt\":\"a\",\"completion\":\"b\"}".to_string(),
            "pas du json".to_string(),
        ];
        assert_eq!(to_chat_jsonl(&lines).lines().count(), 1);
    }
}
