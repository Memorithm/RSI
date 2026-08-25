//! Valeur JSON + parseur + sérialiseur, **100 % std-only**.
//!
//! Suffisant pour le JSON-RPC 2.0 du serveur MCP et la façade [`crate::api`].
//! Parseur à descente récursive conforme à la grammaire JSON (RFC 8259) pour
//! les cas usuels (objets, tableaux, chaînes avec échappements, nombres,
//! `true`/`false`/`null`).

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Valeur JSON.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    // --- constructeurs pratiques ---------------------------------------- //
    pub fn obj() -> Json {
        Json::Obj(BTreeMap::new())
    }

    /// Insère une paire clé/valeur (no-op si `self` n'est pas un objet).
    pub fn set(&mut self, key: &str, value: Json) -> &mut Self {
        if let Json::Obj(m) = self {
            m.insert(key.to_string(), value);
        }
        self
    }

    // --- accesseurs ----------------------------------------------------- //
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        // Rejette NaN, ±∞ et négatifs au lieu de saturer silencieusement à 0
        // (`-1.0 as u64 == 0`, `NaN as u64 == 0`).
        self.as_f64()
            .filter(|n| n.is_finite() && *n >= 0.0)
            .map(|n| n as u64)
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_f64()
            .filter(|n| n.is_finite() && *n >= 0.0)
            .map(|n| n as usize)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    // --- sérialisation -------------------------------------------------- //
    /// Sérialisation compacte.
    #[allow(clippy::inherent_to_string)] // sérialiseur JSON dédié, pas un Display
    pub fn to_string(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    let _ = write!(out, "{}", *n as i64);
                } else {
                    let _ = write!(out, "{}", n);
                }
            }
            Json::Str(s) => write_escaped(s, out),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Obj(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_escaped(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }

    // --- parsing -------------------------------------------------------- //
    /// Parse une chaîne JSON.
    ///
    /// Sécurité : rejette les entrées dont l'imbrication dépasse
    /// [`MAX_DEPTH`], ce qui évite tout dépassement de pile sur une entrée
    /// hostile (le parseur est récursif et traite des données non fiables
    /// côté serveur MCP).
    pub fn parse(input: &str) -> Result<Json, String> {
        let mut p = Parser {
            input,
            chars: input.chars(),
            depth: 0,
        };
        p.skip_ws();
        let v = p.parse_value()?;
        p.skip_ws();
        if p.pos() != p.input.len() {
            return Err(format!("caractères superflus à la position {}", p.pos()));
        }
        Ok(v)
    }
}

/// Profondeur d'imbrication maximale tolérée par le parseur (anti stack-overflow).
const MAX_DEPTH: usize = 128;

fn write_escaped(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    input: &'a str,
    chars: std::str::Chars<'a>,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn pos(&self) -> usize {
        self.input.len() - self.chars.as_str().len()
    }

    fn peek(&self) -> Option<char> {
        self.chars.clone().next()
    }

    fn enter(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(format!("imbrication JSON trop profonde (> {MAX_DEPTH})"));
        }
        Ok(())
    }

    fn next(&mut self) -> Option<char> {
        self.chars.next()
    }

    /// Lit exactement 4 chiffres hexadécimaux (corps d'un échappement `\u`).
    fn read_hex4(&mut self) -> Result<u32, String> {
        let mut code = 0u32;
        for _ in 0..4 {
            let h = self.next().ok_or("\\u tronqué")?;
            code = code * 16 + h.to_digit(16).ok_or("hex invalide dans \\u")?;
        }
        Ok(code)
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.next();
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(Json::Str(self.parse_string()?)),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(format!(
                "token inattendu '{c}' à la position {}",
                self.pos()
            )),
            None => Err("fin d'entrée inattendue".into()),
        }
    }

    fn parse_object(&mut self) -> Result<Json, String> {
        self.enter()?;
        self.next(); // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.next();
            self.depth -= 1;
            return Ok(Json::Obj(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.next() != Some(':') {
                return Err(format!("':' attendu à la position {}", self.pos()));
            }
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.next() {
                Some(',') => continue,
                Some('}') => break,
                _ => return Err(format!("',' ou '}}' attendu à la position {}", self.pos())),
            }
        }
        self.depth -= 1;
        Ok(Json::Obj(map))
    }

    fn parse_array(&mut self) -> Result<Json, String> {
        self.enter()?;
        self.next(); // '['
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.next();
            self.depth -= 1;
            return Ok(Json::Arr(arr));
        }
        loop {
            let val = self.parse_value()?;
            arr.push(val);
            self.skip_ws();
            match self.next() {
                Some(',') => continue,
                Some(']') => break,
                _ => return Err(format!("',' ou ']' attendu à la position {}", self.pos())),
            }
        }
        self.depth -= 1;
        Ok(Json::Arr(arr))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.next() != Some('"') {
            return Err(format!("'\"' attendu à la position {}", self.pos()));
        }
        let mut s = String::new();
        while let Some(c) = self.next() {
            match c {
                '"' => return Ok(s),
                '\\' => {
                    let esc = self.next().ok_or("échappement tronqué")?;
                    match esc {
                        '"' => s.push('"'),
                        '\\' => s.push('\\'),
                        '/' => s.push('/'),
                        'n' => s.push('\n'),
                        'r' => s.push('\r'),
                        't' => s.push('\t'),
                        'b' => s.push('\u{0008}'),
                        'f' => s.push('\u{000C}'),
                        'u' => {
                            let code = self.read_hex4()?;
                            // Paires de surrogates UTF-16 (RFC 8259) : un caractère
                            // hors BMP est encodé `\uD800-DBFF` suivi de
                            // `\uDC00-DFFF`. On les recombine en un scalaire unique.
                            let ch = if (0xD800..=0xDBFF).contains(&code) {
                                if self.next() != Some('\\') || self.next() != Some('u') {
                                    return Err("surrogate haut non suivi de \\u".into());
                                }
                                let low = self.read_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err("surrogate bas invalide".into());
                                }
                                let combined = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                                char::from_u32(combined).unwrap_or('\u{FFFD}')
                            } else if (0xDC00..=0xDFFF).contains(&code) {
                                // surrogate bas isolé : séquence invalide.
                                '\u{FFFD}'
                            } else {
                                char::from_u32(code).unwrap_or('\u{FFFD}')
                            };
                            s.push(ch);
                        }
                        other => return Err(format!("échappement inconnu \\{other}")),
                    }
                }
                c => s.push(c),
            }
        }
        Err("chaîne non terminée".into())
    }

    fn parse_bool(&mut self) -> Result<Json, String> {
        if self.match_literal("true") {
            Ok(Json::Bool(true))
        } else if self.match_literal("false") {
            Ok(Json::Bool(false))
        } else {
            Err(format!(
                "littéral booléen invalide à la position {}",
                self.pos()
            ))
        }
    }

    fn parse_null(&mut self) -> Result<Json, String> {
        if self.match_literal("null") {
            Ok(Json::Null)
        } else {
            Err(format!(
                "littéral null invalide à la position {}",
                self.pos()
            ))
        }
    }

    fn match_literal(&mut self, lit: &str) -> bool {
        if self.chars.as_str().starts_with(lit) {
            let remaining = &self.chars.as_str()[lit.len()..];
            self.chars = remaining.chars();
            true
        } else {
            false
        }
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos();
        if self.peek() == Some('-') {
            self.next();
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                self.next();
            } else {
                break;
            }
        }
        let end = self.pos();
        let slice: &str = &self.input[start..end];
        slice
            .parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("nombre invalide '{slice}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_object() {
        let src = r#"{"a":1,"b":[true,null,"x"],"c":{"d":-2.5}}"#;
        let v = Json::parse(src).unwrap();
        assert_eq!(v.get("a").unwrap().as_f64(), Some(1.0));
        assert_eq!(v.get("b").unwrap().as_array().unwrap().len(), 3);
        assert_eq!(v.get("c").unwrap().get("d").unwrap().as_f64(), Some(-2.5));
        // re-parse de la sérialisation
        let again = Json::parse(&v.to_string()).unwrap();
        assert_eq!(v, again);
    }

    #[test]
    fn parse_escapes() {
        let v = Json::parse(r#""line\n\tA""#).unwrap();
        assert_eq!(v.as_str(), Some("line\n\tA"));
    }

    #[test]
    fn rejects_trailing_garbage() {
        assert!(Json::parse("{} extra").is_err());
    }

    #[test]
    fn rejects_excessive_nesting() {
        // entrée hostile : imbrication profonde → refus propre, pas de crash
        let deep = "[".repeat(10_000);
        let err = Json::parse(&deep).unwrap_err();
        assert!(err.contains("profonde"), "msg = {err}");
    }

    #[test]
    fn accepts_reasonable_nesting() {
        let ok = format!("{}{}", "[".repeat(64), "]".repeat(64));
        assert!(Json::parse(&ok).is_ok());
    }

    #[test]
    fn parses_utf16_surrogate_pairs() {
        // 😀 U+1F600 encodé en paire de surrogates (RFC 8259)
        let v = Json::parse(r#""😀""#).unwrap();
        assert_eq!(v, Json::Str("😀".to_string()));
        // roundtrip : notre sérialiseur émet l'UTF-8 brut (valide), reparsable
        let back = Json::parse(&v.to_string()).unwrap();
        assert_eq!(back, v);
        // BMP normal toujours décodé
        assert_eq!(Json::parse(r#""é""#).unwrap(), Json::Str("é".to_string()));
        // surrogate haut non suivi d'un bas ⇒ erreur propre (pas de panic)
        assert!(Json::parse(r#""\uD83D""#).is_err());
        assert!(Json::parse(r#""\uD83Dx""#).is_err());
    }

    #[test]
    fn parse_never_panics_on_random_input() {
        // Mini-fuzz *in-tree* (RNG déterministe, zéro dépendance) : `Json::parse`
        // ne doit JAMAIS paniquer, quelle que soit l'entrée. On échantillonne dans
        // un alphabet riche en caractères structurellement signifiants.
        //
        // Le volume est paramétrable par `RSI_JSON_FUZZ_ITERS` (défaut 10 000 ;
        // le job nightly CI le monte à 5 M — cf. .github/workflows/fuzz-json.yml).
        use crate::rng::Rng;
        const ALPHABET: &[char] = &[
            '{', '}', '[', ']', ':', ',', '"', '\\', '/', 'u', 'n', 't', 'f', 'e', 'E', '+', '-',
            '.', '0', '1', '9', ' ', '\n', '\t', 'a', 'z', 'é', '😀', '\u{0}',
        ];
        let iters: u64 = std::env::var("RSI_JSON_FUZZ_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000);
        // Graine paramétrable (RSI_JSON_FUZZ_SEED) : sans elle, le job nightly
        // rejouerait exactement les mêmes 5 M entrées à chaque exécution —
        // zéro couverture nouvelle après la première semaine (audit M15).
        // Défaut : dérivé de l'horloge → corpus différent à chaque run.
        let seed: u64 = std::env::var("RSI_JSON_FUZZ_SEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0x5EED)
            });
        eprintln!("[fuzz-json] graine = {seed} ({iters} itérations)");
        let mut rng = Rng::new(seed);
        for i in 0..iters {
            let len = rng.uniform_range(0.0, 48.0) as usize;
            let s: String = (0..len)
                .map(|_| ALPHABET[(rng.uniform_range(0.0, ALPHABET.len() as f64)) as usize])
                .collect();
            // le seul contrat : pas de panic. Ok ou Err, peu importe.
            // En cas de panic, on imprime l'échantillon fautif AVANT de
            // propager — reproductible via RSI_JSON_FUZZ_SEED + itération i.
            let sample = s.clone();
            let idx = i;
            if std::panic::catch_unwind(|| Json::parse(&sample)).is_err() {
                eprintln!(
                    "[fuzz-json] PANIC à l'itération {idx} (graine {seed}) sur :\n{sample:?}"
                );
                panic!("fuzz-json : panic du parseur (échantillon ci-dessus)");
            }
            const PROGRESS_EVERY: u64 = 1_000_000;
            if i > 0 && i % PROGRESS_EVERY == 0 {
                eprintln!("[fuzz-json] {i} entrées hostiles sans panic…");
            }
        }
        // quelques motifs adversariaux ciblés
        let deep = "[".repeat(200);
        for pat in [
            "\"\\u",
            "\"\\uD83D",
            deep.as_str(),
            "{\"a\":",
            "\"\\",
            "1e",
            "-",
            "1.e9",
            "\"\\uZZZZ\"",
        ] {
            let _ = Json::parse(pat);
        }
    }

    #[test]
    fn unsigned_accessors_reject_negative_and_nan() {
        // valeurs valides : conversion normale
        assert_eq!(Json::Num(5.0).as_u64(), Some(5));
        assert_eq!(Json::Num(5.9).as_usize(), Some(5));
        assert_eq!(Json::Num(0.0).as_u64(), Some(0));
        // négatifs : None au lieu de saturer silencieusement à 0
        assert_eq!(Json::Num(-1.0).as_u64(), None);
        assert_eq!(Json::Num(-5.0).as_usize(), None);
        // non-finis : None
        assert_eq!(Json::Num(f64::NAN).as_u64(), None);
        assert_eq!(Json::Num(f64::INFINITY).as_usize(), None);
    }
}
