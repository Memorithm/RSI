//! ⚙️ Loop Engineering — **L5 : checkpoint / reprise / replay**.
//!
//! Sérialise l'**état macro évolutif** de l'agent — `S=(D,M,R,A,C,V)`, substrat
//! (H,O,A,B,C + efficience mesurée), stratégie ℳ courante, compteur `t` — en
//! JSON (via [`crate::json`], sans dépendance). Permet de **reprendre** une
//! boucle (après incident / pause) et sert de point de **rollback** aux
//! disjoncteurs de sûreté (L4).
//!
//! Note déterminisme : le *replay bit-identique* d'une trajectoire complète est
//! garanti par la reproductibilité par graine (même graine + même config ⇒
//! même trajectoire, prouvée par le hash de tête d'audit). Le checkpoint, lui,
//! capture l'état macro (S, substrat, stratégie) pour une **reprise** ; la
//! surface (environnement) et les backends (mémoire/audit) restent fournis par
//! l'agent hôte au moment du `restore`.

use crate::agent::RSIAgent;
use crate::json::Json;
use crate::linalg::Matrix;
use crate::meta::MetaStrategy;
use crate::state::CognitiveState;
use crate::substrate::Substrate;

/// Instantané reprenable de l'état macro d'un agent.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub t: usize,
    pub state: CognitiveState,
    pub substrate: Substrate,
    pub strategy: MetaStrategy,
}

// --- helpers JSON ------------------------------------------------------- //

fn vec_json(v: &[f64]) -> Json {
    Json::Arr(v.iter().map(|&x| Json::Num(x)).collect())
}

fn json_vec(j: &Json) -> Option<Vec<f64>> {
    Some(
        j.as_array()?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0))
            .collect(),
    )
}

fn matrix_json(m: &Matrix) -> Json {
    let mut o = Json::obj();
    o.set("rows", Json::Num(m.rows as f64))
        .set("cols", Json::Num(m.cols as f64))
        .set("data", vec_json(&m.data));
    o
}

fn json_matrix(j: &Json) -> Option<Matrix> {
    let rows = j.get("rows")?.as_usize()?;
    let cols = j.get("cols")?.as_usize()?;
    let data = json_vec(j.get("data")?)?;
    if data.len() != rows * cols {
        return None;
    }
    Some(Matrix::from_vec(rows, cols, data))
}

impl Checkpoint {
    /// Sérialise en JSON.
    pub fn to_json(&self) -> String {
        let mut st = Json::obj();
        st.set("d", vec_json(&self.state.d))
            .set("m", vec_json(&self.state.m))
            .set("r", vec_json(&self.state.r))
            .set("a", vec_json(&self.state.a))
            .set("c", vec_json(&self.state.c))
            .set("v", vec_json(&self.state.v));

        let mut sub = Json::obj();
        sub.set("h", vec_json(&self.substrate.h))
            .set("o", vec_json(&self.substrate.o))
            .set("a", matrix_json(&self.substrate.a))
            .set("b", matrix_json(&self.substrate.b))
            .set("c", matrix_json(&self.substrate.c));
        if let Some(e) = self.substrate.measured_software_eff {
            sub.set("measured_software_eff", Json::Num(e));
        }

        let mut strat = Json::obj();
        strat
            .set(
                "focus",
                Json::Arr(self.strategy.focus.iter().map(|&x| Json::Num(x)).collect()),
            )
            .set("software_edit", vec_json(&self.strategy.software_edit))
            .set("gain", Json::Num(self.strategy.gain));

        let mut root = Json::obj();
        root.set("version", Json::Num(1.0))
            .set("t", Json::Num(self.t as f64))
            .set("state", st)
            .set("substrate", sub)
            .set("strategy", strat);
        root.to_string()
    }

    /// Reconstruit depuis JSON.
    pub fn from_json(src: &str) -> Result<Self, String> {
        let root = Json::parse(src)?;
        let t = root
            .get("t")
            .and_then(|v| v.as_usize())
            .ok_or("champ 't'")?;

        let s = root.get("state").ok_or("champ 'state'")?;
        let comp = |k: &str| -> Result<Vec<f64>, String> {
            json_vec(s.get(k).ok_or(format!("state.{k}"))?).ok_or(format!("state.{k} invalide"))
        };
        let state = CognitiveState {
            d: comp("d")?,
            m: comp("m")?,
            r: comp("r")?,
            a: comp("a")?,
            c: comp("c")?,
            v: comp("v")?,
        };

        let sb = root.get("substrate").ok_or("champ 'substrate'")?;
        let h = json_vec(sb.get("h").ok_or("substrate.h")?).ok_or("substrate.h")?;
        let o = json_vec(sb.get("o").ok_or("substrate.o")?).ok_or("substrate.o")?;
        let a = json_matrix(sb.get("a").ok_or("substrate.a")?).ok_or("substrate.a")?;
        let b = json_matrix(sb.get("b").ok_or("substrate.b")?).ok_or("substrate.b")?;
        let c = json_matrix(sb.get("c").ok_or("substrate.c")?).ok_or("substrate.c")?;
        let mut substrate = Substrate::new(h, o, a, b, c);
        if let Some(e) = sb.get("measured_software_eff").and_then(|v| v.as_f64()) {
            substrate.set_measured_software_eff(Some(e));
        }

        let stj = root.get("strategy").ok_or("champ 'strategy'")?;
        let focus_v =
            json_vec(stj.get("focus").ok_or("strategy.focus")?).ok_or("strategy.focus")?;
        if focus_v.len() != 6 {
            return Err("strategy.focus doit avoir 6 valeurs".into());
        }
        let mut focus = [0.0; 6];
        focus.copy_from_slice(&focus_v);
        let strategy = MetaStrategy {
            focus,
            software_edit: json_vec(stj.get("software_edit").ok_or("strategy.software_edit")?)
                .ok_or("strategy.software_edit")?,
            gain: stj.get("gain").and_then(|v| v.as_f64()).unwrap_or(0.05),
        };

        // Cohérence interne minimale : aucune composante d'état vide (sinon panic
        // ultérieur dans les moyennes / indexations de la surface). La cohérence
        // avec une surface *donnée* est vérifiée par `RSIAgent::try_restore`.
        if [&state.d, &state.m, &state.r, &state.a, &state.c, &state.v]
            .iter()
            .any(|comp| comp.is_empty())
        {
            return Err("state : composante vide".into());
        }
        Ok(Checkpoint {
            t,
            state,
            substrate,
            strategy,
        })
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::from_json(&src)
    }
}

impl RSIAgent {
    /// Capture un instantané reprenable de l'état macro (L5).
    pub fn snapshot(&self) -> Checkpoint {
        Checkpoint {
            t: self.t,
            state: self.state.clone(),
            substrate: self.substrate.clone(),
            strategy: self.strategy.clone(),
        }
    }

    /// Restaure l'état macro depuis un checkpoint (surface/mémoire/audit
    /// conservés). Utilisé pour la reprise et le rollback (L4).
    pub fn restore(&mut self, cp: &Checkpoint) {
        self.state = cp.state.clone();
        self.substrate = cp.substrate.clone();
        self.strategy = cp.strategy.clone();
        self.t = cp.t;
    }

    /// Restaure **en validant** la cohérence dimensionnelle du checkpoint avec
    /// l'agent courant (surface/substrat). Renvoie `Err` au lieu de laisser un
    /// checkpoint incohérent provoquer un panic différé (p. ex. indexation hors
    /// borne dans `si_global`). À préférer pour tout checkpoint *désérialisé*
    /// (`from_json` / `load`) ; `restore` reste le chemin direct pour le rollback
    /// d'un snapshot pris sur le *même* agent (dimensions garanties identiques).
    pub fn try_restore(&mut self, cp: &Checkpoint) -> Result<(), String> {
        let want = [
            self.state.d.len(),
            self.state.m.len(),
            self.state.r.len(),
            self.state.a.len(),
            self.state.c.len(),
            self.state.v.len(),
        ];
        let got = [
            cp.state.d.len(),
            cp.state.m.len(),
            cp.state.r.len(),
            cp.state.a.len(),
            cp.state.c.len(),
            cp.state.v.len(),
        ];
        if got != want {
            return Err(format!(
                "dimensions d'état du checkpoint {got:?} incohérentes avec l'agent {want:?}"
            ));
        }
        if cp.substrate.h.len() != self.substrate.h.len()
            || cp.substrate.o.len() != self.substrate.o.len()
        {
            return Err(format!(
                "dimensions substrat du checkpoint (h={}, o={}) incohérentes avec l'agent (h={}, o={})",
                cp.substrate.h.len(),
                cp.substrate.o.len(),
                self.substrate.h.len(),
                self.substrate.o.len()
            ));
        }
        self.restore(cp);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_restore_preserves_macro_state() {
        let mut agent = RSIAgent::demo(2026);
        agent.run(20);
        let cp = agent.snapshot();
        let si = agent.si_global();
        let t = agent.t;

        // on continue puis on restaure
        agent.run(10);
        assert_ne!(agent.t, t);
        agent.restore(&cp);
        assert_eq!(agent.t, t);
        assert!((agent.si_global() - si).abs() < 1e-12);
    }

    #[test]
    fn checkpoint_json_roundtrip() {
        let mut agent = RSIAgent::demo(7);
        agent.run(15);
        let cp = agent.snapshot();
        let json = cp.to_json();
        let cp2 = Checkpoint::from_json(&json).unwrap();
        assert_eq!(cp.t, cp2.t);
        // SI_global identique après reconstruction (mêmes S/substrat)
        let mut a2 = RSIAgent::demo(7);
        a2.restore(&cp2);
        let mut a1 = RSIAgent::demo(7);
        a1.restore(&cp);
        assert!((a1.si_global() - a2.si_global()).abs() < 1e-9);
    }

    #[test]
    fn resume_continues_improving() {
        let mut agent = RSIAgent::demo(3);
        agent.run(15);
        let cp = agent.snapshot();
        let si_mid = agent.si_global();
        // reprise dans un nouvel agent (même surface via demo(3))
        let mut resumed = RSIAgent::demo(3);
        resumed.restore(&cp);
        assert!((resumed.si_global() - si_mid).abs() < 1e-9);
        resumed.run(30);
        assert!(resumed.si_global() >= si_mid);
    }

    #[test]
    fn try_restore_rejects_mismatched_dims() {
        let mut agent = RSIAgent::demo(2026);
        agent.run(5);
        // checkpoint cohérent : accepté
        let good = agent.snapshot();
        assert!(agent.try_restore(&good).is_ok());
        // checkpoint corrompu (composante d'état tronquée) : refusé proprement
        // au lieu de provoquer un panic différé dans `si_global`.
        let mut bad = agent.snapshot();
        bad.state.d.pop();
        let err = agent.try_restore(&bad).unwrap_err();
        assert!(err.contains("dimensions"), "msg = {err}");
    }

    #[test]
    fn from_json_rejects_empty_state_component() {
        let mut agent = RSIAgent::demo(1);
        agent.run(3);
        let mut cp = agent.snapshot();
        cp.state.r.clear();
        let err = Checkpoint::from_json(&cp.to_json()).unwrap_err();
        assert!(err.contains("vide"), "msg = {err}");
    }
}
