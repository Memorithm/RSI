//! Graphe SVG de la trajectoire (std-only, aucune dépendance).
//!
//! Trace `SI_global`, `SI_safe`, `P_eff` et `Risk_global` en fonction du pas `t`,
//! sous forme de polylignes dans un SVG autonome (ouvrable dans un navigateur).

use std::io;
use std::path::Path;

use crate::agent::StepReport;

const W: f64 = 760.0;
const H: f64 = 380.0;
const ML: f64 = 50.0; // marge gauche
const MR: f64 = 140.0; // marge droite (légende)
const MT: f64 = 40.0;
const MB: f64 = 40.0;

struct Series {
    name: &'static str,
    color: &'static str,
    values: Vec<f64>,
}

fn polyline(values: &[f64], n: usize, color: &str) -> String {
    if values.is_empty() {
        return String::new();
    }
    let pw = W - ML - MR;
    let ph = H - MT - MB;
    let denom = (n.saturating_sub(1)).max(1) as f64;
    let mut pts = String::new();
    for (i, v) in values.iter().enumerate() {
        let x = ML + pw * (i as f64) / denom;
        let y = MT + ph * (1.0 - v.clamp(0.0, 1.0));
        pts.push_str(&format!("{x:.1},{y:.1} "));
    }
    format!(
        "<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"2\" points=\"{}\"/>",
        pts.trim()
    )
}

/// Construit un SVG autonome de la trajectoire.
pub fn trajectory_svg(reports: &[StepReport]) -> String {
    let n = reports.len();
    let series = [
        Series {
            name: "SI_global",
            color: "#2563eb",
            values: reports.iter().map(|r| r.si_global).collect(),
        },
        Series {
            name: "SI_safe",
            color: "#16a34a",
            values: reports.iter().map(|r| r.si_safe).collect(),
        },
        Series {
            name: "P_eff",
            color: "#ea580c",
            values: reports.iter().map(|r| r.p_eff).collect(),
        },
        Series {
            name: "Risk_global",
            color: "#dc2626",
            values: reports.iter().map(|r| r.risk_global).collect(),
        },
    ];

    let pw = W - ML - MR;
    let ph = H - MT - MB;
    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{W}\" height=\"{H}\" \
font-family=\"sans-serif\" font-size=\"12\">"
    ));
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"22\" font-size=\"15\" font-weight=\"bold\">RSI — trajectoire (n={n} pas)</text>",
        ML
    ));

    // grille + axes Y (0..1)
    for k in 0..=4 {
        let val = k as f64 / 4.0;
        let y = MT + ph * (1.0 - val);
        svg.push_str(&format!(
            "<line x1=\"{ML:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"#e5e7eb\"/>",
            ML + pw
        ));
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" fill=\"#6b7280\">{val:.2}</text>",
            ML - 6.0,
            y + 4.0
        ));
    }
    // axe X (label début/fin)
    svg.push_str(&format!(
        "<text x=\"{ML:.1}\" y=\"{:.1}\" fill=\"#6b7280\">t=0</text>",
        H - MB + 18.0
    ));
    svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" fill=\"#6b7280\">t={}</text>",
        ML + pw,
        H - MB + 18.0,
        n.saturating_sub(1)
    ));

    // séries + légende
    for (j, s) in series.iter().enumerate() {
        svg.push_str(&polyline(&s.values, n, s.color));
        let ly = MT + 8.0 + j as f64 * 20.0;
        let lx = ML + pw + 20.0;
        svg.push_str(&format!(
            "<line x1=\"{lx:.1}\" y1=\"{ly:.1}\" x2=\"{:.1}\" y2=\"{ly:.1}\" stroke=\"{}\" stroke-width=\"3\"/>",
            lx + 18.0,
            s.color
        ));
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#111827\">{}</text>",
            lx + 24.0,
            ly + 4.0,
            s.name
        ));
    }

    svg.push_str("</svg>");
    svg
}

/// Écrit le SVG de la trajectoire dans un fichier.
pub fn write_svg<P: AsRef<Path>>(reports: &[StepReport], path: P) -> io::Result<()> {
    std::fs::write(path, trajectory_svg(reports))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RSIAgent;

    #[test]
    fn svg_is_wellformed() {
        let mut agent = RSIAgent::demo(1);
        let reports = agent.run(12);
        let svg = trajectory_svg(&reports);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("SI_global") && svg.contains("Risk_global"));
        assert_eq!(svg.matches("<polyline").count(), 4);
    }

    #[test]
    fn handles_empty() {
        let svg = trajectory_svg(&[]);
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
    }
}
