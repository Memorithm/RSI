//! §1bis — CORPUS DE TÂCHES ANCRÉ (de-stylisation de Ω et Φ)
//!
//! Au lieu d'un espace de tâches synthétique (profils de Dirichlet) et d'une
//! compétence générique (sigmoïde d'un produit scalaire), ce module permet de
//! brancher un **corpus de tâches réel** : chaque tâche porte un **profil
//! d'exigences** explicite sur (D,M,R,A,C,V), une **difficulté** et une
//! **importance**. Le corpus est *chargeable depuis un fichier JSON* (métadonnées
//! de benchmark réelles).
//!
//! La compétence ancrée [`GroundedCapability`] suit la **loi de Liebig** : sur
//! une tâche, l'agent ne vaut que sa capacité *requise la plus faible*
//! (Φ = min des marges), ce qui est plus fidèle qu'un produit scalaire lissé.

use crate::json::Json;
use crate::linalg::sigmoid;
use crate::surface::{CapabilityModel, CeilingModel, IntelligenceSurface, PowerCeiling};

/// Une tâche réelle : exigences par composante, difficulté, importance.
#[derive(Clone, Debug)]
pub struct Task {
    pub name: String,
    /// exigences sur (D, M, R, A, C, V) ∈ [0,1].
    pub requirements: [f64; 6],
    /// difficulté ∈ [0,1] (échelle les exigences et le plafond physique).
    pub difficulty: f64,
    /// importance dans μ (poids).
    pub weight: f64,
}

/// Corpus de tâches (l'espace Ω ancré sur des données).
#[derive(Clone, Debug, Default)]
pub struct TaskCorpus {
    pub tasks: Vec<Task>,
}

impl TaskCorpus {
    pub fn new(tasks: Vec<Task>) -> Self {
        TaskCorpus { tasks }
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Corpus intégré, représentatif d'archétypes de tâches d'un agent cognitif
    /// (chaque archétype sollicite un sous-ensemble distinct de S).
    pub fn builtin() -> Self {
        // (nom, [D,M,R,A,C,V], difficulté, poids)
        let rows: &[(&str, [f64; 6], f64, f64)] = &[
            ("rappel_factuel", [0.9, 0.3, 0.2, 0.1, 0.3, 0.1], 0.3, 1.0),
            (
                "raisonnement_multi_etapes",
                [0.4, 0.7, 0.9, 0.2, 0.4, 0.2],
                0.7,
                1.0,
            ),
            (
                "synthese_long_contexte",
                [0.5, 0.5, 0.5, 0.2, 0.9, 0.2],
                0.6,
                0.9,
            ),
            (
                "planification_autonome",
                [0.4, 0.5, 0.6, 0.9, 0.5, 0.6],
                0.8,
                0.8,
            ),
            (
                "alignement_decision",
                [0.3, 0.3, 0.5, 0.5, 0.3, 0.9],
                0.6,
                0.9,
            ),
            ("generation_code", [0.6, 0.8, 0.8, 0.3, 0.5, 0.3], 0.75, 1.0),
            (
                "apprentissage_nouveau_domaine",
                [0.8, 0.7, 0.6, 0.4, 0.6, 0.3],
                0.7,
                0.7,
            ),
            (
                "dialogue_contextuel",
                [0.5, 0.4, 0.4, 0.3, 0.8, 0.4],
                0.4,
                0.8,
            ),
            (
                "optimisation_outils",
                [0.5, 0.9, 0.6, 0.5, 0.4, 0.3],
                0.7,
                0.7,
            ),
            ("auto_correction", [0.5, 0.6, 0.8, 0.6, 0.6, 0.6], 0.65, 0.8),
        ];
        let tasks = rows
            .iter()
            .map(|(name, req, diff, w)| Task {
                name: (*name).to_string(),
                requirements: *req,
                difficulty: *diff,
                weight: *w,
            })
            .collect();
        TaskCorpus::new(tasks)
    }

    /// Corpus **élargi** (~36 tâches) : chaque archétype intégré est décliné en
    /// plusieurs niveaux de difficulté, donnant un Ω plus grand et moins bruité
    /// (utile pour l'évaluation / les ablations). Pour un vrai benchmark public,
    /// charger les métadonnées via [`TaskCorpus::from_json`].
    pub fn extended() -> Self {
        let base = TaskCorpus::builtin();
        let factors = [0.6_f64, 0.85, 1.0, 1.15];
        let mut tasks = Vec::new();
        for t in &base.tasks {
            for (k, f) in factors.iter().enumerate() {
                let difficulty = (t.difficulty * f).clamp(0.05, 0.98);
                tasks.push(Task {
                    name: format!("{}_d{k}", t.name),
                    requirements: t.requirements,
                    difficulty,
                    // les variantes difficiles pèsent un peu moins
                    weight: t.weight * (1.0 - 0.1 * k as f64),
                });
            }
        }
        TaskCorpus::new(tasks)
    }

    /// Charge un corpus depuis du JSON :
    /// `{"tasks":[{"name":..,"requirements":[6],"difficulty":..,"weight":..}, …]}`.
    pub fn from_json(src: &str) -> Result<Self, String> {
        let root = Json::parse(src)?;
        let arr = root
            .get("tasks")
            .and_then(|v| v.as_array())
            .ok_or("clé 'tasks' (tableau) manquante")?;
        let mut tasks = Vec::with_capacity(arr.len());
        for (i, t) in arr.iter().enumerate() {
            let name = t
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("tâche")
                .to_string();
            let req_arr = t
                .get("requirements")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("tâche {i}: 'requirements' manquant"))?;
            if req_arr.len() != 6 {
                return Err(format!("tâche {i}: 'requirements' doit avoir 6 valeurs"));
            }
            let mut requirements = [0.0; 6];
            for (k, v) in req_arr.iter().enumerate() {
                requirements[k] = v.as_f64().unwrap_or(0.0).clamp(0.0, 1.0);
            }
            let difficulty = t
                .get("difficulty")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
            let weight = t
                .get("weight")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0)
                .max(0.0);
            tasks.push(Task {
                name,
                requirements,
                difficulty,
                weight,
            });
        }
        if tasks.is_empty() {
            return Err("corpus vide".into());
        }
        Ok(TaskCorpus::new(tasks))
    }

    /// Charge un corpus depuis un fichier JSON.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::from_json(&src)
    }
}

/// Compétence ancrée Φ_x(S) — **loi de Liebig** : minimum des marges sur les
/// composantes *requises* par la tâche. La tâche stocke ses exigences déjà
/// mises à l'échelle par la difficulté (cf. [`IntelligenceSurface::from_corpus`]).
#[derive(Clone, Debug)]
pub struct GroundedCapability {
    /// raideur de la transition « capacité insuffisante → suffisante ».
    pub sharpness: f64,
}

impl Default for GroundedCapability {
    fn default() -> Self {
        GroundedCapability { sharpness: 6.0 }
    }
}

impl CapabilityModel for GroundedCapability {
    fn phi(&self, task: &[f64; 6], caps: &[f64; 6]) -> f64 {
        let mut worst = 1.0_f64;
        let mut any = false;
        for i in 0..6 {
            if task[i] > 1e-6 {
                any = true;
                // marge : capacité − exigence effective ; sigmoïde lissée
                let comp = sigmoid((caps[i] - task[i]) * self.sharpness);
                worst = worst.min(comp);
            }
        }
        if any {
            worst
        } else {
            1.0
        }
    }

    fn clone_box(&self) -> Box<dyn CapabilityModel> {
        Box::new(self.clone())
    }
}

impl IntelligenceSurface {
    /// Construit une surface **ancrée sur un corpus réel** (Ω = corpus). Les
    /// exigences sont mises à l'échelle par la difficulté ; le plafond physique
    /// par défaut est la loi de puissance (`PowerCeiling`).
    pub fn from_corpus(corpus: &TaskCorpus) -> Self {
        Self::from_corpus_with(
            corpus,
            Box::new(GroundedCapability::default()),
            Box::new(PowerCeiling),
        )
    }

    /// Variante avec modèles Φ/g personnalisés.
    pub fn from_corpus_with(
        corpus: &TaskCorpus,
        capability: Box<dyn CapabilityModel>,
        ceiling: Box<dyn CeilingModel>,
    ) -> Self {
        let n = corpus.tasks.len().max(1);
        let mut tasks = Vec::with_capacity(n);
        let mut demand = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        let max_diff = corpus
            .tasks
            .iter()
            .map(|t| t.difficulty)
            .fold(f64::MIN, f64::max)
            .max(1e-9);
        for t in &corpus.tasks {
            // exigence effective = exigence × difficulté
            let eff = std::array::from_fn(|i| (t.requirements[i] * t.difficulty).clamp(0.0, 1.0));
            tasks.push(eff);
            demand.push(t.difficulty / max_diff);
            weights.push(t.weight);
        }
        let sum_w: f64 = weights.iter().sum::<f64>().max(1e-12);
        for w in weights.iter_mut() {
            *w /= sum_w;
        }
        IntelligenceSurface {
            tasks,
            demand,
            weights,
            capability,
            ceiling,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::state::{CognitiveState, Dims};
    use crate::substrate::Substrate;

    #[test]
    fn builtin_corpus_surface() {
        let corpus = TaskCorpus::builtin();
        assert!(corpus.len() >= 8);
        let surf = IntelligenceSurface::from_corpus(&corpus);
        assert_eq!(surf.tasks.len(), corpus.len());

        let mut rng = Rng::new(1);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let low = CognitiveState::from_vector(&[0.1; 24], Dims::uniform(4));
        let high = CognitiveState::from_vector(&[0.95; 24], Dims::uniform(4));
        // un agent fort domine un agent faible sur le corpus réel
        assert!(surf.si_global(&high, &sub) > surf.si_global(&low, &sub));
    }

    #[test]
    fn liebig_law_weakest_gates() {
        let cap = GroundedCapability::default();
        // tâche exigeant fortement R (indice 2)
        let task = [0.2, 0.2, 0.9, 0.0, 0.0, 0.0];
        let strong_except_r = [0.9, 0.9, 0.1, 0.9, 0.9, 0.9];
        let balanced_high = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95];
        // la faiblesse en R plombe la compétence malgré le reste élevé,
        // alors qu'un profil équilibré (fort partout) la dépasse largement
        assert!(cap.phi(&task, &strong_except_r) < 0.1);
        assert!(cap.phi(&task, &balanced_high) > 0.5);
        assert!(cap.phi(&task, &balanced_high) > cap.phi(&task, &strong_except_r));
    }

    #[test]
    fn json_roundtrip() {
        let src = r#"{"tasks":[
            {"name":"a","requirements":[0.5,0.5,0.5,0.5,0.5,0.5],"difficulty":0.4,"weight":1.0},
            {"name":"b","requirements":[0.9,0.1,0.1,0.1,0.1,0.1],"difficulty":0.6,"weight":2.0}
        ]}"#;
        let corpus = TaskCorpus::from_json(src).unwrap();
        assert_eq!(corpus.len(), 2);
        assert_eq!(corpus.tasks[1].name, "b");
        let surf = IntelligenceSurface::from_corpus(&corpus);
        assert_eq!(surf.tasks.len(), 2);
    }
}
