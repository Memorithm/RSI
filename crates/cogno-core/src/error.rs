//! Erreurs structurées de cogno-core (contrat §11 : « retourner des erreurs
//! structurées », « les entrées externes ne provoquent pas de panic »).

/// Résultat COGNO fallible.
pub type CognoResult<T> = Result<T, CognoError>;

/// Erreur structurée de l'oracle COGNO.
#[derive(Debug, Clone, PartialEq)]
pub enum CognoError {
    /// Valeur non finie (NaN ou ±infini) dans un calcul.
    NonFinite(&'static str),
    /// Violation de la contrainte non-négative.
    NonNegativeViolation,
    /// Longueur de séquence/vecteur invalide.
    LengthMismatch { expected: usize, got: usize },
    /// Masque invalide (taille, valeurs hors {0,1}, incohérence).
    MaskMismatch(&'static str),
    /// Dépassement arithmétique de taille (multiplication non bornée).
    SizeOverflow,
    /// Dépassement de capacité (cache KV, tampon, budget).
    CapacityOverflow { what: &'static str, capacity: usize, requested: usize },
    /// Entrée invalide (paramètre hors contrat).
    InvalidInput(&'static str),
    /// Backend non disponible / non initialisé.
    BackendUnavailable(&'static str),
}

impl std::fmt::Display for CognoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CognoError::NonFinite(what) => write!(f, "valeur non finie dans {what}"),
            CognoError::NonNegativeViolation => write!(f, "violation de non-négativité"),
            CognoError::LengthMismatch { expected, got } => {
                write!(f, "longueur inattendue : attendu {expected}, reçu {got}")
            }
            CognoError::MaskMismatch(what) => write!(f, "masque invalide : {what}"),
            CognoError::SizeOverflow => write!(f, "dépassement arithmétique de taille"),
            CognoError::CapacityOverflow { what, capacity, requested } => {
                write!(f, "dépassement de capacité {what} : {capacity} < {requested}")
            }
            CognoError::InvalidInput(what) => write!(f, "entrée invalide : {what}"),
            CognoError::BackendUnavailable(what) => write!(f, "backend indisponible : {what}"),
        }
    }
}

impl std::error::Error for CognoError {}

/// Multiplication contrôlée de tailles (contrat §11 : arithmétique contrôlée
/// pour les tailles — pas de `usize * usize` non vérifié).
pub fn checked_mul(a: usize, b: usize) -> CognoResult<usize> {
    a.checked_mul(b).ok_or(CognoError::SizeOverflow)
}

/// Addition contrôlée de tailles.
pub fn checked_add(a: usize, b: usize) -> CognoResult<usize> {
    a.checked_add(b).ok_or(CognoError::SizeOverflow)
}
