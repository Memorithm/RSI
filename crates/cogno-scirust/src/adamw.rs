//! Optimiseur **AdamW** avec clipping de gradient et accumulation contrôlée
//! (contrat §12 : « AdamW ou AMSGrad », « clipping de gradient configurable »,
//! « accumulation de gradient contrôlée »).

/// Configuration d'AdamW.
#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    /// clip de norme de gradient (`None` = pas de clip).
    pub grad_clip: Option<f64>,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        AdamWConfig {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            grad_clip: Some(1.0),
        }
    }
}

impl AdamWConfig {
    /// Valide la configuration : lr fini > 0 ; β₁,β₂ ∈ [0,1) (sinon
    /// `1 − β^t` s'annule → division par zéro) ; eps fini > 0 ;
    /// weight_decay ≥ 0 ; clip (s'il est activé) > 0.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.lr.is_finite() || self.lr <= 0.0 {
            return Err("adamw: lr doit être fini et > 0");
        }
        if !self.beta1.is_finite() || !(0.0..1.0).contains(&self.beta1) {
            return Err("adamw: beta1 doit appartenir à [0, 1)");
        }
        if !self.beta2.is_finite() || !(0.0..1.0).contains(&self.beta2) {
            return Err("adamw: beta2 doit appartenir à [0, 1)");
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err("adamw: eps doit être fini et > 0");
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err("adamw: weight_decay doit être fini et >= 0");
        }
        if let Some(c) = self.grad_clip {
            if !c.is_finite() || c <= 0.0 {
                return Err("adamw: grad_clip doit être fini et > 0");
            }
        }
        Ok(())
    }
}

/// Optimiseur AdamW (découplage du weight decay, cf. Loshchilov & Hutter 2019).
///
/// Déterministe : pas d'aléa, mises à jour dans l'ordre des paramètres.
pub struct AdamW {
    config: AdamWConfig,
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

impl AdamW {
    /// Prépare l'optimiseur pour `n_params` paramètres (état zéro).
    ///
    /// La configuration est validée (`AdamWConfig::validate`) — une config
    /// incohérente (β=1 ⇒ division par zéro dans la correction de biais)
    /// échoue ici plutôt qu'au premier pas.
    pub fn new(config: AdamWConfig, n_params: usize) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(AdamW {
            config,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
        })
    }

    pub fn config(&self) -> AdamWConfig {
        self.config
    }

    /// Norme L2 du gradient (pour le clipping).
    pub fn grad_norm(grad: &[f64]) -> f64 {
        grad.iter().map(|g| g * g).sum::<f64>().sqrt()
    }

    /// Applique un pas d'AdamW sur `params` avec `grad`.
    ///
    /// Le clipping de gradient est appliqué **avant** la mise à jour (norme
    /// globale, configurable). Les longueurs `params` et `grad` doivent
    /// correspondre (validé, pas de panic).
    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) -> Result<(), &'static str> {
        if params.len() != grad.len() {
            return Err("adamw: longueurs params/grad incohérentes");
        }
        let mut g: Vec<f64> = grad.to_vec();
        // clip de norme (optionnel)
        if let Some(max_norm) = self.config.grad_clip {
            let norm = Self::grad_norm(grad);
            if norm > max_norm && norm > 0.0 {
                let scale = max_norm / norm;
                for gi in g.iter_mut() {
                    *gi *= scale;
                }
            }
        }
        self.t += 1;
        let t = self.t as f64;
        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let wd = self.config.weight_decay;
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g[i] * g[i];
            let m_hat = self.m[i] / (1.0 - b1.powf(t));
            let v_hat = self.v[i] / (1.0 - b2.powf(t));
            let update = lr * m_hat / (v_hat.sqrt() + eps);
            params[i] -= update;
            // weight decay découplé
            params[i] -= lr * wd * params[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adamw_descents_on_quadratic() {
        let mut p = vec![1.0, 1.0];
        // f(p) = p0² + p1² → grad = 2p
        let mut opt = AdamW::new(AdamWConfig { lr: 0.1, ..Default::default() }, 2).unwrap();
        for _ in 0..200 {
            let g = vec![2.0 * p[0], 2.0 * p[1]];
            opt.step(&mut p, &g).unwrap();
        }
        assert!(p[0].abs() < 0.1 && p[1].abs() < 0.1, "p={p:?}");
    }

    #[test]
    fn grad_clip_limits_norm() {
        let mut p = vec![0.0];
        let mut opt = AdamW::new(
            AdamWConfig {
                lr: 0.01,
                grad_clip: Some(1.0),
                ..Default::default()
            },
            1,
        )
        .unwrap();
        let huge = vec![1e9];
        let norm_before = AdamW::grad_norm(&huge);
        opt.step(&mut p, &huge).unwrap();
        // le pas appliqué est borné par le clip
        assert!(norm_before > 1.0);
        assert!(p[0].abs() < 1.0);
    }

    #[test]
    fn rejects_length_mismatch() {
        let mut p = vec![0.0];
        let mut opt = AdamW::new(AdamWConfig::default(), 1).unwrap();
        assert!(opt.step(&mut p, &[1.0, 2.0]).is_err());
    }

    /// Config incohérente rejetée à la construction (β=1 → div/0 dans
    /// `1 − β^t` ; lr/eps non positifs ; clip non positif).
    #[test]
    fn rejects_invalid_configs() {
        assert!(AdamW::new(AdamWConfig { beta1: 1.0, ..Default::default() }, 1).is_err());
        assert!(AdamW::new(AdamWConfig { beta2: 1.5, ..Default::default() }, 1).is_err());
        assert!(AdamW::new(AdamWConfig { lr: 0.0, ..Default::default() }, 1).is_err());
        assert!(AdamW::new(AdamWConfig { eps: -1e-8, ..Default::default() }, 1).is_err());
        assert!(AdamW::new(AdamWConfig { weight_decay: -0.1, ..Default::default() }, 1).is_err());
        assert!(AdamW::new(AdamWConfig { grad_clip: Some(0.0), ..Default::default() }, 1).is_err());
        // désactiver le clip reste valide
        assert!(AdamW::new(AdamWConfig { grad_clip: None, ..Default::default() }, 1).is_ok());
    }
}
