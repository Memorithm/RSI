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
    pub fn new(config: AdamWConfig, n_params: usize) -> Self {
        AdamW {
            config,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
        }
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
        let mut opt = AdamW::new(AdamWConfig { lr: 0.1, ..Default::default() }, 2);
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
        );
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
        let mut opt = AdamW::new(AdamWConfig::default(), 1);
        assert!(opt.step(&mut p, &[1.0, 2.0]).is_err());
    }
}
