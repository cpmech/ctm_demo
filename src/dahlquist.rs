use crate::{ModelTrait, StrError};
use std::collections::HashMap;

/// Dahlquist model for testing purposes
///
/// ```text
/// y(x) = e^(-λ x)
/// dy/dx = -λ y
/// ```
pub struct Dahlquist {
    lambda: f64,
}

impl Dahlquist {
    /// Allocates a new instance
    ///
    /// # Parameters
    ///
    /// * `lambda` - decay constant (λ)
    pub fn new(params: HashMap<&str, f64>) -> Result<Self, StrError> {
        let lambda = *params.get("lambda").ok_or("Parameter 'lambda' not found")?;
        Ok(Dahlquist { lambda })
    }

    /// Calculates y(x)
    pub fn analytical_y(lambda: f64, x: f64) -> f64 {
        f64::exp(-lambda * x)
    }

    /// Calculates the analytical consistent tangent modulus
    pub fn analytical_ctm(lambda: f64, y1: f64, ddx: f64) -> f64 {
        -lambda * y1 / (1.0 + ddx * lambda)
    }
}

impl ModelTrait for Dahlquist {
    /// Calculates dy/dx = f(x,y)
    fn calc_f(&self, _x: f64, y: f64) -> f64 {
        -self.lambda * y
    }

    /// Calculates L = ∂f/∂x
    fn calc_ll(&self, _x: f64, _y: f64) -> f64 {
        0.0
    }

    /// Calculates J = ∂f/∂y
    fn calc_jj(&self, _x: f64, _y: f64) -> f64 {
        -self.lambda
    }
}

// tests /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_works() {
        let dahlquist = Dahlquist::new(HashMap::from([("lambda", 1.0)])).unwrap();
        assert_eq!(dahlquist.lambda, 1.0);
    }
}
