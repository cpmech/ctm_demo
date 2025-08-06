use crate::StrError;
use crate::{Dahlquist, HardeningSoftening, ModelTrait, ModelType};
use russell_lab::Vector;
use russell_ode::{Method, OdeSolver, Params, System};
use std::collections::HashMap;

const N_ITERATIONS_MAX: usize = 20;
const BE_TOLERANCE: f64 = 1e-8;
const DELTA: f64 = 1e-5;

/// Represents a stress-strain model with x being strain and y being stress
pub struct Model<'a> {
    actual: Box<dyn ModelTrait>,
    ode_solver: OdeSolver<'a, Box<dyn ModelTrait>>,
}

impl<'a> Model<'a> {
    /// Allocates a new instance
    pub fn new(model_type: ModelType, params: HashMap<&str, f64>, ode_method: Method) -> Result<Self, StrError> {
        let actual: Box<dyn ModelTrait> = match model_type {
            ModelType::Dahlquist => Box::new(Dahlquist::new(params)?),
            ModelType::HardeningSoftening => Box::new(HardeningSoftening::new(params)?),
        };
        let ode_params = Params::new(ode_method);
        let ode_system = System::new(1, |f, x, y, args: &mut Box<dyn ModelTrait>| {
            f[0] = args.calc_f(x, y[0]);
            Ok(())
        });
        let ode_solver = OdeSolver::new(ode_params, ode_system)?;
        Ok(Model { actual, ode_solver })
    }

    /// Performs a backward Euler update
    ///
    /// Calculates x_new and y_new from the total strain increment `Δx`
    pub fn backward_euler_update(&self, x: &mut f64, y: &mut f64, ddx: f64) -> Result<(), StrError> {
        let x0 = *x;
        let y0 = *y;
        let x1 = x0 + ddx;
        let f_trial = self.actual.calc_f(x1, y0);
        let y_trial = y0 + ddx * f_trial;
        *x = x1;
        *y = y_trial;
        let mut converged = false;
        for _ in 0..N_ITERATIONS_MAX {
            let f1 = self.actual.calc_f(*x, *y);
            let r1 = *y - y0 - ddx * f1;
            if f64::abs(r1) < BE_TOLERANCE {
                converged = true;
                break;
            }
            let jj1 = self.actual.calc_jj(*x, *y);
            let dy = -r1 / (1.0 - ddx * jj1);
            *y += dy;
        }
        if !converged {
            return Err("Backward Euler did not converge");
        }
        Ok(())
    }

    /// Performs an update using the ODE solver
    pub fn ode_update(&mut self, x: &mut f64, y: &mut f64, ddx: f64) -> Result<(), StrError> {
        let x0 = *x;
        let x1 = x0 + ddx;
        let mut y0 = Vector::from(&[*y]);
        self.ode_solver.solve(&mut y0, x0, x1, None, &mut self.actual)?;
        *x = x1;
        *y = y0[0];
        Ok(())
    }

    /// Returns the continuous modulus f = dy/dx
    pub fn continuous_modulus(&self, x: f64, y: f64) -> f64 {
        self.actual.calc_f(x, y)
    }

    /// Calculates the consistent tangent modulus @ the update point (x1, y1)
    pub fn consistent_tangent_modulus(&self, x1: f64, y1: f64, ddx: f64) -> f64 {
        let f1 = self.actual.calc_f(x1, y1);
        let ll1 = self.actual.calc_ll(x1, y1);
        let jj1 = self.actual.calc_jj(x1, y1);
        (f1 + ddx * ll1) / (1.0 - ddx * jj1)
    }

    /// Approximates the consistent tangent modulus @ the update point (x1, y1), given the previous point (x0, y0)
    pub fn numerical_consistent_tangent_modulus(
        &mut self,
        x0: f64,
        y0: f64,
        ddx: f64,
        use_ode_solution: bool,
    ) -> Result<f64, StrError> {
        let mut xa = x0;
        let mut ya = y0;
        let mut xb = x0;
        let mut yb = y0;
        if use_ode_solution {
            self.ode_update(&mut xa, &mut ya, ddx)?;
            self.ode_update(&mut xb, &mut yb, ddx + DELTA)?;
        } else {
            self.backward_euler_update(&mut xa, &mut ya, ddx)?;
            self.backward_euler_update(&mut xb, &mut yb, ddx + DELTA)?;
        }
        Ok((yb - ya) / (xb - xa))
    }

    /// Performs a simulation of the model
    ///
    /// Returns `(xx, yy_be, yy_ode, com_list, ctm_list, num_ctm_list, num_ctm_ode_list)` where:
    ///
    /// - `xx` is the vector of x values (strain)
    /// - `yy_be` is the vector of y values (stress) calculated with backward Euler
    /// - `yy_ode` is the vector of y values (stress) calculated with the ODE solver
    /// - `com_list` is the list of continuous moduli
    /// - `ctm_list` is the list of consistent tangent moduli
    /// - `num_ctm_list` is the list of numerical consistent tangent moduli
    /// - `num_ctm_ode_list` is the list of numerical consistent tangent moduli calculated with the ODE solver
    pub fn simulate(
        &mut self,
        x_ini: f64,
        y_ini: f64,
        ddx: f64,
        nd: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), StrError> {
        // Initial values
        let mut x_be = x_ini;
        let mut x_ode = x_ini;
        let mut y_be = y_ini;
        let mut y_ode = y_ini;

        // Perform the backward Euler update
        let mut xx = vec![0.0; nd + 1];
        let mut yy_be = vec![0.0; nd + 1];
        let mut yy_ode = vec![0.0; nd + 1];
        let mut com_list = vec![0.0; nd + 1];
        let mut ctm_list = vec![0.0; nd + 1];
        let mut num_ctm_list = vec![0.0; nd + 1];
        let mut num_ctm_ode_list = vec![0.0; nd + 1];
        let com = self.continuous_modulus(x_be, y_be);
        xx[0] = x_be;
        yy_be[0] = y_be;
        yy_ode[0] = y_ode;
        com_list[0] = com;
        ctm_list[0] = com;
        num_ctm_list[0] = com;
        num_ctm_ode_list[0] = com;
        for k in 1..=nd {
            // x is x0 and y is y0
            let x0 = x_be;
            let y0 = y_be;
            // perform the backward Euler update
            self.backward_euler_update(&mut x_be, &mut y_be, ddx)?;
            // perform the ODE update
            self.ode_update(&mut x_ode, &mut y_ode, ddx)?;
            // x is now x1 and y is now y1
            let x1 = x_be;
            let y1 = y_be;
            // calculate the continuous modulus
            let com = self.continuous_modulus(x1, y1);
            // calculate the consistent tangent modulus
            let ctm = self.consistent_tangent_modulus(x1, y1, ddx);
            let num_ctm = self.numerical_consistent_tangent_modulus(x0, y0, ddx, false)?;
            let num_ctm_ode = self.numerical_consistent_tangent_modulus(x0, y0, ddx, true)?;
            // store the results
            xx[k] = x1;
            yy_be[k] = y1;
            yy_ode[k] = y_ode;
            com_list[k] = com;
            ctm_list[k] = ctm;
            num_ctm_list[k] = num_ctm;
            num_ctm_ode_list[k] = num_ctm_ode;
        }

        // Return the results
        Ok((xx, yy_be, yy_ode, com_list, ctm_list, num_ctm_list, num_ctm_ode_list))
    }
}
