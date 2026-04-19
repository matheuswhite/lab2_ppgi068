use aule::prelude::*;
use std::fmt::Display;

use crate::gaussian_signal::GaussianSignal;

#[derive(Debug, Clone)]
pub struct DifferenceEquation {
    a: Mat<f64>,
    last_outputs: Mat<f64>,
    b: Mat<f64>,
    last_inputs: Mat<f64>,
    c: Mat<f64>,
    last_errors: Mat<f64>,
}

#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,
    pub noises: Vec<f64>,
}

impl SimulationResult {
    pub fn add_noise_at_end(&mut self, mut noise: GaussianSignal) {
        self.noises = self.outputs.iter().map(|_| noise.generate()).collect();
        for (o, n) in self.outputs.iter_mut().zip(self.noises.iter()) {
            *o += n;
        }
    }
}

impl DifferenceEquation {
    pub fn new(a: &[f64], b: &[f64], c: &[f64]) -> Self {
        Self {
            a: Mat::from_fn(1, a.len(), |_, j| a[j]),
            last_outputs: Mat::from_fn(a.len(), 1, |i, _| a[i]),
            b: Mat::from_fn(1, b.len(), |_, j| b[j]),
            last_inputs: Mat::from_fn(b.len(), 1, |i, _| b[i]),
            c: Mat::from_fn(1, c.len(), |_, j| c[j]),
            last_errors: Mat::from_fn(c.len(), 1, |i, _| c[i]),
        }
    }

    pub fn simulate(
        mut self,
        total: usize,
        mut input: impl Block<Input = (), Output = f64>,
        mut noise: impl Block<Input = (), Output = f64>,
    ) -> SimulationResult {
        let mut inputs = vec![];
        let mut outputs = vec![];
        let mut noises = vec![];
        let simulation = Simulation::new(1.0, total as f32);

        for sim_state in simulation {
            let u = input.block((), sim_state);
            inputs.push(u);

            let e = noise.block((), sim_state);
            noises.push(e);

            let y = self.block((u, e), sim_state);
            outputs.push(y);
        }

        SimulationResult {
            inputs,
            outputs,
            noises,
        }
    }

    pub fn parameters(&self) -> Vec<f64> {
        let mut params = vec![];

        params.extend(self.a.col_as_slice(0));
        params.extend(self.b.col_as_slice(0));
        params.extend(self.c.col_as_slice(0));

        params
    }
}

impl Block for DifferenceEquation {
    type Input = (f64, f64); // (input, error)
    type Output = f64;

    fn block(&mut self, input: Self::Input, _sim_state: SimulationState) -> Self::Output {
        if self.b.shape().1 > 0 {
            self.last_inputs[(0, 0)] = input.0;
        }
        if self.c.shape().1 > 0 {
            self.last_errors[(0, 0)] = input.1;
        }

        let y = &self.a * &self.last_outputs
            + &self.b * &self.last_inputs
            + &self.c * &self.last_errors;

        for i in (1..self.a.shape().1).rev() {
            self.last_outputs[(i, 0)] = self.last_outputs[(i - 1, 0)];
        }
        self.last_outputs[(0, 0)] = y[(0, 0)];

        for i in (1..self.b.shape().1).rev() {
            self.last_inputs[(i, 0)] = self.last_inputs[(i - 1, 0)];
        }

        for i in (1..self.c.shape().1).rev() {
            self.last_errors[(i, 0)] = self.last_errors[(i - 1, 0)];
        }

        y[(0, 0)]
    }
}

impl Display for DifferenceEquation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = "y[k] = ".to_string();
        for i in 0..self.a.shape().1 {
            output += &format!("{:.2}y[k-{}] + ", self.a[(0, i)], i + 1);
        }
        for i in 0..self.b.shape().1 {
            if i == 0 {
                output += &format!("{:.2}u[k]", self.b[(0, i)]);
            } else {
                output += &format!("{:.2}u[k-{}]", self.b[(0, i)], i);
            }

            if i != self.b.shape().1 - 1 || self.c.shape().1 > 0 {
                output += " + ";
            }
        }

        for i in 0..self.c.shape().1 {
            if i == 0 {
                output += &format!("{:.2}e[k]", self.c[(0, i)]);
            } else {
                output += &format!("{:.2}e[k-{}]", self.c[(0, i)], i);
            }

            if i != self.c.shape().1 - 1 {
                output += " + ";
            }
        }

        write!(f, "{}", output)
    }
}
