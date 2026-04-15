use aule::prelude::*;
use ndarray::Array2;
use std::fmt::Display;

use crate::gaussian_signal::GaussianSignal;

#[derive(Debug, Clone)]
pub struct DifferenceEquation {
    a: Array2<f64>,
    last_outputs: Array2<f64>,
    b: Array2<f64>,
    last_inputs: Array2<f64>,
    c: Array2<f64>,
    last_errors: Array2<f64>,
}

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
            a: Array2::from_shape_vec((1, a.len()), a.to_vec()).unwrap(),
            last_outputs: Array2::zeros((a.len(), 1)),
            b: Array2::from_shape_vec((1, b.len()), b.to_vec()).unwrap(),
            last_inputs: Array2::zeros((b.len(), 1)),
            c: Array2::from_shape_vec((1, c.len()), c.to_vec()).unwrap(),
            last_errors: Array2::zeros((c.len(), 1)),
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
        let time = Time::new(1.0, total as f32);

        for dt in time {
            let u = input.output(dt);
            inputs.push(u.value);

            let e = noise.output(dt);
            noises.push(e.value);

            let y = self.output((u, e).pack());
            outputs.push(y.value);
        }

        SimulationResult {
            inputs,
            outputs,
            noises,
        }
    }

    pub fn parameters(&self) -> Vec<f64> {
        let mut params = vec![];

        params.extend(self.a.iter());
        params.extend(self.b.iter());
        params.extend(self.c.iter());

        params
    }
}

impl Block for DifferenceEquation {
    type Input = (f64, f64); // (input, error)
    type Output = f64;

    fn output(&mut self, input: Signal<Self::Input>) -> Signal<Self::Output> {
        if self.b.shape()[1] > 0 {
            self.last_inputs[[0, 0]] = input.value.0;
        }
        if self.c.shape()[1] > 0 {
            self.last_errors[[0, 0]] = input.value.1;
        }

        let y = self.a.dot(&self.last_outputs)
            + self.b.dot(&self.last_inputs)
            + self.c.dot(&self.last_errors);

        for i in (1..self.a.shape()[1]).rev() {
            self.last_outputs[[i, 0]] = self.last_outputs[[i - 1, 0]];
        }
        self.last_outputs[[0, 0]] = y[[0, 0]];

        for i in (1..self.b.shape()[1]).rev() {
            self.last_inputs[[i, 0]] = self.last_inputs[[i - 1, 0]];
        }

        for i in (1..self.c.shape()[1]).rev() {
            self.last_errors[[i, 0]] = self.last_errors[[i - 1, 0]];
        }

        input.map(|_| y[[0, 0]])
    }
}

impl Display for DifferenceEquation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = format!("y[k] = ");
        for i in 0..self.a.shape()[1] {
            output += &format!("{:.2}y[k-{}] + ", self.a[[0, i]], i + 1);
        }
        for i in 0..self.b.shape()[1] {
            if i == 0 {
                output += &format!("{:.2}u[k]", self.b[[0, i]]);
            } else {
                output += &format!("{:.2}u[k-{}]", self.b[[0, i]], i);
            }

            if i != self.b.shape()[1] - 1 || self.c.shape()[1] > 0 {
                output += " + ";
            }
        }

        for i in 0..self.c.shape()[1] {
            if i == 0 {
                output += &format!("{:.2}e[k]", self.c[[0, i]]);
            } else {
                output += &format!("{:.2}e[k-{}]", self.c[[0, i]], i);
            }

            if i != self.c.shape()[1] - 1 {
                output += " + ";
            }
        }

        write!(f, "{}", output)
    }
}
