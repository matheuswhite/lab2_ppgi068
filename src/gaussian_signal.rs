use aule::prelude::*;
use rand::{RngExt, rngs::ThreadRng};
use rand_distr::Normal;

pub struct GaussianSignal {
    rng: ThreadRng,
    normal: Normal<f64>,
}

impl GaussianSignal {
    pub fn new(mean: f64, stddev: f64) -> Self {
        Self {
            rng: rand::rng(),
            normal: Normal::new(mean, stddev).unwrap(),
        }
    }

    pub fn generate(&mut self) -> f64 {
        self.rng.sample(&self.normal)
    }
}

impl Block for GaussianSignal {
    type Input = ();
    type Output = f64;

    fn output(&mut self, input: Signal<Self::Input>) -> Signal<Self::Output> {
        let value = self.generate();
        input.map(|_| value)
    }
}
