use aule::prelude::*;
use rand::{RngExt, rngs::ThreadRng};
use std::fmt::Display;

#[derive(Clone)]
pub struct RandomSignal {
    rng: ThreadRng,
    side: rand::distr::Uniform<f64>,
    min: f64,
    max: f64,
}

impl Display for RandomSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RandomSignal({}, {})", self.min, self.max)
    }
}

impl RandomSignal {
    pub fn new(min: f64, max: f64) -> Self {
        let rng = rand::rng();
        let side = rand::distr::Uniform::new(min, max).unwrap();

        Self {
            rng,
            side,
            min,
            max,
        }
    }
}

impl Block for RandomSignal {
    type Input = ();
    type Output = f64;

    fn output(&mut self, input: Signal<Self::Input>) -> Signal<Self::Output> {
        let value = self.rng.sample(self.side);

        input.map(|_| value)
    }
}
