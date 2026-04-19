use aule::prelude::*;

pub struct ArraySignal {
    data: Vec<f64>,
}

impl ArraySignal {
    pub fn new(data: &[f64]) -> Self {
        Self {
            data: data.iter().cloned().rev().collect(),
        }
    }
}

impl Block for ArraySignal {
    type Input = ();
    type Output = f64;

    fn block(&mut self, _input: Self::Input, _sim_state: SimulationState) -> Self::Output {
        self.data.pop().unwrap_or(0.0)
    }
}
