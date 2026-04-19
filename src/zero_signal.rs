use aule::prelude::*;

pub struct ZeroSignal;

impl Block for ZeroSignal {
    type Input = ();
    type Output = f64;

    fn block(&mut self, _input: Self::Input, _sim_state: SimulationState) -> Self::Output {
        0.0
    }
}
