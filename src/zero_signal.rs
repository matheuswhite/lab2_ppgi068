use aule::prelude::*;

pub struct ZeroSignal;

impl Block for ZeroSignal {
    type Input = ();
    type Output = f64;

    fn output(&mut self, input: Signal<Self::Input>) -> Signal<Self::Output> {
        input.map(|_| 0.0)
    }
}
