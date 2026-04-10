use crate::system::{StepResponse, System};
use aule::prelude::*;

pub struct SecondSystem {
    num: Vec<f64>,
    den: Vec<f64>,
}

impl System for SecondSystem {
    fn num(&self) -> &[f64] {
        self.num.as_slice()
    }

    fn den(&self) -> &[f64] {
        self.den.as_slice()
    }

    fn poly(&self) -> &str {
        "x^2 + x + 2.5"
    }

    fn name(&self) -> &str {
        "Second System"
    }
}

impl StepResponse for SecondSystem {
    fn step_response(&mut self, title: impl AsRef<str>, dt: f32, total: f32) {
        let time = Time::new(dt, total);
        let mut step = Step::new(1.0);
        let mut plotter = Plotter::new(title.as_ref().to_string(), ["u(t)", "y(t)"]);
        let mut sys = Tf::new(&self.num, &self.den).to_ss_controllable(RK4);
        let mut output = vec![];

        for dt in time {
            let u = dt * step.as_block();
            let y = sys.output(u);
            output.push(y.value);

            let _ = [u, y].pack() * plotter.as_block();
        }

        plotter.display();
        let _ = plotter.save(&format!("images/{}.png", title.as_ref()));
    }
}

impl Default for SecondSystem {
    fn default() -> Self {
        Self {
            num: vec![2.5],
            den: vec![1.0, 1.0, 2.5],
        }
    }
}
