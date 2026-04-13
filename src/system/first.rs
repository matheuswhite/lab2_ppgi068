use crate::system::{StepResponse, System};
use aule::prelude::*;

pub struct FirstSystem {
    num: Vec<f64>,
    den: Vec<f64>,
}

impl Default for FirstSystem {
    fn default() -> Self {
        Self {
            num: vec![0.5, 2.0, 2.0],
            den: vec![1.0, 3.0, 4.0, 2.0],
        }
    }
}

impl System for FirstSystem {
    fn num(&self) -> &[f64] {
        self.num.as_slice()
    }

    fn den(&self) -> &[f64] {
        self.den.as_slice()
    }

    fn poly(&self) -> &str {
        "x^3 + 3x^2 + 4x + 2"
    }

    fn name(&self) -> &str {
        "First System"
    }
}

impl StepResponse for FirstSystem {
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
        let res = plotter.save(&format!("images/{}.png", title.as_ref()));
        println!("Saved result: {:?}", res);
    }
}
