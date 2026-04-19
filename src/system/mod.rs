use crate::diff_eq::DifferenceEquation;
use aule::{continuous::Polynomial, prelude::*};
use num_complex::Complex;
use rustnomial::Roots;
use std::str::FromStr;

pub mod first;
pub mod second;

pub trait System: StepResponse {
    fn num(&self) -> &[f64];
    fn den(&self) -> &[f64];
    fn poly(&self) -> &str;
    fn name(&self) -> &str;

    fn poles_analysis(&mut self) {
        let roots = self.roots();

        println!("- Roots of the characteristic polynomial:");
        for (i, root) in roots.iter().enumerate() {
            println!("r{}: {}", i + 1, root);
        }

        if roots.iter().any(|r| r.re > 0.0) {
            println!("- The system is unstable.");
        } else if roots.iter().all(|r| r.re < 0.0) {
            println!("- The system is stable.");
        } else {
            println!("- The system is marginally stable.");
        }

        if roots.iter().all(|r| r.im == 0.0) {
            println!("- The system is overdamped.");
        } else if roots.iter().any(|r| r.im != 0.0) {
            println!("- The system is underdamped.");
        } else {
            println!("- The system is critically damped.");
        }

        let damping_ratios: Vec<f64> = roots
            .iter()
            .map(|r| {
                let real_part = r.re;
                let imag_part = r.im;
                let magnitude = (real_part.powi(2) + imag_part.powi(2)).sqrt();
                if magnitude == 0.0 {
                    1.0 // Critically damped
                } else {
                    -real_part / magnitude
                }
            })
            .collect();
        println!("- Damping ratios:");
        for (i, ratio) in damping_ratios.iter().enumerate() {
            println!("ζ{}: {}", i + 1, ratio);
        }

        let natural_frequencies: Vec<f64> = roots
            .iter()
            .map(|r| (r.re.powi(2) + r.im.powi(2)).sqrt())
            .collect();
        println!("- Natural frequencies:");
        for (i, freq) in natural_frequencies.iter().enumerate() {
            println!("ωn{}: {}", i + 1, freq);
        }

        println!();
    }

    fn roots(&self) -> Vec<Complex<f64>> {
        let poly = rustnomial::Polynomial::from_str(self.poly()).unwrap();

        match poly.roots() {
            Roots::ManyComplexRoots(roots) => roots,
            x => panic!("Expected many complex roots: {:?}", x),
        }
    }

    fn to_dtf(&self, dt: f64) -> DTf<f64> {
        let (num, den) = self.get_num_den(dt);
        println!("- Discrete Transfer Function: H(z) =");
        for i in 0..num.len() {
            print!("{:.2e}", num[i]);

            let degree = num.len() - 1 - i;
            if degree == 0 {
                continue;
            }

            if i == num.len() - 1 {
                print!("z^{}", degree);
            } else {
                print!("z^{} + ", degree);
            }
        }
        let dash_len = den.len().max(num.len()) * 12;
        println!("\n{}", "-".repeat(dash_len));
        for i in 0..den.len() {
            print!("{:.2e}", den[i]);

            let degree = den.len() - 1 - i;
            if degree == 0 {
                continue;
            }

            if i == num.len() - 1 {
                print!("z^{}", degree);
            } else {
                print!("z^{} + ", degree);
            }
        }
        println!();
        DTf::new(&num, &den)
    }

    fn to_diff_eq(&self, dt: f64, noise_coeffs: &[f64]) -> DifferenceEquation {
        let (num, den) = self.get_num_den(dt);
        let eq = DifferenceEquation::new(
            den.iter()
                .skip(1)
                .map(|c| -c)
                .collect::<Vec<_>>()
                .as_slice(),
            num.as_slice(),
            noise_coeffs,
        );
        println!("- Difference Equation: {}", eq);
        eq
    }

    fn get_num_den(&self, dt: f64) -> (Vec<f64>, Vec<f64>) {
        let num = Polynomial::new(self.num());
        let den = Polynomial::new(self.den());

        let alpha = 2.0 / dt;
        let m = num.degree() as usize;
        let n = den.degree() as usize;

        if m > n {
            panic!(
                "The degree of the numerator must be less than or equal to the degree of the denominator."
            );
        }

        let mut num_d = vec![];
        let mut den_d = vec![];

        let z_m1 = Polynomial::new(&[1.0, -1.0]);
        let z_p1 = Polynomial::new(&[1.0, 1.0]);

        for i in 0..m + 1 {
            let coeff = num.coeff()[i] * alpha.powi(m as i32 - i as i32);
            let coeff = Polynomial::new(&[coeff]);
            let value = coeff * z_m1.clone().pow(m - i) * z_p1.clone().pow(n - m + i);
            num_d.push(value);
        }

        for i in 0..n + 1 {
            let coeff = den.coeff()[i] * alpha.powi(n as i32 - i as i32);
            let coeff = Polynomial::new(&[coeff]);
            let value = coeff * z_m1.clone().pow(n - i) * z_p1.clone().pow(i);
            den_d.push(value);
        }

        let num = num_d
            .into_iter()
            .fold(Polynomial::new(&[0.0]), |acc, x| acc + x);
        let den = den_d
            .into_iter()
            .fold(Polynomial::new(&[0.0]), |acc, x| acc + x);

        let lead_coeff = den.coeff()[0];
        let num = num
            .coeff()
            .iter()
            .map(|c| c / lead_coeff)
            .collect::<Vec<_>>();
        let den = den
            .coeff()
            .iter()
            .map(|c| c / lead_coeff)
            .collect::<Vec<_>>();

        (num, den)
    }
}

pub trait StepResponse {
    fn step_response(&mut self, title: impl AsRef<str>, dt: f32, total: f32);
}

impl StepResponse for DifferenceEquation {
    fn step_response(&mut self, title: impl AsRef<str>, dt: f32, total: f32) {
        let simulation = Simulation::new(dt, total);
        let mut step = Step::new(1.0);
        let mut plotter = Plotter::new(title.as_ref().to_string(), ["u(t)", "e(t)", "y(t)"]);
        let mut error = Step::new(0.0);
        let mut sys = self.clone();
        let mut output = vec![];

        for sim_state in simulation {
            let u = sim_state * step.as_block();
            let e = sim_state * error.as_block();
            let input = (u, e).pack();
            let y = sys.output(input);
            output.push(y.value);

            let _ = [u, e, y].pack() * plotter.as_block();
        }

        plotter.display();
        let _ = plotter.save(&format!("images/{}.png", title.as_ref()));
    }
}

impl StepResponse for DTf<f64> {
    fn step_response(&mut self, title: impl AsRef<str>, dt: f32, total: f32) {
        let simulation = Simulation::new(dt, total);
        let mut step = Step::new(1.0);
        let mut plotter = Plotter::new(title.as_ref().to_string(), ["u(t)", "y(t)"]);
        let mut sys = self.clone();
        let mut output = vec![];

        for sim_state in simulation {
            let u = sim_state * step.as_block();
            let y = u * sys.as_block();
            output.push(y.value);

            let _ = [u, y].pack() * plotter.as_block();
        }

        plotter.display();
        let _ = plotter.save(&format!("images/{}.png", title.as_ref()));
    }
}
