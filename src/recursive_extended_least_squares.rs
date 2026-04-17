use aule::prelude::*;
use ndarray::Array2;

pub struct RecursiveExtendedLeastSquares {
    last_theta: Array2<f64>,
    last_p: Array2<f64>,
    phi: Array2<f64>,
}

pub struct RELSInput {
    pub output: f64,
    pub input: f64,
    pub noise: f64,
}

impl RecursiveExtendedLeastSquares {
    pub fn new(alpha: f64, order: usize) -> Self {
        Self {
            last_theta: Array2::zeros((3 * order, 1)),
            last_p: Array2::eye(3 * order) * alpha,
            phi: Array2::zeros((3 * order, 1)),
        }
    }

    fn order(&self) -> usize {
        self.last_theta.shape()[0] / 3
    }

    fn update_phi(&mut self, input: &RELSInput) {
        let order = self.order();
        // shift phi down and insert new values at the top
        for i in (0..order * 2 - 1).rev() {
            self.phi[[i + 1, 0]] = self.phi[[i, 0]];
        }

        self.phi[[0, 0]] = -input.output;
        self.phi[[order, 0]] = input.input;
        self.phi[[2 * order, 0]] = input.noise;
    }
}

impl Block for RecursiveExtendedLeastSquares {
    type Input = RELSInput;
    type Output = Vec<f64>;

    fn output(&mut self, mut input: Signal<Self::Input>) -> Signal<Self::Output> {
        input.value.noise = input.value.output - self.phi.t().dot(&self.last_theta)[[0, 0]];

        let kalman_gain_num = self.last_p.dot(&self.phi);
        let kalman_gain_den = 1.0 + self.phi.t().dot(&self.last_p).dot(&self.phi);
        let kalman_gain = kalman_gain_num / kalman_gain_den;

        let y_k = Array2::from_shape_vec((1, 1), vec![input.value.output]).unwrap();
        self.last_theta =
            self.last_theta.clone() + kalman_gain.dot(&(y_k - self.phi.t().dot(&self.last_theta)));
        self.last_p =
            (Array2::eye(3 * self.order()) - kalman_gain.dot(&self.phi.t())).dot(&self.last_p);

        self.update_phi(&input.value);

        input.map(|_| self.last_theta.clone().into_raw_vec())
    }
}

impl Pack<RELSInput> for Signal<(f64, f64, f64)> {
    fn pack(self) -> Signal<RELSInput> {
        self.map(|(output, input, noise)| RELSInput {
            output,
            input,
            noise,
        })
    }
}
