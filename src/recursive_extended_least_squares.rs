use aule::prelude::*;

pub struct RecursiveExtendedLeastSquares {
    last_theta: Mat<f64>,
    last_p: Mat<f64>,
    phi: Mat<f64>,
}

pub struct RELSInput {
    pub output: f64,
    pub input: f64,
    pub noise: f64,
}

impl RecursiveExtendedLeastSquares {
    pub fn new(alpha: f64, order: usize) -> Self {
        Self {
            last_theta: Mat::zeros(3 * order, 1),
            last_p: Mat::<f64>::identity(3 * order, 3 * order) * alpha,
            phi: Mat::zeros(3 * order, 1),
        }
    }

    fn order(&self) -> usize {
        self.last_theta.shape().0 / 3
    }

    fn update_phi(&mut self, input: &RELSInput) {
        let order = self.order();
        // shift phi down and insert new values at the top
        for i in (0..order * 2 - 1).rev() {
            self.phi[(i + 1, 0)] = self.phi[(i, 0)];
        }

        self.phi[(0, 0)] = -input.output;
        self.phi[(order, 0)] = input.input;
        self.phi[(2 * order, 0)] = input.noise;
    }
}

impl Block for RecursiveExtendedLeastSquares {
    type Input = RELSInput;
    type Output = Vec<f64>;

    fn block(&mut self, mut input: Self::Input, _sim_state: SimulationState) -> Self::Output {
        input.noise = input.output - (self.phi.transpose() * &self.last_theta)[(0, 0)];

        let kalman_gain_num = &self.last_p * &self.phi;
        let kalman_gain_den = 1.0 + (self.phi.transpose() * &self.last_p * &self.phi)[(0, 0)];
        let kalman_gain = kalman_gain_num / kalman_gain_den;

        let y_k = mat![[input.output]];
        self.last_theta = self.last_theta.clone()
            + &kalman_gain * &(y_k - self.phi.transpose() * &self.last_theta);
        self.last_p = (Mat::<f64>::identity(3 * self.order(), 3 * self.order())
            - kalman_gain * self.phi.transpose())
            * &self.last_p;

        self.update_phi(&input);

        (0..self.last_theta.nrows())
            .map(|i| self.last_theta[(i, 0)])
            .collect::<Vec<_>>()
    }
}

impl Pack<RELSInput> for Signal<(f64, f64, f64)> {
    fn pack(self) -> Signal<RELSInput> {
        RELSInput {
            output: self.value.0,
            input: self.value.1,
            noise: self.value.2,
        }
        .as_signal(self.sim_state)
    }
}
