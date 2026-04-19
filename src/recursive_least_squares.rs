use aule::prelude::*;

pub struct RecursiveLeastSquares {
    last_theta: Mat<f64>,
    last_p: Mat<f64>,
    phi: Mat<f64>,
}

pub struct RLSInput {
    pub output: f64,
    pub input: f64,
}

impl RecursiveLeastSquares {
    pub fn new(alpha: f64, order: usize) -> Self {
        Self {
            last_theta: Mat::zeros(2 * order, 1),
            last_p: Mat::<f64>::identity(2 * order, 2 * order) * alpha,
            phi: Mat::zeros(2 * order, 1),
        }
    }

    fn order(&self) -> usize {
        self.last_theta.shape().0 / 2
    }

    fn update_phi(&mut self, input: &RLSInput) {
        let order = self.order();
        // shift phi down and insert new values at the top
        for i in (0..order * 2 - 1).rev() {
            self.phi[(i + 1, 0)] = self.phi[(i, 0)];
        }

        self.phi[(0, 0)] = -input.output;
        self.phi[(order, 0)] = input.input;
    }
}

impl Block for RecursiveLeastSquares {
    type Input = RLSInput;
    type Output = Vec<f64>;

    fn block(&mut self, input: Self::Input, _sim_state: SimulationState) -> Self::Output {
        let kalman_gain_num = &self.last_p * &self.phi;
        let kalman_gain_den = 1.0 + (self.phi.transpose() * &self.last_p * &self.phi)[(0, 0)];
        let kalman_gain = kalman_gain_num / kalman_gain_den;

        let y_k = mat![[input.output]];
        self.last_theta = self.last_theta.clone()
            + &kalman_gain * &(y_k - self.phi.transpose() * &self.last_theta);
        self.last_p = (Mat::<f64>::identity(2 * self.order(), 2 * self.order())
            - kalman_gain * self.phi.transpose())
            * &self.last_p;

        self.update_phi(&input);

        (0..self.last_theta.nrows())
            .map(|i| self.last_theta[(i, 0)])
            .collect::<Vec<_>>()
    }
}

impl Pack<RLSInput> for Signal<(f64, f64)> {
    fn pack(self) -> Signal<RLSInput> {
        RLSInput {
            output: self.value.0,
            input: self.value.1,
        }
        .as_signal(self.sim_state)
    }
}
