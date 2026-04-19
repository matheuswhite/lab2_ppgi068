use crate::diff_eq::DifferenceEquation;
use aule::prelude::*;
use faer::linalg::solvers::DenseSolveCore;

pub struct OrdinaryLeastSquares;

impl OrdinaryLeastSquares {
    pub fn identify(
        samples: &[f64],
        input: &[f64],
        error: &[f64],
        output_order: usize,
        input_order: usize,
        noise_order: usize,
    ) -> DifferenceEquation {
        /* theta = (psi_t * psi)^-1 * psi_t * y */

        let n = samples.len().min(input.len()).min(error.len());

        let mut psi = vec![];
        for k in 0..n {
            let mut row = vec![];

            for i in 1..=output_order {
                let y = if k < i { 0.0 } else { samples[k - i] };
                row.push(y);
            }

            for i in 0..input_order {
                let u = if k < i { 0.0 } else { input[k - i] };
                row.push(u);
            }

            for i in 0..noise_order {
                let e = if k < i { 0.0 } else { error[k - i] };
                row.push(e);
            }

            psi.push(row);
        }
        let lines = psi.len();
        let psi = Mat::from_fn(lines, output_order + input_order + noise_order, |i, j| {
            psi[i][j]
        });

        let y = Mat::from_fn(n, 1, |i, _| samples[i]);

        let psi_t = psi.transpose();
        let psi_t_psi = psi_t * &psi;

        let psi_t_psi_inv = psi_t_psi.partial_piv_lu().inverse();
        let psi_t_y = psi_t * &y;
        let theta = &psi_t_psi_inv * &psi_t_y;

        let theta = (0..theta.nrows())
            .map(|i| theta[(i, 0)])
            .collect::<Vec<_>>();
        let a = theta[..output_order].to_vec();
        let b = theta[output_order..output_order + input_order].to_vec();
        let c = theta[output_order + input_order..].to_vec();

        DifferenceEquation::new(&a, &b, &c)
    }
}
