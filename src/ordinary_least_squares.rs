use ndarray::Array2;
use ndarray::s;
use ndarray_inverse::Inverse;

use crate::diff_eq::DifferenceEquation;

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
        let psi = psi.into_iter().flatten().collect::<Vec<_>>();
        let psi =
            Array2::from_shape_vec((lines, output_order + input_order + noise_order), psi).unwrap();

        let y = Array2::from_shape_vec((samples.len(), 1), samples.to_vec()).unwrap();

        let psi_t = psi.t();
        let psi_t_psi = psi_t.dot(&psi);
        let psi_t_psi_inv = psi_t_psi.inv().unwrap();
        let psi_t_y = psi_t.dot(&y);
        let theta = psi_t_psi_inv.dot(&psi_t_y);

        let a = theta.slice(s![..output_order, 0]).to_owned().into_raw_vec();
        let b = theta
            .slice(s![output_order..output_order + input_order, 0])
            .to_owned()
            .into_raw_vec();
        let c = theta
            .slice(s![output_order + input_order.., 0])
            .to_owned()
            .into_raw_vec();

        DifferenceEquation::new(&a, &b, &c)
    }
}
