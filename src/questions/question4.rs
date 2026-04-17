use crate::{
    array_signal::ArraySignal,
    ordinary_least_squares::OrdinaryLeastSquares,
    questions::{SSE, ordinal_str, print_table},
    zero_signal::ZeroSignal,
};
use aule::prelude::*;

pub fn question_4() {
    println!("### Question 4:");

    let (sim_outputs1, arx_mse1) = eval_arx_mse("samples/dados_1.csv");
    let armax_mse1 = eval_armax_mse("samples/dados_1.csv", sim_outputs1);
    print_table("samples/dados_1.csv", &arx_mse1, &armax_mse1);
    print_justification(&arx_mse1, &armax_mse1);

    let (sim_outputs2, arx_mse2) = eval_arx_mse("samples/dados_2.csv");
    let armax_mse2 = eval_armax_mse("samples/dados_2.csv", sim_outputs2);
    print_table("samples/dados_2.csv", &arx_mse2, &armax_mse2);
    print_justification(&arx_mse2, &armax_mse2);
}

fn print_justification(arx_mse: &[f64], armax_mse: &[f64]) {
    let (best_arx_order, best_arx_mse) = arx_mse
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    let (best_armax_order, best_armax_mse) = armax_mse
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    let (best_type, best_order, best_mse) = if best_arx_mse < best_armax_mse {
        ("ARX", best_arx_order + 1, best_arx_mse)
    } else {
        ("ARMAX", best_armax_order + 1, best_armax_mse)
    };
    println!(
        "O modelo {} de ordem {} foi selecionado pois:",
        best_type, best_order
    );
    println!(
        "Apresenta o menor MSE entre os modelos testados ({})\n",
        best_mse
    );
}

fn eval_arx_mse(filename: impl AsRef<str>) -> (Vec<(Vec<f64>, Vec<f64>)>, Vec<f64>) {
    println!("Evaluating ARX MSE for file: {}", filename.as_ref());

    let mut data1_output = FileSamples::from_csv(filename.as_ref(), 0, 1).unwrap();
    let mut data1_input = FileSamples::from_csv(filename.as_ref(), 0, 2).unwrap();

    let mut outputs = vec![];
    let mut inputs = vec![];

    for dt in EndlessTime::new(1.0) {
        let Some(output) = (dt * data1_output.as_block()).unpack() else {
            break;
        };
        outputs.push(output.value);

        let Some(input) = (dt * data1_input.as_block()).unpack() else {
            break;
        };
        inputs.push(input.value);
    }

    let n = outputs.len().min(inputs.len());
    let pivot = n / 2;
    let estimation = (&outputs[..pivot], &inputs[..pivot]);
    let validation = (&outputs[pivot..], &inputs[pivot..]);
    let total = validation.0.len();

    let systems = (1..=5)
        .map(|order| {
            OrdinaryLeastSquares::identify(
                estimation.0,
                estimation.1,
                &vec![0.0; estimation.0.len()],
                order,
                order,
                0,
            )
        })
        .collect::<Vec<_>>();

    let sim_outputs = systems
        .iter()
        .map(|sys| {
            let sim_res_est =
                sys.clone()
                    .simulate(total, ArraySignal::new(estimation.1), ZeroSignal);
            let sim_res_val =
                sys.clone()
                    .simulate(total, ArraySignal::new(validation.1), ZeroSignal);
            (sim_res_est.outputs, sim_res_val.outputs)
        })
        .collect::<Vec<_>>();

    let mse_values = systems
        .into_iter()
        .enumerate()
        .map(|(order, sys)| {
            let sim_res = sys.simulate(total, ArraySignal::new(validation.1), ZeroSignal);
            let mse = SSE::eval(validation.0, &sim_res.outputs) / total as f64;
            println!("{} Order system MSE: {}", ordinal_str(order + 1), mse);
            mse
        })
        .collect::<Vec<_>>();

    let best_order = mse_values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(order, _)| order + 1)
        .unwrap();
    println!(
        "Best order: {} (MSE: {})",
        best_order,
        mse_values[best_order - 1]
    );

    (sim_outputs, mse_values)
}

fn eval_armax_mse(filename: impl AsRef<str>, sim_outputs: Vec<(Vec<f64>, Vec<f64>)>) -> Vec<f64> {
    println!("Evaluating ARMAX MSE for file: {}", filename.as_ref());

    let mut data1_output = FileSamples::from_csv(filename.as_ref(), 0, 1).unwrap();
    let mut data1_input = FileSamples::from_csv(filename.as_ref(), 0, 2).unwrap();

    let mut outputs = vec![];
    let mut inputs = vec![];

    for dt in EndlessTime::new(1.0) {
        let Some(output) = (dt * data1_output.as_block()).unpack() else {
            break;
        };
        outputs.push(output.value);

        let Some(input) = (dt * data1_input.as_block()).unpack() else {
            break;
        };
        inputs.push(input.value);
    }

    let n = outputs.len().min(inputs.len());
    let pivot = n / 2;
    let estimation = (&outputs[..pivot], &inputs[..pivot]);
    let validation = (&outputs[pivot..], &inputs[pivot..]);
    let total = validation.0.len();

    let noises = sim_outputs
        .into_iter()
        .map(|sim_outputs| {
            (
                sim_outputs
                    .0
                    .iter()
                    .zip(estimation.0)
                    .map(|(so, o)| o - so)
                    .collect::<Vec<_>>(),
                sim_outputs
                    .1
                    .iter()
                    .zip(validation.0)
                    .map(|(so, o)| o - so)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    let systems = (1..=5)
        .map(|order| {
            OrdinaryLeastSquares::identify(
                estimation.0,
                estimation.1,
                &noises[order - 1].0,
                order,
                order,
                order,
            )
        })
        .collect::<Vec<_>>();

    let mse_values = systems
        .into_iter()
        .enumerate()
        .map(|(order, sys)| {
            let sim_res = sys.simulate(
                total,
                ArraySignal::new(&validation.1),
                ArraySignal::new(&noises[order].1),
            );
            let mse = SSE::eval(validation.0, &sim_res.outputs) / total as f64;
            println!("{} Order system MSE: {}", ordinal_str(order + 1), mse);
            mse
        })
        .collect::<Vec<_>>();

    let best_order = mse_values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(order, _)| order + 1)
        .unwrap();
    println!(
        "Best order: {} (MSE: {})",
        best_order,
        mse_values[best_order - 1]
    );

    mse_values
}
