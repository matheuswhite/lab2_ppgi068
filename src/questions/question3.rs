use crate::{
    array_signal::ArraySignal,
    diff_eq::SimulationResult,
    gaussian_signal::GaussianSignal,
    ordinary_least_squares::OrdinaryLeastSquares,
    questions::{SSE, ordinal_str, question2::Question2Output},
    zero_signal::ZeroSignal,
};
use aule::prelude::*;
use std::fmt::Display;

pub fn question_3<I>(question2_output: Question2Output, input: I)
where
    I: Block<Input = (), Output = f64> + Clone + Display,
{
    println!("### Question 3 ({}):", input);

    let total = 100;

    println!("### Using the system without noise for estimation and validation...");
    let (diff_eq, simulation_result) = question2_output.system_without_noise;
    let estimation_data = simulation_result.clone();
    let diff_eq = diff_eq.clone();
    let validation_data = diff_eq.simulate(total, input.clone(), ZeroSignal);
    println!("### Evaluating identified systems...");
    eval_systems(&estimation_data, &validation_data, total, 0);

    println!("### Using the system with noise for estimation and validation...");
    let (diff_eq, simulation_result) = question2_output.system_with_noise;
    let estimation_data = simulation_result.clone();
    let diff_eq = diff_eq.clone();
    let validation_data = diff_eq.simulate(total, input.clone(), GaussianSignal::new(0.0, 0.05));
    println!("### Evaluating identified systems...");
    eval_systems(&estimation_data, &validation_data, total, 1);

    println!("### Using the system with noise at the end for estimation and validation...");
    let (diff_eq, simulation_result) = question2_output.system_with_noise_at_end;
    let estimation_data = simulation_result.clone();
    let diff_eq = diff_eq.clone();
    let mut validation_data = diff_eq.simulate(total, input.clone(), ZeroSignal);
    validation_data.add_noise_at_end(GaussianSignal::new(0.0, 0.05));
    println!("### Evaluating identified systems...");
    eval_systems(&estimation_data, &validation_data, total, 1);
}

fn eval_systems(
    estimation_data: &SimulationResult,
    validation_data: &SimulationResult,
    total: usize,
    noise_order: usize,
) {
    for order in 0..5 {
        let order = order + 1;
        let model = OrdinaryLeastSquares::identify(
            &estimation_data.outputs,
            &estimation_data.inputs,
            &estimation_data.noises,
            order,
            order,
            noise_order,
        );
        println!("{} Order identified system: {}", ordinal_str(order), model);

        let inputs = ArraySignal::new(&validation_data.inputs);
        let noises = ArraySignal::new(&validation_data.noises);
        let model_res = model.simulate(total, inputs, noises);

        let sse = SSE::eval(&validation_data.outputs, &model_res.outputs);
        println!("SSE: {}", sse);
        let r2 = CoefficientOfDetermination::eval(&validation_data.outputs, &model_res.outputs);
        println!("R²: {}", r2);
        let snr = SignalToNoiseRatio::eval(&validation_data.outputs, &model_res.outputs);
        println!("SNR: {} dB", snr);
    }
}

struct CoefficientOfDetermination;

impl CoefficientOfDetermination {
    pub fn eval(outputs: &[f64], estimated_outputs: &[f64]) -> f64 {
        let sse = SSE::eval(outputs, estimated_outputs);
        let mean = outputs.iter().sum::<f64>() / outputs.len() as f64;
        let sst = outputs.iter().map(|o| (o - mean).powi(2)).sum::<f64>();
        (1.0 - sse / sst).sqrt()
    }
}

struct SignalToNoiseRatio;

impl SignalToNoiseRatio {
    pub fn eval(outputs: &[f64], estimated_outputs: &[f64]) -> f64 {
        let power_output = outputs.iter().map(|o| o.powi(2)).sum::<f64>();
        let power_error = estimated_outputs
            .iter()
            .zip(outputs)
            .map(|(eo, o)| (o - eo).powi(2))
            .sum::<f64>();
        10.0 * (power_output / power_error).log10()
    }
}
