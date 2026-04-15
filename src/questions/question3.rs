use crate::{
    array_signal::ArraySignal,
    diff_eq::{DifferenceEquation, SimulationResult},
    gaussian_signal::GaussianSignal,
    ordinary_least_squares::OrdinaryLeastSquares,
    questions::{SSE, ordinal_str},
    system::System,
    zero_signal::ZeroSignal,
};
use aule::prelude::*;
use std::fmt::Display;

pub fn question_3<I>(sys: &mut impl System, input: I)
where
    I: Block<Input = (), Output = f64> + Clone + Display,
{
    println!("### Question 3 ({}):", input);

    let dt = 0.1;
    let total = 100;

    let diff_eq = sys.to_diff_eq(dt, &[]);
    println!("### Simulating with zero noise...");
    let original_result = diff_eq.simulate(total, input.clone(), ZeroSignal);
    let systems_without_noise = identify_systems(&original_result, false);
    println!("### Simulating Identified System without Gaussian noise...");
    eval_systems(systems_without_noise, total, original_result);

    let diff_eq = sys.to_diff_eq(dt, &[1.0]);
    println!("### Simulating with Gaussian noise...");
    let original_result = diff_eq.simulate(total, input.clone(), GaussianSignal::new(0.0, 0.05));
    let systems_with_dynamic_noise = identify_systems(&original_result, true);
    println!("### Simulating Identified System with Gaussian noise...");
    eval_systems(systems_with_dynamic_noise, total, original_result);

    let diff_eq = sys.to_diff_eq(dt, &[]);
    println!("### Simulating with Gaussian noise at sensor...");
    let mut original_result = diff_eq.simulate(total, input.clone(), ZeroSignal);
    original_result.add_noise_at_end(GaussianSignal::new(0.0, 0.05));
    let system_with_sensor_noise = identify_systems(&original_result, true);
    println!("### Simulating Identified System with Gaussian noise at sensor...");
    eval_systems(system_with_sensor_noise, total, original_result);
}

fn eval_systems(systems: Vec<DifferenceEquation>, total: usize, original_result: SimulationResult) {
    for (order, sys) in systems.into_iter().enumerate() {
        let order = order + 1;
        println!("{} Order identified system: {}", ordinal_str(order), sys);
        let res = sys.simulate(
            total,
            ArraySignal::new(&original_result.inputs),
            ArraySignal::new(&original_result.noises),
        );

        let sse = SSE::eval(&original_result.outputs, &res.outputs);
        println!("SSE: {}", sse);
        let r2 = CoefficientOfDetermination::eval(&original_result.outputs, &res.outputs);
        println!("R²: {}", r2);
        let snr = SignalToNoiseRatio::eval(&original_result.outputs, &res.outputs);
        println!("SNR: {} dB", snr);
    }
}

pub fn identify_systems(
    simulation_result: &SimulationResult,
    enable_noise: bool,
) -> Vec<DifferenceEquation> {
    let noise_order = if !enable_noise { 0 } else { 1 };
    let mut systems = vec![];

    for order in 1..=5 {
        let sys = OrdinaryLeastSquares::identify(
            &simulation_result.outputs,
            &simulation_result.inputs,
            &simulation_result.noises,
            order,
            order,
            noise_order,
        );
        println!("{} Order identified system: {}", ordinal_str(order), sys);
        systems.push(sys);
    }

    systems
}

struct CoefficientOfDetermination;

impl CoefficientOfDetermination {
    pub fn eval(outputs: &[f64], estimated_outputs: &[f64]) -> f64 {
        let sse = SSE::eval(outputs, estimated_outputs);
        let mean = outputs.iter().sum::<f64>() / outputs.len() as f64;
        let sst = outputs.iter().map(|o| (o - mean).powi(2)).sum::<f64>();
        1.0 - sse / sst
    }
}

struct SignalToNoiseRatio;

impl SignalToNoiseRatio {
    pub fn eval(outputs: &[f64], estimated_outputs: &[f64]) -> f64 {
        let sse = SSE::eval(outputs, estimated_outputs);
        -10.0 * (sse / outputs.len() as f64).log10()
    }
}
