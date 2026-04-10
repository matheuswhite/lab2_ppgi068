use crate::{
    array_signal::ArraySignal, diff_eq::SimulationResult, gaussian_signal::GaussianSignal,
    ordinary_least_squares::OrdinaryLeastSquares, questions::identify_systems, system::System,
    zero_signal::ZeroSignal,
};
use aule::prelude::*;
use std::fmt::Display;

pub fn question_2<I>(sys: &mut impl System, input: I)
where
    I: Block<Input = (), Output = f64> + Clone + Display,
{
    println!("### Question 2 ({}):", input);

    let dt = 0.1;
    let total = 100;

    let diff_eq = sys.to_diff_eq(dt, &[]);
    println!("### Simulating with zero noise...");
    let simulation_result = diff_eq.simulate(total, input.clone(), ZeroSignal);
    identify_systems(&simulation_result, total, false);

    let diff_eq = sys.to_diff_eq(dt, &[1.0]);
    println!("### Simulating with Gaussian noise...");
    let simulation_result = diff_eq.simulate(total, input.clone(), GaussianSignal::new(0.0, 0.05));
    identify_systems(&simulation_result, total, true);

    let diff_eq = sys.to_diff_eq(dt, &[]);
    println!("### Simulating with Gaussian noise at the end...");
    let mut simulation_result = diff_eq.simulate(total, input.clone(), ZeroSignal);
    simulation_result.add_noise_at_end(GaussianSignal::new(0.0, 0.05));
    identify_systems(&simulation_result, total, true);

    let diff_eq = sys.to_diff_eq(dt, &[]);
    for i in 0..=100 {
        println!("### Gaussian Input for Third Order System ({})", i);
        let mut simulation_result = diff_eq.clone().simulate(total, input.clone(), ZeroSignal);
        simulation_result.add_noise_at_end(GaussianSignal::new(0.0, 0.05));
        identify_system_third_system(&simulation_result, total);
    }
}

fn identify_system_third_system(simulation_result: &SimulationResult, total: usize) {
    let sys = OrdinaryLeastSquares::identify(
        &simulation_result.outputs,
        &simulation_result.inputs,
        &simulation_result.noises,
        3,
        3,
        1,
    );
    println!("Third order identified system: {}", sys);
    let SimulationResult {
        outputs: new_outputs,
        ..
    } = sys.simulate(
        total,
        ArraySignal::new(&simulation_result.inputs),
        ArraySignal::new(&simulation_result.noises),
    );
    let residue = simulation_result
        .outputs
        .iter()
        .zip(new_outputs)
        .map(|(o, no)| no - o)
        .collect::<Vec<_>>();
    let residue_mean = residue.iter().sum::<f64>() / residue.len() as f64;
    println!("Residue mean of third order system: {}", residue_mean);
}
