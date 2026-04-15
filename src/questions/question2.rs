use crate::{
    array_signal::ArraySignal,
    diff_eq::{DifferenceEquation, SimulationResult},
    gaussian_signal::GaussianSignal,
    ordinary_least_squares::OrdinaryLeastSquares,
    questions::ordinal_str,
    system::System,
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
    let mut parameters_mean = vec![];
    let mut parameters_std = vec![];
    for i in 1..=100 {
        println!("### Gaussian Input for Third Order System ({})", i);
        let mut simulation_result = diff_eq.clone().simulate(total, input.clone(), ZeroSignal);
        simulation_result.add_noise_at_end(GaussianSignal::new(0.0, 0.05));
        let parameters = identify_system_third_system(&simulation_result, total);

        if parameters_mean.is_empty() {
            parameters_mean = vec![0.0; parameters.len()];
            parameters_std = vec![0.0; parameters.len()];
        }

        for (i, p) in parameters.iter().enumerate() {
            parameters_mean[i] += *p / 100.0;
            parameters_std[i] += p.powi(2) / 100.0;
        }
    }

    for (i, (m, s)) in parameters_mean
        .iter()
        .zip(parameters_std.iter())
        .enumerate()
    {
        println!(
            "Parameter {}: mean = {}, std = {}",
            i + 1,
            m,
            (s - m.powi(2)).sqrt()
        );
    }
}

pub fn identify_systems(
    simulation_result: &SimulationResult,
    total: usize,
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
        let SimulationResult {
            outputs: new_outputs,
            ..
        } = sys.clone().simulate(
            total,
            ArraySignal::new(&simulation_result.inputs),
            ArraySignal::new(&simulation_result.noises),
        );
        let residue = simulation_result
            .outputs
            .iter()
            .zip(new_outputs)
            .map(|(o, no)| o - no)
            .collect::<Vec<_>>();
        let residue_mean = residue.iter().sum::<f64>() / residue.len() as f64;
        println!(
            "Residue mean of {} order system: {}",
            ordinal_str(order),
            residue_mean
        );

        let mut residue = ArraySignal::new(&residue);
        let title = format!("{} Order System Residue", ordinal_str(order));
        let mut plotter = Plotter::new(title.clone(), ["Residue"]);
        for dt in Time::new(1.0, total as f32) {
            let _ = dt * residue.as_block() * plotter.as_block();
        }

        plotter.display();
        let _ = plotter.save(&title.replace(" ", "_").to_lowercase());

        systems.push(sys);
    }

    systems
}

fn identify_system_third_system(simulation_result: &SimulationResult, total: usize) -> Vec<f64> {
    let sys = OrdinaryLeastSquares::identify(
        &simulation_result.outputs,
        &simulation_result.inputs,
        &simulation_result.noises,
        3,
        3,
        1,
    );
    println!("Third order identified system: {}", sys);
    let parameters = sys.parameters();

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
        .map(|(o, no)| o - no)
        .collect::<Vec<_>>();
    let residue_mean = residue.iter().sum::<f64>() / residue.len() as f64;
    println!("Residue mean of third order system: {}", residue_mean);

    parameters
}
