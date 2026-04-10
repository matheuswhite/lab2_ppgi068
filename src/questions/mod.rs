use crate::{
    array_signal::ArraySignal,
    diff_eq::{DifferenceEquation, SimulationResult},
    ordinary_least_squares::OrdinaryLeastSquares,
};

pub mod question1;
pub mod question2;
pub mod question3;
pub mod question4;
pub mod question5;
pub mod question6;

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
            .map(|(o, no)| no - o)
            .collect::<Vec<_>>();
        let residue_mean = residue.iter().sum::<f64>() / residue.len() as f64;
        println!(
            "Residue mean of {} order system: {}",
            ordinal_str(order),
            residue_mean
        );
        systems.push(sys);
    }

    systems
}

pub fn ordinal_str(value: usize) -> String {
    format!(
        "{}{}",
        value,
        match value {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        }
    )
}

pub struct SSE;

impl SSE {
    pub fn eval(outputs: &[f64], estimated_outputs: &[f64]) -> f64 {
        outputs
            .iter()
            .zip(estimated_outputs)
            .map(|(o, eo)| (o - eo).powi(2))
            .sum::<f64>()
    }
}
