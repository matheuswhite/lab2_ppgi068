use crate::{
    array_signal::ArraySignal,
    diff_eq::DifferenceEquation,
    questions::{SSE, ordinal_str, print_table},
    recursive_extended_least_squares::RecursiveExtendedLeastSquares,
    recursive_least_squares::RecursiveLeastSquares,
    zero_signal::ZeroSignal,
};
use aule::prelude::*;

pub fn question_5() {
    println!("### Question 5:");
    let EvalArxMseOutput {
        sim_outputs: sim_outputs3,
        mse_values: arx_mse3,
    } = eval_arx_mse("dados3", "samples/dados_3.csv");
    let armax_mse3 = eval_armax_mse("dados3", "samples/dados_3.csv", sim_outputs3);
    print_table("samples/dados_3.csv", &arx_mse3, &armax_mse3);
    println!(
        "Conclusão dados_3: Os parâmetros não convergem ao longo das 250 amostras - continuam se ajustando até o fim. Isso indica um sistema VARIANTE NO TEMPO.\n"
    );

    let EvalArxMseOutput {
        sim_outputs: sim_outputs4,
        mse_values: arx_mse4,
    } = eval_arx_mse("dados4", "samples/dados_4.csv");
    let armax_mse4 = eval_armax_mse("dados4", "samples/dados_4.csv", sim_outputs4);
    print_table("samples/dados_4.csv", &arx_mse4, &armax_mse4);
    println!(
        "Conclusão dados_4: Os parâmetros convergem após ~50 amostras e permanecem estáveis. Isso indica um sistema INVARIANTE NO TEMPO.\n"
    );
}

fn eval_arx_mse(dataset_name: impl AsRef<str>, filename: impl AsRef<str>) -> EvalArxMseOutput {
    println!("Evaluating ARX SSE for file: {}", filename.as_ref());

    let mut data1_output = FileSamples::from_csv(filename.as_ref(), 0, 1).unwrap();
    let mut data1_input = FileSamples::from_csv(filename.as_ref(), 0, 2).unwrap();

    let mut outputs = vec![];
    let mut inputs = vec![];

    for sim_state in EndlessSimulation::new(1.0) {
        let Some(output) = (sim_state * data1_output.as_block()).unpack() else {
            break;
        };
        outputs.push(output.value);

        let Some(input) = (sim_state * data1_input.as_block()).unpack() else {
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
        .map(|order| identify_arx_rls(estimation.0, estimation.1, order, &dataset_name))
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

    EvalArxMseOutput {
        sim_outputs,
        mse_values,
    }
}

struct EvalArxMseOutput {
    sim_outputs: Vec<(Vec<f64>, Vec<f64>)>,
    mse_values: Vec<f64>,
}

fn identify_arx_rls(
    outputs: &[f64],
    inputs: &[f64],
    order: usize,
    dataset_name: impl AsRef<str>,
) -> DifferenceEquation {
    let total = outputs.len().min(inputs.len());
    let simulation = Simulation::new(1.0, total as f32);
    let mut rls = RecursiveLeastSquares::new(1000.0, order);
    let mut theta = vec![0.0; 2 * order];
    let mut plotter = PlotterDynamic::new(
        format!("RLS ARX on {} order system", ordinal_str(order)),
        (0..2 * order)
            .map(|i| format!("θ{}", i + 1))
            .collect::<Vec<_>>(),
    );

    for (i, sim_state) in simulation.enumerate() {
        let output = outputs[i];
        let input = inputs[i];

        let new_theta = (output, input).as_signal(sim_state).pack() * rls.as_block();
        theta = new_theta.value.clone();

        let _ = new_theta * plotter.as_block();
    }

    plotter.display();
    let _ = plotter.save(&format!(
        "images/RLS_ARX_{}_{}.png",
        dataset_name.as_ref(),
        order
    ));

    let a = theta[..order].to_vec();
    let b = theta[order..].to_vec();
    let c = vec![0.0; order];

    DifferenceEquation::new(&a, &b, &c)
}

fn eval_armax_mse(
    dataset_name: impl AsRef<str>,
    filename: impl AsRef<str>,
    sim_outputs: Vec<(Vec<f64>, Vec<f64>)>,
) -> Vec<f64> {
    println!("Evaluating ARX SSE for file: {}", filename.as_ref());

    let mut data1_output = FileSamples::from_csv(filename.as_ref(), 0, 1).unwrap();
    let mut data1_input = FileSamples::from_csv(filename.as_ref(), 0, 2).unwrap();

    let mut outputs = vec![];
    let mut inputs = vec![];

    for sim_state in EndlessSimulation::new(1.0) {
        let Some(output) = (sim_state * data1_output.as_block()).unpack() else {
            break;
        };
        outputs.push(output.value);

        let Some(input) = (sim_state * data1_input.as_block()).unpack() else {
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
            identify_armax_rels(
                estimation.0,
                estimation.1,
                &noises[order - 1].0,
                order,
                &dataset_name,
            )
        })
        .collect::<Vec<_>>();

    let mse_values = systems
        .into_iter()
        .enumerate()
        .map(|(order, sys)| {
            let sim_res = sys.simulate(
                total,
                ArraySignal::new(validation.1),
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

fn identify_armax_rels(
    outputs: &[f64],
    inputs: &[f64],
    noises: &[f64],
    order: usize,
    dataset_name: impl AsRef<str>,
) -> DifferenceEquation {
    let total = outputs.len().min(inputs.len());
    let simulation = Simulation::new(1.0, total as f32);
    let mut rels = RecursiveExtendedLeastSquares::new(1000.0, order);
    let mut theta = vec![0.0; 3 * order];
    let mut plotter = PlotterDynamic::new(
        format!("RELS ARMAX on {} order system", ordinal_str(order)),
        (0..3 * order)
            .map(|i| format!("θ{}", i + 1))
            .collect::<Vec<_>>(),
    );

    for (i, sim_state) in simulation.enumerate() {
        let output = outputs[i];
        let input = inputs[i];
        let noise = noises[i];

        let rels_input = (output, input, noise).as_signal(sim_state).pack();
        let new_theta = rels_input * rels.as_block();
        theta = new_theta.value.clone();

        let _ = new_theta * plotter.as_block();
    }

    plotter.display();
    let _ = plotter.save(&format!(
        "images/RELS_ARMAX_{}_{}.png",
        dataset_name.as_ref(),
        order
    ));

    let a = theta[..order].to_vec();
    let b = theta[order..2 * order].to_vec();
    let c = theta[2 * order..].to_vec();

    DifferenceEquation::new(&a, &b, &c)
}
