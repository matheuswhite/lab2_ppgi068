use crate::{
    array_signal::ArraySignal,
    diff_eq::DifferenceEquation,
    questions::{AkaikeInformationCriterion, BayesianInformationCriterion, SSE, ordinal_str},
    recursive_extended_least_squares::RecursiveExtendedLeastSquares,
    zero_signal::ZeroSignal,
};
use aule::prelude::*;

pub fn question_6() {
    println!("### Question 6:");
    eval_armax("samples/dados_5.csv", "dados5");
    eval_armax("samples/dados_6.csv", "dados6");
}

fn eval_armax(filename: impl AsRef<str>, dataset_name: impl AsRef<str>) {
    println!("Evaluating ARMAX with RELS for file: {}", filename.as_ref());

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
        .map(|order| identify_armax_rels(estimation.0, estimation.1, order, &dataset_name))
        .collect::<Vec<_>>();

    let sse_values = systems
        .into_iter()
        .enumerate()
        .map(|(order, sys)| {
            let sim_res = sys.simulate(total, ArraySignal::new(validation.1), ZeroSignal);
            let sse = SSE::eval(validation.0, &sim_res.outputs);
            println!("{} Order system SSE: {}", ordinal_str(order + 1), sse);
            sse
        })
        .collect::<Vec<_>>();
    let aic_values = sse_values
        .iter()
        .enumerate()
        .map(|(order, sse)| {
            let aic = AkaikeInformationCriterion::eval(*sse, 3 * (order + 1), total);
            println!("{} Order system AIC: {}", ordinal_str(order + 1), aic);
            aic
        })
        .collect::<Vec<_>>();
    let best_order_aic = aic_values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(order, _)| order + 1)
        .unwrap();
    println!(
        "Best order: {} (AIC: {})",
        best_order_aic,
        aic_values[best_order_aic - 1]
    );

    let bic_values = sse_values
        .iter()
        .enumerate()
        .map(|(order, sse)| {
            let bic = BayesianInformationCriterion::eval(*sse, 3 * (order + 1), total);
            println!("{} Order system BIC: {}", ordinal_str(order + 1), bic);
            bic
        })
        .collect::<Vec<_>>();
    let best_order_bic = bic_values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(order, _)| order + 1)
        .unwrap();
    println!(
        "Best order: {} (BIC: {})",
        best_order_bic,
        bic_values[best_order_bic - 1]
    );

    print_aic_bic_table(filename, &sse_values, &aic_values, &bic_values);
    print_justification_aic_bic(&aic_values, &bic_values);
}

fn identify_armax_rels(
    outputs: &[f64],
    inputs: &[f64],
    order: usize,
    dataset_name: impl AsRef<str>,
) -> DifferenceEquation {
    let total = outputs.len().min(inputs.len());
    let simulation = Simulation::new(1.0, total as f32);
    let mut rels = RecursiveExtendedLeastSquares::new(1000.0, order);
    let mut theta = vec![0.0; 3 * order];
    let mut plotter = PlotterDynamic::new(
        format!("RELS ARMAX on {} order system", ordinal_str(order)),
        (0..order * 3)
            .map(|i| format!("θ{}", i + 1))
            .collect::<Vec<_>>(),
    );

    for (i, sim_state) in simulation.enumerate() {
        let output = outputs[i];
        let input = inputs[i];

        let rls_input = (output, input, 0.0).as_signal(sim_state).pack();
        let new_theta = rls_input * rels.as_block();
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

fn print_aic_bic_table(title: impl AsRef<str>, sse: &[f64], aic: &[f64], bic: &[f64]) {
    println!("{} table:", title.as_ref());
    println!(
        "| {:<10}| {:<12}| {:<12}| {:<12}|",
        "Ordem", "SSE", "AIC", "BIC"
    );
    println!(
        "|-{}|-{}|-{}|-{}|",
        "-".repeat(10),
        "-".repeat(12),
        "-".repeat(12),
        "-".repeat(12)
    );
    for (i, ((s, a), b)) in sse.iter().zip(aic).zip(bic).enumerate() {
        println!("| {:<10}| {:<12.5e}| {:<12.5e}| {:<12.5e}|", i + 1, s, a, b);
    }
}

fn print_justification_aic_bic(aic: &[f64], bic: &[f64]) {
    let (best_aic_order, best_aic) = aic
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    let (best_bic_order, best_bic) = bic
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();

    if best_aic_order == best_bic_order {
        println!(
            "O modelo de ordem {} foi selecionado pois minimiza tanto AIC ({:.4}) quanto BIC ({:.4}).",
            best_aic_order + 1,
            best_aic,
            best_bic
        );
    } else {
        println!(
            "AIC sugere ordem {} (AIC = {:.4}); BIC sugere ordem {} (BIC = {:.4}).",
            best_aic_order + 1,
            best_aic,
            best_bic_order + 1,
            best_bic
        );
        println!(
            "Como BIC penaliza mais a complexidade, optamos pelo modelo de ordem {} (BIC).",
            best_bic_order + 1
        );
    }
}
