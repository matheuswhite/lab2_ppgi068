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
    eval_armax("samples/dados_5.csv");
    eval_armax("samples/dados_6.csv");
}

fn eval_armax(filename: impl AsRef<str>) {
    println!("Evaluating ARMAX with RELS for file: {}", filename.as_ref());

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

    let total = outputs.len().min(inputs.len());
    let systems = (1..=5)
        .map(|order| identify_armax_rels(&outputs, &inputs, order))
        .collect::<Vec<_>>();

    let sse_values = systems
        .into_iter()
        .enumerate()
        .map(|(order, sys)| {
            let sim_res = sys.simulate(total, ArraySignal::new(&inputs), ZeroSignal);
            let sse = SSE::eval(&outputs[..total], &sim_res.outputs);
            println!("{} Order system SSE: {}", ordinal_str(order + 1), sse);
            sse
        })
        .collect::<Vec<_>>();
    let aic_values = sse_values
        .iter()
        .enumerate()
        .map(|(order, sse)| {
            let aic = AkaikeInformationCriterion::eval(*sse, order + 1, total);
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
            let bic = BayesianInformationCriterion::eval(*sse, order + 1, total);
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
}

fn identify_armax_rels(outputs: &[f64], inputs: &[f64], order: usize) -> DifferenceEquation {
    let total = outputs.len().min(inputs.len());
    let time = Time::new(1.0, total as f32);
    let mut rels = RecursiveExtendedLeastSquares::new(1000.0, order);
    let mut theta = vec![0.0; 3 * order];
    let mut plotter = PlotterDynamic::new(
        format!("RELS ARMAX on {} order system", ordinal_str(order)),
        (0..order * 3)
            .map(|i| format!("θ{}", i + 1))
            .collect::<Vec<_>>(),
    );

    for (i, dt) in time.enumerate() {
        let output = outputs[i];
        let input = inputs[i];

        let rls_input = dt.map(|_| (output, input, 0.0)).pack();
        let new_theta = rls_input * rels.as_block();
        theta = new_theta.value.clone();

        let _ = new_theta * plotter.as_block();
    }

    plotter.display();
    let _ = plotter.save(&format!("images/RELS_ARMAX_{}.png", order));

    let a = theta[..order].to_vec();
    let b = theta[order..2 * order].to_vec();
    let c = theta[2 * order..].to_vec();

    DifferenceEquation::new(&a, &b, &c)
}
