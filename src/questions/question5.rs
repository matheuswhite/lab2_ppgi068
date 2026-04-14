use crate::{
    array_signal::ArraySignal,
    diff_eq::DifferenceEquation,
    questions::{SSE, ordinal_str},
    recursive_least_squares::RecursiveLeastSquares,
    zero_signal::ZeroSignal,
};
use aule::prelude::*;

pub fn question_5() {
    println!("### Question 5:");
    eval_arx_sse("samples/dados_3.csv");
    eval_arx_sse("samples/dados_4.csv");
}

fn eval_arx_sse(filename: impl AsRef<str>) {
    println!("Evaluating ARX SSE for file: {}", filename.as_ref());

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
        .map(|order| identify_arx_rls(&outputs, &inputs, order))
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

    let best_order = sse_values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(order, _)| order + 1)
        .unwrap();
    println!(
        "Best order: {} (SSE: {})",
        best_order,
        sse_values[best_order - 1]
    );
}

fn identify_arx_rls(outputs: &[f64], inputs: &[f64], order: usize) -> DifferenceEquation {
    let total = outputs.len().min(inputs.len());
    let time = Time::new(1.0, total as f32);
    let mut rls = RecursiveLeastSquares::new(1000.0, order);
    let mut theta = vec![0.0; 2 * order];
    let mut plotter = PlotterDynamic::new(
        format!("RLS ARX on {} order system", ordinal_str(order)),
        (0..2 * order)
            .map(|i| format!("θ{}", i + 1))
            .collect::<Vec<_>>(),
    );

    for (i, dt) in time.enumerate() {
        let output = outputs[i];
        let input = inputs[i];

        let rls_input = dt.map(|_| (output, input)).pack();
        let new_theta = rls_input * rls.as_block();
        theta = new_theta.value.clone();

        let _ = new_theta * plotter.as_block();
    }

    plotter.display();
    let _ = plotter.save(&format!("images/RLS_ARX_{}.png", order));

    let a = theta[..order].to_vec();
    let b = theta[order..].to_vec();
    let c = vec![0.0; order];

    DifferenceEquation::new(&a, &b, &c)
}
