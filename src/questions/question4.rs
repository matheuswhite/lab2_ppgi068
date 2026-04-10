use aule::prelude::*;

use crate::{
    array_signal::ArraySignal, ordinary_least_squares::OrdinaryLeastSquares, questions::SSE,
    zero_signal::ZeroSignal,
};

pub fn question_4() {
    println!("### Question 4:");
    eval_sse("samples/dados_1.csv");
    eval_sse("samples/dados_2.csv");
}

fn eval_sse(filename: impl AsRef<str>) {
    println!("Evaluating SSE for file: {}", filename.as_ref());

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
        .map(|order| {
            OrdinaryLeastSquares::identify(
                &outputs,
                &inputs,
                &vec![0.0; outputs.len()],
                order,
                order,
                0,
            )
        })
        .collect::<Vec<_>>();

    let sse_values = systems
        .into_iter()
        .enumerate()
        .map(|(order, sys)| {
            let sim_res = sys.simulate(total, ArraySignal::new(&inputs), ZeroSignal);
            let sse = SSE::eval(&outputs[..total], &sim_res.outputs);
            println!("{} Order system SSE: {}", order + 1, sse);
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
