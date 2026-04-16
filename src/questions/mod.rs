pub mod question1;
pub mod question2;
pub mod question3;
pub mod question4;
pub mod question5;
pub mod question6;

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

pub struct AkaikeInformationCriterion;

impl AkaikeInformationCriterion {
    pub fn eval(sse: f64, num_params: usize, num_samples: usize) -> f64 {
        2.0 * num_params as f64 + num_samples as f64 * (sse).ln()
    }
}

pub struct BayesianInformationCriterion;

impl BayesianInformationCriterion {
    pub fn eval(sse: f64, num_params: usize, num_samples: usize) -> f64 {
        (num_params as f64) * (num_samples as f64).ln() + num_samples as f64 * sse.ln()
    }
}
