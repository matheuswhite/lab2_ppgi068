use crate::system::{StepResponse, System};

pub fn question_1(sys: &mut impl System) {
    println!("### Question 1:");

    let dt = 1e-3;
    let name = sys.name().to_string();

    let title = format!("{} Step Response", name);
    sys.step_response(&title, dt as f32, 10.0);

    sys.poles_analysis();

    let dt = 0.1;
    let mut dtf = sys.to_dtf(dt);
    let title = format!("{} Step Response (Discrete)", name);
    dtf.step_response(&title, dt as f32, 10.0);

    let mut diff_eq = sys.to_diff_eq(dt, &[]);
    let title = format!("{} Step Response (Discrete Diff Eq)", name);
    diff_eq.step_response(&title, 1.0, 100.0);
}
