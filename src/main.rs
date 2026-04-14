use crate::{
    questions::{
        question1::question_1, question2::question_2, question3::question_3, question4::question_4,
        question5::question_5, question6::question_6,
    },
    random_signal::RandomSignal,
    system::{first::FirstSystem, second::SecondSystem},
};
use aule::prelude::*;

mod array_signal;
mod diff_eq;
mod gaussian_signal;
mod ordinary_least_squares;
mod questions;
mod random_signal;
mod recursive_extended_least_squares;
mod recursive_least_squares;
mod system;
mod zero_signal;

fn main() {
    println!("# Lab Script 2 - Control Systems Analysis and Design\n");

    println!("## Analyzing the first system:");
    let mut first_system = FirstSystem::default();
    question_1(&mut first_system);
    question_2(&mut first_system, Step::new(1.0));
    question_2(&mut first_system, RandomSignal::new(-1.0, 1.0));
    question_3(&mut first_system, Step::new(1.0));
    question_3(&mut first_system, RandomSignal::new(-1.0, 1.0));

    println!("## Analyzing the second system:");
    let mut second_system = SecondSystem::default();
    question_1(&mut second_system);
    question_2(&mut second_system, Step::new(1.0));
    question_2(&mut second_system, RandomSignal::new(-1.0, 1.0));
    question_3(&mut second_system, Step::new(1.0));
    question_3(&mut second_system, RandomSignal::new(-1.0, 1.0));

    println!("## Analyzing the samples system:");
    question_4();
    question_5();
    question_6();
}
