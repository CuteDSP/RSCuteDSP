// Example demonstrating phase rotation and Hilbert transform

use cute_dsp::phase_rotation::{
    HilbertTransform, PhaseRotator, instantaneous_phase, instantaneous_magnitude, instantaneous_frequency
};
use std::f32::consts::PI;

fn main() {
    println!("Phase Rotation & Hilbert Transform Example");
    println!("==========================================\n");

    // Example 1: Phase Rotator - Simple oscillation
    println!("Example 1: Phase Rotator (1 Hz at 10 Hz sample rate)");
    let mut rotator = PhaseRotator::new(1.0, 10.0);
    let input = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let output: Vec<f32> = input.iter().map(|&s| rotator.process(s)).collect();
    println!("Input:  {:?}", input);
    println!("Output: {:?}\n", output);

    // Example 2: Phase Rotator - Quadrature output
    println!("Example 2: Quadrature Processing (I/Q)");
    let mut rotator_iq = PhaseRotator::new(2.0, 20.0);
    println!("Sample | In-Phase | Quadrature");
    for i in 0..5 {
        let sample = 1.0;
        let (in_phase, quadrature) = rotator_iq.process_quadrature(sample);
        println!("  {}    | {:.3}    | {:.3}", i, in_phase, quadrature);
    }
    println!();

    // Example 3: Hilbert Transform - Creating a simple test signal
    println!("Example 3: Hilbert Transform");
    let mut hilbert = HilbertTransform::new(64);
    let signal = create_test_signal(10);
    println!("Original signal (first 10 samples): {:?}", &signal[..10]);
    
    let hilbert_result = hilbert.transform(&signal);
    println!("Hilbert transformed (first 10 samples): {:?}\n", &hilbert_result[..10]);

    // Example 4: Analytic Signal
    println!("Example 4: Analytic Signal Generation");
    let mut hilbert2 = HilbertTransform::new(64);
    let signal2 = create_test_signal(8);
    let analytic = hilbert2.analytic_signal(&signal2);
    println!("Real (original) | Imaginary (Hilbert) | Magnitude");
    for c in analytic.iter() {
        let mag = (c.re * c.re + c.im * c.im).sqrt();
        println!("{:.3}           | {:.3}              | {:.3}", c.re, c.im, mag);
    }
    println!();

    // Example 5: Instantaneous Phase and Frequency
    println!("Example 5: Instantaneous Phase and Frequency");
    let mut hilbert3 = HilbertTransform::new(128);
    let signal3 = create_chirp_signal(50);
    let analytic3 = hilbert3.analytic_signal(&signal3);
    let phase = instantaneous_phase(&analytic3);
    let magnitude = instantaneous_magnitude(&analytic3);
    let frequency = instantaneous_frequency(&phase, 50.0);
    
    println!("Time | Phase (rad) | Magnitude | Inst. Freq (Hz)");
    for i in (0..phase.len()).step_by(5) {
        println!("{:3} | {:10.3} | {:9.3} | {:14.3}", 
                 i, phase[i], magnitude[i], frequency.get(i).copied().unwrap_or(0.0));
    }
    println!();

    // Example 6: Phase Rotation by fixed angle
    println!("Example 6: Phase Rotation by Fixed Angle");
    let signal6 = vec![1.0, 1.0, 1.0];
    let rotated_90 = PhaseRotator::rotate_by_angle(&signal6, PI / 2.0);
    let rotated_180 = PhaseRotator::rotate_by_angle(&signal6, PI);
    println!("Original:      {:?}", signal6);
    println!("Rotated 90°:   {:?}", rotated_90);
    println!("Rotated 180°:  {:?}", rotated_180);
    println!();

    // Example 7: Block processing
    println!("Example 7: Block Processing with Phase Rotator");
    let mut rotator_block = PhaseRotator::new(5.0, 50.0);
    let input_block = vec![1.0, 0.8, 0.6, 0.4, 0.2];
    let output_block = rotator_block.process_block(&input_block);
    println!("Input block:  {:?}", input_block);
    println!("Output block: {:?}\n", output_block);

    println!("=== Summary ===");
    println!("Phase Rotator: Applies a rotating phase to samples (useful for mixing, modulation)");
    println!("Hilbert Transform: Creates 90° phase-shifted version of signal");
    println!("Analytic Signal: Complex representation with original signal as real part");
    println!("Instantaneous Phase/Frequency: Analysis tools for time-varying signals");
}

/// Create a simple sinusoidal test signal
fn create_test_signal(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| {
            let t = n as f32 / 10.0;
            (2.0 * PI * t).sin()
        })
        .collect()
}

/// Create a chirp signal (frequency increases over time)
fn create_chirp_signal(length: usize) -> Vec<f32> {
    (0..length)
        .map(|n| {
            let t = n as f32 / 50.0;
            // Frequency increases from 2 Hz to 8 Hz
            let freq = 2.0 + (6.0 * t / 1.0).min(6.0);
            (2.0 * PI * freq * t).sin()
        })
        .collect()
}
