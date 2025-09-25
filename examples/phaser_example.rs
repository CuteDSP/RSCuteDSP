//! Phaser Example
//!
//! This example demonstrates how to create a phaser effect using
//! all-pass filters.

use cute_dsp::delay::{Delay, InterpolatorLinear};

struct AllPassFilter {
    delay: Delay<f32, InterpolatorLinear<f32>>,
    feedback: f32,
}

impl AllPassFilter {
    fn new(max_delay_samples: usize, feedback: f32) -> Self {
        Self {
            delay: Delay::new(InterpolatorLinear::new(), max_delay_samples),
            feedback,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.delay.read(5.0); // Fixed delay of 5 samples for simplicity
        let output = input + delayed * self.feedback;
        self.delay.write(output);
        delayed - output * self.feedback
    }
}

struct Phaser {
    allpass_filters: Vec<AllPassFilter>,
    mix: f32,
}

impl Phaser {
    fn new() -> Self {
        let mut allpass_filters = Vec::new();

        // Create 4 all-pass filters with different delays
        for i in 0..4 {
            let max_delay = 10 + i * 2; // Different delay lengths
            allpass_filters.push(AllPassFilter::new(max_delay, 0.7));
        }

        Self {
            allpass_filters,
            mix: 0.5,
        }
    }

    fn set_mix(&mut self, mix: f32) {
        self.mix = mix;
    }

    fn process(&mut self, input: f32) -> f32 {
        // Process through all-pass filters
        let mut output = input;
        for filter in &mut self.allpass_filters {
            output = filter.process(output);
        }

        // Mix dry/wet
        input * (1.0 - self.mix) + output * self.mix
    }
}

fn main() {
    println!("Phaser Example");
    println!("==============");

    const BUFFER_SIZE: usize = 100;

    // Create phaser
    let mut phaser = Phaser::new();

    // Test with impulse
    let mut impulse = vec![0.0; BUFFER_SIZE];
    impulse[0] = 1.0;

    println!("Phaser impulse response (first 20 samples):");
    for i in 0..20 {
        let output = phaser.process(impulse[i]);
        println!("Sample {}: {:.6}", i, output);
    }

    // Test with sine wave
    println!("\nPhaser on sine wave (first 20 samples):");
    phaser = Phaser::new(); // Reset
    for i in 0..20 {
        let input = (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin() * 0.5;
        let output = phaser.process(input);
        println!("Sample {}: {:.6}", i, output);
    }
}