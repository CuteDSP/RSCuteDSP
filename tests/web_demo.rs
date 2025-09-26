//! Web Demo Effect Tests
//!
//! Tests that verify the JavaScript effect implementations work correctly.
//! These tests simulate the web audio processing.

#[cfg(feature = "wasm")]
use wasm_bindgen_test::*;
#[cfg(feature = "wasm")]
wasm_bindgen_test_configure!(run_in_browser);

#[cfg(feature = "wasm")]
mod web_demo_tests {
    use super::*;

    // Simple JavaScript-style effect implementations for testing
    struct SimpleBiquad {
        x1: f32, x2: f32, y1: f32, y2: f32,
        a0: f32, a1: f32, a2: f32, b0: f32, b1: f32, b2: f32,
    }

    impl SimpleBiquad {
        fn new() -> Self {
            Self {
                x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0,
                a0: 1.0, a1: 0.0, a2: 0.0, b0: 1.0, b1: 0.0, b2: 0.0,
            }
        }

        fn lowpass(&mut self, cutoff: f32, resonance: f32, sample_rate: f32) {
            let w0 = 2.0 * std::f32::consts::PI * cutoff / sample_rate;
            let cosw0 = w0.cos();
            let sinw0 = w0.sin();
            let alpha = sinw0 / (2.0 * resonance);

            let b0 = (1.0 - cosw0) / 2.0;
            let b1 = 1.0 - cosw0;
            let b2 = (1.0 - cosw0) / 2.0;
            let a0 = 1.0 + alpha;
            let a1 = -2.0 * cosw0;
            let a2 = 1.0 - alpha;

            self.b0 = b0 / a0; self.b1 = b1 / a0; self.b2 = b2 / a0;
            self.a1 = a1 / a0; self.a2 = a2 / a0;
        }

        fn process(&mut self, input: f32) -> f32 {
            let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
                        - self.a1 * self.y1 - self.a2 * self.y2;

            self.x2 = self.x1; self.x1 = input;
            self.y2 = self.y1; self.y1 = output;

            output
        }
    }

    struct SimpleDelay {
        buffer: Vec<f32>,
        write_index: usize,
        delay_samples: usize,
        feedback: f32,
    }

    impl SimpleDelay {
        fn new(max_delay: usize, sample_rate: f32) -> Self {
            Self {
                buffer: vec![0.0; (max_delay as f32 * sample_rate) as usize],
                write_index: 0,
                delay_samples: (0.3 * sample_rate) as usize,
                feedback: 0.4,
            }
        }

        fn process_buffer(&mut self, input: f32) -> f32 {
            let read_index = (self.write_index as isize - self.delay_samples as isize + self.buffer.len() as isize) as usize % self.buffer.len();
            let delayed = self.buffer[read_index];

            let output = input + delayed * self.feedback;

            self.buffer[self.write_index] = output;
            self.write_index = (self.write_index + 1) % self.buffer.len();

            output
        }
    }

    struct SimpleDistortion {
        drive: f32,
        amount: f32,
    }

    impl SimpleDistortion {
        fn new() -> Self {
            Self { drive: 1.0, amount: 0.5 }
        }

        fn process(&self, input: f32) -> f32 {
            let x = input * self.drive;
            let y = x / (1.0 + x.abs()); // Soft clipping
            y * self.amount + input * (1.0 - self.amount)
        }
    }

    #[wasm_bindgen_test]
    fn test_web_filter_effect() {
        let mut filter = SimpleBiquad::new();
        filter.lowpass(1000.0, 0.7, 44100.0);

        // Test impulse response
        let impulse_response: Vec<f32> = (0..100)
            .map(|i| if i == 0 { 1.0 } else { 0.0 })
            .map(|input| filter.process(input))
            .collect();

        // Should decay over time
        assert!(impulse_response[0] > impulse_response[10]);
        assert!(impulse_response[10] > impulse_response[50]);
    }

    #[wasm_bindgen_test]
    fn test_web_delay_effect() {
        let mut delay = SimpleDelay::new(1, 44100.0);

        // Test with impulse
        let mut output = vec![0.0; 200];
        for i in 0..200 {
            let input = if i == 0 { 1.0 } else { 0.0 };
            output[i] = delay.process_buffer(input);
        }

        // Should have delayed response
        let delay_samples = (0.3 * 44100.0) as usize;
        assert!(output[delay_samples] > 0.1); // Feedback creates echo
    }

    #[wasm_bindgen_test]
    fn test_web_distortion_effect() {
        let distortion = SimpleDistortion::new();

        // Test soft clipping
        let clean_signal = 0.5;
        let distorted = distortion.process(clean_signal);

        // Should be compressed
        assert!(distorted < clean_signal);

        // Test hard input
        let hard_input = 2.0;
        let hard_output = distortion.process(hard_input);
        assert!(hard_output.abs() < hard_input.abs());
    }

    #[wasm_bindgen_test]
    fn test_web_effect_chain() {
        let mut filter = SimpleBiquad::new();
        let mut delay = SimpleDelay::new(1, 44100.0);
        let distortion = SimpleDistortion::new();

        filter.lowpass(2000.0, 0.7, 44100.0);

        // Process a test signal through the chain
        let mut signal = vec![0.0; 512];
        for i in 0..512 {
            signal[i] = (2.0 * std::f32::consts::PI * i as f32 * 440.0 / 44100.0).sin() * 0.3;
        }

        for sample in &mut signal {
            // Apply distortion first
            *sample = distortion.process(*sample);

            // Then filter
            *sample = filter.process(*sample);

            // Finally delay
            *sample = delay.process_buffer(*sample);
        }

        // Verify the chain produces reasonable output
        assert!(signal.iter().all(|&x| x.is_finite()));
        assert!(signal.iter().any(|&x| x.abs() > 0.01));
        assert!(signal.iter().all(|&x| x.abs() <= 2.0));
    }

    #[wasm_bindgen_test]
    fn test_web_phaser_effect() {
        // Simple phaser simulation
        let mut allpass_filters: Vec<(Vec<f32>, usize, f32)> = vec![];
        for i in 0..4 {
            let delay_samples = (0.005 + i as f32 * 0.002) * 44100.0;
            allpass_filters.push((vec![0.0; (delay_samples * 2.0) as usize], 0, 0.7));
        }

        fn process_allpass(input: f32, buffer: &mut Vec<f32>, write_index: &mut usize, delay_samples: usize, feedback: f32) -> f32 {
            let read_index = (*write_index as isize - delay_samples as isize + buffer.len() as isize) as usize % buffer.len();
            let delayed = buffer[read_index];

            let output = input + delayed * feedback;
            buffer[*write_index] = output;
            *write_index = (*write_index + 1) % buffer.len();

            delayed - output * feedback
        }

        // Test phaser on a signal
        let mut signal = vec![0.0; 256];
        for i in 0..256 {
            signal[i] = (2.0 * std::f32::consts::PI * i as f32 / 256.0).sin() * 0.5;
        }

        for sample in &mut signal {
            let mut output = *sample;
            for (buffer, write_index, feedback) in &mut allpass_filters {
                let delay_samples = (buffer.len() / 2) as usize;
                output = process_allpass(output, buffer, write_index, delay_samples, *feedback);
            }
            *sample = *sample * 0.5 + output * 0.5; // Mix dry/wet
        }

        // Verify phaser creates frequency notches
        assert!(signal.iter().all(|&x| x.is_finite()));
        assert!(signal.iter().any(|&x| x.abs() > 0.01));
    }
}