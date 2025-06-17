//! Pitch Shifting Example
//!
//! This example demonstrates how to use the pitch shifting functionality
//! in the Signalsmith DSP library to change the pitch of audio without
//! changing its duration.

use signalsmith_dsp::stretch::{Stretcher, RealtimeStretcher, StretchConfig};
use std::f32::consts::PI;

fn main() {
    println!("Pitch Shifting Example");
    println!("=====================");

    // Example 1: Basic Pitch Shifting
    basic_pitch_shift_example();

    // Example 2: Realtime Pitch Shifting
    realtime_pitch_shift_example();

    // Example 3: Pitch Shifting with Formant Preservation
    formant_preserved_pitch_shift_example();
}

fn basic_pitch_shift_example() {
    println!("\nBasic Pitch Shifting Example:");

    // Create a new stretcher with FFT size 2048 and overlap 4
    let mut stretcher = Stretcher::<f32>::new(2048, 4);

    // Configure the stretcher for pitch shifting
    let mut config = StretchConfig::default();
    config.pitch_shift = 12.0; // Shift up by one octave (12 semitones)
    config.stretch = 1.0; // Keep duration the same
    stretcher.set_config(config);

    // Generate a test signal (sine wave at 440 Hz, 44.1 kHz sample rate)
    let sample_rate = 44100.0;
    let frequency = 440.0;
    let duration_seconds = 1.0;
    let input_samples = (duration_seconds * sample_rate) as usize;

    let mut input_signal = vec![0.0; input_samples];
    for i in 0..input_samples {
        input_signal[i] = (2.0 * PI * frequency * (i as f32) / sample_rate as f32).sin();
    }

    // Process the signal
    let mut output_signal = vec![0.0; input_samples]; // Same size as input (no time stretching)
    stretcher.process(&input_signal, &mut output_signal);

    // Calculate the actual pitch shift factor for display
    let pitch_shift_factor = 2.0f32.powf(stretcher.config().pitch_shift / 12.0);

    // Print information about the pitch shifting
    println!("Original frequency: {} Hz", frequency);
    println!("Shifted frequency: {} Hz", frequency * pitch_shift_factor);
    println!("Pitch shift: {} semitones", stretcher.config().pitch_shift);
    println!("Latency: {} samples", stretcher.latency());

    // Print a few samples of input and output
    println!("\nSample comparison (first 5 samples):");
    println!("Sample | Input    | Output");
    println!("-------|----------|--------");
    for i in 0..5 {
        println!("{:6} | {:8.5} | {:8.5}", i, input_signal[i], output_signal[i]);
    }
}

fn realtime_pitch_shift_example() {
    println!("\nRealtime Pitch Shifting Example:");

    // Create a new realtime stretcher with FFT size 1024, overlap 4, and max block size 512
    let mut stretcher = RealtimeStretcher::<f32>::new(1024, 4, 512);

    // Configure the stretcher for pitch shifting
    let mut config = StretchConfig::default();
    config.pitch_shift = -5.0; // Shift down by 5 semitones
    config.stretch = 1.0; // Keep duration the same
    stretcher.set_config(config);

    // Generate a test signal (sine wave at 440 Hz, 44.1 kHz sample rate)
    let sample_rate = 44100.0;
    let frequency = 440.0;
    let duration_seconds = 0.5;
    let input_samples = (duration_seconds * sample_rate) as usize;

    let mut input_signal = vec![0.0; input_samples];
    for i in 0..input_samples {
        input_signal[i] = (2.0 * PI * frequency * (i as f32) / sample_rate as f32).sin();
    }

    // Create output buffer (same size as input since stretch=1.0)
    let output_samples = input_samples;
    let mut output_signal = vec![0.0; output_samples];

    // Process the signal in blocks to simulate realtime processing
    let block_size = 256;
    let mut input_position = 0;
    let mut output_position = 0;

    while input_position < input_samples {
        // Calculate the current block size
        let current_input_block = std::cmp::min(block_size, input_samples - input_position);
        let current_output_block = std::cmp::min(block_size, output_samples - output_position);

        // Process the block
        stretcher.process(
            &input_signal[input_position..input_position + current_input_block],
            &mut output_signal[output_position..output_position + current_output_block]
        );

        // Move to the next block
        input_position += current_input_block;
        output_position += current_output_block;
    }

    // Calculate the actual pitch shift factor for display
    let pitch_shift_factor = 2.0f32.powf(stretcher.config().pitch_shift / 12.0);

    // Print information about the pitch shifting
    println!("Original frequency: {} Hz", frequency);
    println!("Shifted frequency: {} Hz", frequency * pitch_shift_factor);
    println!("Pitch shift: {} semitones", stretcher.config().pitch_shift);
    println!("Latency: {} samples", stretcher.latency());
}

fn formant_preserved_pitch_shift_example() {
    println!("\nFormant-Preserved Pitch Shifting Example:");
    println!("(Note: This is a simplified simulation of formant preservation)");

    // Create two stretchers - one for pitch shifting and one for formant adjustment
    let mut pitch_stretcher = Stretcher::<f32>::new(2048, 4);
    let mut formant_stretcher = Stretcher::<f32>::new(1024, 4);

    // Configure pitch shifting
    let semitones = 7.0; // Shift up by a perfect fifth
    let mut pitch_config = StretchConfig::default();
    pitch_config.pitch_shift = semitones;
    pitch_config.stretch = 1.0;
    pitch_stretcher.set_config(pitch_config);

    // Configure formant preservation (inverse pitch shift, with higher transient preservation)
    let mut formant_config = StretchConfig::default();
    formant_config.pitch_shift = -semitones; // Opposite direction to preserve formants
    formant_config.stretch = 1.0;
    formant_config.transient_preservation = 0.8;
    formant_stretcher.set_config(formant_config);

    // Generate a test signal (simulated vocal sound - combination of harmonics)
    let sample_rate = 44100.0;
    let fundamental = 220.0; // Fundamental frequency (low A)
    let duration_seconds = 1.0;
    let input_samples = (duration_seconds * sample_rate) as usize;

    let mut input_signal = vec![0.0; input_samples];
    // Create a signal with harmonics to simulate formants
    for i in 0..input_samples {
        let t = i as f32 / sample_rate;
        // Fundamental
        input_signal[i] = (2.0 * PI * fundamental * t).sin() * 0.5;
        // Add harmonics with formant-like envelope
        input_signal[i] += (2.0 * PI * fundamental * 2.0 * t).sin() * 0.3;
        input_signal[i] += (2.0 * PI * fundamental * 3.0 * t).sin() * 0.6; // Emphasized formant
        input_signal[i] += (2.0 * PI * fundamental * 4.0 * t).sin() * 0.8; // Emphasized formant
        input_signal[i] += (2.0 * PI * fundamental * 5.0 * t).sin() * 0.4;
        input_signal[i] += (2.0 * PI * fundamental * 6.0 * t).sin() * 0.2;

        // Normalize
        input_signal[i] /= 3.0;
    }

    // Intermediate buffer for the pitch shifted signal
    let mut pitch_shifted = vec![0.0; input_samples];

    // First apply pitch shift
    pitch_stretcher.process(&input_signal, &mut pitch_shifted);

    // Then apply formant correction
    let mut output_signal = vec![0.0; input_samples];
    formant_stretcher.process(&pitch_shifted, &mut output_signal);

    // Calculate the actual pitch shift factor for display
    let pitch_shift_factor = 2.0f32.powf(semitones / 12.0);

    // Print information about the pitch shifting
    println!("Original fundamental frequency: {} Hz", fundamental);
    println!("Shifted fundamental frequency: {} Hz", fundamental * pitch_shift_factor);
    println!("Pitch shift: {} semitones", semitones);
    println!("Formant preservation: Enabled");

    // Print a few samples of input and output
    println!("\nSample comparison (first 5 samples):");
    println!("Sample | Input    | Output");
    println!("-------|----------|--------");
    for i in 0..5 {
        println!("{:6} | {:8.5} | {:8.5}", i, input_signal[i], output_signal[i]);
    }
}

fn formant_preserved_pitch_shift() {
    let mut stretcher = Stretcher::<f32>::new(2048, 4);
    
    // Configure for 7 semitones up with formant preservation
    stretcher.set_config(StretchConfig {
        pitch_shift: 7.0,
        stretch: 1.0,
        frequency_smoothing: 0.7,  // More frequency smoothing helps preserve formants
        phase_locking: 0.6,
        transient_preservation: 0.8,  // Help preserve the character of the sound
        ..Default::default()
    });
    
    // Create a 220Hz sine wave input (fundamental frequency)
    let mut input = vec![0.0f32; 4096];
    for i in 0..input.len() {
        input[i] = (2.0 * PI * 220.0 * i as f32 / 44100.0).sin();
        // Add some harmonics to simulate formants
        input[i] += 0.5 * (2.0 * PI * 440.0 * i as f32 / 44100.0).sin();
        input[i] += 0.25 * (2.0 * PI * 880.0 * i as f32 / 44100.0).sin();
    }
    
    let mut output = vec![0.0f32; 4096];
    stretcher.process(&input, &mut output);
    
    // Print first 5 samples
    println!("Sample | Input    | Output");
    println!("-------|----------|--------");
    for i in 0..5 {
        println!("{:>6} | {:>8.5} | {:>8.5}", i, input[i], output[i]);
    }
}