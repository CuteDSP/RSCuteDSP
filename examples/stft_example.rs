//! STFT Example
//!
//! This example demonstrates how to use the Short-Time Fourier Transform (STFT)
//! functionality in the Signalsmith DSP library for spectral processing.

use num_complex::Complex;
use cute_dsp::stft::{STFT, WindowShape};
use std::f32::consts::PI;

fn main() {
    println!("STFT Example");
    println!("============");

    // Example 1: Basic STFT Processing
    basic_stft_example();

    // Example 2: Spectral Processing (simple pitch shift)
    spectral_processing_example();
}

fn basic_stft_example() {
    println!("\nBasic STFT Example:");
    
    // Create a new STFT processor (non-modified version)
    let mut stft = STFT::<f32>::new(false);
    
    // Configure the STFT
    // - 1 input channel, 1 output channel
    // - 1024 samples per block
    // - 0 extra input history
    // - 256 samples hop size (75% overlap)
    stft.configure(1, 1, 1024, 0, 256);
    
    // Reset with default settings
    stft.reset_default();
    
    // Set the window shape and interval
    stft.set_interval(256, WindowShape::Kaiser);
    
    // Generate a test signal (sine wave at 440 Hz, 44.1 kHz sample rate)
    let sample_rate = 44100.0;
    let frequency = 440.0;
    let duration_samples = 4096; // About 93 ms
    
    let mut input_signal = vec![0.0; duration_samples];
    for i in 0..duration_samples {
        input_signal[i] = (2.0 * PI * frequency * (i as f32) / sample_rate as f32).sin();
    }
    
    // Process the signal in blocks
    let mut output_signal = vec![0.0; duration_samples];
    let mut position = 0;
    
    while position + stft.block_samples() <= duration_samples {
        // Write input block
        stft.write_input(0, 0, stft.block_samples(), &input_signal[position..position + stft.block_samples()]);
        
        // Process block (input channel 0 to output channel 0)
        stft.process_block(0, 0);
        
        // Read output block
        stft.read_output(0, 0, stft.block_samples(), &mut output_signal[position..position + stft.block_samples()]);
        
        // Move to next block
        position += stft.default_interval();
        stft.move_input(stft.default_interval());
        stft.move_output(stft.default_interval());
    }
    
    // Calculate the error between input and output
    let mut max_error = 0.0;
    for i in stft.latency()..duration_samples - stft.latency() {
        let error = (input_signal[i] - output_signal[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }
    
    println!("Maximum reconstruction error: {}", max_error);
    println!("Total latency: {} samples", stft.latency());
}

fn spectral_processing_example() {
    println!("\nSpectral Processing Example (Simple Pitch Shift):");
    
    // Create a new STFT processor
    let mut stft = STFT::<f32>::new(true); // Use modified STFT for better phase handling
    
    // Configure the STFT
    stft.configure(1, 1, 2048, 0, 512);
    stft.reset_default();
    stft.set_interval(512, WindowShape::Kaiser);
    
    // Generate a test signal (sine wave at 440 Hz, 44.1 kHz sample rate)
    let sample_rate = 44100.0;
    let frequency = 440.0;
    let duration_samples = 8192; // About 186 ms
    
    let mut input_signal = vec![0.0; duration_samples];
    for i in 0..duration_samples {
        input_signal[i] = (2.0 * PI * frequency * (i as f32) / sample_rate as f32).sin();
    }
    
    // Process the signal in blocks with spectral modification
    let mut output_signal = vec![0.0; duration_samples];
    let mut position = 0;
    
    // Pitch shift factor (1.5 = up a perfect fifth)
    let pitch_shift_factor = 1.5;
    
    while position + stft.block_samples() <= duration_samples {
        // Write input block
        stft.write_input(0, 0, stft.block_samples(), &input_signal[position..position + stft.block_samples()]);
        
        // Process block to spectrum
        let spectrum = stft.process_block_to_spectrum(0).to_vec();
        
        // Apply pitch shift (very simple version - just shifts the bins)
        let mut shifted_spectrum = vec![Complex::new(0.0, 0.0); spectrum.len()];
        
        for i in 0..spectrum.len() {
            let shifted_bin = (i as f32 * pitch_shift_factor) as usize;
            if shifted_bin < spectrum.len() {
                shifted_spectrum[shifted_bin] = spectrum[i];
            }
        }
        
        // Process spectrum back to time domain
        stft.process_spectrum_to_block(0, &shifted_spectrum);
        
        // Read output block
        stft.read_output(0, 0, stft.block_samples(), &mut output_signal[position..position + stft.block_samples()]);
        
        // Move to next block
        position += stft.default_interval();
        stft.move_input(stft.default_interval());
        stft.move_output(stft.default_interval());
    }
    
    // Print information about the pitch shift
    println!("Original frequency: {} Hz", frequency);
    println!("Shifted frequency: {} Hz", frequency * pitch_shift_factor);
    println!("Pitch shift factor: {}", pitch_shift_factor);
    
    // Print a few samples of input and output
    println!("\nSample comparison (first 10 samples after latency):");
    println!("Sample | Input    | Output");
    println!("-------|----------|--------");
    let start = stft.latency();
    for i in start..start + 10 {
        println!("{:6} | {:8.5} | {:8.5}", i, input_signal[i], output_signal[i]);
    }
}