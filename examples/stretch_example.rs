//! Time Stretching Example
//!
//! This example demonstrates how to use the time stretching functionality
//! in the Signalsmith DSP library to change the duration of audio without
//! changing its pitch.

use signalsmith_dsp::stretch::{Stretcher, RealtimeStretcher, StretchConfig};
use std::f32::consts::PI;

fn main() {
    println!("Time Stretching Example");
    println!("======================");

    // Example 1: Basic Time Stretching
    basic_stretch_example();

    // Example 2: Realtime Time Stretching
    realtime_stretch_example();

    // Example 3: Pitch Shifting (time stretching + resampling)
    pitch_shift_example();
}

fn basic_stretch_example() {
    println!("\nBasic Time Stretching Example:");
    
    // Create a new stretcher with smaller FFT size for this example
    let fft_size = 1024; // Reduced from 2048
    let overlap = 4;
    let mut stretcher = Stretcher::<f32>::new(fft_size, overlap);
    
    // Configure the stretcher
    let mut config = StretchConfig::default();
    config.stretch = 1.5;
    config.phase_locking = 0.5; // Add phase locking for better quality
    stretcher.set_config(config);
    
    // Generate a test signal (sine wave at 440 Hz, 44.1 kHz sample rate)
    let sample_rate = 44100.0;
    let frequency = 440.0;
    let duration_seconds = 1.0;
    
    // Add extra samples to account for latency
    let input_samples = (duration_seconds * sample_rate) as usize + fft_size;
    
    let mut input_signal = vec![0.0; input_samples];
    for i in 0..input_samples {
        input_signal[i] = (2.0 * PI * frequency * (i as f32) / sample_rate).sin() * 0.9;
    }
    
    // Calculate output size based on stretch ratio plus latency compensation
    let output_samples = ((input_samples - fft_size) as f32 * stretcher.config().stretch) as usize;
    let mut output_signal = vec![0.0; output_samples];
    
    // Process the signal
    stretcher.process(&input_signal, &mut output_signal);
    
    // Skip the latency samples when displaying output
    let skip_samples = stretcher.latency();
    
    println!("Input duration: {} seconds ({} samples)", duration_seconds, input_samples - fft_size);
    println!("Output duration: {} seconds ({} samples)", 
             duration_seconds * stretcher.config().stretch, output_samples);
    println!("Stretch ratio: {}", stretcher.config().stretch);
    println!("Latency: {} samples", stretcher.latency());
    
    // Print samples after the latency period
    println!("\nSample comparison (first 5 samples after latency):");
    println!("Sample | Input    | Output");
    println!("-------|----------|--------");
    for i in 0..5 {
        println!("{:6} | {:8.5} | {:8.5}", 
                i, 
                input_signal[i + skip_samples], 
                output_signal[i + skip_samples]);
    }
}

fn realtime_stretch_example() {
    println!("\nRealtime Time Stretching Example:");
    
    // Create a new realtime stretcher with FFT size 1024, overlap 4, and max block size 512
    let mut stretcher = RealtimeStretcher::<f32>::new(1024, 4, 512);
    
    // Configure the stretcher
    let mut config = StretchConfig::default();
    config.stretch = 0.75; // Compress by 0.75x (make it shorter)
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
    
    // Calculate output size based on stretch ratio
    let output_samples = (input_samples as f32 * stretcher.config().stretch) as usize;
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
    
    // Print information about the stretching
    println!("Input duration: {} seconds ({} samples)", duration_seconds, input_samples);
    println!("Output duration: {} seconds ({} samples)", 
             duration_seconds * stretcher.config().stretch as f32, output_samples);
    println!("Stretch ratio: {}", stretcher.config().stretch);
    println!("Latency: {} samples", stretcher.latency());
}

fn pitch_shift_example() {
    println!("\nPitch Shifting Example (using built-in pitch shifting):");

    // Create a new stretcher with FFT size 2048 and overlap 4
    let mut stretcher = Stretcher::<f32>::new(2048, 4);

    // Configure the stretcher for pitch shifting
    // We can use the built-in pitch_shift parameter (in semitones)
    let semitones = 7.0; // Shift up by a perfect fifth (7 semitones)
    let pitch_shift_factor = 2.0f32.powf(semitones / 12.0); // Convert semitones to factor

    let mut config = StretchConfig::default();
    config.pitch_shift = semitones;
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

    // Calculate output size based on stretch ratio (1.0, so same as input)
    let output_samples = input_samples;
    let mut output_signal = vec![0.0; output_samples];

    // Process the signal directly with pitch shifting
    stretcher.process(&input_signal, &mut output_signal);

    // Print information about the pitch shifting
    println!("Original frequency: {} Hz", frequency);
    println!("Shifted frequency: {} Hz (approximately)", frequency * pitch_shift_factor);
    println!("Pitch shift amount: {} semitones", semitones);
    println!("Pitch shift factor: {}", pitch_shift_factor);
    println!("Time stretch ratio: {}", stretcher.config().stretch);
    
    // Print a few samples of input and output
    println!("\nSample comparison (first 5 samples):");
    println!("Sample | Input    | Output");
    println!("-------|----------|--------");
    for i in 0..5 {
        println!("{:6} | {:8.5} | {:8.5}", i, input_signal[i], output_signal[i]);
    }
}