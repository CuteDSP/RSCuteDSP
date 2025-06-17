//! Filter Example
//!
//! This example demonstrates how to use the filter functionality
//! in the Signalsmith DSP library.

use signalsmith_dsp::filters::{Biquad, BiquadDesign, FilterType, StereoBiquad};

fn main() {
    println!("Filter Example");
    println!("==============");

    // Example 1: Basic Biquad Filter
    basic_filter_example();

    // Example 2: Filter Sweep
    filter_sweep_example();

    // Example 3: Stereo Filter
    stereo_filter_example();
}

fn basic_filter_example() {
    println!("\nBasic Filter Example:");
    
    // Create a new biquad filter
    let mut filter = Biquad::<f32>::new(true); // Use f32 instead of f32
    
    const SAMPLE_RATE: f32 = 44100.0;
    const BUFFER_SIZE: usize = 100;
    
    // Configure as a lowpass filter at 1000 Hz with Q=0.7
    filter.lowpass(1000.0 / SAMPLE_RATE, 0.7, BiquadDesign::Cookbook);
    
    // Process a buffer of audio
    let mut audio_buffer = vec![0.0; BUFFER_SIZE];
    
    // Fill buffer with a simple impulse
    audio_buffer[0] = 1.0;
    
    // Apply the filter
    filter.process_buffer(&mut audio_buffer);
    
    // Print the impulse response
    println!("Lowpass filter impulse response (first 10 samples):");
    for i in 0..10 {
        println!("Sample {}: {:.6}", i, audio_buffer[i]);
    }
}

fn filter_sweep_example() {
    println!("\nFilter Sweep Example:");
    
    // Create filters with different types
    let mut lowpass = Biquad::<f32>::new(true);
    let mut highpass = Biquad::<f32>::new(true);
    let mut bandpass = Biquad::<f32>::new(true);
    let mut notch = Biquad::<f32>::new(true);
    let mut peak = Biquad::<f32>::new(true);
    
    // Configure filters (normalized frequency 0.1 = 4410 Hz at 44.1 kHz sample rate)
    lowpass.lowpass(0.1, 0.7, BiquadDesign::Cookbook);
    highpass.highpass(0.1, 0.7, BiquadDesign::Cookbook);
    bandpass.bandpass(0.1, 1.0); // 1 octave bandwidth
    notch.notch(0.1, 1.0); // 1 octave bandwidth
    peak.peak(0.1, 1.0, 6.0); // 6 dB gain
    
    // Generate a white noise signal
    let mut input = vec![0.0; 100];
    for i in 0..100 {
        input[i] = (i as f32 * 0.1).sin(); // Simple sine wave
    }
    
    // Process with each filter
    let mut lowpass_output = input.clone();
    let mut highpass_output = input.clone();
    let mut bandpass_output = input.clone();
    let mut notch_output = input.clone();
    let mut peak_output = input.clone();
    
    lowpass.process_buffer(&mut lowpass_output);
    highpass.process_buffer(&mut highpass_output);
    bandpass.process_buffer(&mut bandpass_output);
    notch.process_buffer(&mut notch_output);
    peak.process_buffer(&mut peak_output);
    
    // Print results for a few samples
    println!("Filter comparison (first 5 samples):");
    for i in 0..5 {
        println!("Sample {}: Input={:.4}, LP={:.4}, HP={:.4}, BP={:.4}, Notch={:.4}, Peak={:.4}",
            i, input[i], lowpass_output[i], highpass_output[i], 
            bandpass_output[i], notch_output[i], peak_output[i]);
    }
}

fn stereo_filter_example() {
    println!("\nStereo Filter Example:");
    
    // Create a stereo biquad filter
    let mut stereo_filter = StereoBiquad::<f32>::new(true);
    
    // Configure as a lowpass filter
    stereo_filter.lowpass(0.2, 0.7, BiquadDesign::Cookbook);
    
    // Create stereo input (left and right channels)
    let mut left = vec![0.0; 10];
    let mut right = vec![0.0; 10];
    
    // Set different impulses for left and right
    left[0] = 1.0;
    right[5] = 0.8;
    
    // Process the stereo buffer
    stereo_filter.process_buffer(&mut left, &mut right);
    
    // Print the results
    println!("Stereo filter output:");
    println!("      Left    Right");
    for i in 0..10 {
        println!("{}: {:.4}  {:.4}", i, left[i], right[i]);
    }
}