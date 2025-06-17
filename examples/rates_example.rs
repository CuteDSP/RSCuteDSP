//! Sample Rate Conversion Example
//!
//! This example demonstrates how to use the sample rate conversion functionality
//! in the Signalsmith DSP library.

use signalsmith_dsp::rates::{fill_kaiser_sinc, fill_kaiser_sinc_with_centre, Oversampler2xFIR};

fn main() {
    println!("Sample Rate Conversion Example");
    println!("=============================");

    // Example 1: Kaiser-windowed sinc filter
    kaiser_sinc_example();

    // Example 2: 2x Oversampling
    oversampling_example();
}

fn kaiser_sinc_example() {
    println!("\nKaiser-windowed Sinc Filter Example:");
    
    // Create a filter buffer
    let mut filter = vec![0.0f32; 101];
    
    // Fill with a lowpass filter (pass frequencies below 0.2, stop frequencies above 0.3)
    fill_kaiser_sinc(&mut filter, 0.2, 0.3);
    
    // Print the filter coefficients
    println!("Lowpass filter coefficients (first 10 and middle 5):");
    for i in 0..10 {
        println!("Coefficient {}: {:.6}", i, filter[i]);
    }
    println!("...");
    for i in 48..53 {
        println!("Coefficient {}: {:.6}", i, filter[i]);
    }
    
    // Create another filter buffer
    let mut bandpass_filter = vec![0.0f32; 101];
    
    // Fill with a bandpass filter centered at 0.25 (normalized frequency)
    fill_kaiser_sinc_with_centre(&mut bandpass_filter, 0.25);
    
    // Print the bandpass filter coefficients
    println!("\nBandpass filter coefficients (first 10 and middle 5):");
    for i in 0..10 {
        println!("Coefficient {}: {:.6}", i, bandpass_filter[i]);
    }
    println!("...");
    for i in 48..53 {
        println!("Coefficient {}: {:.6}", i, bandpass_filter[i]);
    }
    
    // Analyze the frequency response (simplified)
    println!("\nSimplified frequency response analysis:");
    analyze_frequency_response(&filter, "Lowpass");
    analyze_frequency_response(&bandpass_filter, "Bandpass");
}

fn analyze_frequency_response(filter: &[f32], name: &str) {
    // A very simplified frequency response analysis
    // In a real application, you would use an FFT for this
    let frequencies = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
    
    println!("{} filter response:", name);
    for &freq in &frequencies {
        let mut response = 0.0;
        for (i, &coef) in filter.iter().enumerate() {
            let phase = 2.0 * std::f32::consts::PI * freq * i as f32;
            response += coef * phase.cos();
        }
        println!("  At frequency {:.1}: {:.3}", freq, response);
    }
}

fn oversampling_example() {
    println!("\n2x Oversampling Example:");
    
    // Create a 2x oversampler with 1 channel, max block size 1024, half latency 32, and pass frequency 0.45
    let mut oversampler = Oversampler2xFIR::<f32>::new(1, 1024, 32, 0.45);
    
    // Create a test signal (a simple sine wave)
    let mut input = vec![0.0; 100];
    for i in 0..input.len() {
        input[i] = (i as f32 * 0.1).sin();
    }
    
    // Print some of the original signal
    println!("Original signal (first 10 samples):");
    for i in 0..10 {
        println!("Sample {}: {:.6}", i, input[i]);
    }
    
    let latency = oversampler.latency();
    println!("\nOversampler latency: {} samples", latency);
    
    oversampler.reset();
    
    // Process the signal (upsample) - using up_channel instead of up
    oversampler.up_channel(0, &input, input.len());
    
    let oversampled = oversampler.get_channel_ref(0);
    
    println!("\nOversampled signal (first 20 samples):");
    for i in 0..20 {
        println!("Sample {}: {:.6}", i, oversampled[i]);
    }
    
    let mut output = vec![0.0; input.len()];
    
    oversampler.down_channel(0, &mut output, output.len());
    
    // Rest of the code remains the same...
    println!("\nDownsampled signal (first 10 samples):");
    for i in 0..10 {
        println!("Sample {}: {:.6}", i, output[i]);
    }
    
    println!("\nError analysis (accounting for latency):");
    let mut max_error: f32 = 0.0;
    let mut avg_error = 0.0;
    let valid_samples = input.len() - latency;
    
    for i in 0..valid_samples {
        let error = (input[i] - output[i + latency]).abs();
        max_error = max_error.max(error);
        avg_error += error;
    }
    avg_error /= valid_samples as f32;
    
    println!("Maximum error: {:.6}", max_error);
    println!("Average error: {:.6}", avg_error);
}