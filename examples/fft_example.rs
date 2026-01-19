//! FFT Example
//!
//! This example demonstrates how to use the FFT (Fast Fourier Transform) functionality
//! in the Signalsmith DSP library.

use cute_dsp::fft::{SimpleFFT, SimpleRealFFT};
use num_complex::Complex;

fn main() {
    println!("FFT Example");
    println!("===========");

    // Example 1: Complex FFT
    complex_fft_example();

    // Example 2: Real FFT (more efficient for real-valued signals)
    real_fft_example();
}

fn complex_fft_example() {
    println!("\nComplex FFT Example:");
    
    // Create a new FFT instance for size 1024
    let mut fft = SimpleFFT::<f32>::new(1024);
    
    // Input and output buffers
    let mut time_domain = vec![Complex::new(0.0, 0.0); 1024];
    let mut freq_domain = vec![Complex::new(0.0, 0.0); 1024];
    
    // Fill input with a sine wave at frequency bin 10
    for i in 0..1024 {
        time_domain[i] = Complex::new((i as f32 * 2.0 * std::f32::consts::PI * 10.0 / 1024.0).sin(), 0.0);
    }
    
    // Perform forward FFT
    fft.fft(&time_domain, &mut freq_domain);
    
    // Print the magnitude of the first few frequency bins
    println!("Frequency domain magnitudes (first 15 bins):");
    for i in 0..15 {
        println!("Bin {}: {}", i, (freq_domain[i].norm() / 1024.0));
    }
    
    // Perform inverse FFT
    fft.ifft(&freq_domain, &mut time_domain);
    
    // Check reconstruction error
    let mut max_error = 0.0;
    for i in 0..1024 {
        let original = (i as f32 * 2.0 * std::f32::consts::PI * 10.0 / 1024.0).sin();
        let error = (time_domain[i].re - original).abs();
        if error > max_error {
            max_error = error;
        }
    }
    println!("Maximum reconstruction error: {}", max_error);
}

fn real_fft_example() {
    println!("\nReal FFT Example:");
    
    // Create a new Real FFT instance for size 1024
    let mut real_fft = SimpleRealFFT::<f32>::new(1024);
    
    // Input and output buffers
    let mut time_domain = vec![0.0; 1024];
    let mut freq_domain = vec![Complex::new(0.0, 0.0); 1024/2 + 1]; // Real FFT produces N/2+1 complex values
    
    // Fill input with a sine wave at frequency bin 10
    for i in 0..1024 {
        time_domain[i] = (i as f32 * 2.0 * std::f32::consts::PI * 10.0 / 1024.0).sin();
    }
    
    // Perform forward FFT
    real_fft.fft(&time_domain, &mut freq_domain);
    
    // Print the magnitude of the first few frequency bins
    println!("Frequency domain magnitudes (first 15 bins):");
    for i in 0..15 {
        println!("Bin {}: {}", i, (freq_domain[i].norm() / 1024.0));
    }
    
    // Perform inverse FFT
    real_fft.ifft(&freq_domain, &mut time_domain);
    
    // Check reconstruction error
    let mut max_error = 0.0;
    for i in 0..1024 {
        let original = (i as f32 * 2.0 * std::f32::consts::PI * 10.0 / 1024.0).sin();
        let error = (time_domain[i] - original).abs();
        if error > max_error {
            max_error = error;
        }
    }
    println!("Maximum reconstruction error: {}", max_error);
}