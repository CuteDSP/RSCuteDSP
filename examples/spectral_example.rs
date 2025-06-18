//! Spectral Processing Example
//!
//! This example demonstrates how to use the spectral processing functionality
//! in the Signalsmith DSP library.

use signalsmith_dsp::spectral::{WindowedFFT, SpectralProcessor};
use signalsmith_dsp::windows::Kaiser;
use num_complex::Complex;
use signalsmith_dsp::spectral::utils::{apply_gain, apply_phase_shift, apply_time_shift, complex_to_mag_phase, db_to_linear, linear_to_db, mag_phase_to_complex};

fn main() {
    println!("Spectral Processing Example");
    println!("==========================");

    // Example 1: Windowed FFT
    windowed_fft_example();

    // Example 2: Spectral processor
    spectral_processor_example();

    // Example 3: Spectral utilities
    spectral_utilities_example();
}

fn windowed_fft_example() {
    println!("\nWindowed FFT Example:");
    
    // Create a windowed FFT with size 1024 and no rotation
    let mut windowed_fft = WindowedFFT::<f32>::new(1024, 0);
    
    // Create input and output buffers
    let mut time_domain = vec![0.0; 1024];
    let mut freq_domain = vec![Complex::new(0.0, 0.0); 1024];
    
    // Fill input with a sine wave
    for i in 0..1024 {
        time_domain[i] = (i as f32 * 0.1).sin();
    }
    
    // Print some of the input signal
    println!("Input signal (first 10 samples):");
    for i in 0..10 {
        println!("Sample {}: {:.6}", i, time_domain[i]);
    }
    
    // Perform forward FFT with windowing and scaling
    windowed_fft.fft(&time_domain, &mut freq_domain, true, true);
    
    // Print some of the frequency domain data
    println!("\nFrequency domain (first 10 bins):");
    for i in 0..10 {
        let (mag, phase) = complex_to_mag_phase(freq_domain[i]);
        println!("Bin {}: magnitude={:.6}, phase={:.6}", i, mag, phase);
    }
    
    // Perform inverse FFT with windowing
    windowed_fft.ifft(&freq_domain, &mut time_domain, true);
    
    // Print some of the reconstructed signal
    println!("\nReconstructed signal (first 10 samples):");
    for i in 0..10 {
        println!("Sample {}: {:.6}", i, time_domain[i]);
    }
    
    // Create a windowed FFT with a custom window
    println!("\nCreating a windowed FFT with a custom Kaiser window:");
    let kaiser = Kaiser::new(6.0);
    let mut window_buffer = vec![0.0; 1024];
    kaiser.fill(&mut window_buffer);
    
    let mut windowed_fft_custom = WindowedFFT::<f32>::with_window(1024, |i| {
        let r = (i + 0.5) / 1024.0;
        let arg = (1.0 - (2.0 * r - 1.0).powi(2) as f32).sqrt();
        let beta = 6.0_f32;
        let bessel0_beta = 403.4287934927351_f32; // Precomputed I0(6.0)
        let bessel0_beta_arg = (1.0 + (beta * arg).powi(2) / 4.0_f32).powi(4);
        bessel0_beta_arg / bessel0_beta
    }, 0.0, 0);
    
    println!("Custom windowed FFT created with size: {}", windowed_fft_custom.size());
}

fn spectral_processor_example() {
    println!("\nSpectral Processor Example:");
    
    // Create a spectral processor with FFT size 1024 and 4x overlap
    let mut processor = SpectralProcessor::<f32>::new(1024, 4);
    
    // Create input and output buffers
    let input_size = 2048;
    let mut input = vec![0.0; input_size];
    let mut output = vec![0.0; input_size];
    
    // Fill input with a sine wave
    for i in 0..input_size {
        input[i] = (i as f32 * 0.1).sin();
    }
    
    // Define a spectral processing function (simple low-pass filter)
    let lowpass_filter = |spectrum: &mut [Complex<f32>]| {
        // Apply a simple low-pass filter by attenuating high frequencies
        let cutoff = spectrum.len() / 4;
        for i in cutoff..spectrum.len() {
            // Gradually attenuate higher frequencies
            let factor = 1.0 - ((i - cutoff) as f32 / (spectrum.len() - cutoff) as f32);
            spectrum[i] *= factor;
        }
    };
    
    // Process the signal
    processor.process(&input, &mut output, lowpass_filter);
    
    // Print some of the input and output signals
    println!("Input and output comparison (first 10 samples):");
    for i in 0..10 {
        println!("Sample {}: input={:.6}, output={:.6}", i, input[i], output[i]);
    }
    
    // Reset the processor
    processor.reset();
    println!("\nProcessor reset");
    
    // Print some processor properties
    println!("FFT size: {}", processor.fft_size());
    println!("Hop size: {}", processor.hop_size());
    println!("Overlap: {}", processor.overlap());
    
    // Define a pitch shifting function
    let pitch_shift = |spectrum: &mut [Complex<f32>]| {
        // Simple pitch shifting by frequency domain interpolation
        // This is a very basic implementation and not high quality
        let shift_factor = 1.2; // Shift up by 20%
        let mut shifted = vec![Complex::new(0.0, 0.0); spectrum.len()];
        
        for i in 0..spectrum.len() {
            let shifted_index = (i as f32 * shift_factor) as usize;
            if shifted_index < spectrum.len() {
                shifted[shifted_index] = spectrum[i];
            }
        }
        
        // Copy back to the original spectrum
        for i in 0..spectrum.len() {
            spectrum[i] = shifted[i];
        }
    };
    
    // Process with the pitch shifting function
    processor.process_with_options(&input, &mut output, pitch_shift, true, true);
    
    println!("\nAfter pitch shifting (first 10 samples):");
    for i in 0..10 {
        println!("Sample {}: input={:.6}, output={:.6}", i, input[i], output[i]);
    }
}

fn spectral_utilities_example() {
    println!("\nSpectral Utilities Example:");
    
    // Create a spectrum
    let mut spectrum = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.8, 0.2),
        Complex::new(0.6, 0.4),
        Complex::new(0.4, 0.6),
        Complex::new(0.2, 0.8),
        Complex::new(0.0, 1.0),
    ];
    
    // Print the original spectrum
    println!("Original spectrum:");
    for (i, &c) in spectrum.iter().enumerate() {
        let (mag, phase) = complex_to_mag_phase(c);
        println!("Bin {}: complex=({:.3}, {:.3}), magnitude={:.3}, phase={:.3}", 
                 i, c.re, c.im, mag, phase);
    }
    
    // Convert between magnitude/phase and complex
    println!("\nMagnitude/phase to complex conversion:");
    let mag = 2.0;
    let phase = 0.5;
    let complex = mag_phase_to_complex(mag, phase);
    println!("Magnitude={:.3}, Phase={:.3} -> Complex=({:.3}, {:.3})",
             mag, phase, complex.re, complex.im);
    
    let (mag_back, phase_back) = complex_to_mag_phase(complex);
    println!("Complex=({:.3}, {:.3}) -> Magnitude={:.3}, Phase={:.3}",
             complex.re, complex.im, mag_back, phase_back);
    
    // Convert between linear and dB
    println!("\nLinear to dB conversion:");
    let linear_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
    for &linear in &linear_values {
        let db = linear_to_db(linear);
        println!("Linear={:.3} -> dB={:.3}", linear, db);
        
        let linear_back = db_to_linear(db);
        println!("dB={:.3} -> Linear={:.3}", db, linear_back);
    }
    
    // Apply gain to the spectrum
    println!("\nApplying gain to the spectrum:");
    let gain_db = 6.0; // 6 dB gain
    apply_gain(&mut spectrum, gain_db);
    
    println!("After applying {:.1} dB gain:", gain_db);
    for (i, &c) in spectrum.iter().enumerate() {
        let (mag, phase) = complex_to_mag_phase(c);
        println!("Bin {}: magnitude={:.3}, phase={:.3}", i, mag, phase);
    }
    
    // Apply phase shift to the spectrum
    println!("\nApplying phase shift to the spectrum:");
    let phase_shift = 0.5; // 0.5 radians
    apply_phase_shift(&mut spectrum, phase_shift);
    
    println!("After applying {:.1} radians phase shift:", phase_shift);
    for (i, &c) in spectrum.iter().enumerate() {
        let (mag, phase) = complex_to_mag_phase(c);
        println!("Bin {}: magnitude={:.3}, phase={:.3}", i, mag, phase);
    }
    
    // Apply time shift to the spectrum
    println!("\nApplying time shift to the spectrum:");
    let time_shift = 10.0; // 10 samples
    let sample_rate = 44100.0; // 44.1 kHz
    apply_time_shift(&mut spectrum, time_shift, sample_rate);
    
    println!("After applying {:.1} samples time shift:", time_shift);
    for (i, &c) in spectrum.iter().enumerate() {
        let (mag, phase) = complex_to_mag_phase(c);
        println!("Bin {}: magnitude={:.3}, phase={:.3}", i, mag, phase);
    }
}