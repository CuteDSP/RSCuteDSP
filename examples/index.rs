//! Signalsmith DSP Examples Index
//!
//! This file serves as an index for all the examples in the Signalsmith DSP library.
//! Run this file to see a list of available examples and instructions on how to run them.

fn main() {
    println!("Signalsmith DSP Examples");
    println!("=======================");
    println!();
    println!("This library provides various DSP (Digital Signal Processing) algorithms");
    println!("for audio and signal processing. The following examples demonstrate");
    println!("the key features of the library.");
    println!();
    println!("Available examples:");
    println!();
    println!("1. FFT Example (fft_example.rs)");
    println!("   Demonstrates Fast Fourier Transform functionality");
    println!("   Run with: cargo run --example fft_example");
    println!();
    println!("2. Filter Example (filter_example.rs)");
    println!("   Demonstrates biquad filter functionality");
    println!("   Run with: cargo run --example filter_example");
    println!();
    println!("3. Delay Example (delay_example.rs)");
    println!("   Demonstrates delay line functionality with different interpolation methods");
    println!("   Run with: cargo run --example delay_example");
    println!();
    println!("4. STFT Example (stft_example.rs)");
    println!("   Demonstrates Short-Time Fourier Transform for spectral processing");
    println!("   Run with: cargo run --example stft_example");
    println!();
    println!("5. Time Stretching Example (stretch_example.rs)");
    println!("   Demonstrates time stretching and pitch shifting functionality");
    println!("   Run with: cargo run --example stretch_example");
    println!();
    println!("To run all examples sequentially, use:");
    println!("cargo run --example index --features=\"run-all-examples\"");
    println!();
    println!("Note: Some examples may produce a lot of output. You can redirect the output");
    println!("to a file using: cargo run --example <example_name> > output.txt");
    
    // Optionally run all examples if the "run-all-examples" feature is enabled
    #[cfg(feature = "run-all-examples")]
    {
        println!("\nRunning all examples...\n");
        
        println!("\n\n========== FFT Example ==========\n");
        fft_example::main();
        
        println!("\n\n========== Filter Example ==========\n");
        filter_example::main();
        
        println!("\n\n========== Delay Example ==========\n");
        delay_example::main();
        
        println!("\n\n========== STFT Example ==========\n");
        stft_example::main();
        
        println!("\n\n========== Time Stretching Example ==========\n");
        stretch_example::main();
    }
}

// Include other examples when the "run-all-examples" feature is enabled
#[cfg(feature = "run-all-examples")]
mod fft_example {
    include!("fft_example.rs");
}

#[cfg(feature = "run-all-examples")]
mod filter_example {
    include!("filter_example.rs");
}

#[cfg(feature = "run-all-examples")]
mod delay_example {
    include!("delay_example.rs");
}

#[cfg(feature = "run-all-examples")]
mod stft_example {
    include!("stft_example.rs");
}

#[cfg(feature = "run-all-examples")]
mod stretch_example {
    include!("stretch_example.rs");
}