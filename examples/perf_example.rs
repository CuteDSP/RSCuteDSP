//! Performance Utilities Example
//!
//! This example demonstrates how to use the performance utilities
//! in the Signalsmith DSP library.

use signalsmith_dsp::perf::{StopDenormals, mul, mul_conj};
use num_complex::Complex;
use std::time::{Duration, Instant};

fn main() {
    println!("Performance Utilities Example");
    println!("============================");

    // Example 1: Complex multiplication
    complex_multiplication_example();

    // Example 2: Stop denormals
    stop_denormals_example();

    // Example 3: Performance comparison
    performance_comparison_example();
}

fn complex_multiplication_example() {
    println!("\nComplex Multiplication Example:");
    
    // Create complex numbers
    let a = Complex::new(2.0, 3.0);
    let b = Complex::new(4.0, 5.0);
    
    // Standard complex multiplication
    let standard_result = a * b;
    println!("Standard multiplication: ({}, {}) * ({}, {}) = ({}, {})",
             a.re, a.im, b.re, b.im, standard_result.re, standard_result.im);
    
    // Using the optimized mul function
    let optimized_result = mul(a, b);
    println!("Optimized multiplication: ({}, {}) * ({}, {}) = ({}, {})",
             a.re, a.im, b.re, b.im, optimized_result.re, optimized_result.im);
    
    // Using the optimized mul_conj function (multiplies by the conjugate of b)
    let conj_result = mul_conj(a, b);
    println!("Multiplication with conjugate: ({}, {}) * ({}, {})* = ({}, {})",
             a.re, a.im, b.re, b.im, conj_result.re, conj_result.im);
    
    // Verify that mul_conj is equivalent to multiplying by the conjugate
    let b_conj = Complex::new(b.re, -b.im);
    let standard_conj_result = a * b_conj;
    println!("Standard multiplication with conjugate: ({}, {}) * ({}, {}) = ({}, {})",
             a.re, a.im, b_conj.re, b_conj.im, standard_conj_result.re, standard_conj_result.im);
}

fn stop_denormals_example() {
    println!("\nStop Denormals Example:");
    
    // Create a very small number that might become denormal
    let mut small_value = 1.0e-30;
    println!("Initial small value: {}", small_value);
    
    // Perform operations that might lead to denormals
    for i in 0..10 {
        small_value *= 0.1;
        println!("Iteration {}: value = {}", i, small_value);
    }
    
    println!("\nNow with StopDenormals active:");
    
    // Create a StopDenormals instance to prevent denormals
    let _stop_denormals = StopDenormals::new();
    
    // Reset and perform the same operations
    small_value = 1.0e-30;
    println!("Initial small value: {}", small_value);
    
    // Perform operations that might lead to denormals
    for i in 0..10 {
        small_value *= 0.1;
        println!("Iteration {}: value = {}", i, small_value);
    }
    
    // The StopDenormals instance will be automatically dropped at the end of the scope
    println!("StopDenormals is automatically disabled when it goes out of scope");
}

fn performance_comparison_example() {
    println!("\nPerformance Comparison Example:");
    
    // Parameters for the test
    const NUM_ITERATIONS: usize = 1_000_000;
    const ARRAY_SIZE: usize = 1000;
    
    // Create arrays of complex numbers
    let mut a = vec![Complex::new(0.0, 0.0); ARRAY_SIZE];
    let mut b = vec![Complex::new(0.0, 0.0); ARRAY_SIZE];
    let mut c_standard = vec![Complex::new(0.0, 0.0); ARRAY_SIZE];
    let mut c_optimized = vec![Complex::new(0.0, 0.0); ARRAY_SIZE];
    
    // Initialize with some values
    for i in 0..ARRAY_SIZE {
        a[i] = Complex::new(i as f32 * 0.01, (i as f32 + 0.5) * 0.01);
        b[i] = Complex::new((i as f32 + 0.3) * 0.01, (i as f32 + 0.7) * 0.01);
    }
    
    // Test standard complex multiplication
    println!("Testing standard complex multiplication...");
    let start = Instant::now();
    for _ in 0..NUM_ITERATIONS / ARRAY_SIZE {
        for i in 0..ARRAY_SIZE {
            c_standard[i] = a[i] * b[i];
        }
    }
    let standard_duration = start.elapsed();
    
    // Test optimized complex multiplication
    println!("Testing optimized complex multiplication...");
    let start = Instant::now();
    for _ in 0..NUM_ITERATIONS / ARRAY_SIZE {
        for i in 0..ARRAY_SIZE {
            c_optimized[i] = mul(a[i], b[i]);
        }
    }
    let optimized_duration = start.elapsed();
    
    // Print results
    println!("Standard multiplication time: {:?}", standard_duration);
    println!("Optimized multiplication time: {:?}", optimized_duration);
    
    if optimized_duration < standard_duration {
        let speedup = standard_duration.as_nanos() as f64 / optimized_duration.as_nanos() as f64;
        println!("Optimized version is {:.2}x faster", speedup);
    } else {
        let slowdown = optimized_duration.as_nanos() as f64 / standard_duration.as_nanos() as f64;
        println!("Optimized version is {:.2}x slower", slowdown);
    }
    
    // Now test with StopDenormals
    println!("\nTesting with StopDenormals...");
    
    // Without StopDenormals
    println!("Without StopDenormals:");
    let start = Instant::now();
    for _ in 0..NUM_ITERATIONS / ARRAY_SIZE {
        for i in 0..ARRAY_SIZE {
            c_standard[i] = a[i] * b[i] * Complex::new(1e-30, 1e-30);
        }
    }
    let without_stop_denormals = start.elapsed();
    println!("Time: {:?}", without_stop_denormals);
    
    // With StopDenormals
    println!("With StopDenormals:");
    let _stop_denormals = StopDenormals::new();
    let start = Instant::now();
    for _ in 0..NUM_ITERATIONS / ARRAY_SIZE {
        for i in 0..ARRAY_SIZE {
            c_standard[i] = a[i] * b[i] * Complex::new(1e-30, 1e-30);
        }
    }
    let with_stop_denormals = start.elapsed();
    println!("Time: {:?}", with_stop_denormals);
    
    if with_stop_denormals < without_stop_denormals {
        let speedup = without_stop_denormals.as_nanos() as f64 / with_stop_denormals.as_nanos() as f64;
        println!("With StopDenormals is {:.2}x faster", speedup);
    } else {
        let slowdown = with_stop_denormals.as_nanos() as f64 / without_stop_denormals.as_nanos() as f64;
        println!("With StopDenormals is {:.2}x slower", slowdown);
    }
}