//! Window Functions Example
//!
//! This example demonstrates how to use the window functions
//! in the Signalsmith DSP library.

use signalsmith_dsp::windows::{Kaiser, ApproximateConfinedGaussian, force_perfect_reconstruction};
use std::f32::consts::PI;

fn main() {
    println!("Window Functions Example");
    println!("=======================");

    // Example 1: Kaiser window
    kaiser_window_example();

    // Example 2: Approximate Confined Gaussian window
    acg_window_example();

    // Example 3: Perfect reconstruction for STFT
    perfect_reconstruction_example();

    // Example 4: Window function comparison
    window_comparison_example();
}

fn kaiser_window_example() {
    println!("\nKaiser Window Example:");
    
    // Create a Kaiser window with beta=6.0
    let kaiser = Kaiser::new(6.0);
    
    // Create a buffer to hold the window
    let mut window = vec![0.0f32; 64];
    
    // Fill the buffer with the window function
    kaiser.fill(&mut window);
    
    // Print the window values
    println!("Kaiser window values (beta=6.0):");
    print_window_values(&window);
    
    // Create a Kaiser window with a specific bandwidth
    let kaiser_bw = Kaiser::with_bandwidth(2.5, true);
    
    // Create a buffer to hold the window
    let mut window_bw = vec![0.0f32; 64];
    
    // Fill the buffer with the window function
    kaiser_bw.fill(&mut window_bw);
    
    // Print the window values
    println!("\nKaiser window values (bandwidth=2.5):");
    print_window_values(&window_bw);
    
    // Demonstrate bandwidth to beta conversion
    let bandwidth = 3.0;
    let beta = Kaiser::bandwidth_to_beta(bandwidth, false);
    let beta_heuristic = Kaiser::bandwidth_to_beta(bandwidth, true);
    
    println!("\nBandwidth to beta conversion:");
    println!("Bandwidth: {:.2}", bandwidth);
    println!("Beta (direct): {:.4}", beta);
    println!("Beta (with heuristic): {:.4}", beta_heuristic);
}

fn acg_window_example() {
    println!("\nApproximate Confined Gaussian Window Example:");
    
    // Create an ACG window with sigma=0.2
    let acg = ApproximateConfinedGaussian::new(0.2);
    
    // Create a buffer to hold the window
    let mut window = vec![0.0f32; 64];
    
    // Fill the buffer with the window function
    acg.fill(&mut window);
    
    // Print the window values
    println!("ACG window values (sigma=0.2):");
    print_window_values(&window);
    
    // Create an ACG window with a specific bandwidth
    let acg_bw = ApproximateConfinedGaussian::with_bandwidth(3.0);
    
    // Create a buffer to hold the window
    let mut window_bw = vec![0.0f32; 64];
    
    // Fill the buffer with the window function
    acg_bw.fill(&mut window_bw);
    
    // Print the window values
    println!("\nACG window values (bandwidth=3.0):");
    print_window_values(&window_bw);
    
    // Demonstrate bandwidth to sigma conversion
    let bandwidth = 2.5;
    let sigma = ApproximateConfinedGaussian::bandwidth_to_sigma(bandwidth);
    
    println!("\nBandwidth to sigma conversion:");
    println!("Bandwidth: {:.2}", bandwidth);
    println!("Sigma: {:.4}", sigma);
}

fn perfect_reconstruction_example() {
    println!("\nPerfect Reconstruction Example:");
    
    // Create a window
    let kaiser = Kaiser::new(6.0);
    let mut window = vec![0.0f32; 1024];
    kaiser.fill(&mut window);
    
    // Print some of the original window values
    println!("Original window values (first 5):");
    for i in 0..5 {
        println!("Index {}: {:.6}", i, window[i]);
    }
    
    // Apply perfect reconstruction for 4x overlap
    force_perfect_reconstruction(&mut window, 1024, 256);
    
    // Print some of the modified window values
    println!("\nModified window values for perfect reconstruction (first 5):");
    for i in 0..5 {
        println!("Index {}: {:.6}", i, window[i]);
    }
    
    // Verify perfect reconstruction
    println!("\nVerifying perfect reconstruction:");
    verify_perfect_reconstruction(&window, 256);
}

fn window_comparison_example() {
    println!("\nWindow Function Comparison Example:");
    
    // Create windows of the same size
    let size = 128;
    let mut kaiser_window = vec![0.0f32; size];
    let mut acg_window = vec![0.0f32; size];
    
    // Fill with different window functions
    Kaiser::new(6.0).fill(&mut kaiser_window);
    ApproximateConfinedGaussian::new(0.2).fill(&mut acg_window);
    
    // Compare window properties
    println!("Window properties comparison:");
    
    // Calculate energy
    let kaiser_energy = calculate_window_energy(&kaiser_window);
    let acg_energy = calculate_window_energy(&acg_window);
    
    println!("Kaiser window energy: {:.6}", kaiser_energy);
    println!("ACG window energy: {:.6}", acg_energy);
    
    // Calculate equivalent noise bandwidth
    let kaiser_enbw = calculate_enbw(&kaiser_window);
    let acg_enbw = calculate_enbw(&acg_window);
    
    println!("Kaiser window ENBW: {:.6}", kaiser_enbw);
    println!("ACG window ENBW: {:.6}", acg_enbw);
    
    // Calculate scalloping loss
    let kaiser_scalloping = calculate_scalloping_loss(&kaiser_window);
    let acg_scalloping = calculate_scalloping_loss(&acg_window);
    
    println!("Kaiser window scalloping loss: {:.6} dB", kaiser_scalloping);
    println!("ACG window scalloping loss: {:.6} dB", acg_scalloping);
}

// Helper function to print window values
fn print_window_values(window: &[f32]) {
    // Print a subset of values to avoid too much output
    let indices = [0, 1, 2, 3, window.len()/4, window.len()/2, window.len()*3/4, window.len()-4, window.len()-3, window.len()-2, window.len()-1];
    
    for &i in &indices {
        if i < window.len() {
            println!("Index {}: {:.6}", i, window[i]);
        }
    }
}

// Helper function to verify perfect reconstruction
fn verify_perfect_reconstruction(window: &[f32], hop_size: usize) {
    let mut sum_squared = vec![0.0f32; hop_size];
    
    // Calculate the sum of squared window values at each hop
    for i in 0..window.len() {
        sum_squared[i % hop_size] += window[i] * window[i];
    }
    
    // Check if all values are close to 1.0
    let mut is_perfect = true;
    for (i, &val) in sum_squared.iter().enumerate() {
        println!("Hop {}: Sum of squares = {:.6}", i, val);
        if (val - 1.0).abs() > 1e-5 {
            is_perfect = false;
        }
    }
    
    if is_perfect {
        println!("Perfect reconstruction verified!");
    } else {
        println!("Perfect reconstruction not achieved.");
    }
}

// Helper function to calculate window energy
fn calculate_window_energy(window: &[f32]) -> f32 {
    window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32
}

// Helper function to calculate Equivalent Noise Bandwidth
fn calculate_enbw(window: &[f32]) -> f32 {
    let sum_squared = window.iter().map(|&x| x * x).sum::<f32>();
    let sum = window.iter().sum::<f32>();
    
    window.len() as f32 * sum_squared / (sum * sum)
}

// Helper function to calculate scalloping loss
fn calculate_scalloping_loss(window: &[f32]) -> f32 {
    let n = window.len();
    let mut sum = 0.0;
    
    // Calculate response to a tone at frequency bin + 0.5
    for i in 0..n {
        sum += window[i] * (PI * (i as f32 + 0.5) / n as f32).cos();
    }
    
    // Convert to dB
    20.0 * (sum / window.iter().sum::<f32>()).log10()
}