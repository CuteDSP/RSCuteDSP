//! Delay Example
//!
//! This example demonstrates how to use the delay line functionality
//! in the Signalsmith DSP library.

use signalsmith_dsp::delay::{
    Delay, MultiDelay, 
    InterpolatorNearest, InterpolatorLinear, InterpolatorCubic
};

fn main() {
    println!("Delay Example");
    println!("=============");

    // Example 1: Basic Delay Line
    basic_delay_example();

    // Example 2: Comparing Interpolation Methods
    interpolation_comparison();

    // Example 3: Multi-channel Delay
    multi_delay_example();
}

fn basic_delay_example() {
    println!("\nBasic Delay Example:");
    
    // Create a delay line with cubic interpolation and capacity for 100 samples
    let mut delay = Delay::new(InterpolatorCubic::<f32>::new(), 100);
    
    // Reset the delay line to zeros
    delay.reset(0.0);
    
    // Create an impulse signal (1.0 followed by zeros)
    let mut output_samples = Vec::new();
    
    // Process 20 samples
    for i in 0..20 {
        // Input is an impulse at sample 0
        let input = if i == 0 { 1.0 } else { 0.0 };
        
        // Read from the delay line with a delay of 5.5 samples
        let output = delay.read(5.5);
        output_samples.push(output);
        
        // Write to the delay line
        delay.write(input);
    }
    
    // Print the output
    println!("Delay line output with 5.5 sample delay:");
    for (i, sample) in output_samples.iter().enumerate() {
        println!("Sample {}: {}", i, sample);
    }
}

fn interpolation_comparison() {
    println!("\nInterpolation Comparison:");
    
    // Create delay lines with different interpolation methods
    let mut delay_nearest = Delay::new(InterpolatorNearest::<f32>::new(), 100);
    let mut delay_linear = Delay::new(InterpolatorLinear::<f32>::new(), 100);
    let mut delay_cubic = Delay::new(InterpolatorCubic::<f32>::new(), 100);
    
    // Reset all delay lines
    delay_nearest.reset(0.0);
    delay_linear.reset(0.0);
    delay_cubic.reset(0.0);
    
    // Create storage for output samples
    let mut output_nearest = Vec::new();
    let mut output_linear = Vec::new();
    let mut output_cubic = Vec::new();
    
    // Process 20 samples with a fractional delay of 5.7 samples
    for i in 0..20 {
        // Input is an impulse at sample 0
        let input = if i == 0 { 1.0 } else { 0.0 };
        
        // Read from each delay line
        output_nearest.push(delay_nearest.read(5.7));
        output_linear.push(delay_linear.read(5.7));
        output_cubic.push(delay_cubic.read(5.7));
        
        // Write to each delay line
        delay_nearest.write(input);
        delay_linear.write(input);
        delay_cubic.write(input);
    }
    
    // Print the comparison
    println!("Comparison of interpolation methods with 5.7 sample delay:");
    println!("Sample | Nearest  | Linear   | Cubic");
    println!("-------|----------|----------|--------");
    for i in 0..20 {
        println!("{:6} | {:8.5} | {:8.5} | {:8.5}", 
                 i, output_nearest[i], output_linear[i], output_cubic[i]);
    }
}

fn multi_delay_example() {
    println!("\nMulti-channel Delay Example:");
    
    // Create a multi-channel delay line with 2 channels and capacity for 100 samples
    let mut multi_delay = MultiDelay::new(InterpolatorCubic::<f32>::new(), 2, 100);
    
    // Reset the delay line
    multi_delay.reset(0.0);
    
    // Create storage for output samples
    let mut output_ch0 = Vec::new();
    let mut output_ch1 = Vec::new();
    
    // Process 20 samples
    for i in 0..20 {
        // Input is an impulse at sample 0 for channel 0, and at sample 5 for channel 1
        let input = [
            if i == 0 { 1.0 } else { 0.0 },
            if i == 5 { 1.0 } else { 0.0 }
        ];
        
        // Read from the delay line with different delays for each channel
        let mut output = [0.0, 0.0];
        multi_delay.read_multi(&[3.5, 7.2], &mut output);
        
        output_ch0.push(output[0]);
        output_ch1.push(output[1]);
        
        // Write to the delay line
        multi_delay.write(&input);
    }
    
    // Print the output
    println!("Multi-channel delay output:");
    println!("Sample | Channel 0 | Channel 1");
    println!("-------|-----------|----------");
    for i in 0..20 {
        println!("{:6} | {:9.5} | {:9.5}", i, output_ch0[i], output_ch1[i]);
    }
}