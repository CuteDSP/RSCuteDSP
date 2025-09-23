//! Envelopes Example
//!
//! This example demonstrates how to use the envelope functionality
//! in the Signalsmith DSP library.

use cute_dsp::envelopes::{CubicLfo, BoxFilter, BoxStackFilter, PeakHold, PeakDecayLinear};

fn main() {
    println!("Envelopes Example");
    println!("================");

    // Example 1: Cubic LFO (Low Frequency Oscillator)
    cubic_lfo_example();

    // Example 2: Box Filter (Moving Average)
    box_filter_example();

    // Example 3: Box Stack Filter (Cascaded Box Filters)
    box_stack_filter_example();

    // Example 4: Peak Hold (Envelope Follower)
    peak_hold_example();

    // Example 5: Peak Decay Linear (Envelope Follower)
    peak_decay_linear_example();
}

fn cubic_lfo_example() {
    println!("\nCubic LFO Example:");
    
    // Create a new cubic LFO
    let mut lfo = CubicLfo::new();
    
    // Set LFO parameters: low, high, rate, rate_variation, depth_variation
    lfo.set(0.0, 1.0, 0.1, 0.2, 0.1);
    
    // Generate and print some values
    println!("Generating 10 LFO values:");
    for i in 0..10 {
        let value = lfo.next();
        println!("Value {}: {}", i, value);
    }
    
    // Reset the LFO
    lfo.reset();
    println!("After reset, next value: {}", lfo.next());
    
    // Create an LFO with a specific seed
    let mut seeded_lfo = CubicLfo::with_seed(12345);
    seeded_lfo.set(0.0, 1.0, 0.1, 0.2, 0.1);
    println!("Seeded LFO first value: {}", seeded_lfo.next());
}

fn box_filter_example() {
    println!("\nBox Filter Example:");
    
    // Create a box filter with maximum length 10
    let mut filter = BoxFilter::<f32>::new(10);
    
    // Set the filter length
    filter.set(5);
    
    // Reset the filter with a specific value
    filter.reset(0.0);
    
    // Process some impulse data
    println!("Processing impulse response:");
    let mut value = 1.0;
    for i in 0..15 {
        value = filter.process(if i == 0 { 1.0 } else { 0.0 });
        println!("Output {}: {}", i, value);
    }
    
    // Process a step function
    filter.reset(0.0);
    println!("\nProcessing step response:");
    for i in 0..15 {
        value = filter.process(1.0);
        println!("Output {}: {}", i, value);
    }
}

fn box_stack_filter_example() {
    println!("\nBox Stack Filter Example:");
    
    // Create a box stack filter with maximum size 100 and 3 layers
    let mut filter = BoxStackFilter::<f32>::new(100, 3);
    
    // Set the filter size
    filter.set(50);
    
    // Reset the filter
    filter.reset();
    
    // Process some impulse data
    println!("Processing impulse response:");
    let mut value = 1.0;
    for i in 0..20 {
        value = filter.process(if i == 0 { 1.0 } else { 0.0 });
        println!("Output {}: {}", i, value);
    }
    
    // Process a step function
    filter.reset();
    println!("\nProcessing step response:");
    for i in 0..20 {
        value = filter.process(1.0);
        println!("Output {}: {}", i, value);
    }
    
    // Print some information about the filter
    println!("\nFilter properties:");
    println!("Bandwidth for 3 layers: {}", BoxStackFilter::<f32>::layers_to_bandwidth(3));
    println!("Peak attenuation for 3 layers: {} dB", BoxStackFilter::<f32>::layers_to_peak_db(3));
}

fn peak_hold_example() {
    println!("\nPeak Hold Example:");
    
    // Create a peak hold with maximum length 10
    let mut peak_hold = PeakHold::<f32>::new(10);
    
    // Set the peak hold size
    peak_hold.set(5, false);
    
    // Process some data
    println!("Processing data:");
    let data = [0.1, 0.5, 0.3, 0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0];
    for (i, &value) in data.iter().enumerate() {
        let output = peak_hold.process(value);
        println!("Input: {}, Output: {}", value, output);
    }
    
    // Reset the peak hold
    peak_hold.reset();
    println!("\nAfter reset:");
    println!("Current peak: {}", peak_hold.read());
}

fn peak_decay_linear_example() {
    println!("\nPeak Decay Linear Example:");
    
    // Create a peak decay linear with maximum length 100
    let mut peak_decay = PeakDecayLinear::<f32>::new(100);
    
    // Set the decay length
    peak_decay.set(10.0);
    
    // Reset the peak decay
    peak_decay.reset();
    
    // Process some data
    println!("Processing data:");
    let data = [0.1, 0.5, 0.3, 0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for (i, &value) in data.iter().enumerate() {
        let output = peak_decay.process(value);
        println!("Input: {}, Output: {}", value, output);
    }
}