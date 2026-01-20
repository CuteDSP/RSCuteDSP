//! ADSR Envelope Example
//!
//! This example demonstrates how to use the ADSR envelope generator
//! for synthesizing note envelopes.

use cute_dsp::envelopes::Adsr;

fn main() {
    println!("ADSR Envelope Example");
    println!("====================\n");

    // Example 1: Simple ADSR with standard parameters
    simple_adsr_example();
    println!();

    // Example 2: ADSR with different sustain levels
    sustain_level_example();
    println!();

    // Example 3: ADSR with fast attack and slow release
    percussion_example();
}

fn simple_adsr_example() {
    println!("Example 1: Simple ADSR Envelope");
    println!("---------------------------------");
    
    let sample_rate = 44100.0;
    let mut adsr = Adsr::new(sample_rate);
    
    // Set ADSR parameters: attack=10ms, decay=100ms, sustain=0.7, release=200ms
    adsr.set_times(0.01, 0.1, 0.7, 0.2);
    
    println!("Sample rate: {} Hz", sample_rate as i32);
    println!("Attack: 10ms, Decay: 100ms, Sustain: 0.7, Release: 200ms\n");
    
    // Note on
    println!("Note ON:");
    adsr.gate(true);
    
    // Attack phase (10ms)
    print!("  Attack   (10ms): ");
    for _ in 0..4410 {
        adsr.next();
    }
    println!("value = {:.4}", adsr.value());
    
    // Decay phase (100ms)
    print!("  Decay   (100ms): ");
    for _ in 0..44100 {
        adsr.next();
    }
    println!("value = {:.4}", adsr.value());
    
    // Sustain phase (held for 200ms)
    print!("  Sustain (200ms): ");
    for _ in 0..88200 {
        adsr.next();
    }
    println!("value = {:.4}", adsr.value());
    
    // Note off
    println!("\nNote OFF:");
    adsr.gate(false);
    
    // Release phase (200ms)
    print!("  Release (200ms): ");
    for _ in 0..88200 {
        adsr.next();
    }
    println!("value = {:.4}", adsr.value());
}

fn sustain_level_example() {
    println!("Example 2: Different Sustain Levels");
    println!("------------------------------------");
    
    let sample_rate = 44100.0;
    
    // Generate three notes with different sustain levels
    let sustain_levels = vec![0.3, 0.5, 0.8];
    let labels = vec!["Low", "Medium", "High"];
    
    for (sustain, label) in sustain_levels.iter().zip(labels.iter()) {
        let mut adsr = Adsr::new(sample_rate);
        adsr.set_times(0.005, 0.05, *sustain, 0.1);
        
        adsr.gate(true);
        
        // Skip to sustain phase
        for _ in 0..(5000 + 2205) {
            adsr.next();
        }
        
        println!("{:8} sustain: {:.2}", label, adsr.value());
    }
}

fn percussion_example() {
    println!("Example 3: Percussion-like ADSR (Fast Attack, Slow Release)");
    println!("------------------------------------------------------------");
    
    let sample_rate = 44100.0;
    let mut adsr = Adsr::new(sample_rate);
    
    // Very fast attack, minimal decay/sustain, long release (like a cymbal)
    adsr.set_times(0.001, 0.01, 0.1, 0.5);
    
    println!("Attack: 1ms, Decay: 10ms, Sustain: 0.1, Release: 500ms\n");
    
    adsr.gate(true);
    
    // Just after attack (1ms)
    for _ in 0..44 {
        adsr.next();
    }
    println!("After attack (1ms):  value = {:.4}", adsr.value());
    
    // After decay (10ms more)
    for _ in 0..440 {
        adsr.next();
    }
    println!("After decay (10ms):  value = {:.4}", adsr.value());
    
    // Note off and release
    adsr.gate(false);
    
    // After 200ms of release
    for _ in 0..8820 {
        adsr.next();
    }
    println!("After release (200ms): value = {:.4}", adsr.value());
    
    // After 500ms of release (fully released)
    for _ in 0..13230 {
        adsr.next();
    }
    println!("After full release (500ms): value = {:.4}", adsr.value());
}
