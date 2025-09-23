//! Example usage of Spacing (custom room reverb)

use cute_dsp::spacing::{Spacing, Position};

fn main() {
    example_basic_room();
    example_two_sources();
    sweep_parameters_demo();
    example_with_all_params();
    example_with_long_buffer();
}

/// Basic room reverb example with two sources and two receivers
fn example_basic_room() {
    println!("\n=== Basic Room Example ===");
    let sample_rate = 48000.0;
    let mut spacing = Spacing::new(sample_rate);
    // Set reverb-like parameters
    spacing.set_room_size(1.2); // 20% larger room
    spacing.set_damping(0.2);   // Some high-frequency damping
    spacing.set_diff(0.5);      // 50% input diffusion
    spacing.set_bass(6.0);      // +6 dB bass boost
    spacing.set_decay(0.3);     // moderate decay
    spacing.set_cross(0.4);     // 40% cross-mix
    println!("Room size: {}  Damping: {}  Diff: {}  Bass: {}dB  Decay: {}  Cross: {}", spacing.room_size, spacing.damping, spacing.diff, spacing.bass, spacing.decay, spacing.cross);
    // Add two sources and two receivers
    let src1 = spacing.add_source(Position { x: 0.0, y: 0.0, z: 0.0 });
    let src2 = spacing.add_source(Position { x: 2.0, y: 0.0, z: 0.0 });
    let recv1 = spacing.add_receiver(Position { x: 5.0, y: 0.0, z: 0.0 });
    let recv2 = spacing.add_receiver(Position { x: 5.0, y: 3.0, z: 0.0 });
    // Add direct paths
    spacing.add_path(src1, recv1, 1.0, 0.0);
    spacing.add_path(src2, recv2, 0.8, 0.0);
    // Add a reflection (extra distance)
    spacing.add_path(src1, recv2, 0.5, 3.0); // Simulate a wall reflection
    // Prepare input (impulse) for each source
    let mut input1 = vec![0.0; 32];
    input1[0] = 1.0;
    let mut input2 = vec![0.0; 32];
    // Only src1 emits an impulse in this example
    let inputs: [&[f32]; 2] = [&input1, &input2];
    // Prepare outputs for each receiver
    let mut outputs = vec![vec![0.0; 32]; 2];
    spacing.process(&inputs, &mut outputs);
    // Print first 10 samples for each receiver
    for (idx, out) in outputs.iter().enumerate() {
        println!("Receiver {} output: {:?}", idx, &out[..10]);
    }
}

/// Example with two sources emitting impulses at different times
fn example_two_sources() {
    println!("\n=== Two Sources Example ===");
    let sample_rate = 48000.0;
    let mut spacing = Spacing::new(sample_rate);
    spacing.set_diff(0.7); // Stronger diffusion
    let src1 = spacing.add_source(Position { x: 0.0, y: 0.0, z: 0.0 });
    let src2 = spacing.add_source(Position { x: 1.0, y: 0.0, z: 0.0 });
    let recv = spacing.add_receiver(Position { x: 3.43, y: 0.0, z: 0.0 });
    spacing.add_path(src1, recv, 1.0, 0.0);
    spacing.add_path(src2, recv, 0.5, 0.0);
    let mut input1 = vec![0.0; 64];
    input1[0] = 1.0;
    let mut input2 = vec![0.0; 64];
    input2[20] = 1.0;
    let inputs: [&[f32]; 2] = [&input1, &input2];
    let mut outputs = vec![vec![0.0; 64]];
    spacing.process(&inputs, &mut outputs);
    println!("Receiver output (first 30 samples): {:?}", &outputs[0][..30]);
}

/// Sweep through different parameter settings and print the max output
fn sweep_parameters_demo() {
    println!("\n=== Sweep Spacing parameters ===");
    let sample_rate = 48000.0;
    let mut spacing = Spacing::new(sample_rate);
    let src = spacing.add_source(Position { x: 0.0, y: 0.0, z: 0.0 });
    let recv = spacing.add_receiver(Position { x: 3.43, y: 0.0, z: 0.0 });
    spacing.add_path(src, recv, 1.0, 0.0);
    let mut input = vec![0.0; 500];
    input[0] = 1.0;
    let inputs: [&[f32]; 1] = [&input];
    let mut outputs = vec![vec![0.0; 500]];
    for &room_size in &[1.0, 1.5, 2.0] {
        for &damping in &[0.0, 0.3, 0.7] {
            for &diff in &[0.0, 0.5] {
                for &bass in &[-6.0, 0.0, 6.0] {
                    for &decay in &[0.0, 0.5] {
                        for &cross in &[0.0, 0.7] {
                            spacing.set_room_size(room_size);
                            spacing.set_damping(damping);
                            spacing.set_diff(diff);
                            spacing.set_bass(bass);
                            spacing.set_decay(decay);
                            spacing.set_cross(cross);
                            spacing.clear_paths(); spacing.delays.clear();
                            spacing.add_path(src, recv, 1.0, 0.0);
                            spacing.process(&inputs, &mut outputs);
                            let found = outputs[0].iter().cloned().fold(f32::MIN, f32::max);
                            println!("room={:.1} damp={:.1} diff={:.1} bass={:.1} decay={:.1} cross={:.1} -> max={:.3}", room_size, damping, diff, bass, decay, cross, found);
                        }
                    }
                }
            }
        }
    }
}

/// Example with all parameters set to nonzero values
fn example_with_all_params() {
    println!("\n=== All Parameters Example ===");
    let sample_rate = 48000.0;
    let mut spacing = Spacing::new(sample_rate);
    spacing.set_room_size(1.5);
    spacing.set_damping(0.4);
    spacing.set_diff(0.8);
    spacing.set_bass(-4.0);
    spacing.set_decay(0.6);
    spacing.set_cross(0.9);
    let src1 = spacing.add_source(Position { x: 0.0, y: 0.0, z: 0.0 });
    let src2 = spacing.add_source(Position { x: 2.0, y: 0.0, z: 0.0 });
    let recv1 = spacing.add_receiver(Position { x: 5.0, y: 0.0, z: 0.0 });
    let recv2 = spacing.add_receiver(Position { x: 5.0, y: 3.0, z: 0.0 });
    spacing.add_path(src1, recv1, 1.0, 0.0);
    spacing.add_path(src2, recv2, 0.8, 0.0);
    spacing.add_path(src1, recv2, 0.5, 3.0);
    let mut input1 = vec![0.0; 40];
    input1[0] = 1.0;
    let mut input2 = vec![0.0; 40];
    input2[10] = 1.0;
    let inputs: [&[f32]; 2] = [&input1, &input2];
    let mut outputs = vec![vec![0.0; 40]; 2];
    spacing.process(&inputs, &mut outputs);
    for (idx, out) in outputs.iter().enumerate() {
        println!("Receiver {} output: {:?}", idx, &out[..20]);
    }
}

/// Example with a long buffer to show delayed impulse arrival
fn example_with_long_buffer() {
    println!("\n=== Long Buffer Example (for large room/delay) ===");
    let sample_rate = 48000.0;
    let mut spacing = Spacing::new(sample_rate);
    spacing.set_room_size(2.0); // Large room, long delay
    spacing.set_damping(0.2);
    spacing.set_decay(0.1);
    let src = spacing.add_source(Position { x: 0.0, y: 0.0, z: 0.0 });
    let recv = spacing.add_receiver(Position { x: 10.0, y: 0.0, z: 0.0 }); // 10m away
    spacing.add_path(src, recv, 1.0, 0.0);
    let mut input = vec![0.0; 3000];
    input[0] = 1.0;
    let inputs: [&[f32]; 1] = [&input];
    let mut outputs = vec![vec![0.0; 3000]];
    spacing.process(&inputs, &mut outputs);
    let (max_idx, max_val) = outputs[0].iter().enumerate().fold((0, f32::MIN), |(mi, mv), (i, v)| if *v > mv { (i, *v) } else { (mi, mv) });
    println!("Max output at sample {}: {:.4}", max_idx, max_val);
    println!("First 20 output samples: {:?}", &outputs[0][..20]);
    println!("Output around max: {:?}", &outputs[0][max_idx.saturating_sub(5)..(max_idx+6).min(outputs[0].len())]);
}
