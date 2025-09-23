//! Mix Example
//!
//! This example demonstrates how to use the mixing functionality
//! in the Signalsmith DSP library.

use cute_dsp::mix::{Hadamard, Householder, StereoMultiMixer, cheap_energy_crossfade};

fn main() {
    println!("Mix Example");
    println!("===========");

    // Example 1: Hadamard transform
    hadamard_example();

    // Example 2: Householder transform
    householder_example();

    // Example 3: Stereo to multi-channel mixing
    stereo_multi_mixer_example();

    // Example 4: Energy-preserving crossfade
    crossfade_example();
}

fn hadamard_example() {
    println!("\nHadamard Transform Example:");
    
    // Create a Hadamard transform for size 4
    let hadamard = Hadamard::<f32>::new(4);
    
    // Create test data
    let mut data = [1.0, 2.0, 3.0, 4.0];
    
    // Print original data
    print!("Original data: ");
    for val in data.iter() {
        print!("{:.1} ", val);
    }
    println!();
    
    // Apply Hadamard transform
    hadamard.in_place(&mut data);
    
    // Print transformed data
    print!("Transformed data: ");
    for val in data.iter() {
        print!("{:.1} ", val);
    }
    println!();
    
    // Apply Hadamard transform again to get back the original data (scaled)
    hadamard.in_place(&mut data);
    
    // Print reconstructed data (scaled by the transform)
    print!("Reconstructed data (scaled): ");
    for val in data.iter() {
        print!("{:.1} ", val);
    }
    println!();
    
    // Scale by the inverse of the scaling factor to get the original data
    let scaling = hadamard.scaling_factor();
    for val in data.iter_mut() {
        *val /= scaling;
    }
    
    // Print reconstructed data (unscaled)
    print!("Reconstructed data (unscaled): ");
    for val in data.iter() {
        print!("{:.1} ", val);
    }
    println!();
    
    // Demonstrate unscaled transform
    let mut data = [1.0, 2.0, 3.0, 4.0];
    hadamard.unscaled_in_place(&mut data);
    print!("Unscaled transform: ");
    for val in data.iter() {
        print!("{:.1} ", val);
    }
    println!();
}

fn householder_example() {
    println!("\nHouseholder Transform Example:");
    
    // Create a Householder transform for size 4
    let householder = Householder::<f32>::new(4);
    
    // Create test data
    let mut data = [1.0, 2.0, 3.0, 4.0];
    
    // Print original data
    print!("Original data: ");
    for val in data.iter() {
        print!("{:.1} ", val);
    }
    println!();
    
    // Apply Householder transform
    householder.in_place(&mut data);
    
    // Print transformed data
    print!("Transformed data: ");
    for val in data.iter() {
        print!("{:.1} ", val);
    }
    println!();
    
    // Apply Householder transform again to get back the original data (scaled)
    householder.in_place(&mut data);
    
    // Print reconstructed data (scaled by the transform)
    print!("Reconstructed data (scaled): ");
    for val in data.iter() {
        print!("{:.1} ", val);
    }
    println!();
    
    // Scale by the inverse of the scaling factor to get the original data
    let scaling = householder.scaling_factor();
    for val in data.iter_mut() {
        *val /= scaling;
    }
    
    // Print reconstructed data (unscaled)
    print!("Reconstructed data (unscaled): ");
    for val in data.iter() {
        print!("{:.1} ", val);
    }
    println!();
}

fn stereo_multi_mixer_example() {
    println!("\nStereo Multi Mixer Example:");
    
    // Create a stereo to multi-channel mixer for 5 channels
    let mixer = StereoMultiMixer::<f32>::new(4);
    
    // Create stereo input
    let stereo_input = [0.7, 0.3]; // Left and right channels
    
    // Create multi-channel output buffer
    let mut multi_output = [0.0; 5];
    
    // Convert stereo to multi-channel
    mixer.stereo_to_multi(&stereo_input, &mut multi_output);
    
    // Print multi-channel output
    println!("Stereo input: [{:.1}, {:.1}]", stereo_input[0], stereo_input[1]);
    print!("Multi-channel output: ");
    for val in multi_output.iter() {
        print!("{:.3} ", val);
    }
    println!();
    
    // Create multi-channel input
    let multi_input = [0.2, 0.4, 0.6, 0.8, 1.0];
    
    // Create stereo output buffer
    let mut stereo_output = [0.0, 0.0];
    
    // Convert multi-channel to stereo
    mixer.multi_to_stereo(&multi_input, &mut stereo_output);
    
    // Print stereo output
    print!("Multi-channel input: ");
    for val in multi_input.iter() {
        print!("{:.1} ", val);
    }
    println!();
    println!("Stereo output: [{:.3}, {:.3}]", stereo_output[0], stereo_output[1]);
    
    // Print scaling factors
    println!("Scaling factor 1: {:.3}", mixer.scaling_factor1());
    println!("Scaling factor 2: {:.3}", mixer.scaling_factor2());
}

fn crossfade_example() {
    println!("\nCheap Energy Crossfade Example:");
    
    // Create two signals to crossfade between
    let signal1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let signal2 = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    
    // Create output buffer
    let mut output = [0.0; 10];
    
    // Perform crossfade
    println!("Crossfading from signal1 to signal2:");
    for i in 0..10 {
        // Calculate crossfade coefficient (0 to 1)
        let x = i as f32 / 9.0;
        
        // Calculate coefficients for energy-preserving crossfade
        let mut from_coeff = 0.0;
        let mut to_coeff = 0.0;
        cheap_energy_crossfade(x, &mut to_coeff, &mut from_coeff);
        
        // Apply crossfade
        output[i] = from_coeff * signal1[i] + to_coeff * signal2[i];
        
        println!("Step {}: x={:.2}, from_coeff={:.3}, to_coeff={:.3}, output={:.3}",
                 i, x, from_coeff, to_coeff, output[i]);
    }
    
    // Print final output
    print!("Crossfaded output: ");
    for val in output.iter() {
        print!("{:.3} ", val);
    }
    println!();
}