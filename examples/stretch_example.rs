//! Example usage of SignalsmithStretch for time-stretching and pitch-shifting

use signalsmith_dsp::stretch::SignalsmithStretch;
use std::f32::consts::PI;

fn main() {
    mono_ramp_demo();
    stereo_demo();
    sine_wave_demo();
    formant_demo();
    custom_map_demo();
} 

fn mono_ramp_demo() {
    println!("=== Mono ramp time-stretch & pitch-shift ===");
    let sample_rate = 48000.0_f32;
    let channels = 1;
    let block_samples = 1024;
    let interval_samples = 256;
    let input_len = 2048;
    let mut input: Vec<Vec<f32>> = vec![vec![0.0; input_len]; channels];
    for i in 0..input_len {
        input[0][i] = (i as f32 / input_len as f32) * 2.0 - 1.0;
    }
    let output_len = (input_len as f32 * 1.5) as usize;
    let mut output: Vec<Vec<f32>> = vec![vec![0.0; output_len]; channels];
    let mut stretch = SignalsmithStretch::<f32>::new();
    stretch.configure(channels, block_samples, interval_samples, false);
    stretch.set_transpose_semitones(3.0, 0.0);
    stretch.set_formant_semitones(1.0, false);
    stretch.process(&input, input_len, &mut output, output_len);
    println!("Input (first 10): {:?}", &input[0][..10]);
    println!("Output (first 10): {:?}", &output[0][..10]);
    println!("Output (last 10): {:?}\n", &output[0][output_len-10..]);
}

fn stereo_demo() {
    println!("=== Stereo time-stretch & pitch-shift ===");
    let sample_rate = 48000.0_f32;
    let channels = 2;
    let block_samples = 1024;
    let interval_samples = 256;
    let input_len = 2048;
    let mut input: Vec<Vec<f32>> = vec![vec![0.0; input_len]; channels];
    for i in 0..input_len {
        input[0][i] = ((i as f32 / input_len as f32) * 2.0 - 1.0) * 0.8; // left
        input[1][i] = (((input_len - i) as f32 / input_len as f32) * 2.0 - 1.0) * 0.5; // right
    }
    let output_len = (input_len as f32 * 0.75) as usize;
    let mut output: Vec<Vec<f32>> = vec![vec![0.0; output_len]; channels];
    let mut stretch = SignalsmithStretch::<f32>::new();
    stretch.configure(channels, block_samples, interval_samples, false);
    stretch.set_transpose_semitones(-5.0, 0.0); // pitch down
    stretch.process(&input, input_len, &mut output, output_len);
    println!("Stereo input L (first 10): {:?}", &input[0][..10]);
    println!("Stereo input R (first 10): {:?}", &input[1][..10]);
    println!("Stereo output L (first 10): {:?}", &output[0][..10]);
    println!("Stereo output R (first 10): {:?}\n", &output[1][..10]);
}

fn sine_wave_demo() {
    println!("=== Sine wave pitch-shifting ===");
    let sample_rate = 48000.0_f32;
    let freq = 440.0;
    let block_samples = 1024;
    let interval_samples = 256;
    let input_len = 1024;
    let mut input: Vec<Vec<f32>> = vec![vec![0.0; input_len]];
    for i in 0..input_len {
        input[0][i] = (2.0 * PI * freq * (i as f32) / sample_rate).sin();
    }
    let output_len = input_len;
    let mut output: Vec<Vec<f32>> = vec![vec![0.0; output_len]];
    let mut stretch = SignalsmithStretch::<f32>::new();
    stretch.configure(1, block_samples, interval_samples, false);
    stretch.set_transpose_semitones(12.0, 0.0); // up one octave
    stretch.process(&input, input_len, &mut output, output_len);
    println!("Sine input (first 10): {:?}", &input[0][..10]);
    println!("Sine output (first 10): {:?}\n", &output[0][..10]);
}

fn formant_demo() {
    println!("=== Formant preservation demo ===");
    let sample_rate = 48000.0_f32;
    let block_samples = 1024;
    let interval_samples = 256;
    let input_len = 1024;
    let mut input: Vec<Vec<f32>> = vec![vec![0.0; input_len]];
    // Simulate a "vowel" with two sine waves (formants)
    let f1 = 700.0;
    let f2 = 1200.0;
    for i in 0..input_len {
        input[0][i] = 0.6 * (2.0 * PI * f1 * (i as f32) / sample_rate).sin()
                    + 0.4 * (2.0 * PI * f2 * (i as f32) / sample_rate).sin();
    }
    let output_len = input_len;
    let mut output_no_formant: Vec<Vec<f32>> = vec![vec![0.0; output_len]];
    let mut output_with_formant: Vec<Vec<f32>> = vec![vec![0.0; output_len]];
    let mut stretch = SignalsmithStretch::<f32>::new();
    stretch.configure(1, block_samples, interval_samples, false);
    stretch.set_transpose_semitones(7.0, 0.0); // up a fifth
    // No formant preservation
    stretch.set_formant_semitones(0.0, false);
    stretch.process(&input, input_len, &mut output_no_formant, output_len);
    // With formant preservation
    stretch.set_formant_semitones(-7.0, true); // compensate pitch shift
    stretch.process(&input, input_len, &mut output_with_formant, output_len);
    println!("Formant input (first 10): {:?}", &input[0][..10]);
    println!("Output w/o formant (first 10): {:?}", &output_no_formant[0][..10]);
    println!("Output w/  formant (first 10): {:?}\n", &output_with_formant[0][..10]);
}

fn custom_map_demo() {
    println!("=== Custom frequency map demo ===");
    let sample_rate = 48000.0_f32;
    let block_samples = 1024;
    let interval_samples = 256;
    let input_len = 1024;
    let mut input: Vec<Vec<f32>> = vec![vec![0.0; input_len]];
    for i in 0..input_len {
        input[0][i] = (2.0 * PI * (i as f32) / 32.0).sin();
    }
    let output_len = input_len;
    let mut output: Vec<Vec<f32>> = vec![vec![0.0; output_len]];
    let mut stretch = SignalsmithStretch::<f32>::new();
    stretch.configure(1, block_samples, interval_samples, false);
    // Custom frequency map: invert spectrum (for demonstration)
    stretch.set_freq_map(move |freq| {
        let max_freq = sample_rate / 2.0;
        max_freq - freq
    });
    stretch.process(&input, input_len, &mut output, output_len);
    println!("Custom map input (first 10): {:?}", &input[0][..10]);
    println!("Custom map output (first 10): {:?}", &output[0][..10]);
}