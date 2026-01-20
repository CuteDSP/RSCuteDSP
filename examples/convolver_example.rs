// Example demonstrating the convolver with impulse response

use cute_dsp::convolver::Convolver;

fn main() {
    println!("Convolver / IR Example");
    println!("======================\n");

    // Example 1: Simple impulse response (identity)
    println!("Example 1: Identity IR (passes signal through)");
    let ir_identity = vec![1.0];
    let mut conv1: Convolver<f32> = Convolver::new(ir_identity);
    
    let test_signal = vec![1.0, 0.5, 0.25, 0.125];
    println!("Input:  {:?}", test_signal);
    let output: Vec<f32> = test_signal.iter().map(|&s| conv1.process(s)).collect();
    println!("Output: {:?}\n", output);

    // Example 2: Simple averaging filter (moving average)
    println!("Example 2: Simple Averaging Filter [0.5, 0.5]");
    let ir_average = vec![0.5, 0.5];
    let mut conv2: Convolver<f32> = Convolver::new(ir_average);
    conv2.reset();
    
    let test_signal2 = vec![1.0, 1.0, 0.0, 0.0];
    println!("Input:  {:?}", test_signal2);
    let output2: Vec<f32> = test_signal2.iter().map(|&s| conv2.process(s)).collect();
    println!("Output: {:?}\n", output2);

    // Example 3: Longer impulse response (echo/reverb-like)
    println!("Example 3: Echo IR [1.0, 0.5, 0.25, 0.125]");
    let ir_echo = vec![1.0, 0.5, 0.25, 0.125];
    let mut conv3: Convolver<f32> = Convolver::new(ir_echo);
    conv3.reset();
    
    let test_signal3 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    println!("Input:  {:?}", test_signal3);
    let output3: Vec<f32> = test_signal3.iter().map(|&s| conv3.process(s)).collect();
    println!("Output: {:?}\n", output3);

    // Example 4: Getting IR info
    println!("Example 4: Convolver Information");
    let ir_info = vec![1.0, 0.8, 0.6, 0.4, 0.2];
    let conv4: Convolver<f32> = Convolver::new(ir_info);
    println!("IR length: {}", conv4.ir_len());
    println!("IR coefficients: {:?}", conv4.get_ir());

    println!("\n=== Processing a longer audio stream ===");
    let mut conv5: Convolver<f32> = Convolver::new(vec![1.0, 0.3, 0.1]);
    let audio_stream = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05];
    
    println!("Input stream:  {:?}", audio_stream);
    let processed: Vec<f32> = audio_stream.iter().map(|&s| conv5.process(s)).collect();
    println!("Output stream: {:?}", processed);
}
