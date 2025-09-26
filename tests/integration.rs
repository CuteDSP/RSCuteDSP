//! Integration Tests for DSP Effects
//!
//! Tests that verify the complete DSP effect chains work correctly.

/// Test a complete filter + delay effect chain
#[test]
fn test_filter_delay_chain() {
    use cute_dsp::filters::{Biquad, BiquadDesign};
    use cute_dsp::delay::{Delay, InterpolatorLinear};

    let mut filter = Biquad::<f32>::new(true);
    let mut delay = Delay::new(InterpolatorLinear::new(), 1000);

    // Configure filter as lowpass
    filter.lowpass(1000.0 / 44100.0, 0.7, BiquadDesign::Cookbook);

    // Create test signal
    let mut buffer = vec![0.0; 512];
    for i in 0..512 {
        buffer[i] = (2.0 * std::f32::consts::PI * i as f32 * 440.0 / 44100.0).sin() * 0.5;
    }

    // Process through filter
    filter.process_buffer(&mut buffer);

    // Process through delay
    for sample in &mut buffer {
        let delayed = delay.read(100.0);
        delay.write(*sample + delayed * 0.3);
        *sample = *sample * 0.7 + delayed * 0.3;
    }

    // Verify output is reasonable
    assert!(buffer.iter().all(|&x: &f32| x.abs() <= 1.0));
    assert!(buffer.iter().any(|&x: &f32| x.abs() > 0.01));
}

/// Test basic FFT functionality
#[test]
fn test_fft_processing() {
    use cute_dsp::fft::SimpleRealFFT;
    use num_complex::Complex;

    let mut fft = SimpleRealFFT::<f32>::new(1024);

    // Create test signal
    let mut time_domain = vec![0.0; 1024];
    let mut freq_domain = vec![Complex::new(0.0, 0.0); 1024 / 2 + 1];

    // Fill with sine wave
    for i in 0..1024 {
        time_domain[i] = (2.0 * std::f32::consts::PI * i as f32 * 10.0 / 1024.0).sin();
    }

    // Forward FFT
    fft.fft(&time_domain, &mut freq_domain);

    // Check that we got some non-zero values
    assert!(freq_domain.iter().any(|&c| c.norm() > 0.1));

    // Inverse FFT
    let mut output = vec![0.0; 1024];
    fft.ifft(&freq_domain, &mut output);

    // Verify reconstruction
    assert!(output.iter().all(|&x: &f32| x.is_finite()));
}

/// Test windowing functionality
#[test]
fn test_windowing() {
    use cute_dsp::windows::Kaiser;

    let kaiser = Kaiser::new(6.0);
    let mut buffer = vec![1.0; 1024];

    // Apply Kaiser window
    kaiser.fill(&mut buffer);

    // Check that windowing was applied
    assert_ne!(buffer[0], 1.0);
    assert_ne!(buffer[512], 1.0);
    assert!(buffer.iter().all(|&x: &f32| x >= 0.0 && x <= 1.0));
}

/// Test basic envelope functionality
#[test]
fn test_envelope_processing() {
    use cute_dsp::envelopes::CubicLfo;

    let mut lfo = CubicLfo::new();
    lfo.set(0.0, 1.0, 0.1, 0.0, 0.0); // Simple LFO

    // Generate some values
    let mut values = vec![];
    for _ in 0..10 {
        values.push(lfo.next());
    }

    // Check that we get varying values
    assert!(values.iter().all(|&x: &f32| x >= 0.0 && x <= 1.0));
    assert!(values.iter().any(|&x: &f32| x > 0.1));
}

/// Test performance of DSP operations
#[test]
fn test_dsp_performance() {
    use cute_dsp::filters::{Biquad, BiquadDesign};

    let mut filter = Biquad::<f32>::new(true);
    filter.lowpass(1000.0 / 44100.0, 0.7, BiquadDesign::Cookbook);

    let mut buffer = vec![0.0; 44100]; // 1 second of audio
    for i in 0..44100 {
        buffer[i] = (2.0 * std::f32::consts::PI * i as f32 * 440.0 / 44100.0).sin() * 0.5;
    }

    let start = std::time::Instant::now();
    filter.process_buffer(&mut buffer);
    let duration = start.elapsed();

    // Should process 1 second of audio in less than 100ms on modern hardware
    assert!(duration.as_millis() < 100);
}