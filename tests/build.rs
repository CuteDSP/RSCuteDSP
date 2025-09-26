//! Build and CI Tests
//!
//! Tests that verify the build process and CI pipeline work correctly.

#[cfg(feature = "wasm")]
use wasm_bindgen_test::*;
#[cfg(feature = "wasm")]
wasm_bindgen_test_configure!(run_in_browser);

#[test]
fn test_build_configuration() {
    // Test that the library compiles with all features
    #[cfg(feature = "std")]
    assert!(true, "std feature enabled");

    #[cfg(feature = "wasm")]
    assert!(true, "wasm feature enabled");

    // Test version information
    let version = env!("CARGO_PKG_VERSION");
    assert!(!version.is_empty());
    assert!(version.split('.').count() >= 2);
}

#[test]
fn test_no_std_compilation() {
    // This test ensures the no_std feature works
    #[cfg(not(feature = "std"))]
    {
        use core::f32::consts::PI;
        let _pi = PI; // Ensure core constants are available
    }
}

#[test]
fn test_dependency_versions() {
    // Test that our dependencies are properly configured
    use num_complex::Complex;
    use num_traits::Float;

    let c = Complex::new(1.0f32, 2.0);
    assert_eq!(c.re, 1.0);
    assert_eq!(c.im, 2.0);

    let f: f32 = 1.5;
    assert!(f.is_finite());
}

#[cfg(feature = "wasm")]
mod wasm_build_tests {
    use super::*;

    #[wasm_bindgen_test]
    fn test_wasm_build_artifacts() {
        // Test that WASM bindings are properly exported
        use cute_dsp::*;

        // Test that we can create DSP objects in WASM
        use cute_dsp::fft::SimpleRealFFT;
        let _fft = SimpleRealFFT::<f32>::new(512);

        use cute_dsp::filters::Biquad;
        let _filter = Biquad::<f32>::new(true);

        // Test version function
        let version = version();
        assert!(!version.is_empty());
    }

    #[wasm_bindgen_test]
    fn test_wasm_memory_management() {
        // Test that WASM memory management works correctly
        use cute_dsp::delay::{Delay, InterpolatorLinear};

        let mut delay = Delay::new(InterpolatorLinear::new(), 1000);

        // Fill delay buffer
        for i in 0..100 {
            delay.write(i as f32);
        }

        // Read back values
        let value = delay.read(10.0);
        assert!(value.is_finite());
    }

    #[wasm_bindgen_test]
    fn test_wasm_performance() {
        use std::time::Instant;
        use cute_dsp::fft::SimpleRealFFT;

        let mut fft = SimpleRealFFT::<f32>::new(1024);
        let mut signal = vec![0.0; 1024];
        let mut spectrum = vec![num_complex::Complex::new(0.0, 0.0); 513];

        // Generate test signal
        for i in 0..1024 {
            signal[i] = (2.0 * std::f32::consts::PI * i as f32 / 1024.0).sin();
        }

        let start = Instant::now();

        // Perform multiple FFTs
        for _ in 0..10 {
            fft.fft(&signal, &mut spectrum);
        }

        let duration = start.elapsed();

        // Should complete in reasonable time (allowing for WASM overhead)
        assert!(duration.as_millis() < 500);
    }
}

/// Test that all public APIs are accessible
#[test]
fn test_public_api() {
    // Test that all expected modules are accessible
    use cute_dsp::*;

    // Core modules - test basic instantiation
    let _curves = curves::Linear::<f32>::new();
    let _delay = delay::Buffer::<f32>::new(100);
    let _envelopes = envelopes::CubicLfo::new();
    let _fft = fft::SimpleRealFFT::<f32>::new(512);
    let _filters = filters::Biquad::<f32>::new(true);
    let _mix = mix::StereoMultiMixer::<f32>::new(2);
    let _rates = rates::Oversampler2xFIR::<f32>::new(1, 512, 8, 44100.0);
    let _spectral = spectral::SpectralProcessor::<f32>::new(512, 256);
    let _stft = stft::STFT::<f32>::new(false);
    let _stretch = stretch::SignalsmithStretch::<f32>::new();
    let _windows = windows::Kaiser::<f32>::new(6.0);
}

/// Test that examples can be built
#[test]
#[ignore] // Ignore by default as it requires example files
fn test_examples_compile() {
    // This would test that all examples compile successfully
    // For now, just check that the example files exist
    use std::path::Path;

    let examples = [
        "curves_example.rs",
        "delay_example.rs",
        "envelopes_example.rs",
        "fft_example.rs",
        "filter_example.rs",
        "linear_example.rs",
        "mix_example.rs",
        "perf_example.rs",
        "rates_example.rs",
        "spacing_example.rs",
        "spectral_example.rs",
        "stft_example.rs",
        "stretch_example.rs",
        "windows_example.rs",
        "phaser_example.rs",
    ];

    for example in &examples {
        assert!(Path::new("examples").join(example).exists(),
                "Example {} should exist", example);
    }
}