//! WASM Binding Tests
//!
//! Tests for the WebAssembly bindings and JavaScript interop.

#[cfg(feature = "wasm")]
use wasm_bindgen_test::*;
#[cfg(feature = "wasm")]
wasm_bindgen_test_configure!(run_in_browser);

#[cfg(feature = "wasm")]
mod wasm_tests {
    use super::*;
    use cute_dsp::*;
    use wasm_bindgen::JsValue;

    #[wasm_bindgen_test]
    fn test_wasm_version() {
        // Version function is only available with wasm feature
        let version = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty());
        assert!(version.contains('.'));
    }

    #[wasm_bindgen_test]
    fn test_wasm_fft() {
        use cute_dsp::fft::Fft;
        use cute_dsp::windows::Window;

        let mut fft = Fft::<f32>::new(1024);
        let mut window = Window::<f32>::new(1024);

        // Create a test signal
        let mut signal = vec![0.0; 1024];
        for i in 0..1024 {
            signal[i] = (2.0 * std::f32::consts::PI * i as f32 / 1024.0).sin();
        }

        // Apply window
        window.fill(&mut signal, cute_dsp::windows::WindowType::Hann);

        // Perform FFT
        let mut spectrum = vec![num_complex::Complex::new(0.0, 0.0); 513];
        fft.real_fft(&signal, &mut spectrum);

        // Check that we got some non-zero values
        assert!(spectrum.iter().any(|&c| c.norm() > 0.1));
    }

    #[wasm_bindgen_test]
    fn test_wasm_filters() {
        use cute_dsp::filters::{Biquad, BiquadDesign};

        let mut filter = Biquad::<f32>::new(true);
        filter.lowpass(1000.0 / 44100.0, 0.7, BiquadDesign::Cookbook);

        let mut buffer = vec![1.0; 100];
        filter.process_buffer(&mut buffer);

        // Check that the filter processed the signal
        assert_ne!(buffer[0], 1.0);
        assert!(buffer.iter().any(|&x| x.abs() < 1.0));
    }

    #[wasm_bindgen_test]
    fn test_wasm_delay() {
        use cute_dsp::delay::{Delay, InterpolatorLinear};

        let mut delay = Delay::new(InterpolatorLinear::new(), 1000);
        delay.write(1.0);

        let output = delay.read(10.0);
        assert!(output >= 0.0 && output <= 1.0);
    }

    #[wasm_bindgen_test]
    fn test_wasm_envelopes() {
        use cute_dsp::envelopes::CubicLfo;

        let mut lfo = CubicLfo::new();
        lfo.set_frequency(1.0, 44100.0);

        let value = lfo.process();
        assert!(value >= -1.0 && value <= 1.0);
    }
}

#[cfg(not(feature = "wasm"))]
mod native_tests {
    #[test]
    fn test_native_version() {
        let version = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty());
        assert!(version.contains('.'));
    }
}