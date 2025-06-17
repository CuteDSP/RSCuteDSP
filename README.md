# Signalsmith DSP

A Rust port of the Signalsmith DSP C++ library, providing various DSP (Digital Signal Processing) algorithms for audio and signal processing.

[//]: # ([![Crates.io]&#40;https://img.shields.io/crates/v/signalsmith-dsp.svg&#41;]&#40;https://crates.io/crates/signalsmith-dsp&#41;)

[//]: # ([![Documentation]&#40;https://docs.rs/signalsmith-dsp/badge.svg&#41;]&#40;https://docs.rs/signalsmith-dsp&#41;)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **FFT**: Fast Fourier Transform implementation optimized for sizes that are products of 2^a * 3^b
- **Filters**: Biquad filters with various configurations (lowpass, highpass, bandpass, etc.)
- **Delay Lines**: Efficient delay line implementation with interpolation
- **Curves**: Cubic curve interpolation
- **Windows**: Window functions for spectral processing
- **Envelopes**: LFOs and envelope generators
- **Spectral Processing**: Tools for spectral manipulation
- **Time Stretching**: High-quality time stretching without pitch changes
- **STFT**: Short-time Fourier transform implementation
- **no_std Support**: Can be used in environments without the standard library

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
signalsmith-dsp = "0.0.1"
```

## Usage Examples

### FFT Example

```rust
use signalsmith_dsp::fft::{SimpleFFT, SimpleRealFFT};
use num_complex::Complex;

fn fft_example() {
    // Create a new FFT instance for size 1024
    let fft = SimpleFFT::<f32>::new(1024);

    // Input and output buffers
    let mut time_domain = vec![Complex::new(0.0, 0.0); 1024];
    let mut freq_domain = vec![Complex::new(0.0, 0.0); 1024];

    // Fill input with a sine wave
    for i in 0..1024 {
        time_domain[i] = Complex::new((i as f32 * 0.1).sin(), 0.0);
    }

    // Perform forward FFT
    fft.fft(&time_domain, &mut freq_domain);

    // Process in frequency domain if needed

    // Perform inverse FFT
    fft.ifft(&freq_domain, &mut time_domain);
}
```

### Biquad Filter Example

```rust
use signalsmith_dsp::filters::{Biquad, BiquadDesign, FilterType};

fn filter_example() {
    // Create a new biquad filter
    let mut filter = Biquad::<f32>::new(true);

    // Configure as a lowpass filter at 1000 Hz with Q=0.7 (assuming 44.1 kHz sample rate)
    filter.lowpass(1000.0 / 44100.0, 0.7, BiquadDesign::Cookbook);

    // Process a buffer of audio
    let mut audio_buffer = vec![0.0; 1024];
    // Fill buffer with audio data...

    // Apply the filter
    filter.process_buffer(&mut audio_buffer);
}
```

### Delay Line Example

```rust
use signalsmith_dsp::delay::{Delay, InterpolatorCubic};

fn delay_example() {
    // Create a delay line with cubic interpolation and 1 second capacity at 44.1 kHz
    let mut delay = Delay::new(InterpolatorCubic::<f32>::new(), 44100);

    // Process audio
    let mut output = 0.0;
    for _ in 0..1000 {
        let input = 0.5; // Replace with your input sample

        // Read from the delay line (500 ms delay)
        output = delay.read(22050.0);

        // Write to the delay line
        delay.write(input);

        // Use output...
    }
}
```

## Advanced Usage

For more advanced usage examples, see the examples directory:

- [FFT Example](examples/fft_example.rs) - Fast Fourier Transform
- [Filter Example](examples/filter_example.rs) - Biquad filters
- [Delay Example](examples/delay_example.rs) - Delay lines
- [STFT Example](examples/stft_example.rs) - Short-time Fourier transform
- [Stretch Example](examples/stretch_example.rs) - Time stretching
- [Pitch Shift Example](examples/pitch_shift_example.rs) - Pitch shifting
- [Curves Example](examples/curves_example.rs) - Curve interpolation
- [Envelopes Example](examples/envelopes_example.rs) - LFOs and envelope generators
- [Linear Example](examples/linear_example.rs) - Linear operations
- [Mix Example](examples/mix_example.rs) - Audio mixing utilities
- [Performance Example](examples/perf_example.rs) - Performance optimizations
- [Rates Example](examples/rates_example.rs) - Sample rate conversion
- [Spectral Example](examples/spectral_example.rs) - Spectral processing
- [Windows Example](examples/windows_example.rs) - Window functions

## Feature Flags

- `std` (default): Use the Rust standard library
- `alloc`: Enable allocation without std (for no_std environments)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original C++ library: [Signalsmith Audio DSP Library](https://github.com/signalsmith-audio/dsp)
