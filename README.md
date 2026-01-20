# RSCuteDSP

A Rust port of the Signalsmith DSP C++ library, providing various DSP (Digital Signal Processing) algorithms for audio and signal processing. This library implements the same high-quality algorithms as the original C++ library, optimized for Rust performance and ergonomics.

[![Crates.io](https://img.shields.io/crates/v/cute-dsp)](https://crates.io/crates/cute-dsp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **FFT**: Fast Fourier Transform implementation optimized for sizes that are products of 2^a * 3^b, with both complex and real-valued implementations
- **Filters**: Biquad filters with various configurations (lowpass, highpass, bandpass, etc.) and design methods (Butterworth, cookbook)
- **Delay Lines**: Efficient delay line implementation with multiple interpolation methods (nearest, linear, cubic)
- **Curves**: Cubic curve interpolation with control over slope and curvature
- **Windows**: Window functions for spectral processing (Hann, Hamming, Kaiser, Blackman-Harris, etc.)
- **Envelopes**: LFOs, envelope generators, and **ADSR** envelope with gate control for synthesizer voices
- **Spectral Processing**: Tools for spectral manipulation, phase vocoding, and frequency-domain operations
- **Time Stretching**: High-quality time stretching and pitch shifting using phase vocoder techniques
- **STFT**: Short-time Fourier transform implementation with overlap-add processing
- **Mixing Utilities**: Multi-channel mixing matrices and stereo-to-multi-channel conversion
- **Linear Algebra**: Expression template system for efficient vector operations
- **no_std Support**: Can be used in environments without the standard library, with optional `alloc` feature
- **Spacing (Room Reverb)**: Simulate room acoustics with customizable geometry, early reflections, and multi-channel output

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
cute-dsp = "0.0.2"
```

## WebAssembly Support

This library supports compilation to WebAssembly for use in web browsers and Node.js, with full DSP functionality exposed.

To build for WASM (no-modules target for broad compatibility):

1. Install `wasm-pack`:
   ```bash
   cargo install wasm-pack
   ```

2. Build the WASM package:
   ```bash
   wasm-pack build --target no-modules --out-dir pkg --features wasm
   ```

3. Use in HTML/JavaScript (no-modules target):
   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <meta charset="utf-8">
       <title>CuteDSP WASM Demo</title>
   </head>
   <body>
       <script src="pkg/cute_dsp.js"></script>
       <script>
       async function run() {
           // Initialize WASM
           await wasm_bindgen();

           // Create FFT instance
           const fft = new wasm_bindgen.WasmFFT(1024);

           // Create input arrays
           const realIn = new Float32Array(1024);
           const imagIn = new Float32Array(1024);
           const realOut = new Float32Array(1024);
           const imagOut = new Float32Array(1024);

           // Fill input with a sine wave
           for (let i = 0; i < 1024; i++) {
               realIn[i] = Math.sin(2 * Math.PI * i / 1024);
               imagIn[i] = 0;
           }

           // Perform FFT
           fft.fft_forward(realIn, imagIn, realOut, imagOut);

           // Create and use a biquad filter
           const filter = new wasm_bindgen.WasmBiquad();
           filter.lowpass(0.1, 0.7); // Normalized frequency, Q factor

           const audioIn = new Float32Array(1024);
           const audioOut = new Float32Array(1024);
           // Fill audioIn with audio data...

           filter.process(audioIn, audioOut);

           // Create a delay line
           const delay = new wasm_bindgen.WasmDelay(44100); // Max delay samples
           const delayedSample = delay.process(0.5, 22050.0); // Input sample, delay in samples

           // Create an LFO
           const lfo = new wasm_bindgen.WasmLFO();
           lfo.set_params(0.0, 1.0, 5.0, 0.0, 0.0); // low, high, rate, rate_variation, depth_variation
           const lfoSample = lfo.process();

           // Create window functions
           const kaiser = new wasm_bindgen.WasmKaiser(0.1); // Beta parameter
           const windowData = new Float32Array(512);
           kaiser.fill(windowData);

           // Hann and Hamming windows
           wasm_bindgen.WasmHann.fill(windowData);
           wasm_bindgen.WasmHamming.fill(windowData);

           // Create a delay line
           const delay = new wasm_bindgen.WasmDelay(44100); // Max delay samples
           const delayedSample = delay.process(0.5, 22050.0); // Input sample, delay in samples

           // Create an LFO
           const lfo = new wasm_bindgen.WasmLFO();
           lfo.set_params(0.0, 1.0, 5.0, 0.0, 0.0); // low, high, rate, rate_variation, depth_variation
           const lfoSample = lfo.process();

           // Create window functions
           const kaiser = new wasm_bindgen.WasmKaiser(0.1); // Beta parameter
           const windowData = new Float32Array(512);
           kaiser.fill(windowData);

           // Hann and Hamming windows
           wasm_bindgen.WasmHann.fill(windowData);
           wasm_bindgen.WasmHamming.fill(windowData);

           // Create STFT
           const stft = new wasm_bindgen.WasmSTFT(false); // false for non-modified
           stft.configure(1, 1, 512); // input channels, output channels, block size

           // Linear curve mapping
           const curve = wasm_bindgen.WasmLinearCurve.from_points(0.0, 1.0, 0.0, 100.0);
           const mappedValue = curve.evaluate(0.5);

           // Sample rate conversion filters
           const srcFilter = new Float32Array(256);
           wasm_bindgen.WasmSampleRateConverter.fill_kaiser_sinc_filter(srcFilter, 0.45, 0.55);

           // Spectral utilities
           const complex = wasm_bindgen.WasmSpectralUtils.mag_phase_to_complex(1.0, 0.5);
           const magPhase = wasm_bindgen.WasmSpectralUtils.complex_to_mag_phase(1.0, 0.0);
           const dbValue = wasm_bindgen.WasmSpectralUtils.linear_to_db(10.0);

           // Multi-channel mixing
           const hadamard = new wasm_bindgen.WasmHadamardMixer(4); // 4-channel mixer
           const channels = new Float32Array([1.0, 0.5, 0.3, 0.1]);
           hadamard.mix_in_place(channels);

           console.log('DSP operations completed!');
       }

       run();
       </script>
   </body>
   </html>
   ```

   Or try the included real-time demo: `wasm_demo.html`

## Usage Examples

### FFT Example

```rust
use cute_dsp::fft::{SimpleFFT, SimpleRealFFT};
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
use cute_dsp::filters::{Biquad, BiquadDesign, FilterType};

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
use cute_dsp::delay::{Delay, InterpolatorCubic};

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

### Time Stretching and Pitch Shifting Example

```rust
use cute_dsp::stretch::SignalsmithStretch;

fn time_stretch_example() {
    // Create a new stretch processor
    let mut stretcher = SignalsmithStretch::<f32>::new();

    // Configure for 2x time stretching with pitch shift up 3 semitones
    stretcher.configure(1, 1024, 256, false); // 1 channel, 1024 block size, 256 interval
    stretcher.set_transpose_semitones(3.0, 0.5); // Pitch up 3 semitones
    stretcher.set_formant_semitones(1.0, false); // Formant shift

    // Input and output buffers
    let input_len = 1024;
    let output_len = 2048; // 2x longer output
    let input = vec![vec![0.0; input_len]]; // Fill with audio data
    let mut output = vec![vec![0.0; output_len]];

    // Process the audio
    stretcher.process(&input, input_len, &mut output, output_len);
}
```

### Multichannel Mixing Example

```rust
use cute_dsp::mix::{Hadamard, StereoMultiMixer};

fn mixing_example() {
    // Create a Hadamard matrix for 4-channel mixing
    let hadamard = Hadamard::<f32>::new(4);

    // Mix 4 channels in-place
    let mut data = vec![1.0, 2.0, 3.0, 4.0];
    hadamard.in_place(&mut data);
    // data now contains orthogonal mix of input channels

    // Create a stereo-to-multichannel mixer (must be even number of channels)
    let mixer = StereoMultiMixer::<f32>::new(6);

    // Convert stereo to 6 channels
    let stereo_input = [0.5, 0.8];
    let mut multi_output = vec![0.0; 6];
    mixer.stereo_to_multi(&stereo_input, &mut multi_output);

    // Convert back to stereo
    let mut stereo_output = [0.0, 0.0];
    mixer.multi_to_stereo(&multi_output, &mut stereo_output);

    // Apply energy-preserving crossfade
    let mut to_coeff = 0.0;
    let mut from_coeff = 0.0;
    cute_dsp::mix::cheap_energy_crossfade(0.3, &mut to_coeff, &mut from_coeff);
    // Use coefficients for crossfading between signals
}
```

### ADSR Envelope Example

```rust
use cute_dsp::envelopes::Adsr;

fn adsr_example() {
    // Create ADSR with 44.1kHz sample rate
    let mut adsr = Adsr::new(44100.0);
    
    // Set parameters: attack, decay, sustain, release (in seconds)
    adsr.set_times(0.01, 0.1, 0.7, 0.2);
    
    // Note on (start attack)
    adsr.gate(true);
    
    // Generate envelope samples
    for _ in 0..44100 {
        let envelope_value = adsr.next();
        // Use envelope_value to modulate audio signal
    }
    
    // Note off (start release)
    adsr.gate(false);
    
    // Continue generating samples during release
    for _ in 0..8820 {
        let envelope_value = adsr.next();
    }
}
```

For WebAssembly usage, see [ADSR_WASM.md](ADSR_WASM.md) for complete browser integration examples.

### Spacing (Room Reverb) Example

```rust
use cute_dsp::spacing::{Spacing, Position};

fn spacing_example() {
    // Create a new Spacing effect with a given sample rate
    let mut spacing = Spacing::<f32>::new(48000.0);
    // Add source and receiver positions (in meters)
    let src = spacing.add_source(Position { x: 0.0, y: 0.0, z: 0.0 });
    let recv = spacing.add_receiver(Position { x: 3.43, y: 0.0, z: 0.0 });
    // Add a direct path
    spacing.add_path(src, recv, 1.0, 0.0);
    // Prepare input and output buffers
    let mut input = vec![0.0; 500];
    input[0] = 1.0; // Impulse
    let mut outputs = vec![vec![0.0; 500]];
    // Process the input through the effect
    spacing.process(&[&input], &mut outputs);
    // outputs[0] now contains the processed signal
}
```

## Advanced Usage

For more advanced usage examples, see the examples directory:

- [FFT Example](examples/fft_example.rs) - Fast Fourier Transform
- [Filter Example](examples/filter_example.rs) - Biquad filters
- [Delay Example](examples/delay_example.rs) - Delay lines
- [STFT Example](examples/stft_example.rs) - Short-time Fourier transform
- [Stretch Example](examples/stretch_example.rs) - Time stretching and pitch shifting
- [Curves Example](examples/curves_example.rs) - Curve interpolation
- [Envelopes Example](examples/envelopes_example.rs) - LFOs and envelope generators
- [ADSR Example](examples/adsr_example.rs) - ADSR envelope for synthesizers
- [Linear Example](examples/linear_example.rs) - Linear operations
- [Mix Example](examples/mix_example.rs) - Audio mixing utilities
- [Performance Example](examples/perf_example.rs) - Performance optimizations
- [Rates Example](examples/rates_example.rs) - Sample rate conversion
- [Spectral Example](examples/spectral_example.rs) - Spectral processing
- [Windows Example](examples/windows_example.rs) - Window functions

## Module Overview

### FFT (`fft` module)
Provides Fast Fourier Transform implementations optimized for different use cases:
- `SimpleFFT`: General purpose complex-to-complex FFT
- `SimpleRealFFT`: Optimized for real-valued inputs
- Support for non-power-of-2 sizes (factorizable into 2^a × 3^b)

### Filters (`filters` module)
Digital filter implementations:
- `Biquad`: Second-order filter section with various design methods
- Various filter types: lowpass, highpass, bandpass, notch, peaking, etc.
- Support for filter cascading and multi-channel processing

### Delay (`delay` module)
Delay line utilities:
- Various interpolation methods (nearest, linear, cubic)
- Single and multi-channel delay lines
- Buffer abstractions for efficient memory usage

### Spectral Processing (`spectral` module)
Frequency-domain processing tools:
- Magnitude/phase conversion utilities
- Phase vocoder techniques for pitch and time manipulation
- Frequency-domain filtering and manipulation

### STFT (`stft` module)
Short-time Fourier transform processing:
- Overlap-add processing framework
- Window function application
- Spectral processing utilities

### Stretch (`stretch` module)
Time stretching and pitch shifting:
- High-quality phase vocoder implementation using `SignalsmithStretch`
- Independent control of time and pitch
- Formant preservation and frequency mapping
- Real-time processing capabilities

### Mix (`mix` module)
Multichannel mixing utilities:
- Orthogonal matrices (Hadamard, Householder) for efficient mixing
- Stereo to multichannel conversion
- Energy-preserving crossfading

### Windows (`windows` module)
Window functions for spectral processing:
- Common window types (Hann, Hamming, Kaiser, etc.)
- Window design utilities
- Overlap handling and perfect reconstruction

### Envelopes (`envelopes` module)
Low-frequency oscillators and envelope generators:
- Precise control with minimal aliasing
- Multiple waveform types
- Configurable frequency and amplitude

### Linear (`linear` module)
Linear algebra utilities:
- Expression template system for efficient vector operations
- Optimized mathematical operations

### Curves (`curves` module)
Curve interpolation:
- Cubic curve interpolation with control over slope and curvature
- Smooth parameter transitions

### Rates (`rates` module)
Sample rate conversion:
- High-quality resampling algorithms
- Configurable quality settings

### Spacing (`spacing` module)
Provides a customizable room reverb effect:
- Multi-tap delay network simulating early reflections
- 3D source and receiver positioning
- Adjustable room size, damping, diffusion, bass, decay, and cross-mix
- Suitable for spatial audio and immersive effects

## Feature Flags

- `std` (default): Use the Rust standard library
- `alloc`: Enable allocation without std (for no_std environments)
- `wasm`: Enable WebAssembly bindings

## Testing

### Native Tests

Run the standard test suite:

```bash
cargo test
```

### WebAssembly Tests

The project includes comprehensive WASM tests using `wasm-bindgen-test`:

**Node.js Tests (recommended):**
```bash
# Build for Node.js
wasm-pack build --target nodejs --out-dir pkg --dev -- --features wasm

# Run tests
node test-adsr-wasm.js
```

**Browser Tests:**
```bash
wasm-pack test --headless --chrome --features wasm
```

See [ADSR_WASM_TESTING.md](ADSR_WASM_TESTING.md) for complete testing documentation.

## Performance

This library is designed with performance in mind and offers several optimizations:

- **SIMD Opportunities**: Code structure allows for SIMD optimizations where applicable
- **Cache-Friendly Algorithms**: Algorithms are designed to minimize cache misses
- **Minimal Allocations**: Operations avoid allocations during processing
- **Trade-offs**: Where appropriate, there are options to trade between quality and performance

For maximum performance:
- Use the largest practical buffer sizes for batch processing
- Reuse processor instances rather than creating new ones
- Consider using the `f32` type for most audio applications unless higher precision is needed
- Review the examples in `examples/perf_example.rs` for performance-critical applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original C++ library: [Signalsmith Audio DSP Library](https://github.com/signalsmith-audio/dsp)
- Signalsmith Audio's excellent [technical blog](https://signalsmith-audio.co.uk/writing/) with in-depth explanations of DSP concepts
- The comprehensive [design documentation](https://signalsmith-audio.co.uk/code/stretch/) for the time stretching algorithm
