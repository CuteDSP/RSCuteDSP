//! WebAssembly bindings for the DSP library
//!
//! This module provides WASM-compatible wrappers for the DSP functionality,
//! using concrete types (f32/f64) instead of generics.

use wasm_bindgen::prelude::*;
use num_complex::Complex;
use crate::fft::{SimpleFFT, SimpleRealFFT};
use crate::filters::{Biquad, BiquadDesign, StereoBiquad, FIR};
use crate::windows::{Kaiser, ApproximateConfinedGaussian};
use crate::delay::{Delay, InterpolatorLinear};
use crate::envelopes::{CubicLfo, BoxSum, BoxFilter, BoxStackFilter, PeakHold, PeakDecayLinear, Adsr};
use crate::stft::STFT;
use crate::curves::{Linear, Cubic, CubicSegmentCurve, Reciprocal};
use crate::rates;
use crate::mix::{Hadamard, Householder, StereoMultiMixer};
use crate::perf;
use crate::spacing::{Spacing, Position};
use crate::spectral::{WindowedFFT, SpectralProcessor};
use crate::stretch::SignalsmithStretch;
use crate::convolver::Convolver;
use crate::phase_rotation::{HilbertTransform, PhaseRotator};
use crate::common;

/// FFT functions for f32
#[wasm_bindgen]
pub struct WasmFFT {
    fft: SimpleFFT<f32>,
}

#[wasm_bindgen]
impl WasmFFT {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> WasmFFT {
        WasmFFT {
            fft: SimpleFFT::new(size),
        }
    }

    #[wasm_bindgen]
    pub fn fft_forward(&mut self, real: &[f32], imag: &[f32], out_real: &mut [f32], out_imag: &mut [f32]) {
        let input: Vec<Complex<f32>> = real.iter().zip(imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); input.len()];

        self.fft.fft(&input, &mut output);

        for (i, &c) in output.iter().enumerate() {
            out_real[i] = c.re;
            out_imag[i] = c.im;
        }
    }

    #[wasm_bindgen]
    pub fn fft_inverse(&mut self, real: &[f32], imag: &[f32], out_real: &mut [f32], out_imag: &mut [f32]) {
        let input: Vec<Complex<f32>> = real.iter().zip(imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); input.len()];

        self.fft.ifft(&input, &mut output);

        for (i, &c) in output.iter().enumerate() {
            out_real[i] = c.re;
            out_imag[i] = c.im;
        }
    }
}

/// Real FFT functions for f32
#[wasm_bindgen]
pub struct WasmRealFFT {
    fft: SimpleRealFFT<f32>,
}

#[wasm_bindgen]
impl WasmRealFFT {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> WasmRealFFT {
        WasmRealFFT {
            fft: SimpleRealFFT::new(size),
        }
    }

    #[wasm_bindgen]
    pub fn fft_forward(&mut self, input: &[f32], out_real: &mut [f32], out_imag: &mut [f32]) {
        let mut output = vec![Complex::new(0.0, 0.0); input.len() / 2 + 1];
        self.fft.fft(input, &mut output);

        for (i, &c) in output.iter().enumerate() {
            out_real[i] = c.re;
            out_imag[i] = c.im;
        }
    }

    #[wasm_bindgen]
    pub fn fft_inverse(&mut self, real: &[f32], imag: &[f32], output: &mut [f32]) {
        let input: Vec<Complex<f32>> = real.iter().zip(imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        self.fft.ifft(&input, output);
    }
}

/// Biquad filter for f32
#[wasm_bindgen]
pub struct WasmBiquad {
    filter: Biquad<f32>,
}

#[wasm_bindgen]
impl WasmBiquad {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmBiquad {
        WasmBiquad {
            filter: Biquad::new(true),
        }
    }

    #[wasm_bindgen]
    pub fn lowpass(&mut self, frequency: f32, q: f32, sample_rate: f32) {
        self.filter.lowpass(frequency / sample_rate, q, BiquadDesign::Cookbook);
    }

    #[wasm_bindgen]
    pub fn highpass(&mut self, frequency: f32, q: f32) {
        self.filter.highpass(frequency, q, BiquadDesign::Cookbook);
    }

    #[wasm_bindgen]
    pub fn bandpass(&mut self, frequency: f32, bandwidth_octaves: f32) {
        self.filter.bandpass(frequency, bandwidth_octaves);
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        for (i, &sample) in input.iter().enumerate() {
            output[i] = self.filter.process(sample);
        }
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.filter.reset();
    }
}

/// Stereo Biquad filter for f32
#[wasm_bindgen]
pub struct WasmStereoBiquad {
    filter: StereoBiquad<f32>,
}

#[wasm_bindgen]
impl WasmStereoBiquad {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmStereoBiquad {
        WasmStereoBiquad {
            filter: StereoBiquad::new(true), // Use cookbook bandwidth by default
        }
    }

    #[wasm_bindgen]
    pub fn lowpass(&mut self, frequency: f32, q: f32) {
        self.filter.lowpass(frequency, q, BiquadDesign::Cookbook);
    }

    #[wasm_bindgen]
    pub fn highpass(&mut self, frequency: f32, q: f32) {
        self.filter.highpass(frequency, q, BiquadDesign::Cookbook);
    }

    #[wasm_bindgen]
    pub fn bandpass(&mut self, frequency: f32, bandwidth_octaves: f32) {
        self.filter.bandpass(frequency, bandwidth_octaves);
    }

    #[wasm_bindgen]
    pub fn process(&mut self, left: &[f32], right: &[f32], out_left: &mut [f32], out_right: &mut [f32]) {
        for i in 0..left.len().min(right.len()).min(out_left.len()).min(out_right.len()) {
            let (l, r) = self.filter.process(left[i], right[i]);
            out_left[i] = l;
            out_right[i] = r;
        }
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.filter.reset();
    }
}

/// FIR filter for f32
#[wasm_bindgen]
pub struct WasmFIR {
    filter: FIR<f32>,
}

#[wasm_bindgen]
impl WasmFIR {
    #[wasm_bindgen(constructor)]
    pub fn new(coefficients: &[f32]) -> WasmFIR {
        WasmFIR {
            filter: FIR::new(coefficients.to_vec()),
        }
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        for (i, &sample) in input.iter().enumerate() {
            if i < output.len() {
                output[i] = self.filter.process(sample);
            }
        }
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.filter.reset();
    }
}

/// Kaiser window for f32
#[wasm_bindgen]
pub struct WasmKaiser {
    window: Kaiser<f32>,
}

#[wasm_bindgen]
impl WasmKaiser {
    #[wasm_bindgen(constructor)]
    pub fn new(beta: f32) -> WasmKaiser {
        WasmKaiser {
            window: Kaiser::new(beta),
        }
    }

    #[wasm_bindgen]
    pub fn with_bandwidth(bandwidth: f32) -> WasmKaiser {
        WasmKaiser {
            window: Kaiser::with_bandwidth(bandwidth, true),
        }
    }

    #[wasm_bindgen]
    pub fn fill(&self, data: &mut [f32]) {
        self.window.fill(data);
    }
}

/// Hann window for f32
#[wasm_bindgen]
pub struct WasmHann;

#[wasm_bindgen]
impl WasmHann {
    #[wasm_bindgen]
    pub fn fill(data: &mut [f32]) {
        let size = data.len() as f32;
        for (i, sample) in data.iter_mut().enumerate() {
            let n = i as f32;
            *sample = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n / (size - 1.0)).cos());
        }
    }
}

/// Hamming window for f32
#[wasm_bindgen]
pub struct WasmHamming;

#[wasm_bindgen]
impl WasmHamming {
    #[wasm_bindgen]
    pub fn fill(data: &mut [f32]) {
        let size = data.len() as f32;
        for (i, sample) in data.iter_mut().enumerate() {
            let n = i as f32;
            *sample = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * n / (size - 1.0)).cos();
        }
    }
}

/// Gaussian window for f32
#[wasm_bindgen]
pub struct WasmGaussian {
    window: ApproximateConfinedGaussian<f32>,
}

#[wasm_bindgen]
impl WasmGaussian {
    #[wasm_bindgen(constructor)]
    pub fn new(sigma: f32) -> WasmGaussian {
        WasmGaussian {
            window: ApproximateConfinedGaussian::new(sigma),
        }
    }

    #[wasm_bindgen]
    pub fn with_bandwidth(bandwidth: f32) -> WasmGaussian {
        WasmGaussian {
            window: ApproximateConfinedGaussian::with_bandwidth(bandwidth),
        }
    }

    #[wasm_bindgen]
    pub fn fill(&self, data: &mut [f32]) {
        self.window.fill(data);
    }
}

/// Window utilities
#[wasm_bindgen]
pub struct WasmWindowUtils;

#[wasm_bindgen]
impl WasmWindowUtils {
    /// Force perfect reconstruction for STFT windows
    #[wasm_bindgen]
    pub fn force_perfect_reconstruction(data: &mut [f32], window_length: usize, interval: usize) {
        crate::windows::force_perfect_reconstruction(data, window_length, interval);
    }
}

/// Delay effect for f32
#[wasm_bindgen]
pub struct WasmDelay {
    delay: Delay<f32, InterpolatorLinear<f32>>,
    delay_samples: f32,
    feedback: f32,
    sample_rate: f32,
}

#[wasm_bindgen]
impl WasmDelay {
    #[wasm_bindgen(constructor)]
    pub fn new(max_delay_seconds: f32, sample_rate: f32) -> WasmDelay {
        let max_samples = (max_delay_seconds * sample_rate) as usize;
        let mut delay = WasmDelay {
            delay: Delay::new(InterpolatorLinear::new(), max_samples),
            delay_samples: 0.3 * sample_rate, // default 300ms
            feedback: 0.4, // default feedback
            sample_rate,
        };
        delay.reset(); // Initialize with zeros
        delay
    }

    #[wasm_bindgen]
    pub fn set_delay_time(&mut self, delay_seconds: f32) {
        self.delay_samples = delay_seconds * self.sample_rate;
    }

    #[wasm_bindgen]
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback;
    }

    #[wasm_bindgen]
    pub fn process_buffer(&mut self, input: &[f32], output: &mut [f32]) {
        for i in 0..input.len() {
            // Read delayed sample
            let delayed = self.delay.read(self.delay_samples);
            // Mix input with feedback from delayed output
            let wet = input[i] + delayed * self.feedback;
            // Write to delay line
            self.delay.write(wet);
            // Output is the wet signal
            output[i] = wet;
        }
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.delay.reset(0.0);
    }
}

/// LFO (Low Frequency Oscillator) for f32
#[wasm_bindgen]
pub struct WasmLFO {
    lfo: CubicLfo,
}

#[wasm_bindgen]
impl WasmLFO {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmLFO {
        WasmLFO {
            lfo: CubicLfo::new(),
        }
    }

    #[wasm_bindgen]
    pub fn process(&mut self) -> f32 {
        self.lfo.next()
    }

    #[wasm_bindgen]
    pub fn set_frequency(&mut self, frequency: f32, sample_rate: f32) {
        // Convert frequency to rate (period in samples)
        // The LFO goes up and down in one period, so rate = sample_rate / (2 * frequency)
        let rate = sample_rate / (2.0 * frequency);
        self.lfo.set(-1.0, 1.0, rate, 0.0, 0.0); // low=-1, high=1, no variation
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.lfo.reset();
    }
}

/// ADSR envelope generator for synthesizer voices
#[wasm_bindgen]
pub struct WasmAdsr {
    adsr: Adsr,
}

#[wasm_bindgen]
impl WasmAdsr {
    /// Create a new ADSR envelope with the specified sample rate
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> WasmAdsr {
        WasmAdsr {
            adsr: Adsr::new(sample_rate),
        }
    }

    /// Set all ADSR parameters at once (attack, decay, sustain level, release in seconds)
    #[wasm_bindgen]
    pub fn set_times(&mut self, attack: f32, decay: f32, sustain: f32, release: f32) {
        self.adsr.set_times(attack, decay, sustain, release);
    }

    /// Set attack time in seconds
    #[wasm_bindgen]
    pub fn set_attack(&mut self, attack: f32) {
        self.adsr.set_attack(attack);
    }

    /// Set decay time in seconds
    #[wasm_bindgen]
    pub fn set_decay(&mut self, decay: f32) {
        self.adsr.set_decay(decay);
    }

    /// Set sustain level (0.0 to 1.0)
    #[wasm_bindgen]
    pub fn set_sustain(&mut self, sustain: f32) {
        self.adsr.set_sustain(sustain);
    }

    /// Set release time in seconds
    #[wasm_bindgen]
    pub fn set_release(&mut self, release: f32) {
        self.adsr.set_release(release);
    }

    /// Set gate (true = note on, false = note off)
    #[wasm_bindgen]
    pub fn gate(&mut self, gate: bool) {
        self.adsr.gate(gate);
    }

    /// Process and return the next envelope sample
    #[wasm_bindgen]
    pub fn next(&mut self) -> f32 {
        self.adsr.next()
    }

    /// Get current value without advancing
    #[wasm_bindgen]
    pub fn value(&self) -> f32 {
        self.adsr.value()
    }

    /// Get current stage (0=Attack, 1=Decay, 2=Sustain, 3=Release, 4=Done)
    #[wasm_bindgen]
    pub fn stage(&self) -> u32 {
        self.adsr.stage()
    }

    /// Reset the envelope to zero
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.adsr.reset();
    }

    /// Process a buffer of samples, applying the envelope to each
    #[wasm_bindgen]
    pub fn process_buffer(&mut self, input: &[f32], output: &mut [f32]) {
        for i in 0..input.len().min(output.len()) {
            output[i] = input[i] * self.adsr.next();
        }
    }
}

/// Box filter for envelope smoothing
#[wasm_bindgen]
pub struct WasmBoxFilter {
    filter: BoxFilter<f32>,
}

#[wasm_bindgen]
impl WasmBoxFilter {
    #[wasm_bindgen(constructor)]
    pub fn new(length: usize) -> WasmBoxFilter {
        WasmBoxFilter {
            filter: BoxFilter::new(length),
        }
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: f32) -> f32 {
        self.filter.process(input)
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.filter.reset(0.0);
    }

    #[wasm_bindgen]
    pub fn set_length(&mut self, length: usize) {
        // BoxFilter doesn't have set_length, so we'll recreate it
        *self = WasmBoxFilter::new(length);
    }
}

/// Peak hold for envelope detection
#[wasm_bindgen]
pub struct WasmPeakHold {
    peak_hold: PeakHold<f32>,
}

#[wasm_bindgen]
impl WasmPeakHold {
    #[wasm_bindgen(constructor)]
    pub fn new(hold_samples: usize) -> WasmPeakHold {
        WasmPeakHold {
            peak_hold: PeakHold::new(hold_samples),
        }
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: f32) -> f32 {
        self.peak_hold.process(input)
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.peak_hold.reset();
    }
}

/// Box sum for moving averages
#[wasm_bindgen]
pub struct WasmBoxSum {
    box_sum: BoxSum<f32>,
    max_length: usize,
}

#[wasm_bindgen]
impl WasmBoxSum {
    #[wasm_bindgen(constructor)]
    pub fn new(max_length: usize) -> WasmBoxSum {
        WasmBoxSum {
            box_sum: BoxSum::new(max_length),
            max_length,
        }
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: f32, width: usize) -> f32 {
        self.box_sum.read_write(input, width)
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.box_sum.reset(0.0);
    }

    #[wasm_bindgen]
    pub fn resize(&mut self, max_length: usize) {
        self.box_sum.resize(max_length);
        self.max_length = max_length;
    }

    #[wasm_bindgen]
    pub fn length(&self) -> usize {
        self.max_length
    }
}

/// Box stack filter for multi-layer smoothing
#[wasm_bindgen]
pub struct WasmBoxStackFilter {
    filter: BoxStackFilter<f32>,
}

#[wasm_bindgen]
impl WasmBoxStackFilter {
    #[wasm_bindgen(constructor)]
    pub fn new(max_size: usize, layers: usize) -> WasmBoxStackFilter {
        WasmBoxStackFilter {
            filter: BoxStackFilter::new(max_size, layers),
        }
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: f32) -> f32 {
        self.filter.process(input)
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.filter.reset();
    }

    #[wasm_bindgen]
    pub fn resize(&mut self, max_size: usize, layers: usize) {
        self.filter.resize(max_size, layers);
    }

    #[wasm_bindgen]
    pub fn optimal_ratios(layers: usize) -> Vec<f32> {
        BoxStackFilter::<f32>::optimal_ratios(layers)
    }
}

/// Peak decay with linear decay
#[wasm_bindgen]
pub struct WasmPeakDecayLinear {
    filter: PeakDecayLinear<f32>,
}

#[wasm_bindgen]
impl WasmPeakDecayLinear {
    #[wasm_bindgen(constructor)]
    pub fn new(max_length: usize) -> WasmPeakDecayLinear {
        WasmPeakDecayLinear {
            filter: PeakDecayLinear::new(max_length),
        }
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: f32) -> f32 {
        self.filter.process(input)
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.filter.reset();
    }

    #[wasm_bindgen]
    pub fn resize(&mut self, max_length: usize) {
        self.filter.resize(max_length);
    }

    #[wasm_bindgen]
    pub fn set_length(&mut self, length: f32) {
        self.filter.set(length);
    }
}

/// STFT (Short-time Fourier Transform) for f32
#[wasm_bindgen]
pub struct WasmSTFT {
    stft: STFT<f32>,
}

#[wasm_bindgen]
impl WasmSTFT {
    #[wasm_bindgen(constructor)]
    pub fn new(modified: bool) -> WasmSTFT {
        WasmSTFT {
            stft: STFT::new(modified),
        }
    }

    #[wasm_bindgen]
    pub fn configure(&mut self, in_channels: usize, out_channels: usize, block_samples: usize) {
        self.stft.configure(in_channels, out_channels, block_samples, 0, 0);
    }

    #[wasm_bindgen]
    pub fn process_block_to_spectrum(&mut self, channel: usize, input: &[f32]) -> Vec<f32> {
        self.stft.write_input_simple(channel, input);
        let spectrum = self.stft.process_block_to_spectrum(channel);
        let mut result = Vec::with_capacity(spectrum.len() * 2);
        for &c in spectrum {
            result.push(c.re);
            result.push(c.im);
        }
        result
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.stft.reset_default();
    }
}

/// Linear curve mapping for f32
#[wasm_bindgen]
pub struct WasmLinearCurve {
    curve: Linear<f32>,
}

#[wasm_bindgen]
impl WasmLinearCurve {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmLinearCurve {
        WasmLinearCurve {
            curve: Linear::new(),
        }
    }

    #[wasm_bindgen]
    pub fn from_points(x0: f32, x1: f32, y0: f32, y1: f32) -> WasmLinearCurve {
        WasmLinearCurve {
            curve: Linear::from_points(x0, x1, y0, y1),
        }
    }

    #[wasm_bindgen]
    pub fn with_values(a0: f32, a1: f32) -> WasmLinearCurve {
        WasmLinearCurve {
            curve: Linear::with_values(a0, a1),
        }
    }

    #[wasm_bindgen]
    pub fn evaluate(&self, x: f32) -> f32 {
        self.curve.evaluate(x)
    }

    #[wasm_bindgen]
    pub fn derivative(&self) -> f32 {
        self.curve.derivative()
    }
}

/// Cubic curve mapping for f32
#[wasm_bindgen]
pub struct WasmCubicCurve {
    curve: Cubic<f32>,
}

#[wasm_bindgen]
impl WasmCubicCurve {
    #[wasm_bindgen(constructor)]
    pub fn new(x_start: f32, a0: f32, a1: f32, a2: f32, a3: f32) -> WasmCubicCurve {
        WasmCubicCurve {
            curve: Cubic::new(x_start, a0, a1, a2, a3),
        }
    }

    #[wasm_bindgen]
    pub fn evaluate(&self, x: f32) -> f32 {
        self.curve.evaluate(x)
    }

    #[wasm_bindgen]
    pub fn start(&self) -> f32 {
        self.curve.start()
    }

    /// Create a cubic curve using Hermite interpolation
    #[wasm_bindgen]
    pub fn hermite(x0: f32, x1: f32, y0: f32, y1: f32, g0: f32, g1: f32) -> WasmCubicCurve {
        WasmCubicCurve {
            curve: Cubic::hermite(x0, x1, y0, y1, g0, g1),
        }
    }
}

/// Cubic segment curve for f32
#[wasm_bindgen]
pub struct WasmCubicSegmentCurve {
    curve: CubicSegmentCurve<f32>,
}

#[wasm_bindgen]
impl WasmCubicSegmentCurve {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmCubicSegmentCurve {
        WasmCubicSegmentCurve {
            curve: CubicSegmentCurve::new(),
        }
    }

    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.curve.clear();
    }

    #[wasm_bindgen]
    pub fn add_point(&mut self, x: f32, y: f32, corner: bool) {
        self.curve.add(x, y, corner);
    }

    #[wasm_bindgen]
    pub fn calculate(&mut self) {
        self.curve.update(false, false, 0.0);
    }

    #[wasm_bindgen]
    pub fn evaluate(&self, x: f32) -> f32 {
        self.curve.evaluate(x)
    }

    #[wasm_bindgen]
    pub fn set_gradients(&mut self, low_grad: f32, high_grad: f32) {
        self.curve.low_grad = low_grad;
        self.curve.high_grad = high_grad;
    }
}

/// Reciprocal curve mapping for f32
#[wasm_bindgen]
pub struct WasmReciprocalCurve {
    curve: Reciprocal<f32>,
}

#[wasm_bindgen]
impl WasmReciprocalCurve {
    #[wasm_bindgen(constructor)]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> WasmReciprocalCurve {
        WasmReciprocalCurve {
            curve: Reciprocal::new(a, b, c, d),
        }
    }

    #[wasm_bindgen]
    pub fn from_points(x0: f32, x1: f32, x2: f32, y0: f32, y1: f32, y2: f32) -> WasmReciprocalCurve {
        WasmReciprocalCurve {
            curve: Reciprocal::from_points(x0, x1, x2, y0, y1, y2),
        }
    }

    #[wasm_bindgen]
    pub fn from_y_values(y0: f32, y1: f32, y2: f32) -> WasmReciprocalCurve {
        WasmReciprocalCurve {
            curve: Reciprocal::from_y_values(y0, y1, y2),
        }
    }

    #[wasm_bindgen]
    pub fn bark_scale() -> WasmReciprocalCurve {
        WasmReciprocalCurve {
            curve: Reciprocal::bark_scale(),
        }
    }

    #[wasm_bindgen]
    pub fn bark_range(low_hz: f32, high_hz: f32) -> WasmReciprocalCurve {
        WasmReciprocalCurve {
            curve: Reciprocal::bark_range(low_hz, high_hz),
        }
    }

    #[wasm_bindgen]
    pub fn evaluate(&self, x: f32) -> f32 {
        self.curve.evaluate(x)
    }
}

/// Sample rate conversion utilities
#[wasm_bindgen]
pub struct WasmSampleRateConverter;

#[wasm_bindgen]
impl WasmSampleRateConverter {
    /// Fill buffer with Kaiser-windowed sinc FIR lowpass filter
    #[wasm_bindgen]
    pub fn fill_kaiser_sinc_filter(data: &mut [f32], pass_freq: f32, stop_freq: f32) {
        rates::fill_kaiser_sinc(data, pass_freq, stop_freq);
    }

    /// Fill buffer with Kaiser-windowed sinc FIR lowpass filter using centre frequency
    #[wasm_bindgen]
    pub fn fill_kaiser_sinc_filter_centre(data: &mut [f32], centre_freq: f32) {
        rates::fill_kaiser_sinc_with_centre(data, centre_freq);
    }
}

/// Spectral processing utilities
#[wasm_bindgen]
pub struct WasmSpectralUtils;

#[wasm_bindgen]
impl WasmSpectralUtils {
    /// Convert magnitude and phase to complex number
    #[wasm_bindgen]
    pub fn mag_phase_to_complex(mag: f32, phase: f32) -> Vec<f32> {
        let complex = crate::spectral::utils::mag_phase_to_complex(mag, phase);
        vec![complex.re, complex.im]
    }

    /// Convert complex number to magnitude and phase
    #[wasm_bindgen]
    pub fn complex_to_mag_phase(real: f32, imag: f32) -> Vec<f32> {
        let (mag, phase) = crate::spectral::utils::complex_to_mag_phase(
            Complex::new(real, imag)
        );
        vec![mag, phase]
    }

    /// Convert linear amplitude to decibels
    #[wasm_bindgen]
    pub fn linear_to_db(linear: f32) -> f32 {
        crate::spectral::utils::linear_to_db(linear)
    }

    /// Convert decibels to linear amplitude
    #[wasm_bindgen]
    pub fn db_to_linear(db: f32) -> f32 {
        crate::spectral::utils::db_to_linear(db)
    }

    /// Find the smallest size >= input that is efficient for FFT
    #[wasm_bindgen]
    pub fn fast_size_above(size: usize, divisor: usize) -> usize {
        WindowedFFT::<f32>::fast_size_above(size, divisor)
    }

    /// Find the largest size <= input that is efficient for FFT
    #[wasm_bindgen]
    pub fn fast_size_below(size: usize, divisor: usize) -> usize {
        WindowedFFT::<f32>::fast_size_below(size, divisor)
    }
}

/// Windowed FFT for spectral processing
#[wasm_bindgen]
pub struct WasmWindowedFFT {
    fft: WindowedFFT<f32>,
}

#[wasm_bindgen]
impl WasmWindowedFFT {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize, rotate_samples: usize) -> WasmWindowedFFT {
        WasmWindowedFFT {
            fft: WindowedFFT::new(size, rotate_samples),
        }
    }

    #[wasm_bindgen]
    pub fn set_size(&mut self, size: usize, rotate_samples: usize) {
        self.fft.set_size(size, rotate_samples);
    }

    #[wasm_bindgen]
    pub fn fft_forward(&mut self, input: &[f32], output: &mut [f32], with_window: bool, with_scaling: bool) {
        // Simplified interface for WASM - just copy input to output for now
        // The actual WindowedFFT API is quite complex for WASM binding
        let len = input.len().min(output.len());
        output[..len].copy_from_slice(&input[..len]);
    }

    #[wasm_bindgen]
    pub fn size(&self) -> usize {
        self.fft.size()
    }
}

/// Spectral processor for real-time spectral effects
#[wasm_bindgen]
pub struct WasmSpectralProcessor {
    processor: SpectralProcessor<f32>,
}

#[wasm_bindgen]
impl WasmSpectralProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(fft_size: usize, overlap: usize) -> WasmSpectralProcessor {
        WasmSpectralProcessor {
            processor: SpectralProcessor::new(fft_size, overlap),
        }
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.processor.reset();
    }

    #[wasm_bindgen]
    pub fn fft_size(&self) -> usize {
        self.processor.fft_size()
    }

    #[wasm_bindgen]
    pub fn hop_size(&self) -> usize {
        self.processor.hop_size()
    }

    #[wasm_bindgen]
    pub fn overlap(&self) -> usize {
        self.processor.overlap()
    }
}

/// Multi-channel mixing utilities
#[wasm_bindgen]
pub struct WasmHadamardMixer {
    mixer: Hadamard<f32>,
}

#[wasm_bindgen]
impl WasmHadamardMixer {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> WasmHadamardMixer {
        WasmHadamardMixer {
            mixer: Hadamard::new(size),
        }
    }

    #[wasm_bindgen]
    pub fn mix_in_place(&self, data: &mut [f32]) {
        self.mixer.in_place(data);
    }

    #[wasm_bindgen]
    pub fn scaling_factor(&self) -> f32 {
        self.mixer.scaling_factor()
    }
}

#[wasm_bindgen]
pub struct WasmHouseholderMixer {
    mixer: Householder<f32>,
}

#[wasm_bindgen]
impl WasmHouseholderMixer {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> WasmHouseholderMixer {
        WasmHouseholderMixer {
            mixer: Householder::new(size),
        }
    }

    #[wasm_bindgen]
    pub fn mix_in_place(&self, data: &mut [f32]) {
        self.mixer.in_place(data);
    }
}

/// Stereo to multi-channel mixer
#[wasm_bindgen]
pub struct WasmStereoMultiMixer {
    mixer: StereoMultiMixer<f32>,
}

#[wasm_bindgen]
impl WasmStereoMultiMixer {
    #[wasm_bindgen(constructor)]
    pub fn new(channels: usize) -> WasmStereoMultiMixer {
        WasmStereoMultiMixer {
            mixer: StereoMultiMixer::new(channels),
        }
    }

    #[wasm_bindgen]
    pub fn stereo_to_multi(&self, left: f32, right: f32, output: &mut [f32]) {
        let input = [left, right];
        self.mixer.stereo_to_multi(&input, output);
    }

    #[wasm_bindgen]
    pub fn multi_to_stereo(&self, input: &[f32]) -> Vec<f32> {
        let mut output = [0.0f32, 0.0f32];
        self.mixer.multi_to_stereo(input, &mut output);
        vec![output[0], output[1]]
    }

    #[wasm_bindgen]
    pub fn scaling_factor1(&self) -> f32 {
        self.mixer.scaling_factor1()
    }

    #[wasm_bindgen]
    pub fn scaling_factor2(&self) -> f32 {
        self.mixer.scaling_factor2()
    }
}

/// Mixing utilities
#[wasm_bindgen]
pub struct WasmMixUtils;

#[wasm_bindgen]
impl WasmMixUtils {
    /// Cheap energy-preserving crossfade
    #[wasm_bindgen]
    pub fn cheap_energy_crossfade(x: f32) -> Vec<f32> {
        let mut to_coeff = 0.0f32;
        let mut from_coeff = 0.0f32;
        crate::mix::cheap_energy_crossfade(x, &mut to_coeff, &mut from_coeff);
        vec![to_coeff, from_coeff]
    }
}

/// Performance utilities
#[wasm_bindgen]
pub struct WasmPerfUtils;

#[wasm_bindgen]
impl WasmPerfUtils {
    /// Fast complex multiplication (real part, imag part)
    #[wasm_bindgen]
    pub fn complex_mul(ar: f32, ai: f32, br: f32, bi: f32) -> Vec<f32> {
        let a = Complex::new(ar, ai);
        let b = Complex::new(br, bi);
        let result = perf::mul(a, b);
        vec![result.re, result.im]
    }

    /// Fast complex multiplication with conjugate
    #[wasm_bindgen]
    pub fn complex_mul_conj(ar: f32, ai: f32, br: f32, bi: f32) -> Vec<f32> {
        let a = Complex::new(ar, ai);
        let b = Complex::new(br, bi);
        let result = perf::mul_conj(a, b);
        vec![result.re, result.im]
    }
}

/// Room spacing/reverb effect
#[wasm_bindgen]
pub struct WasmRoomSpacing {
    spacing: Spacing<f32>,
}

#[wasm_bindgen]
impl WasmRoomSpacing {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> WasmRoomSpacing {
        WasmRoomSpacing {
            spacing: Spacing::new(sample_rate),
        }
    }

    #[wasm_bindgen]
    pub fn add_source(&mut self, x: f32, y: f32, z: f32) {
        self.spacing.sources.push(Position { x, y, z });
    }

    #[wasm_bindgen]
    pub fn add_receiver(&mut self, x: f32, y: f32, z: f32) {
        self.spacing.receivers.push(Position { x, y, z });
    }

    #[wasm_bindgen]
    pub fn set_room_size(&mut self, size: f32) {
        self.spacing.set_room_size(size);
    }

    #[wasm_bindgen]
    pub fn set_damping(&mut self, damping: f32) {
        self.spacing.set_damping(damping);
    }

    #[wasm_bindgen]
    pub fn set_diffusion(&mut self, diffusion: f32) {
        self.spacing.set_diff(diffusion);
    }

    #[wasm_bindgen]
    pub fn set_bass(&mut self, bass_db: f32) {
        self.spacing.set_bass(bass_db);
    }

    #[wasm_bindgen]
    pub fn set_decay(&mut self, decay: f32) {
        self.spacing.set_decay(decay);
    }

    #[wasm_bindgen]
    pub fn set_cross_mix(&mut self, cross: f32) {
        self.spacing.set_cross(cross);
    }

    #[wasm_bindgen]
    pub fn prepare(&mut self) {
        // No preparation needed for Spacing
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let inputs = vec![input];
        let mut outputs = vec![vec![0.0f32; output.len()]];
        self.spacing.process(&inputs.iter().map(|x| *x).collect::<Vec<_>>(), &mut outputs);
        output.copy_from_slice(&outputs[0]);
        output.len()
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        // Reset by clearing paths and delays
        self.spacing.clear_paths();
    }
}

/// Time stretching and pitch shifting
#[wasm_bindgen]
pub struct WasmTimeStretch {
    stretcher: SignalsmithStretch<f32>,
    channels: usize,
    block_samples: usize,
    interval_samples: usize,
    split_computation: bool,
}

#[wasm_bindgen]
impl WasmTimeStretch {
    #[wasm_bindgen(constructor)]
    pub fn new(channels: usize, block_size: usize, interval: usize) -> WasmTimeStretch {
        let mut stretcher = SignalsmithStretch::new();
        stretcher.configure(channels, block_size, interval, false);
        WasmTimeStretch { 
            stretcher,
            channels,
            block_samples: block_size,
            interval_samples: interval,
            split_computation: false,
        }
    }

    #[wasm_bindgen]
    pub fn set_stretch_factor(&mut self, factor: f32) {
        // For time stretching, we adjust the interval_samples
        // factor > 1.0 means slower (longer), factor < 1.0 means faster (shorter)
        let new_interval = (self.interval_samples as f32 / factor) as usize;
        self.stretcher.configure(self.channels, self.block_samples, new_interval, self.split_computation);
        self.interval_samples = new_interval;
    }

    #[wasm_bindgen]
    pub fn set_pitch_shift_semitones(&mut self, semitones: f32) {
        self.stretcher.set_transpose_semitones(semitones, 0.5);
    }

    #[wasm_bindgen]
    pub fn set_formant_shift_semitones(&mut self, semitones: f32) {
        self.stretcher.set_formant_semitones(semitones, false);
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32], input_samples: usize, output: &mut [f32], output_samples: usize) -> usize {
        let inputs = vec![input.to_vec()];
        let mut outputs = vec![vec![0.0f32; output_samples]];
        self.stretcher.process(&inputs, input_samples, &mut outputs, output_samples);
        let copy_len = output.len().min(outputs[0].len());
        output[..copy_len].copy_from_slice(&outputs[0][..copy_len]);
        copy_len
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.stretcher.reset();
    }

    #[wasm_bindgen]
    pub fn input_latency(&self) -> usize {
        self.stretcher.input_latency()
    }

    #[wasm_bindgen]
    pub fn output_latency(&self) -> usize {
        self.stretcher.output_latency()
    }
}

/// Convolver with Impulse Response support for WASM
#[wasm_bindgen]
pub struct WasmConvolver {
    convolver: Convolver<f32>,
}

#[wasm_bindgen]
impl WasmConvolver {
    #[wasm_bindgen(constructor)]
    pub fn new(ir: &[f32]) -> WasmConvolver {
        WasmConvolver {
            convolver: Convolver::new(ir.to_vec()),
        }
    }

    #[wasm_bindgen]
    pub fn process(&mut self, sample: f32) -> f32 {
        self.convolver.process(sample)
    }

    #[wasm_bindgen]
    pub fn process_block(&mut self, input: &[f32]) -> Vec<f32> {
        self.convolver.process_block(input)
    }

    #[wasm_bindgen]
    pub fn set_ir(&mut self, ir: &[f32]) {
        self.convolver.set_ir(ir.to_vec());
    }

    #[wasm_bindgen]
    pub fn ir_len(&self) -> usize {
        self.convolver.ir_len()
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.convolver.reset();
    }

    #[wasm_bindgen]
    pub fn get_ir(&self) -> Vec<f32> {
        self.convolver.get_ir().to_vec()
    }
}

/// Hilbert Transform for WASM
#[wasm_bindgen]
pub struct WasmHilbertTransform {
    hilbert: HilbertTransform<f32>,
}

#[wasm_bindgen]
impl WasmHilbertTransform {
    #[wasm_bindgen(constructor)]
    pub fn new(fft_size: usize) -> WasmHilbertTransform {
        WasmHilbertTransform {
            hilbert: HilbertTransform::new(fft_size),
        }
    }

    #[wasm_bindgen]
    pub fn transform(&mut self, signal: &[f32]) -> Vec<f32> {
        self.hilbert.transform(signal)
    }

    /// Get analytic signal (returns interleaved real/imaginary pairs)
    #[wasm_bindgen]
    pub fn analytic_signal(&mut self, signal: &[f32]) -> Vec<f32> {
        let analytic = self.hilbert.analytic_signal(signal);
        let mut result = Vec::with_capacity(analytic.len() * 2);
        for c in analytic {
            result.push(c.re);
            result.push(c.im);
        }
        result
    }
}

/// Phase Rotator for WASM
#[wasm_bindgen]
pub struct WasmPhaseRotator {
    rotator: PhaseRotator<f32>,
}

#[wasm_bindgen]
impl WasmPhaseRotator {
    #[wasm_bindgen(constructor)]
    pub fn new(frequency: f32, sample_rate: f32) -> WasmPhaseRotator {
        WasmPhaseRotator {
            rotator: PhaseRotator::new(frequency, sample_rate),
        }
    }

    #[wasm_bindgen]
    pub fn process(&mut self, sample: f32) -> f32 {
        self.rotator.process(sample)
    }

    /// Get quadrature output (returns [I, Q] pair)
    #[wasm_bindgen]
    pub fn process_quadrature(&mut self, sample: f32) -> Vec<f32> {
        let (i, q) = self.rotator.process_quadrature(sample);
        vec![i, q]
    }

    #[wasm_bindgen]
    pub fn process_block(&mut self, input: &[f32]) -> Vec<f32> {
        self.rotator.process_block(input)
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.rotator.reset();
    }

    #[wasm_bindgen]
    pub fn get_phase(&self) -> f32 {
        self.rotator.get_phase()
    }

    #[wasm_bindgen]
    pub fn set_phase(&mut self, phase: f32) {
        self.rotator.set_phase(phase);
    }

    #[wasm_bindgen]
    pub fn set_frequency(&mut self, frequency: f32, sample_rate: f32) {
        self.rotator.set_frequency(frequency, sample_rate);
    }
}

/// Analysis functions for phase rotation
#[wasm_bindgen]
pub struct WasmPhaseAnalysis;

#[wasm_bindgen]
impl WasmPhaseAnalysis {
    /// Compute instantaneous phase from interleaved complex signal
    /// Input: [r0, i0, r1, i1, ...]
    #[wasm_bindgen]
    pub fn instantaneous_phase(interleaved_complex: &[f32]) -> Vec<f32> {
        let analytic: Vec<Complex<f32>> = interleaved_complex
            .chunks(2)
            .map(|c| Complex::new(c[0], c[1]))
            .collect();
        crate::phase_rotation::instantaneous_phase(&analytic)
    }

    /// Compute instantaneous magnitude from interleaved complex signal
    /// Input: [r0, i0, r1, i1, ...]
    #[wasm_bindgen]
    pub fn instantaneous_magnitude(interleaved_complex: &[f32]) -> Vec<f32> {
        let analytic: Vec<Complex<f32>> = interleaved_complex
            .chunks(2)
            .map(|c| Complex::new(c[0], c[1]))
            .collect();
        crate::phase_rotation::instantaneous_magnitude(&analytic)
    }

    /// Compute instantaneous frequency from phase
    #[wasm_bindgen]
    pub fn instantaneous_frequency(phase: &[f32], sample_rate: f32) -> Vec<f32> {
        crate::phase_rotation::instantaneous_frequency(phase, sample_rate)
    }
}

/// Common utilities and version information
#[wasm_bindgen]
pub struct WasmCommon;

#[wasm_bindgen]
impl WasmCommon {
    /// Get the major version number
    #[wasm_bindgen]
    pub fn version_major() -> u32 {
        common::VERSION_MAJOR
    }

    /// Get the minor version number
    #[wasm_bindgen]
    pub fn version_minor() -> u32 {
        common::VERSION_MINOR
    }

    /// Get the patch version number
    #[wasm_bindgen]
    pub fn version_patch() -> u32 {
        common::VERSION_PATCH
    }

    /// Get the full version string
    #[wasm_bindgen]
    pub fn version_string() -> String {
        common::VERSION_STRING.to_string()
    }

    /// Check if a version is compatible
    #[wasm_bindgen]
    pub fn version_check(major: u32, minor: u32, patch: u32) -> bool {
        common::version_check(major, minor, patch)
    }
}