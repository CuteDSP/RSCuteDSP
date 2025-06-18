//! Short-Time Fourier Transform implementation
//!
//! This module provides a self-normalizing STFT implementation with variable
//! position/window for output blocks.

#![allow(unused_imports)]

#[cfg(feature = "std")]
use std::{f32::consts::PI, vec::Vec, marker::PhantomData};

#[cfg(not(feature = "std"))]
use core::{f32::consts::PI, marker::PhantomData};

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

use num_complex::Complex;
use num_traits::Float;
use num_traits::FromPrimitive;

use crate::fft;
use crate::windows;

/// Window shape for STFT
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowShape {
    /// Ignore window shape (use rectangular window)
    Ignore,
    /// Approximate Confined Gaussian window
    ACG,
    /// Kaiser window
    Kaiser,
}

/// A self-normalizing STFT, with variable position/window for output blocks
pub struct STFT<T: Float> {
    // FFT implementation
    fft: fft::Pow2RealFFT<T>,

    // Configuration
    analysis_channels: usize,
    synthesis_channels: usize,
    block_samples: usize,
    fft_samples: usize,
    fft_bins: usize,
    input_length_samples: usize,
    default_interval: usize,

    // Windows
    analysis_window: Vec<T>,
    synthesis_window: Vec<T>,
    analysis_offset: usize,
    synthesis_offset: usize,

    // Buffers
    input_buffer: Vec<T>,
    input_pos: usize,
    output_buffer: Vec<T>,
    output_pos: usize,
    window_products: Vec<T>,
    spectrum_buffer: Vec<Complex<T>>,
    time_buffer: Vec<T>,

    // Constants
    almost_zero: T,
    modified: bool,
}

#[cfg(feature = "std")]
use std::ops::AddAssign;

#[cfg(not(feature = "std"))]
use core::ops::AddAssign;

impl<T: Float + From<f32> + AddAssign + FromPrimitive> STFT<T> {
    /// Create a new STFT instance
    pub fn new(modified: bool) -> Self {
        Self {
            fft: fft::Pow2RealFFT::new(0),
            analysis_channels: 0,
            synthesis_channels: 0,
            block_samples: 0,
            fft_samples: 0,
            fft_bins: 0,
            input_length_samples: 0,
            default_interval: 0,
            analysis_window: Vec::new(),
            synthesis_window: Vec::new(),
            analysis_offset: 0,
            synthesis_offset: 0,
            input_buffer: Vec::new(),
            input_pos: 0,
            output_buffer: Vec::new(),
            output_pos: 0,
            window_products: Vec::new(),
            spectrum_buffer: Vec::new(),
            time_buffer: Vec::new(),
            almost_zero: <T as From<f32>>::from(1e-20),
            modified,
        }
    }

    /// Configure the STFT
    pub fn configure(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        block_samples: usize,
        extra_input_history: usize,
        interval_samples: usize,
    ) {
        self.analysis_channels = in_channels;
        self.synthesis_channels = out_channels;
        self.block_samples = block_samples;

        // Calculate FFT size (power of 2 >= block_samples)
        let mut fft_samples = 1;
        while fft_samples < block_samples {
            fft_samples *= 2;
        }
        self.fft_samples = fft_samples;
        self.fft.resize(fft_samples);
        self.fft_bins = fft_samples / 2 + 1; // For real FFT

        self.input_length_samples = block_samples + extra_input_history;
        self.input_buffer.resize(self.input_length_samples * in_channels, T::zero());

        self.output_buffer.resize(block_samples * out_channels, T::zero());
        self.window_products.resize(block_samples, T::zero());
        self.spectrum_buffer.resize(self.fft_bins * in_channels.max(out_channels), Complex::new(T::zero(), T::zero()));
        self.time_buffer.resize(fft_samples, T::zero());

        self.analysis_window.resize(block_samples, T::zero());
        self.synthesis_window.resize(block_samples, T::zero());

        // Set default interval if not specified
        let interval = if interval_samples > 0 {
            interval_samples
        } else {
            block_samples / 4
        };
        self.set_interval(interval, WindowShape::ACG);

        self.reset_default();
    }

    /// Get the block size in samples
    pub fn block_samples(&self) -> usize {
        self.block_samples
    }

    /// Get the FFT size in samples
    pub fn fft_samples(&self) -> usize {
        self.fft_samples
    }

    /// Get the default interval between blocks
    pub fn default_interval(&self) -> usize {
        self.default_interval
    }

    /// Get the number of frequency bands
    pub fn bands(&self) -> usize {
        self.fft_bins
    }

    /// Get the analysis latency
    pub fn analysis_latency(&self) -> usize {
        self.block_samples - self.analysis_offset
    }

    /// Get the synthesis latency
    pub fn synthesis_latency(&self) -> usize {
        self.synthesis_offset
    }

    /// Get the total latency
    pub fn latency(&self) -> usize {
        self.synthesis_latency() + self.analysis_latency()
    }

    /// Convert bin index to frequency
    pub fn bin_to_freq(&self, bin: T) -> T {
        if self.modified {
            (bin + <T as From<f32>>::from(0.5)) / <T as From<f32>>::from(self.fft_samples as f32)
        } else {
            bin / <T as From<f32>>::from(self.fft_samples as f32)
        }
    }

    /// Convert frequency to bin index
    pub fn freq_to_bin(&self, freq: T) -> T {
        if self.modified {
            freq * <T as From<f32>>::from(self.fft_samples as f32) - <T as From<f32>>::from(0.5)
        } else {
            freq * <T as From<f32>>::from(self.fft_samples as f32)
        }
    }

    /// Reset the STFT state
    pub fn reset(&mut self, product_weight: T) {
        self.input_pos = self.block_samples;
        self.output_pos = 0;

        // Clear buffers
        for v in &mut self.input_buffer {
            *v = T::zero();
        }
        for v in &mut self.output_buffer {
            *v = T::zero();
        }
        for v in &mut self.spectrum_buffer {
            *v = Complex::new(T::zero(), T::zero());
        }
        for v in &mut self.window_products {
            *v = T::zero();
        }

        // Initialize window products
        self.add_window_product();

        // Accumulate window products for overlapping windows
        for i in (0..self.block_samples - self.default_interval).rev() {
            self.window_products[i] = self.window_products[i] + self.window_products[i + self.default_interval];
        }

        // Scale window products
        for v in &mut self.window_products {
            *v = *v * product_weight + self.almost_zero;
        }

        // Move output position to be ready for first block
        self.move_output(self.default_interval);
    }

    /// Reset the STFT state with default product weight
    pub fn reset_default(&mut self) {
        self.reset(T::one());
    }

    /// Write input samples to a specific channel
    pub fn write_input(&mut self, channel: usize, offset: usize, length: usize, input_array: &[T]) {
        assert!(channel < self.analysis_channels, "Channel index out of bounds");
        assert!(offset + length <= input_array.len(), "Input array too small");

        let buffer_start = channel * self.input_length_samples;
        let offset_pos = (self.input_pos + offset) % self.input_length_samples;

        // Handle wrapping around the circular buffer
        let input_wrap_index = self.input_length_samples - offset_pos;
        let chunk1 = length.min(input_wrap_index);

        // Copy first chunk (before wrap)
        for i in 0..chunk1 {
            let buffer_index = buffer_start + offset_pos + i;
            self.input_buffer[buffer_index] = input_array[i];
        }

        // Copy second chunk (after wrap)
        for i in chunk1..length {
            let buffer_index = buffer_start + i + offset_pos - self.input_length_samples;
            self.input_buffer[buffer_index] = input_array[i];
        }
    }

    /// Write input samples to a specific channel (without offset)
    pub fn write_input_simple(&mut self, channel: usize, input_array: &[T]) {
        self.write_input(channel, 0, input_array.len(), input_array);
    }

    /// Read output samples from a specific channel
    pub fn read_output(&self, channel: usize, offset: usize, length: usize, output_array: &mut [T]) {
        assert!(channel < self.synthesis_channels, "Channel index out of bounds");
        assert!(offset + length <= output_array.len(), "Output array too small");

        let buffer_start = channel * self.block_samples;
        let offset_pos = (self.output_pos + offset) % self.block_samples;

        // Handle wrapping around the circular buffer
        let output_wrap_index = self.block_samples - offset_pos;
        let chunk1 = length.min(output_wrap_index);

        // Copy first chunk (before wrap)
        for i in 0..chunk1 {
            let buffer_index = buffer_start + offset_pos + i;
            output_array[i] = self.output_buffer[buffer_index];
        }

        // Copy second chunk (after wrap)
        for i in chunk1..length {
            let buffer_index = buffer_start + i + offset_pos - self.block_samples;
            output_array[i] = self.output_buffer[buffer_index];
        }
    }

    /// Read output samples from a specific channel (without offset)
    pub fn read_output_simple(&self, channel: usize, output_array: &mut [T]) {
        self.read_output(channel, 0, output_array.len(), output_array);
    }

    /// Move the input position
    pub fn move_input(&mut self, samples: usize) {
        self.input_pos = (self.input_pos + samples) % self.input_length_samples;
    }

    /// Move the output position
    pub fn move_output(&mut self, samples: usize) {
        self.output_pos = (self.output_pos + samples) % self.block_samples;
    }

    /// Set the interval between blocks and update windows
    pub fn set_interval(&mut self, interval: usize, window_shape: WindowShape) {
        self.default_interval = interval;

        // Set window offsets
        self.analysis_offset = self.block_samples / 2;
        self.synthesis_offset = self.block_samples / 2;

        // Create windows
        match window_shape {
            WindowShape::Ignore => {
                // Rectangular window
                for i in 0..self.block_samples {
                    self.analysis_window[i] = T::one();
                    self.synthesis_window[i] = T::one();
                }
            },
            WindowShape::ACG => {
                // Approximate Confined Gaussian window
                let acg = windows::ApproximateConfinedGaussian::with_bandwidth(<T as From<T>>::from(2.5.into()));
                acg.fill(self.analysis_window.as_mut_slice());
                acg.fill(self.synthesis_window.as_mut_slice());
            },
            WindowShape::Kaiser => {
                // Kaiser window
                let kaiser = windows::Kaiser::with_bandwidth(<T as From<f32>>::from(2.5), true);
                kaiser.fill(self.analysis_window.as_mut_slice());
                kaiser.fill(self.synthesis_window.as_mut_slice());
            },
        }

        // Force perfect reconstruction
        windows::force_perfect_reconstruction(&mut self.synthesis_window, self.block_samples, interval);
    }

    /// Add window product to the accumulation buffer
    fn add_window_product(&mut self) {
        for i in 0..self.block_samples {
            self.window_products[i] += self.analysis_window[i] * self.synthesis_window[i];
        }
    }

    /// Process a block of input samples to produce a spectrum
    pub fn process_block_to_spectrum(&mut self, channel: usize) -> &[Complex<T>] {
        assert!(channel < self.analysis_channels, "Channel index out of bounds");

        // Copy input to time buffer with analysis window applied
        let buffer_start = channel * self.input_length_samples;
        for i in 0..self.block_samples {
            let input_index = (self.input_pos + self.block_samples - self.analysis_offset + i) % self.input_length_samples;
            self.time_buffer[i] = self.input_buffer[buffer_start + input_index] * self.analysis_window[i];
        }

        // Zero-pad the rest of the FFT buffer
        for i in self.block_samples..self.fft_samples {
            self.time_buffer[i] = T::zero();
        }

        // Perform FFT
        let spectrum_start = channel * self.fft_bins;
        let spectrum_slice = &mut self.spectrum_buffer[spectrum_start..spectrum_start + self.fft_bins];
        self.fft.fft(&self.time_buffer, spectrum_slice);

        // Return the spectrum for this channel
        &self.spectrum_buffer[spectrum_start..spectrum_start + self.fft_bins]
    }

    /// Process a spectrum to produce a block of output samples
    pub fn process_spectrum_to_block(&mut self, channel: usize, spectrum: &[Complex<T>]) {
        assert!(channel < self.synthesis_channels, "Channel index out of bounds");
        assert!(spectrum.len() >= self.fft_bins, "Spectrum too small");

        // Perform inverse FFT
        self.fft.ifft(spectrum, &mut self.time_buffer);

        // Apply synthesis window and add to output buffer
        let buffer_start = channel * self.block_samples;
        for i in 0..self.block_samples {
            let output_index = (self.output_pos + self.synthesis_offset + i) % self.block_samples;
            let window_product = self.window_products[i];
            let value = self.time_buffer[i] * self.synthesis_window[i] / window_product;
            self.output_buffer[buffer_start + output_index] += value;
        }
    }

    /// Process a block of input samples directly to output
    pub fn process_block(&mut self, in_channel: usize, out_channel: usize) {
        // Process input to spectrum
        let spectrum = self.process_block_to_spectrum(in_channel);

        // Copy spectrum to avoid borrowing issues
        let spectrum_copy = spectrum.to_vec();

        // Process spectrum to output
        self.process_spectrum_to_block(out_channel, &spectrum_copy);
    }

    /// Process multiple channels at once
    pub fn process_channels(&mut self, in_channels: &[usize], out_channels: &[usize]) {
        assert!(in_channels.len() == out_channels.len(), "Channel arrays must have the same length");

        for (in_ch, out_ch) in in_channels.iter().zip(out_channels.iter()) {
            self.process_block(*in_ch, *out_ch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_configuration() {
        let mut stft = STFT::<f32>::new(false);
        stft.configure(2, 2, 1024, 0, 256);

        assert_eq!(stft.block_samples(), 1024);
        assert_eq!(stft.fft_samples(), 1024);
        assert_eq!(stft.default_interval(), 256);
        assert_eq!(stft.bands(), 513); // N/2 + 1 for real FFT
    }

    #[test]
    fn test_stft_io() {
        let mut stft = STFT::<f32>::new(false);
        stft.configure(1, 1, 16, 0, 4);

        // Create a test signal (impulse)
        let mut input = vec![0.0; 16];
        input[0] = 1.0;

        // Write input
        stft.write_input_simple(0, &input);

        // Process block
        stft.process_block(0, 0);

        // Read output
        let mut output = vec![0.0; 16];
        stft.read_output_simple(0, &mut output);

        // The output should have a peak at the synthesis offset
        let max_index = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap();

        assert_eq!(max_index, stft.synthesis_latency());
    }

    #[test]
    fn test_stft_frequency_conversion() {
        let mut stft = STFT::<f32>::new(false);
        stft.configure(1, 1, 1024, 0, 256);

        // Test bin to frequency conversion
        let bin = 100.0;
        let freq = stft.bin_to_freq(bin);
        let bin2 = stft.freq_to_bin(freq);

        assert!((bin - bin2).abs() < 1e-10);
    }
}