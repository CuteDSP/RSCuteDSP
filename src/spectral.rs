//! Spectral Processing
//!
//! This module provides tools for frequency-domain manipulation of audio signals.

#![allow(unused_imports)]

#[cfg(feature = "std")]
use std::{f32::consts::PI, vec::Vec, marker::PhantomData};

#[cfg(feature = "std")]
use std::ops::AddAssign;

#[cfg(not(feature = "std"))]
use core::{f32::consts::PI, marker::PhantomData};

#[cfg(not(feature = "std"))]
use core::ops::AddAssign;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;


use num_complex::Complex;
use num_traits::{Float, FromPrimitive};

use crate::fft;
use crate::windows;
use crate::delay;
use crate::perf;

/// An FFT with built-in windowing and round-trip scaling
///
/// This uses a Modified Real FFT, which applies half-bin shift before the transform.
/// The result therefore has `N/2` bins, centred at the frequencies: `(i + 0.5)/N`.
///
/// This avoids the awkward (real-valued) bands for DC-offset and Nyquist.
pub struct WindowedFFT<T: Float> {
    fft: fft::Pow2RealFFT<T>,
    fft_window: Vec<T>,
    time_buffer: Vec<T>,
    offset_samples: usize,
}

impl<T: Float + num_traits::FloatConst + num_traits::FromPrimitive> WindowedFFT<T> {
    /// Create a new WindowedFFT with the specified size
    pub fn new(size: usize, rotate_samples: usize) -> Self {
        let mut result = Self {
            fft: fft::Pow2RealFFT::new(0),
            fft_window: Vec::new(),
            time_buffer: Vec::new(),
            offset_samples: 0,
        };
        result.set_size(size, rotate_samples);
        result
    }
    
    /// Create a new WindowedFFT with a custom window function
    pub fn with_window<F>(size: usize, window_fn: F, window_offset: T, rotate_samples: usize) -> Self
    where
        F: Fn(T) -> T,
    {
        let mut result = Self {
            fft: fft::Pow2RealFFT::new(0),
            fft_window: Vec::new(),
            time_buffer: Vec::new(),
            offset_samples: 0,
        };
        result.set_size_with_window(size, window_fn, window_offset, rotate_samples);
        result
    }
    
    /// Returns a fast FFT size >= `size`
    pub fn fast_size_above(size: usize, divisor: usize) -> usize {
        // Find the next power of 2 >= size/divisor, then multiply by divisor
        let target = (size + divisor - 1) / divisor; // Ceiling division
        let mut result = 1;
        while result < target {
            result *= 2;
        }
        result * divisor
    }
    
    /// Returns a fast FFT size <= `size`
    pub fn fast_size_below(size: usize, divisor: usize) -> usize {
        // Find the largest power of 2 <= size/divisor, then multiply by divisor
        let target = size / divisor;
        let mut result = 1;
        while result * 2 <= target {
            result *= 2;
        }
        result * divisor
    }
    
    /// Sets the size, returning the window for modification (initially all 1s)
    pub fn set_size_window(&mut self, size: usize, rotate_samples: usize) -> &mut Vec<T> {
        self.fft.resize(size);
        self.fft_window = vec![T::one(); size];
        self.time_buffer.resize(size, T::zero());
        self.offset_samples = rotate_samples % size;
        self.fft_window.as_mut()
    }
    
    /// Sets the FFT size, with a user-defined function for the window
    pub fn set_size_with_window<F>(&mut self, size: usize, window_fn: F, window_offset: T, rotate_samples: usize)
    where
        F: Fn(T) -> T,
    {
        self.set_size_window(size, rotate_samples);
        
        let inv_size = T::from_f32(1.0).unwrap() / T::from_f32(size as f32).unwrap();
        for i in 0..size {
            let r = (T::from_f32(i as f32).unwrap() + window_offset) * inv_size;
            self.fft_window[i] = window_fn(r);
        }
    }
    
    /// Sets the size (using the default Blackman-Harris window)
    pub fn set_size(&mut self, size: usize, rotate_samples: usize) {
        self.set_size_with_window(
            size,
            |x| {
                let phase = T::PI() * T::from_f32(2.0).unwrap() * x;
                // Blackman-Harris
                T::from_f32(0.35875).unwrap() -
                T::from_f32(0.48829).unwrap() * phase.cos() +
                T::from_f32(0.14128).unwrap() * (phase * T::from_f32(2.0).unwrap()).cos() -
                T::from_f32(0.01168).unwrap() * (phase * T::from_f32(3.0).unwrap()).cos()
            },
            T::from_f32(0.5).unwrap(),
            rotate_samples,
        );
    }
    
    /// Get a reference to the window
    pub fn window(&self) -> &[T] {
        &self.fft_window
    }
    
    /// Get the FFT size
    pub fn size(&self) -> usize {
        self.fft_window.len()
    }
    
    /// Performs an FFT, with windowing and rotation (if enabled)
    pub fn fft<I, O>(&mut self, input: I, output: &mut [O], with_window: bool, with_scaling: bool)
    where
        I: AsRef<[T]>,
        O: From<Complex<T>> + Copy,
    {
        let input = input.as_ref();
        let fft_size = self.size();
        let norm = if with_scaling {
            T::from_f32(1.0).unwrap() / T::from_f32(fft_size as f32).unwrap()
        } else {
            T::one()
        };
        
        // Apply window and handle rotation
        for i in 0..self.offset_samples {
            // Inverted polarity since we're using the Modified Real FFT
            self.time_buffer[i + fft_size - self.offset_samples] = 
                -input[i] * norm * if with_window { self.fft_window[i] } else { T::one() };
        }
        for i in self.offset_samples..fft_size {
            self.time_buffer[i - self.offset_samples] = 
                input[i] * norm * if with_window { self.fft_window[i] } else { T::one() };
        }
        
        // Perform FFT
        let mut complex_output = vec![Complex::new(T::zero(), T::zero()); fft_size / 2 + 1];
        self.fft.fft(&self.time_buffer, &mut complex_output);
        
        // Copy to output
        for i in 0..complex_output.len() {
            output[i] = complex_output[i].into();
        }
    }
    
    /// Performs an inverse FFT, with windowing and rotation (if enabled)
    pub fn ifft<I, O>(&mut self, input: &[I], mut output: O, with_window: bool)
    where
        I: Copy + Into<Complex<T>>,
        O: AsMut<[T]>,
    {
        let output = output.as_mut();
        let fft_size = self.size();
        
        // Convert input to complex
        let mut complex_input = vec![Complex::new(T::zero(), T::zero()); fft_size / 2 + 1];
        for i in 0..complex_input.len() {
            complex_input[i] = input[i].into();
        }
        
        // Perform inverse FFT
        self.fft.ifft(&complex_input, &mut self.time_buffer);
        
        // Apply window and handle rotation
        for i in 0..self.offset_samples {
            output[i] = self.time_buffer[i + fft_size - self.offset_samples] * 
                if with_window { self.fft_window[i] } else { T::one() };
        }
        for i in self.offset_samples..fft_size {
            output[i] = self.time_buffer[i - self.offset_samples] * 
                if with_window { self.fft_window[i] } else { T::one() };
        }
    }
}

/// A processor for spectral manipulation of audio
pub struct SpectralProcessor<T: Float> {
    fft: WindowedFFT<T>,
    overlap: usize,
    hop_size: usize,
    input_buffer: Vec<T>,
    output_buffer: Vec<T>,
    spectrum: Vec<Complex<T>>,
    window_sum: Vec<T>,
    steady_state: Vec<T>, // Added for steady-state normalization
}



impl<T: Float + AddAssign + num_traits::FloatConst + FromPrimitive> SpectralProcessor<T> {
    /// Create a new SpectralProcessor with the specified parameters
    pub fn new(fft_size: usize, overlap: usize) -> Self {
        let mut result = Self {
            fft: WindowedFFT::new(fft_size, 0),
            overlap,
            hop_size: fft_size / overlap,
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            spectrum: Vec::new(),
            window_sum: Vec::new(),
            steady_state: Vec::new(), // Added
        };
        result.reset();
        result
    }

    /// Reset the processor state
    pub fn reset(&mut self) {
        let fft_size = self.fft.size();
        self.input_buffer.resize(fft_size, T::zero());
        self.output_buffer.resize(fft_size, T::zero());
        self.spectrum.resize(fft_size / 2 + 1, Complex::new(T::zero(), T::zero()));

        // Calculate window sum for normalization (for each absolute sample in the first fft_size samples)
        self.window_sum = vec![T::zero(); fft_size];
        for i in 0..self.overlap {
            let hop = self.hop_size;
            for j in 0..fft_size {
                let absolute_index = i * hop + j;
                if absolute_index < fft_size {
                    let win_val = self.fft.window()[j];
                    self.window_sum[absolute_index] += win_val * win_val;
                }
            }
        }

        // Precompute steady-state normalization factors for remainders
        self.steady_state = vec![T::zero(); self.hop_size];
        for r in 0..self.hop_size {
            let mut offset = r;
            while offset < fft_size {
                let win_val = self.fft.window()[offset];
                self.steady_state[r] += win_val * win_val;
                offset += self.hop_size;
            }
            // Avoid division by zero in steady-state
            if self.steady_state[r] < T::from_f32(1e-10).unwrap() {
                self.steady_state[r] = T::one();
            }
        }

        // Avoid division by zero in window_sum
        for value in self.window_sum.iter_mut() {
            if *value < T::from_f32(1e-10).unwrap() {
                *value = T::one();
            }
        }
    }
    
    /// Get the FFT size
    pub fn fft_size(&self) -> usize {
        self.fft.size()
    }
    
    /// Get the hop size (distance between consecutive frames)
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }
    
    /// Get the overlap factor
    pub fn overlap(&self) -> usize {
        self.overlap
    }
    
    /// Process a block of input samples with a spectral processing function
    pub fn process<F>(&mut self, input: &[T], output: &mut [T], processor: F)
    where
        F: FnMut(&mut [Complex<T>]),
    {
        self.process_with_options(input, output, processor, true, true);
    }

    /// Process a block of input samples with a spectral processing function and options
    pub fn process_with_options<F>(
        &mut self,
        input: &[T],
        output: &mut [T],
        mut processor: F,
        with_window: bool,
        with_scaling: bool,
    )
    where
        F: FnMut(&mut [Complex<T>]),
    {
        let fft_size = self.fft.size();
        let input_len = input.len();
        let output_len = output.len();

        // Process in overlapping blocks
        for i in (0..input_len).step_by(self.hop_size) {
            // Copy input to buffer with bounds checking
            let copy_len = (input_len - i).min(fft_size);
            self.input_buffer[..copy_len].copy_from_slice(&input[i..i + copy_len]);
            self.input_buffer[copy_len..].fill(T::zero());

            // Perform FFT
            self.fft.fft(&self.input_buffer, &mut self.spectrum, with_window, with_scaling);

            // Apply spectral processing
            processor(&mut self.spectrum);

            // Perform inverse FFT
            self.fft.ifft(&self.spectrum, &mut self.output_buffer, with_window);

            // Overlap-add to output with safe bounds checking
            let output_offset = i;
            let add_len = (output_len.saturating_sub(output_offset)).min(fft_size);
            for j in 0..add_len {
                let abs_index = output_offset + j;
                let norm_factor = if abs_index < fft_size {
                    // Use exact normalization factor for initial samples
                    self.window_sum[abs_index]
                } else {
                    // Use steady-state factor for periodic part
                    let r = abs_index % self.hop_size;
                    self.steady_state[r]
                };
                output[abs_index] += self.output_buffer[j] / norm_factor;
            }
        }
    }
}

/// Utility functions for spectral processing
pub mod utils {
    use super::*;
    
    /// Convert magnitude and phase to complex
    pub fn mag_phase_to_complex<T: Float>(mag: T, phase: T) -> Complex<T> {
        Complex::new(mag * phase.cos(), mag * phase.sin())
    }
    
    /// Convert complex to magnitude and phase
    pub fn complex_to_mag_phase<T: Float>(complex: Complex<T>) -> (T, T) {
        (complex.norm(), complex.arg())
    }
    
    /// Convert linear magnitude to decibels
    pub fn linear_to_db<T: Float>(linear: T) -> T {
        T::from(20.0).unwrap() * linear.log10()
    }
    
    /// Convert decibels to linear magnitude
    pub fn db_to_linear<T: Float>(db: T) -> T {
        T::from(10.0).unwrap().powf(db / T::from(20.0).unwrap())
    }
    
    /// Apply a gain to a spectrum (in decibels)
    pub fn apply_gain<T: Float>(spectrum: &mut [Complex<T>], gain_db: T) {
        let gain_linear = db_to_linear(gain_db);
        for bin in spectrum {
            *bin = *bin * gain_linear;
        }
    }
    
    /// Apply a phase shift to a spectrum (in radians)
    pub fn apply_phase_shift<T: Float>(spectrum: &mut [Complex<T>], phase_shift: T) {
        for bin in spectrum {
            let (mag, phase) = complex_to_mag_phase(*bin);
            *bin = mag_phase_to_complex(mag, phase + phase_shift);
        }
    }
    
    /// Apply a time shift to a spectrum
    pub fn apply_time_shift<T: Float>(spectrum: &mut [Complex<T>], time_shift: T, sample_rate: T) {
        let fft_size = spectrum.len() * 2 - 2;
        let bin_width = sample_rate / T::from(fft_size as f32).unwrap();

        for (i, bin) in spectrum.iter_mut().enumerate() {
            let freq = T::from(i as f32).unwrap() * bin_width;
            let phase_shift = T::from(2.0 * PI).unwrap() * freq * time_shift;
            let (mag, phase) = complex_to_mag_phase(*bin);
            *bin = mag_phase_to_complex(mag, phase + phase_shift);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windowed_fft() {
        let mut fft = WindowedFFT::<f32>::new(1024, 0);

        // Create a test signal (sine wave)
        let mut input = vec![0.0; 1024];
        for i in 0..1024 {
            input[i] = (i as f32 * 0.1).sin();
        }

        // Perform FFT
        let mut output = vec![Complex::new(0.0, 0.0); 513]; // N/2 + 1
        fft.fft(&input, &mut output, true, true);

        // The spectrum should have peaks at the sine wave frequency
        let peak_bin = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
            .map(|(index, _)| index)
            .unwrap();

        // Expected peak at around bin 16-18 (0.1 * 1024 / (2*PI) ≈ 16.3)
        assert!(peak_bin >= 16 && peak_bin <= 18);  // Fixed expected bin range
    }

    #[test]
    fn test_spectral_processor() {
        let mut processor = SpectralProcessor::<f32>::new(1024, 4);

        // Create a test signal (sine wave)
        let mut input = vec![0.0; 2048];
        for i in 0..2048 {
            input[i] = (i as f32 * 0.1).sin();
        }

        // Create output buffer
        let mut output = vec![0.0; 2048];

        // Process with identity function (should reconstruct the input)
        processor.process(&input, &mut output, |_spectrum| {
            // Do nothing (identity)
        });

        // Check that the output approximates the input
        for i in 512..1536 { // Ignore edges due to windowing effects
            assert!((input[i] - output[i]).abs() < 0.1);
        }
    }
    
    #[test]
    fn test_spectral_utils() {
        // Test magnitude/phase conversion
        let complex = Complex::new(3.0, 4.0);
        let (mag, phase) = utils::complex_to_mag_phase(complex);
        let complex2 = utils::mag_phase_to_complex(mag, phase);
        
        assert!((complex.re - complex2.re).abs() < 1e-10);
        assert!((complex.im - complex2.im).abs() < 1e-10);
        
        // Test dB conversion
        let linear = 10.0;
        let db = utils::linear_to_db(linear);
        let linear2 = utils::db_to_linear(db);
        
        assert!((linear - linear2).abs() < 1e-10);
    }
}