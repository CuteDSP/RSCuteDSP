//! Phase Rotation and Hilbert Transform
//!
//! This module provides utilities for phase manipulation, including:
//! - Hilbert transform (90-degree phase shift)
//! - Analytic signal generation
//! - Phase rotation of signals
//! - Phase unwrapping and analysis

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

use num_traits::{Float, FromPrimitive};
use num_complex::Complex;
use crate::fft::SimpleFFT;

/// Hilbert Transform processor
///
/// Produces a 90-degree phase-shifted version of a signal.
/// Can be used to generate analytic signals.
pub struct HilbertTransform<T: Float + FromPrimitive> {
    fft: SimpleFFT<T>,
    fft_size: usize,
}

impl<T: Float + FromPrimitive> HilbertTransform<T> {
    /// Create a new Hilbert transform processor with the specified FFT size
    pub fn new(fft_size: usize) -> Self {
        let fft = SimpleFFT::new(fft_size);
        Self { fft, fft_size }
    }

    /// Compute the Hilbert transform of a signal
    ///
    /// Returns the 90-degree phase-shifted version of the input signal.
    pub fn transform(&mut self, signal: &[T]) -> Vec<T> {
        if signal.is_empty() {
            return Vec::new();
        }

        // Pad to FFT size
        let mut padded = vec![Complex::new(T::zero(), T::zero()); self.fft_size];
        for (i, &val) in signal.iter().enumerate().take(self.fft_size) {
            padded[i] = Complex::new(val, T::zero());
        }

        // Forward FFT
        let mut freq = vec![Complex::new(T::zero(), T::zero()); self.fft_size];
        self.fft.fft(&padded, &mut freq);

        // Apply Hilbert filter: multiply positive frequencies by -j, negative by j
        let mid = self.fft_size / 2;
        
        // DC and Nyquist components stay zero
        freq[0] = Complex::new(T::zero(), T::zero());
        if mid < self.fft_size {
            freq[mid] = Complex::new(T::zero(), T::zero());
        }

        // Multiply positive frequencies by -j (rotate by -90 degrees)
        for k in 1..mid {
            let j_mult = Complex::new(T::zero(), -T::one());
            freq[k] = freq[k] * j_mult;
        }

        // Multiply negative frequencies by j (rotate by +90 degrees)
        for k in (mid + 1)..self.fft_size {
            let j_mult = Complex::new(T::zero(), T::one());
            freq[k] = freq[k] * j_mult;
        }

        // Inverse FFT
        let mut hilbert = vec![Complex::new(T::zero(), T::zero()); self.fft_size];
        self.fft.ifft(&freq, &mut hilbert);

        // Extract real part and scale
        let scale = T::from_f64(2.0).unwrap();
        hilbert
            .iter()
            .take(signal.len())
            .map(|c| c.re * scale)
            .collect()
    }

    /// Create an analytic signal from the input
    ///
    /// Returns a vector of complex numbers where:
    /// - Real part = original signal
    /// - Imaginary part = Hilbert transform (90° phase-shifted version)
    pub fn analytic_signal(&mut self, signal: &[T]) -> Vec<Complex<T>> {
        if signal.is_empty() {
            return Vec::new();
        }

        // Pad to FFT size
        let mut padded = vec![Complex::new(T::zero(), T::zero()); self.fft_size];
        for (i, &val) in signal.iter().enumerate().take(self.fft_size) {
            padded[i] = Complex::new(val, T::zero());
        }

        // Forward FFT
        let mut freq = vec![Complex::new(T::zero(), T::zero()); self.fft_size];
        self.fft.fft(&padded, &mut freq);

        // Zero out negative frequencies and double positive ones
        let mid = self.fft_size / 2;
        for k in (mid + 1)..self.fft_size {
            freq[k] = Complex::new(T::zero(), T::zero());
        }
        
        // Double positive frequencies (except DC and Nyquist)
        let two = T::from_f64(2.0).unwrap();
        for k in 1..mid {
            freq[k] = freq[k] * two;
        }

        // Inverse FFT
        let mut analytic = vec![Complex::new(T::zero(), T::zero()); self.fft_size];
        self.fft.ifft(&freq, &mut analytic);

        analytic.iter().take(signal.len()).copied().collect()
    }
}

/// Phase Rotator for applying phase shifts to signals
pub struct PhaseRotator<T: Float> {
    /// Current phase accumulator
    phase: T,
    /// Phase increment per sample
    phase_increment: T,
    /// Two pi constant
    two_pi: T,
}

impl<T: Float + FromPrimitive> PhaseRotator<T> {
    /// Create a new phase rotator with specified frequency
    ///
    /// # Arguments
    /// * `frequency` - The frequency of the oscillation in Hz
    /// * `sample_rate` - The sample rate in Hz
    pub fn new(frequency: T, sample_rate: T) -> Self {
        let two_pi = T::from_f64(std::f64::consts::PI * 2.0).unwrap();
        let phase_increment = (two_pi * frequency) / sample_rate;
        
        Self {
            phase: T::zero(),
            phase_increment,
            two_pi,
        }
    }

    /// Process a sample and apply phase rotation
    /// Returns the rotated sample
    pub fn process(&mut self, sample: T) -> T {
        let output = sample * self.phase.cos();
        self.advance_phase();
        output
    }

    /// Process a sample with quadrature output (real and imaginary)
    /// Returns (in_phase, quadrature)
    pub fn process_quadrature(&mut self, sample: T) -> (T, T) {
        let in_phase = sample * self.phase.cos();
        let quadrature = sample * self.phase.sin();
        self.advance_phase();
        (in_phase, quadrature)
    }

    /// Process a block of samples
    pub fn process_block(&mut self, input: &[T]) -> Vec<T> {
        input.iter().map(|&s| self.process(s)).collect()
    }

    /// Rotate all samples in a vector by a fixed phase angle
    pub fn rotate_by_angle(signal: &[T], angle: T) -> Vec<T> {
        let cos_angle = angle.cos();
        signal.iter().map(|&s| s * cos_angle).collect()
    }

    /// Advance the phase by one sample
    fn advance_phase(&mut self) {
        self.phase = self.phase + self.phase_increment;
        
        // Wrap phase to [0, 2π)
        let two_pi = self.two_pi;
        if self.phase >= two_pi {
            let cycles = (self.phase / two_pi).floor();
            self.phase = self.phase - (cycles * two_pi);
        }
    }

    /// Reset phase to zero
    pub fn reset(&mut self) {
        self.phase = T::zero();
    }

    /// Get current phase
    pub fn get_phase(&self) -> T {
        self.phase
    }

    /// Set current phase
    pub fn set_phase(&mut self, phase: T) {
        self.phase = phase;
    }

    /// Set frequency
    pub fn set_frequency(&mut self, frequency: T, sample_rate: T) {
        self.phase_increment = (self.two_pi * frequency) / sample_rate;
    }
}

/// Compute instantaneous phase of a signal using analytic signal
pub fn instantaneous_phase(analytic: &[Complex<f32>]) -> Vec<f32> {
    analytic
        .iter()
        .map(|c| c.im.atan2(c.re))
        .collect()
}

/// Compute instantaneous magnitude (amplitude) of a signal
pub fn instantaneous_magnitude(analytic: &[Complex<f32>]) -> Vec<f32> {
    analytic
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect()
}

/// Compute instantaneous frequency using phase derivative
pub fn instantaneous_frequency(
    phase: &[f32],
    sample_rate: f32,
) -> Vec<f32> {
    if phase.len() < 2 {
        return Vec::new();
    }

    let mut freq = Vec::with_capacity(phase.len());
    let two_pi = std::f32::consts::PI * 2.0;

    for i in 0..phase.len() - 1 {
        let mut phase_diff = phase[i + 1] - phase[i];
        
        // Unwrap phase if needed
        if phase_diff > std::f32::consts::PI {
            phase_diff -= two_pi;
        } else if phase_diff < -std::f32::consts::PI {
            phase_diff += two_pi;
        }

        let inst_freq = (phase_diff * sample_rate) / two_pi;
        freq.push(inst_freq);
    }

    // Replicate last value
    if let Some(&last) = freq.last() {
        freq.push(last);
    }

    freq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hilbert_transform_basic() {
        let mut hilbert = HilbertTransform::new(64);
        let signal = vec![1.0; 10];
        let result = hilbert.transform(&signal);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_analytic_signal() {
        let mut hilbert = HilbertTransform::new(64);
        let signal = vec![1.0, 0.5, 0.25, 0.125];
        let analytic = hilbert.analytic_signal(&signal);
        assert_eq!(analytic.len(), 4);
        // Just verify that the analytic signal was computed
        assert!(analytic.iter().any(|c| c.re != 0.0 || c.im != 0.0));
    }

    #[test]
    fn test_phase_rotator() {
        let mut rotator = PhaseRotator::new(1.0, 10.0);
        let sample = 1.0;
        let output = rotator.process(sample);
        assert!(output.is_finite());
    }

    #[test]
    fn test_phase_rotator_quadrature() {
        let mut rotator = PhaseRotator::new(1.0, 10.0);
        let sample = 1.0;
        let (i, q) = rotator.process_quadrature(sample);
        assert!(i.is_finite() && q.is_finite());
    }

    #[test]
    fn test_instantaneous_phase() {
        let analytic = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.707, 0.707),
            Complex::new(0.0, 1.0),
        ];
        let phase = instantaneous_phase(&analytic);
        assert_eq!(phase.len(), 3);
    }

    #[test]
    fn test_phase_rotator_reset() {
        let mut rotator = PhaseRotator::new(1.0, 10.0);
        rotator.process(1.0);
        assert!(rotator.get_phase() > 0.0);
        rotator.reset();
        assert_eq!(rotator.get_phase(), 0.0);
    }
}
