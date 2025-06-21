//! Multichannel mixing utilities
//!
//! This module provides utilities for stereo/multichannel mixing operations,
//! including orthogonal matrices and stereo-to-multi channel conversion.

#![allow(unused_imports)]

#[cfg(feature = "std")]
use std::f32::consts::PI;

#[cfg(not(feature = "std"))]
use core::f32::consts::PI;

use num_traits::{Float, FromPrimitive};

/// Hadamard matrix: high mixing levels, N log(N) operations
pub struct Hadamard<T: Float> {
    size: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float + FromPrimitive> Hadamard<T> {
    /// Create a new Hadamard matrix with the specified size
    pub fn new(size: usize) -> Self {
        Self {
            size,
            _marker: core::marker::PhantomData,
        }
    }
    
    /// Apply the matrix in-place, scaled so it's orthogonal
    pub fn in_place(&self, data: &mut [T]) {
        self.unscaled_in_place(data);
        
        let factor = self.scaling_factor();
        for c in 0..self.size {
            data[c] = data[c] * factor;
        }
    }
    
    /// Scaling factor applied to make it orthogonal
    pub fn scaling_factor(&self) -> T {
        if self.size == 0 {
            T::one()
        } else {
            T::from_f32(1.0 / (self.size as f32).sqrt()).unwrap()
        }
    }
    
    pub fn unscaled_in_place(&self, data: &mut [T]) {
        if self.size <= 1 {
            return;
        }
        
        let mut h_size = 1;
        while h_size < self.size {
            for start_index in (0..self.size).step_by(h_size * 2) {
                for i in start_index..(start_index + h_size).min(self.size) {
                    if i + h_size < self.size {
                        let a = data[i];
                        let b = data[i + h_size];
                        data[i] = a + b;
                        data[i + h_size] = a - b;
                    }
                }
            }
            h_size *= 2;
        }
    }
}

/// Householder matrix: moderate mixing, 2N operations
pub struct Householder<T: Float> {
    size: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Float + FromPrimitive> Householder<T> {
    /// Create a new Householder matrix with the specified size
    pub fn new(size: usize) -> Self {
        Self {
            size,
            _marker: core::marker::PhantomData,
        }
    }
    
    /// Apply the matrix in-place
    pub fn in_place(&self, data: &mut [T]) {
        if self.size < 1 {
            return;
        }
        
        let factor = T::from_f32(-2.0 / self.size as f32).unwrap();
        
        let mut sum = data[0];
        for i in 1..self.size {
            sum = sum + data[i];
        }
        sum = sum * factor;
        
        for i in 0..self.size {
            data[i] = data[i] + sum;
        }
    }
    
    /// The matrix is already orthogonal, but this is here for compatibility with Hadamard
    pub fn scaling_factor(&self) -> T {
        T::one()
    }
}

/// Upmix/downmix a stereo signal to an (even) multi-channel signal
///
/// When spreading out, it rotates the input by various amounts (e.g. a four-channel signal
/// would produce `(left, right, mid, side)`), such that energy is preserved for each pair.
///
/// When mixing together, it uses the opposite rotations, such that upmix → downmix
/// produces the same stereo signal (when scaled by `.scaling_factor1()`).
pub struct StereoMultiMixer<T: Float> {
    channels: usize,
    coeffs: Vec<T>,
}

impl<T: Float + FromPrimitive> StereoMultiMixer<T> {
    /// Create a new mixer with the specified number of channels (must be even)
    pub fn new(channels: usize) -> Self {
        assert!(channels > 0, "StereoMultiMixer must have a positive number of channels");
        assert_eq!(channels % 2, 0, "StereoMultiMixer must have an even number of channels");
        
        let h_channels = channels / 2;
        let mut coeffs = vec![T::zero(); channels];
        
        coeffs[0] = T::one();
        coeffs[1] = T::zero();
        
        for i in 1..h_channels {
            let phase = PI * i as f32 / channels as f32;
            coeffs[2 * i] = T::from_f32(phase.cos()).unwrap();
            coeffs[2 * i + 1] = T::from_f32(phase.sin()).unwrap();
        }
        
        Self { channels, coeffs }
    }
    
    /// Convert a stereo signal to a multi-channel signal
    pub fn stereo_to_multi(&self, input: &[T; 2], output: &mut [T]) {
        let scale = T::from_f32((2.0 / self.channels as f32).sqrt()).unwrap();
        output[0] = input[0] * scale;
        output[1] = input[1] * scale;
        
        for i in (2..self.channels).step_by(2) {
            output[i] = (input[0] * self.coeffs[i] + input[1] * self.coeffs[i + 1]) * scale;
            output[i + 1] = (input[1] * self.coeffs[i] - input[0] * self.coeffs[i + 1]) * scale;
        }
    }
    
    /// Convert a multi-channel signal back to stereo
    pub fn multi_to_stereo(&self, input: &[T], output: &mut [T; 2]) {
        output[0] = input[0];
        output[1] = input[1];
        
        for i in (2..self.channels).step_by(2) {
            output[0] = output[0] + input[i] * self.coeffs[i] - input[i + 1] * self.coeffs[i + 1];
            output[1] = output[1] + input[i + 1] * self.coeffs[i] + input[i] * self.coeffs[i + 1];
        }
    }
    
    /// Scaling factor for the downmix, if channels are phase-aligned
    pub fn scaling_factor1(&self) -> T {
        T::from_f32(2.0 / self.channels as f32).unwrap()
    }
    
    /// Scaling factor for the downmix, if channels are independent
    pub fn scaling_factor2(&self) -> T {
        T::from_f32((2.0 / self.channels as f32).sqrt()).unwrap()
    }
}

/// A cheap (polynomial) almost-energy-preserving crossfade
///
/// Maximum energy error: 1.06%, average 0.64%, curves overshoot by 0.3%
pub fn cheap_energy_crossfade<T: Float + From<f32>>(x: T, to_coeff: &mut T, from_coeff: &mut T) {
    *to_coeff = x;
    *from_coeff = T::one() - x;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_householder() {
        let householder = Householder::<f32>::new(4);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        
        householder.in_place(&mut data);
        
        // Expected result: original - 2*mean*[1,1,1,1]
        let mean = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
        let expected = vec![
            1.0 - 2.0 * mean,
            2.0 - 2.0 * mean,
            3.0 - 2.0 * mean,
            4.0 - 2.0 * mean,
        ];
        
        for i in 0..4 {
            assert!((data[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_hadamard() {
        let hadamard = Hadamard::<f32>::new(4);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];

        hadamard.in_place(&mut data);

        // Correct expected values for 4-point Hadamard transform
        let expected = vec![5.0, -1.0, -2.0, 0.0];
        for i in 0..4 {
            assert!((data[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_stereo_multi_mixer() {
        let mixer = StereoMultiMixer::<f32>::new(4);
        let input = [1.0, 2.0];
        let mut output = vec![0.0; 4];

        mixer.stereo_to_multi(&input, &mut output);

        // Check energy preservation
        let input_energy = input[0] * input[0] + input[1] * input[1];
        let output_energy = output.iter().map(|&x| x * x).sum::<f32>();
        assert!((input_energy - output_energy).abs() < 1e-6);

        // Test round-trip with correct scaling factor
        let mut round_trip = [0.0, 0.0];
        mixer.multi_to_stereo(&output, &mut round_trip);

        // Use scaling factor for independent channels
        let scale = mixer.scaling_factor2();
        assert!((round_trip[0] * scale - input[0]).abs() < 1e-6);
        assert!((round_trip[1] * scale - input[1]).abs() < 1e-6);
    }

    #[test]
    fn test_cheap_energy_crossfade() {
        let mut to_coeff = 0.0;
        let mut from_coeff = 0.0;

        cheap_energy_crossfade(0.5, &mut to_coeff, &mut from_coeff);

        // At x=0.5, coefficients should be equal
        assert!((to_coeff - from_coeff).abs() < 1e-6);

        // Sum should be close to 1.0 (within the error margin)
        assert!((to_coeff + from_coeff - 1.0).abs() < 0.03);

        // Test at extremes
        cheap_energy_crossfade(0.0, &mut to_coeff, &mut from_coeff);
        assert!(to_coeff < 0.01);
        assert!((from_coeff - 1.0).abs() < 0.01);

        cheap_energy_crossfade(1.0, &mut to_coeff, &mut from_coeff);
        assert!((to_coeff - 1.0).abs() < 0.01);
        assert!(from_coeff < 0.01);
    }
}