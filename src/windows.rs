//! Window functions for spectral analysis
//!
//! This module provides window functions commonly used in spectral analysis,
//! such as the Kaiser window and Approximate Confined Gaussian window.

#![allow(unused_imports)]

#[cfg(feature = "std")]
use std::f32::consts::PI;

#[cfg(not(feature = "std"))]
use core::f32::consts::PI;

use num_traits::{Float, FromPrimitive, NumCast};

/// The Kaiser window (almost) maximizes the energy in the main-lobe compared to the side-lobes.
pub struct Kaiser<T: Float> {
    beta: T,
    inv_b0: T,
}

impl<T: Float + FromPrimitive + NumCast> Kaiser<T> {
    /// Create a Kaiser window with a given shape parameter.
    pub fn new(beta: T) -> Self {
        Self {
            beta,
            inv_b0: T::from_f32(1.0).unwrap() / Self::bessel0(beta),
        }
    }

    /// Create a Kaiser window with a specified bandwidth
    pub fn with_bandwidth(bandwidth: T, heuristic_optimal: bool) -> Self {
        Self::new(Self::bandwidth_to_beta(bandwidth, heuristic_optimal))
    }

    /// Convert bandwidth to beta parameter
    pub fn bandwidth_to_beta(bandwidth: T, heuristic_optimal: bool) -> T {
        let bandwidth = if heuristic_optimal {
            Self::heuristic_bandwidth(bandwidth)
        } else {
            bandwidth
        };
        let bandwidth = bandwidth.max(T::from_f32(2.0).unwrap());
        let alpha = (bandwidth * bandwidth * T::from_f32(0.25).unwrap() - T::from_f32(1.0).unwrap()).sqrt();
        alpha * T::from_f32(PI).unwrap()
    }

    /// Fills a slice with a Kaiser window
    pub fn fill(&self, data: &mut [T]) {
        let size = data.len();
        let inv_size = T::from_f32(1.0).unwrap() / T::from_usize(size).unwrap();
        for i in 0..size {
            let r = T::from_usize(2 * i + 1).unwrap() * inv_size - T::from_f32(1.0).unwrap();
            let arg = (T::from_f32(1.0).unwrap() - r * r).sqrt();
            data[i] = Self::bessel0(self.beta * arg) * self.inv_b0;
        }
    }

    // Modified Bessel function of the first kind, order 0
    fn bessel0(x: T) -> T {
        const SIGNIFICANCE_LIMIT: f32 = 1e-4;
        let mut result = T::from_f32(0.0).unwrap();
        let mut term = T::from_f32(1.0).unwrap();
        let mut m = T::from_f32(0.0).unwrap();
        while term > T::from_f32(SIGNIFICANCE_LIMIT).unwrap() {
            result = result + term;
            m = m + T::from_f32(1.0).unwrap();
            term = term * (x * x) / (T::from_f32(4.0).unwrap() * m * m);
        }
        result
    }

    // Heuristic for optimal bandwidth
    fn heuristic_bandwidth(bandwidth: T) -> T {
        bandwidth + T::from_f32(8.0).unwrap() / ((bandwidth + T::from_f32(3.0).unwrap()) * (bandwidth + T::from_f32(3.0).unwrap()))
            + T::from_f32(0.25).unwrap() * (T::from_f32(3.0).unwrap() - bandwidth).max(T::from_f32(0.0).unwrap())
    }
}

/// The Approximate Confined Gaussian window
pub struct ApproximateConfinedGaussian<T: Float> {
    gaussian_factor: T,
}

impl <T: Float> ApproximateConfinedGaussian<T> {
    /// Create an ACG window with a given shape parameter
    pub fn new(sigma: T) -> Self {
        Self {
            gaussian_factor: T::from(0.0625).unwrap() / (sigma * sigma),
        }
    }

    /// Create an ACG window with a specified bandwidth
    pub fn with_bandwidth(bandwidth: T) -> Self {
        Self::new(Self::bandwidth_to_sigma(bandwidth))
    }

    /// Heuristic map from bandwidth to sigma
    pub fn bandwidth_to_sigma(bandwidth: T) -> T {
        T::from(0.3).unwrap() / bandwidth.sqrt()
    }

    /// Fills a slice with an ACG window
    pub fn fill(&self, data: &mut [T]) {
        let size = data.len();
        let inv_size = T::from(1.0).unwrap() / T::from(size as f32).unwrap();
        let offset_scale = self.gaussian(T::from(1.0).unwrap()) / (self.gaussian(T::from(3.0).unwrap()) + self.gaussian(T::from(-1.0).unwrap()));
        let norm = T::from(1.0).unwrap() / (self.gaussian(T::from(0.0).unwrap()) - T::from(2.0).unwrap() * offset_scale * self.gaussian(T::from(2.0).unwrap()));
        
        for i in 0..size {
            let r = (T::from(2.0).unwrap() * T::from(i as f32).unwrap() + T::from(1.0).unwrap()) * inv_size - T::from(1.0).unwrap();
            let value = norm * (self.gaussian(r) - offset_scale * (self.gaussian(r - T::from(2.0).unwrap()) + self.gaussian(r + T::from(2.0).unwrap())));
            data[i] = value;
        }
    }

    // Gaussian function
    fn gaussian(&self, x: T) -> T {
        (-T::from(x * x).unwrap() * self.gaussian_factor).exp()
    }
}

/// Forces STFT perfect-reconstruction on an existing window
pub fn force_perfect_reconstruction<T: Float>(
    data: &mut [T],
    window_length: usize,
    interval: usize,
) {
    for i in 0..interval {
        let mut sum2 = 0.0;
        let mut index = i;
        while index < window_length {
            let val = data[index].to_f32().unwrap();
            sum2 += val * val;
            index += interval;
        }
        let factor = 1.0 / sum2.sqrt();
        
        index = i;
        while index < window_length {
            data[index] = data[index] * T::from(factor).unwrap();
            index += interval;
        }
    }
}

//test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kaiser_window() {
        let mut window = vec![0.0f32; 8];
        let kaiser = Kaiser::new(3.0);
        kaiser.fill(&mut window);
        assert_eq!(window.len(), 8);
        assert!(window.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_acg_window() {
        let mut window = vec![0.0f32; 8];
        let acg = ApproximateConfinedGaussian::new(1.0);
        acg.fill(&mut window);
        assert_eq!(window.len(), 8);
        assert!(window.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
    
    #[test]
    fn test_force_perfect_reconstruction() {
        let mut window = vec![0.1f32; 8];
        force_perfect_reconstruction(&mut window, 8, 2);
        assert_eq!(window.len(), 8);
        assert!(window.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
    
    #[test]
    fn test_bandwidth_to_beta() {
        let beta = Kaiser::<f32>::bandwidth_to_beta(4.0, true);
        assert!(beta > 0.0);
        let kaiser = Kaiser::new(beta);
        let mut window = vec![0.0f32; 8];
        kaiser.fill(&mut window);
        assert_eq!(window.len(), 8);
        assert!(window.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
    
    #[test]
    fn test_bandwidth_to_sigma() {
        let sigma = ApproximateConfinedGaussian::<f32>::bandwidth_to_sigma(4.0);
        assert!(sigma > 0.0);
        let acg = ApproximateConfinedGaussian::new(sigma);
        let mut window = vec![0.0f32; 8];
        acg.fill(&mut window);
        assert_eq!(window.len(), 8);
        assert!(window.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
    
    #[test]
    fn test_heuristic_bandwidth() {
        let bandwidth = Kaiser::<f32>::heuristic_bandwidth(4.0);
        assert!(bandwidth > 0.0);
        let beta = Kaiser::<f32>::bandwidth_to_beta(bandwidth, true);
        assert!(beta > 0.0);
        let kaiser = Kaiser::new(beta);
        let mut window = vec![0.0f32; 8];
        kaiser.fill(&mut window);
        assert_eq!(window.len(), 8);
        assert!(window.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
    
    
}