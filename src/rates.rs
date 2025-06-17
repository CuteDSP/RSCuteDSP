//! Multi-rate processing
//!
//! This module provides utilities for oversampling, upsampling, and downsampling.

#![allow(unused_imports)]

#[cfg(feature = "std")]
use std::{f32::consts::PI, vec::Vec};

#[cfg(not(feature = "std"))]
use core::f32::consts::PI;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

use num_traits::Float;

/// Fills a container with a Kaiser-windowed sinc for an FIR lowpass.
pub fn fill_kaiser_sinc<T: Float + From<f32>>(
    data: &mut [T],
    pass_freq: f32,
    stop_freq: f32,
) {
    let length = data.len();
    if length <= 0 {
        return;
    }
    
    // Calculate Kaiser bandwidth parameter
    let kaiser_bandwidth = <T as From<f32>>::from(stop_freq - pass_freq) * <T as From<f32>>::from(length as f32);
    let kaiser_bandwidth = kaiser_bandwidth + <T as From<f32>>::from(1.25) / kaiser_bandwidth; // heuristic for transition band
    
    // Create Kaiser window
    let kaiser = crate::windows::Kaiser::with_bandwidth(kaiser_bandwidth, false);
    kaiser.fill(data);
    
    // Apply sinc function
    let centre_index = (length - 1) as f32 * 0.5;
    let sinc_scale = PI * (pass_freq + stop_freq);
    let amp_scale = pass_freq + stop_freq;
    
    for i in 0..length {
        let x = i as f32 - centre_index;
        let px = x * sinc_scale;
        let sinc = if px.abs() > 1e-6 {
            px.sin() * amp_scale / px
        } else {
            amp_scale
        };
        data[i] = data[i] * <T as From<f32>>::from(sinc);
    }
}

/// If only the centre frequency is specified, a heuristic is used to balance ripples and transition width.
pub fn fill_kaiser_sinc_with_centre<T: Float + From<f32>>(
    data: &mut [T],
    centre_freq: f32,
) {
    let length = data.len();
    let half_width = 0.45 / (length as f32).sqrt();
    let half_width = if half_width > centre_freq {
        (half_width + centre_freq) * 0.5
    } else {
        half_width
    };
    
    fill_kaiser_sinc(data, centre_freq - half_width, centre_freq + half_width);
}

/// 2x FIR oversampling for block-based processing.
///
/// The oversampled signal is stored inside this object, with channels accessed via `oversampler[c]`.
pub struct Oversampler2xFIR<T: Float> {
    one_way_latency: usize,
    kernel_length: usize,
    channels: usize,
    stride: usize,
    input_stride: usize,
    input_buffer: Vec<T>,
    half_sample_kernel: Vec<T>,
    buffer: Vec<T>,
}

impl<T: Float + From<f32>> Oversampler2xFIR<T> {
    /// Create a new oversampler with the specified parameters
    pub fn new(channels: usize, max_block: usize, half_latency: usize, pass_freq: f32) -> Self {
        let mut result = Self {
            one_way_latency: half_latency,
            kernel_length: half_latency * 2,
            channels,
            stride: 0,
            input_stride: 0,
            input_buffer: Vec::new(),
            half_sample_kernel: Vec::new(),
            buffer: Vec::new(),
        };
        
        result.resize_with_params(channels, max_block, half_latency, pass_freq);
        result
    }
    
    /// Resize the oversampler with new parameters
    pub fn resize(&mut self, channels: usize, max_block_length: usize) {
        self.resize_with_params(channels, max_block_length, self.one_way_latency, 0.43);
    }
    
    /// Resize the oversampler with new parameters including latency and passband frequency
    pub fn resize_with_params(
        &mut self,
        channels: usize,
        max_block_length: usize,
        half_latency: usize,
        pass_freq: f32,
    ) {
        self.one_way_latency = half_latency;
        self.kernel_length = half_latency * 2;
        self.channels = channels;
        
        self.half_sample_kernel.resize(self.kernel_length, T::zero());
        fill_kaiser_sinc(
            &mut self.half_sample_kernel,
            pass_freq,
            1.0 - pass_freq,
        );
        
        self.input_stride = self.kernel_length + max_block_length;
        self.input_buffer.resize(channels * self.input_stride, T::zero());
        
        self.stride = (max_block_length + self.kernel_length) * 2;
        self.buffer.resize(self.stride * channels, T::zero());
    }
    
    /// Reset the oversampler state
    pub fn reset(&mut self) {
        for i in 0..self.input_buffer.len() {
            self.input_buffer[i] = T::zero();
        }
        for i in 0..self.buffer.len() {
            self.buffer[i] = T::zero();
        }
    }
    
    /// Round-trip latency (or equivalently: upsample latency at the higher rate).
    /// This will be twice the value passed into the constructor or `.resize()`.
    pub fn latency(&self) -> usize {
        self.kernel_length
    }
    
    /// Upsamples from a multi-channel input into the internal buffer
    pub fn up<D>(&mut self, data: D, low_samples: usize)
    where
        D: AsRef<[T]>,
        D: core::ops::Index<usize, Output = [T]>,
    {
        for c in 0..self.channels {
            self.up_channel(c, &data[c], low_samples);
        }
    }
    
    /// Upsamples a single-channel input into the internal buffer
    pub fn up_channel<D>(&mut self, c: usize, data: D, low_samples: usize)
    where
        D: AsRef<[T]>,
    {
        let data = data.as_ref();

        // Cache field values to avoid borrowing self multiple times
        let one_way_latency = self.one_way_latency;
        let kernel_length = self.kernel_length;
        let input_stride = self.input_stride;
        let stride = self.stride;

        // Split the borrows to avoid conflicts
        let (input_buffer, buffer, half_sample_kernel) = (
            &mut self.input_buffer,
            &mut self.buffer,
            &self.half_sample_kernel,
        );

        let input_channel = &mut input_buffer[c * input_stride..(c + 1) * input_stride];
        let output = &mut buffer[c * stride + kernel_length * 2..(c + 1) * stride];

        // Copy input data to buffer
        for i in 0..low_samples {
            input_channel[kernel_length + i] = data[i];
        }

        // Process
        for i in 0..low_samples {
            output[2 * i] = input_channel[i + one_way_latency];

            let mut sum = T::zero();
            for o in 0..kernel_length {
                sum = sum + input_channel[i + 1 + o] * half_sample_kernel[o];
            }
            output[2 * i + 1] = sum;
        }

        // Copy the end of the buffer back to the beginning
        for i in 0..kernel_length {
            input_channel[i] = input_channel[low_samples + i];
        }
    }
    
    /// Downsamples from the internal buffer to a multi-channel output
    pub fn down<D>(&mut self, mut data: D, low_samples: usize)
    where
        D: AsMut<[T]>,
        D: core::ops::IndexMut<usize, Output = [T]>,
    {
        for c in 0..self.channels {
            self.down_channel(c, &mut data[c], low_samples);
        }
    }
    
    /// Downsamples a single channel from the internal buffer to a single-channel output
    pub fn down_channel<D>(&mut self, c: usize, mut data: D, low_samples: usize)
    where
        D: AsMut<[T]>,
    {
        let data = data.as_mut();
        let input = &mut self.buffer[c * self.stride..(c + 1) * self.stride];
        
        for i in 0..low_samples {
            let v1 = input[2 * i + self.kernel_length];
            
            let mut sum = T::zero();
            for o in 0..self.kernel_length {
                let v2 = input[2 * (i + o) + 1];
                sum = sum + v2 * self.half_sample_kernel[o];
            }
            
            let v2 = sum;
            let v = (v1 + v2) * <T as From<f32>>::from(0.5);
            data[i] = v;
        }
        
        // Copy the end of the buffer back to the beginning
        for i in 0..self.kernel_length * 2 {
            input[i] = input[low_samples * 2 + i];
        }
    }
    
    /// Gets the samples for a single (higher-rate) channel.
    /// The valid length depends how many input samples were passed into `.up()`/`.up_channel()`.
    pub fn get_channel(&mut self, c: usize) -> &mut [T] {
        &mut self.buffer[c * self.stride + self.kernel_length * 2..
                         (c + 1) * self.stride]
    }
    
    /// Gets the samples for a single (higher-rate) channel (immutable version).
    pub fn get_channel_ref(&self, c: usize) -> &[T] {
        &self.buffer[c * self.stride + self.kernel_length * 2..
                    (c + 1) * self.stride]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fill_kaiser_sinc() {
        let mut data = vec![0.0; 21];
        fill_kaiser_sinc(&mut data, 0.2, 0.3);
        
        // Should be symmetric
        for i in 0..10 {
            assert!((data[i] - data[20 - i]).abs() < 1e-10);
        }
        
        // Center value should be highest
        assert!(data[10] > data[9]);
        assert!(data[10] > data[11]);
    }
    
    #[test]
    fn test_fill_kaiser_sinc_with_centre() {
        let mut data = vec![0.0; 21];
        fill_kaiser_sinc_with_centre(&mut data, 0.25);
        
        // Should be symmetric
        for i in 0..10 {
            assert!((data[i] - data[20 - i]).abs() < 1e-10);
        }
        
        // Center value should be highest
        assert!(data[10] > data[9]);
        assert!(data[10] > data[11]);
    }
    
    #[test]
    fn test_oversampler2xfir() {
        // Create a 2-channel oversampler
        let mut oversampler = Oversampler2xFIR::<f32>::new(2, 16, 8, 0.43);
        
        // Create test input
        let mut input = vec![vec![0.0; 16]; 2];
        for i in 0..16 {
            // Channel 0: sine wave
            input[0][i] = (i as f32 * 0.2).sin();
            // Channel 1: cosine wave
            input[1][i] = (i as f32 * 0.2).cos();
        }
        
        // Upsample
        for c in 0..2 {
            oversampler.up_channel(c, &input[c], 16);
        }
        
        // Check that the upsampled data has the expected properties
        for c in 0..2 {
            let upsampled = oversampler.get_channel_ref(c);
            
            // Check that the even samples match the original (with latency)
            for i in 0..8 {
                assert!((upsampled[i*2] - input[c][i]).abs() < 1e-10);
            }
            
            // Check that the odd samples are interpolated (they should be between adjacent even samples)
            for i in 0..7 {
                let even1 = upsampled[i*2];
                let even2 = upsampled[(i+1)*2];
                let odd = upsampled[i*2 + 1];
                
                // The odd sample should be between the adjacent even samples or at least close
                if (even1 - even2).abs() > 1e-10 {
                    let t = (odd - even1) / (even2 - even1);
                    assert!(t >= -0.1 && t <= 1.1);
                }
            }
        }
        
        // Create output buffer for downsampling
        let mut output = vec![vec![0.0; 16]; 2];
        
        // Downsample
        for c in 0..2 {
            oversampler.down_channel(c, &mut output[c], 16);
        }
        
        // Check that the downsampled data approximates the original
        for c in 0..2 {
            for i in 0..16 {
                // Allow some error due to filtering
                assert!((output[c][i] - input[c][i]).abs() < 0.1);
            }
        }
    }
}