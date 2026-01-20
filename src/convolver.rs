//! Convolver with Impulse Response (IR) support
//!
//! This module provides efficient convolution for applying impulse responses to audio signals.
//! It uses FFT-based convolution (overlap-add) for efficient processing of long impulse responses.

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

use num_traits::Float;
use num_complex::Complex;
use crate::fft::SimpleFFT;
use num_traits::FromPrimitive;

/// Maximum IR length for direct convolution (shorter IRs use direct time-domain convolution)
const DIRECT_CONV_THRESHOLD: usize = 256;

/// A convolver that applies an impulse response to audio signals
///
/// This implementation uses FFT-based convolution (overlap-add method) for long impulse responses
/// and direct convolution for shorter ones. It maintains internal state for processing continuous
/// audio streams.
pub struct Convolver<T: Float + FromPrimitive> {
    /// Impulse response coefficients
    ir: Vec<T>,
    
    /// FFT instance for convolution
    fft: Option<SimpleFFT<T>>,
    
    /// FFT size used for convolution
    fft_size: usize,
    
    /// Buffer for input overlap
    input_buffer: Vec<T>,
    
    /// Buffer for output overlap
    output_buffer: Vec<T>,
    
    /// Pre-computed FFT of IR
    ir_fft: Vec<Complex<T>>,
    
    /// Current position in input buffer
    input_pos: usize,
    
    /// History buffer for direct convolution
    history: Vec<T>,
    
    /// History position for direct convolution
    history_pos: usize,
    
    /// Use FFT-based convolution
    use_fft: bool,
}

impl<T: Float + FromPrimitive> Convolver<T> {
    /// Create a new convolver with the specified impulse response
    ///
    /// # Arguments
    /// * `ir` - The impulse response coefficients
    pub fn new(ir: Vec<T>) -> Self {
        if ir.is_empty() {
            return Self {
                ir,
                fft: None,
                fft_size: 0,
                input_buffer: Vec::new(),
                output_buffer: Vec::new(),
                ir_fft: Vec::new(),
                input_pos: 0,
                history: Vec::new(),
                history_pos: 0,
                use_fft: false,
            };
        }

        let use_fft = ir.len() > DIRECT_CONV_THRESHOLD;
        
        if use_fft {
            // Create FFT-based convolver
            let fft_size = Self::next_power_of_two(ir.len() * 2);
            let mut fft_instance = SimpleFFT::new(fft_size);
            
            // Pre-compute FFT of IR
            let mut ir_buffer = vec![Complex::new(T::zero(), T::zero()); fft_size];
            for (i, &coeff) in ir.iter().enumerate() {
                ir_buffer[i] = Complex::new(coeff, T::zero());
            }
            
            let mut ir_fft = vec![Complex::new(T::zero(), T::zero()); fft_size];
            fft_instance.fft(&ir_buffer, &mut ir_fft);
            
            let input_buffer = vec![T::zero(); fft_size];
            let output_buffer = vec![T::zero(); fft_size];
            
            Self {
                ir: ir,
                fft: Some(fft_instance),
                fft_size,
                input_buffer,
                output_buffer,
                ir_fft,
                input_pos: 0,
                history: Vec::new(),
                history_pos: 0,
                use_fft: true,
            }
        } else {
            // For direct convolution, we need a history buffer
            let history = vec![T::zero(); ir.len()];
            
            Self {
                ir,
                fft: None,
                fft_size: 0,
                input_buffer: Vec::new(),
                output_buffer: Vec::new(),
                ir_fft: Vec::new(),
                input_pos: 0,
                history,
                history_pos: 0,
                use_fft: false,
            }
        }
    }

    /// Process a single sample through the convolver
    pub fn process(&mut self, sample: T) -> T {
        if self.ir.is_empty() {
            return sample;
        }

        if self.use_fft {
            self.process_fft(sample)
        } else {
            self.process_direct(sample)
        }
    }

    /// Process a block of samples
    pub fn process_block(&mut self, input: &[T]) -> Vec<T> {
        input.iter().map(|&s| self.process(s)).collect()
    }

    /// Direct convolution for short impulse responses
    fn process_direct(&mut self, sample: T) -> T {
        // Shift history and insert new sample
        let mut output = T::zero();
        
        // Current sample contributes with first IR coefficient
        output = output + (sample * self.ir[0]);
        
        // Add contributions from history with remaining IR coefficients
        for i in 1..self.ir.len() {
            let hist_idx = (self.history_pos + i) % self.history.len();
            output = output + (self.history[hist_idx] * self.ir[i]);
        }
        
        // Update history with new sample
        self.history[self.history_pos] = sample;
        self.history_pos = (self.history_pos + 1) % self.history.len();
        
        output
    }

    /// FFT-based convolution using overlap-add method
    fn process_fft(&mut self, sample: T) -> T {
        let fft_size = self.fft_size;
        
        // Add input to buffer
        self.input_buffer[self.input_pos] = sample;
        self.input_pos += 1;

        let mut output = T::zero();

        // When we have enough samples for a frame
        if self.input_pos >= fft_size / 2 {
            if let Some(ref mut fft) = self.fft {
                // Prepare input buffer for FFT
                let mut input_fft = vec![Complex::new(T::zero(), T::zero()); fft_size];
                for (i, &val) in self.input_buffer[..self.input_pos].iter().enumerate() {
                    input_fft[i] = Complex::new(val, T::zero());
                }

                // Forward FFT
                let mut input_freq = vec![Complex::new(T::zero(), T::zero()); fft_size];
                fft.fft(&input_fft, &mut input_freq);

                // Multiply with IR FFT (element-wise)
                for i in 0..fft_size {
                    input_freq[i] = input_freq[i] * self.ir_fft[i];
                }

                // Inverse FFT
                let mut output_time = vec![Complex::new(T::zero(), T::zero()); fft_size];
                fft.ifft(&input_freq, &mut output_time);

                // Overlap-add
                for i in 0..fft_size {
                    self.output_buffer[i] = self.output_buffer[i] + output_time[i].re;
                }

                // Extract output
                output = self.output_buffer[0];

                // Shift output buffer
                for i in 0..fft_size - 1 {
                    self.output_buffer[i] = self.output_buffer[i + 1];
                }
                let output_len = self.output_buffer.len();
                self.output_buffer[output_len - 1] = T::zero();

                // Reset input buffer
                self.input_pos = 0;
                for val in &mut self.input_buffer {
                    *val = T::zero();
                }
            }
        } else {
            // Not enough samples yet, pull from output buffer if available
            if self.output_buffer[0] != T::zero() {
                output = self.output_buffer[0];
                for i in 0..self.output_buffer.len() - 1 {
                    self.output_buffer[i] = self.output_buffer[i + 1];
                }
                let output_len = self.output_buffer.len();
                self.output_buffer[output_len - 1] = T::zero();
            }
        }

        output
    }

    /// Set a new impulse response
    pub fn set_ir(&mut self, ir: Vec<T>) {
        *self = Self::new(ir);
    }

    /// Get the current impulse response
    pub fn get_ir(&self) -> &[T] {
        &self.ir
    }

    /// Get the IR length
    pub fn ir_len(&self) -> usize {
        self.ir.len()
    }

    /// Reset the convolver state
    pub fn reset(&mut self) {
        self.input_pos = 0;
        for val in &mut self.input_buffer {
            *val = T::zero();
        }
        for val in &mut self.output_buffer {
            *val = T::zero();
        }
    }

    /// Helper function to find next power of 2
    fn next_power_of_two(n: usize) -> usize {
        let mut power = 1;
        while power < n {
            power *= 2;
        }
        power
    }
}

impl<T: Float + FromPrimitive> Default for Convolver<T> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_convolver() {
        let mut conv: Convolver<f32> = Convolver::new(Vec::new());
        assert_eq!(conv.process(1.0), 1.0);
    }

    #[test]
    fn test_impulse_response() {
        // IR that just passes signal through
        let ir = vec![1.0];
        let mut conv: Convolver<f32> = Convolver::new(ir);
        assert_eq!(conv.process(1.0), 1.0);
    }

    #[test]
    fn test_convolver_with_ir() {
        // Simple IR: [0.5, 0.5] acts as a simple averaging filter
        let ir = vec![0.5, 0.5];
        let mut conv: Convolver<f32> = Convolver::new(ir);
        conv.process(1.0);
        let _output = conv.process(1.0);
        // Basic smoke test - just ensure it processes without panic
    }
}
