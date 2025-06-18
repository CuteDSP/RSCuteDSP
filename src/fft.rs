//! Fast Fourier Transform implementation
//!
//! This module provides FFT implementations optimized for sizes that are products of 2^a * 3^b.
//! It includes both complex and real FFT implementations with various optimizations.

#![allow(unused_imports)]

#[cfg(feature = "std")]
use std::{f64::consts::PI, vec::Vec};


#[cfg(not(feature = "std"))]
use core::f64::consts::PI;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

use num_complex::Complex;
use num_traits::Float;
use num_traits::FromPrimitive;

use crate::perf;

/// Helper functions for complex multiplication and data interleaving
mod helpers {
    use super::*;

    /// Complex multiplication
    pub fn complex_mul<T: Float>(
        a: &mut [Complex<T>],
        b: &[Complex<T>],
        c: &[Complex<T>],
        size: usize,
    ) {
        for i in 0..size {
            let bi = b[i];
            let ci = c[i];
            a[i] = Complex::new(
                bi.re * ci.re - bi.im * ci.im,
                bi.im * ci.re + bi.re * ci.im,
            );
        }
    }

    /// Complex multiplication with conjugate of second argument
    pub fn complex_mul_conj<T: Float>(
        a: &mut [Complex<T>],
        b: &[Complex<T>],
        c: &[Complex<T>],
        size: usize,
    ) {
        for i in 0..size {
            let bi = b[i];
            let ci = c[i];
            a[i] = Complex::new(
                bi.re * ci.re + bi.im * ci.im,
                bi.im * ci.re - bi.re * ci.im,
            );
        }
    }

    /// Complex multiplication with split complex representation
    pub fn complex_mul_split<T: Float>(
        ar: &mut [T],
        ai: &mut [T],
        br: &[T],
        bi: &[T],
        cr: &[T],
        ci: &[T],
        size: usize,
    ) {
        for i in 0..size {
            let rr = br[i] * cr[i] - bi[i] * ci[i];
            let ri = br[i] * ci[i] + bi[i] * cr[i];
            ar[i] = rr;
            ai[i] = ri;
        }
    }

    /// Complex multiplication with conjugate and split complex representation
    pub fn complex_mul_conj_split<T: Float>(
        ar: &mut [T],
        ai: &mut [T],
        br: &[T],
        bi: &[T],
        cr: &[T],
        ci: &[T],
        size: usize,
    ) {
        for i in 0..size {
            let rr = cr[i] * br[i] + ci[i] * bi[i];
            let ri = cr[i] * bi[i] - ci[i] * br[i];
            ar[i] = rr;
            ai[i] = ri;
        }
    }

    /// Interleave copy with fixed stride
    pub fn interleave_copy<T: Copy>(a: &[T], b: &mut [T], a_stride: usize, b_stride: usize) {
        for bi in 0..b_stride {
            for ai in 0..a_stride {
                b[bi + ai * b_stride] = a[bi * a_stride + ai];
            }
        }
    }

    /// Interleave copy with split complex representation
    pub fn interleave_copy_split<T: Copy>(
        a_real: &[T],
        a_imag: &[T],
        b_real: &mut [T],
        b_imag: &mut [T],
        a_stride: usize,
        b_stride: usize,
    ) {
        for bi in 0..b_stride {
            for ai in 0..a_stride {
                b_real[bi + ai * b_stride] = a_real[bi * a_stride + ai];
                b_imag[bi + ai * b_stride] = a_imag[bi * a_stride + ai];
            }
        }
    }
}

/// A simple and portable power-of-2 FFT implementation
pub struct SimpleFFT<T: Float> {
    twiddles: Vec<Complex<T>>,
    working: Vec<Complex<T>>,
}

impl<T: Float + FromPrimitive> SimpleFFT<T> {
    /// Create a new FFT with the specified size
    pub fn new(size: usize) -> Self {
        let mut result = Self {
            twiddles: Vec::new(),
            working: Vec::new(),
        };
        result.resize(size);
        result
    }

    /// Resize the FFT to handle a different size
    pub fn resize(&mut self, size: usize) {
        self.twiddles.resize(size * 3 / 4, Complex::new(T::zero(), T::zero()));
        for i in 0..self.twiddles.len() {
            let twiddle_phase = -T::from_f64(2.0).unwrap() * T::from_f64(PI as f64).unwrap() * T::from_f64(i as f64).unwrap() / T::from_f64(size as f64).unwrap();
            self.twiddles[i] = Complex::new(
                twiddle_phase.cos(),
                twiddle_phase.sin(),
            );
        }
        self.working.resize(size, Complex::new(T::zero(), T::zero()));
    }

    /// Perform a forward FFT
    pub fn fft(&self, time: &[Complex<T>], freq: &mut [Complex<T>]) {
        let size = self.working.len();
        if size <= 1 {
            if size == 1 {
                freq[0] = time[0];
            }
            return;
        }
        self.fft_pass::<false>(size, 1, time, freq, &mut self.working.clone());
    }

    /// Perform an inverse FFT
    pub fn ifft(&self, freq: &[Complex<T>], time: &mut [Complex<T>]) {
        let size = self.working.len();
        if size <= 1 {
            if size == 1 {
                time[0] = freq[0];
            }
            return;
        }
        self.fft_pass::<true>(size, 1, freq, time, &mut self.working.clone());
    }

    /// Perform a forward FFT with split complex representation
    pub fn fft_split(&self, in_r: &[T], in_i: &[T], out_r: &mut [T], out_i: &mut [T]) {
        let size = self.working.len();
        if size <= 1 {
            if size == 1 {
                out_r[0] = in_r[0];
                out_i[0] = in_i[0];
            }
            return;
        }
        
        // Create temporary buffers for working space
        let mut working_r = vec![T::zero(); size];
        let mut working_i = vec![T::zero(); size];
        
        self.fft_pass_split::<false>(size, 1, in_r, in_i, out_r, out_i, &mut working_r, &mut working_i);
    }

    /// Perform an inverse FFT with split complex representation
    pub fn ifft_split(&self, in_r: &[T], in_i: &[T], out_r: &mut [T], out_i: &mut [T]) {
        let size = self.working.len();
        if size <= 1 {
            if size == 1 {
                out_r[0] = in_r[0];
                out_i[0] = in_i[0];
            }
            return;
        }
        
        // Create temporary buffers for working space
        let mut working_r = vec![T::zero(); size];
        let mut working_i = vec![T::zero(); size];
        
        self.fft_pass_split::<true>(size, 1, in_r, in_i, out_r, out_i, &mut working_r, &mut working_i);
    }

    // Internal implementation of FFT pass
    fn fft_pass<const INVERSE: bool>(
        &self,
        size: usize,
        stride: usize,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        working: &mut [Complex<T>],
    ) {
        if size / 4 > 1 {
            // Calculate four quarter-size FFTs
            self.fft_pass::<INVERSE>(size / 4, stride * 4, input, working, output);
            self.combine4::<INVERSE>(size, stride, working, output);
        } else if size == 4 {
            self.combine4::<INVERSE>(4, stride, input, output);
        } else {
            // 2-point FFT
            for s in 0..stride {
                let a = input[s];
                let b = input[s + stride];
                output[s] = a + b;
                output[s + stride] = a - b;
            }
        }
    }

    // Internal implementation of FFT pass with split complex representation
    fn fft_pass_split<const INVERSE: bool>(
        &self,
        size: usize,
        stride: usize,
        in_r: &[T],
        in_i: &[T],
        out_r: &mut [T],
        out_i: &mut [T],
        working_r: &mut [T],
        working_i: &mut [T],
    ) {
        if size / 4 > 1 {
            // Calculate four quarter-size FFTs
            self.fft_pass_split::<INVERSE>(
                size / 4,
                stride * 4,
                in_r,
                in_i,
                working_r,
                working_i,
                out_r,
                out_i,
            );
            self.combine4_split::<INVERSE>(size, stride, working_r, working_i, out_r, out_i);
        } else if size == 4 {
            self.combine4_split::<INVERSE>(4, stride, in_r, in_i, out_r, out_i);
        } else {
            // 2-point FFT
            for s in 0..stride {
                let ar = in_r[s];
                let ai = in_i[s];
                let br = in_r[s + stride];
                let bi = in_i[s + stride];
                out_r[s] = ar + br;
                out_i[s] = ai + bi;
                out_r[s + stride] = ar - br;
                out_i[s + stride] = ai - bi;
            }
        }
    }

    // Combine interleaved results into a single spectrum
    fn combine4<const INVERSE: bool>(
        &self,
        size: usize,
        stride: usize,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
    ) {
        let twiddle_step = self.working.len() / size;
        
        for i in 0..size / 4 {
            let twiddle_b = self.twiddles[i * twiddle_step];
            let twiddle_c = self.twiddles[i * 2 * twiddle_step];
            let twiddle_d = self.twiddles[i * 3 * twiddle_step];
            
            let input_a = &input[4 * i * stride..];
            let input_b = &input[(4 * i + 1) * stride..];
            let input_c = &input[(4 * i + 2) * stride..];
            let input_d = &input[(4 * i + 3) * stride..];
            
            let (output_first_half, output_second_half) = output.split_at_mut((size / 4 * 2) * stride);
            let (output_a_chunk, output_b_chunk) = output_first_half.split_at_mut((size / 4) * stride);
            let (output_c_chunk, output_d_chunk) = output_second_half.split_at_mut((size / 4) * stride);

            let output_a = &mut output_a_chunk[i * stride..];
            let output_b = &mut output_b_chunk[i * stride..];
            let output_c = &mut output_c_chunk[i * stride..];
            let output_d = &mut output_d_chunk[i * stride..];
            
            for s in 0..stride {
                let a = input_a[s];
                let b = if INVERSE {
                    Complex::new(
                        input_b[s].re * twiddle_b.re + input_b[s].im * twiddle_b.im,
                        input_b[s].im * twiddle_b.re - input_b[s].re * twiddle_b.im,
                    )
                } else {
                    Complex::new(
                        input_b[s].re * twiddle_b.re - input_b[s].im * twiddle_b.im,
                        input_b[s].im * twiddle_b.re + input_b[s].re * twiddle_b.im,
                    )
                };
                
                let c = if INVERSE {
                    Complex::new(
                        input_c[s].re * twiddle_c.re + input_c[s].im * twiddle_c.im,
                        input_c[s].im * twiddle_c.re - input_c[s].re * twiddle_c.im,
                    )
                } else {
                    Complex::new(
                        input_c[s].re * twiddle_c.re - input_c[s].im * twiddle_c.im,
                        input_c[s].im * twiddle_c.re + input_c[s].re * twiddle_c.im,
                    )
                };
                
                let d = if INVERSE {
                    Complex::new(
                        input_d[s].re * twiddle_d.re + input_d[s].im * twiddle_d.im,
                        input_d[s].im * twiddle_d.re - input_d[s].re * twiddle_d.im,
                    )
                } else {
                    Complex::new(
                        input_d[s].re * twiddle_d.re - input_d[s].im * twiddle_d.im,
                        input_d[s].im * twiddle_d.re + input_d[s].re * twiddle_d.im,
                    )
                };
                
                let ac0 = a + c;
                let ac1 = a - c;
                let bd0 = b + d;
                let bd1 = if INVERSE { b - d } else { d - b };
                let bd1i = Complex::new(-bd1.im, bd1.re);
                
                output_a[s] = ac0 + bd0;
                output_b[s] = ac1 + bd1i;
                output_c[s] = ac0 - bd0;
                output_d[s] = ac1 - bd1i;
            }
        }
    }

    // Combine interleaved results into a single spectrum with split complex representation
    fn combine4_split<const INVERSE: bool>(
        &self,
        size: usize,
        stride: usize,
        input_r: &[T],
        input_i: &[T],
        output_r: &mut [T],
        output_i: &mut [T],
    ) {
        let twiddle_step = self.working.len() / size;
        
        for i in 0..size / 4 {
            let twiddle_b = self.twiddles[i * twiddle_step];
            let twiddle_c = self.twiddles[i * 2 * twiddle_step];
            let twiddle_d = self.twiddles[i * 3 * twiddle_step];
            
            for s in 0..stride {
                // Get input values
                let a_r = input_r[4 * i * stride + s];
                let a_i = input_i[4 * i * stride + s];
                
                let b_r = input_r[(4 * i + 1) * stride + s];
                let b_i = input_i[(4 * i + 1) * stride + s];
                
                let c_r = input_r[(4 * i + 2) * stride + s];
                let c_i = input_i[(4 * i + 2) * stride + s];
                
                let d_r = input_r[(4 * i + 3) * stride + s];
                let d_i = input_i[(4 * i + 3) * stride + s];
                
                // Apply twiddle factors
                let (b_r_tw, b_i_tw) = if INVERSE {
                    (
                        b_r * twiddle_b.re + b_i * twiddle_b.im,
                        b_i * twiddle_b.re - b_r * twiddle_b.im,
                    )
                } else {
                    (
                        b_r * twiddle_b.re - b_i * twiddle_b.im,
                        b_i * twiddle_b.re + b_r * twiddle_b.im,
                    )
                };
                
                let (c_r_tw, c_i_tw) = if INVERSE {
                    (
                        c_r * twiddle_c.re + c_i * twiddle_c.im,
                        c_i * twiddle_c.re - c_r * twiddle_c.im,
                    )
                } else {
                    (
                        c_r * twiddle_c.re - c_i * twiddle_c.im,
                        c_i * twiddle_c.re + c_r * twiddle_c.im,
                    )
                };
                
                let (d_r_tw, d_i_tw) = if INVERSE {
                    (
                        d_r * twiddle_d.re + d_i * twiddle_d.im,
                        d_i * twiddle_d.re - d_r * twiddle_d.im,
                    )
                } else {
                    (
                        d_r * twiddle_d.re - d_i * twiddle_d.im,
                        d_i * twiddle_d.re + d_r * twiddle_d.im,
                    )
                };
                
                // Butterfly calculations
                let ac0_r = a_r + c_r_tw;
                let ac0_i = a_i + c_i_tw;
                let ac1_r = a_r - c_r_tw;
                let ac1_i = a_i - c_i_tw;
                
                let bd0_r = b_r_tw + d_r_tw;
                let bd0_i = b_i_tw + d_i_tw;
                
                let (bd1_r, bd1_i) = if INVERSE {
                    (b_r_tw - d_r_tw, b_i_tw - d_i_tw)
                } else {
                    (d_r_tw - b_r_tw, d_i_tw - b_i_tw)
                };
                
                let bd1i_r = -bd1_i;
                let bd1i_i = bd1_r;
                
                // Store results
                output_r[i * stride + s] = ac0_r + bd0_r;
                output_i[i * stride + s] = ac0_i + bd0_i;
                
                output_r[(i + size / 4) * stride + s] = ac1_r + bd1i_r;
                output_i[(i + size / 4) * stride + s] = ac1_i + bd1i_i;
                
                output_r[(i + size / 4 * 2) * stride + s] = ac0_r - bd0_r;
                output_i[(i + size / 4 * 2) * stride + s] = ac0_i - bd0_i;
                
                output_r[(i + size / 4 * 3) * stride + s] = ac1_r - bd1i_r;
                output_i[(i + size / 4 * 3) * stride + s] = ac1_i - bd1i_i;
            }
        }
    }
}

/// A wrapper for complex FFT to handle real data
pub struct SimpleRealFFT<T: Float> {
    complex_fft: SimpleFFT<T>,
    tmp_time: Vec<Complex<T>>,
    tmp_freq: Vec<Complex<T>>,
}

impl<T: Float + num_traits::FromPrimitive> SimpleRealFFT<T> {
    /// Create a new real FFT with the specified size
    pub fn new(size: usize) -> Self {
        let mut result = Self {
            complex_fft: SimpleFFT::new(size),
            tmp_time: Vec::new(),
            tmp_freq: Vec::new(),
        };
        result.resize(size);
        result
    }

    /// Resize the FFT to handle a different size
    pub fn resize(&mut self, size: usize) {
        self.complex_fft.resize(size);
        self.tmp_time.resize(size, Complex::new(T::zero(), T::zero()));
        self.tmp_freq.resize(size, Complex::new(T::zero(), T::zero()));
    }

    /// Perform a forward FFT on real data
    pub fn fft(&mut self, time: &[T], freq: &mut [Complex<T>]) {
        let size = self.tmp_time.len();
        
        // Copy real data to complex buffer
        for i in 0..size {
            self.tmp_time[i] = Complex::new(time[i], T::zero());
        }
        
        // Perform complex FFT
        self.complex_fft.fft(&self.tmp_time, &mut self.tmp_freq.clone());
        
        // Extract the result (only half the spectrum is needed due to symmetry)
        for i in 0..size / 2 {
            freq[i] = self.tmp_freq[i];
        }
        
        // Special case for DC and Nyquist
        freq[0] = Complex::new(
            self.tmp_freq[0].re,
            self.tmp_freq[size / 2].re,
        );
    }

    /// Perform a forward FFT on real data with split output
    pub fn fft_split(&self, in_r: &[T], out_r: &mut [T], out_i: &mut [T]) {
        let size = self.tmp_time.len();
        
        // Create temporary zero buffer for imaginary part
        let mut tmp_i = vec![T::zero(); size];
        
        // Perform complex FFT with split representation
        self.complex_fft.fft_split(in_r, &tmp_i, out_r, out_i);
        
        // Special case for Nyquist frequency
        out_i[0] = out_r[size / 2];
    }

    /// Perform an inverse FFT to real data
    pub fn ifft(&mut self, freq: &[Complex<T>], time: &mut [T]) {
        let size = self.tmp_freq.len();
        
        // DC component
        self.tmp_freq[0] = Complex::new(freq[0].re, T::zero());
        
        // Nyquist component
        self.tmp_freq[size / 2] = Complex::new(freq[0].im, T::zero());
        
        // Fill the rest of the spectrum using conjugate symmetry
        for i in 1..size / 2 {
            self.tmp_freq[i] = freq[i];
            self.tmp_freq[size - i] = freq[i].conj();
        }
        
        // Perform inverse complex FFT
        self.complex_fft.ifft(&self.tmp_freq, &mut self.tmp_time.clone());
        
        // Extract real part
        for i in 0..size {
            time[i] = self.tmp_time[i].re;
        }
    }

    /// Perform an inverse FFT from split complex to real data
    pub fn ifft_split(&self, in_r: &[T], in_i: &[T], out_r: &mut [T]) {
        let size = self.tmp_freq.len();
        
        // Create temporary buffers for the full spectrum
        let mut tmp_freq_r = vec![T::zero(); size];
        let mut tmp_freq_i = vec![T::zero(); size];
        
        // DC component
        tmp_freq_r[0] = in_r[0];
        tmp_freq_i[0] = T::zero();
        
        // Nyquist component
        tmp_freq_r[size / 2] = in_i[0];
        tmp_freq_i[size / 2] = T::zero();
        
        // Fill the rest of the spectrum using conjugate symmetry
        for i in 1..size / 2 {
            tmp_freq_r[i] = in_r[i];
            tmp_freq_i[i] = in_i[i];
            tmp_freq_r[size - i] = in_r[i];
            tmp_freq_i[size - i] = -in_i[i];
        }
        
        // Create temporary buffer for imaginary output (will be discarded)
        let mut tmp_out_i = vec![T::zero(); size];
        
        // Perform inverse complex FFT
        self.complex_fft.ifft_split(&tmp_freq_r, &tmp_freq_i, out_r, &mut tmp_out_i);
    }
}

/// A power-of-2 FFT implementation that can be specialized for different platforms
pub struct Pow2FFT<T: Float> {
    simple_fft: SimpleFFT<T>,
    tmp: Vec<Complex<T>>,
}

impl<T: Float+ FromPrimitive> Pow2FFT<T> {
    /// Whether this FFT implementation is faster when given split-complex inputs
    pub const PREFERS_SPLIT: bool = true;

    /// Create a new FFT with the specified size
    pub fn new(size: usize) -> Self {
        let mut result = Self {
            simple_fft: SimpleFFT::new(size),
            tmp: Vec::new(),
        };
        result.resize(size);
        result
    }

    /// Resize the FFT to handle a different size
    pub fn resize(&mut self, size: usize) {
        self.simple_fft.resize(size);
        self.tmp.resize(size, Complex::new(T::zero(), T::zero()));
    }

    /// Perform a forward FFT
    pub fn fft(&self, time: &[Complex<T>], freq: &mut [Complex<T>]) {
        self.simple_fft.fft(time, freq);
    }

    /// Perform a forward FFT with split complex representation
    pub fn fft_split(&self, in_r: &[T], in_i: &[T], out_r: &mut [T], out_i: &mut [T]) {
        self.simple_fft.fft_split(in_r, in_i, out_r, out_i);
    }

    /// Perform an inverse FFT
    pub fn ifft(&self, freq: &[Complex<T>], time: &mut [Complex<T>]) {
        self.simple_fft.ifft(freq, time);
    }

    /// Perform an inverse FFT with split complex representation
    pub fn ifft_split(&self, in_r: &[T], in_i: &[T], out_r: &mut [T], out_i: &mut [T]) {
        self.simple_fft.ifft_split(in_r, in_i, out_r, out_i);
    }
}

/// A power-of-2 real FFT implementation
pub struct Pow2RealFFT<T: Float> {
    simple_real_fft: SimpleRealFFT<T>,
}

impl<T: Float + FromPrimitive> Pow2RealFFT<T> {
    /// Whether this FFT implementation is faster when given split-complex inputs
    pub const PREFERS_SPLIT: bool = Pow2FFT::<T>::PREFERS_SPLIT;

    /// Create a new real FFT with the specified size
    pub fn new(size: usize) -> Self {
        Self {
            simple_real_fft: SimpleRealFFT::new(size),
        }
    }

    /// Resize the FFT to handle a different size
    pub fn resize(&mut self, size: usize) {
        self.simple_real_fft.resize(size);
    }

    /// Perform a forward FFT on real data
    pub fn fft(&mut self, time: &[T], freq: &mut [Complex<T>]) {
        self.simple_real_fft.fft(time, freq);
    }

    /// Perform a forward FFT on real data with split output
    pub fn fft_split(&self, in_r: &[T], out_r: &mut [T], out_i: &mut [T]) {
        self.simple_real_fft.fft_split(in_r, out_r, out_i);
    }

    /// Perform an inverse FFT to real data
    pub fn ifft(&mut self, freq: &[Complex<T>], time: &mut [T]) {
        self.simple_real_fft.ifft(freq, time);
    }

    /// Perform an inverse FFT from split complex to real data
    pub fn ifft_split(&self, in_r: &[T], in_i: &[T], out_r: &mut [T]) {
        self.simple_real_fft.ifft_split(in_r, in_i, out_r);
    }
}

#[cfg(test)]
mod tests {
    use num_complex::ComplexFloat;
    use super::*;
    
    #[test]
    fn test_simple_fft() {
        // Create a 4-point FFT
        let fft = SimpleFFT::<f32>::new(4);
        
        // Create input and output buffers
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let mut output = vec![Complex::new(0.0, 0.0); 4];
        
        // Perform forward FFT
        fft.fft(&input, &mut output);
        
        // All values should be 1.0 for a delta function input
        for i in 0..4 {
            assert!((output[i].re - 1.0).abs() < 1e-10);
            assert!(output[i].im.abs() < 1e-10);
        }
        
        // Create a new input with a sine wave
        let input = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-1.0, 0.0),
        ];
        
        // Perform forward FFT
        fft.fft(&input, &mut output);
        
        // For this input, we should have energy at frequency bin 1
        assert!(output[0].abs() < 1e-10);
        assert!((output[1].im + 2.0).abs() < 1e-10);
        assert!(output[2].abs() < 1e-10);
        assert!((output[3].im - 2.0).abs() < 1e-10);
        
        // Test inverse FFT
        let mut inverse_output = vec![Complex::new(0.0, 0.0); 4];
        fft.ifft(&output, &mut inverse_output);
        
        // Scale by 1/N
        for i in 0..4 {
            inverse_output[i] = inverse_output[i] / 4.0;
        }
        
        // Should recover the original signal
        for i in 0..4 {
            assert!((inverse_output[i].re - input[i].re).abs() < 1e-10);
            assert!((inverse_output[i].im - input[i].im).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_real_fft() {
        // Create an 8-point real FFT
        let mut real_fft = SimpleRealFFT::<f32>::new(8);
        
        // Create input and output buffers
        let input = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut output = vec![Complex::new(0.0, 0.0); 5]; // Only need N/2+1 for real FFT
        
        // Perform forward FFT
        real_fft.fft(&input, &mut output);
        
        // All values should be 1.0 for a delta function input
        for i in 0..5 {
            assert!((output[i].re - 1.0).abs() < 1e-10);
            assert!(output[i].im.abs() < 1e-10);
        }
        
        // Test inverse FFT
        let mut inverse_output = vec![0.0; 8];
        real_fft.ifft(&output, &mut inverse_output);
        
        // Scale by 1/N
        for i in 0..8 {
            inverse_output[i] /= 8.0;
        }
        
        // Should recover the original signal
        for i in 0..8 {
            assert!((inverse_output[i] - input[i]).abs() < 1e-10);
        }
    }
}