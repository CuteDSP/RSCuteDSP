//! Performance helpers for DSP operations
//!
//! This module provides utilities for optimizing performance in DSP operations,
//! including complex multiplication and denormal handling.

#![allow(unused_imports)]

#[cfg(feature = "std")]
use std::marker::PhantomData;

#[cfg(not(feature = "std"))]
use core::marker::PhantomData;

use num_complex::Complex;

/// Complex multiplication without handling NaN/Infinity
///
/// The standard complex multiplication has edge-cases around NaNs which slow things down
/// and prevent auto-vectorization. This function provides a faster alternative.
#[inline]
pub fn mul<T>(a: Complex<T>, b: Complex<T>) -> Complex<T>
where
    T: num_traits::Float,
{
    Complex::new(
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re,
    )
}

/// Complex multiplication with conjugate of the second argument
#[inline]
pub fn mul_conj<T>(a: Complex<T>, b: Complex<T>) -> Complex<T>
where
    T: num_traits::Float,
{
    Complex::new(
        b.re * a.re + b.im * a.im,
        b.re * a.im - b.im * a.re,
    )
}

/// A utility to stop denormal floating-point values
///
/// This struct sets the CPU flags to flush denormals to zero when created,
/// and restores the original state when dropped.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub struct StopDenormals {
    #[cfg(feature = "std")]
    control_status_register: u32,
    #[cfg(not(feature = "std"))]
    _phantom: PhantomData<()>,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl StopDenormals {
    /// Create a new instance, setting CPU flags to flush denormals to zero
    #[cfg(feature = "std")]
    pub fn new() -> Self {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm_getcsr;
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm_setcsr;
        
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm_getcsr;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm_setcsr;
        
        unsafe {
            let csr = _mm_getcsr();
            _mm_setcsr(csr | 0x8040); // Flush-to-Zero and Denormals-Are-Zero
            Self { control_status_register: csr }
        }
    }
    
    /// Create a new instance in no_std mode (does nothing)
    #[cfg(not(feature = "std"))]
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Drop for StopDenormals {
    #[cfg(feature = "std")]
    fn drop(&mut self) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm_setcsr;
        
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::_mm_setcsr;
        
        unsafe {
            _mm_setcsr(self.control_status_register);
        }
    }
    
    #[cfg(not(feature = "std"))]
    fn drop(&mut self) {
        // Do nothing in no_std mode
    }
}

/// ARM implementation of StopDenormals
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub struct StopDenormals {
    #[cfg(feature = "std")]
    status: usize,
    #[cfg(not(feature = "std"))]
    _phantom: PhantomData<()>,
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
impl StopDenormals {
    /// Create a new instance, setting CPU flags to flush denormals to zero
    #[cfg(feature = "std")]
    pub fn new() -> Self {
        let mut status: usize;
        unsafe {
            asm!(
                "mrs {0}, fpcr",
                out(reg) status
            );
            let new_status = status | 0x01000000; // Flush to Zero
            asm!(
                "msr fpcr, {0}",
                in(reg) new_status
            );
        }
        Self { status }
    }
    
    /// Create a new instance in no_std mode (does nothing)
    #[cfg(not(feature = "std"))]
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
impl Drop for StopDenormals {
    #[cfg(feature = "std")]
    fn drop(&mut self) {
        unsafe {
            asm!(
                "msr fpcr, {0}",
                in(reg) self.status
            );
        }
    }
    
    #[cfg(not(feature = "std"))]
    fn drop(&mut self) {
        // Do nothing in no_std mode
    }
}

/// Fallback implementation for other architectures
#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "arm",
    target_arch = "aarch64"
)))]
pub struct StopDenormals {
    _phantom: PhantomData<()>,
}

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "arm",
    target_arch = "aarch64"
)))]
impl StopDenormals {
    /// Create a new instance (does nothing on unsupported architectures)
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_complex_mul() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        
        let result = mul(a, b);
        let expected = a * b;
        
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_complex_mul_conj() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        
        let result = mul_conj(a, b);
        let expected = a * b.conj();
        
        assert_eq!(result, expected);
    }
}