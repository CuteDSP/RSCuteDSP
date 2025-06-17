//! # Signalsmith DSP
//!
//! A Rust port of the Signalsmith DSP C++ library, providing various DSP (Digital Signal Processing)
//! algorithms for audio and signal processing.
//!
//! ## Features
//!
//! - **FFT**: Fast Fourier Transform implementation optimized for sizes that are products of 2^a * 3^b
//! - **Filters**: Biquad filters with various configurations (lowpass, highpass, bandpass, etc.)
//! - **Delay Lines**: Efficient delay line implementation with interpolation
//! - **Curves**: Cubic curve interpolation
//! - **Windows**: Window functions for spectral processing
//! - **Envelopes**: LFOs and envelope generators
//! - **no_std Support**: Can be used in environments without the standard library

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

// Re-export modules
pub mod common;
pub mod curves;
pub mod perf;
pub mod mix;
pub mod rates;
pub mod windows;

pub mod envelopes;


pub mod stretch;

pub mod spectral;

pub mod stft;
pub mod delay;
pub mod fft;
pub mod filters;
