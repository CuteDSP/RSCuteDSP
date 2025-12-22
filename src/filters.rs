//! Basic filters
//!
//! This module provides implementations of common filter types, primarily biquad filters.

#![allow(unused_imports)]

use core::f64;
#[cfg(feature = "std")]
use std::f32::consts::FRAC_1_SQRT_2;
#[cfg(feature = "std")]
use std::f32::consts::PI;
#[cfg(feature = "std")]
use std::f64::consts::TAU;

#[cfg(not(feature = "std"))]
use core::f32::consts::FRAC_1_SQRT_2;
#[cfg(not(feature = "std"))]
use core::f32::consts::PI;
#[cfg(not(feature = "std"))]
use core::f64::consts::TAU;

use num_traits::{Float, NumCast};

/// Filter design methods.
///
/// These differ mostly in how they handle frequency-warping near Nyquist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiquadDesign {
    /// Bilinear transform, adjusting for centre frequency but not bandwidth
    Bilinear,

    /// RBJ's "Audio EQ Cookbook". Based on `Bilinear`, adjusting bandwidth
    /// (for peak/notch/bandpass) to preserve the ratio between upper/lower boundaries.
    /// This performs oddly near Nyquist.
    Cookbook,

    /// Based on `Bilinear`, adjusting bandwidth to preserve the lower boundary
    /// (leaving the upper one loose).
    OneSided,

    /// From Martin Vicanek's "Matched Second Order Digital Filters".
    /// Falls back to `OneSided` for shelf and allpass filters.
    /// This takes the poles from the impulse-invariant approach, and then picks
    /// the zeros to create a better match. This means that Nyquist is not 0dB
    /// for peak/notch (or -Inf for lowpass), but it is a decent match to the
    /// analogue prototype.
    Vicanek,
}

/// A standard biquad filter.
///
/// This is not guaranteed to be stable if modulated at audio rate.
///
/// The default highpass/lowpass bandwidth produces a Butterworth filter
/// when bandwidth-compensation is disabled.
///
/// Bandwidth compensation defaults to `BiquadDesign::OneSided` (or
/// `BiquadDesign::Cookbook` if `cookbook_bandwidth` is enabled) for all
/// filter types aside from highpass/lowpass (which use `BiquadDesign::Bilinear`).
pub struct Biquad<T: Float> {
    a1: T,
    a2: T,
    b0: T,
    b1: T,
    b2: T,
    x1: T,
    x2: T,
    y1: T,
    y2: T,
    cookbook_bandwidth: bool,
}

/// Filter type for the biquad filter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    Highpass,
    Lowpass,
    HighShelf,
    LowShelf,
    Bandpass,
    Notch,
    Peak,
    Allpass,
}

/// Frequency specification for filter design
struct FreqSpec {
    scaled_freq: f32,
    w0: f32,
    sin_w0: f32,
    cos_w0: f32,
    inv_2q: f32,
}

impl FreqSpec {
    fn new(freq: f32, design: BiquadDesign) -> Self {
        let scaled_freq = freq.max(1e-6).min(0.4999);
        let scaled_freq = if design == BiquadDesign::Cookbook {
            scaled_freq.min(0.45)
        } else {
            scaled_freq
        };

        let w0 = 2.0 * PI * scaled_freq;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();

        Self {
            scaled_freq,
            w0,
            sin_w0,
            cos_w0,
            inv_2q: 0.0,
        }
    }

    fn one_sided_comp_q(&mut self) {
        // Ratio between our (digital) lower boundary f1 and centre f0
        let f1_factor = (self.inv_2q * self.inv_2q + 1.0).sqrt() - self.inv_2q;

        // Bilinear means discrete-time freq f = continuous-time freq tan(pi*xf/pi)
        let ct_f1 = (PI * self.scaled_freq * f1_factor).tan();
        let inv_ct_f0 = (1.0 + self.cos_w0) / self.sin_w0;

        let ct_f1_factor = ct_f1 * inv_ct_f0;
        self.inv_2q = 0.5 / ct_f1_factor - 0.5 * ct_f1_factor;
    }
}

impl<T: Float> Biquad<T> {
    /// Create a new biquad filter
    pub fn new(cookbook_bandwidth: bool) -> Self {
        Self {
            a1: T::zero(),
            a2: T::zero(),
            b0: T::one(),
            b1: T::zero(),
            b2: T::zero(),
            x1: T::zero(),
            x2: T::zero(),
            y1: T::zero(),
            y2: T::zero(),
            cookbook_bandwidth,
        }
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.x1 = T::zero();
        self.x2 = T::zero();
        self.y1 = T::zero();
        self.y2 = T::zero();
    }

    /// Process a single sample through the filter
    pub fn process(&mut self, input: T) -> T {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    pub fn get_mag_response(&self, normalized_freq: f64) -> T {
        let b0_sq = self.b0 * self.b0;
        let b1_sq = self.b1 * self.b1;
        let b2_sq = self.b2 * self.b2;
        let a1_sq = self.a1 * self.a1;
        let a2_sq = self.a2 * self.a2;
        let cos_omega = <T as NumCast>::from((TAU * normalized_freq).cos()).unwrap();
        let cos_2omega = <T as NumCast>::from((2.0 * TAU * normalized_freq).cos()).unwrap();

        let nuemenator = b0_sq
            + b1_sq
            + b2_sq
            + T::from(2.0).unwrap() * cos_omega * (self.b0 * self.b1 + self.b1 * self.b2)
            + T::from(2.0).unwrap() * cos_2omega * (self.b0 * self.b2);
        let denumenator = T::from(1.0).unwrap()
            + a1_sq
            + a2_sq
            + T::from(2.0).unwrap() * cos_omega * (self.a1 + self.a1 * self.a2)
            + T::from(2.0).unwrap() * cos_2omega * self.a2;

        //handle case where we're at nyquist, at such a case the denumnator will be very small and might cause nan
        //upstream to user via option as the limit value changes depending on the filter type
        (nuemenator / denumenator).abs().sqrt()
    }

    /// Process a buffer of samples through the filter
    pub fn process_buffer(&mut self, buffer: &mut [T]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }

    /// Create a frequency specification based on octave bandwidth
    fn octave_spec(scaled_freq: f32, octaves: f32, design: BiquadDesign) -> FreqSpec {
        let mut spec = FreqSpec::new(scaled_freq, design);

        let octaves = if design == BiquadDesign::Cookbook {
            // Approximately preserves bandwidth between halfway points
            octaves * spec.w0 / spec.sin_w0
        } else {
            octaves
        };

        spec.inv_2q = (0.5 * octaves * (2.0f32).ln()).sinh(); // 1/(2Q)

        if design == BiquadDesign::OneSided {
            spec.one_sided_comp_q();
        }

        spec
    }

    /// Create a frequency specification based on Q factor
    fn q_spec(scaled_freq: f32, q: f32, design: BiquadDesign) -> FreqSpec {
        let mut spec = FreqSpec::new(scaled_freq, design);

        spec.inv_2q = 0.5 / q;

        if design == BiquadDesign::OneSided {
            spec.one_sided_comp_q();
        }

        spec
    }

    /// Convert decibels to square root of gain
    fn db_to_sqrt_gain(db: f32) -> f32 {
        10.0f32.powf(db * 0.025)
    }

    /// Configure the filter with the given parameters
    fn configure(
        &mut self,
        filter_type: FilterType,
        spec: FreqSpec,
        sqrt_gain: f32,
        _design: BiquadDesign,
    ) -> &mut Self {
        let sin_w0 = spec.sin_w0;
        let cos_w0 = spec.cos_w0;
        let inv_2q = spec.inv_2q;

        let alpha = sin_w0 * inv_2q;
        let a0_inv = 1.0 / (1.0 + alpha);

        match filter_type {
            FilterType::Lowpass => {
                let b1 = (1.0 - cos_w0) * 0.5;
                self.b0 = <T as NumCast>::from(b1 * a0_inv).unwrap();
                self.b1 = <T as NumCast>::from(2.0 * b1 * a0_inv).unwrap();
                self.b2 = <T as NumCast>::from(b1 * a0_inv).unwrap();
                self.a1 = <T as NumCast>::from(-2.0 * cos_w0 * a0_inv).unwrap();
                self.a2 = <T as NumCast>::from((1.0 - alpha) * a0_inv).unwrap();
            }
            FilterType::Highpass => {
                let b0 = (1.0 + cos_w0) * 0.5;
                self.b0 = <T as NumCast>::from(b0 * a0_inv).unwrap();
                self.b1 = <T as NumCast>::from(-2.0 * b0 * a0_inv).unwrap();
                self.b2 = <T as NumCast>::from(b0 * a0_inv).unwrap();
                self.a1 = <T as NumCast>::from(-2.0 * cos_w0 * a0_inv).unwrap();
                self.a2 = <T as NumCast>::from((1.0 - alpha) * a0_inv).unwrap();
            }
            FilterType::Bandpass => {
                self.b0 = <T as NumCast>::from(alpha * a0_inv).unwrap();
                self.b1 = T::zero();
                self.b2 = <T as NumCast>::from(-alpha * a0_inv).unwrap();
                self.a1 = <T as NumCast>::from(-2.0 * cos_w0 * a0_inv).unwrap();
                self.a2 = <T as NumCast>::from((1.0 - alpha) * a0_inv).unwrap();
            }
            FilterType::Notch => {
                self.b0 = <T as NumCast>::from(a0_inv).unwrap();
                self.b1 = <T as NumCast>::from(-2.0 * cos_w0 * a0_inv).unwrap();
                self.b2 = <T as NumCast>::from(a0_inv).unwrap();
                self.a1 = <T as NumCast>::from(-2.0 * cos_w0 * a0_inv).unwrap();
                self.a2 = <T as NumCast>::from((1.0 - alpha) * a0_inv).unwrap();
            }
            FilterType::Peak => {
                let a = sqrt_gain * sqrt_gain;
                let alpha_a = alpha * a;
                let alpha_div_a = alpha / a;

                self.b0 = <T as NumCast>::from((1.0 + alpha_a) * a0_inv).unwrap();
                self.b1 = <T as NumCast>::from(-2.0 * cos_w0 * a0_inv).unwrap();
                self.b2 = <T as NumCast>::from((1.0 - alpha_a) * a0_inv).unwrap();
                self.a1 = <T as NumCast>::from(-2.0 * cos_w0 * a0_inv).unwrap();
                self.a2 = <T as NumCast>::from((1.0 - alpha_div_a) * a0_inv).unwrap();
            }
            FilterType::LowShelf => {
                let a = sqrt_gain * sqrt_gain;
                let sqrt_a = sqrt_gain;

                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;

                let a0_inv = 1.0 / a0;

                self.b0 = <T as NumCast>::from(b0 * a0_inv).unwrap();
                self.b1 = <T as NumCast>::from(b1 * a0_inv).unwrap();
                self.b2 = <T as NumCast>::from(b2 * a0_inv).unwrap();
                self.a1 = <T as NumCast>::from(a1 * a0_inv).unwrap();
                self.a2 = <T as NumCast>::from(a2 * a0_inv).unwrap();
            }
            FilterType::HighShelf => {
                let a = sqrt_gain * sqrt_gain;
                let sqrt_a = sqrt_gain;

                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;

                let a0_inv = 1.0 / a0;

                self.b0 = <T as NumCast>::from(b0 * a0_inv).unwrap();
                self.b1 = <T as NumCast>::from(b1 * a0_inv).unwrap();
                self.b2 = <T as NumCast>::from(b2 * a0_inv).unwrap();
                self.a1 = <T as NumCast>::from(a1 * a0_inv).unwrap();
                self.a2 = <T as NumCast>::from(a2 * a0_inv).unwrap();
            }
            FilterType::Allpass => {
                self.b0 = <T as NumCast>::from((1.0 - alpha) * a0_inv).unwrap();
                self.b1 = <T as NumCast>::from(-2.0 * cos_w0 * a0_inv).unwrap();
                self.b2 = <T as NumCast>::from((1.0 + alpha) * a0_inv).unwrap();
                self.a1 = <T as NumCast>::from(-2.0 * cos_w0 * a0_inv).unwrap();
                self.a2 = <T as NumCast>::from((1.0 - alpha) * a0_inv).unwrap();
            }
        }

        self
    }

    /// Default bandwidth for highpass/lowpass filters (Butterworth when bandwidth compensation is disabled)

    pub const DEFAULT_BANDWIDTH: f32 = FRAC_1_SQRT_2; // 1/sqrt(2)

    /// Configure a lowpass filter
    pub fn lowpass(&mut self, freq: T, q: T, design: BiquadDesign) -> &mut Self {
        let spec = Self::q_spec(
            <f32 as NumCast>::from(freq).unwrap(),
            <f32 as NumCast>::from(q).unwrap(),
            design,
        );
        self.configure(FilterType::Lowpass, spec, 1.0, design)
    }

    /// Configure a highpass filter
    pub fn highpass(&mut self, freq: T, q: T, design: BiquadDesign) -> &mut Self {
        let spec = Self::q_spec(
            <f32 as NumCast>::from(freq).unwrap(),
            <f32 as NumCast>::from(q).unwrap(),
            design,
        );
        self.configure(FilterType::Highpass, spec, 1.0, design)
    }

    /// Configure a bandpass filter
    pub fn bandpass(&mut self, freq: T, bandwidth_octaves: T) -> &mut Self {
        let bw_design = if self.cookbook_bandwidth {
            BiquadDesign::Cookbook
        } else {
            BiquadDesign::OneSided
        };
        let spec = Self::octave_spec(
            <f32 as NumCast>::from(freq).unwrap(),
            <f32 as NumCast>::from(bandwidth_octaves).unwrap(),
            bw_design,
        );
        self.configure(FilterType::Bandpass, spec, 1.0, bw_design)
    }

    /// Configure a notch filter
    pub fn notch(&mut self, freq: T, bandwidth_octaves: T) -> &mut Self {
        let bw_design = if self.cookbook_bandwidth {
            BiquadDesign::Cookbook
        } else {
            BiquadDesign::OneSided
        };
        let spec = Self::octave_spec(
            <f32 as NumCast>::from(freq).unwrap(),
            <f32 as NumCast>::from(bandwidth_octaves).unwrap(),
            bw_design,
        );
        self.configure(FilterType::Notch, spec, 1.0, bw_design)
    }

    /// Configure a peak filter
    pub fn peak(&mut self, freq: T, bandwidth_octaves: T, gain_db: T) -> &mut Self {
        let bw_design = if self.cookbook_bandwidth {
            BiquadDesign::Cookbook
        } else {
            BiquadDesign::OneSided
        };
        let spec = Self::octave_spec(
            <f32 as NumCast>::from(freq).unwrap(),
            <f32 as NumCast>::from(bandwidth_octaves).unwrap(),
            bw_design,
        );
        let sqrt_gain = Self::db_to_sqrt_gain(<f32 as NumCast>::from(gain_db).unwrap());
        self.configure(FilterType::Peak, spec, sqrt_gain, bw_design)
    }

    /// Configure a low shelf filter
    pub fn low_shelf(&mut self, freq: T, gain_db: T) -> &mut Self {
        let bw_design = if self.cookbook_bandwidth {
            BiquadDesign::Cookbook
        } else {
            BiquadDesign::OneSided
        };
        let mut spec = FreqSpec::new(<f32 as NumCast>::from(freq).unwrap(), bw_design);
        spec.inv_2q = 0.5;
        let sqrt_gain = Self::db_to_sqrt_gain(<f32 as NumCast>::from(gain_db).unwrap());
        self.configure(FilterType::LowShelf, spec, sqrt_gain, bw_design)
    }

    /// Configure a high shelf filter
    pub fn high_shelf(&mut self, freq: T, gain_db: T) -> &mut Self {
        let bw_design = if self.cookbook_bandwidth {
            BiquadDesign::Cookbook
        } else {
            BiquadDesign::OneSided
        };
        let mut spec = FreqSpec::new(<f32 as NumCast>::from(freq).unwrap(), bw_design);
        spec.inv_2q = 0.5;
        let sqrt_gain = Self::db_to_sqrt_gain(<f32 as NumCast>::from(gain_db).unwrap());
        self.configure(FilterType::HighShelf, spec, sqrt_gain, bw_design)
    }

    /// Configure an allpass filter
    pub fn allpass(&mut self, freq: T, q: T) -> &mut Self {
        let bw_design = if self.cookbook_bandwidth {
            BiquadDesign::Cookbook
        } else {
            BiquadDesign::OneSided
        };
        let spec = Self::q_spec(
            <f32 as NumCast>::from(freq).unwrap(),
            <f32 as NumCast>::from(q).unwrap(),
            bw_design,
        );
        self.configure(FilterType::Allpass, spec, 1.0, bw_design)
    }
}

/// A stereo biquad filter
pub struct StereoBiquad<T: Float> {
    left: Biquad<T>,
    right: Biquad<T>,
}

impl<T: Float> StereoBiquad<T> {
    /// Create a new stereo biquad filter
    pub fn new(cookbook_bandwidth: bool) -> Self {
        Self {
            left: Biquad::new(cookbook_bandwidth),
            right: Biquad::new(cookbook_bandwidth),
        }
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
    }

    /// Process a stereo sample through the filter
    pub fn process(&mut self, left: T, right: T) -> (T, T) {
        (self.left.process(left), self.right.process(right))
    }

    /// Process a stereo buffer through the filter
    pub fn process_buffer(&mut self, left: &mut [T], right: &mut [T]) {
        assert_eq!(left.len(), right.len(), "Buffer lengths must match");

        for i in 0..left.len() {
            let (l, r) = self.process(left[i], right[i]);
            left[i] = l;
            right[i] = r;
        }
    }

    /// Configure a lowpass filter
    pub fn lowpass(&mut self, freq: T, q: T, design: BiquadDesign) -> &mut Self {
        self.left.lowpass(freq, q, design);
        self.right.lowpass(freq, q, design);
        self
    }

    /// Configure a highpass filter
    pub fn highpass(&mut self, freq: T, q: T, design: BiquadDesign) -> &mut Self {
        self.left.highpass(freq, q, design);
        self.right.highpass(freq, q, design);
        self
    }

    /// Configure a bandpass filter
    pub fn bandpass(&mut self, freq: T, bandwidth_octaves: T) -> &mut Self {
        self.left.bandpass(freq, bandwidth_octaves);
        self.right.bandpass(freq, bandwidth_octaves);
        self
    }

    /// Configure a notch filter
    pub fn notch(&mut self, freq: T, bandwidth_octaves: T) -> &mut Self {
        self.left.notch(freq, bandwidth_octaves);
        self.right.notch(freq, bandwidth_octaves);
        self
    }

    /// Configure a peak filter
    pub fn peak(&mut self, freq: T, bandwidth_octaves: T, gain_db: T) -> &mut Self {
        self.left.peak(freq, bandwidth_octaves, gain_db);
        self.right.peak(freq, bandwidth_octaves, gain_db);
        self
    }

    /// Configure a low shelf filter
    pub fn low_shelf(&mut self, freq: T, gain_db: T) -> &mut Self {
        self.left.low_shelf(freq, gain_db);
        self.right.low_shelf(freq, gain_db);
        self
    }

    /// Configure a high shelf filter
    pub fn high_shelf(&mut self, freq: T, gain_db: T) -> &mut Self {
        self.left.high_shelf(freq, gain_db);
        self.right.high_shelf(freq, gain_db);
        self
    }

    /// Configure an allpass filter
    pub fn allpass(&mut self, freq: T, q: T) -> &mut Self {
        self.left.allpass(freq, q);
        self.right.allpass(freq, q);
        self
    }
}

/// A direct-form FIR (impulse response) filter
pub struct FIR<T: Float> {
    kernel: Vec<T>,
    buffer: Vec<T>,
    pos: usize,
}

impl<T: Float> FIR<T> {
    /// Create a new FIR filter with the given impulse response (kernel)
    pub fn new(kernel: Vec<T>) -> Self {
        let len = kernel.len();
        Self {
            kernel,
            buffer: vec![T::zero(); len],
            pos: 0,
        }
    }

    /// Reset the filter state (clears the buffer)
    pub fn reset(&mut self) {
        for v in &mut self.buffer {
            *v = T::zero();
        }
        self.pos = 0;
    }

    /// Process a single sample through the FIR filter
    pub fn process(&mut self, input: T) -> T {
        let len = self.kernel.len();
        self.buffer[self.pos] = input;
        let mut acc = T::zero();
        let mut i = self.pos;
        for k in 0..len {
            acc = acc + self.kernel[k] * self.buffer[i];
            if i == 0 {
                i = len - 1;
            } else {
                i -= 1;
            }
        }
        self.pos = (self.pos + 1) % len;
        acc
    }

    /// Process a buffer of samples through the FIR filter (in-place)
    pub fn process_buffer(&mut self, buffer: &mut [T]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }

    /// Get the kernel (impulse response)
    pub fn kernel(&self) -> &[T] {
        &self.kernel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_lowpass() {
        let mut filter = Biquad::<f32>::new(false);
        filter.lowpass(0.25, 0.7071, BiquadDesign::Bilinear);

        // Process DC signal to reach steady state
        let mut output = 0.0;
        for _ in 0..100 {
            output = filter.process(1.0);
        }
        // DC should pass through
        assert!((output - 1.0).abs() < 1e-6);

        // Reset and test with a sine wave at Nyquist
        filter.reset();
        for _ in 0..100 {
            // Alternating 1.0, -1.0 (Nyquist frequency)
            filter.process(1.0);
            output = filter.process(-1.0);
        }

        // Nyquist should be heavily attenuated
        assert!(output.abs() < 0.1);
    }

    #[test]
    fn test_mag_response_sanity() {
        let mut filter = Biquad::<f32>::new(false);

        filter.allpass(0.25, 1.0);
        let resp = filter.get_mag_response(0.1);

        assert!((resp - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mag_response_6db_point() {
        let mut filter = Biquad::<f64>::new(false);
        let test_freq = 0.1;

        filter.lowpass(0.1, 0.5, BiquadDesign::Bilinear);

        let resp = filter.get_mag_response(test_freq);
        let expected = 0.5;
        assert!((resp - expected).abs() < 1e-6);
    }

    #[test]
    fn test_mag_response_low_cutoff() {
        let mut filter = Biquad::<f64>::new(false);
        let test_freq = 1.0/480.0;

        filter.lowpass(1.0/480.0, 0.5, BiquadDesign::Bilinear);

        
        
        let resp = filter.get_mag_response(test_freq);

        let expected = 0.5;
        assert!((resp - expected).abs() < 1e-3);
    }


    #[test]
    fn test_mag_response_notch() {
        let mut filter = Biquad::<f32>::new(false);

        filter.notch(1000.0 / 48000.0, 1.0);
        let res = filter.get_mag_response(996.539611 / 48000.0);

        assert!(!res.is_nan())
    }

    #[test]
    fn test_biquad_highpass() {
        let mut filter = Biquad::<f32>::new(false);
        filter.highpass(0.25, 0.7071, BiquadDesign::Bilinear);

        // DC should be blocked
        let mut output = 1.0;
        for _ in 0..100 {
            output = filter.process(1.0);
        }
        assert!(output.abs() < 0.1);

        // Reset and test with a sine wave at Nyquist
        filter.reset();
        for _ in 0..100 {
            // Alternating 1.0, -1.0 (Nyquist frequency)
            filter.process(1.0);
            output = filter.process(-1.0);
        }

        // Nyquist should pass through
        assert!(output.abs() > 0.9);
    }

    #[test]
    fn test_stereo_biquad() {
        let mut filter = StereoBiquad::<f32>::new(false);
        filter.lowpass(0.25, 0.7071, BiquadDesign::Bilinear);

        // Process multiple samples to reach steady state
        let (mut left, mut right) = (0.0, 0.0);
        for _ in 0..100 {
            (left, right) = filter.process(1.0, 0.5);
        }

        // Both channels should be processed similarly
        assert!((left - 1.0).abs() < 1e-6);
        assert!((right - 0.5).abs() < 1e-6);

        // Test buffer processing
        filter.reset();
        let mut left_buffer = vec![1.0; 100];
        let mut right_buffer = vec![0.5; 100];

        filter.process_buffer(&mut left_buffer, &mut right_buffer);

        // Check last samples after stabilization
        assert!((left_buffer[99] - 1.0).abs() < 1e-6);
        assert!((right_buffer[99] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_fir_filter() {
        use super::FIR;
        // Impulse response: [1.0, 0.5, 0.25]
        let kernel = vec![1.0, 0.5, 0.25];
        let mut fir = FIR::new(kernel.clone());
        fir.reset();
        // Feed an impulse (1.0, then zeros)
        let mut output = vec![];
        output.push(fir.process(1.0));
        output.push(fir.process(0.0));
        output.push(fir.process(0.0));
        // Should match the kernel
        for (o, k) in output.iter().zip(kernel.iter()) {
            assert!((o - k).abs() < 1e-6);
        }
        // Test moving average (box filter)
        let kernel = vec![1.0 / 3.0; 3];
        let mut fir = FIR::new(kernel);
        fir.reset();
        let input = vec![3.0, 3.0, 3.0, 3.0];
        let mut output = vec![];
        for x in input {
            output.push(fir.process(x));
        }
        // After initial fill, output should be 3.0
        assert!((output[2] - 3.0).abs() < 1e-6);
        assert!((output[3] - 3.0).abs() < 1e-6);
    }
}
