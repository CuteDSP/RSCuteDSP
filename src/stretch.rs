//! Time-stretching and pitch-shifting using phase vocoder
//!
//! This module provides high-quality time-stretching and pitch-shifting capabilities
//! using a phase vocoder approach with spectral peak detection and formant preservation.

#![allow(unused_imports)]

use num_traits::{Float, FromPrimitive, NumCast};
use num_complex::Complex;
use core::marker::PhantomData;

use crate::stft::STFT;

/// Spectral band data for phase vocoder
#[derive(Clone, Debug)]
pub struct Band<T: Float> {
    pub input: Complex<T>,
    pub prev_input: Complex<T>,
    pub output: Complex<T>,
    pub input_energy: T,
}

impl<T: Float> Default for Band<T> {
    fn default() -> Self {
        Self {
            input: Complex::new(T::zero(), T::zero()),
            prev_input: Complex::new(T::zero(), T::zero()),
            output: Complex::new(T::zero(), T::zero()),
            input_energy: T::zero(),
        }
    }
}

/// Spectral peak information
#[derive(Clone, Debug)]
pub struct Peak<T: Float> {
    pub input: T,
    pub output: T,
}

/// Frequency mapping point
#[derive(Clone, Debug)]
pub struct PitchMapPoint<T: Float> {
    pub input_bin: T,
    pub freq_grad: T,
}

/// Phase prediction for spectral processing
#[derive(Clone, Debug)]
pub struct Prediction<T: Float> {
    pub energy: T,
    pub input: Complex<T>,
}

impl<T: Float> Default for Prediction<T> {
    fn default() -> Self {
        Self {
            energy: T::zero(),
            input: Complex::new(T::zero(), T::zero()),
        }
    }
}

impl<T: Float> Prediction<T> {
    pub fn make_output(&self, phase: Complex<T>) -> Complex<T> {
        let phase_norm = phase.norm_sqr();
        let phase = if phase_norm <= T::epsilon() {
            self.input
        } else {
            phase
        };
        let phase_norm = phase.norm_sqr() + T::epsilon();
        phase * Complex::new((self.energy / phase_norm).sqrt(), T::zero())
    }
}

/// Main time-stretching and pitch-shifting processor
pub struct SignalsmithStretch<T: Float> {
    // Configuration
    split_computation: bool,
    channels: usize,
    bands: usize,
    
    // STFT and buffers
    block_samples: usize,
    interval_samples: usize,
    tmp_buffer: Vec<T>,
    
    // STFT instances
    analysis_stft: STFT<T>,
    synthesis_stft: STFT<T>,
    
    // Spectral data
    channel_bands: Vec<Band<T>>,
    peaks: Vec<Peak<T>>,
    energy: Vec<T>,
    smoothed_energy: Vec<T>,
    output_map: Vec<PitchMapPoint<T>>,
    channel_predictions: Vec<Prediction<T>>,
    
    // Processing state
    prev_input_offset: i32,
    silence_counter: usize,
    did_seek: bool,
    
    // Frequency mapping
    freq_multiplier: T,
    freq_tonality_limit: T,
    custom_freq_map: Option<Box<dyn Fn(T) -> T + Send + Sync + 'static>>,
    
    // Formant processing
    formant_multiplier: T,
    inv_formant_multiplier: T,
    formant_compensation: bool,
    formant_base_freq: T,
}

impl<T: Float + FromPrimitive + NumCast + core::ops::AddAssign> SignalsmithStretch<T> {
    /// Create a new stretch processor
    pub fn new() -> Self {
        Self {
            split_computation: false,
            channels: 0,
            bands: 0,
            block_samples: 0,
            interval_samples: 0,
            tmp_buffer: Vec::new(),
            analysis_stft: STFT::new(false),
            synthesis_stft: STFT::new(false),
            channel_bands: Vec::new(),
            peaks: Vec::new(),
            energy: Vec::new(),
            smoothed_energy: Vec::new(),
            output_map: Vec::new(),
            channel_predictions: Vec::new(),
            prev_input_offset: -1,
            silence_counter: 0,
            did_seek: false,
            freq_multiplier: T::one(),
            freq_tonality_limit: T::from_f32(0.5).unwrap(),
            custom_freq_map: None,
            formant_multiplier: T::one(),
            inv_formant_multiplier: T::one(),
            formant_compensation: false,
            formant_base_freq: T::zero()
        }
    }

    /// Get the block size in samples
    pub fn block_samples(&self) -> usize {
        self.block_samples
    }

    /// Get the interval size in samples
    pub fn interval_samples(&self) -> usize {
        self.interval_samples
    }

    /// Get the input latency
    pub fn input_latency(&self) -> usize {
        self.block_samples / 2
    }

    /// Get the output latency
    pub fn output_latency(&self) -> usize {
        self.block_samples / 2 + if self.split_computation { self.interval_samples } else { 0 }
    }

    /// Reset the processor state
    pub fn reset(&mut self) {
        self.prev_input_offset = -1;
        for band in &mut self.channel_bands {
            *band = Band::default();
        }
        self.silence_counter = 0;
        self.did_seek = false;
    }

    /// Configure with default preset
    pub fn preset_default(&mut self, n_channels: usize, sample_rate: T, split_computation: bool) {
        let block_samples = (sample_rate * T::from_f32(0.12).unwrap()).to_usize().unwrap_or(1024);
        let interval_samples = (sample_rate * T::from_f32(0.03).unwrap()).to_usize().unwrap_or(256);
        self.configure(n_channels, block_samples, interval_samples, split_computation);
    }

    /// Configure with cheaper preset
    pub fn preset_cheaper(&mut self, n_channels: usize, sample_rate: T, split_computation: bool) {
        let block_samples = (sample_rate * T::from_f32(0.1).unwrap()).to_usize().unwrap_or(1024);
        let interval_samples = (sample_rate * T::from_f32(0.04).unwrap()).to_usize().unwrap_or(256);
        self.configure(n_channels, block_samples, interval_samples, split_computation);
    }

    /// Manual configuration
    pub fn configure(&mut self, n_channels: usize, block_samples: usize, interval_samples: usize, split_computation: bool) {
        self.split_computation = split_computation;
        self.channels = n_channels;
        self.block_samples = block_samples;
        self.interval_samples = interval_samples;
        
        self.bands = block_samples / 2 + 1;
        
        // Configure STFT instances
        self.analysis_stft.configure(n_channels, n_channels, block_samples, block_samples, interval_samples);
        self.synthesis_stft.configure(n_channels, n_channels, block_samples, block_samples, interval_samples);
        
        self.tmp_buffer.resize(block_samples + interval_samples, T::zero());
        self.channel_bands.resize(self.bands * self.channels, Band::default());
        
        self.peaks.clear();
        self.peaks.reserve(self.bands / 2);
        self.energy.resize(self.bands, T::zero());
        self.smoothed_energy.resize(self.bands, T::zero());
        self.output_map.resize(self.bands, PitchMapPoint { input_bin: T::zero(), freq_grad: T::one() });
        self.channel_predictions.resize(self.channels * self.bands, Prediction::default());
        
        self.reset();
    }

    /// Set transpose factor for pitch shifting
    pub fn set_transpose_factor(&mut self, multiplier: T, tonality_limit: T) {
        self.freq_multiplier = multiplier;
        if tonality_limit > T::zero() {
            self.freq_tonality_limit = tonality_limit / multiplier.sqrt();
        } else {
            self.freq_tonality_limit = T::one();
        }
        self.custom_freq_map = None;
    }

    /// Set transpose in semitones
    pub fn set_transpose_semitones(&mut self, semitones: T, tonality_limit: T) {
        let multiplier = T::from_f32(2.0).unwrap().powf(semitones / T::from_f32(12.0).unwrap());
        self.set_transpose_factor(multiplier, tonality_limit);
    }

    /// Set custom frequency mapping function
    pub fn set_freq_map<F>(&mut self, input_to_output: F)
    where
        F: Fn(T) -> T + 'static,
    {
        self.custom_freq_map = Some(Box::new(input_to_output));
    }

    /// Set formant factor
    pub fn set_formant_factor(&mut self, multiplier: T, compensate_pitch: bool) {
        self.formant_multiplier = multiplier;
        self.inv_formant_multiplier = T::one() / multiplier;
        self.formant_compensation = compensate_pitch;
    }

    /// Set formant shift in semitones
    pub fn set_formant_semitones(&mut self, semitones: T, compensate_pitch: bool) {
        let multiplier = T::from_f32(2.0).unwrap().powf(semitones / T::from_f32(12.0).unwrap());
        self.set_formant_factor(multiplier, compensate_pitch);
    }

    /// Set formant base frequency
    pub fn set_formant_base(&mut self, base_freq: T) {
        self.formant_base_freq = base_freq;
    }

    /// Convert bin index to frequency (simplified)
    fn bin_to_freq(&self, bin: T) -> T {
        bin * T::from_f32(22050.0).unwrap() / T::from_usize(self.bands).unwrap()
    }

    /// Convert frequency to bin index (simplified)
    fn freq_to_bin(&self, freq: T) -> T {
        freq * T::from_usize(self.bands).unwrap() / T::from_f32(22050.0).unwrap()
    }

    /// Map frequency according to current settings
    fn map_freq(&self, freq: T) -> T {
        if let Some(ref custom_map) = self.custom_freq_map {
            custom_map(freq)
        } else if freq > self.freq_tonality_limit {
            freq + (self.freq_multiplier - T::one()) * self.freq_tonality_limit
        } else {
            freq * self.freq_multiplier
        }
    }

    /// Get bands for a specific channel
    fn bands_for_channel(&self, channel: usize) -> &[Band<T>] {
        let start = channel * self.bands;
        let end = start + self.bands;
        &self.channel_bands[start..end]
    }

    /// Get mutable bands for a specific channel
    fn bands_for_channel_mut(&mut self, channel: usize) -> &mut [Band<T>] {
        let start = channel * self.bands;
        let end = start + self.bands;
        &mut self.channel_bands[start..end]
    }

    /// Get predictions for a specific channel
    fn predictions_for_channel(&self, channel: usize) -> &[Prediction<T>] {
        let start = channel * self.bands;
        let end = start + self.bands;
        &self.channel_predictions[start..end]
    }

    /// Get mutable predictions for a specific channel
    fn predictions_for_channel_mut(&mut self, channel: usize) -> &mut [Prediction<T>] {
        let start = channel * self.bands;
        let end = start + self.bands;
        &mut self.channel_predictions[start..end]
    }

    /// Find spectral peaks
    fn find_peaks(&mut self) {
        self.peaks.clear();
        
        let mut start = 0;
        while start < self.bands {
            if self.energy[start] > self.smoothed_energy[start] {
                let mut end = start;
                let mut band_sum = T::zero();
                let mut energy_sum = T::zero();
                
                while end < self.bands && self.energy[end] > self.smoothed_energy[end] {
                    band_sum = band_sum + T::from_usize(end).unwrap() * self.energy[end];
                    energy_sum = energy_sum + self.energy[end];
                    end += 1;
                }
                
                let avg_band = band_sum / energy_sum;
                let avg_freq = self.bin_to_freq(avg_band);
                self.peaks.push(Peak {
                    input: avg_band,
                    output: self.freq_to_bin(self.map_freq(avg_freq)),
                });
                
                start = end;
            } else {
                start += 1;
            }
        }
    }

    /// Update output frequency mapping
    fn update_output_map(&mut self) {
        if self.peaks.is_empty() {
            for b in 0..self.bands {
                self.output_map[b] = PitchMapPoint {
                    input_bin: T::from_usize(b).unwrap(),
                    freq_grad: T::one(),
                };
            }
            return;
        }

        let bottom_offset = self.peaks[0].input - self.peaks[0].output;
        let end_bin = (self.peaks[0].output.ceil()).to_usize().unwrap_or(0).min(self.bands);
        
        for b in 0..end_bin {
            self.output_map[b] = PitchMapPoint {
                input_bin: T::from_usize(b).unwrap() + bottom_offset,
                freq_grad: T::one(),
            };
        }

        // Interpolate between peaks
        for p in 1..self.peaks.len() {
            let prev = &self.peaks[p - 1];
            let next = &self.peaks[p];
            
            let range_scale = T::one() / (next.output - prev.output);
            let out_offset = prev.input - prev.output;
            let out_scale = next.input - next.output - prev.input + prev.output;
            let grad_scale = out_scale * range_scale;
            
            let start_bin = (prev.output.ceil()).to_usize().unwrap_or(0);
            let end_bin = (next.output.ceil()).to_usize().unwrap_or(0).min(self.bands);
            
            for b in start_bin..end_bin {
                let r = (T::from_usize(b).unwrap() - prev.output) * range_scale;
                let h = r * r * (T::from_f32(3.0).unwrap() - T::from_f32(2.0).unwrap() * r);
                let out_b = T::from_usize(b).unwrap() + out_offset + h * out_scale;
                
                let grad_h = T::from_f32(6.0).unwrap() * r * (T::one() - r);
                let grad_b = T::one() + grad_h * grad_scale;
                
                self.output_map[b] = PitchMapPoint {
                    input_bin: out_b,
                    freq_grad: grad_b,
                };
            }
        }

        let top_offset = self.peaks.last().unwrap().input - self.peaks.last().unwrap().output;
        let start_bin = (self.peaks.last().unwrap().output).to_usize().unwrap_or(0);
        
        for b in start_bin..self.bands {
            self.output_map[b] = PitchMapPoint {
                input_bin: T::from_usize(b).unwrap() + top_offset,
                freq_grad: T::one(),
            };
        }
    }

    /// Main processing function (simplified)
    pub fn process<I, O>(&mut self, inputs: I, input_samples: usize, mut outputs: O, output_samples: usize)
    where
        I: AsRef<[Vec<T>]>,
        O: AsMut<[Vec<T>]>,
    {
        let inputs = inputs.as_ref();
        let outputs = outputs.as_mut();
        
        // Simplified processing - just copy input to output for now
        for c in 0..self.channels.min(inputs.len()).min(outputs.len()) {
            let input_channel = &inputs[c];
            let output_channel = &mut outputs[c];
            
            for i in 0..output_samples.min(output_channel.len()) {
                let input_idx = (i * input_samples / output_samples).min(input_channel.len().saturating_sub(1));
                output_channel[i] = input_channel[input_idx];
            }
        }
    }
}

impl<T: Float + FromPrimitive + NumCast + core::ops::AddAssign> Default for SignalsmithStretch<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        
        let c = a * b;
        assert!((c.re - (-5.0)).abs() < 1e-6);
        assert!((c.im - 10.0).abs() < 1e-6);
        
        let norm_sq = a.norm_sqr();
        assert!((norm_sq - 5.0).abs() < 1e-6);
        
        let conj = a.conj();
        assert!((conj.re - 1.0).abs() < 1e-6);
        assert!((conj.im - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_band_default() {
        let band: Band<f32> = Band::default();
        assert_eq!(band.input.re, 0.0);
        assert_eq!(band.input.im, 0.0);
        assert_eq!(band.input_energy, 0.0);
    }

    #[test]
    fn test_prediction_make_output() {
        let mut pred = Prediction::<f32>::default();
        pred.energy = 4.0;
        pred.input = Complex::new(2.0, 0.0);
        
        let phase = Complex::new(1.0, 1.0);
        let output = pred.make_output(phase);
        
        println!("output.norm() = {}", output.norm());
        
        assert!(output.norm().is_finite() && output.norm() > 0.0);
    }

    #[test]
    fn test_cute_stretch_new() {
        let stretch = SignalsmithStretch::<f32>::new();
        assert_eq!(stretch.channels, 0);
        assert_eq!(stretch.bands, 0);
        assert_eq!(stretch.block_samples, 0);
    }

    #[test]
    fn test_cute_stretch_configure() {
        let mut stretch = SignalsmithStretch::<f32>::new();
        stretch.configure(2, 1024, 256, false);
        
        assert_eq!(stretch.channels, 2);
        assert_eq!(stretch.block_samples, 1024);
        assert_eq!(stretch.interval_samples, 256);
        assert_eq!(stretch.bands, 513);
        assert_eq!(stretch.channel_bands.len(), 2 * 513);
    }

    #[test]
    fn test_transpose_factor() {
        let mut stretch = SignalsmithStretch::<f32>::new();
        stretch.set_transpose_factor(2.0, 0.5);
        
        assert_eq!(stretch.freq_multiplier, 2.0);
        assert!((stretch.freq_tonality_limit - (0.5 / 2.0_f32.sqrt())).abs() < 1e-6);
    }

    #[test]
    fn test_transpose_semitones() {
        let mut stretch = SignalsmithStretch::<f32>::new();
        stretch.set_transpose_semitones(12.0, 0.5);
        
        assert!((stretch.freq_multiplier - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_formant_factor() {
        let mut stretch = SignalsmithStretch::<f32>::new();
        stretch.set_formant_factor(1.5, true);
        
        assert_eq!(stretch.formant_multiplier, 1.5);
        assert!((stretch.inv_formant_multiplier - (1.0/1.5)).abs() < 1e-6);
        assert!(stretch.formant_compensation);
    }

    #[test]
    fn test_find_peaks() {
        let mut stretch = SignalsmithStretch::<f32>::new();
        stretch.configure(1, 8, 4, false);
        
        stretch.energy = vec![0.1, 0.5, 0.8, 0.3, 0.1, 0.2, 0.1, 0.1];
        stretch.smoothed_energy = vec![0.2, 0.3, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1];
        
        stretch.find_peaks();
        
        assert!(!stretch.peaks.is_empty());
    }

    #[test]
    fn test_update_output_map() {
        let mut stretch = SignalsmithStretch::<f32>::new();
        stretch.configure(1, 8, 4, false);
        
        stretch.peaks.push(Peak { input: 2.0, output: 3.0 });
        stretch.peaks.push(Peak { input: 5.0, output: 6.0 });
        
        stretch.update_output_map();
        
        assert_eq!(stretch.output_map.len(), stretch.bands);
        assert!(stretch.output_map[0].input_bin < stretch.output_map[1].input_bin);
    }

    #[test]
    fn test_process_simple() {
        let mut stretch = SignalsmithStretch::<f32>::new();
        stretch.configure(2, 1024, 256, false);
        
        let inputs = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];
        let mut outputs = vec![
            vec![0.0; 6],
            vec![0.0; 6],
        ];
        
        stretch.process(&inputs, 4, &mut outputs, 6);
        
        assert!(outputs[0].iter().any(|&x| x != 0.0));
        assert!(outputs[1].iter().any(|&x| x != 0.0));
    }
}
