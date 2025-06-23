//! Spacing: Custom Room Reverb Effect
//!
//! This module provides a reverb-like effect with support for custom source and receiver positions.
//! It uses a multi-tap delay network to simulate early reflections and room size.

use crate::delay::{Delay, InterpolatorLinear};
use num_traits::{Float, FromPrimitive};
use crate::filters::{Biquad, BiquadDesign};
use crate::mix::{Hadamard, Householder};

/// 3D position for source/receiver
#[derive(Clone, Copy, Debug)]
pub struct Position<T: Float> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Float> Position<T> {
    pub fn distance(&self, other: &Position<T>) -> T {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2)).sqrt()
    }
}

/// A single path from source to receiver (could be direct or a reflection)
#[derive(Clone, Debug)]
pub struct Path<T: Float> {
    pub delay_samples: T,
    pub gain: T,
    // Future: add filter, wall absorption, etc.
}

/// Main Spacing effect struct
pub struct Spacing<T: Float> {
    pub sample_rate: T,
    pub sources: Vec<Position<T>>,
    pub receivers: Vec<Position<T>>,
    pub paths: Vec<(usize, usize, Path<T>)>, // (source_idx, receiver_idx, Path)
    pub delays: Vec<Delay<T, InterpolatorLinear<T>>>,
    // Reverb-like parameters
    pub room_size: T, // scales all distances
    pub damping: T,   // 0 = no damping, 1 = max damping
    // New parameters
    pub diff: T,      // 0 = no diffusion, 1 = max
    pub bass: T,      // 0 = flat, >0 = boost, <0 = cut
    pub decay: T,     // 0 = no decay, 1 = max decay
    pub cross: T,     // 0 = no cross-mix, 1 = max
    // State for filters/mixers
    pub bass_filters: Vec<Biquad<T>>,
    pub cross_mixer: Option<Hadamard<T>>,
}

impl<T: Float + FromPrimitive> Spacing<T> {
    /// Create a new Spacing effect with a given sample rate
    pub fn new(sample_rate: T) -> Self {
        Self {
            sample_rate,
            sources: Vec::new(),
            receivers: Vec::new(),
            paths: Vec::new(),
            delays: Vec::new(),
            room_size: T::one(),
            damping: T::zero(),
            diff: T::zero(),
            bass: T::zero(),
            decay: T::zero(),
            cross: T::zero(),
            bass_filters: Vec::new(),
            cross_mixer: None,
        }
    }

    /// Set room size (scales all distances)
    pub fn set_room_size(&mut self, size: T) {
        let min = T::from(0.01).unwrap();
        self.room_size = if size < min { min } else { size };
    }
    /// Set damping (0 = no damping, 1 = max damping)
    pub fn set_damping(&mut self, damping: T) {
        self.damping = if damping < T::zero() {
            T::zero()
        } else if damping > T::one() {
            T::one()
        } else {
            damping
        };
    }
    /// Set diffusion amount
    pub fn set_diff(&mut self, diff: T) {
        self.diff = if diff < T::zero() { T::zero() } else if diff > T::one() { T::one() } else { diff };
    }
    /// Set bass boost/cut (dB)
    pub fn set_bass(&mut self, bass: T) {
        self.bass = bass;
        // Reconfigure filters
        self.bass_filters.clear();
        for _ in 0..self.receivers.len() {
            let mut biq = Biquad::new(false);
            let freq = T::from_f32(200.0).unwrap() / self.sample_rate; // 200 Hz cutoff
            biq.low_shelf(freq, bass);
            self.bass_filters.push(biq);
        }
    }
    /// Set decay (0 = no decay, 1 = max decay)
    pub fn set_decay(&mut self, decay: T) {
        self.decay = if decay < T::zero() { T::zero() } else if decay > T::one() { T::one() } else { decay };
    }
    /// Set cross-mix (0 = none, 1 = max)
    pub fn set_cross(&mut self, cross: T) {
        self.cross = if cross < T::zero() { T::zero() } else if cross > T::one() { T::one() } else { cross };
        if self.cross > T::zero() {
            self.cross_mixer = Some(Hadamard::new(self.receivers.len()));
        } else {
            self.cross_mixer = None;
        }
    }

    /// Add a source position
    pub fn add_source(&mut self, pos: Position<T>) -> usize {
        self.sources.push(pos);
        self.sources.len() - 1
    }

    /// Add a receiver position
    pub fn add_receiver(&mut self, pos: Position<T>) -> usize {
        self.receivers.push(pos);
        // Add a bass filter for the new receiver
        let mut biq = Biquad::new(false);
        let freq = T::from_f32(200.0).unwrap() / self.sample_rate;
        biq.low_shelf(freq, self.bass);
        self.bass_filters.push(biq);
        self.receivers.len() - 1
    }

    /// Add a path (direct or reflection) between a source and receiver
    pub fn add_path(&mut self, source_idx: usize, receiver_idx: usize, gain: T, extra_distance: T) {
        let src = self.sources[source_idx];
        let recv = self.receivers[receiver_idx];
        let distance = (src.distance(&recv) + extra_distance) * self.room_size;
        let speed_of_sound = T::from(343.0).unwrap();
        let delay_samples = distance / speed_of_sound * self.sample_rate;
        self.paths.push((source_idx, receiver_idx, Path { delay_samples, gain }));
        let delay_len = delay_samples.ceil().to_usize().unwrap_or(1) + 1;
        self.delays.push(Delay::new(InterpolatorLinear::new(), delay_len));
    }

    /// Clear all paths and delays
    pub fn clear_paths(&mut self) {
        self.paths.clear();
        self.delays.clear();
    }

    /// Process a buffer for all receivers (multi-source, multi-output)
    pub fn process(&mut self, inputs: &[&[T]], outputs: &mut [Vec<T>]) {
        let len = if let Some(input) = inputs.get(0) { input.len() } else { 0 };
        for out in outputs.iter_mut() {
            out.clear();
            out.resize(len, T::zero());
        }
        // Optionally apply input diffusion (pre-mix)
        let mut diff_inputs: Vec<Vec<T>> = vec![];
        if self.diff > T::zero() && inputs.len() > 1 {
            diff_inputs = inputs.iter().map(|x| x.to_vec()).collect();
            let mut frame: Vec<T> = diff_inputs.iter().map(|v| v[0]).collect();
            let hadamard = Hadamard::new(inputs.len());
            for i in 0..len {
                for (j, v) in diff_inputs.iter_mut().enumerate() {
                    frame[j] = v[i];
                }
                hadamard.in_place(&mut frame);
                for (j, v) in diff_inputs.iter_mut().enumerate() {
                    v[i] = frame[j] * self.diff + v[i] * (T::one() - self.diff);
                }
            }
        }
        let use_inputs: Vec<&[T]> = if self.diff > T::zero() && inputs.len() > 1 {
            diff_inputs.iter().map(|v| v.as_slice()).collect()
        } else {
            inputs.iter().map(|x| *x).collect()
        };
        for i in 0..len {
            let mut wet_sum = vec![T::zero(); outputs.len()];
            for ((path_idx, (source_idx, receiver_idx, path)), delay) in self.paths.iter().enumerate().zip(self.delays.iter_mut()) {
                let sample = if let Some(input) = use_inputs.get(*source_idx) {
                    input[i]
                } else {
                    T::zero()
                };
                delay.write(sample);
                let mut delayed = delay.read(path.delay_samples) * path.gain;
                // Apply decay
                if self.decay > T::zero() {
                    let decay_factor = T::one() - self.decay * T::from_f32(0.001).unwrap();
                    delayed = delayed * decay_factor.powi(i as i32);
                }
                // Apply damping
                if self.damping > T::zero() {
                    delayed = delayed * (T::one() - self.damping).powi(i as i32);
                }
                wet_sum[*receiver_idx] = wet_sum[*receiver_idx] + delayed;
            }
            // Apply bass filter per receiver
            for (r, wet) in wet_sum.iter_mut().enumerate() {
                if let Some(filt) = self.bass_filters.get_mut(r) {
                    *wet = filt.process(*wet);
                }
            }
            // Optionally apply cross-mix (Hadamard)
            if let Some(mixer) = &self.cross_mixer {
                let mut frame = wet_sum.clone();
                mixer.in_place(&mut frame);
                for (r, out) in outputs.iter_mut().enumerate() {
                    out[i] = frame[r] * self.cross + wet_sum[r] * (T::one() - self.cross);
                }
            } else {
                for (r, out) in outputs.iter_mut().enumerate() {
                    out[i] = wet_sum[r];
                }
            }
        }
    }

    // Future: add methods to add reflections, set wall positions, multi-channel output, etc.
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_direct_path() {
        let sample_rate = 48000.0f32;
        let mut spacing = Spacing::<f32>::new(sample_rate);
        let src = spacing.add_source(Position { x: 0.0, y: 0.0, z: 0.0 });
        let recv = spacing.add_receiver(Position { x: 3.43, y: 0.0, z: 0.0 }); // 3.43m = 480 samples
        spacing.add_path(src, recv, 1.0, 0.0);
        let mut input = vec![0.0; 500];
        input[0] = 1.0;
        let mut outputs = vec![vec![0.0; 500]];
        spacing.process(&[&input], &mut outputs);
        let found = outputs[0][478..=482].iter().cloned().fold(f32::MIN, f32::max);
        assert!((found - 1.0).abs() < 1e-5, "max in window around 480: {}", found);
    }
    #[test]
    fn test_multiple_receivers_and_reflection() {
        let sample_rate = 48000.0f32;
        let mut spacing = Spacing::<f32>::new(sample_rate);
        let src = spacing.add_source(Position { x: 0.0, y: 0.0, z: 0.0 });
        let recv1 = spacing.add_receiver(Position { x: 3.43, y: 0.0, z: 0.0 });
        let recv2 = spacing.add_receiver(Position { x: 3.43, y: 3.0, z: 0.0 });
        spacing.add_path(src, recv1, 1.0, 0.0);
        spacing.add_path(src, recv2, 0.5, 3.0); // Reflection: extra 3m
        let mut input = vec![0.0; 1200];
        input[0] = 1.0;
        let mut outputs = vec![vec![0.0; 1200]; 2];
        spacing.process(&[&input], &mut outputs);
        let found1 = outputs[0][478..=482].iter().cloned().fold(f32::MIN, f32::max);
        assert!((found1 - 1.0).abs() < 1e-5, "max in window around 480: {}", found1);
        let d = (3.43f32.powi(2) + 3.0f32.powi(2)).sqrt() + 3.0;
        let delay = (d / 343.0 * sample_rate).round() as usize;
        if delay < outputs[1].len() - 1 {
            let window = &outputs[1][delay.saturating_sub(2)..(delay+3).min(outputs[1].len())];
            let max_val = window.iter().cloned().fold(f32::MIN, f32::max);
            assert!((max_val > 0.2 && max_val < 0.3), "max_val={} in window around delay={}", max_val, delay);
        } else {
            panic!("Reflection delay {} exceeds output buffer length {}", delay, outputs[1].len());
        }
    }
    #[test]
    fn test_damping_reduces_amplitude() {
        let sample_rate = 48000.0f32;
        let mut spacing = Spacing::<f32>::new(sample_rate);
        let src = spacing.add_source(Position { x: 0.0, y: 0.0, z: 0.0 });
        let recv = spacing.add_receiver(Position { x: 3.43, y: 0.0, z: 0.0 });
        spacing.add_path(src, recv, 1.0, 0.0);
        let mut input = vec![0.0; 500];
        input[0] = 1.0;
        let mut outputs = vec![vec![0.0; 500]];
        spacing.set_damping(0.0);
        spacing.process(&[&input], &mut outputs);
        let found_no_damping = outputs[0][478..=482].iter().cloned().fold(f32::MIN, f32::max);
        spacing.clear_paths(); spacing.delays.clear();
        spacing.add_path(src, recv, 1.0, 0.0);
        spacing.set_damping(0.5);
        let mut outputs2 = vec![vec![0.0; 500]];
        spacing.process(&[&input], &mut outputs2);
        let found_damping = outputs2[0][478..=482].iter().cloned().fold(f32::MIN, f32::max);
        assert!(found_damping < found_no_damping, "Damping should reduce amplitude: {} vs {}", found_damping, found_no_damping);
    }
    #[test]
    fn test_room_size_affects_delay() {
        let sample_rate = 48000.0f32;
        let mut spacing = Spacing::<f32>::new(sample_rate);
        let src = spacing.add_source(Position { x: 0.0, y: 0.0, z: 0.0 });
        let recv = spacing.add_receiver(Position { x: 3.43, y: 0.0, z: 0.0 });
        spacing.set_room_size(1.0);
        spacing.add_path(src, recv, 1.0, 0.0);
        let mut input = vec![0.0; 2500];
        input[0] = 1.0;
        let mut outputs = vec![vec![0.0; 2500]];
        spacing.process(&[&input], &mut outputs);
        let _found1 = outputs[0][478..=482].iter().cloned().fold(f32::MIN, f32::max);
        spacing.clear_paths(); spacing.delays.clear();
        spacing.set_room_size(2.0);
        spacing.add_path(src, recv, 1.0, 0.0);
        let mut input = vec![0.0; 2500];
        input[0] = 1.0;
        let mut outputs2 = vec![vec![0.0; 2500]];
        spacing.process(&[&input], &mut outputs2);
        let found2 = outputs2[0][958..=962].iter().cloned().fold(f32::MIN, f32::max);
        assert!(found2 > 0.2, "Impulse at double room size should be present: {}", found2);
        assert!(outputs2[0][478..=482].iter().all(|&v| v < 1e-3), "Impulse should not be at original delay");
    }
    #[test]
    fn test_multiple_sources() {
        let sample_rate = 48000.0f32;
        let mut spacing = Spacing::<f32>::new(sample_rate);
        let src1 = spacing.add_source(Position { x: 0.0, y: 0.0, z: 0.0 });
        let src2 = spacing.add_source(Position { x: 1.0, y: 0.0, z: 0.0 });
        let recv = spacing.add_receiver(Position { x: 3.43, y: 0.0, z: 0.0 });
        spacing.add_path(src1, recv, 1.0, 0.0);
        spacing.add_path(src2, recv, 0.5, 0.0);
        let mut input1 = vec![0.0; 540];
        input1[0] = 1.0;
        let mut input2 = vec![0.0; 540];
        input2[30] = 1.0;
        let mut outputs = vec![vec![0.0; 540]];
        spacing.process(&[&input1, &input2], &mut outputs);
        let peak1 = outputs[0][478..=482].iter().cloned().fold(f32::MIN, f32::max);
        let peak2 = outputs[0][368..=372].iter().cloned().fold(f32::MIN, f32::max);
        assert!(peak1 > 0.4 && peak2 > 0.2, "Both impulses should be present: peak1={}, peak2={}", peak1, peak2);
        assert!(peak1 + peak2 > 0.9, "Sum of peaks should be > 0.9: {}", peak1 + peak2);
    }
} 