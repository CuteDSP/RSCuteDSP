//! LFOs and envelope generators
//!
//! This module provides LFOs, envelopes, and filters for manipulating them.

#![allow(unused_imports)]

#[cfg(feature = "std")]
use std::{vec::Vec, marker::PhantomData, f32::consts::E};

#[cfg(not(feature = "std"))]
use core::{marker::PhantomData, f32::consts::E};

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

use num_traits::Float;

/// An LFO based on cubic segments.
///
/// You can randomize the rate and/or the depth. Randomizing the depth past `0.5` means
/// it no longer neatly alternates sides.
///
/// Without randomization, it is approximately sine-like.
pub struct CubicLfo {
    ratio: f32,
    ratio_step: f32,
    
    value_from: f32,
    value_to: f32,
    value_range: f32,
    
    target_low: f32,
    target_high: f32,
    target_rate: f32,
    
    rate_random: f32,
    depth_random: f32,
    
    fresh_reset: bool,
    
    // Random number generation
    seed: u64,
}

impl CubicLfo {
    /// Create a new LFO with a random seed
    pub fn new() -> Self {
        let mut lfo = Self {
            ratio: 0.0,
            ratio_step: 0.0,
            value_from: 0.0,
            value_to: 1.0,
            value_range: 1.0,
            target_low: 0.0,
            target_high: 1.0,
            target_rate: 0.0,
            rate_random: 0.5,
            depth_random: 0.0,
            fresh_reset: true,
            seed: 0,
        };
        
        // Use a simple time-based seed
        #[cfg(feature = "std")]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
            lfo.seed = now.as_secs() ^ (now.subsec_nanos() as u64);
        }
        
        #[cfg(not(feature = "std"))]
        {
            // In no_std environments, use a fixed seed
            lfo.seed = 12345;
        }
        
        lfo.reset();
        lfo
    }
    
    /// Create a new LFO with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        let mut lfo = Self {
            ratio: 0.0,
            ratio_step: 0.0,
            value_from: 0.0,
            value_to: 1.0,
            value_range: 1.0,
            target_low: 0.0,
            target_high: 1.0,
            target_rate: 0.0,
            rate_random: 0.5,
            depth_random: 0.0,
            fresh_reset: true,
            seed,
        };
        lfo.reset();
        lfo
    }
    
    // Simple xorshift random number generator
    fn random(&mut self) -> f32 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        (self.seed & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }
    
    fn random_rate(&mut self) -> f32 {
        self.target_rate * (E.powf(self.rate_random * (self.random() - 0.5)))
    }
    
    fn random_target(&mut self, previous: f32) -> f32 {
        let random_offset = self.depth_random * self.random() * (self.target_low - self.target_high);
        if previous < (self.target_low + self.target_high) * 0.5 {
            self.target_high + random_offset
        } else {
            self.target_low - random_offset
        }
    }
    
    /// Resets the LFO state, starting with random phase.
    pub fn reset(&mut self) {
        self.ratio = self.random();
        self.ratio_step = self.random_rate();
        
        if self.random() < 0.5 {
            self.value_from = self.target_low;
            self.value_to = self.target_high;
        } else {
            self.value_from = self.target_high;
            self.value_to = self.target_low;
        }
        
        self.value_range = self.value_to - self.value_from;
        self.fresh_reset = true;
    }
    
    /// Smoothly updates the LFO parameters.
    ///
    /// If called directly after `.reset()`, oscillation will immediately start within the specified range.
    /// Otherwise, it will remain smooth and fit within the new range after at most one cycle.
    ///
    /// The LFO will complete a full oscillation in (approximately) `1/rate` samples.
    /// `rate_variation` can be any number, but 0-1 is a good range.
    ///
    /// `depth_variation` must be in the range [0, 1], where ≤ 0.5 produces random amplitude
    /// but still alternates up/down.
    pub fn set(&mut self, low: f32, high: f32, rate: f32, rate_variation: f32, depth_variation: f32) {
        let rate = rate * 2.0; // We want to go up and down during this period
        self.target_rate = rate;
        self.target_low = low.min(high);
        self.target_high = low.max(high);
        self.rate_random = rate_variation;
        self.depth_random = depth_variation.max(0.0).min(1.0);
        
        // If we haven't called .next() yet, don't bother being smooth.
        if self.fresh_reset {
            return self.reset();
        }
        
        // Only update the current rate if it's outside our new random-variation range
        let max_random_ratio = E.powf(0.5 * self.rate_random);
        if self.ratio_step > rate * max_random_ratio || self.ratio_step < rate / max_random_ratio {
            self.ratio_step = self.random_rate();
        }
    }
    
    /// Returns the next output sample
    pub fn next(&mut self) -> f32 {
        self.fresh_reset = false;
        let result = self.ratio * self.ratio * (3.0 - 2.0 * self.ratio) * self.value_range + self.value_from;
        
        self.ratio += self.ratio_step;
        while self.ratio >= 1.0 {
            self.ratio -= 1.0;
            self.ratio_step = self.random_rate();
            self.value_from = self.value_to;
            self.value_to = self.random_target(self.value_from);
            self.value_range = self.value_to - self.value_from;
        }
        
        result
    }
}

/// Variable-width rectangular sum
pub struct BoxSum<T: Float> {
    buffer_length: usize,
    index: usize,
    buffer: Vec<T>,
    sum: T,
    wrap_jump: T,
}

impl<T: Float> BoxSum<T> {
    /// Create a new box sum with the specified maximum length
    pub fn new(max_length: usize) -> Self {
        let mut result = Self {
            buffer_length: 0,
            index: 0,
            buffer: Vec::new(),
            sum: T::zero(),
            wrap_jump: T::zero(),
        };
        result.resize(max_length);
        result
    }
    
    /// Sets the maximum size (and reset contents)
    pub fn resize(&mut self, max_length: usize) {
        self.buffer_length = max_length + 1;
        self.buffer.resize(self.buffer_length, T::zero());
        self.reset(T::zero());
    }
    
    /// Resets (with an optional "fill" value)
    pub fn reset(&mut self, value: T) {
        self.index = 0;
        self.sum = T::zero();
        
        for i in 0..self.buffer.len() {
            self.buffer[i] = self.sum;
            self.sum = self.sum + value;
        }
        
        self.wrap_jump = self.sum;
        self.sum = T::zero();
    }
    
    /// Read a sum of the last `width` samples
    pub fn read(&self, width: usize) -> T {
        let mut read_index = self.index as isize - width as isize;
        let mut result = self.sum;
        
        if read_index < 0 {
            result = result + self.wrap_jump;
            read_index += self.buffer_length as isize;
        }
        
        result - self.buffer[read_index as usize]
    }
    
    /// Write a new sample
    pub fn write(&mut self, value: T) {
        self.index += 1;
        if self.index == self.buffer_length {
            self.index = 0;
            self.wrap_jump = self.sum;
            self.sum = T::zero();
        }
        
        self.sum = self.sum + value;
        self.buffer[self.index] = self.sum;
    }
    
    /// Read and write in one operation
    pub fn read_write(&mut self, value: T, width: usize) -> T {
        self.write(value);
        self.read(width)
    }
}

/// Rectangular moving average filter (FIR).
///
/// A filter of length 1 has order 0 (i.e. does nothing).
pub struct BoxFilter<T: Float> {
    box_sum: BoxSum<T>,
    length: usize,
    max_length: usize,
    multiplier: T,
}

impl<T: Float> BoxFilter<T> {
    /// Create a new box filter with the specified maximum length
    pub fn new(max_length: usize) -> Self {
        let mut result = Self {
            box_sum: BoxSum::new(max_length),
            length: 0,
            max_length: 0,
            multiplier: T::one(),
        };
        result.resize(max_length);
        result
    }
    
    /// Sets the maximum size (and current size, and resets)
    pub fn resize(&mut self, max_length: usize) {
        self.max_length = max_length;
        self.box_sum.resize(max_length);
        self.set(max_length);
    }
    
    /// Sets the current size (expanding/allocating only if needed)
    pub fn set(&mut self, length: usize) {
        self.length = length;
        self.multiplier = T::one() / T::from(length).unwrap();
        
        if length > self.max_length {
            self.resize(length);
        }
    }
    
    /// Resets (with an optional "fill" value)
    pub fn reset(&mut self, fill: T) {
        self.box_sum.reset(fill);
    }
    
    /// Process a sample
    pub fn process(&mut self, v: T) -> T {
        self.box_sum.read_write(v, self.length) * self.multiplier
    }
}

/// FIR filter made from a stack of `BoxFilter`s.
///
/// This filter has a non-negative impulse (monotonic step response), making it useful
/// for smoothing positive-only values.
pub struct BoxStackFilter<T: Float> {
    size: usize,
    layers: Vec<BoxStackLayer<T>>,
}

struct BoxStackLayer<T: Float> {
    ratio: f32,
    length_error: f32,
    length: usize,
    filter: BoxFilter<T>,
}

impl<T: Float> BoxStackFilter<T> {
    /// Create a new box stack filter with the specified maximum size and number of layers
    pub fn new(max_size: usize, layers: usize) -> Self {
        let mut result = Self {
            size: 0,
            layers: Vec::new(),
        };
        result.resize(max_size, layers);
        result
    }
    
    /// Returns an optimal set of length ratios (heuristic for larger depths)
    pub fn optimal_ratios(layer_count: usize) -> Vec<f32> {
        // Coefficients up to 6, found through numerical search
        static HARDCODED: [f32; 21] = [
            1.0, 0.58224186169, 0.41775813831, 0.404078562416, 0.334851475794, 0.261069961789,
            0.307944914938, 0.27369945234, 0.22913263601, 0.189222996712, 0.248329349789,
            0.229253789144, 0.201191468123, 0.173033035122, 0.148192357821, 0.205275202874,
            0.198413552119, 0.178256637764, 0.157821404506, 0.138663023387, 0.121570179349
        ];
        
        if layer_count <= 0 {
            return Vec::new();
        } else if layer_count <= 6 {
            let start = layer_count * (layer_count - 1) / 2;
            return HARDCODED[start..start + layer_count].to_vec();
        }
        
        let mut result = vec![0.0; layer_count];
        
        let inv_n = 1.0 / layer_count as f32;
        let sqrt_n = (layer_count as f32).sqrt();
        let p = 1.0 - inv_n;
        let k = 1.0 + 4.5 / sqrt_n + 0.08 * sqrt_n;
        
        let mut sum = 0.0;
        for i in 0..layer_count {
            let x = i as f32 * inv_n;
            let power = -x * (1.0 - p * (-x * k).exp());
            let length = 2.0f32.powf(power);
            result[i] = length;
            sum += length;
        }
        
        let factor = 1.0 / sum;
        for r in &mut result {
            *r *= factor;
        }
        
        result
    }
    
    /// Approximate (optimal) bandwidth for a given number of layers
    pub fn layers_to_bandwidth(layers: usize) -> f32 {
        1.58 * (layers as f32 + 0.1)
    }
    
    /// Approximate (optimal) peak in the stop-band
    pub fn layers_to_peak_db(layers: usize) -> f32 {
        5.0 - layers as f32 * 18.0
    }
    
    /// Sets size using an optimal (heuristic at larger sizes) set of length ratios
    pub fn resize(&mut self, max_size: usize, layer_count: usize) {
        self.resize_with_ratios(max_size, Self::optimal_ratios(layer_count));
    }
    
    /// Sets the maximum (and current) impulse response length and explicit length ratios
    pub fn resize_with_ratios(&mut self, max_size: usize, ratios: Vec<f32>) {
        self.setup_layers(ratios);
        
        for layer in &mut self.layers {
            layer.filter.resize(0);
        }
        
        self.size = 0;
        self.set(max_size);
        self.reset();
    }
    
    fn setup_layers(&mut self, ratios: Vec<f32>) {
        self.layers.clear();
        
        let mut sum = 0.0;
        for ratio in &ratios {
            self.layers.push(BoxStackLayer {
                ratio: *ratio,
                length_error: 0.0,
                length: 0,
                filter: BoxFilter::new(0),
            });
            sum += ratio;
        }
        
        let factor = 1.0 / sum;
        for layer in &mut self.layers {
            layer.ratio *= factor;
        }
    }
    
    /// Sets the impulse response length (does not reset if `size` ≤ `max_size`)
    pub fn set(&mut self, size: usize) {
        if self.layers.is_empty() {
            return;
        }
        
        if self.size == size {
            return;
        }
        
        self.size = size;
        let order = size - 1;
        let mut total_order = 0;
        
        for layer in &mut self.layers {
            let layer_order_fractional = layer.ratio * order as f32;
            let layer_order = layer_order_fractional as usize;
            layer.length = layer_order + 1;
            layer.length_error = layer_order as f32 - layer_order_fractional;
            total_order += layer_order;
        }
        
        // Round some of them up, so the total is correct
        while total_order < order {
            let mut min_index = 0;
            let mut min_error = self.layers[0].length_error;
            
            for i in 1..self.layers.len() {
                if self.layers[i].length_error < min_error {
                    min_error = self.layers[i].length_error;
                    min_index = i;
                }
            }
            
            self.layers[min_index].length += 1;
            self.layers[min_index].length_error += 1.0;
            total_order += 1;
        }
        
        for layer in &mut self.layers {
            layer.filter.set(layer.length);
        }
    }
    
    /// Resets the filter
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.filter.reset(T::zero());
        }
    }
    
    /// Process a sample
    pub fn process(&mut self, v: T) -> T {
        let mut result = v;
        for layer in &mut self.layers {
            result = layer.filter.process(result);
        }
        result
    }
}

/// Peak-hold filter.
///
/// The size is variable, and can be changed instantly with `.set()`,
/// or by using `.push()`/`.pop()` in an unbalanced way.
///
/// This has complexity O(1) every sample when the length remains constant
/// (balanced `.push()`/`.pop()`, or using `filter(v)`), and amortised O(1)
/// complexity otherwise.
pub struct PeakHold<T: Float> {
    buffer_mask: usize,
    buffer: Vec<T>,
    back_index: isize,
    middle_start: isize,
    working_index: isize,
    middle_end: isize,
    front_index: isize,
    front_max: T,
    working_max: T,
    middle_max: T,
}

impl<T: Float> PeakHold<T> {
    /// Create a new peak-hold filter with the specified maximum length
    pub fn new(max_length: usize) -> Self {
        let mut result = Self {
            buffer_mask: 0,
            buffer: Vec::new(),
            back_index: 0,
            middle_start: 0,
            working_index: 0,
            middle_end: 0,
            front_index: 0,
            front_max: T::min_value(),
            working_max: T::min_value(),
            middle_max: T::min_value(),
        };
        result.resize(max_length);
        result
    }
    
    /// Get the current size of the filter
    pub fn size(&self) -> usize {
        (self.front_index - self.back_index) as usize
    }
    
    /// Resize the filter to a new maximum length
    pub fn resize(&mut self, max_length: usize) {
        let mut buffer_length = 1;
        while buffer_length < max_length {
            buffer_length *= 2;
        }
        
        self.buffer.resize(buffer_length, T::min_value());
        self.buffer_mask = buffer_length - 1;
        
        self.front_index = self.back_index + max_length as isize;
        self.reset();
    }
    
    /// Reset the filter
    pub fn reset(&mut self) {
        let prev_size = self.size();
        
        for i in 0..self.buffer.len() {
            self.buffer[i] = T::min_value();
        }
        
        self.front_max = T::min_value();
        self.working_max = T::min_value();
        self.middle_max = T::min_value();
        
        self.middle_end = 0;
        self.working_index = 0;
        self.front_index = 0;
        self.middle_start = self.middle_end - (prev_size as isize / 2);
        self.back_index = self.front_index - prev_size as isize;
    }
    
    /// Sets the size immediately.
    ///
    /// Must be `0 <= new_size <= max_length` (see constructor and `.resize()`).
    ///
    /// Shrinking doesn't destroy information, and if you expand again (with `preserve_current_peak=false`),
    /// you will get the same output as before shrinking. Expanding when `preserve_current_peak` is enabled
    /// is destructive, re-writing its history such that the current output value is unchanged.
    pub fn set(&mut self, new_size: usize, preserve_current_peak: bool) {
        while self.size() < new_size {
            let back_prev_idx = (self.back_index as usize) & self.buffer_mask;
            let back_prev = self.buffer[back_prev_idx];
            
            self.back_index -= 1;
            
            let back_idx = (self.back_index as usize) & self.buffer_mask;
            if preserve_current_peak {
                self.buffer[back_idx] = back_prev;
            } else {
                self.buffer[back_idx] = self.buffer[back_idx].max(back_prev);
            }
        }
        
        while self.size() > new_size {
            self.pop();
        }
    }
    
    /// Push a new value onto the filter
    pub fn push(&mut self, v: T) {
        let front_idx = (self.front_index as usize) & self.buffer_mask;
        self.buffer[front_idx] = v;
        self.front_index += 1;
        self.front_max = self.front_max.max(v);
    }
    
    /// Pop a value from the filter
    pub fn pop(&mut self) {
        if self.back_index == self.middle_start {
            // Move along the maximums
            self.working_max = T::min_value();
            self.middle_max = self.front_max;
            self.front_max = T::min_value();
            
            let prev_front_length = self.front_index - self.middle_end;
            let prev_middle_length = self.middle_end - self.middle_start;
            
            if prev_front_length <= prev_middle_length + 1 {
                // Swap over simply
                self.middle_start = self.middle_end;
                self.middle_end = self.front_index;
                self.working_index = self.middle_end;
            } else {
                // The front is longer than the middle - only happens if unbalanced
                // We don't move *all* of the front over, keeping half the surplus in the front
                let middle_length = (self.front_index - self.middle_start) / 2;
                self.middle_start = self.middle_end;
                self.middle_end += middle_length;
                
                // Working index is close enough that it will be finished by the time the back is empty
                let back_length = self.middle_start - self.back_index;
                let working_length = back_length.min(self.middle_end - self.middle_start);
                self.working_index = self.middle_start + working_length;
                
                // Since the front was not completely consumed, we re-calculate the front's maximum
                for i in self.middle_end..self.front_index {
                    let idx = (i as usize) & self.buffer_mask;
                    self.front_max = self.front_max.max(self.buffer[idx]);
                }
                
                // The index might not start at the end of the working block - compute the last bit immediately
                for i in (self.working_index..self.middle_end).rev() {
                    let idx = (i as usize) & self.buffer_mask;
                    self.buffer[idx] = self.working_max;
                    self.working_max = self.working_max.max(self.buffer[idx]);
                }
            }
            
            // Is the new back (previous middle) empty? Only happens if unbalanced
            if self.back_index == self.middle_start {
                // swap over again (front's empty, no change)
                self.working_max = T::min_value();
                self.middle_max = self.front_max;
                self.front_max = T::min_value();
                self.middle_start = self.middle_end;
                self.working_index = self.middle_end;
                
                if self.back_index == self.middle_start {
                    self.back_index -= 1; // Only happens if you pop from an empty list - fail nicely
                }
            }
            
            // In case of length 0, when everything points at this value
            let front_idx = (self.front_index as usize) & self.buffer_mask;
            self.buffer[front_idx] = T::min_value();
        }
        
        self.back_index += 1;
        
        if self.working_index != self.middle_start {
            self.working_index -= 1;
            let idx = (self.working_index as usize) & self.buffer_mask;
            self.buffer[idx] = self.working_max;
            self.working_max = self.working_max.max(self.buffer[idx]);
        }
    }
    
    /// Read the current maximum value
    pub fn read(&self) -> T {
        let back_idx = (self.back_index as usize) & self.buffer_mask;
        let back_max = self.buffer[back_idx];
        back_max.max(self.middle_max).max(self.front_max)
    }
    
    /// Process a sample (push, pop, and read)
    pub fn process(&mut self, v: T) -> T {
        self.push(v);
        self.pop();
        self.read()
    }
}

/// Peak-decay filter with a linear shape and fixed-time return to constant value.
///
/// This is equivalent to a `BoxFilter` which resets itself whenever the output
/// would be less than the input.
pub struct PeakDecayLinear<T: Float> {
    peak_hold: PeakHold<T>,
    value: T,
    step_multiplier: T,
}

impl<T: Float> PeakDecayLinear<T> {
    /// Create a new peak-decay filter with the specified maximum length
    pub fn new(max_length: usize) -> Self {
        let mut result = Self {
            peak_hold: PeakHold::new(max_length),
            value: T::min_value(),
            step_multiplier: T::one(),
        };
        result.set(max_length as f32);
        result
    }

    /// Resize the filter to a new maximum length
    pub fn resize(&mut self, max_length: usize) {
        self.peak_hold.resize(max_length);
        self.reset();
    }

    /// Set the filter length
    pub fn set(&mut self, length: f32) {
        let window_size = length.ceil() as usize;
        self.peak_hold.set(window_size, true);
        self.step_multiplier = T::from(1.0 / window_size as f32).unwrap();
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.peak_hold.reset();
        self.set(self.peak_hold.size() as f32);
        self.value = T::min_value();
    }

    /// Process a sample
    pub fn process(&mut self, v: T) -> T {
        let peak = self.peak_hold.read();
        self.peak_hold.process(v);

        // Calculate decay step based on peak value and filter length
        let decay_step = peak * self.step_multiplier;
        self.value = v.max(self.value - decay_step);
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_lfo() {
        let mut lfo = CubicLfo::with_seed(12345);

        // Set parameters
        lfo.set(0.0, 1.0, 0.1, 0.0, 0.0);

        // Generate some samples
        let mut samples = Vec::new();
        for _ in 0..100 {
            samples.push(lfo.next());
        }

        // Check that samples are within range
        for sample in &samples {
            assert!(*sample >= 0.0 && *sample <= 1.0);
        }

        // Check that the LFO oscillates (has both high and low values)
        let min = samples.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = samples.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        assert!(min < 0.1);
        assert!(max > 0.9);
    }

    #[test]
    fn test_box_filter() {
        let mut filter = BoxFilter::<f32>::new(10);

        // Set length to 5
        filter.set(5);

        // Process a step function
        let mut output = Vec::new();
        for i in 0..20 {
            let input = if i >= 10 { 1.0 } else { 0.0 };
            output.push(filter.process(input));
        }

        // Check that the output ramps up over 5 samples
        assert_eq!(output[9], 0.0);
        assert_eq!(output[10], 0.2);
        assert_eq!(output[11], 0.4);
        assert_eq!(output[12], 0.6);
        assert_eq!(output[13], 0.8);
        assert_eq!(output[14], 1.0);
    }

    #[test]
    fn test_box_stack_filter() {
        let mut filter = BoxStackFilter::<f32>::new(20, 3);

        // Process a step function
        let mut output = Vec::new();
        for i in 0..40 {
            let input = if i >= 20 { 1.0 } else { 0.0 };
            output.push(filter.process(input));
        }

        // Check that the output eventually reaches 1.0
        assert!(output[39] > 0.99);

        // Check that the transition is smooth (no overshoots)
        for i in 1..output.len() {
            assert!(output[i] >= output[i-1]);
        }
    }

    fn test_peak_hold() {
        let mut filter = PeakHold::<f32>::new(5);

        // Process a sequence
        let input = vec![0.1, 0.5, 0.3, 0.2, 0.4, 0.1, 0.0];
        let mut output = Vec::new();

        for &v in &input {
            output.push(filter.process(v));
        }

        // The peak should hold for 5 samples before starting to decay
        assert_eq!(output[0], 0.1);  // First sample is the input
        assert_eq!(output[1], 0.5);  // Peak value
        assert_eq!(output[2], 0.5);  // Hold peak
        assert_eq!(output[3], 0.5);  // Hold peak
        assert_eq!(output[4], 0.5);  // Hold peak
        assert_eq!(output[5], 0.4);  // New peak after window moves
        assert_eq!(output[6], 0.4);  // Continue with new peak
    }

    #[test]
    fn test_peak_decay_linear() {
        let mut filter = PeakDecayLinear::<f32>::new(10);

        // Process a sequence with known values
        let input = vec![0.1, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut output = Vec::new();

        // Reset filter to ensure clean state
        filter.reset();

        for &v in &input {
            output.push(filter.process(v));
        }

        // The output should follow a linear decay from peak over 10 samples
        let decay_per_sample = 0.5 / 10.0;  // 0.05 per sample

        // Check values with appropriate epsilon
        assert!((output[0] - 0.1).abs() < 1e-6);  // First sample
        assert!((output[1] - 0.5).abs() < 1e-6);  // Peak value
        assert!((output[2] - 0.45).abs() < 1e-5,  // After first decay step
                "Sample 2 mismatch: expected 0.45, got {}", output[2]);
        assert!((output[3] - 0.40).abs() < 1e-5);
        assert!((output[4] - 0.35).abs() < 1e-5);
        assert!((output[5] - 0.30).abs() < 1e-5);
        assert!((output[6] - 0.25).abs() < 1e-5);
        assert!((output[7] - 0.20).abs() < 1e-5);
        assert!((output[8] - 0.15).abs() < 1e-5);
        assert!((output[9] - 0.10).abs() < 1e-5);
    }
}