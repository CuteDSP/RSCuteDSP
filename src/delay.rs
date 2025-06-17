//! Delay utilities
//!
//! This module provides standalone structures for implementing delay lines with
//! various interpolation methods.

#![allow(unused_imports)]

#[cfg(feature = "std")]
use std::{vec::Vec, marker::PhantomData};

#[cfg(not(feature = "std"))]
use core::marker::PhantomData;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

use num_traits::Float;

/// Single-channel delay buffer
///
/// Access is used with `buffer[index]`, relative to the internal read/write position ("head").
/// This head is moved using `buffer.advance(1)` (or `buffer.advance(n)`), such that
/// `buffer[1] == buffer.advance(1)[0]`.
///
/// The capacity includes both positive and negative indices. For example, a capacity of 100
/// would support using any of the ranges:
///
/// * `buffer[-99]` to `buffer[0]`
/// * `buffer[-50]` to `buffer[49]`
/// * `buffer[0]` to `buffer[99]`
///
/// Although buffers are usually used with historical samples accessed using negative indices
/// e.g. `buffer[-10]`, you could equally use it flipped around (moving the head backwards
/// through the buffer using `buffer.advance(-1)`).
pub struct Buffer<T: Float> {
    buffer_index: usize,
    buffer_mask: usize,
    buffer: Vec<T>,
}

impl<T: Float> Buffer<T> {
    /// Create a new buffer with the specified minimum capacity
    pub fn new(min_capacity: usize) -> Self {
        let mut result = Self {
            buffer_index: 0,
            buffer_mask: 0,
            buffer: Vec::new(),
        };
        result.resize(min_capacity, T::zero());
        result
    }

    /// Resize the buffer to have at least the specified capacity
    pub fn resize(&mut self, min_capacity: usize, value: T) {
        let mut buffer_length = 1;
        while buffer_length < min_capacity {
            buffer_length *= 2;
        }
        self.buffer = vec![value; buffer_length];
        self.buffer_mask = buffer_length - 1;
        self.buffer_index = 0;
    }

    /// Reset the buffer to a specific value
    pub fn reset(&mut self, value: T) {
        for i in 0..self.buffer.len() {
            self.buffer[i] = value;
        }
    }

    /// Access a sample relative to the current position
    pub fn get(&self, offset: isize) -> T {
        let index = (self.buffer_index as isize + offset) as usize & self.buffer_mask;
        self.buffer[index]
    }

    /// Set a sample relative to the current position
    pub fn set(&mut self, offset: isize, value: T) {
        let index = (self.buffer_index as isize + offset) as usize & self.buffer_mask;
        self.buffer[index] = value;
    }

    /// Advance the buffer position by the specified amount
    pub fn advance(&mut self, amount: isize) -> &mut Self {
        if amount >= 0 {
            self.buffer_index = self.buffer_index.wrapping_add(amount as usize);
        } else {
            self.buffer_index = self.buffer_index.wrapping_sub((-amount) as usize);
        }
        self
    }

    /// Write data into the buffer
    pub fn write<D>(&mut self, data: &[D], length: usize)
    where
        D: Copy + Into<T>,
    {
        for i in 0..length {
            self.set(i as isize, data[i].into());
        }
    }

    /// Read data out from the buffer
    pub fn read<D>(&self, length: usize, data: &mut [D])
    where
        T: Into<D>,
        D: Copy,
    {
        for i in 0..length {
            data[i] = self.get(i as isize).into();
        }
    }

    /// Create a view at the current position
    pub fn view(&self) -> View<T> {
        View {
            buffer: self,
            offset: 0,
        }
    }

    /// Create a view at a specific offset from the current position
    pub fn view_at(&self, offset: isize) -> View<T> {
        View {
            buffer: self,
            offset,
        }
    }
}

/// A view into a buffer at a specific position
pub struct View<'a, T: Float> {
    buffer: &'a Buffer<T>,
    offset: isize,
}

impl<'a, T: Float> View<'a, T> {
    /// Access a sample relative to this view's position
    pub fn get(&self, offset: isize) -> T {
        self.buffer.get(self.offset + offset)
    }

    /// Create a new view at an offset from this view
    pub fn offset(&self, offset: isize) -> View<'a, T> {
        View {
            buffer: self.buffer,
            offset: self.offset + offset,
        }
    }
}

/// Multi-channel delay buffer
///
/// This behaves similarly to the single-channel `Buffer`, with the following differences:
///
/// * `buffer.channel(c)` returns a view for a single channel
/// * The constructor and `.resize()` take an additional first `channels` argument.
pub struct MultiBuffer<T: Float> {
    channels: usize,
    stride: usize,
    buffer: Buffer<T>,
}

impl<T: Float> MultiBuffer<T> {
    /// Create a new multi-channel buffer
    pub fn new(channels: usize, capacity: usize) -> Self {
        Self {
            channels,
            stride: capacity,
            buffer: Buffer::new(channels * capacity),
        }
    }

    /// Resize the buffer
    pub fn resize(&mut self, channels: usize, capacity: usize, value: T) {
        self.channels = channels;
        self.stride = capacity;
        self.buffer.resize(channels * capacity, value);
    }

    /// Reset the buffer to a specific value
    pub fn reset(&mut self, value: T) {
        self.buffer.reset(value);
    }

    /// Advance the buffer position
    pub fn advance(&mut self, amount: isize) -> &mut Self {
        self.buffer.advance(amount);
        self
    }

    /// Get a view for a specific channel
    pub fn channel(&self, channel: usize) -> View<T> {
        self.buffer.view_at((channel * self.stride) as isize)
    }

    /// Get a sample at a specific channel and offset
    pub fn get(&self, channel: usize, offset: isize) -> T {
        self.buffer.get((channel * self.stride) as isize + offset)
    }

    /// Set a sample at a specific channel and offset
    pub fn set(&mut self, channel: usize, offset: isize, value: T) {
        self.buffer.set((channel * self.stride) as isize + offset, value);
    }
}

/// Nearest-neighbour interpolator
pub struct InterpolatorNearest<T: Float> {
    _marker: PhantomData<T>,
}

impl<T: Float> InterpolatorNearest<T> {
    /// Create a new nearest-neighbour interpolator
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    /// The number of input samples required
    pub const INPUT_LENGTH: usize = 1;

    /// The latency introduced by the interpolator
    pub fn latency() -> T {
        -T::from(0.5).unwrap()
    }

    /// Interpolate a fractional sample
    pub fn fractional<D>(&self, data: &D, _fractional: T) -> T
    where
        D: core::ops::Index<usize, Output = T>,
    {
        data[0]
    }
}

/// Linear interpolator
pub struct InterpolatorLinear<T: Float> {
    _marker: PhantomData<T>,
}

impl<T: Float> InterpolatorLinear<T> {
    /// Create a new linear interpolator
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    /// The number of input samples required
    pub const INPUT_LENGTH: usize = 2;

    /// The latency introduced by the interpolator
    pub fn latency() -> T {
        T::zero()
    }

    /// Interpolate a fractional sample
    pub fn fractional<D>(&self, data: &D, fractional: T) -> T
    where
        D: core::ops::Index<usize, Output = T>,
    {
        let a = data[0];
        let b = data[1];
        a + fractional * (b - a)
    }
}

/// Spline cubic interpolator
pub struct InterpolatorCubic<T: Float> {
    _marker: PhantomData<T>,
}

impl<T: Float> InterpolatorCubic<T> {
    /// Create a new cubic interpolator
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }

    /// The number of input samples required
    pub const INPUT_LENGTH: usize = 4;

    /// The latency introduced by the interpolator
    pub fn latency() -> T {
        T::one()
    }

    /// Interpolate a fractional sample
    pub fn fractional<D>(&self, data: &D, fractional: T) -> T
    where
        D: core::ops::Index<usize, Output = T>,
    {
        let a = data[0];
        let b = data[1];
        let c = data[2];
        let d = data[3];

        let cb_diff = c - b;
        let half = T::from(0.5).unwrap();
        let k1 = (c - a) * half;
        let two = T::one() + T::one();
        let k3 = k1 + (d - b) * half - cb_diff * two;
        let k2 = cb_diff - k3 - k1;

        b + fractional * (k1 + fractional * (k2 + fractional * k3))
    }
}

/// A delay-line reader which uses an external buffer
pub struct Reader<T: Float, I> {
    interpolator: I,
    _marker: PhantomData<T>,
}

impl<T: Float, I> Reader<T, I> {
    /// Create a new reader with the default interpolator
    pub fn new(interpolator: I) -> Self {
        Self {
            interpolator,
            _marker: PhantomData,
        }
    }

    /// Read a sample from the buffer with the specified delay
    pub fn read<B>(&self, buffer: &B, delay_samples: T) -> T
    where
        I: InterpolatorTrait<T>,
        B: BufferTrait<T>,
    {
        let start_index = delay_samples.floor().to_usize().unwrap_or(0);
        let remainder = delay_samples - T::from(start_index).unwrap();

        // Create a flipped view for the interpolator
        struct Flipped<'a, T: Float, B: BufferTrait<T>> {
            view: &'a B,
            start_index: usize,
            _marker: PhantomData<T>,
        }

        impl<'a, T: Float, B: BufferTrait<T>> core::ops::Index<usize> for Flipped<'a, T, B> {
            type Output = T;

            fn index(&self, index: usize) -> &Self::Output {
                // Delay buffers use negative indices, but interpolators use positive ones
                let offset = -(index as isize) - (self.start_index as isize);
                self.view.get_ref(offset)
            }
        }

        let flipped = Flipped {
            view: buffer,
            start_index,
            _marker: PhantomData,
        };

        self.interpolator.fractional(&flipped, remainder)
    }
}

/// A trait for buffer types that can be used with readers
pub trait BufferTrait<T: Float> {
    /// Get a reference to a sample at the specified offset
    fn get_ref(&self, offset: isize) -> &T;
}

impl<T: Float> BufferTrait<T> for Buffer<T> {
    fn get_ref(&self, offset: isize) -> &T {
        let index = (self.buffer_index as isize + offset) as usize & self.buffer_mask;
        &self.buffer[index]
    }
}

impl<'a, T: Float> BufferTrait<T> for View<'a, T> {
    fn get_ref(&self, offset: isize) -> &T {
        let index = (self.buffer.buffer_index as isize + self.offset + offset) as usize 
            & self.buffer.buffer_mask;
        &self.buffer.buffer[index]
    }
}

/// A trait for interpolator types
pub trait InterpolatorTrait<T: Float> {
    /// Interpolate a fractional sample
    fn fractional<D>(&self, data: &D, fractional: T) -> T
    where
        D: core::ops::Index<usize, Output = T>;
}

impl<T: Float> InterpolatorTrait<T> for InterpolatorNearest<T> {
    fn fractional<D>(&self, data: &D, fractional: T) -> T
    where
        D: core::ops::Index<usize, Output = T>,
    {
        self.fractional(data, fractional)
    }
}

impl<T: Float> InterpolatorTrait<T> for InterpolatorLinear<T> {
    fn fractional<D>(&self, data: &D, fractional: T) -> T
    where
        D: core::ops::Index<usize, Output = T>,
    {
        self.fractional(data, fractional)
    }
}

impl<T: Float> InterpolatorTrait<T> for InterpolatorCubic<T> {
    fn fractional<D>(&self, data: &D, fractional: T) -> T
    where
        D: core::ops::Index<usize, Output = T>,
    {
        self.fractional(data, fractional)
    }
}

/// A single-channel delay-line containing its own buffer
pub struct Delay<T: Float, I> {
    reader: Reader<T, I>,
    buffer: Buffer<T>,
}

impl<T: Float, I: InterpolatorTrait<T>> Delay<T, I> {
    /// Create a new delay line with the specified capacity and interpolator
    pub fn new(interpolator: I, capacity: usize) -> Self {
        Self {
            reader: Reader::new(interpolator),
            buffer: Buffer::new(capacity),
        }
    }

    /// Reset the delay line to a specific value
    pub fn reset(&mut self, value: T) {
        self.buffer.reset(value);
    }

    /// Resize the delay line
    pub fn resize(&mut self, min_capacity: usize, value: T) {
        self.buffer.resize(min_capacity, value);
    }

    /// Read a sample from the delay line
    pub fn read(&self, delay_samples: T) -> T {
        self.reader.read(&self.buffer, delay_samples)
    }

    /// Write a sample to the delay line
    pub fn write(&mut self, value: T) -> &mut Self {
        self.buffer.advance(1);
        self.buffer.set(0, value);
        self
    }
}

/// A multi-channel delay-line with its own buffer
pub struct MultiDelay<T: Float, I> {
    reader: Reader<T, I>,
    channels: usize,
    buffer: MultiBuffer<T>,
}

impl<T: Float, I: InterpolatorTrait<T>> MultiDelay<T, I> {
    /// Create a new multi-channel delay line
    pub fn new(interpolator: I, channels: usize, capacity: usize) -> Self {
        Self {
            reader: Reader::new(interpolator),
            channels,
            buffer: MultiBuffer::new(channels, capacity),
        }
    }

    /// Reset the delay line to a specific value
    pub fn reset(&mut self, value: T) {
        self.buffer.reset(value);
    }

    /// Resize the delay line
    pub fn resize(&mut self, channels: usize, capacity: usize, value: T) {
        self.channels = channels;
        self.buffer.resize(channels, capacity, value);
    }

    /// Read a sample from a specific channel
    pub fn read_channel(&self, channel: usize, delay_samples: T) -> T {
        self.reader.read(&self.buffer.channel(channel), delay_samples)
    }

    /// Read samples from all channels with the same delay
    pub fn read(&self, delay_samples: T, output: &mut [T]) {
        for c in 0..self.channels {
            output[c] = self.read_channel(c, delay_samples);
        }
    }

    /// Read samples from all channels with different delays
    pub fn read_multi(&self, delays: &[T], output: &mut [T]) {
        for c in 0..self.channels {
            output[c] = self.read_channel(c, delays[c]);
        }
    }

    /// Write samples to all channels
    pub fn write(&mut self, data: &[T]) -> &mut Self {
        self.buffer.advance(1);
        for c in 0..self.channels {
            self.buffer.set(c, 0, data[c]);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer() {
        let mut buffer = Buffer::<f32>::new(16);

        // Write some values
        buffer.set(0, 1.0);
        buffer.set(1, 2.0);
        buffer.set(2, 3.0);

        // Check values
        assert_eq!(buffer.get(0), 1.0);
        assert_eq!(buffer.get(1), 2.0);
        assert_eq!(buffer.get(2), 3.0);

        // Advance and check again
        buffer.advance(1);
        assert_eq!(buffer.get(-1), 1.0);
        assert_eq!(buffer.get(0), 2.0);
        assert_eq!(buffer.get(1), 3.0);
    }

    #[test]
    fn test_multi_buffer() {
        let mut buffer = MultiBuffer::<f32>::new(2, 16);

        // Write some values
        buffer.set(0, 0, 1.0);
        buffer.set(0, 1, 2.0);
        buffer.set(1, 0, 3.0);
        buffer.set(1, 1, 4.0);

        // Check values
        assert_eq!(buffer.get(0, 0), 1.0);
        assert_eq!(buffer.get(0, 1), 2.0);
        assert_eq!(buffer.get(1, 0), 3.0);
        assert_eq!(buffer.get(1, 1), 4.0);

        // Advance and check again
        buffer.advance(1);
        assert_eq!(buffer.get(0, -1), 1.0);
        assert_eq!(buffer.get(0, 0), 2.0);
        assert_eq!(buffer.get(1, -1), 3.0);
        assert_eq!(buffer.get(1, 0), 4.0);
    }

    #[test]
    fn test_interpolators() {
        // Create some test data
        let data = [1.0, 2.0, 3.0, 4.0];

        // Test nearest interpolator
        let nearest = InterpolatorNearest::<f32>::new();
        assert_eq!(nearest.fractional(&data, 0.0), 1.0);
        assert_eq!(nearest.fractional(&data, 0.4), 1.0);
        assert_eq!(nearest.fractional(&data, 0.6), 1.0);

        // Test linear interpolator
        let linear = InterpolatorLinear::<f32>::new();
        assert_eq!(linear.fractional(&data, 0.0), 1.0);
        assert_eq!(linear.fractional(&data, 0.5), 1.5);
        assert_eq!(linear.fractional(&data, 1.0), 2.0);

        // Test cubic interpolator
        let cubic = InterpolatorCubic::<f32>::new();
        assert_eq!(cubic.fractional(&data, 0.0), 2.0);
        // The result of cubic interpolation at 0.5 should be between 2.0 and 3.0
        let cubic_result = cubic.fractional(&data, 0.5);
        assert!(cubic_result > 2.0 && cubic_result < 3.0);
    }

    #[test]
    fn test_delay() {
        let interpolator = InterpolatorLinear::<f32>::new();
        let mut delay = Delay::new(interpolator, 16);

        // Write some values
        delay.write(1.0).write(2.0).write(3.0);

        // Read with different delays
        assert_eq!(delay.read(0.0), 3.0);
        assert_eq!(delay.read(1.0), 2.0);
        assert_eq!(delay.read(2.0), 1.0);
        assert_eq!(delay.read(0.5), 2.5); // Interpolated
    }

    #[test]
    fn test_multi_delay() {
        let interpolator = InterpolatorLinear::<f32>::new();
        let mut delay = MultiDelay::new(interpolator, 2, 16);

        // Write some values
        delay.write(&[1.0, 3.0]).write(&[2.0, 4.0]);

        // Read with different delays
        let mut output = [0.0, 0.0];
        delay.read(0.0, &mut output);
        assert_eq!(output, [2.0, 4.0]);

        delay.read(1.0, &mut output);
        assert_eq!(output, [1.0, 3.0]);

        // Read with different delays per channel
        delay.read_multi(&[0.0, 1.0], &mut output);
        assert_eq!(output, [2.0, 3.0]);
    }
}
