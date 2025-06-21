//! User-defined mapping functions
//!
//! This module provides various curve implementations for mapping values,
//! including linear, cubic, and reciprocal functions.

#![allow(unused_imports)]


#[cfg(feature = "std")]
use std::{vec::Vec, cmp::Ordering};

#[cfg(not(feature = "std"))]
use core::cmp::Ordering;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

use num_complex::Complex;
use num_traits::Float;

/// Linear map for real values
#[derive(Clone, Copy, Debug)]
pub struct Linear<T: Float> {
    a1: T,
    a0: T,
}

impl<T: Float> Linear<T> {
    pub fn new() -> Self {
        Self::with_values(T::zero(), T::one())
    }

    pub fn with_values(a0: T, a1: T) -> Self {
        Self { a1, a0 }
    }

    /// Construct by from/to value pairs
    pub fn from_points(x0: T, x1: T, y0: T, y1: T) -> Self {
        let a1 = if x0 == x1 {
            T::zero()
        } else {
            (y1 - y0) / (x1 - x0)
        };
        let a0 = y0 - x0 * a1;
        Self { a1, a0 }
    }

    pub fn evaluate(&self, x: T) -> T {
        self.a0 + x * self.a1
    }

    pub fn derivative(&self) -> T {
        self.a1
    }

    /// Returns the inverse map (with some numerical error)
    pub fn inverse(&self) -> Self {
        let inv_a1 = T::one() / self.a1;
        Self::with_values(-self.a0 * inv_a1, inv_a1)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cubic<T: Float> {
    x_start: T,
    a0: T,
    a1: T,
    a2: T,
    a3: T,
}

impl<T: Float> Cubic<T> {
    pub fn new(x_start: T, a0: T, a1: T, a2: T, a3: T) -> Self {
        Self {
            x_start,
            a0,
            a1,
            a2,
            a3,
        }
    }

    pub fn evaluate(&self, x: T) -> T {
        let x = x - self.x_start;
        self.a0 + x * (self.a1 + x * (self.a2 + x * self.a3))
    }

    pub fn start(&self) -> T {
        self.x_start
    }

    pub fn derivative(&self) -> Self {
        Self::new(
            self.x_start,
            self.a1,
            self.a2 * (T::one() + T::one()),
            self.a3 * (T::one() + T::one() + T::one()),
            T::zero(),
        )
    }

    pub fn derivative_at(&self, x: T) -> T {
        let x = x - self.x_start;
        self.a1 + x * (self.a2 * (T::one() + T::one()) + x * self.a3 * (T::one() + T::one() + T::one()))
    }

    fn gradient(x0: T, x1: T, y0: T, y1: T) -> T {
        (y1 - y0) / (x1 - x0)
    }

    fn ensure_monotonic(curve_grad: &mut T, grad_a: T, grad_b: T) {
        if (grad_a <= T::zero() && grad_b >= T::zero())
            || (grad_a >= T::zero() && grad_b <= T::zero())
        {
            *curve_grad = T::zero();
        } else {
            if curve_grad.abs() > grad_a * (T::one() + T::one() + T::one()) {
                *curve_grad = grad_a * (T::one() + T::one() + T::one());
            }
            if curve_grad.abs() > grad_b * (T::one() + T::one() + T::one()) {
                *curve_grad = grad_b * (T::one() + T::one() + T::one());
            }
        }
    }

    pub fn hermite(x0: T, x1: T, y0: T, y1: T, g0: T, g1: T) -> Self {
        let x_scale = T::one() / (x1 - x0);
        let x_scale_sq = x_scale * x_scale;
        
        let three = T::one() + T::one() + T::one();
        let two = T::one() + T::one();

        Self::new(
            x0,
            y0,
            g0,
            (three * (y1 - y0) * x_scale - two * g0 - g1) * x_scale,
            (two * (y0 - y1) * x_scale + g0 + g1) * x_scale_sq,
        )
    }
}

/// A point in a cubic segment curve
#[derive(Clone, Debug)]
struct Point<T: Float> {
    x: T,
    y: T,
    line_grad: T,
    curve_grad: T,
    has_curve_grad: bool,
}

impl<T: Float> Point<T> {
    fn new(x: T, y: T) -> Self {
        Self {
            x,
            y,
            line_grad: T::zero(),
            curve_grad: T::zero(),
            has_curve_grad: false,
        }
    }
}

impl<T: Float> PartialEq for Point<T> {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
    }
}

impl<T: Float> PartialOrd for Point<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.x.partial_cmp(&other.x)
    }
}

/// Smooth interpolation (optionally monotonic) between points, using cubic segments.
#[derive(Clone, Debug)]
pub struct CubicSegmentCurve<T: Float> {
    points: Vec<Point<T>>,
    first: Point<T>,
    last: Point<T>,
    segments: Vec<Cubic<T>>,
    pub low_grad: T,
    pub high_grad: T,
}

impl<T: Float> CubicSegmentCurve<T> {
    /// Create a new empty cubic segment curve
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            first: Point::new(T::zero(), T::zero()),
            last: Point::new(T::zero(), T::zero()),
            segments: Vec::new(),
            low_grad: T::zero(),
            high_grad: T::zero(),
        }
    }
    
    /// Clear existing points and segments
    pub fn clear(&mut self) {
        self.points.clear();
        self.segments.clear();
        self.first = Point::new(T::zero(), T::zero());
        self.last = Point::new(T::zero(), T::zero());
    }
    
    /// Add a new point, but does not recalculate the segments
    pub fn add(&mut self, x: T, y: T, corner: bool) -> &mut Self {
        self.points.push(Point::new(x, y));
        if corner {
            self.points.push(Point::new(x, y));
        }
        self
    }
    
    /// Find the segment containing a given x value
    fn find_segment(&self, x: T) -> &Cubic<T> {
        // Binary search
        let mut low = 0;
        let mut high = self.segments.len();
        
        while low + 1 < high {
            let mid = (low + high) / 2;
            if self.segments[mid].start() <= x {
                low = mid;
            } else {
                high = mid;
            }
        }
        
        &self.segments[low]
    }
    
    /// Reads a value out from the curve
    pub fn evaluate(&self, x: T) -> T {
        if x <= self.first.x {
            self.first.y + (x - self.first.x) * self.low_grad
        } else if x >= self.last.x {
            self.last.y + (x - self.last.x) * self.high_grad
        } else {
            self.find_segment(x).evaluate(x)
        }
    }
    
    // Make sure update() properly sets the gradients
    /// Recalculates the segments
    pub fn update(&mut self, monotonic: bool, extend_grad: bool, monotonic_factor: T) {
        if self.points.is_empty() {
            self.add(T::zero(), T::zero(), false);
        }
        
        // Sort points by x value
        self.points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));
        
        self.segments.clear();
        
        // Calculate the point-to-point gradients
        for i in 1..self.points.len() {
            let prev_x = self.points[i-1].x;
            let next_x = self.points[i].x;
            let prev_y = self.points[i-1].y;
            let next_y = self.points[i].y;
            
            if prev_x != next_x {
                self.points[i-1].line_grad = (next_y - prev_y) / (next_x - prev_x);
            } else {
                self.points[i-1].line_grad = T::zero();
            }
        }
        
        // Reset curve gradients
        for p in &mut self.points {
            p.has_curve_grad = false;
        }
        
        // Set boundary gradients
        if !self.points.is_empty() {
            self.points[0].curve_grad = self.low_grad;
            self.points[0].has_curve_grad = true;
            let last_idx = self.points.len() - 1;
            self.points[last_idx].curve_grad = self.high_grad;
            self.points[last_idx].has_curve_grad = true;
        }
        
        // Calculate curve gradient where we know it
        for i in 1..self.points.len().saturating_sub(1) {
            let (left, right) = self.points.split_at_mut(i);
            let p0 = &left[i-1];
            let (p1_slice, p2_slice) = right.split_at_mut(1);
            let p1 = &mut p1_slice[0];
            let p2 = &p2_slice[0];

            if p0.x != p1.x && p1.x != p2.x {
                let half = T::from(0.5).unwrap();
                p1.curve_grad = (p0.line_grad + p1.line_grad) * half;
                p1.has_curve_grad = true;
            }
        }
        
        // Fill in missing curve gradients
        for i in 1..self.points.len() {
            let (p1, p2) = {
                let (left, right) = self.points.split_at_mut(i);
                (&mut left[i-1], &mut right[0])
            };
            
            if p1.x == p2.x {
                continue;
            }
            
            if p1.has_curve_grad {
                if !p2.has_curve_grad {
                    let two = T::one() + T::one();
                    p2.curve_grad = two * p1.line_grad - p1.curve_grad;
                }
            } else if p2.has_curve_grad {
                let two = T::one() + T::one();
                p1.curve_grad = two * p1.line_grad - p2.curve_grad;
            } else {
                p1.curve_grad = p1.line_grad;
                p2.curve_grad = p1.line_grad;
            }
        }
        
        // Apply monotonicity constraints if requested
        if monotonic {
            for i in 1..self.points.len() {
                let (p1, p2) = {
                    let (left, right) = self.points.split_at_mut(i);
                    (&mut left[i-1], &mut right[0])
                };
                
                if p1.x != p2.x {
                    if p1.line_grad >= T::zero() {
                        p1.curve_grad = T::zero().max(p1.curve_grad.min(p1.line_grad * monotonic_factor));
                        p2.curve_grad = T::zero().max(p2.curve_grad.min(p1.line_grad * monotonic_factor));
                    } else {
                        p1.curve_grad = T::zero().min(p1.curve_grad.max(p1.line_grad * monotonic_factor));
                        p2.curve_grad = T::zero().min(p2.curve_grad.max(p1.line_grad * monotonic_factor));
                    }
                }
            }
        }
        
        // Create segments
        for i in 1..self.points.len() {
            let p1 = &self.points[i-1];
            let p2 = &self.points[i];
            
            if p1.x != p2.x {
                self.segments.push(Cubic::hermite(
                    p1.x, p2.x, p1.y, p2.y, p1.curve_grad, p2.curve_grad
                ));
            }
        }
        
        // Store first and last points
        if !self.points.is_empty() {
            self.first = self.points[0].clone();
            self.last = self.points.last().unwrap().clone();
            
            // Update gradients if requested
            if extend_grad && !self.segments.is_empty() {
                // Calculate initial gradient from first segment
                self.low_grad = if self.points.len() > 1 {
                    // Get gradient from first segment at start point
                    let first_segment = &self.segments[0];
                    first_segment.derivative_at(self.first.x)
                } else {
                    T::zero()
                };
                
                if self.points.len() > 1 {
                    let last_idx = self.points.len() - 1;
                    let last2_idx = last_idx - 1;
                    if self.points[last_idx].x != self.points[last2_idx].x || 
                       self.points[last_idx].y == self.points[last2_idx].y {
                        self.high_grad = self.segments.last().unwrap().derivative_at(self.last.x);
                    }
                }
            }
        }
        
        // Ensure proper gradient calculation at endpoints
        if extend_grad && !self.segments.is_empty() {
            if self.points.len() > 1 {
                self.low_grad = self.segments[0].derivative_at(self.first.x);
                self.high_grad = self.segments.last().unwrap().derivative_at(self.last.x);
            }
        }
    }
    
    /// Get the derivative of the curve
    pub fn derivative(&self) -> Self {
        let mut result = self.clone();
        result.first.y = self.low_grad;
        result.last.y = self.high_grad;
        result.low_grad = T::zero();
        result.high_grad = T::zero();
        
        for i in 0..result.segments.len() {
            result.segments[i] = self.segments[i].derivative();
        }
        
        result
    }
    
    /// Evaluate the derivative at a point
    pub fn derivative_at(&self, x: T) -> T {
        if x < self.first.x {
            return self.low_grad;
        }
        if x >= self.last.x {
            return self.high_grad;
        }
        self.find_segment(x).derivative_at(x)
    }
    
    /// Get access to the segments
    pub fn segments(&self) -> &[Cubic<T>] {
        &self.segments
    }
}

/// A warped-range map, based on 1/x
#[derive(Clone, Copy, Debug)]
pub struct Reciprocal<T: Float> {
    a: T,
    b: T,
    c: T,
    d: T,
}

impl<T: Float> Reciprocal<T> {
    /// Create a new reciprocal map with the given coefficients
    pub fn new(a: T, b: T, c: T, d: T) -> Self {
        Self { a, b, c, d }
    }
    
    /// Create a reciprocal map from three points
    pub fn from_points(x0: T, x1: T, x2: T, y0: T, y1: T, y2: T) -> Self {
        let kx = (x1 - x0) / (x2 - x1);
        let ky = (y1 - y0) / (y2 - y1);
        
        let a = (kx * x2) * y0 - (ky * x0) * y2;
        let b = ky * y2 - kx * y0;
        let c = kx * x2 - ky * x0;
        let d = ky - kx;
        
        Self { a, b, c, d }
    }
    
    /// Create a reciprocal map from three y values (using default x range 0, 0.5, 1)
    pub fn from_y_values(y0: T, y1: T, y2: T) -> Self {
        Self::from_points(
            T::zero(),
            T::from(0.5).unwrap(),
            T::one(),
            y0, y1, y2
        )
    }
    
    /// Decent approximation to the Bark scale
    pub fn bark_scale() -> Self {
        Self::from_points(
            T::one(),
            T::from(10.0).unwrap(),
            T::from(24.0).unwrap(),
            T::from(60.0).unwrap(),
            T::from(1170.0).unwrap(),
            T::from(13500.0).unwrap()
        )
    }
    
    /// Returns a map from 0-1 to the given (non-negative) Hz range.
    pub fn bark_range(low_hz: T, high_hz: T) -> Self {
        let bark = Self::bark_scale();
        let low_bark = bark.inverse_value(low_hz);
        let high_bark = bark.inverse_value(high_hz);
        let half = T::from(0.5).unwrap();
        
        Self::from_y_values(low_bark, (low_bark + high_bark) * half, high_bark)
            .then(&bark)
    }
    
    /// Evaluate the reciprocal function at point x
    pub fn evaluate(&self, x: T) -> T {
        (self.a + self.b * x) / (self.c + self.d * x)
    }
    
    /// Returns the inverse map
    pub fn inverse(&self) -> Self {
        Self::new(-self.a, self.c, self.b, -self.d)
    }
    
    /// Evaluate the inverse at point y
    pub fn inverse_value(&self, y: T) -> T {
        (self.c * y - self.a) / (self.b - self.d * y)
    }
    
    /// Evaluate the derivative at point x
    pub fn derivative_at(&self, x: T) -> T {
        let l = self.c + self.d * x;
        (self.b * self.c - self.a * self.d) / (l * l)
    }
    
    /// Combine two `Reciprocal`s together in sequence
    pub fn then(&self, other: &Self) -> Self {
        Self::new(
            other.a * self.c + other.b * self.a,
            other.a * self.d + other.b * self.b,
            other.c * self.c + other.d * self.a,
            other.c * self.d + other.d * self.b
        )
    }
}

// make test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let linear = Linear::from_points(0.0, 1.0, 0.0, 2.0);
        assert_eq!(linear.evaluate(0.5), 1.0);
        assert_eq!(linear.derivative(), 2.0);
        let inverse = linear.inverse();
        assert_eq!(inverse.evaluate(1.0), 0.5);
    }

    #[test]
    fn test_cubic() {
        let cubic = Cubic::new(0.0, 1.0, 2.0, 3.0, 4.0);
        let x = 1.5;
        let expected_value = 1.0 + (x * (2.0 + x * (3.0 + x * 4.0))); // cubic.evaluate(1.5)
        let expected_derivative = 2.0 + x * (2.0 * 3.0 + x * 3.0 * 4.0); // cubic.derivative_at(1.5)

        assert!((cubic.evaluate(x) - expected_value).abs() < 1e-6);
        assert!((cubic.derivative_at(x) - expected_derivative).abs() < 1e-6);
    }

    #[test]
    fn test_reciprocal() {
        let reciprocal = Reciprocal::from_points(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let x = 2.5;
        let value = reciprocal.evaluate(x);
        let derivative = reciprocal.derivative_at(x);

        // Validate that values are finite (not NaN or Inf)
        assert!(value.is_finite());
        assert!(derivative.is_finite());
    }

    
}