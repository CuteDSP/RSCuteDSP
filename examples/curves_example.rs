//! Curves Example
//!
//! This example demonstrates how to use the curve interpolation functionality
//! in the Signalsmith DSP library.

use signalsmith_dsp::curves::{Linear, Cubic, CubicSegmentCurve, Reciprocal};

fn main() {
    println!("Curves Example");
    println!("==============");

    // Example 1: Linear interpolation
    linear_example();

    // Example 2: Cubic curve interpolation
    cubic_example();

    // Example 3: Cubic segment curve
    cubic_segment_curve_example();

    // Example 4: Reciprocal curve (useful for audio frequency mapping)
    reciprocal_example();
}

fn linear_example() {
    println!("\nLinear Interpolation Example:");

    // Create a linear curve from points (0, 1) to (1, 3)
    let linear = Linear::<f32>::from_points(0.0, 1.0, 1.0, 3.0);

    // Evaluate at different points
    println!("Value at x=0.0: {}", linear.evaluate(0.0));
    println!("Value at x=0.5: {}", linear.evaluate(0.5));
    println!("Value at x=1.0: {}", linear.evaluate(1.0));

    // Get the derivative (slope)
    println!("Derivative (slope): {}", linear.derivative());

    // Create an inverse mapping
    let inverse = linear.inverse();
    println!("Inverse mapping of y=2.0: {}", inverse.evaluate(2.0));
}

fn cubic_example() {
    println!("\nCubic Curve Example:");

    // Create a cubic curve using Hermite interpolation
    // Parameters: x0, x1, y0, y1, g0, g1 (start x, end x, start y, end y, start gradient, end gradient)
    let cubic = Cubic::<f32>::hermite(0.0, 1.0, 0.0, 1.0, 0.0, 2.0);

    // Evaluate at different points
    println!("Value at x=0.0: {}", cubic.evaluate(0.0));
    println!("Value at x=0.25: {}", cubic.evaluate(0.25));
    println!("Value at x=0.5: {}", cubic.evaluate(0.5));
    println!("Value at x=0.75: {}", cubic.evaluate(0.75));
    println!("Value at x=1.0: {}", cubic.evaluate(1.0));

    // Get the derivative at a point
    println!("Derivative at x=0.5: {}", cubic.derivative_at(0.5));

    // Approximate a smooth cubic curve through 4 points by chaining two Hermite cubics
    // This will evaluate at x=1.5 (between x1=1.0, x2=2.0) using a Hermite curve
    let x0 = 0.0;
    let x1 = 1.0;
    let x2 = 2.0;
    let x3 = 3.0;
    let y0 = 0.0;
    let y1 = 2.0;
    let y2 = 1.0;
    let y3 = 3.0;

    // Estimate tangents for Hermite: central difference where possible
    let g1 = (y2 - y0) / (x2 - x0);
    let g2 = (y3 - y1) / (x3 - x1);

    // Hermite curve between (x1,y1) - (x2,y2) using g1, g2
    let smooth = Cubic::<f32>::hermite(x1, x2, y1, y2, g1, g2);
    println!("Smooth curve value at x=1.5: {}", smooth.evaluate(1.5));
}

fn cubic_segment_curve_example() {
    println!("\nCubic Segment Curve Example:");

    // Create a new cubic segment curve
    let mut curve = CubicSegmentCurve::<f32>::new();

    // Add points to the curve
    curve.add(0.0, 0.0, false)
         .add(1.0, 2.0, false)
         .add(2.0, 1.0, false)
         .add(3.0, 3.0, false);

    // Update the curve (compute the segments)
    curve.update(true, false, 1.0);

    // Evaluate at different points
    println!("Value at x=0.5: {}", curve.evaluate(0.5));
    println!("Value at x=1.5: {}", curve.evaluate(1.5));
    println!("Value at x=2.5: {}", curve.evaluate(2.5));

    // Get the derivative at a point
    println!("Derivative at x=1.0: {}", curve.derivative_at(1.0));

    // Get the number of segments
    println!("Number of segments: {}", curve.segments().len());
}

fn reciprocal_example() {
    println!("\nReciprocal Curve Example:");

    // Create a reciprocal curve from points
    // Parameters: x0, x1, x2, y0, y1, y2
    let reciprocal = Reciprocal::<f32>::from_points(100.0, 1000.0, 10000.0, 0.0, 1.0, 2.0);

    // Evaluate at different frequencies
    println!("Value at 100 Hz: {}", reciprocal.evaluate(100.0));
    println!("Value at 500 Hz: {}", reciprocal.evaluate(500.0));
    println!("Value at 1000 Hz: {}", reciprocal.evaluate(1000.0));
    println!("Value at 5000 Hz: {}", reciprocal.evaluate(5000.0));
    println!("Value at 10000 Hz: {}", reciprocal.evaluate(10000.0));

    // Create a Bark scale mapping (psychoacoustic frequency scale)
    let bark = Reciprocal::<f32>::bark_scale();
    println!("Bark value at 100 Hz: {}", bark.evaluate(100.0));
    println!("Bark value at 1000 Hz: {}", bark.evaluate(1000.0));
    println!("Bark value at 10000 Hz: {}", bark.evaluate(10000.0));

    // Create a Bark scale mapping for a specific frequency range
    let bark_range = Reciprocal::<f32>::bark_range(20.0, 20000.0);
    println!("Normalized Bark value at 100 Hz: {}", bark_range.evaluate(100.0));
    println!("Normalized Bark value at 1000 Hz: {}", bark_range.evaluate(1000.0));
    println!("Normalized Bark value at 10000 Hz: {}", bark_range.evaluate(10000.0));
}
