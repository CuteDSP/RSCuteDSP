//! Linear Example
//!
//! This example demonstrates how to use the linear algebra and expression template system
//! in the Signalsmith DSP library.

use cute_dsp::linear::*;
use num_complex::Complex;

fn main() {
    println!("Linear Example");
    println!("==============");

    // Example 1: Constant expression
    constant_expression_example();

    // Example 2: Math operations
    math_ops_example();

    // Example 3: Binary operations
    binary_ops_example();

    // Example 4: Split pointer usage
    split_pointer_example();

    // Example 5: Fill real array from expression
    fill_real_example();
}

fn constant_expression_example() {
    println!("\nConstant Expression Example:");
    let expr = Expression::new(ConstantExpr { value: 3.14f32 });
    println!("expr.get(0) = {:.2}", expr.get(0));
    println!("expr.get(10) = {:.2}", expr.get(10));
}

fn math_ops_example() {
    println!("\nMath Operations Example:");
    let expr = Expression::new(ConstantExpr { value: 4.0f32 });
    println!("abs: {:.2}", expr.abs().get(0));
    println!("sqrt: {:.2}", expr.sqrt().get(0));
    println!("norm: {:.2}", expr.norm().get(0));
}

fn binary_ops_example() {
    println!("\nBinary Operations Example:");
    let expr1 = Expression::new(ConstantExpr { value: 5.0f32 });
    let expr2 = Expression::new(ConstantExpr { value: 2.0f32 });
    println!("add: {:.2}", expr1.add(&expr2).get(0));
    println!("sub: {:.2}", expr1.sub(&expr2).get(0));
    println!("mul: {:.2}", expr1.mul(&expr2).get(0));
    println!("div: {:.2}", expr1.div(&expr2).get(0));
}

fn split_pointer_example() {
    println!("\nSplit Pointer Example:");
    let mut real = [1.0f32, 2.0, 3.0];
    let mut imag = [4.0f32, 5.0, 6.0];
    let split_ptr = SplitPointer::new(real.as_mut_ptr(), imag.as_mut_ptr());
    unsafe {
        for i in 0..3 {
            let c = split_ptr.get(i);
            println!("split_ptr[{}] = Complex {{ re: {:.1}, im: {:.1} }}", i, c.re, c.im);
        }
    }
}

fn fill_real_example() {
    println!("\nFill Real Array Example:");
    let linear = Linear::new();
    let expr = Expression::new(ConstantExpr { value: 7.0f32 });
    let mut data = [0.0f32; 5];
    linear.fill_real(data.as_mut_ptr(), &expr, 5);
    print!("Filled data: ");
    for v in &data {
        print!("{:.1} ", v);
    }
    println!();
}
