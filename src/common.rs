//! Common definitions and helper functions used by the rest of the library

#![allow(unused_imports)]

#[cfg(feature = "std")]
use std::f32::consts::PI;

#[cfg(not(feature = "std"))]
use core::f32::consts::PI;

/// The version of the library
pub const VERSION_MAJOR: u32 = 1;
pub const VERSION_MINOR: u32 = 6;
pub const VERSION_PATCH: u32 = 2;
pub const VERSION_STRING: &str = "1.6.2";

/// Check if the library version is compatible (semver).
/// Major versions are not compatible with each other.
/// Minor and patch versions are backwards-compatible.
#[inline]
pub const fn version_check(major: u32, minor: u32, patch: u32) -> bool {
    major == VERSION_MAJOR
        && (VERSION_MINOR > minor
            || (VERSION_MINOR == minor && VERSION_PATCH >= patch))
}

/// Macro to check the library version is compatible (semver).
#[macro_export]
macro_rules! version_check {
    ($major:expr, $minor:expr, $patch:expr) => {
        const _: bool = $crate::common::version_check($major, $minor, $patch);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_check() {
        assert!(version_check(1, 6, 2));
        assert!(version_check(1, 6, 1));
        assert!(version_check(1, 5, 0));
        assert!(!version_check(2, 0, 0));
        assert!(!version_check(1, 7, 0));
    }
}