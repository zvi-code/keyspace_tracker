//! Prefix Tracker - High-performance bitmap-based existence tracking.
//!
//! This crate provides efficient tracking of which IDs exist for given prefixes,
//! supporting billions of IDs with nanosecond to single-digit microsecond latency.

pub mod prefix_tracker;

pub use prefix_tracker::*;
