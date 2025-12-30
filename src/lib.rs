//! # prefix_tracker
//!
//! High-performance, lock-free bitmap-based existence tracking for key-value systems.
//!
//! ## Features
//!
//! - **Sub-nanosecond operations**: Core operations (test, set, claim) in 2-7ns
//! - **Lock-free concurrency**: Atomic operations with no mutex overhead
//! - **SIMD optimized**: ARM NEON and x86_64 AVX2/POPCNT acceleration
//! - **Memory efficient**: 1 bit per ID, auto-growing bitmaps
//! - **Hierarchical support**: Track (id, sub_id) pairs efficiently
//! - **Rich iteration**: Sequential, random, write, delete, partitioned iterators
//!
//! ## Quick Start
//!
//! ```rust
//! use prefix_tracker::{PrefixGroupsTracker, TrackerConfig};
//!
//! // Create a registry and register a tracker
//! let groups = PrefixGroupsTracker::new();
//! groups.register(TrackerConfig::simple("user:"));
//!
//! // Get tracker and add IDs
//! let users = groups.get("user:").unwrap();
//! users.add(42);
//! users.add(1337);
//!
//! // Check existence
//! assert!(users.exists(42));
//! assert!(!users.exists(999));
//!
//! // Iterate over set IDs
//! for (id, _) in users.iter().set_only().sequential() {
//!     println!("User ID: {}", id);
//! }
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   PrefixGroupsTracker                       │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
//! │  │ "user:"     │ │ "session:"  │ │ "cache:"    │  ...      │
//! │  │ PrefixTracker│ │ PrefixTracker│ │ PrefixTracker│          │
//! │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘           │
//! │         │               │               │                   │
//! │         ▼               ▼               ▼                   │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
//! │  │AtomicBitmap │ │AtomicBitmap │ │AtomicBitmap │           │
//! │  │ [0100101...] │ │ [1100010...]│ │ [0011100...]│           │
//! │  └─────────────┘ └─────────────┘ └─────────────┘           │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Core Types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`AtomicBitmap`] | Lock-free atomic bitmap with SIMD-accelerated operations |
//! | [`PrefixTracker`] | Single-prefix tracker (simple or hierarchical mode) |
//! | [`PrefixGroupsTracker`] | Registry of multiple trackers with group operations |
//! | [`TrackerConfig`] | Configuration for tracker behavior |
//!
//! ## Iterators
//!
//! | Iterator | Description |
//! |----------|-------------|
//! | [`SequentialIter`] | In-order iteration over IDs |
//! | [`RandomIter`] | Pseudo-random traversal (no allocation) |
//! | [`WriteIter`] | Atomically claim unique IDs |
//! | [`DeleteIter`] | Exclusive iterator that clears IDs |
//! | [`PartitionedIter`] | Parallel iteration with disjoint ranges |
//! | [`GroupIter`] | Iterate across multiple prefixes |
//! | [`GroupWriteIter`] | Claim IDs across prefixes with policies |
//!
//! ## Examples
//!
//! ### Simple Tracking
//!
//! ```rust
//! use prefix_tracker::PrefixTracker;
//!
//! let tracker = PrefixTracker::simple("vec:");
//!
//! // Add IDs
//! tracker.add(0);
//! tracker.add(100);
//! tracker.add(999);
//!
//! // Check and remove
//! assert!(tracker.exists(100));
//! tracker.remove(100);
//! assert!(!tracker.exists(100));
//!
//! // Count
//! assert_eq!(tracker.count(), 2);
//! ```
//!
//! ### Hierarchical Tracking
//!
//! ```rust
//! use prefix_tracker::PrefixTracker;
//!
//! // Track (id, sub_id) pairs - e.g., hash fields
//! let tracker = PrefixTracker::hierarchical("hash:");
//!
//! tracker.add_pair(1, 0);    // hash:1 field 0
//! tracker.add_pair(1, 5);    // hash:1 field 5
//! tracker.add_pair(2, 10);   // hash:2 field 10
//!
//! assert!(tracker.exists_pair(1, 5));
//! assert!(!tracker.exists_pair(1, 99));
//! ```
//!
//! ### Concurrent ID Claiming
//!
//! ```rust
//! use prefix_tracker::PrefixTracker;
//! use std::sync::Arc;
//! use std::thread;
//!
//! let tracker = Arc::new(PrefixTracker::simple("id:"));
//!
//! // Multiple threads claim unique IDs
//! let handles: Vec<_> = (0..4).map(|_| {
//!     let t = tracker.clone();
//!     thread::spawn(move || {
//!         let mut claimed = Vec::new();
//!         for id in 0..1000u64 {
//!             if t.claim(id) {
//!                 claimed.push(id);
//!             }
//!         }
//!         claimed
//!     })
//! }).collect();
//!
//! let all_claimed: Vec<Vec<u64>> = handles.into_iter()
//!     .map(|h| h.join().unwrap())
//!     .collect();
//!
//! // Each ID claimed by exactly one thread
//! let total: usize = all_claimed.iter().map(|v| v.len()).sum();
//! assert_eq!(total, 1000);
//! ```
//!
//! ### Group Iteration with Policies
//!
//! ```rust
//! use prefix_tracker::{PrefixGroupsTracker, TrackerConfig, ClaimPolicy};
//!
//! let groups = PrefixGroupsTracker::new();
//! groups.register(TrackerConfig::simple("a:"));
//! groups.register(TrackerConfig::simple("b:"));
//!
//! // Claim IDs round-robin across prefixes
//! let mut writer = groups.iter().write(ClaimPolicy::RoundRobin);
//! let mut count = 0;
//! while let Some(item) = writer.next() {
//!     println!("Claimed {}:{}", item.prefix, item.id);
//!     count += 1;
//!     if count >= 10 { break; }
//! }
//! ```
//!
//! ## Performance
//!
//! | Operation | Latency | Notes |
//! |-----------|---------|-------|
//! | `exists` | 1.6 ns | L1 cache hit |
//! | `add` | 7.2 ns | Atomic OR |
//! | `claim` | 2.4 ns | CAS loop |
//! | `find_next_set` | 4.0 ns | SIMD accelerated |
//!
//! ## Platform Optimizations
//!
//! Enable native CPU features for best performance:
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo build --release
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod prefix_tracker;

pub use prefix_tracker::*;
