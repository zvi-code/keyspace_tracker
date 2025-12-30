//! Prefix Tracker - High-performance bitmap-based existence tracking.
//!
//! This module provides efficient tracking of which IDs exist for given prefixes,
//! supporting billions of IDs with nanosecond to single-digit microsecond latency.
//!
//! # Architecture
//!
//! - `AtomicBitmap`: Lock-free atomic bitmap with concurrent read/write support
//! - `PrefixTracker`: Single-prefix tracker (simple or hierarchical)
//! - `PrefixGroupsTracker`: Registry of trackers, one per prefix
//! - Various iterators for sequential, random, write, delete, and partitioned access
//!
//! # Example
//!
//! ```
//! use prefix_tracker::{PrefixGroupsTracker, TrackerConfig};
//!
//! let groups = PrefixGroupsTracker::new();
//!
//! // Register a simple tracker
//! groups.register(TrackerConfig::simple("vec:"));
//! let tracker = groups.get("vec:").unwrap();
//!
//! // Add IDs
//! tracker.add(0);
//! tracker.add(100);
//!
//! // Iterate set bits
//! for (id, _) in tracker.iter().set_only().sequential() {
//!     println!("ID: {}", id);
//! }
//! ```

mod bitmap;
mod bitmap_simd;
mod config;
mod group_iter;
mod iterators;
mod tracker;

pub use bitmap::AtomicBitmap;
pub use config::{
    BitFilter, ClaimPolicy, IdRange, IterOrder, PrefixFilter, PrefixWeight, TrackerConfig,
};
pub use group_iter::{
    GroupDeleteIter, GroupItem, GroupIter, GroupIterBuilder, GroupPartitionedIter, GroupWriteIter,
};
pub use iterators::{
    DeleteIter, PartitionedIter, RandomIter, SequentialIter, TrackerItem, TrackerIterBuilder,
    WriteIter,
};
pub use tracker::PrefixTracker;

use std::sync::Arc;

use ahash::RandomState as AHasher;
use dashmap::DashMap;

/// Type alias for DashMap with AHash (2-10x faster than default SipHash)
type FastDashMap<K, V> = DashMap<K, V, AHasher>;

/// Registry of [`PrefixTracker`]s, one per prefix.
///
/// `PrefixGroupsTracker` is a thread-safe collection that manages multiple
/// trackers, each identified by a unique prefix string. It provides:
///
/// - Concurrent registration and lookup of trackers
/// - Group-level iteration across all prefixes
/// - Automatic tracker creation with default configuration
///
/// # Thread Safety
///
/// All operations are thread-safe. The internal map uses [`DashMap`] with
/// [AHash](https://docs.rs/ahash) for high-performance concurrent access.
///
/// # Examples
///
/// ## Basic Usage
///
/// ```rust
/// use prefix_tracker::{PrefixGroupsTracker, TrackerConfig};
///
/// let groups = PrefixGroupsTracker::new();
///
/// // Register trackers with custom configuration
/// groups.register(TrackerConfig::simple("user:").with_max_id(1_000_000));
/// groups.register(TrackerConfig::simple("session:").with_max_id(100_000));
///
/// // Get and use trackers
/// let users = groups.get("user:").unwrap();
/// users.add(42);
/// ```
///
/// ## Auto-Creation with Default Config
///
/// ```rust
/// use prefix_tracker::{PrefixGroupsTracker, TrackerConfig};
///
/// // Set a default config for auto-created trackers
/// let groups = PrefixGroupsTracker::with_default_config(
///     TrackerConfig::simple("").with_max_id(10_000)
/// );
///
/// // get_or_create auto-creates if not exists
/// let tracker = groups.get_or_create("new_prefix:");
/// tracker.add(1);
/// ```
///
/// ## Group Iteration
///
/// ```rust
/// use prefix_tracker::{PrefixGroupsTracker, TrackerConfig, ClaimPolicy};
///
/// let groups = PrefixGroupsTracker::new();
/// groups.register(TrackerConfig::simple("a:"));
/// groups.register(TrackerConfig::simple("b:"));
///
/// // Iterate across all prefixes
/// for item in groups.iter().set_only().horizontal().build() {
///     println!("{}:{}", item.prefix, item.id);
/// }
///
/// // Claim IDs with round-robin policy
/// let mut writer = groups.iter().write(ClaimPolicy::RoundRobin);
/// while let Some(item) = writer.next() {
///     // Each claimed ID is unique across all threads
/// }
/// ```
pub struct PrefixGroupsTracker {
    /// Map from prefix string to tracker.
    trackers: FastDashMap<String, Arc<PrefixTracker>>,

    /// Default configuration for auto-created trackers.
    default_config: TrackerConfig,
}

impl PrefixGroupsTracker {
    /// Create a new empty registry.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use prefix_tracker::PrefixGroupsTracker;
    ///
    /// let groups = PrefixGroupsTracker::new();
    /// assert_eq!(groups.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            trackers: FastDashMap::default(),
            default_config: TrackerConfig::default(),
        }
    }

    /// Create a new registry with custom default configuration.
    pub fn with_default_config(config: TrackerConfig) -> Self {
        Self {
            trackers: FastDashMap::default(),
            default_config: config,
        }
    }

    /// Register a new tracker with the given configuration.
    ///
    /// If a tracker with the same prefix already exists, returns the existing one.
    pub fn register(&self, config: TrackerConfig) -> Arc<PrefixTracker> {
        let prefix = config.prefix.clone();

        self.trackers
            .entry(prefix)
            .or_insert_with(|| Arc::new(PrefixTracker::new(config)))
            .clone()
    }

    /// Get an existing tracker by prefix.
    pub fn get(&self, prefix: &str) -> Option<Arc<PrefixTracker>> {
        self.trackers.get(prefix).map(|r| r.value().clone())
    }

    /// Get an existing tracker or create one with default configuration.
    pub fn get_or_create(&self, prefix: &str) -> Arc<PrefixTracker> {
        self.trackers
            .entry(prefix.to_string())
            .or_insert_with(|| {
                let mut config = self.default_config.clone();
                config.prefix = prefix.to_string();
                Arc::new(PrefixTracker::new(config))
            })
            .clone()
    }

    /// Remove a tracker by prefix.
    ///
    /// Returns the removed tracker if it existed.
    pub fn remove(&self, prefix: &str) -> Option<Arc<PrefixTracker>> {
        self.trackers.remove(prefix).map(|(_, v)| v)
    }

    /// Get list of all registered prefixes.
    pub fn prefixes(&self) -> Vec<String> {
        self.trackers.iter().map(|r| r.key().clone()).collect()
    }

    /// Get number of registered trackers.
    #[inline]
    pub fn len(&self) -> usize {
        self.trackers.len()
    }

    /// Check if registry is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.trackers.is_empty()
    }

    /// Get total count across all trackers.
    pub fn total_count(&self) -> u64 {
        self.trackers.iter().map(|r| r.value().count()).sum()
    }

    /// Clear all trackers (removes all registrations).
    pub fn clear(&self) {
        self.trackers.clear();
    }

    /// Create a group-level iterator builder.
    pub fn iter(&self) -> GroupIterBuilder<'_> {
        let trackers: Vec<Arc<PrefixTracker>> =
            self.trackers.iter().map(|r| r.value().clone()).collect();

        GroupIterBuilder::new(trackers)
    }

    /// Iterate over all registered (prefix, tracker) pairs.
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(&str, &Arc<PrefixTracker>),
    {
        for entry in self.trackers.iter() {
            f(entry.key(), entry.value());
        }
    }
}

impl Default for PrefixGroupsTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PrefixGroupsTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixGroupsTracker")
            .field("len", &self.len())
            .field("total_count", &self.total_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        let groups = PrefixGroupsTracker::new();

        let tracker = groups.register(TrackerConfig::simple("vec:"));
        assert_eq!(tracker.prefix(), "vec:");

        let retrieved = groups.get("vec:").unwrap();
        assert_eq!(retrieved.prefix(), "vec:");

        // Same instance
        tracker.add(42);
        assert!(retrieved.exists(42));
    }

    #[test]
    fn test_get_or_create() {
        let groups = PrefixGroupsTracker::new();

        // First call creates
        let t1 = groups.get_or_create("doc:");
        assert_eq!(t1.prefix(), "doc:");

        // Second call returns same instance
        let t2 = groups.get_or_create("doc:");
        t1.add(100);
        assert!(t2.exists(100));
    }

    #[test]
    fn test_remove() {
        let groups = PrefixGroupsTracker::new();

        groups.register(TrackerConfig::simple("vec:"));
        assert!(groups.get("vec:").is_some());

        let removed = groups.remove("vec:");
        assert!(removed.is_some());
        assert!(groups.get("vec:").is_none());
    }

    #[test]
    fn test_prefixes() {
        let groups = PrefixGroupsTracker::new();

        groups.register(TrackerConfig::simple("a:"));
        groups.register(TrackerConfig::simple("b:"));
        groups.register(TrackerConfig::simple("c:"));

        let mut prefixes = groups.prefixes();
        prefixes.sort();

        assert_eq!(prefixes, vec!["a:", "b:", "c:"]);
    }

    #[test]
    fn test_total_count() {
        let groups = PrefixGroupsTracker::new();

        let t1 = groups.register(TrackerConfig::simple("a:"));
        let t2 = groups.register(TrackerConfig::simple("b:"));

        t1.add(0);
        t1.add(1);
        t2.add(0);

        assert_eq!(groups.total_count(), 3);
    }

    #[test]
    fn test_clear() {
        let groups = PrefixGroupsTracker::new();

        groups.register(TrackerConfig::simple("a:"));
        groups.register(TrackerConfig::simple("b:"));
        assert_eq!(groups.len(), 2);

        groups.clear();
        assert!(groups.is_empty());
    }

    #[test]
    fn test_concurrent_registration() {
        use std::sync::Arc;
        use std::thread;

        let groups = Arc::new(PrefixGroupsTracker::new());

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let groups = groups.clone();
                thread::spawn(move || {
                    let prefix = format!("prefix{}:", i);
                    let tracker = groups.register(TrackerConfig::simple(&prefix));

                    for j in 0..100 {
                        tracker.add(j);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(groups.len(), 8);
        assert_eq!(groups.total_count(), 800);
    }

    #[test]
    fn test_group_iter_integration() {
        let groups = PrefixGroupsTracker::new();

        let t1 = groups.register(TrackerConfig::simple("vec:").with_max_id(10));
        let t2 = groups.register(TrackerConfig::simple("doc:").with_max_id(10));

        for i in 0..5 {
            t1.add(i);
            t2.add(i);
        }

        // Horizontal iteration
        let items: Vec<_> = groups.iter().set_only().horizontal().build().collect();
        assert_eq!(items.len(), 10);

        // Prefix filter
        let vec_items: Vec<_> = groups
            .iter()
            .prefixes(&["vec:"])
            .set_only()
            .horizontal()
            .build()
            .collect();
        assert_eq!(vec_items.len(), 5);
    }

    #[test]
    fn test_hierarchical_integration() {
        let groups = PrefixGroupsTracker::new();

        let tracker = groups.register(
            TrackerConfig::hierarchical("hash:")
                .with_max_id(100)
                .with_max_sub_id(100),
        );

        tracker.add_pair(1, 0);
        tracker.add_pair(1, 1);
        tracker.add_pair(2, 0);

        assert_eq!(tracker.count(), 3);
        assert_eq!(tracker.primary_count(), 2);
        assert_eq!(tracker.sub_count(1), 2);

        // Iterate
        let items: Vec<_> = tracker.iter().set_only().sequential().collect();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], (1, Some(0)));
        assert_eq!(items[1], (1, Some(1)));
        assert_eq!(items[2], (2, Some(0)));
    }

    #[test]
    fn test_write_iter_integration() {
        let groups = PrefixGroupsTracker::new();

        groups.register(TrackerConfig::simple("a:").with_max_id(50));
        groups.register(TrackerConfig::simple("b:").with_max_id(50));

        let mut iter = groups.iter().write(ClaimPolicy::RoundRobin);

        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
            if count >= 20 {
                break;
            }
        }

        assert_eq!(count, 20);
        assert_eq!(groups.total_count(), 20);
    }

    #[test]
    fn test_delete_iter_integration() {
        let groups = PrefixGroupsTracker::new();

        let t1 = groups.register(TrackerConfig::simple("a:").with_max_id(10));
        let t2 = groups.register(TrackerConfig::simple("b:").with_max_id(10));

        for i in 0..5 {
            t1.add(i);
            t2.add(i);
        }

        assert_eq!(groups.total_count(), 10);

        if let Some(mut iter) = groups.iter().delete(ClaimPolicy::RoundRobin) {
            while iter.next().is_some() {}
        }

        assert_eq!(groups.total_count(), 0);
    }
}
