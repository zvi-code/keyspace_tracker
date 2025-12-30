//! Prefix Tracker implementation.
//!
//! Provides `PrefixTracker` which wraps `AtomicBitmap` with prefix metadata
//! and supports both simple (single ID) and hierarchical (id, sub_id) tracking.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use ahash::RandomState as AHasher;
use dashmap::DashMap;

use super::bitmap::AtomicBitmap;
use super::config::TrackerConfig;
use super::iterators::TrackerIterBuilder;

/// Type alias for DashMap with AHash (2-10x faster than default SipHash)
type FastDashMap<K, V> = DashMap<K, V, AHasher>;

/// Tracks existence of IDs for a specific prefix.
///
/// `PrefixTracker` provides efficient, thread-safe tracking of which IDs exist
/// for a given key prefix. It supports two modes:
///
/// - **Simple mode**: Tracks individual IDs (u64)
/// - **Hierarchical mode**: Tracks (id, sub_id) pairs
///
/// # Thread Safety
///
/// All operations are thread-safe:
/// - Reads use atomic loads
/// - Writes use atomic compare-and-swap (CAS)
/// - Growth operations are synchronized via mutex
/// - Multiple write iterators can run concurrently
/// - Only one delete iterator can be active at a time
///
/// # Performance
///
/// | Operation | Simple Mode | Hierarchical Mode |
/// |-----------|------------|-------------------|
/// | `exists` | ~1.6 ns | ~17 ns |
/// | `add` | ~7 ns | ~39 ns |
/// | `claim` | ~2.4 ns | ~20 ns |
///
/// # Examples
///
/// ## Simple Tracker
///
/// ```rust
/// use prefix_tracker::PrefixTracker;
///
/// let tracker = PrefixTracker::simple("vec:");
///
/// // Add IDs
/// tracker.add(0);
/// tracker.add(100);
///
/// // Check existence
/// assert!(tracker.exists(100));
/// assert!(!tracker.exists(50));
///
/// // Atomic claim (test-and-set)
/// assert!(tracker.claim(200));  // Returns true, ID now set
/// assert!(!tracker.claim(200)); // Returns false, already set
///
/// // Remove
/// tracker.remove(100);
/// assert!(!tracker.exists(100));
/// ```
///
/// ## Hierarchical Tracker
///
/// ```rust
/// use prefix_tracker::PrefixTracker;
///
/// // For hash fields: hash:id -> { field0, field1, ... }
/// let tracker = PrefixTracker::hierarchical("hash:");
///
/// tracker.add_pair(1, 0);   // hash:1 has field 0
/// tracker.add_pair(1, 5);   // hash:1 has field 5
/// tracker.add_pair(2, 10);  // hash:2 has field 10
///
/// assert!(tracker.exists_pair(1, 5));
/// assert!(!tracker.exists_pair(1, 99));
///
/// // Count total (id, sub_id) pairs
/// assert_eq!(tracker.count(), 3);
/// ```
///
/// ## Concurrent Claiming
///
/// ```rust
/// use prefix_tracker::PrefixTracker;
/// use std::sync::Arc;
/// use std::thread;
///
/// let tracker = Arc::new(PrefixTracker::simple("id:"));
///
/// let handles: Vec<_> = (0..4).map(|_| {
///     let t = tracker.clone();
///     thread::spawn(move || {
///         (0..100u64).filter(|&id| t.claim(id)).count()
///     })
/// }).collect();
///
/// let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
/// assert_eq!(total, 100); // Each ID claimed exactly once
/// ```
pub struct PrefixTracker {
    /// Configuration.
    config: TrackerConfig,

    /// For simple trackers: bitmap for all IDs.
    /// For hierarchical: bitmap tracking which primary IDs have any sub_ids.
    primary_bitmap: AtomicBitmap,

    /// For hierarchical trackers: sub-bitmaps per primary ID.
    /// None for simple trackers.
    sub_bitmaps: Option<FastDashMap<u64, AtomicBitmap>>,

    /// Total count for hierarchical trackers (sum of all sub-bitmap counts).
    /// For simple trackers, use primary_bitmap.count() directly.
    hierarchical_count: AtomicU64,

    /// Atomic flag for exclusive delete iterator.
    delete_iter_active: AtomicBool,

    /// Shared cursor for write iterators (word index).
    write_cursor: AtomicU64,
}

impl PrefixTracker {
    /// Create a new tracker with the given configuration.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use prefix_tracker::{PrefixTracker, TrackerConfig};
    ///
    /// let config = TrackerConfig::simple("myprefix:")
    ///     .with_max_id(1_000_000)
    ///     .with_initial_capacity(4096);
    ///
    /// let tracker = PrefixTracker::new(config);
    /// ```
    pub fn new(config: TrackerConfig) -> Self {
        let sub_bitmaps = if config.hierarchical {
            Some(FastDashMap::default())
        } else {
            None
        };

        Self {
            primary_bitmap: AtomicBitmap::with_capacity(config.initial_capacity),
            sub_bitmaps,
            hierarchical_count: AtomicU64::new(0),
            delete_iter_active: AtomicBool::new(false),
            write_cursor: AtomicU64::new(0),
            config,
        }
    }

    /// Create a simple (non-hierarchical) tracker.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use prefix_tracker::PrefixTracker;
    ///
    /// let tracker = PrefixTracker::simple("vec:");
    /// tracker.add(42);
    /// assert!(tracker.exists(42));
    /// ```
    #[inline]
    pub fn simple(prefix: impl Into<String>) -> Self {
        Self::new(TrackerConfig::simple(prefix))
    }

    /// Create a hierarchical tracker for (id, sub_id) pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use prefix_tracker::PrefixTracker;
    ///
    /// let tracker = PrefixTracker::hierarchical("hash:");
    /// tracker.add_pair(1, 5);
    /// assert!(tracker.exists_pair(1, 5));
    /// ```
    #[inline]
    pub fn hierarchical(prefix: impl Into<String>) -> Self {
        Self::new(TrackerConfig::hierarchical(prefix))
    }

    /// Get the prefix string.
    #[inline]
    pub fn prefix(&self) -> &str {
        &self.config.prefix
    }

    /// Check if this is a hierarchical tracker.
    #[inline]
    pub fn is_hierarchical(&self) -> bool {
        self.config.hierarchical
    }

    /// Get maximum primary ID if set.
    #[inline]
    pub fn max_id(&self) -> Option<u64> {
        self.config.max_id
    }

    /// Get maximum sub_id if set (hierarchical only).
    #[inline]
    pub fn max_sub_id(&self) -> Option<u64> {
        self.config.max_sub_id
    }

    /// Get the tracker configuration.
    #[inline]
    pub fn config(&self) -> &TrackerConfig {
        &self.config
    }

    // ========================================================================
    // Simple tracker operations
    // ========================================================================

    /// Add an ID to the tracker (mark as existing).
    ///
    /// Returns true if newly added, false if already existed.
    ///
    /// # Panics
    /// Panics if this is a hierarchical tracker.
    #[inline]
    pub fn add(&self, id: u64) -> bool {
        debug_assert!(
            !self.is_hierarchical(),
            "use add_pair() for hierarchical trackers"
        );

        if let Some(max) = self.config.max_id {
            if id >= max {
                return false;
            }
        }

        !self.primary_bitmap.set(id as usize)
    }

    /// Remove an ID from the tracker.
    ///
    /// Returns true if was present, false if was not present.
    ///
    /// # Panics
    /// Panics if this is a hierarchical tracker.
    #[inline]
    pub fn remove(&self, id: u64) -> bool {
        debug_assert!(
            !self.is_hierarchical(),
            "use remove_pair() for hierarchical trackers"
        );

        self.primary_bitmap.clear(id as usize)
    }

    /// Check if an ID exists.
    ///
    /// # Panics
    /// Panics if this is a hierarchical tracker.
    #[inline]
    pub fn exists(&self, id: u64) -> bool {
        debug_assert!(
            !self.is_hierarchical(),
            "use exists_pair() for hierarchical trackers"
        );

        self.primary_bitmap.test(id as usize)
    }

    /// Atomically claim an ID: mark as existing only if not already.
    ///
    /// Returns true if successfully claimed (was not present).
    ///
    /// # Panics
    /// Panics if this is a hierarchical tracker.
    #[inline]
    pub fn claim(&self, id: u64) -> bool {
        debug_assert!(
            !self.is_hierarchical(),
            "use claim_pair() for hierarchical trackers"
        );

        if let Some(max) = self.config.max_id {
            if id >= max {
                return false;
            }
        }

        self.primary_bitmap.test_and_set(id as usize)
    }

    // ========================================================================
    // Hierarchical tracker operations
    // ========================================================================

    /// Get or create sub-bitmap for a primary ID.
    fn get_or_create_sub_bitmap(
        &self,
        id: u64,
    ) -> dashmap::mapref::one::RefMut<'_, u64, AtomicBitmap> {
        let sub_bitmaps = self.sub_bitmaps.as_ref().expect("hierarchical tracker");

        sub_bitmaps.entry(id).or_insert_with(|| {
            let capacity = self.config.max_sub_id.unwrap_or(4096) as usize;
            AtomicBitmap::with_capacity(capacity.min(4096))
        })
    }

    /// Get sub-bitmap for a primary ID if it exists.
    fn get_sub_bitmap(&self, id: u64) -> Option<dashmap::mapref::one::Ref<'_, u64, AtomicBitmap>> {
        self.sub_bitmaps.as_ref()?.get(&id)
    }

    /// Add a (id, sub_id) pair to the tracker.
    ///
    /// Returns true if newly added, false if already existed.
    ///
    /// # Panics
    /// Panics if this is a simple (non-hierarchical) tracker.
    #[inline]
    pub fn add_pair(&self, id: u64, sub_id: u64) -> bool {
        debug_assert!(self.is_hierarchical(), "use add() for simple trackers");

        if let Some(max) = self.config.max_id {
            if id >= max {
                return false;
            }
        }
        if let Some(max) = self.config.max_sub_id {
            if sub_id >= max {
                return false;
            }
        }

        let sub_bitmap = self.get_or_create_sub_bitmap(id);
        let was_unset = !sub_bitmap.set(sub_id as usize);

        if was_unset {
            self.hierarchical_count.fetch_add(1, Ordering::Relaxed);

            // Mark primary ID as having sub_ids
            self.primary_bitmap.set(id as usize);
        }

        was_unset
    }

    /// Remove a (id, sub_id) pair from the tracker.
    ///
    /// Returns true if was present, false if was not present.
    ///
    /// # Panics
    /// Panics if this is a simple (non-hierarchical) tracker.
    #[inline]
    pub fn remove_pair(&self, id: u64, sub_id: u64) -> bool {
        debug_assert!(self.is_hierarchical(), "use remove() for simple trackers");

        let sub_bitmaps = self.sub_bitmaps.as_ref().expect("hierarchical tracker");

        if let Some(sub_bitmap) = sub_bitmaps.get(&id) {
            let was_set = sub_bitmap.clear(sub_id as usize);
            if was_set {
                self.hierarchical_count.fetch_sub(1, Ordering::Relaxed);

                // If sub-bitmap is now empty, clear primary bit
                if sub_bitmap.is_empty() {
                    self.primary_bitmap.clear(id as usize);
                }
            }
            was_set
        } else {
            false
        }
    }

    /// Check if a (id, sub_id) pair exists.
    ///
    /// # Panics
    /// Panics if this is a simple (non-hierarchical) tracker.
    #[inline]
    pub fn exists_pair(&self, id: u64, sub_id: u64) -> bool {
        debug_assert!(self.is_hierarchical(), "use exists() for simple trackers");

        if let Some(sub_bitmap) = self.get_sub_bitmap(id) {
            sub_bitmap.test(sub_id as usize)
        } else {
            false
        }
    }

    /// Atomically claim a (id, sub_id) pair.
    ///
    /// Returns true if successfully claimed (was not present).
    ///
    /// # Panics
    /// Panics if this is a simple (non-hierarchical) tracker.
    #[inline]
    pub fn claim_pair(&self, id: u64, sub_id: u64) -> bool {
        debug_assert!(self.is_hierarchical(), "use claim() for simple trackers");

        if let Some(max) = self.config.max_id {
            if id >= max {
                return false;
            }
        }
        if let Some(max) = self.config.max_sub_id {
            if sub_id >= max {
                return false;
            }
        }

        let sub_bitmap = self.get_or_create_sub_bitmap(id);
        let claimed = sub_bitmap.test_and_set(sub_id as usize);

        if claimed {
            self.hierarchical_count.fetch_add(1, Ordering::Relaxed);
            self.primary_bitmap.set(id as usize);
        }

        claimed
    }

    /// Check if a primary ID has any sub_ids (hierarchical only).
    #[inline]
    pub fn id_exists(&self, id: u64) -> bool {
        self.primary_bitmap.test(id as usize)
    }

    // ========================================================================
    // Count and capacity
    // ========================================================================

    /// Get total count of set bits.
    ///
    /// For simple trackers: number of IDs.
    /// For hierarchical: total number of (id, sub_id) pairs.
    #[inline]
    pub fn count(&self) -> u64 {
        if self.is_hierarchical() {
            self.hierarchical_count.load(Ordering::Relaxed)
        } else {
            self.primary_bitmap.count()
        }
    }

    /// Get number of primary IDs with any sub_ids (hierarchical only).
    ///
    /// For simple trackers, this is the same as count().
    #[inline]
    pub fn primary_count(&self) -> u64 {
        self.primary_bitmap.count()
    }

    /// Get count of sub_ids for a given primary ID (hierarchical only).
    ///
    /// Returns 0 for simple trackers or if ID doesn't exist.
    #[inline]
    pub fn sub_count(&self, id: u64) -> u64 {
        if let Some(sub_bitmap) = self.get_sub_bitmap(id) {
            sub_bitmap.count()
        } else {
            0
        }
    }

    /// Get the effective max sub_id for a given primary ID.
    ///
    /// Returns the minimum of:
    /// - config.max_sub_id (if set)
    /// - sub_bitmap capacity for this id (if exists)
    /// - 0 if not hierarchical or no sub_bitmap exists
    #[inline]
    pub(crate) fn effective_max_sub_id_for(&self, id: u64) -> u64 {
        if !self.config.hierarchical {
            return 0;
        }
        
        let config_max = self.config.max_sub_id.unwrap_or(u64::MAX);
        
        if let Some(sub_bitmap) = self.get_sub_bitmap(id) {
            config_max.min(sub_bitmap.capacity() as u64)
        } else {
            // No sub_bitmap exists for this id, so no sub_ids to iterate
            0
        }
    }

    /// Get current capacity (primary bitmap).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.primary_bitmap.capacity()
    }

    /// Check if tracker is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Ensure capacity for the given ID.
    #[inline]
    pub fn ensure_capacity(&self, id: u64) {
        self.primary_bitmap.ensure_capacity(id as usize);
    }

    /// Clear all tracked IDs.
    ///
    /// This requires exclusive access (&mut self).
    pub fn clear(&mut self) {
        self.primary_bitmap.clear_all();

        if let Some(ref mut sub_bitmaps) = self.sub_bitmaps {
            sub_bitmaps.clear();
        }

        *self.hierarchical_count.get_mut() = 0;
        *self.write_cursor.get_mut() = 0;
    }

    // ========================================================================
    // Iterator support
    // ========================================================================

    /// Create an iterator builder for this tracker.
    pub fn iter(&self) -> TrackerIterBuilder<'_> {
        TrackerIterBuilder::new(self)
    }

    /// Try to acquire exclusive delete iterator access.
    ///
    /// Returns true if acquired, false if another delete iterator is active.
    pub(crate) fn try_acquire_delete_lock(&self) -> bool {
        self.delete_iter_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }

    /// Release exclusive delete iterator access.
    pub(crate) fn release_delete_lock(&self) {
        self.delete_iter_active.store(false, Ordering::Release);
    }

    /// Get reference to primary bitmap for iteration.
    pub(crate) fn primary_bitmap(&self) -> &AtomicBitmap {
        &self.primary_bitmap
    }

    /// Get reference to sub-bitmaps for iteration.
    pub(crate) fn sub_bitmaps(&self) -> Option<&FastDashMap<u64, AtomicBitmap>> {
        self.sub_bitmaps.as_ref()
    }

    /// Get write cursor reference.
    pub(crate) fn write_cursor(&self) -> &AtomicU64 {
        &self.write_cursor
    }

    /// Reset write cursor (call before starting iteration).
    pub(crate) fn reset_write_cursor(&self) {
        self.write_cursor.store(0, Ordering::Release);
    }

    /// Get effective max ID for iteration.
    pub(crate) fn effective_max_id(&self) -> u64 {
        self.config
            .max_id
            .unwrap_or(self.primary_bitmap.capacity() as u64)
    }

    /// Get effective max sub_id for iteration.
    pub(crate) fn effective_max_sub_id(&self) -> u64 {
        self.config.max_sub_id.unwrap_or(u64::MAX)
    }
}

impl std::fmt::Debug for PrefixTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixTracker")
            .field("prefix", &self.config.prefix)
            .field("hierarchical", &self.config.hierarchical)
            .field("count", &self.count())
            .field("capacity", &self.capacity())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tracker() {
        let tracker = PrefixTracker::simple("vec:");

        assert_eq!(tracker.prefix(), "vec:");
        assert!(!tracker.is_hierarchical());
        assert!(tracker.is_empty());

        assert!(tracker.add(0));
        assert!(!tracker.add(0)); // Already exists
        assert!(tracker.exists(0));
        assert_eq!(tracker.count(), 1);

        assert!(tracker.add(100));
        assert_eq!(tracker.count(), 2);

        assert!(tracker.remove(0));
        assert!(!tracker.exists(0));
        assert_eq!(tracker.count(), 1);
    }

    #[test]
    fn test_simple_claim() {
        let tracker = PrefixTracker::simple("vec:");

        assert!(tracker.claim(42));
        assert!(!tracker.claim(42)); // Already claimed
        assert!(tracker.exists(42));
    }

    #[test]
    fn test_simple_with_max_id() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));

        assert!(tracker.add(50));
        assert!(!tracker.add(100)); // At max, should fail
        assert!(!tracker.add(200)); // Above max, should fail
        assert_eq!(tracker.count(), 1);
    }

    #[test]
    fn test_hierarchical_tracker() {
        let tracker = PrefixTracker::hierarchical("hash:");

        assert!(tracker.is_hierarchical());
        assert!(tracker.is_empty());

        assert!(tracker.add_pair(1, 0));
        assert!(tracker.add_pair(1, 42));
        assert!(!tracker.add_pair(1, 42)); // Already exists

        assert!(tracker.exists_pair(1, 0));
        assert!(tracker.exists_pair(1, 42));
        assert!(!tracker.exists_pair(1, 100));
        assert!(!tracker.exists_pair(2, 0));

        assert_eq!(tracker.count(), 2);
        assert_eq!(tracker.primary_count(), 1);
        assert_eq!(tracker.sub_count(1), 2);

        assert!(tracker.add_pair(2, 0));
        assert_eq!(tracker.count(), 3);
        assert_eq!(tracker.primary_count(), 2);
    }

    #[test]
    fn test_hierarchical_remove() {
        let tracker = PrefixTracker::hierarchical("hash:");

        tracker.add_pair(1, 0);
        tracker.add_pair(1, 1);

        assert!(tracker.id_exists(1));

        assert!(tracker.remove_pair(1, 0));
        assert!(!tracker.exists_pair(1, 0));
        assert!(tracker.id_exists(1)); // Still has sub_id 1

        assert!(tracker.remove_pair(1, 1));
        assert!(!tracker.id_exists(1)); // No more sub_ids
    }

    #[test]
    fn test_hierarchical_claim() {
        let tracker = PrefixTracker::hierarchical("hash:");

        assert!(tracker.claim_pair(1, 42));
        assert!(!tracker.claim_pair(1, 42)); // Already claimed
        assert!(tracker.exists_pair(1, 42));
    }

    #[test]
    fn test_clear() {
        let mut tracker = PrefixTracker::simple("vec:");

        tracker.add(0);
        tracker.add(100);
        assert_eq!(tracker.count(), 2);

        tracker.clear();
        assert_eq!(tracker.count(), 0);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_hierarchical_clear() {
        let mut tracker = PrefixTracker::hierarchical("hash:");

        tracker.add_pair(1, 0);
        tracker.add_pair(1, 1);
        tracker.add_pair(2, 0);
        assert_eq!(tracker.count(), 3);

        tracker.clear();
        assert_eq!(tracker.count(), 0);
        assert_eq!(tracker.primary_count(), 0);
    }

    #[test]
    fn test_concurrent_simple() {
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(PrefixTracker::simple("vec:"));

        let handles: Vec<_> = (0..8)
            .map(|t| {
                let tracker = tracker.clone();
                thread::spawn(move || {
                    for i in 0..1000 {
                        tracker.add(t * 1000 + i);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(tracker.count(), 8000);
    }

    #[test]
    fn test_concurrent_claim() {
        use std::sync::atomic::AtomicU64;
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(PrefixTracker::simple("vec:"));
        let success_count = Arc::new(AtomicU64::new(0));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let tracker = tracker.clone();
                let success = success_count.clone();
                thread::spawn(move || {
                    for i in 0..1000 {
                        if tracker.claim(i) {
                            success.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Exactly 1000 successful claims
        assert_eq!(success_count.load(Ordering::Relaxed), 1000);
        assert_eq!(tracker.count(), 1000);
    }
}
