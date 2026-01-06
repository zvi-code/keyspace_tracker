//! Single-tracker iterators.
//!
//! Provides various iterator types for iterating over a `PrefixTracker`:
//! - `SequentialIter`: Ascending order iteration
//! - `RandomIter`: Pseudo-random order iteration
//! - `WriteIter`: Concurrent write with atomic claim-and-set
//! - `DeleteIter`: Exclusive delete iteration
//! - `PartitionedIter`: Parallel iteration with disjoint ranges
//! - `SamplingIter`: Wrapper for mixed-ratio and probabilistic sampling

use std::sync::atomic::Ordering;

use super::config::{BitFilter, FilterContext, IdRange, SamplingConfig};
use super::tracker::PrefixTracker;

/// Item yielded by tracker iterators.
///
/// For simple trackers: `(id, None)`
/// For hierarchical trackers: `(id, Some(sub_id))`
pub type TrackerItem = (u64, Option<u64>);

/// Builder for configuring single-tracker iterators.
pub struct TrackerIterBuilder<'a> {
    tracker: &'a PrefixTracker,
    range: IdRange,
    filter: BitFilter,
    sampling: SamplingConfig,
}

impl<'a> TrackerIterBuilder<'a> {
    /// Create a new iterator builder.
    pub(crate) fn new(tracker: &'a PrefixTracker) -> Self {
        Self {
            tracker,
            range: IdRange::unbounded(),
            filter: BitFilter::All,
            sampling: SamplingConfig::new(),
        }
    }

    /// Set minimum and maximum primary ID range.
    #[inline]
    pub fn id_range(mut self, min: u64, max: u64) -> Self {
        self.range.id_min = min;
        self.range.id_max = max;
        self
    }

    /// Set minimum and maximum sub_id range (hierarchical only).
    #[inline]
    pub fn sub_id_range(mut self, min: u64, max: u64) -> Self {
        self.range.sub_id_min = min;
        self.range.sub_id_max = max;
        self
    }

    /// Set both ID and sub_id ranges.
    #[inline]
    pub fn range(mut self, id_range: (u64, u64), sub_id_range: (u64, u64)) -> Self {
        self.range = IdRange::new(id_range, sub_id_range);
        self
    }

    /// Filter to only iterate over set (existing) IDs.
    #[inline]
    pub fn set_only(mut self) -> Self {
        self.filter = BitFilter::Set;
        self
    }

    /// Filter to only iterate over unset (non-existing) IDs.
    #[inline]
    pub fn unset_only(mut self) -> Self {
        self.filter = BitFilter::Unset;
        self
    }

    /// Set the bit filter explicitly.
    #[inline]
    pub fn filter(mut self, filter: BitFilter) -> Self {
        self.filter = filter;
        self
    }

    // ========================================================================
    // Sampling Configuration
    // ========================================================================

    /// Set mixed ratio: proportion of set bits vs unset bits to return.
    ///
    /// For example, `mixed_ratio(0.9)` returns 90% existing (set) and 10% new (unset).
    /// This is useful for "90% overwrite + 10% new write" benchmark patterns.
    ///
    /// Note: This overrides `set_only()` / `unset_only()` filters.
    #[inline]
    pub fn mixed_ratio(mut self, set_ratio: f64) -> Self {
        self.sampling = self.sampling.with_set_ratio(set_ratio.clamp(0.0, 1.0));
        self
    }

    /// Limit the number of items returned.
    #[inline]
    pub fn limit(mut self, n: u64) -> Self {
        self.sampling = self.sampling.with_limit(n);
        self
    }

    /// Set probabilistic sampling: each item has `probability` chance of being returned.
    ///
    /// For example, `sample(0.5)` returns approximately 50% of matching items.
    #[inline]
    pub fn sample(mut self, probability: f64) -> Self {
        self.sampling = self.sampling.with_sample_probability(probability.clamp(0.0, 1.0));
        self
    }

    /// Set random seed for reproducible iteration.
    #[inline]
    pub fn seed(mut self, seed: u64) -> Self {
        self.sampling = self.sampling.with_seed(seed);
        self
    }

    /// Set full sampling config.
    #[inline]
    pub fn with_sampling(mut self, config: SamplingConfig) -> Self {
        self.sampling = config;
        self
    }

    /// Set access distribution for random iteration.
    ///
    /// Use Zipfian for cache workloads, Exponential for session stores, etc.
    #[inline]
    pub fn distribution(mut self, dist: super::config::AccessDistribution) -> Self {
        self.sampling.distribution = dist;
        self
    }

    /// Enable overlapping iteration (contention testing).
    ///
    /// Multiple threads may visit the same keys, useful for testing
    /// concurrent access patterns and lock contention.
    #[inline]
    pub fn overlapping(mut self) -> Self {
        self.sampling.overlapping = true;
        self
    }

    /// Set ID range using percentage of the keyspace.
    ///
    /// For example, `range_percent(0.5, 1.0)` iterates the upper 50% of keyspace.
    #[inline]
    pub fn range_percent(mut self, start_pct: f64, end_pct: f64) -> Self {
        let max = self.tracker.effective_max_id();
        let start = (max as f64 * start_pct.clamp(0.0, 1.0)) as u64;
        let end = (max as f64 * end_pct.clamp(0.0, 1.0)) as u64;
        self.range.id_min = start;
        self.range.id_max = end;
        self
    }

    /// Build a sequential (ascending order) iterator.
    pub fn sequential(self) -> SequentialIter<'a> {
        let max_id = self.range.id_max.min(self.tracker.effective_max_id());

        SequentialIter {
            tracker: self.tracker,
            range: self.range,
            ctx: FilterContext::new(self.filter, self.sampling),
            current_id: self.range.id_min,
            current_sub_id: self.range.sub_id_min,
            max_id,
            wrap_around: false,
        }
    }

    /// Build a random-order iterator.
    ///
    /// Uses multiplicative hashing for memory-efficient pseudo-random traversal.
    /// If sampling is configured, returns a `SamplingIter<RandomIter>` instead.
    pub fn random(self) -> RandomIter<'a> {
        let max_id = self.range.id_max.min(self.tracker.effective_max_id());
        let range_size = max_id.saturating_sub(self.range.id_min);

        // Choose a multiplier coprime with range_size for full coverage
        let multiplier = Self::find_coprime(range_size);

        // Use seeded RNG if configured, otherwise random
        let mut rng = self.sampling.make_rng();
        let offset = rng.u64(..) % range_size.max(1);

        RandomIter {
            tracker: self.tracker,
            range: self.range,
            ctx: FilterContext::new(self.filter, self.sampling),
            range_size,
            multiplier,
            offset,
            current_step: 0,
            max_steps: range_size,
        }
    }

    /// Find a number coprime with n for multiplicative permutation.
    ///
    /// Uses known good LCG multipliers that are coprime with powers of 2.
    /// Falls back to binary GCD search if needed.
    #[inline]
    fn find_coprime(n: u64) -> u64 {
        if n <= 1 {
            return 1;
        }

        // Large odd primes - always coprime with powers of 2
        // These are specifically chosen LCG multipliers with good statistical properties
        const CANDIDATES: [u64; 5] = [
            6364136223846793005, // Knuth MMIX LCG
            2862933555777941757, // Steele, Vigna SplitMix
            0x5851F42D4C957F2D,  // 64-bit LCG from PCG
            0x2545F4914F6CDD1D,  // Another good LCG multiplier
            0x14057B7EF767814F,  // Mersenne-related
        ];

        // Fast path: if n is a power of 2, any odd number is coprime
        if n.is_power_of_two() {
            return CANDIDATES[0];
        }

        // Check candidates using binary GCD
        for &c in &CANDIDATES {
            if Self::binary_gcd(c, n) == 1 {
                return c;
            }
        }

        // Fallback: find smallest coprime > 1
        // Start with 3 (2 might share factor with n), check odd numbers only
        let mut c = 3u64;
        while c < n {
            if Self::binary_gcd(c, n) == 1 {
                return c;
            }
            c += 2;
        }

        1
    }

    /// Binary GCD algorithm (Stein's algorithm).
    /// Faster than Euclidean GCD on modern CPUs - uses only subtraction and bit shifts.
    #[inline]
    fn binary_gcd(mut a: u64, mut b: u64) -> u64 {
        if a == 0 {
            return b;
        }
        if b == 0 {
            return a;
        }

        // Find common factors of 2
        let shift = (a | b).trailing_zeros();

        // Remove all factors of 2 from a
        a >>= a.trailing_zeros();

        loop {
            // Remove all factors of 2 from b
            b >>= b.trailing_zeros();

            // Ensure a <= b
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }

            b -= a;

            if b == 0 {
                return a << shift;
            }
        }
    }

    /// Build a write iterator with atomic claim-and-set semantics.
    ///
    /// Multiple `WriteIter`s can run concurrently. Each call to `next()`
    /// atomically claims a unique ID.
    ///
    /// **Note:** This resets the write cursor to 0 before iteration begins.
    /// For concurrent scenarios where multiple iterators should share cursor
    /// position, use [`continue_write()`](Self::continue_write) instead.
    pub fn write(self) -> WriteIter<'a> {
        // Reset cursor at start of iteration
        self.tracker.reset_write_cursor();

        let max_id = self.range.id_max.min(self.tracker.effective_max_id());

        WriteIter {
            tracker: self.tracker,
            range: self.range,
            filter: self.filter,
            max_id,
            wrap_around: false,
        }
    }

    /// Build a write iterator that continues from the current cursor position.
    ///
    /// Unlike [`write()`](Self::write), this does **not** reset the write cursor.
    /// Use this when multiple workers need to share cursor state across
    /// independently-created iterators.
    ///
    /// # Example: Concurrent workers
    ///
    /// ```
    /// use keyspace_tracker::PrefixTracker;
    ///
    /// let tracker = PrefixTracker::new(100);
    /// tracker.reset_write_cursor(); // Reset once at the start
    ///
    /// // Multiple workers can create iterators without resetting cursor
    /// let mut iter1 = tracker.iter().continue_write();
    /// let mut iter2 = tracker.iter().continue_write();
    ///
    /// // Each claim_next_id() atomically advances the shared cursor
    /// let id1 = iter1.claim_next_id();
    /// let id2 = iter2.claim_next_id();
    /// assert_ne!(id1, id2); // Different IDs guaranteed
    /// ```
    pub fn continue_write(self) -> WriteIter<'a> {
        let max_id = self.range.id_max.min(self.tracker.effective_max_id());

        WriteIter {
            tracker: self.tracker,
            range: self.range,
            filter: self.filter,
            max_id,
            wrap_around: false,
        }
    }

    /// Build an exclusive delete iterator.
    ///
    /// Only one `DeleteIter` can exist per tracker at a time.
    /// Returns `None` if another delete iterator is active.
    pub fn delete(self) -> Option<DeleteIter<'a>> {
        if !self.tracker.try_acquire_delete_lock() {
            return None;
        }

        let max_id = self.range.id_max.min(self.tracker.effective_max_id());

        Some(DeleteIter {
            tracker: self.tracker,
            range: self.range,
            current_id: self.range.id_min,
            current_sub_id: self.range.sub_id_min,
            max_id,
        })
    }

    /// Build partitioned iterators for parallel processing.
    ///
    /// Returns `n` iterators with disjoint ranges covering the full space.
    /// Each partition gets its own RNG seeded from the base seed + partition index.
    pub fn partitioned(self, n: usize) -> Vec<PartitionedIter<'a>> {
        let max_id = self.range.id_max.min(self.tracker.effective_max_id());
        let total = max_id.saturating_sub(self.range.id_min);
        
        // In overlapping mode, each partition iterates the full range
        let overlapping = self.sampling.overlapping;
        let chunk_size = if overlapping {
            total // Each partition gets full range
        } else {
            (total + n as u64 - 1) / n as u64
        };

        // Per-partition limit if overall limit is set
        let per_partition_limit = if overlapping {
            self.sampling.limit // Each partition gets full limit in overlapping mode
        } else {
            self.sampling.limit.map(|l| (l + n as u64 - 1) / n as u64)
        };

        (0..n)
            .map(|i| {
                let (start, end) = if overlapping {
                    // All partitions get the full range
                    (self.range.id_min, max_id)
                } else {
                    // Non-overlapping: divide range among partitions
                    let start = self.range.id_min + i as u64 * chunk_size;
                    let end = (start + chunk_size).min(max_id);
                    (start, end)
                };

                // Create per-partition sampling config with adjusted limit and seed
                let mut partition_sampling = self.sampling;
                partition_sampling.limit = per_partition_limit;
                // Seed each partition differently for better randomness
                if let Some(seed) = self.sampling.seed {
                    partition_sampling.seed = Some(seed.wrapping_add(i as u64));
                }

                let range = IdRange {
                    id_min: start,
                    id_max: end,
                    sub_id_min: self.range.sub_id_min,
                    sub_id_max: self.range.sub_id_max,
                };

                PartitionedIter {
                    tracker: self.tracker,
                    range,
                    ctx: FilterContext::new(self.filter, partition_sampling),
                    current_id: start,
                    current_sub_id: self.range.sub_id_min,
                }
            })
            .collect()
    }

    /// Build a single partition iterator for independent parallel processing.
    ///
    /// Unlike `partitioned(n)` which returns all partitions, this method creates
    /// a single partition iterator given the partition index and total count.
    /// This enables each worker thread to independently create its own iterator
    /// without coordination or access to other partitions.
    ///
    /// # Arguments
    ///
    /// * `index` - The partition index (0-based, must be < `total`)
    /// * `total` - Total number of partitions
    ///
    /// # Panics
    ///
    /// Panics if `index >= total` or `total == 0`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use keyspace_tracker::PrefixTracker;
    /// use std::thread;
    /// use std::sync::Arc;
    ///
    /// let tracker = Arc::new(PrefixTracker::simple("data:"));
    /// for i in 0..1000u64 {
    ///     tracker.add(i);
    /// }
    ///
    /// let num_workers = 4;
    /// let handles: Vec<_> = (0..num_workers)
    ///     .map(|worker_id| {
    ///         let t = tracker.clone();
    ///         thread::spawn(move || {
    ///             // Each worker independently creates its own partition
    ///             let mut count = 0u64;
    ///             for (id, _) in t.iter()
    ///                 .set_only()
    ///                 .partition(worker_id, num_workers)
    ///             {
    ///                 count += 1;
    ///             }
    ///             count
    ///         })
    ///     })
    ///     .collect();
    ///
    /// let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    /// assert_eq!(total, 1000);
    /// ```
    pub fn partition(self, index: usize, total: usize) -> PartitionedIter<'a> {
        assert!(total > 0, "total partitions must be > 0");
        assert!(index < total, "partition index {} must be < total {}", index, total);

        let max_id = self.range.id_max.min(self.tracker.effective_max_id());
        let range_size = max_id.saturating_sub(self.range.id_min);

        // In overlapping mode, each partition iterates the full range
        let overlapping = self.sampling.overlapping;
        let chunk_size = if overlapping {
            range_size // Each partition gets full range
        } else {
            (range_size + total as u64 - 1) / total as u64
        };

        let (start, end) = if overlapping {
            (self.range.id_min, max_id)
        } else {
            let start = self.range.id_min + index as u64 * chunk_size;
            let end = (start + chunk_size).min(max_id);
            (start, end)
        };

        // Per-partition limit if overall limit is set
        let per_partition_limit = if overlapping {
            self.sampling.limit
        } else {
            self.sampling.limit.map(|l| (l + total as u64 - 1) / total as u64)
        };

        // Create per-partition sampling config
        let mut partition_sampling = self.sampling;
        partition_sampling.limit = per_partition_limit;
        if let Some(seed) = self.sampling.seed {
            partition_sampling.seed = Some(seed.wrapping_add(index as u64));
        }

        let range = IdRange {
            id_min: start,
            id_max: end,
            sub_id_min: self.range.sub_id_min,
            sub_id_max: self.range.sub_id_max,
        };

        PartitionedIter {
            tracker: self.tracker,
            range,
            ctx: FilterContext::new(self.filter, partition_sampling),
            current_id: start,
            current_sub_id: self.range.sub_id_min,
        }
    }
}

// ============================================================================
// Sequential Iterator
// ============================================================================

/// Sequential iterator over IDs in ascending order.
///
/// Thread-safe: multiple instances can run concurrently for reads.
/// Supports sampling configuration for mixed-ratio, probabilistic sampling, and limits.
pub struct SequentialIter<'a> {
    tracker: &'a PrefixTracker,
    range: IdRange,
    ctx: FilterContext,
    current_id: u64,
    current_sub_id: u64,
    max_id: u64,
    wrap_around: bool,
}

impl<'a> SequentialIter<'a> {
    /// Enable wrap-around mode: when iteration reaches the end, restart from beginning.
    ///
    /// This is useful for benchmarks that need indefinite iteration over existing keys.
    ///
    /// # Example
    ///
    /// ```
    /// use keyspace_tracker::PrefixTracker;
    ///
    /// let tracker = PrefixTracker::new(100);
    /// for i in 0..50 {
    ///     tracker.add(i);
    /// }
    ///
    /// let mut iter = tracker.iter().set_only().sequential().wrap_around();
    ///
    /// // Iterate more than keyspace size - wraps around
    /// for _ in 0..200 {
    ///     let (id, _) = iter.next().unwrap();
    ///     // IDs cycle through set bits repeatedly
    /// }
    /// ```
    pub fn wrap_around(mut self) -> Self {
        self.wrap_around = true;
        self
    }

    /// Get next ID for simple (non-hierarchical) trackers.
    fn next_simple(&mut self) -> Option<TrackerItem> {
        let bitmap = self.tracker.primary_bitmap();

        // SIMD fast path: set_only without sampling
        if self.ctx.can_use_find_next_set() {
            loop {
                if !self.ctx.check_limit() { return None; }
                if self.current_id >= self.max_id {
                    if self.wrap_around { self.current_id = self.range.id_min; continue; }
                    return None;
                }
                if let Some(id) = bitmap.find_next_set(self.current_id as usize).filter(|&id| (id as u64) < self.max_id) {
                    self.current_id = id as u64 + 1;
                    self.ctx.record_yield();
                    return Some((id as u64, None));
                }
                if self.wrap_around { self.current_id = self.range.id_min; continue; }
                return None;
            }
        }

        // SIMD fast path: unset_only without sampling
        if self.ctx.can_use_find_next_unset() {
            loop {
                if !self.ctx.check_limit() { return None; }
                if self.current_id >= self.max_id {
                    if self.wrap_around { self.current_id = self.range.id_min; continue; }
                    return None;
                }
                if let Some(id) = bitmap.find_next_unset(self.current_id as usize, self.max_id as usize) {
                    self.current_id = id as u64 + 1;
                    self.ctx.record_yield();
                    return Some((id as u64, None));
                }
                if self.wrap_around { self.current_id = self.range.id_min; continue; }
                return None;
            }
        }

        // Standard path
        loop {
            while self.current_id < self.max_id {
                let id = self.current_id;
                self.current_id += 1;
                if self.ctx.should_yield(bitmap.test(id as usize)) {
                    return Some((id, None));
                }
            }
            if self.wrap_around { self.current_id = self.range.id_min; continue; }
            return None;
        }
    }

    /// Get next (id, sub_id) for hierarchical trackers.
    fn next_hierarchical(&mut self) -> Option<TrackerItem> {
        let sub_bitmaps = self.tracker.sub_bitmaps()?;
        let max_sub_base = self.range.sub_id_max.min(self.tracker.effective_max_sub_id());

        loop {
            if self.current_id >= self.max_id {
                if self.wrap_around {
                    self.current_id = self.range.id_min;
                    self.current_sub_id = self.range.sub_id_min;
                    continue;
                }
                return None;
            }

            if let Some(sub_bitmap) = sub_bitmaps.get(&self.current_id) {
                let max_sub_id = max_sub_base.min(sub_bitmap.capacity() as u64);

                while self.current_sub_id < max_sub_id {
                    let sub_id = self.current_sub_id;
                    self.current_sub_id += 1;
                    if sub_id < self.range.sub_id_min { continue; }
                    if self.ctx.should_yield(sub_bitmap.test(sub_id as usize)) {
                        return Some((self.current_id, Some(sub_id)));
                    }
                }
            }
            self.current_id += 1;
            self.current_sub_id = self.range.sub_id_min;
        }
    }
}

impl Iterator for SequentialIter<'_> {
    type Item = TrackerItem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.tracker.is_hierarchical() {
            self.next_hierarchical()
        } else {
            self.next_simple()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // If limit is set, use it as upper bound
        if let Some(limit) = self.ctx.sampling.limit {
            let remaining = limit.saturating_sub(self.ctx.yielded()) as usize;
            return (0, Some(remaining));
        }

        // For set_only without sampling, we can estimate based on count
        if self.ctx.can_use_find_next_set() && !self.tracker.is_hierarchical() {
            let total_set = self.tracker.count() as usize;
            // Upper bound is total set bits minus what we've yielded
            let remaining = total_set.saturating_sub(self.ctx.yielded() as usize);
            return (0, Some(remaining));
        }

        // For unset_only, estimate based on remaining range
        if self.ctx.can_use_find_next_unset() && !self.tracker.is_hierarchical() {
            let remaining_range = self.max_id.saturating_sub(self.current_id) as usize;
            let set_bits = self.tracker.count() as usize;
            let estimated_unset = remaining_range.saturating_sub(set_bits);
            return (0, Some(estimated_unset));
        }

        // Fallback: remaining range as upper bound
        let remaining = self.max_id.saturating_sub(self.current_id) as usize;
        (0, Some(remaining))
    }
}

// ============================================================================
// Random Iterator
// ============================================================================

/// Random-order iterator over IDs.
///
/// Uses multiplicative hashing for memory-efficient pseudo-random traversal
/// without storing all indices. Supports sampling configuration for:
/// - Mixed-ratio iteration (X% set, Y% unset)
/// - Probabilistic sampling
/// - Limit on returned items
pub struct RandomIter<'a> {
    tracker: &'a PrefixTracker,
    range: IdRange,
    ctx: FilterContext,
    range_size: u64,
    multiplier: u64,
    offset: u64,
    current_step: u64,
    max_steps: u64,
}

impl<'a> RandomIter<'a> {
    fn next_index(&mut self) -> Option<u64> {
        if !self.ctx.check_limit() {
            return None;
        }

        while self.current_step < self.max_steps {
            self.current_step += 1;

            // Select index based on distribution
            let index = if self.ctx.has_distribution() {
                // Use configured distribution (Zipfian, Exponential, etc.)
                self.ctx.sample_index(self.range_size)
            } else {
                // Use multiplicative permutation for uniform coverage
                if self.range_size > 0 {
                    ((self.current_step - 1)
                        .wrapping_mul(self.multiplier)
                        .wrapping_add(self.offset))
                        % self.range_size
                } else {
                    return None;
                }
            };

            let id = self.range.id_min + index;

            if !self.range.contains_id(id) {
                continue;
            }

            let is_set = self.tracker.primary_bitmap().test(id as usize);

            if !self.ctx.matches_filter(is_set) {
                continue;
            }

            if !self.ctx.passes_sampling() {
                continue;
            }

            self.ctx.record_yield();
            return Some(id);
        }

        None
    }
}

impl Iterator for RandomIter<'_> {
    type Item = TrackerItem;

    fn next(&mut self) -> Option<Self::Item> {
        // For simplicity, random iterator only handles primary IDs
        // Hierarchical random iteration would need different approach
        let id = self.next_index()?;

        if self.tracker.is_hierarchical() {
            // For hierarchical, return first sub_id for this ID
            if let Some(sub_bitmaps) = self.tracker.sub_bitmaps() {
                if let Some(sub_bitmap) = sub_bitmaps.get(&id) {
                    if let Some(sub_id) = sub_bitmap.find_next_set(0) {
                        return Some((id, Some(sub_id as u64)));
                    }
                }
            }
            // ID exists in primary but has no sub_ids yet
            Some((id, Some(0)))
        } else {
            Some((id, None))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // If limit is set, use it as upper bound
        if let Some(limit) = self.ctx.sampling.limit {
            let remaining = limit.saturating_sub(self.ctx.yielded()) as usize;
            return (0, Some(remaining));
        }

        // Remaining steps as upper bound
        let remaining_steps = self.max_steps.saturating_sub(self.current_step) as usize;
        (0, Some(remaining_steps))
    }
}

// ============================================================================
// Write Iterator
// ============================================================================

/// Write iterator with atomic claim-and-set semantics.
///
/// Multiple `WriteIter`s can run concurrently. Each call to `next()`
/// atomically claims a unique ID that was previously unset.
pub struct WriteIter<'a> {
    tracker: &'a PrefixTracker,
    range: IdRange,
    filter: BitFilter,
    max_id: u64,
    wrap_around: bool,
}

impl<'a> WriteIter<'a> {
    /// Enable wrap-around mode: when keyspace is exhausted, reset cursor to 0
    /// and clear the bitmap to allow re-claiming IDs.
    ///
    /// This is useful for benchmarks that need to iterate indefinitely.
    ///
    /// # Example
    ///
    /// ```
    /// use keyspace_tracker::PrefixTracker;
    ///
    /// let tracker = PrefixTracker::new(100);
    /// let mut iter = tracker.iter().write().wrap_around();
    ///
    /// // Will never return None - wraps around when exhausted
    /// for _ in 0..1000 {
    ///     let (id, _) = iter.next().unwrap();
    ///     // Process id (0-99, then wraps back to 0)
    /// }
    /// ```
    pub fn wrap_around(mut self) -> Self {
        self.wrap_around = true;
        self
    }

    /// Atomically claim next ID and mark as set.
    ///
    /// Returns `None` when range is exhausted (unless `wrap_around()` is enabled).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<TrackerItem> {
        if self.tracker.is_hierarchical() {
            self.next_hierarchical()
        } else {
            self.next_simple()
        }
    }

    fn next_simple(&mut self) -> Option<TrackerItem> {
        let bitmap = self.tracker.primary_bitmap();
        let cursor = self.tracker.write_cursor();

        match self.filter {
            BitFilter::Unset => {
                // Find and claim next unset bit
                loop {
                    let start = cursor.load(Ordering::Acquire);
                    if start >= self.max_id {
                        if self.wrap_around {
                            // Reset cursor and clear bitmap for next cycle
                            bitmap.clear_range(self.range.id_min as usize, self.max_id as usize);
                            cursor.store(self.range.id_min, Ordering::Release);
                            continue;
                        }
                        return None;
                    }

                    // Find next unset bit (within bitmap capacity)
                    // If cursor is beyond capacity, all bits from cursor to max_id are unset
                    let capacity = bitmap.capacity() as u64;
                    let id = if start >= capacity {
                        // Beyond bitmap capacity - bit is implicitly unset
                        start
                    } else if let Some(found) = bitmap.find_next_unset(start as usize, self.max_id as usize) {
                        found as u64
                    } else if capacity < self.max_id {
                        // No unset bits in bitmap, but there are IDs beyond capacity
                        capacity
                    } else {
                        // No more unset bits
                        if self.wrap_around {
                            bitmap.clear_range(self.range.id_min as usize, self.max_id as usize);
                            cursor.store(self.range.id_min, Ordering::Release);
                            continue;
                        }
                        cursor.store(self.max_id, Ordering::Release);
                        return None;
                    };

                    // Bounds check (capacity can grow during iteration)
                    if id >= self.max_id {
                        if self.wrap_around {
                            bitmap.clear_range(self.range.id_min as usize, self.max_id as usize);
                            cursor.store(self.range.id_min, Ordering::Release);
                            continue;
                        }
                        cursor.store(self.max_id, Ordering::Release);
                        return None;
                    }

                    if id < self.range.id_min {
                        cursor.fetch_max(self.range.id_min, Ordering::AcqRel);
                        continue;
                    }

                    // Try to claim it
                    if bitmap.test_and_set(id as usize) {
                        // Advance cursor past this bit
                        cursor.fetch_max(id + 1, Ordering::AcqRel);
                        return Some((id, None));
                    }
                    // Someone else claimed it, retry
                }
            }
            BitFilter::All => {
                // Claim any bit (set if unset, or just return if already set)
                loop {
                    let id = cursor.fetch_add(1, Ordering::AcqRel);
                    if id >= self.max_id {
                        if self.wrap_around {
                            // Reset cursor for next cycle (don't clear - All mode doesn't care)
                            cursor.store(self.range.id_min, Ordering::Release);
                            continue;
                        }
                        return None;
                    }
                    if id < self.range.id_min {
                        cursor.fetch_max(self.range.id_min, Ordering::AcqRel);
                        continue;
                    }

                    // Set the bit (may already be set)
                    bitmap.set(id as usize);
                    return Some((id, None));
                }
            }
            BitFilter::Set => {
                // WriteIter with Set filter doesn't make sense (can't claim what's already set)
                None
            }
        }
    }

    fn next_hierarchical(&mut self) -> Option<TrackerItem> {
        // For hierarchical, we iterate through (id, sub_id) pairs
        let max_sub_id = self
            .range
            .sub_id_max
            .min(self.tracker.effective_max_sub_id());

        match self.filter {
            BitFilter::Unset => {
                // Find and claim next unset (id, sub_id) pair
                let cursor = self.tracker.write_cursor();

                loop {
                    // Get next candidate
                    let combined = cursor.fetch_add(1, Ordering::AcqRel);

                    // Decode combined cursor into (id, sub_id)
                    let id = combined / max_sub_id.max(1);
                    let sub_id = combined % max_sub_id.max(1);

                    if id >= self.max_id {
                        return None;
                    }

                    if !self.range.contains(id, sub_id) {
                        continue;
                    }

                    // Try to claim this pair
                    if self.tracker.claim_pair(id, sub_id) {
                        return Some((id, Some(sub_id)));
                    }
                    // Already claimed, try next
                }
            }
            BitFilter::All => {
                // Similar to above but always set
                let cursor = self.tracker.write_cursor();
                let combined = cursor.fetch_add(1, Ordering::AcqRel);

                let id = combined / max_sub_id.max(1);
                let sub_id = combined % max_sub_id.max(1);

                if id >= self.max_id {
                    return None;
                }

                if !self.range.contains(id, sub_id) {
                    return self.next(); // Recurse to find valid pair
                }

                self.tracker.add_pair(id, sub_id);
                Some((id, Some(sub_id)))
            }
            BitFilter::Set => None,
        }
    }
}

// ============================================================================
// Delete Iterator
// ============================================================================

/// Exclusive delete iterator.
///
/// Only one can exist per tracker at a time. Atomically claims and clears bits.
pub struct DeleteIter<'a> {
    tracker: &'a PrefixTracker,
    range: IdRange,
    current_id: u64,
    current_sub_id: u64,
    max_id: u64,
}

impl<'a> DeleteIter<'a> {
    /// Atomically claim next set ID and clear it.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<TrackerItem> {
        if self.tracker.is_hierarchical() {
            self.next_hierarchical()
        } else {
            self.next_simple()
        }
    }

    fn next_simple(&mut self) -> Option<TrackerItem> {
        let bitmap = self.tracker.primary_bitmap();

        while self.current_id < self.max_id {
            let id = self.current_id;
            self.current_id += 1;

            if !self.range.contains_id(id) {
                continue;
            }

            // Try to atomically clear
            if bitmap.test_and_clear(id as usize) {
                return Some((id, None));
            }
        }

        None
    }

    fn next_hierarchical(&mut self) -> Option<TrackerItem> {
        let sub_bitmaps = self.tracker.sub_bitmaps()?;

        while self.current_id < self.max_id {
            if let Some(sub_bitmap) = sub_bitmaps.get(&self.current_id) {
                // Use actual bitmap capacity, clamped by range
                let bitmap_cap = sub_bitmap.capacity() as u64;
                let max_sub_id = self
                    .range
                    .sub_id_max
                    .min(self.tracker.effective_max_sub_id())
                    .min(bitmap_cap);

                while self.current_sub_id < max_sub_id {
                    let sub_id = self.current_sub_id;
                    self.current_sub_id += 1;

                    if !self.range.contains(self.current_id, sub_id) {
                        continue;
                    }

                    // Try to atomically clear
                    if sub_bitmap.test_and_clear(sub_id as usize) {
                        return Some((self.current_id, Some(sub_id)));
                    }
                }
            }

            self.current_id += 1;
            self.current_sub_id = self.range.sub_id_min;
        }

        None
    }
}

impl Drop for DeleteIter<'_> {
    fn drop(&mut self) {
        self.tracker.release_delete_lock();
    }
}

// ============================================================================
// Partitioned Iterator
// ============================================================================

/// Partitioned iterator for parallel processing with guaranteed disjoint ranges.
/// Supports sampling configuration for mixed-ratio, probabilistic sampling, and limits.
pub struct PartitionedIter<'a> {
    tracker: &'a PrefixTracker,
    range: IdRange,
    ctx: FilterContext,
    current_id: u64,
    current_sub_id: u64,
}

impl<'a> PartitionedIter<'a> {
    /// Get the range covered by this partition.
    pub fn range(&self) -> &IdRange {
        &self.range
    }
}

impl Iterator for PartitionedIter<'_> {
    type Item = TrackerItem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.tracker.is_hierarchical() {
            self.next_hierarchical()
        } else {
            self.next_simple()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // If limit is set, use it as upper bound
        if let Some(limit) = self.ctx.sampling.limit {
            let remaining = limit.saturating_sub(self.ctx.yielded()) as usize;
            return (0, Some(remaining));
        }

        // Partition range as upper bound
        let remaining = self.range.id_max.saturating_sub(self.current_id) as usize;
        (0, Some(remaining))
    }
}

impl<'a> PartitionedIter<'a> {
    fn next_simple(&mut self) -> Option<TrackerItem> {
        let bitmap = self.tracker.primary_bitmap();
        let max = self.range.id_max;

        // SIMD fast path: set_only without sampling
        if self.ctx.can_use_find_next_set() {
            if !self.ctx.check_limit() || self.current_id >= max { return None; }
            let id = bitmap.find_next_set(self.current_id as usize)? as u64;
            if id >= max { return None; }
            self.current_id = id + 1;
            self.ctx.record_yield();
            return Some((id, None));
        }

        // SIMD fast path: unset_only without sampling
        if self.ctx.can_use_find_next_unset() {
            if !self.ctx.check_limit() || self.current_id >= max { return None; }
            let id = bitmap.find_next_unset(self.current_id as usize, max as usize)? as u64;
            self.current_id = id + 1;
            self.ctx.record_yield();
            return Some((id, None));
        }

        // Standard path
        while self.current_id < max {
            let id = self.current_id;
            self.current_id += 1;
            if self.ctx.should_yield(bitmap.test(id as usize)) {
                return Some((id, None));
            }
        }
        None
    }

    fn next_hierarchical(&mut self) -> Option<TrackerItem> {
        let sub_bitmaps = self.tracker.sub_bitmaps()?;
        let max_sub_base = self.range.sub_id_max.min(self.tracker.effective_max_sub_id());

        while self.current_id < self.range.id_max {
            if let Some(sub_bitmap) = sub_bitmaps.get(&self.current_id) {
                let max_sub_id = max_sub_base.min(sub_bitmap.capacity() as u64);

                while self.current_sub_id < max_sub_id {
                    let sub_id = self.current_sub_id;
                    self.current_sub_id += 1;
                    if sub_id < self.range.sub_id_min { continue; }
                    if self.ctx.should_yield(sub_bitmap.test(sub_id as usize)) {
                        return Some((self.current_id, Some(sub_id)));
                    }
                }
            }
            self.current_id += 1;
            self.current_sub_id = self.range.sub_id_min;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TrackerConfig;

    // ========================================================================
    // Size Hint Tests
    // ========================================================================

    #[test]
    fn test_size_hint_sequential_set_only() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(1000));
        for i in 0..100 {
            tracker.add(i);
        }

        let iter = tracker.iter().set_only().sequential();
        let (_, upper) = iter.size_hint();

        // Upper bound should be the count of set bits
        assert_eq!(upper, Some(100));
    }

    #[test]
    fn test_size_hint_sequential_with_limit() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(1000));
        for i in 0..100 {
            tracker.add(i);
        }

        let iter = tracker.iter().set_only().limit(50).sequential();
        let (_, upper) = iter.size_hint();

        // Upper bound should be the limit
        assert_eq!(upper, Some(50));
    }

    #[test]
    fn test_size_hint_random() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(1000));
        for i in 0..100 {
            tracker.add(i);
        }

        let iter = tracker.iter().set_only().random();
        let (_, upper) = iter.size_hint();

        // Upper bound should be max_steps (range_size)
        assert_eq!(upper, Some(1000));
    }

    #[test]
    fn test_size_hint_partitioned() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(1000));
        for i in 0..100 {
            tracker.add(i);
        }

        let iter = tracker.iter().set_only().partition(0, 4);
        let (_, upper) = iter.size_hint();

        // Upper bound should be partition range (1000 / 4 = 250)
        assert_eq!(upper, Some(250));
    }

    // ========================================================================
    // Sampling Tests
    // ========================================================================

    #[test]
    fn test_limit_sequential() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));
        for i in 0..100 {
            tracker.add(i);
        }

        let items: Vec<_> = tracker.iter().set_only().limit(10).sequential().collect();
        assert_eq!(items.len(), 10);
    }

    #[test]
    fn test_limit_random() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));
        for i in 0..100 {
            tracker.add(i);
        }

        let items: Vec<_> = tracker.iter().set_only().limit(10).random().collect();
        assert_eq!(items.len(), 10);
    }

    #[test]
    fn test_sample_probability() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(1000));
        for i in 0..1000 {
            tracker.add(i);
        }

        // Sample ~50% - should get roughly 400-600 items
        let items: Vec<_> = tracker
            .iter()
            .set_only()
            .sample(0.5)
            .seed(12345) // Reproducible
            .sequential()
            .collect();

        assert!(items.len() > 300 && items.len() < 700);
    }

    #[test]
    fn test_seeded_random_reproducible() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));
        for i in 0..100 {
            tracker.add(i);
        }

        // Same seed should produce same sequence
        let items1: Vec<_> = tracker
            .iter()
            .set_only()
            .seed(42)
            .random()
            .take(20)
            .collect();

        let items2: Vec<_> = tracker
            .iter()
            .set_only()
            .seed(42)
            .random()
            .take(20)
            .collect();

        assert_eq!(items1, items2);
    }

    #[test]
    fn test_mixed_ratio() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));
        // Set half the bits
        for i in 0..50 {
            tracker.add(i);
        }

        // 80% set, 20% unset
        let items: Vec<_> = tracker
            .iter()
            .mixed_ratio(0.8)
            .seed(12345)
            .limit(100)
            .random()
            .collect();

        let set_count = items.iter().filter(|(id, _)| tracker.exists(*id)).count();
        let unset_count = items.len() - set_count;

        // Should have more set than unset (roughly 80/20)
        assert!(set_count > unset_count * 2);
    }

    #[test]
    fn test_range_percent() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));
        for i in 0..100 {
            tracker.add(i);
        }

        // Upper 50%
        let items: Vec<_> = tracker
            .iter()
            .set_only()
            .range_percent(0.5, 1.0)
            .sequential()
            .collect();

        assert!(items.iter().all(|(id, _)| *id >= 50));
        assert_eq!(items.len(), 50);
    }

    #[test]
    fn test_partitioned_with_sampling() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(1000));
        for i in 0..1000 {
            tracker.add(i);
        }

        // Each partition should get ~25 items (100 total / 4 partitions)
        let partitions = tracker.iter().set_only().limit(100).partitioned(4);

        let total: usize = partitions.into_iter().map(|p| p.count()).sum();
        // Total should be close to 100 (may vary slightly due to partitioning)
        assert!(total >= 96 && total <= 104);
    }

    // ========================================================================
    // Workload & Distribution Tests
    // ========================================================================

    #[test]
    fn test_zipfian_distribution() {
        use crate::AccessDistribution;

        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(10000));
        for i in 0..10000 {
            tracker.add(i);
        }

        // Zipfian should heavily favor low-index keys
        let items: Vec<_> = tracker
            .iter()
            .set_only()
            .distribution(AccessDistribution::Zipfian { skew: 1.0 })
            .seed(42)
            .limit(1000)
            .random()
            .collect();

        assert_eq!(items.len(), 1000);

        // Count how many are in first 10% of keyspace
        let low_count = items.iter().filter(|(id, _)| *id < 1000).count();
        // Should be more than 50% (Zipfian is heavily skewed)
        assert!(low_count > 500, "Zipfian should favor low indices, got {} in first 10%", low_count);
    }

    #[test]
    fn test_hotspot_distribution() {
        use crate::AccessDistribution;

        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(10000));
        for i in 0..10000 {
            tracker.add(i);
        }

        // 10% hot keys should get 90% of accesses
        let items: Vec<_> = tracker
            .iter()
            .set_only()
            .distribution(AccessDistribution::Hotspot { hot_pct: 0.1, hot_prob: 0.9 })
            .seed(42)
            .limit(1000)
            .random()
            .collect();

        let hot_count = items.iter().filter(|(id, _)| *id < 1000).count();
        // Should be around 90%
        assert!(hot_count > 800, "Hotspot should strongly favor hot keys, got {} hot", hot_count);
    }

    #[test]
    fn test_overlapping_mode() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));
        for i in 0..100 {
            tracker.add(i);
        }

        // With overlapping, multiple partitions can see same keys
        let partitions = tracker
            .iter()
            .set_only()
            .overlapping()
            .seed(42)
            .partitioned(4);

        let mut all_items: Vec<u64> = Vec::new();
        for partition in partitions {
            for (id, _) in partition {
                all_items.push(id);
            }
        }

        // All 100 items should be visited by each partition
        assert_eq!(all_items.len(), 400);
    }

    // ========================================================================
    // Original Tests
    // ========================================================================

    #[test]
    fn test_sequential_iter_simple() {
        let tracker = PrefixTracker::simple("vec:");
        tracker.add(0);
        tracker.add(5);
        tracker.add(10);

        let items: Vec<_> = tracker.iter().set_only().sequential().collect();
        assert_eq!(items, vec![(0, None), (5, None), (10, None)]);
    }

    #[test]
    fn test_sequential_iter_unset() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(10));
        tracker.add(2);
        tracker.add(5);

        let items: Vec<_> = tracker.iter().unset_only().sequential().collect();
        assert_eq!(
            items,
            vec![
                (0, None),
                (1, None),
                (3, None),
                (4, None),
                (6, None),
                (7, None),
                (8, None),
                (9, None)
            ]
        );
    }

    #[test]
    fn test_sequential_iter_range() {
        let tracker = PrefixTracker::simple("vec:");
        for i in 0..100 {
            tracker.add(i);
        }

        let items: Vec<_> = tracker
            .iter()
            .id_range(10, 15)
            .set_only()
            .sequential()
            .collect();
        assert_eq!(
            items,
            vec![
                (10, None),
                (11, None),
                (12, None),
                (13, None),
                (14, None)
            ]
        );
    }

    #[test]
    fn test_sequential_iter_hierarchical() {
        let tracker = PrefixTracker::hierarchical("hash:");
        tracker.add_pair(1, 0);
        tracker.add_pair(1, 5);
        tracker.add_pair(2, 10);

        let items: Vec<_> = tracker.iter().set_only().sequential().collect();
        assert_eq!(items, vec![(1, Some(0)), (1, Some(5)), (2, Some(10))]);
    }

    #[test]
    fn test_random_iter() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));
        for i in 0..50 {
            tracker.add(i * 2);
        }

        let items: Vec<_> = tracker.iter().set_only().random().collect();

        // Should have all 50 items, but in random order
        assert_eq!(items.len(), 50);

        // All items should be even numbers
        for (id, _) in &items {
            assert!(id % 2 == 0);
        }
    }

    #[test]
    fn test_write_iter_simple() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));

        let mut iter = tracker.iter().unset_only().write();
        let mut claimed = Vec::new();

        while let Some((id, _)) = iter.next() {
            claimed.push(id);
            if claimed.len() >= 10 {
                break;
            }
        }

        assert_eq!(claimed.len(), 10);
        assert_eq!(tracker.count(), 10);

        // All claimed IDs should now be set
        for id in &claimed {
            assert!(tracker.exists(*id));
        }
    }

    #[test]
    fn test_write_iter_concurrent() {
        use std::sync::atomic::AtomicU64;
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(PrefixTracker::new(
            TrackerConfig::simple("vec:").with_max_id(1000),
        ));
        let claimed_count = Arc::new(AtomicU64::new(0));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let tracker = tracker.clone();
                let claimed = claimed_count.clone();
                thread::spawn(move || {
                    let mut iter = tracker.iter().unset_only().write();
                    while iter.next().is_some() {
                        claimed.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All 1000 IDs should be claimed exactly once
        assert_eq!(claimed_count.load(Ordering::Relaxed), 1000);
        assert_eq!(tracker.count(), 1000);
    }

    #[test]
    fn test_delete_iter_exclusive() {
        let tracker = PrefixTracker::simple("vec:");
        tracker.add(0);
        tracker.add(1);

        let iter1 = tracker.iter().delete();
        assert!(iter1.is_some());

        // Second delete iterator should fail
        let iter2 = tracker.iter().delete();
        assert!(iter2.is_none());

        drop(iter1);

        // Now it should work
        let iter3 = tracker.iter().delete();
        assert!(iter3.is_some());
    }

    #[test]
    fn test_delete_iter() {
        let tracker = PrefixTracker::simple("vec:");
        tracker.add(0);
        tracker.add(5);
        tracker.add(10);

        let mut deleted = Vec::new();
        if let Some(mut iter) = tracker.iter().delete() {
            while let Some((id, _)) = iter.next() {
                deleted.push(id);
            }
        }

        assert_eq!(deleted, vec![0, 5, 10]);
        assert_eq!(tracker.count(), 0);
    }

    #[test]
    fn test_partitioned_iter() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));
        for i in 0..100 {
            tracker.add(i);
        }

        let partitions = tracker.iter().set_only().partitioned(4);
        assert_eq!(partitions.len(), 4);

        let mut all_items: Vec<_> = partitions.into_iter().flatten().collect();
        all_items.sort_by_key(|(id, _)| *id);

        assert_eq!(all_items.len(), 100);
        for (i, (id, _)) in all_items.iter().enumerate() {
            assert_eq!(*id, i as u64);
        }
    }

    #[test]
    fn test_partitioned_no_overlap() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(1000));
        for i in 0..1000 {
            tracker.add(i);
        }

        let partitions = tracker.iter().set_only().partitioned(8);

        // Check that ranges don't overlap
        for i in 0..partitions.len() {
            for j in (i + 1)..partitions.len() {
                let r1 = partitions[i].range();
                let r2 = partitions[j].range();

                assert!(r1.id_max <= r2.id_min || r2.id_max <= r1.id_min);
            }
        }
    }

    #[test]
    fn test_single_partition() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));
        for i in 0..100 {
            tracker.add(i);
        }

        // Each worker independently creates its partition
        let total = 4;
        let mut all_items = Vec::new();

        for worker_id in 0..total {
            let items: Vec<_> = tracker
                .iter()
                .set_only()
                .partition(worker_id, total)
                .collect();
            all_items.extend(items);
        }

        all_items.sort_by_key(|(id, _)| *id);
        assert_eq!(all_items.len(), 100);
    }

    #[test]
    fn test_partition_equals_partitioned() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(1000));
        for i in 0..1000 {
            tracker.add(i);
        }

        let total = 8;
        
        // Get all partitions at once
        let all_partitions = tracker.iter().set_only().partitioned(total);
        
        // Get each partition independently
        for (index, partition) in all_partitions.into_iter().enumerate() {
            let independent = tracker.iter().set_only().partition(index, total);
            
            // Ranges should match
            assert_eq!(partition.range().id_min, independent.range().id_min);
            assert_eq!(partition.range().id_max, independent.range().id_max);
        }
    }

    #[test]
    fn test_partition_disjoint_ranges() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(1000));
        for i in 0..1000 {
            tracker.add(i);
        }

        let total = 8;
        let mut ranges = Vec::new();

        for i in 0..total {
            let p = tracker.iter().set_only().partition(i, total);
            ranges.push((p.range().id_min, p.range().id_max));
        }

        // Check disjoint
        for i in 0..total {
            for j in (i + 1)..total {
                assert!(
                    ranges[i].1 <= ranges[j].0 || ranges[j].1 <= ranges[i].0,
                    "Partitions {} and {} overlap", i, j
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "partition index")]
    fn test_partition_index_out_of_bounds() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));
        tracker.iter().partition(5, 4); // index >= total should panic
    }

    #[test]
    #[should_panic(expected = "total partitions must be > 0")]
    fn test_partition_zero_total() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));
        tracker.iter().partition(0, 0); // total == 0 should panic
    }

    #[test]
    fn test_partition_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(PrefixTracker::new(
            TrackerConfig::simple("vec:").with_max_id(10000),
        ));
        for i in 0..10000 {
            tracker.add(i);
        }

        let num_workers = 8;
        let handles: Vec<_> = (0..num_workers)
            .map(|worker_id| {
                let t = tracker.clone();
                thread::spawn(move || {
                    t.iter()
                        .set_only()
                        .partition(worker_id, num_workers)
                        .count() as u64
                })
            })
            .collect();

        let total: u64 = handles.into_iter().map(|h: thread::JoinHandle<u64>| h.join().unwrap()).sum();
        assert_eq!(total, 10000);
    }

    #[test]
    fn test_continue_write_multithreaded() {
        use std::collections::HashSet;
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(PrefixTracker::new(
            TrackerConfig::simple("vec:").with_max_id(10000),
        ));

        // Reset cursor once before spawning workers
        tracker.reset_write_cursor();

        let num_workers = 8;
        let claims_per_worker = 100;

        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let t = tracker.clone();
                thread::spawn(move || {
                    let mut claimed = Vec::with_capacity(claims_per_worker);
                    let mut iter = t.iter().continue_write();

                    for _ in 0..claims_per_worker {
                        if let Some((id, _sub_id)) = iter.next() {
                            claimed.push(id);
                        }
                    }
                    claimed
                })
            })
            .collect();

        // Collect all claimed IDs
        let all_claimed: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        // Verify no duplicates - each ID should be claimed exactly once
        let unique: HashSet<_> = all_claimed.iter().collect();
        assert_eq!(
            unique.len(),
            all_claimed.len(),
            "Duplicate IDs claimed! Got {} claims but only {} unique",
            all_claimed.len(),
            unique.len()
        );

        // Should have claimed exactly num_workers * claims_per_worker IDs
        assert_eq!(all_claimed.len(), num_workers * claims_per_worker);
    }

    #[test]
    fn test_continue_write_vs_write_cursor_behavior() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(100));

        // First write() resets cursor to 0
        let mut iter1 = tracker.iter().write();
        let (id1, _) = iter1.next().unwrap();
        assert_eq!(id1, 0);

        // Second write() also resets cursor to 0 - gets same ID!
        let mut iter2 = tracker.iter().write();
        let (id2, _) = iter2.next().unwrap();
        assert_eq!(id2, 0, "write() should reset cursor");

        // Now use continue_write() - manually reset first
        tracker.reset_write_cursor();
        let mut iter3 = tracker.iter().continue_write();
        let (id3, _) = iter3.next().unwrap();
        assert_eq!(id3, 0);

        // continue_write() does NOT reset - continues from cursor
        let mut iter4 = tracker.iter().continue_write();
        let (id4, _) = iter4.next().unwrap();
        assert_eq!(id4, 1, "continue_write() should NOT reset cursor");

        let mut iter5 = tracker.iter().continue_write();
        let (id5, _) = iter5.next().unwrap();
        assert_eq!(id5, 2, "continue_write() should continue advancing");
    }

    #[test]
    fn test_write_wrap_around() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(10));

        // Without wrap_around - stops at exhaustion
        let mut iter = tracker.iter().write();
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 10);
        assert!(iter.next().is_none(), "Should stop at exhaustion");

        // With wrap_around - continues indefinitely
        tracker.reset_write_cursor();
        let mut iter = tracker.iter().write().wrap_around();
        
        // Claim more than keyspace size
        let mut ids = Vec::new();
        for _ in 0..25 {
            let (id, _) = iter.next().unwrap();
            ids.push(id);
        }

        // Should have wrapped around at least twice
        assert_eq!(ids.len(), 25);
        
        // Check wraparound occurred - IDs should cycle through 0-9
        assert!(ids[10..20].iter().all(|&id| id < 10), "Second cycle should be 0-9");
        assert!(ids[20..25].iter().all(|&id| id < 10), "Third cycle should be 0-9");
    }

    #[test]
    fn test_write_wrap_around_multithreaded() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(PrefixTracker::new(
            TrackerConfig::simple("vec:").with_max_id(100),
        ));
        let total_claimed = Arc::new(AtomicU64::new(0));

        // Reset cursor once before spawning workers
        tracker.reset_write_cursor();

        let num_workers = 4;
        let claims_per_worker = 500; // 2000 total claims for 100 IDs = 20 full cycles

        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let t = tracker.clone();
                let counter = total_claimed.clone();
                thread::spawn(move || {
                    // Use continue_write() with wrap_around for concurrent access
                    let mut iter = t.iter().continue_write().wrap_around();

                    for _ in 0..claims_per_worker {
                        let (id, _) = iter.next().unwrap();
                        assert!(id < 100, "ID should be within range");
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // All workers should have completed their claims
        assert_eq!(
            total_claimed.load(Ordering::SeqCst),
            (num_workers * claims_per_worker) as u64
        );
    }

    #[test]
    fn test_sequential_wrap_around() {
        let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(10));
        
        // Add some IDs
        for i in 0..5 {
            tracker.add(i);
        }

        // Without wrap_around - stops at end
        let count = tracker.iter().set_only().sequential().count();
        assert_eq!(count, 5);

        // With wrap_around - continues indefinitely (use limit to stop)
        let mut iter = tracker.iter().set_only().sequential().wrap_around();
        let mut ids = Vec::new();
        for _ in 0..15 {
            let (id, _) = iter.next().unwrap();
            ids.push(id);
        }

        // Should cycle through set IDs (0,1,2,3,4) three times
        assert_eq!(ids.len(), 15);
        assert_eq!(ids[0..5], [0, 1, 2, 3, 4]);
        assert_eq!(ids[5..10], [0, 1, 2, 3, 4]);
        assert_eq!(ids[10..15], [0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sequential_wrap_around_multithreaded() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(PrefixTracker::new(
            TrackerConfig::simple("vec:").with_max_id(100),
        ));

        // Add all IDs
        for i in 0..100 {
            tracker.add(i);
        }

        let total_read = Arc::new(AtomicU64::new(0));
        let num_workers = 4;
        let reads_per_worker = 500;

        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let t = tracker.clone();
                let counter = total_read.clone();
                thread::spawn(move || {
                    let mut iter = t.iter().set_only().sequential().wrap_around();

                    for _ in 0..reads_per_worker {
                        let (id, _) = iter.next().unwrap();
                        assert!(id < 100, "ID should be within range");
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(
            total_read.load(Ordering::SeqCst),
            (num_workers * reads_per_worker) as u64
        );
    }
}
