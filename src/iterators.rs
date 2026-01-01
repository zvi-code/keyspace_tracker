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

use super::config::{BitFilter, IdRange, SamplingConfig};
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

    /// Check if sampling is configured.
    #[inline]
    fn has_sampling(&self) -> bool {
        self.sampling.has_sampling()
    }

    /// Build a sequential (ascending order) iterator.
    pub fn sequential(self) -> SequentialIter<'a> {
        let max_id = self.range.id_max.min(self.tracker.effective_max_id());

        SequentialIter {
            tracker: self.tracker,
            range: self.range,
            filter: self.filter,
            sampling: self.sampling,
            current_id: self.range.id_min,
            current_sub_id: self.range.sub_id_min,
            max_id,
            rng: self.sampling.make_rng(),
            yielded: 0,
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
            filter: self.filter,
            sampling: self.sampling,
            range_size,
            multiplier,
            offset,
            current_step: 0,
            max_steps: range_size,
            rng,
            yielded: 0,
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
    pub fn write(self) -> WriteIter<'a> {
        // Reset cursor at start of iteration
        self.tracker.reset_write_cursor();

        let max_id = self.range.id_max.min(self.tracker.effective_max_id());

        WriteIter {
            tracker: self.tracker,
            range: self.range,
            filter: self.filter,
            max_id,
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

                PartitionedIter {
                    tracker: self.tracker,
                    range: IdRange {
                        id_min: start,
                        id_max: end,
                        sub_id_min: self.range.sub_id_min,
                        sub_id_max: self.range.sub_id_max,
                    },
                    filter: self.filter,
                    sampling: partition_sampling,
                    current_id: start,
                    current_sub_id: self.range.sub_id_min,
                    rng: partition_sampling.make_rng(),
                    yielded: 0,
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

        PartitionedIter {
            tracker: self.tracker,
            range: IdRange {
                id_min: start,
                id_max: end,
                sub_id_min: self.range.sub_id_min,
                sub_id_max: self.range.sub_id_max,
            },
            filter: self.filter,
            sampling: partition_sampling,
            current_id: start,
            current_sub_id: self.range.sub_id_min,
            rng: partition_sampling.make_rng(),
            yielded: 0,
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
    filter: BitFilter,
    sampling: SamplingConfig,
    current_id: u64,
    current_sub_id: u64,
    max_id: u64,
    rng: fastrand::Rng,
    yielded: u64,
}

impl<'a> SequentialIter<'a> {
    /// Check if we should continue iterating (respects limit).
    #[inline]
    fn check_limit(&self) -> bool {
        if let Some(limit) = self.sampling.limit {
            self.yielded < limit
        } else {
            true
        }
    }

    /// Check if item matches filter (with mixed-ratio support).
    #[inline]
    fn matches_filter(&mut self, is_set: bool) -> bool {
        if let Some(set_ratio) = self.sampling.set_ratio {
            // Mixed-ratio mode
            let want_set = self.rng.f64() < set_ratio;
            want_set == is_set
        } else {
            match self.filter {
                BitFilter::All => true,
                BitFilter::Set => is_set,
                BitFilter::Unset => !is_set,
            }
        }
    }

    /// Check probabilistic sampling.
    #[inline]
    fn passes_sampling(&mut self) -> bool {
        if self.sampling.sample_probability < 1.0 {
            self.rng.f64() < self.sampling.sample_probability
        } else {
            true
        }
    }

    /// Get next ID for simple (non-hierarchical) trackers.
    fn next_simple(&mut self) -> Option<TrackerItem> {
        if !self.check_limit() {
            return None;
        }

        let bitmap = self.tracker.primary_bitmap();

        while self.current_id < self.max_id {
            let id = self.current_id;
            self.current_id += 1;

            let is_set = bitmap.test(id as usize);

            if !self.matches_filter(is_set) {
                continue;
            }

            if !self.passes_sampling() {
                continue;
            }

            self.yielded += 1;
            return Some((id, None));
        }

        None
    }

    /// Get next (id, sub_id) for hierarchical trackers.
    fn next_hierarchical(&mut self) -> Option<TrackerItem> {
        if !self.check_limit() {
            return None;
        }

        let sub_bitmaps = self.tracker.sub_bitmaps()?;

        loop {
            // Check if we've exhausted all primary IDs
            if self.current_id >= self.max_id {
                return None;
            }

            // Check if current primary ID has a sub-bitmap
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

                    if sub_id < self.range.sub_id_min {
                        continue;
                    }

                    let is_set = sub_bitmap.test(sub_id as usize);

                    if !self.matches_filter(is_set) {
                        continue;
                    }

                    if !self.passes_sampling() {
                        continue;
                    }

                    // Re-check limit after potential skip
                    if !self.check_limit() {
                        return None;
                    }

                    self.yielded += 1;
                    return Some((self.current_id, Some(sub_id)));
                }
            }

            // Move to next primary ID
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
    filter: BitFilter,
    sampling: SamplingConfig,
    range_size: u64,
    multiplier: u64,
    offset: u64,
    current_step: u64,
    max_steps: u64,
    rng: fastrand::Rng,
    yielded: u64,
}

impl<'a> RandomIter<'a> {
    fn next_index(&mut self) -> Option<u64> {
        // Check limit
        if let Some(limit) = self.sampling.limit {
            if self.yielded >= limit {
                return None;
            }
        }

        while self.current_step < self.max_steps {
            self.current_step += 1;

            // Select index based on distribution
            let index = if self.sampling.has_distribution() {
                // Use configured distribution (Zipfian, Exponential, etc.)
                self.sampling.sample_index(&mut self.rng, self.range_size)
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

            // Apply filter logic (with mixed-ratio support)
            let matches = if let Some(set_ratio) = self.sampling.set_ratio {
                // Mixed-ratio mode: probabilistically select set vs unset
                let want_set = self.rng.f64() < set_ratio;
                want_set == is_set
            } else {
                // Standard filter mode
                match self.filter {
                    BitFilter::All => true,
                    BitFilter::Set => is_set,
                    BitFilter::Unset => !is_set,
                }
            };

            if !matches {
                continue;
            }

            // Apply probabilistic sampling
            if self.sampling.sample_probability < 1.0 {
                if self.rng.f64() >= self.sampling.sample_probability {
                    continue;
                }
            }

            self.yielded += 1;
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
}

impl<'a> WriteIter<'a> {
    /// Atomically claim next ID and mark as set.
    ///
    /// Returns `None` when range is exhausted.
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
                        return None;
                    }

                    // Find next unset bit
                    if let Some(id) = bitmap.find_next_unset(start as usize, self.max_id as usize) {
                        let id = id as u64;
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
                    } else {
                        // No more unset bits
                        cursor.store(self.max_id, Ordering::Release);
                        return None;
                    }
                }
            }
            BitFilter::All => {
                // Claim any bit (set if unset, or just return if already set)
                loop {
                    let id = cursor.fetch_add(1, Ordering::AcqRel);
                    if id >= self.max_id {
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
    filter: BitFilter,
    sampling: SamplingConfig,
    current_id: u64,
    current_sub_id: u64,
    rng: fastrand::Rng,
    yielded: u64,
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
}

impl<'a> PartitionedIter<'a> {
    /// Check if we should continue iterating (respects limit).
    #[inline]
    fn check_limit(&self) -> bool {
        if let Some(limit) = self.sampling.limit {
            self.yielded < limit
        } else {
            true
        }
    }

    /// Check if item matches filter (with mixed-ratio support).
    #[inline]
    fn matches_filter(&mut self, is_set: bool) -> bool {
        if let Some(set_ratio) = self.sampling.set_ratio {
            let want_set = self.rng.f64() < set_ratio;
            want_set == is_set
        } else {
            match self.filter {
                BitFilter::All => true,
                BitFilter::Set => is_set,
                BitFilter::Unset => !is_set,
            }
        }
    }

    /// Check probabilistic sampling.
    #[inline]
    fn passes_sampling(&mut self) -> bool {
        if self.sampling.sample_probability < 1.0 {
            self.rng.f64() < self.sampling.sample_probability
        } else {
            true
        }
    }

    fn next_simple(&mut self) -> Option<TrackerItem> {
        if !self.check_limit() {
            return None;
        }

        let bitmap = self.tracker.primary_bitmap();

        while self.current_id < self.range.id_max {
            let id = self.current_id;
            self.current_id += 1;

            let is_set = bitmap.test(id as usize);

            if !self.matches_filter(is_set) {
                continue;
            }

            if !self.passes_sampling() {
                continue;
            }

            self.yielded += 1;
            return Some((id, None));
        }

        None
    }

    fn next_hierarchical(&mut self) -> Option<TrackerItem> {
        if !self.check_limit() {
            return None;
        }

        let sub_bitmaps = self.tracker.sub_bitmaps()?;

        while self.current_id < self.range.id_max {
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

                    if sub_id < self.range.sub_id_min {
                        continue;
                    }

                    let is_set = sub_bitmap.test(sub_id as usize);

                    if !self.matches_filter(is_set) {
                        continue;
                    }

                    if !self.passes_sampling() {
                        continue;
                    }

                    if !self.check_limit() {
                        return None;
                    }

                    self.yielded += 1;
                    return Some((self.current_id, Some(sub_id)));
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
}
