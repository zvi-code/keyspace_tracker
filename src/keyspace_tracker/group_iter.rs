//! Group-level iterators.
//!
//! Provides iterators that span multiple `PrefixTracker`s within a `PrefixGroupsTracker`:
//! - `GroupIter`: Read iteration with horizontal/vertical/random ordering
//! - `GroupWriteIter`: Concurrent write with policy-based prefix selection
//! - `GroupDeleteIter`: Exclusive delete across multiple prefixes
//! - `GroupPartitionedIter`: Parallel iteration across prefixes

use std::sync::Arc;

use super::config::{BitFilter, ClaimPolicy, IdRange, IterOrder, PrefixFilter, PrefixWeight};
use super::tracker::PrefixTracker;

/// Item yielded by group iterators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupItem {
    /// The prefix string.
    pub prefix: Arc<str>,
    /// The primary ID.
    pub id: u64,
    /// The sub_id (None for simple trackers).
    pub sub_id: Option<u64>,
}

/// Builder for group-level iteration.
pub struct GroupIterBuilder<'a> {
    trackers: Vec<Arc<PrefixTracker>>,
    prefix_filter: PrefixFilter,
    range: IdRange,
    bit_filter: BitFilter,
    order: IterOrder,
    weight: PrefixWeight,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> GroupIterBuilder<'a> {
    /// Create a new group iterator builder.
    pub(crate) fn new(trackers: Vec<Arc<PrefixTracker>>) -> Self {
        Self {
            trackers,
            prefix_filter: PrefixFilter::All,
            range: IdRange::unbounded(),
            bit_filter: BitFilter::All,
            order: IterOrder::Horizontal,
            weight: PrefixWeight::Uniform,
            _marker: std::marker::PhantomData,
        }
    }

    /// Include all registered prefixes.
    #[inline]
    pub fn all_prefixes(mut self) -> Self {
        self.prefix_filter = PrefixFilter::All;
        self
    }

    /// Include only the specified prefixes.
    pub fn prefixes(mut self, prefixes: &[&str]) -> Self {
        self.prefix_filter = PrefixFilter::Exact(prefixes.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Include prefixes matching the predicate.
    pub fn filter_prefixes<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.prefix_filter = PrefixFilter::Predicate(Arc::new(f));
        self
    }

    /// Set ID and sub_id ranges.
    #[inline]
    pub fn range(mut self, id_range: (u64, u64), sub_id_range: (u64, u64)) -> Self {
        self.range = IdRange::new(id_range, sub_id_range);
        self
    }

    /// Filter to only iterate over set (existing) IDs.
    #[inline]
    pub fn set_only(mut self) -> Self {
        self.bit_filter = BitFilter::Set;
        self
    }

    /// Filter to only iterate over unset (non-existing) IDs.
    #[inline]
    pub fn unset_only(mut self) -> Self {
        self.bit_filter = BitFilter::Unset;
        self
    }

    /// Set horizontal iteration order (all IDs for prefix₀, then prefix₁, ...).
    #[inline]
    pub fn horizontal(mut self) -> Self {
        self.order = IterOrder::Horizontal;
        self
    }

    /// Set vertical iteration order (ID₀ across all prefixes, then ID₁, ...).
    #[inline]
    pub fn vertical(mut self) -> Self {
        self.order = IterOrder::Vertical;
        self
    }

    /// Set random iteration order.
    #[inline]
    pub fn random(mut self) -> Self {
        self.order = IterOrder::Random;
        self
    }

    /// Set weighting strategy for random/vertical iteration.
    #[inline]
    pub fn weighted(mut self, weight: PrefixWeight) -> Self {
        self.weight = weight;
        self
    }

    /// Get filtered trackers based on prefix filter.
    fn filtered_trackers(&self) -> Vec<Arc<PrefixTracker>> {
        self.trackers
            .iter()
            .filter(|t| match &self.prefix_filter {
                PrefixFilter::All => true,
                PrefixFilter::Exact(prefixes) => prefixes.contains(&t.prefix().to_string()),
                PrefixFilter::Predicate(f) => f(t.prefix()),
            })
            .cloned()
            .collect()
    }

    /// Build a read iterator.
    pub fn build(self) -> GroupIter {
        let trackers = self.filtered_trackers();

        GroupIter {
            trackers,
            range: self.range,
            bit_filter: self.bit_filter,
            order: self.order,
            weight: self.weight,
            rng: fastrand::Rng::new(),
            prefix_index: 0,
            current_id: self.range.id_min,
            current_sub_id: self.range.sub_id_min,
            exhausted: false,
        }
    }

    /// Build a write iterator with the specified claim policy.
    pub fn write(self, policy: ClaimPolicy) -> GroupWriteIter {
        let trackers = self.filtered_trackers();
        GroupWriteIter::new(trackers, self.range, policy)
    }

    /// Build an exclusive delete iterator with the specified claim policy.
    ///
    /// Returns `None` if any of the trackers already has an active delete iterator.
    pub fn delete(self, _policy: ClaimPolicy) -> Option<GroupDeleteIter> {
        let trackers = self.filtered_trackers();

        // Try to acquire delete locks on all trackers
        let mut acquired = Vec::new();
        for t in &trackers {
            if t.try_acquire_delete_lock() {
                acquired.push(t.clone());
            } else {
                // Release already acquired locks
                for acq in acquired {
                    acq.release_delete_lock();
                }
                return None;
            }
        }

        Some(GroupDeleteIter {
            trackers,
            range: self.range,
            prefix_index: 0,
            current_id: self.range.id_min,
            current_sub_id: self.range.sub_id_min,
        })
    }

    /// Build partitioned iterators for parallel processing.
    pub fn partitioned(self, n: usize) -> Vec<GroupPartitionedIter> {
        let trackers = self.filtered_trackers();

        // Partition by ID range, each partition gets all prefixes
        let total = self.range.id_max.saturating_sub(self.range.id_min);
        let chunk_size = (total + n as u64 - 1) / n as u64;

        (0..n)
            .map(|i| {
                let start = self.range.id_min + i as u64 * chunk_size;
                let end = (start + chunk_size).min(self.range.id_max);

                GroupPartitionedIter {
                    trackers: trackers.clone(),
                    range: IdRange {
                        id_min: start,
                        id_max: end,
                        sub_id_min: self.range.sub_id_min,
                        sub_id_max: self.range.sub_id_max,
                    },
                    bit_filter: self.bit_filter,
                    prefix_index: 0,
                    current_id: start,
                    current_sub_id: self.range.sub_id_min,
                }
            })
            .collect()
    }
}

// ============================================================================
// Group Iterator
// ============================================================================

/// Read iterator over multiple prefixes.
pub struct GroupIter {
    trackers: Vec<Arc<PrefixTracker>>,
    range: IdRange,
    bit_filter: BitFilter,
    order: IterOrder,
    weight: PrefixWeight,
    rng: fastrand::Rng,
    prefix_index: usize,
    current_id: u64,
    current_sub_id: u64,
    exhausted: bool,
}

impl GroupIter {
    fn next_horizontal(&mut self) -> Option<GroupItem> {
        while self.prefix_index < self.trackers.len() {
            let tracker = &self.trackers[self.prefix_index];
            let max_id = self.range.id_max.min(tracker.effective_max_id());

            while self.current_id < max_id {
                let id = self.current_id;

                if tracker.is_hierarchical() {
                    // Use per-id max based on sub_bitmap capacity
                    let max_sub_id = self.range.sub_id_max.min(tracker.effective_max_sub_id_for(id));

                    while self.current_sub_id < max_sub_id {
                        let sub_id = self.current_sub_id;
                        self.current_sub_id += 1;

                        if !self.range.contains(id, sub_id) {
                            continue;
                        }

                        let is_set = tracker.exists_pair(id, sub_id);
                        if self.matches_filter(is_set) {
                            return Some(GroupItem {
                                prefix: tracker.prefix().into(),
                                id,
                                sub_id: Some(sub_id),
                            });
                        }
                    }

                    self.current_id += 1;
                    self.current_sub_id = self.range.sub_id_min;
                } else {
                    self.current_id += 1;

                    if !self.range.contains_id(id) {
                        continue;
                    }

                    let is_set = tracker.exists(id);
                    if self.matches_filter(is_set) {
                        return Some(GroupItem {
                            prefix: tracker.prefix().into(),
                            id,
                            sub_id: None,
                        });
                    }
                }
            }

            // Move to next prefix
            self.prefix_index += 1;
            self.current_id = self.range.id_min;
            self.current_sub_id = self.range.sub_id_min;
        }

        None
    }

    fn next_vertical(&mut self) -> Option<GroupItem> {
        if self.trackers.is_empty() {
            return None;
        }

        // Find max ID across all trackers
        let max_id = self
            .trackers
            .iter()
            .map(|t| self.range.id_max.min(t.effective_max_id()))
            .max()
            .unwrap_or(0);

        while self.current_id < max_id {
            while self.prefix_index < self.trackers.len() {
                let tracker = &self.trackers[self.prefix_index];
                self.prefix_index += 1;

                let tracker_max_id = self.range.id_max.min(tracker.effective_max_id());
                if self.current_id >= tracker_max_id {
                    continue;
                }

                if tracker.is_hierarchical() {
                    // For hierarchical in vertical mode, we return one sub_id at a time
                    let is_set = tracker.exists_pair(self.current_id, self.current_sub_id);
                    if self.matches_filter(is_set) {
                        return Some(GroupItem {
                            prefix: tracker.prefix().into(),
                            id: self.current_id,
                            sub_id: Some(self.current_sub_id),
                        });
                    }
                } else {
                    let is_set = tracker.exists(self.current_id);
                    if self.matches_filter(is_set) {
                        return Some(GroupItem {
                            prefix: tracker.prefix().into(),
                            id: self.current_id,
                            sub_id: None,
                        });
                    }
                }
            }

            // Move to next ID
            self.prefix_index = 0;
            self.current_id += 1;
        }

        None
    }

    fn next_random(&mut self) -> Option<GroupItem> {
        if self.trackers.is_empty() || self.exhausted {
            return None;
        }

        // Cache range values to avoid borrow conflicts
        let id_min = self.range.id_min;
        let id_max = self.range.id_max;
        let sub_id_min = self.range.sub_id_min;
        let sub_id_max = self.range.sub_id_max;

        // Simple random: pick random prefix, then random ID
        let max_attempts = 1000;
        for _ in 0..max_attempts {
            let tracker_idx = self.select_weighted_tracker_idx()?;
            let tracker = &self.trackers[tracker_idx];

            let max_id = id_max.min(tracker.effective_max_id());
            if id_min >= max_id {
                continue;
            }

            let id = self.rng.u64(id_min..max_id);

            if tracker.is_hierarchical() {
                // Use per-id max based on sub_bitmap capacity
                let max_sub_id = sub_id_max.min(tracker.effective_max_sub_id_for(id));
                if sub_id_min >= max_sub_id {
                    continue;
                }

                let sub_id = self.rng.u64(sub_id_min..max_sub_id);
                let is_set = tracker.exists_pair(id, sub_id);

                if self.matches_filter(is_set) {
                    return Some(GroupItem {
                        prefix: tracker.prefix().into(),
                        id,
                        sub_id: Some(sub_id),
                    });
                }
            } else {
                let is_set = tracker.exists(id);
                if self.matches_filter(is_set) {
                    return Some(GroupItem {
                        prefix: tracker.prefix().into(),
                        id,
                        sub_id: None,
                    });
                }
            }
        }

        self.exhausted = true;
        None
    }

    fn select_weighted_tracker_idx(&mut self) -> Option<usize> {
        if self.trackers.is_empty() {
            return None;
        }

        match self.weight {
            PrefixWeight::Uniform => {
                let idx = self.rng.usize(..self.trackers.len());
                Some(idx)
            }
            PrefixWeight::ByCapacity => {
                let weights: Vec<u64> = self.trackers.iter().map(|t| t.capacity() as u64).collect();
                self.weighted_select_idx(&weights)
            }
            PrefixWeight::ByCount => {
                let weights: Vec<u64> = self.trackers.iter().map(|t| t.count()).collect();
                self.weighted_select_idx(&weights)
            }
            PrefixWeight::ByUnsetCount => {
                let weights: Vec<u64> = self
                    .trackers
                    .iter()
                    .map(|t| (t.capacity() as u64).saturating_sub(t.count()))
                    .collect();
                self.weighted_select_idx(&weights)
            }
        }
    }

    fn weighted_select_idx(&mut self, weights: &[u64]) -> Option<usize> {
        let total: u64 = weights.iter().sum();
        if total == 0 {
            // Fallback to uniform
            let idx = self.rng.usize(..self.trackers.len());
            return Some(idx);
        }

        let mut pick = self.rng.u64(..total);
        for (i, &w) in weights.iter().enumerate() {
            if pick < w {
                return Some(i);
            }
            pick -= w;
        }

        Some(self.trackers.len() - 1)
    }

    #[inline]
    fn matches_filter(&self, is_set: bool) -> bool {
        match self.bit_filter {
            BitFilter::All => true,
            BitFilter::Set => is_set,
            BitFilter::Unset => !is_set,
        }
    }
}

impl Iterator for GroupIter {
    type Item = GroupItem;

    fn next(&mut self) -> Option<Self::Item> {
        match self.order {
            IterOrder::Horizontal => self.next_horizontal(),
            IterOrder::Vertical => self.next_vertical(),
            IterOrder::Random => self.next_random(),
        }
    }
}

// ============================================================================
// Group Write Iterator
// ============================================================================

/// Write iterator with policy-based prefix selection.
pub struct GroupWriteIter {
    trackers: Vec<Arc<PrefixTracker>>,
    range: IdRange,
    policy: ClaimPolicy,
    rng: fastrand::Rng,
    round_robin_index: usize,
    /// Per-tracker cursor positions for efficient scanning
    cursors: Vec<u64>,
    /// Tracks which trackers are exhausted (cursor >= max_id)
    exhausted: Vec<bool>,
}

impl GroupWriteIter {
    /// Create new GroupWriteIter with per-tracker cursors
    pub(crate) fn new(
        trackers: Vec<Arc<PrefixTracker>>,
        range: IdRange,
        policy: ClaimPolicy,
    ) -> Self {
        let n = trackers.len();
        Self {
            trackers,
            range,
            policy,
            rng: fastrand::Rng::new(),
            round_robin_index: 0,
            cursors: vec![range.id_min; n],
            exhausted: vec![false; n],
        }
    }
    
    /// Atomically claim next (prefix, id, sub_id) and mark as set.
    pub fn next(&mut self) -> Option<GroupItem> {
        if self.trackers.is_empty() {
            return None;
        }

        // Try up to trackers.len() times to find a non-exhausted tracker
        for _ in 0..self.trackers.len() {
            let tracker_idx = self.select_tracker_index()?;

            if let Some(item) = self.try_claim_at(tracker_idx) {
                return Some(item);
            }
            // try_claim_at marks tracker as exhausted if no more unset bits
        }

        None
    }

    fn select_tracker_index(&mut self) -> Option<usize> {
        if self.trackers.is_empty() {
            return None;
        }

        // Check if all exhausted
        if self.exhausted.iter().all(|&e| e) {
            return None;
        }

        match self.policy {
            ClaimPolicy::RoundRobin => {
                let start = self.round_robin_index;
                for i in 0..self.trackers.len() {
                    let idx = (start + i) % self.trackers.len();
                    if !self.exhausted[idx] {
                        self.round_robin_index = (idx + 1) % self.trackers.len();
                        return Some(idx);
                    }
                }
                None
            }
            ClaimPolicy::Random => {
                // Collect non-exhausted indices
                let available: Vec<usize> = self.exhausted
                    .iter()
                    .enumerate()
                    .filter(|(_, &e)| !e)
                    .map(|(i, _)| i)
                    .collect();
                
                if available.is_empty() {
                    return None;
                }
                Some(available[self.rng.usize(..available.len())])
            }
            ClaimPolicy::WeightedRandom(weight) => {
                let weights: Vec<u64> = self
                    .trackers
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        if self.exhausted[i] {
                            0
                        } else {
                            match weight {
                                PrefixWeight::Uniform => 1,
                                PrefixWeight::ByCapacity => t.capacity() as u64,
                                PrefixWeight::ByCount => t.count(),
                                PrefixWeight::ByUnsetCount => {
                                    (t.capacity() as u64).saturating_sub(t.count())
                                }
                            }
                        }
                    })
                    .collect();

                let total: u64 = weights.iter().sum();
                if total == 0 {
                    return None;
                }

                let mut pick = self.rng.u64(..total);
                for (i, &w) in weights.iter().enumerate() {
                    if pick < w {
                        return Some(i);
                    }
                    pick -= w;
                }

                Some(self.trackers.len() - 1)
            }
            ClaimPolicy::LeastLoaded => self
                .trackers
                .iter()
                .enumerate()
                .filter(|(i, _)| !self.exhausted[*i])
                .max_by_key(|(_, t)| (t.capacity() as u64).saturating_sub(t.count()))
                .map(|(i, _)| i),
            ClaimPolicy::MostLoaded => self
                .trackers
                .iter()
                .enumerate()
                .filter(|(i, _)| !self.exhausted[*i])
                .max_by_key(|(_, t)| t.count())
                .map(|(i, _)| i),
        }
    }

    fn try_claim_at(&mut self, tracker_idx: usize) -> Option<GroupItem> {
        let tracker = &self.trackers[tracker_idx];
        let max_id = self.range.id_max.min(tracker.effective_max_id());
        let prefix: Arc<str> = tracker.prefix().into();
        let is_hierarchical = tracker.is_hierarchical();

        if is_hierarchical {
            // For hierarchical, use cursor as combined (id * max_sub_id + sub_id)
            let max_sub_id = self.range.sub_id_max.min(tracker.effective_max_sub_id()).max(1);
            let cursor = self.cursors[tracker_idx];
            
            let mut id = cursor / max_sub_id;
            let mut sub_id = cursor % max_sub_id;
            
            while id < max_id {
                while sub_id < max_sub_id {
                    if tracker.claim_pair(id, sub_id) {
                        // Update cursor past this position
                        self.cursors[tracker_idx] = id * max_sub_id + sub_id + 1;
                        return Some(GroupItem {
                            prefix,
                            id,
                            sub_id: Some(sub_id),
                        });
                    }
                    sub_id += 1;
                }
                id += 1;
                sub_id = self.range.sub_id_min;
            }
            
            // Exhausted
            self.exhausted[tracker_idx] = true;
            self.cursors[tracker_idx] = max_id * max_sub_id;
        } else {
            // Simple mode: use find_next_unset for O(density) instead of O(n)
            let bitmap = tracker.primary_bitmap();
            let cursor = self.cursors[tracker_idx];
            
            if cursor >= max_id {
                self.exhausted[tracker_idx] = true;
                return None;
            }
            
            // Find next unset bit from cursor
            if let Some(id) = bitmap.find_next_unset(cursor as usize, max_id as usize) {
                let id = id as u64;
                
                // Try to claim it atomically
                if bitmap.test_and_set(id as usize) {
                    // Update cursor past this bit
                    self.cursors[tracker_idx] = id + 1;
                    return Some(GroupItem {
                        prefix,
                        id,
                        sub_id: None,
                    });
                } else {
                    // Someone else claimed it, advance cursor and let next() retry
                    self.cursors[tracker_idx] = id + 1;
                    return None;
                }
            } else {
                // No more unset bits
                self.exhausted[tracker_idx] = true;
                self.cursors[tracker_idx] = max_id;
            }
        }

        None
    }
}

// ============================================================================
// Group Delete Iterator
// ============================================================================

/// Exclusive delete iterator over multiple prefixes.
pub struct GroupDeleteIter {
    trackers: Vec<Arc<PrefixTracker>>,
    range: IdRange,
    prefix_index: usize,
    current_id: u64,
    current_sub_id: u64,
}

impl GroupDeleteIter {
    /// Atomically claim next set ID and clear it.
    pub fn next(&mut self) -> Option<GroupItem> {
        if self.trackers.is_empty() {
            return None;
        }

        // For delete, we use sequential scanning (simpler than write)
        while self.prefix_index < self.trackers.len() {
            let tracker = &self.trackers[self.prefix_index];
            let max_id = self.range.id_max.min(tracker.effective_max_id());

            while self.current_id < max_id {
                if tracker.is_hierarchical() {
                    // Use per-id max based on sub_bitmap capacity
                    let max_sub_id = self.range.sub_id_max.min(tracker.effective_max_sub_id_for(self.current_id));

                    while self.current_sub_id < max_sub_id {
                        let id = self.current_id;
                        let sub_id = self.current_sub_id;
                        self.current_sub_id += 1;

                        if tracker.remove_pair(id, sub_id) {
                            return Some(GroupItem {
                                prefix: tracker.prefix().into(),
                                id,
                                sub_id: Some(sub_id),
                            });
                        }
                    }

                    self.current_id += 1;
                    self.current_sub_id = self.range.sub_id_min;
                } else {
                    let id = self.current_id;
                    self.current_id += 1;

                    if tracker.remove(id) {
                        return Some(GroupItem {
                            prefix: tracker.prefix().into(),
                            id,
                            sub_id: None,
                        });
                    }
                }
            }

            self.prefix_index += 1;
            self.current_id = self.range.id_min;
            self.current_sub_id = self.range.sub_id_min;
        }

        None
    }
}

impl Drop for GroupDeleteIter {
    fn drop(&mut self) {
        for t in &self.trackers {
            t.release_delete_lock();
        }
    }
}

// ============================================================================
// Group Partitioned Iterator
// ============================================================================

/// Partitioned iterator for parallel processing across prefixes.
pub struct GroupPartitionedIter {
    trackers: Vec<Arc<PrefixTracker>>,
    range: IdRange,
    bit_filter: BitFilter,
    prefix_index: usize,
    current_id: u64,
    current_sub_id: u64,
}

impl GroupPartitionedIter {
    /// Get the ID range covered by this partition.
    pub fn range(&self) -> &IdRange {
        &self.range
    }
}

impl Iterator for GroupPartitionedIter {
    type Item = GroupItem;

    fn next(&mut self) -> Option<Self::Item> {
        while self.prefix_index < self.trackers.len() {
            let tracker = &self.trackers[self.prefix_index];
            let max_id = self.range.id_max.min(tracker.effective_max_id());

            while self.current_id < max_id {
                if tracker.is_hierarchical() {
                    // Use per-id max based on sub_bitmap capacity
                    let max_sub_id = self.range.sub_id_max.min(tracker.effective_max_sub_id_for(self.current_id));

                    while self.current_sub_id < max_sub_id {
                        let id = self.current_id;
                        let sub_id = self.current_sub_id;
                        self.current_sub_id += 1;

                        let is_set = tracker.exists_pair(id, sub_id);
                        let matches = match self.bit_filter {
                            BitFilter::All => true,
                            BitFilter::Set => is_set,
                            BitFilter::Unset => !is_set,
                        };

                        if matches {
                            return Some(GroupItem {
                                prefix: tracker.prefix().into(),
                                id,
                                sub_id: Some(sub_id),
                            });
                        }
                    }

                    self.current_id += 1;
                    self.current_sub_id = self.range.sub_id_min;
                } else {
                    let id = self.current_id;
                    self.current_id += 1;

                    let is_set = tracker.exists(id);
                    let matches = match self.bit_filter {
                        BitFilter::All => true,
                        BitFilter::Set => is_set,
                        BitFilter::Unset => !is_set,
                    };

                    if matches {
                        return Some(GroupItem {
                            prefix: tracker.prefix().into(),
                            id,
                            sub_id: None,
                        });
                    }
                }
            }

            self.prefix_index += 1;
            self.current_id = self.range.id_min;
            self.current_sub_id = self.range.sub_id_min;
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyspace_tracker::TrackerConfig;

    fn create_test_trackers() -> Vec<Arc<PrefixTracker>> {
        let t1 = Arc::new(PrefixTracker::new(
            TrackerConfig::simple("vec:").with_max_id(100),
        ));
        let t2 = Arc::new(PrefixTracker::new(
            TrackerConfig::simple("doc:").with_max_id(100),
        ));

        // Add some data
        for i in 0..10 {
            t1.add(i);
        }
        for i in 0..5 {
            t2.add(i);
        }

        vec![t1, t2]
    }

    #[test]
    fn test_group_iter_horizontal() {
        let trackers = create_test_trackers();

        let items: Vec<_> = GroupIterBuilder::new(trackers)
            .set_only()
            .horizontal()
            .build()
            .collect();

        // Should have all 15 items, first 10 from vec:, then 5 from doc:
        assert_eq!(items.len(), 15);

        // First 10 should be vec:
        for i in 0..10 {
            assert_eq!(items[i].prefix.as_ref(), "vec:");
            assert_eq!(items[i].id, i as u64);
        }

        // Next 5 should be doc:
        for i in 0..5 {
            assert_eq!(items[10 + i].prefix.as_ref(), "doc:");
            assert_eq!(items[10 + i].id, i as u64);
        }
    }

    #[test]
    fn test_group_iter_vertical() {
        let trackers = create_test_trackers();

        let items: Vec<_> = GroupIterBuilder::new(trackers)
            .set_only()
            .vertical()
            .build()
            .collect();

        // Should have items interleaved by ID
        assert_eq!(items.len(), 15);

        // IDs 0-4 should appear for both prefixes
        let id0_items: Vec<_> = items.iter().filter(|i| i.id == 0).collect();
        assert_eq!(id0_items.len(), 2);
    }

    #[test]
    fn test_group_iter_random() {
        let trackers = create_test_trackers();

        let items: Vec<_> = GroupIterBuilder::new(trackers)
            .set_only()
            .random()
            .build()
            .take(100)
            .collect();

        // Should get some items (random may not cover all)
        assert!(!items.is_empty());

        // All items should be from valid prefixes
        for item in &items {
            assert!(item.prefix.as_ref() == "vec:" || item.prefix.as_ref() == "doc:");
        }
    }

    #[test]
    fn test_group_iter_prefix_filter() {
        let trackers = create_test_trackers();

        let items: Vec<_> = GroupIterBuilder::new(trackers)
            .prefixes(&["vec:"])
            .set_only()
            .horizontal()
            .build()
            .collect();

        // Should only have vec: items
        assert_eq!(items.len(), 10);
        for item in &items {
            assert_eq!(item.prefix.as_ref(), "vec:");
        }
    }

    #[test]
    fn test_group_write_round_robin() {
        let t1 = Arc::new(PrefixTracker::new(
            TrackerConfig::simple("a:").with_max_id(10),
        ));
        let t2 = Arc::new(PrefixTracker::new(
            TrackerConfig::simple("b:").with_max_id(10),
        ));
        let trackers = vec![t1.clone(), t2.clone()];

        let mut iter = GroupIterBuilder::new(trackers).write(ClaimPolicy::RoundRobin);

        let mut claimed = Vec::new();
        while let Some(item) = iter.next() {
            claimed.push(item);
            if claimed.len() >= 6 {
                break;
            }
        }

        // Should have items from both prefixes
        let a_count = claimed.iter().filter(|i| i.prefix.as_ref() == "a:").count();
        let b_count = claimed.iter().filter(|i| i.prefix.as_ref() == "b:").count();

        assert!(a_count > 0);
        assert!(b_count > 0);
    }

    #[test]
    fn test_group_delete() {
        let trackers = create_test_trackers();
        let t1 = trackers[0].clone();
        let t2 = trackers[1].clone();

        let initial_count = t1.count() + t2.count();

        let mut deleted = Vec::new();
        if let Some(mut iter) = GroupIterBuilder::new(trackers).delete(ClaimPolicy::RoundRobin) {
            while let Some(item) = iter.next() {
                deleted.push(item);
            }
        }

        assert_eq!(deleted.len(), initial_count as usize);
        assert_eq!(t1.count(), 0);
        assert_eq!(t2.count(), 0);
    }

    #[test]
    fn test_group_partitioned() {
        let trackers = create_test_trackers();

        let partitions = GroupIterBuilder::new(trackers)
            .range((0, 100), (0, 100))
            .set_only()
            .partitioned(4);

        assert_eq!(partitions.len(), 4);

        // Collect all items from all partitions
        let all_items: Vec<_> = partitions.into_iter().flatten().collect();

        // Should have all 15 items
        assert_eq!(all_items.len(), 15);
    }
}
