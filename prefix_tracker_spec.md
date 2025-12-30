# PrefixTracker Feature Specification

## Overview

Implement a high-performance, memory-efficient bitmap-based existence tracker for key-value systems. The system tracks which IDs exist for given prefixes, supporting billions of IDs with nanosecond to single-digit microsecond latency.

**Primary Use Case**: Benchmark worker threads iterating over key spaces to execute commands (GET/SET/HSET/QUERY/DELETE) against a key-value store, where keys follow the format `prefix{cluster_tag}:zero_padded_id`.

## Architecture

```
src/prefix_tracker/
├── mod.rs           # PrefixGroupsTracker + module re-exports
├── bitmap.rs        # AtomicBitmap - low-level atomic bitmap implementation
├── tracker.rs       # PrefixTracker - wraps bitmap with metadata
└── iterators.rs     # All iterator types (read, write, delete, partitioned)
```

## Module 1: `bitmap.rs` - AtomicBitmap

### Purpose
Low-level atomic bitmap supporting lock-free concurrent bit operations. Foundation for all higher-level tracking.

### Struct Definition

```rust
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Cache-line aligned atomic bitmap for concurrent bit operations.
/// Uses AtomicU64 words for efficient SIMD-friendly memory layout.
pub struct AtomicBitmap {
    /// Bitmap storage: Vec<AtomicU64>, each word holds 64 bits.
    /// Capacity in bits = words.len() * 64.
    /// Use UnsafeCell or raw pointer for interior mutability during growth.
    words: Box<[AtomicU64]>,
    
    /// Current capacity in bits (always multiple of 64).
    capacity: AtomicUsize,
    
    /// Population count (number of set bits). Updated atomically on set/clear.
    /// May have momentary inconsistency but eventually consistent.
    popcount: AtomicU64,
}
```

### Constants

```rust
const BITS_PER_WORD: usize = 64;
const INITIAL_CAPACITY_BITS: usize = 4096;  // 4K bits = 64 words = 512 bytes
const WORD_SHIFT: usize = 6;  // log2(64) for fast division
const WORD_MASK: usize = 63;  // For fast modulo
```

### Required Methods

#### Construction & Capacity

```rust
impl AtomicBitmap {
    /// Create bitmap with initial capacity (rounded up to next multiple of 64).
    /// Allocates zeroed memory.
    pub fn new() -> Self;
    
    /// Create with specific initial capacity in bits.
    pub fn with_capacity(capacity_bits: usize) -> Self;
    
    /// Current capacity in bits.
    #[inline]
    pub fn capacity(&self) -> usize;
    
    /// Ensure capacity for given bit index. Grows if necessary.
    /// Growth strategy: max(current * 2, required_capacity rounded to 64).
    /// Thread-safe but may allocate. Returns true if growth occurred.
    pub fn ensure_capacity(&self, bit_index: usize) -> bool;
}
```

#### Bit Operations (All O(1), Lock-Free)

```rust
impl AtomicBitmap {
    /// Set bit at index. Returns previous value (false if was unset).
    /// Auto-grows if index >= capacity.
    /// Atomically increments popcount if bit changed 0→1.
    #[inline]
    pub fn set(&self, index: usize) -> bool;
    
    /// Clear bit at index. Returns previous value (true if was set).
    /// No-op if index >= capacity.
    /// Atomically decrements popcount if bit changed 1→0.
    #[inline]
    pub fn clear(&self, index: usize) -> bool;
    
    /// Test bit at index. Returns false if index >= capacity.
    #[inline]
    pub fn test(&self, index: usize) -> bool;
    
    /// Atomically test-and-set: if bit is 0, set to 1 and return true.
    /// If bit is already 1, return false. Used for claim operations.
    /// Auto-grows if index >= capacity.
    #[inline]
    pub fn test_and_set(&self, index: usize) -> bool;
    
    /// Atomically test-and-clear: if bit is 1, set to 0 and return true.
    /// If bit is already 0, return false.
    #[inline]
    pub fn test_and_clear(&self, index: usize) -> bool;
    
    /// Get current population count (number of set bits).
    #[inline]
    pub fn count(&self) -> u64;
    
    /// Check if bitmap is empty (no bits set).
    #[inline]
    pub fn is_empty(&self) -> bool;
}
```

#### Bulk Operations

```rust
impl AtomicBitmap {
    /// Clear all bits and reset popcount to 0.
    /// Deallocates storage and resets to initial capacity.
    /// NOT thread-safe with concurrent bit operations.
    pub fn clear_all(&mut self);
    
    /// Clear all bits but retain allocated capacity.
    /// Faster than clear_all() when capacity will be reused.
    pub fn reset(&mut self);
}
```

### Implementation Notes

1. **Memory Ordering**:
   - Use `Ordering::Relaxed` for popcount updates (eventual consistency acceptable)
   - Use `Ordering::AcqRel` for CAS operations in test_and_set/test_and_clear
   - Use `Ordering::Acquire` for reads that guard subsequent operations
   - Use `Ordering::Release` for writes that publish data

2. **Growth Strategy**:
   - Growth requires synchronization. Use `RwLock` or similar for resize.
   - Copy-on-grow: allocate new storage, copy old data, atomic swap pointer.
   - Consider using `crossbeam-epoch` or `arc-swap` for lock-free growth if performance critical.
   - Simple approach: `Mutex<()>` for growth only, all other ops lock-free.

3. **Bit Manipulation Patterns**:
   ```rust
   // Word and bit index from global bit index
   let word_idx = index >> WORD_SHIFT;  // index / 64
   let bit_idx = index & WORD_MASK;     // index % 64
   let mask = 1u64 << bit_idx;
   
   // Atomic set with popcount update
   let word = &self.words[word_idx];
   let old = word.fetch_or(mask, Ordering::AcqRel);
   if (old & mask) == 0 {
       self.popcount.fetch_add(1, Ordering::Relaxed);
   }
   ```

4. **Find First Set/Unset in Word**:
   ```rust
   // Find first set bit in word (0-63), or 64 if none
   fn find_first_set(word: u64) -> u32 {
       if word == 0 { 64 } else { word.trailing_zeros() }
   }
   
   // Find first unset bit in word (0-63), or 64 if all set
   fn find_first_unset(word: u64) -> u32 {
       find_first_set(!word)
   }
   ```

---

## Module 2: `tracker.rs` - PrefixTracker

### Purpose
Wraps `AtomicBitmap` with prefix metadata and provides high-level ID tracking interface.

### Struct Definition

```rust
use std::sync::Arc;
use super::bitmap::AtomicBitmap;

/// Tracks existence of IDs for a specific prefix.
/// Thread-safe for all operations.
pub struct PrefixTracker {
    /// The prefix string (e.g., "vec:", "doc:")
    prefix: String,
    
    /// Underlying bitmap storage
    bitmap: AtomicBitmap,
    
    /// Optional maximum ID (exclusive). None = unbounded (limited by memory).
    max_id: Option<u64>,
}
```

### Required Methods

```rust
impl PrefixTracker {
    /// Create tracker for given prefix.
    pub fn new(prefix: impl Into<String>) -> Self;
    
    /// Create with maximum ID bound.
    pub fn with_max_id(prefix: impl Into<String>, max_id: u64) -> Self;
    
    /// Get the prefix string.
    #[inline]
    pub fn prefix(&self) -> &str;
    
    /// Get maximum ID if set.
    #[inline]
    pub fn max_id(&self) -> Option<u64>;
    
    /// Add ID to tracker (mark as existing). Returns true if newly added.
    #[inline]
    pub fn add(&self, id: u64) -> bool;
    
    /// Remove ID from tracker. Returns true if was present.
    #[inline]
    pub fn remove(&self, id: u64) -> bool;
    
    /// Check if ID exists.
    #[inline]
    pub fn exists(&self, id: u64) -> bool;
    
    /// Atomically claim ID: mark as existing only if not already.
    /// Returns true if successfully claimed (was not present).
    #[inline]
    pub fn claim(&self, id: u64) -> bool;
    
    /// Get count of existing IDs.
    #[inline]
    pub fn count(&self) -> u64;
    
    /// Check if tracker is empty.
    #[inline]
    pub fn is_empty(&self) -> bool;
    
    /// Get current capacity.
    #[inline]
    pub fn capacity(&self) -> usize;
    
    /// Clear all tracked IDs. Resets to initial state.
    /// NOT thread-safe with concurrent operations.
    pub fn clear(&mut self);
    
    /// Ensure capacity for given ID.
    pub fn ensure_capacity(&self, id: u64);
}
```

### Iterator Factory Methods

```rust
impl PrefixTracker {
    /// Create iterator configuration builder.
    pub fn iter(&self) -> IteratorBuilder<'_>;
}

/// Builder for configuring iterators.
pub struct IteratorBuilder<'a> {
    tracker: &'a PrefixTracker,
    min_id: u64,
    max_id: u64,
    filter: BitFilter,
}

#[derive(Clone, Copy, Default)]
pub enum BitFilter {
    #[default]
    All,        // Iterate all IDs in range
    Set,        // Only existing IDs
    Unset,      // Only non-existing IDs
}

impl<'a> IteratorBuilder<'a> {
    /// Set minimum ID (inclusive). Default: 0.
    pub fn min(self, min_id: u64) -> Self;
    
    /// Set maximum ID (exclusive). Default: tracker.max_id or capacity.
    pub fn max(self, max_id: u64) -> Self;
    
    /// Set ID range [min, max).
    pub fn range(self, min_id: u64, max_id: u64) -> Self;
    
    /// Filter to only set bits (existing IDs).
    pub fn set_only(self) -> Self;
    
    /// Filter to only unset bits (non-existing IDs).
    pub fn unset_only(self) -> Self;
    
    /// Build sequential read iterator.
    pub fn sequential(self) -> SequentialIter<'a>;
    
    /// Build random-access read iterator (shuffled order).
    pub fn random(self) -> RandomIter<'a>;
    
    /// Build write iterator (claim-and-set semantics).
    /// Multiple writers can run concurrently.
    pub fn write(self) -> WriteIter<'a>;
    
    /// Build exclusive delete iterator.
    /// Only one delete iterator can exist at a time.
    /// Returns None if another delete iterator is active.
    pub fn delete(self) -> Option<DeleteIter<'a>>;
    
    /// Build partitioned iterators for parallel processing.
    /// Returns `n` iterators with disjoint ranges covering the full space.
    pub fn partitioned(self, n: usize) -> Vec<PartitionedIter<'a>>;
}
```

---

## Module 3: `iterators.rs` - Iterator Types

### Common Traits

```rust
/// Base trait for all prefix tracker iterators.
pub trait TrackerIterator {
    /// Get next ID, or None if exhausted.
    fn next_id(&mut self) -> Option<u64>;
    
    /// Get approximate remaining count (may be inaccurate for filtered iterators).
    fn remaining(&self) -> usize;
    
    /// Check if iterator is exhausted.
    fn is_exhausted(&self) -> bool;
}
```

### Iterator Type 1: SequentialIter (Read)

```rust
/// Sequential iterator over IDs in ascending order.
/// Thread-safe: multiple instances can run concurrently.
/// Does NOT provide unique IDs across instances.
pub struct SequentialIter<'a> {
    tracker: &'a PrefixTracker,
    current: u64,
    max_id: u64,
    filter: BitFilter,
}

impl<'a> SequentialIter<'a> {
    /// Standard Iterator impl for convenience.
    /// Note: implements Iterator<Item = u64>
}

impl Iterator for SequentialIter<'_> {
    type Item = u64;
    fn next(&mut self) -> Option<u64>;
}
```

**Implementation**:
- For `BitFilter::All`: return `current++` until `max_id`
- For `BitFilter::Set`: scan forward to find next set bit
- For `BitFilter::Unset`: scan forward to find next unset bit
- Use word-level operations for efficiency: skip fully-set/unset words

### Iterator Type 2: RandomIter (Read)

```rust
/// Random-order iterator over IDs.
/// Uses Fisher-Yates shuffle with LCG for deterministic pseudo-random order.
/// Thread-safe: multiple instances can run concurrently.
pub struct RandomIter<'a> {
    tracker: &'a PrefixTracker,
    indices: Vec<u64>,  // Pre-shuffled list of valid IDs
    position: usize,
}
```

**Implementation**:
- On construction: collect all valid IDs (respecting filter), shuffle
- `next()` returns `indices[position++]`
- Memory cost: O(n) where n = number of IDs in range matching filter
- For very large ranges, consider block-based shuffling to reduce memory

**Alternative (Memory-Efficient)**:
```rust
/// Memory-efficient random iterator using multiplicative hashing.
/// Generates pseudo-random permutation on-the-fly without storing all indices.
pub struct RandomIterLowMem<'a> {
    tracker: &'a PrefixTracker,
    range_size: u64,
    current_step: u64,
    multiplier: u64,  // Coprime with range_size
    offset: u64,
    filter: BitFilter,
    count_returned: u64,
    max_count: u64,
}
```
- Use multiplicative group permutation: `next_idx = (current * multiplier + offset) % range_size`
- For filtered iteration: skip non-matching until found or exhausted
- Trade-off: O(1) memory but may have gaps requiring extra iterations

### Iterator Type 3: WriteIter (Concurrent Write)

```rust
/// Write iterator with atomic claim-and-set semantics.
/// Multiple WriteIters can run concurrently.
/// Each call to next() atomically claims a unique ID.
pub struct WriteIter<'a> {
    tracker: &'a PrefixTracker,
    /// Shared atomic cursor across all WriteIter instances for this tracker.
    /// Points to next word to scan.
    cursor: &'a AtomicU64,
    min_id: u64,
    max_id: u64,
    filter: BitFilter,
}
```

**Implementation**:
```rust
impl WriteIter<'_> {
    /// Atomically claim next ID and mark it as set.
    /// Returns None when range exhausted.
    pub fn next(&mut self) -> Option<u64> {
        match self.filter {
            BitFilter::Unset => self.claim_next_unset(),
            BitFilter::All => self.claim_next_any(),
            BitFilter::Set => panic!("WriteIter with Set filter is invalid"),
        }
    }
    
    fn claim_next_unset(&mut self) -> Option<u64> {
        // Atomically find and claim next unset bit
        loop {
            let word_idx = self.cursor.load(Ordering::Acquire);
            if word_idx * 64 >= self.max_id {
                return None;  // Exhausted
            }
            
            let word_ptr = &self.tracker.bitmap.words[word_idx as usize];
            let word = word_ptr.load(Ordering::Acquire);
            
            if word == u64::MAX {
                // Word fully set, advance cursor
                let _ = self.cursor.compare_exchange(
                    word_idx, word_idx + 1,
                    Ordering::AcqRel, Ordering::Relaxed
                );
                continue;
            }
            
            // Find first unset bit
            let bit_idx = (!word).trailing_zeros();
            let global_id = word_idx * 64 + bit_idx as u64;
            
            if global_id >= self.max_id || global_id < self.min_id {
                // Outside range, advance cursor
                self.cursor.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            
            // Attempt to claim
            let mask = 1u64 << bit_idx;
            let old = word_ptr.fetch_or(mask, Ordering::AcqRel);
            if (old & mask) == 0 {
                // Successfully claimed
                self.tracker.bitmap.popcount.fetch_add(1, Ordering::Relaxed);
                return Some(global_id);
            }
            // Someone else claimed this bit, retry same word
        }
    }
}
```

### Iterator Type 4: DeleteIter (Exclusive)

```rust
/// Exclusive delete iterator. Only one can exist per PrefixTracker.
/// Atomically claims and clears bits.
pub struct DeleteIter<'a> {
    tracker: &'a PrefixTracker,
    /// Guard that releases exclusive lock on drop.
    _guard: DeleteIterGuard<'a>,
    cursor: u64,
    max_id: u64,
    filter: BitFilter,
}

/// RAII guard for exclusive delete iterator access.
struct DeleteIterGuard<'a> {
    lock: &'a AtomicBool,
}

impl Drop for DeleteIterGuard<'_> {
    fn drop(&mut self) {
        self.lock.store(false, Ordering::Release);
    }
}
```

**Implementation**:
- `PrefixTracker` has `delete_iter_active: AtomicBool`
- `delete()` builder method does CAS on flag, returns `None` if already active
- `DeleteIter::next()` finds next set bit, atomically clears it, returns ID
- On drop, guard releases flag

### Iterator Type 5: PartitionedIter (Parallel with Disjoint Ranges)

```rust
/// Partitioned iterator for parallel processing with guaranteed disjoint ranges.
/// Each partition covers [start, end) with no overlap.
pub struct PartitionedIter<'a> {
    tracker: &'a PrefixTracker,
    start: u64,
    end: u64,
    current: u64,
    filter: BitFilter,
}
```

**Factory Implementation**:
```rust
impl<'a> IteratorBuilder<'a> {
    pub fn partitioned(self, n: usize) -> Vec<PartitionedIter<'a>> {
        let total = self.max_id - self.min_id;
        let chunk_size = (total + n as u64 - 1) / n as u64;  // Ceiling division
        
        (0..n).map(|i| {
            let start = self.min_id + i as u64 * chunk_size;
            let end = (start + chunk_size).min(self.max_id);
            PartitionedIter {
                tracker: self.tracker,
                start,
                end,
                current: start,
                filter: self.filter,
            }
        }).collect()
    }
}
```

---

## Module 4: `mod.rs` - PrefixGroupsTracker

### Purpose
Manages multiple `PrefixTracker` instances, one per prefix. Thread-safe registry.

### Struct Definition

```rust
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use super::tracker::PrefixTracker;

/// Registry of PrefixTrackers, one per prefix.
/// Thread-safe for concurrent access.
pub struct PrefixGroupsTracker {
    /// Map from prefix string to tracker.
    /// Using RwLock: reads (get) are frequent, writes (create) are rare.
    trackers: RwLock<HashMap<String, Arc<PrefixTracker>>>,
    
    /// Default max_id for new trackers. None = unbounded.
    default_max_id: Option<u64>,
}
```

### Required Methods

```rust
impl PrefixGroupsTracker {
    /// Create empty registry.
    pub fn new() -> Self;
    
    /// Create with default max_id for new trackers.
    pub fn with_default_max_id(max_id: u64) -> Self;
    
    /// Get or create tracker for prefix.
    /// Creates new tracker if not exists.
    pub fn get_or_create(&self, prefix: &str) -> Arc<PrefixTracker>;
    
    /// Get tracker for prefix if exists.
    pub fn get(&self, prefix: &str) -> Option<Arc<PrefixTracker>>;
    
    /// Add new tracker with custom configuration.
    /// Returns existing tracker if prefix already registered.
    pub fn add(&self, tracker: PrefixTracker) -> Arc<PrefixTracker>;
    
    /// Remove tracker for prefix. Returns removed tracker if existed.
    pub fn remove(&self, prefix: &str) -> Option<Arc<PrefixTracker>>;
    
    /// List all registered prefixes.
    pub fn prefixes(&self) -> Vec<String>;
    
    /// Get count of registered prefixes.
    pub fn len(&self) -> usize;
    
    /// Check if empty.
    pub fn is_empty(&self) -> bool;
    
    /// Clear all trackers.
    pub fn clear(&self);
    
    /// Get total count across all trackers.
    pub fn total_count(&self) -> u64;
}

impl Default for PrefixGroupsTracker {
    fn default() -> Self {
        Self::new()
    }
}
```

---

## Performance Requirements

### Latency Targets

| Operation | Target Latency |
|-----------|---------------|
| `test(id)` | < 10 ns |
| `set(id)` (no growth) | < 20 ns |
| `test_and_set(id)` | < 30 ns |
| `count()` | < 5 ns |
| `SequentialIter::next()` | < 50 ns |
| `WriteIter::next()` (low contention) | < 100 ns |
| `get_or_create(prefix)` (exists) | < 50 ns |

### Memory Targets

| Metric | Target |
|--------|--------|
| Bits per ID | 1 bit (dense) |
| Overhead per tracker | < 256 bytes |
| Initial allocation | 512 bytes (4K bits) |

### Concurrency

- All read operations: fully lock-free
- Set/clear/claim: lock-free (atomic CAS)
- Growth: brief lock acquisition, non-blocking for readers
- Iterator creation: lock-free except DeleteIter

---

## Testing Requirements

### Unit Tests (`bitmap.rs`)

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_set_clear_test();
    
    #[test]
    fn test_test_and_set_returns_false_if_already_set();
    
    #[test]
    fn test_popcount_accuracy();
    
    #[test]
    fn test_auto_growth();
    
    #[test]
    fn test_clear_all_resets_state();
    
    #[test]
    fn test_concurrent_set_different_bits();
    
    #[test]
    fn test_concurrent_set_same_bit();
}
```

### Unit Tests (`tracker.rs`)

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_add_remove_exists();
    
    #[test]
    fn test_claim_returns_false_if_exists();
    
    #[test]
    fn test_max_id_enforcement();
    
    #[test]
    fn test_iterator_builder_range();
}
```

### Unit Tests (`iterators.rs`)

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_sequential_iter_all();
    
    #[test]
    fn test_sequential_iter_set_only();
    
    #[test]
    fn test_sequential_iter_unset_only();
    
    #[test]
    fn test_write_iter_claims_unique_ids();
    
    #[test]
    fn test_write_iter_concurrent_no_duplicates();
    
    #[test]
    fn test_delete_iter_exclusive();
    
    #[test]
    fn test_partitioned_iter_full_coverage();
    
    #[test]
    fn test_partitioned_iter_no_overlap();
}
```

### Integration Tests

```rust
#[test]
fn test_prefix_groups_concurrent_access();

#[test]
fn test_write_iter_followed_by_read_iter();

#[test]
fn test_large_scale_bitmap_millions();
```

### Benchmarks (Criterion)

```rust
fn bench_bitmap_set(c: &mut Criterion);
fn bench_bitmap_test(c: &mut Criterion);
fn bench_bitmap_test_and_set(c: &mut Criterion);
fn bench_sequential_iter(c: &mut Criterion);
fn bench_write_iter_single_thread(c: &mut Criterion);
fn bench_write_iter_contention(c: &mut Criterion);
fn bench_get_or_create(c: &mut Criterion);
```

---

## Error Handling

- No panics in hot paths
- `max_id` violations: silently ignore or return false (not error)
- Growth failure (OOM): propagate as `Result` where feasible, or panic (acceptable for OOM)
- Invalid iterator configurations (e.g., min > max): panic in debug, clamp in release

---

## Dependencies

**Required**:
- None (use std only for core implementation)

**Optional/Recommended**:
- `parking_lot`: faster Mutex/RwLock (if RwLock needed for growth)
- `criterion`: benchmarking
- `rand` or `fastrand`: for RandomIter shuffle

**Avoid**:
- `crossbeam` (unless lock-free growth becomes critical)
- Heavy allocation in hot paths

---

## Implementation Order

1. **`bitmap.rs`**: Core atomic bitmap with tests
2. **`tracker.rs`**: PrefixTracker wrapping bitmap with tests  
3. **`iterators.rs`**: Sequential and Write iterators first, then Random/Delete/Partitioned
4. **`mod.rs`**: PrefixGroupsTracker registry
5. **Benchmarks**: Validate performance targets
6. **Integration with existing code**: Replace `ClusterTagMap` usage

---

## Code Style

- `#[inline]` on all hot-path methods
- `#[must_use]` on methods returning values that shouldn't be ignored
- Explicit `Ordering` on all atomic operations (no defaults)
- Document memory ordering choices in comments
- Use `debug_assert!` for invariant checks in hot paths
- Prefer `u64` for IDs consistently (no mixing with `usize` in public API)

---

## Example Usage

```rust
use prefix_tracker::{PrefixGroupsTracker, BitFilter};

// Create registry
let groups = PrefixGroupsTracker::with_default_max_id(1_000_000);

// Get or create tracker for prefix
let tracker = groups.get_or_create("vec:");

// Add some IDs
tracker.add(0);
tracker.add(100);
tracker.add(999);

// Check existence
assert!(tracker.exists(100));
assert!(!tracker.exists(50));

// Count
assert_eq!(tracker.count(), 3);

// Sequential iteration over set bits
for id in tracker.iter().set_only().sequential() {
    println!("Existing ID: {}", id);
}

// Parallel write iteration (claim unset IDs)
let writers: Vec<_> = (0..4).map(|_| {
    let t = tracker.clone();
    std::thread::spawn(move || {
        let mut iter = t.iter().unset_only().write();
        let mut claimed = 0;
        while let Some(id) = iter.next() {
            // Simulate work
            claimed += 1;
        }
        claimed
    })
}).collect();

let total_claimed: u64 = writers.into_iter()
    .map(|h| h.join().unwrap())
    .sum();

// Partitioned parallel read
let partitions = tracker.iter().set_only().partitioned(8);
let handles: Vec<_> = partitions.into_iter().map(|mut part| {
    std::thread::spawn(move || {
        let mut count = 0;
        while let Some(_id) = part.next_id() {
            count += 1;
        }
        count
    })
}).collect();
```

---

## Appendix: Atomic Operations Reference

```rust
// Set bit atomically, return true if changed 0→1
fn atomic_set(word: &AtomicU64, bit: u32, popcount: &AtomicU64) -> bool {
    let mask = 1u64 << bit;
    let old = word.fetch_or(mask, Ordering::AcqRel);
    let changed = (old & mask) == 0;
    if changed {
        popcount.fetch_add(1, Ordering::Relaxed);
    }
    changed
}

// Clear bit atomically, return true if changed 1→0
fn atomic_clear(word: &AtomicU64, bit: u32, popcount: &AtomicU64) -> bool {
    let mask = 1u64 << bit;
    let old = word.fetch_and(!mask, Ordering::AcqRel);
    let changed = (old & mask) != 0;
    if changed {
        popcount.fetch_sub(1, Ordering::Relaxed);
    }
    changed
}

// Test bit (no modification)
fn atomic_test(word: &AtomicU64, bit: u32) -> bool {
    let mask = 1u64 << bit;
    (word.load(Ordering::Acquire) & mask) != 0
}

// Test-and-set with CAS (for claim semantics)
fn atomic_test_and_set(word: &AtomicU64, bit: u32, popcount: &AtomicU64) -> bool {
    let mask = 1u64 << bit;
    loop {
        let old = word.load(Ordering::Acquire);
        if (old & mask) != 0 {
            return false;  // Already set
        }
        match word.compare_exchange_weak(
            old, old | mask,
            Ordering::AcqRel, Ordering::Relaxed
        ) {
            Ok(_) => {
                popcount.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            Err(_) => continue,  // Retry
        }
    }
}
```
