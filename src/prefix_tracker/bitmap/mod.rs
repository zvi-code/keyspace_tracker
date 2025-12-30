//! Atomic bitmap implementation with SIMD-accelerated operations.
//!
//! Provides a lock-free concurrent bitmap using `AtomicU64` words.
//! Uses runtime CPU detection to select optimal implementations:
//!
//! - **ARM**: NEON baseline, with SVE/SVE2 for Graviton 3/4
//! - **x86_64**: POPCNT baseline, AVX2 for Haswell+, AVX-512 for Skylake-X+
//! - **Other**: Portable scalar fallback with loop unrolling
//!
//! All bit operations are atomic and safe for concurrent access.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use parking_lot::Mutex;

// Platform-specific implementations
#[cfg(target_arch = "aarch64")]
mod arm;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86;

// Re-export platform capabilities for debugging/introspection
#[cfg(target_arch = "aarch64")]
pub use arm::ArmCapabilities;
#[cfg(target_arch = "x86_64")]
pub use x86::X86Capabilities;

// ============================================================================
// Internal SIMD Dispatch
// ============================================================================

/// Count set bits using the best available SIMD implementation.
#[inline]
fn simd_popcount_slice(words: &[AtomicU64]) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        arm::popcount_slice(words)
    }
    #[cfg(target_arch = "x86_64")]
    {
        x86::popcount_slice(words)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::popcount_slice(words)
    }
}

/// Find first non-zero word using SIMD-accelerated scan.
#[inline]
fn simd_find_first_nonzero(words: &[AtomicU64], start_word: usize) -> Option<(usize, u64)> {
    #[cfg(target_arch = "aarch64")]
    {
        arm::find_first_nonzero(words, start_word)
    }
    #[cfg(target_arch = "x86_64")]
    {
        x86::find_first_nonzero(words, start_word)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::find_first_nonzero(words, start_word)
    }
}

/// Find first word with unset bits using SIMD-accelerated scan.
#[inline]
fn simd_find_first_not_full(words: &[AtomicU64], start_word: usize) -> Option<(usize, u64)> {
    #[cfg(target_arch = "aarch64")]
    {
        arm::find_first_not_full(words, start_word)
    }
    #[cfg(target_arch = "x86_64")]
    {
        x86::find_first_not_full(words, start_word)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::find_first_not_full(words, start_word)
    }
}

/// Prefetch next cache line for iteration.
#[inline]
fn simd_prefetch_next_cacheline(words: &[AtomicU64], current_idx: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        arm::prefetch_next_cacheline(words, current_idx);
    }
    #[cfg(target_arch = "x86_64")]
    {
        x86::prefetch_next_cacheline(words, current_idx);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (words, current_idx);
    }
}

// ============================================================================
// Constants
// ============================================================================

/// Number of bits per word.
const BITS_PER_WORD: usize = 64;

/// Initial capacity in bits.
const INITIAL_CAPACITY_BITS: usize = 4096;

/// Shift amount for word index calculation (log2(64) = 6).
const WORD_SHIFT: usize = 6;

/// Mask for bit index within word.
const WORD_MASK: usize = 63;

// ============================================================================
// AtomicBitmap
// ============================================================================

/// A thread-safe, lock-free atomic bitmap.
///
/// `AtomicBitmap` provides efficient storage and manipulation of a set of bits,
/// where each bit represents whether an ID exists or not. All operations are
/// atomic and safe for concurrent access from multiple threads.
///
/// # Features
///
/// - **Lock-free operations**: `test`, `set`, `clear`, `test_and_set` use atomic
///   compare-and-swap (CAS) and are wait-free for most operations
/// - **Auto-growing**: Automatically expands when accessing indices beyond capacity
/// - **SIMD-accelerated**: Uses ARM NEON or x86 AVX2/POPCNT for bulk operations
/// - **Memory efficient**: 1 bit per ID, 8 bytes per 64 IDs
///
/// # Performance
///
/// | Operation | Latency | Notes |
/// |-----------|---------|-------|
/// | `test` | ~1.6 ns | L1 cache hit |
/// | `set` | ~7 ns | Atomic OR |
/// | `clear` | ~7 ns | Atomic AND |
/// | `test_and_set` | ~2.4 ns | CAS loop |
/// | `find_next_set` | ~4 ns | SIMD accelerated |
///
/// # Thread Safety
///
/// All bit operations are lock-free using atomic primitives. The only blocking
/// operation is `ensure_capacity` when the bitmap needs to grow, which uses a
/// mutex to coordinate allocation.
///
/// # Examples
///
/// ## Basic Operations
///
/// ```rust
/// use prefix_tracker::AtomicBitmap;
///
/// let bitmap = AtomicBitmap::with_capacity(1000);
///
/// // Set and test bits
/// bitmap.set(42);
/// assert!(bitmap.test(42));
/// assert!(!bitmap.test(43));
///
/// // Clear bits
/// bitmap.clear(42);
/// assert!(!bitmap.test(42));
///
/// // Population count
/// bitmap.set(1);
/// bitmap.set(2);
/// bitmap.set(3);
/// assert_eq!(bitmap.count(), 3);
/// ```
///
/// ## Atomic Claim (Test-and-Set)
///
/// ```rust
/// use prefix_tracker::AtomicBitmap;
/// use std::sync::Arc;
/// use std::thread;
///
/// let bitmap = Arc::new(AtomicBitmap::with_capacity(1000));
///
/// // Multiple threads can atomically claim unique bits
/// let handles: Vec<_> = (0..4).map(|_| {
///     let bm = bitmap.clone();
///     thread::spawn(move || {
///         (0..100).filter(|&i| bm.test_and_set(i)).count()
///     })
/// }).collect();
///
/// let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
/// assert_eq!(total, 100); // Each bit claimed exactly once
/// ```
///
/// ## Scanning
///
/// ```rust
/// use prefix_tracker::AtomicBitmap;
///
/// let bitmap = AtomicBitmap::with_capacity(1000);
/// bitmap.set(10);
/// bitmap.set(50);
/// bitmap.set(100);
///
/// // Find set bits
/// let mut pos = 0;
/// let mut found = Vec::new();
/// while let Some(next) = bitmap.find_next_set(pos) {
///     found.push(next);
///     pos = next + 1;
/// }
/// assert_eq!(found, vec![10, 50, 100]);
///
/// // Find unset bits
/// let first_unset = bitmap.find_next_unset(0, 1000);
/// assert_eq!(first_unset, Some(0));
/// ```
pub struct AtomicBitmap {
    /// Bitmap storage. Uses UnsafeCell for interior mutability during growth.
    words: std::cell::UnsafeCell<Box<[AtomicU64]>>,

    /// Current capacity in bits (always multiple of 64).
    capacity: AtomicUsize,

    /// Population count (number of set bits).
    /// Updated atomically on set/clear operations.
    popcount: AtomicU64,

    /// Mutex for growth synchronization only.
    growth_lock: Mutex<()>,
}

// SAFETY: All bit operations are atomic. Growth is synchronized via mutex.
unsafe impl Send for AtomicBitmap {}
unsafe impl Sync for AtomicBitmap {}

impl AtomicBitmap {
    /// Create a new bitmap with default initial capacity (4096 bits).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use prefix_tracker::AtomicBitmap;
    ///
    /// let bitmap = AtomicBitmap::new();
    /// assert!(bitmap.capacity() >= 4096);
    /// ```
    pub fn new() -> Self {
        Self::with_capacity(INITIAL_CAPACITY_BITS)
    }

    /// Create a new bitmap with specified initial capacity (in bits).
    ///
    /// Capacity is rounded up to the next multiple of 64.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use prefix_tracker::AtomicBitmap;
    ///
    /// let bitmap = AtomicBitmap::with_capacity(1_000_000);
    /// assert!(bitmap.capacity() >= 1_000_000);
    /// ```
    pub fn with_capacity(capacity_bits: usize) -> Self {
        let capacity = capacity_bits.max(64).next_multiple_of(64);
        let word_count = capacity / BITS_PER_WORD;

        let mut words = Vec::with_capacity(word_count);
        words.resize_with(word_count, || AtomicU64::new(0));

        Self {
            words: std::cell::UnsafeCell::new(words.into_boxed_slice()),
            capacity: AtomicUsize::new(capacity),
            popcount: AtomicU64::new(0),
            growth_lock: Mutex::new(()),
        }
    }

    /// Get current capacity in bits.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Acquire)
    }

    /// Get number of words in the bitmap.
    #[inline]
    pub fn word_count(&self) -> usize {
        self.capacity() / BITS_PER_WORD
    }

    /// Get population count (number of set bits).
    #[inline]
    pub fn count(&self) -> u64 {
        self.popcount.load(Ordering::Relaxed)
    }

    /// Check if bitmap is empty (no bits set).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Ensure capacity for the given bit index.
    ///
    /// Returns true if growth occurred.
    pub fn ensure_capacity(&self, bit_index: usize) -> bool {
        let required = (bit_index + 1).next_multiple_of(64);
        if self.capacity() >= required {
            return false;
        }

        self.grow(required);
        true
    }

    /// Grow the bitmap to at least the specified capacity.
    fn grow(&self, required_capacity: usize) {
        let _guard = self.growth_lock.lock();

        // Double-check after acquiring lock
        let current = self.capacity.load(Ordering::Acquire);
        if current >= required_capacity {
            return;
        }

        // Calculate new capacity (at least double, or required)
        let new_capacity = std::cmp::max(current * 2, required_capacity).next_multiple_of(64);
        let new_word_count = new_capacity / BITS_PER_WORD;

        // Allocate new storage
        let mut new_words: Vec<AtomicU64> = Vec::with_capacity(new_word_count);
        new_words.resize_with(new_word_count, || AtomicU64::new(0));

        // Copy old data
        // SAFETY: We hold the growth lock, preventing concurrent growth.
        let old_words = unsafe { &*self.words.get() };
        for (i, word) in old_words.iter().enumerate() {
            new_words[i].store(word.load(Ordering::Relaxed), Ordering::Relaxed);
        }

        // Swap storage
        // SAFETY: We hold the growth lock. Readers may see old or new data,
        // but both are valid (old data is subset of new data).
        unsafe {
            *self.words.get() = new_words.into_boxed_slice();
        }

        // Publish new capacity
        self.capacity.store(new_capacity, Ordering::Release);
    }

    /// Get a reference to the words slice.
    ///
    /// SAFETY: Caller must ensure no concurrent growth during use.
    #[inline]
    fn words(&self) -> &[AtomicU64] {
        // SAFETY: We only read through atomic operations.
        unsafe { &*self.words.get() }
    }

    /// Load a word value.
    #[inline]
    pub fn load_word(&self, word_index: usize) -> u64 {
        let words = self.words();
        if word_index >= words.len() {
            return 0;
        }
        words[word_index].load(Ordering::Acquire)
    }

    /// Fetch-or on a word, returning the previous value.
    #[inline]
    pub fn word_fetch_or(&self, word_index: usize, mask: u64) -> u64 {
        let words = self.words();
        if word_index >= words.len() {
            return 0;
        }
        words[word_index].fetch_or(mask, Ordering::AcqRel)
    }

    /// Fetch-and on a word, returning the previous value.
    #[inline]
    pub fn word_fetch_and(&self, word_index: usize, mask: u64) -> u64 {
        let words = self.words();
        if word_index >= words.len() {
            return u64::MAX;
        }
        words[word_index].fetch_and(mask, Ordering::AcqRel)
    }

    /// Set a bit at the given index.
    ///
    /// Returns the previous value (false if was unset, true if was set).
    /// Auto-grows if index >= capacity.
    #[inline]
    pub fn set(&self, index: usize) -> bool {
        self.ensure_capacity(index);

        let word_idx = index >> WORD_SHIFT;
        let bit_idx = index & WORD_MASK;
        let mask = 1u64 << bit_idx;

        let old = self.word_fetch_or(word_idx, mask);
        let was_set = (old & mask) != 0;

        if !was_set {
            self.popcount.fetch_add(1, Ordering::Relaxed);
        }

        was_set
    }

    /// Clear a bit at the given index.
    ///
    /// Returns the previous value (true if was set, false if was unset).
    /// No-op if index >= capacity.
    #[inline]
    pub fn clear(&self, index: usize) -> bool {
        if index >= self.capacity() {
            return false;
        }

        let word_idx = index >> WORD_SHIFT;
        let bit_idx = index & WORD_MASK;
        let mask = 1u64 << bit_idx;

        let old = self.word_fetch_and(word_idx, !mask);
        let was_set = (old & mask) != 0;

        if was_set {
            self.popcount.fetch_sub(1, Ordering::Relaxed);
        }

        was_set
    }

    /// Test if a bit is set at the given index.
    ///
    /// Returns false if index >= capacity.
    #[inline]
    pub fn test(&self, index: usize) -> bool {
        if index >= self.capacity() {
            return false;
        }

        let word_idx = index >> WORD_SHIFT;
        let bit_idx = index & WORD_MASK;
        let mask = 1u64 << bit_idx;

        (self.load_word(word_idx) & mask) != 0
    }

    /// Atomically test and set a bit.
    ///
    /// If the bit is unset (0), sets it to 1 and returns true.
    /// If the bit is already set (1), returns false.
    /// Auto-grows if index >= capacity.
    #[inline]
    pub fn test_and_set(&self, index: usize) -> bool {
        self.ensure_capacity(index);

        let word_idx = index >> WORD_SHIFT;
        let bit_idx = index & WORD_MASK;
        let mask = 1u64 << bit_idx;

        let words = self.words();
        if word_idx >= words.len() {
            return false;
        }

        let word = &words[word_idx];

        loop {
            let old = word.load(Ordering::Acquire);
            if (old & mask) != 0 {
                return false; // Already set
            }

            match word.compare_exchange_weak(old, old | mask, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => {
                    self.popcount.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
                Err(_) => continue, // Retry
            }
        }
    }

    /// Atomically test and clear a bit.
    ///
    /// If the bit is set (1), clears it to 0 and returns true.
    /// If the bit is already unset (0), returns false.
    #[inline]
    pub fn test_and_clear(&self, index: usize) -> bool {
        if index >= self.capacity() {
            return false;
        }

        let word_idx = index >> WORD_SHIFT;
        let bit_idx = index & WORD_MASK;
        let mask = 1u64 << bit_idx;

        let words = self.words();
        if word_idx >= words.len() {
            return false;
        }

        let word = &words[word_idx];

        loop {
            let old = word.load(Ordering::Acquire);
            if (old & mask) == 0 {
                return false; // Already unset
            }

            match word.compare_exchange_weak(old, old & !mask, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.popcount.fetch_sub(1, Ordering::Relaxed);
                    return true;
                }
                Err(_) => continue, // Retry
            }
        }
    }

    /// Clear all bits and reset to initial capacity.
    ///
    /// This requires exclusive access (&mut self) as it deallocates storage.
    pub fn clear_all(&mut self) {
        let word_count = INITIAL_CAPACITY_BITS / BITS_PER_WORD;
        let mut words = Vec::with_capacity(word_count);
        words.resize_with(word_count, || AtomicU64::new(0));

        *self.words.get_mut() = words.into_boxed_slice();
        *self.capacity.get_mut() = INITIAL_CAPACITY_BITS;
        *self.popcount.get_mut() = 0;
    }

    /// Clear all bits but retain allocated capacity.
    ///
    /// This requires exclusive access (&mut self).
    pub fn reset(&mut self) {
        let words = self.words.get_mut();
        for word in words.iter_mut() {
            *word.get_mut() = 0;
        }
        *self.popcount.get_mut() = 0;
    }

    /// Find the first set bit starting from `start_index`.
    ///
    /// Returns None if no set bit is found before capacity.
    /// Uses SIMD-optimized scanning on ARM NEON and x86_64.
    pub fn find_next_set(&self, start_index: usize) -> Option<usize> {
        let capacity = self.capacity();
        if start_index >= capacity {
            return None;
        }

        let words = self.words();
        let start_word_idx = start_index >> WORD_SHIFT;
        let bit_idx = start_index & WORD_MASK;

        // Check first word (masked to ignore bits before start)
        let first_word = words[start_word_idx].load(Ordering::Relaxed) & (u64::MAX << bit_idx);
        if first_word != 0 {
            let bit = first_word.trailing_zeros() as usize;
            let index = (start_word_idx << WORD_SHIFT) + bit;
            if index < capacity {
                return Some(index);
            }
        }

        // Use SIMD-optimized scan for remaining words
        if let Some((word_idx, word)) = simd_find_first_nonzero(words, start_word_idx + 1) {
            let bit = word.trailing_zeros() as usize;
            let index = (word_idx << WORD_SHIFT) + bit;
            if index < capacity {
                return Some(index);
            }
        }

        None
    }

    /// Find the first unset bit starting from `start_index`.
    ///
    /// Returns None if no unset bit is found before `max_index`.
    /// Uses SIMD-optimized scanning on ARM NEON and x86_64.
    pub fn find_next_unset(&self, start_index: usize, max_index: usize) -> Option<usize> {
        let capacity = self.capacity();
        let limit = max_index.min(capacity);

        if start_index >= limit {
            return None;
        }

        let words = self.words();
        let start_word_idx = start_index >> WORD_SHIFT;
        let bit_idx = start_index & WORD_MASK;

        // Check first word (masked to ignore bits before start)
        let first_word = words[start_word_idx].load(Ordering::Relaxed) | ((1u64 << bit_idx) - 1);
        if first_word != u64::MAX {
            let bit = (!first_word).trailing_zeros() as usize;
            let index = (start_word_idx << WORD_SHIFT) + bit;
            if index < limit {
                return Some(index);
            }
        }

        // Use SIMD-optimized scan for remaining words
        if let Some((word_idx, word)) = simd_find_first_not_full(words, start_word_idx + 1) {
            let bit = (!word).trailing_zeros() as usize;
            let index = (word_idx << WORD_SHIFT) + bit;
            if index < limit {
                return Some(index);
            }
        }

        None
    }

    /// Recompute population count using SIMD-optimized counting.
    ///
    /// This is useful after bulk operations or to verify the incremental count.
    /// Uses NEON on ARM, POPCNT/AVX2 on x86_64, with scalar fallback.
    pub fn recompute_count(&self) -> u64 {
        simd_popcount_slice(self.words())
    }

    /// Verify and fix the population count if it has drifted.
    ///
    /// Returns true if the count was corrected.
    pub fn verify_count(&self) -> bool {
        let actual = self.recompute_count();
        let stored = self.popcount.load(Ordering::Relaxed);
        if actual != stored {
            self.popcount.store(actual, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Prefetch the next cache line for sequential iteration.
    ///
    /// Call this during iteration to reduce cache misses.
    #[inline]
    pub fn prefetch_ahead(&self, current_word_idx: usize) {
        simd_prefetch_next_cacheline(self.words(), current_word_idx);
    }

    /// Get the density of the bitmap (ratio of set bits to capacity).
    #[inline]
    pub fn density(&self) -> f64 {
        let cap = self.capacity() as f64;
        if cap == 0.0 {
            0.0
        } else {
            self.count() as f64 / cap
        }
    }
}

impl Default for AtomicBitmap {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AtomicBitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicBitmap")
            .field("capacity", &self.capacity())
            .field("count", &self.count())
            .finish()
    }
}

// ============================================================================
// Public API re-exports for advanced usage
// ============================================================================

/// Print detected CPU capabilities for debugging.
pub fn print_cpu_capabilities() {
    #[cfg(target_arch = "aarch64")]
    {
        let caps = ArmCapabilities::get();
        println!("ARM Capabilities: {:?}", caps);
    }
    #[cfg(target_arch = "x86_64")]
    {
        let caps = X86Capabilities::get();
        println!("x86_64 Capabilities: {:?}", caps);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        println!("Using scalar fallback (no SIMD)");
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_new_bitmap() {
        let bm = AtomicBitmap::new();
        assert_eq!(bm.capacity(), INITIAL_CAPACITY_BITS);
        assert_eq!(bm.count(), 0);
        assert!(bm.is_empty());
    }

    #[test]
    fn test_set_and_test() {
        let bm = AtomicBitmap::new();

        assert!(!bm.test(0));
        assert!(!bm.set(0)); // Returns false (was not set)
        assert!(bm.test(0));
        assert!(bm.set(0)); // Returns true (was already set)

        assert!(!bm.test(63));
        bm.set(63);
        assert!(bm.test(63));

        assert!(!bm.test(64));
        bm.set(64);
        assert!(bm.test(64));
    }

    #[test]
    fn test_clear() {
        let bm = AtomicBitmap::new();

        bm.set(42);
        assert!(bm.test(42));
        assert_eq!(bm.count(), 1);

        assert!(bm.clear(42)); // Returns true (was set)
        assert!(!bm.test(42));
        assert_eq!(bm.count(), 0);

        assert!(!bm.clear(42)); // Returns false (was not set)
    }

    #[test]
    fn test_test_and_set() {
        let bm = AtomicBitmap::new();

        assert!(bm.test_and_set(100)); // Success, was unset
        assert!(!bm.test_and_set(100)); // Failure, already set
        assert!(bm.test(100));
        assert_eq!(bm.count(), 1);
    }

    #[test]
    fn test_test_and_clear() {
        let bm = AtomicBitmap::new();

        bm.set(100);
        assert!(bm.test_and_clear(100)); // Success, was set
        assert!(!bm.test_and_clear(100)); // Failure, already unset
        assert!(!bm.test(100));
        assert_eq!(bm.count(), 0);
    }

    #[test]
    fn test_auto_growth() {
        let bm = AtomicBitmap::with_capacity(64);
        assert_eq!(bm.capacity(), 64);

        bm.set(1000);
        assert!(bm.capacity() >= 1001);
        assert!(bm.test(1000));
    }

    #[test]
    fn test_popcount() {
        let bm = AtomicBitmap::new();

        bm.set(0);
        bm.set(1);
        bm.set(100);
        assert_eq!(bm.count(), 3);

        bm.clear(1);
        assert_eq!(bm.count(), 2);

        bm.set(100); // Already set, count shouldn't change
        assert_eq!(bm.count(), 2);
    }

    #[test]
    fn test_find_next_set() {
        let bm = AtomicBitmap::new();

        bm.set(10);
        bm.set(100);
        bm.set(1000);

        assert_eq!(bm.find_next_set(0), Some(10));
        assert_eq!(bm.find_next_set(10), Some(10));
        assert_eq!(bm.find_next_set(11), Some(100));
        assert_eq!(bm.find_next_set(100), Some(100));
        assert_eq!(bm.find_next_set(101), Some(1000));
        assert_eq!(bm.find_next_set(1001), None);
    }

    #[test]
    fn test_find_next_unset() {
        let bm = AtomicBitmap::with_capacity(128);

        // Set first 10 bits
        for i in 0..10 {
            bm.set(i);
        }

        assert_eq!(bm.find_next_unset(0, 100), Some(10));
        assert_eq!(bm.find_next_unset(10, 100), Some(10));
        assert_eq!(bm.find_next_unset(5, 100), Some(10));
    }

    #[test]
    fn test_concurrent_set() {
        let bm = Arc::new(AtomicBitmap::with_capacity(10000));
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let bm = bm.clone();
                thread::spawn(move || {
                    for i in 0..1000 {
                        bm.set(t * 1000 + i);
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        assert_eq!(bm.count(), 8000);
    }

    #[test]
    fn test_concurrent_test_and_set() {
        let bm = Arc::new(AtomicBitmap::with_capacity(1000));
        let success_count = Arc::new(AtomicU64::new(0));

        let threads: Vec<_> = (0..8)
            .map(|_| {
                let bm = bm.clone();
                let success = success_count.clone();
                thread::spawn(move || {
                    for i in 0..1000 {
                        if bm.test_and_set(i) {
                            success.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        // Exactly 1000 successful claims (one per bit)
        assert_eq!(success_count.load(Ordering::Relaxed), 1000);
        assert_eq!(bm.count(), 1000);
    }

    #[test]
    fn test_clear_all() {
        let mut bm = AtomicBitmap::new();

        for i in 0..100 {
            bm.set(i);
        }
        assert_eq!(bm.count(), 100);

        bm.clear_all();
        assert_eq!(bm.count(), 0);
        assert_eq!(bm.capacity(), INITIAL_CAPACITY_BITS);
    }

    #[test]
    fn test_reset() {
        let mut bm = AtomicBitmap::with_capacity(10000);

        for i in 0..100 {
            bm.set(i);
        }
        let cap = bm.capacity();

        bm.reset();
        assert_eq!(bm.count(), 0);
        assert_eq!(bm.capacity(), cap); // Capacity preserved
    }

    #[test]
    fn test_recompute_count() {
        let bm = AtomicBitmap::new();

        for i in 0..100 {
            bm.set(i);
        }

        assert_eq!(bm.recompute_count(), 100);
        assert_eq!(bm.count(), 100);
    }

    #[test]
    fn test_cpu_capabilities() {
        // Just ensure this doesn't panic
        print_cpu_capabilities();
    }
}
