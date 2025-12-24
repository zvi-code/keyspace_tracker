//! SIMD-optimized bitmap operations.
//!
//! Provides vectorized implementations for:
//! - Bulk popcount (count set bits)
//! - Fast scanning (find_next_set/unset)
//! - Prefetching for sequential access
//!
//! Supports:
//! - ARM NEON (aarch64)
//! - x86_64 AVX2/POPCNT
//! - Fallback scalar implementation

use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Cache Line Constants
// ============================================================================

/// Cache line size in bytes (64 bytes for ARM Graviton and modern x86)
pub const CACHE_LINE_SIZE: usize = 64;

/// Number of u64 words per cache line
pub const WORDS_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / 8;

// ============================================================================
// Prefetch Hints
// ============================================================================

/// Prefetch data for reading
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // PRFM PLDL1KEEP - prefetch for load, L1 cache, keep in cache
        std::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags)
        );
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr; // Suppress unused warning
    }
}

/// Prefetch data for writing
#[inline(always)]
pub fn prefetch_write<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // PRFM PSTL1KEEP - prefetch for store, L1 cache, keep in cache
        std::arch::asm!(
            "prfm pstl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags)
        );
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        _mm_prefetch(ptr as *const i8, _MM_HINT_ET0);
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

/// Prefetch next cache line during iteration
#[inline(always)]
pub fn prefetch_next_cacheline(words: &[AtomicU64], current_idx: usize) {
    let next_cacheline_idx = (current_idx + WORDS_PER_CACHE_LINE) & !(WORDS_PER_CACHE_LINE - 1);
    if next_cacheline_idx < words.len() {
        prefetch_read(words[next_cacheline_idx..].as_ptr());
    }
}

// ============================================================================
// SIMD Popcount - ARM NEON
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// Count set bits in a slice of AtomicU64 using NEON SIMD.
    /// Processes 2 u64s (128 bits) per iteration.
    #[inline]
    pub fn popcount_slice(words: &[AtomicU64]) -> u64 {
        let mut total: u64 = 0;
        let mut i = 0;
        let len = words.len();

        // Process 4 words (256 bits) per iteration for better ILP
        while i + 4 <= len {
            // Prefetch ahead
            if i + WORDS_PER_CACHE_LINE < len {
                prefetch_read(words[i + WORDS_PER_CACHE_LINE..].as_ptr());
            }

            unsafe {
                // Load 4 u64s
                let w0 = words[i].load(Ordering::Relaxed);
                let w1 = words[i + 1].load(Ordering::Relaxed);
                let w2 = words[i + 2].load(Ordering::Relaxed);
                let w3 = words[i + 3].load(Ordering::Relaxed);

                // Pack into NEON vectors
                let v0: uint64x2_t = vcombine_u64(vcreate_u64(w0), vcreate_u64(w1));
                let v1: uint64x2_t = vcombine_u64(vcreate_u64(w2), vcreate_u64(w3));

                // Reinterpret as bytes and count bits per byte
                let cnt0: uint8x16_t = vcntq_u8(vreinterpretq_u8_u64(v0));
                let cnt1: uint8x16_t = vcntq_u8(vreinterpretq_u8_u64(v1));

                // Horizontal add: sum all bytes
                // vpaddlq_u8: pairwise add u8 -> u16
                // vpaddlq_u16: pairwise add u16 -> u32
                // vpaddlq_u32: pairwise add u32 -> u64
                let sum0: uint64x2_t = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(cnt0)));
                let sum1: uint64x2_t = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(cnt1)));

                // Extract and accumulate
                total += vgetq_lane_u64(sum0, 0) + vgetq_lane_u64(sum0, 1);
                total += vgetq_lane_u64(sum1, 0) + vgetq_lane_u64(sum1, 1);
            }

            i += 4;
        }

        // Handle remaining words with scalar popcount
        while i < len {
            total += words[i].load(Ordering::Relaxed).count_ones() as u64;
            i += 1;
        }

        total
    }

    /// Find first non-zero word starting from `start_word` using NEON.
    /// Returns (word_index, word_value) or None if all zero.
    #[inline]
    pub fn find_first_nonzero(words: &[AtomicU64], start_word: usize) -> Option<(usize, u64)> {
        let len = words.len();
        if start_word >= len {
            return None;
        }

        let mut i = start_word;

        // Process 4 words at a time
        while i + 4 <= len {
            // Prefetch ahead
            if i + WORDS_PER_CACHE_LINE < len {
                prefetch_read(words[i + WORDS_PER_CACHE_LINE..].as_ptr());
            }

            unsafe {
                let w0 = words[i].load(Ordering::Relaxed);
                let w1 = words[i + 1].load(Ordering::Relaxed);
                let w2 = words[i + 2].load(Ordering::Relaxed);
                let w3 = words[i + 3].load(Ordering::Relaxed);

                // Quick check: OR all words together
                let any_set = w0 | w1 | w2 | w3;
                if any_set != 0 {
                    // At least one is non-zero, find which one
                    if w0 != 0 { return Some((i, w0)); }
                    if w1 != 0 { return Some((i + 1, w1)); }
                    if w2 != 0 { return Some((i + 2, w2)); }
                    return Some((i + 3, w3));
                }
            }

            i += 4;
        }

        // Scalar fallback for remaining words
        while i < len {
            let w = words[i].load(Ordering::Relaxed);
            if w != 0 {
                return Some((i, w));
            }
            i += 1;
        }

        None
    }

    /// Find first word with unset bits (not all 1s) starting from `start_word`.
    /// Returns (word_index, word_value) or None if all full.
    #[inline]
    pub fn find_first_not_full(words: &[AtomicU64], start_word: usize) -> Option<(usize, u64)> {
        let len = words.len();
        if start_word >= len {
            return None;
        }

        let mut i = start_word;
        const ALL_ONES: u64 = !0u64;

        // Process 4 words at a time
        while i + 4 <= len {
            if i + WORDS_PER_CACHE_LINE < len {
                prefetch_read(words[i + WORDS_PER_CACHE_LINE..].as_ptr());
            }

            let w0 = words[i].load(Ordering::Relaxed);
            let w1 = words[i + 1].load(Ordering::Relaxed);
            let w2 = words[i + 2].load(Ordering::Relaxed);
            let w3 = words[i + 3].load(Ordering::Relaxed);

            // Quick check: AND all words - if result is all 1s, all are full
            let all_full = w0 & w1 & w2 & w3;
            if all_full != ALL_ONES {
                // At least one has unset bits
                if w0 != ALL_ONES { return Some((i, w0)); }
                if w1 != ALL_ONES { return Some((i + 1, w1)); }
                if w2 != ALL_ONES { return Some((i + 2, w2)); }
                return Some((i + 3, w3));
            }

            i += 4;
        }

        // Scalar fallback
        while i < len {
            let w = words[i].load(Ordering::Relaxed);
            if w != ALL_ONES {
                return Some((i, w));
            }
            i += 1;
        }

        None
    }
}

// ============================================================================
// SIMD Popcount - x86_64 AVX2/POPCNT
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;

    /// Count set bits using hardware POPCNT instruction.
    /// Most x86_64 CPUs since ~2008 support this.
    #[inline]
    #[target_feature(enable = "popcnt")]
    unsafe fn popcnt64(x: u64) -> u64 {
        use std::arch::x86_64::_popcnt64;
        _popcnt64(x as i64) as u64
    }

    /// Count set bits in a slice of AtomicU64.
    /// Uses hardware POPCNT with prefetching and unrolling.
    #[inline]
    pub fn popcount_slice(words: &[AtomicU64]) -> u64 {
        // Check for POPCNT support at runtime
        if !is_x86_feature_detected!("popcnt") {
            return popcount_scalar(words);
        }

        let mut total: u64 = 0;
        let mut i = 0;
        let len = words.len();

        // Process 8 words per iteration for better ILP
        while i + 8 <= len {
            // Prefetch ahead
            if i + WORDS_PER_CACHE_LINE < len {
                prefetch_read(words[i + WORDS_PER_CACHE_LINE..].as_ptr());
            }

            unsafe {
                let w0 = words[i].load(Ordering::Relaxed);
                let w1 = words[i + 1].load(Ordering::Relaxed);
                let w2 = words[i + 2].load(Ordering::Relaxed);
                let w3 = words[i + 3].load(Ordering::Relaxed);
                let w4 = words[i + 4].load(Ordering::Relaxed);
                let w5 = words[i + 5].load(Ordering::Relaxed);
                let w6 = words[i + 6].load(Ordering::Relaxed);
                let w7 = words[i + 7].load(Ordering::Relaxed);

                // Parallel POPCNT - CPU can execute multiple in parallel
                total += popcnt64(w0) + popcnt64(w1) + popcnt64(w2) + popcnt64(w3);
                total += popcnt64(w4) + popcnt64(w5) + popcnt64(w6) + popcnt64(w7);
            }

            i += 8;
        }

        // Handle remaining words
        while i < len {
            unsafe {
                total += popcnt64(words[i].load(Ordering::Relaxed));
            }
            i += 1;
        }

        total
    }

    /// Find first non-zero word using AVX2 comparison.
    #[inline]
    pub fn find_first_nonzero(words: &[AtomicU64], start_word: usize) -> Option<(usize, u64)> {
        let len = words.len();
        if start_word >= len {
            return None;
        }

        let mut i = start_word;

        // Process 4 words at a time (same as NEON for consistency)
        while i + 4 <= len {
            if i + WORDS_PER_CACHE_LINE < len {
                prefetch_read(words[i + WORDS_PER_CACHE_LINE..].as_ptr());
            }

            let w0 = words[i].load(Ordering::Relaxed);
            let w1 = words[i + 1].load(Ordering::Relaxed);
            let w2 = words[i + 2].load(Ordering::Relaxed);
            let w3 = words[i + 3].load(Ordering::Relaxed);

            let any_set = w0 | w1 | w2 | w3;
            if any_set != 0 {
                if w0 != 0 { return Some((i, w0)); }
                if w1 != 0 { return Some((i + 1, w1)); }
                if w2 != 0 { return Some((i + 2, w2)); }
                return Some((i + 3, w3));
            }

            i += 4;
        }

        // Scalar fallback
        while i < len {
            let w = words[i].load(Ordering::Relaxed);
            if w != 0 {
                return Some((i, w));
            }
            i += 1;
        }

        None
    }

    /// Find first word with unset bits.
    #[inline]
    pub fn find_first_not_full(words: &[AtomicU64], start_word: usize) -> Option<(usize, u64)> {
        let len = words.len();
        if start_word >= len {
            return None;
        }

        let mut i = start_word;
        const ALL_ONES: u64 = !0u64;

        while i + 4 <= len {
            if i + WORDS_PER_CACHE_LINE < len {
                prefetch_read(words[i + WORDS_PER_CACHE_LINE..].as_ptr());
            }

            let w0 = words[i].load(Ordering::Relaxed);
            let w1 = words[i + 1].load(Ordering::Relaxed);
            let w2 = words[i + 2].load(Ordering::Relaxed);
            let w3 = words[i + 3].load(Ordering::Relaxed);

            let all_full = w0 & w1 & w2 & w3;
            if all_full != ALL_ONES {
                if w0 != ALL_ONES { return Some((i, w0)); }
                if w1 != ALL_ONES { return Some((i + 1, w1)); }
                if w2 != ALL_ONES { return Some((i + 2, w2)); }
                return Some((i + 3, w3));
            }

            i += 4;
        }

        while i < len {
            let w = words[i].load(Ordering::Relaxed);
            if w != ALL_ONES {
                return Some((i, w));
            }
            i += 1;
        }

        None
    }

    /// Scalar fallback for systems without POPCNT
    fn popcount_scalar(words: &[AtomicU64]) -> u64 {
        words.iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as u64)
            .sum()
    }
}

// ============================================================================
// Scalar Fallback
// ============================================================================

mod scalar {
    use super::*;

    #[inline]
    pub fn popcount_slice(words: &[AtomicU64]) -> u64 {
        let mut total: u64 = 0;
        let mut i = 0;
        let len = words.len();

        // Unroll by 4 for better ILP
        while i + 4 <= len {
            let w0 = words[i].load(Ordering::Relaxed);
            let w1 = words[i + 1].load(Ordering::Relaxed);
            let w2 = words[i + 2].load(Ordering::Relaxed);
            let w3 = words[i + 3].load(Ordering::Relaxed);

            total += w0.count_ones() as u64;
            total += w1.count_ones() as u64;
            total += w2.count_ones() as u64;
            total += w3.count_ones() as u64;

            i += 4;
        }

        while i < len {
            total += words[i].load(Ordering::Relaxed).count_ones() as u64;
            i += 1;
        }

        total
    }

    #[inline]
    pub fn find_first_nonzero(words: &[AtomicU64], start_word: usize) -> Option<(usize, u64)> {
        for i in start_word..words.len() {
            let w = words[i].load(Ordering::Relaxed);
            if w != 0 {
                return Some((i, w));
            }
        }
        None
    }

    #[inline]
    pub fn find_first_not_full(words: &[AtomicU64], start_word: usize) -> Option<(usize, u64)> {
        const ALL_ONES: u64 = !0u64;
        for i in start_word..words.len() {
            let w = words[i].load(Ordering::Relaxed);
            if w != ALL_ONES {
                return Some((i, w));
            }
        }
        None
    }
}

// ============================================================================
// Public API - Auto-dispatches to best implementation
// ============================================================================

/// Count total set bits in a slice of AtomicU64.
/// Automatically uses SIMD when available.
#[inline]
pub fn popcount_slice(words: &[AtomicU64]) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::popcount_slice(words)
    }

    #[cfg(target_arch = "x86_64")]
    {
        avx2::popcount_slice(words)
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::popcount_slice(words)
    }
}

/// Find first non-zero word starting from `start_word`.
/// Returns (word_index, word_value) or None if all zero.
#[inline]
pub fn find_first_nonzero(words: &[AtomicU64], start_word: usize) -> Option<(usize, u64)> {
    #[cfg(target_arch = "aarch64")]
    {
        neon::find_first_nonzero(words, start_word)
    }

    #[cfg(target_arch = "x86_64")]
    {
        avx2::find_first_nonzero(words, start_word)
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::find_first_nonzero(words, start_word)
    }
}

/// Find first word with unset bits starting from `start_word`.
/// Returns (word_index, word_value) or None if all full.
#[inline]
pub fn find_first_not_full(words: &[AtomicU64], start_word: usize) -> Option<(usize, u64)> {
    #[cfg(target_arch = "aarch64")]
    {
        neon::find_first_not_full(words, start_word)
    }

    #[cfg(target_arch = "x86_64")]
    {
        avx2::find_first_not_full(words, start_word)
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::find_first_not_full(words, start_word)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_words(values: &[u64]) -> Vec<AtomicU64> {
        values.iter().map(|&v| AtomicU64::new(v)).collect()
    }

    #[test]
    fn test_popcount_empty() {
        let words = make_words(&[]);
        assert_eq!(popcount_slice(&words), 0);
    }

    #[test]
    fn test_popcount_single() {
        let words = make_words(&[0b1010_1010]);
        assert_eq!(popcount_slice(&words), 4);
    }

    #[test]
    fn test_popcount_multiple() {
        let words = make_words(&[0xFF, 0xFF, 0xFF, 0xFF]); // 8 bits each
        assert_eq!(popcount_slice(&words), 32);
    }

    #[test]
    fn test_popcount_large() {
        // Test with more than 8 words to exercise SIMD paths
        let words = make_words(&[!0u64; 16]); // All bits set
        assert_eq!(popcount_slice(&words), 64 * 16);
    }

    #[test]
    fn test_find_nonzero_empty() {
        let words = make_words(&[]);
        assert_eq!(find_first_nonzero(&words, 0), None);
    }

    #[test]
    fn test_find_nonzero_all_zero() {
        let words = make_words(&[0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(find_first_nonzero(&words, 0), None);
    }

    #[test]
    fn test_find_nonzero_first() {
        let words = make_words(&[1, 0, 0, 0]);
        assert_eq!(find_first_nonzero(&words, 0), Some((0, 1)));
    }

    #[test]
    fn test_find_nonzero_middle() {
        let words = make_words(&[0, 0, 0, 0, 0, 42, 0, 0]);
        assert_eq!(find_first_nonzero(&words, 0), Some((5, 42)));
    }

    #[test]
    fn test_find_nonzero_with_offset() {
        let words = make_words(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(find_first_nonzero(&words, 3), Some((3, 4)));
    }

    #[test]
    fn test_find_not_full_empty() {
        let words = make_words(&[]);
        assert_eq!(find_first_not_full(&words, 0), None);
    }

    #[test]
    fn test_find_not_full_all_full() {
        let words = make_words(&[!0u64; 8]);
        assert_eq!(find_first_not_full(&words, 0), None);
    }

    #[test]
    fn test_find_not_full_first() {
        let words = make_words(&[0, !0u64, !0u64, !0u64]);
        assert_eq!(find_first_not_full(&words, 0), Some((0, 0)));
    }

    #[test]
    fn test_find_not_full_middle() {
        let words = make_words(&[!0u64, !0u64, !0u64, !0u64, !0u64, 42, !0u64, !0u64]);
        assert_eq!(find_first_not_full(&words, 0), Some((5, 42)));
    }

    #[test]
    fn test_prefetch_does_not_crash() {
        let words = make_words(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        prefetch_read(words.as_ptr());
        prefetch_write(words.as_ptr());
        prefetch_next_cacheline(&words, 0);
        prefetch_next_cacheline(&words, 8);
        // Just verify no crash
    }
}
