//! Configuration types for PrefixTracker.
//!
//! This module contains all configuration structs and enums used throughout
//! the prefix_tracker module.

use std::sync::Arc;

/// Configuration for registering a prefix tracker.
#[derive(Clone, Debug)]
pub struct TrackerConfig {
    /// The prefix string (e.g., "vec:", "hash:user:")
    pub prefix: String,

    /// Whether this tracker uses hierarchical (id, sub_id) pairs.
    /// If false, only single u64 IDs are tracked.
    pub hierarchical: bool,

    /// Optional maximum primary ID (exclusive). None = unbounded.
    pub max_id: Option<u64>,

    /// Optional maximum sub_id (exclusive). Only used if hierarchical=true.
    pub max_sub_id: Option<u64>,

    /// Initial capacity hint for primary IDs (in bits).
    pub initial_capacity: usize,
}

// ============================================================================
// Sampling Configuration
// ============================================================================

/// Sampling configuration for iterators.
///
/// Controls how items are selected during iteration, enabling mixed-ratio
/// and percentage-based sampling patterns common in benchmarks.
///
/// # Examples
///
/// ```rust
/// use keyspace_tracker::{SamplingConfig, AccessDistribution};
///
/// // 90% overwrites (existing) + 10% new writes
/// let config = SamplingConfig::new()
///     .with_set_ratio(0.9)
///     .with_seed(42); // reproducible
///
/// // Zipfian distribution (cache-like hot/cold)
/// let config = SamplingConfig::new()
///     .with_distribution(AccessDistribution::Zipfian { skew: 0.99 });
///
/// // Sample ~50% of keys randomly
/// let config = SamplingConfig::new()
///     .with_sample_probability(0.5);
///
/// // Take exactly 1000 items
/// let config = SamplingConfig::new()
///     .with_limit(1000);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    /// Ratio of set bits to return (0.0-1.0).
    /// - 1.0 = only set bits (equivalent to BitFilter::Set)
    /// - 0.0 = only unset bits (equivalent to BitFilter::Unset)
    /// - 0.9 = 90% set, 10% unset (for "90% overwrite + 10% new" patterns)
    /// - None = use BitFilter instead
    pub set_ratio: Option<f64>,

    /// Maximum number of items to return. None = unlimited.
    pub limit: Option<u64>,

    /// Duration limit for iteration (in milliseconds). None = unlimited.
    /// When set, iteration continues until the duration expires.
    /// Combines with `limit` - iteration stops when either is reached.
    pub duration_ms: Option<u64>,

    /// Probabilistic sampling ratio (0.0-1.0).
    /// Each matching item has this probability of being returned.
    /// 1.0 = return all matching, 0.5 = return ~50% of matching.
    pub sample_probability: f64,

    /// Random seed for reproducible iteration. None = random seed.
    pub seed: Option<u64>,

    /// Access distribution for non-uniform key selection.
    /// Only applies to random iteration mode.
    pub distribution: AccessDistribution,

    /// Enable overlapping mode: multiple threads may visit same keys.
    /// Default is false (disjoint iteration for write safety).
    pub overlapping: bool,
}

impl SamplingConfig {
    /// Create default config (all items, no sampling).
    #[inline]
    pub const fn new() -> Self {
        Self {
            set_ratio: None,
            limit: None,
            duration_ms: None,
            sample_probability: 1.0,
            seed: None,
            distribution: AccessDistribution::Uniform,
            overlapping: false,
        }
    }

    /// Set the ratio of set vs unset bits to return.
    ///
    /// This enables mixed-ratio iteration, overriding BitFilter.
    /// - 0.9 means 90% of returned items will be set (existing)
    /// - 0.1 means 10% of returned items will be set
    #[inline]
    pub const fn with_set_ratio(mut self, ratio: f64) -> Self {
        self.set_ratio = Some(ratio);
        self
    }

    /// Limit the number of items returned.
    #[inline]
    pub const fn with_limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set duration limit for iteration (in milliseconds).
    ///
    /// Iteration will continue (with wraparound) until the duration expires.
    /// This enables time-based workloads like "run for 60 seconds".
    #[inline]
    pub const fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Set probabilistic sampling (each item has `prob` chance of being returned).
    ///
    /// Use this to randomly sample a percentage of the keyspace.
    /// For example, 0.5 returns approximately 50% of matching items.
    #[inline]
    pub const fn with_sample_probability(mut self, prob: f64) -> Self {
        self.sample_probability = prob;
        self
    }

    /// Set seed for reproducible random iteration.
    #[inline]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set access distribution for non-uniform key selection.
    ///
    /// This affects how keys are selected in random iteration mode.
    /// Use Zipfian for cache-like workloads, Exponential for session stores, etc.
    #[inline]
    pub const fn with_distribution(mut self, dist: AccessDistribution) -> Self {
        self.distribution = dist;
        self
    }

    /// Enable overlapping mode for contention testing.
    ///
    /// When true, multiple threads may visit the same keys.
    /// Useful for testing lock contention, race conditions.
    #[inline]
    pub const fn with_overlapping(mut self, enabled: bool) -> Self {
        self.overlapping = enabled;
        self
    }

    /// Check if this config uses mixed-ratio mode.
    #[inline]
    pub fn is_mixed_ratio(&self) -> bool {
        matches!(self.set_ratio, Some(r) if r > 0.0 && r < 1.0)
    }

    /// Check if any sampling/limiting is configured.
    #[inline]
    pub fn has_sampling(&self) -> bool {
        self.limit.is_some() 
            || self.duration_ms.is_some()
            || self.sample_probability < 1.0 
            || self.set_ratio.is_some()
            || !matches!(self.distribution, AccessDistribution::Uniform)
    }

    /// Check if non-uniform distribution is configured.
    #[inline]
    pub fn has_distribution(&self) -> bool {
        !matches!(self.distribution, AccessDistribution::Uniform)
    }

    /// Create RNG from seed (or random if no seed).
    #[inline]
    pub fn make_rng(&self) -> fastrand::Rng {
        match self.seed {
            Some(seed) => fastrand::Rng::with_seed(seed),
            None => fastrand::Rng::new(),
        }
    }

    /// Sample an index using the configured distribution.
    #[inline]
    pub fn sample_index(&self, rng: &mut fastrand::Rng, range_size: u64) -> u64 {
        self.distribution.sample(rng, range_size)
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Filter Context - Consolidated filter/sampling logic
// ============================================================================

/// Encapsulates filter and sampling state for consistent behavior across iterators.
///
/// This consolidates the `check_limit()`, `matches_filter()`, and `passes_sampling()`
/// logic that is common to `SequentialIter`, `RandomIter`, `PartitionedIter`, etc.
///
/// # Example
///
/// ```ignore
/// let mut ctx = FilterContext::new(BitFilter::Set, sampling_config);
/// 
/// // In iteration loop:
/// if !ctx.check_limit() { return None; }
/// if !ctx.matches_filter(is_set) { continue; }
/// if !ctx.passes_sampling() { continue; }
/// ctx.record_yield();
/// ```
#[derive(Clone)]
pub struct FilterContext {
    filter: BitFilter,
    /// Sampling configuration (pub for size_hint access in iterators).
    pub(crate) sampling: SamplingConfig,
    rng: fastrand::Rng,
    yielded: u64,
    /// Start time for duration-based iteration (milliseconds since UNIX epoch).
    start_time_ms: Option<u64>,
}

impl FilterContext {
    /// Create a new filter context.
    #[inline]
    pub fn new(filter: BitFilter, sampling: SamplingConfig) -> Self {
        let rng = sampling.make_rng();
        let start_time_ms = sampling.duration_ms.map(|_| Self::current_time_ms());
        Self {
            filter,
            sampling,
            rng,
            yielded: 0,
            start_time_ms,
        }
    }

    /// Get current time in milliseconds since UNIX epoch.
    #[inline]
    fn current_time_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Check if we should continue iterating (respects limit and duration).
    #[inline]
    pub fn check_limit(&self) -> bool {
        // Check count limit
        if let Some(limit) = self.sampling.limit {
            if self.yielded >= limit {
                return false;
            }
        }
        // Check duration limit
        if let (Some(duration_ms), Some(start_time_ms)) = (self.sampling.duration_ms, self.start_time_ms) {
            let elapsed = Self::current_time_ms().saturating_sub(start_time_ms);
            if elapsed >= duration_ms {
                return false;
            }
        }
        true
    }

    /// Check if item matches filter (with mixed-ratio support).
    #[inline]
    pub fn matches_filter(&mut self, is_set: bool) -> bool {
        if let Some(set_ratio) = self.sampling.set_ratio {
            // Mixed-ratio mode: probabilistically select based on ratio
            let want_set = self.rng.f64() < set_ratio;
            want_set == is_set
        } else {
            // Standard filter mode
            match self.filter {
                BitFilter::All => true,
                BitFilter::Set => is_set,
                BitFilter::Unset => !is_set,
            }
        }
    }

    /// Check probabilistic sampling.
    #[inline]
    pub fn passes_sampling(&mut self) -> bool {
        if self.sampling.sample_probability < 1.0 {
            self.rng.f64() < self.sampling.sample_probability
        } else {
            true
        }
    }

    /// Record that an item was yielded. Call after successfully returning an item.
    #[inline]
    pub fn record_yield(&mut self) {
        self.yielded += 1;
    }

    /// Get current yield count.
    #[inline]
    pub fn yielded(&self) -> u64 {
        self.yielded
    }

    /// Sample an index using configured distribution.
    #[inline]
    pub fn sample_index(&mut self, range_size: u64) -> u64 {
        self.sampling.distribution.sample(&mut self.rng, range_size)
    }

    /// Get a random f64 in [0, 1).
    #[inline]
    pub fn random_f64(&mut self) -> f64 {
        self.rng.f64()
    }

    /// Check if using non-uniform distribution.
    #[inline]
    pub fn has_distribution(&self) -> bool {
        self.sampling.has_distribution()
    }

    /// Check if we can use SIMD-accelerated `find_next_set` for iteration.
    /// 
    /// Returns true when filter is `Set` with no mixed-ratio sampling,
    /// enabling O(density) iteration instead of O(n).
    #[inline]
    pub fn can_use_find_next_set(&self) -> bool {
        self.filter == BitFilter::Set 
            && self.sampling.set_ratio.is_none()
            && self.sampling.sample_probability >= 1.0
    }

    /// Check if we can use SIMD-accelerated `find_next_unset` for iteration.
    #[inline]
    pub fn can_use_find_next_unset(&self) -> bool {
        self.filter == BitFilter::Unset 
            && self.sampling.set_ratio.is_none()
            && self.sampling.sample_probability >= 1.0
    }

    /// Get the filter type.
    #[inline]
    pub fn filter(&self) -> BitFilter {
        self.filter
    }

    /// Combined check: filter + sampling + limit, then record yield if passed.
    /// 
    /// Returns true if the item should be yielded. Automatically records the yield.
    /// Use this to simplify the common iteration pattern:
    /// 
    /// ```ignore
    /// // Before: 4 separate calls
    /// if !ctx.check_limit() { return None; }
    /// if !ctx.matches_filter(is_set) { continue; }
    /// if !ctx.passes_sampling() { continue; }
    /// ctx.record_yield();
    /// 
    /// // After: single call
    /// if !ctx.should_yield(is_set) { continue; }
    /// ```
    #[inline]
    pub fn should_yield(&mut self, is_set: bool) -> bool {
        if !self.check_limit() { return false; }
        if !self.matches_filter(is_set) { return false; }
        if !self.passes_sampling() { return false; }
        self.record_yield();
        true
    }
}

impl Default for FilterContext {
    fn default() -> Self {
        Self::new(BitFilter::All, SamplingConfig::new())
    }
}

// ============================================================================
// Access Distribution Patterns
// ============================================================================

/// Access distribution for realistic workload simulation.
///
/// Different workloads exhibit different access patterns:
/// - **Uniform**: Equal probability for all keys (synthetic benchmarks)
/// - **Zipfian**: Few hot keys, long tail of cold keys (caches, social media)
/// - **Exponential**: Recent items accessed more (session stores, time-series)
/// - **Normal**: Clustered around a center point (range queries)
/// - **Hotspot**: Explicit hot/cold partitioning (mixed workloads)
///
/// # Examples
///
/// ```rust
/// use keyspace_tracker::AccessDistribution;
///
/// // Cache-like access (80% of requests hit 20% of keys)
/// let dist = AccessDistribution::Zipfian { skew: 1.0 };
///
/// // Session store (recent sessions accessed more)
/// let dist = AccessDistribution::Exponential { lambda: 0.1 };
///
/// // 10% of keys handle 90% of traffic
/// let dist = AccessDistribution::Hotspot { hot_pct: 0.1, hot_prob: 0.9 };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AccessDistribution {
    /// Equal probability for all keys in range.
    #[default]
    Uniform,

    /// Zipfian distribution: P(k) ∝ 1/k^s where s is skew.
    /// - skew=0.0: uniform
    /// - skew=0.99: typical web cache (~80/20 rule)
    /// - skew=1.2: highly skewed (social media hot posts)
    Zipfian {
        /// Skew parameter (0.0-2.0 typical, higher = more skewed)
        skew: f64,
    },

    /// Exponential decay from start of range.
    /// P(k) ∝ e^(-λk) - recent items more likely.
    /// Useful for session stores, time-series data.
    Exponential {
        /// Decay rate (0.01-1.0 typical, higher = steeper decay)
        lambda: f64,
    },

    /// Normal/Gaussian distribution centered at mean_pct of range.
    /// Useful for range-query workloads.
    Normal {
        /// Center point as percentage of range (0.0-1.0)
        mean_pct: f64,
        /// Standard deviation as percentage of range (0.01-0.5 typical)
        std_pct: f64,
    },

    /// Explicit hotspot: hot_pct of keys receive hot_prob of accesses.
    /// Simple model for mixed hot/cold workloads.
    Hotspot {
        /// Fraction of keys that are "hot" (0.0-1.0)
        hot_pct: f64,
        /// Probability of accessing a hot key (0.0-1.0)
        hot_prob: f64,
    },

    /// Latest-N bias: strongly prefer the most recently added keys.
    /// Models append-heavy workloads like logs, streams.
    Latest {
        /// Percentage of keyspace considered "recent" (0.0-1.0)
        recent_pct: f64,
        /// Probability of accessing recent vs old (0.0-1.0)
        recent_prob: f64,
    },
}

impl AccessDistribution {
    /// Sample an index from this distribution given range [0, n).
    ///
    /// Returns a value in [0, n) biased according to the distribution.
    #[inline]
    pub fn sample(&self, rng: &mut fastrand::Rng, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }

        match self {
            Self::Uniform => rng.u64(0..n),

            Self::Zipfian { skew } => {
                // Approximate Zipfian using rejection sampling
                // For better performance, could use precomputed CDF table
                Self::sample_zipfian(rng, n, *skew)
            }

            Self::Exponential { lambda } => {
                // Inverse CDF: x = -ln(1-u)/λ, clamped to [0,n)
                let u = rng.f64();
                let x = -(1.0 - u).ln() / lambda;
                (x as u64).min(n - 1)
            }

            Self::Normal { mean_pct, std_pct } => {
                // Box-Muller transform for normal distribution
                let mean = *mean_pct * n as f64;
                let std = *std_pct * n as f64;
                let (u1, u2) = (rng.f64(), rng.f64());
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let x = mean + z * std;
                (x.max(0.0) as u64).min(n - 1)
            }

            Self::Hotspot { hot_pct, hot_prob } => {
                let hot_count = (n as f64 * hot_pct).max(1.0) as u64;
                if rng.f64() < *hot_prob {
                    // Access hot region
                    rng.u64(0..hot_count)
                } else {
                    // Access cold region
                    if hot_count < n {
                        hot_count + rng.u64(0..(n - hot_count))
                    } else {
                        rng.u64(0..n)
                    }
                }
            }

            Self::Latest { recent_pct, recent_prob } => {
                let recent_count = (n as f64 * recent_pct).max(1.0) as u64;
                let recent_start = n.saturating_sub(recent_count);
                if rng.f64() < *recent_prob && recent_start < n {
                    // Access recent region (end of range)
                    recent_start + rng.u64(0..recent_count.min(n - recent_start))
                } else {
                    // Access older region
                    if recent_start > 0 {
                        rng.u64(0..recent_start)
                    } else {
                        rng.u64(0..n)
                    }
                }
            }
        }
    }

    /// Approximate Zipfian sampling using rejection method.
    #[inline]
    fn sample_zipfian(rng: &mut fastrand::Rng, n: u64, skew: f64) -> u64 {
        if skew <= 0.0 {
            return rng.u64(0..n);
        }

        // Use inverse transform with approximation
        // CDF ≈ k^(1-s) for Zipf, so inverse is u^(1/(1-s))
        let u = rng.f64();
        if skew >= 1.0 {
            // For s >= 1, use modified formula
            let exp = 1.0 / skew;
            let x = (u * (n as f64).powf(1.0 - exp)).powf(1.0 / (1.0 - exp));
            (x as u64).min(n - 1)
        } else {
            // For s < 1
            let x = ((n as f64).powf(1.0 - skew) * u).powf(1.0 / (1.0 - skew));
            (x as u64).min(n - 1)
        }
    }

    /// Create a Zipfian distribution with typical web cache skew.
    #[inline]
    pub const fn web_cache() -> Self {
        Self::Zipfian { skew: 0.99 }
    }

    /// Create an exponential distribution for session-like access.
    #[inline]
    pub const fn session_store() -> Self {
        Self::Exponential { lambda: 0.1 }
    }

    /// Create a hotspot distribution (X% keys get Y% traffic).
    #[inline]
    pub const fn hotspot(hot_pct: f64, hot_prob: f64) -> Self {
        Self::Hotspot { hot_pct, hot_prob }
    }
}

// ============================================================================
// Fragmentation Patterns (for defrag testing)
// ============================================================================

/// Pattern for creating fragmented keyspaces.
///
/// Used to simulate memory fragmentation scenarios for testing
/// Valkey's defragmentation capabilities.
///
/// # Examples
///
/// ```rust
/// use keyspace_tracker::FragmentationPattern;
///
/// // Every other key deleted (checkerboard pattern)
/// let pattern = FragmentationPattern::Alternating { stride: 2 };
///
/// // Random 30% holes
/// let pattern = FragmentationPattern::Sparse { density: 0.7 };
///
/// // Clusters of 100 keys with gaps
/// let pattern = FragmentationPattern::Clustered { cluster_size: 100, gap_size: 50 };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum FragmentationPattern {
    /// No fragmentation - all keys present.
    #[default]
    None,

    /// Random holes with specified density (0.0-1.0 = fraction present).
    Sparse {
        /// Fraction of keys that should be present (0.0-1.0)
        density: f64,
    },

    /// Every Nth key present (creates regular holes).
    Alternating {
        /// Stride between present keys (2 = every other key)
        stride: u64,
    },

    /// Clusters of adjacent keys with gaps between.
    Clustered {
        /// Number of adjacent keys in each cluster
        cluster_size: u64,
        /// Number of missing keys between clusters
        gap_size: u64,
    },

    /// Simulate aging: older keys (lower IDs) have more holes.
    Aged {
        /// Minimum density at start of range
        min_density: f64,
        /// Maximum density at end of range
        max_density: f64,
    },

    /// Random variable-size holes (realistic fragmentation).
    VariableHoles {
        /// Average hole size
        avg_hole_size: u64,
        /// Average gap between holes
        avg_gap_size: u64,
    },
}

impl FragmentationPattern {
    /// Check if a key at given index should be present.
    #[inline]
    pub fn should_exist(&self, index: u64, max_index: u64, rng: &mut fastrand::Rng) -> bool {
        match self {
            Self::None => true,

            Self::Sparse { density } => rng.f64() < *density,

            Self::Alternating { stride } => index % stride == 0,

            Self::Clustered { cluster_size, gap_size } => {
                let cycle = cluster_size + gap_size;
                (index % cycle) < *cluster_size
            }

            Self::Aged { min_density, max_density } => {
                if max_index == 0 {
                    return true;
                }
                let progress = index as f64 / max_index as f64;
                let density = min_density + (max_density - min_density) * progress;
                rng.f64() < density
            }

            Self::VariableHoles { avg_hole_size, avg_gap_size } => {
                // Simplified: use position in pseudo-random cycle
                let cycle = avg_hole_size + avg_gap_size;
                let pos = index % cycle;
                pos >= *avg_hole_size
            }
        }
    }

    /// Calculate expected density for this pattern.
    pub fn expected_density(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Sparse { density } => *density,
            Self::Alternating { stride } => 1.0 / *stride as f64,
            Self::Clustered { cluster_size, gap_size } => {
                *cluster_size as f64 / (cluster_size + gap_size) as f64
            }
            Self::Aged { min_density, max_density } => (min_density + max_density) / 2.0,
            Self::VariableHoles { avg_hole_size, avg_gap_size } => {
                *avg_gap_size as f64 / (avg_hole_size + avg_gap_size) as f64
            }
        }
    }
}

// ============================================================================
// Workload Profile (Composite Configuration)
// ============================================================================

/// Composite workload profile for realistic benchmark simulation.
///
/// Combines access distribution, operation mix, and fragmentation
/// patterns into a single configuration.
///
/// # Examples
///
/// ```rust
/// use keyspace_tracker::{WorkloadProfile, AccessDistribution, FragmentationPattern};
///
/// // Session store workload
/// let profile = WorkloadProfile::session_store();
///
/// // Cache workload with 90% reads
/// let profile = WorkloadProfile::cache(0.9);
///
/// // Custom mixed workload
/// let profile = WorkloadProfile::new()
///     .with_distribution(AccessDistribution::Zipfian { skew: 1.0 })
///     .with_read_ratio(0.8)
///     .with_fragmentation(FragmentationPattern::Sparse { density: 0.7 });
/// ```
#[derive(Clone, Debug)]
pub struct WorkloadProfile {
    /// Access distribution for key selection.
    pub distribution: AccessDistribution,

    /// Ratio of read operations (0.0-1.0).
    pub read_ratio: f64,

    /// Ratio of write operations to existing keys (overwrite).
    /// Remaining writes go to new keys.
    pub overwrite_ratio: f64,

    /// Ratio of delete operations (from remaining after reads).
    pub delete_ratio: f64,

    /// Fragmentation pattern for initial state.
    pub fragmentation: FragmentationPattern,

    /// Whether keys have TTL (enables expiration simulation).
    pub has_ttl: bool,

    /// Average TTL in "generations" (iteration cycles).
    pub avg_ttl_generations: u64,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for WorkloadProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkloadProfile {
    /// Create a new default workload profile.
    pub const fn new() -> Self {
        Self {
            distribution: AccessDistribution::Uniform,
            read_ratio: 0.5,
            overwrite_ratio: 0.5,
            delete_ratio: 0.0,
            fragmentation: FragmentationPattern::None,
            has_ttl: false,
            avg_ttl_generations: 100,
            seed: None,
        }
    }

    /// Set access distribution.
    #[inline]
    pub const fn with_distribution(mut self, dist: AccessDistribution) -> Self {
        self.distribution = dist;
        self
    }

    /// Set read ratio (0.0-1.0).
    #[inline]
    pub const fn with_read_ratio(mut self, ratio: f64) -> Self {
        self.read_ratio = ratio;
        self
    }

    /// Set overwrite ratio for writes (0.0-1.0).
    #[inline]
    pub const fn with_overwrite_ratio(mut self, ratio: f64) -> Self {
        self.overwrite_ratio = ratio;
        self
    }

    /// Set delete ratio (0.0-1.0).
    #[inline]
    pub const fn with_delete_ratio(mut self, ratio: f64) -> Self {
        self.delete_ratio = ratio;
        self
    }

    /// Set fragmentation pattern.
    #[inline]
    pub const fn with_fragmentation(mut self, pattern: FragmentationPattern) -> Self {
        self.fragmentation = pattern;
        self
    }

    /// Enable TTL simulation.
    #[inline]
    pub const fn with_ttl(mut self, avg_generations: u64) -> Self {
        self.has_ttl = true;
        self.avg_ttl_generations = avg_generations;
        self
    }

    /// Set random seed.
    #[inline]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    // ========================================================================
    // Preset Profiles
    // ========================================================================

    /// Session store: exponential access, high TTL churn.
    pub const fn session_store() -> Self {
        Self {
            distribution: AccessDistribution::Exponential { lambda: 0.1 },
            read_ratio: 0.8,
            overwrite_ratio: 0.3,
            delete_ratio: 0.1,
            fragmentation: FragmentationPattern::None,
            has_ttl: true,
            avg_ttl_generations: 50,
            seed: None,
        }
    }

    /// Cache workload: Zipfian access, configurable read ratio.
    pub const fn cache(read_ratio: f64) -> Self {
        Self {
            distribution: AccessDistribution::Zipfian { skew: 0.99 },
            read_ratio,
            overwrite_ratio: 0.8,
            delete_ratio: 0.0,
            fragmentation: FragmentationPattern::None,
            has_ttl: false,
            avg_ttl_generations: 100,
            seed: None,
        }
    }

    /// Time-series: append-heavy, latest keys hot.
    pub const fn time_series() -> Self {
        Self {
            distribution: AccessDistribution::Latest { recent_pct: 0.1, recent_prob: 0.9 },
            read_ratio: 0.7,
            overwrite_ratio: 0.0, // Append-only
            delete_ratio: 0.05,   // Trim old data
            fragmentation: FragmentationPattern::None,
            has_ttl: true,
            avg_ttl_generations: 200,
            seed: None,
        }
    }

    /// Defragmentation test: creates fragmented keyspace.
    pub const fn defrag_test() -> Self {
        Self {
            distribution: AccessDistribution::Uniform,
            read_ratio: 0.5,
            overwrite_ratio: 0.3,
            delete_ratio: 0.3,
            fragmentation: FragmentationPattern::Sparse { density: 0.6 },
            has_ttl: false,
            avg_ttl_generations: 100,
            seed: None,
        }
    }

    /// Mixed workload: hot/cold data, some deletes.
    pub const fn mixed() -> Self {
        Self {
            distribution: AccessDistribution::Hotspot { hot_pct: 0.2, hot_prob: 0.8 },
            read_ratio: 0.6,
            overwrite_ratio: 0.5,
            delete_ratio: 0.1,
            fragmentation: FragmentationPattern::None,
            has_ttl: false,
            avg_ttl_generations: 100,
            seed: None,
        }
    }

    // ========================================================================
    // Operation Selection
    // ========================================================================

    /// Select operation type based on profile ratios.
    ///
    /// Returns: (is_read, is_overwrite_if_write, is_delete)
    #[inline]
    pub fn select_operation(&self, rng: &mut fastrand::Rng) -> WorkloadOperation {
        let r = rng.f64();

        if r < self.read_ratio {
            WorkloadOperation::Read
        } else {
            let write_delete_ratio = 1.0 - self.read_ratio;
            let delete_threshold = self.read_ratio + (write_delete_ratio * self.delete_ratio);

            if r < delete_threshold {
                WorkloadOperation::Delete
            } else if rng.f64() < self.overwrite_ratio {
                WorkloadOperation::Overwrite
            } else {
                WorkloadOperation::Insert
            }
        }
    }
}

/// Operation type selected from workload profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadOperation {
    /// Read existing key.
    Read,
    /// Write to existing key (overwrite).
    Overwrite,
    /// Write to new key (insert).
    Insert,
    /// Delete existing key.
    Delete,
}

impl WorkloadOperation {
    /// Whether this operation targets existing keys.
    #[inline]
    pub const fn targets_existing(&self) -> bool {
        matches!(self, Self::Read | Self::Overwrite | Self::Delete)
    }

    /// Whether this operation targets non-existing keys.
    #[inline]
    pub const fn targets_new(&self) -> bool {
        matches!(self, Self::Insert)
    }

    /// Whether this operation is a write.
    #[inline]
    pub const fn is_write(&self) -> bool {
        matches!(self, Self::Overwrite | Self::Insert)
    }
}

// ============================================================================
// Snapshot & Diff Support (for benchmark state tracking)
// ============================================================================

/// Immutable snapshot of bitmap state.
///
/// Generic state capture for:
/// - Detecting which keys were added/removed
/// - Backfilling missing keys
/// - Set operations with external reference sets
///
/// # Examples
///
/// ```rust
/// use keyspace_tracker::{PrefixTracker, TrackerConfig};
///
/// let tracker = PrefixTracker::new(TrackerConfig::simple("vec:").with_max_id(1000));
/// tracker.add_range(0, 1000);
///
/// // Take snapshot before workload
/// let snapshot = tracker.snapshot();
///
/// // Run workload that deletes some keys
/// tracker.remove_range(100, 200);
///
/// // Find what was removed
/// let removed = tracker.removed_since(&snapshot);
/// assert_eq!(removed.len(), 100);
/// ```
#[derive(Clone)]
pub struct BitmapSnapshot {
    /// Packed bitmap data (non-atomic copy)
    data: Vec<u64>,
    /// Original capacity
    capacity: usize,
    /// Population count at snapshot time
    count: u64,
}

impl BitmapSnapshot {
    /// Create snapshot from raw data.
    pub(crate) fn from_raw(data: Vec<u64>, capacity: usize, count: u64) -> Self {
        Self { data, capacity, count }
    }

    /// Get capacity of snapshot.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get population count.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Test if a bit is set in snapshot.
    #[inline]
    pub fn test(&self, index: usize) -> bool {
        if index >= self.capacity {
            return false;
        }
        let word_idx = index >> 6;
        let bit_idx = index & 63;
        if word_idx >= self.data.len() {
            return false;
        }
        (self.data[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Iterate all set bits in snapshot.
    pub fn iter_set(&self) -> impl Iterator<Item = u64> + '_ {
        SnapshotSetIter {
            snapshot: self,
            current_word: 0,
            current_bits: if self.data.is_empty() { 0 } else { self.data[0] },
        }
    }

    /// Count set bits in range [start, end).
    pub fn count_range(&self, start: usize, end: usize) -> u64 {
        let end = end.min(self.capacity);
        if start >= end {
            return 0;
        }

        let mut count = 0u64;
        for id in start..end {
            if self.test(id) {
                count += 1;
            }
        }
        count
    }

    /// Get raw data for set operations.
    #[inline]
    pub fn raw_data(&self) -> &[u64] {
        &self.data
    }

    // ========================================================================
    // Set Operations
    // ========================================================================

    /// Compute intersection with another snapshot.
    /// Returns IDs that are set in both.
    pub fn intersection(&self, other: &BitmapSnapshot) -> Vec<u64> {
        // Pre-allocate based on intersection count to avoid reallocations
        let estimated = self.intersection_count(other) as usize;
        let mut result = Vec::with_capacity(estimated);
        let min_words = self.data.len().min(other.data.len());

        for word_idx in 0..min_words {
            let mut both = self.data[word_idx] & other.data[word_idx];
            while both != 0 {
                let bit = both.trailing_zeros() as usize;
                both &= both - 1;
                result.push(((word_idx << 6) + bit) as u64);
            }
        }
        result
    }

    /// Compute difference: IDs in self but not in other.
    pub fn difference(&self, other: &BitmapSnapshot) -> Vec<u64> {
        // Pre-allocate based on difference count to avoid reallocations
        let estimated = self.difference_count(other) as usize;
        let mut result = Vec::with_capacity(estimated);

        for word_idx in 0..self.data.len() {
            let other_word = if word_idx < other.data.len() { other.data[word_idx] } else { 0 };
            let mut diff = self.data[word_idx] & !other_word;
            while diff != 0 {
                let bit = diff.trailing_zeros() as usize;
                diff &= diff - 1;
                result.push(((word_idx << 6) + bit) as u64);
            }
        }
        result
    }

    /// Count intersection size (without allocating).
    pub fn intersection_count(&self, other: &BitmapSnapshot) -> u64 {
        let min_words = self.data.len().min(other.data.len());
        let mut count = 0u64;

        for word_idx in 0..min_words {
            count += (self.data[word_idx] & other.data[word_idx]).count_ones() as u64;
        }
        count
    }

    /// Count difference size (without allocating).
    pub fn difference_count(&self, other: &BitmapSnapshot) -> u64 {
        let mut count = 0u64;

        for word_idx in 0..self.data.len() {
            let other_word = if word_idx < other.data.len() { other.data[word_idx] } else { 0 };
            count += (self.data[word_idx] & !other_word).count_ones() as u64;
        }
        count
    }
}

/// Iterator over set bits in a snapshot.
struct SnapshotSetIter<'a> {
    snapshot: &'a BitmapSnapshot,
    current_word: usize,
    current_bits: u64,
}

impl Iterator for SnapshotSetIter<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_bits != 0 {
                let bit = self.current_bits.trailing_zeros() as usize;
                self.current_bits &= self.current_bits - 1;
                let index = (self.current_word << 6) + bit;
                if index < self.snapshot.capacity {
                    return Some(index as u64);
                }
            }

            self.current_word += 1;
            if self.current_word >= self.snapshot.data.len() {
                return None;
            }
            self.current_bits = self.snapshot.data[self.current_word];
        }
    }
}

impl std::fmt::Debug for BitmapSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitmapSnapshot")
            .field("capacity", &self.capacity)
            .field("count", &self.count)
            .finish()
    }
}

// ============================================================================
// Reference Set (generic external ID set for filtering)
// ============================================================================

/// A set of reference IDs for filtering iteration.
///
/// Generic container for any external ID set - the application
/// decides what these IDs represent (ground truth, query vectors,
/// hot keys, etc.).
///
/// # Examples
///
/// ```rust
/// use keyspace_tracker::ReferenceSet;
///
/// // Application creates reference set from its domain data
/// let important_keys = ReferenceSet::from_iter([10, 20, 30, 40, 50]);
///
/// // O(1) membership test
/// assert!(important_keys.contains(20));
/// assert!(!important_keys.contains(25));
///
/// // Iterate
/// for id in important_keys.iter() {
///     println!("Important key: {}", id);
/// }
/// ```
#[derive(Clone)]
pub struct ReferenceSet {
    /// Bitmap for O(1) membership test
    bitmap: Vec<u64>,
    /// Maximum ID + 1
    max_id: u64,
    /// Count of IDs in set
    count: u64,
}

impl ReferenceSet {
    /// Create empty reference set with given capacity.
    pub fn with_capacity(max_id: u64) -> Self {
        let word_count = ((max_id + 63) / 64) as usize;
        Self {
            bitmap: vec![0u64; word_count],
            max_id,
            count: 0,
        }
    }

    /// Create from iterator of IDs.
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter(ids: impl IntoIterator<Item = u64>) -> Self {
        let ids: Vec<u64> = ids.into_iter().collect();
        let max_id = ids.iter().copied().max().unwrap_or(0) + 1;
        let mut set = Self::with_capacity(max_id);
        for id in ids {
            set.insert(id);
        }
        set
    }

    /// Create from slice of IDs.
    pub fn from_slice(ids: &[u64]) -> Self {
        Self::from_iter(ids.iter().copied())
    }

    /// Insert an ID.
    pub fn insert(&mut self, id: u64) -> bool {
        if id >= self.max_id {
            let new_max = (id + 1).next_power_of_two();
            let new_word_count = ((new_max + 63) / 64) as usize;
            self.bitmap.resize(new_word_count, 0);
            self.max_id = new_max;
        }

        let word_idx = (id >> 6) as usize;
        let bit_idx = id & 63;
        let mask = 1u64 << bit_idx;

        if (self.bitmap[word_idx] & mask) == 0 {
            self.bitmap[word_idx] |= mask;
            self.count += 1;
            true
        } else {
            false
        }
    }

    /// Remove an ID.
    pub fn remove(&mut self, id: u64) -> bool {
        if id >= self.max_id {
            return false;
        }

        let word_idx = (id >> 6) as usize;
        let bit_idx = id & 63;
        let mask = 1u64 << bit_idx;

        if (self.bitmap[word_idx] & mask) != 0 {
            self.bitmap[word_idx] &= !mask;
            self.count -= 1;
            true
        } else {
            false
        }
    }

    /// Check if ID is in set. O(1).
    #[inline]
    pub fn contains(&self, id: u64) -> bool {
        if id >= self.max_id {
            return false;
        }
        let word_idx = (id >> 6) as usize;
        let bit_idx = id & 63;
        (self.bitmap[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Get number of IDs in set.
    #[inline]
    pub fn len(&self) -> u64 {
        self.count
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get maximum ID capacity.
    #[inline]
    pub fn max_id(&self) -> u64 {
        self.max_id
    }

    /// Get raw bitmap data.
    #[inline]
    pub fn raw_data(&self) -> &[u64] {
        &self.bitmap
    }

    /// Iterate over all IDs in set.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        ReferenceSetIter {
            bitmap: &self.bitmap,
            max_id: self.max_id,
            current_word: 0,
            current_bits: if self.bitmap.is_empty() { 0 } else { self.bitmap[0] },
        }
    }

    // ========================================================================
    // Set Operations with Snapshots
    // ========================================================================

    /// Count how many IDs from this set exist in the snapshot.
    pub fn count_existing_in(&self, snapshot: &BitmapSnapshot) -> u64 {
        let min_words = self.bitmap.len().min(snapshot.data.len());
        let mut count = 0u64;

        for i in 0..min_words {
            count += (self.bitmap[i] & snapshot.data[i]).count_ones() as u64;
        }
        count
    }

    /// Count how many IDs from this set are missing in the snapshot.
    #[inline]
    pub fn count_missing_in(&self, snapshot: &BitmapSnapshot) -> u64 {
        self.count - self.count_existing_in(snapshot)
    }

    /// Get IDs from this set that are missing in the snapshot.
    pub fn missing_in(&self, snapshot: &BitmapSnapshot) -> Vec<u64> {
        let mut missing = Vec::new();
        let min_words = self.bitmap.len().min(snapshot.data.len());

        // Process words that exist in both
        for word_idx in 0..min_words {
            let mut diff = self.bitmap[word_idx] & !snapshot.data[word_idx];
            while diff != 0 {
                let bit = diff.trailing_zeros() as usize;
                diff &= diff - 1;
                missing.push(((word_idx << 6) + bit) as u64);
            }
        }

        // Any words beyond snapshot are all missing
        for word_idx in min_words..self.bitmap.len() {
            let mut bits = self.bitmap[word_idx];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                missing.push(((word_idx << 6) + bit) as u64);
            }
        }

        missing
    }

    /// Get IDs from this set that exist in the snapshot.
    pub fn existing_in(&self, snapshot: &BitmapSnapshot) -> Vec<u64> {
        let mut existing = Vec::new();
        let min_words = self.bitmap.len().min(snapshot.data.len());

        for word_idx in 0..min_words {
            let mut both = self.bitmap[word_idx] & snapshot.data[word_idx];
            while both != 0 {
                let bit = both.trailing_zeros() as usize;
                both &= both - 1;
                existing.push(((word_idx << 6) + bit) as u64);
            }
        }
        existing
    }

    // ========================================================================
    // Set Operations with Other ReferenceSets
    // ========================================================================

    /// Intersection with another ReferenceSet.
    pub fn intersection(&self, other: &ReferenceSet) -> ReferenceSet {
        let min_words = self.bitmap.len().min(other.bitmap.len());
        let mut result = ReferenceSet::with_capacity(self.max_id.min(other.max_id));

        for word_idx in 0..min_words {
            result.bitmap[word_idx] = self.bitmap[word_idx] & other.bitmap[word_idx];
            result.count += result.bitmap[word_idx].count_ones() as u64;
        }
        result
    }

    /// Union with another ReferenceSet.
    pub fn union(&self, other: &ReferenceSet) -> ReferenceSet {
        let max_words = self.bitmap.len().max(other.bitmap.len());
        let mut result = ReferenceSet::with_capacity(self.max_id.max(other.max_id));

        for word_idx in 0..max_words {
            let a = if word_idx < self.bitmap.len() { self.bitmap[word_idx] } else { 0 };
            let b = if word_idx < other.bitmap.len() { other.bitmap[word_idx] } else { 0 };
            result.bitmap[word_idx] = a | b;
            result.count += result.bitmap[word_idx].count_ones() as u64;
        }
        result
    }

    /// Difference: IDs in self but not in other.
    pub fn difference(&self, other: &ReferenceSet) -> ReferenceSet {
        let mut result = ReferenceSet::with_capacity(self.max_id);

        for word_idx in 0..self.bitmap.len() {
            let other_word = if word_idx < other.bitmap.len() { other.bitmap[word_idx] } else { 0 };
            result.bitmap[word_idx] = self.bitmap[word_idx] & !other_word;
            result.count += result.bitmap[word_idx].count_ones() as u64;
        }
        result
    }
}

impl std::fmt::Debug for ReferenceSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReferenceSet")
            .field("count", &self.count)
            .field("max_id", &self.max_id)
            .finish()
    }
}

/// Iterator over ReferenceSet.
struct ReferenceSetIter<'a> {
    bitmap: &'a [u64],
    max_id: u64,
    current_word: usize,
    current_bits: u64,
}

impl Iterator for ReferenceSetIter<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_bits != 0 {
                let bit = self.current_bits.trailing_zeros() as usize;
                self.current_bits &= self.current_bits - 1;
                let index = ((self.current_word << 6) + bit) as u64;
                if index < self.max_id {
                    return Some(index);
                }
            }

            self.current_word += 1;
            if self.current_word >= self.bitmap.len() {
                return None;
            }
            self.current_bits = self.bitmap[self.current_word];
        }
    }
}

impl TrackerConfig {
    /// Create a simple (non-hierarchical) tracker configuration.
    #[inline]
    pub fn simple(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            hierarchical: false,
            max_id: None,
            max_sub_id: None,
            initial_capacity: 4096,
        }
    }

    /// Create a hierarchical tracker configuration for (id, sub_id) pairs.
    #[inline]
    pub fn hierarchical(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            hierarchical: true,
            max_id: None,
            max_sub_id: None,
            initial_capacity: 4096,
        }
    }

    /// Set maximum primary ID (exclusive).
    #[inline]
    pub fn with_max_id(mut self, max_id: u64) -> Self {
        self.max_id = Some(max_id);
        self
    }

    /// Set maximum sub_id (exclusive). Only meaningful for hierarchical trackers.
    #[inline]
    pub fn with_max_sub_id(mut self, max_sub_id: u64) -> Self {
        self.max_sub_id = Some(max_sub_id);
        self
    }

    /// Set initial capacity (in bits).
    #[inline]
    pub fn with_initial_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = capacity;
        self
    }
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            prefix: String::new(),
            hierarchical: false,
            max_id: None,
            max_sub_id: None,
            initial_capacity: 4096,
        }
    }
}

/// Filter for which bits to iterate over.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BitFilter {
    /// Iterate all IDs in range regardless of set/unset state.
    #[default]
    All,
    /// Only iterate over set (existing) IDs.
    Set,
    /// Only iterate over unset (non-existing) IDs.
    Unset,
}

/// Iteration order for group-level iteration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IterOrder {
    /// All IDs for prefix₀, then prefix₁, etc.
    #[default]
    Horizontal,
    /// ID₀ across all prefixes, then ID₁, etc.
    Vertical,
    /// Shuffled across entire space.
    Random,
}

/// Weighting strategy for prefix selection in group iteration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefixWeight {
    /// Equal probability per prefix.
    #[default]
    Uniform,
    /// Proportional to tracker.capacity().
    ByCapacity,
    /// Proportional to tracker.count() (set bits).
    ByCount,
    /// Proportional to capacity - count (unset bits).
    ByUnsetCount,
}

/// Policy for selecting next prefix in write/delete group iterators.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClaimPolicy {
    /// Cycle through prefixes in order.
    #[default]
    RoundRobin,
    /// Random prefix selection (uniform).
    Random,
    /// Weighted random selection.
    WeightedRandom(PrefixWeight),
    /// Prefer prefix with most unset bits.
    LeastLoaded,
    /// Prefer prefix with most set bits.
    MostLoaded,
}

/// Range specification for iteration.
///
/// For simple trackers, only `id_min` and `id_max` are used.
/// For hierarchical trackers, all four bounds apply.
#[derive(Clone, Copy, Debug)]
pub struct IdRange {
    /// Minimum primary ID (inclusive).
    pub id_min: u64,
    /// Maximum primary ID (exclusive).
    pub id_max: u64,
    /// Minimum sub_id (inclusive).
    pub sub_id_min: u64,
    /// Maximum sub_id (exclusive).
    pub sub_id_max: u64,
}

impl IdRange {
    /// Create a new range with specified bounds.
    #[inline]
    pub fn new(id_range: (u64, u64), sub_id_range: (u64, u64)) -> Self {
        Self {
            id_min: id_range.0,
            id_max: id_range.1,
            sub_id_min: sub_id_range.0,
            sub_id_max: sub_id_range.1,
        }
    }

    /// Create an unbounded range (0 to u64::MAX for both dimensions).
    #[inline]
    pub fn unbounded() -> Self {
        Self {
            id_min: 0,
            id_max: u64::MAX,
            sub_id_min: 0,
            sub_id_max: u64::MAX,
        }
    }

    /// Create a range for simple (non-hierarchical) trackers.
    #[inline]
    pub fn simple(min: u64, max: u64) -> Self {
        Self {
            id_min: min,
            id_max: max,
            sub_id_min: 0,
            sub_id_max: u64::MAX,
        }
    }

    /// Check if an (id, sub_id) pair is within this range.
    #[inline]
    pub fn contains(&self, id: u64, sub_id: u64) -> bool {
        id >= self.id_min
            && id < self.id_max
            && sub_id >= self.sub_id_min
            && sub_id < self.sub_id_max
    }

    /// Check if a simple id is within range.
    #[inline]
    pub fn contains_id(&self, id: u64) -> bool {
        id >= self.id_min && id < self.id_max
    }
}

impl Default for IdRange {
    fn default() -> Self {
        Self::unbounded()
    }
}

// ============================================================================
// Iteration Cursor
// ============================================================================

/// Cursor for iterating over (id, sub_id) pairs within a range.
///
/// Unifies the common pattern of iterating through IDs, handling both
/// simple (id-only) and hierarchical (id, sub_id) cases.
#[derive(Clone, Debug)]
pub struct IdCursor {
    /// Current primary ID position.
    pub id: u64,
    /// Current sub_id position (for hierarchical iteration).
    pub sub_id: u64,
    /// Minimum sub_id to reset to when advancing id.
    pub sub_id_min: u64,
}

impl IdCursor {
    /// Create a new cursor starting at the given range minimum.
    #[inline]
    pub fn new(range: &IdRange) -> Self {
        Self {
            id: range.id_min,
            sub_id: range.sub_id_min,
            sub_id_min: range.sub_id_min,
        }
    }

    /// Advance to next sub_id, wrapping to next id when sub_id_max reached.
    /// Returns true if advanced within bounds, false if exhausted.
    #[inline]
    pub fn advance_sub(&mut self, sub_id_max: u64, id_max: u64) -> bool {
        self.sub_id += 1;
        if self.sub_id >= sub_id_max {
            self.sub_id = self.sub_id_min;
            self.id += 1;
        }
        self.id < id_max
    }

    /// Advance to next id (simple mode).
    /// Returns true if within bounds, false if exhausted.
    #[inline]
    pub fn advance_id(&mut self, id_max: u64) -> bool {
        self.id += 1;
        self.id < id_max
    }

    /// Get current position as (id, sub_id).
    #[inline]
    pub fn position(&self) -> (u64, u64) {
        (self.id, self.sub_id)
    }

    /// Reset to start of range.
    #[inline]
    pub fn reset(&mut self, range: &IdRange) {
        self.id = range.id_min;
        self.sub_id = range.sub_id_min;
        self.sub_id_min = range.sub_id_min;
    }

    /// Move directly to a position.
    #[inline]
    pub fn seek(&mut self, id: u64, sub_id: u64) {
        self.id = id;
        self.sub_id = sub_id;
    }
}

/// Filter for selecting which prefixes to iterate in group iteration.
#[derive(Default)]
pub enum PrefixFilter {
    /// Include all registered prefixes.
    #[default]
    All,
    /// Include only these exact prefixes.
    Exact(Vec<String>),
    /// Include prefixes matching this predicate.
    Predicate(Arc<dyn Fn(&str) -> bool + Send + Sync>),
}

impl std::fmt::Debug for PrefixFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::All => write!(f, "PrefixFilter::All"),
            Self::Exact(v) => write!(f, "PrefixFilter::Exact({:?})", v),
            Self::Predicate(_) => write!(f, "PrefixFilter::Predicate(<fn>)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_config_simple() {
        let config = TrackerConfig::simple("vec:");
        assert_eq!(config.prefix, "vec:");
        assert!(!config.hierarchical);
        assert!(config.max_id.is_none());
    }

    #[test]
    fn test_tracker_config_hierarchical() {
        let config = TrackerConfig::hierarchical("hash:user:")
            .with_max_id(1_000_000)
            .with_max_sub_id(1000);

        assert_eq!(config.prefix, "hash:user:");
        assert!(config.hierarchical);
        assert_eq!(config.max_id, Some(1_000_000));
        assert_eq!(config.max_sub_id, Some(1000));
    }

    #[test]
    fn test_id_range_contains() {
        let range = IdRange::new((10, 100), (0, 50));

        assert!(range.contains(10, 0));
        assert!(range.contains(50, 25));
        assert!(range.contains(99, 49));
        assert!(!range.contains(9, 0));
        assert!(!range.contains(100, 0));
        assert!(!range.contains(50, 50));
    }

    #[test]
    fn test_id_range_unbounded() {
        let range = IdRange::unbounded();

        assert!(range.contains(0, 0));
        assert!(range.contains(u64::MAX - 1, u64::MAX - 1));
    }
}
