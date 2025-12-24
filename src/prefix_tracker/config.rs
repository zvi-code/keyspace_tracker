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

/// Filter for selecting which prefixes to iterate in group iteration.
pub enum PrefixFilter {
    /// Include all registered prefixes.
    All,
    /// Include only these exact prefixes.
    Exact(Vec<String>),
    /// Include prefixes matching this predicate.
    Predicate(Arc<dyn Fn(&str) -> bool + Send + Sync>),
}

impl Default for PrefixFilter {
    fn default() -> Self {
        Self::All
    }
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
