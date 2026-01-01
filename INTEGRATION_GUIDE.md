I'll update the specification to reflect these capabilities without specifying implementation details.

# Keyspace Tracker Integration Specification

Design specification for integrating `keyspace_tracker` with vector database benchmarking tools.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Concepts](#core-concepts)
4. [Integration Points](#integration-points)
5. [Use Cases](#use-cases)
6. [API Reference](#api-reference)
7. [Implementation Guide](#implementation-guide)

---

## Overview

### Purpose

The `keyspace_tracker` crate provides efficient, thread-safe tracking of key existence and flexible iteration patterns for database benchmarking. It enables:

- **State tracking**: Know which keys exist without querying the database
- **Scan-based initialization**: Populate tracker state from database scans
- **Workload generation**: Iterate keys with realistic access patterns (Zipfian, hotspot, etc.)
- **Multi-threaded coordination**: Partition keyspace across threads without overlap
- **Slot-aware tracking**: Optional per-slot or slot-range trackers for cluster environments
- **Reference set protection**: Mark subsets of keys (e.g., ground truth) for protection during operations
- **Benchmark lifecycle**: Track state changes across load/workload/query phases

### Design Principles

1. **Domain-agnostic**: Tracker provides primitives; application assigns semantic meaning
2. **Zero-allocation hot paths**: Iteration uses bitmaps, not heap allocations
3. **Lock-free operations**: Atomic bitmap operations for concurrent access
4. **Composable**: Build complex workloads from simple primitives
5. **Cluster-friendly**: Support slot-based distribution without requiring hashtags

### Key Abstractions

| Abstraction | Purpose |
|-------------|---------|
| `PrefixTracker` | Tracks existence of IDs under a key prefix |
| `TrackerGroups` | Manages multiple trackers (by prefix, by slot range, etc.) |
| `BitmapSnapshot` | Immutable state capture for before/after comparison |
| `ReferenceSet` | External ID set for filtering and protection (replaces protected ID lists) |
| `AccessDistribution` | Statistical distribution for realistic access patterns |
| `SamplingConfig` | Controls iteration behavior (limits, ratios, filters) |

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Benchmark Application                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  vec-load   │  │  vec-query  │  │ vec-delete  │  │ vec-update  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
└─────────┼────────────────┼────────────────┼────────────────┼────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         TrackerGroups                                │
│                                                                      │
│  Option A: Single tracker (standalone or uniform distribution)       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ data_tracker: prefix="vec:", max_id=10M                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Option B: Per-slot or slot-range trackers (cluster mode)            │
│  ┌──────────────┐  ┌──────────────┐       ┌──────────────┐          │
│  │ slot_0_4095  │  │ slot_4096_   │  ...  │ slot_12288_  │          │
│  │              │  │ 8191         │       │ 16383        │          │
│  └──────────────┘  └──────────────┘       └──────────────┘          │
│                                                                      │
│  Reference Sets (protection, filtering)                              │
│  ┌─────────────────┐  ┌─────────────────┐                           │
│  │ ground_truth    │  │ hot_keys        │                           │
│  │ (protected)     │  │ (priority)      │                           │
│  └─────────────────┘  └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AtomicBitmap                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ [0101101001...] Lock-free, SIMD-accelerated, thread-safe     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Distribution Model

Keys are distributed across cluster slots by hashing the full key (no hashtag required):

```
Key: "vec:000000000042"
      └───────┬───────┘
              │
         CRC16 hash
              │
              ▼
         slot = 8734
```

The tracker can operate in two modes:

| Mode | Structure | Use Case |
|------|-----------|----------|
| **Unified** | Single tracker for all IDs | Standalone, or when slot distribution is uniform |
| **Slot-partitioned** | Multiple trackers by slot/range | Per-node state tracking, slot-aware operations |

### Thread Model

```
┌────────────────────────────────────────────────────────────────────┐
│                      Orchestrator Thread                            │
│  - Runs SCAN to discover existing keys                             │
│  - Initializes tracker(s) from scan results                        │
│  - Loads reference sets (ground truth, hot keys)                   │
│  - Takes snapshots before/after workloads                          │
│  - Computes metrics using reference set operations                 │
└────────────────────────────────────────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Worker 0    │      │  Worker 1    │      │  Worker N    │
│              │      │              │      │              │
│ Option A:    │      │ Option A:    │      │ Option A:    │
│ partition by │      │ partition by │      │ partition by │
│ thread index │      │ thread index │      │ thread index │
│              │      │              │      │              │
│ Option B:    │      │ Option B:    │      │ Option B:    │
│ partition by │      │ partition by │      │ partition by │
│ slot range   │      │ slot range   │      │ slot range   │
└──────────────┘      └──────────────┘      └──────────────┘
       │                      │                      │
       └──────────────────────┴──────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Shared Tracker  │
                    │  (lock-free ops) │
                    └──────────────────┘
```

---

## Core Concepts

### Key Identification

Keys are decomposed for efficient tracking:

```
Key: "vec:000000000042"
      ├─┬─┴──────┬──────┘
      │ │        │
   prefix       id (u64)
```

For slot-aware mode, the full key determines slot assignment:

```
Key: "vec:000000000042"
              │
         CRC16(key) % 16384
              │
              ▼
         slot = 8734  →  tracker for slot range [8192, 12287]
```

### Existence States

Each ID has one of four states relative to the tracker:

| State | In Tracker | In Database | Use Case |
|-------|------------|-------------|----------|
| **Set** | ✓ | ✓ | Normal existing key |
| **Unset** | ✗ | ✗ | Key never created or deleted |
| **Stale** | ✓ | ✗ | Tracker thinks exists, but deleted externally |
| **Missing** | ✗ | ✓ | Key exists but tracker unaware |

**Scan-based initialization** eliminates Stale/Missing states by synchronizing tracker with actual database state.

### Reference Sets for Protection and Filtering

`ReferenceSet` provides efficient membership testing and set operations for any collection of IDs:

| Use Case | Description |
|----------|-------------|
| **Ground truth protection** | IDs that must not be deleted during workloads |
| **Hot key identification** | IDs that should receive prioritized access |
| **Query vector set** | IDs to iterate for query workloads |
| **Backfill targets** | IDs that need restoration after workloads |

Reference sets support:
- O(1) membership testing
- Intersection/difference with tracker snapshots
- Iteration in sorted order
- Set algebra (union, intersection, difference)

### Access Distributions

Statistical distributions for realistic workload simulation:

| Distribution | Parameters | Use Case | Access Pattern |
|--------------|------------|----------|----------------|
| `Uniform` | - | Baseline testing | Equal probability |
| `Zipfian` | `skew: f64` | Web cache, popular items | ~80% traffic to ~20% keys |
| `Exponential` | `lambda: f64` | Session stores | Recent keys accessed more |
| `Hotspot` | `hot_pct, hot_prob` | Cache hot keys | X% keys get Y% traffic |
| `Normal` | `mean_pct, std_pct` | Range queries | Bell curve around mean |
| `Latest` | `recent_pct, recent_prob` | Append workloads | Newest keys most active |

### Iteration Modes

| Mode | Method | Thread-Safe | Use Case |
|------|--------|-------------|----------|
| **Sequential** | `.sequential()` | Yes (partitioned) | Cache warming, full scan |
| **Random** | `.random()` | Yes (partitioned) | Random access patterns |
| **Overlapping** | `.overlapping().random()` | Yes | Contention testing |
| **Claim** | `.claim()` | Yes (atomic) | Exactly-once processing |

### Partitioning Strategies

| Strategy | Method | Use Case |
|----------|--------|----------|
| **Independent workers** | `.partition(index, total)` | Each worker computes its own shard |
| **Central distribution** | `.partitioned(n)` | Coordinator distributes partitions |
| **Slot-based** | Multiple trackers by slot range | Align with cluster topology |
| **Range-based** | `.id_range(start, end)` | Process specific ID ranges |

**Independent Worker Pattern** (preferred for benchmarks):
```rust
// Each worker independently creates its partition - no coordination needed
for (id, _) in tracker.iter()
    .set_only()
    .partition(worker_id, num_workers)  // Returns PartitionedIter directly
{
    // Process this worker's disjoint range
}
```

**Central Distribution Pattern**:
```rust
// Coordinator creates all partitions, then distributes
let partitions = tracker.iter().set_only().partitioned(num_workers);
for (worker_id, partition) in partitions.into_iter().enumerate() {
    // Send partition to worker
}
```

---

## Integration Points

### Benchmark Phases

```
Phase 1: INITIALIZATION
├── Option A: Cold start (no existing data)
│   └── Create empty tracker(s)
├── Option B: Warm start (existing data)
│   ├── Run SCAN to discover existing keys
│   ├── Parse IDs from scanned keys
│   └── Populate tracker(s) from scan results
├── Load reference sets from dataset
│   ├── Ground truth neighbors → ReferenceSet (for protection)
│   └── Query vector IDs → ReferenceSet (for iteration)
└── Take initial snapshot (optional)

Phase 2: DATA LOADING (vec-load)
├── Iterate: tracker.iter().unset_only().partitioned(...)
├── For each ID: INSERT into database, tracker.add(id)
└── Bulk: tracker.add_range(start, end)

Phase 3: WORKLOAD EXECUTION (vec-delete, vec-update, mixed)
├── Take pre-workload snapshot
├── Execute workload operations
│   ├── Delete: check reference set, skip if protected, tracker.remove(id)
│   ├── Update: (no tracker change, key still exists)
│   └── Insert: tracker.add(id)
└── Compute diff: added_since(), removed_since()

Phase 4: PRE-QUERY VALIDATION
├── Check reference set coverage in current state
├── Option: Backfill missing reference set members
└── Option: Abort if coverage below threshold

Phase 5: QUERY BENCHMARK (vec-query)
├── Iterate query vectors
├── Execute searches
├── Compare results to reference set (ground truth)
└── Compute recall metrics

Phase 6: CLEANUP
├── Delete index (optional)
├── Clear keys (optional)
└── Report final statistics
```

### Scan-Based Initialization

The tracker supports initialization from database scans:

```
SCAN 0 MATCH "vec:*" COUNT 10000
  │
  ▼
Parse each key → extract ID
  │
  ▼
tracker.add(id) for each discovered key
  │
  ▼
Tracker state = actual database state
```

For slot-aware mode:
```
For each slot range [start, end]:
  SCAN on node owning slots
    │
    ▼
  Parse keys → extract IDs
    │
    ▼
  slot_tracker[range].add(id)
```

### CLI Integration Points

```bash
# New CLI options for keyspace tracking
--keyspace-tracker           # Enable keyspace tracking
--tracker-init <MODE>        # cold|scan (default: cold)
--reference-set-backfill     # Restore missing reference set members before query
--reference-set-check        # Verify reference set coverage, report status
--workload-snapshot          # Take snapshots before/after workload phases
--protect-reference-set      # Skip reference set members during delete operations
```

---

## Use Cases

### Use Case 1: Scan and Initialize Tracker

**Scenario**: Database has existing data, need to sync tracker before operations.

```rust
use keyspace_tracker::{PrefixTracker, TrackerConfig};

// Create tracker
let tracker = PrefixTracker::new(
    TrackerConfig::simple("vec:").with_max_id(10_000_000)
);

// Scan database and populate tracker
let mut cursor = 0;
loop {
    let (next_cursor, keys) = valkey.scan(cursor, "vec:*", 10000);
    
    for key in keys {
        if let Some(id) = parse_id_from_key(&key) {
            tracker.add(id);
        }
    }
    
    if next_cursor == 0 {
        break;
    }
    cursor = next_cursor;
}

println!("Discovered {} existing keys", tracker.count());
```

### Use Case 2: Reference Set Protection During Deletes

**Scenario**: Delete keys but protect ground truth vectors needed for recall measurement.

```rust
use keyspace_tracker::{PrefixTracker, ReferenceSet, AccessDistribution};

// Build reference set from ground truth
let mut protected = ReferenceSet::with_capacity(dataset.num_vectors() as u64);
for query_idx in 0..dataset.num_queries() {
    for neighbor_id in dataset.ground_truth(query_idx) {
        protected.insert(neighbor_id as u64);
    }
}
println!("Protected {} ground truth vectors", protected.len());

// Delete with protection
let mut deleted = 0u64;
let mut skipped = 0u64;

for (id, _) in tracker.iter()
    .set_only()
    .distribution(AccessDistribution::Zipfian { skew: 0.99 })
    .sample(0.1)  // Delete ~10% of keys
    .random()
{
    // Skip protected IDs
    if protected.contains(id) {
        skipped += 1;
        continue;
    }
    
    valkey.del(format!("vec:{}", id));
    tracker.remove(id);
    deleted += 1;
}

println!("Deleted: {}, Skipped (protected): {}", deleted, skipped);
```

### Use Case 3: Reference Set Coverage Check

**Scenario**: Verify ground truth coverage before running queries.

```rust
let snapshot = tracker.snapshot();

// Count coverage
let existing = protected.count_existing_in(&snapshot);
let missing = protected.count_missing_in(&snapshot);
let total = protected.len();
let coverage = existing as f64 / total as f64;

println!("Reference Set Coverage:");
println!("  Existing: {} ({:.1}%)", existing, coverage * 100.0);
println!("  Missing:  {} ({:.1}%)", missing, (1.0 - coverage) * 100.0);

// Abort if coverage too low
if coverage < 0.95 {
    eprintln!("ERROR: Coverage below 95%, recall measurement invalid");
    std::process::exit(1);
}
```

### Use Case 4: Reference Set Backfill

**Scenario**: Restore missing reference set members before query phase.

```rust
let snapshot = tracker.snapshot();
let missing_ids = protected.missing_in(&snapshot);

println!("Backfilling {} missing reference set members", missing_ids.len());

for id in missing_ids {
    let vector = dataset.get_vector(id as usize);
    valkey.hset(format!("vec:{}", id), "embedding", vector);
    tracker.add(id);
}

// Verify complete
let new_snapshot = tracker.snapshot();
assert_eq!(protected.count_missing_in(&new_snapshot), 0);
```

### Use Case 5: Slot-Aware Tracking (Cluster Mode)

**Scenario**: Track keys per slot range for cluster-aware operations.

```rust
use keyspace_tracker::{TrackerGroups, TrackerConfig};

// Create tracker group with slot-range partitioning
let groups = TrackerGroups::new();

// Register trackers for slot ranges (example: 4 ranges)
let slot_ranges = [
    (0, 4095),
    (4096, 8191),
    (8192, 12287),
    (12288, 16383),
];

let trackers: Vec<_> = slot_ranges.iter()
    .map(|(start, end)| {
        groups.register(
            TrackerConfig::simple("vec:")
                .with_max_id(max_ids_per_range)
                .with_slot_range(*start, *end)
        )
    })
    .collect();

// Initialize each tracker from its node's scan
for (i, (slot_start, slot_end)) in slot_ranges.iter().enumerate() {
    let node = cluster.node_for_slot(*slot_start);
    
    // Scan on specific node
    for key in node.scan("vec:*") {
        let id = parse_id(&key);
        let slot = crc16(&key) % 16384;
        
        if slot >= *slot_start && slot <= *slot_end {
            trackers[i].add(id);
        }
    }
}

// Operations automatically route to correct tracker
fn get_tracker_for_key(key: &str) -> &PrefixTracker {
    let slot = crc16(key) % 16384;
    let range_idx = (slot / 4096) as usize;
    &trackers[range_idx]
}
```

### Use Case 6: Multi-Threaded Load with Partitioning

**Scenario**: Load vectors across multiple threads with disjoint key assignment.

```rust
use std::sync::Arc;
use std::thread;

let tracker = Arc::new(tracker);
let num_threads = 16;

let handles: Vec<_> = (0..num_threads).map(|thread_id| {
    let tracker = tracker.clone();
    let dataset = dataset.clone();
    
    thread::spawn(move || {
        let mut count = 0u64;
        
        // Each thread independently computes its own disjoint range
        for (id, _) in tracker.iter()
            .unset_only()
            .partition(thread_id, num_threads)  // Returns PartitionedIter directly
        {
            let vector = dataset.get_vector(id as usize);
            valkey.hset(format!("vec:{}", id), "embedding", vector);
            tracker.add(id);
            count += 1;
        }
        
        count
    })
}).collect();

let total: u64 = handles.into_iter()
    .map(|h| h.join().unwrap())
    .sum();
    
println!("Loaded {} vectors across {} threads", total, num_threads);
```

### Use Case 7: Workload with Snapshot Diff

**Scenario**: Track changes during workload execution.

```rust
// Snapshot before workload
let pre_workload = tracker.snapshot();
println!("Pre-workload: {} keys", pre_workload.count());

// Execute mixed workload
let profile = WorkloadProfile::cache(0.8);  // 80% reads
for _ in 0..100_000 {
    match profile.select_operation(&mut rng) {
        WorkloadOperation::Read => { /* ... */ }
        WorkloadOperation::Insert => {
            // Find unused ID and insert
            for (id, _) in tracker.iter().unset_only().limit(1).random() {
                valkey.hset(format!("vec:{}", id), "embedding", random_vector());
                tracker.add(id);
            }
        }
        WorkloadOperation::Delete => {
            for (id, _) in tracker.iter().set_only().limit(1).random() {
                if !protected.contains(id) {
                    valkey.del(format!("vec:{}", id));
                    tracker.remove(id);
                }
            }
        }
        _ => {}
    }
}

// Compute changes
let added = tracker.added_count_since(&pre_workload);
let removed = tracker.removed_count_since(&pre_workload);
println!("Workload complete: +{} -{}", added, removed);
```

### Use Case 8: Concurrent Claim Workers with Shared Cursor

**Scenario**: Multiple workers atomically claim unique IDs without coordination, using `continue_write()`.

When workers independently create iterators via `write()`, each call resets the cursor to 0, causing duplicate claims. Use `continue_write()` to share cursor state across workers:

```rust
use std::sync::Arc;
use std::thread;

let tracker = Arc::new(tracker);
let num_workers = 8;
let claims_per_worker = 1000;

// Reset cursor once before spawning workers
tracker.reset_write_cursor();

let handles: Vec<_> = (0..num_workers)
    .map(|_| {
        let t = tracker.clone();
        thread::spawn(move || {
            let mut claimed = Vec::with_capacity(claims_per_worker);
            
            // continue_write() does NOT reset cursor - shares state across workers
            let mut iter = t.iter().continue_write();
            
            for _ in 0..claims_per_worker {
                if let Some((id, _)) = iter.next() {
                    // ID is atomically claimed and set in bitmap
                    valkey.hset(format!("vec:{}", id), "embedding", random_vector());
                    claimed.push(id);
                }
            }
            claimed
        })
    })
    .collect();

// All claimed IDs are unique - no duplicates across workers
let all_claimed: Vec<u64> = handles.into_iter()
    .flat_map(|h| h.join().unwrap())
    .collect();
    
println!("Claimed {} unique IDs", all_claimed.len());
```

**Key difference:**
| Method | Cursor Behavior | Use Case |
|--------|-----------------|----------|
| `write()` | Resets cursor to 0 | Single iterator, fresh start |
| `continue_write()` | Keeps current position | Multiple workers sharing cursor |

### Use Case 9: Iterate Only Reference Set Members

**Scenario**: Run queries only on vectors that exist in both tracker and reference set.

```rust
// Iterate intersection of tracker and reference set
for id in tracker.iter_intersection(&query_vectors) {
    let vector = dataset.get_query(id as usize);
    let results = valkey.ft_search(index, vector, k);
    // Process results...
}

// Or iterate reference set members missing from tracker
for id in tracker.iter_missing_from_reference(&protected) {
    println!("Missing protected ID: {}", id);
}
```

---

## API Reference

### TrackerConfig

```rust
/// Configuration for a PrefixTracker.
pub struct TrackerConfig {
    pub prefix: String,
    pub max_id: Option<u64>,
    pub max_sub_id: Option<u64>,
    pub slot_range: Option<(u16, u16)>,  // For slot-aware mode
    pub initial_capacity: usize,
}

impl TrackerConfig {
    /// Simple (non-hierarchical) tracker.
    pub fn simple(prefix: impl Into<String>) -> Self;
    
    /// Hierarchical tracker (prefix:id:sub_id).
    pub fn hierarchical(prefix: impl Into<String>) -> Self;
    
    /// Set maximum ID (pre-allocates bitmap).
    pub fn with_max_id(self, max_id: u64) -> Self;
    
    /// Set maximum sub_id for hierarchical trackers.
    pub fn with_max_sub_id(self, max_sub_id: u64) -> Self;
    
    /// Set slot range for cluster-aware tracking.
    pub fn with_slot_range(self, start: u16, end: u16) -> Self;
}
```

### PrefixTracker

```rust
/// Thread-safe existence tracker for keys under a prefix.
pub struct PrefixTracker { /* ... */ }

impl PrefixTracker {
    // === Basic Operations ===
    pub fn add(&self, id: u64) -> bool;           // Returns true if newly added
    pub fn remove(&self, id: u64) -> bool;        // Returns true if was present
    pub fn exists(&self, id: u64) -> bool;        // O(1) existence check
    pub fn claim(&self, id: u64) -> bool;         // Atomic test-and-set
    pub fn count(&self) -> u64;                   // Population count
    pub fn density(&self) -> f64;                 // count / max_id
    
    // === Bulk Operations ===
    pub fn add_range(&self, start: u64, end: u64) -> u64;
    pub fn remove_range(&self, start: u64, end: u64) -> u64;
    pub fn clear(&self);
    
    // === Cursor Control ===
    pub fn reset_write_cursor(&self);             // Reset to 0 before spawning workers
    
    // === Snapshot & Diff ===
    pub fn snapshot(&self) -> BitmapSnapshot;
    pub fn added_since(&self, snapshot: &BitmapSnapshot) -> Vec<u64>;
    pub fn removed_since(&self, snapshot: &BitmapSnapshot) -> Vec<u64>;
    pub fn added_count_since(&self, snapshot: &BitmapSnapshot) -> u64;
    pub fn removed_count_since(&self, snapshot: &BitmapSnapshot) -> u64;
    
    // === Reference Set Operations ===
    pub fn restore_from_reference(&self, reference: &ReferenceSet) -> u64;
    pub fn iter_intersection<'a>(&'a self, reference: &'a ReferenceSet) 
        -> impl Iterator<Item = u64> + 'a;
    pub fn iter_missing_from_reference<'a>(&'a self, reference: &'a ReferenceSet) 
        -> impl Iterator<Item = u64> + 'a;
    
    // === Iteration ===
    pub fn iter(&self) -> TrackerIterBuilder;
}
```

### TrackerIterBuilder

```rust
/// Builder for configuring iteration over a tracker.
pub struct TrackerIterBuilder<'a> { /* ... */ }

impl<'a> TrackerIterBuilder<'a> {
    // === Filters ===
    pub fn set_only(self) -> Self;
    pub fn unset_only(self) -> Self;
    pub fn mixed_ratio(self, existing_ratio: f64) -> Self;
    
    // === Partitioning ===
    /// Create a single partition for independent worker processing.
    /// Each worker can independently call partition(my_id, total) without coordination.
    pub fn partition(self, index: usize, total: usize) -> PartitionedIter<'a>;
    /// Create all partitions at once (for central distribution).
    pub fn partitioned(self, n: usize) -> Vec<PartitionedIter<'a>>;
    pub fn overlapping(self) -> Self;
    
    // === Sampling ===
    pub fn sample(self, probability: f64) -> Self;
    pub fn limit(self, max_items: u64) -> Self;
    pub fn seed(self, seed: u64) -> Self;
    
    // === Distribution ===
    pub fn distribution(self, dist: AccessDistribution) -> Self;
    
    // === ID Range ===
    pub fn id_range(self, start: u64, end: u64) -> Self;
    
    // === Terminal Operations ===
    pub fn sequential(self) -> SequentialIter<'a>;
    pub fn random(self) -> RandomIter<'a>;
    
    /// Build write iterator (resets cursor to 0).
    pub fn write(self) -> WriteIter<'a>;
    
    /// Build write iterator (continues from current cursor position).
    /// Use when multiple workers share cursor state.
    pub fn continue_write(self) -> WriteIter<'a>;
}
```

### BitmapSnapshot

```rust
/// Immutable snapshot of tracker state.
#[derive(Clone, Debug)]
pub struct BitmapSnapshot { /* ... */ }

impl BitmapSnapshot {
    pub fn capacity(&self) -> usize;
    pub fn count(&self) -> u64;
    pub fn test(&self, index: usize) -> bool;
    pub fn iter_set(&self) -> impl Iterator<Item = u64> + '_;
    
    // Set operations
    pub fn intersection(&self, other: &BitmapSnapshot) -> Vec<u64>;
    pub fn difference(&self, other: &BitmapSnapshot) -> Vec<u64>;
    pub fn intersection_count(&self, other: &BitmapSnapshot) -> u64;
    pub fn difference_count(&self, other: &BitmapSnapshot) -> u64;
}
```

### ReferenceSet

```rust
/// External ID set for filtering and protection.
///
/// Replaces domain-specific protected ID tracking with generic,
/// efficient set operations.
#[derive(Clone, Debug)]
pub struct ReferenceSet { /* ... */ }

impl ReferenceSet {
    // === Construction ===
    pub fn with_capacity(max_id: u64) -> Self;
    pub fn from_iter(ids: impl IntoIterator<Item = u64>) -> Self;
    pub fn from_slice(ids: &[u64]) -> Self;
    
    // === Mutation ===
    pub fn insert(&mut self, id: u64) -> bool;
    pub fn remove(&mut self, id: u64) -> bool;
    
    // === Query ===
    pub fn contains(&self, id: u64) -> bool;  // O(1) - use for protection checks
    pub fn len(&self) -> u64;
    pub fn is_empty(&self) -> bool;
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_;
    
    // === Operations with Snapshots ===
    pub fn count_existing_in(&self, snapshot: &BitmapSnapshot) -> u64;
    pub fn count_missing_in(&self, snapshot: &BitmapSnapshot) -> u64;
    pub fn missing_in(&self, snapshot: &BitmapSnapshot) -> Vec<u64>;
    pub fn existing_in(&self, snapshot: &BitmapSnapshot) -> Vec<u64>;
    
    // === Set Algebra ===
    pub fn intersection(&self, other: &ReferenceSet) -> ReferenceSet;
    pub fn union(&self, other: &ReferenceSet) -> ReferenceSet;
    pub fn difference(&self, other: &ReferenceSet) -> ReferenceSet;
}
```

### AccessDistribution

```rust
/// Statistical distribution for key access patterns.
#[derive(Clone, Debug)]
pub enum AccessDistribution {
    Uniform,
    Zipfian { skew: f64 },
    Exponential { lambda: f64 },
    Normal { mean_pct: f64, std_pct: f64 },
    Hotspot { hot_pct: f64, hot_prob: f64 },
    Latest { recent_pct: f64, recent_prob: f64 },
}

impl AccessDistribution {
    pub fn web_cache() -> Self;      // Zipfian { skew: 0.99 }
    pub fn session_store() -> Self;  // Exponential { lambda: 0.1 }
    pub fn hotspot() -> Self;        // Hotspot { hot_pct: 0.2, hot_prob: 0.8 }
}
```

### WorkloadProfile

```rust
/// Composite workload configuration.
#[derive(Clone, Debug)]
pub struct WorkloadProfile {
    pub distribution: AccessDistribution,
    pub read_ratio: f64,
    pub overwrite_ratio: f64,
    pub delete_ratio: f64,
    pub fragmentation: FragmentationPattern,
    pub seed: Option<u64>,
}

impl WorkloadProfile {
    pub fn session_store() -> Self;
    pub fn cache(read_ratio: f64) -> Self;
    pub fn time_series() -> Self;
    pub fn defrag_test() -> Self;
    pub fn mixed() -> Self;
    
    pub fn select_operation(&self, rng: &mut impl Rng) -> WorkloadOperation;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadOperation {
    Read,
    Overwrite,
    Insert,
    Delete,
}
```

---

## Implementation Guide

### Replacing Protected ID Lists

The `ReferenceSet` replaces any domain-specific protected ID tracking:

**Before (domain-specific):**
```rust
struct ProtectedVectorIds {
    ids: HashSet<u64>,
}

impl ProtectedVectorIds {
    fn is_protected(&self, id: u64) -> bool {
        self.ids.contains(&id)
    }
}
```

**After (generic):**
```rust
// Build reference set from any source
let protected = ReferenceSet::from_iter(ground_truth_ids);

// O(1) protection check
if protected.contains(id) {
    // Skip protected ID
}

// Additional capabilities
let coverage = protected.count_existing_in(&tracker.snapshot());
let missing = protected.missing_in(&tracker.snapshot());
```

### Scan-Based Initialization Pattern

```rust
/// Initialize tracker from database scan.
pub fn init_tracker_from_scan(
    client: &mut Connection,
    prefix: &str,
    max_id: u64,
) -> PrefixTracker {
    let tracker = PrefixTracker::new(
        TrackerConfig::simple(prefix).with_max_id(max_id)
    );
    
    let pattern = format!("{}*", prefix);
    let mut cursor = 0;
    
    loop {
        let (next, keys): (u64, Vec<String>) = redis::cmd("SCAN")
            .arg(cursor)
            .arg("MATCH").arg(&pattern)
            .arg("COUNT").arg(10000)
            .query(client)
            .unwrap();
        
        for key in keys {
            if let Some(id) = extract_id_from_key(&key, prefix) {
                tracker.add(id);
            }
        }
        
        if next == 0 {
            break;
        }
        cursor = next;
    }
    
    tracker
}
```

### Slot-Aware Initialization Pattern

```rust
/// Initialize trackers per slot range from cluster scan.
pub fn init_slot_trackers(
    cluster: &ClusterConnection,
    prefix: &str,
    max_id_per_range: u64,
) -> Vec<(SlotRange, PrefixTracker)> {
    let mut trackers = Vec::new();
    
    for node in cluster.nodes() {
        let slots = node.slot_ranges();
        
        for range in slots {
            let tracker = PrefixTracker::new(
                TrackerConfig::simple(prefix)
                    .with_max_id(max_id_per_range)
                    .with_slot_range(range.start, range.end)
            );
            
            // Scan this node
            let pattern = format!("{}*", prefix);
            for key in node.scan(&pattern) {
                let slot = crc16_slot(&key);
                if range.contains(slot) {
                    if let Some(id) = extract_id_from_key(&key, prefix) {
                        tracker.add(id);
                    }
                }
            }
            
            trackers.push((range, tracker));
        }
    }
    
    trackers
}
```

---

## Performance Considerations

### Memory Usage

| Component | Memory per ID | 10M IDs |
|-----------|---------------|---------|
| `PrefixTracker` | 1 bit | ~1.25 MB |
| `ReferenceSet` | 1 bit | ~1.25 MB |
| `BitmapSnapshot` | 1 bit (copy) | ~1.25 MB |

### Operation Costs

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `add/remove/exists` | O(1) | Atomic, lock-free |
| `ReferenceSet.contains` | O(1) | Use for protection checks |
| `add_range/remove_range` | O(n/64) | SIMD-accelerated |
| `count` | O(1) | Cached, atomic read |
| `snapshot` | O(n/64) | Copy bitmap data |
| `intersection/difference` | O(n/64) | SIMD popcount |
| `iter().sequential()` | O(n) | Bitmap scan |
| `iter().random()` | O(k) | k = items yielded |

### Best Practices

1. **Scan once at startup**: Initialize tracker from database scan to ensure consistency
2. **Pre-allocate capacity**: Use `with_max_id()` to avoid bitmap growth
3. **Use reference sets for protection**: O(1) membership check vs O(n) list search
4. **Partition iterations**: Use `.partition()` or slot-based trackers for parallelism
5. **Avoid frequent snapshots**: Snapshot creation copies bitmap data
6. **Batch reference set operations**: Build set once, query many times

---

## Appendix: Command Reference

### Tracker CLI Commands

```bash
# Initialize tracker from database scan
valkey-bench-rs -h $HOST --cluster -t vec-load \
  --keyspace-tracker \
  --tracker-init scan \
  --schema dataset.yaml --data dataset.bin \
  --search-index idx -n 1000000 -c 100

# Query with reference set coverage check
valkey-bench-rs -h $HOST --cluster -t vec-query \
  --keyspace-tracker \
  --reference-set-check \
  --schema dataset.yaml --data dataset.bin \
  --search-index idx -k 10 -n 10000

# Query with automatic reference set backfill
valkey-bench-rs -h $HOST --cluster -t vec-query \
  --keyspace-tracker \
  --reference-set-backfill \
  --schema dataset.yaml --data dataset.bin \
  --search-index idx -k 10 -n 10000

# Delete with reference set protection
valkey-bench-rs -h $HOST --cluster -t vec-delete \
  --keyspace-tracker \
  --protect-reference-set \
  --workload-snapshot \
  --schema dataset.yaml --data dataset.bin \
  --search-index idx -n 10000

# Mixed workload with access distribution
valkey-bench-rs -h $HOST --cluster \
  --keyspace-tracker \
  --tracker-init scan \
  --parallel "get:80,set:20" \
  --iteration "zipfian:0.99" \
  -n 1000000 -c 100
```