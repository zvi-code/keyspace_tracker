# keyspace_tracker

[![Crates.io](https://img.shields.io/crates/v/keyspace_tracker.svg)](https://crates.io/crates/keyspace_tracker)
[![Documentation](https://docs.rs/keyspace_tracker/badge.svg)](https://docs.rs/keyspace_tracker)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Simulate real-world application workloads. Track keyspace state and iterate with composable patterns: thread partitioning, Zipfian/hotspot distributions, atomic claiming, sampling—lock-free primitives for building realistic benchmarks.

Note: Initially this was designed to integrate in https://github.com/zvi-code/valkey-bench-rs, but written as independant tool for other possible usages

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   PrefixGroupsTracker                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│  │ "user:"      │ │ "session:"   │ │ "cache:"     │  .....  │
│  │ PrefixTracker│ │ PrefixTracker│ │ PrefixTracker│         │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘         │
│         │               │                 │                 │
│         ▼               ▼                 ▼                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│  │AtomicBitmap  │ │AtomicBitmap  │ │AtomicBitmap  │         │
│  │ [0100101...] │ │ [1100010...] │ │ [0011100...] │         │
│  └──────────────┘ └──────────────┘ └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Sub-nanosecond operations**: Core operations (test, set, claim) in 2-7ns
- **Lock-free concurrency**: Atomic operations with no mutex overhead
- **O(1) scaling**: Constant performance from 1K to 10M+ elements
- **SIMD optimized**: ARM NEON and x86_64 AVX2/POPCNT acceleration
- **Memory efficient**: 1 bit per ID, auto-growing bitmaps
- **Hierarchical support**: Track (id, sub_id) pairs efficiently
- **Rich iteration**: Sequential, random, write (claim), delete, partitioned iterators
- **Sampling & distributions**: Mixed-ratio, probabilistic sampling, Zipfian, hotspot patterns
- **Bulk operations**: Set/clear ranges efficiently with SIMD acceleration
- **Group operations**: Iterate across multiple prefixes with various policies

## Use Cases

- **Database benchmarking**: Track key existence for workload generation (see [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md))
- **Cache existence tracking**: Know which keys exist without fetching values
- **Backfill completion tracking**: Track which IDs have been processed
- **Distributed ID allocation**: Claim unique IDs across threads/processes
- **Bloom filter alternative**: Exact membership testing (no false positives)

## Quick Start

```rust
use keyspace_tracker::{PrefixGroupsTracker, TrackerConfig, ClaimPolicy};

// Create a registry for multiple prefixes
let groups = PrefixGroupsTracker::new();

// Register trackers for different key types
groups.register(TrackerConfig::simple("user:").with_max_id(1_000_000));
groups.register(TrackerConfig::simple("session:").with_max_id(100_000));

// Get a tracker and add IDs
let users = groups.get("user:").unwrap();
users.add(42);
users.add(1337);

// Check existence
assert!(users.exists(42));
assert!(!users.exists(999));

// Iterate over set IDs
for (id, _) in users.iter().set_only().sequential() {
    println!("User ID: {}", id);
}
```

## Core Types

### `AtomicBitmap`

Low-level lock-free bitmap with atomic operations:

```rust
use keyspace_tracker::AtomicBitmap;

let bitmap = AtomicBitmap::with_capacity(1_000_000);

// Basic operations
bitmap.set(42);                    // Set bit, returns true if was unset
bitmap.test(42);                   // Check if set
bitmap.clear(42);                  // Clear bit

// Atomic claim (test-and-set)
if bitmap.test_and_set(100) {
    println!("Successfully claimed ID 100");
}

// Find next set/unset bit
let next_set = bitmap.find_next_set(0);
let next_unset = bitmap.find_next_unset(0, 1_000_000);

// Bulk operations (SIMD accelerated)
bitmap.set_range(0, 10_000);        // Set bits [0, 10000)
bitmap.clear_range(5_000, 7_500);   // Clear bits [5000, 7500)
let count = bitmap.count_range(0, 10_000);  // Count set bits in range
```

### `PrefixTracker`

Single-prefix tracker with simple or hierarchical mode:

```rust
use keyspace_tracker::{PrefixTracker, TrackerConfig};

// Simple mode: flat ID space
let simple = PrefixTracker::new(
    TrackerConfig::simple("vec:")
        .with_max_id(1_000_000)
        .with_initial_capacity(4096)
);

simple.add(0);
simple.add(100);
assert!(simple.exists(100));

// Hierarchical mode: (id, sub_id) pairs
let hier = PrefixTracker::hierarchical("hash:");
hier.add_pair(1, 0);    // hash:1 -> field 0
hier.add_pair(1, 5);    // hash:1 -> field 5
hier.add_pair(2, 10);   // hash:2 -> field 10

assert!(hier.exists_pair(1, 5));
assert!(!hier.exists_pair(1, 99));
```

### `PrefixGroupsTracker`

Registry of multiple trackers with group-level operations:

```rust
use keyspace_tracker::{PrefixGroupsTracker, TrackerConfig, ClaimPolicy};

let groups = PrefixGroupsTracker::new();

// Register multiple prefixes
groups.register(TrackerConfig::simple("a:"));
groups.register(TrackerConfig::simple("b:"));
groups.register(TrackerConfig::simple("c:"));

// Add data
groups.get("a:").unwrap().add(1);
groups.get("b:").unwrap().add(2);

// Iterate across all prefixes
for item in groups.iter().set_only().horizontal().build() {
    println!("{}:{}", item.prefix, item.id);
}

// Claim IDs round-robin across prefixes
let mut writer = groups.iter().write(ClaimPolicy::RoundRobin);
while let Some(item) = writer.next() {
    println!("Claimed {}:{}", item.prefix, item.id);
    if item.id > 100 { break; }
}
```

## Iterators

### Sequential Iterator

Iterate in ID order:

```rust
// All IDs (set and unset)
for (id, sub_id) in tracker.iter().sequential() {
    // ...
}

// Only set IDs
for (id, _) in tracker.iter().set_only().sequential() {
    // ...
}

// Only unset IDs in range
for (id, _) in tracker.iter()
    .range((0, 1000), (0, u64::MAX))
    .unset_only()
    .sequential() 
{
    // ...
}
```

### Random Iterator

Memory-efficient pseudo-random traversal:

```rust
// Random order iteration (no allocation)
for (id, _) in tracker.iter().set_only().random() {
    // Visits each set ID exactly once in random order
}

// Reproducible random with seed
for (id, _) in tracker.iter().set_only().seed(42).random() {
    // Same sequence every time with seed 42
}
```

### Sampling Iterator

Flexible sampling for benchmark workloads:

```rust
// Limit number of items returned
for (id, _) in tracker.iter().set_only().limit(1000).sequential() {
    // Returns at most 1000 items
}

// Probabilistic sampling (~50% of keys)
for (id, _) in tracker.iter().set_only().sample(0.5).sequential() {
    // Each key has 50% chance of being returned
}

// Mixed-ratio: 90% existing + 10% new keys
for (id, _) in tracker.iter().mixed_ratio(0.9).random() {
    // 90% of returned IDs are set, 10% are unset
}

// Range-relative iteration (upper 50% of keyspace)
for (id, _) in tracker.iter().set_only().range_percent(0.5, 1.0).sequential() {
    // Only IDs in upper half of keyspace
}

// Combine options
for (id, _) in tracker.iter()
    .mixed_ratio(0.9)
    .sample(0.5)
    .limit(10_000)
    .seed(42)
    .random()
{
    // Complex workload pattern
}
```

### Access Distributions

Realistic workload patterns for benchmarking:

```rust
use keyspace_tracker::{SamplingConfig, AccessDistribution};

// Zipfian distribution (popular keys accessed more)
let config = SamplingConfig::new()
    .with_distribution(AccessDistribution::Zipfian { skew: 0.99 });

// Hotspot (20% of keys get 80% of traffic)
let config = SamplingConfig::new()
    .with_distribution(AccessDistribution::Hotspot { 
        hot_pct: 0.2, 
        hot_prob: 0.8 
    });

// Latest (recent keys accessed more)
let config = SamplingConfig::new()
    .with_distribution(AccessDistribution::Latest { 
        recent_pct: 0.1, 
        recent_prob: 0.9 
    });

for (id, _) in tracker.iter().with_sampling(config).random() {
    // Access pattern follows distribution
}
```

### Write Iterator

Atomically claim unique IDs:

```rust
// Claim unset IDs atomically (thread-safe)
let mut writer = tracker.iter().unset_only().write();
let mut claimed = Vec::new();

while let Some((id, _)) = writer.next() {
    claimed.push(id);
    if claimed.len() >= 100 { break; }
}
// All IDs in `claimed` are now set and unique
```

#### Concurrent Workers with `continue_write()`

When multiple workers need to share cursor state across independently-created iterators,
use `continue_write()` instead of `write()`:

```rust
use std::sync::Arc;
use std::thread;

let tracker = Arc::new(tracker);

// Reset cursor once before spawning workers
tracker.reset_write_cursor();

let handles: Vec<_> = (0..8)
    .map(|_| {
        let t = tracker.clone();
        thread::spawn(move || {
            // continue_write() does NOT reset cursor - shares state across workers
            let mut iter = t.iter().continue_write();
            let mut claimed = Vec::new();
            
            for _ in 0..100 {
                if let Some((id, _)) = iter.next() {
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
```

| Method | Cursor Behavior | Use Case |
|--------|-----------------|----------|
| `write()` | Resets cursor to 0 | Single iterator, fresh start |
| `continue_write()` | Keeps current position | Multiple workers sharing cursor |

### Delete Iterator

Exclusive iterator that clears IDs:

```rust
// Atomically claim and clear set IDs
if let Some(mut deleter) = tracker.iter().delete() {
    while let Some((id, _)) = deleter.next() {
        println!("Deleted ID: {}", id);
    }
}
// Returns None if another delete iterator is active
```

### Partitioned Iterator

Parallel iteration with disjoint ranges:

```rust
use std::thread;

// Option 1: Create all partitions centrally, then distribute
let partitions = tracker.iter().set_only().partitioned(4);

let handles: Vec<_> = partitions
    .into_iter()
    .map(|mut part| {
        thread::spawn(move || {
            let mut count = 0;
            while let Some(_) = part.next() {
                count += 1;
            }
            count
        })
    })
    .collect();

let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
```

### Independent Worker Partitioning

Each worker independently creates its own partition without coordination:

```rust
use std::sync::Arc;
use std::thread;

let tracker = Arc::new(tracker);
let num_workers = 8;

let handles: Vec<_> = (0..num_workers)
    .map(|worker_id| {
        let t = tracker.clone();
        thread::spawn(move || {
            // Each worker computes its own range - no central coordination
            let mut count = 0u64;
            for (id, _) in t.iter()
                .set_only()
                .partition(worker_id, num_workers)  // Single partition by index
            {
                count += 1;
            }
            count
        })
    })
    .collect();

let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
```

### Overlapping Mode (Contention Testing)

All partitions iterate the full range for lock contention testing:

```rust
// Each partition sees all keys (for contention benchmarks)
let partitions = tracker.iter()
    .set_only()
    .overlapping()  // Enable overlapping mode
    .seed(42)
    .partitioned(4);

// Each partition iterates the full keyspace
for partition in partitions {
    for (id, _) in partition {
        // All threads may access same keys
    }
}
```

## Group Iteration

### Horizontal (prefix-by-prefix)

```rust
// All IDs for prefix₀, then prefix₁, etc.
for item in groups.iter().horizontal().set_only().build() {
    println!("{}:{}", item.prefix, item.id);
}
```

### Vertical (round-robin across prefixes)

```rust
// ID₀ across all prefixes, then ID₁, etc.
for item in groups.iter().vertical().set_only().build() {
    println!("{}:{}", item.prefix, item.id);
}
```

### Random

```rust
for item in groups.iter().random().set_only().build().take(1000) {
    // Random sampling across all prefixes
}
```

### Claim Policies

```rust
use keyspace_tracker::ClaimPolicy;

// Round-robin across prefixes
let mut w = groups.iter().write(ClaimPolicy::RoundRobin);

// Random prefix selection
let mut w = groups.iter().write(ClaimPolicy::Random);

// Prefer prefixes with more free slots
let mut w = groups.iter().write(ClaimPolicy::LeastLoaded);

// Prefer prefixes with more used slots
let mut w = groups.iter().write(ClaimPolicy::MostLoaded);
```

## Performance

Benchmarked on Apple M2 (ARM64):

| Operation | Latency | Notes |
|-----------|---------|-------|
| `test` (exists) | 1.6 ns | L1 cache hit |
| `set` (add) | 7.2 ns | Atomic OR |
| `test_and_set` (claim) | 2.4 ns | CAS loop |
| `find_next_set` | 4.0 ns | SIMD accelerated |
| `find_next_unset` | Varies | O(density) |
| Hierarchical `exists_pair` | 17 ns | DashMap + bitmap |
| Hierarchical `add_pair` | 39 ns | DashMap + bitmap |
| Iterator per-key | 2.5-5 ns | All iteration modes |

### Scaling

Constant O(1) performance regardless of bitmap size:

| Size | Throughput | Latency |
|------|------------|---------|
| 1K | 430 Mops/s | 2.3 ns |
| 10K | 428 Mops/s | 2.3 ns |
| 100K | 425 Mops/s | 2.4 ns |
| 1M | 423 Mops/s | 2.4 ns |
| 10M | 427 Mops/s | 2.3 ns |

## Platform Optimizations

The library automatically uses platform-specific optimizations:

- **ARM64 (NEON)**: Vectorized popcount, prefetching
- **x86_64 (AVX2/POPCNT)**: Hardware popcount, prefetching
- **Fallback**: Optimized scalar with loop unrolling

### Benchmarking for Maximum Performance

```bash
# Maximum performance benchmark (recommended)
RUSTFLAGS="-C target-cpu=native" cargo bench --release

# View specific benchmark group
RUSTFLAGS="-C target-cpu=native" cargo bench --release -- bitmap/
RUSTFLAGS="-C target-cpu=native" cargo bench --release -- simd/
```

### Build Configuration

The crate includes aggressive optimization profiles in `Cargo.toml`:

```toml
[profile.bench]
opt-level = 3
lto = "fat"              # Full cross-crate inlining
codegen-units = 1        # Global optimization  
panic = "abort"          # No unwinding overhead
overflow-checks = false  # No runtime checks
```

### Platform-Specific Tuning

Edit `.cargo/config.toml` for your target:

```toml
# Apple Silicon (M1/M2/M3/M4)
[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=apple-m1"]

# AWS Graviton3
[target.aarch64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=neoverse-v1"]

# Modern x86_64 (Haswell+)
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=haswell"]
```

### Nightly Optimizations (Optional)

```bash
# Extra tuning with nightly
RUSTFLAGS="-C target-cpu=native -Z tune-cpu=native" \
  cargo +nightly bench --release
```

### Stable Benchmark Tips

For consistent results:
- Close background applications
- Plug in power (avoid CPU throttling)
- Run multiple times to check variance
- Use `criterion`'s built-in statistical analysis

## Thread Safety

All types are `Send + Sync`:

- `AtomicBitmap`: Lock-free atomic operations
- `PrefixTracker`: Lock-free reads/writes, mutex only for growth
- `PrefixGroupsTracker`: Concurrent access via DashMap with AHash
- `WriteIter`: Multiple writers can run concurrently
- `DeleteIter`: Exclusive (only one per tracker)

## Documentation

- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Detailed guide for database benchmarking integration
- [BENCHMARKING.md](BENCHMARKING.md) - Performance tuning and benchmark configuration

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.
