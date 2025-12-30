use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use keyspace_tracker::{
    AtomicBitmap, PrefixTracker, PrefixGroupsTracker, TrackerConfig, ClaimPolicy,
};
use std::sync::Arc;

// =============================================================================
// AtomicBitmap Benchmarks
// =============================================================================

fn bench_bitmap_test(c: &mut Criterion) {
    let bitmap = AtomicBitmap::with_capacity(1_000_000);
    
    // Pre-populate 50% of bits
    for i in (0..500_000).step_by(2) {
        bitmap.set(i);
    }

    c.bench_function("bitmap/test (exists check)", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let idx = (i % 1_000_000) as usize;
            i = i.wrapping_add(1);
            black_box(bitmap.test(idx))
        })
    });
}

fn bench_bitmap_set(c: &mut Criterion) {
    let bitmap = AtomicBitmap::with_capacity(1_000_000);

    c.bench_function("bitmap/set", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let idx = (i % 1_000_000) as usize;
            i = i.wrapping_add(1);
            black_box(bitmap.set(idx))
        })
    });
}

fn bench_bitmap_test_and_set(c: &mut Criterion) {
    let bitmap = AtomicBitmap::with_capacity(10_000_000);

    c.bench_function("bitmap/test_and_set (claim)", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let idx = (i % 10_000_000) as usize;
            i = i.wrapping_add(1);
            black_box(bitmap.test_and_set(idx))
        })
    });
}

fn bench_bitmap_test_and_clear(c: &mut Criterion) {
    let bitmap = AtomicBitmap::with_capacity(1_000_000);
    
    // Pre-set all bits
    for i in 0..1_000_000 {
        bitmap.set(i);
    }

    c.bench_function("bitmap/test_and_clear", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let idx = (i % 1_000_000) as usize;
            i = i.wrapping_add(1);
            // Re-set to keep benchmark consistent
            bitmap.set(idx);
            black_box(bitmap.test_and_clear(idx))
        })
    });
}

fn bench_bitmap_find_next_set(c: &mut Criterion) {
    let bitmap = AtomicBitmap::with_capacity(1_000_000);
    
    // Sparse population: 1% set
    for i in (0..1_000_000).step_by(100) {
        bitmap.set(i);
    }

    c.bench_function("bitmap/find_next_set", |b| {
        let mut start = 0usize;
        b.iter(|| {
            let result = bitmap.find_next_set(start);
            start = (start + 1) % 1_000_000;
            black_box(result)
        })
    });
}

fn bench_bitmap_find_next_unset(c: &mut Criterion) {
    let bitmap = AtomicBitmap::with_capacity(1_000_000);
    
    // Dense population: 99% set
    for i in 0..990_000 {
        bitmap.set(i);
    }

    c.bench_function("bitmap/find_next_unset", |b| {
        let mut start = 0usize;
        b.iter(|| {
            let result = bitmap.find_next_unset(start, 1_000_000);
            start = (start + 1) % 1_000_000;
            black_box(result)
        })
    });
}

// =============================================================================
// PrefixTracker Simple Mode Benchmarks
// =============================================================================

fn bench_tracker_simple_add(c: &mut Criterion) {
    let tracker = PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(10_000_000)
    );

    c.bench_function("tracker/simple/add", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let id = i % 10_000_000;
            i = i.wrapping_add(1);
            black_box(tracker.add(id))
        })
    });
}

fn bench_tracker_simple_exists(c: &mut Criterion) {
    let tracker = PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(1_000_000)
    );
    
    // Pre-populate 50%
    for i in (0..500_000).step_by(2) {
        tracker.add(i);
    }

    c.bench_function("tracker/simple/exists", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let id = i % 1_000_000;
            i = i.wrapping_add(1);
            black_box(tracker.exists(id))
        })
    });
}

fn bench_tracker_simple_claim(c: &mut Criterion) {
    let tracker = PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(10_000_000)
    );

    c.bench_function("tracker/simple/claim", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let id = i % 10_000_000;
            i = i.wrapping_add(1);
            black_box(tracker.claim(id))
        })
    });
}

fn bench_tracker_simple_remove(c: &mut Criterion) {
    let tracker = PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(1_000_000)
    );
    
    // Pre-populate all
    for i in 0..1_000_000 {
        tracker.add(i);
    }

    c.bench_function("tracker/simple/remove", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let id = i % 1_000_000;
            i = i.wrapping_add(1);
            // Re-add to keep benchmark consistent
            tracker.add(id);
            black_box(tracker.remove(id))
        })
    });
}

// =============================================================================
// PrefixTracker Hierarchical Mode Benchmarks
// =============================================================================

fn bench_tracker_hierarchical_add_pair(c: &mut Criterion) {
    let tracker = PrefixTracker::new(
        TrackerConfig::hierarchical("hash:")
            .with_max_id(10_000)
            .with_max_sub_id(1_000)
    );

    c.bench_function("tracker/hierarchical/add_pair", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let id = i % 10_000;
            let sub_id = (i / 10_000) % 1_000;
            i = i.wrapping_add(1);
            black_box(tracker.add_pair(id, sub_id))
        })
    });
}

fn bench_tracker_hierarchical_exists_pair(c: &mut Criterion) {
    let tracker = PrefixTracker::new(
        TrackerConfig::hierarchical("hash:")
            .with_max_id(1_000)
            .with_max_sub_id(1_000)
    );
    
    // Pre-populate
    for id in 0..1_000 {
        for sub_id in (0..1_000).step_by(2) {
            tracker.add_pair(id, sub_id);
        }
    }

    c.bench_function("tracker/hierarchical/exists_pair", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let id = i % 1_000;
            let sub_id = (i / 1_000) % 1_000;
            i = i.wrapping_add(1);
            black_box(tracker.exists_pair(id, sub_id))
        })
    });
}

fn bench_tracker_hierarchical_claim_pair(c: &mut Criterion) {
    let tracker = PrefixTracker::new(
        TrackerConfig::hierarchical("hash:")
            .with_max_id(10_000)
            .with_max_sub_id(1_000)
    );

    c.bench_function("tracker/hierarchical/claim_pair", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let id = i % 10_000;
            let sub_id = (i / 10_000) % 1_000;
            i = i.wrapping_add(1);
            black_box(tracker.claim_pair(id, sub_id))
        })
    });
}

// =============================================================================
// Iterator Benchmarks
// =============================================================================

fn bench_iter_sequential_set(c: &mut Criterion) {
    let tracker = PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(100_000)
    );
    
    // Populate 10%
    for i in (0..100_000).step_by(10) {
        tracker.add(i);
    }

    c.bench_function("iter/sequential/set_only", |b| {
        b.iter(|| {
            let count = tracker.iter().set_only().sequential().count();
            black_box(count)
        })
    });
}

fn bench_iter_sequential_all(c: &mut Criterion) {
    let tracker = PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(10_000)
    );
    
    for i in (0..10_000).step_by(10) {
        tracker.add(i);
    }

    c.bench_function("iter/sequential/all", |b| {
        b.iter(|| {
            let count = tracker.iter().sequential().count();
            black_box(count)
        })
    });
}

fn bench_iter_random(c: &mut Criterion) {
    let tracker = PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(100_000)
    );
    
    for i in (0..100_000).step_by(10) {
        tracker.add(i);
    }

    c.bench_function("iter/random/set_only", |b| {
        b.iter(|| {
            let count = tracker.iter().set_only().random().take(1000).count();
            black_box(count)
        })
    });
}

fn bench_iter_write(c: &mut Criterion) {
    c.bench_function("iter/write/claim_sequential", |b| {
        b.iter_custom(|iters| {
            // Fresh tracker ensures sparse bitmap
            let tracker = Arc::new(PrefixTracker::new(
                TrackerConfig::simple("vec:").with_max_id(10_000_000)
            ));
            
            let mut iter = tracker.iter().unset_only().write();
            let start = std::time::Instant::now();
            for _ in 0..iters {
                black_box(iter.next());
            }
            start.elapsed()
        })
    });
}

fn bench_iter_write_saturation(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter/write_saturation");
    
    for fill_pct in [0, 50, 90, 99] {
        group.bench_with_input(BenchmarkId::new("claim", format!("{}%", fill_pct)), &fill_pct, |b, &fill_pct| {
            let capacity = 100_000u64;
            let tracker = PrefixTracker::new(
                TrackerConfig::simple("vec:").with_max_id(capacity)
            );
            
            // Pre-fill to target percentage
            let fill_count = (capacity * fill_pct as u64) / 100;
            for i in 0..fill_count {
                tracker.add(i);
            }
            
            b.iter(|| {
                let mut iter = tracker.iter().unset_only().write();
                for _ in 0..100 {
                    black_box(iter.next());
                }
            })
        });
    }
    
    group.finish();
}

// =============================================================================
// Group Iteration Benchmarks
// =============================================================================

fn bench_group_iter_horizontal(c: &mut Criterion) {
    let groups = PrefixGroupsTracker::new();
    
    for prefix in ["vec:", "doc:", "hash:", "set:"] {
        let t = groups.register(TrackerConfig::simple(prefix).with_max_id(10_000));
        for i in 0..1_000 {
            t.add(i);
        }
    }

    c.bench_function("group_iter/horizontal/set_only", |b| {
        b.iter(|| {
            let count = groups.iter().set_only().horizontal().build().count();
            black_box(count)
        })
    });
}

fn bench_group_iter_vertical(c: &mut Criterion) {
    let groups = PrefixGroupsTracker::new();
    
    for prefix in ["vec:", "doc:", "hash:", "set:"] {
        let t = groups.register(TrackerConfig::simple(prefix).with_max_id(10_000));
        for i in 0..1_000 {
            t.add(i);
        }
    }

    c.bench_function("group_iter/vertical/set_only", |b| {
        b.iter(|| {
            let count = groups.iter().set_only().vertical().build().count();
            black_box(count)
        })
    });
}

fn bench_group_write_round_robin(c: &mut Criterion) {
    // Measure per-claim cost with fresh sparse bitmaps
    c.bench_function("group_iter/write/round_robin", |b| {
        b.iter_custom(|iters| {
            // Fresh trackers - large capacity ensures sparse bitmap throughout
            let groups = PrefixGroupsTracker::new();
            for prefix in ["a:", "b:", "c:", "d:"] {
                groups.register(TrackerConfig::simple(prefix).with_max_id(10_000_000));
            }
            
            let mut iter = groups.iter().write(ClaimPolicy::RoundRobin);
            let start = std::time::Instant::now();
            for _ in 0..iters {
                black_box(iter.next());
            }
            start.elapsed()
        })
    });
}

fn bench_group_write_saturation(c: &mut Criterion) {
    // Measure degradation as bitmap fills up
    let mut group = c.benchmark_group("group_iter/write_saturation");
    
    for fill_pct in [0, 50, 90, 99] {
        group.bench_with_input(BenchmarkId::new("round_robin", format!("{}%", fill_pct)), &fill_pct, |b, &fill_pct| {
            let groups = PrefixGroupsTracker::new();
            let capacity = 100_000u64;
            
            for prefix in ["a:", "b:", "c:", "d:"] {
                let t = groups.register(TrackerConfig::simple(prefix).with_max_id(capacity));
                // Pre-fill to target percentage
                let fill_count = (capacity * fill_pct as u64) / 100;
                for i in 0..fill_count {
                    t.add(i);
                }
            }
            
            b.iter(|| {
                let mut iter = groups.iter().write(ClaimPolicy::RoundRobin);
                // Try to claim 100 items
                for _ in 0..100 {
                    black_box(iter.next());
                }
            })
        });
    }
    
    group.finish();
}

// =============================================================================
// Scaling Benchmarks
// =============================================================================

fn bench_bitmap_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitmap/scaling");
    
    for size in [1_000, 10_000, 100_000, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(BenchmarkId::new("test", size), &size, |b, &size| {
            let bitmap = AtomicBitmap::with_capacity(size);
            for i in (0..size).step_by(2) {
                bitmap.set(i);
            }
            
            let mut i = 0u64;
            b.iter(|| {
                for _ in 0..1000 {
                    let idx = (i as usize) % size;
                    i = i.wrapping_add(1);
                    black_box(bitmap.test(idx));
                }
            })
        });
    }
    
    group.finish();
}

fn bench_tracker_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tracker/scaling");
    
    for size in [1_000u64, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(BenchmarkId::new("exists", size), &size, |b, &size| {
            let tracker = PrefixTracker::new(
                TrackerConfig::simple("test:").with_max_id(size)
            );
            for i in (0..size).step_by(2) {
                tracker.add(i);
            }
            
            let mut i = 0u64;
            b.iter(|| {
                for _ in 0..1000 {
                    let id = i % size;
                    i = i.wrapping_add(1);
                    black_box(tracker.exists(id));
                }
            })
        });
    }
    
    group.finish();
}

// =============================================================================
// Concurrent Benchmarks
// =============================================================================

fn bench_concurrent_claims(c: &mut Criterion) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    let mut group = c.benchmark_group("concurrent");
    
    for threads in [1, 2, 4, 8] {
        group.bench_with_input(BenchmarkId::new("claim", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let tracker = Arc::new(PrefixTracker::new(
                    TrackerConfig::simple("test:").with_max_id(iters * threads as u64 * 100)
                ));
                let counter = Arc::new(AtomicU64::new(0));
                
                let start = std::time::Instant::now();
                
                let handles: Vec<_> = (0..threads).map(|_| {
                    let tracker = tracker.clone();
                    let counter = counter.clone();
                    thread::spawn(move || {
                        for _ in 0..iters {
                            let id = counter.fetch_add(1, Ordering::Relaxed);
                            black_box(tracker.claim(id));
                        }
                    })
                }).collect();
                
                for h in handles {
                    h.join().unwrap();
                }
                
                start.elapsed()
            })
        });
    }
    
    group.finish();
}

// =============================================================================
// Criterion Groups
// =============================================================================

criterion_group!(
    bitmap_benches,
    bench_bitmap_test,
    bench_bitmap_set,
    bench_bitmap_test_and_set,
    bench_bitmap_test_and_clear,
    bench_bitmap_find_next_set,
    bench_bitmap_find_next_unset,
);

criterion_group!(
    tracker_simple_benches,
    bench_tracker_simple_add,
    bench_tracker_simple_exists,
    bench_tracker_simple_claim,
    bench_tracker_simple_remove,
);

criterion_group!(
    tracker_hierarchical_benches,
    bench_tracker_hierarchical_add_pair,
    bench_tracker_hierarchical_exists_pair,
    bench_tracker_hierarchical_claim_pair,
);

criterion_group!(
    iter_benches,
    bench_iter_sequential_set,
    bench_iter_sequential_all,
    bench_iter_random,
    bench_iter_write,
    bench_iter_write_saturation,
);

criterion_group!(
    group_benches,
    bench_group_iter_horizontal,
    bench_group_iter_vertical,
    bench_group_write_round_robin,
    bench_group_write_saturation,
);

criterion_group!(
    scaling_benches,
    bench_bitmap_scaling,
    bench_tracker_scaling,
);

criterion_group!(
    concurrent_benches,
    bench_concurrent_claims,
);

// =============================================================================
// SIMD Benchmarks
// =============================================================================

fn bench_simd_popcount(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/popcount");
    
    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let bitmap = AtomicBitmap::with_capacity(size);
        
        // Set 50% of bits
        for i in (0..size).step_by(2) {
            bitmap.set(i);
        }
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("recompute", size), &size, |b, _| {
            b.iter(|| {
                black_box(bitmap.recompute_count())
            })
        });
    }
    
    group.finish();
}

fn bench_simd_find_next(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/find_next");
    
    // Sparse bitmap (1% set)
    let sparse = AtomicBitmap::with_capacity(1_000_000);
    for i in (0..1_000_000).step_by(100) {
        sparse.set(i);
    }
    
    // Dense bitmap (99% set)
    let dense = AtomicBitmap::with_capacity(1_000_000);
    for i in 0..1_000_000 {
        if i % 100 != 0 {
            dense.set(i);
        }
    }
    
    group.bench_function("find_set/sparse_1pct", |b| {
        b.iter(|| {
            let mut pos = 0;
            let mut count = 0;
            while let Some(next) = sparse.find_next_set(pos) {
                count += 1;
                pos = next + 1;
                if count >= 100 { break; }
            }
            black_box(count)
        })
    });
    
    group.bench_function("find_unset/dense_99pct", |b| {
        b.iter(|| {
            let mut pos = 0;
            let mut count = 0;
            while let Some(next) = dense.find_next_unset(pos, 1_000_000) {
                count += 1;
                pos = next + 1;
                if count >= 100 { break; }
            }
            black_box(count)
        })
    });
    
    group.finish();
}

fn bench_simd_scan_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/full_scan");
    
    for size in [10_000, 100_000, 1_000_000] {
        let bitmap = AtomicBitmap::with_capacity(size);
        
        // Set every 10th bit
        for i in (0..size).step_by(10) {
            bitmap.set(i);
        }
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("find_all_set", size), &size, |b, &size| {
            b.iter(|| {
                let mut pos = 0;
                let mut count = 0;
                while let Some(next) = bitmap.find_next_set(pos) {
                    count += 1;
                    pos = next + 1;
                }
                black_box(count)
            })
        });
    }
    
    group.finish();
}

fn bench_hash_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash");
    
    // Benchmark hierarchical tracker operations (uses DashMap with AHash)
    let tracker = PrefixTracker::hierarchical("bench:");
    
    // Pre-populate
    for id in 0..1000u64 {
        for sub_id in 0..10u64 {
            tracker.add_pair(id, sub_id);
        }
    }
    
    group.bench_function("hierarchical/exists_pair", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let id = i % 1000;
            let sub_id = i % 10;
            i = i.wrapping_add(1);
            black_box(tracker.exists_pair(id, sub_id))
        })
    });
    
    group.bench_function("hierarchical/add_pair_existing", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let id = i % 1000;
            let sub_id = i % 10;
            i = i.wrapping_add(1);
            black_box(tracker.add_pair(id, sub_id))
        })
    });
    
    // Benchmark PrefixGroupsTracker operations
    let groups = PrefixGroupsTracker::new();
    for i in 0..100 {
        groups.register(TrackerConfig::simple(&format!("prefix{}:", i)));
    }
    
    group.bench_function("groups/get_existing", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let prefix = format!("prefix{}:", i % 100);
            i = i.wrapping_add(1);
            black_box(groups.get(&prefix))
        })
    });
    
    group.finish();
}

fn bench_random_iter_coprime(c: &mut Criterion) {
    // Benchmark the random iterator which uses coprime calculation
    let mut group = c.benchmark_group("random_iter");
    
    for size in [1_000, 10_000, 100_000] {
        let tracker = PrefixTracker::new(TrackerConfig::simple("rand:").with_max_id(size as u64));
        
        // Set 50% of bits
        for i in (0..size).step_by(2) {
            tracker.add(i as u64);
        }
        
        group.throughput(Throughput::Elements(size as u64 / 2));
        group.bench_with_input(BenchmarkId::new("iterate_set", size), &size, |b, _| {
            b.iter(|| {
                let items: Vec<_> = tracker.iter().set_only().random().collect();
                black_box(items.len())
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    simd_benches,
    bench_simd_popcount,
    bench_simd_find_next,
    bench_simd_scan_full,
);

criterion_group!(
    hash_benches,
    bench_hash_operations,
    bench_random_iter_coprime,
);

criterion_main!(
    bitmap_benches,
    tracker_simple_benches,
    tracker_hierarchical_benches,
    iter_benches,
    group_benches,
    scaling_benches,
    concurrent_benches,
    simd_benches,
    hash_benches,
);
