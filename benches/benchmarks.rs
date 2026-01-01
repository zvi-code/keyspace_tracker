use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use keyspace_tracker::{
    AtomicBitmap, PrefixTracker, PrefixGroupsTracker, TrackerConfig, ClaimPolicy, SamplingConfig,
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

    let mut group = c.benchmark_group("concurrent/claim");
    
    for threads in [1, 2, 4, 8, 16] {
        group.bench_with_input(BenchmarkId::new("disjoint", threads), &threads, |b, &threads| {
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
        
        // Contention: all threads claim from same small range
        group.bench_with_input(BenchmarkId::new("contention", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let tracker = Arc::new(PrefixTracker::new(
                    TrackerConfig::simple("test:").with_max_id(1000) // Small range = high contention
                ));
                
                let start = std::time::Instant::now();
                
                let handles: Vec<_> = (0..threads).map(|_| {
                    let tracker = tracker.clone();
                    thread::spawn(move || {
                        for i in 0..iters {
                            let id = i % 1000;
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

/// Benchmark concurrent bitmap set operations
fn bench_concurrent_bitmap_set(c: &mut Criterion) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    let mut group = c.benchmark_group("concurrent/bitmap");
    
    for threads in [1, 2, 4, 8, 16] {
        // Disjoint: each thread sets different bits
        group.bench_with_input(BenchmarkId::new("set_disjoint", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let bitmap = Arc::new(AtomicBitmap::with_capacity((iters * threads as u64 * 100) as usize));
                let counter = Arc::new(AtomicU64::new(0));
                
                let start = std::time::Instant::now();
                
                let handles: Vec<_> = (0..threads).map(|_| {
                    let bitmap = bitmap.clone();
                    let counter = counter.clone();
                    thread::spawn(move || {
                        for _ in 0..iters {
                            let idx = counter.fetch_add(1, Ordering::Relaxed) as usize;
                            black_box(bitmap.set(idx));
                        }
                    })
                }).collect();
                
                for h in handles {
                    h.join().unwrap();
                }
                
                start.elapsed()
            })
        });
        
        // Contention: all threads hit same bits
        group.bench_with_input(BenchmarkId::new("set_contention", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let bitmap = Arc::new(AtomicBitmap::with_capacity(1000));
                
                let start = std::time::Instant::now();
                
                let handles: Vec<_> = (0..threads).map(|_| {
                    let bitmap = bitmap.clone();
                    thread::spawn(move || {
                        for i in 0..iters {
                            let idx = (i % 1000) as usize;
                            black_box(bitmap.set(idx));
                        }
                    })
                }).collect();
                
                for h in handles {
                    h.join().unwrap();
                }
                
                start.elapsed()
            })
        });
        
        // test_and_set (CAS) with contention
        group.bench_with_input(BenchmarkId::new("test_and_set_contention", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let bitmap = Arc::new(AtomicBitmap::with_capacity(10000));
                
                let start = std::time::Instant::now();
                
                let handles: Vec<_> = (0..threads).map(|_| {
                    let bitmap = bitmap.clone();
                    thread::spawn(move || {
                        for i in 0..iters {
                            let idx = (i % 10000) as usize;
                            black_box(bitmap.test_and_set(idx));
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

/// Benchmark concurrent partitioned iteration (the main use case)
fn bench_concurrent_partition_iter(c: &mut Criterion) {
    use std::thread;

    let mut group = c.benchmark_group("concurrent/partition");
    group.sample_size(50); // Reduce sample size for longer-running benchmarks
    
    let size = 1_000_000u64;
    
    for threads in [1, 2, 4, 8, 16] {
        // Each thread iterates its own partition - no contention
        group.throughput(Throughput::Elements(size));
        group.bench_with_input(BenchmarkId::new("sequential_set_only", threads), &threads, |b, &threads| {
            // Pre-create tracker with data
            let tracker = Arc::new(PrefixTracker::new(
                TrackerConfig::simple("part:").with_max_id(size)
            ));
            for i in 0..size {
                tracker.add(i);
            }
            
            b.iter(|| {
                let handles: Vec<_> = (0..threads).map(|thread_id| {
                    let tracker = tracker.clone();
                    thread::spawn(move || {
                        let mut count = 0u64;
                        for (id, _) in tracker.iter()
                            .set_only()
                            .partition(thread_id, threads)
                        {
                            black_box(id);
                            count += 1;
                        }
                        count
                    })
                }).collect();
                
                let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
                black_box(total)
            })
        });
        
        // Partitioned iteration (PartitionedIter is already a concrete iterator)
        group.bench_with_input(BenchmarkId::new("partitioned_iterate", threads), &threads, |b, &threads| {
            let tracker = Arc::new(PrefixTracker::new(
                TrackerConfig::simple("part:").with_max_id(size)
            ));
            for i in 0..size {
                tracker.add(i);
            }
            
            b.iter(|| {
                let handles: Vec<_> = (0..threads).map(|thread_id| {
                    let tracker = tracker.clone();
                    thread::spawn(move || {
                        let mut count = 0u64;
                        // PartitionedIter is already a sequential iterator over its range
                        for (id, _) in tracker.iter()
                            .set_only()
                            .partition(thread_id, threads)
                        {
                            black_box(id);
                            count += 1;
                        }
                        count
                    })
                }).collect();
                
                let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
                black_box(total)
            })
        });
    }
    
    group.finish();
}

/// Benchmark concurrent write iteration (claiming unset IDs)
fn bench_concurrent_write_iter(c: &mut Criterion) {
    use std::thread;

    let mut group = c.benchmark_group("concurrent/write_iter");
    group.sample_size(50);
    
    let size = 100_000u64;
    
    for threads in [1, 2, 4, 8, 16] {
        group.throughput(Throughput::Elements(size));
        
        // WriteIter claiming from empty tracker
        group.bench_with_input(BenchmarkId::new("claim_empty", threads), &threads, |b, &threads| {
            b.iter(|| {
                let tracker = Arc::new(PrefixTracker::new(
                    TrackerConfig::simple("write:").with_max_id(size)
                ));
                
                let handles: Vec<_> = (0..threads).map(|_| {
                    let tracker = tracker.clone();
                    thread::spawn(move || {
                        let mut count = 0u64;
                        let mut writer = tracker.iter().unset_only().write();
                        while let Some(item) = writer.next() {
                            black_box(item);
                            count += 1;
                        }
                        count
                    })
                }).collect();
                
                let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
                assert_eq!(total, size);
                black_box(total)
            })
        });
        
        // Partitioned claiming (each thread claims IDs in its partition)
        group.bench_with_input(BenchmarkId::new("claim_partitioned", threads), &threads, |b, &threads| {
            b.iter(|| {
                let tracker = Arc::new(PrefixTracker::new(
                    TrackerConfig::simple("write:").with_max_id(size)
                ));
                
                let handles: Vec<_> = (0..threads).map(|thread_id| {
                    let tracker = tracker.clone();
                    thread::spawn(move || {
                        let mut count = 0u64;
                        // Iterate partition and claim each ID
                        for (id, _) in tracker.iter()
                            .unset_only()
                            .partition(thread_id, threads)
                        {
                            if tracker.claim(id) {
                                black_box(id);
                                count += 1;
                            }
                        }
                        count
                    })
                }).collect();
                
                let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
                black_box(total)
            })
        });
    }
    
    group.finish();
}

/// Benchmark concurrent tracker add/exists operations
fn bench_concurrent_tracker_ops(c: &mut Criterion) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    let mut group = c.benchmark_group("concurrent/tracker");
    
    let size = 1_000_000u64;
    
    for threads in [1, 2, 4, 8, 16] {
        // Concurrent adds to different IDs
        group.bench_with_input(BenchmarkId::new("add_disjoint", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let tracker = Arc::new(PrefixTracker::new(
                    TrackerConfig::simple("test:").with_max_id(size)
                ));
                let counter = Arc::new(AtomicU64::new(0));
                
                let start = std::time::Instant::now();
                
                let handles: Vec<_> = (0..threads).map(|_| {
                    let tracker = tracker.clone();
                    let counter = counter.clone();
                    thread::spawn(move || {
                        for _ in 0..iters {
                            let id = counter.fetch_add(1, Ordering::Relaxed) % size;
                            black_box(tracker.add(id));
                        }
                    })
                }).collect();
                
                for h in handles {
                    h.join().unwrap();
                }
                
                start.elapsed()
            })
        });
        
        // Concurrent exists checks (read-heavy, should scale well)
        group.bench_with_input(BenchmarkId::new("exists_readonly", threads), &threads, |b, &threads| {
            let tracker = Arc::new(PrefixTracker::new(
                TrackerConfig::simple("test:").with_max_id(size)
            ));
            // Pre-populate 50%
            for i in (0..size).step_by(2) {
                tracker.add(i);
            }
            
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                
                let handles: Vec<_> = (0..threads).map(|thread_id| {
                    let tracker = tracker.clone();
                    thread::spawn(move || {
                        let offset = thread_id as u64 * 12345; // Different starting points
                        for i in 0..iters {
                            let id = (offset + i) % size;
                            black_box(tracker.exists(id));
                        }
                    })
                }).collect();
                
                for h in handles {
                    h.join().unwrap();
                }
                
                start.elapsed()
            })
        });
        
        // Mixed read/write (80% read, 20% write)
        group.bench_with_input(BenchmarkId::new("mixed_80_20", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let tracker = Arc::new(PrefixTracker::new(
                    TrackerConfig::simple("test:").with_max_id(size)
                ));
                // Pre-populate 50%
                for i in (0..size).step_by(2) {
                    tracker.add(i);
                }
                
                let start = std::time::Instant::now();
                
                let handles: Vec<_> = (0..threads).map(|thread_id| {
                    let tracker = tracker.clone();
                    thread::spawn(move || {
                        let offset = thread_id as u64 * 12345;
                        for i in 0..iters {
                            let id = (offset + i) % size;
                            if i % 5 == 0 {
                                // 20% writes
                                black_box(tracker.add(id));
                            } else {
                                // 80% reads
                                black_box(tracker.exists(id));
                            }
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

/// Benchmark concurrent group operations
fn bench_concurrent_group_ops(c: &mut Criterion) {
    use std::thread;

    let mut group = c.benchmark_group("concurrent/group");
    group.sample_size(50);
    
    for threads in [1, 2, 4, 8] {
        // Concurrent registration and lookup
        group.bench_with_input(BenchmarkId::new("register_lookup", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let groups = Arc::new(PrefixGroupsTracker::new());
                
                // Pre-register some prefixes
                for i in 0..100 {
                    groups.register(TrackerConfig::simple(&format!("prefix{}:", i)));
                }
                
                let start = std::time::Instant::now();
                
                let handles: Vec<_> = (0..threads).map(|thread_id| {
                    let groups = groups.clone();
                    thread::spawn(move || {
                        for i in 0..iters {
                            let prefix = format!("prefix{}:", (thread_id as u64 * 1000 + i) % 100);
                            if i % 10 == 0 {
                                // 10% new registrations
                                black_box(groups.register(TrackerConfig::simple(&format!("new{}_{}", thread_id, i))));
                            } else {
                                // 90% lookups
                                black_box(groups.get(&prefix));
                            }
                        }
                    })
                }).collect();
                
                for h in handles {
                    h.join().unwrap();
                }
                
                start.elapsed()
            })
        });
        
        // Concurrent group iteration with write
        group.bench_with_input(BenchmarkId::new("write_round_robin", threads), &threads, |b, &threads| {
            b.iter(|| {
                let groups = Arc::new(PrefixGroupsTracker::new());
                for i in 0..4 {
                    groups.register(TrackerConfig::simple(&format!("g{}:", i)).with_max_id(10000));
                }
                
                let handles: Vec<_> = (0..threads).map(|_| {
                    let groups = groups.clone();
                    thread::spawn(move || {
                        let mut writer = groups.iter().write(ClaimPolicy::RoundRobin);
                        let mut count = 0u64;
                        while let Some(item) = writer.next() {
                            black_box(item);
                            count += 1;
                        }
                        count
                    })
                }).collect();
                
                let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
                black_box(total)
            })
        });
    }
    
    group.finish();
}

/// Benchmark concurrent hierarchical tracker operations
fn bench_concurrent_hierarchical(c: &mut Criterion) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    let mut group = c.benchmark_group("concurrent/hierarchical");
    
    for threads in [1, 2, 4, 8] {
        // Concurrent add_pair operations
        group.bench_with_input(BenchmarkId::new("add_pair", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let tracker = Arc::new(PrefixTracker::new(
                    TrackerConfig::hierarchical("hash:")
                        .with_max_id(10000)
                        .with_max_sub_id(100)
                ));
                let counter = Arc::new(AtomicU64::new(0));
                
                let start = std::time::Instant::now();
                
                let handles: Vec<_> = (0..threads).map(|_| {
                    let tracker = tracker.clone();
                    let counter = counter.clone();
                    thread::spawn(move || {
                        for _ in 0..iters {
                            let n = counter.fetch_add(1, Ordering::Relaxed);
                            let id = n % 10000;
                            let sub_id = n % 100;
                            black_box(tracker.add_pair(id, sub_id));
                        }
                    })
                }).collect();
                
                for h in handles {
                    h.join().unwrap();
                }
                
                start.elapsed()
            })
        });
        
        // Concurrent claim_pair operations
        group.bench_with_input(BenchmarkId::new("claim_pair", threads), &threads, |b, &threads| {
            b.iter_custom(|iters| {
                let tracker = Arc::new(PrefixTracker::new(
                    TrackerConfig::hierarchical("hash:")
                        .with_max_id(10000)
                        .with_max_sub_id(100)
                ));
                let counter = Arc::new(AtomicU64::new(0));
                
                let start = std::time::Instant::now();
                
                let handles: Vec<_> = (0..threads).map(|_| {
                    let tracker = tracker.clone();
                    let counter = counter.clone();
                    thread::spawn(move || {
                        for _ in 0..iters {
                            let n = counter.fetch_add(1, Ordering::Relaxed);
                            let id = n % 10000;
                            let sub_id = n % 100;
                            black_box(tracker.claim_pair(id, sub_id));
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
    bench_concurrent_bitmap_set,
    bench_concurrent_partition_iter,
    bench_concurrent_write_iter,
    bench_concurrent_tracker_ops,
    bench_concurrent_group_ops,
    bench_concurrent_hierarchical,
);

// =============================================================================
// Sampling & New Iterator Benchmarks
// =============================================================================

/// Benchmark limit-based iteration (per-key cost)
fn bench_iter_limit(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/limit");
    
    for size in [10_000u64, 100_000, 1_000_000] {
        let tracker = PrefixTracker::new(
            TrackerConfig::simple("vec:").with_max_id(size)
        );
        for i in 0..size {
            tracker.add(i);
        }
        
        // Benchmark: iterate with limit (take first N)
        let limit = 1000u64;
        group.throughput(Throughput::Elements(limit));
        
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |b, _| {
            b.iter(|| {
                let count = tracker.iter()
                    .set_only()
                    .limit(limit)
                    .sequential()
                    .count();
                black_box(count)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("random", size), &size, |b, _| {
            b.iter(|| {
                let count = tracker.iter()
                    .set_only()
                    .limit(limit)
                    .random()
                    .count();
                black_box(count)
            })
        });
    }
    
    group.finish();
}

/// Benchmark probabilistic sampling (per-key cost)
fn bench_iter_sample_probability(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/probability");
    
    let size = 100_000u64;
    let tracker = PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(size)
    );
    for i in 0..size {
        tracker.add(i);
    }
    
    for probability in [0.1, 0.5, 0.9] {
        let expected = (size as f64 * probability) as u64;
        group.throughput(Throughput::Elements(expected));
        
        group.bench_with_input(
            BenchmarkId::new("sequential", format!("{:.0}%", probability * 100.0)),
            &probability,
            |b, &prob| {
                b.iter(|| {
                    let count = tracker.iter()
                        .set_only()
                        .sample(prob)
                        .seed(42)
                        .sequential()
                        .count();
                    black_box(count)
                })
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("random", format!("{:.0}%", probability * 100.0)),
            &probability,
            |b, &prob| {
                b.iter(|| {
                    let count = tracker.iter()
                        .set_only()
                        .sample(prob)
                        .seed(42)
                        .random()
                        .count();
                    black_box(count)
                })
            }
        );
    }
    
    group.finish();
}

/// Benchmark mixed-ratio iteration (X% existing + Y% new)
fn bench_iter_mixed_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/mixed_ratio");
    
    let size = 100_000u64;
    let tracker = PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(size)
    );
    // Set 50% of bits
    for i in 0..size / 2 {
        tracker.add(i);
    }
    
    let iterations = 10_000u64;
    group.throughput(Throughput::Elements(iterations));
    
    for ratio in [0.1, 0.5, 0.9] {
        group.bench_with_input(
            BenchmarkId::new("random", format!("{:.0}%_set", ratio * 100.0)),
            &ratio,
            |b, &ratio| {
                b.iter(|| {
                    let count = tracker.iter()
                        .mixed_ratio(ratio)
                        .seed(42)
                        .limit(iterations)
                        .random()
                        .count();
                    black_box(count)
                })
            }
        );
    }
    
    group.finish();
}

/// Benchmark seeded random iteration (reproducibility)
fn bench_iter_seeded_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/seeded");
    
    for size in [10_000u64, 100_000, 1_000_000] {
        let tracker = PrefixTracker::new(
            TrackerConfig::simple("vec:").with_max_id(size)
        );
        for i in (0..size).step_by(2) {
            tracker.add(i);
        }
        
        let iterations = 1000u64;
        group.throughput(Throughput::Elements(iterations));
        
        group.bench_with_input(BenchmarkId::new("with_seed", size), &size, |b, _| {
            b.iter(|| {
                let count = tracker.iter()
                    .set_only()
                    .seed(12345)
                    .limit(iterations)
                    .random()
                    .count();
                black_box(count)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("without_seed", size), &size, |b, _| {
            b.iter(|| {
                let count = tracker.iter()
                    .set_only()
                    .limit(iterations)
                    .random()
                    .count();
                black_box(count)
            })
        });
    }
    
    group.finish();
}

/// Benchmark range_percent iteration (upper/lower X% of keyspace)
fn bench_iter_range_percent(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/range_percent");
    
    let size = 100_000u64;
    let tracker = PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(size)
    );
    for i in 0..size {
        tracker.add(i);
    }
    
    // Benchmark different range slices
    for (start_pct, end_pct) in [(0.0, 0.5), (0.5, 1.0), (0.25, 0.75)] {
        let expected = (size as f64 * (end_pct - start_pct)) as u64;
        group.throughput(Throughput::Elements(expected));
        
        group.bench_with_input(
            BenchmarkId::new("sequential", format!("{:.0}-{:.0}%", start_pct * 100.0, end_pct * 100.0)),
            &(start_pct, end_pct),
            |b, &(start, end)| {
                b.iter(|| {
                    let count = tracker.iter()
                        .set_only()
                        .range_percent(start, end)
                        .sequential()
                        .count();
                    black_box(count)
                })
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("random", format!("{:.0}-{:.0}%", start_pct * 100.0, end_pct * 100.0)),
            &(start_pct, end_pct),
            |b, &(start, end)| {
                b.iter(|| {
                    let count = tracker.iter()
                        .set_only()
                        .range_percent(start, end)
                        .random()
                        .count();
                    black_box(count)
                })
            }
        );
    }
    
    group.finish();
}

/// Benchmark partitioned iteration with sampling
fn bench_iter_partitioned_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/partitioned");
    
    let size = 100_000u64;
    let tracker = Arc::new(PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(size)
    ));
    for i in 0..size {
        tracker.add(i);
    }
    
    for num_partitions in [2, 4, 8] {
        group.throughput(Throughput::Elements(size));
        
        // Without sampling
        group.bench_with_input(
            BenchmarkId::new("plain", num_partitions),
            &num_partitions,
            |b, &n| {
                b.iter(|| {
                    let partitions = tracker.iter().set_only().partitioned(n);
                    let total: usize = partitions.into_iter()
                        .map(|p| p.count())
                        .sum();
                    black_box(total)
                })
            }
        );
        
        // With limit per partition
        let limit_per_part = 1000u64;
        group.throughput(Throughput::Elements(limit_per_part * num_partitions as u64));
        
        group.bench_with_input(
            BenchmarkId::new("with_limit", num_partitions),
            &num_partitions,
            |b, &n| {
                b.iter(|| {
                    let partitions = tracker.iter()
                        .set_only()
                        .limit(limit_per_part)
                        .partitioned(n);
                    let total: usize = partitions.into_iter()
                        .map(|p| p.count())
                        .sum();
                    black_box(total)
                })
            }
        );
    }
    
    group.finish();
}

/// Benchmark bulk bitmap operations (set_range, clear_range, count_range)
fn bench_bitmap_bulk_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitmap/bulk");
    
    for size in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        
        // set_range benchmark
        group.bench_with_input(BenchmarkId::new("set_range", size), &size, |b, &size| {
            let bitmap = AtomicBitmap::with_capacity(size);
            b.iter(|| {
                bitmap.clear_range(0, size); // Reset for fair comparison
                let count = bitmap.set_range(0, size);
                black_box(count)
            })
        });
        
        // clear_range benchmark
        group.bench_with_input(BenchmarkId::new("clear_range", size), &size, |b, &size| {
            let bitmap = AtomicBitmap::with_capacity(size);
            bitmap.set_range(0, size); // Pre-set all
            b.iter(|| {
                bitmap.set_range(0, size); // Reset to all set
                let count = bitmap.clear_range(0, size);
                black_box(count)
            })
        });
        
        // count_range benchmark
        group.bench_with_input(BenchmarkId::new("count_range", size), &size, |b, &size| {
            let bitmap = AtomicBitmap::with_capacity(size);
            // Set 50% of bits
            for i in (0..size).step_by(2) {
                bitmap.set(i);
            }
            b.iter(|| {
                let count = bitmap.count_range(0, size);
                black_box(count)
            })
        });
    }
    
    group.finish();
}

/// Benchmark with_sampling config (SamplingConfig struct)
fn bench_iter_with_sampling_config(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/config");
    
    let size = 100_000u64;
    let tracker = PrefixTracker::new(
        TrackerConfig::simple("vec:").with_max_id(size)
    );
    for i in 0..size {
        tracker.add(i);
    }
    
    // Complex sampling config: 90% set, 50% sample, limit 1000, seeded
    let config = SamplingConfig::new()
        .with_set_ratio(0.9)
        .with_sample_probability(0.5)
        .with_limit(1000)
        .with_seed(42);
    
    group.throughput(Throughput::Elements(1000));
    
    group.bench_function("complex_config/random", |b| {
        b.iter(|| {
            let count = tracker.iter()
                .with_sampling(config)
                .random()
                .count();
            black_box(count)
        })
    });
    
    group.bench_function("complex_config/sequential", |b| {
        b.iter(|| {
            let count = tracker.iter()
                .with_sampling(config)
                .sequential()
                .count();
            black_box(count)
        })
    });
    
    group.finish();
}

/// Per-key latency benchmark (< 1µs requirement)
fn bench_per_key_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency/per_key");
    group.sample_size(1000);
    
    // Test various sizes to ensure consistent per-key performance
    for size in [10_000u64, 100_000, 1_000_000, 10_000_000] {
        let tracker = PrefixTracker::new(
            TrackerConfig::simple("vec:").with_max_id(size)
        );
        // Set 50% of bits
        for i in (0..size).step_by(2) {
            tracker.add(i);
        }
        
        let ops_per_iter = 10_000u64;
        group.throughput(Throughput::Elements(ops_per_iter));
        
        // Sequential iteration per-key
        group.bench_with_input(
            BenchmarkId::new("sequential/set_only", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let count = tracker.iter()
                        .set_only()
                        .limit(ops_per_iter)
                        .sequential()
                        .count();
                    black_box(count)
                })
            }
        );
        
        // Random iteration per-key
        group.bench_with_input(
            BenchmarkId::new("random/set_only", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let count = tracker.iter()
                        .set_only()
                        .seed(42)
                        .limit(ops_per_iter)
                        .random()
                        .count();
                    black_box(count)
                })
            }
        );
        
        // Mixed ratio iteration per-key
        group.bench_with_input(
            BenchmarkId::new("mixed_ratio/90_10", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let count = tracker.iter()
                        .mixed_ratio(0.9)
                        .seed(42)
                        .limit(ops_per_iter)
                        .random()
                        .count();
                    black_box(count)
                })
            }
        );
        
        // Sampling iteration per-key
        group.bench_with_input(
            BenchmarkId::new("sample/50pct", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let count = tracker.iter()
                        .set_only()
                        .sample(0.5)
                        .seed(42)
                        .limit(ops_per_iter)
                        .sequential()
                        .count();
                    black_box(count)
                })
            }
        );
    }
    
    group.finish();
}

/// Benchmark delete iterator with sampling
fn bench_iter_delete_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/delete");
    
    group.bench_function("delete_50pct", |b| {
        b.iter_custom(|iters| {
            let size = 10_000u64;
            let tracker = PrefixTracker::new(
                TrackerConfig::simple("vec:").with_max_id(size)
            );
            for i in 0..size {
                tracker.add(i);
            }
            
            let delete_count = size / 2;
            let start = std::time::Instant::now();
            
            if let Some(mut del_iter) = tracker.iter().delete() {
                for _ in 0..iters.min(delete_count) {
                    black_box(del_iter.next());
                }
            }
            
            start.elapsed()
        })
    });
    
    group.finish();
}

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
        group.bench_with_input(BenchmarkId::new("find_all_set", size), &size, |b, _size| {
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

criterion_group!(
    sampling_benches,
    bench_iter_limit,
    bench_iter_sample_probability,
    bench_iter_mixed_ratio,
    bench_iter_seeded_random,
    bench_iter_range_percent,
    bench_iter_partitioned_sampling,
    bench_bitmap_bulk_operations,
    bench_iter_with_sampling_config,
    bench_per_key_latency,
    bench_iter_delete_sampling,
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
    sampling_benches,
);
