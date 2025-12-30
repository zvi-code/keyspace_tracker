# Benchmarking Guide for keyspace_tracker

## TL;DR - Maximum Performance

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --release
```

This single command enables:
- Native CPU instruction set (NEON on ARM, AVX2 on x86)
- Full LTO (cross-crate inlining)
- Single codegen unit (global optimization)
- Panic=abort (no unwinding overhead)

---

## Optimization Flags Explained

### 1. CPU Target (`-C target-cpu=native`)

**This is the most impactful flag.**

| Flag | Effect |
|------|--------|
| `native` | Auto-detect CPU, use all available features |
| `apple-m1` | Apple Silicon (M1/M2/M3/M4) |
| `neoverse-v1` | AWS Graviton3 |
| `neoverse-n1` | AWS Graviton2 |
| `haswell` | Intel Haswell+ / AMD Zen+ |

On Apple M2, `native` enables:
- ARMv8.5 instructions
- Aggressive NEON vectorization
- Apple-specific instruction scheduling

### 2. Link-Time Optimization (`lto = "fat"`)

```toml
[profile.bench]
lto = "fat"
```

| Option | Effect |
|--------|--------|
| `"fat"` | Full LTO, best optimization, slowest compile |
| `"thin"` | Parallel LTO, good optimization, faster compile |
| `false` | No LTO |

Fat LTO allows:
- Cross-crate inlining
- Dead code elimination
- Hot loop optimization across module boundaries

### 3. Codegen Units (`codegen-units = 1`)

```toml
[profile.bench]
codegen-units = 1
```

LLVM optimizes the entire crate as a single unit instead of parallel shards.
Trade-off: Slower compilation.

### 4. Panic Strategy (`panic = "abort"`)

```toml
[profile.bench]
panic = "abort"
```

Skips unwinding machinery. Saves ~1-2% on hot paths.
Only use if you don't need panic recovery.

### 5. Overflow Checks (`overflow-checks = false`)

```toml
[profile.bench]
overflow-checks = false
```

Disables integer overflow checks in release. Already default in release, but explicit is clearer.

---

## Platform-Specific Configuration

### Apple Silicon (M1/M2/M3/M4)

`.cargo/config.toml`:
```toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=apple-m1"]
```

Or use `native` which auto-detects:
```bash
RUSTFLAGS="-C target-cpu=native" cargo bench
```

### AWS Graviton

| Instance | CPU | Config |
|----------|-----|--------|
| c6g, m6g, r6g | Graviton2 | `target-cpu=neoverse-n1` |
| c7g, m7g, r7g | Graviton3 | `target-cpu=neoverse-v1` |
| c8g, m8g, r8g | Graviton4 | `target-cpu=neoverse-v2` |

`.cargo/config.toml`:
```toml
[target.aarch64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=neoverse-v1"]
```

### x86_64

| CPU | Config |
|-----|--------|
| Intel Haswell+ | `target-cpu=haswell` |
| Intel Skylake-X | `target-cpu=skylake-avx512` |
| AMD Zen3 | `target-cpu=znver3` |
| AMD Zen4 | `target-cpu=znver4` |

---

## Advanced Optimizations

### Nightly-Only Flags

```bash
# Extra CPU tuning
RUSTFLAGS="-C target-cpu=native -Z tune-cpu=native" \
  cargo +nightly bench --release
```

### Fast-Math (Floating Point)

⚠️ Only if you don't need strict IEEE compliance:

```bash
RUSTFLAGS="-C target-cpu=native -C llvm-args=-enable-no-nans-fp-math" \
  cargo bench --release
```

This enables:
- Reordering of FP operations
- Assuming no NaNs/infinities
- Fused multiply-add

### Verify Vectorization

Check if NEON/AVX instructions are generated:

```bash
# Install cargo-asm
cargo install cargo-asm

# View assembly for a function
cargo asm keyspace_tracker::keyspace_tracker::bitmap_simd::neon::popcount_slice
```

Look for:
- ARM: `cnt`, `fmla`, `ld1`, `st1`
- x86: `vpopcnt`, `vpand`, `vmovdqu`

---

## Benchmark Stability Tips

### macOS

```bash
# Disable App Nap
defaults write NSGlobalDomain NSAppSleepDisabled -bool YES

# Close heavy apps (Chrome, Docker, Slack)

# Plug in power (avoid thermal throttling)

# Use taskpolicy for consistent scheduling
taskpolicy -c utility cargo bench --release
```

### Linux

```bash
# Disable CPU frequency scaling
sudo cpupower frequency-set --governor performance

# Pin to specific CPUs
taskset -c 0-3 cargo bench --release

# Disable turbo boost for consistency
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

### General

- Run benchmarks multiple times
- Check the confidence intervals in criterion output
- Avoid running during system updates/background tasks
- Warm up the CPU before measuring

---

## Running Specific Benchmarks

```bash
# All benchmarks
RUSTFLAGS="-C target-cpu=native" cargo bench --release

# Bitmap operations only
cargo bench --release -- bitmap/

# SIMD-specific tests
cargo bench --release -- simd/

# Group operations
cargo bench --release -- group/

# Scaling tests
cargo bench --release -- scaling/
```

---

## Comparing Results

Criterion saves results in `target/criterion/`. To compare:

```bash
# Run baseline
cargo bench --release -- --save-baseline main

# Make changes, then compare
cargo bench --release -- --baseline main
```

---

## Expected Performance

With full optimizations on Apple M2:

| Operation | Expected Latency |
|-----------|-----------------|
| `test` (exists) | ~1.5 ns |
| `set` (add) | ~7 ns |
| `test_and_set` (claim) | ~2.5 ns |
| `find_next_set` | ~3-5 ns |
| Hierarchical exists | ~15-20 ns |

Throughput scales linearly with core count for parallel operations.

---

## Cargo.toml Reference

```toml
[profile.bench]
opt-level = 3           # Maximum optimization
lto = "fat"             # Full link-time optimization
codegen-units = 1       # Single compilation unit
panic = "abort"         # No unwinding
overflow-checks = false # No integer overflow checks
debug = false           # No debug symbols
```

---

## Troubleshooting

### Benchmark variance too high

- Check for background processes
- Disable turbo boost
- Use `taskset` to pin to specific cores

### Not seeing SIMD instructions

- Verify `target-cpu` is set correctly
- Check `cargo asm` output
- Ensure using `--release`

### LTO taking too long

- Use `lto = "thin"` for faster builds
- Increase RAM if swapping during link
