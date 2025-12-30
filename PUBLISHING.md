# Publishing prefix_tracker to crates.io

## Prerequisites

1. **Create a crates.io account**: Go to https://crates.io and sign in with GitHub

2. **Get your API token**: 
   - Go to https://crates.io/settings/tokens
   - Click "New Token"
   - Save the token securely

3. **Login with cargo**:
   ```bash
   cargo login <your-api-token>
   ```

## Pre-publish Checklist

### 1. Update Cargo.toml

Edit these fields in `Cargo.toml`:

```toml
[package]
authors = ["Your Name <your.email@example.com>"]
repository = "https://github.com/yourusername/prefix_tracker"
```

### 2. Verify the package

```bash
# Check what will be published
cargo package --list

# Verify package builds
cargo package

# Run all tests
cargo test

# Run doc tests
cargo test --doc

# Check documentation renders correctly
cargo doc --open
```

### 3. Check for issues

```bash
# Lint
cargo clippy

# Check formatting
cargo fmt --check

# Verify all examples compile
cargo build --examples  # if you have examples/
```

## Publishing

### First-time publish

```bash
cargo publish
```

### Dry run (recommended first)

```bash
cargo publish --dry-run
```

## After Publishing

Your crate will be available at:
- **Crates.io**: https://crates.io/crates/prefix_tracker
- **Documentation**: https://docs.rs/prefix_tracker (auto-generated)

## Version Bumping

For subsequent releases:

1. Update `version` in `Cargo.toml` (follow SemVer)
2. Update CHANGELOG.md (if you have one)
3. Commit and tag:
   ```bash
   git add -A
   git commit -m "Release v0.2.0"
   git tag v0.2.0
   git push && git push --tags
   ```
4. Publish:
   ```bash
   cargo publish
   ```

## Generating Documentation Locally

```bash
# Generate docs
cargo doc --no-deps

# Open in browser
cargo doc --no-deps --open

# Generate with all features documented
RUSTDOCFLAGS="--cfg docsrs" cargo doc --all-features --no-deps
```

## Documentation Tips

### Doc comments

- Use `///` for item documentation
- Use `//!` for module-level documentation
- Use ` ```rust ` for code examples (auto-tested!)
- Use ` ```rust,ignore ` for examples that shouldn't be tested
- Use ` ```rust,no_run ` for examples that compile but shouldn't run

### Intra-doc links

Link to other items in your crate:
```rust
/// See [`AtomicBitmap`] for the underlying implementation.
/// Use [`PrefixTracker::new`] to create with custom config.
```

### Hiding implementation details

```rust
#[doc(hidden)]
pub fn internal_function() { }
```

### Feature flags in docs

```rust
#[cfg(feature = "advanced")]
#[cfg_attr(docsrs, doc(cfg(feature = "advanced")))]
pub fn advanced_feature() { }
```

## Common Issues

### "crate already exists"
- Crate names are unique globally
- Pick a different name or ask the owner to transfer

### "invalid category"
- Check valid categories at https://crates.io/category_slugs

### Documentation not building on docs.rs
- Check the docs.rs build log at https://docs.rs/crate/prefix_tracker/latest/builds
- Ensure examples compile without external dependencies

## File Structure for Publishing

```
prefix_tracker/
├── Cargo.toml          # Package metadata
├── Cargo.lock          # Dependency lock (optional for libs)
├── LICENSE             # Required
├── README.md           # Shown on crates.io
├── CHANGELOG.md        # Recommended
├── src/
│   ├── lib.rs          # Crate root with //! docs
│   └── ...
├── benches/            # Benchmarks
├── examples/           # Example programs (optional)
└── tests/              # Integration tests (optional)
```

## Useful Commands

```bash
# Check package contents before publishing
cargo package --list

# Install your crate locally to test
cargo install --path .

# Yank a version (remove from index, doesn't delete)
cargo yank --version 0.1.0

# Un-yank
cargo yank --version 0.1.0 --undo
```
