# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Run Commands

- Build all crates: `cargo build --release`
- Build specific crate: `cargo build --release -p core`
- Run core crate: `cargo run --release -p core`
- Run with ticker: `cargo run --release -p core -- -t AAPL`
- Test all crates: `cargo test`
- Test specific crate: `cargo test -p core` or `cargo test -p simulations`
- Run single test: `cargo test -p core test_name`
- Run ignored test: `cargo test -p core -- --ignored test_name`
- Benchmark core: `cargo bench -p core`
- Benchmark simulations: `cargo bench -p simulations`
- Format all crates: `cargo fmt`
- Lint all crates: `cargo clippy`

## Code Style Guidelines

- Use Rust 2024 edition style
- Use workspace dependencies in Cargo.toml files
- Structure code with core crate for main functionality and simulations for calculation logic
- Group imports with std first, then external crates, alphabetically within groups
- Use doc comments (`///`) for public functions/structs with complete parameter documentation
- Use descriptive variable names in snake_case
- Prefer Result/Option for error handling over panics
- Use rayon for parallelism when processing data
- Include test cases for public functionality
- Write robust error handling for network operations
- Cache API responses when possible to reduce network calls
- Use strongly typed structs when deserializing external data
