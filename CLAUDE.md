# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Run Commands

- Build: `cargo build --release`
- Run: `cargo run --release`
- Run with ticker: `cargo run --release -- -t AAPL`
- Test: `cargo test`
- Run single test: `cargo test test_name`
- Run ignored test: `cargo test -- --ignored test_name`
- Benchmark: `cargo bench`
- Format: `cargo fmt`
- Lint: `cargo clippy`

## Code Style Guidelines

- Use Rust 2021 edition style
- Group imports with std first, then external crates
- Use doc comments (`///`) for public functions/structs
- Use descriptive variable names in snake_case
- Prefer Result/Option for error handling over panics
- Use rayon for parallelism when processing data
- In Python code, follow PEP 8 style guidelines
- Write robust error handling for network operations
- Cache API responses when possible to reduce network calls
- Use strongly typed structs when deserializing external data
