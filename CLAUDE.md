# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- `cargo build --release`: Build the Rust project in release mode
- `cargo run --release`: Run the Rust simulation
- `cargo run --release -- -t <TICKER>`: Run with a specific stock ticker
- `cargo run --release -- -h`: Display help and command options
- `cargo test`: Run all Rust tests
- `python main.py`: Run the legacy Python version

## Code Style Guidelines

### Rust Guidelines
- Use snake_case for variables, functions, and file names
- Use struct-based organization for data
- Handle errors with anyhow::Result and ? operator
- Use descriptive variable names that reflect their purpose
- Always validate user inputs
- Include inline documentation for functions
- Use the clap crate for command-line parsing

### Python Guidelines (legacy code)
- Follow PEP 8 style for Python code
- Use snake_case for variables and functions
- Group related functionality into separate functions
- Handle exceptions appropriately
- Use pandas for data manipulation
- Use numpy for numerical operations

### General
- Add TODO/FIXME comments for future improvements
- Use constants for magic numbers
- Ensure proper error messages are displayed to users
- Maintain type safety and use appropriate data structures
- Follow Repository conventions for file organization