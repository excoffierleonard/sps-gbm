# Stock Price Simulation (SPS-GBM)

A tool for simulating future stock prices using Geometric Brownian Motion (GBM) in Rust.

## Overview

This tool simulates potential future stock prices based on historical data using the Geometric Brownian Motion model, a standard approach in financial modeling.

## Features

- Fetch historical stock data from Alpha Vantage API
- Calculate drift and volatility parameters from historical data
- Run multiple simulations in parallel using Rayon
- Visualize results with price paths and distribution histograms
- Cached API responses to reduce network calls
- Command-line interface with customizable parameters

## Installation

```bash
git clone https://github.com/excoffierleonard/sps-gbm.git
cd sps-gbm
cargo build --release
```

You'll need to create a `.env` file with your Alpha Vantage API key:

```
ALPHAVANTAGE_API_KEY=your_api_key_here
```

## Usage

```bash
# Run with default parameters (AAPL stock)
cargo run --release

# Run with a specific ticker
cargo run --release -- -t AAPL

# Run with custom parameters
cargo run --release -- -t MSFT -f 2023-01-01 -u 2023-12-31 -s 200 -n 500
```

### Command Line Options

- `-t, --ticker`: Stock ticker symbol (default: AAPL)
- `-f, --start-date`: Start date for historical data (default: 2023-01-01)
- `-u, --end-date`: End date for historical data (default: 2024-12-31)
- `-s, --steps`: Number of simulation steps (default: 100)
- `-n, --paths`: Number of simulation paths (default: 100)

## Example Output

```
Stock Ticker: AAPL
Annualized Mean Return (μ): 0.5785
Annualized Volatility (σ): 0.1984
Most Recent Closing Price: 192.02

Summary of Predicted Stock Prices after 366 days (on 2025-01-01):
Mean Final Price: 345.32 (79.83%)
Median Final Price: 339.59
Standard Deviation of Final Prices: 70.44
95% Confidence Interval: (207.26, 483.38)
Percentiles: 10th 257.40, 25th 293.76, 75th 386.72, 90th 443.00
```

![Simulated Stock Price Paths & Summary](example.png)

## License

MIT License - See [LICENSE](LICENSE) file for details.
