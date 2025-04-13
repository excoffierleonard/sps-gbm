# Stock Price Simulation (SPS-GBM)

A tool for simulating future stock prices using Geometric Brownian Motion (GBM) in Rust.

## Overview

This tool simulates potential future stock prices based on historical data using the Geometric Brownian Motion model, a standard approach in financial modeling.

## Features

- Fetch historical stock data from Alpha Vantage
- Calculate drift and volatility parameters from historical data
- Run multiple simulations to generate price distribution
- Visualize results with price paths and histograms
- Generate summary statistics for predicted prices

## Installation

```bash
git clone https://github.com/excoffierleonard/sps-gbm.git
cd sps-gbm
cargo build --release
```

## Usage

```bash
# Run with interactive prompts
cargo run --release

# Run with a specific ticker
cargo run --release -- -t AAPL
```

Follow the prompts to enter:

- Stock ticker symbol
- Historical date range
- Prediction date
- Number of simulations

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
