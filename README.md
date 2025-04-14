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

```bash
Simulation Results:
Ticker: AAPL
Mean Price: 233.61
Median Price: 232.33
Standard Deviation: 23.35
Confidence Interval (95%): [233.15, 234.07]
Percentiles 10th: 204.17, 25th: 217.19, 75th: 248.73, 90th: 264.01
/var/folders/bc/8p9v9s1575b7f_bm71jvxps00000gn/T/.tmp3S3gUa.png
```

![Simulated Stock Price Paths & Summary](example.png)

## License

MIT License - See [LICENSE](LICENSE) file for details.
