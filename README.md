# Stock Price Simulation using Geometric Brownian Motion (GBM)

## Overview

This project simulates the future stock prices of a user-specified ticker using the Geometric Brownian Motion (GBM) model. The GBM model is widely used in quantitative finance to model stock price trajectories due to its ability to capture both the drift (expected return) and volatility (random fluctuations) of stock prices. The simulation utilizes historical stock data to estimate the parameters for drift and volatility and projects future prices based on user inputs.

## Features

- Fetch historical stock data from Yahoo Finance.
- Calculate annualized mean return (drift) and volatility.
- Simulate future stock prices using the GBM model.
- Plot the simulated stock prices with dates for clear visualization.
- User-friendly input prompts for stock ticker, historical data dates, and prediction period.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- yfinance

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/stock-price-simulation.git
   cd stock-price-simulation
   ```

2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib yfinance
   ```

## Usage

1. Run the script:

   ```bash
   python stock_simulation.py
   ```

2. Follow the prompts:

   - **Stock Ticker**: Enter the ticker symbol of the stock (e.g., `AAPL` for Apple).
   - **Start Date**: Enter the start date for historical data in `YYYY-MM-DD` format (default is 5 years ago).
   - **End Date**: Enter the end date for historical data in `YYYY-MM-DD` format (default is the most recent close date).
   - **Prediction Period**: Enter the number of days to project the stock price (default is 5 years).

3. View the results:
   - The script will display the calculated annualized mean return (drift) and volatility.
   - It will plot the simulated future stock prices with corresponding dates.

## Example

Running the script with prompt inputs:

```
Enter the stock ticker: AAPL
Enter start date for historical data (YYYY-MM-DD) [default: 2018-09-30]:
Enter end date for historical data (YYYY-MM-DD) [default: 2023-09-30]:
Enter the prediction period in days [default: 1260]:
```

Output:

- Annualized Mean Return (mu): 0.3045
- Annualized Volatility (sigma): 0.2710
- Most Recent Closing Price: 150.30

The plot displays the simulated stock price path over the next 1260 days (5 years).

![Simulated Stock Price Path](plot.png)

## How It Works

1. **Input**: The user provides a stock ticker, date range for historical data, and the prediction period.
2. **Fetch Data**: The script fetches historical stock prices using the `yfinance` library.
3. **Calculate Parameters**: The script calculates the daily returns, then computes the annualized mean return (drift, \(\mu\)) and volatility (\(\sigma\)).
4. **Simulate Future Prices**: Using the GBM formula, the script projects future stock prices based on the calculated parameters.
5. **Plot**: The script plots the simulated stock prices with the corresponding dates for easy interpretation.

## Contribution

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/stock-price-simulation/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- This project utilizes the `yfinance` library to fetch historical stock data.
- Inspiration from "Stochastic Calculus for Finance 1" for the GBM model.
