# Stock Price Simulation using Geometric Brownian Motion (GBM)

<!-- TODO: Update Usage, Example, and Possible Improvements to reflect Monte-Carlo and enhancements made. -->

## Overview

This project simulates the future stock prices of a user-specified ticker using the Geometric Brownian Motion (GBM) model. The GBM model is widely used in quantitative finance to model stock price trajectories due to its ability to capture both the drift (expected return) and volatility (random fluctuations) of stock prices. The simulation utilizes historical stock data to estimate the parameters for drift and volatility and projects future prices based on user inputs.

## Features

- Fetch historical stock data from Yahoo Finance.
- Calculate key metrics: annualized mean return (mu) and volatility (sigma).
- Simulate future stock prices using the Geometric Brownian Motion (GBM) model with customizable parameters.
- Perform multiple simulations to estimate the distribution of future stock prices.
- Calculate and display summary statistics including mean, median, standard deviation, confidence intervals, and percentiles of predicted prices.
- Plot the distribution of simulated final stock prices and their paths over time.
- User-friendly input prompts for stock ticker, historical data dates, and prediction period.
- Interactive user inputs for number of simulations and prediction period.
- Dynamically display calculated parameters and simulation results for robust analysis.
- Detailed visualization of simulation results, including a histogram of final prices and multiple price projection paths.

## Requirements

- Python 3.x
- Matplotlib
- NumPy
- yfinance
- SciPy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/excoffierleonard/sps-gbm.git
   cd sps-gbm
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the script:

   ```bash
   python main.py
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
Enter the stock ticker: MSFT
Enter start date for historical data (YYYY-MM-DD) [default: 2019-05-21]:
Enter end date for historical data (YYYY-MM-DD) [default: 2024-05-19]:
Enter the prediction period in days [default: 1260]:
```

Output:

- Annualized Mean Return (mu): 0.3440
- Annualized Volatility (sigma): 0.3038
- Most Recent Closing Price: 420.21

The plot displays the simulated stock price path over the next 1260 days (5 years).

![Simulated Stock Price Path](example.png)

## How It Works

1. **Input**: The user provides a stock ticker, date range for historical data, and the prediction period.
2. **Fetch Data**: The script fetches historical stock prices using the `yfinance` library.
3. **Calculate Parameters**: The script calculates the daily returns, then computes the annualized mean return (drift, \(\mu\)) and volatility (\(\sigma\)).
4. **Simulate Future Prices**: Using the GBM formula, the script projects future stock prices based on the calculated parameters.
5. **Plot**: The script plots the simulated stock prices with the corresponding dates for easy interpretation.

## Possible Improvements

There are several enhancements we can make to this project to increase its functionality and robustness:

1. **Monte Carlo Simulation**:

   - Run multiple simulations to better understand the variability and range of possible future stock prices.
   - Aggregate and visualize the results to see potential future price distributions and confidence intervals.

2. **Parameter Estimation Methods**:

   - Improve the calculation of drift (\(\mu\)) and volatility (\(\sigma\)) using more advanced statistical methods.
   - Consider using exponential moving averages or GARCH models to capture more dynamic aspects of financial time series.

3. **User Interface**:

   - Create a graphical user interface (GUI) to make the application more user-friendly.
   - Implement interactive plots where users can hover over points to see exact values.

4. **Report Generation**:

   - Automatically generate detailed reports with charts and statistics after running the simulations.
   - Include summary statistics and risk metrics such as Value at Risk (VaR) and Expected Shortfall (ES).

5. **Extend to Other Asset Classes**:

   - Adapt the model to simulate other types of financial assets like commodities, bonds, or cryptocurrencies.

6. **Integration with Data Sources**:

   - Integrate with real-time data APIs to fetch the most recent stock prices and perform live simulations.
   - Use historical fundamental data to enhance the parameter estimation.

7. **Sensitivity Analysis**:

   - Implement sensitivity analysis to understand how changes in input parameters (drift, volatility) affect the simulation outcomes.

8. **Backtesting**:
   - Develop backtesting functionality to compare the simulation results with actual historical outcomes to validate the model.

By adding these features, we can make this project more comprehensive and useful for various financial modeling and forecasting scenarios.

## Contribution

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/excoffierleonard/sps-gbm/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- This project utilizes the `yfinance` library to fetch historical stock data.
- Inspiration from "Stochastic Calculus for Finance 1" for the GBM model.
