# Stock Price Simulation using Geometric Brownian Motion (GBM)

<!-- TODO: Rewrite whole README.md when program major fix done. -->

## Overview

This project simulates the future stock prices of a user-specified ticker using the Geometric Brownian Motion (GBM) model. The GBM model is widely used in quantitative finance to model stock price trajectories due to its ability to capture both the drift (expected return) and volatility (random fluctuations) of stock prices. The simulation utilizes historical stock data to estimate the parameters for drift and volatility and projects future prices based on user inputs.

## Features

- **Fetch Historical Data**: Retrieve historical stock data from Yahoo Finance using the `yfinance` library.
- **Calculate Key Metrics**: Compute annualized mean return (µ) and volatility (σ) from historical data.
- **Simulate Future Stock Prices**: Use the Geometric Brownian Motion (GBM) model to simulate future stock price trajectories.
- **Multiple Simulations**: Perform multiple simulations to generate a distribution of future stock prices.
- **Summary Statistics**: Compute and display summary statistics including mean, median, standard deviation, confidence intervals, and percentiles of the predicted prices.
- **Visualizations**: Generate detailed visualizations:
  - Plot multiple simulated future stock price paths.
  - Display a histogram of the final predicted stock prices.
  - Annotate plots with statistics like mean and median prices.
- **User-Friendly Inputs**: Interactive prompts for user inputs including stock ticker, historical data range, prediction period, and number of simulations.
- **Dynamic Displays**: Dynamically display calculated parameters and results for robust analysis.

## Requirements

- Python 3
- Matplotlib
- Pandas
- Pandas Market Calendars
- Yfinance
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
   - **Start Date**: Enter the start date for historical data in `YYYY-MM-DD` format (default is 1 years ago).
   - **End Date**: Enter the end date for historical data in `YYYY-MM-DD` format (default is the most recent close date).
   - **Prediction Period**: Enter the number of days to project the stock price (default is 1 year).
   - **Number of Simulations**: Enter the number of simulations to perform (default is 1000).

3. View the results:
   - The script will display the calculated annualized mean return (µ) and annualized volatility (σ).
   - It will plot multiple simulated stock price trajectories over the prediction period.
   - A histogram will visualize the distribution of the final predicted stock prices.
   - Summary statistics including mean, median, standard deviation, confidence intervals, and percentiles of the predicted prices will be displayed.

## Example

Input:

```
Enter the stock ticker: AAPL
Enter start date for historical data (YYYY-MM-DD) [default: 2019-01-01]:
Enter end date for historical data (YYYY-MM-DD) [default: 2024-01-01]:
Enter the prediction period in days [default: 252]:
Enter the number of simulations to perform [default: 1000]:
```

Output:

```
Stock Ticker: AAPL
Annualized Mean Return (µ): 0.4590
Annualized Volatility (σ): 0.3223
Most Recent Closing Price: 192.02

Summary of Predicted Stock Prices after 252 days:
Mean Final Price: 308.67 (+60.75%)
Median Final Price: 293.68
Standard Deviation of Final Prices: 105.85
95% Confidence Interval: (101.21, 516.13)
10th Percentile: 185.49
25th Percentile: 230.15
75th Percentile: 367.25
90th Percentile: 450.23
```

![Simulated Stock Price Paths & Summary](example.png)

## How It Works

1. **User Inputs**: The user provides a stock ticker, the date range for historical data, prediction period, and the number of simulations through interactive prompts.
2. **Fetch Historical Data**: The script retrieves historical stock prices for the specified date range using the `yfinance` library.
3. **Calculate Parameters**: Daily stock returns are computed to determine the annualized mean return (µ) and volatility (σ).
4. **Set Simulation Parameters**: The time period in years (T) and the number of steps (N) are calculated based on the prediction period in days.
5. **Run Simulations**: The Geometric Brownian Motion (GBM) model is used to simulate future stock prices over the prediction period. This involves generating multiple simulation paths.
6. **Perform Multiple Simulations**: The specified number of simulations is run to obtain a distribution of final predicted stock prices.
7. **Calculate Summary Statistics**: Key statistics are computed, including the mean, median, standard deviation, confidence intervals, and percentiles of the final predicted prices.
8. **Display Results**: The script displays the calculated parameters and summary statistics.
9. **Plot Results**: The script generates and displays plots:
   - **Simulation Paths**: Visualize multiple simulated stock price trajectories over the prediction period.
   - **Histogram of Final Prices**: Show the distribution of the final predicted stock prices.

## Possible Improvements

1. **Parameter Estimation Methods**:

   - Improve the calculation of drift (µ) and volatility (σ) using more advanced statistical methods.
   - Consider using exponential moving averages or GARCH models to capture more dynamic aspects of financial time series.

2. **User Interface**:

   - Create a graphical user interface (GUI) to make the application more user-friendly.
   - Implement interactive plots where users can hover over points to see exact values.

3. **Complete Report Generation**:

   - Include risk metrics such as Value at Risk (VaR) and Expected Shortfall (ES) in generated reports.
   - Create complete Stock analysis report and export it in a pretty pdf.\*\*

4. **Extend to Other Asset Classes**:

   - Adapt the model to simulate other types of financial assets like commodities, bonds, or cryptocurrencies.

5. **Integration with Data Sources**:

   - Integrate with real-time data APIs to fetch the most recent stock prices and perform live simulations.
   - Use historical fundamental data to enhance the parameter estimation.

6. **Sensitivity Analysis**:

   - Implement sensitivity analysis to understand how changes in input parameters (drift, volatility) affect the simulation outcomes.

7. **Backtesting**:
   - Develop backtesting functionality to compare the simulation results with actual historical outcomes to validate the model.
   - Implement rollover of monte-carlo over past steps to have a rolling prediction and improve model.\*\*

## Contribution

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/excoffierleonard/sps-gbm/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- This project utilizes the `yfinance` library to fetch historical stock data.
- Inspiration from "Stochastic Calculus for Finance II" for the GBM model.
