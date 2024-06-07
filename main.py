from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from matplotlib.gridspec import GridSpec
from scipy import stats


# Function to get the market calendar based on the stock ticker
def get_market_calendar(ticker):
    stock = yf.Ticker(ticker)
    exchange = stock.info.get("exchange")

    if not exchange:
        print("Exchange not found, defaulting to NYSE")
        exchange = "NYSE"  # Default to NYSE if not found

    # Map yfinance exchange info to pandas_market_calendars
    if exchange in ["NMS", "NASDAQ"]:
        return mcal.get_calendar("NASDAQ")
    elif exchange in ["NYQ", "NYSE"]:
        return mcal.get_calendar("NYSE")
    elif exchange in ["AMS", "AEX"]:
        return mcal.get_calendar("Euronext")
    elif exchange in ["LSE", "LON"]:
        return mcal.get_calendar("LSE")
    else:
        print("Exchange calendar not found, defaulting to NYSE")
        return mcal.get_calendar("NYSE")  # Default fallback


# Function to get user inputs
def get_inputs():
    ticker = input("Enter the stock ticker: ").strip()
    default_end_date = datetime.today().date()
    default_start_date = default_end_date - timedelta(days=5 * 365)
    start_date = input(
        f"Enter start date for historical data (YYYY-MM-DD) [default: {default_start_date}]: "
    ).strip()
    end_date = input(
        f"Enter end date for historical data (YYYY-MM-DD) [default: {default_end_date}]: "
    ).strip()
    if not start_date:
        start_date = default_start_date
    if not end_date:
        end_date = default_end_date

    default_prediction_days = 365  # Adjusted default prediction period
    prediction_days_input = input(
        f"Enter the prediction period in days [default: {default_prediction_days}]: "
    ).strip()
    prediction_days = (
        int(prediction_days_input) if prediction_days_input else default_prediction_days
    )

    default_num_simulations = 1000
    num_simulations_input = input(
        f"Enter the number of simulations to perform [default: {default_num_simulations}]: "
    ).strip()
    num_simulations = (
        int(num_simulations_input) if num_simulations_input else default_num_simulations
    )

    return ticker, start_date, end_date, prediction_days, num_simulations


# Function to fetch historical data, calculate mu and sigma, and set simulation parameters
def calculate_parameters_and_setup_simulation(
    ticker, start_date, end_date, prediction_days
):
    data = yf.download(ticker, start=start_date, end=end_date)
    data["Return"] = data["Adj Close"].pct_change()
    returns = data["Return"].dropna()
    mu_daily = returns.mean()
    sigma_daily = returns.std()

    # Calculate the number of trading days in the historical data
    trading_days_per_year = len(returns) / (
        (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    )

    mu_annual = (1 + mu_daily) ** trading_days_per_year - 1
    sigma_annual = sigma_daily * np.sqrt(trading_days_per_year)
    S0 = data["Adj Close"].iloc[-1]

    # Get the appropriate market calendar for the stock ticker
    market_calendar = get_market_calendar(ticker)
    market_schedule = market_calendar.schedule(
        start_date=datetime.today().date(),
        end_date=datetime.today().date() + timedelta(days=prediction_days),
    )
    trading_days_count = len(
        market_schedule
    )  # Actual trading days count for the prediction period
    trading_dates = market_schedule.index.tolist()

    # Add initial date
    last_historical_date = data.index[-1].date()
    trading_dates.insert(0, last_historical_date)

    T = trading_days_count / trading_days_per_year  # Time period in years
    N = trading_days_count  # Number of steps (days to project)

    return mu_annual, sigma_annual, S0, T, N, trading_dates


# Function to run and perform simulations
def simulate_and_perform(S0, mu, sigma, T, N, num_simulations):
    dt = T / N
    simulations = np.zeros((num_simulations, N + 1))
    Z = np.random.standard_normal((num_simulations, N))
    simulations[:, 0] = S0

    for t in range(1, N + 1):
        simulations[:, t] = simulations[:, t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )

    final_prices = simulations[:, -1]
    mean_final_price = np.mean(final_prices)
    median_final_price = np.median(final_prices)
    std_final_price = np.std(final_prices)
    return (
        simulations,
        final_prices,
        mean_final_price,
        median_final_price,
        std_final_price,
    )


# Function to calculate summary statistics
def calculate_summary_stats(
    S0,
    final_prices,
    mean_final_price,
    std_final_price,
    confidence_level=0.95,
    percentiles=(10, 25, 75, 90),
):
    confidence_interval = stats.norm.interval(
        confidence_level, loc=mean_final_price, scale=std_final_price
    )
    calculated_percentiles = np.percentile(final_prices, percentiles)
    percent_change = ((mean_final_price - S0) / S0) * 100
    return confidence_interval, calculated_percentiles, percent_change


# Unified function to display parameters and summary statistics
def display_results(
    ticker,
    prediction_days,
    mu,
    sigma,
    S0,
    mean_final_price,
    median_final_price,
    std_final_price,
    confidence_interval,
    percentiles,
    percent_change,
):
    print(f"\nStock Ticker: {ticker}")
    print(f"Annualized Mean Return (µ): {mu:.4f}")
    print(f"Annualized Volatility (σ): {sigma:.4f}")
    print(f"Most Recent Closing Price: {S0:.2f}")
    print(f"\nSummary of Predicted Stock Prices after {prediction_days} days:")

    # Determine the sign for the percentage change
    sign = "+" if percent_change >= 0 else "-"
    percent_change_abs = abs(percent_change)

    print(f"Mean Final Price: {mean_final_price:.2f} ({sign}{percent_change_abs:.2f}%)")
    print(f"Median Final Price: {median_final_price:.2f}")
    print(f"Standard Deviation of Final Prices: {std_final_price:.2f}")
    print(
        f"95% Confidence Interval: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})"
    )
    print(f"10th Percentile: {percentiles[0]:.2f}")
    print(f"25th Percentile: {percentiles[1]:.2f}")
    print(f"75th Percentile: {percentiles[2]:.2f}")
    print(f"90th Percentile: {percentiles[3]:.2f}")


# Function to plot simulation results
def plot_results(
    ticker,
    prediction_days,
    simulations,
    final_prices,
    mean_final_price,
    median_final_price,
    trading_dates,  # Add trading dates as input
):
    N = simulations.shape[1] - 1  # Number of steps

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0)  # Set wspace to 0

    ax_sim = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharey=ax_sim)

    # Plot simulations
    for future_prices in simulations:
        ax_sim.plot(trading_dates, future_prices, alpha=0.3)
    ax_sim.set_title(
        f"Stock Price Simulations for {ticker} using GBM ({simulations.shape[0]} Simulations)"
    )
    ax_sim.set_xlabel("Date")
    ax_sim.set_ylabel("Stock Price")
    ax_sim.grid(True)

    # Ensure the limits are tight around the data and prediction days
    ax_sim.set_xlim([trading_dates[0], trading_dates[-1]])
    ax_sim.set_ylim([np.min(final_prices), np.max(final_prices)])

    # Plot histogram of final prices
    ax_hist.hist(
        final_prices, bins=50, alpha=0.75, edgecolor="k", orientation="horizontal"
    )
    ax_hist.set_title("Distribution of Final Predicted Stock Prices")
    ax_hist.set_xlabel("Frequency")

    # Color code the mean final price based on increased or decrease
    color_mean = "green" if mean_final_price >= simulations[:, 0][0] else "red"

    ax_hist.axhline(
        mean_final_price,
        color=color_mean,
        linestyle="dashed",
        linewidth=1,  # Thinner line
        label="Mean Final Price",
    )
    ax_hist.axhline(
        median_final_price,
        color="blue",
        linestyle="dashed",
        linewidth=1,  # Thinner line
        label="Median Final Price",
    )
    ax_hist.legend()
    ax_hist.grid(True)

    # Annotate the mean final price on the right of the distribution
    ax_hist.annotate(
        f"{mean_final_price:.2f}",
        xy=(1, mean_final_price),
        xycoords=("axes fraction", "data"),
        xytext=(5, 0),
        textcoords="offset points",
        color=color_mean,
        ha="left",
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor=color_mean, boxstyle="round,pad=0.3"),
    )

    # Hide y-axis ticks labels of the histogram to avoid duplication
    plt.setp(ax_hist.get_yticklabels(), visible=False)  # Hide yticks for hist plot

    # Reduce margins to make the plots appear connected
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10, wspace=0)

    # Set a title for the entire figure
    fig.suptitle(f"Stock Price Simulation and Prediction for {ticker}", fontsize=16)

    plt.show()


# Main function to orchestrate the workflow
def main():
    # Get user inputs
    ticker, start_date, end_date, prediction_days, num_simulations = get_inputs()

    # Calculate parameters and set up simulation
    mu, sigma, S0, T, N, trading_dates = calculate_parameters_and_setup_simulation(
        ticker, start_date, end_date, prediction_days
    )

    # Perform simulations
    simulations, final_prices, mean_final_price, median_final_price, std_final_price = (
        simulate_and_perform(S0, mu, sigma, T, N, num_simulations)
    )

    # Calculate summary statistics
    confidence_interval, percentiles, percent_change = calculate_summary_stats(
        S0, final_prices, mean_final_price, std_final_price
    )

    # Display results
    display_results(
        ticker,
        prediction_days,
        mu,
        sigma,
        S0,
        mean_final_price,
        median_final_price,
        std_final_price,
        confidence_interval,
        percentiles,
        percent_change,
    )

    # Plot results
    plot_results(
        ticker,
        prediction_days,
        simulations,
        final_prices,
        mean_final_price,
        median_final_price,
        trading_dates,  # Pass the trading_dates
    )


if __name__ == "__main__":
    main()
