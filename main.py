from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy import stats


# TODO: Consider alligning prediction days with trading days, rather than just displaying it as steps.
# TODO: Enhance summary to upside %.
# TODO: Create complete Stock analysis report and export it in a pretty pdf.
# TODO: Implement rollover of monte-carlo over past steps to have a rolling prediction and improve model.
# Function to fetch historical data and calculate mu and sigma
def calculate_parameters(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data["Return"] = data["Adj Close"].pct_change()
    returns = data["Return"].dropna()
    mu_daily = returns.mean()
    sigma_daily = returns.std()
    trading_days = 252
    mu_annual = (1 + mu_daily) ** trading_days - 1
    sigma_annual = sigma_daily * np.sqrt(trading_days)
    return mu_annual, sigma_annual, data["Adj Close"].iloc[-1]


# Function to simulate future stock prices
def simulate_stock_price(S0, mu, sigma, T, N):
    dt = T / N
    Z = np.random.standard_normal(N)
    S = np.zeros(N + 1)
    S[0] = S0
    for t in range(1, N + 1):
        S[t] = S[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t - 1]
        )
    return S


# Function to get and default user inputs
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
    return ticker, start_date, end_date


# Function to get prediction parameters
def get_prediction_parameters():
    default_prediction_days = 1 * 252
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
    return prediction_days, num_simulations


# Function to perform simulations and display results
def perform_simulations(S0, mu, sigma, T, N, num_simulations):
    simulations = [
        simulate_stock_price(S0, mu, sigma, T, N) for _ in range(num_simulations)
    ]
    final_prices = [simulation[-1] for simulation in simulations]
    mean_final_price = np.mean(final_prices)
    median_final_price = np.median(final_prices)
    std_final_price = np.std(final_prices)
    return simulations, mean_final_price, median_final_price, std_final_price


# Function to calculate summary statistics
def calculate_summary_stats(
    final_prices,
    mean_final_price,
    std_final_price,
    target_price=None,
    confidence_level=0.95,
    percentiles=(10, 25, 75, 90),
):
    confidence_interval = stats.norm.interval(
        confidence_level, loc=mean_final_price, scale=std_final_price
    )
    calculated_percentiles = np.percentile(final_prices, percentiles)
    probability_of_profit = None
    if target_price:
        probability_of_profit = np.mean(np.array(final_prices) > target_price)
    return confidence_interval, calculated_percentiles, probability_of_profit


# Function to print summary statistics
def print_summary(
    prediction_days,
    mean_final_price,
    median_final_price,
    std_final_price,
    confidence_interval,
    percentiles,
    probability_of_profit=None,
    target_price=None,
):
    print(f"\nSummary of Predicted Stock Prices after {prediction_days} days:")
    print(f"Mean Final Price: {mean_final_price:.2f}")
    print(f"Median Final Price: {median_final_price:.2f}")
    print(f"Standard Deviation of Final Prices: {std_final_price:.2f}")
    print(
        f"95% Confidence Interval: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})"
    )
    print(f"10th Percentile: {percentiles[0]:.2f}")
    print(f"25th Percentile: {percentiles[1]:.2f}")
    print(f"75th Percentile: {percentiles[2]:.2f}")
    print(f"90th Percentile: {percentiles[3]:.2f}")
    if target_price:
        print(
            f"Probability of Exceeding Target Price ({target_price}): {probability_of_profit:.2%}"
        )


# Function to plot histogram of final prices
def plot_histogram(
    final_prices, mean_final_price, median_final_price, prediction_days, ax_hist
):
    ax_hist.hist(final_prices, bins=50, alpha=0.75, edgecolor="k")
    ax_hist.set_title(
        f"Distribution of Final Predicted Stock Prices after {prediction_days} days"
    )
    ax_hist.set_xlabel("Stock Price")
    ax_hist.set_ylabel("Frequency")
    ax_hist.axvline(
        mean_final_price,
        color="r",
        linestyle="dashed",
        linewidth=1,
        label="Mean Final Price",
    )
    ax_hist.axvline(
        median_final_price,
        color="g",
        linestyle="dashed",
        linewidth=1,
        label="Median Final Price",
    )
    ax_hist.legend()
    ax_hist.grid(True)


# Refactored display summary function
def display_summary(
    prediction_days,
    final_prices,
    mean_final_price,
    median_final_price,
    std_final_price,
    ax_hist=None,
    target_price=None,
):
    confidence_interval, percentiles, probability_of_profit = calculate_summary_stats(
        final_prices, mean_final_price, std_final_price, target_price
    )

    print_summary(
        prediction_days,
        mean_final_price,
        median_final_price,
        std_final_price,
        confidence_interval,
        percentiles,
        probability_of_profit,
        target_price,
    )

    if ax_hist:
        plot_histogram(
            final_prices, mean_final_price, median_final_price, prediction_days, ax_hist
        )


# TODO: Fix Date not displaying on x axis.
# Function to plot the simulation
def plot_simulation(ticker, future_dates, simulations, ax_sim):
    for future_prices in simulations:
        ax_sim.plot(future_dates, future_prices, alpha=0.3)
    ax_sim.set_title(
        f"Stock Price Simulations for {ticker} using GBM ({len(simulations)} Simulations)"
    )
    ax_sim.set_xlabel("Date")
    ax_sim.set_ylabel("Stock Price")
    ax_sim.grid(True)
    plt.gcf().autofmt_xdate()


# Function to display calculated parameters
def display_parameters(mu, sigma, S0):
    print(f"Annualized Mean Return (mu): {mu:.4f}")
    print(f"Annualized Volatility (sigma): {sigma:.4f}")
    print(f"Most Recent Closing Price: {S0:.2f}")


# Function to set simulation parameters
def set_simulation_parameters(prediction_days):
    T = prediction_days / 252  # Time period in years
    N = prediction_days  # Number of steps (days to project)
    return T, N


# Function to run multiple simulations
def run_simulations(S0, mu, sigma, T, N, num_simulations):
    simulations = [
        simulate_stock_price(S0, mu, sigma, T, N) for _ in range(num_simulations)
    ]
    final_prices = [simulation[-1] for simulation in simulations]
    return simulations, final_prices


# Function to calculate final statistics
def calculate_final_stats(final_prices):
    mean_final_price = np.mean(final_prices)
    median_final_price = np.median(final_prices)
    std_final_price = np.std(final_prices)
    return mean_final_price, median_final_price, std_final_price


# Function to plot and display results
def plot_and_display_results(
    ticker,
    prediction_days,
    simulations,
    final_prices,
    mean_final_price,
    median_final_price,
    std_final_price,
    N,
):
    fig, (ax_sim, ax_hist) = plt.subplots(2, 1, figsize=(10, 12))
    future_dates = [datetime.today().date() + timedelta(days=i) for i in range(N + 1)]
    plot_simulation(ticker, future_dates, simulations, ax_sim)
    display_summary(
        prediction_days,
        final_prices,
        mean_final_price,
        median_final_price,
        std_final_price,
        ax_hist=ax_hist,
    )
    plt.tight_layout()
    plt.show()


# Refactored main function
def main():
    ticker, start_date, end_date = get_inputs()
    mu, sigma, S0 = calculate_parameters(ticker, start_date, end_date)
    display_parameters(mu, sigma, S0)
    prediction_days, num_simulations = get_prediction_parameters()
    T, N = set_simulation_parameters(prediction_days)
    simulations, final_prices = run_simulations(S0, mu, sigma, T, N, num_simulations)
    mean_final_price, median_final_price, std_final_price = calculate_final_stats(
        final_prices
    )
    plot_and_display_results(
        ticker,
        prediction_days,
        simulations,
        final_prices,
        mean_final_price,
        median_final_price,
        std_final_price,
        N,
    )


if __name__ == "__main__":
    main()
