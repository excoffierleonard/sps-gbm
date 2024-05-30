from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy import stats


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
    default_prediction_days = 5 * 252
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


# Function to plot the simulation
def plot_simulation(ticker, future_dates, simulations):
    plt.figure(figsize=(10, 6))
    for future_prices in simulations:
        plt.plot(future_dates, future_prices, alpha=0.3)
    plt.title(
        f"Stock Price Simulations for {ticker} using GBM ({len(simulations)} Simulations)"
    )
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.show()


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


# Function to display summary statistics
def display_summary(
    prediction_days,
    final_prices,
    mean_final_price,
    median_final_price,
    std_final_price,
    target_price=None,
):
    from scipy import stats

    # Confidence Interval Calculation
    confidence_level = 0.95
    confidence_interval = stats.norm.interval(
        confidence_level, loc=mean_final_price, scale=std_final_price
    )

    # Percentiles Calculation
    percentiles = np.percentile(final_prices, [10, 25, 75, 90])

    # Probability of Profit Calculation
    if target_price:
        probability_of_profit = np.mean(np.array(final_prices) > target_price)

    # Print the summary
    print(f"\nSummary of Predicted Stock Prices after {prediction_days} days:")
    print(f"Mean Final Price: {mean_final_price:.2f}")
    print(f"Median Final Price: {median_final_price:.2f}")
    print(f"Standard Deviation of Final Prices: {std_final_price:.2f}")
    print(
        f"{confidence_level*100}% Confidence Interval: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})"
    )
    print(f"10th Percentile: {percentiles[0]:.2f}")
    print(f"25th Percentile: {percentiles[1]:.2f}")
    print(f"75th Percentile: {percentiles[2]:.2f}")
    print(f"90th Percentile: {percentiles[3]:.2f}")
    if target_price:
        print(
            f"Probability of Exceeding Target Price ({target_price}): {probability_of_profit:.2%}"
        )

    # Plot a histogram of final prices
    plt.figure(figsize=(10, 6))
    plt.hist(final_prices, bins=50, alpha=0.75, edgecolor="k")
    plt.title(
        f"Distribution of Final Predicted Stock Prices after {prediction_days} days"
    )
    plt.xlabel("Stock Price")
    plt.ylabel("Frequency")
    plt.axvline(
        mean_final_price,
        color="r",
        linestyle="dashed",
        linewidth=1,
        label="Mean Final Price",
    )
    plt.axvline(
        median_final_price,
        color="g",
        linestyle="dashed",
        linewidth=1,
        label="Median Final Price",
    )
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function
def main():
    # Get user inputs
    ticker, start_date, end_date = get_inputs()

    # Calculate parameters
    mu, sigma, S0 = calculate_parameters(ticker, start_date, end_date)

    # Display the calculated parameters
    print(f"Annualized Mean Return (mu): {mu:.4f}")
    print(f"Annualized Volatility (sigma): {sigma:.4f}")
    print(f"Most Recent Closing Price: {S0:.2f}")

    # Get prediction parameters
    prediction_days, num_simulations = get_prediction_parameters()

    # Set the time period and number of steps
    T = prediction_days / 252  # Time period in years
    N = prediction_days  # Number of steps (days to project)

    # Perform multiple simulations
    simulations = []
    for _ in range(num_simulations):
        future_prices = simulate_stock_price(S0, mu, sigma, T, N)
        simulations.append(future_prices)

    # Extract the final prices from each simulation
    final_prices = [simulation[-1] for simulation in simulations]

    # Calculate the mean, median, and standard deviation of the final prices
    mean_final_price = np.mean(final_prices)
    median_final_price = np.median(final_prices)
    std_final_price = np.std(final_prices)

    # Display the summary statistics
    display_summary(
        prediction_days,
        final_prices,
        mean_final_price,
        median_final_price,
        std_final_price,
    )

    # Generate dates for the x-axis
    future_dates = [datetime.today().date() + timedelta(days=i) for i in range(N + 1)]

    # Plot the stock price paths for all simulations
    plot_simulation(ticker, future_dates, simulations)


if __name__ == "__main__":
    main()
