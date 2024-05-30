from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf


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
    prediction_days, mean_final_price, median_final_price, std_final_price
):
    print(f"\nSummary of Predicted Stock Prices after {prediction_days} days:")
    print(f"Mean Final Price: {mean_final_price:.2f}")
    print(f"Median Final Price: {median_final_price:.2f}")
    print(f"Standard Deviation of Final Prices: {std_final_price:.2f}")


# Main function
def main():
    ticker, start_date, end_date = get_inputs()
    mu, sigma, S0 = calculate_parameters(ticker, start_date, end_date)
    print(f"Annualized Mean Return (mu): {mu:.4f}")
    print(f"Annualized Volatility (sigma): {sigma:.4f}")
    print(f"Most Recent Closing Price: {S0:.2f}")
    prediction_days, num_simulations = get_prediction_parameters()
    T = prediction_days / 252
    N = prediction_days
    simulations, mean_final_price, median_final_price, std_final_price = (
        perform_simulations(S0, mu, sigma, T, N, num_simulations)
    )
    display_summary(
        prediction_days, mean_final_price, median_final_price, std_final_price
    )
    future_dates = [datetime.today().date() + timedelta(days=i) for i in range(N + 1)]
    plot_simulation(ticker, future_dates, simulations)


if __name__ == "__main__":
    main()
