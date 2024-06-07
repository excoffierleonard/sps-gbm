# TODO: Implement some data validation for the inputs.
# TODO: Gracefully handle exit and other errors in the main function.
# FIXME: Stop rendering of non-trading days on the plot.

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from scipy import stats


def get_market_calendar(ticker):
    stock = yf.Ticker(ticker)
    exchange = stock.info.get("exchange", "NYSE")
    exchange_calendar = {
        "NMS": "NASDAQ",
        "NASDAQ": "NASDAQ",
        "NYQ": "NYSE",
        "NYSE": "NYSE",
        "AMS": "Euronext",
        "AEX": "Euronext",
        "LSE": "LSE",
        "LON": "LSE",
    }
    return mcal.get_calendar(exchange_calendar.get(exchange, "NYSE"))


def get_inputs():
    days_in_year = 365
    default_simulations_count = 1000
    today = datetime.today().date()
    default_start_date = today - timedelta(days=5 * days_in_year)
    ticker = input("Enter the stock ticker: ").strip()
    start_date = input(
        f"Enter start date (YYYY-MM-DD) [default: {default_start_date}]: "
    ).strip() or str(default_start_date)
    end_date = input(
        f"Enter end date (YYYY-MM-DD) [default: {today}]: "
    ).strip() or str(today)
    prediction_days = int(
        input(
            f"Enter the prediction period in days [default: {days_in_year}]: "
        ).strip()
        or days_in_year
    )
    num_simulations = int(
        input(
            f"Enter the number of simulations [default: {default_simulations_count}]: "
        ).strip()
        or default_simulations_count
    )
    return (
        ticker,
        start_date,
        end_date,
        prediction_days,
        num_simulations,
    )


def calculate_parameters_and_setup_simulation(
    ticker, start_date, end_date, prediction_days
):
    data = yf.download(ticker, start=start_date, end=end_date)
    returns = data["Adj Close"].pct_change().dropna()
    mu_daily, sigma_daily = returns.mean(), returns.std()
    actual_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    trading_days_per_year = len(returns) / actual_days * 365
    mu_annual = ((1 + mu_daily) ** trading_days_per_year) - 1
    sigma_annual = sigma_daily * np.sqrt(trading_days_per_year)
    S0 = data["Adj Close"].iloc[-1]

    today = datetime.today().date()
    market_calendar = get_market_calendar(ticker)
    market_schedule = market_calendar.schedule(
        start_date=today, end_date=today + timedelta(days=prediction_days)
    )
    trading_days_count = len(market_schedule)
    trading_dates = [data.index[-1].date()] + market_schedule.index.tolist()
    T, N = trading_days_count / trading_days_per_year, trading_days_count

    return mu_annual, sigma_annual, S0, T, N, trading_dates


def simulate_and_perform(S0, mu, sigma, T, N, num_simulations):
    dt = T / N
    Z = np.random.standard_normal((num_simulations, N))
    paths = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    paths = np.cumprod(np.insert(paths, 0, 1, axis=1), axis=1) * S0
    final_prices = paths[:, -1]
    return (
        paths,
        final_prices,
        final_prices.mean(),
        np.median(final_prices),
        final_prices.std(),
    )


def calculate_summary_stats(
    S0, final_prices, mean_final_price, std_final_price, confidence_level=0.95
):
    confidence_interval = stats.norm.interval(
        confidence_level, loc=mean_final_price, scale=std_final_price
    )
    percentiles = np.percentile(final_prices, [10, 25, 75, 90])
    percent_change = (mean_final_price - S0) / S0 * 100
    return confidence_interval, percentiles, percent_change


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
    print(f"Mean Final Price: {mean_final_price:.2f} ({percent_change:.2f}%)")
    print(f"Median Final Price: {median_final_price:.2f}")
    print(f"Standard Deviation of Final Prices: {std_final_price:.2f}")
    print(
        f"95% Confidence Interval: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})"
    )
    print(
        f"Percentiles: 10th {percentiles[0]:.2f}, 25th {percentiles[1]:.2f}, 75th {percentiles[2]:.2f}, 90th {percentiles[3]:.2f}"
    )


def plot_results(
    ticker,
    S0,
    simulations,
    final_prices,
    mean_final_price,
    median_final_price,
    trading_dates,
):
    fig, (ax_sim, ax_hist) = plt.subplots(
        1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]}
    )

    for future_prices in simulations:
        ax_sim.plot(trading_dates, future_prices, alpha=0.3)
    ax_sim.set(
        title=f"Stock Price Simulations for {ticker}",
        xlabel="Date",
        ylabel="Stock Price",
        xlim=[trading_dates[0], trading_dates[-1]],
        ylim=[np.min(final_prices), np.max(final_prices)],
    )
    ax_sim.grid(True)

    ax_hist.hist(
        final_prices, bins=50, alpha=0.75, edgecolor="k", orientation="horizontal"
    )
    ax_hist.set(
        title="Distribution of Final Predicted Stock Prices", xlabel="Frequency"
    )
    ax_hist.grid(True)
    color_mean = "green" if mean_final_price >= S0 else "red"
    ax_hist.axhline(
        mean_final_price,
        color=color_mean,
        linestyle="dashed",
        linewidth=1,
        label="Mean Final Price",
    )
    ax_hist.axhline(
        median_final_price,
        color="blue",
        linestyle="dashed",
        linewidth=1,
        label="Median Final Price",
    )
    ax_hist.legend()

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
    plt.setp(ax_hist.get_yticklabels(), visible=False)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10, wspace=0)
    fig.suptitle(f"Stock Price Simulation and Prediction for {ticker}", fontsize=16)
    plt.show()


def main():
    ticker, start_date, end_date, prediction_days, num_simulations = get_inputs()
    mu, sigma, S0, T, N, trading_dates = calculate_parameters_and_setup_simulation(
        ticker, start_date, end_date, prediction_days
    )
    simulations, final_prices, mean_final_price, median_final_price, std_final_price = (
        simulate_and_perform(S0, mu, sigma, T, N, num_simulations)
    )
    confidence_interval, percentiles, percent_change = calculate_summary_stats(
        S0, final_prices, mean_final_price, std_final_price
    )
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
    plot_results(
        ticker,
        S0,
        simulations,
        final_prices,
        mean_final_price,
        median_final_price,
        trading_dates,
    )


if __name__ == "__main__":
    main()
