from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from scipy import stats

DAYS_IN_YEAR = 365
HISTORICAL_DATA_PERIOD = 1 * DAYS_IN_YEAR
DEFAULT_SIMULATIONS_COUNT = 1000


def get_inputs():
    def is_valid_date(date_str):
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def is_positive_integer(value):
        try:
            return int(value) > 0
        except ValueError:
            return False

    def is_existing_ticker(ticker):
        try:
            stock = yf.Ticker(ticker)
            return not stock.history(period="1d").empty
        except Exception:
            return False

    today = datetime.today().date()
    default_start_date = today - timedelta(days=HISTORICAL_DATA_PERIOD)

    while True:
        ticker = input("Enter the stock ticker: ").strip()
        if ticker and is_existing_ticker(ticker):
            break
        print("Error: The stock ticker does not exist. Please enter a valid ticker.")

    while True:
        start_date = input(
            f"Enter historical start date (YYYY-MM-DD) [default: {default_start_date}]: "
        ).strip() or str(default_start_date)
        if (
            is_valid_date(start_date)
            and datetime.strptime(start_date, "%Y-%m-%d").date() <= today
        ):
            break
        print(
            "Error: Invalid historical start date. Please use YYYY-MM-DD format and ensure it is not in the future."
        )

    while True:
        end_date = input(
            f"Enter historical end date (YYYY-MM-DD) [default: {today}]: "
        ).strip() or str(today)
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        if (
            is_valid_date(end_date)
            and end_date_obj <= today
            and end_date_obj >= datetime.strptime(start_date, "%Y-%m-%d").date()
        ):
            break
        print(
            "Error: Invalid historical end date. Please use YYYY-MM-DD format, ensure it is not in the future, and not before the start date."
        )

    historical_time_span = (
        end_date_obj - datetime.strptime(start_date, "%Y-%m-%d").date()
    )
    default_prediction_date = end_date_obj + historical_time_span

    while True:
        prediction_date = input(
            f"Enter the prediction date (YYYY-MM-DD) [default: {default_prediction_date}]: "
        ).strip() or str(default_prediction_date)
        if (
            is_valid_date(prediction_date)
            and datetime.strptime(prediction_date, "%Y-%m-%d").date() > end_date_obj
        ):
            prediction_date_obj = datetime.strptime(prediction_date, "%Y-%m-%d").date()
            prediction_days = (prediction_date_obj - end_date_obj).days
            break
        print(
            "Error: Prediction date must be after the historical end date and use YYYY-MM-DD format."
        )

    while True:
        num_simulations = input(
            f"Enter the number of simulations [default: {DEFAULT_SIMULATIONS_COUNT}]: "
        ).strip() or str(DEFAULT_SIMULATIONS_COUNT)
        if is_positive_integer(num_simulations):
            num_simulations = int(num_simulations)
            break
        print("Error: Number of simulations must be a positive integer.")

    return (
        ticker,
        start_date,
        end_date,
        prediction_days,
        prediction_date,
        num_simulations,
    )


def calculate_parameters_and_setup_simulation(
    ticker, start_date, end_date, prediction_days
):
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

    data = yf.download(ticker, start=start_date, end=end_date)
    returns = data["Adj Close"].pct_change().dropna()
    mu_daily, sigma_daily = returns.mean(), returns.std()
    actual_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    trading_days_per_year = len(returns) / actual_days * DAYS_IN_YEAR
    mu_annual = ((1 + mu_daily) ** trading_days_per_year) - 1
    sigma_annual = sigma_daily * np.sqrt(trading_days_per_year)
    S0 = data["Adj Close"].iloc[-1]

    market_calendar = get_market_calendar(ticker)
    market_schedule = market_calendar.schedule(
        start_date=pd.to_datetime(end_date),
        end_date=pd.to_datetime(end_date) + timedelta(days=prediction_days),
    )
    trading_days_count = len(market_schedule)
    trading_dates = [market_schedule.index[0].date()] + market_schedule.index.tolist()
    T, N = trading_days_count / trading_days_per_year, trading_days_count

    return mu_annual, sigma_annual, S0, T, N, trading_dates, data


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
    prediction_date,
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
    print(
        f"\nSummary of Predicted Stock Prices after {prediction_days} days (on {prediction_date}):"
    )
    print(f"Mean Final Price: {mean_final_price:.2f} ({percent_change:.2f}%)")
    print(f"Median Final Price: {median_final_price:.2f}")
    print(f"Standard Deviation of Final Prices: {std_final_price:.2f}")
    print(
        f"95% Confidence Interval: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})"
    )
    print(
        f"Percentiles: 10th {percentiles[0]:.2f}, 25th {percentiles[1]:.2f}, 75th {percentiles[2]:.2f}, 90th {percentiles[3]:.2f}"
    )


# FIXME: Stop rendering of non-trading days on the plot, the issue is likely on the proportional time delta rendering rather that equidistantly indexing the data points.
def plot_results(
    ticker,
    S0,
    simulations,
    final_prices,
    mean_final_price,
    median_final_price,
    trading_dates,
    historical_data,
):
    fig, (ax_sim, ax_hist) = plt.subplots(
        1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]}
    )

    initial_dates = historical_data.index
    historical_prices = historical_data["Adj Close"].values
    trading_dates = pd.to_datetime(trading_dates)
    whole_dates = initial_dates.append(trading_dates[1:])

    ax_sim.plot(
        initial_dates,
        historical_prices,
        color="blue",
        alpha=0.5,
        label="Historical Prices",
    )
    for future_prices in simulations:
        ax_sim.plot(whole_dates[-len(future_prices) :], future_prices, alpha=0.3)
    ax_sim.set(
        title=f"Stock Price Simulations for {ticker}",
        xlabel="Date",
        ylabel="Stock Price",
        xlim=[whole_dates[0], whole_dates[-1]],
        ylim=[
            np.min(np.concatenate((historical_prices, simulations.flatten()))),
            np.max(simulations),
        ],
    )
    ax_sim.legend(loc="upper left")
    ax_sim.grid(True)
    plt.setp(ax_sim.xaxis.get_majorticklabels(), rotation=45)

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
    ax_hist.set_ylim(ax_sim.get_ylim())
    plt.setp(ax_hist.get_yticklabels(), visible=False)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0)
    fig.suptitle(f"Stock Price Simulation and Prediction for {ticker}", fontsize=16)
    plt.show()


def main():
    try:
        (
            ticker,
            start_date,
            end_date,
            prediction_days,
            prediction_date,
            num_simulations,
        ) = get_inputs()
        mu, sigma, S0, T, N, trading_dates, historical_data = (
            calculate_parameters_and_setup_simulation(
                ticker, start_date, end_date, prediction_days
            )
        )
        (
            simulations,
            final_prices,
            mean_final_price,
            median_final_price,
            std_final_price,
        ) = simulate_and_perform(S0, mu, sigma, T, N, num_simulations)
        confidence_interval, percentiles, percent_change = calculate_summary_stats(
            S0, final_prices, mean_final_price, std_final_price
        )

        display_results(
            ticker,
            prediction_days,
            prediction_date,
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
            historical_data,
        )
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Exiting program.")


if __name__ == "__main__":
    main()
