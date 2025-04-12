use anyhow::{anyhow, Result};
use chrono::{Duration, Local};
use clap::Parser;
use plotters::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal as StatNormal;
use std::fs;
use std::process;

const DAYS_IN_YEAR: f64 = 252.0; // Trading days in a year
const DEFAULT_SIMULATIONS_COUNT: usize = 1000;
const CONFIDENCE_LEVEL: f64 = 0.95;

/// Stock Price Simulator using Geometric Brownian Motion
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Stock ticker symbol (e.g., AAPL, MSFT)
    #[clap(short, long)]
    ticker: String,

    /// Number of days to predict into the future
    #[clap(short, long, default_value_t = 30)]
    days: i64,

    /// Number of simulations to run
    #[clap(short = 'n', long, default_value_t = DEFAULT_SIMULATIONS_COUNT)]
    simulations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StockData {
    date: String,
    close: f64,
}

struct SimulationResults {
    simulations: Vec<Vec<f64>>,
    mean_final_price: f64,
    median_final_price: f64,
    std_final_price: f64,
}

/// Fetch historical stock data from an external financial API
///
/// # Parameters
/// - `ticker`: The stock symbol (e.g., "AAPL", "MSFT")
/// - `days`: Number of historical days to fetch data for
///
/// # Returns
/// A vector of StockData structs containing historical price data in chronological order
/// (oldest first, most recent last)
fn fetch_stock_data(_ticker: &str, _days: i64) -> Result<Vec<StockData>> {
    // This is a placeholder implementation
    // Replace this with your own implementation that fetches data from your preferred source

    Err(anyhow!(
        "You need to implement fetch_stock_data with your preferred stock data source"
    ))
}

fn simulate_gbm(
    s0: f64,
    mu: f64,
    sigma: f64,
    days: usize,
    num_simulations: usize,
) -> SimulationResults {
    let dt = 1.0 / DAYS_IN_YEAR;
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    let mut simulations = vec![vec![s0; days + 1]; num_simulations];
    let mut final_prices = Vec::with_capacity(num_simulations);

    for simulation in simulations.iter_mut().take(num_simulations) {
        for j in 1..=days {
            let z = normal.sample(&mut rng);
            let return_val = (drift + vol * z).exp();
            simulation[j] = simulation[j - 1] * return_val;
        }
        final_prices.push(simulation[days]);
    }

    // Calculate statistics
    final_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_final_price = final_prices.iter().sum::<f64>() / final_prices.len() as f64;
    let median_final_price = if final_prices.len() % 2 == 0 {
        (final_prices[final_prices.len() / 2 - 1] + final_prices[final_prices.len() / 2]) / 2.0
    } else {
        final_prices[final_prices.len() / 2]
    };

    let variance = final_prices
        .iter()
        .map(|&price| (price - mean_final_price).powi(2))
        .sum::<f64>()
        / final_prices.len() as f64;
    let std_final_price = variance.sqrt();

    SimulationResults {
        simulations,
        mean_final_price,
        median_final_price,
        std_final_price,
    }
}

fn calculate_confidence_interval(mean: f64, std_dev: f64) -> (f64, f64) {
    let normal = StatNormal::new(mean, std_dev).unwrap();
    let alpha = 1.0 - CONFIDENCE_LEVEL;
    let critical_value = normal.inverse_cdf(1.0 - alpha / 2.0);

    (
        mean - critical_value * std_dev,
        mean + critical_value * std_dev,
    )
}

fn plot_results(ticker: &str, s0: f64, results: &SimulationResults, days: i64) -> Result<()> {
    // Create output directory if it doesn't exist
    fs::create_dir_all("output")?;

    // Create a new drawing area
    let root = BitMapBackend::new("output/stock_simulation.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Calculate dates for x-axis
    let today = Local::now().date_naive();
    let end_date = today + Duration::days(days);

    // Find min and max values for y-axis
    let min_price = results
        .simulations
        .iter()
        .flatten()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let max_price = results
        .simulations
        .iter()
        .flatten()
        .fold(0.0f64, |a, &b| a.max(b));

    // Set up the chart
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} Stock Price Simulations", ticker),
            ("sans-serif", 24),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(today..end_date, min_price * 0.9..max_price * 1.1)?;

    chart
        .configure_mesh()
        .x_labels(5)
        .y_labels(10)
        .x_label_formatter(&|x| x.format("%Y-%m-%d").to_string())
        .y_label_formatter(&|y| format!("${:.2}", y))
        .draw()?;

    // Plot current price point
    chart.draw_series(PointSeries::of_element(
        vec![(today, s0)],
        5,
        &BLUE,
        &|c, s, st| {
            EmptyElement::at(c)
                + Circle::new((0, 0), s, st.filled())
                + Text::new(format!("${:.2}", s0), (10, 0), ("sans-serif", 15))
        },
    ))?;

    // Plot simulations (limited to 50 to avoid cluttering)
    let n_simulations_to_plot = results.simulations.len().min(50);
    for i in 0..n_simulations_to_plot {
        let dates = (0..=days)
            .map(|d| today + Duration::days(d))
            .collect::<Vec<_>>();

        chart.draw_series(LineSeries::new(
            dates
                .iter()
                .zip(results.simulations[i].iter())
                .map(|(x, y)| (*x, *y)),
            RGBColor(100, 100, 100).mix(0.3),
        ))?;
    }

    // Draw mean line
    let dates = (0..=days)
        .map(|d| today + Duration::days(d))
        .collect::<Vec<_>>();

    let mean_values = (0..=days as usize)
        .map(|d| {
            results.simulations.iter().map(|sim| sim[d]).sum::<f64>()
                / results.simulations.len() as f64
        })
        .collect::<Vec<_>>();

    chart
        .draw_series(LineSeries::new(
            dates.iter().zip(mean_values.iter()).map(|(x, y)| (*x, *y)),
            RED.mix(0.8),
        ))?
        .label("Mean")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.8)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    println!("\nPlot saved to output/stock_simulation.png");

    Ok(())
}

fn calculate_annual_params(stock_data: &[StockData]) -> Result<(f64, f64, f64)> {
    if stock_data.len() < 2 {
        return Err(anyhow!("Not enough stock data for calculation"));
    }

    // Calculate daily returns
    let mut returns = Vec::new();
    for i in 1..stock_data.len() {
        let previous = stock_data[i - 1].close;
        let current = stock_data[i].close;
        returns.push((current / previous) - 1.0);
    }

    // Calculate daily parameters
    let mu_daily = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance_daily =
        returns.iter().map(|&r| (r - mu_daily).powi(2)).sum::<f64>() / returns.len() as f64;
    let sigma_daily = variance_daily.sqrt();

    // Annualize parameters
    let mu_annual = ((1.0 + mu_daily).powf(DAYS_IN_YEAR)) - 1.0;
    let sigma_annual = sigma_daily * DAYS_IN_YEAR.sqrt();

    // Get current stock price
    let s0 = stock_data.last().unwrap().close;

    Ok((mu_annual, sigma_annual, s0))
}

fn main() -> Result<()> {
    println!("Stock Price Simulator using Geometric Brownian Motion");
    println!("====================================================");

    let args = Args::parse();
    let ticker = args.ticker.to_uppercase();
    let days = args.days;

    println!("Stock ticker: {}", ticker);
    println!("Prediction period: {} days", days);

    // Fetch historical stock data
    let stock_data = match fetch_stock_data(&ticker, days) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    };

    println!(
        "Calculating parameters from {} data points...",
        stock_data.len()
    );
    let (mu, sigma, s0) = calculate_annual_params(&stock_data)?;

    println!("Running {} simulations...", args.simulations);
    let results = simulate_gbm(s0, mu, sigma, days as usize, args.simulations);

    // Display results
    println!("\nStarting Price: ${:.2}", s0);
    println!("Annual Return (μ): {:.2}%", mu * 100.0);
    println!("Annual Volatility (σ): {:.2}%", sigma * 100.0);

    println!("\nSimulation Results (after {} days):", days);
    println!("Mean Final Price: ${:.2}", results.mean_final_price);
    println!("Median Final Price: ${:.2}", results.median_final_price);

    let percent_change = (results.mean_final_price - s0) / s0 * 100.0;
    println!("Expected Change: {:.2}%", percent_change);

    let ci = calculate_confidence_interval(results.mean_final_price, results.std_final_price);
    println!("95% Confidence Interval: (${:.2}, ${:.2})", ci.0, ci.1);

    // Generate plot
    plot_results(&ticker, s0, &results, days)?;

    println!("Simulation completed successfully!");
    Ok(())
}
