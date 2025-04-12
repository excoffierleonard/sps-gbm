use anyhow::{anyhow, Context, Result};
use chrono::{Duration, Local};
use clap::Parser;
use plotters::prelude::*;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sps_gbm::SimulationResults;
use sps_gbm::simulate_gbm;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal as StatNormal;
use std::fs;
use std::path::Path;
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

/// Fetch historical stock data from Alpha Vantage API
///
/// # Parameters
/// - `ticker`: The stock symbol (e.g., "AAPL", "MSFT")
/// - `days`: Number of historical days to fetch data for
///
/// # Returns
/// A vector of StockData structs containing historical price data in chronological order
/// (oldest first, most recent last)
fn fetch_stock_data(ticker: &str, days: i64) -> Result<Vec<StockData>> {
    dotenvy::dotenv().ok();

    // Create cache directory if it doesn't exist
    let cache_dir = Path::new("cache");
    if !cache_dir.exists() {
        fs::create_dir_all(cache_dir)?;
    }

    // Check if we have cached data for this ticker
    let cache_file = cache_dir.join(format!("{}.json", ticker));

    if cache_file.exists() {
        println!("Using cached data for {}", ticker);
        let cached_data = fs::read_to_string(&cache_file)
            .with_context(|| format!("Failed to read cache file for {}", ticker))?;

        let stock_data: Vec<StockData> = serde_json::from_str(&cached_data)
            .with_context(|| format!("Failed to parse cached data for {}", ticker))?;

        // Return the most recent 'days' of stock data
        return Ok(stock_data
            .into_iter()
            .rev()
            .take(days as usize)
            .rev()
            .collect());
    }

    // If no cached data, fetch from API
    println!("Fetching data for {} from Alpha Vantage...", ticker);

    // Get API key from environment variable (set via .env file)
    let api_key = std::env::var("ALPHAVANTAGE_API_KEY").unwrap();

    let url = format!(
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey={}",
        ticker, api_key
    );

    // Make the API request
    let client = Client::new();
    let response = client
        .get(&url)
        .send()
        .with_context(|| format!("Failed to fetch data for {}", ticker))?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "API request failed with status: {}",
            response.status()
        ));
    }

    let json: Value = response
        .json()
        .with_context(|| "Failed to parse API response as JSON")?;

    // Check for error response
    if json.get("Error Message").is_some() {
        let error_msg = json["Error Message"].as_str().unwrap_or("Unknown error");
        return Err(anyhow!("API error: {}", error_msg));
    }

    if json.get("Information").is_some() && json.get("Time Series (Daily)").is_none() {
        let info = json["Information"].as_str().unwrap_or("API limit reached");
        return Err(anyhow!("API limit issue: {}", info));
    }

    // Extract time series data
    let time_series = json["Time Series (Daily)"]
        .as_object()
        .ok_or_else(|| anyhow!("Failed to extract time series data"))?;

    // Convert to vector of StockData
    let mut stock_data: Vec<StockData> = Vec::new();

    for (date, data) in time_series {
        let close_price = data["4. close"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing close price data"))?
            .parse::<f64>()
            .with_context(|| "Failed to parse close price as float")?;

        stock_data.push(StockData {
            date: date.to_string(),
            close: close_price,
        });
    }

    // Sort by date (oldest first)
    stock_data.sort_by(|a, b| a.date.cmp(&b.date));

    // Cache the data for future use
    let cached_json = serde_json::to_string(&stock_data)
        .with_context(|| "Failed to serialize stock data for caching")?;

    fs::write(&cache_file, cached_json)
        .with_context(|| format!("Failed to write cache file for {}", ticker))?;

    // Return the most recent 'days' of data
    Ok(stock_data
        .into_iter()
        .rev()
        .take(days as usize)
        .rev()
        .collect())
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
