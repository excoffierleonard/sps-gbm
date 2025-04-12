use anyhow::{anyhow, Result};
use chrono::Datelike;
use chrono::{Duration, Local, NaiveDate};
use clap::Parser;
use csv::ReaderBuilder;
use plotters::prelude::*;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use serde::{Deserialize, Serialize};
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal as StatNormal;
use std::collections::HashSet;
use std::fs::{self};
use std::thread::sleep;
use std::time::Duration as StdDuration;

const DAYS_IN_YEAR: f64 = 365.0;
const HISTORICAL_DATA_PERIOD: i64 = 365;
const DEFAULT_SIMULATIONS_COUNT: usize = 1000;
const CONFIDENCE_LEVEL: f64 = 0.95;
const MAX_RETRIES: usize = 3;
const RETRY_DELAY_MS: u64 = 1000;

/// Stock Price Simulator using Geometric Brownian Motion
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Stock ticker symbol (e.g., AAPL, MSFT)
    #[clap(short, long)]
    ticker: String,

    /// Historical start date in YYYY-MM-DD format
    #[clap(short = 's', long, value_parser = parse_date)]
    start_date: Option<NaiveDate>,

    /// Historical end date in YYYY-MM-DD format
    #[clap(short = 'e', long, value_parser = parse_date)]
    end_date: Option<NaiveDate>,

    /// Prediction date in YYYY-MM-DD format
    #[clap(short = 'p', long, value_parser = parse_date)]
    prediction_date: Option<NaiveDate>,

    /// Number of simulations to run
    #[clap(short = 'n', long, default_value_t = DEFAULT_SIMULATIONS_COUNT)]
    simulations: usize,
    
    /// Display verbose debug information
    #[clap(short, long)]
    verbose: bool,
}

fn parse_date(date_str: &str) -> Result<NaiveDate, String> {
    NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
        .map_err(|e| format!("Invalid date format: {}", e))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StockData {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    adj_close: f64,
    volume: u64,
}

struct SimulationResults {
    simulations: Vec<Vec<f64>>,
    final_prices: Vec<f64>,
    mean_final_price: f64,
    median_final_price: f64,
    std_final_price: f64,
}

// Results container to avoid too_many_arguments warning
struct DisplayData<'a> {
    ticker: &'a str,
    prediction_days: i64,
    prediction_date: NaiveDate,
    mu: f64,
    sigma: f64,
    s0: f64,
    results: &'a SimulationResults,
    stats: &'a SummaryStats,
}

struct SummaryStats {
    confidence_interval: (f64, f64),
    percentiles: Vec<f64>,
    percent_change: f64,
}

fn create_client() -> Result<Client> {
    let mut headers = HeaderMap::new();
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"),
    );
    
    Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))
}

fn fetch_with_retry<F, T>(f: F, verbose: bool) -> Result<T> 
where 
    F: Fn() -> Result<T>
{
    let mut last_error = None;
    
    for attempt in 1..=MAX_RETRIES {
        if verbose && attempt > 1 {
            println!("Retry attempt {} of {}", attempt, MAX_RETRIES);
        }
        
        match f() {
            Ok(result) => return Ok(result),
            Err(e) => {
                if verbose {
                    println!("Attempt {} failed: {}", attempt, e);
                }
                last_error = Some(e);
                
                if attempt < MAX_RETRIES {
                    let backoff = RETRY_DELAY_MS * attempt as u64;
                    if verbose {
                        println!("Waiting {}ms before retry...", backoff);
                    }
                    sleep(StdDuration::from_millis(backoff));
                }
            }
        }
    }
    
    Err(last_error.unwrap_or_else(|| anyhow!("All retry attempts failed")))
}

fn get_market_days(start_date: NaiveDate, end_date: NaiveDate) -> Result<HashSet<NaiveDate>> {
    // Simplified version that considers Monday-Friday as trading days
    // Except major US holidays (simplified)
    let mut trading_days = HashSet::new();
    let mut current_date = start_date;

    while current_date <= end_date {
        let weekday = current_date.weekday();
        if weekday != chrono::Weekday::Sat && weekday != chrono::Weekday::Sun {
            // This is a simplification - a real implementation would check holidays too
            trading_days.insert(current_date);
        }
        current_date = current_date.succ_opt().unwrap();
    }

    Ok(trading_days)
}

fn validate_and_setup_dates(args: &Args) -> Result<(NaiveDate, NaiveDate, i64, NaiveDate)> {
    let today = Local::now().date_naive();
    let default_start_date = today - Duration::days(HISTORICAL_DATA_PERIOD);
    
    // Validate start_date
    let start_date = match args.start_date {
        Some(date) => {
            if date > today {
                return Err(anyhow!("Start date cannot be in the future"));
            }
            date
        },
        None => default_start_date,
    };
    
    // Validate end_date
    let end_date = match args.end_date {
        Some(date) => {
            if date > today {
                return Err(anyhow!("End date cannot be in the future"));
            }
            if date < start_date {
                return Err(anyhow!("End date must be after start date"));
            }
            date
        },
        None => today,
    };
    
    // Calculate default prediction date
    let historical_time_span = end_date.signed_duration_since(start_date).num_days();
    let default_prediction_date = end_date + Duration::days(historical_time_span);
    
    // Validate prediction_date
    let prediction_date = match args.prediction_date {
        Some(date) => {
            if date <= end_date {
                return Err(anyhow!("Prediction date must be after the historical end date"));
            }
            date
        },
        None => default_prediction_date,
    };
    
    // Calculate prediction days
    let prediction_days = prediction_date.signed_duration_since(end_date).num_days();
    
    Ok((start_date, end_date, prediction_days, prediction_date))
}

fn fetch_stock_data(
    ticker: &str,
    start_date: NaiveDate,
    end_date: NaiveDate,
    verbose: bool,
) -> Result<Vec<StockData>> {
    if verbose {
        println!("Attempting to fetch data for {} from {} to {}", ticker, start_date, end_date);
    }
    
    let period1 = start_date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
    let period2 = end_date.and_hms_opt(23, 59, 59).unwrap().and_utc().timestamp();

    let url = format!(
        "https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history",
        ticker, period1, period2
    );

    let client = create_client()?;
    
    let response = fetch_with_retry(|| {
        if verbose {
            println!("Sending request to: {}", url);
        }
        
        client.get(&url)
            .send()
            .map_err(|e| anyhow!("Request failed: {}", e))
            .and_then(|resp| {
                if verbose {
                    println!("Response status: {}", resp.status());
                }
                
                if !resp.status().is_success() {
                    return Err(anyhow!("Failed to download data for {}: {}", ticker, resp.status()));
                }
                
                Ok(resp)
            })
    }, verbose)?;
    
    let csv_data = response.text()?;
    
    if verbose {
        println!("CSV data length: {} bytes", csv_data.len());
    }
    
    // Check if we got actual CSV data with a header row
    if !csv_data.contains("Date,Open,High,Low,Close,Adj Close,Volume") {
        return Err(anyhow!("Invalid data format received from Yahoo Finance API"));
    }
    
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(csv_data.as_bytes());

    let mut stock_data = Vec::new();
    for result in reader.records() {
        let record = result?;

        // Skip records with null/na values
        if record.iter().any(|field| field == "null" || field == "na") {
            continue;
        }

        // Parse the record
        let date = record
            .get(0)
            .ok_or_else(|| anyhow!("Missing date"))?
            .to_string();
        let open = record
            .get(1)
            .ok_or_else(|| anyhow!("Missing open"))?
            .parse::<f64>()?;
        let high = record
            .get(2)
            .ok_or_else(|| anyhow!("Missing high"))?
            .parse::<f64>()?;
        let low = record
            .get(3)
            .ok_or_else(|| anyhow!("Missing low"))?
            .parse::<f64>()?;
        let close = record
            .get(4)
            .ok_or_else(|| anyhow!("Missing close"))?
            .parse::<f64>()?;
        let adj_close = record
            .get(5)
            .ok_or_else(|| anyhow!("Missing adj_close"))?
            .parse::<f64>()?;
        let volume = record
            .get(6)
            .ok_or_else(|| anyhow!("Missing volume"))?
            .parse::<u64>()?;

        stock_data.push(StockData {
            date,
            open,
            high,
            low,
            close,
            adj_close,
            volume,
        });
    }

    // Sort by date in ascending order
    stock_data.sort_by(|a, b| a.date.cmp(&b.date));
    
    if verbose {
        println!("Fetched {} data points", stock_data.len());
    }
    
    if stock_data.is_empty() {
        return Err(anyhow!("No stock data found for {} in the specified date range", ticker));
    }

    Ok(stock_data)
}

fn calculate_parameters(
    stock_data: &[StockData],
    prediction_days: i64,
) -> Result<(f64, f64, f64, f64, usize, Vec<NaiveDate>)> {
    if stock_data.is_empty() {
        return Err(anyhow!("No stock data available"));
    }

    // Calculate returns
    let mut returns = Vec::new();
    for i in 1..stock_data.len() {
        let previous = stock_data[i - 1].adj_close;
        let current = stock_data[i].adj_close;
        returns.push((current / previous) - 1.0);
    }

    // Calculate parameters
    let mu_daily = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance_daily: f64 =
        returns.iter().map(|&r| (r - mu_daily).powi(2)).sum::<f64>() / returns.len() as f64;
    let sigma_daily = variance_daily.sqrt();

    // Annualize parameters
    let actual_days = stock_data.len() as f64;
    let trading_days_per_year = actual_days / DAYS_IN_YEAR * DAYS_IN_YEAR;
    let mu_annual = ((1.0 + mu_daily).powf(trading_days_per_year)) - 1.0;
    let sigma_annual = sigma_daily * (trading_days_per_year.sqrt());

    // Get the most recent closing price
    let s0 = stock_data.last().unwrap().adj_close;

    // Calculate trading days in prediction period
    let start_date = NaiveDate::parse_from_str(&stock_data.last().unwrap().date, "%Y-%m-%d")?;
    let end_date = start_date + Duration::days(prediction_days);
    let trading_dates = get_trading_dates(start_date, end_date)?;
    let trading_days_count = trading_dates.len();

    // Calculate time parameters for GBM
    let t = trading_days_count as f64 / trading_days_per_year;

    Ok((
        mu_annual,
        sigma_annual,
        s0,
        t,
        trading_days_count,
        trading_dates,
    ))
}

fn get_trading_dates(start_date: NaiveDate, end_date: NaiveDate) -> Result<Vec<NaiveDate>> {
    let market_days = get_market_days(start_date, end_date)?;
    let mut dates: Vec<NaiveDate> = market_days.into_iter().collect();
    dates.sort();
    Ok(dates)
}

fn simulate_gbm(
    s0: f64,
    mu: f64,
    sigma: f64,
    t: f64,
    n: usize,
    num_simulations: usize,
) -> SimulationResults {
    let dt = t / n as f64;
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    let mut simulations = vec![vec![s0; n + 1]; num_simulations];
    let mut final_prices = Vec::with_capacity(num_simulations);

    for simulation in simulations.iter_mut().take(num_simulations) {
        for j in 1..=n {
            let z = normal.sample(&mut rng);
            let return_val = (drift + vol * z).exp();
            simulation[j] = simulation[j - 1] * return_val;
        }
        final_prices.push(simulation[n]);
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
        final_prices,
        mean_final_price,
        median_final_price,
        std_final_price,
    }
}

fn calculate_summary_stats(s0: f64, results: &SimulationResults) -> SummaryStats {
    let normal = StatNormal::new(results.mean_final_price, results.std_final_price).unwrap();
    let alpha = 1.0 - CONFIDENCE_LEVEL;
    let critical_value = normal.inverse_cdf(1.0 - alpha / 2.0);
    let confidence_interval = (
        results.mean_final_price - critical_value * results.std_final_price,
        results.mean_final_price + critical_value * results.std_final_price,
    );

    let percentiles = vec![
        results.final_prices[results.final_prices.len() / 10],
        results.final_prices[results.final_prices.len() / 4],
        results.final_prices[results.final_prices.len() * 3 / 4],
        results.final_prices[results.final_prices.len() * 9 / 10],
    ];

    let percent_change = (results.mean_final_price - s0) / s0 * 100.0;

    SummaryStats {
        confidence_interval,
        percentiles,
        percent_change,
    }
}

fn display_results(data: &DisplayData) {
    println!("\nStock Ticker: {}", data.ticker);
    println!("Annualized Mean Return (µ): {:.4}", data.mu);
    println!("Annualized Volatility (σ): {:.4}", data.sigma);
    println!("Most Recent Closing Price: {:.2}", data.s0);
    println!(
        "\nSummary of Predicted Stock Prices after {} days (on {}):",
        data.prediction_days, data.prediction_date
    );
    println!(
        "Mean Final Price: {:.2} ({:.2}%)",
        data.results.mean_final_price, data.stats.percent_change
    );
    println!("Median Final Price: {:.2}", data.results.median_final_price);
    println!(
        "Standard Deviation of Final Prices: {:.2}",
        data.results.std_final_price
    );
    println!(
        "95% Confidence Interval: ({:.2}, {:.2})",
        data.stats.confidence_interval.0, data.stats.confidence_interval.1
    );
    println!(
        "Percentiles: 10th {:.2}, 25th {:.2}, 75th {:.2}, 90th {:.2}",
        data.stats.percentiles[0], data.stats.percentiles[1], data.stats.percentiles[2], data.stats.percentiles[3]
    );
}

fn plot_results(
    ticker: &str,
    stock_data: &[StockData],
    results: &SimulationResults,
    trading_dates: &[NaiveDate],
) -> Result<()> {
    // Create output directory if it doesn't exist
    fs::create_dir_all("output")?;

    // Create a new drawing area
    let root = BitMapBackend::new("output/stock_simulation.png", (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Split the drawing area
    let (upper, lower) = root.split_vertically(450);

    // Prepare data for plotting
    let mut historical_dates = Vec::new();
    let mut historical_prices = Vec::new();

    for data in stock_data {
        let date = NaiveDate::parse_from_str(&data.date, "%Y-%m-%d")?;
        historical_dates.push(date);
        historical_prices.push(data.adj_close);
    }

    // Find min and max values for y-axis
    let min_price = historical_prices
        .iter()
        .chain(results.simulations.iter().flatten())
        .fold(f64::INFINITY, |a, &b| a.min(b));

    let max_price = results
        .simulations
        .iter()
        .flatten()
        .fold(0.0f64, |a, &b| a.max(b));

    // Set up the chart for simulations
    let mut chart = ChartBuilder::on(&upper)
        .caption(
            format!("Stock Price Simulations for {}", ticker),
            ("sans-serif", 24),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            historical_dates[0]..*trading_dates.last().unwrap(),
            min_price * 0.9..max_price * 1.1,
        )?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| x.format("%Y-%m-%d").to_string())
        .y_label_formatter(&|y| format!("${:.2}", y))
        .draw()?;

    // Plot historical data
    chart
        .draw_series(LineSeries::new(
            historical_dates
                .iter()
                .zip(historical_prices.iter())
                .map(|(x, y)| (*x, *y)),
            BLUE.mix(0.5),
        ))?
        .label("Historical Prices")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.mix(0.5)));

    // Plot simulations (limited to 100 to avoid cluttering)
    let n_simulations_to_plot = results.simulations.len().min(100);
    for i in 0..n_simulations_to_plot {
        let simulation_dates = trading_dates
            .iter()
            .take(results.simulations[i].len())
            .cloned()
            .collect::<Vec<_>>();

        chart.draw_series(LineSeries::new(
            simulation_dates
                .iter()
                .zip(results.simulations[i].iter())
                .map(|(x, y)| (*x, *y)),
            RGBColor(100, 100, 100).mix(0.3),
        ))?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // Set up the histogram for final prices
    let mut chart = ChartBuilder::on(&lower)
        .caption(
            "Distribution of Final Predicted Stock Prices",
            ("sans-serif", 24),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0..results.final_prices.len() as u32,
            min_price * 0.9..max_price * 1.1,
        )?;

    chart
        .configure_mesh()
        .x_desc("Frequency")
        .y_labels(10)
        .y_label_formatter(&|y| format!("${:.2}", y))
        .disable_x_mesh()
        .draw()?;

    // Group final prices into bins for histogram
    let bin_count = 50;
    let bin_width = (max_price - min_price) / bin_count as f64;
    let mut bins = vec![0; bin_count];

    for &price in &results.final_prices {
        let bin_idx = ((price - min_price) / bin_width) as usize;
        match bin_idx.cmp(&bin_count) {
            std::cmp::Ordering::Less => bins[bin_idx] += 1,
            std::cmp::Ordering::Equal => bins[bin_count - 1] += 1, // Edge case for maximum value
            std::cmp::Ordering::Greater => {/* Out of range, ignore */},
        }
    }

    // Draw histogram
    chart.draw_series(bins.iter().enumerate().map(|(i, &count)| {
        let x0 = i as u32;
        let y0 = min_price + (i as f64) * bin_width;
        Rectangle::new(
            [(x0, y0), (x0 + count as u32, y0 + bin_width)],
            BLUE.mix(0.5).filled(),
        )
    }))?;

    // Draw mean line
    let mean_color = if results.mean_final_price >= stock_data.last().unwrap().adj_close {
        GREEN
    } else {
        RED
    };

    chart
        .draw_series(LineSeries::new(
            vec![
                (0, results.mean_final_price),
                (results.final_prices.len() as u32, results.mean_final_price),
            ],
            mean_color.mix(0.8),
        ))?
        .label("Mean Final Price")
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], mean_color.mix(0.8)));

    // Draw median line
    chart
        .draw_series(LineSeries::new(
            vec![
                (0, results.median_final_price),
                (
                    results.final_prices.len() as u32,
                    results.median_final_price,
                ),
            ],
            BLUE.mix(0.8),
        ))?
        .label("Median Final Price")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.mix(0.8)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    println!("\nPlot saved to output/stock_simulation.png");

    Ok(())
}

fn main() -> Result<()> {
    println!("Stock Price Simulator using Geometric Brownian Motion");
    println!("====================================================");

    let args = Args::parse();
    let verbose = args.verbose;
    
    // Create ticker to uppercase
    let ticker = args.ticker.to_uppercase();
    println!("Stock ticker: {}", ticker);
    
    // Validate dates
    match validate_and_setup_dates(&args) {
        Ok((start_date, end_date, prediction_days, prediction_date)) => {
            println!("Fetching historical data for {} from {} to {}...", ticker, start_date, end_date);

            // Fetch real stock data from Yahoo Finance
            let stock_data = fetch_stock_data(&ticker, start_date, end_date, verbose)?;

            // Continue with the simulation
            println!("Calculating parameters...");
            let (mu, sigma, s0, t, n, trading_dates) =
                calculate_parameters(&stock_data, prediction_days)?;

            println!("Running {} simulations...", args.simulations);
            let results = simulate_gbm(s0, mu, sigma, t, n, args.simulations);

            let stats = calculate_summary_stats(s0, &results);

            let display_data = DisplayData {
                ticker: &ticker,
                prediction_days,
                prediction_date,
                mu,
                sigma,
                s0,
                results: &results,
                stats: &stats,
            };
            display_results(&display_data);

            println!("\nGenerating plot...");
            plot_results(&ticker, &stock_data, &results, &trading_dates)?;

            println!("Simulation completed successfully!");
            Ok(())
        }
        Err(e) => {
            Err(anyhow!("Date error: {}", e))
        }
    }
}