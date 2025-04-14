use std::env;

use sps_gbm::generate_simulation;

use clap::Parser;

/// Geometric Brownian Motion Stock Price Simulator
#[derive(Parser)]
#[command(version)]
struct Cli {
    /// The stock ticker symbol (e.g., AAPL, MSFT)
    #[arg(short, long, default_value = "AAPL")]
    ticker: String,

    /// Start date for historical data (YYYY-MM-DD)
    #[arg(short = 'f', long, default_value = "2023-01-01")]
    start_date: String,

    /// End date for historical data (YYYY-MM-DD)
    #[arg(short = 'u', long, default_value = "2024-12-31")]
    end_date: String,

    /// Number of simulation steps
    #[arg(short = 's', long, default_value_t = 100)]
    steps: usize,

    /// Number of simulation paths
    #[arg(short = 'n', long, default_value_t = 100)]
    paths: usize,
}

fn main() {
    // Parse command line arguments
    let args = Cli::parse();

    // Load environment variables from .env file if it exists
    dotenvy::dotenv().ok();

    // Get API key from environment
    let api_key = env::var("ALPHAVANTAGE_API_KEY").expect(
        "ALPHAVANTAGE_API_KEY environment variable not set. Please set it or create a .env file.",
    );

    // Run simulation and generate plot
    let simulation_result = generate_simulation(
        &args.ticker,
        &api_key,
        &args.start_date,
        &args.end_date,
        args.steps,
        args.paths,
    );

    let stats = simulation_result.summary_stats;

    // Print the simulation results
    println!("Simulation Results:");
    println!("Ticker: {}", args.ticker);
    println!("Mean Price: {}", stats.mean);
    println!("Median Price: {}", stats.median);
    println!("Standard Deviation: {}", stats.std_dev);
    println!(
        "Confidence Interval (95%): [{}, {}]",
        stats.confidence_interval_95.lower_bound, stats.confidence_interval_95.upper_bound
    );
    println!(
        "Percentiles 10th: {}, 25th: {}, 75th: {}, 90th: {}",
        stats.percentiles.p10, stats.percentiles.p25, stats.percentiles.p75, stats.percentiles.p90
    );

    println!("{}", simulation_result.plot_path.display());
}
