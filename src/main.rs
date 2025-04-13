use std::env;

use sps_gbm::simulate_and_plot;

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
    let output_path = simulate_and_plot(
        &args.ticker,
        &api_key,
        &args.start_date,
        &args.end_date,
        args.steps,
        args.paths,
    );

    println!("{}", output_path.display());
}
