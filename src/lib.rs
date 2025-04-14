mod calculations;
mod data;
mod plot;

use std::path::PathBuf;

pub use calculations::{
    SummaryStats, calculate_summary_stats, estimate_gbm_parameters, gbm_step,
    generate_gbm_paths_from_prices, simulate_gbm_path,
};
pub use data::{cache_prices, fetch_historical_prices_alphavantage, get_cached_prices};
pub use plot::plot_results;

use chrono::NaiveDate;

pub struct SimulationResult {
    // The path to the generated plot image
    pub plot_path: PathBuf,
    // The summary statistics of the simulation
    pub summary_stats: SummaryStats,
}

/// End-to-end, high-level function to simulate GBM paths and plot results
///
/// # Arguments
/// * `symbol` - The stock symbol to fetch prices for
/// * `api_key` - Your Alpha Vantage API key
/// * `start_date` - The start date for fetching prices (YYYY-MM-DD)
/// * `end_date` - The end date for fetching prices (YYYY-MM-DD)
/// * `num_steps` - The number of steps to simulate
/// * `num_paths` - The number of paths to simulate
///
/// # Returns
/// The result  of the simulation
pub fn generate_simulation(
    symbol: &str,
    api_key: &str,
    start_date: &str,
    end_date: &str,
    num_steps: usize,
    num_paths: usize,
) -> SimulationResult {
    // Fetch historical prices
    let historical_prices =
        fetch_historical_prices_alphavantage(symbol, api_key, start_date, end_date);

    // Generate simulated paths
    let paths = generate_gbm_paths_from_prices(&historical_prices, num_steps, num_paths);

    // Create future dates for simulation
    let last_historical_date = NaiveDate::parse_from_str(end_date, "%Y-%m-%d").unwrap();
    let future_dates: Vec<NaiveDate> = (0..=num_steps)
        .map(|i| {
            last_historical_date
                .checked_add_days(chrono::Days::new(i as u64))
                .unwrap()
        })
        .collect();

    // Combine paths with dates
    let dated_paths: Vec<Vec<(NaiveDate, f64)>> = paths
        .iter()
        .map(|path| {
            path.iter()
                .enumerate()
                .map(|(i, &price)| (future_dates[i], price))
                .collect()
        })
        .collect();

    SimulationResult {
        plot_path: plot_results(symbol, &dated_paths),
        summary_stats: calculate_summary_stats(
            &paths
                .iter()
                .map(|path| path.last().copied().unwrap())
                .collect::<Vec<f64>>(),
        ),
    }
}
