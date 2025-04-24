// External crates
use chrono::NaiveDate;

// Standard library
use std::path::PathBuf;

// Local crates
use simulations::GbmSimulator;

// Module declarations
mod calculations;
mod data;
mod plot;

// Re-exports
pub use calculations::SummaryStats;
pub use data::{AlphaVantage, DateRange, PriceProvider};
pub use plot::SimulatedDatedPaths;

pub struct Simulation {
    // The path to the generated plot image
    pub plot_path: PathBuf,
    // The summary statistics of the simulation
    pub summary_stats: SummaryStats,
}

impl Simulation {
    /// End-to-end, high-level function to simulate GBM paths and plot results
    ///
    /// # Arguments
    /// * `symbol` - The stock symbol to fetch prices for
    /// * `api_key` - Your Alpha Vantage API key
    /// * `start_date` - The start date for fetching prices (YYYY-MM-DD)
    /// * `end_date` - The end date for fetching prices (YYYY-MM-DD)
    /// * `num_steps` - The number of steps to simulate
    /// * `num_paths` - The number of paths to simulate
    pub fn generate(
        symbol: &str,
        api_key: &str,
        start_date: &str,
        end_date: &str,
        num_steps: usize,
        num_paths: usize,
    ) -> Simulation {
        // Fetch historical prices
        let historical_prices = AlphaVantage::new(api_key.to_string())
            .fetch_prices(symbol, &DateRange::new(start_date, end_date));

        // Generate simulated paths
        let paths =
            GbmSimulator::from_prices(&historical_prices, 1.0).simulate_paths(num_steps, num_paths);

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

        Simulation {
            plot_path: SimulatedDatedPaths::from_paths(dated_paths).plot(symbol),
            summary_stats: SummaryStats::from_prices(
                &paths
                    .iter()
                    .map(|path| path.last().copied().unwrap())
                    .collect::<Vec<f64>>(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    #[ignore] // Requires a valid API key and network connection
    fn generate_simulation_test() {
        dotenvy::dotenv().ok();

        let api_key = env::var("ALPHAVANTAGE_API_KEY").unwrap();

        let simulation_result =
            Simulation::generate("AAPL", &api_key, "2025-01-01", "2025-04-01", 100, 100);

        let plot_path = simulation_result.plot_path;
        assert!(plot_path.exists());
        assert_eq!(plot_path.extension(), Some("png".as_ref()));

        let stats = simulation_result.summary_stats;
        assert!(stats.mean > 0.0);
        assert!(stats.median > 0.0);
        assert!(stats.std_dev > 0.0);
        assert!(stats.confidence_interval_95.lower_bound < stats.mean);
        assert!(stats.confidence_interval_95.upper_bound > stats.mean);
        assert!(stats.percentiles.p5 < stats.percentiles.p10);
        assert!(stats.percentiles.p10 < stats.percentiles.p25);
        assert!(stats.percentiles.p25 < stats.percentiles.p50);
        assert!(stats.percentiles.p50 < stats.percentiles.p75);
        assert!(stats.percentiles.p75 < stats.percentiles.p90);
        assert!(stats.percentiles.p90 < stats.percentiles.p95);
    }
}
