use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

/// Calculates a single step of geometric Brownian motion
///
/// # Arguments
///
/// * `current_value` - The current value S(t)
/// * `drift` - The drift parameter μ
/// * `volatility` - The volatility parameter σ
/// * `dt` - The time step Δt
/// * `z` - Standard normal random variable (N(0,1))
///
/// # Returns
///
/// The next value S(t+Δt)
pub fn gbm_step(current_value: f64, drift: f64, volatility: f64, dt: f64, z: f64) -> f64 {
    let drift_term = (drift - 0.5 * volatility * volatility) * dt;
    let diffusion_term = volatility * dt.sqrt() * z;

    current_value * (drift_term + diffusion_term).exp()
}

/// Simulates a path of geometric Brownian motion
///
/// # Arguments
///
/// * `initial_value` - The initial value S(0)
/// * `drift` - The drift parameter μ
/// * `volatility` - The volatility parameter σ
/// * `dt` - The time step Δt
/// * `num_steps` - The number of steps to simulate
///
/// # Returns
///
/// A vector containing the simulated path of values
pub fn simulate_gbm_path(
    initial_value: f64,
    drift: f64,
    volatility: f64,
    dt: f64,
    num_steps: usize,
) -> Vec<f64> {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Pregenerate all random z values
    let z_values: Vec<f64> = (0..num_steps).map(|_| normal.sample(&mut rng)).collect();

    let mut path = Vec::with_capacity(num_steps + 1);
    path.push(initial_value);

    for &z in z_values.iter() {
        let next_value = gbm_step(path.last().copied().unwrap(), drift, volatility, dt, z);
        path.push(next_value);
    }

    path
}

pub struct GBMParameters {
    pub drift: f64,
    pub volatility: f64,
}

/// Estimates GBM parameters (drift and volatility) from historical prices
///
/// # Arguments
///
/// * `prices` - Vector of historical prices
/// * `dt` - The time step (fraction of a year) between each price observation
///
/// # Returns
///
/// A tuple (drift, volatility) with annualized parameters for GBM
pub fn estimate_gbm_parameters(prices: &[f64], dt: f64) -> GBMParameters {
    let log_returns: Vec<f64> = prices
        .windows(2)
        .map(|window| (window[1] / window[0]).ln())
        .collect();

    let mean_log_return = log_returns.iter().copied().sum::<f64>() / log_returns.len() as f64;
    let variance_log_return = log_returns
        .iter()
        .map(|&x| (x - mean_log_return).powi(2))
        .sum::<f64>()
        / (log_returns.len() - 1) as f64;

    let volatility = variance_log_return.sqrt() / dt.sqrt();
    // Add the volatility adjustment to get the correct drift
    let drift = mean_log_return / dt + 0.5 * volatility.powi(2);

    GBMParameters { drift, volatility }
}

/// Generate GBM paths from historical prices
///
/// # Arguments
///
/// * `prices` - Vector of historical prices
/// * `dt` - The time step (fraction of a year) between each price observation
/// * `num_steps` - The number of steps to simulate
/// * `num_paths` - The number of paths to simulate
///
/// # Returns
///
/// A vector of vectors, where each inner vector represents a simulated path
pub fn generate_gbm_paths_from_prices(
    prices: &[f64],
    dt: f64,
    num_steps: usize,
    num_paths: usize,
) -> Vec<Vec<f64>> {
    let gbm_parameters = estimate_gbm_parameters(prices, dt);
    let initial_value = prices[0];

    (0..num_paths)
        .into_par_iter()
        .map(|_| {
            simulate_gbm_path(
                initial_value,
                gbm_parameters.drift,
                gbm_parameters.volatility,
                dt,
                num_steps,
            )
        })
        .collect()
}

#[derive(Deserialize, Serialize)]
struct DailyData {
    #[serde(rename = "4. close")]
    close: String,
}

#[derive(Deserialize, Serialize)]
struct ApiResponse {
    #[serde(rename = "Time Series (Daily)")]
    time_series: BTreeMap<String, DailyData>,
}

/// Fetches historical prices from the Alpha Vantage API
///
/// # Arguments
/// * `symbol` - The stock symbol to fetch prices for
/// * `api_key` - Your Alpha Vantage API key
/// * `start_date` - The start date for fetching prices (YYYY-MM-DD)
/// * `end_date` - The end date for fetching prices (YYYY-MM-DD)
///
/// # Returns
/// A vector of historical closing prices in chronological order (oldest to newest)
pub fn fetch_historical_prices(
    symbol: &str,
    api_key: &str,
    start_date: &str,
    end_date: &str,
) -> Vec<f64> {
    // Create cache directory if it doesn't exist
    let cache_dir = Path::new("cache");
    if !cache_dir.exists() {
        fs::create_dir_all(cache_dir).unwrap();
    }

    // Check if we have cached data for this ticker
    let cache_file = cache_dir.join(format!("{}.json", symbol));

    // TODO: Verify that the funciton does indeed check for date bounds being present

    if cache_file.exists() {
        println!("Using cached data for {}", symbol);
        let cached_data = fs::read_to_string(&cache_file).unwrap();

        let api_data: ApiResponse = serde_json::from_str(&cached_data).unwrap();

        // Filter dates within the specified range and extract closing prices
        let mut prices: Vec<(String, f64)> = api_data
            .time_series
            .iter()
            .filter(|(date, _)| date.as_str() >= start_date && date.as_str() <= end_date)
            .map(|(date, data)| (date.clone(), data.close.parse::<f64>().unwrap_or(0.0)))
            .collect();

        // Sort by date (oldest first)
        prices.sort_by(|(date_a, _), (date_b, _)| date_a.cmp(date_b));

        // Return just the prices
        return prices.into_iter().map(|(_, price)| price).collect();
    }

    // If no cached data, fetch from API
    println!("Fetching data for {} from Alpha Vantage...", symbol);

    let url = format!(
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey={}&outputsize=full",
        symbol, api_key
    );

    let client = Client::new();
    let response = client.get(&url).send().unwrap();
    let api_data: ApiResponse = response.json().unwrap();

    // Cache the data for future use
    let cached_json = serde_json::to_string(&api_data).unwrap();
    fs::write(&cache_file, cached_json).unwrap();

    // Filter dates within the specified range and extract closing prices
    let mut prices: Vec<(String, f64)> = api_data
        .time_series
        .iter()
        .filter(|(date, _)| date.as_str() >= start_date && date.as_str() <= end_date)
        .map(|(date, data)| (date.clone(), data.close.parse::<f64>().unwrap_or(0.0)))
        .collect();

    // Sort by date (oldest first)
    prices.sort_by(|(date_a, _), (date_b, _)| date_a.cmp(date_b));

    // Return just the prices
    prices.into_iter().map(|(_, price)| price).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbm_step_formula() {
        struct TestCase {
            current_value: f64,
            drift: f64,
            volatility: f64,
            dt: f64,
            z: f64,
            expected: f64,
        }

        let test_cases = [
            TestCase {
                current_value: 100.0,
                drift: 0.05,
                volatility: 0.2,
                dt: 1.0,
                z: 0.5,
                expected: 113.88283833246217,
            },
            TestCase {
                current_value: 150.0,
                drift: 0.03,
                volatility: 0.15,
                dt: 1.0,
                z: -0.3,
                expected: 146.1137304422672,
            },
            TestCase {
                current_value: 200.0,
                drift: 0.07,
                volatility: 0.25,
                dt: 0.5,
                z: 1.0,
                expected: 243.3422921483655,
            },
        ];

        for tc in test_cases.iter() {
            let next_value = gbm_step(tc.current_value, tc.drift, tc.volatility, tc.dt, tc.z);
            assert_eq!(next_value, tc.expected);
        }
    }

    #[test]
    fn simulate_gbm_path_correct() {
        let initial_value = 100.0;
        let drift = 0.05;
        let volatility = 0.2;
        let dt = 1.0;
        let num_steps = 10;

        let path = simulate_gbm_path(initial_value, drift, volatility, dt, num_steps);

        assert_eq!(path.len(), num_steps + 1);
        assert_eq!(path[0], initial_value);

        for i in 1..path.len() {
            assert!(path[i] > 0.0);
        }
    }

    #[test]
    fn estimate_gbm_parameters_formulas() {
        struct TestCase {
            prices: Vec<f64>,
            dt: f64,
            expected_drift: f64,
            expected_volatility: f64,
        }

        let test_cases = [
            TestCase {
                prices: vec![100.0, 105.0, 110.0],
                dt: 1.0,
                expected_drift: 0.0476563782957547,
                expected_volatility: 0.0016052374230733303,
            },
            TestCase {
                prices: vec![200.0, 210.0, 220.0],
                dt: 1.0,
                expected_drift: 0.0476563782957547,
                expected_volatility: 0.0016052374230733303,
            },
        ];

        for tc in test_cases.iter() {
            let gbm_parameters = estimate_gbm_parameters(&tc.prices, tc.dt);
            assert_eq!(gbm_parameters.drift, tc.expected_drift);
            assert_eq!(gbm_parameters.volatility, tc.expected_volatility);
        }
    }

    #[test]
    fn generate_gbm_paths_from_prices_correct() {
        let prices = vec![100.0, 105.0, 110.0];
        let dt = 1.0;
        let num_steps = 10;
        let num_paths = 5;

        let paths = generate_gbm_paths_from_prices(&prices, dt, num_steps, num_paths);

        assert_eq!(paths.len(), num_paths);
        for path in paths.iter() {
            assert_eq!(path.len(), num_steps + 1);
            assert_eq!(path[0], prices[0]);
            for i in 1..path.len() {
                assert!(path[i] > 0.0);
            }
        }
    }

    #[test]
    #[ignore] // Requires a valid API key and network connection
    fn fetch_historical_prices_test() {
        use std::env;
        dotenvy::dotenv().ok();

        let api_key = env::var("ALPHAVANTAGE_API_KEY").unwrap();

        let result = fetch_historical_prices("AAPL", &api_key, "2025-03-01", "2025-04-01");

        assert!(!result.is_empty());
        assert!(result.iter().all(|&price| price > 0.0));
    }
}
