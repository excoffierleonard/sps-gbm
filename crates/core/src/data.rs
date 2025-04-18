use std::{collections::BTreeMap, fs, path::Path};

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

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

#[derive(Deserialize, Serialize, Clone)]
pub struct PriceData {
    pub date: String,
    pub price: f64,
}

/// Retrieves price data from the cache if available
///
/// # Arguments
/// * `symbol` - The stock symbol to get cached data for
/// * `start_date` - The start date for filtering prices (YYYY-MM-DD)
/// * `end_date` - The end date for filtering prices (YYYY-MM-DD)
///
/// # Returns
/// An Option containing a vector of historical closing prices if cache exists,
/// or None if no cache is available
pub fn get_cached_prices(symbol: &str, start_date: &str, end_date: &str) -> Option<Vec<f64>> {
    // Create cache directory if it doesn't exist
    let cache_dir = Path::new("cache");
    if !cache_dir.exists() {
        fs::create_dir_all(cache_dir).unwrap();
    }

    // Check if we have cached data for this ticker
    let cache_file = cache_dir.join(format!("{}.json", symbol));

    if cache_file.exists() {
        let cached_data = fs::read_to_string(&cache_file).unwrap();
        let price_data: Vec<PriceData> = serde_json::from_str(&cached_data).unwrap();

        // Filter dates within the specified range
        let mut filtered_prices: Vec<PriceData> = price_data
            .into_iter()
            .filter(|data| data.date.as_str() >= start_date && data.date.as_str() <= end_date)
            .collect();

        // Sort by date (oldest first)
        filtered_prices.sort_by(|a, b| a.date.cmp(&b.date));

        // Return just the prices
        return Some(filtered_prices.into_iter().map(|data| data.price).collect());
    }

    None
}

/// Stores price data in the cache
///
/// # Arguments
/// * `symbol` - The stock symbol the data belongs to
/// * `price_data` - Vector of price data to cache
///
/// # Returns
/// Result indicating success or failure
pub fn cache_prices(symbol: &str, price_data: Vec<PriceData>) -> std::io::Result<()> {
    let cache_dir = Path::new("cache");
    if !cache_dir.exists() {
        fs::create_dir_all(cache_dir)?;
    }

    let cache_file = cache_dir.join(format!("{}.json", symbol));
    let cached_json = serde_json::to_string(&price_data).unwrap();
    fs::write(&cache_file, cached_json)
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
pub fn fetch_historical_prices_alphavantage(
    symbol: &str,
    api_key: &str,
    start_date: &str,
    end_date: &str,
) -> Vec<f64> {
    // Try to get prices from cache first
    if let Some(cached_prices) = get_cached_prices(symbol, start_date, end_date) {
        return cached_prices;
    }

    // If no cache hit, fetch from API
    let url = format!(
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey={}&outputsize=full",
        symbol, api_key
    );

    let client = Client::new();
    let response = client.get(&url).send().unwrap();

    // Parse API response using the old format
    let api_data: ApiResponse = response.json().unwrap();

    // Transform the data to the new format
    let price_data: Vec<PriceData> = api_data
        .time_series
        .iter()
        .map(|(date, data)| PriceData {
            date: date.clone(),
            price: data.close.parse::<f64>().unwrap_or(0.0),
        })
        .collect();

    // Cache the data using the new format
    let _ = cache_prices(symbol, price_data.clone());

    // Filter dates within the specified range
    let mut filtered_prices: Vec<PriceData> = price_data
        .into_iter()
        .filter(|data| data.date.as_str() >= start_date && data.date.as_str() <= end_date)
        .collect();

    // Sort by date (oldest first)
    filtered_prices.sort_by(|a, b| a.date.cmp(&b.date));

    // Return just the prices
    filtered_prices.into_iter().map(|data| data.price).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    #[ignore] // Requires a valid API key and network connection
    fn fetch_historical_prices_test() {
        dotenvy::dotenv().ok();

        let api_key = env::var("ALPHAVANTAGE_API_KEY").unwrap();

        let result =
            fetch_historical_prices_alphavantage("AAPL", &api_key, "2025-03-01", "2025-04-01");

        assert!(!result.is_empty());
        assert!(result.iter().all(|&price| price > 0.0));
    }
}
