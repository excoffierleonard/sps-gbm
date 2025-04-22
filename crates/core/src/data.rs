use std::{collections::BTreeMap, fs, path::Path};

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

pub struct DateRange {
    start: String,
    end: String,
}

impl DateRange {
    pub fn new(start: &str, end: &str) -> Self {
        // Validate date format (YYYY-MM-DD)
        if !start.is_empty() && !end.is_empty() {
            if start.len() != 10 || end.len() != 10 {
                panic!("Invalid date format. Use YYYY-MM-DD.");
            }
            if start > end {
                panic!("Start date must be before or equal to end date.");
            }
        }

        // Create DateRange
        Self {
            start: start.to_string(),
            end: end.to_string(),
        }
    }
}

/// Represents a provider that can fetch historical prices.
pub trait PriceProvider {
    /// Returns (chronological) daily closing prices for the specified symbol and date range.
    fn fetch_prices(&self, symbol: &str, range: &DateRange) -> Vec<f64>;
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

#[derive(Deserialize, Serialize, Clone)]
pub struct PriceData {
    pub date: String,
    pub price: f64,
}

pub struct AlphaVantage {
    api_key: String,
}

impl AlphaVantage {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
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
    fn get_cached_prices(symbol: &str, start_date: &str, end_date: &str) -> Option<Vec<f64>> {
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
    fn cache_prices(symbol: &str, price_data: Vec<PriceData>) -> () {
        let cache_dir = Path::new("cache");
        if !cache_dir.exists() {
            fs::create_dir_all(cache_dir).unwrap();
        }

        let cache_file = cache_dir.join(format!("{}.json", symbol));
        let cached_json = serde_json::to_string(&price_data).unwrap();
        fs::write(&cache_file, cached_json).unwrap()
    }
}

impl PriceProvider for AlphaVantage {
    /// Fetches historical prices from the Alpha Vantage API
    ///
    /// # Arguments
    /// * `symbol` - The stock symbol to fetch prices for
    /// * `range` - The date range for which to fetch prices
    ///
    /// # Returns
    /// A vector of historical closing prices in chronological order (oldest to newest)
    fn fetch_prices(&self, symbol: &str, range: &DateRange) -> Vec<f64> {
        let api_key = &self.api_key;
        let start_date = &range.start;
        let end_date = &range.end;

        // Try to get prices from cache first
        if let Some(cached_prices) = Self::get_cached_prices(symbol, start_date, end_date) {
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
        let _ = Self::cache_prices(symbol, price_data.clone());

        // Filter dates within the specified range
        let mut filtered_prices: Vec<PriceData> = price_data
            .into_iter()
            .filter(|data| {
                data.date.as_str() >= start_date.as_str() && data.date.as_str() <= end_date.as_str()
            })
            .collect();

        // Sort by date (oldest first)
        filtered_prices.sort_by(|a, b| a.date.cmp(&b.date));

        // Return just the prices
        filtered_prices.into_iter().map(|data| data.price).collect()
    }
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

        let result = AlphaVantage::new(api_key)
            .fetch_prices("AAPL", &DateRange::new("2024-03-01", "2024-03-31"));

        assert!(!result.is_empty());
        assert!(result.iter().all(|&price| price > 0.0));
    }
}
