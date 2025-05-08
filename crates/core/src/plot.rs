use chrono::{Days, NaiveDate};
use plotters::prelude::*;
use tempfile::NamedTempFile;

use std::path::PathBuf;

struct FinalPrices {
    /// The final prices of the simulated paths
    final_prices: Vec<f64>,
}

impl FinalPrices {
    /// Creates a new instance of FinalPrices
    fn new(prices: Vec<f64>) -> Self {
        Self {
            final_prices: prices,
        }
    }

    /// From simulated paths
    fn from_simulated_paths(paths: &SimulatedDatedPaths) -> Self {
        let final_prices = paths
            .paths
            .iter()
            .map(|path| path.last().unwrap().1)
            .collect::<Vec<f64>>();
        Self::new(final_prices)
    }

    /// Compute the distribution of final prices across n intervals
    ///
    /// # Arguments
    /// * `n_intervals` - The number of intervals to divide the price range into
    ///
    /// # Returns
    /// A vector of frequencies (as percentages) for each interval
    fn compute_distribution(&self, n_intervals: usize) -> Vec<f64> {
        if self.final_prices.is_empty() || n_intervals == 0 {
            return Vec::new();
        }

        // Find min and max prices
        let min_price = self
            .final_prices
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = self
            .final_prices
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate interval width
        let range = max_price - min_price;
        let interval_width = range / n_intervals as f64;

        // Initialize counters for each interval
        let mut counts = vec![0; n_intervals];

        // Count occurrences in each interval
        for &price in &self.final_prices {
            // Handle edge case for the maximum price
            let interval = if price == max_price {
                n_intervals - 1
            } else {
                ((price - min_price) / interval_width).floor() as usize
            };
            counts[interval] += 1;
        }

        // Convert counts to percentages
        let total = self.final_prices.len() as f64;
        counts
            .into_iter()
            .map(|count| count as f64 / total)
            .collect()
    }
}

pub struct SimulatedDatedPaths {
    paths: Vec<Vec<(NaiveDate, f64)>>,
}

impl SimulatedDatedPaths {
    /// Creates a new instance of SimulatedDatedPaths
    ///
    /// # Arguments
    /// * `paths` - A vector of vectors containing the simulated paths
    /// * `end_date` - The last date of the historical data (the beginning of the simulation)
    pub fn from_paths(paths: &Vec<Vec<f64>>, end_date: &str) -> Self {
        // Create future dates for simulation
        let last_historical_date = NaiveDate::parse_from_str(end_date, "%Y-%m-%d").unwrap();
        let future_dates: Vec<NaiveDate> = (0..paths[0].len())
            .map(|i| {
                last_historical_date
                    .checked_add_days(Days::new(i as u64))
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
        Self { paths: dated_paths }
    }

    /// Plots the results of the simulation
    ///
    /// # Arguments
    /// * `symbol` - The stock symbol
    ///
    /// # Returns
    /// A PathBuf to the generated plot image
    pub fn plot(self, symbol: &str) -> PathBuf {
        let simulated_paths = &self.paths;
        let final_prices = FinalPrices::from_simulated_paths(&self);

        // Create a temporary file for the output
        let output_path = PathBuf::from(format!(
            "{}.png",
            NamedTempFile::new()
                .unwrap()
                .path()
                .to_path_buf()
                .to_string_lossy()
        ));

        // Create the chart
        let root = BitMapBackend::new(&output_path, (1920, 1080)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let chart_title = format!("Price Simulation for {}", symbol);

        let min_date = simulated_paths[0][0].0;
        let max_date = simulated_paths[0].last().unwrap().0;
        let min_price = simulated_paths
            .iter()
            .flat_map(|path| path.iter().map(|(_, price)| *price))
            .fold(f64::INFINITY, |a, b| a.min(b));
        let max_price = simulated_paths
            .iter()
            .flat_map(|path| path.iter().map(|(_, price)| *price))
            .fold(0.0_f64, |a, b| a.max(b));

        let mut chart = ChartBuilder::on(&root)
            .caption(chart_title, ("SF Mono", 30).into_font())
            .margin(10)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(min_date..max_date, min_price..max_price)
            .unwrap();

        // Add some styling and formatting
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_label_formatter(&|x| x.format("%Y-%m-%d").to_string())
            .y_label_formatter(&|y| format!("{:.2}", y))
            .x_desc("Date")
            .y_desc("Price")
            .draw()
            .unwrap();

        // Trace the paths by drawing each simulation as a line series
        simulated_paths.iter().enumerate().for_each(|(i, path)| {
            chart
                .draw_series(LineSeries::new(
                    path.iter().map(|(date, price)| (*date, *price)),
                    Palette99::pick(i),
                ))
                .unwrap();
        });

        root.present().unwrap();

        output_path.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plot_results_test() {
        let symbol = "AAPL";
        let simulated_paths = vec![
            vec![100.0, 105.0, 108.0, 110.0],
            vec![100.0, 95.0, 98.0, 102.0],
        ];

        // Using the first simulation date as the end_date for the simulation
        let output_path =
            SimulatedDatedPaths::from_paths(&simulated_paths, "2025-03-01").plot(symbol);

        assert!(output_path.exists());
        assert_eq!(output_path.extension(), Some("png".as_ref()));
    }

    #[test]
    fn test_distribution_computation() {
        // Create sample final prices
        let prices = vec![
            100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0,
        ];
        let final_prices = FinalPrices::new(prices);

        // Compute distribution with 5 intervals
        let distribution = final_prices.compute_distribution(5);

        // Expected results: 10 values distributed evenly across 5 intervals = 20% each

        // Check result properties
        assert_eq!(distribution.len(), 5);

        // Check that each interval has 20% of the values (0.2 as a fraction)
        for percentage in &distribution {
            assert!(
                (percentage - 0.2).abs() < 1e-10,
                "Expected 0.2 but got {}",
                percentage
            );
        }

        // Test with uneven distribution
        let uneven_prices = vec![100.0, 100.0, 100.0, 100.0, 140.0, 140.0];
        let uneven_final_prices = FinalPrices::new(uneven_prices);
        let uneven_distribution = uneven_final_prices.compute_distribution(2);

        assert_eq!(uneven_distribution.len(), 2);
        assert!(
            (uneven_distribution[0] - 2.0 / 3.0).abs() < 1e-10,
            "Expected 2/3 but got {}",
            uneven_distribution[0]
        );
        assert!(
            (uneven_distribution[1] - 1.0 / 3.0).abs() < 1e-10,
            "Expected 1/3 but got {}",
            uneven_distribution[1]
        );

        // Test edge case with empty prices
        let empty_prices = FinalPrices::new(vec![]);
        assert!(empty_prices.compute_distribution(5).is_empty());

        // Test edge case with zero intervals
        assert!(final_prices.compute_distribution(0).is_empty());
    }
}
