use chrono::{Days, NaiveDate};
use plotters::prelude::*;
use tempfile::NamedTempFile;

use std::path::PathBuf;

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
        let simulated_paths = self.paths;

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
}
