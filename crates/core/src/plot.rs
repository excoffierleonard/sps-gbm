use std::path::PathBuf;

use chrono::NaiveDate;
use plotters::prelude::*;
use tempfile::NamedTempFile;

pub struct SimulatedDatedPaths {
    paths: Vec<Vec<(NaiveDate, f64)>>,
}

impl SimulatedDatedPaths {
    /// Creates a new instance of SimulatedDatedPaths
    ///
    /// # Arguments
    /// * `paths` - A vector of simulated paths, where each path is a vector of (date, price) tuples
    pub fn from_paths(paths: Vec<Vec<(NaiveDate, f64)>>) -> Self {
        Self { paths }
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
            vec![
                (NaiveDate::from_ymd_opt(2025, 3, 1).unwrap(), 100.0),
                (NaiveDate::from_ymd_opt(2025, 3, 2).unwrap(), 105.0),
                (NaiveDate::from_ymd_opt(2025, 3, 3).unwrap(), 108.0),
                (NaiveDate::from_ymd_opt(2025, 3, 4).unwrap(), 110.0),
            ],
            vec![
                (NaiveDate::from_ymd_opt(2025, 3, 1).unwrap(), 100.0),
                (NaiveDate::from_ymd_opt(2025, 3, 2).unwrap(), 95.0),
                (NaiveDate::from_ymd_opt(2025, 3, 3).unwrap(), 98.0),
                (NaiveDate::from_ymd_opt(2025, 3, 4).unwrap(), 102.0),
            ],
        ];

        let output_path = SimulatedDatedPaths::from_paths(simulated_paths).plot(symbol);

        assert!(output_path.exists());
        assert_eq!(output_path.extension(), Some("png".as_ref()));
    }
}
