use simulations::GbmSimulator;

use statrs::statistics::{Data, Distribution as StatsDistribution, Median, OrderStatistics};

pub struct GBMParameters {
    pub initial_value: f64,
    pub drift: f64,
    pub volatility: f64,
    pub dt: f64,
}

impl GBMParameters {
    pub fn new(initial_value: f64, drift: f64, volatility: f64, dt: f64) -> Self {
        Self {
            initial_value,
            drift,
            volatility,
            dt,
        }
    }

    /// Estimates GBM parameters (drift and volatility) from historical prices
    ///
    /// # Arguments
    ///
    /// * `prices` - Vector of historical prices
    /// * `dt` - The time step (fraction of a year) between each price observation
    pub fn from_prices(prices: &[f64], dt: f64) -> Self {
        let initial_value = prices[0];
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

        GBMParameters {
            initial_value,
            drift,
            volatility,
            dt,
        }
    }
}

/// Generate GBM paths from historical prices
///
/// # Arguments
///
/// * `prices` - Vector of historical prices
/// * `num_steps` - The number of steps to simulate
/// * `num_paths` - The number of paths to simulate
///
/// # Returns
///
/// A vector of vectors, where each inner vector represents a simulated path
pub fn generate_gbm_paths_from_prices(
    prices: &[f64],
    num_steps: usize,
    num_paths: usize,
) -> Vec<Vec<f64>> {
    // Hardcoded dt value since here the time step between the historical prices and the simulated prices are the same
    let dt = 1.0;

    let gbm_parameters = GBMParameters::from_prices(prices, dt);

    GbmSimulator::new(
        gbm_parameters.initial_value,
        gbm_parameters.drift,
        gbm_parameters.volatility,
        gbm_parameters.dt,
    )
    .simulate_paths(num_steps, num_paths)
    .into_vec_of_vec()
}

#[derive(Debug)]
pub struct ConfidenceInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
}

impl ConfidenceInterval {
    pub fn new(lower_bound: f64, upper_bound: f64) -> Self {
        Self {
            lower_bound,
            upper_bound,
        }
    }
}

#[derive(Debug)]
pub struct Percenticles {
    pub p5: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
}

impl Percenticles {
    pub fn new(p5: f64, p10: f64, p25: f64, p50: f64, p75: f64, p90: f64, p95: f64) -> Self {
        Self {
            p5,
            p10,
            p25,
            p50,
            p75,
            p90,
            p95,
        }
    }
}

#[derive(Debug)]
pub struct SummaryStats {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub confidence_interval_95: ConfidenceInterval,
    pub percentiles: Percenticles,
}

impl SummaryStats {
    pub fn new(
        mean: f64,
        median: f64,
        std_dev: f64,
        confidence_interval_95: ConfidenceInterval,
        percentiles: Percenticles,
    ) -> Self {
        Self {
            mean,
            median,
            std_dev,
            confidence_interval_95,
            percentiles,
        }
    }

    /// Calculates summary statistics for a vector of final prices
    ///
    /// # Arguments
    /// * `prices` - A slice of f64 values representing the final prices of all simulated paths
    ///
    /// # Returns
    /// A SummaryStats struct each field representing a summary statistic
    pub fn from_prices(prices: &[f64]) -> Self {
        let mut data = Data::new(prices.to_vec());

        let mean = data.mean().unwrap();
        let std_dev = data.std_dev().unwrap();
        let n = prices.len() as f64;
        let std_error = std_dev / n.sqrt();

        // Using 1.96 for 95% confidence interval (normal distribution)
        let z_score = 1.96;

        SummaryStats {
            mean,
            median: data.median(),
            std_dev,
            // TODO: Fix, Might not be correct calculation for CI
            confidence_interval_95: ConfidenceInterval::new(
                mean - z_score * std_error,
                mean + z_score * std_error,
            ),
            percentiles: Percenticles::new(
                data.percentile(5),
                data.percentile(10),
                data.percentile(25),
                data.percentile(50),
                data.percentile(75),
                data.percentile(90),
                data.percentile(95),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_summary_stats_test() {
        let prices = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = SummaryStats::from_prices(&prices);

        assert_eq!(stats.mean, 30.0);
        assert_eq!(stats.median, 30.0);
        assert_eq!(stats.std_dev, 15.811388300841896);
        assert_eq!(stats.confidence_interval_95.lower_bound, 16.140707088743667);
        assert_eq!(stats.confidence_interval_95.upper_bound, 43.85929291125633);
        assert_eq!(stats.percentiles.p5, 10.0);
        assert_eq!(stats.percentiles.p10, 10.0);
        assert_eq!(stats.percentiles.p25, 16.666666666666664);
        assert_eq!(stats.percentiles.p50, 30.0);
        assert_eq!(stats.percentiles.p75, 43.33333333333333);
        assert_eq!(stats.percentiles.p90, 50.0);
        assert_eq!(stats.percentiles.p95, 50.0);
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
            let gbm_parameters = GBMParameters::from_prices(&tc.prices, tc.dt);
            assert_eq!(gbm_parameters.drift, tc.expected_drift);
            assert_eq!(gbm_parameters.volatility, tc.expected_volatility);
        }
    }

    #[test]
    fn generate_gbm_paths_from_prices_correct() {
        let prices = vec![100.0, 105.0, 110.0];
        let num_steps = 10;
        let num_paths = 5;

        let paths = generate_gbm_paths_from_prices(&prices, num_steps, num_paths);

        assert_eq!(paths.len(), num_paths);
        for path in paths.iter() {
            assert_eq!(path.len(), num_steps + 1);
            assert_eq!(path[0], prices[0]);
            for i in 1..path.len() {
                assert!(path[i] > 0.0);
            }
        }
    }
}
