use statrs::statistics::{Data, Distribution as StatsDistribution, Median, OrderStatistics};

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
}
