use std::env;

use sps_gbm::generate_simulation;

#[test]
#[ignore] // Requires a valid API key and network connection
fn generate_simulation_test() {
    dotenvy::dotenv().ok();

    let api_key = env::var("ALPHAVANTAGE_API_KEY").unwrap();

    let simulation_result =
        generate_simulation("AAPL", &api_key, "2025-01-01", "2025-04-01", 100, 100);

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
