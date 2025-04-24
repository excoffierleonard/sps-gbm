use core::{SimulatedDatedPaths, Simulation, SummaryStats};

use chrono::NaiveDate;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use dotenvy::dotenv;

fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("GBM");
    g.throughput(Throughput::Elements(1));

    // No benchmark for fetch_historical_prices as it requires network access and api key, might test the caching fetching later
    g.bench_function("Plot Results", |b| {
        b.iter(|| {
            SimulatedDatedPaths::from_paths(vec![
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
            ])
            .plot("AAPL")
        })
    });
    g.bench_function("Calculate Summary Stats", |b| {
        b.iter(|| SummaryStats::from_prices(black_box(&vec![10.0, 20.0, 30.0, 40.0, 50.0])))
    });
}

// This benchmark requires a valid API key and network access
fn main_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("GBM");
    g.throughput(Throughput::Elements(1));

    dotenv().ok();
    let alphavantage_api_key = std::env::var("ALPHAVANTAGE_API_KEY").unwrap();

    g.bench_function("End-to-End function", |b| {
        b.iter(|| {
            Simulation::generate(
                black_box("AAPL"),
                black_box(&alphavantage_api_key),
                black_box("2025-01-01"),
                black_box("2025-04-01"),
                black_box(1_000),
                black_box(1_000),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark, main_benchmark);
criterion_main!(benches);
