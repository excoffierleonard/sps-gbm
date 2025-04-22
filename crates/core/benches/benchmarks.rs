use core::{GBMParameters, Prices, SimulatedDatedPaths, SummaryStats};

use chrono::NaiveDate;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("GBM");
    g.throughput(Throughput::Elements(1));

    g.bench_function("Calculate Parameters", |b| {
        b.iter(|| {
            GBMParameters::from_prices(black_box(&[100.0, 105.0, 110.0, 115.0]), black_box(1.0))
        })
    });
    g.bench_function("Generate 1,000 Paths from Prices", |b| {
        b.iter(|| {
            Prices::from_slice(black_box(&[100.0, 105.0, 110.0, 115.0]))
                .simulate_paths(black_box(1_000), black_box(1_000))
        })
    });

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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
