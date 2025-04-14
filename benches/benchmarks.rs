use sps_gbm::{
    calculate_summary_stats, estimate_gbm_parameters, gbm_step, generate_gbm_paths_from_prices,
    plot_results, simulate_gbm_path,
};

use chrono::NaiveDate;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("GBM");
    g.throughput(Throughput::Elements(1));

    g.bench_function("Step", |b| {
        b.iter(|| {
            gbm_step(
                black_box(100.0),
                black_box(0.05),
                black_box(0.2),
                black_box(1.0),
                black_box(0.5),
            )
        })
    });
    g.bench_function("Path | 1,000 Steps", |b| {
        b.iter(|| {
            simulate_gbm_path(
                black_box(100.0),
                black_box(0.05),
                black_box(0.2),
                black_box(1.0),
                black_box(1_000),
            )
        })
    });
    g.bench_function("Calculate Parameters", |b| {
        b.iter(|| estimate_gbm_parameters(black_box(&[100.0, 105.0, 110.0, 115.0]), black_box(1.0)))
    });
    g.bench_function("Generate 10,000 Paths from Prices", |b| {
        b.iter(|| {
            generate_gbm_paths_from_prices(
                black_box(&[100.0, 105.0, 110.0, 115.0]),
                black_box(1_000),
                black_box(10_000),
            )
        })
    });
    // No benchmark for fetch_historical_prices as it requires network access and api key, might test the caching fetching later
    g.bench_function("Plot Results", |b| {
        b.iter(|| {
            plot_results(
                black_box("AAPL"),
                black_box(&vec![
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
                ]),
            )
        })
    });
    g.bench_function("Calculate Summary Stats", |b| {
        b.iter(|| calculate_summary_stats(black_box(&vec![10.0, 20.0, 30.0, 40.0, 50.0])))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
