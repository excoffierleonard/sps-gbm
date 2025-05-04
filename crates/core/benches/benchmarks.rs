use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use dotenvy::dotenv;
use rand::Rng;
use rand::rng;

use std::env;

use core::{SimulatedDatedPaths, Simulation, SummaryStats};

fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("GBM");
    g.throughput(Throughput::Elements(1));
    g.sample_size(10);

    // Generate 1000 simulated paths, each with 1000 steps using a simple random generator.
    let mut rng = rng();
    let paths: Vec<Vec<f64>> = (0..1000)
        .map(|_| (0..1000).map(|_| rng.random_range(90.0..110.0)).collect())
        .collect();

    // TOFIX: Why does this take 2 seconds but e2e takes <100ms ?
    g.bench_function("Plot Results", |b| {
        b.iter(|| {
            SimulatedDatedPaths::from_paths(black_box(&paths), black_box("2025-04-01"))
                .plot(black_box("AAPL"))
        })
    });
    g.bench_function("Calculate Summary Stats", |b| {
        b.iter(|| SummaryStats::from_prices(black_box(&paths[999])))
    });
}

// This benchmark requires a valid API key and network access
fn main_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("GBM");
    g.throughput(Throughput::Elements(1));

    dotenv().ok();
    let alphavantage_api_key = env::var("ALPHAVANTAGE_API_KEY").unwrap();

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
