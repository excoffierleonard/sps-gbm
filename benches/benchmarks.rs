use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use sps_gbm::{
    estimate_gbm_parameters, gbm_step, generate_gbm_paths_from_prices, simulate_gbm_path,
};

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
    g.bench_function("Generate Paths from Prices", |b| {
        b.iter(|| {
            generate_gbm_paths_from_prices(
                black_box(&[100.0, 105.0, 110.0, 115.0]),
                black_box(1.0),
                black_box(10),
                black_box(5),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
