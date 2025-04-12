use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use sps_gbm::{gbm_step, simulate_gbm_path};

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
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
