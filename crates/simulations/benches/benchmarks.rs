use simulations::simulate_gbm_paths;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("Simulations");
    g.throughput(Throughput::Elements(1));

    g.bench_function("1 Simulation | 1,000 Paths | 1,000 Steps", |b| {
        b.iter(|| {
            simulate_gbm_paths(
                black_box(100.0),
                black_box(0.05),
                black_box(0.2),
                black_box(1.0),
                black_box(1_000),
                black_box(1_000),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
