use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use sps_gbm::gbm_step;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("GBM Step");
    group.throughput(Throughput::Elements(1));

    group.bench_function("GBM Function", |b| {
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
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
