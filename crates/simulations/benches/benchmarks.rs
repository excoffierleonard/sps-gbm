use simulations::GbmSimulator;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("1 Simulation");
    g.throughput(Throughput::Elements(1));

    let params = GbmSimulator::new(100.0, 0.05, 0.2, 1.0);

    g.bench_function("1,000 Steps | 1,000 Paths ", |b| {
        b.iter(|| params.simulate_paths(black_box(1_000), black_box(1_000)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
