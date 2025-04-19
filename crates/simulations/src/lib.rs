use rand::rng;
use rand_distr::{Distribution as RandDistribution, StandardNormal};
use rayon::prelude::*;

/// Calculates a single step of geometric Brownian motion
///
/// # Arguments
///
/// * `current_value` - The current value S(t)
/// * `drift` - The drift parameter μ
/// * `volatility` - The volatility parameter σ
/// * `dt` - The time step Δt
/// * `z` - Standard normal random variable (N(0,1))
///
/// # Returns
///
/// The next value S(t+Δt)
fn gbm_step(current_value: f64, drift: f64, volatility: f64, dt: f64, z: f64) -> f64 {
    let drift_term = (drift - 0.5 * volatility * volatility) * dt;
    let diffusion_term = volatility * dt.sqrt() * z;

    current_value * (drift_term + diffusion_term).exp()
}

/// Generates a vector of standard normal random variables
///
/// # Arguments
///
/// * `num_steps` - The number of random variables to generate
///
/// # Returns
///
/// A vector of standard normal random variables
fn generate_random_normal_zs(num_steps: usize) -> Vec<f64> {
    let mut rng = rng();
    (0..num_steps)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect()
}

/// Simulates a path of geometric Brownian motion
///
/// # Arguments
///
/// * `initial_value` - The initial value S(0)
/// * `drift` - The drift parameter μ
/// * `volatility` - The volatility parameter σ
/// * `dt` - The time step Δt
/// * `num_steps` - The number of steps to simulate
///
/// # Returns
///
/// A vector containing the simulated path of values
fn simulate_gbm_path(
    initial_value: f64,
    drift: f64,
    volatility: f64,
    dt: f64,
    num_steps: usize,
) -> Vec<f64> {
    // Pregenerate all random z values
    let z_values = generate_random_normal_zs(num_steps);

    // Iterate through the z values to calculate the path
    let mut path = Vec::with_capacity(num_steps + 1);
    path.push(initial_value);
    let mut current_value = initial_value;
    for &z in &z_values {
        let next_value = gbm_step(current_value, drift, volatility, dt, z);
        path.push(next_value);
        current_value = next_value;
    }
    path
}

/// Simulates multiple paths of geometric Brownian motion
///
/// # Arguments
///
/// * `initial_value` - The initial value S(0)
/// * `drift` - The drift parameter μ
/// * `volatility` - The volatility parameter σ
/// * `dt` - The time step Δt
/// * `num_steps` - The number of steps to simulate
/// * `num_paths` - The number of paths to simulate
///
/// # Returns
///
/// A vector of vectors, where each inner vector represents a simulated path
pub fn simulate_gbm_paths(
    initial_value: f64,
    drift: f64,
    volatility: f64,
    dt: f64,
    num_steps: usize,
    num_paths: usize,
) -> Vec<Vec<f64>> {
    (0..num_paths)
        .into_par_iter()
        .map(|_| simulate_gbm_path(initial_value, drift, volatility, dt, num_steps))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbm_step_formula() {
        struct TestCase {
            current_value: f64,
            drift: f64,
            volatility: f64,
            dt: f64,
            z: f64,
            expected: f64,
        }

        let test_cases = [
            TestCase {
                current_value: 100.0,
                drift: 0.05,
                volatility: 0.2,
                dt: 1.0,
                z: 0.5,
                expected: 113.88283833246217,
            },
            TestCase {
                current_value: 150.0,
                drift: 0.03,
                volatility: 0.15,
                dt: 1.0,
                z: -0.3,
                expected: 146.1137304422672,
            },
            TestCase {
                current_value: 200.0,
                drift: 0.07,
                volatility: 0.25,
                dt: 0.5,
                z: 1.0,
                expected: 243.3422921483655,
            },
        ];

        for tc in test_cases.iter() {
            let next_value = gbm_step(tc.current_value, tc.drift, tc.volatility, tc.dt, tc.z);
            assert_eq!(next_value, tc.expected);
        }
    }

    #[test]
    fn generate_random_normal_zs_correct() {
        let num_steps = 1000;
        let zs = generate_random_normal_zs(num_steps);

        assert_eq!(zs.len(), num_steps);
        for &z in &zs {
            assert!(z.is_finite());
        }
    }

    #[test]
    fn simulate_gbm_path_correct() {
        let initial_value = 100.0;
        let drift = 0.05;
        let volatility = 0.2;
        let dt = 1.0;
        let num_steps = 10;

        let path = simulate_gbm_path(initial_value, drift, volatility, dt, num_steps);

        assert_eq!(path.len(), num_steps + 1);
        assert_eq!(path[0], initial_value);

        for i in 1..path.len() {
            assert!(path[i] > 0.0);
        }
    }

    #[test]
    fn simulate_gbm_paths_correct() {
        let initial_value = 100.0;
        let drift = 0.05;
        let volatility = 0.2;
        let dt = 1.0;
        let num_steps = 10;
        let num_paths = 5;

        let paths = simulate_gbm_paths(initial_value, drift, volatility, dt, num_steps, num_paths);

        assert_eq!(paths.len(), num_paths);
        for path in paths {
            assert_eq!(path.len(), num_steps + 1);
            assert_eq!(path[0], initial_value);
            for value in path.iter().skip(1) {
                assert!(*value > 0.0);
            }
        }
    }
}
