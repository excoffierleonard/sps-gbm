use rand::thread_rng;
use rand_distr::{Distribution, Normal};

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
pub fn gbm_step(current_value: f64, drift: f64, volatility: f64, dt: f64, z: f64) -> f64 {
    let drift_term = (drift - 0.5 * volatility * volatility) * dt;
    let diffusion_term = volatility * dt.sqrt() * z;

    current_value * (drift_term + diffusion_term).exp()
}

pub fn simulate_gbm_path(
    initial_value: f64,
    drift: f64,
    volatility: f64,
    dt: f64,
    num_steps: usize,
) -> Vec<f64> {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    
    // Pregenerate all random z values
    let z_values: Vec<f64> = (0..num_steps)
        .map(|_| normal.sample(&mut rng))
        .collect();

    let mut path = Vec::with_capacity(num_steps + 1);
    path.push(initial_value);

    for i in 0..num_steps {
        let next_value = gbm_step(path.last().copied().unwrap(), drift, volatility, dt, z_values[i]);
        path.push(next_value);
    }

    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbm_step_formula() {
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
            assert!((next_value - tc.expected).abs() < 1e-2);
        }
    }

    #[test]
    fn test_simulate_gbm_path() {
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
}
