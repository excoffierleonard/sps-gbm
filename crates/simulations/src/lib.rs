use std::ops::Deref;

use rand::rng;
use rand_distr::{Distribution as RandDistribution, StandardNormal};
use rayon::prelude::*;

/// A simulated path of values
pub struct SimulatedPath {
    /// The simulated path of values
    path: Vec<f64>,
}

impl Deref for SimulatedPath {
    type Target = Vec<f64>;
    fn deref(&self) -> &Self::Target {
        &self.path
    }
}

impl From<Vec<f64>> for SimulatedPath {
    fn from(path: Vec<f64>) -> Self {
        Self { path }
    }
}

impl From<SimulatedPath> for Vec<f64> {
    fn from(sim_path: SimulatedPath) -> Self {
        sim_path.path
    }
}

/// A collection of simulated paths
pub struct SimulatedPaths {
    /// The simulated paths of values
    paths: Vec<SimulatedPath>,
}

impl Deref for SimulatedPaths {
    type Target = Vec<SimulatedPath>;
    fn deref(&self) -> &Self::Target {
        &self.paths
    }
}

impl From<Vec<SimulatedPath>> for SimulatedPaths {
    fn from(paths: Vec<SimulatedPath>) -> Self {
        Self { paths }
    }
}

impl From<SimulatedPaths> for Vec<SimulatedPath> {
    fn from(sim_paths: SimulatedPaths) -> Self {
        sim_paths.paths
    }
}

impl SimulatedPaths {
    /// Converts the simulated paths into a vector of vectors
    pub fn into_vec_of_vec(self) -> Vec<Vec<f64>> {
        self.paths.into_iter().map(Into::into).collect()
    }
}

/// A simulator for geometric Brownian motion (GBM)
pub struct GbmSimulator {
    /// The initial value S(0)
    initial_value: f64,
    /// The drift parameter μ
    drift: f64,
    /// The volatility parameter σ
    volatility: f64,
    /// The time step Δt
    dt: f64,
}

impl GbmSimulator {
    /// Creates a new GBM simulator with the given parameters
    ///
    /// # Arguments
    ///
    /// * `initial_value` - The initial value S(0)
    /// * `drift` - The drift parameter μ
    /// * `volatility` - The volatility parameter σ
    /// * `dt` - The time step Δt
    pub fn new(initial_value: f64, drift: f64, volatility: f64, dt: f64) -> Self {
        Self {
            initial_value,
            drift,
            volatility,
            dt,
        }
    }

    /// Calculates a single step of geometric Brownian motion
    ///
    /// # Arguments
    ///
    /// * `current_value` - The current value S(t)
    /// * `z` - Standard normal random variable (N(0,1))
    ///
    /// # Returns
    ///
    /// The next value S(t+Δt)
    fn gbm_step(&self, current_value: f64, z: f64) -> f64 {
        let drift_term = (self.drift - 0.5 * self.volatility * self.volatility) * self.dt;
        let diffusion_term = self.volatility * self.dt.sqrt() * z;

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
    /// * `num_steps` - The number of steps to simulate
    ///
    /// # Returns
    ///
    /// A vector containing the simulated path of values
    fn simulate_path(&self, num_steps: usize) -> Vec<f64> {
        // Pregenerate all random z values
        let z_values = Self::generate_random_normal_zs(num_steps);

        // Iterate through the z values to calculate the path
        let mut path = Vec::with_capacity(num_steps + 1);
        path.push(self.initial_value);
        let mut current_value = self.initial_value;
        for &z in &z_values {
            let next_value = self.gbm_step(current_value, z);
            path.push(next_value);
            current_value = next_value;
        }
        path
    }

    /// Simulates multiple paths of geometric Brownian motion
    ///
    /// # Arguments
    ///
    /// * `num_steps` - The number of steps to simulate
    /// * `num_paths` - The number of paths to simulate
    ///
    /// # Returns
    ///
    /// A collection of simulated paths
    pub fn simulate_paths(&self, num_steps: usize, num_paths: usize) -> SimulatedPaths {
        let paths: Vec<SimulatedPath> = (0..num_paths)
            .into_par_iter()
            .map(|_| {
                let simulated_path = self.simulate_path(num_steps);
                SimulatedPath::from(simulated_path)
            })
            .collect();
        SimulatedPaths::from(paths)
    }
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
            let simulator = GbmSimulator::new(tc.current_value, tc.drift, tc.volatility, tc.dt);
            let next_value = simulator.gbm_step(tc.current_value, tc.z);
            assert_eq!(next_value, tc.expected);
        }
    }

    #[test]
    fn generate_random_normal_zs_correct() {
        let num_steps = 1000;
        let zs = GbmSimulator::generate_random_normal_zs(num_steps);

        assert_eq!(zs.len(), num_steps);
        for &z in &zs {
            assert!(z.is_finite());
        }
    }

    #[test]
    fn simulate_path_correct() {
        let initial_value = 100.0;
        let drift = 0.05;
        let volatility = 0.2;
        let dt = 1.0;
        let num_steps = 10;

        let simulator = GbmSimulator::new(initial_value, drift, volatility, dt);
        let path = simulator.simulate_path(num_steps);

        assert_eq!(path.len(), num_steps + 1);
        assert_eq!(path[0], initial_value);

        for i in 1..path.len() {
            assert!(path[i] > 0.0);
        }
    }

    #[test]
    fn simulate_paths_correct() {
        let initial_value = 100.0;
        let drift = 0.05;
        let volatility = 0.2;
        let dt = 1.0;
        let num_steps = 10;
        let num_paths = 5;

        let simulator = GbmSimulator::new(initial_value, drift, volatility, dt);
        let paths = simulator.simulate_paths(num_steps, num_paths);

        assert_eq!(paths.len(), num_paths);
        for path in paths.iter() {
            assert_eq!(path.len(), num_steps + 1);
            assert_eq!(path[0], initial_value);
            for value in path.iter().skip(1) {
                assert!(*value > 0.0);
            }
        }
    }
}
