use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

const DAYS_IN_YEAR: f64 = 252.0; // Trading days in a year

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResults {
    pub simulations: Vec<Vec<f64>>,
    pub mean_final_price: f64,
    pub median_final_price: f64,
    pub std_final_price: f64,
}

/// Simulates stock price evolution using Geometric Brownian Motion model
///
/// # Parameters
/// - `s0`: Initial stock price
/// - `mu`: Annual drift (expected return)
/// - `sigma`: Annual volatility
/// - `days`: Number of days to simulate
/// - `num_simulations`: Number of simulation paths to generate
///
/// # Returns
/// A SimulationResults struct containing the simulation paths and statistics
pub fn simulate_gbm(
    s0: f64,
    mu: f64,
    sigma: f64,
    days: usize,
    num_simulations: usize,
) -> SimulationResults {
    let dt = 1.0 / DAYS_IN_YEAR;
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    let mut simulations = vec![vec![s0; days + 1]; num_simulations];
    let mut final_prices = Vec::with_capacity(num_simulations);

    for simulation in simulations.iter_mut().take(num_simulations) {
        for j in 1..=days {
            let z = normal.sample(&mut rng);
            let return_val = (drift + vol * z).exp();
            simulation[j] = simulation[j - 1] * return_val;
        }
        final_prices.push(simulation[days]);
    }

    // Calculate statistics
    final_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_final_price = final_prices.iter().sum::<f64>() / final_prices.len() as f64;
    let median_final_price = if final_prices.len() % 2 == 0 {
        (final_prices[final_prices.len() / 2 - 1] + final_prices[final_prices.len() / 2]) / 2.0
    } else {
        final_prices[final_prices.len() / 2]
    };

    let variance = final_prices
        .iter()
        .map(|&price| (price - mean_final_price).powi(2))
        .sum::<f64>()
        / final_prices.len() as f64;
    let std_final_price = variance.sqrt();

    SimulationResults {
        simulations,
        mean_final_price,
        median_final_price,
        std_final_price,
    }
}