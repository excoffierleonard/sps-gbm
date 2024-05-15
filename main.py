import matplotlib.pyplot as plt
import numpy as np

# Parameters
S0 = 172.02  # Initial stock price
mu = 0.32  # Drift
sigma = 0.3  # Volatility
T = 1.0  # Time period (one year)
N = 252  # Number of steps (e.g., trading days in a year)

# Time increment
dt = T / N

# Generate random steps
np.random.seed()  # For reproducibility
Z = np.random.standard_normal(N)  # Standard normal random variables
# Calculate price path
S = np.zeros(N + 1)
S[0] = S0
for t in range(1, N + 1):
    S[t] = S[t - 1] * np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t - 1]
    )

# Plot the stock price path
plt.plot(S)
plt.title("Stock Price Simulation using GBM")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.grid(True)
plt.show()
