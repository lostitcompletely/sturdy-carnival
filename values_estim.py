import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell, norm, gamma

# Load saved results
RESULTS_FILE = "combined_piece_values.pkl"

with open(RESULTS_FILE, "rb") as f:
    all_data = pickle.load(f)

LABELS = {
    1: "Pawn",
    2: "Knight",
    3: "Bishop",
    4: "Rook",
    5: "Queen",
}

# Plotting function
def plot_piece_distribution(piece_type, data):
    data = np.array(data)
    data = data[np.isfinite(data)]
    # Remove non-positive values
    data = data[data > 0]

    # remove extreme outliers for stability
    data = np.clip(data, 0, np.percentile(data, 99))
    
    if len(data) < 5:
        print(f"Skipping {LABELS[piece_type]} — too few samples.")
        return

    # Fit a Maxwell–Boltzmann distribution
    params = maxwell.fit(data, floc=0)
    g_params = gamma.fit(data, floc=0)
    x = np.linspace(0, np.max(data), 200)
    pdf = maxwell.pdf(x, *params)

    # Also overlay a normal fit for comparison
    mu, sigma = np.mean(data), np.std(data)
    gamma_pdf = gamma.pdf(x, *g_params)

    # plotting histogram and fits
    plt.figure(figsize=(7,4))
    plt.hist(data, bins=20, density=True, alpha=0.4, label="Empirical data")
    plt.plot(x, pdf, "r-", label="Maxwell fit")
    plt.plot(x, gamma_pdf, "g--", label="Gamma fit")
    plt.title(f"{LABELS[piece_type]} Value Distribution")
    plt.xlabel("Value (in pawns)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plot all pieces 
for pt, values in all_data.items():
    if len(values) == 0:
        continue
    plot_piece_distribution(pt, values)
