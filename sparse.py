import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os

# =========================
# 1. Load Data (Force numeric + skip headers/indices)
# =========================
def load_excel_numeric(filename):
    """Read Excel file, skip first row/column, and force numeric."""
    df = pd.read_excel(filename, header=None)
    # Drop first row and first column if they are headers or indices
    df = df.iloc[1:, 1:]
    df = df.apply(pd.to_numeric, errors='coerce')  # convert all to float
    return df.fillna(0).values  # replace NaN with 0

S_true = load_excel_numeric("Source.xlsx")       # 16 × 1000
G = load_excel_numeric("Leadfield.xlsx")         # 32 × 16
Y_measured = load_excel_numeric("EEG Data.xlsx") # 32 × 1000
Noise = load_excel_numeric("Noise.xlsx")         # 32 × 1000

print("Shapes check after cleaning:")
print("G:", G.shape)
print("S_true:", S_true.shape)
print("Y_measured:", Y_measured.shape)
print("Noise:", Noise.shape)

# =========================
# 2. Define Data Conditions
# =========================
Y_noise_free = G @ S_true           # No noise
Y_noisy = Y_noise_free + Noise      # With added noise

# =========================
# 3. Sparse Inverse Solution (L1 / LASSO)
# =========================
lasso_nf = MultiTaskLassoCV(cv=5, max_iter=5000, n_jobs=-1, verbose=1)
lasso_nf.fit(G, Y_noise_free)
S_hat_nf = lasso_nf.coef_.T   # (16 × 1000)

lasso_noisy = MultiTaskLassoCV(cv=5, max_iter=5000, n_jobs=-1, verbose=1)
lasso_noisy.fit(G, Y_noisy)
S_hat_noisy = lasso_noisy.coef_.T

print("\nOptimal α (noise-free):", lasso_nf.alpha_)
print("Optimal α (noisy):", lasso_noisy.alpha_)

# =========================
# 4. Accuracy Metrics
# =========================
def compute_metrics(S_true, S_hat):
    rows = []
    for i in range(S_true.shape[0]):
        s_true, s_hat = S_true[i], S_hat[i]
        mse = mean_squared_error(s_true, s_hat)
        rmse = np.sqrt(mse)
        nrmse = rmse / (np.std(s_true) + 1e-12)
        r, _ = pearsonr(s_true, s_hat)
        rows.append({"Source": i+1, "MSE": mse, "RMSE": rmse, "NRMSE": nrmse, "Correlation": r})
    return pd.DataFrame(rows)

metrics_nf = compute_metrics(S_true, S_hat_nf)
metrics_noisy = compute_metrics(S_true, S_hat_noisy)

# =========================
# 5. Save Outputs
# =========================
os.makedirs("Results", exist_ok=True)

np.savetxt("Results/S_hat_noise_free.csv", S_hat_nf, delimiter=",")
np.savetxt("Results/S_hat_noisy.csv", S_hat_noisy, delimiter=",")
metrics_nf.to_excel("Results/Metrics_NoiseFree.xlsx", index=False)
metrics_noisy.to_excel("Results/Metrics_Noisy.xlsx", index=False)

# =========================
# 6. Visualization
# =========================
selected_sources = [0, 1, 2, 3]  # You can change this list
for s in selected_sources:
    plt.figure(figsize=(10,3))
    plt.plot(S_true[s], label="True Source")
    plt.plot(S_hat_nf[s], label="Estimated (No Noise)")
    plt.plot(S_hat_noisy[s], label="Estimated (With Noise)", linestyle='--')
    plt.title(f"Source {s+1} Reconstruction")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Results/Source_{s+1}_Comparison.png", dpi=150)
    plt.show()

# =========================
# 7. Summary
# =========================
print("\n=== Reconstruction Summary ===")
print("Average correlation (no noise):", metrics_nf["Correlation"].mean())
print("Average correlation (with noise):", metrics_noisy["Correlation"].mean())
print("Results saved in 'Results/' folder.")
