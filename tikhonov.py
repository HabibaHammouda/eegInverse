import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os


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
Noise = load_excel_numeric("Noise.xlsx")   
print("Shapes check after cleaning:")
print("G:", G.shape)
print("S_true:", S_true.shape)
print("Y_measured:", Y_measured.shape)
print("Noise:", Noise.shape)
Y_noise_free = G @ S_true
Y_noisy = Y_noise_free + Noise

# =========================
# 3. Define Tikhonov (Ridge) Inverse Function
# =========================
def tikhonov_inverse(G, Y, lam=0.1):
    """Compute Tikhonov (ridge) inverse estimate."""
    n_sources = G.shape[1]
    return np.linalg.inv(G.T @ G + lam * np.eye(n_sources)) @ G.T @ Y

# =========================
# 4. Apply Tikhonov Inverse
# =========================
lambda_value = 0.1  # you can tune this later
S_hat_nf = tikhonov_inverse(G, Y_noise_free, lam=lambda_value)
S_hat_noisy = tikhonov_inverse(G, Y_noisy, lam=lambda_value)

# =========================
# 5. Accuracy Metrics
# =========================
def compute_metrics(S_true, S_hat):
    rows = []
    for i in range(S_true.shape[0]):
        s_true, s_hat = S_true[i], S_hat[i]
        mse = mean_squared_error(s_true, s_hat)
        rmse = np.sqrt(mse)
        nrmse = rmse / (np.std(s_true) + 1e-12)
        r, _ = pearsonr(s_true, s_hat)
        rows.append({
            "Source": i + 1,
            "MSE": mse,
            "RMSE": rmse,
            "NRMSE": nrmse,
            "Correlation": r
        })
    return pd.DataFrame(rows)

metrics_nf = compute_metrics(S_true, S_hat_nf)
metrics_noisy = compute_metrics(S_true, S_hat_noisy)

# =========================
# 6. Save Outputs
# =========================
os.makedirs("Results_Tikhonov", exist_ok=True)

np.savetxt("Results_Tikhonov/S_hat_noise_free.csv", S_hat_nf, delimiter=",")
np.savetxt("Results_Tikhonov/S_hat_noisy.csv", S_hat_noisy, delimiter=",")
metrics_nf.to_excel("Results_Tikhonov/Metrics_NoiseFree.xlsx", index=False)
metrics_noisy.to_excel("Results_Tikhonov/Metrics_Noisy.xlsx", index=False)

# =========================
# 7. Visualization
# =========================
selected_sources = [0, 1, 2, 3]
for s in selected_sources:
    plt.figure(figsize=(12, 4))

    # --- Left: noise-free reconstruction ---
    plt.subplot(1, 2, 1)
    plt.plot(S_true[s], label="True Source", color="black")
    plt.plot(S_hat_nf[s], label="Tikhonov (No Noise)", color="blue")
    plt.title(f"Source {s + 1} - No Noise")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()

    # --- Right: noisy reconstruction ---
    plt.subplot(1, 2, 2)
    plt.plot(S_true[s], label="True Source", color="black")
    plt.plot(S_hat_noisy[s], label="Tikhonov (With Noise)", color="red", linestyle="--")
    plt.title(f"Source {s + 1} - With Noise")
    plt.xlabel("Time (samples)")
    plt.legend()

    plt.suptitle(f"Tikhonov Reconstruction Comparison (λ={lambda_value})")
    plt.tight_layout()
    plt.savefig(f"Results_Tikhonov/Source_{s + 1}_SideBySide.png", dpi=150)
    plt.show()

# =========================
# 8. Summary
# =========================
print("\n=== Tikhonov Reconstruction Summary ===")
print("λ =", lambda_value)
print("Average correlation (no noise):", metrics_nf["Correlation"].mean())
print("Average correlation (with noise):", metrics_noisy["Correlation"].mean())
print("Results saved in 'Results_Tikhonov/' folder.")