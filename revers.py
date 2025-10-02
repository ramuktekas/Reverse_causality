import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ----------------------------
# Directories
# ----------------------------
forward_dir = "MV_Yeo7_acw"
reverse_dir = "MV_Yeo7_acw"

# ----------------------------
# Yeo network names
# ----------------------------
yeo_names = {
    1: "VIS",
    2: "SOM",
    3: "DA",
    4: "VA",
    5: "LIM",
    6: "CON",
    7: "DMN"
}

# ----------------------------
# Functions for metrics
# ----------------------------
def frob_ratio(num, den):
    num_norm = np.linalg.norm(num, 'fro')
    den_norm = np.linalg.norm(den, 'fro')
    if den_norm == 0:
        return np.nan
    return num_norm / den_norm

def compute_metrics(A, B):
    AT = A.T
    BT = B.T
    dsym = frob_ratio(A - AT, A + AT)
    dnorm = frob_ratio(A @ AT - AT @ A, A @ AT + AT @ A)
    dcsym = frob_ratio(A - B, A + B)
    dcnorm = frob_ratio(A - BT, A + BT)
    return dsym, dnorm, dcsym, dcnorm

def rescale(matrix, s=0.8):
    """
    Rescale a square matrix by s / lambda_max to ensure spectral radius <= s.
    
    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to rescale.
    s : float
        Target spectral radius (0 < s < 1). Default is 0.8.
    
    Returns
    -------
    np.ndarray
        Rescaled matrix.
    """
    eigvals = np.linalg.eigvals(matrix)
    lam_max = np.max(np.abs(eigvals))
    if lam_max == 0:
        return matrix.copy()  # nothing to rescale
    factor = s / lam_max
    return matrix * factor

# ----------------------------
# Collect metrics for all 7 networks
# ----------------------------
metrics = []
for net_num in range(1, 8):
    fpath = os.path.join(forward_dir, f"mv_yeo7_network{net_num}.npy")
    rpath = os.path.join(reverse_dir, f"mv_yeo7_network{net_num}_rev.npy")
    if not (os.path.exists(fpath) and os.path.exists(rpath)):
        print(f"Missing network {net_num}, skipping...")
        continue

    A = np.load(fpath)
    A = rescale(A)
    B = np.load(rpath)
    B = rescale(B)
    
    dsym, dnorm, dcsym, dcnorm = compute_metrics(A, B)
    metrics.append({
        "network": net_num,
        "dsym": dsym,
        "dnorm": dnorm,
        "dcsym": dcsym,
        "dcnorm": dcnorm
    })

df = pd.DataFrame(metrics)
print(df)

# ----------------------------
# Plot dcnorm vs dnorm (all points, one regression)
# ----------------------------
plt.figure(figsize=(10,8))

x = df["dnorm"].values
y = df["dcnorm"].values

# scatter points with network names
for i, row in df.iterrows():
    name = yeo_names.get(row["network"], f"Network {row['network']}")
    plt.scatter(row["dnorm"], row["dcnorm"], label=name, s=100)

# linear regression across all points
slope, intercept, r_value, p_value, std_err = linregress(x, y)
x_fit = np.linspace(min(x)*0.9, max(x)*1.1, 100)
y_fit = intercept + slope * x_fit
plt.plot(x_fit, y_fit, color='black', linestyle='--', label=f"Fit: $R^2$={r_value**2:.2f}, p={p_value:.3f}")

plt.xlabel("Deviation from normality (dnorm{A})", fontweight='bold')
plt.ylabel("Non-reversibilty under time-reversal (dcnorm{A})", fontweight='bold')
plt.title("Causal structure reversal under time-reversal of the 7 Yeo networks (Multivariate Granger sense)- Movie watching dACW", fontweight='bold',fontsize=9)
plt.legend(fontsize=9)
plt.grid(False)
plt.tight_layout()
plt.savefig("acwmv_yeo7_reversal_Movie2.png",dpi=600)

plt.show()

# ----------------------------
# Plot dcsym vs dsym (all points, one regression)
# ----------------------------
plt.figure(figsize=(10,8))

x = df["dsym"].values
y = df["dcsym"].values

for i, row in df.iterrows():
    name = yeo_names.get(row["network"], f"Network {row['network']}")
    plt.scatter(row["dsym"], row["dcsym"], label=name, s=100)

slope, intercept, r_value, p_value, std_err = linregress(x, y)
x_fit = np.linspace(min(x)*0.9, max(x)*1.1, 100)
y_fit = intercept + slope * x_fit
plt.plot(x_fit, y_fit, color='black', linestyle='--', label=f"Fit: $R^2$={r_value**2:.2f}, p={p_value:.3f}")

plt.xlabel("Deviation from symmetry (dsym{A})", fontweight='bold')
plt.ylabel("Non-conservablity under time-reversal (dcsym{A})", fontweight='bold')
plt.title("Causal structure conservation under time-reversal of the 7 Yeo networks (Multivariate Granger sense)-Movie watching dACW", fontweight='bold', fontsize=9)
plt.legend(fontsize=9)
plt.grid(False)
plt.tight_layout()
plt.savefig("acwmv_yeo7_conservation_Movie2.png",dpi=600)
plt.show()
