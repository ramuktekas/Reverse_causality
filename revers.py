import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Directories
# ----------------------------
forward_dir = "Granger_Yeo7"
reverse_dir = "Granger_Yeo7_rev"

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

# ----------------------------
# Collect metrics for all 7 networks
# ----------------------------
metrics = []
for net_num in range(1, 8):
    fpath = os.path.join(forward_dir, f"Yeo7_network{net_num}_gc_matrix.npy")
    rpath = os.path.join(reverse_dir, f"Yeo7_network{net_num}_gc_matrix_rev.npy")
    if not (os.path.exists(fpath) and os.path.exists(rpath)):
        print(f"Missing network {net_num}, skipping...")
        continue

    A = np.load(fpath)
    B = np.load(rpath)

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
# Define Yeo7 network names
# ----------------------------
yeo7_names = {
    1: "Visual",
    2: "SomMot",
    3: "DorsAttn",
    4: "SalVentAttn",
    5: "Limbic",
    6: "Cont",
    7: "DMN"
}

# ----------------------------
# Plot dcnorm vs dnorm
# ----------------------------
plt.figure(figsize=(7,6))
for _, row in df.iterrows():
    plt.scatter(row["dnorm"], row["dcnorm"],
                label=yeo7_names.get(row["network"], f"Network {row['network']}"),
                s=100)
plt.xlabel("dnorm(A)")
plt.ylabel("dcnorm(A)")
plt.title("dcnorm vs dnorm (Yeo7)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dev_norm_7yeonet.png")

plt.show()

# ----------------------------
# Plot dcsym vs dsym
# ----------------------------
plt.figure(figsize=(7,6))
for _, row in df.iterrows():
    plt.scatter(row["dsym"], row["dcsym"],
                label=yeo7_names.get(row["network"], f"Network {row['network']}"),
                s=100)
plt.xlabel("dsym(A)")
plt.ylabel("dcsym(A)")
plt.title("dcsym vs dsym (Yeo7)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dev_sym_7yeonet.png")

plt.show()
