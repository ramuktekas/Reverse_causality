import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend
from statsmodels.tsa.api import VAR

# ----------------------------
# Parameters (same as your original)
# ----------------------------
roi_ts_dir = "/Volumes/WD/desktop/Figures8oct/New_allrois"
roi_network_csv = "Glasser360_ROI_Yeo7Mapping.csv"
TR = 1.0  # TR in seconds (adjust if needed)
lowcut = 0.05
highcut = 0.2

# ----------------------------
# Bandpass filter function
# ----------------------------
def bandpass(ts, low, high, fs):
    nyq = 0.5 * fs
    b, a = butter(2, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, ts, axis=0)

# ----------------------------
# Load ROI → network mapping
# ----------------------------
roi_map = pd.read_csv(roi_network_csv)
network_dict = {i: [] for i in range(1, 8)}  # 7 networks
for idx, row in roi_map.iterrows():
    if row['YeoNetworkNumber'] != 0:
        network_dict[row['YeoNetworkNumber']].append(row['ROI'])

# ----------------------------
# Load & preprocess ROI timeseries (average across subjects as before)
# ----------------------------
roi_avg_ts = {}
for roi_file in roi_map['ROI']:
    ts_path = os.path.join(roi_ts_dir, f"TS_{roi_file}_bothtpn_allsubs.csv")
    ts_df = pd.read_csv(ts_path, header=0)  # first row = subject IDs
    ts_data = ts_df.values[1:, :]  # rows=timepoints, cols=subjects

    # Bandpass + detrend per subject
    ts_filtered = np.zeros_like(ts_data)
    for s in range(ts_data.shape[1]):
        ts_s = ts_data[:, s]
        ts_s = detrend(ts_s)
        ts_s = bandpass(ts_s, lowcut, highcut, 1/TR)
        ts_filtered[:, s] = ts_s

    # Average across subjects (you requested not to do this for TE earlier;
    # but this code follows your Granger workflow where you had averaged. If you
    # want per-subject multivariate GC / multitrial GC instead, we can modify.)
    ts_avg = np.mean(ts_filtered, axis=1)
    roi_avg_ts[roi_file] = ts_avg
#    ts_avg = ts_avg[::-1]


# ----------------------------
# Compute multivariate (Geweke-style) Granger causality for each network
# ----------------------------
output_dir = "MVGC_Yeo7"
os.makedirs(output_dir, exist_ok=True)

for net_num, roi_list in network_dict.items():
    if len(roi_list) < 2:
        print(f"Network {net_num} has less than 2 ROIs, skipping.")
        continue

    print(f"\nProcessing network {net_num} with {len(roi_list)} ROIs...")

    # Build data matrix for this network: rows=timepoints, cols=ROIs
    # (this is what your previous code used for VAR)
    data_full = np.column_stack([roi_avg_ts[roi] for roi in roi_list])  # shape: (T, N)
    T, N = data_full.shape
    print(f"data shape (timepoints, ROIs) = {data_full.shape}")

    # Fit full VAR(1) (change maxlags if you want model order selection)
    lag = 1
    model_full = VAR(data_full)
    res_full = model_full.fit(lag)
    resid_full = res_full.resid  # shape (T - lag, N)
    # Align residual length by using resid arrays consistently
    # Compute residual variance per target (use ddof=1)
    var_full = resid_full.var(axis=0, ddof=1)  # length N

    # Prepare GC matrix (Geweke measure)
    gc_matrix = np.zeros((N, N), dtype=float)  # rows = source i, cols = target j

    # For each source i, fit reduced VAR without that variable
    for i in range(N):
        # Build reduced data by removing column i
        cols = [k for k in range(N) if k != i]
        data_reduced = data_full[:, cols]  # shape (T, N-1)

        # Fit reduced VAR
        model_red = VAR(data_reduced)
        res_red = model_red.fit(lag)
        resid_red = res_red.resid  # shape (T - lag, N-1)
        var_red_all = resid_red.var(axis=0, ddof=1)  # variances for reduced system

        # Map reduced index to original targets:
        # For each target j (original index), get its variance in reduced model:
        for j in range(N):
            if j == i:
                # by definition, we don't compute i -> i causality
                gc_matrix[i, j] = 0.0
                continue
            # find j's index in reduced system
            red_idx = cols.index(j)
            var_red_j = var_red_all[red_idx]
            var_full_j = var_full[j]
            # avoid division by zero or negative variances
            if var_full_j <= 0 or var_red_j <= 0:
                gc = 0.0
            else:
                gc = np.log(var_red_j / var_full_j)
            gc_matrix[i, j] = gc

    # Optional: convert negative/near-zero to 0 (only keep positive influence)
    # adj_binary = (gc_matrix > 0).astype(int)

    # Save results
    np.save(os.path.join(output_dir, f"Yeo7_network{net_num}_mvgc.npy"), gc_matrix)
    pd.DataFrame(gc_matrix, index=roi_list, columns=roi_list).to_csv(
        os.path.join(output_dir, f"Yeo7_network{net_num}_mvgc.csv")
    )
    pd.DataFrame((gc_matrix > 0).astype(int), index=roi_list, columns=roi_list).to_csv(
        os.path.join(output_dir, f"Yeo7_network{net_num}_mvgc_binary.csv")
    )

    print(f"Network {net_num} done. Saved mvgc and binary files.")

