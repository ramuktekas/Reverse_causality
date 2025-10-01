import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend
from statsmodels.tsa.api import VAR

# ----------------------------
# Parameters
# ----------------------------
roi_ts_dir = "/Volumes/WD/desktop/Figures8oct/New_acwts"
roi_network_csv = "Glasser360_ROI_Yeo7Mapping.csv"
TR = 1.0  # TR in seconds
lowcut = 0.05
highcut = 0.2
lag = 1  # VAR(1)
output_dir_bv = "BV_yeo7_acw"
output_dir_mv = "MV_yeo7_acw"

os.makedirs(output_dir_bv, exist_ok=True)
os.makedirs(output_dir_mv, exist_ok=True)

# ----------------------------
# Bandpass filter function
# ----------------------------
def bandpass(ts, low, high, fs):
    return ts
#    nyq = 0.5 * fs
#    b, a = butter(2, [low/nyq, high/nyq], btype='band')
#    return filtfilt(b, a, ts, axis=0)

# ----------------------------
# Load ROI → network mapping
# ----------------------------
roi_map = pd.read_csv(roi_network_csv)
network_dict = {i: [] for i in range(1, 8)}  # 7 networks
for idx, row in roi_map.iterrows():
    if row['YeoNetworkNumber'] != 0:
        network_dict[row['YeoNetworkNumber']].append(row['ROI'])

# ----------------------------
# Load & preprocess ROI timeseries
# ----------------------------
roi_avg_ts = {}
for roi_file in roi_map['ROI']:
    ts_path = os.path.join(roi_ts_dir, f"TS_{roi_file}_bothacw_allsubs.csv")
    ts_df = pd.read_csv(ts_path, header=0)
    ts_data = ts_df.values[1:, :]  # rows=timepoints, cols=subjects

    # Detrend + bandpass per subject
    ts_filtered = np.zeros_like(ts_data)
    for s in range(ts_data.shape[1]):
        ts_s = ts_data[:, s]
        ts_s = detrend(ts_s)
        ts_s = bandpass(ts_s, lowcut, highcut, 1/TR)
        ts_filtered[:, s] = ts_s

    # Average across subjects
    ts_avg = np.mean(ts_filtered, axis=1)
    roi_avg_ts[roi_file] = ts_avg

# ----------------------------
# Helper to build BV matrix from all pairs
# ----------------------------
def build_bv_matrix(roi_list, roi_avg_ts, reverse=False):
    N = len(roi_list)
    bv_matrix = np.zeros((N, N))
    # optionally reverse time series
    ts_data = {roi: roi_avg_ts[roi][::-1] if reverse else roi_avg_ts[roi] for roi in roi_list}

    for i in range(N-1):
        for j in range(i+1, N):
            pair_data = np.column_stack([ts_data[roi_list[i]], ts_data[roi_list[j]]])
            model = VAR(pair_data)
            res = model.fit(lag)
            A = res.params[1:].T  # shape 2x2, transpose to align rows=target, cols=lagged predictors
            # Insert into bv_matrix
            bv_matrix[i, i] = A[0,0]
            bv_matrix[i, j] = A[0,1]
            bv_matrix[j, i] = A[1,0]
            bv_matrix[j, j] = A[1,1]

    return bv_matrix

# ----------------------------
# Helper to build MV matrix
# ----------------------------
def build_mv_matrix(roi_list, roi_avg_ts, reverse=False):
    N = len(roi_list)
    ts_data = np.column_stack([roi_avg_ts[roi][::-1] if reverse else roi_avg_ts[roi] for roi in roi_list])
    model = VAR(ts_data)
    res = model.fit(lag)
    A = res.params[1:].T  # shape N x N
    return A

# ----------------------------
# Process each network
# ----------------------------
for net_num, roi_list in network_dict.items():
    if len(roi_list) < 2:
        print(f"Network {net_num} has less than 2 ROIs, skipping.")
        continue

    print(f"\nProcessing network {net_num} with {len(roi_list)} ROIs...")

    # ---------------- BV case ----------------
    bv_matrix = build_bv_matrix(roi_list, roi_avg_ts, reverse=False)
    bv_matrix_rev = build_bv_matrix(roi_list, roi_avg_ts, reverse=True)

    np.save(os.path.join(output_dir_bv, f"acwbv_yeo7_network{net_num}.npy"), bv_matrix)
    pd.DataFrame(bv_matrix, index=roi_list, columns=roi_list).to_csv(
        os.path.join(output_dir_bv, f"acwbv_yeo7_network{net_num}.csv")
    )

    np.save(os.path.join(output_dir_bv, f"acwbv_yeo7_network{net_num}_rev.npy"), bv_matrix_rev)
    pd.DataFrame(bv_matrix_rev, index=roi_list, columns=roi_list).to_csv(
        os.path.join(output_dir_bv, f"acwbv_yeo7_network{net_num}_rev.csv")
    )

    # ---------------- MV case ----------------
    mv_matrix = build_mv_matrix(roi_list, roi_avg_ts, reverse=False)
    mv_matrix_rev = build_mv_matrix(roi_list, roi_avg_ts, reverse=True)

    np.save(os.path.join(output_dir_mv, f"acwmv_yeo7_network{net_num}.npy"), mv_matrix)
    pd.DataFrame(mv_matrix, index=roi_list, columns=roi_list).to_csv(
        os.path.join(output_dir_mv, f"acwmv_yeo7_network{net_num}.csv")
    )

    np.save(os.path.join(output_dir_mv, f"acwmv_yeo7_network{net_num}_rev.npy"), mv_matrix_rev)
    pd.DataFrame(mv_matrix_rev, index=roi_list, columns=roi_list).to_csv(
        os.path.join(output_dir_mv, f"acwmv_yeo7_network{net_num}_rev.csv")
    )

    print(f"Network {net_num} done. BV and MV matrices saved.")
