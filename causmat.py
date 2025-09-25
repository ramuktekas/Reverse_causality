import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend
from statsmodels.tsa.api import VAR

# ----------------------------
# Parameters
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
network_dict = {i: [] for i in range(1,8)}  # 7 networks

for idx, row in roi_map.iterrows():
    if row['YeoNetworkNumber'] != 0:
        network_dict[row['YeoNetworkNumber']].append(row['ROI'])

# ----------------------------
# Load & preprocess ROI timeseries
# ----------------------------
roi_avg_ts = {}  # store 1D timeseries per ROI

for roi_file in roi_map['ROI']:
    ts_path = os.path.join(roi_ts_dir, f"TS_{roi_file}_bothtpn_allsubs.csv")
    ts_df = pd.read_csv(ts_path, header=0)  # first row = subject IDs
    ts_data = ts_df.values[1:, :]  # exclude header row, rows=timepoints, cols=subjects
#    ts_data = ts_data[::-1, :]

    # Bandpass + detrend per subject
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
# Construct network VAR(1) and Granger matrices
# ----------------------------
granger_matrices = {}

for net_num, roi_list in network_dict.items():
    if len(roi_list) < 2:
        print(f"Network {net_num} has less than 2 ROIs, skipping VAR.")
        continue

    # Stack timeseries: rows=timepoints, cols=ROIs
    ts_matrix = np.column_stack([roi_avg_ts[roi] for roi in roi_list])

    # Fit VAR(1)
    model = VAR(ts_matrix)
    var_res = model.fit(1)

    # Compute pairwise Granger causality F-statistics
    # Using .test_causality from statsmodels for each pair
    n = ts_matrix.shape[1]
    gc_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                test = var_res.test_causality(causing=i, caused=j, kind='f')
                gc_matrix[i, j] = test.test_statistic

    granger_matrices[net_num] = {
        "roi_list": roi_list,
        "gc_matrix": gc_matrix
    }
    print(f"Network {net_num} ({len(roi_list)} ROIs) done.")

# ----------------------------
# Save Granger matrices
# ----------------------------
output_dir = "Granger_Yeo7"
os.makedirs(output_dir, exist_ok=True)

for net_num, data in granger_matrices.items():
    gc_mat = data['gc_matrix']
    roi_names = data['roi_list']
    np.save(os.path.join(output_dir, f"Yeo7_network{net_num}_gc_matrix.npy"), gc_mat)
    pd.DataFrame(gc_mat, index=roi_names, columns=roi_names).to_csv(
        os.path.join(output_dir, f"Yeo7_network{net_num}_gc_matrix_rev.csv")
    )

print("All 7 networks processed. Granger matrices saved.")
