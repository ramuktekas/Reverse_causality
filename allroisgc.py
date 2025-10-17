import os
import numpy as np
import pandas as pd
from scipy.signal import detrend, butter, filtfilt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf

# ----------------------------
# Parameters
# ----------------------------

TR = 1.0
lowcut = 0.05
highcut = 0.2
lag = 1  # VAR(1)
acw = False
sw = False
if acw:
    roi_ts_dir = "/Volumes/WD/desktop/Figures8oct/New_acwts"
else:
    roi_ts_dir = "/Volumes/WD/desktop/Figures8oct/New_allrois"
roi_network_csv = "Glasser360_ROI_Yeo7Mapping.csv"
# ----------------------------
# Bandpass function
# ----------------------------
def bandpass(ts, low, high, fs):
    nyq = 0.5 * fs
    b, a = butter(2, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, ts, axis=0)
# ----------------------------
# get acw timeseries
# ----------------------------
def calculate_acw(ts):
    sampling_rate = 1.0
    acw_func = acf(ts, nlags=len(ts)-1, qstat=False, alpha=None, fft=True)
    acw_0 = np.argmax(acw_func <= 0) / sampling_rate
    return acw_0
def get_acw_time_series(time_series, window_size=600, step_size=10):
    acw_series = []
    for i in range(0, len(time_series) - window_size + 1, step_size):
        window = time_series[i:i+window_size]
        acw_0 = calculate_acw(window)
        acw_series.append(acw_0)
    return np.array(acw_series)
# ----------------------------
# Load ROI timeseries
# ----------------------------
roi_map = pd.read_csv(roi_network_csv)
roi_ts_subjects = {}
for roi_file in roi_map['ROI']:
    if acw:
        ts_path = os.path.join(roi_ts_dir, f"TS_{roi_file}_bothacw_allsubs.csv")
    else:
        ts_path = os.path.join(roi_ts_dir, f"TS_{roi_file}_bothtpn_allsubs.csv")
    ts_df = pd.read_csv(ts_path, header=0)
    ts_data = ts_df.values[1:, :]  # rows=time, cols=subjects
    ts_filtered = np.zeros_like(ts_data)
    for s in range(ts_data.shape[1]):
        ts_s = ts_data[:, s]
        ts_s = detrend(ts_s)
        ts_s = bandpass(ts_s, lowcut, highcut, 1/TR)
        # Step 1: zero-mean
        # ts_s = ts_s - np.mean(ts_s)

        # # Step 2: scale to fit within [-1, 1]
        # max_abs = np.max(np.abs(ts_s))
        # if max_abs != 0:
        #     ts_s = ts_s / max_abs
        ts_filtered[:, s] = ts_s
    roi_ts_subjects[roi_file] = ts_filtered

# ----------------------------
# Load input/source time series
# ----------------------------
#input_ts = pd.read_csv("/Volumes/WD/desktop/Figures8oct/Sentence_Similarity_7T_MOVIE2_HO1.csv")['Cosine_Similarity']
## Trim edges
#input_ts = input_ts.iloc[200:-200].dropna().values




# ----------------------------
# Functions for metrics
# ----------------------------
def frob_ratio(num, den):
    num_norm = np.linalg.norm(num, 'fro')
    den_norm = np.linalg.norm(den, 'fro')
    return num_norm / den_norm if den_norm != 0 else np.nan

def compute_metrics(A, B):
    AT = A.T
    BT = B.T
    dsym = frob_ratio(A - AT, A + AT)
    dnorm = frob_ratio(A @ AT - AT @ A, A @ AT + AT @ A)
    dcsym = frob_ratio(A - B, A + B)
    dcnorm = frob_ratio(A - BT, A + BT)
    return dsym, dnorm, dcsym, dcnorm

def rescale(matrix, s=0.8):
    eigvals = np.linalg.eigvals(matrix)
    lam_max = np.max(np.abs(eigvals))
    return matrix * (s / lam_max) if lam_max != 0 else matrix.copy()

import matplotlib.pyplot as plt
from scipy.stats import linregress

# ----------------------------
# Network mapping for plotting
# ----------------------------
roi_map = pd.read_csv(roi_network_csv)
roi_to_net = dict(zip(roi_map['ROI'], roi_map['YeoNetworkNumber']))

yeo_names = {1: "VIS", 2: "SOM", 3: "DA", 4: "VA", 5: "LIM", 6: "CON", 7: "DMN"}
yeo_colors = {
    1: "#E41A1C",  # red
    2: "#377EB8",  # blue
    3: "#4DAF4A",  # green
    4: "#FF7F00",  # orange
    5: "#984EA3",  # purple
    6: "#A65628",  # brown
    7: "#F781BF"   # pink/coral
}

# ----------------------------
# Function to compute metrics given an input_ts
# ----------------------------
def compute_gc_for_input(input_ts, filename_csv, gcwindow=100, gcstep=1):
    # preprocess input
    input_ts = input_ts.iloc[200:-200].dropna().values
    input_ts = np.diff(input_ts, axis=0)
    # input_ts = (input_ts - input_ts.min()) / (input_ts.max() - input_ts.min())
    # Step 1: zero-mean
    # input_ts = input_ts - np.mean(input_ts)
    

    # # Step 2: scale to fit within [-1, 1]
    # max_abs = np.max(np.abs(input_ts))
    # if max_abs != 0:
    #     input_ts = input_ts / max_abs

    if not acw:
        # Downsample: average every 10 samples
        n = 10
        trim_len = len(input_ts) - (len(input_ts) % n)
        input_ts = input_ts[:trim_len].reshape(-1, n).mean(axis=1)
    else:
        input_ts = get_acw_time_series(input_ts, 600, 10)
        input_ts = detrend(input_ts)
        input_ts = bandpass(ts_s, lowcut, highcut, 1/TR)

    results = []

    for roi_file, ts_data in roi_ts_subjects.items():
        if sw:
            ts_avg = np.mean(ts_data, axis=1)
            min_len = min(len(ts_avg), len(input_ts))
            ts_roi = ts_avg[:min_len]
            ts_in = input_ts[:min_len]

            # collect metrics over sliding windows
            dsym_vals, dnorm_vals, dcsym_vals, dcnorm_vals = [], [], [], []

            for start in range(0, min_len - gcwindow + 1, gcstep):
                roi_win = ts_roi[start:start+gcwindow]
                in_win  = ts_in[start:start+gcwindow]

                ts_fwd = np.column_stack([roi_win, in_win])
                ts_rev = ts_fwd[::-1, :]

                try:
                    var_fwd = VAR(ts_fwd).fit(lag)
                    var_rev = VAR(ts_rev).fit(lag)
                except Exception:
                    continue  # skip if VAR fails

                A_fwd = rescale(var_fwd.params[1:].T)
                A_rev = rescale(var_rev.params[1:].T)

                dsym, dnorm, dcsym, dcnorm = compute_metrics(A_fwd, A_rev)

                dsym_vals.append(dsym)
                dnorm_vals.append(dnorm)
                dcsym_vals.append(dcsym)
                dcnorm_vals.append(dcnorm)

            # compute mean and std across windows
            results.append({
                "roi": roi_file,
                "network": roi_to_net.get(roi_file, 0),
                "dsym": np.nanmean(dsym_vals),
                "dnorm": np.nanmean(dnorm_vals),
                "dcsym": np.nanmean(dcsym_vals),
                "dcnorm": np.nanmean(dcnorm_vals),
                "dsym_std": np.nanstd(dsym_vals),
                "dnorm_std": np.nanstd(dnorm_vals),
                "dcsym_std": np.nanstd(dcsym_vals),
                "dcnorm_std": np.nanstd(dcnorm_vals),
            })
        else:
            ts_avg = np.mean(ts_data, axis=1)

            min_len = min(len(ts_avg), len(input_ts))
            ts_roi = ts_avg[:min_len]
            ts_in = input_ts[:min_len]

            ts_fwd = np.column_stack([ts_roi, ts_in])
            ts_rev = ts_fwd[::-1, :]

            var_fwd = VAR(ts_fwd).fit(lag)
            var_rev = VAR(ts_rev).fit(lag)

            A_fwd = rescale(var_fwd.params[1:].T)
            A_rev = rescale(var_rev.params[1:].T)

            dsym, dnorm, dcsym, dcnorm = compute_metrics(A_fwd, A_rev)

            results.append({
                "roi": roi_file,
                "network": roi_to_net.get(roi_file, 0),
                "dsym": dsym,
                "dnorm": dnorm,
                "dcsym": dcsym,
                "dcnorm": dcnorm
            })

    df = pd.DataFrame(results)
    df.to_csv(filename_csv, index=False)
    if sw:
        print(f"Saved {filename_csv} with windowed GC metrics (win={gcwindow}, step={gcstep}).")
    else:
        print(f"Saved {filename_csv}")
    return df


# ----------------------------
# Compute for both inputs
# ----------------------------
df_ss = compute_gc_for_input(
    pd.read_csv("/Volumes/WD/desktop/Figures8oct/Sentence_Similarity_7T_MOVIE2_HO1.csv")['Cosine_Similarity'],
    "360_SS_diff.csv"
)
df_wd = compute_gc_for_input(
    pd.read_csv("/Volumes/WD/desktop/Figures8oct/WD_MOVIE2_HOI.csv")['Word_depth'],
    "360_WD_diff.csv"
)

# ----------------------------
# Plotting function
# ----------------------------
def scatter_with_fit(ax, df, x_col, y_col, title, x_label, y_label):
    # scatter by network
    for net in sorted(df['network'].unique()):
        if net == 0:
            continue
        net_df = df[df['network'] == net]
        ax.scatter(net_df[x_col], net_df[y_col],
                   color=yeo_colors[net],
                   alpha=0.6,
                   label=yeo_names[net] if ax == axes[0,0] else "",
                   s=15)
    # regression line
    x = df[x_col].values
    y = df[y_col].values
    slope, intercept, r, p, _ = linregress(x, y)
    x_fit = np.linspace(min(x)*0.9, max(x)*1.1, 100)
    y_fit = intercept + slope*x_fit
    ax.plot(x_fit, y_fit, color="black", linestyle="--")
    
    # custom labels & title
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")


# ----------------------------
# Create 2x2 plots
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

scatter_with_fit(
    axes[0,0], df_ss, "dnorm", "dcnorm",
    "Sentence Similarity: Reversibility",
    "Deviation from normality (dnorm)",
    "Non-reversibilty (dcnorm)"
)
scatter_with_fit(
    axes[0,1], df_ss, "dsym", "dcsym",
    "Sentence Similarity: Conservability",
    "Deviation from symmetry (dsym)",
    "Non-conservability (dcsym)"
)
scatter_with_fit(
    axes[1,0], df_wd, "dnorm", "dcnorm",
    "Word Depth: Reversibility",
    "Deviation from normality (dnorm)",
    "Irreversibilty (dcnorm)"
)
scatter_with_fit(
    axes[1,1], df_wd, "dsym", "dcsym",
    "Word Depth: Conservability",
    "Deviation from symmetry (dsym)",
    "Non-conservability (dcsym)"
)


handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=9)

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("causality_movie2_diff.png", dpi=600)
plt.show()
