import numpy as np
import pandas as pd
import mienc

if __name__ == "__main__":
    # --- Load CSV ---
    csv_path = "/Volumes/WD/desktop/c255/dataset_SS.csv"
    df = pd.read_csv(csv_path)

    # The dataset shape must be (samples, rois, sessions)
    # Each ROI is one column, timepoints go down the rows
    data_array = df.values[:, :, np.newaxis]  # (877 x 362 x 1)

    # --- Initialize NonLinearEstimator ---
    nle = mienc.NonLinearEstimator(
        config_file="/Volumes/WD/desktop/c255/config_SS.ini",
        estimator="bin",
        parameter=0,
        surrogates=99,
        cache="/Volumes/WD/desktop/c255/cache_dir",
        save_out="in_memory",  # <-- FIXED: must be "in_memory" (underscore!)
        verbose=True,
        EC=0,
        random_state=42,
        workers=4,
    )

    # --- Run estimation ---
    stats = nle.estimate(
        data_array,
        display=False,
        extended_stats=True,
        compute_shadow=True,
        retrieve=True,
        steps=200,
        iterations=50000,
        samples=0,
        # workers=4,
    )
    # --- Extract results safely ---
    if isinstance(stats, dict):
        # Single-session: MI matrix is stored under key 0
        total_mi = stats.get(0, None)
        # Extract the MI data
        total_mi = stats.get(0, None)

        # Inspect
        print("Type:", type(total_mi))
        if isinstance(total_mi, np.ndarray):
            print("Shape:", total_mi.shape)
            total_mi = total_mi.squeeze()
            pd.DataFrame(total_mi).to_csv("/Volumes/WD/desktop/c255/totalMI.csv", index=False)
            print("✅ Saved totalMI.csv (shape:", total_mi.shape, ")")
        elif isinstance(total_mi, dict):
            for k, v in total_mi.items():
                if isinstance(v, np.ndarray):
                    pd.DataFrame(v.squeeze()).to_csv(f"/Volumes/WD/desktop/c255/{k}.csv", index=False)
                    print(f"✅ Saved {k}.csv (shape: {v.shape})")
        else:
            print("⚠️ Unrecognized type for total_mi:", type(total_mi))
