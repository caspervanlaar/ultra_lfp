import os
import json
import numpy as np
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyze

# --- 1. SETUP SOBOL PROBLEM ---
T = 1000
sobol_problem = {
    'num_vars': 5,
    'names': ['LAMBDA_SLOW', 'H_INERTIA', 'BASE_STRENGTH', 'PERIOD', 'JITTER_SCALE'],    
    'bounds': [[0.0001, 0.0015], [0.92, 0.99], [0.02, 0.15], [T*1.0, T*5.0], [0.01, 0.10]]
}

N_baseline = 8 
param_values = sobol_sampler.sample(sobol_problem, N_baseline)
TOTAL_RUNS = len(param_values)

SSD_BACKUP = "/home/casper/Documents/A_Casper/Brein/CS_Ai_Wolfhampton/Dissertation/Adding_problem/SOBOL96_H128/TEST/"
PATH_HISTORY = os.path.join(SSD_BACKUP, "SOBOL_FULL_HISTORY.json")

# --- 2. LOAD DATA ---
with open(PATH_HISTORY, "r") as f:
    history_data = json.load(f)

target_metrics = {
    "mse": "mse",
    "rank": "rank",
    "sync": "sync",
    "acorr": "acorr",
    "intf": "intf",
    "entropy": "entropy"
}

# Initialize with NaNs so we can identify missing runs later
Y_vectors = {v: np.full(TOTAL_RUNS, np.nan) for v in target_metrics.values()}

for entry in history_data:
    idx = entry.get("run_id")
    epochs = entry.get("epochs", [])
    
    if idx is not None and idx < TOTAL_RUNS and len(epochs) > 0:
        final_stats = epochs[-1]
        for json_key, var_name in target_metrics.items():
            val = final_stats.get(json_key)
            # Only assign if it's a real number
            if val is not None and np.isfinite(float(val)):
                Y_vectors[var_name][idx] = float(val)

# --- 3. THE "NAN" FIX (Imputation) ---
print("\n--- Cleaning Data for Analysis ---")
for var_name, vec in Y_vectors.items():
    nan_mask = ~np.isfinite(vec)
    nan_count = np.sum(nan_mask)
    
    if nan_count > 0:
        # For MSE, NaNs (explosions) are bad. Replace with max observed + some penalty.
        # For Rank/Entropy, we replace with the global mean or a safe boundary.
        if var_name == "mse":
            fill_val = np.nanmax(vec) * 2 if not np.all(nan_mask) else 10.0
        else:
            fill_val = np.nanmedian(vec) if not np.all(nan_mask) else 0.0
            
        vec[nan_mask] = fill_val
        print(f"  [!] {var_name}: Replaced {nan_count} NaN/Inf values with {fill_val:.4f}")

# --- 4. ANALYZE ---
Si_results = {}
for var_name in target_metrics.values():
    print(f"Computing Sobol indices for: {var_name}")
    try:
        Si_results[var_name] = sobol_analyze.analyze(sobol_problem, Y_vectors[var_name])
    except Exception as e:
        print(f"  [!] Failed {var_name}: {e}")

# --- 5. SAVE ---
def _numpy_fix(obj):
    return obj.tolist() if hasattr(obj, 'tolist') else float(obj)

final_path = os.path.join(SSD_BACKUP, "SOBOL_FINAL_RESULTS.json")
with open(final_path, "w") as f:
    json.dump(Si_results, f, indent=4, default=_numpy_fix)

print(f"\n[CLEANED & COMPLETE] Results saved to: {final_path}")