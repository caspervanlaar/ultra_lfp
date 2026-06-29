import subprocess
import os
import json
import numpy as np
import shutil
import sys
from SALib.sample import saltelli

# --- 1. SETUP SOBOL PROBLEM ---
T             = 1024
sobol_problem = {
    'num_vars': 5,
    'names': ['LAMBDA_SLOW', 'H_INERTIA', 'BASE_STRENGTH', 'PERIOD', 'JITTER_SCALE'],
'bounds': [
        [0.001, 0.05],
        [0.01, 0.99],
        [0.01, 0.2],
        [T/13, T/2],
        [0.1, 1.2]
    ]
}
N_baseline   = 32
param_values = saltelli.sample(sobol_problem, N_baseline)
TOTAL_RUNS   = len(param_values)

RAM_PATH   = "/mnt/ramdisk"
SSD_BACKUP = "/home/casper/sobol_cifar_backup"

if not os.path.exists(SSD_BACKUP):
    os.makedirs(SSD_BACKUP)

PATH_Y       = os.path.join(SSD_BACKUP, "SOBOL_PROGRESS_Y.json")
PATH_HISTORY = os.path.join(SSD_BACKUP, "SOBOL_FULL_HISTORY.json")

metric_keys = ["acc", "rank", "sync", "intf", "entr"]

print(f"Total planned runs: {TOTAL_RUNS}")

# --- 2. LOAD EXISTING ARCHIVE INTO Y AND HISTORY ---
# Backfills Y for all already-completed runs so the final Sobol analysis
# sees a complete, correct Y vector -- not zeros for the first 288 slots.
Y = {m: np.zeros(TOTAL_RUNS) for m in metric_keys}
full_history_archive = []
already_done = set()

for run_id in range(TOTAL_RUNS):
    fpath = os.path.join(SSD_BACKUP, f"SOBOL_RUN_{run_id}.json")
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            worker_data = json.load(f)
        for k in metric_keys:
            Y[k][run_id] = worker_data.get(k, 0.0)
        full_history_archive.append(worker_data)
        already_done.add(run_id)

print(f"Loaded {len(already_done)} existing runs from archive: "
      f"{min(already_done) if already_done else '-'} -- {max(already_done) if already_done else '-'}")

remaining = [i for i in range(TOTAL_RUNS) if i not in already_done]
print(f"Runs remaining: {len(remaining)}  "
      f"({remaining[0] if remaining else 'none'} -- {remaining[-1] if remaining else 'none'})")

# --- 3. EXECUTION LOOP (only missing runs) ---
for run_id in remaining:
    print(f"\n>>> Starting Run {run_id + 1} of {TOTAL_RUNS}")

    cmd = [sys.executable, "CIFAR_sobol_worker.py", str(run_id), "SESSION_CIFAR"]
    subprocess.run(cmd)

    # --- 4. AGGREGATE RESULT ---
    filename = f"SOBOL_RUN_{run_id}.json"
    src = os.path.join(RAM_PATH, filename)
    dst = os.path.join(SSD_BACKUP, filename)

    if os.path.exists(src):
        with open(src, "r") as f:
            worker_data = json.load(f)
        for k in metric_keys:
            Y[k][run_id] = worker_data.get(k, 0.0)
        full_history_archive.append(worker_data)
        shutil.move(src, dst)
    else:
        print(f"    [!] Warning: {filename} not found on RAM disk. Run marked failed.")
        for k in metric_keys:
            Y[k][run_id] = 0.0

    # --- 5. SAVE MASTER PROGRESS (ATOMIC WRITE) ---
    with open(PATH_Y + ".tmp", "w") as f:
        json.dump({k: v.tolist() for k, v in Y.items()}, f, indent=2)
    os.replace(PATH_Y + ".tmp", PATH_Y)

    with open(PATH_HISTORY + ".tmp", "w") as f:
        json.dump(full_history_archive, f, indent=2)
    os.replace(PATH_HISTORY + ".tmp", PATH_HISTORY)

    os.system("sync")
    print(f">>> Run {run_id} complete. Progress synced to {SSD_BACKUP}.")

# --- 6. FINAL SOBOL ANALYSIS ---
from SALib.analyze import sobol as sobol_analyze

print("\n--- ALL RUNS COMPLETE. Processing Sobol indices... ---")
Si_results = {}
for k in metric_keys:
    if np.any(Y[k]):
        try:
            Si_results[k] = sobol_analyze.analyze(sobol_problem, Y[k])
        except Exception as e:
            print(f"  [!] Sobol analysis failed for metric '{k}': {e}")

def _numpy_fix(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return float(obj)

final_path = os.path.join(SSD_BACKUP, "SOBOL_FINAL_RESULTS.json")
with open(final_path, "w") as f:
    json.dump(Si_results, f, indent=4, default=_numpy_fix)

print(f"Sobol indices saved to {final_path}")
print("Analysis complete.")