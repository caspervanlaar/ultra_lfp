import subprocess
import os
import json
import numpy as np
import shutil
import sys
from SALib.sample import saltelli

# --- 1. SETUP SOBOL PROBLEM ---
T = 1000
sobol_problem = {
    'num_vars': 5,
    'names': ['LAMBDA_SLOW', 'H_INERTIA', 'BASE_STRENGTH', 'PERIOD', 'JITTER_SCALE'],    
    'bounds': [
        [0.0001, 0.0015],  # LAMBDA_SLOW
        [0.92, 0.99],      # H_INERTIA
        [0.02, 0.15],      # BASE_STRENGTH
        [T*1.0, T*5.0],    # PERIOD
        [0.01, 0.10]       # JITTER_SCALE
    ]
}

N_baseline = 8 
param_values = saltelli.sample(sobol_problem, N_baseline)
RAM_PATH = "/mnt/ramdisk"
SSD_BACKUP = "/home/casper/sobol_adding_backup"

if not os.path.exists(SSD_BACKUP):
    os.makedirs(SSD_BACKUP)

# --- MASTER FILES ---
PATH_Y = os.path.join(SSD_BACKUP, "SOBOL_PROGRESS_Y.json")
PATH_HISTORY = os.path.join(SSD_BACKUP, "SOBOL_FULL_HISTORY.json")

# Initialize master containers
Y_mse = np.zeros(len(param_values))
full_history_archive = []

# --- 2. EXECUTION LOOP ---
# We now iterate one by one to ensure a clean RAM/VRAM state every time
for run_id in range(len(param_values)):
    print(f"\n>>> Starting Run {run_id + 1} of {len(param_values)}")
    
    # LAUNCH WORKER FOR A SINGLE RUN
    # This ensures TensorFlow completely exits and releases the GPU after every run
    cmd = [sys.executable, "Adding_sobol_worker.py", str(run_id), str(run_id + 1)]
    subprocess.run(cmd)

    # --- 3. AGGREGATE RESULT ---
    filename = f"SOBOL_RUN_{run_id}.json"
    src = os.path.join(RAM_PATH, filename)
    dst = os.path.join(SSD_BACKUP, filename)
    
    if os.path.exists(src):
        with open(src, "r") as f:
            worker_data = json.load(f)
        
        # Extract MSE from the last epoch
        if "epochs" in worker_data and len(worker_data["epochs"]) > 0:
            Y_mse[run_id] = worker_data["epochs"][-1]["mse"]
        else:
            Y_mse[run_id] = 999.0
        
        full_history_archive.append(worker_data)
        
        # Archive the individual run file
        shutil.move(src, dst)
    else:
        print(f"    [!] Warning: {filename} not found.")
        Y_mse[run_id] = 999.0

    # --- 4. SAVE MASTER PROGRESS (ATOMIC) ---
    # Save after every run so you don't lose data if the power goes out
    with open(PATH_Y + ".tmp", "w") as f:
        json.dump(Y_mse[:run_id+1].tolist(), f, indent=4)
    os.replace(PATH_Y + ".tmp", PATH_Y)
        
    with open(PATH_HISTORY + ".tmp", "w") as f:
        json.dump(full_history_archive, f, indent=4)
    os.replace(PATH_HISTORY + ".tmp", PATH_HISTORY)

    # --- 5. SYSTEM MAINTENANCE ---
    # No sudo needed: 'sync' flushes writes to disk
    os.system("sync")
    print(f">>> Run {run_id} complete. OS reclaimed memory. Progress synced.")

print("\n--- ALL RUNS COMPLETE ---")