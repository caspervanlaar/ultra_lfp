import subprocess
import os
import json
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import gc

# Configuration
sobol_problem = {
    'num_vars': 5,
    'names': ['LAMBDA_SLOW', 'H_INERTIA', 'BASE_STRENGTH', 'PERIOD', 'JITTER_SCALE'],
    'bounds': [[0.001, 0.05], [0.01, 0.99], [0.01, 0.99], [78.4, 392.0], [0.1, 1.2]]
}

N_baseline = 64 # Set back to 64 for your full sweep
param_values = saltelli.sample(sobol_problem, N_baseline)
metric_keys = ["acc", "rank", "sync", "intf", "entr"]
Y = {m: np.zeros(len(param_values)) for m in metric_keys}

print(f"Total planned runs: {len(param_values)}")



# --- 1. SETUP ARCHIVE ---
# This will hold the FULL data (epochs + params) for every run
full_history_archive = [] 
Y = {m: np.zeros(len(param_values)) for m in metric_keys}

for i, params in enumerate(param_values):
    print(f"\n>>> Starting Run {i}/{len(param_values)}")
    
    cmd = ["python", "sobol_worker.py", str(i), "SESSION_01"] + [str(p) for p in params]
    subprocess.run(cmd)
    
    temp_file = f"SOBOL_RUN_{i}.json"
    if os.path.exists(temp_file):
        with open(temp_file, "r") as f:
            worker_data = json.load(f)
            
            # A) KEEP SALib HAPPY: Store only the last epoch value in Y
            for k in metric_keys:
                Y[k][i] = worker_data[k]
            
            # B) KEEP YOU HAPPY: Append the entire worker JSON to our archive
            full_history_archive.append(worker_data)
        
        # Now it's safe to delete the temp file
        os.remove(temp_file)
        
        # --- 2. SAVE PROGRESS (Both Files) ---
        # File 1: For SALib / Quick check of final values
        with open("SOBOL_PROGRESS_Y.json", "w") as f:
            json.dump({k: v.tolist() for k, v in Y.items()}, f, indent=2)
            
        # File 2: THE HOLY GRAIL (Every epoch of every run)
        with open("SOBOL_FULL_HISTORY.json", "w") as f:
            json.dump(full_history_archive, f, indent=2)
            
    else:
        print(f"!!! CRITICAL: Run {i} failed. No JSON produced.")
        # Fill Y with 0.0 so SALib doesn't crash later
        for k in metric_keys: Y[k][i] = 0.0
         
 

    
    os.system("sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches")    
    os.system("sudo swapoff -a && sudo swapon -a") 
    gc.collect()

# --- FINAL SOBOL ANALYSIS ---
print("\nProcessing final Sobol indices...")
Si_results = {}
for k in metric_keys:
    # Check if we have enough non-zero data to analyze
    if np.any(Y[k]):
        Si_results[k] = sobol.analyze(sobol_problem, Y[k])

with open("SOBOL_FINAL_RESULTS.json", "w") as f:
    json.dump(Si_results, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

print("Analysis Complete.")