import subprocess
import os
import json
import numpy as np
import shutil
from SALib.sample import saltelli

# 1. Setup Sobol Problem - MATCH T TO THE WORKER
T = 1000 
sobol_problem = {
    'num_vars': 5,
    'names': ['LAMBDA_SLOW', 'H_INERTIA', 'BASE_STRENGTH', 'PERIOD', 'JITTER_SCALE'],
    'bounds': [
        [0.001, 0.05],   # LAMBDA_SLOW
        [0.01, 0.99],    # H_INERTIA
        [0.01, 0.99],    # BASE_STRENGTH
        [T/10, T/2],     # PERIOD (100 to 500)
        [0.1, 1.2]       # JITTER_SCALE
    ]
}

N_baseline = 64 # Total samples = N * (2D + 2) = 64 * 12 = 768 runs
param_values = saltelli.sample(sobol_problem, N_baseline)
RAM_PATH = "/mnt/ramdisk"
SSD_BACKUP = os.path.expanduser("~/sobol_adding_backup")
BATCH_SIZE = 50 # Subprocess restarts every 50 runs to purge VRAM

if not os.path.exists(SSD_BACKUP):
    os.makedirs(SSD_BACKUP)

# 2. Execution Loop
for start_idx in range(0, len(param_values), BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, len(param_values))
    
    # LAUNCH WORKER
    cmd = ["python", "Adding_sobol_worker.py", str(start_idx), str(end_idx)]
    subprocess.run(cmd)

    # BACKUP RAMDISK TO SSD
    for f in os.listdir(RAM_PATH):
        if f.endswith(".json"):
            shutil.move(os.path.join(RAM_PATH, f), os.path.join(SSD_BACKUP, f))
    
    # Deep system cache flush
    os.system("sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches")