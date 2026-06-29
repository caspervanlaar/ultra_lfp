import subprocess
import os
import json
import shutil
import sys
import glob
import time
from datetime import datetime

# --- 1. SETUP PROBING MODES ---
MODES      = ["active"]
# Keeps SESSION_ID consistent across all three sub-worker calls
SESSION_ID = f"ADDING_PROBE_{datetime.now().strftime('%y%m%d_%H%M')}"

RAM_PATH   = "/mnt/ramdisk"
SSD_BACKUP = "/home/casper/Documents/A_Casper/Brein/CS_Ai_Wolfhampton/Dissertation/Adding_problem/H32_probing/"

if not os.path.exists(SSD_BACKUP):
    os.makedirs(SSD_BACKUP)

PATH_HISTORY = os.path.join(SSD_BACKUP, "PROBING_FULL_HISTORY.json")

print(f"Session: {SESSION_ID}")
print(f"Targeting Modes: {', '.join(MODES)}")

# --- 2. EXECUTION LOOP ---
full_history_archive = []

for mode in MODES:
    print(f"\n" + "="*50)
    print(f">>> Executing: {mode.upper()}")
    print("="*50)

    # Calling the script we just optimized
    cmd = [sys.executable, "Adding_probing_worker.py", mode, SESSION_ID]
    result = subprocess.run(cmd)

    # Check if the worker actually finished correctly
    if result.returncode != 0:
        print(f"    [!] Worker failed for mode {mode} with exit code {result.returncode}")

    # --- 3. MIGRATE JSON RESULT ---
    json_filename = f"PROBE_RUN_{mode.upper()}.json"
    src_json = os.path.join(RAM_PATH, json_filename)
    dst_json = os.path.join(SSD_BACKUP, json_filename)

    if os.path.exists(src_json):
        with open(src_json, "r") as f:
            worker_data = json.load(f)
        full_history_archive.append(worker_data)
        # Using move instead of copy to keep RAM disk clean
        shutil.move(src_json, dst_json)
        print(f"    [OK] Moved JSON: {json_filename}")
    else:
        print(f"    [!] Warning: JSON result {json_filename} missing.")

    # --- 4. MIGRATE ALL .H5 WEIGHT FILES ---
    # This glob targets the weights specific to this session
    h5_pattern = os.path.join(RAM_PATH, f"*{SESSION_ID}*.h5")
    h5_files = glob.glob(h5_pattern)
    
    if h5_files:
        for f in h5_files:
            shutil.move(f, os.path.join(SSD_BACKUP, os.path.basename(f)))
            print(f"    [OK] Moved weights: {os.path.basename(f)}")
    else:
        print("    [!] No .h5 files found for this session.")

    # --- 5. ATOMIC HISTORY UPDATE ---
    with open(PATH_HISTORY + ".tmp", "w") as f:
        json.dump(full_history_archive, f, indent=2)
    os.replace(PATH_HISTORY + ".tmp", PATH_HISTORY)

    # --- 6. COOL-DOWN PERIOD ---
    # Brief pause to allow the GPU driver to fully reset the context 
    # and clear any residual memory before the next mode starts.
    print(f">>> Mode {mode.upper()} finalized. Reclaiming VRAM...")
    time.sleep(2)
    os.system("sync")

print("\n--- ALL SESSIONS COMPLETE ---")