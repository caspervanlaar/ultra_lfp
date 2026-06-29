import subprocess
import os
import json
import shutil
import sys
import glob
from datetime import datetime

# --- 1. SETUP PROBING MODES ---
MODES = ["active", "probe", "passive"]
SESSION_ID = f"CIFAR_PROBE_{datetime.now().strftime('%y%m%d_%H%M')}"

RAM_PATH   = "/mnt/ramdisk"
SSD_BACKUP = "/home/casper/Documents/A_Casper/Brein/CS_Ai_Wolfhampton/Dissertation/CIFAR_P_RESULTS/H128_probing/"

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

    # Call the CIFAR_PROBING.py worker
    cmd = [sys.executable, "CIFAR_PROBING.py", mode, SESSION_ID]
    subprocess.run(cmd)

    # --- 3. MIGRATE ALL JSON RESULTS ---
    # We look for the specific JSON expected from the worker
    json_filename = f"PROBE_RUN_{mode.upper()}.json"
    src_json = os.path.join(RAM_PATH, json_filename)
    dst_json = os.path.join(SSD_BACKUP, json_filename)

    if os.path.exists(src_json):
        with open(src_json, "r") as f:
            worker_data = json.load(f)
        full_history_archive.append(worker_data)
        shutil.move(src_json, dst_json)
    else:
        print(f"    [!] Warning: JSON result {json_filename} missing.")

    # --- 4. MIGRATE ALL .H5 FILES (BROAD SWEEP) ---
    # Captures any and all weight files regardless of naming convention
    h5_files = glob.glob(os.path.join(RAM_PATH, "*.h5"))
    if h5_files:
        for f in h5_files:
            shutil.move(f, os.path.join(SSD_BACKUP, os.path.basename(f)))
            print(f"    [OK] Moved weights: {os.path.basename(f)}")
    else:
        print("    [!] No .h5 files found to move.")

    # --- 5. ATOMIC HISTORY UPDATE ---
    with open(PATH_HISTORY + ".tmp", "w") as f:
        json.dump(full_history_archive, f, indent=2)
    os.replace(PATH_HISTORY + ".tmp", PATH_HISTORY)

    os.system("sync")
    print(f">>> Mode {mode.upper()} finalized.")

print("\n--- ALL SESSIONS COMPLETE ---")