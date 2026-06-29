"""
audit_sobol_runs.py

Compares the parameters stored in each archived SOBOL_RUN_{i}.json against
the correct N=64 Sobol table. Prints a summary and writes two files:
  - audit_mismatched.json  : run_ids where params deviate beyond tolerance
  - audit_ok.json          : run_ids that are clean

Usage:
    python audit_sobol_runs.py [backup_dir]

Default backup_dir: /home/casper/sobol_cifar_backup
"""

import os
import sys
import json
import numpy as np
from SALib.sample import saltelli

# --- CONFIG ---
BACKUP_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/casper/sobol_cifar_backup"
TOL        = 1e-3   # float tolerance for parameter comparison

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
# Ground-truth table (N=64 -- what the orchestrator used)
param_values_64 = saltelli.sample(sobol_problem, 64)
# Wrong table (N=32 -- what the worker used)
param_values_32 = saltelli.sample(sobol_problem, 32)

PARAM_KEYS = ['lambda', 'inertia', 'strength', 'period', 'jitter']

# --- SCAN ARCHIVE ---
all_files = sorted(
    [f for f in os.listdir(BACKUP_DIR) if f.startswith("SOBOL_RUN_") and f.endswith(".json")],
    key=lambda f: int(f.replace("SOBOL_RUN_", "").replace(".json", ""))
)

print(f"Found {len(all_files)} archived run files in {BACKUP_DIR}")
print(f"Correct table size  (N=64): {len(param_values_64)} rows")
print(f"Wrong   table size  (N=32): {len(param_values_32)} rows")
print("-" * 72)

ok          = []
mismatched  = []
missing     = []

for fname in all_files:
    run_id = int(fname.replace("SOBOL_RUN_", "").replace(".json", ""))
    fpath  = os.path.join(BACKUP_DIR, fname)

    with open(fpath, "r") as f:
        data = json.load(f)

    stored = data.get("parameters", {})
    stored_vec = np.array([
        stored.get("lambda",   np.nan),
        stored.get("inertia",  np.nan),
        stored.get("strength", np.nan),
        stored.get("period",   np.nan),
        stored.get("jitter",   np.nan),
    ])

    if np.any(np.isnan(stored_vec)):
        print(f"  [MISSING PARAMS] run {run_id:4d} -- no parameters key in JSON")
        missing.append(run_id)
        continue

    # Compare against N=64 table
    if run_id < len(param_values_64):
        correct_vec = param_values_64[run_id]
        max_err_64  = float(np.max(np.abs(stored_vec - correct_vec)))
    else:
        correct_vec = None
        max_err_64  = float('inf')

    # Compare against N=32 table (to confirm what table was actually used)
    if run_id < len(param_values_32):
        wrong_vec  = param_values_32[run_id]
        max_err_32 = float(np.max(np.abs(stored_vec - wrong_vec)))
    else:
        wrong_vec  = None
        max_err_32 = float('inf')

    is_wrong   = max_err_64 > TOL
    matches_32 = max_err_32 <= TOL

    if is_wrong:
        mismatched.append({
            "run_id":       run_id,
            "max_err_vs_64": max_err_64,
            "max_err_vs_32": max_err_32,
            "matches_n32":  matches_32,
            "stored":       stored_vec.tolist(),
            "correct_n64":  correct_vec.tolist() if correct_vec is not None else None,
        })
        tag = "matches N=32 table" if matches_32 else "unknown origin"
        print(f"  [MISMATCH] run {run_id:4d} | err_vs_64={max_err_64:.2e} | {tag}")
    else:
        ok.append(run_id)

# --- SUMMARY ---
print("\n" + "=" * 72)
print(f"  OK (match N=64)  : {len(ok)}")
print(f"  MISMATCHED       : {len(mismatched)}")
print(f"  MISSING params   : {len(missing)}")
print(f"  Total scanned    : {len(all_files)}")

if mismatched:
    first_bad = mismatched[0]["run_id"]
    last_bad  = mismatched[-1]["run_id"]
    n32_count = sum(1 for m in mismatched if m["matches_n32"])
    print(f"\n  Bad run range    : {first_bad} -- {last_bad}")
    print(f"  Confirmed N=32   : {n32_count} / {len(mismatched)}")
    print(f"\n  Runs to rerun    : {[m['run_id'] for m in mismatched]}")

# --- WRITE OUTPUT FILES ---
out_mismatch = os.path.join(BACKUP_DIR, "audit_mismatched.json")
out_ok       = os.path.join(BACKUP_DIR, "audit_ok.json")

with open(out_mismatch, "w") as f:
    json.dump(mismatched, f, indent=2)

with open(out_ok, "w") as f:
    json.dump(ok, f, indent=2)

print(f"\n  Mismatch details : {out_mismatch}")
print(f"  OK list          : {out_ok}")
print("=" * 72)

# --- RERUN HELPER ---
# Prints a ready-to-paste command to requeue only the bad runs
if mismatched:
    bad_ids = [m["run_id"] for m in mismatched]
    print("\nRerun snippet (paste into a shell script or orchestrator):")
    print("  bad_ids = " + str(bad_ids))
    print()
    print("  for run_id in bad_ids:")
    print("      subprocess.run([sys.executable, 'CIFAR_sobol_worker.py',")
    print("                      str(run_id), 'SESSION_CIFAR'])")