import sys
import os
import gc
import json
import math
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import mixed_precision

# --- 1. ENVIRONMENT ---
def setup_environment():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth error: {e}")
    mixed_precision.set_global_policy('float32')
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    os.environ['PYTHONHASHSEED'] = str(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)

setup_environment()

# =============================================================================
# --- 2. CONFIGURATION --- Edit these to match your session
# =============================================================================
# Which mode to test: "active", "probe", or "passive"
MODE       = "passive"
SESSION_ID = "260423_1223"   # The timestamp portion of your session folder name

SSD_BACKUP = "/home/casper/Documents/A_Casper/Brein/CS_Ai_Wolfhampton/Dissertation/Adding_problem/H64_probing/"
RAM_DISK   = "/home/casper/Documents/A_Casper/Brein/CS_Ai_Wolfhampton/Dissertation/Adding_problem/H64_probing/"

# Sobol-optimized constants (must match training)
T             = 1000
HIDDEN        = 64
LAMBDA_SLOW   = 0.0007
H_INERTIA     = 0.965
BASE_STRENGTH = 0.076
PERIOD        = 2500.0
JITTER_SCALE  = 0.055
REST_BASELINE = 1.0
BATCH_SIZE    = 256

# Full session tag as produced by the orchestrator
FULL_SESSION  = f"ADDING_PROBE_{SESSION_ID}"

# Weight file saved by ModelCheckpoint in the worker
WEIGHTS_PATH  = os.path.join(
    SSD_BACKUP,
    f"best_weights_{MODE.upper()}_{FULL_SESSION}.weights.h5"
)

# JSON metadata file saved by the worker
JSON_PATH = os.path.join(SSD_BACKUP, f"PROBE_RUN_{MODE.upper()}.json")

# --- Fallback: also check RAM disk in case orchestrator hasn't moved it yet ---
if not os.path.exists(WEIGHTS_PATH):
    WEIGHTS_PATH_RAM = os.path.join(
        RAM_DISK,
        f"best_weights_{MODE.upper()}_{FULL_SESSION}.weights.h5"
    )
    if os.path.exists(WEIGHTS_PATH_RAM):
        print(f"[INFO] Weights found on ramdisk, using: {WEIGHTS_PATH_RAM}")
        WEIGHTS_PATH = WEIGHTS_PATH_RAM

if not os.path.exists(JSON_PATH):
    JSON_PATH_RAM = os.path.join(RAM_DISK, f"PROBE_RUN_{MODE.upper()}.json")
    if os.path.exists(JSON_PATH_RAM):
        print(f"[INFO] JSON found on ramdisk, using: {JSON_PATH_RAM}")
        JSON_PATH = JSON_PATH_RAM

print(f"\n{'='*60}")
print(f" ADDING TASK PROBING TESTER")
print(f" Mode:     {MODE.upper()}")
print(f" Session:  {FULL_SESSION}")
print(f" Weights:  {WEIGHTS_PATH}")
print(f" JSON:     {JSON_PATH}")
print(f"{'='*60}\n")

# =============================================================================
# --- 3. CELL ARCHITECTURE (must be identical to worker) ---
# =============================================================================
class JitteredFeedbackCell(tf.keras.layers.Layer):
    def __init__(self, units, strength, period, lambda_slow, jitter_scale,
                 h_inertia, rest_baseline=1.0, mode="active", **kwargs):
        super().__init__(**kwargs)
        self.units         = units
        self.strength      = strength
        self.period        = period
        self.lambda_slow   = lambda_slow
        self.jitter_scale  = jitter_scale
        self.h_inertia     = h_inertia
        self.rest_baseline = rest_baseline
        self.mode          = mode
        self.state_size    = [units, units, 1]

    def build(self, input_shape):
        self.w_in  = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer="glorot_uniform", name="w_in")
        self.w_rec = self.add_weight(shape=(self.units, self.units),
                                     initializer=tf.keras.initializers.Orthogonal(gain=1.1),
                                     name="w_rec")
        self.bias  = self.add_weight(shape=(self.units,),
                                     initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                     name="bias")
        self.neuron_gain = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.1),
            trainable=True, name="n_gain")

    def call(self, inputs, states):
        prev_h, prev_G, prev_phase = states
        half       = self.units // 2
        raw_signal = tf.concat([prev_h[:, half:], prev_h[:, :half]], axis=1)

        source_signal = tf.stop_gradient(raw_signal) if self.mode == "probe" else raw_signal

        new_G  = (1.0 - self.lambda_slow) * prev_G + self.lambda_slow * source_signal
        G_norm = (new_G - tf.reduce_mean(new_G, axis=-1, keepdims=True)) / \
                 (tf.math.reduce_std(new_G, axis=-1, keepdims=True) + 1e-6)

        new_phase  = prev_phase + (2.0 * math.pi / self.period)
        oscillator = tf.math.sin(new_phase)

        if self.mode == "active":
            bias_signal     = tf.reduce_mean(source_signal, axis=-1, keepdims=True) - 0.1
            combined_signal = oscillator + (self.jitter_scale * bias_signal)
        else:
            combined_signal = oscillator

        if self.mode != "passive":
            field_effect = (self.rest_baseline +
                            (self.strength * combined_signal * tf.tanh(G_norm))) * self.neuron_gain
            field_effect = tf.clip_by_value(field_effect, 0.1, 5.0)
        else:
            field_effect = tf.ones_like(prev_h) * self.rest_baseline

        z          = tf.matmul(inputs, self.w_in) + \
                     (tf.matmul(prev_h, self.w_rec) * field_effect) + self.bias
        new_h_cand = tf.nn.leaky_relu(z, alpha=0.01)
        h          = (self.h_inertia * prev_h) + ((1.0 - self.h_inertia) * new_h_cand)
        h          = tf.clip_by_value(h, -15.0, 15.0)
        return h, [h, new_G, new_phase]


def build_model(hidden, strength, period, lambda_slow, h_inertia, jitter_scale, mode):
    cell_kwargs = dict(units=hidden, strength=strength, period=period,
                       lambda_slow=lambda_slow, jitter_scale=jitter_scale,
                       h_inertia=h_inertia, mode=mode)
    inputs = tf.keras.Input(shape=(T, 2))
    h_seq  = tf.keras.layers.RNN(JitteredFeedbackCell(**cell_kwargs),
                                 return_sequences=True)(inputs)
    output = tf.keras.layers.Dense(1)(h_seq[:, -1, :])
    return tf.keras.Model(inputs=inputs, outputs=[output, h_seq])


# =============================================================================
# --- 4. METRICS (identical to worker) ---
# =============================================================================
@tf.function(jit_compile=True)
def calculate_metrics_gpu(h_seq_v):
    h_final  = h_seq_v[:, -1, :]
    batch_f  = tf.cast(tf.shape(h_final)[0], tf.float32)
    hidden_f = tf.cast(tf.shape(h_final)[1], tf.float32)

    s        = tf.linalg.svd(h_final, compute_uv=False) + 1e-12
    p_rank   = s / (tf.reduce_sum(s) + 1e-10)
    eff_rank = tf.exp(-tf.reduce_sum(p_rank * tf.math.log(p_rank + 1e-10)))

    h_norm   = (h_final - tf.reduce_mean(h_final, axis=0)) / \
               (tf.math.reduce_std(h_final, axis=0) + 1e-8)
    corr_mat = tf.matmul(h_norm, h_norm, transpose_a=True) / batch_f
    sync_val = (tf.reduce_sum(tf.abs(corr_mat)) - hidden_f) / (hidden_f**2 - hidden_f)

    sample_seq    = h_seq_v[0]
    x_t           = sample_seq[1:, :]
    x_t_prev      = sample_seq[:-1, :]

    def z_score(x):
        return (x - tf.reduce_mean(x, axis=0)) / (tf.math.reduce_std(x, axis=0) + 1e-8)

    x_t_norm      = z_score(x_t)
    x_t_prev_norm = z_score(x_t_prev)
    neuron_persist = tf.reduce_mean(x_t_norm * x_t_prev_norm, axis=0)
    acorr_val      = tf.reduce_mean(neuron_persist)

    mean_field       = tf.reduce_mean(h_norm, axis=1, keepdims=True)
    neuron_field_cov = tf.matmul(h_norm, mean_field, transpose_a=True) / batch_f
    intf_val         = tf.reduce_mean(tf.abs(neuron_field_cov))

    h_var    = tf.math.reduce_variance(h_final) + 1e-10
    entr_val = 0.5 * tf.math.log(2.0 * math.pi * math.e * h_var) / tf.math.log(2.0)

    return eff_rank, sync_val, acorr_val, intf_val, entr_val


# =============================================================================
# --- 5. DATA (same seed as worker to get identical test split) ---
# =============================================================================
def get_adding_data(num_samples=15000, length=T):
    X_val  = np.random.uniform(0, 1, (num_samples, length, 1))
    X_mask = np.zeros((num_samples, length, 1))
    Y      = np.zeros((num_samples, 1))
    for i in range(num_samples):
        idx = np.random.choice(length, 2, replace=False)
        X_mask[i, idx, 0] = 1.0
        Y[i, 0] = np.sum(X_val[i, idx, 0])
    X = np.concatenate([X_val, X_mask], axis=-1)
    return tf.constant(X, dtype=tf.float32), tf.constant(Y, dtype=tf.float32)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating data (seed=42 matches worker)...")
X_ALL, Y_ALL   = get_adding_data()
VAL_X,  VAL_Y  = X_ALL[12000:13000], Y_ALL[12000:13000]
TEST_X, TEST_Y = X_ALL[13000:],      Y_ALL[13000:]


# =============================================================================
# --- 6. LOAD MODEL ---
# =============================================================================
if not os.path.exists(WEIGHTS_PATH):
    print(f"[ERROR] Weights not found: {WEIGHTS_PATH}")
    print("Check MODE, SESSION_ID, and SSD_BACKUP path at the top of this script.")
    sys.exit(1)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Building {MODE.upper()} model...")
model = build_model(
    hidden=HIDDEN, strength=BASE_STRENGTH, period=PERIOD,
    lambda_slow=LAMBDA_SLOW, h_inertia=H_INERTIA,
    jitter_scale=JITTER_SCALE, mode=MODE
)
# Build graph with a dummy forward pass before loading weights
_ = model(tf.zeros((1, T, 2)), training=False)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading weights from {WEIGHTS_PATH}...")
model.load_weights(WEIGHTS_PATH)
print("[OK] Weights loaded.\n")


# =============================================================================
# --- 7. BATCHED EVALUATION HELPER ---
# =============================================================================
def batched_mse(model, X, Y, batch_size=BATCH_SIZE):
    """MSE over the full set in batches to avoid OOM."""
    total_sq_err = 0.0
    n = X.shape[0]
    for start in range(0, n, batch_size):
        xb = X[start:start+batch_size]
        yb = Y[start:start+batch_size]
        preds, _ = model(xb, training=False)
        total_sq_err += float(tf.reduce_sum(tf.square(preds - yb)))
    return total_sq_err / n


def batched_metrics(model, X, batch_size=BATCH_SIZE):
    """Aggregate representational metrics across batches (mean of per-batch values)."""
    ranks, syncs, acorrs, intfs, entrs = [], [], [], [], []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        xb = X[start:start+batch_size]
        if xb.shape[0] < 2:      # SVD needs at least 2 samples
            continue
        _, h_seq = model(xb, training=False)
        e_rank, s_val, acorr, intf, entr = calculate_metrics_gpu(h_seq)
        ranks.append(float(e_rank))
        syncs.append(float(s_val))
        acorrs.append(float(acorr))
        intfs.append(float(intf))
        entrs.append(float(entr))
    return {
        "rank":  float(np.mean(ranks)),
        "sync":  float(np.mean(syncs)),
        "acorr": float(np.mean(acorrs)),
        "intf":  float(np.mean(intfs)),
        "entr":  float(np.mean(entrs)),
    }


# =============================================================================
# --- 8. RUN EVALUATION ---
# =============================================================================
ts = datetime.now().strftime('%H:%M:%S')
print(f"[{ts}] Running validation evaluation ({VAL_X.shape[0]} samples)...")
val_mse   = batched_mse(model, VAL_X, VAL_Y)
val_stats = batched_metrics(model, VAL_X)

ts = datetime.now().strftime('%H:%M:%S')
print(f"[{ts}] Running test evaluation ({TEST_X.shape[0]} samples)...")
test_mse   = batched_mse(model, TEST_X, TEST_Y)
test_stats = batched_metrics(model, TEST_X)

# =============================================================================
# --- 9. PRINT SUMMARY ---
# =============================================================================
def fmt_block(title, mse, stats):
    lines = [
        f"\n{'='*55}",
        f"  {title}",
        f"{'='*55}",
        f"  MSE        : {mse:.6f}",
        f"  Eff. Rank  : {stats['rank']:.4f}",
        f"  Synchrony  : {stats['sync']:.4f}",
        f"  Autocorr   : {stats['acorr']:.4f}",
        f"  Interference: {stats['intf']:.4f}",
        f"  Entropy    : {stats['entr']:.4f}",
        f"{'='*55}",
    ]
    return "\n".join(lines)

print(fmt_block(f"VALIDATION  |  Mode: {MODE.upper()}  |  Session: {FULL_SESSION}",
                val_mse, val_stats))
print(fmt_block(f"TEST        |  Mode: {MODE.upper()}  |  Session: {FULL_SESSION}",
                test_mse, test_stats))

# =============================================================================
# --- 10. PATCH JSON WITH TEST RESULTS ---
# =============================================================================
def numpy_fix(obj):
    if isinstance(obj, (np.integer, np.floating, np.ndarray, np.generic)):
        return obj.item() if hasattr(obj, 'item') else obj.tolist()
    return float(obj)

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r") as f:
        result_data = json.load(f)
    print(f"\n[INFO] Loaded existing JSON from {JSON_PATH}")
else:
    print(f"\n[WARN] JSON not found at {JSON_PATH}. Creating fresh entry.")
    result_data = {
        "run_id":   MODE.upper(),
        "session":  FULL_SESSION,
        "mode":     MODE,
        "status":   "evaluated",
        "parameters": {
            "lambda":   LAMBDA_SLOW,
            "inertia":  H_INERTIA,
            "strength": BASE_STRENGTH,
            "period":   PERIOD,
            "jitter":   JITTER_SCALE
        },
        "epochs": []
    }

# Patch in the test/val results
result_data["status"]            = "evaluated"
result_data["test_mse"]          = test_mse
result_data["val_mse_final"]     = val_mse
result_data["evaluated_at"]      = datetime.now().isoformat()
result_data["weights_loaded"]    = WEIGHTS_PATH
result_data["test_metrics"]      = test_stats
result_data["val_metrics"]       = val_stats

# Write back to SSD (keep the original JSON on SSD, not RAM disk)
ssd_json_path = os.path.join(SSD_BACKUP, f"PROBE_RUN_{MODE.upper()}.json")
tmp_path      = ssd_json_path + ".tmp"
try:
    with open(tmp_path, "w") as f:
        json.dump(result_data, f, default=numpy_fix, indent=4)
    os.replace(tmp_path, ssd_json_path)
    print(f"[OK] JSON updated: {ssd_json_path}")
except Exception as e:
    print(f"[ERROR] Could not write JSON: {e}")

# =============================================================================
# --- 11. CLEANUP ---
# =============================================================================
del model
tf.keras.backend.clear_session()
gc.collect()

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done.")
