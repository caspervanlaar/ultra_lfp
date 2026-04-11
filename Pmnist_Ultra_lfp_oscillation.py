import numpy as np
from tqdm import tqdm
import time
import os
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
import gc
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import scipy.linalg
import math
import os, gc, random
import matplotlib.pyplot as plt
import json
import pickle
import time
from datetime import datetime
from IPython.display import clear_output
def clear():
    # Clears terminal for Windows (cls) and Linux/Mac (clear)
    os.system('cls' if os.name == 'nt' else 'clear')
tf.keras.mixed_precision.set_global_policy('float32')

# Note: Your model will now output float16. 
# Ensure your last layer is float32 for stability:
# self.out = tf.keras.layers.Dense(num_classes, dtype='float32')

# Standard Conda path for libdevice
conda_cuda_path = os.path.join(os.environ.get('CONDA_PREFIX', ''), "Library", "bin")
if os.path.exists(conda_cuda_path):
    os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={conda_cuda_path}"
    print(f"XLA Path set to: {conda_cuda_path}")

# EMERGENCY BYPASS: If it still fails, disable JIT for now. 
# It will be slightly slower, but it will NOT crash.
USE_JIT = False

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir="C:/Users/caspe/anaconda3/envs/SPIKEDETEC/Library/bin"'

# 1. Load MNIST
(x_all, y_all), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Flatten and Normalize (Standard step)
x_all = x_all.reshape(-1, 784).astype('uint8') / 255.0
x_test = x_test.reshape(-1, 784).astype('uint8') / 255.0

# 3. Create FIXED Permutation (The "p" in pMNIST)
rng = np.random.RandomState(42)
perm = rng.permutation(784)

x_all = x_all[:, perm]
x_test = x_test[:, perm]

# --- NEW: Train/Val Split (90% Train, 10% Val) ---
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_all, y_all, test_size=0.1, random_state=42, stratify=y_all
)

# 4. Reshape to [Batch, Time, Channels] for your RNN
x_train = x_train[:, :, np.newaxis]
x_val   = x_val[:, :, np.newaxis]
x_test  = x_test[:, :, np.newaxis]

# 5. Labels to int32
y_train = y_train.astype('int32')
y_val   = y_val.astype('int32')
y_test  = y_test.astype('int32')

# 6. Build datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(256)
val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(256)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

print(f"pMNIST Ready. Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

import tensorflow as tf
# --- 1. CONFIGURATION ---


# Forcefully disable the strict determinism flag at the TF level
try:
    tf.config.experimental.enable_op_determinism(False)
    print("Success: Strict determinism disabled.")
except:
    print("Warning: Could not toggle flag. If Phase 1 fails, restart kernel.")
def reset_seeds():
    import os
    import gc
    import random
    import numpy as np

    # 1. Clear session
    tf.keras.backend.clear_session()
    
    # 2. Disable strict determinism to allow GPU kernels to run
    os.environ['TF_DETERMINISTIC_OPS'] = '0' 
    
    np.random.seed(42)
    tf.random.set_seed(42)
  
    np.random.seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
  
  
    # 3. Hard-lock all seeds
    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42) 
    
    gc.collect()
    print("Environment Reset: Seeds locked at 42. Strict determinism disabled.")
    
def print_param_report(hidden_size, input_dim=1, num_classes=10):
    """
    Calculates and prints a detailed breakdown of parameters for the 
    OscillatingResonator architecture.
    """
    # RNN Layer 1: Inputs from data
    rnn1_params = (input_dim * hidden_size) + (hidden_size * hidden_size) + hidden_size
    
    # RNN Layer 2 & 3: Inputs from previous hidden layer
    rnn_mid_params = (hidden_size * hidden_size) + (hidden_size * hidden_size) + hidden_size
    
    # Final Dense Layer
    dense_params = (hidden_size * num_classes) + num_classes
    
    total = rnn1_params + (2 * rnn_mid_params) + dense_params
    
    print(f"--- PARAMETER COUNT REPORT (Hidden: {hidden_size}) ---")
    print(f"RNN Layer 1: {rnn1_params:>8,}")
    print(f"RNN Layer 2: {rnn_mid_params:>8,}")
    print(f"RNN Layer 3: {rnn_mid_params:>8,}")
    print(f"Dense Out:   {dense_params:>8,}")
    print("-" * 35)
    print(f"TOTAL PARAMS: {total:>8,}")
    print(f"Estimated Memory: {total * 4 / 1024:.2f} KB (float32)")
    print("=" * 35 + "\n")


 
import numpy as np
import tensorflow as tf
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import gc
from datetime import datetime
import json
DATA_PERCENT = 0.1
BATCH_SIZE = 4 * 64
EPOCHS = 12         
HIDDEN = 32 
REST_BASELINE = 1.0
LEARNING_RATE = 1e-3
#Changed in sobol
BASE_STRENGTH =0
LAMBDA_SLOW = 0
PERIOD= 0 
JITTER_SCALE = 0
N_baseline = 64 # T
# --- 1.1 AUTO-GENERATED SESSION NAME ---
# Creates a name like: RES_H32_S0.40_J1.15_T123456
now = datetime.now()
readable_ts = now.strftime("%y%m%d_%H%M") 
SESSION_ID = f"RES_H{HIDDEN}_S{BASE_STRENGTH}_J{JITTER_SCALE}_{readable_ts}"
print(f" SESSION INITIALIZED: {SESSION_ID}")
print_param_report(HIDDEN)

sobol_problem = {
    'num_vars': 5,
    'names': ['LAMBDA_SLOW', 'H_INERTIA','BASE_STRENGTH', 'PERIOD', 'JITTER_SCALE'],
    'bounds': [
        [0.001, 0.05], 
        [0.01, 0.99],
        [0.01, 0.99],      
        [784/10, 784/2], 
        [0.1, 1.2]
    ]
}

def calc_sobol_samples(problem, N, second_order=True):
    """
    Calculates total runs for a SALib Saltelli/Sobol sequence.
    Formula: N * (2D + 2) if second_order=True
             N * (D + 2)  if second_order=False
    """
    D = problem['num_vars']
    if second_order:
        total = N * (2 * D + 2)
    else:
        total = N * (D + 2)
    return total

# Your specific setup

total_runs = calc_sobol_samples(sobol_problem, N_baseline, second_order=True)

print(f"--- SOBOL RUN ESTIMATION ---")
print(f"Variables (D): {sobol_problem['num_vars']}")
print(f"Baseline (N):  {N_baseline}")
print(f"Total Model Evaluations: {total_runs}")

# Time estimation (Optional but helpful for dissertations)
avg_time_per_run = EPOCHS*24 # seconds (Estimate based on your Epochs)
total_hours = (total_runs * avg_time_per_run) / 3600
total_days = total_hours/24
print(f"Estimated Time: {total_hours:.2f} hours")
print(f"Estimated Time: {total_days:.2f} days")






# --- 2. THE JITTERED CELL ---
class JitteredFeedbackCell(tf.keras.layers.Layer):
    def __init__(self, units=16, strength=0.0, period=256.0, lambda_slow=0.05, mode="active", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = [units, units, 1] 
        self.strength = strength 
        self.period = period
        self.lambda_slow = lambda_slow
        self.mode = mode

    def build(self, input_shape):
        self.w_in = self.add_weight(shape=(input_shape[-1], self.units), name="w_in", initializer="glorot_uniform")
        self.w_rec = self.add_weight(shape=(self.units, self.units), name="w_rec",
                                     initializer=tf.keras.initializers.Orthogonal(gain=1.0))
        self.bias = self.add_weight(shape=(self.units,), name="bias", initializer="zeros")

    def call(self, inputs, states):
        prev_h, prev_G, prev_phase = states
        half = self.units // 2
        raw_signal = tf.concat([prev_h[:, half:], prev_h[:, :half]], axis=1)
        source_signal = tf.stop_gradient(raw_signal) if self.mode == "probe" else raw_signal
        new_G = (1.0 - self.lambda_slow) * prev_G + self.lambda_slow * source_signal
        G_mean = tf.reduce_mean(new_G, axis=-1, keepdims=True)
        G_std = tf.math.reduce_std(new_G, axis=-1, keepdims=True) + 1e-6
        G_norm = (new_G - G_mean) / G_std
        new_phase = prev_phase + (2.0 * math.pi / self.period)
        oscillator = tf.math.sin(new_phase)
        
        if self.mode == "active":
            bias_signal = tf.reduce_mean(source_signal, axis=-1, keepdims=True) - 0.1
            combined_signal = oscillator + (JITTER_SCALE * bias_signal)
        else:
            combined_signal = oscillator

        current_strength = self.strength * combined_signal if self.mode != "passive" else 0.0
        field_effect = REST_BASELINE + (current_strength * tf.tanh(G_norm))
        z = (tf.matmul(inputs, self.w_in) + tf.matmul(prev_h, self.w_rec) + self.bias) * field_effect
        h = (H_SCALE[0] * prev_h) + (H_SCALE[1] * tf.nn.elu(z))
        h = tf.clip_by_value(h, -20.0, 20.0)
        return h, [h, new_G, new_phase]

class OscillatingResonator(tf.keras.Model):
    def __init__(self, hidden=16, num_classes=10, strength=0.0, mode="active"):
        super().__init__()
        self.cell_ref = JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode)
        self.rnn1 = tf.keras.layers.RNN(self.cell_ref, return_sequences=True)
        self.rnn2 = tf.keras.layers.RNN(JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode), return_sequences=True)
        self.rnn3 = tf.keras.layers.RNN(JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode), return_sequences=True)
        self.out = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        h1 = self.rnn1(x, training=training)
        h2 = self.rnn2(h1, training=training)
        h3 = self.rnn3(h2, training=training)
        return self.out(h3[:, -1, :]), h3

# --- 3. UPDATED LOGGING UTILITY ---
def print_history_summary(history, model, model_name="Model", test_acc=None):
    cell = model.rnn1.cell
    total_params = model.count_params()
    
    print(f"\n DATA LOG: {model_name}")
    print(f" CONFIG: Hidden: {cell.units} | Params: {total_params:,} | F-Wgt: {cell.strength:.4f} | "
          f"L-Slow (τ): {cell.lambda_slow:.3f} | Jitter: {JITTER_SCALE:.2f} | Period: {cell.period:.1f} | Data: {DATA_PERCENT*100:.2f}%")
    print("="*145)
    header = f"{'Epoch':<6} | {'Loss':<7} | {'Val-Acc%':<8} | {'Rank':<6} | {'Sync':<6} | {'Entrp':<6} | {'A-Corr':<7} | {'Intf':<6} | {'F-Wgt':<6}"
    print(header)
    print("-" * 145)

    for i in range(len(history['loss'])):
        m = history['hidden_metrics'][i]
        print(f"{i+1:<6} | {history['loss'][i]:<7.3f} | {history['acc'][i]*100:<8.2f} | "
              f"{m['effective_rank']:<6.2f} | {m['synchrony']:<6.3f} | {m['entropy']:<6.2f} | "
              f"{m['a_corr']:<7.3f} | {m['interference']:<6.3f} | {cell.strength:<6.3f}")
    
    print("-" * 145)
    test_str = f"{test_acc*100:.2f}%" if test_acc is not None else "N/A"
    print(f" FINAL PERFORMANCE: Validation Acc: {history['acc'][-1]*100:.2f}% | TEST ACCURACY: {test_str}")
    print("="*145 + "\n")

# --- 4. TRAINING PHASE WITH SNAPSHOT & LIVE LOGS ---
# --- UPDATED TRAINING PHASE ---
def train_phase(model, train_data, val_data, epochs=3, name="model"):
    current_lr = LEARNING_RATE
    optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    history = {"loss": [], "acc": [], "hidden_metrics": []}
    best_val_acc = 0.0

    @tf.function
    def train_step(x, y, lr):
        optimizer.learning_rate.assign(lr)
        with tf.GradientTape() as tape:
            logits, _ = model(x, training=True)
            loss_v = loss_fn(y, logits)
        grads = tape.gradient(loss_v, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_v, logits


    print(f"\n{'Epoch':<6} | {'Loss':<7} | {'Val-Acc%':<8} | {'Rank':<6} | {'Sync':<6} | {'Entrp':<6} | {'A-Corr':<7}")
    print("-" * 75)

    for epoch in range(epochs):
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_loss = []
        pbar = tqdm(train_data, desc=f"EPOCH {epoch+1}/{epochs}", leave=False)
        
        for x_b, y_b in pbar:
            loss_v, logits = train_step(x_b, y_b, current_lr)
            acc_metric.update_state(y_b, logits)
            epoch_loss.append(float(loss_v))
            pbar.set_postfix({"loss": f"{np.mean(epoch_loss):.4f}", "acc": f"{acc_metric.result():.2%}"})

        # --- VALIDATION SNAPSHOT ---
        # --- VALIDATION SNAPSHOT (GPU VERSION) ---
        for x_v, y_v in val_data.take(1):
            logits_v, h_seq_v = model(x_v, training=False)
            val_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_v, axis=-1, output_type=tf.int32), y_v), tf.float32))
            
            # h_final is [Batch, Hidden]
            h_final = h_seq_v[:, -1, :] 
            
            # 1. Effective Rank (GPU)
            s = tf.linalg.svd(h_final, compute_uv=False) + 1e-12
            p_rank = s / tf.reduce_sum(s)
            eff_rank = tf.exp(-tf.reduce_sum(p_rank * tf.math.log(p_rank + 1e-10)))
            
            # 2. Entropy (Approx via Histogram on GPU)
            # Note: TF histogram is slightly different, but faster for GSA trends
            counts = tf.histogram_fixed_width(h_final, [-20.0, 20.0], nbins=50)
            p_ent = tf.cast(counts, tf.float32) / (tf.cast(tf.size(h_final), tf.float32) + 1e-10)
            entropy_val = -tf.reduce_sum(p_ent * tf.math.log(p_ent + 1e-10) / tf.math.log(2.0))
            
            # 3. Synchrony (GPU Correlation)
            # We transpose to get [Hidden, Batch] to correlate neurons
            h_t = tf.transpose(h_final)
            # Subtract mean and divide by std for correlation
            h_norm = h_t - tf.reduce_mean(h_t, axis=1, keepdims=True)
            h_std = tf.math.reduce_std(h_t, axis=1, keepdims=True) + 1e-8
            corr_mat = tf.matmul(h_norm, h_norm, transpose_b=True) / (tf.cast(tf.shape(h_t)[1], tf.float32) * h_std * tf.transpose(h_std))
            sync_val = (tf.reduce_sum(tf.abs(corr_mat)) - HIDDEN) / (HIDDEN**2 - HIDDEN)

            # 4. Interference (GPU)
            mean_field = tf.reduce_mean(h_final, axis=1, keepdims=True) # [Batch, 1]
            # Correlate each neuron with the mean field
            h_final_norm = h_final - tf.reduce_mean(h_final, axis=0, keepdims=True)
            mf_norm = mean_field - tf.reduce_mean(mean_field, axis=0)
            
            # Simplified Interference calculation on GPU
            cov = tf.reduce_mean(h_final_norm * mf_norm, axis=0)
            std_prod = tf.math.reduce_std(h_final, axis=0) * tf.math.reduce_std(mean_field) + 1e-8
            interference_val = tf.reduce_mean(tf.abs(cov / std_prod))

            # Convert to float for history (minimal overhead)
            history["loss"].append(float(np.mean(epoch_losses)))
            history["acc"].append(float(val_acc))
            history["hidden_metrics"].append({
                "effective_rank": float(eff_rank),
                "synchrony": float(sync_val),
                "entropy": float(entropy_val),
                "interference": float(interference_val)
            })

            print(f"{epoch+1:<6} | {np.mean(epoch_loss):<7.3f} | {val_acc*100:<8.2f} | "
                  f"{eff_rank:<6.2f} | {sync_val:<6.3f} | {entropy_val:<6.2f} | {acorr_val:<7.3f}{status_msg}")
    
    # 2. SAVE LATEST WEIGHTS (Post-training)
    model.save_weights(f"latest_{name}_{SESSION_ID}.weights.h5")
    print(f"--- Finished {name}: Best and Latest weights saved. ---")
            
    return history

# --- 5. SAVE ---
# This map will hold everything: configs, histories, and final scores
master_results = {
    "config": {
        "hidden": HIDDEN,
        "base_strength": BASE_STRENGTH,
        "jitter": JITTER_SCALE,
        "period": PERIOD,
        "epochs": EPOCHS,
        "data_pct": DATA_PERCENT
    },
    "runs": {}
}

def safe_evaluate(model, dataset):
    """Evaluates accuracy in batches to prevent GPU OOM errors."""
    accs = []
    for x_batch, y_batch in dataset:
        logits, _ = model(x_batch, training=False)
        preds = np.argmax(logits.numpy(), axis=-1)
        accs.append(np.mean(preds == y_batch.numpy()))
    return np.mean(accs)


# --- 6. EXECUTION ---
def reset_env():
    tf.keras.backend.clear_session()
    random.seed(42); np.random.seed(42); tf.random.set_seed(42)
    gc.collect()

# SETUP DATA
num_train = int(len(x_train) * DATA_PERCENT)
num_val = int(len(x_val)*DATA_PERCENT)
num_test  = int(len(x_test) * DATA_PERCENT)

train_ds = tf.data.Dataset.from_tensor_slices((x_train[:num_train], y_train[:num_train])) \
    .shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val[:1000], y_val[:1000])) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds_subset = tf.data.Dataset.from_tensor_slices((x_test[:num_test], y_test[:num_test])) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Test Run Ready: Samples -> Train: {num_train}, Val: {num_val}, Data %: {DATA_PERCENT*100}")

# --- RUN 1: ACTIVE ---
# --- 5. EXECUTION & UNIQUE SAVING ---

master_results = {
    "session_id": SESSION_ID,
    "config": {
        "hidden": HIDDEN, "base_strength": BASE_STRENGTH,
        "jitter": JITTER_SCALE, "period": PERIOD, "epochs": EPOCHS
    },
    "runs": {}
}

def save_metrics_to_files(history, model, model_name, test_acc):
    cell = model.rnn1.cell
    total_params = model.count_params()
    
    # 1. GENERATE THE TEXT TABLE 
    log_lines = []
    log_lines.append(f"\n DATA LOG: {model_name}")
    log_lines.append(f" CONFIG: Hidden: {cell.units} | Params: {total_params:,} | F-Wgt: {cell.strength:.4f} | "
                     f"L-Slow (τ): {cell.lambda_slow:.3f} | Jitter: {JITTER_SCALE:.2f} | Period: {cell.period:.1f} | Data: {DATA_PERCENT*100:.2f}%")
    log_lines.append("="*145)
    header = f"{'Epoch':<6} | {'Loss':<7} | {'Val-Acc%':<8} | {'Rank':<6} | {'Sync':<6} | {'Entrp':<6} | {'A-Corr':<7} | {'Intf':<6} | {'F-Wgt':<6}"
    log_lines.append(header)
    log_lines.append("-" * 145)

    for i in range(len(history['loss'])):
        m = history['hidden_metrics'][i]
        line = (f"{i+1:<6} | {history['loss'][i]:<7.3f} | {history['acc'][i]*100:<8.2f} | "
                f"{m['effective_rank']:<6.2f} | {m['synchrony']:<6.3f} | {m['entropy']:<6.2f} | "
                f"{m['a_corr']:<7.3f} | {m['interference']:<6.3f} | {cell.strength:<6.3f}")
        log_lines.append(line)
    
    log_lines.append("-" * 145)
    test_str = f"{test_acc*100:.2f}%" if test_acc is not None else "N/A"
    log_lines.append(f" FINAL PERFORMANCE: Validation Acc: {history['acc'][-1]*100:.2f}% | TEST ACCURACY: {test_str}")
    log_lines.append("="*145 + "\n")

    # Save to TXT (Appends so you don't lose previous runs)
    with open(f"log_{SESSION_ID}.txt", "a") as f:
        f.write("\n".join(log_lines))
    
    # Print to console as usual
    print("\n".join(log_lines))

    # 2. SAVE RAW DATA TO JSON
    json_data = {
        "session_id": SESSION_ID,
        "model_name": model_name,
        "config": {
            "hidden": cell.units, "strength": cell.strength, "tau": cell.lambda_slow,
            "jitter": JITTER_SCALE, "period": PERIOD, "data_pct": DATA_PERCENT
        },
        "history": history,
        "test_accuracy": float(test_acc)
    }
    
    # Simple conversion for numpy types
    def clean_types(obj):
        if isinstance(obj, dict): return {k: clean_types(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean_types(i) for i in obj]
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        return obj

    with open(f"results_{SESSION_ID}_{model_name}.json", "w") as f:
        json.dump(clean_types(json_data), f, indent=4)

import tensorflow as tf
import numpy as np
import scipy.linalg
import json
import gc
import math
from tqdm.auto import tqdm

# --- 1. GLOBAL MASTER LOGGING ---
SOBOL_MASTER_DATA = {"session_id": SESSION_ID, "runs": {}, "Si_all": {}}

def update_json_log(history, model, run_id):
    """Saves every epoch's progress into one master 'Big Ass' JSON file."""
    global SOBOL_MASTER_DATA
    if run_id is None: return # Skip if not a Sobol run
    
    cell = model.rnn1.cell
    SOBOL_MASTER_DATA["runs"][str(run_id)] = {
        "config": {
            "tau": float(cell.lambda_slow),
            "strength": float(cell.strength),
            "jitter": float(JITTER_SCALE),
            "period": float(cell.period),
            #"learning rate": float(LEARNING_RATE),
            "h_inertia": float(H_SCALE[0])
        },
        "history": history
    }

    def clean(obj):
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(i) for i in obj]
        if isinstance(obj, (np.float32, np.float64, np.ndarray)): 
            return float(obj) if np.isscalar(obj) else obj.tolist()
        return obj

    with open(f"SOBOL_MASTER_{SESSION_ID}.json", "w") as f:
        json.dump(clean(SOBOL_MASTER_DATA), f, indent=4)

# --- 2. THE JITTERED CELL & MODEL ---
class JitteredFeedbackCell(tf.keras.layers.Layer):
    def __init__(self, units=16, strength=0.0, period=256.0, lambda_slow=0.05, mode="active", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = [units, units, 1] 
        self.strength = strength 
        self.period = period
        self.lambda_slow = lambda_slow
        self.mode = mode

    def build(self, input_shape):
        self.w_in = self.add_weight(shape=(input_shape[-1], self.units), name="w_in", initializer="glorot_uniform")
        self.w_rec = self.add_weight(shape=(self.units, self.units), name="w_rec",
                                     initializer=tf.keras.initializers.Orthogonal(gain=1.0))
        self.bias = self.add_weight(shape=(self.units,), name="bias", initializer="zeros")

    def call(self, inputs, states):
        prev_h, prev_G, prev_phase = states
        half = self.units // 2
        raw_signal = tf.concat([prev_h[:, half:], prev_h[:, :half]], axis=1)
        source_signal = tf.stop_gradient(raw_signal) if self.mode == "probe" else raw_signal
        
        new_G = (1.0 - self.lambda_slow) * prev_G + self.lambda_slow * source_signal
        G_norm = (new_G - tf.reduce_mean(new_G, axis=-1, keepdims=True)) / (tf.math.reduce_std(new_G, axis=-1, keepdims=True) + 1e-6)
        
        new_phase = prev_phase + (2.0 * math.pi / self.period)
        oscillator = tf.math.sin(new_phase)
        
        if self.mode == "active":
            bias_signal = tf.reduce_mean(source_signal, axis=-1, keepdims=True) - 0.1
            combined_signal = oscillator + (JITTER_SCALE * bias_signal)
        else:
            combined_signal = oscillator

        current_strength = self.strength * combined_signal if self.mode != "passive" else 0.0
        field_effect = REST_BASELINE + (current_strength * tf.tanh(G_norm))
        
        z = (tf.matmul(inputs, self.w_in) + tf.matmul(prev_h, self.w_rec) + self.bias) * field_effect
        h = (H_SCALE[0] * prev_h) + (H_SCALE[1] * tf.nn.elu(z))
        h = tf.clip_by_value(h, -20.0, 20.0)
        return h, [h, new_G, new_phase]

class OscillatingResonator(tf.keras.Model):
    def __init__(self, hidden=16, num_classes=10, strength=0.0, mode="active"):
        super().__init__()
        # Use cell_ref so we can access hyperparameters easily later
        self.cell_ref = JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode)
        self.rnn1 = tf.keras.layers.RNN(self.cell_ref, return_sequences=True)
        self.rnn2 = tf.keras.layers.RNN(JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode), return_sequences=True)
        self.rnn3 = tf.keras.layers.RNN(JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode), return_sequences=True)
        self.out = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        h1 = self.rnn1(x, training=training)
        h2 = self.rnn2(h1, training=training)
        h3 = self.rnn3(h2, training=training)
        return self.out(h3[:, -1, :]), h3

# --- 3. UPDATED TRAINING PHASE ---
def train_phase(model, train_data, val_data, epochs=3, name="model", run_id=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    history = {"loss": [], "acc": [], "hidden_metrics": []}
    best_val_acc = 0.0

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits, _ = model(x, training=True)
            loss_v = loss_fn(y, logits)
        grads = tape.gradient(loss_v, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_v, logits

    # Added 'Intf' to the console header
    print(f"\n{'Epoch':<6} | {'Loss':<7} | {'Val-Acc%':<8} | {'Rank':<6} | {'Sync':<6} | {'Entrp':<6} | {'A-Corr':<7} | {'Intf':<6}")
    print("-" * 85)

    for epoch in range(epochs):
        epoch_losses = []
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        pbar = tqdm(train_data, desc=f"EPOCH {epoch+1}/{epochs}", leave=False,disable=True)
        
        for x_b, y_b in pbar:
            loss_v, logits = train_step(x_b, y_b)
            acc_metric.update_state(y_b, logits)
            epoch_losses.append(float(loss_v))
            pbar.set_postfix({"loss": f"{np.mean(epoch_losses):.4f}", "acc": f"{acc_metric.result():.2%}"})

        # --- VALIDATION SNAPSHOT ---
        for x_v, y_v in val_data.take(1):
            logits_v, h_seq_v = model(x_v, training=False)
            val_acc = np.mean(np.argmax(logits_v.numpy(), axis=-1) == y_v.numpy())            
            
            # --- METRIC CALCULATIONS ---
            h_final = h_seq_v.numpy()[:, -1, :]
            
            # 1. Effective Rank
            s = scipy.linalg.svdvals(h_final) + 1e-12
            p_rank = s / (np.sum(s) + 1e-10)
            eff_rank = np.exp(-np.sum(p_rank * np.log(p_rank + 1e-10)))
            
            # 2. Entropy
            counts, _ = np.histogram(h_final, bins=50)
            p_ent = counts / (h_final.size + 1e-10) 
            entropy_val = -np.sum(p_ent * np.log2(p_ent + 1e-10))
            
            # 3. Synchrony (Inter-neuron correlation)
            sync_val = (np.sum(np.abs(np.corrcoef(h_final.T + 1e-8))) - HIDDEN) / (HIDDEN**2 - HIDDEN)
            
            # 4. Auto-Correlation (Temporal slowness)
            acorr_val = np.mean(np.abs(np.corrcoef(h_seq_v.numpy()[0].T + 1e-8)))

            # 5. NEW: Interference (Neuron-to-Field alignment)
            # Measures how much neurons are driven by the common field vs unique input
            mean_field = np.mean(h_final, axis=1, keepdims=True)
            neuron_to_field_corrs = [
                np.abs(np.corrcoef(h_final[:, j], mean_field[:, 0])[0, 1]) 
                for j in range(h_final.shape[1])
            ]
            interference_val = np.mean(np.nan_to_num(neuron_to_field_corrs))

        # --- HISTORY UPDATE ---
        avg_loss = np.mean(epoch_losses)
        history["loss"].append(float(avg_loss))
        history["acc"].append(float(val_acc))
        history["hidden_metrics"].append({
            "effective_rank": float(eff_rank),
            "synchrony": float(sync_val),
            "entropy": float(entropy_val),
            "a_corr": float(acorr_val),
            "interference": float(interference_val)
        })

        # --- THE BIG JSON SAVE ---
        update_json_log(history, model, run_id)
        
        # Unified Console Output
        print(f"EP {epoch+1}/{epochs}: {avg_loss:<7.3f} | {val_acc:<8.2%} | {eff_rank:<6.2f} | "
              f"{sync_val:<6.3f} | {entropy_val:<6.2f} | {acorr_val:<7.3f} | {interference_val:<6.3f}")

    return history


# --- 7. SOBOL SENSITIVITY ANALYSIS IMPLEMENTATION (FIXED) ---

SOBOL_MASTER_DATA = {
    "session_id": SESSION_ID,
    "problem": sobol_problem,
    "runs": {},
    "Si_all": {} 
}

def evaluate_sobol_run(params, run_id):
    # Map back to the globals the model uses
    global LEARNING_RATE, JITTER_SCALE, BASE_STRENGTH, PERIOD, LAMBDA_SLOW, H_INERTIA, H_SCALE
    
    current_lslow, h_inertia, current_strength, current_period, current_jitter = params
    
    LAMBDA_SLOW = current_lslow
    H_INERTIA = h_inertia          
    BASE_STRENGTH = current_strength
    PERIOD = current_period
    JITTER_SCALE = current_jitter
    
    # Critical derivation for architectural parity
    H_SCALE = [H_INERTIA, 1.0 - H_INERTIA] 

    # Clear memory from previous run to prevent VRAM accumulation
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Build model (Will use the updated globals)
    # Reset seeds here for identical initialization across configurations
    tf.keras.utils.set_random_seed(42)
    model = OscillatingResonator(hidden=HIDDEN, num_classes=10, strength=BASE_STRENGTH, mode="active")
    
    # Standard Header for the Run
    print(f"\n[Sobol Run {run_id}/{total_runs}] "
          f"H_Inertia: {H_INERTIA:.4f} | "
          f"BASE_S: {BASE_STRENGTH:.4f} | "
          f"Tau: {LAMBDA_SLOW:.4f} | "
          f"Jitter: {JITTER_SCALE:.2f}")

    # Execute training
    # Note: ensure train_phase uses print(f"\r...", end="", flush=True) for epoch updates
    history = train_phase(model, train_ds, val_ds, epochs=EPOCHS, name="sobol", run_id=run_id)
    
    # Final newline to clear the 'flush' line from train_phase
    print("") 

    # Safely extract metrics from the final epoch
    metrics = {
        "acc": float(history['acc'][-1]),
        "rank": float(history['hidden_metrics'][-1]['effective_rank']),
        "sync": float(history['hidden_metrics'][-1]['synchrony']),
        "entr": float(history['hidden_metrics'][-1]['entropy']),
        "acorr": float(history['hidden_metrics'][-1]['a_corr']),
        "Intf": float(history['hidden_metrics'][-1]['interference'])
    }
    
    # Cleanup model reference
    del model
    clear()
    return metrics

def run_sobol_analysis():
    print(f"--- STARTING MULTI-METRIC SOBOL ANALYSIS ({SESSION_ID}) ---")
    
    # SALib generates N * (2D + 2) samples
    # For D=5, N=8, this results in 8 * (10 + 2) = 96 runs
    param_values = saltelli.sample(sobol_problem, N_baseline, calc_second_order=True) 
    num_runs = len(param_values)
    
    metric_keys = ["acc", "rank", "sync", "entr", "acorr", "Intf"]
    Y = {m: np.zeros(num_runs) for m in metric_keys}

    for i, params in enumerate(param_values):
        try:
            res = evaluate_sobol_run(params, run_id=i)
            for key in metric_keys: 
                Y[key][i] = res[key]
        except Exception as e:
            # Move to next line so the error doesn't overwrite the progress line
            print(f"\n!! Run {i} failed: {e}")
            # Fill with NaN so the analyzer can handle the missing data
            for key in metric_keys: Y[key][i] = np.nan 

    # Calculate Sensitivity Indices for all tracked metrics
    Si_all = {}
    for key in metric_keys:
        y_clean = Y[key]
        nan_mask = np.isnan(y_clean)
        n_failed = np.sum(nan_mask)
        
        if n_failed > 0:
            print(f"Warning: {n_failed} failed runs for metric '{key}'")
        
        # If more than 10% of runs failed, the indices might be unreliable
        if n_failed / len(y_clean) > 0.1:
            print(f"Skipping {key}: failure rate too high ({n_failed}/{len(y_clean)})")
            Si_all[key] = None
            continue
            
        # Perform Sobol Analysis
        Si = sobol.analyze(sobol_problem, y_clean, print_to_console=False)
        Si_all[key] = {
            "S1": Si['S1'].tolist(),
            "ST": Si['ST'].tolist(),
            "S2": Si['S2'].tolist() if 'S2' in Si else None,
            "n_failed": int(n_failed)
        }

    SOBOL_MASTER_DATA["Si_all"] = Si_all
    
    # JSON Serialization helper for Numpy/Tensorflow types
    def clean(obj):
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(i) for i in obj]
        if isinstance(obj, (np.float32, np.float64, np.ndarray)): 
            return float(obj) if np.isscalar(obj) else obj.tolist()
        return obj

    output_path = f"SOBOL_MASTER_{SESSION_ID}.json"
    with open(output_path, "w") as f:
        json.dump(clean(SOBOL_MASTER_DATA), f, indent=4)
        
    print(f"\nFINISH. Full sensitivity results saved to: {output_path}")
    return Si_all

# --- EXECUTION ---
Si_all = run_sobol_analysis()

