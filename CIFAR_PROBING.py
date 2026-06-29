import sys
import os
import gc
import json
import math
import time
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import mixed_precision

# --- 1. ENVIRONMENT & OOM MITIGATION ---
def setup_environment():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth error: {e}")

    conda_cuda_path = os.path.join(os.environ.get('CONDA_PREFIX', ''), "Library", "bin")
    if os.path.exists(conda_cuda_path):
        os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={conda_cuda_path}"

    mixed_precision.set_global_policy('float32')
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    os.environ['PYTHONHASHSEED'] = str(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)

setup_environment()

# --- 2. COMMAND LINE ARGUMENTS ---
try:
    MODE       = sys.argv[1].lower() # passive, active, or probe
    SESSION_ID = sys.argv[2]
    RUN_ID     = MODE.upper()        # Using mode as the ID for clarity
except IndexError:
    print("Usage: python CIFAR_PROBING.py <mode> <session_id>")
    sys.exit(1)

# --- 3. FIXED HYPERPARAMETERS ---
BASE_STRENGTH = 0.012968750000000001 
PERIOD        = 424.0 
LAMBDA_SLOW   = 0.024734375000000003  
JITTER_SCALE  = 0.7703124999999998 
REST_BASELINE = 1.0
H_INERTIA     = 0.9746875 

LEARNING_RATE = 5e-3
PATIENCE      = 2  # How many epochs to wait before halving
LR_FACTOR     = 0.5 # Halve the rate
MIN_LR        = 1e-6
INPUT_DIM     = 3
NUM_CLASSES   = 10
HIDDEN        = 128
EPOCHS        = 25
BATCH_SIZE    = 32
DATA_PERCENT  = 1 
PATIENCE      = 2
RAM_DISK      = "/mnt/ramdisk"
T             = 1024

# --- 5. DATA LOADER ---
def get_cifar_sequential():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize and Reshape
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, T, INPUT_DIM)
    x_test  = x_test.reshape(-1, T, INPUT_DIM)
    y_train = y_train.flatten().astype('int32')
    y_test  = y_test.flatten().astype('int32')

    # Training Dataset
    num_train = int(len(x_train) * DATA_PERCENT)
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train[:num_train], y_train[:num_train]))
        .shuffle(10000, seed=42)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Validation Subset (Used by NeuroCallback)
    val_subset_x = tf.constant(x_test[:256], dtype=tf.float32)
    val_subset_y = tf.constant(y_test[:256], dtype=tf.int32)
    
    num_test = int(len(x_test) * DATA_PERCENT)
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test[:num_test], y_test[:num_test]))
        .shuffle(10000, seed=42)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


  

    return train_ds, val_subset_x, val_subset_y, test_ds

# Unpack with the new test_ds
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading CIFAR-10...")
train_ds, val_subset_x, val_subset_y, test_ds = get_cifar_sequential()



# --- 6. MODEL DEFINITION ---
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
        self.w_in  = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", name="w_in")
        self.w_rec = self.add_weight(shape=(self.units, self.units), initializer=tf.keras.initializers.Orthogonal(gain=1.1), name="w_rec")
        self.bias  = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.RandomNormal(stddev=0.01), name="bias")
        self.neuron_gain = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.1), name="n_gain")

    def call(self, inputs, states):
        prev_h, prev_G, prev_phase = states
        half = self.units // 2
        raw_signal = tf.concat([prev_h[:, half:], prev_h[:, :half]], axis=1)
        source_signal = tf.stop_gradient(raw_signal) if self.mode == "probe" else raw_signal

        new_G  = (1.0 - self.lambda_slow) * prev_G + self.lambda_slow * source_signal
        G_norm = (new_G - tf.reduce_mean(new_G, axis=-1, keepdims=True)) / (tf.math.reduce_std(new_G, axis=-1, keepdims=True) + 1e-6)

        new_phase  = prev_phase + (2.0 * math.pi / self.period)
        oscillator = tf.math.sin(new_phase)

        if self.mode == "active":
            bias_signal = tf.reduce_mean(source_signal, axis=-1, keepdims=True) - 0.1
            combined_signal = oscillator + (self.jitter_scale * bias_signal)
        else:
            combined_signal = oscillator

        if self.mode != "passive":
            field_effect = (self.rest_baseline + (self.strength * combined_signal * tf.tanh(G_norm))) * self.neuron_gain
            field_effect = tf.clip_by_value(field_effect, 0.1, 5.0)
        else:
            field_effect = tf.ones_like(prev_h) * self.rest_baseline

        z = tf.matmul(inputs, self.w_in) + (tf.matmul(prev_h, self.w_rec) * field_effect) + self.bias
        new_h_candidate = tf.nn.leaky_relu(z, alpha=0.01)
        h = (self.h_inertia * prev_h) + ((1.0 - self.h_inertia) * new_h_candidate)
        h = tf.clip_by_value(h, -15.0, 15.0)
        return h, [h, new_G, new_phase]

def build_model(hidden, num_classes, strength, period, lambda_slow, h_inertia, jitter_scale, mode):
    cell_kwargs = dict(units=hidden, strength=strength, period=period, lambda_slow=lambda_slow, 
                       jitter_scale=jitter_scale, h_inertia=h_inertia, mode=mode)
    inputs = tf.keras.Input(shape=(T, INPUT_DIM))
    h1     = tf.keras.layers.RNN(JitteredFeedbackCell(**cell_kwargs), return_sequences=True)(inputs)
    h2     = tf.keras.layers.RNN(JitteredFeedbackCell(**cell_kwargs), return_sequences=True)(h1)
    h3     = tf.keras.layers.RNN(JitteredFeedbackCell(**cell_kwargs), return_sequences=True)(h2)
    logits = tf.keras.layers.Dense(num_classes, dtype='float32')(h3[:, -1, :])
    return tf.keras.Model(inputs=inputs, outputs=[logits, h3])

# --- 7. METRIC FUNCTIONS ---
@tf.function(jit_compile=True)
def calculate_metrics_gpu(h_seq_v):
    h_final  = h_seq_v[:, -1, :]
    batch_f  = tf.cast(tf.shape(h_final)[0], tf.float32)
    hidden_f = tf.cast(tf.shape(h_final)[1], tf.float32)

    s        = tf.linalg.svd(h_final, compute_uv=False) + 1e-12
    p_rank   = s / (tf.reduce_sum(s) + 1e-10)
    eff_rank = tf.exp(-tf.reduce_sum(p_rank * tf.math.log(p_rank + 1e-10)))

    h_norm   = (h_final - tf.reduce_mean(h_final, axis=0)) / (tf.math.reduce_std(h_final, axis=0) + 1e-8)
    corr_mat = tf.matmul(h_norm, h_norm, transpose_a=True) / batch_f
    sync_val = (tf.reduce_sum(tf.abs(corr_mat)) - hidden_f) / (hidden_f**2 - hidden_f)

    sample_seq = h_seq_v[0]
    s_norm     = (sample_seq - tf.reduce_mean(sample_seq, axis=0)) / (tf.math.reduce_std(sample_seq, axis=0) + 1e-8)
    t_corr     = tf.matmul(s_norm, s_norm, transpose_a=True) / tf.cast(tf.shape(s_norm)[0], tf.float32)
    # 1. Take a sample [Time, Hidden]
    sample_seq = h_seq_v[0] 

    # 2. Slice into t and t-1
    # x_t is steps 1 to T; x_t_minus_1 is steps 0 to T-1
    x_t = sample_seq[1:, :]
    x_t_prev = sample_seq[:-1, :]

    # 3. Normalize (Z-score) across the time dimension
    def z_score(x):
        return (x - tf.reduce_mean(x, axis=0)) / (tf.math.reduce_std(x, axis=0) + 1e-8)

    x_t_norm = z_score(x_t)
    x_t_prev_norm = z_score(x_t_prev)

    # 4. Element-wise correlation per neuron (persistence)
    # Multiply t and t-1, then average over time
    # Result is [Hidden]
    neuron_persistence = tf.reduce_mean(x_t_norm * x_t_prev_norm, axis=0)

    # 5. Scalar metric: Average persistence across all neurons
    acorr_val = tf.reduce_mean(neuron_persistence)

    mean_field       = tf.reduce_mean(h_norm, axis=1, keepdims=True)
    neuron_field_cov = tf.matmul(h_norm, mean_field, transpose_a=True) / batch_f
    intf_val         = tf.reduce_mean(tf.abs(neuron_field_cov))

    h_var    = tf.math.reduce_variance(h_final) + 1e-10
    entr_val = 0.5 * tf.math.log(2.0 * math.pi * math.e * h_var) / tf.math.log(2.0)

    return eff_rank, sync_val, acorr_val, intf_val, entr_val


# --- 8. CALLBACK ---
class NeuroCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_x, val_y, run_id, params):
        super().__init__()
        self.val_x   = val_x
        self.val_y   = val_y
        self.run_id  = run_id
        self.params_ = params
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        # Bulletproof unpacking for 2 outputs
        outputs = self.model(self.val_x, training=False)
        logits, h_seq = outputs[0], outputs[1]
        
        e_rank, s_val, acorr, intf, entr = calculate_metrics_gpu(h_seq)

        # Fallback accuracy calculation if Keras routing gets confused
        val_acc = logs.get("val_dense_sparse_categorical_accuracy", logs.get("val_sparse_categorical_accuracy", 0.0))
        if val_acc == 0.0:
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(preds, self.val_y), tf.float32))
            val_acc = float(correct / tf.cast(tf.shape(self.val_y)[0], tf.float32))

        m = {
            "epoch": epoch + 1,
            "loss":  float(logs.get("loss", 0.0)),
            "acc":   float(val_acc),
            "rank":  float(e_rank),
            "sync":  float(s_val),
            "acorr": float(acorr),
            "intf":  float(intf),
            "entr":  float(entr)
        }
        self.history.append(m)

        l_slow, h_inert, b_strength, period, jitter = self.params_
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n| RUN {self.run_id} [{ts}] Ep {epoch+1}")
        print(f"| Lam:{l_slow:.4f} | H_in:{h_inert:.3f} | Str:{b_strength:.3f} | Per:{period:.1f} | Jit:{jitter:.3f} |")
        print(f"| Loss:{m['loss']:.4f} | Acc:{m['acc']:.4f} | Rank:{m['rank']:.3f} | Entr:{m['entr']:.3f} |")
        print(f"| Syn:{m['sync']:.4f} | ACor:{m['acorr']:.4f} | Intf:{m['intf']:.4f} |")


# --- 9. BUILD & TRAIN ---
# Standardizing on uppercase 'MODE' as defined in your arg parsing block
print(f"\n[MODE: {MODE.upper()}] lam={LAMBDA_SLOW:.4f} | h_in={H_INERTIA:.3f} | "
      f"str={BASE_STRENGTH:.3f} | per={PERIOD:.1f} | jit={JITTER_SCALE:.3f}")

model = build_model(
    hidden=HIDDEN, num_classes=NUM_CLASSES,
    strength=BASE_STRENGTH, period=PERIOD,
    lambda_slow=LAMBDA_SLOW, h_inertia=H_INERTIA,
    jitter_scale=JITTER_SCALE, mode=MODE # Changed 'mode' to 'MODE'
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
    loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), None],
    metrics=[['sparse_categorical_accuracy'], []] 
)

# Use 'MODE' and the 'RUN_ID' defined in Section 2
# 1. Define the Learning Rate Halver
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_sparse_categorical_accuracy', # Targets the accuracy of the first output
    factor=LR_FACTOR,
    patience=PATIENCE,
    min_lr=MIN_LR,
    verbose=1  # This will print "Epoch 0001: ReduceLROnPlateau reducing learning rate to..."
)

# 2. Add it to your callbacks list
params_list = [LAMBDA_SLOW, H_INERTIA, BASE_STRENGTH, PERIOD, JITTER_SCALE]
cb = NeuroCallback(val_subset_x, val_subset_y, RUN_ID, params_list)

# 3. Run the training with BOTH callbacks
model.fit(
    train_ds, 
    epochs=EPOCHS, 
    validation_data=(val_subset_x, val_subset_y), # Required for LR scheduler to see val_acc
    callbacks=[cb, lr_callback], 
    verbose=0
)

# --- 10. SAVE RESULTS ---
final_acc  = cb.history[-1]["acc"]  if cb.history else 0.0
final_loss = cb.history[-1]["loss"] if cb.history else float('nan')

# Determine status for the orchestrator
if math.isnan(final_loss) or math.isinf(final_loss):
    status = "exploded"
elif final_acc < 0.15:
    status = "failed"
else:
    status = "complete"

# Keep all metrics (rank, sync, acorr, intf, entr) in the final JSON
result_data = {
    "run_id":           RUN_ID,
    "session":          SESSION_ID,
    "mode":             MODE,
    "status":           status,
    "completed_epochs": len(cb.history),
    "parameters": {
        "lambda":   LAMBDA_SLOW,
        "inertia":  H_INERTIA,
        "strength": BASE_STRENGTH,
        "period":   PERIOD,
        "jitter":   JITTER_SCALE
    },
    "acc":    final_acc,
    "rank":   cb.history[-1]["rank"]  if cb.history else 0.0,
    "sync":   cb.history[-1]["sync"]  if cb.history else 0.0,
    "acorr":  cb.history[-1]["acorr"] if cb.history else 0.0,
    "intf":   cb.history[-1]["intf"]  if cb.history else 0.0,
    "entr":   cb.history[-1]["entr"]  if cb.history else 0.0,
    "epochs": cb.history
}

print(f"\n[STATUS] Run {RUN_ID}: {status} | Epochs: {len(cb.history)} | Final Acc: {final_acc:.4f}")

def numpy_fix(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    return float(obj)

# Output filename specifically for the Probing Orchestrator
out_path = os.path.join(RAM_DISK, f"PROBE_RUN_{MODE.upper()}.json")
try:
    with open(out_path, 'w') as f:
        json.dump(result_data, f, default=numpy_fix, indent=4)
    print(f"Result saved: {out_path}")
except Exception as e:
    print(f"Error saving JSON: {e}")

# --- 10. SAVE RESULTS & EVALUATE ---
final_acc  = cb.history[-1]["acc"]  if cb.history else 0.0
final_loss = cb.history[-1]["loss"] if cb.history else float('nan')

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running batched test evaluation...")


# 3. Final Evaluation
test_scores = model.evaluate(test_ds, verbose=0)
test_acc = float(test_scores[1])

# Determine status
if math.isnan(final_loss) or math.isinf(final_loss):
    status = "exploded"
elif test_acc < 0.15:
    status = "failed"
else:
    status = "complete"

# Consolidate results
result_data = {
    "run_id":           RUN_ID,
    "session":          SESSION_ID,
    "mode":             MODE,
    "status":           status,
    "test_acc":         test_acc, 
    "val_acc_history":  final_acc,
    "completed_epochs": len(cb.history),
    "parameters": {
        "lambda":   LAMBDA_SLOW,
        "inertia":  H_INERTIA,
        "strength": BASE_STRENGTH,
        "period":   PERIOD,
        "jitter":   JITTER_SCALE
    },
    "metrics": {
        "rank":   cb.history[-1]["rank"]  if cb.history else 0.0,
        "sync":   cb.history[-1]["sync"]  if cb.history else 0.0,
        "acorr":  cb.history[-1]["acorr"] if cb.history else 0.0,
        "intf":   cb.history[-1]["intf"]  if cb.history else 0.0,
        "entr":   cb.history[-1]["entr"]  if cb.history else 0.0,
    },
    "epochs": cb.history
}

# --- SAVE MODEL WEIGHTS ---
model_path = os.path.join(RAM_DISK, f"model_{MODE.upper()}_{SESSION_ID}.weights.h5")
try:
    model.save_weights(model_path)
    print(f"Weights saved: {model_path}")
except Exception as e:
    print(f"Weights save failed: {e}")

def numpy_fix(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    return float(obj)

# --- SAVE JSON RESULTS ---
out_path = os.path.join(RAM_DISK, f"PROBE_RUN_{MODE.upper()}.json")
try:
    with open(out_path, 'w') as f:
        json.dump(result_data, f, default=numpy_fix, indent=4)
    print(f"\n[COMPLETE] Mode {MODE.upper()} saved to {out_path}")
    print(f"[STATS] Test Acc: {test_acc:.4f} | Status: {status}")
except Exception as e:
    print(f"Error saving JSON: {e}")

# Cleanup
del model, cb
tf.keras.backend.clear_session()
gc.collect()
sys.exit(0)
