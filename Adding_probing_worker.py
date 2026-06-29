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

# --- 2. COMMAND LINE ARGUMENTS ---
try:
    MODE       = sys.argv[1].lower()   # "active" or "probe"
    SESSION_ID = sys.argv[2]
    RUN_ID     = MODE.upper()
except IndexError:
    print("Usage: python Adding_probing_worker.py <mode> <session_id>")
    sys.exit(1)

# --- 3. FIXED HYPERPARAMETERS (Optimized via Sobol) ---
LAMBDA_SLOW   = 0.0007          
H_INERTIA     = 0.98#0.915#0.955#0.965           
BASE_STRENGTH = 0.15#0.13#0.09#0.076           
PERIOD        = 2500.0          
JITTER_SCALE  = 0.07#0.1#0.07#0.055           
REST_BASELINE = 1.0

T             = 1000
HIDDEN        = 32
EPOCHS        = 250             
BATCH_SIZE    = 256
LEARNING_RATE = 2e-4
LR_PATIENCE   = 10              # Epochs to wait before halving LR
LR_FACTOR     = 0.5
MIN_LR        = 1e-6
RAM_DISK      = "/mnt/ramdisk"

# --- 4. DATA GENERATION ---
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

print(f"[{datetime.now().strftime('%H:%M:%S')}] Pre-loading Adding data...")
X_ALL, Y_ALL     = get_adding_data()
TRAIN_X, TRAIN_Y = X_ALL[:12000], Y_ALL[:12000]
VAL_X,   VAL_Y   = X_ALL[12000:13000], Y_ALL[12000:13000]
TEST_X,  TEST_Y  = X_ALL[13000:],      Y_ALL[13000:]

train_ds = (
    tf.data.Dataset.from_tensor_slices((TRAIN_X, TRAIN_Y))
    .cache()
    .shuffle(1000, seed=42)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# --- 5. CELL ARCHITECTURE ---
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

        z            = tf.matmul(inputs, self.w_in) + \
                       (tf.matmul(prev_h, self.w_rec) * field_effect) + self.bias
        new_h_cand   = tf.nn.leaky_relu(z, alpha=0.01)
        h            = (self.h_inertia * prev_h) + ((1.0 - self.h_inertia) * new_h_cand)
        h            = tf.clip_by_value(h, -15.0, 15.0)
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

# --- 6. METRICS ---

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

# --- 7. CALLBACKS ---
class NeuroCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_x, val_y, run_id, params):
        super().__init__()
        self.val_x, self.val_y, self.run_id, self.params_ = val_x, val_y, run_id, params
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        outputs = self.model(self.val_x[:256], training=False)
        e_rank, s_val, acorr, intf, entr = calculate_metrics_gpu(outputs[1])
        val_mse = float(tf.reduce_mean(tf.square(outputs[0] - self.val_y[:256])))
        m = {"epoch": epoch + 1, "mse": val_mse, "rank": float(e_rank), 
             "sync": float(s_val), "acorr": float(acorr), "intf": float(intf), "entr": float(entr)}
        self.history.append(m)
        print(f"| Ep {epoch+1} | MSE:{val_mse:.4f} | Rank:{m['rank']:.2f} | Syn:{m['sync']:.3f} |")

# --- 8. BUILD & TRAIN ---
model = build_model(HIDDEN, BASE_STRENGTH, PERIOD, LAMBDA_SLOW, H_INERTIA, JITTER_SCALE, MODE)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0), loss=['mse', None])

checkpoint_path = os.path.join(RAM_DISK, f"best_weights_{MODE.upper()}_{SESSION_ID}.weights.h5")
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)

lr_decay = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR, verbose=1)

cb = NeuroCallback(VAL_X, VAL_Y, RUN_ID, [LAMBDA_SLOW, H_INERTIA, BASE_STRENGTH, PERIOD, JITTER_SCALE])

model.fit(train_ds, epochs=EPOCHS, validation_data=(VAL_X, VAL_Y),
          callbacks=[cb, lr_decay, checkpoint_cb, tf.keras.callbacks.TerminateOnNaN()], verbose=0)

# --- 9. SAVE & EVALUATE (OOM-Safe Handover) ---


# 1. Capture history and final validation state while in memory
history_snapshot = cb.history
final_val_mse = history_snapshot[-1]["mse"] if history_snapshot else float('nan')

# 2. Determine status based on training history
status = "success" if not math.isnan(final_val_mse) else "failed"

# 3. Final JSON Assembly (Placeholder for test_mse)
result_data = {
    "run_id": RUN_ID, 
    "session": SESSION_ID, 
    "mode": MODE, 
    "status": status,
    "test_mse": None,  # To be calculated later from saved weights
    "val_mse_final": final_val_mse,
    "parameters": {
        "lambda": LAMBDA_SLOW, 
        "inertia": H_INERTIA, 
        "strength": BASE_STRENGTH, 
        "period": PERIOD, 
        "jitter": JITTER_SCALE
    },
    "metrics": history_snapshot[-1] if history_snapshot else {}, 
    "epochs": history_snapshot
}

def numpy_fix(obj):
    if isinstance(obj, (np.integer, np.floating, np.ndarray, np.generic)):
        return obj.item() if hasattr(obj, 'item') else obj.tolist()
    return float(obj)

# 4. Write to RAM_DISK immediately
out_path = os.path.join(RAM_DISK, f"PROBE_RUN_{MODE.upper()}.json")
try:
    with open(out_path, 'w') as f:
        json.dump(result_data, f, default=numpy_fix, indent=4)
    print(f"\n[METADATA SAVED] {out_path}")
    print(f"Checkpoint remains at: {checkpoint_path}")
except Exception as e:
    print(f"JSON Error: {e}")

# 5. Clean up and exit
print(f"[{datetime.now().strftime('%H:%M:%S')}] Final VRAM reclamation...")
del model
tf.keras.backend.clear_session()
gc.collect()

sys.exit(0)