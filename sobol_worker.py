import sys
import os
import gc
import json
import math
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import mixed_precision

# --- 1. ENVIRONMENT & SEEDING ---
def setup_environment():
    # Standard Conda path for libdevice (XLA support)
    conda_cuda_path = os.path.join(os.environ.get('CONDA_PREFIX', ''), "Library", "bin")
    if os.path.exists(conda_cuda_path):
        os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={conda_cuda_path}"
    
    # Mixed precision for performance
    mixed_precision.set_global_policy('float32')
    
    # Disable strict determinism to allow GPU kernels to run
    os.environ['TF_DETERMINISTIC_OPS'] = '0' 
    try:
        tf.config.experimental.enable_op_determinism(False)
    except:
        pass

    # Hard-lock all seeds for reproducible permutations
    os.environ['PYTHONHASHSEED'] = str(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42) 

setup_environment()

# --- 2. COMMAND LINE ARGUMENTS ---
try:
    RUN_ID        = sys.argv[1]
    SESSION_ID    = sys.argv[2]
    LAMBDA_SLOW   = float(sys.argv[3])
    H_INERTIA     = float(sys.argv[4])
    BASE_STRENGTH = float(sys.argv[5])
    PERIOD        = float(sys.argv[6])
    JITTER_SCALE  = float(sys.argv[7])
except IndexError:
    print("Usage: python sobol_worker.py <RUN_ID> <SESSION_ID> <LAMBDA> <INERTIA> <STRENGTH> <PERIOD> <JITTER>")
    sys.exit(1)

# Derived Constants
HIDDEN = 32
EPOCHS = 10
BATCH_SIZE = 512
DATA_PERCENT = 0.1
LEARNING_RATE = 5e-3
REST_BASELINE = 1.0
H_SCALE = [H_INERTIA, 1.0 - H_INERTIA]

# --- 3. DATA LOADER ---
def get_pmnist_data():
    (x_all, y_all), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_all = x_all.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    # Fixed Permutation
    #rng = np.random.RandomState(42)
    #perm = rng.permutation(784)
    #x_all = x_all[:, perm]
    #x_test = x_test[:, perm]

    # Train/Val Split (90/10)
    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.1, random_state=42, stratify=y_all
    )

    x_train = x_train[:, :, np.newaxis]
    x_val   = x_val[:, :, np.newaxis]
    
    y_train = y_train.astype('int32')
    y_val   = y_val.astype('int32')

    num_train = int(len(x_train) * DATA_PERCENT)
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train[:num_train], y_train[:num_train])) \
        .shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # 1000 sample fixed subset for rapid validation metrics
    val_ds = tf.data.Dataset.from_tensor_slices((x_val[:1000], y_val[:1000])) \
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # We need a single batch representation for the metric extractions
    val_subset_x, val_subset_y = next(iter(val_ds.unbatch().batch(256)))

    return train_ds, val_ds, val_subset_x, val_subset_y

train_ds, val_ds, val_subset_x, val_subset_y = get_pmnist_data()


# --- 4. MODEL DEFINITION ---
class JitteredFeedbackCell(tf.keras.layers.Layer):
    def __init__(self, units, strength, period, lambda_slow, mode="active", **kwargs):
        super().__init__(**kwargs)
        self.units, self.strength, self.period, self.lambda_slow, self.mode = units, strength, period, lambda_slow, mode
        self.state_size = [units, units, 1] 

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
        self.rnn1 = tf.keras.layers.RNN(JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode), return_sequences=True)
        self.rnn2 = tf.keras.layers.RNN(JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode), return_sequences=True)
        self.rnn3 = tf.keras.layers.RNN(JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode), return_sequences=True)
        self.out = tf.keras.layers.Dense(num_classes, dtype='float32')

    def call(self, x, training=False):
        h1 = self.rnn1(x, training=training)
        h2 = self.rnn2(h1, training=training)
        h3 = self.rnn3(h2, training=training)
        return self.out(h3[:, -1, :]), h3

# --- 5. TRAINING LOOP & METRICS ---
model = OscillatingResonator(hidden=HIDDEN, strength=BASE_STRENGTH)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits, _ = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y, logits)
    return loss

@tf.function(jit_compile=True)
def calculate_metrics_gpu(x_val, y_val):
    logits_v, h_seq_v = model(x_val, training=False)
    h_final = h_seq_v[:, -1, :]
    batch_f = tf.cast(tf.shape(h_final)[0], tf.float32)

    # 1. Effective Rank - GPU SVD
    s = tf.linalg.svd(h_final, compute_uv=False) + 1e-12
    p_rank = s / (tf.reduce_sum(s) + 1e-10)
    eff_rank = tf.exp(-tf.reduce_sum(p_rank * tf.math.log(p_rank + 1e-10)))

    # 2. Synchrony
    h_centered = h_final - tf.reduce_mean(h_final, axis=0)
    h_std = tf.math.reduce_std(h_final, axis=0) + 1e-8
    h_norm = h_centered / h_std
    corr_mat = tf.matmul(h_norm, h_norm, transpose_a=True) / batch_f
    sync_val = (tf.reduce_sum(tf.abs(corr_mat)) - tf.cast(HIDDEN, tf.float32)) / tf.cast(HIDDEN**2 - HIDDEN, tf.float32)

    # 3. Auto-Correlation
    sample_seq = h_seq_v[0]
    s_norm = (sample_seq - tf.reduce_mean(sample_seq, axis=0)) / (tf.math.reduce_std(sample_seq, axis=0) + 1e-8)
    t_corr = tf.matmul(s_norm, s_norm, transpose_a=True) / tf.cast(tf.shape(s_norm)[0], tf.float32)
    acorr_val = tf.reduce_mean(tf.abs(t_corr))

    # 4. Interference
    mean_field = tf.reduce_mean(h_norm, axis=1, keepdims=True)
    neuron_field_cov = tf.matmul(h_norm, mean_field, transpose_a=True) / batch_f
    interference_val = tf.reduce_mean(tf.abs(neuron_field_cov))

    return eff_rank, sync_val, acorr_val, interference_val, h_final, logits_v

def calculate_entropy_cpu(h_final_np):
    counts, _ = np.histogram(h_final_np, bins=50)
    p_ent = counts / (h_final_np.size + 1e-10)
    return float(-np.sum(p_ent * np.log2(p_ent + 1e-10)))

@tf.function
def get_val_acc(y_val, logits_v):
    val_acc_metric.update_state(y_val, logits_v)
    return val_acc_metric.result()

# --- 6. EXECUTION ---
epoch_history = []
for epoch in range(EPOCHS):
    train_loss_metric.reset_state()
    train_acc_metric.reset_state()
    val_acc_metric.reset_state()

    for x_b, y_b in train_ds:
        train_step(x_b, y_b)

    e_rank, s_val, acorr, interf, h_final, logits_v = calculate_metrics_gpu(val_subset_x, val_subset_y)
    v_acc = float(get_val_acc(val_subset_y, logits_v).numpy())
    entr = calculate_entropy_cpu(h_final.numpy())
    e_rank, s_val, acorr, interf = [float(t.numpy()) for t in [e_rank, s_val, acorr, interf]]

    final_loss = float(train_loss_metric.result().numpy())

    current_snapshot = {
        "epoch": epoch + 1,
        "loss": final_loss,
        "acc": v_acc, "rank": e_rank, "sync": s_val,
        "entr": entr, "acorr": acorr, "intf": interf
    }
    epoch_history.append(current_snapshot)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Epoch {epoch+1}/{EPOCHS} | Run: {RUN_ID}")
    print(f"  > [STATS] Loss: {final_loss:.4f} | Acc: {v_acc:.4f} | Rank: {e_rank:.2f} | Sync: {s_val:.4f}")
    print(f"  > [EXTRA] Entr: {entr:.4f} | ACorr: {acorr:.4f} | Intf: {interf:.4f}")
    print(f"{'-'*70}")

    if np.isnan(final_loss) or e_rank < 1.01:
        print(f"!!! CRITICAL INSTABILITY DETECTED !!!")
        break

# --- 7. FINAL JSON DUMP ---
# This builds the master object for this specific Sobol run
final_loss = epoch_history[-1]["loss"]
final_acc = epoch_history[-1]["acc"]
status = "failed" if (np.isnan(final_loss) or np.isinf(final_loss) or final_acc < 0.11) else "complete"

result_data = {
    "run_id": int(RUN_ID),
    "session": SESSION_ID,
    "status": status,
    "completed_epochs": len(epoch_history),
    "parameters": {
        "lambda": float(LAMBDA_SLOW),
        "inertia": float(H_INERTIA),
        "strength": float(BASE_STRENGTH),
        "period": float(PERIOD),
        "jitter": float(JITTER_SCALE)
    },
    "acc": epoch_history[-1]["acc"],
    "rank": epoch_history[-1]["rank"],
    "sync": epoch_history[-1]["sync"],
    "entr": epoch_history[-1]["entr"],
    "acorr": epoch_history[-1]["acorr"],
    "intf": epoch_history[-1]["intf"],
    "epochs": epoch_history
}

print(f"[STATUS] Run {RUN_ID}: {status} | Epochs: {len(epoch_history)} | Final Acc: {final_acc:.4f}")

def numpy_fix(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    return float(obj)

try:
    with open(f"SOBOL_RUN_{RUN_ID}.json", 'w') as f:
        json.dump(result_data, f, default=numpy_fix, indent=4)
    print(f"File saved: SOBOL_RUN_{RUN_ID}.json")
except Exception as e:
    print(f"Error saving JSON: {e}")

del model
gc.collect()
sys.exit(0)
