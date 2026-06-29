import sys
import os
import gc
import json
import math
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import mixed_precision
from SALib.sample import saltelli

# --- 1. ENVIRONMENT & SEEDING ---
def setup_environment():
    conda_cuda_path = os.path.join(os.environ.get('CONDA_PREFIX', ''), "Library", "bin")
    if os.path.exists(conda_cuda_path):
        os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={conda_cuda_path}"

    mixed_precision.set_global_policy('float32')
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    try:
        tf.config.experimental.enable_op_determinism(False)
    except Exception:
        pass

    os.environ['PYTHONHASHSEED'] = str(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)

setup_environment()

# --- 2. COMMAND LINE ARGUMENTS ---
try:
    RUN_ID     = int(sys.argv[1])
    SESSION_ID = sys.argv[2]
except (IndexError, ValueError):
    print("Usage: python CIFAR_sobol_worker.py <run_id> <session_id>")
    sys.exit(1)

# --- 3. SOBOL PROBLEM ---
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
N_baseline   = 32
param_values = saltelli.sample(sobol_problem, N_baseline)

p             = param_values[RUN_ID]
LAMBDA_SLOW   = float(p[0])
H_INERTIA     = float(p[1])
BASE_STRENGTH = float(p[2])
PERIOD        = float(p[3])
JITTER_SCALE  = float(p[4])

# --- 4. HYPERPARAMETERS ---
INPUT_DIM     = 3
NUM_CLASSES   = 10
HIDDEN        = 64
EPOCHS        = 10
BATCH_SIZE    = 128
LEARNING_RATE = 5e-3
DATA_PERCENT  = 0.1
REST_BASELINE = 1.0
PATIENCE      = 2
RAM_DISK      = "/mnt/ramdisk"

# --- 5. DATA LOADER ---
def get_cifar_sequential():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

    x_train = x_train.reshape(-1, T, INPUT_DIM)
    x_test  = x_test.reshape(-1, T, INPUT_DIM)

    y_train = y_train.flatten().astype('int32')
    y_test  = y_test.flatten().astype('int32')

    num_train = int(len(x_train) * DATA_PERCENT)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train[:num_train], y_train[:num_train]))
        .shuffle(10000, seed=42)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_subset_x = tf.constant(x_test[:256], dtype=tf.float32)
    val_subset_y = tf.constant(y_test[:256], dtype=tf.int32)
    val_fit_x = tf.constant(x_test[:1000], dtype=tf.float32)
    val_fit_y = tf.constant(y_test[:1000], dtype=tf.int32)

    return train_ds, val_subset_x, val_subset_y, val_fit_x, val_fit_y

print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading CIFAR-10...")
train_ds, val_subset_x, val_subset_y, val_fit_x, val_fit_y = get_cifar_sequential()

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
        self.w_in  = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer="glorot_uniform", name="w_in")
        # Gain 1.1 matches Adding task for better energy
        self.w_rec = self.add_weight(shape=(self.units, self.units),
                                     initializer=tf.keras.initializers.Orthogonal(gain=1.1),
                                     name="w_rec")
        self.bias  = self.add_weight(shape=(self.units,), 
                                     initializer=tf.keras.initializers.RandomNormal(stddev=0.01), 
                                     name="bias")
        # The key to breaking synchrony collapse
        self.neuron_gain = self.add_weight(shape=(self.units,), 
                                           initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.1),
                                           trainable=True, name="n_gain")

    def call(self, inputs, states):
        prev_h, prev_G, prev_phase = states

        half          = self.units // 2
        raw_signal    = tf.concat([prev_h[:, half:], prev_h[:, :half]], axis=1)
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
            # Apply neuron-specific gain and clip the field effect
            field_effect = (self.rest_baseline + (self.strength * combined_signal * tf.tanh(G_norm))) * self.neuron_gain
            field_effect = tf.clip_by_value(field_effect, 0.1, 5.0)
        else:
            field_effect = tf.ones_like(prev_h) * self.rest_baseline

        # Modulate ONLY the recurrent term, leave input alone
        z = tf.matmul(inputs, self.w_in) + (tf.matmul(prev_h, self.w_rec) * field_effect) + self.bias
        
        # Leaky ReLU and wider bounds prevent rank collapse
        new_h_candidate = tf.nn.leaky_relu(z, alpha=0.01)
        h = (self.h_inertia * prev_h) + ((1.0 - self.h_inertia) * new_h_candidate)
        h = tf.clip_by_value(h, -15.0, 15.0)

        return h, [h, new_G, new_phase]

def build_model(hidden, num_classes, strength, period, lambda_slow, h_inertia, jitter_scale):
    cell_kwargs = dict(
        units=hidden, strength=strength, period=period,
        lambda_slow=lambda_slow, jitter_scale=jitter_scale,
        h_inertia=h_inertia, mode="active"
    )
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
    acorr_val  = tf.reduce_mean(tf.abs(t_corr))

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
print(f"\n[RUN {RUN_ID}] lam={LAMBDA_SLOW:.4f} | h_in={H_INERTIA:.3f} | "
      f"str={BASE_STRENGTH:.3f} | per={PERIOD:.1f} | jit={JITTER_SCALE:.3f}")

model = build_model(
    hidden=HIDDEN, num_classes=NUM_CLASSES,
    strength=BASE_STRENGTH, period=PERIOD,
    lambda_slow=LAMBDA_SLOW, h_inertia=H_INERTIA,
    jitter_scale=JITTER_SCALE
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
    loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), None],
    metrics=[['sparse_categorical_accuracy'], []] 
)

# Passed val_y for manual fallback accuracy calculation
cb        = NeuroCallback(val_subset_x, val_subset_y, RUN_ID, p)
lr_decay  = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=PATIENCE, min_lr=1e-6, verbose=1
)
nan_guard = tf.keras.callbacks.TerminateOnNaN()

model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=(val_fit_x, val_fit_y),
    callbacks=[cb, lr_decay, nan_guard],
    verbose=0
)

# --- 10. SAVE RESULTS ---
final_acc  = cb.history[-1]["acc"]  if cb.history else 0.0
final_loss = cb.history[-1]["loss"] if cb.history else float('nan')

if math.isnan(final_loss) or math.isinf(final_loss):
    status = "exploded"
elif final_acc < 0.15:
    status = "failed"
else:
    status = "complete"

result_data = {
    "run_id":           RUN_ID,
    "session":          SESSION_ID,
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
    "intf":   cb.history[-1]["intf"]  if cb.history else 0.0,
    "entr":   cb.history[-1]["entr"]  if cb.history else 0.0,
    "epochs": cb.history
}

print(f"\n[STATUS] Run {RUN_ID}: {status} | Epochs: {len(cb.history)} | Final Acc: {final_acc:.4f}")

def numpy_fix(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    return float(obj)

out_path = os.path.join(RAM_DISK, f"SOBOL_RUN_{RUN_ID}.json")
try:
    with open(out_path, 'w') as f:
        json.dump(result_data, f, default=numpy_fix, indent=4)
    print(f"Result saved: {out_path}")
except Exception as e:
    print(f"Error saving JSON: {e}")

del model, cb
tf.keras.backend.clear_session()
gc.collect()
sys.exit(0)