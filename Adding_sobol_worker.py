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

# --- 1. ENVIRONMENT & CONFIG ---
T = 1000
RAM_DISK = "/mnt/ramdisk"
mixed_precision.set_global_policy('float32')
tf.keras.utils.set_random_seed(42)

# --- 2. DERIVED CONSTANTS (Global Defaults) ---
HIDDEN = 64
EPOCHS = 10
BATCH_SIZE = 256  
LEARNING_RATE = 1e-3
DATA_PERCENT = 1.0  
REST_BASELINE = 1.0

# --- 3. RECONSTRUCT SOBOL TABLE ---
sobol_problem = {
    'num_vars': 5,
    'names': ['LAMBDA_SLOW', 'H_INERTIA', 'BASE_STRENGTH', 'PERIOD', 'JITTER_SCALE'],
    'bounds': [[0.001, 0.05], [0.01, 0.99], [0.01, 0.99], [T/10, T/2], [0.1, 1.2]]
}
param_values = saltelli.sample(sobol_problem, 64) 

# --- 4. DATA LOADER ---
def get_adding_data(num_samples=15000, length=1000):
    X_val = np.random.uniform(0, 1, (num_samples, length, 1))
    X_mask = np.zeros((num_samples, length, 1))
    Y = np.zeros((num_samples, 1))
    for i in range(num_samples):
        idx = np.random.choice(length, 2, replace=False)
        X_mask[i, idx, 0] = 1.0
        Y[i, 0] = np.sum(X_val[i, idx, 0])
    X = np.concatenate([X_val, X_mask], axis=-1)
    return tf.constant(X, dtype=tf.float32), tf.constant(Y, dtype=tf.float32)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Pre-loading Data...")
X_ALL, Y_ALL = get_adding_data()
num_train = int(12000 * DATA_PERCENT)
TRAIN_X, TRAIN_Y = X_ALL[:num_train], Y_ALL[:num_train]
VAL_X, VAL_Y = X_ALL[12000:13000], Y_ALL[12000:13000]

# --- 5. BATCH RUN LOOP ---
class JitteredFeedbackCell(tf.keras.layers.Layer):
    def __init__(self, units, strength, period, lambda_slow, jitter, h_inertia, rest_baseline=1.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.strength = strength
        self.period = period
        self.lambda_slow = lambda_slow
        self.jitter = jitter
        self.h_scale = [h_inertia, 1.0 - h_inertia]
        self.rest_baseline = rest_baseline
        # State: [hidden_state, global_field, oscillator_phase]
        self.state_size = [units, units, 1] 

    def build(self, input_shape):
        self.w_in = self.add_weight(
            shape=(input_shape[-1], self.units), 
            initializer="glorot_uniform", 
            name="w_in"
        )
        self.w_rec = self.add_weight(
            shape=(self.units, self.units), 
            initializer=tf.keras.initializers.Orthogonal(gain=1.0), 
            name="w_rec"
        )
        self.bias = self.add_weight(
            shape=(self.units,), 
            initializer="zeros", 
            name="bias"
        )

    def call(self, inputs, states):
        prev_h, prev_G, prev_phase = states
        
        # 1. Global Field Update (Inter-unit communication)
        half = self.units // 2
        raw_signal = tf.concat([prev_h[:, half:], prev_h[:, :half]], axis=1)
        new_G = (1.0 - self.lambda_slow) * prev_G + self.lambda_slow * raw_signal
        
        # 2. Field Normalization
        G_norm = (new_G - tf.reduce_mean(new_G, axis=-1, keepdims=True)) / \
                 (tf.math.reduce_std(new_G, axis=-1, keepdims=True) + 1e-6)
        
        # 3. Oscillatory + Jitter Modulation
        new_phase = prev_phase + (2.0 * math.pi / self.period)
        oscillator = tf.math.sin(new_phase)
        bias_signal = tf.reduce_mean(raw_signal, axis=-1, keepdims=True) - 0.1
        combined_signal = oscillator + (self.jitter * bias_signal)

        # --- THE FIELD EFFECT ---
        # Multiplicative modulation of the neural drive
        field_effect = self.rest_baseline + (self.strength * combined_signal * tf.tanh(G_norm))
        
        # 4. State Update
        # Z is scaled by the field before passing through the activation (ELU)
        z = (tf.matmul(inputs, self.w_in) + tf.matmul(prev_h, self.w_rec) + self.bias) * field_effect
        
        # Leaky integration using the H_SCALE inertia
        h = (self.h_scale[0] * prev_h) + (self.h_scale[1] * tf.nn.elu(z))
        h = tf.clip_by_value(h, -20.0, 20.0)
        
        return h, [h, new_G, new_phase]

# --- 5. TRAINING LOOP & METRICS ---
@tf.function(jit_compile=True)
def calculate_metrics_gpu(h_seq_v):
    h_final = h_seq_v[:, -1, :]
    batch_f = tf.cast(tf.shape(h_final)[0], tf.float32)

    # 1. EFFECTIVE RANK (Spectral Entropy)
    s = tf.linalg.svd(h_final, compute_uv=False) + 1e-12
    p_rank = s / (tf.reduce_sum(s) + 1e-10)
    eff_rank = tf.exp(-tf.reduce_sum(p_rank * tf.math.log(p_rank + 1e-10)))

    # 2. SYNCHRONY
    h_centered = h_final - tf.reduce_mean(h_final, axis=0)
    h_std = tf.math.reduce_std(h_final, axis=0) + 1e-8
    h_norm = h_centered / h_std
    corr_mat = tf.matmul(h_norm, h_norm, transpose_a=True) / batch_f
    sync_val = (tf.reduce_sum(tf.abs(corr_mat)) - tf.cast(HIDDEN, tf.float32)) / \
               (tf.cast(HIDDEN**2 - HIDDEN, tf.float32))

    # 3. AUTO-CORRELATION
    sample_seq = h_seq_v[0]
    s_norm_temp = (sample_seq - tf.reduce_mean(sample_seq, axis=0)) / \
                  (tf.math.reduce_std(sample_seq, axis=0) + 1e-8)
    t_corr = tf.matmul(s_norm_temp, s_norm_temp, transpose_a=True) / \
             tf.cast(tf.shape(s_norm_temp)[0], tf.float32)
    acorr_val = tf.reduce_mean(tf.abs(t_corr))

    # 4. INTERFERENCE
    mean_field = tf.reduce_mean(h_norm, axis=1, keepdims=True)
    neuron_field_cov = tf.matmul(h_norm, mean_field, transpose_a=True) / batch_f
    interference_val = tf.reduce_mean(tf.abs(neuron_field_cov))

    # 5. ACTIVATION ENTROPY  
    h_var = tf.math.reduce_variance(h_final) + 1e-10
    entropy_val = 0.5 * tf.math.log(2.0 * math.pi * math.e * h_var) / tf.math.log(2.0)

    return eff_rank, sync_val, acorr_val, interference_val, entropy_val


START_IDX, END_IDX = int(sys.argv[1]), int(sys.argv[2])

for run_id in range(START_IDX, END_IDX):
    p = param_values[run_id]
    l_slow, h_inert, b_strength, period, jitter = p
    
    # PER-RUN DERIVED CONSTANTS
    # These depend on the specific Sobol-sampled inertia
    H_SCALE = [h_inert, 1.0 - h_inert] 

    print(f"\n[INIT] RUN {run_id} | λ={l_slow:.4f}, H_Scale={H_SCALE}, Str={b_strength:.2f}")
# --- 6. BUILD MODEL ---    
    cell = JitteredFeedbackCell(
        units=HIDDEN, 
        lambda_slow=l_slow, 
        h_inertia=h_inert, 
        strength=b_strength, 
        period=period, 
        jitter=jitter
    )
    
    inputs = tf.keras.Input(shape=(T, 2))
    rnn_out, h_seq = tf.keras.layers.RNN(cell, return_sequences=True)(inputs)
    outputs = tf.keras.layers.Dense(1)(rnn_out[:, -1, :])
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, h_seq])
    
    # Track actual parameter size for the specific run
    total_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    loss_fn = tf.keras.losses.MeanSquaredError()
    epoch_history = []

    # --- 7. TRAINING LOOP ---
    for epoch in range(EPOCHS):
        total_train_loss = 0
        num_batches = 0
        
        for i in range(0, len(TRAIN_X), BATCH_SIZE):
            xb, yb = TRAIN_X[i:i+BATCH_SIZE], TRAIN_Y[i:i+BATCH_SIZE]
            with tf.GradientTape() as tape:
                preds, _ = model(xb, training=True)
                loss = loss_fn(yb, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_train_loss += loss.numpy()
            num_batches += 1
        
        # Validation and Metrics extraction
        val_preds, val_h_seq = model(VAL_X, training=False)
        val_mse = loss_fn(VAL_Y, val_preds).numpy()
        e_rank, s_val, acorr, interf, entr = calculate_metrics_gpu(val_h_seq) 
        
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Ep {epoch+1}/{EPOCHS} | Run: {run_id} | Params: {total_params}")
        print(f"  > [STATS] Train Loss: {total_train_loss/num_batches:.4f} | Val MSE: {val_mse:.4f}")
        print(f"  > [METRICS] Rank: {e_rank:.2f} | Entropy: {entr:.2f} | Sync: {s_val:.4f}")
        print("-" * 70)

        epoch_history.append({
            "epoch": epoch+1, 
            "mse": float(val_mse), 
            "rank": float(e_rank),
            "sync": float(s_val), 
            "acorr": float(acorr), 
            "intf": float(interf),
            "entropy": float(entr),
            "params": int(total_params)
        })
    # Save to RAM
    out_file = f"{RAM_DISK}/SOBOL_RUN_{run_id}.json"
    with open(out_file, 'w') as f:
        json.dump({"run_id": run_id, "params": p.tolist(), "history": epoch_history}, f)

    tf.keras.backend.clear_session()
    gc.collect()

sys.exit(0)