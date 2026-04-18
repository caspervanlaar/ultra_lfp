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

# --- 1. ENVIRONMENT & CONFIG ---
T = 1000
RAM_DISK = "/mnt/ramdisk"
mixed_precision.set_global_policy('float32')
tf.keras.utils.set_random_seed(42)

HIDDEN = 128
EPOCHS = 150
BATCH_SIZE = 256  
LEARNING_RATE = 2e-4  
DATA_PERCENT = 1  
N_baseline = 8

# --- 2. SOBOL SETUP ---
sobol_problem = {
    'num_vars': 5,
    'names': ['LAMBDA_SLOW', 'H_INERTIA', 'BASE_STRENGTH', 'PERIOD', 'JITTER_SCALE'],
    'bounds': [
        [0.0001, 0.0015],  # LAMBDA_SLOW: Tightened. 0.002 exploded; stay below that.
        [0.92, 0.99],      # H_INERTIA: 0.50 is too low. 0.92-0.99 captures the "survival cliff."
        [0.02, 0.15],      # BASE_STRENGTH: 0.25 is very high; 0.076 was golden. 0.15 is plenty.
        [T*1.0, T*5.0],    # PERIOD: Ensure at least 1 full sequence length per oscillation.
        [0.01, 0.10]       # JITTER_SCALE: Slightly wider to test the rank-diversity limit.
    ]
}
param_values = saltelli.sample(sobol_problem, N_baseline) 

# --- 3. DATA LOADER ---
def get_adding_data(num_samples=15000, length=T):
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
TRAIN_X, TRAIN_Y = X_ALL[:12000], Y_ALL[:12000]
VAL_X, VAL_Y = X_ALL[12000:13000], Y_ALL[12000:13000]

# --- 4. UPDATED CELL ---
class JitteredFeedbackCell(tf.keras.layers.Layer):
    def __init__(self, units, strength, period, lambda_slow, jitter, h_inertia, rest_baseline=1.0, **kwargs):
        super().__init__(**kwargs)
        self.units, self.strength, self.period = units, strength, period
        self.lambda_slow, self.jitter, self.h_inertia = lambda_slow, jitter, h_inertia
        self.rest_baseline = rest_baseline
        self.state_size = [units, units, 1] 

    def build(self, input_shape):
        # Standard Glorot/He to get more initial energy than 0.01
        self.w_in = self.add_weight(shape=(input_shape[-1], self.units), 
                                    initializer="glorot_uniform", 
                                    name="w_in")
        # ORTHOGONAL is the key to Rank > 1.3
        self.w_rec = self.add_weight(shape=(self.units, self.units), 
                                    initializer=tf.keras.initializers.Orthogonal(gain=1.1), 
                                    name="w_rec")
        self.bias = self.add_weight(
        shape=(self.units,), 
        initializer=tf.keras.initializers.RandomNormal(stddev=0.01), 
        name="bias"
        )
        self.neuron_gain = self.add_weight(
            shape=(self.units,), 
            initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.1),
            trainable=True, 
            name="n_gain"
        )

    def call(self, inputs, states):
        prev_h, prev_G, prev_phase = states
        half = self.units // 2
        raw_signal = tf.concat([prev_h[:, half:], prev_h[:, :half]], axis=1)
        new_G = (1.0 - self.lambda_slow) * prev_G + self.lambda_slow * raw_signal
        
        G_norm = (new_G - tf.reduce_mean(new_G, axis=-1, keepdims=True)) / (tf.math.reduce_std(new_G, axis=-1, keepdims=True) + 1e-6)
        new_phase = prev_phase + (2.0 * math.pi / self.period)
        
        bias_sig = tf.reduce_mean(raw_signal, axis=-1, keepdims=True) - 0.1
        combined_signal = tf.math.sin(new_phase) + (self.jitter * bias_sig)

        # Apply modulation with neuron-specific gains to shatter synchrony
        field_effect = (self.rest_baseline + (self.strength * combined_signal * tf.tanh(G_norm))) * self.neuron_gain
        field_effect = tf.clip_by_value(field_effect, 0.1, 5.0)
        
        # Instead of ELU, try a gated-like update or pure tanh
        z = tf.matmul(inputs, self.w_in) + (tf.matmul(prev_h, self.w_rec) * field_effect) + self.bias
        new_h_candidate = tf.nn.leaky_relu(z, alpha=0.01)
        h = (self.h_inertia * prev_h) + ((1.0 - self.h_inertia) * new_h_candidate)
        
        return tf.clip_by_value(h, -15.0, 15.0), [h, new_G, new_phase]

@tf.function(jit_compile=True)
def calculate_metrics_gpu(h_seq_v):
    h_seq_v = h_seq_v[:256] 
    h_final = h_seq_v[:, -1, :]
    batch_f = tf.cast(tf.shape(h_final)[0], tf.float32)
    hidden_f = tf.cast(tf.shape(h_final)[1], tf.float32)

    s = tf.linalg.svd(h_final, compute_uv=False) + 1e-12
    p_rank = s / (tf.reduce_sum(s) + 1e-10)
    eff_rank = tf.exp(-tf.reduce_sum(p_rank * tf.math.log(p_rank + 1e-10)))
    h_var = tf.math.reduce_variance(h_final) + 1e-10
    entropy_val = 0.5 * tf.math.log(2.0 * math.pi * math.e * h_var) / tf.math.log(2.0)

    h_norm = (h_final - tf.reduce_mean(h_final, axis=0)) / (tf.math.reduce_std(h_final, axis=0) + 1e-8)
    sync_val = (tf.reduce_sum(tf.abs(tf.matmul(h_norm, h_norm, transpose_a=True) / batch_f)) - hidden_f) / (hidden_f**2 - hidden_f)

    sample_seq = h_seq_v[0]
    s_norm = (sample_seq - tf.reduce_mean(sample_seq, axis=0)) / (tf.math.reduce_std(sample_seq, axis=0) + 1e-8)
    t_corr = tf.matmul(s_norm, s_norm, transpose_a=True) / tf.cast(tf.shape(s_norm)[0], tf.float32)
    acorr_val = tf.reduce_mean(tf.abs(t_corr))

    mean_field = tf.reduce_mean(h_norm, axis=1, keepdims=True)
    neuron_field_cov = tf.matmul(h_norm, mean_field, transpose_a=True) / batch_f
    intf_val = tf.reduce_mean(tf.abs(neuron_field_cov))

    return eff_rank, sync_val, acorr_val, intf_val, entropy_val

class NeuroCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_x, run_id, total_params, params):
        super().__init__()
        self.val_x = val_x
        self.run_id = run_id
        self.total_params = total_params
        self.p = params
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        _, val_h_seq = self.model(self.val_x[:256], training=False)
        e_rank, s_val, acorr, intf, entr = calculate_metrics_gpu(val_h_seq)
        val_mse = logs.get("val_loss", 0.0)
        
        m = {
            "epoch": epoch + 1, "mse": float(val_mse), "rank": float(e_rank),
            "sync": float(s_val), "acorr": float(acorr), "intf": float(intf), "entropy": float(entr)
        }
        self.history.append(m)
        
        ts = datetime.now().strftime("%H:%M:%S") 
        print(f"\n| RUN {self.run_id} [{ts}] Ep {epoch+1}")       
        print(f"| Lam:{l_slow:6.3f} | H_in:{h_inert:6.3f} | Str: {b_strength:6.3f} | Per:{period:6.1f} | Jit:{jitter:5.3f} |")
        print(f"| MSE:{val_mse:6.3f} | Rank:{e_rank:6.3f} | Entr:{entr:6.3f} |")
        print(f"| Syn:{s_val:6.3f} | ACor:{acorr:6.3f} | Intf:{intf:6.3f} |")

# --- 6. RUN LOOP ---
START_IDX, END_IDX = int(sys.argv[1]), int(sys.argv[2])
train_ds = tf.data.Dataset.from_tensor_slices((TRAIN_X, TRAIN_Y)).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

for run_id in range(START_IDX, END_IDX):
    p = param_values[run_id]
    l_slow, h_inert, b_strength, period, jitter = p

    cell = JitteredFeedbackCell(units=HIDDEN, strength=b_strength, period=period, 
                                lambda_slow=l_slow, jitter=jitter, h_inertia=h_inert)
    inputs = tf.keras.Input(shape=(T, 2))
    h_seq = tf.keras.layers.RNN(cell, return_sequences=True)(inputs)
    outputs = tf.keras.layers.Dense(1)(h_seq[:, -1, :]) 
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, h_seq])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0), 
        loss=['mse', None])
    
    # --- ADDED CALLBACKS ---
    # 1. Decay LR by half if validation loss doesn't improve for 10 epochs
    lr_decay = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=10, 
        min_lr=1e-6, 
        verbose=1
    )
    
    # 2. Stop immediately if NaNs appear (saves time/resources)
    nan_guard = tf.keras.callbacks.TerminateOnNaN()
    
    cb = NeuroCallback(VAL_X, run_id, model.count_params(), p)
    
    # Added the new callbacks to the list
    model.fit(
        train_ds, 
        epochs=EPOCHS, 
        validation_data=(VAL_X, VAL_Y), 
        callbacks=[cb, lr_decay, nan_guard], 
        verbose=0
    )

    # Status check for JSON output
    final_mse = cb.history[-1]["mse"] if cb.history else 999
    status = "complete" if final_mse < 0.1 else "failed"
    if math.isnan(final_mse):
        status = "exploded"

    result = {
        "run_id": int(run_id), 
        "status": status,
        "parameters": {"lambda": float(l_slow), "inertia": float(h_inert), "strength": float(b_strength), "period": float(period), "jitter": float(jitter)},
        "epochs": cb.history
    }
    
    with open(os.path.join(RAM_DISK, f"SOBOL_RUN_{run_id}.json"), 'w') as f:
        json.dump(result, f, indent=4)

    del model, cb
    tf.keras.backend.clear_session()
    gc.collect()

print("Batch complete.")
sys.exit(0)