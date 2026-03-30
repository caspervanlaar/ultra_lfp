{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "72ca62dc",
   "metadata": {},
   "source": [
    "## Liberaries and data settup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "32112755",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "from tqdm import tqdm\n",
    "import time\n",
    "import os\n",
    "import os\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras import mixed_precision\n",
    "import gc\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "from tqdm import tqdm\n",
    "import scipy.linalg\n",
    "import math\n",
    "import os, gc, random\n",
    "import matplotlib.pyplot as plt\n",
    "import json\n",
    "import pickle\n",
    "import time\n",
    "from datetime import datetime\n",
    "\n",
    "tf.keras.mixed_precision.set_global_policy('float32')\n",
    "\n",
    "# Note: Your model will now output float16. \n",
    "# Ensure your last layer is float32 for stability:\n",
    "# self.out = tf.keras.layers.Dense(num_classes, dtype='float32')\n",
    "\n",
    "# Standard Conda path for libdevice\n",
    "conda_cuda_path = os.path.join(os.environ.get('CONDA_PREFIX', ''), \"Library\", \"bin\")\n",
    "if os.path.exists(conda_cuda_path):\n",
    "    os.environ['XLA_FLAGS'] = f\"--xla_gpu_cuda_data_dir={conda_cuda_path}\"\n",
    "    print(f\"XLA Path set to: {conda_cuda_path}\")\n",
    "\n",
    "# EMERGENCY BYPASS: If it still fails, disable JIT for now. \n",
    "# It will be slightly slower, but it will NOT crash.\n",
    "USE_JIT = False\n",
    "\n",
    "import os\n",
    "os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=\"C:/Users/caspe/anaconda3/envs/SPIKEDETEC/Library/bin\"'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "458125e2",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "I0000 00:00:1775302859.325166    4464 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 2156 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3050 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "pMNIST Ready. Train: (54000, 784, 1), Val: (6000, 784, 1), Test: (10000, 784, 1)\n"
     ]
    }
   ],
   "source": [
    "# 1. Load MNIST\n",
    "(x_all, y_all), (x_test, y_test) = tf.keras.datasets.mnist.load_data()\n",
    "\n",
    "# 2. Flatten and Normalize (Standard step)\n",
    "x_all = x_all.reshape(-1, 784).astype('uint8') / 255.0\n",
    "x_test = x_test.reshape(-1, 784).astype('uint8') / 255.0\n",
    "\n",
    "# 3. Create FIXED Permutation (The \"p\" in pMNIST)\n",
    "rng = np.random.RandomState(42)\n",
    "perm = rng.permutation(784)\n",
    "\n",
    "x_all = x_all[:, perm]\n",
    "x_test = x_test[:, perm]\n",
    "\n",
    "# --- NEW: Train/Val Split (90% Train, 10% Val) ---\n",
    "from sklearn.model_selection import train_test_split\n",
    "x_train, x_val, y_train, y_val = train_test_split(\n",
    "    x_all, y_all, test_size=0.1, random_state=42, stratify=y_all\n",
    ")\n",
    "\n",
    "# 4. Reshape to [Batch, Time, Channels] for your RNN\n",
    "x_train = x_train[:, :, np.newaxis]\n",
    "x_val   = x_val[:, :, np.newaxis]\n",
    "x_test  = x_test[:, :, np.newaxis]\n",
    "\n",
    "# 5. Labels to int32\n",
    "y_train = y_train.astype('int32')\n",
    "y_val   = y_val.astype('int32')\n",
    "y_test  = y_test.astype('int32')\n",
    "\n",
    "# 6. Build datasets\n",
    "train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(256)\n",
    "val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(256)\n",
    "test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)\n",
    "\n",
    "print(f\"pMNIST Ready. Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfe41b19",
   "metadata": {},
   "source": [
    "## Clean environment and drift calc\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "44b6aa62",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Warning: Could not toggle flag. If Phase 1 fails, restart kernel.\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "# --- 1. CONFIGURATION ---\n",
    "\n",
    "\n",
    "# Forcefully disable the strict determinism flag at the TF level\n",
    "try:\n",
    "    tf.config.experimental.enable_op_determinism(False)\n",
    "    print(\"Success: Strict determinism disabled.\")\n",
    "except:\n",
    "    print(\"Warning: Could not toggle flag. If Phase 1 fails, restart kernel.\")\n",
    "def reset_seeds():\n",
    "    import os\n",
    "    import gc\n",
    "    import random\n",
    "    import numpy as np\n",
    "\n",
    "    # 1. Clear session\n",
    "    tf.keras.backend.clear_session()\n",
    "    \n",
    "    # 2. Disable strict determinism to allow GPU kernels to run\n",
    "    os.environ['TF_DETERMINISTIC_OPS'] = '0' \n",
    "    \n",
    "    # 3. Hard-lock all seeds\n",
    "    os.environ['PYTHONHASHSEED'] = str(42)\n",
    "    random.seed(42)\n",
    "    np.random.seed(42)\n",
    "    tf.random.set_seed(42)\n",
    "    tf.keras.utils.set_random_seed(42) \n",
    "    \n",
    "    gc.collect()\n",
    "    print(\"Environment Reset: Seeds locked at 42. Strict determinism disabled.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "839e114c",
   "metadata": {},
   "source": [
    "## Variables etc"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "644e2f11",
   "metadata": {},
   "source": [
    "## Parameter count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "25c87a42",
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_param_report(hidden_size, input_dim=1, num_classes=10):\n",
    "    \"\"\"\n",
    "    Calculates and prints a detailed breakdown of parameters for the \n",
    "    OscillatingResonator architecture.\n",
    "    \"\"\"\n",
    "    # RNN Layer 1: Inputs from data\n",
    "    rnn1_params = (input_dim * hidden_size) + (hidden_size * hidden_size) + hidden_size\n",
    "    \n",
    "    # RNN Layer 2 & 3: Inputs from previous hidden layer\n",
    "    rnn_mid_params = (hidden_size * hidden_size) + (hidden_size * hidden_size) + hidden_size\n",
    "    \n",
    "    # Final Dense Layer\n",
    "    dense_params = (hidden_size * num_classes) + num_classes\n",
    "    \n",
    "    total = rnn1_params + (2 * rnn_mid_params) + dense_params\n",
    "    \n",
    "    print(f\"--- PARAMETER COUNT REPORT (Hidden: {hidden_size}) ---\")\n",
    "    print(f\"RNN Layer 1: {rnn1_params:>8,}\")\n",
    "    print(f\"RNN Layer 2: {rnn_mid_params:>8,}\")\n",
    "    print(f\"RNN Layer 3: {rnn_mid_params:>8,}\")\n",
    "    print(f\"Dense Out:   {dense_params:>8,}\")\n",
    "    print(\"-\" * 35)\n",
    "    print(f\"TOTAL PARAMS: {total:>8,}\")\n",
    "    print(f\"Estimated Memory: {total * 4 / 1024:.2f} KB (float32)\")\n",
    "    print(\"=\" * 35 + \"\\n\")\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "43329e12",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " SESSION INITIALIZED: RES_H16_S0.22_J0.65_260404_2241\n",
      "--- PARAMETER COUNT REPORT (Hidden: 16) ---\n",
      "RNN Layer 1:      288\n",
      "RNN Layer 2:      528\n",
      "RNN Layer 3:      528\n",
      "Dense Out:        170\n",
      "-----------------------------------\n",
      "TOTAL PARAMS:    1,514\n",
      "Estimated Memory: 5.91 KB (float32)\n",
      "===================================\n",
      "\n"
     ]
    }
   ],
   "source": [
    "DATA_PERCENT = 0.01\n",
    "BATCH_SIZE = 4 * 64\n",
    "EPOCHS = 2         \n",
    "HIDDEN = 16 \n",
    "BASE_STRENGTH = 0.22 \n",
    "PERIOD = 784/4 \n",
    "LAMBDA_SLOW = 0.015  \n",
    "JITTER_SCALE = 0.65 \n",
    "REST_BASELINE = 1.0\n",
    "LEARNING_RATE = 8e-4\n",
    "H_SCALE = [0.94, 0.06]    \n",
    "\n",
    "\n",
    "# --- 1.1 AUTO-GENERATED SESSION NAME ---\n",
    "# Creates a name like: RES_H32_S0.40_J1.15_T123456\n",
    "now = datetime.now()\n",
    "readable_ts = now.strftime(\"%y%m%d_%H%M\") \n",
    "SESSION_ID = f\"RES_H{HIDDEN}_S{BASE_STRENGTH}_J{JITTER_SCALE}_{readable_ts}\"\n",
    "print(f\" SESSION INITIALIZED: {SESSION_ID}\")\n",
    "print_param_report(HIDDEN)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f18bc1c1",
   "metadata": {},
   "source": [
    "##  Oscillatory global field block "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "74bf4173",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test Run Ready: Samples -> Train: 5400, Val: 600, Data %: 10.0\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "\n",
    "\n",
    "# --- 2. THE JITTERED CELL ---\n",
    "class JitteredFeedbackCell(tf.keras.layers.Layer):\n",
    "    def __init__(self, units=16, strength=0.0, period=256.0, lambda_slow=0.05, mode=\"active\", **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.units = units\n",
    "        self.state_size = [units, units, 1] \n",
    "        self.strength = strength \n",
    "        self.period = period\n",
    "        self.lambda_slow = lambda_slow\n",
    "        self.mode = mode\n",
    "\n",
    "    def build(self, input_shape):\n",
    "        self.w_in = self.add_weight(shape=(input_shape[-1], self.units), name=\"w_in\", initializer=\"glorot_uniform\")\n",
    "        self.w_rec = self.add_weight(shape=(self.units, self.units), name=\"w_rec\",\n",
    "                                     initializer=tf.keras.initializers.Orthogonal(gain=1.0))\n",
    "        self.bias = self.add_weight(shape=(self.units,), name=\"bias\", initializer=\"zeros\")\n",
    "\n",
    "    def call(self, inputs, states):\n",
    "        prev_h, prev_G, prev_phase = states\n",
    "        half = self.units // 2\n",
    "        raw_signal = tf.concat([prev_h[:, half:], prev_h[:, :half]], axis=1)\n",
    "        source_signal = tf.stop_gradient(raw_signal) if self.mode == \"probe\" else raw_signal\n",
    "        new_G = (1.0 - self.lambda_slow) * prev_G + self.lambda_slow * source_signal\n",
    "        G_mean = tf.reduce_mean(new_G, axis=-1, keepdims=True)\n",
    "        G_std = tf.math.reduce_std(new_G, axis=-1, keepdims=True) + 1e-6\n",
    "        G_norm = (new_G - G_mean) / G_std\n",
    "        new_phase = prev_phase + (2.0 * math.pi / self.period)\n",
    "        oscillator = tf.math.sin(new_phase)\n",
    "        \n",
    "        if self.mode == \"active\":\n",
    "            bias_signal = tf.reduce_mean(source_signal, axis=-1, keepdims=True) - 0.1\n",
    "            combined_signal = oscillator + (JITTER_SCALE * bias_signal)\n",
    "        else:\n",
    "            combined_signal = oscillator\n",
    "\n",
    "        current_strength = self.strength * combined_signal if self.mode != \"passive\" else 0.0\n",
    "        field_effect = REST_BASELINE + (current_strength * tf.tanh(G_norm))\n",
    "        z = (tf.matmul(inputs, self.w_in) + tf.matmul(prev_h, self.w_rec) + self.bias) * field_effect\n",
    "        h = (H_SCALE[0] * prev_h) + (H_SCALE[1] * tf.nn.elu(z))\n",
    "        h = tf.clip_by_value(h, -20.0, 20.0)\n",
    "        return h, [h, new_G, new_phase]\n",
    "\n",
    "class OscillatingResonator(tf.keras.Model):\n",
    "    def __init__(self, hidden=16, num_classes=10, strength=0.0, mode=\"active\"):\n",
    "        super().__init__()\n",
    "        self.cell_ref = JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode)\n",
    "        self.rnn1 = tf.keras.layers.RNN(self.cell_ref, return_sequences=True)\n",
    "        self.rnn2 = tf.keras.layers.RNN(JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode), return_sequences=True)\n",
    "        self.rnn3 = tf.keras.layers.RNN(JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode), return_sequences=True)\n",
    "        self.out = tf.keras.layers.Dense(num_classes)\n",
    "\n",
    "    def call(self, x, training=False):\n",
    "        h1 = self.rnn1(x, training=training)\n",
    "        h2 = self.rnn2(h1, training=training)\n",
    "        h3 = self.rnn3(h2, training=training)\n",
    "        return self.out(h3[:, -1, :]), h3\n",
    "\n",
    "# --- 3. UPDATED LOGGING UTILITY ---\n",
    "def print_history_summary(history, model, model_name=\"Model\", test_acc=None):\n",
    "    cell = model.rnn1.cell\n",
    "    total_params = model.count_params()\n",
    "    \n",
    "    print(f\"\\n DATA LOG: {model_name}\")\n",
    "    print(f\" CONFIG: Hidden: {cell.units} | Params: {total_params:,} | F-Wgt: {cell.strength:.4f} | \"\n",
    "          f\"L-Slow (τ): {cell.lambda_slow:.3f} | Jitter: {JITTER_SCALE:.2f} | Period: {cell.period:.1f} | Data: {DATA_PERCENT*100:.2f}%\")\n",
    "    print(\"=\"*145)\n",
    "    header = f\"{'Epoch':<6} | {'Loss':<7} | {'Val-Acc%':<8} | {'Rank':<6} | {'Sync':<6} | {'Entrp':<6} | {'A-Corr':<7} | {'Intf':<6} | {'F-Wgt':<6}\"\n",
    "    print(header)\n",
    "    print(\"-\" * 145)\n",
    "\n",
    "    for i in range(len(history['loss'])):\n",
    "        m = history['hidden_metrics'][i]\n",
    "        print(f\"{i+1:<6} | {history['loss'][i]:<7.3f} | {history['acc'][i]*100:<8.2f} | \"\n",
    "              f\"{m['effective_rank']:<6.2f} | {m['synchrony']:<6.3f} | {m['entropy']:<6.2f} | \"\n",
    "              f\"{m['a_corr']:<7.3f} | {m['interference']:<6.3f} | {cell.strength:<6.3f}\")\n",
    "    \n",
    "    print(\"-\" * 145)\n",
    "    test_str = f\"{test_acc*100:.2f}%\" if test_acc is not None else \"N/A\"\n",
    "    print(f\" FINAL PERFORMANCE: Validation Acc: {history['acc'][-1]*100:.2f}% | TEST ACCURACY: {test_str}\")\n",
    "    print(\"=\"*145 + \"\\n\")\n",
    "\n",
    "# --- 4. TRAINING PHASE WITH SNAPSHOT & LIVE LOGS ---\n",
    "# --- UPDATED TRAINING PHASE ---\n",
    "def train_phase(model, train_data, val_data, epochs=3, name=\"model\"):\n",
    "    current_lr = LEARNING_RATE\n",
    "    optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)\n",
    "    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)\n",
    "    history = {\"loss\": [], \"acc\": [], \"hidden_metrics\": []}\n",
    "    best_val_acc = 0.0\n",
    "\n",
    "    @tf.function\n",
    "    def train_step(x, y, lr):\n",
    "        optimizer.learning_rate.assign(lr)\n",
    "        with tf.GradientTape() as tape:\n",
    "            logits, _ = model(x, training=True)\n",
    "            loss_v = loss_fn(y, logits)\n",
    "        grads = tape.gradient(loss_v, model.trainable_variables)\n",
    "        grads, _ = tf.clip_by_global_norm(grads, 1.0)\n",
    "        optimizer.apply_gradients(zip(grads, model.trainable_variables))\n",
    "        return loss_v, logits\n",
    "\n",
    "    print(f\"\\n{'Epoch':<6} | {'Loss':<7} | {'Val-Acc%':<8} | {'Rank':<6} | {'Sync':<6} | {'Entrp':<6} | {'A-Corr':<7}\")\n",
    "    print(\"-\" * 75)\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()\n",
    "        epoch_loss = []\n",
    "        pbar = tqdm(train_data, desc=f\"EPOCH {epoch+1}/{epochs}\", leave=False)\n",
    "        \n",
    "        for x_b, y_b in pbar:\n",
    "            loss_v, logits = train_step(x_b, y_b, current_lr)\n",
    "            acc_metric.update_state(y_b, logits)\n",
    "            epoch_loss.append(float(loss_v))\n",
    "            pbar.set_postfix({\"loss\": f\"{np.mean(epoch_loss):.4f}\", \"acc\": f\"{acc_metric.result():.2%}\"})\n",
    "\n",
    "        # --- VALIDATION SNAPSHOT ---\n",
    "        for x_v, y_v in val_data.take(1):\n",
    "            logits_v, h_seq_v = model(x_v, training=False)\n",
    "            val_acc = np.mean(np.argmax(logits_v.numpy(), axis=-1) == y_v.numpy())\n",
    "            \n",
    "            # 1. SAVE BEST WEIGHTS\n",
    "            if val_acc > best_val_acc:\n",
    "                best_val_acc = val_acc\n",
    "                model.save_weights(f\"best_{name}_{SESSION_ID}.weights.h5\")\n",
    "                status_msg = f\" (New Best!)\"\n",
    "            else:\n",
    "                status_msg = \"\"\n",
    "            \n",
    "            # Metric Calculations\n",
    "            h_final = h_seq_v.numpy()[:, -1, :]\n",
    "            s = scipy.linalg.svdvals(h_final) + 1e-12\n",
    "            p_rank = s / (np.sum(s) + 1e-10)\n",
    "            eff_rank = np.exp(-np.sum(p_rank * np.log(p_rank + 1e-10)))\n",
    "            \n",
    "            counts, _ = np.histogram(h_final, bins=50)\n",
    "            p_ent = counts / (h_final.size + 1e-10) \n",
    "            entropy_val = -np.sum(p_ent * np.log2(p_ent + 1e-10))\n",
    "            \n",
    "            sync_val = (np.sum(np.abs(np.corrcoef(h_final.T + 1e-8))) - HIDDEN) / (HIDDEN**2 - HIDDEN)\n",
    "            acorr_val = np.mean(np.abs(np.corrcoef(h_seq_v.numpy()[0].T + 1e-8)))\n",
    "\n",
    "\n",
    "            h_final = h_seq_v.numpy()[:, -1, :]\n",
    "            pop_mean = np.mean(h_final, axis=1, keepdims=True)\n",
    "            correlations = [\n",
    "                np.corrcoef(h_final[:, i], pop_mean[:, 0])[0, 1]\n",
    "                for i in range(h_final.shape[1])\n",
    "            ]\n",
    "            interference = np.mean(np.abs(correlations))\n",
    "\n",
    "            history[\"loss\"].append(np.mean(epoch_loss))\n",
    "            history[\"acc\"].append(val_acc)\n",
    "            history[\"hidden_metrics\"].append({\n",
    "                \"effective_rank\": eff_rank,\n",
    "                \"synchrony\": sync_val,\n",
    "                \"entropy\": entropy_val,\n",
    "                \"a_corr\": acorr_val,\n",
    "                \"interference\": interference\n",
    "            })\n",
    "\n",
    "            print(f\"{epoch+1:<6} | {np.mean(epoch_loss):<7.3f} | {val_acc*100:<8.2f} | \"\n",
    "                  f\"{eff_rank:<6.2f} | {sync_val:<6.3f} | {entropy_val:<6.2f} | {acorr_val:<7.3f}{status_msg}\")\n",
    "    \n",
    "    # 2. SAVE LATEST WEIGHTS (Post-training)\n",
    "    model.save_weights(f\"latest_{name}_{SESSION_ID}.weights.h5\")\n",
    "    print(f\"--- Finished {name}: Best and Latest weights saved. ---\")\n",
    "            \n",
    "    return history\n",
    "\n",
    "# --- 5. SAVE ---\n",
    "# This map will hold everything: configs, histories, and final scores\n",
    "master_results = {\n",
    "    \"config\": {\n",
    "        \"hidden\": HIDDEN,\n",
    "        \"base_strength\": BASE_STRENGTH,\n",
    "        \"jitter\": JITTER_SCALE,\n",
    "        \"period\": PERIOD,\n",
    "        \"epochs\": EPOCHS,\n",
    "        \"data_pct\": DATA_PERCENT\n",
    "    },\n",
    "    \"runs\": {}\n",
    "}\n",
    "\n",
    "def safe_evaluate(model, dataset):\n",
    "    \"\"\"Evaluates accuracy in batches to prevent GPU OOM errors.\"\"\"\n",
    "    accs = []\n",
    "    for x_batch, y_batch in dataset:\n",
    "        logits, _ = model(x_batch, training=False)\n",
    "        preds = np.argmax(logits.numpy(), axis=-1)\n",
    "        accs.append(np.mean(preds == y_batch.numpy()))\n",
    "    return np.mean(accs)\n",
    "\n",
    "\n",
    "# --- 6. EXECUTION ---\n",
    "def reset_env():\n",
    "    tf.keras.backend.clear_session()\n",
    "    random.seed(42); np.random.seed(42); tf.random.set_seed(42)\n",
    "    gc.collect()\n",
    "\n",
    "# SETUP DATA\n",
    "num_train = int(len(x_train) * DATA_PERCENT)\n",
    "num_val = int(len(x_val)*DATA_PERCENT)\n",
    "num_test  = int(len(x_test) * DATA_PERCENT)\n",
    "\n",
    "train_ds = tf.data.Dataset.from_tensor_slices((x_train[:num_train], y_train[:num_train])) \\\n",
    "    .shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)\n",
    "\n",
    "val_ds = tf.data.Dataset.from_tensor_slices((x_val[:1000], y_val[:1000])) \\\n",
    "    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)\n",
    "\n",
    "test_ds_subset = tf.data.Dataset.from_tensor_slices((x_test[:num_test], y_test[:num_test])) \\\n",
    "    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)\n",
    "\n",
    "print(f\"Test Run Ready: Samples -> Train: {num_train}, Val: {num_val}, Data %: {DATA_PERCENT*100}\")\n",
    "\n",
    "# --- RUN 1: ACTIVE ---\n",
    "# --- 5. EXECUTION & UNIQUE SAVING ---\n",
    "\n",
    "master_results = {\n",
    "    \"session_id\": SESSION_ID,\n",
    "    \"config\": {\n",
    "        \"hidden\": HIDDEN, \"base_strength\": BASE_STRENGTH,\n",
    "        \"jitter\": JITTER_SCALE, \"period\": PERIOD, \"epochs\": EPOCHS\n",
    "    },\n",
    "    \"runs\": {}\n",
    "}\n",
    "\n",
    "def save_metrics_to_files(history, model, model_name, test_acc):\n",
    "    cell = model.rnn1.cell\n",
    "    total_params = model.count_params()\n",
    "    \n",
    "    # 1. GENERATE THE TEXT TABLE \n",
    "    log_lines = []\n",
    "    log_lines.append(f\"\\n DATA LOG: {model_name}\")\n",
    "    log_lines.append(f\" CONFIG: Hidden: {cell.units} | Params: {total_params:,} | F-Wgt: {cell.strength:.4f} | \"\n",
    "                     f\"L-Slow (τ): {cell.lambda_slow:.3f} | Jitter: {JITTER_SCALE:.2f} | Period: {cell.period:.1f} | Data: {DATA_PERCENT*100:.2f}%\")\n",
    "    log_lines.append(\"=\"*145)\n",
    "    header = f\"{'Epoch':<6} | {'Loss':<7} | {'Val-Acc%':<8} | {'Rank':<6} | {'Sync':<6} | {'Entrp':<6} | {'A-Corr':<7} | {'Intf':<6} | {'F-Wgt':<6}\"\n",
    "    log_lines.append(header)\n",
    "    log_lines.append(\"-\" * 145)\n",
    "\n",
    "    for i in range(len(history['loss'])):\n",
    "        m = history['hidden_metrics'][i]\n",
    "        line = (f\"{i+1:<6} | {history['loss'][i]:<7.3f} | {history['acc'][i]*100:<8.2f} | \"\n",
    "                f\"{m['effective_rank']:<6.2f} | {m['synchrony']:<6.3f} | {m['entropy']:<6.2f} | \"\n",
    "                f\"{m['a_corr']:<7.3f} | {m['interference']:<6.3f} | {cell.strength:<6.3f}\")\n",
    "        log_lines.append(line)\n",
    "    \n",
    "    log_lines.append(\"-\" * 145)\n",
    "    test_str = f\"{test_acc*100:.2f}%\" if test_acc is not None else \"N/A\"\n",
    "    log_lines.append(f\" FINAL PERFORMANCE: Validation Acc: {history['acc'][-1]*100:.2f}% | TEST ACCURACY: {test_str}\")\n",
    "    log_lines.append(\"=\"*145 + \"\\n\")\n",
    "\n",
    "    # Save to TXT (Appends so you don't lose previous runs)\n",
    "    with open(f\"log_{SESSION_ID}.txt\", \"a\") as f:\n",
    "        f.write(\"\\n\".join(log_lines))\n",
    "    \n",
    "    # Print to console as usual\n",
    "    print(\"\\n\".join(log_lines))\n",
    "\n",
    "    # 2. SAVE RAW DATA TO JSON\n",
    "    json_data = {\n",
    "        \"session_id\": SESSION_ID,\n",
    "        \"model_name\": model_name,\n",
    "        \"config\": {\n",
    "            \"hidden\": cell.units, \"strength\": cell.strength, \"tau\": cell.lambda_slow,\n",
    "            \"jitter\": JITTER_SCALE, \"period\": PERIOD, \"data_pct\": DATA_PERCENT\n",
    "        },\n",
    "        \"history\": history,\n",
    "        \"test_accuracy\": float(test_acc)\n",
    "    }\n",
    "    \n",
    "    # Simple conversion for numpy types\n",
    "    def clean_types(obj):\n",
    "        if isinstance(obj, dict): return {k: clean_types(v) for k, v in obj.items()}\n",
    "        if isinstance(obj, list): return [clean_types(i) for i in obj]\n",
    "        if isinstance(obj, (np.float32, np.float64)): return float(obj)\n",
    "        if isinstance(obj, (np.int32, np.int64)): return int(obj)\n",
    "        return obj\n",
    "\n",
    "    with open(f\"results_{SESSION_ID}_{model_name}.json\", \"w\") as f:\n",
    "        json.dump(clean_types(json_data), f, indent=4)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cd9cdb87",
   "metadata": {},
   "source": [
    "## Confusion matrix, salience map and symmetry calculation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "086bf395",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "def plot_resonator_confusion(model, x_test_full, y_test_full, data_percent=DATA_PERCENT, reversed_mode=False, session_name=\"Session\"):\n",
    "    # 1. Respect the data constraint\n",
    "    num_test = int(len(x_test_full) * data_percent)\n",
    "    x_subset = x_test_full[:num_test]\n",
    "    y_true = y_test_full[:num_test]\n",
    "\n",
    "    # 2. Handle Reverse Mode (Internal labeling only)\n",
    "    title_prefix = \"REVERSED\" if reversed_mode else \"STANDARD\"\n",
    "    if reversed_mode:\n",
    "        x_subset = x_subset[:, ::-1, :]\n",
    "\n",
    "    # 3. Get predictions\n",
    "    print(f\"--> Predicting for {title_prefix}...\")\n",
    "    logits, _ = model.predict(x_subset, batch_size=BATCH_SIZE, verbose=0)\n",
    "    y_pred = np.argmax(logits, axis=-1)\n",
    "    y_true_np = y_true.numpy() if hasattr(y_true, 'numpy') else y_true\n",
    "\n",
    "    # 4. Build ASCII Confusion Matrix\n",
    "    cm = confusion_matrix(y_true_np, y_pred, labels=range(10))\n",
    "    \n",
    "    cm_lines = []\n",
    "    cm_lines.append(f\"\\n{'#'*70}\")\n",
    "    cm_lines.append(f\" {title_prefix} EVALUATION | SESSION: {session_name}\")\n",
    "    cm_lines.append(f\"{'#'*70}\")\n",
    "    cm_lines.append(\"Pred:    0    1    2    3    4    5    6    7    8    9\")\n",
    "    cm_lines.append(\"True |\" + \"----\" * 12)\n",
    "    \n",
    "    for i, row in enumerate(cm):\n",
    "        row_str = f\"{i:>3}  | \" + \" \".join([f\"{val:>4}\" for val in row])\n",
    "        cm_lines.append(row_str)\n",
    "    \n",
    "    cm_lines.append(\"-\" * 60)\n",
    "    \n",
    "    # 5. Calculate Bias Score\n",
    "    most_common_idx = np.argmax(np.sum(cm, axis=0))\n",
    "    bias_pct = (np.sum(cm[:, most_common_idx]) / np.sum(cm)) * 100\n",
    "    cm_lines.append(f\"Primary Bias: Digit '{most_common_idx}' accounts for {bias_pct:.1f}% of all predictions.\")\n",
    "    cm_lines.append(f\"{'#'*70}\\n\")\n",
    "    \n",
    "    ascii_matrix = \"\\n\".join(cm_lines)\n",
    "\n",
    "    # 6. APPEND to the main session log ONLY (stripping any stray prefixes)\n",
    "    # This ensures it goes into log_RES_H16...txt even if called weirdly\n",
    "    clean_session = session_name.replace(\"Standard_\", \"\").replace(\"Reversed_\", \"\")\n",
    "    log_filename = f\"log_{clean_session}.txt\"\n",
    "    \n",
    "    with open(log_filename, \"a\") as f:\n",
    "        f.write(ascii_matrix)\n",
    "    \n",
    "    # Still print to console for the live check\n",
    "    print(ascii_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "79e5e305",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_field_jitter(model, sample_batch):\n",
    "    # 1. Extract hidden sequences\n",
    "    logits, h_seq = model(sample_batch[:1], training=False)\n",
    "    h_np = h_seq[0].numpy() # Shape: (784, HIDDEN)\n",
    "    \n",
    "    # 2. Reconstruct physics components\n",
    "    base_delta = (2.0 * np.pi) / PERIOD\n",
    "    steps = np.arange(h_np.shape[0])\n",
    "    phase = steps * base_delta\n",
    "    \n",
    "    # Raw Oscillator and Neuronal Bias\n",
    "    ghost_field = np.sin(phase)\n",
    "    activity_bias = JITTER_SCALE * np.mean(h_np, axis=-1) \n",
    "    combined_signal = ghost_field + activity_bias\n",
    "    \n",
    "    # Reconstruct G_norm (Simplified EMA for visualization)\n",
    "    # This approximates the 'new_G' EMA in your cell logic\n",
    "    ema_g = 0\n",
    "    g_history = []\n",
    "    for step_h in h_np:\n",
    "        # Shuffling the signal as per your 'half' concat logic\n",
    "        half = len(step_h) // 2\n",
    "        shuffled = np.concatenate([step_h[half:], step_h[:half]])\n",
    "        ema_g = (1.0 - LAMBDA_SLOW) * ema_g + LAMBDA_SLOW * shuffled\n",
    "        g_history.append(ema_g)\n",
    "    \n",
    "    g_history = np.array(g_history)\n",
    "    # Z-score normalization for the tanh gating\n",
    "    g_norm = (g_history - np.mean(g_history)) / (np.std(g_history) + 1e-6)\n",
    "    \n",
    "    # Calculate Final Gain Modulation (Field Effect)\n",
    "    # FE = 1 + (strength * combined * tanh(G_norm))\n",
    "    # We take the mean across neurons for the final visualization\n",
    "    field_effect = 1.0 + (BASE_STRENGTH * combined_signal[:, None] * np.tanh(g_norm))\n",
    "    field_effect_mean = np.mean(field_effect, axis=-1)\n",
    "\n",
    "    # 3. Plotting\n",
    "    fig, axes = plt.subplots(1, 3, figsize=(20, 5))\n",
    "    \n",
    "    # Plot 1: Interference Pattern\n",
    "    axes[0].plot(ghost_field, label=\"Pure Sine (Ghost)\", color='gray', alpha=0.4, linestyle='--')\n",
    "    axes[0].plot(combined_signal, label=\"Combined (Active)\", color='#3498db', linewidth=1.5)\n",
    "    axes[0].set_title(\"Carrier vs. Signal Interference\")\n",
    "    axes[0].legend()\n",
    "\n",
    "    # Plot 2: Neuronal Bias (The DC Offset)\n",
    "    axes[1].axhline(0, color='black', lw=1, alpha=0.3)\n",
    "    axes[1].fill_between(steps, activity_bias, color='#e74c3c', alpha=0.3, label=\"Neuronal Bias\")\n",
    "    axes[1].plot(activity_bias, color='#e74c3c', lw=2)\n",
    "    axes[1].set_title(f\"Hidden State Pressure (Scale: {JITTER_SCALE})\")\n",
    "    axes[1].legend()\n",
    "\n",
    "    # Plot 3: Resulting Gain Modulation (The Field Effect)\n",
    "    axes[2].axhline(1.0, color='black', lw=1, alpha=0.3) # 1.0 is neutral gain\n",
    "    axes[2].plot(field_effect_mean, color='#2ecc71', lw=2, label=\"Field Effect (Gain)\")\n",
    "    axes[2].fill_between(steps, 1.0, field_effect_mean, color='#2ecc71', alpha=0.2)\n",
    "    axes[2].set_title(\"Total Gain Modulation ($FE_t$)\")\n",
    "    axes[2].set_ylabel(\"Multiplicative Factor\")\n",
    "    axes[2].legend()\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# Run it!\n",
    "sample_x, sample_y = next(iter(test_ds_subset))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9101c0b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def visualize_pmnist_complete(model, image, label, permutation):\n",
    "    # 1. Setup Inverse Permutation\n",
    "    inverse_perm = np.argsort(permutation)\n",
    "    img_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)\n",
    "    \n",
    "    # 2. Manual hidden state extraction\n",
    "    h1_seq = model.rnn1(img_tensor, training=False)\n",
    "    h2_seq = model.rnn2(h1_seq, training=False)\n",
    "    h3_seq = model.rnn3(h2_seq, training=False)\n",
    "    \n",
    "    # 3. Get Prediction and Gradients for Saliency\n",
    "    with tf.GradientTape() as tape:\n",
    "        tape.watch(img_tensor)        \n",
    "        output = model(img_tensor, training=False)\n",
    "        logits = output[0] if isinstance(output, tuple) else output\n",
    "        \n",
    "        # Calculate Prediction and Confidence\n",
    "        probs = tf.nn.softmax(logits, axis=-1)\n",
    "        pred_label = np.argmax(probs.numpy(), axis=-1)[0]\n",
    "        confidence = np.max(probs.numpy(), axis=-1)[0]\n",
    "        \n",
    "        loss = logits[0, label]\n",
    "\n",
    "    grads = tape.gradient(loss, img_tensor)\n",
    "    saliency_flat = tf.reduce_max(tf.abs(grads), axis=-1).numpy().flatten()\n",
    "    \n",
    "    # 4. Prepare Images (Clean vs Scrambled)\n",
    "    unscrambled_img = image.flatten()[inverse_perm].reshape(28, 28)\n",
    "    unscrambled_sal = saliency_flat[inverse_perm].reshape(28, 28)\n",
    "    scrambled_img = image.reshape(28, 28)\n",
    "    scrambled_sal = saliency_flat.reshape(28, 28)\n",
    "\n",
    "    # --- PLOTTING ---\n",
    "    fig = plt.figure(figsize=(22, 12))\n",
    "    \n",
    "    # Color the title based on correctness\n",
    "    result_color = \"green\" if pred_label == label else \"red\"\n",
    "    \n",
    "    # COLUMN 1: UNSCRAMBLED (The \"Truth\")\n",
    "    ax1 = plt.subplot2grid((3, 5), (0, 0))\n",
    "    ax1.imshow(unscrambled_img, cmap='gray')\n",
    "    ax1.set_title(f\"Target: {label} | PRED: {pred_label}\\n({confidence*100:.1f}% Conf)\", \n",
    "                  fontsize=14, color=result_color, fontweight='bold')\n",
    "    \n",
    "    ax2 = plt.subplot2grid((3, 5), (1, 0))\n",
    "    ax2.imshow(unscrambled_sal, cmap='hot')\n",
    "    ax2.set_title(\"Focus (Unscrambled)\")\n",
    "\n",
    "    # COLUMN 2: SCRAMBLED (The \"Reality\")\n",
    "    ax3 = plt.subplot2grid((3, 5), (0, 1))\n",
    "    ax3.imshow(scrambled_img, cmap='gray')\n",
    "    ax3.set_title(\"Input (Scrambled)\")\n",
    "    \n",
    "    ax4 = plt.subplot2grid((3, 5), (1, 1))\n",
    "    ax4.imshow(scrambled_sal, cmap='hot')\n",
    "    ax4.set_title(\"Focus (Scrambled)\")\n",
    "\n",
    "    # COLUMN 3-5: THE HIDDEN HEARTBEATS\n",
    "    layers = [h1_seq, h2_seq, h3_seq]\n",
    "    layer_names = [\"Layer 1\", \"Layer 2\", \"Layer 3: Decision State\"]\n",
    "    \n",
    "    for i, (h_seq, name) in enumerate(zip(layers, layer_names)):\n",
    "        ax = plt.subplot2grid((3, 5), (i, 2), colspan=3)\n",
    "        activity = tf.transpose(h_seq[0]).numpy()\n",
    "        im = ax.imshow(activity, aspect='auto', cmap='magma', interpolation='nearest')\n",
    "        ax.set_title(name)\n",
    "        ax.set_ylabel(\"Neuron ID\")\n",
    "        if i == 2: ax.set_xlabel(\"Time (784 Steps)\")\n",
    "        plt.colorbar(im, ax=ax)\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# Run it!\n",
    "#visualize_pmnist_complete(model_active, x_test[101], y_test[101], perm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "df6d3968",
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy.spatial.distance import euclidean\n",
    "from scipy.stats import pearsonr\n",
    "\n",
    "def correlate_jitter_fields(model, image_idx, x_data, period, strength, lambda_slow):\n",
    "    # 1. Get Forward Jitter\n",
    "    sample_fwd = x_data[image_idx:image_idx+1]\n",
    "    _, h_fwd = model(sample_fwd, training=False)\n",
    "    fe_fwd = calculate_field_effect(h_fwd[0].numpy(), period, strength, lambda_slow)\n",
    "\n",
    "    # 2. Get Reversed Jitter\n",
    "    sample_rev = sample_fwd[:, ::-1, :]\n",
    "    _, h_rev = model(sample_rev, training=False)\n",
    "    fe_rev = calculate_field_effect(h_rev[0].numpy(), period, strength, lambda_slow)\n",
    "    \n",
    "    # 3. Flip the Reversed signal back to compare 1:1 with Forward\n",
    "    fe_rev_flipped = fe_rev[::-1]\n",
    "\n",
    "    # 4. Math Metrics\n",
    "    corr, _ = pearsonr(fe_fwd, fe_rev_flipped)\n",
    "    dist = euclidean(fe_fwd, fe_rev_flipped)\n",
    "    \n",
    "    print(f\"\\n--- SYMMETRY ANALYSIS (Hash {image_idx}) ---\")\n",
    "    print(f\"Correlation Score (R): {corr:.4f}  (1.0 = Perfect Symmetry)\")\n",
    "    print(f\"Euclidean Distance:    {dist:.4f}  (0.0 = Identical Field)\")\n",
    "    \n",
    "    return corr, dist\n",
    "\n",
    "# Helper to match your plot_field_jitter logic\n",
    "def calculate_field_effect(h_np, period, strength, lambda_slow):\n",
    "    steps = np.arange(len(h_np))\n",
    "    ghost = np.sin(steps * (2.0 * np.pi / period))\n",
    "    bias = np.mean(h_np, axis=-1)\n",
    "    combined = ghost + bias\n",
    "    \n",
    "    # Simple EMA to simulate G_norm\n",
    "    ema_g = 0\n",
    "    g_history = []\n",
    "    for step_h in h_np:\n",
    "        half = len(step_h) // 2\n",
    "        shuffled = np.concatenate([step_h[half:], step_h[:half]])\n",
    "        ema_g = (1.0 - lambda_slow) * ema_g + lambda_slow * shuffled\n",
    "        g_history.append(ema_g)\n",
    "    \n",
    "    g_norm = (np.array(g_history) - np.mean(g_history)) / (np.std(g_history) + 1e-6)\n",
    "    # Mean Field Effect across neurons\n",
    "    fe = 1.0 + (strength * combined[:, None] * np.tanh(g_norm))\n",
    "    return np.mean(fe, axis=-1)\n",
    "\n",
    "from scipy.stats import pearsonr\n",
    "from scipy.spatial.distance import euclidean\n",
    "\n",
    "def analyze_symmetry(model, hash_std, hash_rev, x_data):\n",
    "    # 1. Generate Fields\n",
    "    def get_fe(idx, reversed=False):\n",
    "        batch = x_data[idx:idx+1]\n",
    "        if reversed: batch = batch[:, ::-1, :]\n",
    "        _, h_seq = model(batch, training=False)\n",
    "        return calculate_field_effect(h_seq[0].numpy(), PERIOD, BASE_STRENGTH, LAMBDA_SLOW)\n",
    "\n",
    "    fe_std = get_fe(hash_std, reversed=False)\n",
    "    fe_rev = get_fe(hash_rev, reversed=True)\n",
    "    \n",
    "    # Flip the reversed field to align 1:1 with the standard timeline\n",
    "    fe_rev_aligned = fe_rev[::-1]\n",
    "\n",
    "    # 2. Math Metrics\n",
    "    corr, _ = pearsonr(fe_std, fe_rev_aligned)\n",
    "    dist = euclidean(fe_std, fe_rev_aligned)\n",
    "\n",
    "    print(f\"\\n{'='*40}\")\n",
    "    print(f\" FIELD SYMMETRY: Hash {hash_std} vs Hash {hash_rev}\")\n",
    "    print(f\"{'='*40}\")\n",
    "    print(f\" Pearson Correlation: {corr:.4f}  (Target: 1.0)\")\n",
    "    print(f\" Euclidean Distance:  {dist:.4f}  (Target: 0.0)\")\n",
    "    print(f\"{'='*40}\\n\")\n",
    "    \n",
    "    # Run your existing plots\n",
    "    plot_field_jitter(model, x_data[hash_std:hash_std+1])\n",
    "    plot_field_jitter(model, x_data[hash_rev:hash_rev+1][:, ::-1, :])\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dafa8ae4",
   "metadata": {},
   "source": [
    "## Execute"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "86c23a1a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- RUN 1: ACTIVE ---\n",
    "\n",
    "print(f\"\\n[Phase 1] Training Active for {SESSION_ID}...\")\n",
    "\n",
    "model_active = OscillatingResonator(hidden=HIDDEN, strength=BASE_STRENGTH, mode=\"active\")\n",
    "_ = model_active(tf.zeros((1, 784, 1))) \n",
    "\n",
    "# We pass the SESSION_ID into the name so the .h5 file is unique\n",
    "hist_active = train_phase(model_active, train_ds, val_ds, epochs=EPOCHS, name=f\"{SESSION_ID}_active\")\n",
    "\n",
    "test_acc_active = safe_evaluate(model_active, test_ds_subset)\n",
    "\n",
    "# Store everything in the map using the unique weight filename\n",
    "master_results[\"runs\"][\"active\"] = {\n",
    "    \"history\": hist_active,\n",
    "    \"test_acc\": float(test_acc_active),\n",
    "    \"weight_path\": f\"best_{SESSION_ID}_active.weights.h5\"\n",
    "}\n",
    "\n",
    "# --- FINAL GLOBAL SAVE ---\n",
    "\n",
    "print(f\"\\n SESSION COMPLETE!\")\n",
    "print(f\"Weights: best_{SESSION_ID}_active.weights.h5\")\n",
    "save_metrics_to_files(hist_active, model_active, \"Active\", test_acc_active)\n",
    "\n",
    " \n",
    "# --- RUN 2: PROBE ---\n",
    "\n",
    "reset_env()\n",
    "print(f\"\\n[Phase 2] Training Probe for {SESSION_ID}...\")\n",
    "\n",
    "model_probe = OscillatingResonator(hidden=HIDDEN, strength=BASE_STRENGTH, mode=\"probe\")\n",
    "_ = model_probe(tf.zeros((1, 784, 1))) \n",
    "\n",
    "# Train with unique ID\n",
    "hist_probe = train_phase(model_probe, train_ds, val_ds, epochs=EPOCHS, name=f\"{SESSION_ID}_probe\")\n",
    "\n",
    "test_acc_probe = safe_evaluate(model_probe, test_ds_subset)\n",
    "\n",
    "# Store in master map\n",
    "master_results[\"runs\"][\"probe\"] = {\n",
    "    \"history\": hist_probe,\n",
    "    \"test_acc\": float(test_acc_probe),\n",
    "    \"weight_path\": f\"best_{SESSION_ID}_probe.weights.h5\"\n",
    "}\n",
    "\n",
    "# Log to files\n",
    "save_metrics_to_files(hist_probe, model_probe, \"Probe\", test_acc_probe)\n",
    "\n",
    "# --- RUN 3: PASSIVE ---\n",
    "\n",
    "reset_env()\n",
    "print(f\"\\n[Phase 3] Training Passive for {SESSION_ID}...\")\n",
    "\n",
    "model_passive = OscillatingResonator(hidden=HIDDEN, strength=BASE_STRENGTH, mode=\"passive\")\n",
    "_ = model_passive(tf.zeros((1, 784, 1))) \n",
    "\n",
    "# Train with unique ID\n",
    "hist_passive = train_phase(model_passive, train_ds, val_ds, epochs=EPOCHS, name=f\"{SESSION_ID}_passive\")\n",
    "\n",
    "test_acc_passive = safe_evaluate(model_passive, test_ds_subset)\n",
    "\n",
    "# Store in master map\n",
    "master_results[\"runs\"][\"passive\"] = {\n",
    "    \"history\": hist_passive,\n",
    "    \"test_acc\": float(test_acc_passive),\n",
    "    \"weight_path\": f\"best_{SESSION_ID}_passive.weights.h5\"\n",
    "}\n",
    "\n",
    "# Log to files\n",
    "save_metrics_to_files(hist_passive, model_passive, \"Passive\", test_acc_passive)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f892ff77",
   "metadata": {},
   "source": [
    "## Evaluate Confusion matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "746f2462",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Configuration for this eval run\n",
    "DATA_PERCENT = 0.05\n",
    "BATCH_SIZE = 128 \n",
    "SESSION_ID = 'RES_H16_S0.22_J0.65_260324_1110'\n",
    "# 2. Dynamic weight pathing using the SESSION_ID\n",
    "# This ensures we aren't accidentally evaluating the wrong model version\n",
    "# --- EVALUATION BLOCK (Mode-Agnostic Version) ---\n",
    "\n",
    "# Set this once at the top of your cell depending on which you are testing\n",
    "CURRENT_MODE = \"active\"  # or \"probe\" or \"passive\"\n",
    "\n",
    "# 1. Dynamic pathing (No elif needed!)\n",
    "best_weight_path = f\"best_{SESSION_ID}_{CURRENT_MODE}_{SESSION_ID}.weights.h5\" \n",
    "\n",
    "print(f\"--> Target Mode: {CURRENT_MODE}\")\n",
    "print(f\"--> Weight Path: {best_weight_path}\")\n",
    "\n",
    "# 2. Universal Instantiation\n",
    "if f'model_{CURRENT_MODE}' not in locals():\n",
    "    print(f\"--> Creating model_{CURRENT_MODE}...\")\n",
    "    # This works for any mode!\n",
    "    model_to_eval = OscillatingResonator(hidden=HIDDEN, strength=BASE_STRENGTH, mode=CURRENT_MODE)\n",
    "    _ = model_to_eval(tf.zeros((1, 784, 1))) \n",
    "else:\n",
    "    model_to_eval = locals()[f'model_{CURRENT_MODE}']\n",
    "\n",
    "# 3. Universal Load\n",
    "if os.path.exists(best_weight_path):\n",
    "    model_to_eval.load_weights(best_weight_path)\n",
    "    print(f\"--> [SUCCESS] Weights loaded for {CURRENT_MODE}\")\n",
    "else:\n",
    "    print(f\"--> [ERROR] Weights NOT FOUND at {best_weight_path}\")\n",
    "\n",
    "\n",
    "if 'model_active' not in locals():\n",
    "    print(f\"--> model_active not found. Re-instantiating for {SESSION_ID}...\")\n",
    "    model_active = OscillatingResonator(hidden=HIDDEN, strength=BASE_STRENGTH, mode=\"active\")\n",
    "    _ = model_active(tf.zeros((1, 784, 1))) \n",
    "\n",
    "if os.path.exists(best_weight_path):\n",
    "    print(f\"--> Restoring weights from: {best_weight_path}\")\n",
    "    model_active.load_weights(best_weight_path)\n",
    "else:\n",
    "    if os.path.exists(\"best_active.weights.h5\"):\n",
    "        print(\"--> Using generic best_active.weights.h5\")\n",
    "        model_active.load_weights(\"best_active.weights.h5\")\n",
    "    else:\n",
    "        print(f\"--> CRITICAL: No weights found for {SESSION_ID}!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2facb0fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# 3. Standard Evaluation\n",
    "print(\"\\nEvaluating Standard Test Set...\")\n",
    "num_test = int(len(x_test) * DATA_PERCENT)\n",
    "x_subset = x_test[:num_test]\n",
    "y_subset = y_test[:num_test]\n",
    "\n",
    "# Using predict() for OOM safety\n",
    "logits_std, _ = model_active.predict(x_subset, batch_size=BATCH_SIZE, verbose=1)\n",
    "y_pred_std = np.argmax(logits_std, axis=-1)\n",
    "test_acc_active = np.mean(y_pred_std == y_subset)\n",
    "\n",
    "# 4. Reversed Evaluation\n",
    "print(\"\\nEvaluating Reversed Test Set...\")\n",
    "x_rev = x_subset[:, ::-1, :]\n",
    "logits_rev, _ = model_active.predict(x_rev, batch_size=BATCH_SIZE, verbose=1)\n",
    "y_pred_rev = np.argmax(logits_rev, axis=-1)\n",
    "rev_acc_active = np.mean(y_pred_rev == y_subset) * 100\n",
    "\n",
    "# 5. Final Stats & Logging logic\n",
    "retention_ratio = rev_acc_active / (test_acc_active * 100 + 1e-9)\n",
    "\n",
    "# Format the summary string\n",
    "summary_box = [\n",
    "    \"\\n\" + \"=\"*60,\n",
    "    f\" SESSION: {SESSION_ID}_{CURRENT_MODE}\",\n",
    "    f\" REVERSAL MEMORY CHECK (Data: {DATA_PERCENT*100:.1f}%)\",\n",
    "    \"=\"*60,\n",
    "    f\" Standard pMNIST Acc : {test_acc_active*100:>6.2f}%\",\n",
    "    f\" Reversed pMNIST Acc : {rev_acc_active:>6.2f}%\",\n",
    "    f\" Retention Ratio     : {retention_ratio:>6.2f}x\",\n",
    "    \"=\"*60 + \"\\n\"\n",
    "]\n",
    "summary_text = \"\\n\".join(summary_box)\n",
    "\n",
    "# APPEND to the text file\n",
    "log_filename = f\"log_{SESSION_ID}.txt\"\n",
    "with open(log_filename, \"a\") as f:\n",
    "    f.write(summary_text)\n",
    "\n",
    "print(summary_text)\n",
    "\n",
    "# 6. Confusion Matrices (Already appends to log internally)\n",
    "plot_resonator_confusion(model_to_eval, x_test, y_test, \n",
    "                         data_percent=DATA_PERCENT, \n",
    "                         reversed_mode=False, \n",
    "                         session_name=f\"Standard_{SESSION_ID}\")\n",
    "\n",
    "plot_resonator_confusion(model_to_eval, x_test, y_test, \n",
    "                         data_percent=DATA_PERCENT, \n",
    "                         reversed_mode=True, \n",
    "                         session_name=f\"Reversed_{SESSION_ID}\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1521d7be",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "\n",
    "# --- 1. CONFIGURATION ---\n",
    "DATA_PERCENT = 0.1  \n",
    "BATCH_SIZE = 64\n",
    "SESSION_ID = 'RES_H16_S0.22_J0.65_260324_1110'\n",
    "MODES_TO_EVAL = [\"probe\", \"passive\", \"active\"]\n",
    "USE_LATEST = False  \n",
    "\n",
    "# Define the exact log filename\n",
    "log_filename = f\"log_{SESSION_ID}.txt\"\n",
    "\n",
    "# Helper to print to console AND append to log file simultaneously\n",
    "def log_print(msg):\n",
    "    print(msg)\n",
    "    with open(log_filename, \"a\") as f:\n",
    "        f.write(str(msg) + \"\\n\")\n",
    "\n",
    "# --- 2. EVALUATION LOOP ---\n",
    "for CURRENT_MODE in MODES_TO_EVAL:\n",
    "    log_print(\"\\n\" + \"=\"*70)\n",
    "    log_print(f\" INITIALIZING EVALUATION: {CURRENT_MODE.upper()}\")\n",
    "    log_print(\"=\"*70)\n",
    "\n",
    "    # Pathing logic for 'latest' vs 'best'\n",
    "    if USE_LATEST:\n",
    "        weight_path = f\"latest_{SESSION_ID}_{CURRENT_MODE}_{SESSION_ID}.weights.h5\"\n",
    "    else:\n",
    "        weight_path = f\"best_{SESSION_ID}_{CURRENT_MODE}_{SESSION_ID}.weights.h5\"\n",
    "        #best_RES_H16_S0.22_J0.65_260223_2220_probe_RES_H16_S0.22_J0.65_260223_2220.weights\n",
    "        #best_probe_RES_H16_S0.22_J0.65_260223_2220.weights.h5\n",
    "\n",
    "    log_print(f\"--> Target Mode: {CURRENT_MODE}\")\n",
    "    log_print(f\"--> Weight Path: {weight_path}\")\n",
    "\n",
    "    # 3. Model Setup\n",
    "    tf.keras.backend.clear_session()\n",
    "    model_to_eval = OscillatingResonator(hidden=HIDDEN, strength=BASE_STRENGTH, mode=CURRENT_MODE)\n",
    "    _ = model_to_eval(tf.zeros((1, 784, 1))) \n",
    "\n",
    "    # 4. Load Weights\n",
    "    if os.path.exists(weight_path):\n",
    "        model_to_eval.load_weights(weight_path)\n",
    "        log_print(f\"--> [SUCCESS] Weights loaded for {CURRENT_MODE}\")\n",
    "    else:\n",
    "        log_print(f\"--> [ERROR] Weights NOT FOUND at {weight_path}. Skipping...\")\n",
    "        continue\n",
    "\n",
    "    # 5. Standard Evaluation\n",
    "    log_print(f\"\\nEvaluating Standard Test Set ({CURRENT_MODE})...\")\n",
    "    num_test = int(len(x_test) * DATA_PERCENT)\n",
    "    x_subset = x_test[:num_test]\n",
    "    y_subset = y_test[:num_test]\n",
    "\n",
    "    logits_std, _ = model_to_eval.predict(x_subset, batch_size=BATCH_SIZE, verbose=1)\n",
    "    y_pred_std = np.argmax(logits_std, axis=-1)\n",
    "    test_acc_raw = np.mean(y_pred_std == y_subset)\n",
    "\n",
    "    # 6. Reversed Evaluation\n",
    "    log_print(f\"\\nEvaluating Reversed Test Set ({CURRENT_MODE})...\")\n",
    "    x_rev = x_subset[:, ::-1, :] \n",
    "    logits_rev, _ = model_to_eval.predict(x_rev, batch_size=BATCH_SIZE, verbose=1)\n",
    "    y_pred_rev = np.argmax(logits_rev, axis=-1)\n",
    "    rev_acc_pct = np.mean(y_pred_rev == y_subset) * 100\n",
    "\n",
    "    # 7. Stats calculation\n",
    "    retention_ratio = rev_acc_pct / (test_acc_raw * 100 + 1e-9)\n",
    "    weight_type_label = \"LATEST\" if USE_LATEST else \"BEST\"\n",
    "\n",
    "    summary_box = [\n",
    "        \"\\n\" + \"=\"*60,\n",
    "        f\" SESSION        : {SESSION_ID}\",\n",
    "        f\" MODE           : {CURRENT_MODE.upper()} ({weight_type_label})\",\n",
    "        f\" EVAL DATA %    : {DATA_PERCENT*100:.1f}%\",\n",
    "        \"=\"*60,\n",
    "        f\" Standard Acc   : {test_acc_raw*100:>6.2f}%\",\n",
    "        f\" Reversed Acc   : {rev_acc_pct:>6.2f}%\",\n",
    "        f\" Retention Ratio: {retention_ratio:>6.2f}x\",\n",
    "        \"=\"*60 + \"\\n\"\n",
    "    ]\n",
    "    \n",
    "    # Log the summary box\n",
    "    for line in summary_box:\n",
    "        log_print(line)\n",
    "\n",
    "    # 8. Confusion Matrices \n",
    "    # (Assuming these handle their own internal logging/plotting)\n",
    "    plot_resonator_confusion(model_to_eval, x_test, y_test, \n",
    "                             data_percent=DATA_PERCENT, \n",
    "                             reversed_mode=False, \n",
    "                             session_name=f\"Standard_{SESSION_ID}_{CURRENT_MODE}\")\n",
    "\n",
    "    plot_resonator_confusion(model_to_eval, x_test, y_test, \n",
    "                             data_percent=DATA_PERCENT, \n",
    "                             reversed_mode=True, \n",
    "                             session_name=f\"Reversed_{SESSION_ID}_{CURRENT_MODE}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d9e9a0df",
   "metadata": {},
   "source": [
    "## Salience map code"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "392da46f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Expanded scan to find the first 5 unique indices (Hashes) for each digit\n",
    "def get_sample_summary_expanded(y_data):\n",
    "    summary = {}\n",
    "    # Target order: 1, 2, 3, 4, 5, 6, 7, 8, 9, 0\n",
    "    target_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]\n",
    "    \n",
    "    for digit in target_order:\n",
    "        # Find all indices where y_test matches the digit\n",
    "        # Using y_data.numpy() if it's a tensor to ensure compatibility\n",
    "        y_vals = y_data.numpy() if hasattr(y_data, 'numpy') else y_data\n",
    "        indices = np.where(y_vals == digit)[0]\n",
    "        summary[digit] = indices[:5].tolist() # Take the first five\n",
    "        \n",
    "    print(f\"{'='*65}\")\n",
    "    print(f\"            EXPANDED RESONATOR HASH DIRECTORY (Top 5)\")\n",
    "    print(f\"{'='*65}\")\n",
    "    print(f\"Digit | Hash 1 | Hash 2 | Hash 3 | Hash 4 | Hash 5\")\n",
    "    print(f\"------|--------|--------|--------|--------|--------\")\n",
    "    for digit in target_order:\n",
    "        h = summary[digit]\n",
    "        # Padding in case some digits have fewer than 5 samples in the subset\n",
    "        h += [\"N/A\"] * (5 - len(h))\n",
    "        print(f\"  {digit}   | {h[0]:>6} | {h[1]:>6} | {h[2]:>6} | {h[3]:>6} | {h[4]:>6}\")\n",
    "    print(f\"{'='*65}\")\n",
    "\n",
    "# Run this on your subset\n",
    "get_sample_summary_expanded(y_subset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68ae784e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "\n",
    "# --- 1. CONFIGURATION ---\n",
    "DATA_PERCENT = 0.01  \n",
    "BATCH_SIZE = 128\n",
    "SESSION_ID = 'RES_H16_S0.22_J0.65_260324_1110'\n",
    "MODES_TO_EVAL = [\"active\"]\n",
    "USE_LATEST = True  \n",
    "\n",
    "# Define the exact log filename\n",
    "log_filename = f\"log_{SESSION_ID}.txt\"\n",
    "\n",
    "# Helper to print to console AND append to log file simultaneously\n",
    "def log_print(msg):\n",
    "    print(msg)\n",
    "    with open(log_filename, \"a\") as f:\n",
    "        f.write(str(msg) + \"\\n\")\n",
    "\n",
    "# --- 2. EVALUATION LOOP ---\n",
    "for CURRENT_MODE in MODES_TO_EVAL:\n",
    "    log_print(\"\\n\" + \"=\"*70)\n",
    "    log_print(f\" INITIALIZING EVALUATION: {CURRENT_MODE.upper()}\")\n",
    "    log_print(\"=\"*70)\n",
    "\n",
    "    # Pathing logic for 'latest' vs 'best'\n",
    "    if USE_LATEST:\n",
    "        weight_path = f\"latest_{SESSION_ID}_{CURRENT_MODE}_{SESSION_ID}.weights.h5\"\n",
    "    else:\n",
    "        weight_path = f\"best_{SESSION_ID}_{CURRENT_MODE}_{SESSION_ID}.weights.h5\"\n",
    "        #best_RES_H16_S0.22_J0.65_260223_2220_probe_RES_H16_S0.22_J0.65_260223_2220.weights\n",
    "        #best_probe_RES_H16_S0.22_J0.65_260223_2220.weights.h5\n",
    "\n",
    "    log_print(f\"--> Target Mode: {CURRENT_MODE}\")\n",
    "    log_print(f\"--> Weight Path: {weight_path}\")\n",
    "\n",
    "    # 3. Model Setup\n",
    "    tf.keras.backend.clear_session()\n",
    "    model_to_eval = OscillatingResonator(hidden=HIDDEN, strength=BASE_STRENGTH, mode=CURRENT_MODE)\n",
    "    _ = model_to_eval(tf.zeros((1, 784, 1))) \n",
    "\n",
    "    # 4. Load Weights\n",
    "    if os.path.exists(weight_path):\n",
    "        model_to_eval.load_weights(weight_path)\n",
    "        log_print(f\"--> [SUCCESS] Weights loaded for {CURRENT_MODE}\")\n",
    "    else:\n",
    "        log_print(f\"--> [ERROR] Weights NOT FOUND at {weight_path}. Skipping...\")\n",
    "        continue\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9e5be8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize normal permuted sequence\n",
    "\n",
    "Hash_p = 5\n",
    "sample_batch = x_test[Hash_p:Hash_p+1]\n",
    "visualize_pmnist_complete(model_to_eval, x_test[Hash_p], y_test[Hash_p], perm)\n",
    "\n",
    "# Visualize reversed permuted sequence\n",
    "Hash_rev = 5\n",
    "rev_batch = x_test[Hash_rev:Hash_rev+1]\n",
    "sample_batch_rev = rev_batch[:, ::-1, :]\n",
    "visualize_pmnist_complete(model_to_eval, x_rev[Hash_rev], y_test[Hash_rev], perm[::-1])\n",
    "\n",
    "# Analyze symetry:\n",
    "analyze_symmetry(model_to_eval, Hash_p, Hash_rev, x_test)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6ebe724a",
   "metadata": {},
   "source": [
    "## SOBOL analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "af36a4d6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " SESSION INITIALIZED: RES_H32_S0.22_J0.65_260404_2241\n",
      "--- PARAMETER COUNT REPORT (Hidden: 32) ---\n",
      "RNN Layer 1:    1,088\n",
      "RNN Layer 2:    2,080\n",
      "RNN Layer 3:    2,080\n",
      "Dense Out:        330\n",
      "-----------------------------------\n",
      "TOTAL PARAMS:    5,578\n",
      "Estimated Memory: 21.79 KB (float32)\n",
      "===================================\n",
      "\n",
      "--- SOBOL RUN ESTIMATION ---\n",
      "Variables (D): 5\n",
      "Baseline (N):  64\n",
      "Total Model Evaluations: 768\n",
      "Estimated Time: 61.44 hours\n",
      "Estimated Time: 2.56 days\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from SALib.sample import saltelli\n",
    "from SALib.analyze import sobol\n",
    "import matplotlib.pyplot as plt\n",
    "import gc\n",
    "from datetime import datetime\n",
    "import json\n",
    "DATA_PERCENT = 0.1\n",
    "BATCH_SIZE = 4 * 64\n",
    "EPOCHS = 12         \n",
    "HIDDEN = 32 \n",
    "REST_BASELINE = 1.0\n",
    "LEARNING_RATE = 1e-3\n",
    "#BASE_STRENGTH =0.6\n",
    "#LAMBDA_SLOW = 0.04\n",
    "#PERIOD= 784/4 \n",
    "#JITTER_SCALE = 0.6\n",
    "N_baseline = 64 # This is the number you pass to saltelli.sample\n",
    "\n",
    "# --- 1.1 AUTO-GENERATED SESSION NAME ---\n",
    "# Creates a name like: RES_H32_S0.40_J1.15_T123456\n",
    "now = datetime.now()\n",
    "readable_ts = now.strftime(\"%y%m%d_%H%M\") \n",
    "SESSION_ID = f\"RES_H{HIDDEN}_S{BASE_STRENGTH}_J{JITTER_SCALE}_{readable_ts}\"\n",
    "print(f\" SESSION INITIALIZED: {SESSION_ID}\")\n",
    "print_param_report(HIDDEN)\n",
    "\n",
    "sobol_problem = {\n",
    "    'num_vars': 5,\n",
    "    'names': ['LAMBDA_SLOW', 'H_INERTIA','BASE_STRENGTH', 'PERIOD', 'JITTER_SCALE'],\n",
    "    'bounds': [\n",
    "        [0.001, 0.05], \n",
    "        [0.01, 0.99],\n",
    "        [0.01, 0.99],      \n",
    "        [784/10, 784/2], \n",
    "        [0.1, 1.2]\n",
    "    ]\n",
    "}\n",
    "\n",
    "def calc_sobol_samples(problem, N, second_order=True):\n",
    "    \"\"\"\n",
    "    Calculates total runs for a SALib Saltelli/Sobol sequence.\n",
    "    Formula: N * (2D + 2) if second_order=True\n",
    "             N * (D + 2)  if second_order=False\n",
    "    \"\"\"\n",
    "    D = problem['num_vars']\n",
    "    if second_order:\n",
    "        total = N * (2 * D + 2)\n",
    "    else:\n",
    "        total = N * (D + 2)\n",
    "    return total\n",
    "\n",
    "# Your specific setup\n",
    "\n",
    "total_runs = calc_sobol_samples(sobol_problem, N_baseline, second_order=True)\n",
    "\n",
    "print(f\"--- SOBOL RUN ESTIMATION ---\")\n",
    "print(f\"Variables (D): {sobol_problem['num_vars']}\")\n",
    "print(f\"Baseline (N):  {N_baseline}\")\n",
    "print(f\"Total Model Evaluations: {total_runs}\")\n",
    "\n",
    "# Time estimation (Optional but helpful for dissertations)\n",
    "avg_time_per_run = EPOCHS*24 # seconds (Estimate based on your Epochs)\n",
    "total_hours = (total_runs * avg_time_per_run) / 3600\n",
    "total_days = total_hours/24\n",
    "print(f\"Estimated Time: {total_hours:.2f} hours\")\n",
    "print(f\"Estimated Time: {total_days:.2f} days\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d8cac84",
   "metadata": {},
   "source": [
    "## Oscilatory "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "db5f85f3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/casper/micromamba/envs/rs/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import scipy.linalg\n",
    "import json\n",
    "import gc\n",
    "import math\n",
    "from tqdm.auto import tqdm\n",
    "\n",
    "# --- 1. GLOBAL MASTER LOGGING ---\n",
    "SOBOL_MASTER_DATA = {\"session_id\": SESSION_ID, \"runs\": {}, \"Si_all\": {}}\n",
    "\n",
    "def update_json_log(history, model, run_id):\n",
    "    \"\"\"Saves every epoch's progress into one master 'Big Ass' JSON file.\"\"\"\n",
    "    global SOBOL_MASTER_DATA\n",
    "    if run_id is None: return # Skip if not a Sobol run\n",
    "    \n",
    "    cell = model.rnn1.cell\n",
    "    SOBOL_MASTER_DATA[\"runs\"][str(run_id)] = {\n",
    "        \"config\": {\n",
    "            \"tau\": float(cell.lambda_slow),\n",
    "            \"strength\": float(cell.strength),\n",
    "            \"jitter\": float(JITTER_SCALE),\n",
    "            \"period\": float(cell.period),\n",
    "            #\"learning rate\": float(LEARNING_RATE),\n",
    "            \"h_inertia\": float(H_SCALE[0])\n",
    "        },\n",
    "        \"history\": history\n",
    "    }\n",
    "\n",
    "    def clean(obj):\n",
    "        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}\n",
    "        if isinstance(obj, list): return [clean(i) for i in obj]\n",
    "        if isinstance(obj, (np.float32, np.float64, np.ndarray)): \n",
    "            return float(obj) if np.isscalar(obj) else obj.tolist()\n",
    "        return obj\n",
    "\n",
    "    with open(f\"SOBOL_MASTER_{SESSION_ID}.json\", \"w\") as f:\n",
    "        json.dump(clean(SOBOL_MASTER_DATA), f, indent=4)\n",
    "\n",
    "# --- 2. THE JITTERED CELL & MODEL ---\n",
    "class JitteredFeedbackCell(tf.keras.layers.Layer):\n",
    "    def __init__(self, units=16, strength=0.0, period=256.0, lambda_slow=0.05, mode=\"active\", **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.units = units\n",
    "        self.state_size = [units, units, 1] \n",
    "        self.strength = strength \n",
    "        self.period = period\n",
    "        self.lambda_slow = lambda_slow\n",
    "        self.mode = mode\n",
    "\n",
    "    def build(self, input_shape):\n",
    "        self.w_in = self.add_weight(shape=(input_shape[-1], self.units), name=\"w_in\", initializer=\"glorot_uniform\")\n",
    "        self.w_rec = self.add_weight(shape=(self.units, self.units), name=\"w_rec\",\n",
    "                                     initializer=tf.keras.initializers.Orthogonal(gain=1.0))\n",
    "        self.bias = self.add_weight(shape=(self.units,), name=\"bias\", initializer=\"zeros\")\n",
    "\n",
    "    def call(self, inputs, states):\n",
    "        prev_h, prev_G, prev_phase = states\n",
    "        half = self.units // 2\n",
    "        raw_signal = tf.concat([prev_h[:, half:], prev_h[:, :half]], axis=1)\n",
    "        source_signal = tf.stop_gradient(raw_signal) if self.mode == \"probe\" else raw_signal\n",
    "        \n",
    "        new_G = (1.0 - self.lambda_slow) * prev_G + self.lambda_slow * source_signal\n",
    "        G_norm = (new_G - tf.reduce_mean(new_G, axis=-1, keepdims=True)) / (tf.math.reduce_std(new_G, axis=-1, keepdims=True) + 1e-6)\n",
    "        \n",
    "        new_phase = prev_phase + (2.0 * math.pi / self.period)\n",
    "        oscillator = tf.math.sin(new_phase)\n",
    "        \n",
    "        if self.mode == \"active\":\n",
    "            bias_signal = tf.reduce_mean(source_signal, axis=-1, keepdims=True) - 0.1\n",
    "            combined_signal = oscillator + (JITTER_SCALE * bias_signal)\n",
    "        else:\n",
    "            combined_signal = oscillator\n",
    "\n",
    "        current_strength = self.strength * combined_signal if self.mode != \"passive\" else 0.0\n",
    "        field_effect = REST_BASELINE + (current_strength * tf.tanh(G_norm))\n",
    "        \n",
    "        z = (tf.matmul(inputs, self.w_in) + tf.matmul(prev_h, self.w_rec) + self.bias) * field_effect\n",
    "        h = (H_SCALE[0] * prev_h) + (H_SCALE[1] * tf.nn.elu(z))\n",
    "        h = tf.clip_by_value(h, -20.0, 20.0)\n",
    "        return h, [h, new_G, new_phase]\n",
    "\n",
    "class OscillatingResonator(tf.keras.Model):\n",
    "    def __init__(self, hidden=16, num_classes=10, strength=0.0, mode=\"active\"):\n",
    "        super().__init__()\n",
    "        # Use cell_ref so we can access hyperparameters easily later\n",
    "        self.cell_ref = JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode)\n",
    "        self.rnn1 = tf.keras.layers.RNN(self.cell_ref, return_sequences=True)\n",
    "        self.rnn2 = tf.keras.layers.RNN(JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode), return_sequences=True)\n",
    "        self.rnn3 = tf.keras.layers.RNN(JitteredFeedbackCell(hidden, strength, PERIOD, LAMBDA_SLOW, mode=mode), return_sequences=True)\n",
    "        self.out = tf.keras.layers.Dense(num_classes)\n",
    "\n",
    "    def call(self, x, training=False):\n",
    "        h1 = self.rnn1(x, training=training)\n",
    "        h2 = self.rnn2(h1, training=training)\n",
    "        h3 = self.rnn3(h2, training=training)\n",
    "        return self.out(h3[:, -1, :]), h3\n",
    "\n",
    "# --- 3. UPDATED TRAINING PHASE ---\n",
    "def train_phase(model, train_data, val_data, epochs=3, name=\"model\", run_id=None):\n",
    "    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)\n",
    "    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)\n",
    "    history = {\"loss\": [], \"acc\": [], \"hidden_metrics\": []}\n",
    "    best_val_acc = 0.0\n",
    "\n",
    "    @tf.function\n",
    "    def train_step(x, y):\n",
    "        with tf.GradientTape() as tape:\n",
    "            logits, _ = model(x, training=True)\n",
    "            loss_v = loss_fn(y, logits)\n",
    "        grads = tape.gradient(loss_v, model.trainable_variables)\n",
    "        grads, _ = tf.clip_by_global_norm(grads, 1.0)\n",
    "        optimizer.apply_gradients(zip(grads, model.trainable_variables))\n",
    "        return loss_v, logits\n",
    "\n",
    "    # Added 'Intf' to the console header\n",
    "    print(f\"\\n{'Epoch':<6} | {'Loss':<7} | {'Val-Acc%':<8} | {'Rank':<6} | {'Sync':<6} | {'Entrp':<6} | {'A-Corr':<7} | {'Intf':<6}\")\n",
    "    print(\"-\" * 85)\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        epoch_losses = []\n",
    "        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()\n",
    "        pbar = tqdm(train_data, desc=f\"EPOCH {epoch+1}/{epochs}\", leave=False)\n",
    "        \n",
    "        for x_b, y_b in pbar:\n",
    "            loss_v, logits = train_step(x_b, y_b)\n",
    "            acc_metric.update_state(y_b, logits)\n",
    "            epoch_losses.append(float(loss_v))\n",
    "            pbar.set_postfix({\"loss\": f\"{np.mean(epoch_losses):.4f}\", \"acc\": f\"{acc_metric.result():.2%}\"})\n",
    "\n",
    "        # --- VALIDATION SNAPSHOT ---\n",
    "        for x_v, y_v in val_data.take(1):\n",
    "            logits_v, h_seq_v = model(x_v, training=False)\n",
    "            val_acc = np.mean(np.argmax(logits_v.numpy(), axis=-1) == y_v.numpy())            \n",
    "            \n",
    "            # --- METRIC CALCULATIONS ---\n",
    "            h_final = h_seq_v.numpy()[:, -1, :]\n",
    "            \n",
    "            # 1. Effective Rank\n",
    "            s = scipy.linalg.svdvals(h_final) + 1e-12\n",
    "            p_rank = s / (np.sum(s) + 1e-10)\n",
    "            eff_rank = np.exp(-np.sum(p_rank * np.log(p_rank + 1e-10)))\n",
    "            \n",
    "            # 2. Entropy\n",
    "            counts, _ = np.histogram(h_final, bins=50)\n",
    "            p_ent = counts / (h_final.size + 1e-10) \n",
    "            entropy_val = -np.sum(p_ent * np.log2(p_ent + 1e-10))\n",
    "            \n",
    "            # 3. Synchrony (Inter-neuron correlation)\n",
    "            sync_val = (np.sum(np.abs(np.corrcoef(h_final.T + 1e-8))) - HIDDEN) / (HIDDEN**2 - HIDDEN)\n",
    "            \n",
    "            # 4. Auto-Correlation (Temporal slowness)\n",
    "            acorr_val = np.mean(np.abs(np.corrcoef(h_seq_v.numpy()[0].T + 1e-8)))\n",
    "\n",
    "            # 5. NEW: Interference (Neuron-to-Field alignment)\n",
    "            # Measures how much neurons are driven by the common field vs unique input\n",
    "            mean_field = np.mean(h_final, axis=1, keepdims=True)\n",
    "            neuron_to_field_corrs = [\n",
    "                np.abs(np.corrcoef(h_final[:, j], mean_field[:, 0])[0, 1]) \n",
    "                for j in range(h_final.shape[1])\n",
    "            ]\n",
    "            interference_val = np.mean(np.nan_to_num(neuron_to_field_corrs))\n",
    "\n",
    "        # --- HISTORY UPDATE ---\n",
    "        avg_loss = np.mean(epoch_losses)\n",
    "        history[\"loss\"].append(float(avg_loss))\n",
    "        history[\"acc\"].append(float(val_acc))\n",
    "        history[\"hidden_metrics\"].append({\n",
    "            \"effective_rank\": float(eff_rank),\n",
    "            \"synchrony\": float(sync_val),\n",
    "            \"entropy\": float(entropy_val),\n",
    "            \"a_corr\": float(acorr_val),\n",
    "            \"interference\": float(interference_val)\n",
    "        })\n",
    "\n",
    "        # --- THE BIG JSON SAVE ---\n",
    "        update_json_log(history, model, run_id)\n",
    "        \n",
    "        # Unified Console Output\n",
    "        print(f\"EP {epoch+1}/{epochs}: {avg_loss:<7.3f} | {val_acc:<8.2%} | {eff_rank:<6.2f} | \"\n",
    "              f\"{sync_val:<6.3f} | {entropy_val:<6.2f} | {acorr_val:<7.3f} | {interference_val:<6.3f}\")\n",
    "\n",
    "    return history"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2cdbe07b",
   "metadata": {},
   "source": [
    "## Execute SOBOL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b0ae593e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_4464/1967379887.py:67: DeprecationWarning: `salib.sample.saltelli` will be removed in SALib 1.5.1 Please use `salib.sample.sobol`\n",
      "  param_values = saltelli.sample(sobol_problem, N_baseline, calc_second_order=True)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--- STARTING MULTI-METRIC SOBOL ANALYSIS (RES_H32_S0.22_J0.65_260404_2241) ---\n",
      "\n",
      "[Sobol Run 0/768] H_Inertia: 0.4005 | BASE_S: 0.8139 | Tau: 0.0021 | Jitter: 0.81\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "EPOCH 1/12: 100%|██████████| 22/22 [00:27<00:00,  1.30s/it, loss=nan, acc=9.09%]   2026-04-04 22:42:05.245217: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n",
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 0 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 1/768] H_Inertia: 0.4005 | BASE_S: 0.8139 | Tau: 0.0228 | Jitter: 0.81\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "EPOCH 1/12: 100%|██████████| 22/22 [00:25<00:00,  1.33s/it, loss=nan, acc=9.56%]    2026-04-04 22:42:37.070616: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n",
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 1 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 2/768] H_Inertia: 0.6608 | BASE_S: 0.8139 | Tau: 0.0021 | Jitter: 0.81\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-04-04 22:43:14.014989: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 12.847  | 16.02%   | 12.98  | 0.277  | 3.61   | 0.341   | 0.462 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/casper/micromamba/envs/rs/lib/python3.11/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide\n",
      "  c /= stddev[:, None]\n",
      "/home/casper/micromamba/envs/rs/lib/python3.11/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide\n",
      "  c /= stddev[None, :]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 6.784   | 9.38%    | 1.91   | nan    | 3.47   | 0.348   | 0.399 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-04-04 22:44:08.882723: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 5.066   | 10.16%   | 5.73   | 0.394  | 2.71   | 0.386   | 0.512 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.306   | 26.95%   | 6.78   | 0.552  | 2.59   | 0.360   | 0.651 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.476   | 29.30%   | 6.90   | 0.518  | 2.28   | 0.349   | 0.573 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.009   | 33.20%   | 9.67   | 0.469  | 3.25   | 0.347   | 0.491 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-04-04 22:45:59.026372: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.960   | 34.38%   | 7.34   | 0.405  | 2.77   | 0.346   | 0.434 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.860   | 35.55%   | 7.52   | 0.386  | 2.82   | 0.359   | 0.508 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.939   | 35.94%   | 9.98   | 0.514  | 3.32   | 0.343   | 0.408 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 2.439   | 19.53%   | 22.07  | 0.257  | 3.07   | 0.367   | 0.507 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 7.853   | 13.28%   | 25.73  | 0.179  | 3.53   | 0.322   | 0.435 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 12.574  | 13.28%   | 21.69  | 0.187  | 3.37   | 0.331   | 0.394 \n",
      "\n",
      "\n",
      "[Sobol Run 3/768] H_Inertia: 0.4005 | BASE_S: 0.2167 | Tau: 0.0021 | Jitter: 0.81\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.174   | 25.39%   | 9.17   | 0.578  | 3.29   | 0.612   | 0.618 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.023   | 31.64%   | 11.48  | 0.469  | 3.97   | 0.605   | 0.281 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-04-04 22:49:37.788740: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.892   | 34.77%   | 11.13  | 0.383  | 2.64   | 0.375   | 0.418 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.788   | 37.11%   | 12.81  | 0.422  | 3.97   | 0.437   | 0.366 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.701   | 41.02%   | 12.40  | 0.413  | 2.76   | 0.392   | 0.369 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.662   | 33.98%   | 13.24  | 0.416  | 4.06   | 0.545   | 0.341 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.648   | 39.84%   | 13.11  | 0.404  | 3.62   | 0.430   | 0.353 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.552   | 46.88%   | 12.71  | 0.408  | 3.62   | 0.385   | 0.375 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.553   | 42.97%   | 12.73  | 0.410  | 2.94   | 0.394   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.509   | 48.83%   | 13.31  | 0.363  | 3.11   | 0.390   | 0.336 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.460   | 49.22%   | 12.12  | 0.378  | 3.11   | 0.379   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.450   | 50.00%   | 13.01  | 0.385  | 3.33   | 0.372   | 0.344 \n",
      "\n",
      "\n",
      "[Sobol Run 4/768] H_Inertia: 0.4005 | BASE_S: 0.8139 | Tau: 0.0021 | Jitter: 0.81\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 4 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 5/768] H_Inertia: 0.4005 | BASE_S: 0.8139 | Tau: 0.0021 | Jitter: 0.23\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.598   | 14.06%   | 7.70   | 0.511  | 4.59   | 0.534   | 0.665 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.148   | 23.44%   | 7.71   | 0.642  | 4.58   | 0.585   | 0.730 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.013   | 27.34%   | 9.06   | 0.542  | 4.54   | 0.583   | 0.633 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.993   | 36.72%   | 8.85   | 0.488  | 4.27   | 0.583   | 0.556 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.912   | 29.69%   | 9.56   | 0.521  | 4.72   | 0.589   | 0.578 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.906   | 35.55%   | 10.41  | 0.486  | 4.43   | 0.620   | 0.542 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "EPOCH 7/12: 100%|██████████| 22/22 [00:20<00:00,  1.07it/s, loss=1.8115, acc=34.17%]2026-04-04 22:57:12.269663: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n",
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.812   | 40.23%   | 10.34  | 0.467  | 4.44   | 0.584   | 0.391 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.751   | 44.53%   | 10.69  | 0.474  | 4.56   | 0.589   | 0.463 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.703   | 43.75%   | 10.82  | 0.444  | 4.54   | 0.529   | 0.369 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.683   | 44.14%   | 11.04  | 0.402  | 4.37   | 0.519   | 0.315 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.633   | 46.88%   | 11.27  | 0.409  | 4.65   | 0.471   | 0.329 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.649   | 47.27%   | 11.15  | 0.415  | 4.48   | 0.481   | 0.331 \n",
      "\n",
      "\n",
      "[Sobol Run 6/768] H_Inertia: 0.6608 | BASE_S: 0.2167 | Tau: 0.0021 | Jitter: 0.23\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.184   | 29.30%   | 7.38   | 0.666  | 4.51   | 0.572   | 0.484 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.958   | 34.77%   | 8.06   | 0.578  | 4.81   | 0.536   | 0.595 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.842   | 37.89%   | 7.61   | 0.552  | 4.87   | 0.577   | 0.507 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.749   | 39.06%   | 7.67   | 0.543  | 4.75   | 0.563   | 0.486 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.677   | 39.45%   | 8.46   | 0.484  | 4.80   | 0.520   | 0.457 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.618   | 44.53%   | 8.01   | 0.482  | 4.72   | 0.491   | 0.473 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.603   | 35.94%   | 8.59   | 0.522  | 4.53   | 0.483   | 0.438 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.660   | 42.58%   | 9.76   | 0.452  | 4.73   | 0.495   | 0.446 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.551   | 46.09%   | 8.31   | 0.480  | 4.62   | 0.476   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.528   | 45.31%   | 8.73   | 0.522  | 4.57   | 0.460   | 0.420 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.473   | 48.05%   | 9.83   | 0.451  | 4.58   | 0.425   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.437   | 51.95%   | 9.88   | 0.448  | 4.66   | 0.426   | 0.379 \n",
      "\n",
      "\n",
      "[Sobol Run 7/768] H_Inertia: 0.4005 | BASE_S: 0.2167 | Tau: 0.0228 | Jitter: 0.23\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.204   | 30.47%   | 10.82  | 0.535  | 4.39   | 0.502   | 0.637 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.952   | 35.94%   | 11.66  | 0.524  | 4.31   | 0.437   | 0.501 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.808   | 40.23%   | 12.66  | 0.474  | 4.29   | 0.413   | 0.358 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.736   | 44.14%   | 11.69  | 0.490  | 4.14   | 0.395   | 0.403 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.662   | 44.14%   | 12.12  | 0.450  | 4.15   | 0.383   | 0.363 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.583   | 42.58%   | 11.68  | 0.446  | 3.98   | 0.398   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.548   | 48.05%   | 12.91  | 0.405  | 4.10   | 0.354   | 0.385 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.505   | 49.61%   | 13.17  | 0.406  | 4.33   | 0.346   | 0.393 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.456   | 48.83%   | 13.52  | 0.408  | 4.18   | 0.348   | 0.355 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.446   | 50.39%   | 13.17  | 0.388  | 4.23   | 0.341   | 0.377 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.402   | 50.78%   | 13.49  | 0.386  | 4.27   | 0.338   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.374   | 50.00%   | 13.07  | 0.377  | 4.13   | 0.343   | 0.384 \n",
      "\n",
      "\n",
      "[Sobol Run 8/768] H_Inertia: 0.6608 | BASE_S: 0.8139 | Tau: 0.0228 | Jitter: 0.23\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.775   | 11.72%   | 7.82   | 0.423  | 3.90   | 0.338   | 0.532 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.316   | 18.75%   | 4.77   | 0.805  | 4.48   | 0.537   | 0.678 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "EPOCH 3/12: 100%|██████████| 22/22 [00:19<00:00,  1.13it/s, loss=2.1398, acc=22.50%]2026-04-04 23:11:19.039801: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n",
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.140   | 28.52%   | 6.49   | 0.675  | 4.00   | 0.499   | 0.194 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.065   | 29.30%   | 7.88   | 0.638  | 3.86   | 0.518   | 0.450 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.024   | 26.56%   | 7.44   | 0.651  | 3.92   | 0.556   | 0.385 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.958   | 33.59%   | 8.94   | 0.494  | 3.83   | 0.546   | 0.552 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.933   | 32.42%   | 9.50   | 0.501  | 3.68   | 0.547   | 0.426 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.889   | 24.22%   | 7.69   | 0.559  | 3.78   | 0.535   | 0.493 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.986   | 35.55%   | 9.79   | 0.499  | 3.47   | 0.517   | 0.361 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.857   | 36.72%   | 10.89  | 0.433  | 3.57   | 0.506   | 0.332 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.872   | 35.55%   | 11.09  | 0.424  | 3.47   | 0.505   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.766   | 39.84%   | 11.37  | 0.387  | 3.68   | 0.502   | 0.396 \n",
      "\n",
      "\n",
      "[Sobol Run 9/768] H_Inertia: 0.6608 | BASE_S: 0.2167 | Tau: 0.0228 | Jitter: 0.23\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.138   | 31.64%   | 9.64   | 0.507  | 4.40   | 0.489   | 0.608 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.994   | 33.20%   | 9.60   | 0.455  | 4.76   | 0.505   | 0.461 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.881   | 32.42%   | 10.05  | 0.497  | 4.78   | 0.501   | 0.592 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.812   | 41.02%   | 10.52  | 0.436  | 4.81   | 0.431   | 0.444 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.700   | 40.23%   | 10.34  | 0.417  | 4.74   | 0.442   | 0.455 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.665   | 43.36%   | 10.49  | 0.386  | 4.41   | 0.424   | 0.482 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.617   | 46.48%   | 10.57  | 0.374  | 4.71   | 0.374   | 0.434 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.532   | 49.61%   | 10.21  | 0.405  | 4.56   | 0.402   | 0.569 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.480   | 48.44%   | 9.97   | 0.415  | 4.49   | 0.398   | 0.581 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.470   | 48.83%   | 10.63  | 0.388  | 4.44   | 0.384   | 0.520 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.425   | 46.88%   | 10.47  | 0.376  | 4.47   | 0.389   | 0.479 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.396   | 48.44%   | 10.39  | 0.394  | 4.48   | 0.402   | 0.517 \n",
      "\n",
      "\n",
      "[Sobol Run 10/768] H_Inertia: 0.6608 | BASE_S: 0.2167 | Tau: 0.0228 | Jitter: 0.81\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.148   | 30.86%   | 7.88   | 0.536  | 4.23   | 0.557   | 0.574 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.998   | 35.55%   | 8.56   | 0.529  | 4.25   | 0.575   | 0.312 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.945   | 37.50%   | 9.37   | 0.521  | 4.35   | 0.504   | 0.426 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.864   | 38.28%   | 9.59   | 0.490  | 4.75   | 0.461   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.749   | 41.41%   | 10.99  | 0.408  | 4.41   | 0.495   | 0.326 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.655   | 33.20%   | 9.78   | 0.397  | 4.69   | 0.489   | 0.478 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.035   | 30.86%   | 9.96   | 0.382  | 3.01   | 0.482   | 0.503 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.944   | 44.92%   | 11.72  | 0.430  | 4.61   | 0.460   | 0.398 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.628   | 42.58%   | 11.08  | 0.386  | 4.66   | 0.438   | 0.376 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.573   | 47.66%   | 11.49  | 0.392  | 4.82   | 0.474   | 0.385 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.489   | 47.27%   | 11.70  | 0.406  | 4.56   | 0.441   | 0.375 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.451   | 49.61%   | 11.66  | 0.397  | 4.65   | 0.467   | 0.445 \n",
      "\n",
      "\n",
      "[Sobol Run 11/768] H_Inertia: 0.6608 | BASE_S: 0.2167 | Tau: 0.0228 | Jitter: 0.23\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.172   | 22.66%   | 6.89   | 0.547  | 4.57   | 0.529   | 0.563 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.078   | 30.47%   | 7.82   | 0.542  | 4.45   | 0.524   | 0.487 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.924   | 35.55%   | 9.03   | 0.509  | 4.18   | 0.521   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.824   | 39.84%   | 10.36  | 0.440  | 4.45   | 0.516   | 0.354 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.735   | 42.58%   | 10.92  | 0.431  | 4.53   | 0.488   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.639   | 40.23%   | 10.59  | 0.390  | 4.07   | 0.536   | 0.287 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.580   | 44.53%   | 11.27  | 0.405  | 4.28   | 0.496   | 0.387 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.537   | 44.53%   | 11.47  | 0.376  | 4.43   | 0.535   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.468   | 48.44%   | 10.87  | 0.405  | 4.34   | 0.504   | 0.489 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.509   | 46.48%   | 10.74  | 0.394  | 4.28   | 0.447   | 0.449 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.499   | 45.31%   | 11.54  | 0.379  | 4.41   | 0.547   | 0.382 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.401   | 48.44%   | 11.35  | 0.373  | 4.29   | 0.483   | 0.361 \n",
      "\n",
      "\n",
      "[Sobol Run 12/768] H_Inertia: 0.8905 | BASE_S: 0.3239 | Tau: 0.0266 | Jitter: 0.26\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.243   | 25.39%   | 6.81   | 0.726  | 5.16   | 0.440   | 0.672 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.033   | 30.86%   | 8.96   | 0.591  | 4.79   | 0.455   | 0.621 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.978   | 33.59%   | 9.03   | 0.570  | 4.81   | 0.442   | 0.523 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.915   | 32.81%   | 9.66   | 0.550  | 4.66   | 0.464   | 0.532 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.846   | 41.02%   | 10.98  | 0.512  | 4.71   | 0.466   | 0.487 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.784   | 39.06%   | 11.17  | 0.492  | 4.58   | 0.423   | 0.457 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.709   | 42.58%   | 11.67  | 0.422  | 4.84   | 0.401   | 0.465 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.642   | 43.75%   | 12.34  | 0.428  | 4.84   | 0.411   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.555   | 46.48%   | 12.57  | 0.429  | 4.86   | 0.409   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.497   | 44.14%   | 12.85  | 0.409  | 4.62   | 0.403   | 0.374 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.439   | 45.70%   | 12.55  | 0.417  | 4.61   | 0.399   | 0.353 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.417   | 44.92%   | 12.83  | 0.420  | 4.69   | 0.403   | 0.373 \n",
      "\n",
      "\n",
      "[Sobol Run 13/768] H_Inertia: 0.8905 | BASE_S: 0.3239 | Tau: 0.0473 | Jitter: 0.26\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.221   | 21.09%   | 5.19   | 0.771  | 4.45   | 0.425   | 0.235 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.123   | 22.66%   | 6.15   | 0.779  | 4.34   | 0.441   | 0.240 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.062   | 24.61%   | 6.41   | 0.698  | 4.27   | 0.504   | 0.383 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.027   | 30.08%   | 8.68   | 0.562  | 4.51   | 0.440   | 0.502 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.900   | 34.77%   | 9.96   | 0.512  | 4.40   | 0.411   | 0.514 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.786   | 40.62%   | 9.79   | 0.495  | 4.42   | 0.400   | 0.459 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "EPOCH 7/12: 100%|██████████| 22/22 [00:19<00:00,  1.11it/s, loss=1.6489, acc=40.94%]2026-04-04 23:38:52.876071: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n",
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.649   | 46.48%   | 11.54  | 0.446  | 4.50   | 0.407   | 0.467 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.563   | 48.44%   | 10.97  | 0.431  | 4.66   | 0.403   | 0.434 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.481   | 46.88%   | 10.82  | 0.426  | 4.53   | 0.397   | 0.393 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.477   | 50.00%   | 10.44  | 0.427  | 4.44   | 0.394   | 0.341 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.415   | 50.39%   | 12.00  | 0.425  | 4.52   | 0.387   | 0.360 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.381   | 55.86%   | 10.73  | 0.425  | 4.41   | 0.396   | 0.238 \n",
      "\n",
      "\n",
      "[Sobol Run 14/768] H_Inertia: 0.1708 | BASE_S: 0.3239 | Tau: 0.0266 | Jitter: 0.26\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.172   | 31.64%   | 13.59  | 0.443  | 4.00   | 0.451   | 0.377 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.938   | 35.16%   | 13.09  | 0.465  | 3.51   | 0.415   | 0.335 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.833   | 38.67%   | 14.31  | 0.392  | 3.51   | 0.374   | 0.299 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.747   | 42.97%   | 15.18  | 0.351  | 3.77   | 0.371   | 0.225 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.710   | 42.19%   | 14.27  | 0.340  | 3.53   | 0.338   | 0.297 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.589   | 42.97%   | 15.02  | 0.324  | 3.60   | 0.385   | 0.336 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.490   | 50.39%   | 14.62  | 0.348  | 3.76   | 0.408   | 0.356 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.438   | 49.22%   | 15.30  | 0.306  | 4.23   | 0.400   | 0.288 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.389   | 52.73%   | 14.67  | 0.336  | 4.13   | 0.431   | 0.277 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.341   | 50.39%   | 15.33  | 0.313  | 4.09   | 0.416   | 0.284 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.287   | 57.81%   | 14.82  | 0.326  | 3.88   | 0.465   | 0.311 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.252   | 57.42%   | 14.53  | 0.329  | 4.04   | 0.465   | 0.275 \n",
      "\n",
      "\n",
      "[Sobol Run 15/768] H_Inertia: 0.8905 | BASE_S: 0.7067 | Tau: 0.0266 | Jitter: 0.26\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.265   | 22.66%   | 9.00   | 0.614  | 4.95   | 0.343   | 0.282 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.097   | 28.52%   | 7.21   | 0.807  | 4.65   | 0.378   | 0.793 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.084   | 19.92%   | 6.35   | 0.715  | 4.66   | 0.365   | 0.575 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.016   | 25.00%   | 7.38   | 0.714  | 4.62   | 0.349   | 0.674 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.911   | 33.59%   | 7.71   | 0.640  | 4.54   | 0.363   | 0.672 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.833   | 36.72%   | 8.38   | 0.563  | 4.45   | 0.359   | 0.533 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.751   | 37.89%   | 9.02   | 0.546  | 4.58   | 0.353   | 0.539 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.686   | 39.06%   | 8.62   | 0.539  | 4.60   | 0.339   | 0.308 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.604   | 39.84%   | 9.19   | 0.522  | 4.57   | 0.342   | 0.377 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.586   | 38.28%   | 8.69   | 0.502  | 4.56   | 0.340   | 0.297 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.580   | 39.06%   | 9.13   | 0.499  | 4.54   | 0.348   | 0.250 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.600   | 44.14%   | 8.75   | 0.496  | 4.56   | 0.350   | 0.275 \n",
      "\n",
      "\n",
      "[Sobol Run 16/768] H_Inertia: 0.8905 | BASE_S: 0.3239 | Tau: 0.0266 | Jitter: 0.26\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.232   | 22.66%   | 5.70   | 0.759  | 4.90   | 0.436   | 0.745 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.079   | 25.78%   | 7.55   | 0.647  | 4.58   | 0.496   | 0.654 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.939   | 30.86%   | 8.51   | 0.487  | 4.57   | 0.452   | 0.579 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.829   | 41.02%   | 8.74   | 0.460  | 4.28   | 0.446   | 0.496 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.710   | 32.81%   | 9.73   | 0.427  | 4.39   | 0.445   | 0.461 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.602   | 41.80%   | 9.53   | 0.427  | 4.37   | 0.441   | 0.432 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.574   | 45.70%   | 10.37  | 0.417  | 4.63   | 0.438   | 0.413 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.505   | 45.31%   | 10.09  | 0.408  | 4.54   | 0.417   | 0.420 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.453   | 48.44%   | 10.75  | 0.405  | 4.45   | 0.398   | 0.359 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.390   | 52.34%   | 11.01  | 0.398  | 4.59   | 0.395   | 0.380 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.351   | 52.34%   | 11.20  | 0.401  | 4.60   | 0.381   | 0.358 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.318   | 51.17%   | 10.92  | 0.391  | 4.59   | 0.381   | 0.361 \n",
      "\n",
      "\n",
      "[Sobol Run 17/768] H_Inertia: 0.8905 | BASE_S: 0.3239 | Tau: 0.0266 | Jitter: 0.78\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.271   | 23.44%   | 6.60   | 0.704  | 4.42   | 0.452   | 0.282 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.097   | 26.95%   | 7.35   | 0.695  | 4.55   | 0.389   | 0.430 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.019   | 27.34%   | 8.62   | 0.513  | 4.32   | 0.382   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.928   | 38.28%   | 10.64  | 0.516  | 4.67   | 0.415   | 0.503 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.794   | 41.02%   | 11.81  | 0.464  | 4.79   | 0.398   | 0.550 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.658   | 44.14%   | 12.08  | 0.442  | 4.77   | 0.410   | 0.508 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.587   | 46.88%   | 12.69  | 0.405  | 4.86   | 0.382   | 0.495 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.554   | 46.09%   | 11.72  | 0.401  | 4.77   | 0.399   | 0.499 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.531   | 46.48%   | 12.15  | 0.409  | 4.89   | 0.410   | 0.472 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.469   | 48.44%   | 12.65  | 0.404  | 4.70   | 0.410   | 0.471 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.487   | 50.39%   | 13.51  | 0.390  | 4.64   | 0.404   | 0.456 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.436   | 49.61%   | 12.54  | 0.419  | 4.70   | 0.438   | 0.460 \n",
      "\n",
      "\n",
      "[Sobol Run 18/768] H_Inertia: 0.1708 | BASE_S: 0.7067 | Tau: 0.0266 | Jitter: 0.78\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 18 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 19/768] H_Inertia: 0.8905 | BASE_S: 0.7067 | Tau: 0.0473 | Jitter: 0.78\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.390   | 24.22%   | 7.68   | 0.597  | 4.49   | 0.410   | 0.546 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.059   | 25.39%   | 7.22   | 0.696  | 4.49   | 0.423   | 0.774 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.969   | 31.64%   | 8.06   | 0.594  | 4.73   | 0.382   | 0.682 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.892   | 30.86%   | 9.00   | 0.655  | 4.39   | 0.379   | 0.752 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.772   | 33.20%   | 9.79   | 0.553  | 4.51   | 0.404   | 0.658 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.667   | 40.62%   | 10.31  | 0.529  | 4.74   | 0.400   | 0.599 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.583   | 43.75%   | 10.44  | 0.498  | 4.73   | 0.422   | 0.506 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.511   | 50.39%   | 10.92  | 0.466  | 4.57   | 0.413   | 0.464 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.451   | 46.88%   | 10.44  | 0.501  | 4.79   | 0.434   | 0.391 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.455   | 44.92%   | 10.29  | 0.448  | 4.76   | 0.408   | 0.432 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.407   | 51.95%   | 10.48  | 0.459  | 4.83   | 0.389   | 0.399 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.383   | 51.56%   | 10.99  | 0.465  | 4.75   | 0.407   | 0.373 \n",
      "\n",
      "\n",
      "[Sobol Run 20/768] H_Inertia: 0.1708 | BASE_S: 0.3239 | Tau: 0.0473 | Jitter: 0.78\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.251   | 23.05%   | 11.05  | 0.489  | 4.55   | 0.420   | 0.542 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.008   | 33.98%   | 12.94  | 0.443  | 4.39   | 0.358   | 0.478 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.818   | 39.45%   | 14.58  | 0.366  | 4.50   | 0.295   | 0.353 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.685   | 42.19%   | 15.40  | 0.357  | 4.36   | 0.325   | 0.403 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.582   | 44.53%   | 15.45  | 0.323  | 4.10   | 0.295   | 0.365 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.508   | 47.27%   | 15.79  | 0.325  | 4.03   | 0.288   | 0.311 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.448   | 45.31%   | 16.08  | 0.309  | 4.29   | 0.279   | 0.323 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.426   | 46.48%   | 15.68  | 0.289  | 4.21   | 0.283   | 0.338 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.375   | 49.22%   | 15.58  | 0.294  | 3.13   | 0.273   | 0.331 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.351   | 48.44%   | 15.26  | 0.292  | 4.06   | 0.276   | 0.329 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.332   | 50.78%   | 15.58  | 0.278  | 3.20   | 0.263   | 0.344 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.460   | 45.31%   | 9.28   | 0.310  | 3.39   | 0.432   | 0.432 \n",
      "\n",
      "\n",
      "[Sobol Run 21/768] H_Inertia: 0.1708 | BASE_S: 0.7067 | Tau: 0.0473 | Jitter: 0.78\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 21 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 22/768] H_Inertia: 0.1708 | BASE_S: 0.7067 | Tau: 0.0473 | Jitter: 0.26\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.326   | 22.66%   | 12.00  | 0.517  | 4.65   | 0.363   | 0.596 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.080   | 33.20%   | 10.87  | 0.537  | 4.06   | 0.393   | 0.708 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.913   | 34.38%   | 14.44  | 0.390  | 4.34   | 0.317   | 0.564 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.794   | 38.28%   | 14.91  | 0.341  | 4.58   | 0.281   | 0.515 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.695   | 41.41%   | 15.34  | 0.338  | 4.46   | 0.282   | 0.498 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.597   | 38.28%   | 15.10  | 0.337  | 4.45   | 0.292   | 0.499 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.506   | 43.75%   | 15.75  | 0.314  | 4.52   | 0.286   | 0.439 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.479   | 47.27%   | 15.96  | 0.320  | 4.44   | 0.288   | 0.454 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.440   | 46.09%   | 14.64  | 0.322  | 4.34   | 0.289   | 0.487 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.417   | 44.14%   | 15.03  | 0.318  | 4.59   | 0.287   | 0.446 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.385   | 50.00%   | 15.56  | 0.305  | 4.61   | 0.280   | 0.439 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.334   | 50.78%   | 15.81  | 0.308  | 4.59   | 0.276   | 0.401 \n",
      "\n",
      "\n",
      "[Sobol Run 23/768] H_Inertia: 0.1708 | BASE_S: 0.7067 | Tau: 0.0473 | Jitter: 0.78\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.468   | 20.31%   | 9.41   | 0.316  | 5.23   | 0.363   | 0.392 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.197   | 23.05%   | 11.46  | 0.440  | 4.70   | 0.294   | 0.552 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.212   | 25.39%   | 10.28  | 0.475  | 4.66   | 0.338   | 0.632 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.078   | 29.69%   | 12.80  | 0.384  | 4.51   | 0.374   | 0.525 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.042   | 35.55%   | 12.30  | 0.364  | 4.60   | 0.315   | 0.452 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.942   | 35.94%   | 14.36  | 0.346  | 4.59   | 0.319   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.886   | 35.55%   | 13.73  | 0.347  | 4.40   | 0.293   | 0.337 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 9.056   | 9.38%    | 7.74   | 0.273  | 3.06   | 0.211   | 0.396 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 23 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 24/768] H_Inertia: 0.1555 | BASE_S: 0.5689 | Tau: 0.0389 | Jitter: 0.54\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.357   | 26.95%   | 11.14  | 0.557  | 4.81   | 0.361   | 0.723 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.988   | 35.94%   | 11.99  | 0.486  | 4.90   | 0.386   | 0.502 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.856   | 37.89%   | 13.09  | 0.451  | 4.90   | 0.356   | 0.530 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.220   | 40.62%   | 8.30   | 0.371  | 3.10   | 0.298   | 0.475 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.652   | 41.41%   | 13.31  | 0.386  | 4.91   | 0.353   | 0.512 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.635   | 44.53%   | 14.18  | 0.367  | 4.92   | 0.384   | 0.343 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.606   | 44.53%   | 13.73  | 0.361  | 4.89   | 0.330   | 0.470 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.543   | 48.05%   | 14.51  | 0.362  | 4.75   | 0.338   | 0.440 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.486   | 45.70%   | 14.53  | 0.356  | 4.81   | 0.350   | 0.424 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.445   | 48.83%   | 15.38  | 0.326  | 4.75   | 0.341   | 0.392 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.402   | 49.61%   | 14.96  | 0.314  | 4.82   | 0.340   | 0.394 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.356   | 50.39%   | 15.17  | 0.327  | 4.79   | 0.339   | 0.403 \n",
      "\n",
      "\n",
      "[Sobol Run 25/768] H_Inertia: 0.1555 | BASE_S: 0.5689 | Tau: 0.0351 | Jitter: 0.54\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.365   | 23.83%   | 10.29  | 0.445  | 4.66   | 0.436   | 0.585 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.045   | 29.30%   | 11.06  | 0.490  | 4.64   | 0.392   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.914   | 25.39%   | 11.11  | 0.484  | 4.97   | 0.376   | 0.343 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.938   | 37.89%   | 11.01  | 0.442  | 4.76   | 0.424   | 0.355 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.763   | 42.97%   | 11.80  | 0.423  | 4.62   | 0.415   | 0.278 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.661   | 41.80%   | 11.57  | 0.418  | 4.62   | 0.432   | 0.334 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.633   | 44.92%   | 12.44  | 0.372  | 4.71   | 0.351   | 0.326 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.549   | 46.88%   | 13.40  | 0.374  | 4.64   | 0.435   | 0.358 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.494   | 50.00%   | 12.95  | 0.372  | 4.72   | 0.413   | 0.333 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 2.788   | 13.28%   | 3.91   | 0.381  | 3.65   | 0.273   | 0.430 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 8.535   | 10.55%   | 1.20   | nan    | 2.94   | 0.357   | 0.356 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 2.617   | 20.70%   | 1.22   | nan    | 2.56   | 0.370   | 0.240 \n",
      "\n",
      "\n",
      "[Sobol Run 26/768] H_Inertia: 0.9058 | BASE_S: 0.5689 | Tau: 0.0389 | Jitter: 0.54\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.303   | 23.44%   | 4.30   | 0.803  | 4.87   | 0.496   | 0.881 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.096   | 26.56%   | 5.92   | 0.686  | 4.41   | 0.460   | 0.805 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.980   | 30.08%   | 8.55   | 0.511  | 4.43   | 0.427   | 0.632 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.826   | 34.38%   | 9.78   | 0.450  | 4.53   | 0.403   | 0.529 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-04-05 00:34:51.261772: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.728   | 39.06%   | 10.40  | 0.432  | 4.50   | 0.412   | 0.489 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.622   | 39.06%   | 10.38  | 0.417  | 4.83   | 0.414   | 0.496 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.606   | 40.62%   | 10.57  | 0.423  | 4.81   | 0.407   | 0.477 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.514   | 44.53%   | 10.71  | 0.428  | 4.58   | 0.417   | 0.513 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.485   | 46.48%   | 11.41  | 0.421  | 4.75   | 0.402   | 0.418 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.449   | 45.70%   | 10.94  | 0.424  | 4.70   | 0.415   | 0.478 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.399   | 50.39%   | 11.33  | 0.411  | 4.81   | 0.404   | 0.453 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.379   | 52.73%   | 11.49  | 0.415  | 4.74   | 0.414   | 0.441 \n",
      "\n",
      "\n",
      "[Sobol Run 27/768] H_Inertia: 0.1555 | BASE_S: 0.9517 | Tau: 0.0389 | Jitter: 0.54\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 27 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 28/768] H_Inertia: 0.1555 | BASE_S: 0.5689 | Tau: 0.0389 | Jitter: 0.54\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.008   | 26.17%   | 13.81  | 0.364  | 4.86   | 0.566   | 0.233 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.081   | 33.20%   | 15.59  | 0.347  | 4.71   | 0.461   | 0.314 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.834   | 37.11%   | 16.94  | 0.313  | 4.41   | 0.488   | 0.352 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.707   | 43.36%   | 17.34  | 0.302  | 4.12   | 0.457   | 0.349 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.621   | 42.58%   | 17.12  | 0.301  | 4.17   | 0.485   | 0.340 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.531   | 51.17%   | 17.59  | 0.278  | 4.10   | 0.457   | 0.334 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.518   | 46.09%   | 16.92  | 0.303  | 3.06   | 0.420   | 0.411 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.721   | 40.23%   | 18.51  | 0.287  | 4.25   | 0.294   | 0.245 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.500   | 47.27%   | 17.59  | 0.277  | 4.04   | 0.499   | 0.371 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.460   | 50.78%   | 17.92  | 0.270  | 4.43   | 0.404   | 0.315 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.394   | 52.73%   | 17.52  | 0.276  | 4.34   | 0.432   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.346   | 53.91%   | 17.00  | 0.272  | 4.09   | 0.403   | 0.389 \n",
      "\n",
      "\n",
      "[Sobol Run 29/768] H_Inertia: 0.1555 | BASE_S: 0.5689 | Tau: 0.0389 | Jitter: 1.05\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.650   | 13.28%   | 6.63   | 0.357  | 5.23   | 0.506   | 0.486 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.169   | 27.34%   | 9.07   | 0.411  | 4.80   | 0.449   | 0.554 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.979   | 36.33%   | 10.90  | 0.397  | 4.73   | 0.410   | 0.483 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.812   | 43.75%   | 11.69  | 0.374  | 4.74   | 0.387   | 0.442 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.689   | 44.14%   | 13.24  | 0.366  | 4.65   | 0.365   | 0.415 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.626   | 43.75%   | 12.98  | 0.334  | 4.70   | 0.386   | 0.381 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.579   | 44.14%   | 13.31  | 0.353  | 3.17   | 0.335   | 0.520 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 29 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 30/768] H_Inertia: 0.9058 | BASE_S: 0.9517 | Tau: 0.0389 | Jitter: 1.05\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 13.028  | 8.98%    | 4.11   | 0.761  | 2.31   | 0.373   | 0.867 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.304   | 19.53%   | 5.61   | 0.592  | 4.74   | 0.383   | 0.731 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.153   | 19.92%   | 5.53   | 0.664  | 4.50   | 0.415   | 0.762 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.088   | 21.48%   | 6.16   | 0.584  | 4.70   | 0.431   | 0.706 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.022   | 22.66%   | 6.42   | 0.565  | 4.71   | 0.401   | 0.670 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.988   | 24.61%   | 7.97   | 0.546  | 4.58   | 0.375   | 0.643 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.938   | 28.52%   | 8.07   | 0.504  | 4.51   | 0.346   | 0.609 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.964   | 25.39%   | 8.02   | 0.526  | 4.55   | 0.370   | 0.571 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.903   | 29.30%   | 7.28   | 0.514  | 4.59   | 0.365   | 0.612 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.863   | 25.00%   | 8.83   | 0.469  | 4.19   | 0.332   | 0.529 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.822   | 32.03%   | 9.17   | 0.479  | 4.37   | 0.372   | 0.625 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.789   | 33.20%   | 9.92   | 0.487  | 4.54   | 0.361   | 0.619 \n",
      "\n",
      "\n",
      "[Sobol Run 31/768] H_Inertia: 0.1555 | BASE_S: 0.9517 | Tau: 0.0351 | Jitter: 1.05\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 31 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 32/768] H_Inertia: 0.9058 | BASE_S: 0.5689 | Tau: 0.0351 | Jitter: 1.05\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.958   | 10.94%   | 11.55  | 0.421  | 2.67   | 0.465   | 0.623 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 9.354   | 12.11%   | 23.68  | 0.224  | 4.09   | 0.403   | 0.476 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 16.310  | 11.33%   | 23.22  | 0.225  | 4.03   | 0.330   | 0.491 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 20.594  | 10.55%   | 24.56  | 0.140  | 3.91   | 0.316   | 0.394 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 22.358  | 11.72%   | 22.25  | 0.128  | 3.85   | 0.252   | 0.344 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 23.189  | 8.98%    | 22.50  | 0.147  | 3.82   | 0.267   | 0.393 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 21.437  | 11.33%   | 22.70  | 0.162  | 3.79   | 0.342   | 0.408 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 20.890  | 10.94%   | 23.63  | 0.132  | 3.83   | 0.278   | 0.359 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 21.971  | 10.94%   | 20.96  | 0.127  | 3.53   | 0.350   | 0.336 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 23.027  | 12.11%   | 17.25  | nan    | 3.44   | 0.304   | 0.228 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 21.482  | 11.33%   | 15.18  | nan    | 3.65   | 0.276   | 0.326 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 21.065  | 13.67%   | 11.79  | nan    | 2.78   | 0.283   | 0.440 \n",
      "\n",
      "\n",
      "[Sobol Run 33/768] H_Inertia: 0.9058 | BASE_S: 0.9517 | Tau: 0.0351 | Jitter: 1.05\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 12.302  | 15.23%   | 25.23  | 0.260  | 4.11   | 0.245   | 0.530 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 15.938  | 11.33%   | 25.09  | 0.204  | 4.11   | 0.313   | 0.467 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 16.345  | 10.55%   | 25.37  | 0.179  | 4.09   | 0.224   | 0.437 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 16.745  | 8.98%    | 24.89  | 0.216  | 4.07   | 0.247   | 0.473 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 17.511  | 10.55%   | 24.94  | 0.189  | 4.14   | 0.254   | 0.448 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 16.074  | 7.03%    | 25.21  | 0.193  | 4.11   | 0.232   | 0.460 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 17.426  | 9.77%    | 25.27  | 0.194  | 4.18   | 0.372   | 0.452 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 17.109  | 7.81%    | 24.89  | 0.204  | 4.09   | 0.421   | 0.466 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 17.494  | 8.59%    | 24.74  | 0.214  | 4.09   | 0.325   | 0.473 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 16.434  | 6.64%    | 23.90  | 0.211  | 4.05   | 0.292   | 0.455 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 13.609  | 11.33%   | 24.07  | 0.189  | 4.09   | 0.340   | 0.433 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 13.872  | 12.50%   | 24.62  | 0.253  | 4.17   | 0.311   | 0.516 \n",
      "\n",
      "\n",
      "[Sobol Run 34/768] H_Inertia: 0.9058 | BASE_S: 0.9517 | Tau: 0.0351 | Jitter: 0.54\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.049   | 25.00%   | 7.37   | 0.617  | 4.74   | 0.347   | 0.740 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.038   | 29.69%   | 8.14   | 0.506  | 4.69   | 0.344   | 0.648 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.956   | 26.17%   | 8.61   | 0.566  | 4.51   | 0.347   | 0.711 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.910   | 28.12%   | 8.49   | 0.519  | 4.42   | 0.374   | 0.635 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.881   | 32.42%   | 8.89   | 0.528  | 4.51   | 0.389   | 0.637 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.848   | 30.86%   | 8.81   | 0.526  | 4.58   | 0.344   | 0.653 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.760   | 36.72%   | 8.84   | 0.516  | 4.55   | 0.349   | 0.630 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.729   | 34.77%   | 9.21   | 0.509  | 4.61   | 0.344   | 0.606 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.698   | 33.98%   | 9.08   | 0.507  | 4.42   | 0.336   | 0.593 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.720   | 35.55%   | 9.65   | 0.480  | 4.10   | 0.361   | 0.548 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.699   | 37.89%   | 9.78   | 0.513  | 4.70   | 0.327   | 0.618 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.644   | 32.81%   | 10.06  | 0.482  | 3.94   | 0.339   | 0.564 \n",
      "\n",
      "\n",
      "[Sobol Run 35/768] H_Inertia: 0.9058 | BASE_S: 0.9517 | Tau: 0.0351 | Jitter: 1.05\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 25.137  | 13.28%   | 25.71  | 0.080  | 3.87   | 0.335   | 0.298 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 25.330  | 12.11%   | 25.95  | 0.072  | 3.87   | 0.245   | 0.272 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 25.171  | 10.16%   | 25.79  | 0.094  | 3.88   | 0.253   | 0.319 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 23.497  | 9.38%    | 25.84  | 0.229  | 3.97   | 0.436   | 0.502 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 22.877  | 12.50%   | 25.57  | 0.156  | 3.86   | 0.395   | 0.406 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 24.310  | 9.77%    | 25.72  | 0.171  | 3.96   | 0.353   | 0.423 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 25.102  | 11.33%   | 25.90  | 0.083  | 3.93   | 0.322   | 0.303 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 25.032  | 11.72%   | 25.94  | 0.082  | 3.94   | 0.454   | 0.304 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 25.340  | 10.94%   | 25.81  | 0.081  | 3.92   | 0.300   | 0.301 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 25.349  | 13.67%   | 25.86  | 0.096  | 3.98   | 0.322   | 0.330 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 24.500  | 8.20%    | 25.98  | 0.070  | 3.91   | 0.383   | 0.270 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 24.473  | 11.33%   | 25.88  | 0.083  | 3.93   | 0.236   | 0.300 \n",
      "\n",
      "\n",
      "[Sobol Run 36/768] H_Inertia: 0.6455 | BASE_S: 0.0789 | Tau: 0.0144 | Jitter: 1.09\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.243   | 27.73%   | 8.47   | 0.600  | 3.92   | 0.513   | 0.607 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.062   | 33.98%   | 11.09  | 0.459  | 4.60   | 0.439   | 0.319 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.927   | 35.16%   | 11.85  | 0.442  | 4.66   | 0.409   | 0.349 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.803   | 41.41%   | 12.66  | 0.404  | 4.75   | 0.383   | 0.428 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.702   | 44.53%   | 13.07  | 0.380  | 4.74   | 0.384   | 0.491 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.619   | 45.31%   | 13.34  | 0.369  | 4.54   | 0.371   | 0.401 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.577   | 49.22%   | 13.14  | 0.364  | 4.61   | 0.365   | 0.418 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.538   | 39.84%   | 12.55  | 0.362  | 4.76   | 0.369   | 0.355 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.533   | 47.66%   | 13.93  | 0.365  | 4.61   | 0.342   | 0.371 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.499   | 50.00%   | 13.69  | 0.362  | 4.69   | 0.357   | 0.358 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.448   | 51.17%   | 14.17  | 0.342  | 4.60   | 0.363   | 0.416 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.444   | 50.39%   | 14.24  | 0.346  | 4.68   | 0.344   | 0.385 \n",
      "\n",
      "\n",
      "[Sobol Run 37/768] H_Inertia: 0.6455 | BASE_S: 0.0789 | Tau: 0.0106 | Jitter: 1.09\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.230   | 24.22%   | 8.41   | 0.642  | 3.77   | 0.531   | 0.694 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.066   | 33.20%   | 10.25  | 0.555  | 4.11   | 0.494   | 0.565 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.902   | 34.38%   | 11.49  | 0.463  | 4.71   | 0.434   | 0.425 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.779   | 38.67%   | 11.96  | 0.442  | 4.45   | 0.387   | 0.374 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.725   | 43.36%   | 12.15  | 0.401  | 4.59   | 0.377   | 0.414 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.652   | 41.02%   | 12.42  | 0.389  | 4.58   | 0.369   | 0.425 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.632   | 44.53%   | 12.78  | 0.390  | 4.54   | 0.362   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.546   | 48.44%   | 12.76  | 0.393  | 4.63   | 0.363   | 0.357 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.520   | 48.44%   | 12.74  | 0.389  | 4.52   | 0.335   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.480   | 50.39%   | 13.14  | 0.370  | 4.63   | 0.330   | 0.397 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.454   | 48.83%   | 13.35  | 0.365  | 4.70   | 0.331   | 0.398 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.406   | 52.34%   | 12.68  | 0.359  | 4.72   | 0.323   | 0.441 \n",
      "\n",
      "\n",
      "[Sobol Run 38/768] H_Inertia: 0.4158 | BASE_S: 0.0789 | Tau: 0.0144 | Jitter: 1.09\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.152   | 30.86%   | 9.64   | 0.592  | 3.76   | 0.539   | 0.699 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.981   | 34.77%   | 12.19  | 0.492  | 4.19   | 0.410   | 0.528 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.860   | 41.41%   | 13.45  | 0.393  | 4.33   | 0.359   | 0.441 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.767   | 44.14%   | 13.91  | 0.398  | 4.04   | 0.360   | 0.454 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.656   | 47.27%   | 14.75  | 0.363  | 3.94   | 0.345   | 0.469 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.585   | 46.88%   | 15.10  | 0.326  | 3.78   | 0.332   | 0.401 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.506   | 49.61%   | 15.09  | 0.326  | 3.78   | 0.332   | 0.440 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.437   | 50.00%   | 14.82  | 0.327  | 3.94   | 0.325   | 0.446 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.403   | 50.39%   | 14.76  | 0.328  | 3.88   | 0.325   | 0.454 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.354   | 54.30%   | 14.20  | 0.326  | 3.94   | 0.316   | 0.450 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.319   | 55.08%   | 14.89  | 0.316  | 4.05   | 0.320   | 0.424 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.294   | 54.30%   | 14.86  | 0.323  | 4.27   | 0.317   | 0.411 \n",
      "\n",
      "\n",
      "[Sobol Run 39/768] H_Inertia: 0.6455 | BASE_S: 0.4617 | Tau: 0.0144 | Jitter: 1.09\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.194   | 26.95%   | 10.20  | 0.581  | 4.35   | 0.453   | 0.468 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.042   | 31.64%   | 10.11  | 0.544  | 4.26   | 0.430   | 0.441 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.957   | 34.77%   | 10.38  | 0.475  | 4.06   | 0.409   | 0.249 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.913   | 36.72%   | 11.36  | 0.432  | 4.70   | 0.390   | 0.422 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.811   | 41.02%   | 11.62  | 0.436  | 4.35   | 0.364   | 0.386 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.714   | 40.62%   | 11.47  | 0.424  | 4.06   | 0.352   | 0.297 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.659   | 46.09%   | 12.37  | 0.408  | 4.38   | 0.356   | 0.327 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.596   | 43.36%   | 12.60  | 0.383  | 4.49   | 0.337   | 0.295 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.545   | 46.88%   | 12.45  | 0.416  | 4.36   | 0.351   | 0.340 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.481   | 47.27%   | 12.77  | 0.358  | 2.89   | 0.346   | 0.360 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.473   | 52.34%   | 12.78  | 0.398  | 4.39   | 0.342   | 0.379 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.464   | 50.39%   | 12.36  | 0.417  | 4.42   | 0.350   | 0.360 \n",
      "\n",
      "\n",
      "[Sobol Run 40/768] H_Inertia: 0.6455 | BASE_S: 0.0789 | Tau: 0.0144 | Jitter: 1.09\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.270   | 23.05%   | 6.28   | 0.722  | 3.96   | 0.634   | 0.787 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.084   | 31.64%   | 9.10   | 0.547  | 4.05   | 0.509   | 0.499 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.945   | 35.55%   | 9.91   | 0.517  | 4.20   | 0.501   | 0.314 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.872   | 37.11%   | 10.63  | 0.509  | 4.52   | 0.452   | 0.398 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.779   | 44.14%   | 11.78  | 0.426  | 4.29   | 0.377   | 0.301 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.703   | 46.09%   | 12.73  | 0.387  | 4.33   | 0.367   | 0.350 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.606   | 46.09%   | 13.18  | 0.357  | 4.65   | 0.355   | 0.267 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.552   | 43.75%   | 12.68  | 0.353  | 4.51   | 0.351   | 0.325 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.575   | 45.70%   | 12.42  | 0.351  | 4.48   | 0.366   | 0.325 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.547   | 41.02%   | 9.02   | 0.413  | 4.34   | 0.394   | 0.507 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.645   | 47.66%   | 13.24  | 0.358  | 4.45   | 0.363   | 0.258 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.485   | 52.73%   | 11.73  | 0.383  | 4.41   | 0.343   | 0.313 \n",
      "\n",
      "\n",
      "[Sobol Run 41/768] H_Inertia: 0.6455 | BASE_S: 0.0789 | Tau: 0.0144 | Jitter: 0.50\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.225   | 26.17%   | 7.47   | 0.670  | 3.82   | 0.549   | 0.731 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.032   | 28.12%   | 10.35  | 0.566  | 4.29   | 0.492   | 0.217 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.914   | 35.16%   | 10.79  | 0.486  | 4.41   | 0.446   | 0.283 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.820   | 37.11%   | 11.31  | 0.443  | 4.21   | 0.379   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.754   | 41.02%   | 12.48  | 0.415  | 4.33   | 0.371   | 0.350 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.700   | 43.36%   | 12.86  | 0.401  | 4.47   | 0.363   | 0.329 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.647   | 43.75%   | 12.70  | 0.407  | 4.47   | 0.366   | 0.317 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.590   | 48.44%   | 12.81  | 0.398  | 4.55   | 0.381   | 0.300 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.579   | 43.36%   | 13.21  | 0.399  | 4.51   | 0.392   | 0.316 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.549   | 46.09%   | 12.96  | 0.374  | 4.60   | 0.368   | 0.297 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.566   | 50.78%   | 12.67  | 0.393  | 4.60   | 0.376   | 0.299 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.529   | 47.66%   | 12.91  | 0.390  | 4.58   | 0.377   | 0.269 \n",
      "\n",
      "\n",
      "[Sobol Run 42/768] H_Inertia: 0.4158 | BASE_S: 0.4617 | Tau: 0.0144 | Jitter: 0.50\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.365   | 19.53%   | 7.60   | 0.635  | 3.98   | 0.438   | 0.770 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.133   | 24.22%   | 9.96   | 0.575  | 3.73   | 0.449   | 0.629 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.022   | 29.69%   | 9.40   | 0.552  | 3.92   | 0.455   | 0.638 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.935   | 37.89%   | 11.95  | 0.406  | 4.08   | 0.388   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.843   | 35.94%   | 11.75  | 0.431  | 3.99   | 0.379   | 0.401 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.781   | 43.36%   | 13.21  | 0.386  | 3.96   | 0.357   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.664   | 45.70%   | 13.22  | 0.354  | 3.86   | 0.323   | 0.422 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.597   | 51.17%   | 13.34  | 0.342  | 3.73   | 0.296   | 0.413 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.544   | 50.78%   | 13.03  | 0.358  | 3.79   | 0.321   | 0.439 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.526   | 51.56%   | 13.01  | 0.345  | 3.73   | 0.319   | 0.421 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.451   | 53.91%   | 13.56  | 0.337  | 3.69   | 0.312   | 0.462 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.424   | 55.08%   | 13.59  | 0.331  | 3.75   | 0.309   | 0.432 \n",
      "\n",
      "\n",
      "[Sobol Run 43/768] H_Inertia: 0.6455 | BASE_S: 0.4617 | Tau: 0.0106 | Jitter: 0.50\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.323   | 16.02%   | 4.74   | 0.673  | 4.68   | 0.529   | 0.425 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.148   | 23.83%   | 4.64   | 0.762  | 4.63   | 0.450   | 0.581 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.068   | 27.73%   | 5.82   | 0.649  | 4.64   | 0.487   | 0.446 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.003   | 28.12%   | 5.23   | 0.637  | 4.59   | 0.479   | 0.354 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.030   | 33.59%   | 7.00   | 0.567  | 4.54   | 0.419   | 0.415 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.946   | 33.98%   | 7.06   | 0.479  | 4.71   | 0.390   | 0.481 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.834   | 33.20%   | 7.24   | 0.489  | 4.79   | 0.381   | 0.504 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.772   | 31.64%   | 7.78   | 0.487  | 4.70   | 0.379   | 0.454 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.767   | 38.28%   | 7.25   | 0.478  | 4.65   | 0.385   | 0.423 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.806   | 31.25%   | 7.86   | 0.453  | 4.80   | 0.368   | 0.437 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.760   | 39.84%   | 8.15   | 0.473  | 4.59   | 0.371   | 0.405 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.677   | 42.97%   | 8.03   | 0.454  | 4.64   | 0.362   | 0.479 \n",
      "\n",
      "\n",
      "[Sobol Run 44/768] H_Inertia: 0.4158 | BASE_S: 0.0789 | Tau: 0.0106 | Jitter: 0.50\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.157   | 30.08%   | 10.91  | 0.517  | 3.76   | 0.542   | 0.587 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.939   | 33.98%   | 11.37  | 0.480  | 4.01   | 0.426   | 0.355 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.809   | 43.75%   | 12.39  | 0.407  | 3.71   | 0.450   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.742   | 41.80%   | 11.69  | 0.431  | 3.73   | 0.471   | 0.441 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.677   | 44.92%   | 13.93  | 0.369  | 4.00   | 0.406   | 0.423 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.607   | 42.97%   | 13.00  | 0.369  | 3.83   | 0.414   | 0.433 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.608   | 48.05%   | 14.44  | 0.346  | 4.04   | 0.370   | 0.377 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.507   | 50.78%   | 14.00  | 0.351  | 4.05   | 0.373   | 0.426 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.490   | 50.78%   | 14.72  | 0.325  | 4.18   | 0.355   | 0.336 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.450   | 50.00%   | 14.04  | 0.331  | 4.13   | 0.364   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.419   | 51.56%   | 13.80  | 0.350  | 3.91   | 0.386   | 0.443 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.408   | 46.48%   | 14.63  | 0.317  | 3.99   | 0.351   | 0.395 \n",
      "\n",
      "\n",
      "[Sobol Run 45/768] H_Inertia: 0.4158 | BASE_S: 0.4617 | Tau: 0.0106 | Jitter: 0.50\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.275   | 16.02%   | 6.22   | 0.673  | 4.23   | 0.530   | 0.707 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.110   | 25.78%   | 9.74   | 0.637  | 3.85   | 0.434   | 0.197 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.955   | 35.94%   | 10.64  | 0.541  | 3.61   | 0.408   | 0.282 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.868   | 34.38%   | 11.18  | 0.496  | 3.36   | 0.368   | 0.387 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.815   | 38.67%   | 12.21  | 0.433  | 3.32   | 0.357   | 0.393 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.771   | 39.45%   | 11.42  | 0.442  | 3.13   | 0.379   | 0.354 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.688   | 48.05%   | 12.10  | 0.449  | 3.41   | 0.374   | 0.460 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.647   | 48.83%   | 12.24  | 0.423  | 3.15   | 0.368   | 0.428 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.615   | 46.88%   | 12.50  | 0.395  | 3.45   | 0.364   | 0.442 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.599   | 48.83%   | 12.09  | 0.403  | 3.01   | 0.371   | 0.403 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.572   | 50.78%   | 11.99  | 0.438  | 3.00   | 0.367   | 0.405 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.519   | 49.22%   | 12.37  | 0.372  | 3.07   | 0.362   | 0.397 \n",
      "\n",
      "\n",
      "[Sobol Run 46/768] H_Inertia: 0.4158 | BASE_S: 0.4617 | Tau: 0.0106 | Jitter: 1.09\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.722   | 23.44%   | 8.28   | 0.638  | 4.98   | 0.509   | 0.514 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.099   | 28.52%   | 10.84  | 0.550  | 4.43   | 0.581   | 0.563 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.022   | 33.20%   | 11.74  | 0.503  | 4.34   | 0.556   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.101   | 19.92%   | 11.21  | 0.396  | 3.53   | 0.534   | 0.568 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 16.389  | 10.55%   | 5.98   | 0.264  | 3.57   | 0.280   | 0.364 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 15.674  | 21.09%   | 15.97  | 0.261  | 3.81   | 0.533   | 0.469 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 17.343  | 10.94%   | 15.86  | 0.254  | 3.73   | 0.314   | 0.459 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 20.978  | 12.89%   | 20.52  | 0.314  | 3.49   | 0.349   | 0.574 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 11.545  | 12.89%   | 5.95   | 0.418  | 4.50   | 0.450   | 0.531 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 2.177   | 33.20%   | 10.75  | 0.473  | 4.35   | 0.410   | 0.378 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.924   | 36.33%   | 12.11  | 0.430  | 4.23   | 0.502   | 0.424 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.844   | 39.45%   | 13.21  | 0.411  | 4.53   | 0.466   | 0.266 \n",
      "\n",
      "\n",
      "[Sobol Run 47/768] H_Inertia: 0.4158 | BASE_S: 0.4617 | Tau: 0.0106 | Jitter: 0.50\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.310   | 24.22%   | 8.52   | 0.646  | 3.89   | 0.446   | 0.641 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.070   | 29.69%   | 8.61   | 0.663  | 3.88   | 0.477   | 0.625 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.010   | 28.12%   | 9.82   | 0.593  | 4.07   | 0.494   | 0.593 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.952   | 30.47%   | 9.21   | 0.547  | 3.85   | 0.410   | 0.578 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.916   | 34.77%   | 11.45  | 0.444  | 3.73   | 0.395   | 0.472 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.916   | 36.33%   | 12.18  | 0.430  | 4.25   | 0.389   | 0.512 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.861   | 38.67%   | 11.64  | 0.423  | 3.99   | 0.372   | 0.452 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.823   | 30.47%   | 11.69  | 0.385  | 3.84   | 0.352   | 0.486 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.788   | 42.19%   | 12.13  | 0.428  | 3.99   | 0.377   | 0.415 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.729   | 44.14%   | 13.01  | 0.387  | 4.00   | 0.336   | 0.404 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.704   | 46.09%   | 12.76  | 0.380  | 4.02   | 0.389   | 0.473 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.646   | 45.70%   | 12.85  | 0.374  | 3.90   | 0.348   | 0.454 \n",
      "\n",
      "\n",
      "[Sobol Run 48/768] H_Inertia: 0.0330 | BASE_S: 0.4464 | Tau: 0.0205 | Jitter: 0.95\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 48 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 49/768] H_Inertia: 0.0330 | BASE_S: 0.4464 | Tau: 0.0167 | Jitter: 0.95\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 11.378  | 9.38%    | 15.26  | 0.234  | 3.38   | 0.256   | 0.441 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 21.892  | 14.84%   | 20.57  | 0.241  | 3.41   | 0.293   | 0.478 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 17.700  | 14.45%   | 21.28  | 0.257  | 3.69   | 0.350   | 0.510 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 19.217  | 8.59%    | 20.90  | 0.335  | 3.68   | 0.475   | 0.572 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 3.092   | 26.56%   | 16.92  | 0.408  | 4.86   | 0.378   | 0.363 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.115   | 27.34%   | 11.84  | 0.359  | 4.79   | 0.385   | 0.303 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.913   | 38.67%   | 16.38  | 0.318  | 4.75   | 0.284   | 0.290 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.691   | 42.19%   | 18.38  | 0.284  | 4.62   | 0.262   | 0.280 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.529   | 46.09%   | 16.94  | 0.287  | 4.53   | 0.279   | 0.242 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.428   | 46.88%   | 18.28  | 0.271  | 4.55   | 0.331   | 0.258 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.336   | 51.95%   | 18.50  | 0.265  | 4.56   | 0.301   | 0.219 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.293   | 53.52%   | 18.54  | 0.266  | 4.55   | 0.298   | 0.222 \n",
      "\n",
      "\n",
      "[Sobol Run 50/768] H_Inertia: 0.7833 | BASE_S: 0.4464 | Tau: 0.0205 | Jitter: 0.95\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.173   | 26.56%   | 8.71   | 0.607  | 4.20   | 0.346   | 0.582 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.027   | 33.20%   | 9.18   | 0.548  | 3.76   | 0.381   | 0.457 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.900   | 33.59%   | 8.70   | 0.520  | 4.02   | 0.396   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.895   | 37.89%   | 8.99   | 0.560  | 4.32   | 0.422   | 0.630 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.816   | 39.06%   | 10.52  | 0.495  | 4.02   | 0.405   | 0.521 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.769   | 38.28%   | 11.04  | 0.473  | 3.97   | 0.416   | 0.540 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.686   | 46.48%   | 12.25  | 0.425  | 3.68   | 0.395   | 0.405 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.682   | 48.44%   | 12.81  | 0.428  | 3.99   | 0.357   | 0.360 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.584   | 48.83%   | 12.63  | 0.414  | 3.97   | 0.337   | 0.393 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.535   | 43.36%   | 11.74  | 0.423  | 4.22   | 0.393   | 0.401 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.486   | 49.22%   | 12.64  | 0.398  | 4.18   | 0.393   | 0.349 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-04-05 02:26:44.538703: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.439   | 46.48%   | 13.15  | 0.394  | 4.39   | 0.388   | 0.299 \n",
      "\n",
      "\n",
      "[Sobol Run 51/768] H_Inertia: 0.0330 | BASE_S: 0.8292 | Tau: 0.0205 | Jitter: 0.95\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 51 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 52/768] H_Inertia: 0.0330 | BASE_S: 0.4464 | Tau: 0.0205 | Jitter: 0.95\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.193   | 29.69%   | 12.55  | 0.443  | 4.58   | 0.417   | 0.517 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.875   | 39.84%   | 14.99  | 0.398  | 4.37   | 0.349   | 0.402 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.707   | 39.45%   | 15.77  | 0.322  | 4.40   | 0.314   | 0.383 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.621   | 44.53%   | 15.46  | 0.332  | 4.22   | 0.341   | 0.355 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.565   | 50.00%   | 15.37  | 0.335  | 4.82   | 0.306   | 0.351 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.485   | 52.34%   | 15.17  | 0.306  | 4.75   | 0.308   | 0.309 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.406   | 51.17%   | 16.02  | 0.301  | 4.86   | 0.293   | 0.286 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.312   | 53.91%   | 17.04  | 0.276  | 4.84   | 0.289   | 0.245 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.250   | 54.30%   | 17.66  | 0.269  | 4.72   | 0.274   | 0.262 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.177   | 59.77%   | 17.19  | 0.273  | 4.76   | 0.262   | 0.252 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.128   | 58.20%   | 18.26  | 0.255  | 4.86   | 0.237   | 0.253 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.121   | 57.42%   | 18.98  | 0.247  | 4.69   | 0.241   | 0.246 \n",
      "\n",
      "\n",
      "[Sobol Run 53/768] H_Inertia: 0.0330 | BASE_S: 0.4464 | Tau: 0.0205 | Jitter: 0.92\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.179   | 37.11%   | 16.93  | 0.302  | 4.11   | 0.342   | 0.414 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.826   | 42.58%   | 17.85  | 0.306  | 4.17   | 0.373   | 0.352 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.639   | 47.27%   | 18.74  | 0.283  | 4.30   | 0.406   | 0.241 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.462   | 53.12%   | 19.47  | 0.258  | 4.25   | 0.343   | 0.227 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.391   | 53.52%   | 18.87  | 0.261  | 3.97   | 0.331   | 0.258 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.321   | 57.42%   | 20.47  | 0.246  | 4.19   | 0.324   | 0.223 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.242   | 56.25%   | 20.27  | 0.250  | 4.39   | 0.338   | 0.238 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.198   | 59.38%   | 19.69  | 0.244  | 4.47   | 0.334   | 0.254 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.157   | 55.86%   | 20.41  | 0.248  | 4.38   | 0.322   | 0.244 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.122   | 60.55%   | 19.99  | 0.254  | 4.39   | 0.313   | 0.259 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.096   | 59.77%   | 20.55  | 0.236  | 4.55   | 0.307   | 0.243 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.060   | 60.16%   | 20.19  | 0.236  | 4.45   | 0.296   | 0.259 \n",
      "\n",
      "\n",
      "[Sobol Run 54/768] H_Inertia: 0.7833 | BASE_S: 0.8292 | Tau: 0.0205 | Jitter: 0.92\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.148   | 12.89%   | 21.62  | 0.487  | 3.14   | 0.411   | 0.708 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 14.130  | 12.89%   | 23.75  | 0.263  | 3.82   | 0.221   | 0.530 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 17.320  | 7.03%    | 23.44  | 0.201  | 3.74   | 0.215   | 0.464 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 21.400  | 10.55%   | 23.39  | 0.164  | 3.53   | 0.198   | 0.416 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 21.771  | 11.72%   | 23.02  | 0.180  | 3.59   | 0.241   | 0.409 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 20.491  | 14.45%   | 22.54  | 0.224  | 3.59   | 0.220   | 0.466 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 16.234  | 10.55%   | 22.27  | 0.182  | 3.45   | 0.209   | 0.411 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 15.560  | 11.72%   | 20.05  | 0.221  | 3.47   | 0.239   | 0.456 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 10.557  | 12.50%   | 19.74  | 0.214  | 3.43   | 0.218   | 0.449 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 13.089  | 12.89%   | 21.49  | 0.223  | 3.67   | 0.210   | 0.457 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 12.362  | 8.98%    | 20.98  | 0.229  | 3.58   | 0.276   | 0.466 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 12.920  | 12.89%   | 17.91  | 0.205  | 3.33   | 0.216   | 0.385 \n",
      "\n",
      "\n",
      "[Sobol Run 55/768] H_Inertia: 0.0330 | BASE_S: 0.8292 | Tau: 0.0167 | Jitter: 0.92\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                  \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 55 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 56/768] H_Inertia: 0.7833 | BASE_S: 0.4464 | Tau: 0.0167 | Jitter: 0.92\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.255   | 20.70%   | 4.07   | 0.760  | 4.35   | 0.570   | 0.778 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.109   | 25.00%   | 5.90   | 0.649  | 4.23   | 0.468   | 0.396 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.031   | 28.52%   | 6.97   | 0.577  | 4.12   | 0.408   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.952   | 25.78%   | 6.69   | 0.568  | 4.15   | 0.431   | 0.418 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.863   | 30.47%   | 6.63   | 0.539  | 3.96   | 0.414   | 0.468 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.796   | 35.16%   | 6.65   | 0.580  | 4.01   | 0.396   | 0.460 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.777   | 30.08%   | 7.58   | 0.542  | 3.79   | 0.427   | 0.411 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.804   | 36.33%   | 8.14   | 0.496  | 3.88   | 0.410   | 0.423 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.727   | 41.41%   | 7.95   | 0.522  | 4.13   | 0.384   | 0.398 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.634   | 38.67%   | 8.43   | 0.440  | 4.02   | 0.392   | 0.412 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.699   | 48.05%   | 9.61   | 0.406  | 3.67   | 0.381   | 0.427 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.594   | 44.92%   | 9.66   | 0.407  | 4.12   | 0.374   | 0.438 \n",
      "\n",
      "\n",
      "[Sobol Run 57/768] H_Inertia: 0.7833 | BASE_S: 0.8292 | Tau: 0.0167 | Jitter: 0.92\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.476   | 23.83%   | 7.39   | 0.513  | 4.60   | 0.347   | 0.638 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.148   | 22.66%   | 7.34   | 0.558  | 4.86   | 0.333   | 0.669 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.085   | 15.62%   | 5.43   | 0.547  | 4.72   | 0.321   | 0.593 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.173   | 25.00%   | 8.07   | 0.522  | 4.68   | 0.322   | 0.637 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.131   | 22.27%   | 9.99   | 0.454  | 4.45   | 0.438   | 0.545 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.631   | 12.50%   | 10.62  | 0.453  | 2.39   | 0.548   | 0.659 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.363   | 23.44%   | 6.30   | 0.771  | 4.45   | 0.576   | 0.502 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.067   | 32.03%   | 7.01   | 0.636  | 4.04   | 0.464   | 0.645 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.969   | 30.47%   | 7.57   | 0.601  | 3.75   | 0.441   | 0.542 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.908   | 26.95%   | 7.34   | 0.596  | 3.64   | 0.447   | 0.687 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.866   | 31.25%   | 8.56   | 0.504  | 4.17   | 0.364   | 0.567 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.783   | 40.62%   | 9.95   | 0.455  | 4.26   | 0.360   | 0.476 \n",
      "\n",
      "\n",
      "[Sobol Run 58/768] H_Inertia: 0.7833 | BASE_S: 0.8292 | Tau: 0.0167 | Jitter: 0.95\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 15.202  | 11.72%   | 26.92  | 0.163  | 3.70   | 0.209   | 0.423 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 17.126  | 10.94%   | 26.92  | 0.142  | 3.57   | 0.210   | 0.382 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 16.581  | 8.59%    | 27.14  | 0.150  | 3.68   | 0.212   | 0.407 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 16.097  | 12.11%   | 26.87  | 0.178  | 3.63   | 0.304   | 0.447 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 13.075  | 9.38%    | 4.45   | 0.578  | 4.32   | 0.386   | 0.410 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.311   | 20.31%   | 6.52   | 0.725  | 5.14   | 0.438   | 0.846 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.134   | 21.88%   | 6.30   | 0.712  | 4.95   | 0.462   | 0.831 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.190   | 22.66%   | 8.36   | 0.556  | 4.90   | 0.375   | 0.685 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 2.127   | 26.56%   | 9.23   | 0.525  | 4.75   | 0.463   | 0.631 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.987   | 33.98%   | 9.52   | 0.465  | 5.10   | 0.392   | 0.613 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.808   | 35.16%   | 10.67  | 0.425  | 4.80   | 0.460   | 0.557 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.708   | 41.02%   | 11.50  | 0.402  | 4.84   | 0.388   | 0.492 \n",
      "\n",
      "\n",
      "[Sobol Run 59/768] H_Inertia: 0.7833 | BASE_S: 0.8292 | Tau: 0.0167 | Jitter: 0.92\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 12.217  | 9.77%    | 27.13  | 0.194  | 3.72   | 0.255   | 0.464 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 5.284   | 18.36%   | 2.49   | 0.558  | 4.54   | 0.391   | 0.650 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.284   | 18.75%   | 2.72   | 0.810  | 4.98   | 0.474   | 0.893 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.154   | 23.05%   | 4.33   | 0.706  | 4.48   | 0.455   | 0.796 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.112   | 24.61%   | 4.35   | 0.620  | 4.63   | 0.492   | 0.711 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.016   | 28.12%   | 5.27   | 0.665  | 4.92   | 0.430   | 0.761 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.985   | 29.69%   | 5.65   | 0.662  | 4.77   | 0.429   | 0.722 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.951   | 32.03%   | 7.05   | 0.574  | 4.64   | 0.426   | 0.675 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.901   | 32.03%   | 7.15   | 0.560  | 4.77   | 0.419   | 0.591 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.805   | 35.16%   | 7.76   | 0.542  | 4.86   | 0.409   | 0.572 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.764   | 36.72%   | 8.08   | 0.545  | 4.87   | 0.406   | 0.538 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 3.823   | 12.89%   | 23.13  | 0.297  | 3.64   | 0.288   | 0.557 \n",
      "\n",
      "\n",
      "[Sobol Run 60/768] H_Inertia: 0.5230 | BASE_S: 0.9364 | Tau: 0.0450 | Jitter: 0.40\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.824   | 9.77%    | 1.61   | 0.587  | 4.78   | 0.301   | 0.719 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.297   | 20.31%   | 2.09   | 0.599  | 4.87   | 0.304   | 0.601 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.275   | 19.92%   | 2.98   | 0.676  | 4.92   | 0.303   | 0.599 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.200   | 23.44%   | 3.88   | 0.627  | 4.95   | 0.292   | 0.526 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.141   | 23.44%   | 4.03   | 0.677  | 4.72   | 0.289   | 0.685 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.114   | 16.80%   | 5.03   | 0.579  | 4.92   | 0.293   | 0.690 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.235   | 24.61%   | 5.72   | 0.534  | 4.73   | 0.287   | 0.572 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.078   | 25.39%   | 6.03   | 0.563  | 4.94   | 0.290   | 0.617 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 2.021   | 26.17%   | 6.36   | 0.607  | 4.94   | 0.294   | 0.581 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.996   | 16.80%   | 6.06   | 0.455  | 4.91   | 0.305   | 0.463 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.988   | 28.52%   | 7.78   | 0.516  | 4.79   | 0.306   | 0.532 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.903   | 29.30%   | 7.32   | 0.493  | 4.97   | 0.329   | 0.467 \n",
      "\n",
      "\n",
      "[Sobol Run 61/768] H_Inertia: 0.5230 | BASE_S: 0.9364 | Tau: 0.0412 | Jitter: 0.40\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.039   | 9.77%    | 7.21   | 0.482  | 3.24   | 0.278   | 0.656 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.875   | 8.20%    | 1.39   | 0.556  | 4.42   | 0.408   | 0.636 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.321   | 7.03%    | 1.50   | 0.777  | 4.51   | 0.385   | 0.771 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.321   | 20.31%   | 2.67   | 0.625  | 4.29   | 0.395   | 0.737 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.262   | 19.92%   | 3.08   | 0.623  | 4.37   | 0.408   | 0.761 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.648   | 12.11%   | 11.42  | 0.369  | 2.09   | 0.342   | 0.404 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.625   | 19.53%   | 4.48   | 0.391  | 5.14   | 0.484   | 0.473 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.259   | 26.95%   | 7.15   | 0.631  | 4.60   | 0.403   | 0.765 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 2.106   | 27.73%   | 7.31   | 0.634  | 4.61   | 0.395   | 0.762 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.996   | 30.47%   | 9.36   | 0.541  | 4.58   | 0.365   | 0.690 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.923   | 28.52%   | 8.34   | 0.586  | 4.67   | 0.374   | 0.708 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.919   | 25.00%   | 9.12   | 0.553  | 4.60   | 0.370   | 0.646 \n",
      "\n",
      "\n",
      "[Sobol Run 62/768] H_Inertia: 0.2933 | BASE_S: 0.9364 | Tau: 0.0450 | Jitter: 0.40\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.919   | 19.53%   | 3.63   | 0.781  | 4.88   | 0.262   | 0.564 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.356   | 12.89%   | 2.94   | 0.709  | 4.91   | 0.290   | 0.591 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.414   | 10.55%   | 3.04   | 0.729  | 5.01   | 0.271   | 0.706 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.330   | 22.66%   | 5.30   | 0.560  | 4.33   | 0.282   | 0.466 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.567   | 12.11%   | 11.50  | 0.431  | 4.55   | 0.290   | 0.444 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.183   | 20.70%   | 6.74   | 0.698  | 4.71   | 0.503   | 0.640 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.091   | 24.61%   | 8.56   | 0.666  | 4.76   | 0.496   | 0.757 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.002   | 29.69%   | 11.18  | 0.463  | 4.79   | 0.477   | 0.447 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.881   | 32.03%   | 11.68  | 0.456  | 4.74   | 0.507   | 0.440 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.784   | 32.03%   | 12.47  | 0.418  | 4.84   | 0.380   | 0.337 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.751   | 40.23%   | 13.67  | 0.383  | 4.78   | 0.359   | 0.245 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.706   | 41.80%   | 13.67  | 0.367  | 4.90   | 0.357   | 0.257 \n",
      "\n",
      "\n",
      "[Sobol Run 63/768] H_Inertia: 0.5230 | BASE_S: 0.3392 | Tau: 0.0450 | Jitter: 0.40\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.275   | 20.70%   | 4.52   | 0.650  | 4.09   | 0.524   | 0.737 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.212   | 22.66%   | 6.01   | 0.578  | 4.09   | 0.444   | 0.695 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.004   | 31.64%   | 8.40   | 0.531  | 4.16   | 0.505   | 0.440 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.871   | 37.89%   | 10.04  | 0.470  | 4.10   | 0.413   | 0.369 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.801   | 39.84%   | 10.08  | 0.467  | 4.05   | 0.444   | 0.394 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.747   | 37.89%   | 11.15  | 0.468  | 4.18   | 0.506   | 0.325 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.850   | 38.28%   | 11.50  | 0.458  | 4.20   | 0.357   | 0.262 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.726   | 44.92%   | 10.66  | 0.458  | 3.99   | 0.366   | 0.280 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.632   | 46.88%   | 10.26  | 0.449  | 3.96   | 0.384   | 0.307 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.572   | 45.70%   | 10.93  | 0.442  | 4.00   | 0.367   | 0.315 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.536   | 48.83%   | 10.46  | 0.443  | 3.95   | 0.367   | 0.302 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.477   | 47.66%   | 10.45  | 0.401  | 4.10   | 0.359   | 0.300 \n",
      "\n",
      "\n",
      "[Sobol Run 64/768] H_Inertia: 0.5230 | BASE_S: 0.9364 | Tau: 0.0450 | Jitter: 0.40\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 4.279   | 12.50%   | 2.74   | 0.729  | 3.66   | 0.353   | 0.812 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.441   | 15.62%   | 2.29   | 0.817  | 4.26   | 0.345   | 0.901 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.354   | 17.58%   | 2.07   | 0.775  | 4.05   | 0.389   | 0.848 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.211   | 23.44%   | 3.05   | 0.744  | 3.77   | 0.400   | 0.825 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.211   | 25.39%   | 2.93   | 0.692  | 3.78   | 0.407   | 0.813 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.160   | 26.95%   | 3.12   | 0.645  | 3.46   | 0.423   | 0.745 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.108   | 24.22%   | 3.21   | 0.639  | 3.43   | 0.406   | 0.752 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.060   | 26.17%   | 3.86   | 0.597  | 3.52   | 0.393   | 0.724 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.990   | 32.81%   | 4.36   | 0.535  | 3.49   | 0.365   | 0.636 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.938   | 21.88%   | 5.30   | 0.452  | 3.32   | 0.366   | 0.566 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 2.040   | 28.52%   | 4.62   | 0.483  | 3.44   | 0.363   | 0.620 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.980   | 27.34%   | 4.82   | 0.464  | 3.39   | 0.353   | 0.540 \n",
      "\n",
      "\n",
      "[Sobol Run 65/768] H_Inertia: 0.5230 | BASE_S: 0.9364 | Tau: 0.0450 | Jitter: 0.37\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.523   | 21.48%   | 5.06   | 0.568  | 4.97   | 0.319   | 0.338 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.120   | 24.22%   | 6.76   | 0.655  | 4.20   | 0.319   | 0.450 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.000   | 25.78%   | 6.25   | 0.653  | 3.92   | 0.312   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.013   | 29.69%   | 7.20   | 0.581  | 3.98   | 0.315   | 0.430 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.005   | 29.30%   | 7.81   | 0.586  | 3.79   | 0.264   | 0.448 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.532   | 24.61%   | 8.75   | 0.569  | 4.72   | 0.330   | 0.695 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.223   | 16.41%   | 7.21   | 0.424  | 2.64   | 0.350   | 0.586 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.281   | 22.66%   | 10.50  | 0.500  | 3.80   | 0.462   | 0.671 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 2.110   | 21.88%   | 14.87  | 0.378  | 3.63   | 0.382   | 0.426 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 2.055   | 29.30%   | 12.79  | 0.462  | 4.11   | 0.371   | 0.479 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.930   | 35.94%   | 13.44  | 0.390  | 3.78   | 0.346   | 0.500 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.944   | 24.61%   | 8.78   | 0.488  | 2.81   | 0.375   | 0.666 \n",
      "\n",
      "\n",
      "[Sobol Run 66/768] H_Inertia: 0.2933 | BASE_S: 0.3392 | Tau: 0.0450 | Jitter: 0.37\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.209   | 29.30%   | 10.80  | 0.565  | 4.71   | 0.490   | 0.470 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.973   | 39.45%   | 11.68  | 0.474  | 4.33   | 0.579   | 0.489 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.827   | 42.97%   | 12.63  | 0.446  | 4.54   | 0.511   | 0.397 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.743   | 44.92%   | 11.42  | 0.426  | 3.43   | 0.537   | 0.472 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.634   | 44.53%   | 13.40  | 0.406  | 3.92   | 0.521   | 0.407 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.621   | 44.53%   | 14.33  | 0.393  | 4.20   | 0.495   | 0.377 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.593   | 44.53%   | 12.38  | 0.397  | 3.32   | 0.502   | 0.477 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.528   | 49.22%   | 12.65  | 0.374  | 3.84   | 0.489   | 0.427 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.521   | 49.61%   | 12.99  | 0.385  | 3.49   | 0.485   | 0.393 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.497   | 46.09%   | 13.65  | 0.358  | 4.23   | 0.483   | 0.421 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.477   | 46.48%   | 13.09  | 0.368  | 3.97   | 0.466   | 0.392 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.436   | 44.53%   | 14.59  | 0.341  | 4.09   | 0.463   | 0.404 \n",
      "\n",
      "\n",
      "[Sobol Run 67/768] H_Inertia: 0.5230 | BASE_S: 0.3392 | Tau: 0.0412 | Jitter: 0.37\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.161   | 30.08%   | 10.01  | 0.515  | 4.58   | 0.471   | 0.534 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.037   | 31.25%   | 10.88  | 0.458  | 4.45   | 0.495   | 0.385 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.921   | 35.94%   | 10.24  | 0.491  | 4.28   | 0.488   | 0.490 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.853   | 37.11%   | 11.02  | 0.469  | 4.22   | 0.488   | 0.396 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.833   | 38.67%   | 11.14  | 0.478  | 4.35   | 0.455   | 0.438 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.777   | 40.23%   | 11.78  | 0.447  | 4.16   | 0.425   | 0.394 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.731   | 41.02%   | 11.76  | 0.428  | 4.08   | 0.393   | 0.359 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.676   | 39.84%   | 11.52  | 0.416  | 3.56   | 0.401   | 0.374 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.617   | 43.75%   | 11.25  | 0.410  | 3.72   | 0.379   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.599   | 41.41%   | 12.00  | 0.394  | 3.65   | 0.384   | 0.381 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.557   | 42.58%   | 10.92  | 0.411  | 3.40   | 0.372   | 0.462 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.540   | 45.31%   | 11.38  | 0.404  | 3.53   | 0.383   | 0.465 \n",
      "\n",
      "\n",
      "[Sobol Run 68/768] H_Inertia: 0.2933 | BASE_S: 0.9364 | Tau: 0.0412 | Jitter: 0.37\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.410   | 21.48%   | 14.28  | 0.368  | 4.28   | 0.461   | 0.342 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.130   | 26.95%   | 14.18  | 0.359  | 4.38   | 0.485   | 0.473 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.038   | 14.45%   | 16.93  | 0.305  | 4.45   | 0.331   | 0.304 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.327   | 14.45%   | 7.27   | 0.603  | 4.87   | 0.501   | 0.703 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.174   | 23.05%   | 6.17   | 0.615  | 4.93   | 0.496   | 0.765 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.075   | 21.09%   | 9.79   | 0.530  | 3.55   | 0.494   | 0.363 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.075   | 29.30%   | 9.24   | 0.537  | 4.95   | 0.428   | 0.657 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.002   | 34.77%   | 11.68  | 0.395  | 3.41   | 0.381   | 0.327 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.879   | 35.16%   | 11.01  | 0.357  | 2.75   | 0.347   | 0.364 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.825   | 39.84%   | 10.87  | 0.379  | 2.89   | 0.366   | 0.419 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.814   | 43.36%   | 12.05  | 0.348  | 2.71   | 0.360   | 0.377 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.731   | 34.38%   | 12.86  | 0.339  | 2.83   | 0.342   | 0.412 \n",
      "\n",
      "\n",
      "[Sobol Run 69/768] H_Inertia: 0.2933 | BASE_S: 0.3392 | Tau: 0.0412 | Jitter: 0.37\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.328   | 20.31%   | 12.41  | 0.440  | 4.57   | 0.375   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.098   | 31.25%   | 10.19  | 0.566  | 3.75   | 0.580   | 0.489 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.973   | 34.38%   | 11.85  | 0.453  | 3.66   | 0.470   | 0.408 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.886   | 37.11%   | 11.76  | 0.477  | 3.66   | 0.485   | 0.480 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.804   | 41.02%   | 13.40  | 0.423  | 3.74   | 0.430   | 0.424 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.693   | 42.19%   | 13.29  | 0.426  | 3.80   | 0.393   | 0.397 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.630   | 44.14%   | 13.10  | 0.387  | 3.94   | 0.352   | 0.466 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.555   | 43.36%   | 14.57  | 0.355  | 4.02   | 0.344   | 0.355 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.512   | 47.27%   | 12.92  | 0.376  | 3.98   | 0.354   | 0.457 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.511   | 46.09%   | 13.87  | 0.394  | 4.15   | 0.342   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.482   | 49.22%   | 14.15  | 0.364  | 4.02   | 0.342   | 0.360 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.423   | 51.17%   | 14.75  | 0.335  | 3.99   | 0.330   | 0.379 \n",
      "\n",
      "\n",
      "[Sobol Run 70/768] H_Inertia: 0.2933 | BASE_S: 0.3392 | Tau: 0.0412 | Jitter: 0.40\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.212   | 29.69%   | 12.30  | 0.473  | 4.92   | 0.619   | 0.184 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.936   | 35.55%   | 12.37  | 0.389  | 4.61   | 0.560   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.755   | 41.41%   | 12.91  | 0.391  | 4.00   | 0.523   | 0.422 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.659   | 43.36%   | 13.84  | 0.366  | 4.19   | 0.522   | 0.374 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.609   | 42.19%   | 12.38  | 0.399  | 4.03   | 0.487   | 0.424 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.562   | 48.05%   | 13.31  | 0.372  | 4.24   | 0.499   | 0.370 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.523   | 48.44%   | 12.96  | 0.374  | 4.08   | 0.446   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.491   | 50.39%   | 14.15  | 0.350  | 3.98   | 0.451   | 0.360 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.468   | 49.22%   | 12.73  | 0.385  | 4.12   | 0.431   | 0.394 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.454   | 48.83%   | 14.05  | 0.346  | 4.00   | 0.445   | 0.316 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.407   | 50.00%   | 12.94  | 0.367  | 4.11   | 0.455   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.430   | 48.44%   | 14.73  | 0.340  | 4.04   | 0.448   | 0.308 \n",
      "\n",
      "\n",
      "[Sobol Run 71/768] H_Inertia: 0.2933 | BASE_S: 0.3392 | Tau: 0.0412 | Jitter: 0.37\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.219   | 29.69%   | 11.05  | 0.549  | 4.06   | 0.670   | 0.597 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.017   | 34.38%   | 11.48  | 0.467  | 4.23   | 0.586   | 0.490 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.864   | 39.06%   | 11.70  | 0.438  | 3.92   | 0.582   | 0.514 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.744   | 39.45%   | 11.42  | 0.446  | 3.54   | 0.549   | 0.528 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.732   | 42.19%   | 13.70  | 0.394  | 4.07   | 0.566   | 0.491 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.629   | 42.58%   | 13.81  | 0.381  | 3.94   | 0.562   | 0.430 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.582   | 48.05%   | 13.77  | 0.361  | 3.98   | 0.496   | 0.440 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.538   | 50.39%   | 13.12  | 0.357  | 3.60   | 0.536   | 0.443 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.488   | 47.66%   | 12.59  | 0.353  | 3.71   | 0.558   | 0.453 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.468   | 48.44%   | 13.67  | 0.336  | 3.68   | 0.544   | 0.420 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.441   | 49.22%   | 12.58  | 0.343  | 3.43   | 0.578   | 0.474 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.403   | 51.17%   | 13.25  | 0.338  | 3.63   | 0.553   | 0.449 \n",
      "\n",
      "\n",
      "[Sobol Run 72/768] H_Inertia: 0.2780 | BASE_S: 0.2014 | Tau: 0.0328 | Jitter: 0.13\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.201   | 19.14%   | 9.13   | 0.559  | 4.34   | 0.506   | 0.522 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.079   | 26.95%   | 10.42  | 0.523  | 4.35   | 0.476   | 0.415 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.907   | 35.55%   | 10.75  | 0.506  | 4.42   | 0.545   | 0.406 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.818   | 39.84%   | 11.40  | 0.498  | 4.56   | 0.483   | 0.282 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.745   | 38.67%   | 11.80  | 0.449  | 4.42   | 0.483   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.650   | 42.19%   | 12.85  | 0.426  | 4.37   | 0.426   | 0.310 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.604   | 45.31%   | 12.53  | 0.415  | 4.46   | 0.458   | 0.364 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.568   | 47.27%   | 13.29  | 0.393  | 4.34   | 0.413   | 0.341 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.521   | 46.48%   | 12.48  | 0.364  | 4.46   | 0.430   | 0.395 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.497   | 48.83%   | 13.17  | 0.393  | 4.38   | 0.409   | 0.294 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.486   | 51.95%   | 12.72  | 0.397  | 4.48   | 0.394   | 0.322 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.421   | 50.78%   | 13.84  | 0.342  | 4.42   | 0.338   | 0.300 \n",
      "\n",
      "\n",
      "[Sobol Run 73/768] H_Inertia: 0.2780 | BASE_S: 0.2014 | Tau: 0.0289 | Jitter: 0.13\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.210   | 23.83%   | 8.51   | 0.617  | 4.32   | 0.513   | 0.610 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.955   | 37.89%   | 11.75  | 0.452  | 4.35   | 0.424   | 0.258 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.728   | 39.06%   | 13.30  | 0.417  | 4.47   | 0.441   | 0.294 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.629   | 44.14%   | 12.93  | 0.381  | 4.57   | 0.437   | 0.298 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.568   | 48.05%   | 12.80  | 0.400  | 4.49   | 0.380   | 0.328 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.493   | 49.61%   | 13.38  | 0.392  | 4.49   | 0.387   | 0.315 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.485   | 47.27%   | 13.34  | 0.368  | 4.27   | 0.365   | 0.380 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.461   | 50.78%   | 11.75  | 0.390  | 4.10   | 0.367   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.425   | 51.95%   | 12.63  | 0.380  | 4.07   | 0.359   | 0.341 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.411   | 50.39%   | 12.74  | 0.354  | 3.95   | 0.342   | 0.386 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.389   | 52.34%   | 11.58  | 0.362  | 4.14   | 0.525   | 0.372 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.361   | 50.39%   | 11.36  | 0.368  | 4.23   | 0.499   | 0.384 \n",
      "\n",
      "\n",
      "[Sobol Run 74/768] H_Inertia: 0.5383 | BASE_S: 0.2014 | Tau: 0.0328 | Jitter: 0.13\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.138   | 28.12%   | 7.72   | 0.523  | 4.11   | 0.467   | 0.481 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.979   | 34.38%   | 9.48   | 0.508  | 4.17   | 0.496   | 0.480 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.918   | 32.03%   | 8.85   | 0.511  | 4.30   | 0.494   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.829   | 34.77%   | 9.44   | 0.475  | 4.29   | 0.423   | 0.334 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.790   | 40.23%   | 10.06  | 0.464  | 4.16   | 0.494   | 0.424 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.728   | 38.67%   | 10.91  | 0.404  | 4.47   | 0.356   | 0.356 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.702   | 42.19%   | 11.20  | 0.383  | 4.17   | 0.346   | 0.302 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.678   | 43.36%   | 12.07  | 0.400  | 4.42   | 0.340   | 0.323 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.607   | 45.31%   | 11.65  | 0.377  | 4.31   | 0.348   | 0.297 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.611   | 42.97%   | 11.52  | 0.386  | 4.30   | 0.345   | 0.309 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.543   | 47.27%   | 11.64  | 0.378  | 4.42   | 0.329   | 0.316 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.549   | 48.05%   | 11.94  | 0.387  | 4.75   | 0.322   | 0.316 \n",
      "\n",
      "\n",
      "[Sobol Run 75/768] H_Inertia: 0.2780 | BASE_S: 0.0942 | Tau: 0.0328 | Jitter: 0.13\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.212   | 26.95%   | 11.04  | 0.543  | 4.28   | 0.486   | 0.524 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.929   | 32.81%   | 13.65  | 0.404  | 4.65   | 0.384   | 0.349 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.750   | 36.33%   | 13.72  | 0.398  | 4.65   | 0.364   | 0.356 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.655   | 44.92%   | 13.82  | 0.381  | 4.37   | 0.351   | 0.399 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.597   | 38.28%   | 13.39  | 0.356  | 4.32   | 0.348   | 0.406 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.559   | 46.88%   | 13.74  | 0.359  | 4.23   | 0.344   | 0.401 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.515   | 50.39%   | 13.64  | 0.362  | 4.20   | 0.355   | 0.438 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.474   | 51.56%   | 13.17  | 0.342  | 4.15   | 0.344   | 0.424 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.436   | 52.34%   | 12.77  | 0.334  | 4.06   | 0.331   | 0.400 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.414   | 49.22%   | 14.08  | 0.337  | 4.27   | 0.340   | 0.402 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.378   | 53.12%   | 13.54  | 0.327  | 4.32   | 0.327   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.356   | 52.73%   | 12.48  | 0.331  | 4.15   | 0.320   | 0.375 \n",
      "\n",
      "\n",
      "[Sobol Run 76/768] H_Inertia: 0.2780 | BASE_S: 0.2014 | Tau: 0.0328 | Jitter: 0.13\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.363   | 19.53%   | 11.26  | 0.460  | 4.18   | 0.372   | 0.551 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.101   | 25.78%   | 12.85  | 0.408  | 4.18   | 0.386   | 0.276 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.943   | 29.30%   | 13.44  | 0.385  | 3.48   | 0.379   | 0.393 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.835   | 33.98%   | 12.39  | 0.402  | 3.30   | 0.371   | 0.479 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.786   | 35.16%   | 11.13  | 0.437  | 3.38   | 0.383   | 0.527 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.751   | 37.50%   | 13.57  | 0.359  | 3.47   | 0.374   | 0.426 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.699   | 42.58%   | 13.82  | 0.362  | 3.58   | 0.384   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.753   | 37.11%   | 14.68  | 0.335  | 3.64   | 0.390   | 0.274 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.677   | 38.67%   | 12.74  | 0.388  | 3.50   | 0.368   | 0.490 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.688   | 35.55%   | 9.57   | 0.422  | 3.53   | 0.375   | 0.568 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.611   | 45.31%   | 12.79  | 0.352  | 3.60   | 0.356   | 0.457 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.560   | 46.48%   | 14.34  | 0.342  | 3.62   | 0.367   | 0.349 \n",
      "\n",
      "\n",
      "[Sobol Run 77/768] H_Inertia: 0.2780 | BASE_S: 0.2014 | Tau: 0.0328 | Jitter: 0.64\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.199   | 25.78%   | 9.41   | 0.551  | 4.52   | 0.513   | 0.360 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.953   | 32.81%   | 10.42  | 0.534  | 4.64   | 0.557   | 0.302 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.871   | 39.45%   | 11.03  | 0.502  | 4.49   | 0.531   | 0.332 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.796   | 37.50%   | 11.07  | 0.513  | 4.48   | 0.465   | 0.328 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.724   | 33.59%   | 11.75  | 0.466  | 4.69   | 0.447   | 0.325 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.681   | 39.45%   | 12.88  | 0.468  | 4.43   | 0.460   | 0.308 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.610   | 42.97%   | 12.30  | 0.447  | 4.62   | 0.377   | 0.278 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.583   | 43.75%   | 12.62  | 0.429  | 4.55   | 0.380   | 0.257 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.517   | 41.41%   | 12.35  | 0.376  | 4.58   | 0.402   | 0.352 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.536   | 45.70%   | 13.38  | 0.359  | 4.51   | 0.320   | 0.350 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.504   | 43.36%   | 13.11  | 0.378  | 4.44   | 0.320   | 0.253 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.492   | 50.39%   | 13.86  | 0.415  | 4.56   | 0.359   | 0.251 \n",
      "\n",
      "\n",
      "[Sobol Run 78/768] H_Inertia: 0.5383 | BASE_S: 0.0942 | Tau: 0.0328 | Jitter: 0.64\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.238   | 28.12%   | 7.59   | 0.689  | 3.94   | 0.656   | 0.751 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.018   | 32.42%   | 11.10  | 0.464  | 4.03   | 0.503   | 0.531 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.881   | 38.67%   | 11.98  | 0.418  | 3.99   | 0.460   | 0.439 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.777   | 41.02%   | 12.78  | 0.413  | 4.14   | 0.420   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.713   | 44.92%   | 12.97  | 0.400  | 4.13   | 0.388   | 0.367 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.638   | 47.66%   | 13.00  | 0.396  | 4.00   | 0.378   | 0.360 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.604   | 42.58%   | 12.85  | 0.404  | 4.04   | 0.382   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.561   | 48.05%   | 13.39  | 0.375  | 4.12   | 0.358   | 0.341 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.511   | 51.56%   | 13.50  | 0.377  | 4.03   | 0.376   | 0.350 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.490   | 46.09%   | 14.56  | 0.356  | 4.21   | 0.369   | 0.374 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.448   | 52.34%   | 14.46  | 0.344  | 4.30   | 0.369   | 0.351 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.396   | 53.12%   | 14.00  | 0.361  | 4.48   | 0.377   | 0.337 \n",
      "\n",
      "\n",
      "[Sobol Run 79/768] H_Inertia: 0.2780 | BASE_S: 0.0942 | Tau: 0.0289 | Jitter: 0.64\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.246   | 26.17%   | 11.13  | 0.563  | 4.06   | 0.638   | 0.671 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.959   | 33.59%   | 11.89  | 0.428  | 4.13   | 0.499   | 0.554 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.798   | 39.06%   | 13.17  | 0.406  | 4.03   | 0.495   | 0.488 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.715   | 43.75%   | 13.53  | 0.377  | 4.24   | 0.416   | 0.482 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.661   | 44.53%   | 13.48  | 0.373  | 4.05   | 0.475   | 0.483 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.632   | 45.70%   | 12.32  | 0.382  | 3.96   | 0.463   | 0.504 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.592   | 48.83%   | 13.35  | 0.365  | 4.11   | 0.423   | 0.478 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.557   | 46.88%   | 13.66  | 0.351  | 4.07   | 0.406   | 0.452 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.521   | 47.27%   | 13.23  | 0.355  | 3.97   | 0.413   | 0.451 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.501   | 47.66%   | 13.18  | 0.343  | 4.12   | 0.444   | 0.430 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.434   | 50.00%   | 13.33  | 0.364  | 4.13   | 0.412   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.435   | 47.27%   | 13.10  | 0.356  | 4.08   | 0.414   | 0.409 \n",
      "\n",
      "\n",
      "[Sobol Run 80/768] H_Inertia: 0.5383 | BASE_S: 0.2014 | Tau: 0.0289 | Jitter: 0.64\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.311   | 22.66%   | 9.24   | 0.573  | 3.63   | 0.767   | 0.696 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.064   | 33.98%   | 9.01   | 0.592  | 3.81   | 0.640   | 0.708 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.926   | 35.55%   | 10.60  | 0.472  | 3.94   | 0.583   | 0.503 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.808   | 42.58%   | 11.62  | 0.420  | 3.87   | 0.527   | 0.455 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.737   | 44.14%   | 11.91  | 0.384  | 3.84   | 0.535   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.657   | 48.44%   | 12.84  | 0.388  | 3.85   | 0.509   | 0.382 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.611   | 47.66%   | 12.29  | 0.397  | 3.93   | 0.492   | 0.416 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.577   | 46.09%   | 11.78  | 0.408  | 3.90   | 0.455   | 0.483 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.554   | 49.61%   | 12.76  | 0.387  | 3.98   | 0.443   | 0.457 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.533   | 51.17%   | 11.95  | 0.415  | 4.00   | 0.423   | 0.480 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.508   | 49.61%   | 12.61  | 0.389  | 4.01   | 0.412   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.452   | 45.70%   | 13.46  | 0.381  | 4.14   | 0.415   | 0.424 \n",
      "\n",
      "\n",
      "[Sobol Run 81/768] H_Inertia: 0.5383 | BASE_S: 0.0942 | Tau: 0.0289 | Jitter: 0.64\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.184   | 28.91%   | 8.76   | 0.579  | 4.15   | 0.537   | 0.381 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.988   | 36.72%   | 10.60  | 0.488  | 4.06   | 0.518   | 0.264 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.855   | 37.89%   | 11.15  | 0.436  | 4.18   | 0.459   | 0.298 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.750   | 40.62%   | 11.40  | 0.387  | 4.21   | 0.406   | 0.352 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.705   | 39.06%   | 11.79  | 0.394  | 4.32   | 0.405   | 0.314 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.659   | 44.92%   | 11.72  | 0.386  | 4.25   | 0.395   | 0.350 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.642   | 44.53%   | 11.80  | 0.375  | 4.26   | 0.373   | 0.367 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.588   | 45.31%   | 10.92  | 0.390  | 4.17   | 0.374   | 0.432 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.583   | 45.31%   | 11.24  | 0.365  | 4.27   | 0.372   | 0.413 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.555   | 45.31%   | 11.93  | 0.374  | 4.31   | 0.371   | 0.338 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.533   | 45.31%   | 11.39  | 0.361  | 4.24   | 0.339   | 0.385 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.489   | 44.53%   | 12.16  | 0.366  | 4.12   | 0.356   | 0.366 \n",
      "\n",
      "\n",
      "[Sobol Run 82/768] H_Inertia: 0.5383 | BASE_S: 0.0942 | Tau: 0.0289 | Jitter: 0.13\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.261   | 31.64%   | 9.01   | 0.624  | 4.18   | 0.621   | 0.725 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.992   | 31.64%   | 10.75  | 0.497  | 4.18   | 0.497   | 0.438 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.870   | 41.41%   | 11.21  | 0.457  | 4.27   | 0.474   | 0.486 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.780   | 40.23%   | 11.37  | 0.423  | 4.46   | 0.421   | 0.379 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.699   | 42.58%   | 10.65  | 0.432  | 4.52   | 0.431   | 0.420 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.639   | 46.48%   | 11.48  | 0.415  | 4.58   | 0.417   | 0.354 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.602   | 44.92%   | 11.76  | 0.433  | 4.52   | 0.391   | 0.331 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.563   | 48.83%   | 11.23  | 0.408  | 4.56   | 0.444   | 0.403 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.541   | 49.61%   | 11.83  | 0.399  | 4.49   | 0.421   | 0.359 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.503   | 47.27%   | 11.95  | 0.399  | 4.24   | 0.391   | 0.302 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.473   | 46.88%   | 11.97  | 0.390  | 4.36   | 0.412   | 0.347 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.445   | 47.27%   | 12.31  | 0.366  | 4.44   | 0.389   | 0.350 \n",
      "\n",
      "\n",
      "[Sobol Run 83/768] H_Inertia: 0.5383 | BASE_S: 0.0942 | Tau: 0.0289 | Jitter: 0.64\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.262   | 31.25%   | 9.56   | 0.586  | 4.17   | 0.578   | 0.526 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.019   | 33.20%   | 10.63  | 0.503  | 4.37   | 0.457   | 0.558 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.879   | 38.28%   | 11.99  | 0.429  | 4.35   | 0.412   | 0.484 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.767   | 36.72%   | 11.31  | 0.442  | 4.21   | 0.423   | 0.450 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.686   | 46.09%   | 11.41  | 0.408  | 4.32   | 0.423   | 0.504 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.648   | 45.70%   | 11.94  | 0.413  | 4.36   | 0.413   | 0.422 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.580   | 47.27%   | 11.09  | 0.402  | 4.27   | 0.447   | 0.398 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.529   | 47.27%   | 11.54  | 0.391  | 4.39   | 0.402   | 0.402 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.510   | 46.88%   | 11.71  | 0.403  | 4.30   | 0.401   | 0.318 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.480   | 47.66%   | 11.86  | 0.375  | 4.33   | 0.418   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.450   | 47.27%   | 11.83  | 0.397  | 4.29   | 0.393   | 0.336 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.416   | 49.61%   | 12.48  | 0.379  | 4.35   | 0.403   | 0.356 \n",
      "\n",
      "\n",
      "[Sobol Run 84/768] H_Inertia: 0.7680 | BASE_S: 0.6914 | Tau: 0.0083 | Jitter: 0.68\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 4.310   | 12.11%   | 12.26  | 0.305  | 3.14   | 0.321   | 0.505 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 5.620   | 12.11%   | 7.79   | 0.542  | 4.93   | 0.362   | 0.363 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.170   | 21.48%   | 6.00   | 0.703  | 4.73   | 0.364   | 0.722 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.113   | 24.22%   | 6.24   | 0.452  | 2.26   | 0.406   | 0.445 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 10.712  | 16.80%   | 3.59   | 0.378  | 2.92   | 0.326   | 0.530 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 9.919   | 9.38%    | 1.18   | nan    | 3.07   | 0.298   | 0.258 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 5.560   | 15.23%   | 1.58   | nan    | 3.39   | 0.319   | 0.274 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 3.737   | 8.20%    | 1.20   | nan    | 3.46   | 0.359   | 0.409 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 2.414   | 18.36%   | 1.15   | nan    | 3.30   | 0.388   | 0.549 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 2.244   | 25.39%   | 1.40   | nan    | 3.48   | 0.347   | 0.622 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 2.181   | 26.56%   | 1.52   | nan    | 3.42   | 0.367   | 0.626 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 2.067   | 27.73%   | 1.74   | nan    | 3.57   | 0.328   | 0.677 \n",
      "\n",
      "\n",
      "[Sobol Run 85/768] H_Inertia: 0.7680 | BASE_S: 0.6914 | Tau: 0.0044 | Jitter: 0.68\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.334   | 17.97%   | 4.28   | 0.768  | 4.99   | 0.466   | 0.313 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.113   | 24.22%   | 5.65   | 0.663  | 4.78   | 0.400   | 0.584 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.041   | 21.88%   | 4.37   | 0.742  | 4.54   | 0.395   | 0.381 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.005   | 25.78%   | 5.59   | 0.722  | 4.79   | 0.522   | 0.337 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.938   | 28.12%   | 5.37   | 0.676  | 4.80   | 0.484   | 0.387 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.939   | 29.69%   | 6.10   | 0.655  | 4.90   | 0.493   | 0.570 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.894   | 35.16%   | 6.34   | 0.683  | 4.83   | 0.488   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.972   | 27.73%   | 7.17   | 0.599  | 4.41   | 0.485   | 0.601 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.893   | 34.38%   | 6.18   | 0.637  | 4.93   | 0.468   | 0.558 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.812   | 31.64%   | 7.53   | 0.610  | 4.86   | 0.452   | 0.561 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.857   | 30.86%   | 7.42   | 0.606  | 4.79   | 0.458   | 0.378 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.777   | 37.50%   | 7.16   | 0.583  | 4.80   | 0.454   | 0.458 \n",
      "\n",
      "\n",
      "[Sobol Run 86/768] H_Inertia: 0.0483 | BASE_S: 0.6914 | Tau: 0.0083 | Jitter: 0.68\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.738   | 18.75%   | 11.13  | 0.379  | 2.36   | 0.319   | 0.559 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.973   | 14.84%   | 9.39   | 0.349  | 3.64   | 0.298   | 0.529 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 86 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 87/768] H_Inertia: 0.7680 | BASE_S: 0.5842 | Tau: 0.0083 | Jitter: 0.68\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.301   | 22.27%   | 6.53   | 0.805  | 4.60   | 0.399   | 0.436 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.107   | 26.95%   | 6.49   | 0.683  | 4.87   | 0.419   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.057   | 29.30%   | 7.48   | 0.683  | 4.81   | 0.523   | 0.466 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.973   | 33.59%   | 9.29   | 0.636  | 4.80   | 0.531   | 0.525 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 3.376   | 11.33%   | 7.14   | 0.405  | 3.35   | 0.445   | 0.583 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 11.301  | 16.02%   | 4.16   | 0.501  | 3.81   | 0.339   | 0.584 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 7.575   | 13.67%   | 6.96   | nan    | 3.47   | 0.461   | 0.580 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 11.962  | 14.45%   | 4.16   | 0.596  | 3.15   | 0.397   | 0.768 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 8.176   | 25.78%   | 4.74   | 0.754  | 4.89   | 0.424   | 0.231 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 2.124   | 24.22%   | 7.98   | 0.694  | 4.75   | 0.401   | 0.533 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 2.045   | 28.91%   | 9.14   | 0.629  | 4.73   | 0.416   | 0.370 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.917   | 33.59%   | 10.68  | 0.491  | 4.77   | 0.408   | 0.421 \n",
      "\n",
      "\n",
      "[Sobol Run 88/768] H_Inertia: 0.7680 | BASE_S: 0.6914 | Tau: 0.0083 | Jitter: 0.68\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.845   | 17.58%   | 3.52   | 0.757  | 4.15   | 0.348   | 0.790 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.341   | 22.66%   | 3.20   | 0.747  | 4.13   | 0.341   | 0.688 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.274   | 17.19%   | 3.29   | 0.652  | 3.94   | 0.337   | 0.756 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.234   | 18.75%   | 3.93   | 0.698  | 4.32   | 0.353   | 0.387 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.575   | 9.77%    | 2.24   | 0.682  | 4.00   | 0.341   | 0.781 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.371   | 11.33%   | 3.36   | 0.645  | 3.13   | 0.375   | 0.712 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.216   | 21.09%   | 3.68   | 0.769  | 3.82   | 0.359   | 0.700 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.165   | 15.62%   | 3.25   | 0.832  | 4.83   | 0.369   | 0.836 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 2.183   | 23.05%   | 4.26   | 0.725  | 3.07   | 0.364   | 0.587 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 2.227   | 20.70%   | 3.46   | 0.737  | 3.39   | 0.373   | 0.722 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 2.259   | 19.14%   | 4.28   | 0.775  | 4.08   | 0.377   | 0.356 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 2.158   | 16.02%   | 3.73   | 0.706  | 3.81   | 0.379   | 0.644 \n",
      "\n",
      "\n",
      "[Sobol Run 89/768] H_Inertia: 0.7680 | BASE_S: 0.6914 | Tau: 0.0083 | Jitter: 1.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 14.324  | 8.20%    | 26.64  | 0.187  | 3.66   | 0.324   | 0.451 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 15.548  | 9.77%    | 26.81  | 0.197  | 3.61   | 0.212   | 0.467 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 11.128  | 10.16%   | 22.18  | 0.291  | 3.46   | 0.267   | 0.550 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 5.540   | 11.33%   | 7.74   | 0.350  | 3.82   | 0.434   | 0.443 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 6.436   | 6.64%    | 18.72  | 0.296  | 3.50   | 0.298   | 0.525 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 12.066  | 14.06%   | 26.81  | 0.155  | 3.53   | 0.247   | 0.415 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 17.231  | 9.77%    | 27.37  | 0.143  | 3.56   | 0.211   | 0.403 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 19.554  | 8.98%    | 26.93  | 0.155  | 3.44   | 0.237   | 0.418 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 21.347  | 10.55%   | 26.99  | 0.136  | 3.47   | 0.252   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 17.820  | 11.33%   | 27.25  | 0.204  | 3.61   | 0.252   | 0.478 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 15.064  | 11.72%   | 26.99  | 0.207  | 3.54   | 0.220   | 0.480 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 18.001  | 11.33%   | 27.38  | 0.122  | 3.31   | 0.223   | 0.373 \n",
      "\n",
      "\n",
      "[Sobol Run 90/768] H_Inertia: 0.0483 | BASE_S: 0.5842 | Tau: 0.0083 | Jitter: 1.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 90 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 91/768] H_Inertia: 0.7680 | BASE_S: 0.5842 | Tau: 0.0044 | Jitter: 1.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 8.778   | 13.67%   | 23.22  | 0.227  | 3.77   | 0.307   | 0.480 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 10.716  | 10.55%   | 17.59  | 0.263  | 3.74   | 0.327   | 0.468 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 6.602   | 12.11%   | 18.86  | 0.310  | 3.48   | 0.330   | 0.540 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 11.206  | 8.98%    | 25.27  | 0.200  | 3.80   | 0.263   | 0.444 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 15.408  | 10.94%   | 24.69  | 0.206  | 3.65   | 0.406   | 0.445 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 16.047  | 11.33%   | 26.14  | 0.176  | 3.68   | 0.316   | 0.419 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 17.187  | 9.77%    | 25.58  | 0.210  | 3.79   | 0.271   | 0.462 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 17.145  | 11.33%   | 25.02  | 0.211  | 3.75   | 0.239   | 0.464 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 13.220  | 10.55%   | 17.04  | 0.255  | 3.94   | 0.252   | 0.477 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 12.838  | 10.94%   | 23.98  | 0.218  | 3.85   | 0.307   | 0.454 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 13.828  | 10.55%   | 24.83  | 0.216  | 3.78   | 0.361   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 13.396  | 8.59%    | 15.03  | 0.248  | 3.73   | 0.301   | 0.446 \n",
      "\n",
      "\n",
      "[Sobol Run 92/768] H_Inertia: 0.0483 | BASE_S: 0.6914 | Tau: 0.0044 | Jitter: 1.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 92 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 93/768] H_Inertia: 0.0483 | BASE_S: 0.5842 | Tau: 0.0044 | Jitter: 1.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 93 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 94/768] H_Inertia: 0.0483 | BASE_S: 0.5842 | Tau: 0.0044 | Jitter: 0.68\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.055   | 10.16%   | 8.01   | 0.352  | 3.26   | 0.516   | 0.409 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 4.016   | 6.64%    | 10.33  | 0.383  | 3.47   | 0.314   | 0.545 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 94 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 95/768] H_Inertia: 0.0483 | BASE_S: 0.5842 | Tau: 0.0044 | Jitter: 1.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 95 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 96/768] H_Inertia: 0.2167 | BASE_S: 0.1402 | Tau: 0.0113 | Jitter: 0.33\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.263   | 28.52%   | 13.23  | 0.440  | 5.01   | 0.496   | 0.359 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.975   | 31.25%   | 11.65  | 0.440  | 4.30   | 0.408   | 0.480 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.807   | 41.41%   | 13.34  | 0.381  | 4.41   | 0.395   | 0.428 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.699   | 45.31%   | 13.28  | 0.382  | 4.49   | 0.398   | 0.404 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.629   | 44.14%   | 13.56  | 0.359  | 4.75   | 0.370   | 0.378 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.580   | 45.31%   | 13.51  | 0.350  | 4.69   | 0.364   | 0.380 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.531   | 44.53%   | 13.32  | 0.342  | 4.58   | 0.363   | 0.377 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.506   | 48.83%   | 13.81  | 0.341  | 4.33   | 0.344   | 0.410 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.463   | 48.44%   | 13.41  | 0.337  | 4.36   | 0.360   | 0.408 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.413   | 50.78%   | 13.59  | 0.335  | 4.42   | 0.354   | 0.392 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.368   | 53.12%   | 13.57  | 0.320  | 4.65   | 0.330   | 0.341 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.331   | 53.52%   | 14.09  | 0.318  | 4.61   | 0.341   | 0.303 \n",
      "\n",
      "\n",
      "[Sobol Run 97/768] H_Inertia: 0.2167 | BASE_S: 0.1402 | Tau: 0.0075 | Jitter: 0.33\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.287   | 19.92%   | 10.25  | 0.526  | 4.37   | 0.635   | 0.529 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.049   | 28.91%   | 12.47  | 0.420  | 4.57   | 0.393   | 0.411 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.936   | 28.52%   | 10.13  | 0.453  | 4.13   | 0.426   | 0.541 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.859   | 32.03%   | 12.48  | 0.398  | 4.70   | 0.393   | 0.390 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.779   | 39.45%   | 12.54  | 0.394  | 4.33   | 0.423   | 0.413 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.721   | 38.67%   | 13.61  | 0.367  | 4.51   | 0.389   | 0.441 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.654   | 41.80%   | 12.29  | 0.399  | 4.44   | 0.442   | 0.395 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.645   | 41.02%   | 13.05  | 0.343  | 4.50   | 0.367   | 0.448 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.600   | 44.14%   | 15.30  | 0.324  | 4.55   | 0.341   | 0.444 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.550   | 44.14%   | 14.62  | 0.324  | 4.50   | 0.339   | 0.447 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.493   | 44.92%   | 14.80  | 0.311  | 4.58   | 0.329   | 0.414 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.419   | 47.27%   | 15.37  | 0.307  | 4.49   | 0.326   | 0.376 \n",
      "\n",
      "\n",
      "[Sobol Run 98/768] H_Inertia: 0.8445 | BASE_S: 0.1402 | Tau: 0.0113 | Jitter: 0.33\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.203   | 25.78%   | 6.98   | 0.587  | 4.64   | 0.474   | 0.476 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.009   | 26.56%   | 8.96   | 0.526  | 4.27   | 0.542   | 0.557 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.953   | 31.25%   | 7.99   | 0.529  | 4.05   | 0.638   | 0.485 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.846   | 39.06%   | 8.76   | 0.485  | 4.41   | 0.588   | 0.473 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.745   | 40.62%   | 10.32  | 0.441  | 4.35   | 0.496   | 0.436 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.654   | 42.97%   | 10.83  | 0.434  | 4.54   | 0.458   | 0.452 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.610   | 39.45%   | 11.31  | 0.419  | 4.71   | 0.454   | 0.442 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.574   | 41.02%   | 11.38  | 0.416  | 4.74   | 0.446   | 0.404 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.549   | 38.67%   | 10.93  | 0.416  | 4.49   | 0.446   | 0.346 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.512   | 45.31%   | 11.78  | 0.417  | 4.74   | 0.434   | 0.364 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.467   | 44.92%   | 11.66  | 0.401  | 4.52   | 0.456   | 0.436 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.427   | 47.27%   | 11.46  | 0.394  | 4.57   | 0.460   | 0.442 \n",
      "\n",
      "\n",
      "[Sobol Run 99/768] H_Inertia: 0.2167 | BASE_S: 0.7680 | Tau: 0.0113 | Jitter: 0.33\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.372   | 10.55%   | 1.89   | 0.361  | 4.88   | 0.333   | 0.476 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.392   | 21.09%   | 2.15   | 0.546  | 4.71   | 0.392   | 0.685 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.304   | 15.62%   | 4.78   | 0.450  | 2.79   | 0.344   | 0.559 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.258   | 15.23%   | 4.43   | 0.575  | 4.25   | 0.332   | 0.710 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.207   | 28.12%   | 4.71   | 0.524  | 4.37   | 0.334   | 0.625 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.218   | 18.75%   | 10.93  | 0.427  | 4.52   | 0.314   | 0.293 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.163   | 31.25%   | 5.97   | 0.528  | 4.46   | 0.335   | 0.603 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.045   | 30.47%   | 6.39   | 0.480  | 4.49   | 0.332   | 0.517 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 2.045   | 26.56%   | 6.05   | 0.517  | 4.17   | 0.340   | 0.586 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 2.038   | 31.25%   | 5.80   | 0.512  | 4.20   | 0.342   | 0.491 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.963   | 33.59%   | 6.67   | 0.493  | 4.02   | 0.327   | 0.532 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.962   | 34.38%   | 6.63   | 0.497  | 4.14   | 0.330   | 0.489 \n",
      "\n",
      "\n",
      "[Sobol Run 100/768] H_Inertia: 0.2167 | BASE_S: 0.1402 | Tau: 0.0113 | Jitter: 0.33\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.212   | 23.44%   | 12.18  | 0.452  | 4.44   | 0.394   | 0.221 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.945   | 35.55%   | 12.07  | 0.416  | 4.46   | 0.356   | 0.316 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.773   | 42.97%   | 13.83  | 0.338  | 4.52   | 0.336   | 0.323 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.672   | 46.88%   | 12.56  | 0.358  | 4.30   | 0.323   | 0.301 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.603   | 51.95%   | 13.94  | 0.342  | 4.42   | 0.323   | 0.332 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.500   | 50.78%   | 12.64  | 0.341  | 4.12   | 0.314   | 0.352 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.496   | 53.91%   | 14.55  | 0.309  | 4.43   | 0.316   | 0.330 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.458   | 54.30%   | 13.39  | 0.309  | 4.02   | 0.306   | 0.344 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.386   | 55.08%   | 14.25  | 0.296  | 4.16   | 0.308   | 0.351 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.347   | 56.64%   | 13.90  | 0.321  | 4.26   | 0.310   | 0.324 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.339   | 56.64%   | 14.03  | 0.288  | 4.18   | 0.294   | 0.358 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-04-05 05:13:01.857485: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.269   | 55.08%   | 14.14  | 0.279  | 4.27   | 0.300   | 0.356 \n",
      "\n",
      "\n",
      "[Sobol Run 101/768] H_Inertia: 0.2167 | BASE_S: 0.1402 | Tau: 0.0113 | Jitter: 0.44\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.281   | 25.00%   | 12.63  | 0.462  | 3.98   | 0.482   | 0.402 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.027   | 27.73%   | 13.04  | 0.421  | 4.48   | 0.441   | 0.329 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.912   | 35.16%   | 13.52  | 0.409  | 4.32   | 0.447   | 0.352 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.800   | 37.89%   | 13.93  | 0.373  | 4.37   | 0.377   | 0.372 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.698   | 39.06%   | 12.64  | 0.354  | 4.09   | 0.363   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.611   | 45.70%   | 14.38  | 0.325  | 4.19   | 0.322   | 0.387 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.529   | 44.53%   | 14.47  | 0.304  | 4.09   | 0.303   | 0.374 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.454   | 47.27%   | 14.47  | 0.304  | 4.04   | 0.340   | 0.366 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.414   | 48.05%   | 15.04  | 0.287  | 4.31   | 0.305   | 0.354 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.372   | 51.95%   | 14.11  | 0.304  | 4.18   | 0.314   | 0.376 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.326   | 53.12%   | 14.98  | 0.288  | 4.27   | 0.297   | 0.352 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.285   | 53.12%   | 15.02  | 0.284  | 4.38   | 0.298   | 0.304 \n",
      "\n",
      "\n",
      "[Sobol Run 102/768] H_Inertia: 0.8445 | BASE_S: 0.7680 | Tau: 0.0113 | Jitter: 0.44\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.509   | 26.17%   | 6.11   | 0.633  | 4.33   | 0.423   | 0.601 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.033   | 27.73%   | 6.14   | 0.610  | 4.27   | 0.419   | 0.578 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.933   | 30.47%   | 7.82   | 0.536  | 4.37   | 0.356   | 0.433 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.848   | 36.72%   | 9.47   | 0.513  | 4.23   | 0.355   | 0.449 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.772   | 39.84%   | 9.21   | 0.498  | 4.08   | 0.362   | 0.451 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.731   | 44.53%   | 9.63   | 0.453  | 3.69   | 0.360   | 0.405 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.674   | 45.70%   | 9.54   | 0.470  | 3.88   | 0.342   | 0.378 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.601   | 44.92%   | 10.68  | 0.423  | 3.78   | 0.330   | 0.410 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.528   | 46.88%   | 10.47  | 0.436  | 3.76   | 0.331   | 0.388 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.479   | 48.83%   | 10.82  | 0.399  | 3.90   | 0.329   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.477   | 48.83%   | 10.94  | 0.405  | 3.65   | 0.347   | 0.440 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.426   | 55.86%   | 11.17  | 0.398  | 3.63   | 0.331   | 0.401 \n",
      "\n",
      "\n",
      "[Sobol Run 103/768] H_Inertia: 0.2167 | BASE_S: 0.7680 | Tau: 0.0075 | Jitter: 0.44\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.739   | 10.55%   | 9.39   | 0.355  | 4.73   | 0.327   | 0.427 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.171   | 30.47%   | 12.18  | 0.467  | 5.04   | 0.320   | 0.391 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.913   | 33.98%   | 12.73  | 0.366  | 2.97   | 0.302   | 0.419 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.522   | 20.70%   | 13.00  | 0.300  | 3.40   | 0.321   | 0.450 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 3.089   | 25.00%   | 14.34  | 0.337  | 2.98   | 0.333   | 0.409 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.928   | 32.81%   | 13.38  | 0.443  | 5.00   | 0.430   | 0.318 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.754   | 40.23%   | 13.06  | 0.484  | 4.84   | 0.383   | 0.479 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.685   | 47.27%   | 13.37  | 0.415  | 4.73   | 0.393   | 0.282 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.534   | 46.48%   | 14.80  | 0.400  | 4.55   | 0.347   | 0.331 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.436   | 43.36%   | 14.11  | 0.395  | 4.59   | 0.357   | 0.273 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.427   | 50.00%   | 14.14  | 0.411  | 4.60   | 0.359   | 0.311 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.354   | 48.05%   | 14.72  | 0.386  | 4.51   | 0.322   | 0.283 \n",
      "\n",
      "\n",
      "[Sobol Run 104/768] H_Inertia: 0.8445 | BASE_S: 0.1402 | Tau: 0.0075 | Jitter: 0.44\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.228   | 20.70%   | 4.93   | 0.763  | 4.97   | 0.548   | 0.771 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.100   | 29.69%   | 8.34   | 0.557  | 4.56   | 0.564   | 0.449 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.955   | 35.94%   | 9.64   | 0.568  | 4.48   | 0.530   | 0.523 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.839   | 35.55%   | 10.73  | 0.535  | 4.70   | 0.512   | 0.499 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.749   | 39.45%   | 11.50  | 0.475  | 4.82   | 0.461   | 0.528 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.665   | 39.84%   | 11.51  | 0.457  | 4.57   | 0.451   | 0.465 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.601   | 41.80%   | 12.27  | 0.433  | 4.41   | 0.427   | 0.443 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.532   | 44.14%   | 11.88  | 0.430  | 4.37   | 0.420   | 0.444 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.486   | 37.50%   | 12.52  | 0.422  | 4.36   | 0.423   | 0.437 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.459   | 44.92%   | 12.03  | 0.414  | 4.35   | 0.405   | 0.440 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.485   | 48.44%   | 12.09  | 0.422  | 4.51   | 0.401   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.480   | 46.09%   | 11.89  | 0.411  | 4.34   | 0.402   | 0.440 \n",
      "\n",
      "\n",
      "[Sobol Run 105/768] H_Inertia: 0.8445 | BASE_S: 0.7680 | Tau: 0.0075 | Jitter: 0.44\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.394   | 16.41%   | 4.46   | 0.636  | 4.98   | 0.383   | 0.772 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.102   | 27.73%   | 7.41   | 0.470  | 4.83   | 0.325   | 0.535 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.022   | 29.30%   | 6.76   | 0.504  | 4.78   | 0.327   | 0.628 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.945   | 28.91%   | 8.53   | 0.461  | 4.53   | 0.342   | 0.441 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.859   | 31.25%   | 8.21   | 0.432  | 4.58   | 0.375   | 0.558 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.819   | 35.94%   | 9.83   | 0.419  | 4.64   | 0.356   | 0.468 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.739   | 35.16%   | 10.23  | 0.416  | 4.61   | 0.343   | 0.430 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.705   | 38.28%   | 10.11  | 0.415  | 4.46   | 0.348   | 0.414 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.648   | 41.80%   | 11.44  | 0.396  | 4.47   | 0.342   | 0.361 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.584   | 42.97%   | 11.31  | 0.392  | 4.59   | 0.327   | 0.359 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.523   | 44.14%   | 11.74  | 0.401  | 4.39   | 0.329   | 0.392 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.490   | 47.66%   | 11.47  | 0.387  | 4.66   | 0.341   | 0.418 \n",
      "\n",
      "\n",
      "[Sobol Run 106/768] H_Inertia: 0.8445 | BASE_S: 0.7680 | Tau: 0.0075 | Jitter: 0.33\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.154   | 25.00%   | 7.75   | 0.620  | 4.94   | 0.340   | 0.704 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.008   | 33.98%   | 8.27   | 0.558  | 4.27   | 0.398   | 0.483 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.886   | 35.55%   | 8.33   | 0.499  | 4.15   | 0.430   | 0.445 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.778   | 33.20%   | 7.79   | 0.436  | 4.09   | 0.403   | 0.395 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.746   | 44.14%   | 8.52   | 0.460  | 4.08   | 0.388   | 0.442 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.650   | 41.41%   | 8.12   | 0.446  | 3.77   | 0.376   | 0.417 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.581   | 43.75%   | 8.16   | 0.446  | 4.06   | 0.360   | 0.396 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.534   | 45.70%   | 8.82   | 0.416  | 3.66   | 0.351   | 0.417 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.505   | 47.66%   | 8.90   | 0.435  | 4.02   | 0.357   | 0.433 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.462   | 46.88%   | 9.06   | 0.400  | 3.76   | 0.349   | 0.397 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.423   | 47.27%   | 9.39   | 0.398  | 3.68   | 0.334   | 0.400 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.393   | 51.56%   | 9.28   | 0.395  | 3.57   | 0.327   | 0.404 \n",
      "\n",
      "\n",
      "[Sobol Run 107/768] H_Inertia: 0.8445 | BASE_S: 0.7680 | Tau: 0.0075 | Jitter: 0.44\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.172   | 26.95%   | 8.04   | 0.607  | 4.63   | 0.347   | 0.671 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.999   | 30.08%   | 8.09   | 0.604  | 4.62   | 0.413   | 0.648 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.884   | 35.55%   | 7.89   | 0.491  | 4.51   | 0.415   | 0.446 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.782   | 40.23%   | 8.92   | 0.491  | 4.40   | 0.436   | 0.533 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.683   | 39.84%   | 9.34   | 0.521  | 4.18   | 0.425   | 0.552 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.634   | 42.58%   | 8.61   | 0.491  | 4.16   | 0.411   | 0.455 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.558   | 44.53%   | 9.51   | 0.498  | 3.95   | 0.454   | 0.512 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.541   | 47.66%   | 9.23   | 0.478  | 3.88   | 0.422   | 0.469 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.500   | 50.39%   | 9.13   | 0.440  | 3.88   | 0.433   | 0.464 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.457   | 49.22%   | 9.37   | 0.455  | 3.81   | 0.402   | 0.452 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.449   | 49.22%   | 9.88   | 0.445  | 3.78   | 0.409   | 0.434 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.390   | 53.12%   | 9.65   | 0.431  | 3.70   | 0.399   | 0.434 \n",
      "\n",
      "\n",
      "[Sobol Run 108/768] H_Inertia: 0.7067 | BASE_S: 0.6302 | Tau: 0.0358 | Jitter: 0.88\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.259   | 25.78%   | 10.52  | 0.465  | 3.29   | 0.387   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.075   | 24.22%   | 9.50   | 0.466  | 3.30   | 0.373   | 0.475 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.976   | 32.03%   | 9.69   | 0.437  | 3.30   | 0.385   | 0.366 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.899   | 39.84%   | 8.82   | 0.441  | 3.10   | 0.391   | 0.426 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.789   | 38.67%   | 9.56   | 0.442  | 3.31   | 0.378   | 0.368 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.775   | 38.28%   | 9.05   | 0.427  | 3.67   | 0.374   | 0.459 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.665   | 42.97%   | 9.01   | 0.414  | 2.94   | 0.362   | 0.396 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.624   | 41.41%   | 10.20  | 0.414  | 3.35   | 0.353   | 0.394 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.537   | 45.70%   | 9.45   | 0.400  | 3.07   | 0.362   | 0.383 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.516   | 39.84%   | 9.86   | 0.419  | 3.51   | 0.360   | 0.393 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.491   | 47.27%   | 9.76   | 0.385  | 3.07   | 0.345   | 0.393 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.482   | 46.88%   | 9.70   | 0.376  | 3.16   | 0.360   | 0.408 \n",
      "\n",
      "\n",
      "[Sobol Run 109/768] H_Inertia: 0.7067 | BASE_S: 0.6302 | Tau: 0.0320 | Jitter: 0.88\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.239   | 23.44%   | 10.60  | 0.533  | 3.29   | 0.364   | 0.342 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.031   | 32.03%   | 10.94  | 0.535  | 3.11   | 0.386   | 0.326 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.967   | 32.42%   | 12.42  | 0.478  | 2.88   | 0.368   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.862   | 34.77%   | 10.20  | 0.469  | 2.86   | 0.374   | 0.462 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.782   | 37.50%   | 10.80  | 0.438  | 3.23   | 0.361   | 0.358 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.742   | 39.45%   | 11.47  | 0.397  | 2.89   | 0.347   | 0.397 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.698   | 41.80%   | 11.48  | 0.402  | 3.51   | 0.344   | 0.413 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.651   | 40.62%   | 11.95  | 0.390  | 3.06   | 0.358   | 0.404 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.550   | 47.27%   | 12.71  | 0.362  | 3.24   | 0.360   | 0.390 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.552   | 44.92%   | 13.19  | 0.338  | 3.80   | 0.331   | 0.330 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.478   | 50.39%   | 12.67  | 0.347  | 3.96   | 0.350   | 0.360 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.430   | 51.17%   | 13.02  | 0.343  | 4.14   | 0.344   | 0.345 \n",
      "\n",
      "\n",
      "[Sobol Run 110/768] H_Inertia: 0.3545 | BASE_S: 0.6302 | Tau: 0.0358 | Jitter: 0.88\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 110 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 111/768] H_Inertia: 0.7067 | BASE_S: 0.2780 | Tau: 0.0358 | Jitter: 0.88\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.200   | 26.95%   | 9.70   | 0.611  | 4.11   | 0.458   | 0.407 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.063   | 25.78%   | 9.22   | 0.625  | 4.40   | 0.464   | 0.315 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.995   | 27.73%   | 9.22   | 0.604  | 4.40   | 0.503   | 0.512 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.915   | 32.42%   | 9.37   | 0.611  | 4.58   | 0.465   | 0.522 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.885   | 38.67%   | 10.19  | 0.597  | 4.56   | 0.452   | 0.587 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.793   | 35.94%   | 10.69  | 0.502  | 4.39   | 0.395   | 0.492 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.753   | 41.80%   | 11.49  | 0.435  | 4.64   | 0.382   | 0.492 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.662   | 42.58%   | 11.89  | 0.438  | 4.53   | 0.364   | 0.430 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.592   | 48.05%   | 12.21  | 0.412  | 4.43   | 0.367   | 0.430 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.588   | 44.14%   | 12.30  | 0.432  | 4.35   | 0.365   | 0.438 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.516   | 46.88%   | 12.39  | 0.408  | 4.29   | 0.360   | 0.406 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.474   | 48.83%   | 12.45  | 0.413  | 4.32   | 0.353   | 0.407 \n",
      "\n",
      "\n",
      "[Sobol Run 112/768] H_Inertia: 0.7067 | BASE_S: 0.6302 | Tau: 0.0358 | Jitter: 0.88\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.064   | 24.61%   | 9.87   | 0.480  | 4.23   | 0.448   | 0.537 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.066   | 29.30%   | 10.92  | 0.485  | 4.34   | 0.552   | 0.348 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.948   | 31.64%   | 11.83  | 0.453  | 4.26   | 0.446   | 0.425 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.877   | 33.59%   | 11.92  | 0.430  | 4.52   | 0.420   | 0.495 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.834   | 34.77%   | 12.05  | 0.466  | 3.84   | 0.407   | 0.348 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.778   | 37.50%   | 11.79  | 0.465  | 3.67   | 0.423   | 0.618 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 5.512   | 14.06%   | 22.87  | 0.202  | 3.88   | 0.355   | 0.428 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 12.273  | 11.72%   | 19.85  | 0.223  | 3.89   | 0.403   | 0.416 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 9.435   | 10.55%   | 16.37  | 0.271  | 3.56   | 0.401   | 0.369 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 10.280  | 12.50%   | 17.34  | 0.270  | 3.77   | 0.316   | 0.486 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 10.157  | 17.97%   | 22.77  | 0.241  | 3.85   | 0.288   | 0.471 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 10.023  | 9.77%    | 17.73  | 0.403  | 3.91   | 0.345   | 0.587 \n",
      "\n",
      "\n",
      "[Sobol Run 113/768] H_Inertia: 0.7067 | BASE_S: 0.6302 | Tau: 0.0358 | Jitter: 0.99\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.249   | 26.56%   | 11.37  | 0.513  | 3.17   | 0.412   | 0.277 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.025   | 35.16%   | 12.07  | 0.484  | 3.93   | 0.371   | 0.399 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.932   | 33.98%   | 12.14  | 0.470  | 3.41   | 0.394   | 0.333 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 6.492   | 8.98%    | 13.08  | 0.331  | 2.89   | 0.416   | 0.510 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 3.979   | 11.33%   | 5.42   | 0.545  | 2.35   | 0.531   | 0.699 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 3.148   | 6.25%    | 10.26  | 0.373  | 3.30   | 0.433   | 0.558 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.534   | 20.31%   | 8.20   | 0.752  | 4.03   | 0.444   | 0.733 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.079   | 26.56%   | 9.05   | 0.650  | 3.62   | 0.446   | 0.667 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.968   | 31.25%   | 9.37   | 0.585  | 3.55   | 0.408   | 0.615 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.895   | 32.42%   | 10.03  | 0.549  | 3.60   | 0.392   | 0.600 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.802   | 35.55%   | 10.03  | 0.499  | 3.87   | 0.370   | 0.568 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.785   | 39.45%   | 11.06  | 0.463  | 3.77   | 0.366   | 0.516 \n",
      "\n",
      "\n",
      "[Sobol Run 114/768] H_Inertia: 0.3545 | BASE_S: 0.2780 | Tau: 0.0358 | Jitter: 0.99\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.205   | 25.78%   | 11.07  | 0.454  | 4.50   | 0.524   | 0.388 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.997   | 32.42%   | 12.63  | 0.505  | 4.84   | 0.530   | 0.214 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.876   | 32.81%   | 12.42  | 0.438  | 4.73   | 0.519   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.785   | 39.06%   | 12.97  | 0.435  | 4.46   | 0.447   | 0.334 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.697   | 38.28%   | 13.22  | 0.381  | 3.93   | 0.388   | 0.462 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.686   | 44.53%   | 13.14  | 0.466  | 4.55   | 0.401   | 0.316 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.577   | 46.88%   | 13.45  | 0.456  | 4.20   | 0.385   | 0.309 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.530   | 50.78%   | 13.94  | 0.449  | 4.34   | 0.373   | 0.304 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.501   | 48.44%   | 14.10  | 0.400  | 4.38   | 0.357   | 0.339 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.459   | 51.17%   | 13.98  | 0.411  | 4.04   | 0.366   | 0.314 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.431   | 51.56%   | 14.94  | 0.386  | 4.21   | 0.361   | 0.301 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.377   | 56.25%   | 13.87  | 0.432  | 4.17   | 0.385   | 0.362 \n",
      "\n",
      "\n",
      "[Sobol Run 115/768] H_Inertia: 0.7067 | BASE_S: 0.2780 | Tau: 0.0320 | Jitter: 0.99\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.196   | 26.95%   | 8.41   | 0.574  | 4.50   | 0.481   | 0.405 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.988   | 36.33%   | 8.70   | 0.572  | 4.69   | 0.467   | 0.603 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.885   | 36.72%   | 9.96   | 0.520  | 4.58   | 0.421   | 0.526 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.801   | 39.84%   | 10.25  | 0.510  | 4.38   | 0.382   | 0.460 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.727   | 44.92%   | 10.59  | 0.488  | 4.55   | 0.387   | 0.518 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.652   | 45.70%   | 11.04  | 0.462  | 4.67   | 0.411   | 0.479 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.613   | 41.80%   | 9.83   | 0.501  | 4.40   | 0.393   | 0.428 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.598   | 45.31%   | 11.37  | 0.464  | 4.53   | 0.403   | 0.425 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.546   | 46.09%   | 10.39  | 0.479  | 4.45   | 0.405   | 0.428 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.502   | 46.88%   | 11.28  | 0.454  | 4.18   | 0.422   | 0.332 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.534   | 50.00%   | 10.50  | 0.456  | 4.60   | 0.420   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.479   | 48.83%   | 10.69  | 0.479  | 4.28   | 0.422   | 0.385 \n",
      "\n",
      "\n",
      "[Sobol Run 116/768] H_Inertia: 0.3545 | BASE_S: 0.6302 | Tau: 0.0320 | Jitter: 0.99\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 116 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 117/768] H_Inertia: 0.3545 | BASE_S: 0.2780 | Tau: 0.0320 | Jitter: 0.99\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.201   | 18.36%   | 7.67   | 0.540  | 5.00   | 0.405   | 0.479 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.050   | 33.59%   | 10.57  | 0.466  | 4.56   | 0.401   | 0.504 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.947   | 28.91%   | 9.49   | 0.500  | 4.44   | 0.414   | 0.596 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.953   | 33.59%   | 11.64  | 0.444  | 4.49   | 0.381   | 0.483 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.979   | 37.11%   | 12.28  | 0.482  | 4.59   | 0.444   | 0.544 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.808   | 39.84%   | 12.36  | 0.447  | 4.47   | 0.399   | 0.455 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.745   | 43.75%   | 13.68  | 0.400  | 4.54   | 0.345   | 0.526 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.746   | 45.31%   | 14.45  | 0.382  | 4.97   | 0.360   | 0.368 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.646   | 48.05%   | 14.57  | 0.376  | 4.42   | 0.349   | 0.475 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.572   | 46.88%   | 14.49  | 0.364  | 4.48   | 0.337   | 0.467 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.532   | 42.58%   | 13.35  | 0.362  | 4.08   | 0.339   | 0.494 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.541   | 50.78%   | 14.81  | 0.349  | 4.45   | 0.356   | 0.410 \n",
      "\n",
      "\n",
      "[Sobol Run 118/768] H_Inertia: 0.3545 | BASE_S: 0.2780 | Tau: 0.0320 | Jitter: 0.88\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.221   | 27.73%   | 11.01  | 0.508  | 4.70   | 0.434   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.976   | 31.64%   | 10.04  | 0.468  | 4.47   | 0.475   | 0.420 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.858   | 37.50%   | 11.81  | 0.453  | 4.33   | 0.463   | 0.302 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.762   | 35.94%   | 8.72   | 0.376  | 3.09   | 0.455   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 7.600   | 20.31%   | 7.15   | 0.414  | 3.59   | 0.452   | 0.494 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.014   | 38.67%   | 13.06  | 0.457  | 4.12   | 0.427   | 0.427 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.707   | 37.11%   | 13.04  | 0.463  | 4.18   | 0.432   | 0.405 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.631   | 42.19%   | 12.62  | 0.455  | 4.39   | 0.448   | 0.371 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.595   | 41.80%   | 12.94  | 0.417  | 4.13   | 0.361   | 0.439 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.552   | 45.31%   | 13.58  | 0.410  | 4.05   | 0.367   | 0.344 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.497   | 46.09%   | 13.42  | 0.380  | 3.87   | 0.379   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.490   | 45.70%   | 13.95  | 0.391  | 4.24   | 0.366   | 0.334 \n",
      "\n",
      "\n",
      "[Sobol Run 119/768] H_Inertia: 0.3545 | BASE_S: 0.2780 | Tau: 0.0320 | Jitter: 0.99\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.284   | 23.05%   | 11.75  | 0.484  | 4.59   | 0.488   | 0.559 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.025   | 34.77%   | 11.63  | 0.488  | 4.16   | 0.404   | 0.262 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.882   | 32.81%   | 12.96  | 0.438  | 4.11   | 0.386   | 0.335 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.798   | 39.45%   | 12.18  | 0.436  | 4.00   | 0.377   | 0.430 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.723   | 40.23%   | 12.85  | 0.388  | 4.06   | 0.375   | 0.408 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.676   | 44.92%   | 12.81  | 0.385  | 4.02   | 0.367   | 0.357 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.605   | 48.05%   | 12.57  | 0.382  | 3.87   | 0.378   | 0.434 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.561   | 48.44%   | 12.56  | 0.368  | 3.83   | 0.361   | 0.411 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.557   | 48.44%   | 13.14  | 0.364  | 3.80   | 0.359   | 0.374 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.495   | 51.17%   | 12.63  | 0.374  | 3.79   | 0.348   | 0.324 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.460   | 50.39%   | 12.34  | 0.356  | 3.53   | 0.351   | 0.344 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.449   | 51.95%   | 12.98  | 0.346  | 3.48   | 0.337   | 0.318 \n",
      "\n",
      "\n",
      "[Sobol Run 120/768] H_Inertia: 0.4617 | BASE_S: 0.3852 | Tau: 0.0481 | Jitter: 1.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.354   | 26.17%   | 7.63   | 0.477  | 4.44   | 0.359   | 0.575 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.111   | 27.73%   | 9.94   | 0.449  | 2.75   | 0.389   | 0.487 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.101   | 34.77%   | 11.26  | 0.453  | 4.08   | 0.424   | 0.552 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.902   | 35.16%   | 10.63  | 0.444  | 4.15   | 0.442   | 0.559 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.824   | 35.94%   | 11.42  | 0.412  | 3.90   | 0.435   | 0.453 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.765   | 40.23%   | 12.97  | 0.366  | 3.97   | 0.392   | 0.449 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.659   | 45.31%   | 13.16  | 0.333  | 3.53   | 0.349   | 0.417 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.608   | 45.31%   | 12.55  | 0.344  | 3.29   | 0.334   | 0.434 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 6.326   | 10.55%   | 20.04  | 0.215  | 3.74   | 0.242   | 0.441 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 18.745  | 8.59%    | 21.15  | 0.153  | 3.43   | 0.228   | 0.311 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 19.700  | 10.94%   | 21.92  | 0.190  | 3.58   | 0.207   | 0.426 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 18.760  | 11.72%   | 21.70  | 0.142  | 3.45   | 0.224   | 0.299 \n",
      "\n",
      "\n",
      "[Sobol Run 121/768] H_Inertia: 0.4617 | BASE_S: 0.3852 | Tau: 0.0443 | Jitter: 1.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.547   | 16.41%   | 3.55   | 0.616  | 4.69   | 0.556   | 0.661 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.200   | 24.22%   | 5.95   | 0.763  | 4.63   | 0.524   | 0.633 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.040   | 30.47%   | 7.18   | 0.577  | 4.57   | 0.486   | 0.400 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.968   | 26.56%   | 6.49   | 0.595  | 4.55   | 0.488   | 0.358 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.906   | 30.47%   | 6.54   | 0.609  | 4.34   | 0.487   | 0.191 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.919   | 34.38%   | 6.90   | 0.537  | 4.42   | 0.475   | 0.202 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.814   | 28.91%   | 6.91   | 0.498  | 3.96   | 0.454   | 0.395 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.203   | 32.42%   | 8.35   | 0.489  | 4.27   | 0.361   | 0.321 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.857   | 37.89%   | 8.58   | 0.460  | 4.53   | 0.463   | 0.221 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.707   | 41.02%   | 8.34   | 0.456  | 4.09   | 0.422   | 0.247 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.661   | 38.28%   | 8.38   | 0.431  | 3.99   | 0.388   | 0.321 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.802   | 48.83%   | 9.68   | 0.409  | 4.29   | 0.412   | 0.294 \n",
      "\n",
      "\n",
      "[Sobol Run 122/768] H_Inertia: 0.5995 | BASE_S: 0.3852 | Tau: 0.0481 | Jitter: 1.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.215   | 30.08%   | 10.67  | 0.562  | 4.36   | 0.553   | 0.367 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.086   | 32.42%   | 11.21  | 0.542  | 4.70   | 0.467   | 0.480 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.978   | 35.16%   | 10.29  | 0.563  | 4.48   | 0.452   | 0.596 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.957   | 37.11%   | 10.86  | 0.495  | 4.50   | 0.425   | 0.420 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.836   | 38.67%   | 11.38  | 0.457  | 4.02   | 0.425   | 0.488 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 3.140   | 10.55%   | 13.86  | 0.313  | 3.98   | 0.395   | 0.487 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 6.054   | 22.27%   | 8.71   | 0.456  | 4.95   | 0.484   | 0.526 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.987   | 31.25%   | 11.15  | 0.493  | 4.94   | 0.500   | 0.454 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.853   | 35.55%   | 13.22  | 0.450  | 4.77   | 0.422   | 0.468 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.759   | 35.55%   | 13.58  | 0.400  | 4.69   | 0.444   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.656   | 40.62%   | 14.86  | 0.389  | 4.56   | 0.354   | 0.439 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.671   | 42.58%   | 13.94  | 0.396  | 4.66   | 0.411   | 0.339 \n",
      "\n",
      "\n",
      "[Sobol Run 123/768] H_Inertia: 0.4617 | BASE_S: 0.0330 | Tau: 0.0481 | Jitter: 1.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.178   | 30.08%   | 10.14  | 0.553  | 4.01   | 0.546   | 0.633 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.960   | 39.06%   | 12.13  | 0.431  | 4.00   | 0.427   | 0.406 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.839   | 42.58%   | 13.15  | 0.407  | 4.25   | 0.392   | 0.348 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.772   | 43.36%   | 13.93  | 0.378  | 4.12   | 0.365   | 0.301 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.683   | 46.09%   | 13.79  | 0.379  | 3.81   | 0.356   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.624   | 49.22%   | 14.44  | 0.341  | 3.90   | 0.336   | 0.407 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.579   | 45.31%   | 13.61  | 0.357  | 3.74   | 0.345   | 0.407 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.564   | 50.00%   | 14.36  | 0.343  | 3.73   | 0.333   | 0.446 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.483   | 51.95%   | 14.67  | 0.319  | 3.68   | 0.328   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.449   | 52.34%   | 14.56  | 0.318  | 3.81   | 0.321   | 0.426 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.414   | 50.00%   | 14.54  | 0.349  | 3.98   | 0.320   | 0.350 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.406   | 51.17%   | 14.25  | 0.327  | 3.81   | 0.317   | 0.458 \n",
      "\n",
      "\n",
      "[Sobol Run 124/768] H_Inertia: 0.4617 | BASE_S: 0.3852 | Tau: 0.0481 | Jitter: 1.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.184   | 25.78%   | 9.46   | 0.670  | 4.08   | 0.595   | 0.624 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.996   | 33.20%   | 9.21   | 0.467  | 2.52   | 0.521   | 0.524 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.886   | 41.02%   | 11.92  | 0.433  | 3.97   | 0.515   | 0.351 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.843   | 41.41%   | 12.46  | 0.434  | 4.41   | 0.476   | 0.254 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.764   | 41.02%   | 12.47  | 0.425  | 4.38   | 0.469   | 0.288 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.148   | 32.42%   | 13.62  | 0.402  | 3.39   | 0.428   | 0.623 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.063   | 42.58%   | 13.30  | 0.432  | 4.52   | 0.418   | 0.267 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.674   | 44.92%   | 13.79  | 0.379  | 3.41   | 0.411   | 0.395 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.590   | 48.83%   | 13.06  | 0.372  | 3.02   | 0.388   | 0.421 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.551   | 45.31%   | 14.04  | 0.359  | 4.34   | 0.391   | 0.264 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.503   | 50.78%   | 14.29  | 0.349  | 3.72   | 0.403   | 0.319 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.468   | 53.91%   | 14.28  | 0.354  | 3.57   | 0.410   | 0.280 \n",
      "\n",
      "\n",
      "[Sobol Run 125/768] H_Inertia: 0.4617 | BASE_S: 0.3852 | Tau: 0.0481 | Jitter: 0.71\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.184   | 30.47%   | 10.87  | 0.460  | 4.29   | 0.468   | 0.572 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.981   | 33.98%   | 11.02  | 0.455  | 4.02   | 0.579   | 0.580 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.870   | 40.23%   | 11.10  | 0.464  | 3.36   | 0.576   | 0.598 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.779   | 41.41%   | 11.08  | 0.448  | 3.74   | 0.548   | 0.601 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.680   | 43.75%   | 10.36  | 0.399  | 3.35   | 0.474   | 0.532 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.619   | 46.09%   | 11.46  | 0.388  | 3.27   | 0.477   | 0.521 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.582   | 43.75%   | 10.81  | 0.384  | 3.51   | 0.488   | 0.548 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.544   | 49.61%   | 11.61  | 0.369  | 3.42   | 0.475   | 0.525 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.494   | 50.78%   | 12.03  | 0.361  | 3.41   | 0.484   | 0.448 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.476   | 51.56%   | 11.48  | 0.361  | 3.52   | 0.411   | 0.505 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.461   | 50.78%   | 14.12  | 0.320  | 3.59   | 0.458   | 0.456 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.417   | 57.03%   | 12.82  | 0.332  | 3.64   | 0.498   | 0.456 \n",
      "\n",
      "\n",
      "[Sobol Run 126/768] H_Inertia: 0.5995 | BASE_S: 0.0330 | Tau: 0.0481 | Jitter: 0.71\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.220   | 30.08%   | 9.04   | 0.591  | 3.98   | 0.547   | 0.609 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.010   | 32.03%   | 9.93   | 0.522  | 4.12   | 0.484   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.883   | 30.47%   | 11.02  | 0.461  | 4.12   | 0.424   | 0.327 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.794   | 37.50%   | 12.13  | 0.409  | 4.22   | 0.400   | 0.402 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.715   | 38.67%   | 12.23  | 0.395  | 4.38   | 0.379   | 0.455 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.670   | 41.02%   | 12.25  | 0.393  | 4.34   | 0.386   | 0.438 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.628   | 43.75%   | 12.64  | 0.379  | 4.39   | 0.358   | 0.398 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.588   | 42.58%   | 12.44  | 0.365  | 4.35   | 0.354   | 0.390 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.558   | 46.09%   | 12.27  | 0.380  | 4.47   | 0.371   | 0.370 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.531   | 42.19%   | 12.63  | 0.369  | 4.28   | 0.364   | 0.367 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.542   | 45.70%   | 12.46  | 0.355  | 4.27   | 0.357   | 0.342 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.491   | 49.61%   | 12.77  | 0.354  | 4.23   | 0.346   | 0.346 \n",
      "\n",
      "\n",
      "[Sobol Run 127/768] H_Inertia: 0.4617 | BASE_S: 0.0330 | Tau: 0.0443 | Jitter: 0.71\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.180   | 28.12%   | 9.65   | 0.601  | 3.86   | 0.644   | 0.728 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.960   | 32.81%   | 11.11  | 0.472  | 4.12   | 0.424   | 0.392 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.828   | 42.19%   | 12.90  | 0.410  | 4.28   | 0.410   | 0.352 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.752   | 42.97%   | 13.88  | 0.378  | 4.13   | 0.347   | 0.455 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.696   | 41.80%   | 14.04  | 0.383  | 4.06   | 0.368   | 0.371 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.609   | 47.27%   | 14.29  | 0.366  | 3.95   | 0.341   | 0.462 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.564   | 47.27%   | 14.55  | 0.352  | 3.82   | 0.338   | 0.466 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.528   | 47.66%   | 15.11  | 0.331  | 4.18   | 0.329   | 0.411 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.466   | 48.83%   | 14.37  | 0.357  | 3.94   | 0.334   | 0.480 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.430   | 50.78%   | 14.92  | 0.331  | 4.01   | 0.322   | 0.390 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.409   | 52.73%   | 14.37  | 0.327  | 3.97   | 0.333   | 0.452 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.364   | 52.34%   | 14.88  | 0.316  | 4.11   | 0.314   | 0.375 \n",
      "\n",
      "\n",
      "[Sobol Run 128/768] H_Inertia: 0.5995 | BASE_S: 0.3852 | Tau: 0.0443 | Jitter: 0.71\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.256   | 22.66%   | 6.16   | 0.779  | 4.34   | 0.566   | 0.698 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.101   | 29.30%   | 7.33   | 0.704  | 4.44   | 0.642   | 0.287 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.060   | 34.38%   | 9.03   | 0.614  | 4.17   | 0.617   | 0.313 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.982   | 33.59%   | 8.88   | 0.567  | 4.20   | 0.512   | 0.482 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.952   | 35.55%   | 10.22  | 0.521  | 4.03   | 0.480   | 0.426 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.894   | 32.81%   | 9.32   | 0.532  | 4.12   | 0.498   | 0.402 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.906   | 36.33%   | 11.17  | 0.460  | 3.99   | 0.433   | 0.371 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.829   | 39.06%   | 12.05  | 0.431  | 3.93   | 0.512   | 0.339 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.781   | 36.72%   | 11.74  | 0.422  | 3.80   | 0.439   | 0.382 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.746   | 41.41%   | 12.71  | 0.406  | 4.03   | 0.411   | 0.359 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.696   | 43.75%   | 12.46  | 0.414  | 4.01   | 0.417   | 0.338 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.688   | 43.75%   | 12.33  | 0.427  | 4.28   | 0.432   | 0.333 \n",
      "\n",
      "\n",
      "[Sobol Run 129/768] H_Inertia: 0.5995 | BASE_S: 0.0330 | Tau: 0.0443 | Jitter: 0.71\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.213   | 29.30%   | 8.57   | 0.586  | 4.16   | 0.530   | 0.598 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.979   | 32.81%   | 9.59   | 0.527  | 4.12   | 0.497   | 0.333 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.876   | 36.72%   | 11.28  | 0.411  | 4.09   | 0.407   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.823   | 37.89%   | 11.35  | 0.441  | 4.19   | 0.407   | 0.287 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.756   | 38.28%   | 12.38  | 0.396  | 4.32   | 0.375   | 0.333 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.670   | 44.53%   | 12.26  | 0.381  | 4.39   | 0.354   | 0.382 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.609   | 41.80%   | 12.33  | 0.389  | 4.40   | 0.353   | 0.344 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.556   | 43.36%   | 12.10  | 0.403  | 4.34   | 0.341   | 0.343 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.538   | 43.75%   | 12.27  | 0.391  | 4.33   | 0.351   | 0.368 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.520   | 42.97%   | 12.14  | 0.395  | 4.26   | 0.346   | 0.382 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.520   | 44.92%   | 11.81  | 0.387  | 4.37   | 0.346   | 0.365 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.505   | 43.75%   | 12.35  | 0.376  | 4.27   | 0.339   | 0.383 \n",
      "\n",
      "\n",
      "[Sobol Run 130/768] H_Inertia: 0.5995 | BASE_S: 0.0330 | Tau: 0.0443 | Jitter: 1.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.237   | 32.03%   | 8.09   | 0.651  | 4.18   | 0.581   | 0.714 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.021   | 33.20%   | 9.20   | 0.573  | 4.41   | 0.535   | 0.203 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.895   | 39.06%   | 10.31  | 0.504  | 4.42   | 0.456   | 0.308 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.809   | 42.97%   | 10.79  | 0.462  | 4.40   | 0.409   | 0.321 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.770   | 42.19%   | 11.47  | 0.445  | 4.36   | 0.399   | 0.311 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.727   | 40.62%   | 12.10  | 0.422  | 4.32   | 0.380   | 0.363 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.659   | 40.62%   | 12.45  | 0.425  | 4.31   | 0.361   | 0.380 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.642   | 42.97%   | 12.08  | 0.386  | 4.51   | 0.366   | 0.347 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.597   | 46.09%   | 12.17  | 0.365  | 4.36   | 0.353   | 0.368 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.578   | 46.48%   | 12.20  | 0.379  | 4.29   | 0.361   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.526   | 47.66%   | 12.64  | 0.364  | 4.33   | 0.351   | 0.342 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.514   | 45.31%   | 12.59  | 0.395  | 4.34   | 0.350   | 0.357 \n",
      "\n",
      "\n",
      "[Sobol Run 131/768] H_Inertia: 0.5995 | BASE_S: 0.0330 | Tau: 0.0443 | Jitter: 0.71\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.228   | 27.73%   | 8.84   | 0.602  | 4.00   | 0.551   | 0.607 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.000   | 32.81%   | 10.05  | 0.531  | 4.29   | 0.501   | 0.240 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.874   | 33.20%   | 11.04  | 0.450  | 4.19   | 0.442   | 0.298 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.786   | 39.06%   | 11.31  | 0.420  | 4.39   | 0.398   | 0.370 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.711   | 39.84%   | 11.86  | 0.410  | 4.37   | 0.375   | 0.378 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.629   | 44.14%   | 11.72  | 0.384  | 4.44   | 0.358   | 0.368 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.614   | 41.41%   | 12.11  | 0.371  | 4.40   | 0.350   | 0.367 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.588   | 46.48%   | 12.02  | 0.403  | 4.37   | 0.358   | 0.388 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.534   | 46.09%   | 11.87  | 0.368  | 4.33   | 0.348   | 0.348 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.515   | 44.92%   | 12.07  | 0.388  | 4.39   | 0.345   | 0.360 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.495   | 48.05%   | 12.19  | 0.379  | 4.43   | 0.355   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.458   | 45.70%   | 12.28  | 0.372  | 4.41   | 0.343   | 0.363 \n",
      "\n",
      "\n",
      "[Sobol Run 132/768] H_Inertia: 0.9517 | BASE_S: 0.8752 | Tau: 0.0236 | Jitter: 0.61\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.360   | 23.83%   | 8.81   | 0.538  | 4.43   | 0.365   | 0.261 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.089   | 28.52%   | 9.25   | 0.498  | 4.97   | 0.360   | 0.577 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.985   | 29.30%   | 9.35   | 0.502  | 4.96   | 0.339   | 0.500 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.907   | 36.72%   | 9.43   | 0.510  | 4.77   | 0.362   | 0.549 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.853   | 37.11%   | 9.39   | 0.490  | 4.77   | 0.364   | 0.552 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.807   | 35.55%   | 9.99   | 0.489  | 4.63   | 0.336   | 0.509 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.781   | 34.38%   | 8.98   | 0.499  | 4.57   | 0.337   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.753   | 33.20%   | 10.68  | 0.464  | 4.43   | 0.336   | 0.462 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.735   | 37.89%   | 10.77  | 0.463  | 4.62   | 0.367   | 0.530 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.653   | 41.41%   | 11.75  | 0.438  | 4.58   | 0.350   | 0.434 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.641   | 43.75%   | 10.96  | 0.434  | 4.40   | 0.337   | 0.400 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.587   | 40.62%   | 11.55  | 0.449  | 4.54   | 0.363   | 0.393 \n",
      "\n",
      "\n",
      "[Sobol Run 133/768] H_Inertia: 0.9517 | BASE_S: 0.8752 | Tau: 0.0198 | Jitter: 0.61\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.339   | 26.56%   | 9.08   | 0.575  | 4.17   | 0.365   | 0.439 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.061   | 28.52%   | 8.84   | 0.540  | 4.07   | 0.400   | 0.407 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.944   | 30.08%   | 10.19  | 0.492  | 4.04   | 0.442   | 0.371 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.859   | 33.59%   | 10.89  | 0.480  | 4.36   | 0.434   | 0.492 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.781   | 37.89%   | 11.51  | 0.440  | 4.21   | 0.376   | 0.433 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.697   | 37.50%   | 11.58  | 0.431  | 4.05   | 0.373   | 0.453 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.661   | 37.89%   | 10.95  | 0.434  | 4.08   | 0.381   | 0.446 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.597   | 43.75%   | 11.31  | 0.436  | 4.23   | 0.358   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.554   | 42.58%   | 11.69  | 0.427  | 3.98   | 0.392   | 0.427 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.543   | 45.70%   | 11.27  | 0.419  | 4.15   | 0.386   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.516   | 43.36%   | 11.33  | 0.405  | 4.08   | 0.389   | 0.396 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.480   | 48.44%   | 12.04  | 0.409  | 4.17   | 0.387   | 0.406 \n",
      "\n",
      "\n",
      "[Sobol Run 134/768] H_Inertia: 0.1095 | BASE_S: 0.8752 | Tau: 0.0236 | Jitter: 0.61\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 134 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 135/768] H_Inertia: 0.9517 | BASE_S: 0.5230 | Tau: 0.0236 | Jitter: 0.61\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.230   | 20.31%   | 4.47   | 0.662  | 4.63   | 0.425   | 0.267 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.067   | 26.17%   | 6.11   | 0.656  | 4.33   | 0.478   | 0.401 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.941   | 29.30%   | 6.40   | 0.638  | 4.43   | 0.518   | 0.410 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.877   | 32.42%   | 7.36   | 0.567  | 4.37   | 0.482   | 0.276 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.807   | 33.98%   | 7.75   | 0.565  | 4.35   | 0.552   | 0.383 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.739   | 37.50%   | 8.78   | 0.490  | 4.52   | 0.508   | 0.352 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.713   | 38.28%   | 8.79   | 0.475  | 4.62   | 0.490   | 0.424 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.678   | 40.23%   | 8.80   | 0.486  | 4.56   | 0.468   | 0.276 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.645   | 41.41%   | 9.17   | 0.460  | 4.62   | 0.457   | 0.333 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.614   | 45.70%   | 9.13   | 0.461  | 4.72   | 0.429   | 0.319 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.570   | 48.05%   | 9.35   | 0.458  | 4.62   | 0.397   | 0.207 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.547   | 45.31%   | 9.26   | 0.480  | 4.71   | 0.403   | 0.265 \n",
      "\n",
      "\n",
      "[Sobol Run 136/768] H_Inertia: 0.9517 | BASE_S: 0.8752 | Tau: 0.0236 | Jitter: 0.61\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.220   | 20.70%   | 5.35   | 0.852  | 4.88   | 0.630   | 0.402 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.105   | 24.61%   | 6.04   | 0.759  | 4.65   | 0.557   | 0.353 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.068   | 21.88%   | 5.61   | 0.767  | 4.66   | 0.482   | 0.514 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.043   | 24.61%   | 7.37   | 0.756  | 4.89   | 0.536   | 0.409 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.956   | 27.73%   | 9.77   | 0.626  | 4.79   | 0.511   | 0.446 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.901   | 33.20%   | 10.32  | 0.536  | 4.59   | 0.458   | 0.391 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.803   | 34.38%   | 10.67  | 0.504  | 4.58   | 0.458   | 0.361 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.692   | 42.58%   | 10.55  | 0.530  | 4.58   | 0.495   | 0.379 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.612   | 43.75%   | 10.47  | 0.518  | 4.57   | 0.476   | 0.348 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.589   | 44.92%   | 9.89   | 0.525  | 4.55   | 0.455   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.562   | 44.53%   | 10.51  | 0.527  | 4.54   | 0.457   | 0.355 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.558   | 44.14%   | 9.89   | 0.521  | 4.52   | 0.464   | 0.331 \n",
      "\n",
      "\n",
      "[Sobol Run 137/768] H_Inertia: 0.9517 | BASE_S: 0.8752 | Tau: 0.0236 | Jitter: 0.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.350   | 21.09%   | 6.44   | 0.673  | 4.58   | 0.396   | 0.545 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.041   | 25.78%   | 8.00   | 0.533  | 4.42   | 0.341   | 0.412 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.944   | 34.38%   | 10.41  | 0.472  | 4.25   | 0.340   | 0.363 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.848   | 37.89%   | 9.11   | 0.475  | 4.11   | 0.360   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.825   | 35.55%   | 10.08  | 0.468  | 4.14   | 0.319   | 0.372 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.731   | 41.02%   | 10.32  | 0.451  | 4.03   | 0.390   | 0.368 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.641   | 42.19%   | 9.93   | 0.471  | 3.94   | 0.386   | 0.370 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.616   | 44.92%   | 9.87   | 0.427  | 3.91   | 0.387   | 0.350 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.588   | 47.27%   | 10.57  | 0.429  | 3.93   | 0.371   | 0.330 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.567   | 46.88%   | 10.70  | 0.426  | 3.96   | 0.414   | 0.347 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.561   | 44.14%   | 10.72  | 0.418  | 3.98   | 0.368   | 0.338 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.532   | 48.44%   | 10.82  | 0.425  | 4.04   | 0.360   | 0.327 \n",
      "\n",
      "\n",
      "[Sobol Run 138/768] H_Inertia: 0.1095 | BASE_S: 0.5230 | Tau: 0.0236 | Jitter: 0.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.280   | 23.05%   | 13.08  | 0.348  | 4.62   | 0.264   | 0.322 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.998   | 38.67%   | 12.81  | 0.409  | 4.09   | 0.291   | 0.237 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.759   | 40.23%   | 12.61  | 0.395  | 4.00   | 0.296   | 0.258 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.624   | 42.58%   | 13.35  | 0.378  | 4.14   | 0.288   | 0.208 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.552   | 49.22%   | 11.68  | 0.420  | 3.71   | 0.300   | 0.212 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.450   | 53.91%   | 11.35  | 0.403  | 3.70   | 0.293   | 0.239 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.372   | 52.73%   | 10.18  | 0.419  | 3.68   | 0.296   | 0.215 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.325   | 55.08%   | 10.74  | 0.389  | 3.71   | 0.284   | 0.234 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.254   | 55.08%   | 9.43   | 0.396  | 3.86   | 0.290   | 0.241 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.239   | 50.00%   | 11.15  | 0.386  | 3.77   | 0.295   | 0.258 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.253   | 59.38%   | 10.41  | 0.388  | 3.93   | 0.281   | 0.226 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.174   | 58.98%   | 10.62  | 0.375  | 3.95   | 0.274   | 0.217 \n",
      "\n",
      "\n",
      "[Sobol Run 139/768] H_Inertia: 0.9517 | BASE_S: 0.5230 | Tau: 0.0198 | Jitter: 0.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.236   | 19.53%   | 3.72   | 0.783  | 4.64   | 0.732   | 0.336 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.124   | 24.22%   | 5.27   | 0.789  | 4.58   | 0.558   | 0.303 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.063   | 23.44%   | 5.61   | 0.742  | 4.63   | 0.520   | 0.409 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.991   | 26.95%   | 7.30   | 0.645  | 4.39   | 0.501   | 0.387 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.930   | 31.64%   | 8.93   | 0.548  | 4.24   | 0.477   | 0.327 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.821   | 37.11%   | 9.68   | 0.508  | 4.35   | 0.447   | 0.353 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.762   | 37.89%   | 9.61   | 0.512  | 4.31   | 0.456   | 0.278 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.715   | 39.06%   | 9.66   | 0.507  | 4.40   | 0.445   | 0.283 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.707   | 38.28%   | 9.47   | 0.515  | 4.32   | 0.449   | 0.301 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.656   | 42.19%   | 10.46  | 0.498  | 4.36   | 0.464   | 0.257 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.679   | 41.80%   | 10.29  | 0.486  | 4.28   | 0.446   | 0.283 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.619   | 42.19%   | 9.81   | 0.497  | 4.29   | 0.437   | 0.288 \n",
      "\n",
      "\n",
      "[Sobol Run 140/768] H_Inertia: 0.1095 | BASE_S: 0.8752 | Tau: 0.0198 | Jitter: 0.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.314   | 13.28%   | 6.03   | 0.338  | 4.88   | 0.466   | 0.428 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.145   | 26.56%   | 12.12  | 0.413  | 3.45   | 0.356   | 0.459 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.069   | 31.25%   | 14.25  | 0.398  | 4.30   | 0.284   | 0.551 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.887   | 37.89%   | 15.08  | 0.402  | 4.76   | 0.278   | 0.514 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.787   | 43.75%   | 15.42  | 0.345  | 4.59   | 0.277   | 0.395 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.858   | 43.36%   | 14.97  | 0.368  | 4.54   | 0.272   | 0.460 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.621   | 46.88%   | 14.79  | 0.342  | 4.50   | 0.253   | 0.433 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.556   | 45.31%   | 14.37  | 0.346  | 4.47   | 0.254   | 0.435 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.501   | 51.17%   | 13.90  | 0.348  | 4.23   | 0.269   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.455   | 50.39%   | 14.61  | 0.313  | 4.44   | 0.263   | 0.422 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.386   | 53.52%   | 14.16  | 0.294  | 4.05   | 0.281   | 0.357 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.355   | 56.64%   | 13.49  | 0.332  | 4.11   | 0.275   | 0.481 \n",
      "\n",
      "\n",
      "[Sobol Run 141/768] H_Inertia: 0.1095 | BASE_S: 0.5230 | Tau: 0.0198 | Jitter: 0.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.830   | 19.53%   | 6.24   | 0.420  | 5.04   | 0.568   | 0.562 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.150   | 27.34%   | 11.95  | 0.427  | 4.83   | 0.488   | 0.476 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.003   | 32.03%   | 15.89  | 0.372  | 3.83   | 0.394   | 0.363 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.934   | 37.89%   | 15.10  | 0.380  | 4.59   | 0.400   | 0.445 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.892   | 39.45%   | 15.25  | 0.378  | 4.00   | 0.394   | 0.397 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.803   | 38.28%   | 13.75  | 0.398  | 4.66   | 0.412   | 0.502 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.771   | 46.48%   | 16.49  | 0.347  | 4.56   | 0.346   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.619   | 37.89%   | 13.66  | 0.390  | 4.65   | 0.387   | 0.506 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.581   | 45.70%   | 15.11  | 0.383  | 4.09   | 0.344   | 0.405 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.517   | 43.36%   | 14.81  | 0.347  | 3.81   | 0.280   | 0.341 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.464   | 51.95%   | 16.12  | 0.336  | 4.34   | 0.314   | 0.346 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.384   | 50.78%   | 15.25  | 0.354  | 4.23   | 0.339   | 0.407 \n",
      "\n",
      "\n",
      "[Sobol Run 142/768] H_Inertia: 0.1095 | BASE_S: 0.5230 | Tau: 0.0198 | Jitter: 0.61\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.255   | 24.61%   | 14.51  | 0.368  | 4.63   | 0.347   | 0.187 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.990   | 35.16%   | 14.07  | 0.420  | 3.26   | 0.348   | 0.339 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.804   | 40.62%   | 16.12  | 0.342  | 4.05   | 0.329   | 0.288 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.706   | 45.31%   | 16.41  | 0.312  | 3.15   | 0.336   | 0.272 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.596   | 48.44%   | 17.29  | 0.321  | 4.12   | 0.349   | 0.335 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.539   | 45.31%   | 16.56  | 0.298  | 4.36   | 0.322   | 0.251 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.468   | 50.39%   | 16.26  | 0.312  | 4.14   | 0.315   | 0.239 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.386   | 51.56%   | 16.58  | 0.319  | 4.08   | 0.316   | 0.253 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.337   | 52.73%   | 16.11  | 0.329  | 3.97   | 0.311   | 0.262 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.272   | 56.25%   | 16.51  | 0.311  | 4.28   | 0.293   | 0.253 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.226   | 56.25%   | 15.49  | 0.319  | 4.03   | 0.290   | 0.236 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.196   | 55.47%   | 16.38  | 0.308  | 4.12   | 0.281   | 0.259 \n",
      "\n",
      "\n",
      "[Sobol Run 143/768] H_Inertia: 0.1095 | BASE_S: 0.5230 | Tau: 0.0198 | Jitter: 0.16\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.250   | 22.27%   | 12.82  | 0.354  | 4.52   | 0.331   | 0.341 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.048   | 28.91%   | 12.63  | 0.381  | 4.36   | 0.297   | 0.421 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.875   | 38.28%   | 14.46  | 0.353  | 3.80   | 0.321   | 0.363 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.737   | 35.55%   | 15.99  | 0.328  | 4.42   | 0.306   | 0.260 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.650   | 40.23%   | 14.34  | 0.331  | 3.99   | 0.315   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.597   | 48.05%   | 14.38  | 0.342  | 3.88   | 0.328   | 0.322 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.514   | 46.48%   | 16.12  | 0.302  | 4.37   | 0.322   | 0.253 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.468   | 44.14%   | 14.70  | 0.297  | 4.27   | 0.259   | 0.319 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.449   | 48.44%   | 15.62  | 0.319  | 4.17   | 0.286   | 0.325 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.336   | 53.12%   | 14.33  | 0.358  | 4.33   | 0.268   | 0.342 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.295   | 53.91%   | 14.62  | 0.335  | 4.33   | 0.271   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.214   | 54.30%   | 14.88  | 0.328  | 4.30   | 0.267   | 0.331 \n",
      "\n",
      "\n",
      "[Sobol Run 144/768] H_Inertia: 0.3392 | BASE_S: 0.5077 | Tau: 0.0175 | Jitter: 0.47\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.419   | 20.31%   | 6.50   | 0.493  | 4.80   | 0.406   | 0.657 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.131   | 26.17%   | 8.03   | 0.544  | 4.16   | 0.388   | 0.683 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.055   | 23.44%   | 8.52   | 0.576  | 4.41   | 0.343   | 0.710 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.080   | 25.78%   | 7.65   | 0.597  | 4.44   | 0.351   | 0.676 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.033   | 26.95%   | 5.86   | 0.685  | 4.69   | 0.357   | 0.790 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.965   | 27.73%   | 5.84   | 0.657  | 4.66   | 0.372   | 0.758 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.965   | 27.34%   | 6.89   | 0.640  | 4.83   | 0.348   | 0.757 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.930   | 32.03%   | 6.96   | 0.616  | 4.75   | 0.346   | 0.728 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.875   | 35.94%   | 7.85   | 0.569  | 4.67   | 0.352   | 0.678 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.856   | 34.38%   | 8.50   | 0.522  | 4.68   | 0.342   | 0.660 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.801   | 37.89%   | 9.32   | 0.523  | 4.62   | 0.339   | 0.493 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.743   | 35.55%   | 9.47   | 0.498  | 4.75   | 0.343   | 0.648 \n",
      "\n",
      "\n",
      "[Sobol Run 145/768] H_Inertia: 0.3392 | BASE_S: 0.5077 | Tau: 0.0014 | Jitter: 0.47\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.371   | 16.02%   | 8.67   | 0.544  | 4.02   | 0.449   | 0.690 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.115   | 21.88%   | 9.95   | 0.512  | 4.34   | 0.408   | 0.629 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.058   | 29.30%   | 9.64   | 0.516  | 4.21   | 0.444   | 0.661 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.007   | 26.56%   | 13.08  | 0.412  | 3.56   | 0.433   | 0.383 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.955   | 32.81%   | 11.06  | 0.534  | 4.36   | 0.398   | 0.510 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.877   | 29.69%   | 12.42  | 0.473  | 4.03   | 0.356   | 0.223 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.890   | 34.38%   | 13.61  | 0.406  | 4.39   | 0.381   | 0.453 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.840   | 36.33%   | 13.55  | 0.402  | 4.66   | 0.362   | 0.477 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.742   | 36.72%   | 13.77  | 0.388  | 4.68   | 0.379   | 0.491 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.663   | 39.84%   | 14.95  | 0.371  | 4.82   | 0.364   | 0.456 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.630   | 43.36%   | 14.62  | 0.357  | 4.72   | 0.361   | 0.387 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.610   | 44.53%   | 14.78  | 0.349  | 4.76   | 0.367   | 0.436 \n",
      "\n",
      "\n",
      "[Sobol Run 146/768] H_Inertia: 0.7220 | BASE_S: 0.5077 | Tau: 0.0175 | Jitter: 0.47\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.298   | 17.19%   | 8.05   | 0.509  | 3.41   | 0.449   | 0.560 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.051   | 32.42%   | 8.77   | 0.565  | 3.51   | 0.463   | 0.457 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.876   | 35.94%   | 10.13  | 0.458  | 3.81   | 0.469   | 0.470 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.787   | 40.62%   | 9.26   | 0.445  | 3.94   | 0.463   | 0.483 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.678   | 46.09%   | 9.92   | 0.413  | 3.76   | 0.408   | 0.418 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.595   | 45.70%   | 10.33  | 0.414  | 3.97   | 0.422   | 0.387 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.523   | 49.61%   | 10.25  | 0.420  | 3.96   | 0.417   | 0.401 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.468   | 46.48%   | 10.49  | 0.421  | 4.02   | 0.417   | 0.398 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.423   | 51.56%   | 11.11  | 0.392  | 3.85   | 0.374   | 0.321 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.391   | 48.05%   | 10.53  | 0.387  | 3.95   | 0.378   | 0.280 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.339   | 50.39%   | 10.33  | 0.406  | 3.86   | 0.366   | 0.288 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.329   | 53.52%   | 10.94  | 0.423  | 4.04   | 0.408   | 0.357 \n",
      "\n",
      "\n",
      "[Sobol Run 147/768] H_Inertia: 0.3392 | BASE_S: 0.1555 | Tau: 0.0175 | Jitter: 0.47\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.250   | 23.05%   | 9.46   | 0.656  | 4.96   | 0.747   | 0.402 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.060   | 26.56%   | 10.29  | 0.577  | 3.95   | 0.574   | 0.330 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.963   | 32.03%   | 10.83  | 0.507  | 3.75   | 0.552   | 0.470 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.877   | 31.64%   | 11.06  | 0.493  | 3.73   | 0.531   | 0.493 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.794   | 40.62%   | 11.89  | 0.455  | 3.64   | 0.472   | 0.414 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.757   | 40.23%   | 12.84  | 0.438  | 3.85   | 0.466   | 0.355 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.674   | 41.80%   | 12.80  | 0.494  | 4.16   | 0.469   | 0.367 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.635   | 41.02%   | 12.72  | 0.472  | 3.84   | 0.465   | 0.380 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.592   | 44.14%   | 12.58  | 0.458  | 3.70   | 0.457   | 0.405 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.566   | 44.53%   | 12.80  | 0.457  | 3.98   | 0.391   | 0.340 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.519   | 45.31%   | 13.58  | 0.453  | 4.08   | 0.435   | 0.335 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.497   | 45.70%   | 13.10  | 0.432  | 3.98   | 0.378   | 0.320 \n",
      "\n",
      "\n",
      "[Sobol Run 148/768] H_Inertia: 0.3392 | BASE_S: 0.5077 | Tau: 0.0175 | Jitter: 0.47\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.428   | 19.53%   | 5.56   | 0.675  | 3.62   | 0.476   | 0.617 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.200   | 20.31%   | 9.87   | 0.485  | 3.35   | 0.543   | 0.424 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.110   | 33.20%   | 7.90   | 0.616  | 3.70   | 0.484   | 0.369 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.048   | 30.08%   | 7.46   | 0.593  | 3.51   | 0.483   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.979   | 34.38%   | 8.99   | 0.543  | 3.21   | 0.439   | 0.441 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.914   | 27.73%   | 10.01  | 0.501  | 3.55   | 0.446   | 0.518 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.904   | 35.94%   | 9.14   | 0.515  | 3.16   | 0.438   | 0.525 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.934   | 37.50%   | 10.01  | 0.494  | 3.21   | 0.438   | 0.509 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.822   | 41.41%   | 9.85   | 0.502  | 3.39   | 0.454   | 0.543 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.824   | 37.50%   | 10.83  | 0.481  | 3.50   | 0.471   | 0.469 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.834   | 37.50%   | 11.06  | 0.468  | 3.52   | 0.430   | 0.510 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.817   | 42.97%   | 12.07  | 0.456  | 3.53   | 0.428   | 0.456 \n",
      "\n",
      "\n",
      "[Sobol Run 149/768] H_Inertia: 0.3392 | BASE_S: 0.5077 | Tau: 0.0175 | Jitter: 1.12\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.606   | 17.58%   | 6.46   | 0.349  | 5.07   | 0.353   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.233   | 21.48%   | 11.52  | 0.577  | 4.43   | 0.350   | 0.327 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.073   | 27.34%   | 9.94   | 0.615  | 4.62   | 0.375   | 0.600 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.989   | 32.81%   | 9.46   | 0.544  | 4.82   | 0.348   | 0.669 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.002   | 29.69%   | 12.41  | 0.407  | 4.40   | 0.330   | 0.359 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.870   | 35.55%   | 12.41  | 0.393  | 3.92   | 0.327   | 0.441 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.816   | 37.11%   | 12.03  | 0.434  | 4.35   | 0.319   | 0.522 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.717   | 43.36%   | 13.24  | 0.401  | 3.85   | 0.295   | 0.503 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.672   | 42.58%   | 14.15  | 0.359  | 3.62   | 0.287   | 0.351 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.692   | 32.03%   | 21.96  | 0.377  | 3.38   | 0.341   | 0.625 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 149 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 150/768] H_Inertia: 0.7220 | BASE_S: 0.1555 | Tau: 0.0175 | Jitter: 1.12\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.274   | 23.44%   | 8.19   | 0.676  | 3.98   | 0.562   | 0.585 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.094   | 32.81%   | 10.19  | 0.555  | 4.12   | 0.497   | 0.304 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.918   | 39.45%   | 12.52  | 0.492  | 4.38   | 0.418   | 0.363 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.809   | 42.97%   | 12.70  | 0.448  | 4.36   | 0.377   | 0.349 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.733   | 44.53%   | 12.23  | 0.433  | 4.34   | 0.379   | 0.372 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.679   | 45.70%   | 12.81  | 0.404  | 4.31   | 0.379   | 0.351 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.617   | 45.70%   | 12.72  | 0.401  | 4.25   | 0.369   | 0.353 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.582   | 45.31%   | 12.88  | 0.405  | 4.30   | 0.380   | 0.347 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.564   | 48.05%   | 13.08  | 0.388  | 4.40   | 0.377   | 0.365 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.497   | 52.34%   | 12.86  | 0.397  | 4.45   | 0.383   | 0.350 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.462   | 50.78%   | 13.23  | 0.362  | 4.25   | 0.375   | 0.324 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.439   | 53.91%   | 13.15  | 0.370  | 4.30   | 0.369   | 0.289 \n",
      "\n",
      "\n",
      "[Sobol Run 151/768] H_Inertia: 0.3392 | BASE_S: 0.1555 | Tau: 0.0014 | Jitter: 1.12\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.261   | 23.05%   | 9.90   | 0.697  | 4.76   | 0.545   | 0.197 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.021   | 31.64%   | 9.43   | 0.527  | 4.18   | 0.467   | 0.424 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.887   | 39.84%   | 11.08  | 0.512  | 4.27   | 0.410   | 0.282 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.763   | 41.80%   | 11.70  | 0.452  | 3.93   | 0.384   | 0.322 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.679   | 44.14%   | 11.46  | 0.423  | 3.88   | 0.372   | 0.423 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.623   | 46.88%   | 11.98  | 0.426  | 3.94   | 0.379   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.573   | 50.39%   | 12.25  | 0.406  | 3.82   | 0.383   | 0.366 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.544   | 44.53%   | 12.69  | 0.364  | 3.87   | 0.389   | 0.395 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.514   | 49.22%   | 12.31  | 0.430  | 4.10   | 0.394   | 0.386 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.478   | 53.52%   | 12.52  | 0.401  | 3.77   | 0.383   | 0.374 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.461   | 50.78%   | 12.27  | 0.380  | 3.89   | 0.386   | 0.382 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.433   | 52.34%   | 12.80  | 0.382  | 3.85   | 0.384   | 0.363 \n",
      "\n",
      "\n",
      "[Sobol Run 152/768] H_Inertia: 0.7220 | BASE_S: 0.5077 | Tau: 0.0014 | Jitter: 1.12\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 6.339   | 13.28%   | 8.15   | 0.339  | 3.30   | 0.550   | 0.438 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 11.859  | 9.38%    | 6.85   | nan    | 3.87   | 0.363   | 0.566 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 5.776   | 9.77%    | 1.43   | nan    | 3.06   | 0.377   | 0.518 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 3.217   | 10.55%   | 1.18   | nan    | 3.13   | 0.356   | 0.746 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.408   | 12.50%   | 1.26   | nan    | 3.15   | 0.362   | 0.760 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.231   | 19.14%   | 1.41   | nan    | 3.25   | 0.365   | 0.709 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.158   | 19.53%   | 1.42   | nan    | 3.28   | 0.357   | 0.625 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.125   | 18.75%   | 1.42   | nan    | 3.22   | 0.356   | 0.521 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 2.129   | 26.17%   | 1.58   | nan    | 3.35   | 0.342   | 0.652 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 2.133   | 16.41%   | 1.54   | nan    | 3.24   | 0.327   | 0.533 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 2.144   | 22.66%   | 1.46   | nan    | 3.20   | 0.334   | 0.715 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 2.112   | 25.78%   | 1.58   | nan    | 3.33   | 0.339   | 0.696 \n",
      "\n",
      "\n",
      "[Sobol Run 153/768] H_Inertia: 0.7220 | BASE_S: 0.1555 | Tau: 0.0014 | Jitter: 1.12\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.258   | 19.92%   | 5.85   | 0.702  | 4.27   | 0.542   | 0.542 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.101   | 30.86%   | 7.71   | 0.617  | 4.19   | 0.478   | 0.531 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.992   | 38.67%   | 9.83   | 0.472  | 4.00   | 0.399   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.922   | 40.23%   | 11.06  | 0.413  | 4.18   | 0.378   | 0.437 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.841   | 38.28%   | 11.77  | 0.371  | 4.20   | 0.415   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.831   | 42.97%   | 12.31  | 0.388  | 4.09   | 0.424   | 0.427 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.774   | 41.80%   | 12.05  | 0.388  | 4.11   | 0.378   | 0.391 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.705   | 43.75%   | 13.39  | 0.353  | 3.90   | 0.400   | 0.406 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.641   | 48.05%   | 13.56  | 0.354  | 3.91   | 0.385   | 0.413 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.625   | 41.80%   | 12.55  | 0.338  | 3.77   | 0.368   | 0.398 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.610   | 49.61%   | 13.27  | 0.351  | 3.83   | 0.367   | 0.396 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.539   | 48.83%   | 13.84  | 0.354  | 3.92   | 0.365   | 0.403 \n",
      "\n",
      "\n",
      "[Sobol Run 154/768] H_Inertia: 0.7220 | BASE_S: 0.1555 | Tau: 0.0014 | Jitter: 0.47\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.226   | 28.91%   | 7.84   | 0.636  | 4.34   | 0.576   | 0.553 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.074   | 25.78%   | 8.36   | 0.548  | 4.01   | 0.432   | 0.327 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.016   | 39.84%   | 10.16  | 0.501  | 4.00   | 0.430   | 0.276 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.887   | 38.67%   | 11.28  | 0.405  | 4.26   | 0.380   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.808   | 42.19%   | 11.96  | 0.383  | 4.11   | 0.390   | 0.385 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.718   | 51.95%   | 12.91  | 0.372  | 4.24   | 0.399   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.587   | 48.83%   | 13.36  | 0.355  | 4.15   | 0.385   | 0.401 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.539   | 50.39%   | 12.20  | 0.374  | 4.14   | 0.399   | 0.393 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.546   | 52.34%   | 12.67  | 0.344  | 4.06   | 0.414   | 0.386 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.435   | 53.91%   | 12.62  | 0.356  | 4.04   | 0.393   | 0.379 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.389   | 56.25%   | 12.63  | 0.360  | 4.13   | 0.394   | 0.386 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.339   | 55.47%   | 12.94  | 0.354  | 4.23   | 0.385   | 0.380 \n",
      "\n",
      "\n",
      "[Sobol Run 155/768] H_Inertia: 0.7220 | BASE_S: 0.1555 | Tau: 0.0014 | Jitter: 1.12\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.216   | 23.05%   | 6.49   | 0.738  | 4.60   | 0.647   | 0.691 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.081   | 30.86%   | 9.36   | 0.551  | 4.35   | 0.444   | 0.469 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.977   | 38.67%   | 9.50   | 0.521  | 4.13   | 0.506   | 0.449 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.879   | 41.02%   | 10.39  | 0.473  | 4.09   | 0.448   | 0.460 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.774   | 45.70%   | 11.59  | 0.437  | 4.17   | 0.398   | 0.350 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.710   | 45.31%   | 12.15  | 0.401  | 4.44   | 0.399   | 0.371 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.625   | 48.83%   | 12.15  | 0.425  | 4.38   | 0.388   | 0.356 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.580   | 48.83%   | 12.43  | 0.407  | 4.41   | 0.388   | 0.357 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.556   | 48.44%   | 12.64  | 0.382  | 4.26   | 0.397   | 0.345 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.511   | 51.95%   | 12.25  | 0.393  | 4.31   | 0.384   | 0.354 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.498   | 51.95%   | 12.10  | 0.395  | 4.31   | 0.376   | 0.320 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.467   | 53.52%   | 12.16  | 0.388  | 4.38   | 0.378   | 0.342 \n",
      "\n",
      "\n",
      "[Sobol Run 156/768] H_Inertia: 0.8292 | BASE_S: 0.0177 | Tau: 0.0420 | Jitter: 1.02\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.225   | 26.56%   | 8.12   | 0.625  | 4.68   | 0.501   | 0.655 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.003   | 30.08%   | 9.08   | 0.566  | 4.37   | 0.511   | 0.406 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.905   | 37.11%   | 10.57  | 0.507  | 4.25   | 0.470   | 0.321 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.785   | 35.94%   | 9.73   | 0.449  | 4.35   | 0.426   | 0.471 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.692   | 41.02%   | 11.97  | 0.433  | 4.46   | 0.433   | 0.338 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.618   | 41.02%   | 11.89  | 0.407  | 4.28   | 0.441   | 0.379 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.552   | 43.75%   | 12.13  | 0.415  | 4.44   | 0.438   | 0.397 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.530   | 41.80%   | 12.31  | 0.409  | 4.46   | 0.434   | 0.348 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.531   | 44.14%   | 12.45  | 0.409  | 4.56   | 0.447   | 0.390 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.521   | 43.36%   | 12.51  | 0.406  | 4.51   | 0.443   | 0.415 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.493   | 46.48%   | 12.36  | 0.405  | 4.60   | 0.423   | 0.339 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.460   | 46.09%   | 12.81  | 0.397  | 4.35   | 0.418   | 0.376 \n",
      "\n",
      "\n",
      "[Sobol Run 157/768] H_Inertia: 0.8292 | BASE_S: 0.0177 | Tau: 0.0259 | Jitter: 1.02\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.242   | 20.31%   | 5.53   | 0.690  | 4.64   | 0.511   | 0.585 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.038   | 34.38%   | 9.55   | 0.576  | 4.47   | 0.522   | 0.424 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.872   | 35.16%   | 9.55   | 0.509  | 4.04   | 0.435   | 0.448 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.799   | 39.45%   | 10.76  | 0.462  | 4.21   | 0.414   | 0.422 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.691   | 43.75%   | 10.84  | 0.440  | 4.23   | 0.401   | 0.459 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.599   | 45.70%   | 11.01  | 0.426  | 4.30   | 0.393   | 0.489 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.540   | 43.75%   | 12.34  | 0.409  | 4.38   | 0.406   | 0.423 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.505   | 44.92%   | 11.59  | 0.415  | 4.46   | 0.401   | 0.431 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.505   | 46.88%   | 10.41  | 0.431  | 4.50   | 0.401   | 0.474 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.494   | 44.14%   | 12.06  | 0.416  | 4.51   | 0.413   | 0.411 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.498   | 45.31%   | 12.59  | 0.400  | 4.55   | 0.413   | 0.392 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.439   | 49.22%   | 11.50  | 0.408  | 4.53   | 0.397   | 0.421 \n",
      "\n",
      "\n",
      "[Sobol Run 158/768] H_Inertia: 0.2320 | BASE_S: 0.0177 | Tau: 0.0420 | Jitter: 1.02\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.217   | 29.69%   | 11.68  | 0.487  | 4.51   | 0.432   | 0.572 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.909   | 33.20%   | 13.01  | 0.415  | 4.30   | 0.359   | 0.466 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.784   | 38.28%   | 13.19  | 0.379  | 4.45   | 0.360   | 0.427 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.712   | 42.58%   | 13.32  | 0.374  | 4.35   | 0.358   | 0.455 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.624   | 42.58%   | 13.83  | 0.369  | 4.43   | 0.340   | 0.426 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.590   | 44.92%   | 13.36  | 0.358  | 4.47   | 0.337   | 0.418 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.536   | 46.09%   | 13.06  | 0.366  | 4.20   | 0.331   | 0.441 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.480   | 51.17%   | 13.61  | 0.358  | 4.33   | 0.320   | 0.430 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.458   | 46.88%   | 12.93  | 0.340  | 4.27   | 0.313   | 0.399 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.394   | 50.78%   | 14.11  | 0.328  | 4.43   | 0.302   | 0.394 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.362   | 50.00%   | 13.05  | 0.331  | 4.42   | 0.323   | 0.417 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.366   | 51.56%   | 12.69  | 0.323  | 4.29   | 0.303   | 0.401 \n",
      "\n",
      "\n",
      "[Sobol Run 159/768] H_Inertia: 0.8292 | BASE_S: 0.6455 | Tau: 0.0420 | Jitter: 1.02\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.043   | 13.67%   | 9.71   | 0.519  | 4.93   | 0.544   | 0.376 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.100   | 18.36%   | 8.99   | 0.540  | 4.93   | 0.463   | 0.492 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.021   | 28.12%   | 10.74  | 0.543  | 4.95   | 0.457   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.876   | 38.28%   | 11.81  | 0.471  | 4.72   | 0.432   | 0.546 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.791   | 37.11%   | 12.35  | 0.452  | 4.91   | 0.378   | 0.479 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.772   | 42.58%   | 12.14  | 0.484  | 4.80   | 0.401   | 0.422 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.680   | 36.72%   | 12.69  | 0.457  | 4.52   | 0.411   | 0.381 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.723   | 42.97%   | 13.04  | 0.403  | 4.42   | 0.432   | 0.386 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.658   | 46.88%   | 13.41  | 0.402  | 4.62   | 0.426   | 0.343 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.605   | 45.70%   | 14.02  | 0.406  | 4.31   | 0.381   | 0.290 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.542   | 49.22%   | 13.76  | 0.379  | 4.54   | 0.375   | 0.324 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.506   | 46.09%   | 14.07  | 0.408  | 4.67   | 0.437   | 0.421 \n",
      "\n",
      "\n",
      "[Sobol Run 160/768] H_Inertia: 0.8292 | BASE_S: 0.0177 | Tau: 0.0420 | Jitter: 1.02\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.226   | 21.48%   | 5.69   | 0.727  | 4.50   | 0.585   | 0.683 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.057   | 27.73%   | 9.22   | 0.528  | 4.33   | 0.464   | 0.476 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.920   | 30.86%   | 9.68   | 0.550  | 4.14   | 0.486   | 0.316 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.810   | 37.50%   | 11.35  | 0.458  | 4.25   | 0.455   | 0.343 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.678   | 41.41%   | 12.00  | 0.433  | 4.41   | 0.448   | 0.358 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.620   | 42.19%   | 11.99  | 0.432  | 4.36   | 0.435   | 0.356 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.558   | 44.14%   | 11.99  | 0.428  | 4.39   | 0.428   | 0.330 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.571   | 41.02%   | 12.30  | 0.454  | 4.53   | 0.465   | 0.347 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.538   | 43.36%   | 12.42  | 0.422  | 4.32   | 0.447   | 0.353 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.478   | 46.88%   | 12.36  | 0.423  | 4.28   | 0.416   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.484   | 46.88%   | 12.17  | 0.418  | 4.32   | 0.401   | 0.320 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.455   | 46.09%   | 12.46  | 0.418  | 4.34   | 0.389   | 0.321 \n",
      "\n",
      "\n",
      "[Sobol Run 161/768] H_Inertia: 0.8292 | BASE_S: 0.0177 | Tau: 0.0420 | Jitter: 0.57\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.235   | 22.66%   | 6.40   | 0.691  | 4.43   | 0.518   | 0.355 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.049   | 30.86%   | 9.10   | 0.556  | 4.38   | 0.464   | 0.381 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.906   | 32.42%   | 9.65   | 0.515  | 4.31   | 0.461   | 0.390 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.769   | 40.62%   | 11.29  | 0.436  | 4.28   | 0.422   | 0.375 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.674   | 41.41%   | 11.34  | 0.416  | 4.37   | 0.415   | 0.444 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.593   | 42.97%   | 12.10  | 0.404  | 4.40   | 0.436   | 0.361 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.547   | 41.41%   | 12.33  | 0.414  | 4.55   | 0.443   | 0.366 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.532   | 42.97%   | 11.65  | 0.410  | 4.39   | 0.423   | 0.369 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.510   | 44.53%   | 11.92  | 0.405  | 4.29   | 0.418   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.459   | 44.53%   | 12.38  | 0.403  | 4.41   | 0.414   | 0.340 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.454   | 44.14%   | 11.98  | 0.402  | 4.37   | 0.425   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.444   | 46.09%   | 12.57  | 0.393  | 4.53   | 0.417   | 0.340 \n",
      "\n",
      "\n",
      "[Sobol Run 162/768] H_Inertia: 0.2320 | BASE_S: 0.6455 | Tau: 0.0420 | Jitter: 0.57\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.139   | 9.77%    | 12.74  | 0.383  | 4.00   | 0.286   | 0.364 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.240   | 19.53%   | 7.82   | 0.700  | 4.50   | 0.411   | 0.792 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.206   | 25.78%   | 9.02   | 0.709  | 4.40   | 0.391   | 0.820 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.097   | 32.42%   | 11.88  | 0.560  | 4.98   | 0.352   | 0.698 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.912   | 37.89%   | 12.82  | 0.504  | 4.53   | 0.321   | 0.634 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.788   | 35.94%   | 12.18  | 0.429  | 4.79   | 0.307   | 0.230 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.764   | 41.41%   | 12.35  | 0.473  | 4.69   | 0.286   | 0.481 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.666   | 41.41%   | 14.02  | 0.426  | 4.79   | 0.281   | 0.526 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.642   | 43.75%   | 13.51  | 0.431  | 4.72   | 0.271   | 0.476 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.584   | 48.05%   | 13.75  | 0.409  | 4.46   | 0.270   | 0.334 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.493   | 47.66%   | 13.59  | 0.424  | 4.68   | 0.283   | 0.408 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.440   | 50.00%   | 13.43  | 0.422  | 4.51   | 0.271   | 0.371 \n",
      "\n",
      "\n",
      "[Sobol Run 163/768] H_Inertia: 0.8292 | BASE_S: 0.6455 | Tau: 0.0259 | Jitter: 0.57\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.198   | 23.05%   | 7.53   | 0.643  | 4.31   | 0.392   | 0.315 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.971   | 30.86%   | 6.89   | 0.565  | 4.17   | 0.428   | 0.348 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.828   | 34.77%   | 6.78   | 0.540  | 4.19   | 0.424   | 0.414 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.739   | 34.77%   | 7.29   | 0.515  | 4.22   | 0.419   | 0.483 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.726   | 35.16%   | 7.29   | 0.498  | 4.31   | 0.413   | 0.572 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.675   | 37.11%   | 7.67   | 0.501  | 4.28   | 0.405   | 0.498 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.606   | 40.62%   | 8.91   | 0.483  | 4.43   | 0.370   | 0.574 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.518   | 46.88%   | 8.29   | 0.451  | 4.33   | 0.392   | 0.555 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.508   | 50.00%   | 9.40   | 0.434  | 4.40   | 0.374   | 0.545 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.407   | 49.61%   | 8.57   | 0.461  | 4.61   | 0.372   | 0.602 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.368   | 53.12%   | 9.17   | 0.459  | 4.56   | 0.359   | 0.575 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.344   | 52.34%   | 8.85   | 0.445  | 4.47   | 0.341   | 0.550 \n",
      "\n",
      "\n",
      "[Sobol Run 164/768] H_Inertia: 0.2320 | BASE_S: 0.0177 | Tau: 0.0259 | Jitter: 0.57\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.218   | 23.05%   | 8.50   | 0.573  | 4.53   | 0.465   | 0.598 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.014   | 27.73%   | 11.93  | 0.427  | 4.22   | 0.405   | 0.497 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.860   | 33.20%   | 11.64  | 0.416  | 4.04   | 0.373   | 0.503 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.734   | 38.67%   | 12.54  | 0.368  | 4.15   | 0.369   | 0.480 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.653   | 42.19%   | 13.38  | 0.348  | 4.05   | 0.332   | 0.438 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.580   | 45.70%   | 13.42  | 0.341  | 4.00   | 0.327   | 0.413 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.518   | 47.27%   | 14.16  | 0.330  | 4.15   | 0.317   | 0.385 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.467   | 51.17%   | 14.66  | 0.313  | 4.18   | 0.312   | 0.363 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.426   | 51.17%   | 13.89  | 0.327  | 4.14   | 0.304   | 0.347 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.406   | 48.83%   | 13.91  | 0.318  | 4.13   | 0.298   | 0.354 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.345   | 53.52%   | 14.43  | 0.307  | 4.18   | 0.297   | 0.377 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.326   | 53.91%   | 14.27  | 0.299  | 4.26   | 0.298   | 0.377 \n",
      "\n",
      "\n",
      "[Sobol Run 165/768] H_Inertia: 0.2320 | BASE_S: 0.6455 | Tau: 0.0259 | Jitter: 0.57\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 4.091   | 15.23%   | 13.57  | 0.383  | 5.07   | 0.436   | 0.324 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.103   | 26.95%   | 11.45  | 0.556  | 4.94   | 0.541   | 0.376 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.936   | 32.03%   | 12.83  | 0.463  | 4.92   | 0.435   | 0.282 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.819   | 33.98%   | 13.55  | 0.446  | 4.84   | 0.536   | 0.327 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.776   | 35.55%   | 13.38  | 0.431  | 4.70   | 0.448   | 0.289 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.739   | 41.02%   | 14.81  | 0.409  | 4.68   | 0.522   | 0.346 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.744   | 39.06%   | 14.97  | 0.365  | 4.66   | 0.376   | 0.416 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.678   | 41.02%   | 14.54  | 0.406  | 4.65   | 0.416   | 0.324 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.661   | 41.80%   | 14.61  | 0.389  | 4.52   | 0.429   | 0.347 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.589   | 48.83%   | 14.99  | 0.386  | 4.45   | 0.442   | 0.276 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.546   | 49.61%   | 15.05  | 0.390  | 4.46   | 0.483   | 0.261 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.885   | 35.16%   | 15.58  | 0.270  | 3.42   | 0.329   | 0.472 \n",
      "\n",
      "\n",
      "[Sobol Run 166/768] H_Inertia: 0.2320 | BASE_S: 0.6455 | Tau: 0.0259 | Jitter: 1.02\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 166 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 167/768] H_Inertia: 0.2320 | BASE_S: 0.6455 | Tau: 0.0259 | Jitter: 0.57\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.671   | 19.14%   | 13.41  | 0.445  | 4.87   | 0.386   | 0.369 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.251   | 11.72%   | 15.96  | 0.333  | 3.48   | 0.355   | 0.256 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.238   | 23.05%   | 10.91  | 0.630  | 4.43   | 0.366   | 0.726 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.068   | 26.56%   | 11.70  | 0.535  | 4.55   | 0.362   | 0.595 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.988   | 33.59%   | 13.26  | 0.493  | 4.63   | 0.347   | 0.582 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.878   | 39.45%   | 14.04  | 0.459  | 4.34   | 0.336   | 0.555 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.804   | 41.80%   | 13.36  | 0.470  | 4.31   | 0.344   | 0.606 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.754   | 41.80%   | 15.00  | 0.430  | 4.55   | 0.330   | 0.512 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.682   | 44.53%   | 16.14  | 0.393  | 4.52   | 0.328   | 0.452 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.610   | 45.70%   | 15.76  | 0.409  | 4.44   | 0.317   | 0.516 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.539   | 44.92%   | 15.68  | 0.405  | 4.45   | 0.322   | 0.476 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.498   | 44.92%   | 16.01  | 0.392  | 4.30   | 0.318   | 0.386 \n",
      "\n",
      "\n",
      "[Sobol Run 168/768] H_Inertia: 0.0942 | BASE_S: 0.7527 | Tau: 0.0297 | Jitter: 0.74\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                  \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 168 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 169/768] H_Inertia: 0.0942 | BASE_S: 0.7527 | Tau: 0.0381 | Jitter: 0.74\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                  \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 169 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 170/768] H_Inertia: 0.9670 | BASE_S: 0.7527 | Tau: 0.0297 | Jitter: 0.74\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.208   | 21.88%   | 4.87   | 0.841  | 4.92   | 0.708   | 0.692 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.079   | 22.27%   | 5.66   | 0.727  | 4.92   | 0.628   | 0.686 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.034   | 25.00%   | 5.88   | 0.716  | 4.67   | 0.586   | 0.486 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.996   | 28.12%   | 7.32   | 0.650  | 4.47   | 0.489   | 0.357 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.940   | 29.30%   | 8.78   | 0.576  | 4.61   | 0.489   | 0.472 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.873   | 32.03%   | 9.07   | 0.510  | 4.51   | 0.501   | 0.423 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.843   | 32.42%   | 8.44   | 0.521  | 4.39   | 0.487   | 0.252 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.790   | 35.55%   | 9.62   | 0.487  | 4.43   | 0.481   | 0.257 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.752   | 34.77%   | 9.91   | 0.477  | 4.36   | 0.464   | 0.216 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.659   | 37.11%   | 10.44  | 0.467  | 4.50   | 0.474   | 0.320 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.624   | 37.89%   | 9.83   | 0.476  | 4.40   | 0.484   | 0.230 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.595   | 38.28%   | 9.87   | 0.470  | 4.48   | 0.469   | 0.237 \n",
      "\n",
      "\n",
      "[Sobol Run 171/768] H_Inertia: 0.0942 | BASE_S: 0.8905 | Tau: 0.0297 | Jitter: 0.74\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                   \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 171 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 172/768] H_Inertia: 0.0942 | BASE_S: 0.7527 | Tau: 0.0297 | Jitter: 0.74\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                  \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "!! Run 172 failed: array must not contain infs or NaNs\n",
      "\n",
      "[Sobol Run 173/768] H_Inertia: 0.0942 | BASE_S: 0.7527 | Tau: 0.0297 | Jitter: 0.30\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.224   | 29.69%   | 11.49  | 0.491  | 4.64   | 0.356   | 0.317 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.023   | 28.12%   | 10.80  | 0.466  | 4.47   | 0.324   | 0.458 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.892   | 37.50%   | 13.43  | 0.407  | 4.60   | 0.328   | 0.333 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.749   | 33.59%   | 11.85  | 0.404  | 4.40   | 0.314   | 0.413 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.685   | 47.66%   | 15.13  | 0.341  | 4.47   | 0.306   | 0.297 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.538   | 48.83%   | 14.69  | 0.333  | 4.60   | 0.296   | 0.285 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.443   | 53.12%   | 14.68  | 0.337  | 4.69   | 0.272   | 0.317 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.390   | 50.39%   | 13.55  | 0.326  | 4.52   | 0.281   | 0.365 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.330   | 53.52%   | 13.66  | 0.345  | 4.47   | 0.278   | 0.307 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.278   | 56.25%   | 13.50  | 0.344  | 4.40   | 0.290   | 0.333 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.198   | 53.91%   | 14.31  | 0.304  | 4.54   | 0.286   | 0.307 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.172   | 51.95%   | 13.26  | 0.296  | 4.39   | 0.287   | 0.315 \n",
      "\n",
      "\n",
      "[Sobol Run 174/768] H_Inertia: 0.9670 | BASE_S: 0.8905 | Tau: 0.0297 | Jitter: 0.30\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.238   | 20.31%   | 5.44   | 0.702  | 5.00   | 0.476   | 0.306 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.051   | 26.17%   | 7.70   | 0.654  | 4.97   | 0.492   | 0.655 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.947   | 29.30%   | 8.21   | 0.576  | 4.97   | 0.507   | 0.553 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.902   | 31.64%   | 8.45   | 0.570  | 4.97   | 0.505   | 0.474 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.825   | 34.38%   | 8.12   | 0.554  | 4.84   | 0.501   | 0.439 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.795   | 31.64%   | 8.19   | 0.526  | 4.77   | 0.454   | 0.324 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.750   | 39.06%   | 8.48   | 0.493  | 4.87   | 0.434   | 0.419 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.740   | 34.77%   | 8.09   | 0.540  | 4.74   | 0.458   | 0.330 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.702   | 41.80%   | 8.81   | 0.517  | 4.78   | 0.474   | 0.399 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.671   | 39.06%   | 8.34   | 0.500  | 4.72   | 0.447   | 0.421 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.665   | 44.14%   | 8.97   | 0.485  | 4.85   | 0.477   | 0.414 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.619   | 36.72%   | 9.64   | 0.504  | 4.74   | 0.503   | 0.353 \n",
      "\n",
      "\n",
      "[Sobol Run 175/768] H_Inertia: 0.0942 | BASE_S: 0.8905 | Tau: 0.0381 | Jitter: 0.30\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 3.345   | 8.20%    | 4.70   | 0.516  | 3.86   | 0.285   | 0.294 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.481   | 9.77%    | 3.66   | 0.638  | 4.16   | 0.304   | 0.523 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.309   | 19.53%   | 8.05   | 0.381  | 4.25   | 0.336   | 0.291 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.130   | 24.22%   | 7.60   | 0.371  | 4.52   | 0.318   | 0.332 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 2.128   | 26.56%   | 8.40   | 0.342  | 4.52   | 0.357   | 0.229 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.979   | 32.03%   | 11.67  | 0.296  | 4.37   | 0.310   | 0.214 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.809   | 41.41%   | 12.07  | 0.294  | 4.20   | 0.300   | 0.335 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.647   | 40.23%   | 12.83  | 0.270  | 4.12   | 0.276   | 0.294 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.610   | 44.14%   | 13.31  | 0.267  | 4.09   | 0.283   | 0.278 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.544   | 41.41%   | 13.22  | 0.274  | 4.20   | 0.289   | 0.256 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.533   | 42.97%   | 13.64  | 0.276  | 4.06   | 0.284   | 0.317 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.514   | 45.31%   | 13.96  | 0.259  | 4.11   | 0.275   | 0.181 \n",
      "\n",
      "\n",
      "[Sobol Run 176/768] H_Inertia: 0.9670 | BASE_S: 0.7527 | Tau: 0.0381 | Jitter: 0.30\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.194   | 21.09%   | 5.47   | 0.771  | 4.79   | 0.498   | 0.678 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.023   | 23.83%   | 5.66   | 0.714  | 4.70   | 0.434   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.952   | 26.95%   | 6.25   | 0.684  | 4.72   | 0.448   | 0.504 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.914   | 30.47%   | 6.69   | 0.661  | 4.70   | 0.471   | 0.372 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.882   | 33.20%   | 7.33   | 0.585  | 4.64   | 0.409   | 0.260 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.823   | 33.20%   | 7.75   | 0.578  | 4.55   | 0.396   | 0.239 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.756   | 32.81%   | 7.45   | 0.504  | 4.53   | 0.415   | 0.418 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.693   | 39.06%   | 9.01   | 0.495  | 4.54   | 0.392   | 0.401 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.665   | 33.59%   | 7.75   | 0.470  | 4.69   | 0.424   | 0.440 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.661   | 37.50%   | 8.51   | 0.487  | 4.65   | 0.428   | 0.450 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.632   | 42.97%   | 8.33   | 0.456  | 4.64   | 0.421   | 0.444 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.612   | 37.11%   | 7.17   | 0.475  | 4.63   | 0.450   | 0.458 \n",
      "\n",
      "\n",
      "[Sobol Run 177/768] H_Inertia: 0.9670 | BASE_S: 0.8905 | Tau: 0.0381 | Jitter: 0.30\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.224   | 21.09%   | 3.97   | 0.754  | 4.47   | 0.746   | 0.551 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.085   | 26.17%   | 5.88   | 0.655  | 4.44   | 0.595   | 0.643 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.040   | 24.22%   | 5.46   | 0.658  | 4.29   | 0.565   | 0.534 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.013   | 26.17%   | 7.40   | 0.567  | 4.46   | 0.511   | 0.591 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.921   | 34.38%   | 7.64   | 0.568  | 4.27   | 0.454   | 0.361 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.838   | 35.94%   | 8.45   | 0.514  | 4.25   | 0.445   | 0.328 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.786   | 28.91%   | 8.63   | 0.486  | 4.61   | 0.446   | 0.494 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.776   | 35.55%   | 9.22   | 0.477  | 4.25   | 0.436   | 0.315 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.735   | 33.98%   | 8.83   | 0.456  | 4.26   | 0.421   | 0.375 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.690   | 32.81%   | 8.72   | 0.480  | 4.21   | 0.441   | 0.348 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.662   | 37.50%   | 9.31   | 0.465  | 4.22   | 0.435   | 0.340 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.626   | 41.02%   | 9.84   | 0.444  | 4.30   | 0.444   | 0.283 \n",
      "\n",
      "\n",
      "[Sobol Run 178/768] H_Inertia: 0.9670 | BASE_S: 0.8905 | Tau: 0.0381 | Jitter: 0.74\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.197   | 25.00%   | 7.21   | 0.642  | 4.70   | 0.520   | 0.555 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.035   | 28.12%   | 7.71   | 0.650  | 4.65   | 0.511   | 0.595 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.959   | 25.39%   | 7.09   | 0.636  | 4.80   | 0.464   | 0.436 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.890   | 28.52%   | 7.97   | 0.556  | 4.67   | 0.440   | 0.367 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.836   | 35.55%   | 8.39   | 0.522  | 4.30   | 0.443   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.795   | 35.16%   | 9.81   | 0.542  | 4.55   | 0.479   | 0.322 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.758   | 35.16%   | 7.71   | 0.489  | 4.35   | 0.430   | 0.471 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.725   | 32.81%   | 7.32   | 0.433  | 4.17   | 0.389   | 0.570 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.701   | 35.55%   | 8.48   | 0.468  | 4.30   | 0.407   | 0.433 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.655   | 36.72%   | 8.88   | 0.479  | 4.21   | 0.422   | 0.408 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.638   | 39.45%   | 8.39   | 0.461  | 4.00   | 0.416   | 0.449 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.605   | 37.11%   | 8.68   | 0.480  | 4.19   | 0.426   | 0.423 \n",
      "\n",
      "\n",
      "[Sobol Run 179/768] H_Inertia: 0.9670 | BASE_S: 0.8905 | Tau: 0.0381 | Jitter: 0.30\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.208   | 22.27%   | 7.18   | 0.643  | 5.08   | 0.527   | 0.720 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.087   | 26.56%   | 6.91   | 0.711  | 4.86   | 0.469   | 0.651 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.007   | 29.69%   | 7.77   | 0.618  | 4.89   | 0.504   | 0.608 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.928   | 34.77%   | 7.82   | 0.603  | 4.70   | 0.498   | 0.326 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.835   | 38.28%   | 8.53   | 0.543  | 4.43   | 0.478   | 0.302 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.752   | 34.77%   | 8.69   | 0.505  | 4.44   | 0.416   | 0.361 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.691   | 40.62%   | 9.24   | 0.484  | 4.16   | 0.426   | 0.371 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.688   | 39.84%   | 8.69   | 0.484  | 4.29   | 0.413   | 0.404 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.642   | 40.23%   | 9.11   | 0.479  | 4.25   | 0.432   | 0.383 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.614   | 39.45%   | 8.58   | 0.471  | 4.25   | 0.430   | 0.441 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.598   | 41.02%   | 9.19   | 0.470  | 4.24   | 0.434   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.575   | 39.84%   | 9.97   | 0.435  | 4.09   | 0.429   | 0.448 \n",
      "\n",
      "\n",
      "[Sobol Run 180/768] H_Inertia: 0.5842 | BASE_S: 0.2627 | Tau: 0.0052 | Jitter: 0.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.206   | 30.08%   | 9.29   | 0.668  | 3.85   | 0.505   | 0.736 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.980   | 37.11%   | 10.14  | 0.520  | 3.84   | 0.458   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.867   | 36.33%   | 10.93  | 0.441  | 3.44   | 0.386   | 0.412 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.809   | 41.41%   | 11.17  | 0.399  | 3.25   | 0.394   | 0.390 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.758   | 42.58%   | 11.59  | 0.393  | 3.24   | 0.434   | 0.408 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.743   | 46.09%   | 11.54  | 0.384  | 3.53   | 0.351   | 0.419 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.664   | 43.36%   | 12.05  | 0.357  | 3.67   | 0.346   | 0.415 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.681   | 46.88%   | 11.74  | 0.364  | 3.32   | 0.353   | 0.421 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.673   | 44.53%   | 10.64  | 0.380  | 3.36   | 0.333   | 0.462 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.638   | 46.88%   | 11.20  | 0.384  | 3.52   | 0.348   | 0.357 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.598   | 50.00%   | 10.53  | 0.387  | 3.37   | 0.340   | 0.402 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.559   | 48.05%   | 11.21  | 0.358  | 3.43   | 0.325   | 0.413 \n",
      "\n",
      "\n",
      "[Sobol Run 181/768] H_Inertia: 0.5842 | BASE_S: 0.2627 | Tau: 0.0136 | Jitter: 0.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.195   | 32.03%   | 9.48   | 0.592  | 3.57   | 0.498   | 0.704 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.991   | 31.25%   | 9.19   | 0.503  | 3.66   | 0.464   | 0.580 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.895   | 41.02%   | 12.08  | 0.389  | 4.03   | 0.447   | 0.351 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.837   | 42.19%   | 12.53  | 0.391  | 3.93   | 0.463   | 0.340 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.774   | 45.70%   | 11.84  | 0.400  | 3.94   | 0.427   | 0.322 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.702   | 45.70%   | 12.48  | 0.392  | 4.07   | 0.439   | 0.306 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.667   | 42.19%   | 12.34  | 0.404  | 4.37   | 0.436   | 0.288 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.664   | 46.48%   | 12.59  | 0.384  | 4.19   | 0.443   | 0.300 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.628   | 46.48%   | 13.23  | 0.360  | 4.21   | 0.416   | 0.333 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.633   | 44.92%   | 11.98  | 0.387  | 4.22   | 0.413   | 0.257 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.597   | 44.14%   | 12.58  | 0.364  | 4.20   | 0.348   | 0.369 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.568   | 47.27%   | 12.48  | 0.361  | 4.19   | 0.353   | 0.337 \n",
      "\n",
      "\n",
      "[Sobol Run 182/768] H_Inertia: 0.4770 | BASE_S: 0.2627 | Tau: 0.0052 | Jitter: 0.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.171   | 32.42%   | 9.05   | 0.596  | 4.01   | 0.474   | 0.664 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.984   | 33.20%   | 10.67  | 0.497  | 4.07   | 0.469   | 0.397 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.891   | 33.20%   | 11.02  | 0.460  | 4.05   | 0.438   | 0.348 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.830   | 37.89%   | 11.09  | 0.426  | 4.29   | 0.488   | 0.274 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.749   | 42.97%   | 12.63  | 0.409  | 4.16   | 0.434   | 0.314 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.780   | 45.31%   | 12.06  | 0.381  | 4.17   | 0.414   | 0.314 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.714   | 48.44%   | 12.20  | 0.396  | 4.19   | 0.397   | 0.328 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.638   | 43.75%   | 12.15  | 0.416  | 4.17   | 0.406   | 0.321 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.615   | 47.27%   | 12.41  | 0.374  | 4.31   | 0.404   | 0.297 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.601   | 50.78%   | 12.82  | 0.374  | 4.08   | 0.369   | 0.307 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.532   | 46.48%   | 13.53  | 0.365  | 4.46   | 0.385   | 0.318 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.485   | 52.34%   | 13.41  | 0.353  | 4.37   | 0.365   | 0.280 \n",
      "\n",
      "\n",
      "[Sobol Run 183/768] H_Inertia: 0.5842 | BASE_S: 0.4005 | Tau: 0.0052 | Jitter: 0.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.199   | 26.56%   | 9.16   | 0.566  | 3.65   | 0.471   | 0.436 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.029   | 32.03%   | 6.87   | 0.553  | 2.95   | 0.505   | 0.585 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.940   | 35.55%   | 8.44   | 0.551  | 3.25   | 0.520   | 0.545 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.910   | 35.55%   | 7.79   | 0.530  | 2.82   | 0.524   | 0.574 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.862   | 40.23%   | 8.12   | 0.508  | 2.48   | 0.512   | 0.598 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.820   | 38.28%   | 8.80   | 0.460  | 2.57   | 0.497   | 0.559 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.760   | 43.36%   | 9.34   | 0.454  | 2.56   | 0.489   | 0.549 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.730   | 40.62%   | 10.00  | 0.445  | 2.71   | 0.462   | 0.478 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.699   | 42.97%   | 10.28  | 0.422  | 2.67   | 0.457   | 0.504 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.659   | 44.14%   | 11.54  | 0.398  | 3.01   | 0.425   | 0.453 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.623   | 44.14%   | 11.49  | 0.386  | 3.03   | 0.431   | 0.444 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.608   | 43.75%   | 12.32  | 0.349  | 3.04   | 0.436   | 0.422 \n",
      "\n",
      "\n",
      "[Sobol Run 184/768] H_Inertia: 0.5842 | BASE_S: 0.2627 | Tau: 0.0052 | Jitter: 0.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.191   | 34.38%   | 8.46   | 0.574  | 4.00   | 0.588   | 0.442 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.034   | 31.64%   | 8.55   | 0.572  | 3.60   | 0.580   | 0.550 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.926   | 35.94%   | 10.67  | 0.495  | 3.82   | 0.583   | 0.382 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.861   | 41.02%   | 11.23  | 0.413  | 3.69   | 0.499   | 0.344 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.806   | 40.62%   | 11.07  | 0.403  | 3.76   | 0.529   | 0.379 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.752   | 39.84%   | 11.21  | 0.427  | 3.93   | 0.499   | 0.359 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.695   | 44.53%   | 12.66  | 0.363  | 3.91   | 0.466   | 0.334 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.668   | 44.14%   | 12.33  | 0.390  | 4.05   | 0.466   | 0.299 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.659   | 47.66%   | 12.90  | 0.364  | 4.15   | 0.438   | 0.310 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.613   | 47.66%   | 12.92  | 0.371  | 4.06   | 0.422   | 0.288 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.572   | 44.53%   | 12.65  | 0.376  | 4.10   | 0.434   | 0.281 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.545   | 42.58%   | 12.32  | 0.384  | 4.01   | 0.430   | 0.301 \n",
      "\n",
      "\n",
      "[Sobol Run 185/768] H_Inertia: 0.5842 | BASE_S: 0.2627 | Tau: 0.0052 | Jitter: 0.85\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.187   | 30.08%   | 7.73   | 0.574  | 4.65   | 0.555   | 0.536 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.988   | 35.55%   | 8.20   | 0.595  | 4.12   | 0.584   | 0.343 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.918   | 35.55%   | 8.96   | 0.588  | 4.23   | 0.553   | 0.373 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.860   | 38.28%   | 9.09   | 0.542  | 4.14   | 0.544   | 0.369 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.841   | 41.02%   | 10.34  | 0.491  | 4.23   | 0.506   | 0.284 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 2.029   | 43.36%   | 7.06   | 0.434  | 2.91   | 0.435   | 0.536 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.728   | 42.58%   | 10.71  | 0.465  | 3.86   | 0.448   | 0.253 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.680   | 42.97%   | 11.04  | 0.435  | 4.34   | 0.446   | 0.293 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.625   | 42.19%   | 12.10  | 0.409  | 4.68   | 0.429   | 0.294 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.634   | 40.62%   | 11.18  | 0.410  | 4.36   | 0.428   | 0.324 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.598   | 46.88%   | 11.41  | 0.428  | 4.56   | 0.447   | 0.255 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.559   | 48.83%   | 11.65  | 0.416  | 4.28   | 0.441   | 0.276 \n",
      "\n",
      "\n",
      "[Sobol Run 186/768] H_Inertia: 0.4770 | BASE_S: 0.4005 | Tau: 0.0052 | Jitter: 0.85\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.189   | 27.73%   | 6.83   | 0.677  | 3.65   | 0.399   | 0.734 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.069   | 27.73%   | 10.06  | 0.503  | 3.75   | 0.415   | 0.518 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 3.556   | 13.28%   | 3.86   | 0.408  | 3.48   | 0.298   | 0.535 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 12.009  | 7.42%    | 6.72   | nan    | 3.45   | 0.323   | 0.469 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 10.187  | 7.81%    | 1.69   | nan    | 2.71   | 0.296   | 0.473 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 6.726   | 6.25%    | 2.91   | 0.761  | 3.04   | 0.601   | 0.865 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.298   | 28.12%   | 9.80   | 0.570  | 4.84   | 0.481   | 0.502 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.000   | 30.86%   | 11.92  | 0.530  | 4.91   | 0.414   | 0.251 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.877   | 36.72%   | 14.72  | 0.367  | 4.83   | 0.342   | 0.323 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.787   | 41.02%   | 15.09  | 0.372  | 4.67   | 0.329   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 4.040   | 15.23%   | 3.00   | 0.469  | 3.22   | 0.337   | 0.638 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 4.384   | 15.23%   | 1.59   | 0.447  | 2.81   | 0.406   | 0.558 \n",
      "\n",
      "\n",
      "[Sobol Run 187/768] H_Inertia: 0.5842 | BASE_S: 0.4005 | Tau: 0.0136 | Jitter: 0.85\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.273   | 22.66%   | 6.71   | 0.766  | 4.73   | 0.624   | 0.664 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.117   | 26.56%   | 7.66   | 0.726  | 4.58   | 0.548   | 0.515 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.054   | 26.56%   | 7.89   | 0.664  | 4.13   | 0.543   | 0.379 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 2.050   | 30.47%   | 9.77   | 0.521  | 4.60   | 0.602   | 0.494 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.998   | 34.77%   | 10.20  | 0.512  | 3.75   | 0.523   | 0.329 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.868   | 35.94%   | 11.55  | 0.427  | 3.78   | 0.466   | 0.344 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.010   | 33.98%   | 12.30  | 0.479  | 4.38   | 0.441   | 0.477 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.808   | 38.67%   | 13.00  | 0.424  | 4.30   | 0.434   | 0.420 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.765   | 43.36%   | 12.81  | 0.419  | 3.82   | 0.405   | 0.379 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 9.024   | 21.09%   | 4.18   | 0.563  | 3.37   | 0.333   | 0.730 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 2.319   | 39.06%   | 12.85  | 0.420  | 4.63   | 0.420   | 0.363 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.699   | 41.80%   | 13.31  | 0.403  | 4.39   | 0.465   | 0.318 \n",
      "\n",
      "\n",
      "[Sobol Run 188/768] H_Inertia: 0.4770 | BASE_S: 0.2627 | Tau: 0.0136 | Jitter: 0.85\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.146   | 30.47%   | 9.44   | 0.577  | 4.21   | 0.533   | 0.392 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 1.936   | 38.67%   | 11.11  | 0.499  | 3.68   | 0.523   | 0.390 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.836   | 36.72%   | 11.47  | 0.471  | 3.64   | 0.472   | 0.353 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.764   | 39.84%   | 11.53  | 0.412  | 3.59   | 0.388   | 0.365 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.686   | 37.89%   | 12.21  | 0.365  | 3.53   | 0.425   | 0.395 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.690   | 43.36%   | 12.84  | 0.362  | 3.61   | 0.367   | 0.401 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.585   | 45.31%   | 12.93  | 0.350  | 3.17   | 0.421   | 0.434 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.575   | 48.44%   | 12.87  | 0.345  | 3.11   | 0.481   | 0.448 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.491   | 51.95%   | 13.65  | 0.328  | 3.27   | 0.395   | 0.412 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.451   | 51.56%   | 13.03  | 0.328  | 3.14   | 0.366   | 0.402 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.779   | 33.20%   | 5.53   | 0.368  | 3.44   | 0.337   | 0.526 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 4.717   | 50.78%   | 13.74  | 0.334  | 3.90   | 0.378   | 0.458 \n",
      "\n",
      "\n",
      "[Sobol Run 189/768] H_Inertia: 0.4770 | BASE_S: 0.4005 | Tau: 0.0136 | Jitter: 0.85\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.237   | 21.48%   | 7.65   | 0.613  | 4.15   | 0.499   | 0.648 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.081   | 28.12%   | 10.64  | 0.396  | 4.19   | 0.522   | 0.327 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 2.056   | 29.30%   | 11.23  | 0.444  | 4.11   | 0.483   | 0.271 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.937   | 30.86%   | 12.48  | 0.397  | 3.84   | 0.427   | 0.391 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.855   | 33.59%   | 12.54  | 0.407  | 3.87   | 0.409   | 0.362 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.811   | 35.55%   | 12.78  | 0.396  | 3.53   | 0.378   | 0.317 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.979   | 30.86%   | 14.55  | 0.353  | 3.61   | 0.383   | 0.299 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.933   | 34.77%   | 14.03  | 0.348  | 3.97   | 0.365   | 0.370 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.733   | 42.58%   | 13.32  | 0.329  | 2.79   | 0.368   | 0.326 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.771   | 39.06%   | 13.93  | 0.338  | 2.80   | 0.381   | 0.354 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.674   | 39.45%   | 14.62  | 0.343  | 3.18   | 0.401   | 0.358 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.605   | 46.88%   | 14.45  | 0.369  | 3.27   | 0.432   | 0.337 \n",
      "\n",
      "\n",
      "[Sobol Run 190/768] H_Inertia: 0.4770 | BASE_S: 0.4005 | Tau: 0.0136 | Jitter: 0.19\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.212   | 29.69%   | 7.99   | 0.598  | 3.62   | 0.606   | 0.739 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.030   | 40.23%   | 10.69  | 0.444  | 3.60   | 0.570   | 0.466 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.878   | 41.41%   | 11.36  | 0.443  | 3.85   | 0.541   | 0.384 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.785   | 44.53%   | 12.03  | 0.430  | 3.74   | 0.405   | 0.375 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 1.711   | 48.05%   | 12.40  | 0.379  | 3.71   | 0.372   | 0.425 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 1.682   | 45.31%   | 12.00  | 0.376  | 3.68   | 0.392   | 0.391 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 1.631   | 45.70%   | 11.37  | 0.375  | 3.57   | 0.357   | 0.488 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 1.559   | 45.70%   | 11.94  | 0.359  | 3.65   | 0.366   | 0.425 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 1.498   | 47.66%   | 12.32  | 0.350  | 3.62   | 0.362   | 0.435 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 10/12: 1.463   | 48.05%   | 12.91  | 0.323  | 3.51   | 0.363   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 11/12: 1.415   | 48.44%   | 12.46  | 0.323  | 3.63   | 0.453   | 0.389 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 12/12: 1.395   | 48.44%   | 12.33  | 0.344  | 3.80   | 0.358   | 0.410 \n",
      "\n",
      "\n",
      "[Sobol Run 191/768] H_Inertia: 0.4770 | BASE_S: 0.4005 | Tau: 0.0136 | Jitter: 0.85\n",
      "\n",
      "Epoch  | Loss    | Val-Acc% | Rank   | Sync   | Entrp  | A-Corr  | Intf  \n",
      "-------------------------------------------------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 1/12: 2.201   | 30.86%   | 8.65   | 0.551  | 3.26   | 0.573   | 0.589 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 2/12: 2.058   | 34.38%   | 9.49   | 0.491  | 3.20   | 0.632   | 0.510 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                     \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 3/12: 1.969   | 39.45%   | 10.41  | 0.476  | 3.39   | 0.601   | 0.429 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                    \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 4/12: 1.846   | 43.36%   | 11.28  | 0.420  | 3.15   | 0.553   | 0.374 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                      \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 5/12: 7.422   | 12.11%   | 1.35   | nan    | 2.90   | 0.436   | 0.649 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                          \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 6/12: 4.223   | 12.89%   | 1.52   | nan    | 2.69   | 0.355   | 0.685 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "EPOCH 7/12:   0%|          | 0/22 [00:00<?, ?it/s]2026-04-05 16:58:29.166559: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:452] ShuffleDatasetV3:19: Filling up shuffle buffer (this may take a while): 136 of 5000\n",
      "2026-04-05 16:58:29.214415: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:482] Shuffle buffer filled.\n",
      "                                                                                       \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 7/12: 2.468   | 16.80%   | 1.09   | nan    | 2.57   | 0.365   | 0.393 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                        \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 8/12: 2.276   | 18.36%   | 2.06   | nan    | 3.26   | 0.385   | 0.673 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                         \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "EP 9/12: 2.538   | 16.80%   | 1.81   | nan    | 2.67   | 0.388   | 0.641 \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "EPOCH 10/12:  18%|█▊        | 4/22 [8:12:29<30:34:20, 6114.46s/it, loss=2.1954, acc=16.31%] "
     ]
    }
   ],
   "source": [
    "# --- 7. SOBOL SENSITIVITY ANALYSIS IMPLEMENTATION (FIXED) ---\n",
    "\n",
    "SOBOL_MASTER_DATA = {\n",
    "    \"session_id\": SESSION_ID,\n",
    "    \"problem\": sobol_problem,\n",
    "    \"runs\": {},\n",
    "    \"Si_all\": {} \n",
    "}\n",
    "\n",
    "def evaluate_sobol_run(params, run_id):\n",
    "    # Map back to the globals the model uses\n",
    "    global LEARNING_RATE, JITTER_SCALE, BASE_STRENGTH, PERIOD, LAMBDA_SLOW, H_INERTIA, H_SCALE\n",
    "    \n",
    "    current_lslow, h_inertia, current_strength, current_period, current_jitter = params\n",
    "    \n",
    "    LAMBDA_SLOW = current_lslow\n",
    "    H_INERTIA = h_inertia          \n",
    "    BASE_STRENGTH = current_strength\n",
    "    PERIOD = current_period\n",
    "    JITTER_SCALE = current_jitter\n",
    "    \n",
    "    # Critical derivation for architectural parity\n",
    "    H_SCALE = [H_INERTIA, 1.0 - H_INERTIA] \n",
    "\n",
    "    # Clear memory from previous run to prevent VRAM accumulation\n",
    "    tf.keras.backend.clear_session()\n",
    "    gc.collect()\n",
    "    \n",
    "    # Build model (Will use the updated globals)\n",
    "    # Reset seeds here for identical initialization across configurations\n",
    "    tf.keras.utils.set_random_seed(42)\n",
    "    model = OscillatingResonator(hidden=HIDDEN, num_classes=10, strength=BASE_STRENGTH, mode=\"active\")\n",
    "    \n",
    "    # Standard Header for the Run\n",
    "    print(f\"\\n[Sobol Run {run_id}/{total_runs}] \"\n",
    "          f\"H_Inertia: {H_INERTIA:.4f} | \"\n",
    "          f\"BASE_S: {BASE_STRENGTH:.4f} | \"\n",
    "          f\"Tau: {LAMBDA_SLOW:.4f} | \"\n",
    "          f\"Jitter: {JITTER_SCALE:.2f}\")\n",
    "\n",
    "    # Execute training\n",
    "    # Note: ensure train_phase uses print(f\"\\r...\", end=\"\", flush=True) for epoch updates\n",
    "    history = train_phase(model, train_ds, val_ds, epochs=EPOCHS, name=\"sobol\", run_id=run_id)\n",
    "    \n",
    "    # Final newline to clear the 'flush' line from train_phase\n",
    "    print(\"\") \n",
    "\n",
    "    # Safely extract metrics from the final epoch\n",
    "    metrics = {\n",
    "        \"acc\": float(history['acc'][-1]),\n",
    "        \"rank\": float(history['hidden_metrics'][-1]['effective_rank']),\n",
    "        \"sync\": float(history['hidden_metrics'][-1]['synchrony']),\n",
    "        \"entr\": float(history['hidden_metrics'][-1]['entropy']),\n",
    "        \"acorr\": float(history['hidden_metrics'][-1]['a_corr']),\n",
    "        \"Intf\": float(history['hidden_metrics'][-1]['interference'])\n",
    "    }\n",
    "    \n",
    "    # Cleanup model reference\n",
    "    del model\n",
    "    return metrics\n",
    "\n",
    "def run_sobol_analysis():\n",
    "    print(f\"--- STARTING MULTI-METRIC SOBOL ANALYSIS ({SESSION_ID}) ---\")\n",
    "    \n",
    "    # SALib generates N * (2D + 2) samples\n",
    "    # For D=5, N=8, this results in 8 * (10 + 2) = 96 runs\n",
    "    param_values = saltelli.sample(sobol_problem, N_baseline, calc_second_order=True) \n",
    "    num_runs = len(param_values)\n",
    "    \n",
    "    metric_keys = [\"acc\", \"rank\", \"sync\", \"entr\", \"acorr\", \"Intf\"]\n",
    "    Y = {m: np.zeros(num_runs) for m in metric_keys}\n",
    "\n",
    "    for i, params in enumerate(param_values):\n",
    "        try:\n",
    "            res = evaluate_sobol_run(params, run_id=i)\n",
    "            for key in metric_keys: \n",
    "                Y[key][i] = res[key]\n",
    "        except Exception as e:\n",
    "            # Move to next line so the error doesn't overwrite the progress line\n",
    "            print(f\"\\n!! Run {i} failed: {e}\")\n",
    "            # Fill with NaN so the analyzer can handle the missing data\n",
    "            for key in metric_keys: Y[key][i] = np.nan \n",
    "\n",
    "    # Calculate Sensitivity Indices for all tracked metrics\n",
    "    Si_all = {}\n",
    "    for key in metric_keys:\n",
    "        y_clean = Y[key]\n",
    "        nan_mask = np.isnan(y_clean)\n",
    "        n_failed = np.sum(nan_mask)\n",
    "        \n",
    "        if n_failed > 0:\n",
    "            print(f\"Warning: {n_failed} failed runs for metric '{key}'\")\n",
    "        \n",
    "        # If more than 10% of runs failed, the indices might be unreliable\n",
    "        if n_failed / len(y_clean) > 0.1:\n",
    "            print(f\"Skipping {key}: failure rate too high ({n_failed}/{len(y_clean)})\")\n",
    "            Si_all[key] = None\n",
    "            continue\n",
    "            \n",
    "        # Perform Sobol Analysis\n",
    "        Si = sobol.analyze(sobol_problem, y_clean, print_to_console=False)\n",
    "        Si_all[key] = {\n",
    "            \"S1\": Si['S1'].tolist(),\n",
    "            \"ST\": Si['ST'].tolist(),\n",
    "            \"S2\": Si['S2'].tolist() if 'S2' in Si else None,\n",
    "            \"n_failed\": int(n_failed)\n",
    "        }\n",
    "\n",
    "    SOBOL_MASTER_DATA[\"Si_all\"] = Si_all\n",
    "    \n",
    "    # JSON Serialization helper for Numpy/Tensorflow types\n",
    "    def clean(obj):\n",
    "        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}\n",
    "        if isinstance(obj, list): return [clean(i) for i in obj]\n",
    "        if isinstance(obj, (np.float32, np.float64, np.ndarray)): \n",
    "            return float(obj) if np.isscalar(obj) else obj.tolist()\n",
    "        return obj\n",
    "\n",
    "    output_path = f\"SOBOL_MASTER_{SESSION_ID}.json\"\n",
    "    with open(output_path, \"w\") as f:\n",
    "        json.dump(clean(SOBOL_MASTER_DATA), f, indent=4)\n",
    "        \n",
    "    print(f\"\\nFINISH. Full sensitivity results saved to: {output_path}\")\n",
    "    return Si_all\n",
    "\n",
    "# --- EXECUTION ---\n",
    "Si_all = run_sobol_analysis()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fbc9af16",
   "metadata": {},
   "source": [
    "## Evaluate SOBOL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d377e35",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# --- 1. LOAD THE DATA ---\n",
    "FILE_PATH = r\"SOBOL_RECOVERED_S2.json\"\n",
    "\n",
    "with open(FILE_PATH, \"r\") as f:\n",
    "    data = json.load(f)\n",
    "\n",
    "# --- 2. HEATMAP: SENSITIVITY INDICES ---\n",
    "def plot_sensitivity_heatmap(data):\n",
    "    si_all = data.get(\"Si_all\", {})\n",
    "    param_names = data[\"problem\"][\"names\"]\n",
    "    \n",
    "    # Restructure data for Seaborn\n",
    "    heatmap_data = []\n",
    "    metrics = list(si_all.keys()) # ['acc', 'rank', 'sync', 'entr', 'acorr', 'Intf']\n",
    "    \n",
    "    for metric in metrics:\n",
    "        # We focus on ST (Total-Order Sensitivity) for the big picture\n",
    "        st_values = si_all[metric][\"ST\"]\n",
    "        heatmap_data.append(st_values)\n",
    "        \n",
    "    df = pd.DataFrame(heatmap_data, index=metrics, columns=param_names)\n",
    "    \n",
    "    plt.figure(figsize=(10, 6))\n",
    "    sns.heatmap(df, annot=True, cmap=\"viridis\", fmt=\".2f\", linewidths=0.5)\n",
    "    plt.title(\"Total-Order Sensitivity ($S_T$) Across All Metrics\", pad=15)\n",
    "    plt.ylabel(\"Measurement Metric\")\n",
    "    plt.xlabel(\"Hyperparameter\")\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# --- 3. EVOLUTION PLOTS: EPOCH TRAJECTORIES ---\n",
    "def plot_epoch_evolution(data):\n",
    "    runs = data.get(\"runs\", {})\n",
    "    \n",
    "    # Setup a 2x3 grid for the 6 measurements\n",
    "    fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
    "    fig.suptitle(\"Epoch-by-Epoch Evolution Across All Sobol Runs\", fontsize=16)\n",
    "    axes = axes.flatten()\n",
    "    \n",
    "    metrics_to_plot = [\n",
    "        (\"acc\", \"Accuracy\"),\n",
    "        (\"effective_rank\", \"Effective Rank\"),\n",
    "        (\"synchrony\", \"Synchrony\"),\n",
    "        (\"entropy\", \"Entropy\"),\n",
    "        (\"a_corr\", \"Auto-Correlation\"),\n",
    "        (\"interference\", \"Interference\")\n",
    "    ]\n",
    "    \n",
    "    for ax, (metric_key, title) in zip(axes, metrics_to_plot):\n",
    "        ax.set_title(title)\n",
    "        ax.set_xlabel(\"Epoch\")\n",
    "        ax.set_ylabel(\"Value\")\n",
    "        ax.grid(True, linestyle=\"--\", alpha=0.6)\n",
    "        \n",
    "        # Plot every run as a semi-transparent line\n",
    "        for run_id, run_data in runs.items():\n",
    "            try:\n",
    "                if metric_key == \"acc\":\n",
    "                    y_vals = run_data[\"history\"][\"acc\"]\n",
    "                else:\n",
    "                    y_vals = [epoch_data[metric_key] for epoch_data in run_data[\"history\"][\"hidden_metrics\"]]\n",
    "                \n",
    "                x_vals = range(1, len(y_vals) + 1)\n",
    "                # alpha=0.15 creates a \"density\" effect when hundreds of lines overlap\n",
    "                ax.plot(x_vals, y_vals, color=\"blue\", alpha=0.15, linewidth=1)\n",
    "            except KeyError:\n",
    "                continue # Skip if a run crashed or is missing data\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.subplots_adjust(top=0.9)\n",
    "    plt.show()\n",
    "\n",
    "# --- EXECUTE ---\n",
    "if __name__ == \"__main__\":\n",
    "    plot_sensitivity_heatmap(data)\n",
    "    plot_epoch_evolution(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68618e38",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import numpy as np\n",
    "import os\n",
    "from SALib.analyze import sobol\n",
    "\n",
    "# --- CONFIGURATION ---\n",
    "FILE_NAME = \"SOBOL_RECOVERED_S2.json\"\n",
    "METRIC_KEYS = [\"acc\", \"rank\", \"sync\", \"entr\", \"acorr\", \"Intf\"]\n",
    "\n",
    "# --- STEP 1: LOAD DATA ---\n",
    "if not os.path.exists(FILE_NAME):\n",
    "    print(f\"Error: {FILE_NAME} not found.\")\n",
    "    exit()\n",
    "\n",
    "with open(FILE_NAME, \"r\") as f:\n",
    "    master_data = json.load(f)\n",
    "\n",
    "problem = master_data[\"problem\"]\n",
    "runs = master_data[\"runs\"]\n",
    "\n",
    "# Filter for runs that actually have history data\n",
    "valid_ids = sorted([k for k in runs.keys() if \"history\" in runs[k]], key=int)\n",
    "num_runs = len(valid_ids)\n",
    "\n",
    "print(f\"Loaded {num_runs} valid runs. Processing metrics...\")\n",
    "\n",
    "# --- STEP 2: EXTRACT Y-VALUES ---\n",
    "# We map your JSON keys to the internal Y_data keys\n",
    "Y_data = {m: np.zeros(num_runs) for m in METRIC_KEYS}\n",
    "\n",
    "for i, run_id in enumerate(valid_ids):\n",
    "    history = runs[run_id][\"history\"]\n",
    "    \n",
    "    # Extracting from the LAST entry of the lists (Final Epoch)\n",
    "    # acc is a direct list in history\n",
    "    Y_data[\"acc\"][i] = history[\"acc\"][-1]\n",
    "    \n",
    "    # hidden_metrics is a list of dicts; we take the last dict\n",
    "    last_metrics = history[\"hidden_metrics\"][-1]\n",
    "    \n",
    "    Y_data[\"rank\"][i]  = last_metrics[\"effective_rank\"]\n",
    "    Y_data[\"sync\"][i]  = last_metrics[\"synchrony\"]\n",
    "    Y_data[\"entr\"][i]  = last_metrics[\"entropy\"]\n",
    "    Y_data[\"acorr\"][i] = last_metrics[\"a_corr\"]\n",
    "    Y_data[\"Intf\"][i]  = last_metrics[\"interference\"]\n",
    "\n",
    "# --- STEP 3: SOBOL ANALYSIS ---\n",
    "Si_results = {}\n",
    "\n",
    "for key in METRIC_KEYS:\n",
    "    print(f\"Calculating Sobol (S1, ST, S2) for: {key}...\")\n",
    "    \n",
    "    # calc_second_order=True requires the specific sample size N*(2D+2)\n",
    "    # SALib will throw an error if your num_runs doesn't match the problem definition\n",
    "    try:\n",
    "        Si = sobol.analyze(problem, Y_data[key], calc_second_order=True, print_to_console=False)\n",
    "        \n",
    "        # S2 is a NxN matrix, we must convert it to a list for JSON\n",
    "        Si_results[key] = {\n",
    "            \"S1\": Si['S1'].tolist(),\n",
    "            \"ST\": Si['ST'].tolist(),\n",
    "            \"S2\": Si['S2'].tolist() if Si['S2'] is not None else None,\n",
    "            \"S1_conf\": Si['S1_conf'].tolist(),\n",
    "            \"ST_conf\": Si['ST_conf'].tolist()\n",
    "        }\n",
    "    except Exception as e:\n",
    "        print(f\"  [!] Failed to calculate S2 for {key}: {e}\")\n",
    "\n",
    "# --- STEP 4: SAVE ---\n",
    "master_data[\"Si_all\"] = Si_results\n",
    "output_file = FILE_NAME.replace(\".json\", \"_FIXED_S2.json\")\n",
    "\n",
    "with open(output_file, \"w\") as f:\n",
    "    json.dump(master_data, f, indent=4)\n",
    "\n",
    "print(f\"\\nSUCCESS: Results saved to {output_file}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4aa95a66",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import numpy as np\n",
    "import re\n",
    "import os\n",
    "from SALib.analyze import sobol\n",
    "\n",
    "#FILE_NAME = \"SOBOL_MASTER_RES_H32_S0.22_J0.6_260319_1405.json\"\n",
    "FINAL_OUTPUT = \"SOBOL_RECOVERED_S2.json\"\n",
    "\n",
    "def load_with_nan_fix(filename):\n",
    "    print(f\"Loading and sanitizing {filename}...\")\n",
    "    if not os.path.exists(filename):\n",
    "        raise FileNotFoundError(f\"Could not find {filename}\")\n",
    "        \n",
    "    with open(filename, 'r', encoding='utf-8') as f:\n",
    "        content = f.read()\n",
    "\n",
    "    # The Fix: Replace unquoted NaN with null so the JSON parser accepts it\n",
    "    sanitized = re.sub(r':\\s*NaN', ': null', content, flags=re.IGNORECASE)\n",
    "    sanitized = re.sub(r':\\s*nan', ': null', sanitized)\n",
    "    \n",
    "    try:\n",
    "        return json.loads(sanitized)\n",
    "    except json.JSONDecodeError:\n",
    "        # If it still fails, the file might be mid-write. Seal it.\n",
    "        print(\"JSON incomplete, applying emergency seal...\")\n",
    "        if not sanitized.strip().endswith('}'):\n",
    "            sanitized = sanitized.strip().rstrip(',') + \"}}}}\"\n",
    "        return json.loads(sanitized)\n",
    "\n",
    "# --- EXECUTION ---\n",
    "try:\n",
    "    master_data = load_with_nan_fix(FILE_NAME)\n",
    "except Exception as e:\n",
    "    print(f\"FATAL ERROR during load: {e}\")\n",
    "    exit()\n",
    "\n",
    "runs = master_data[\"runs\"]\n",
    "problem = master_data[\"problem\"]\n",
    "param_names = problem['names']\n",
    "\n",
    "# Sort IDs numerically to ensure Y matches the Sobol sequence\n",
    "valid_ids = sorted(runs.keys(), key=int)\n",
    "num_runs = len(valid_ids)\n",
    "\n",
    "print(f\"Processing {num_runs} runs...\")\n",
    "\n",
    "# Define targets\n",
    "metric_map = {\n",
    "    \"acc\": \"acc\",\n",
    "    \"rank\": \"effective_rank\",\n",
    "    \"sync\": \"synchrony\",\n",
    "    \"entr\": \"entropy\",\n",
    "    \"acorr\": \"a_corr\",\n",
    "    \"Intf\": \"interference\"\n",
    "}\n",
    "\n",
    "Y_data = {k: np.zeros(num_runs) for k in metric_map.keys()}\n",
    "\n",
    "for i, rid in enumerate(valid_ids):\n",
    "    hist = runs[rid].get(\"history\", {})\n",
    "    \n",
    "    # 1. Grab last Accuracy\n",
    "    acc_list = hist.get(\"acc\", [])\n",
    "    Y_data[\"acc\"][i] = acc_list[-1] if acc_list else 0\n",
    "    \n",
    "    # 2. Grab last epoch of Hidden Metrics\n",
    "    metrics_list = hist.get(\"hidden_metrics\", [])\n",
    "    if metrics_list:\n",
    "        last_epoch = metrics_list[-1]\n",
    "        for short_key, json_key in metric_map.items():\n",
    "            if short_key == \"acc\": continue\n",
    "            \n",
    "            val = last_epoch.get(json_key)\n",
    "            # If val is None (from our 'null' fix) or NaN, default to 0\n",
    "            if val is None or (isinstance(val, float) and np.isnan(val)):\n",
    "                Y_data[short_key][i] = 0.0\n",
    "            else:\n",
    "                Y_data[short_key][i] = float(val)\n",
    "\n",
    "# --- STEP 3: DYNAMIC TRIMMING & ANALYSIS ---\n",
    "Si_results = {}\n",
    "\n",
    "D = problem['num_vars']\n",
    "calc_second_order = True \n",
    "# For 5 vars: (2*5 + 2) = 12 runs per block\n",
    "step_size = (2 * D + 2) if calc_second_order else (D + 2)\n",
    "\n",
    "for key in Y_data.keys():\n",
    "    total_available = len(Y_data[key])\n",
    "    num_complete_blocks = total_available // step_size\n",
    "    valid_cutoff = num_complete_blocks * step_size\n",
    "    \n",
    "    if valid_cutoff == 0:\n",
    "        print(f\"Skipping {key}: Need at least {step_size} runs, but only have {total_available}.\")\n",
    "        continue\n",
    "\n",
    "    # Trim to valid Sobol structure\n",
    "    Y_trimmed = Y_data[key][:valid_cutoff]\n",
    "    \n",
    "    print(f\"\\nAnalyzing {key.upper()}: Using {valid_cutoff} runs...\")\n",
    "\n",
    "    try:\n",
    "        Si = sobol.analyze(problem, Y_trimmed, calc_second_order=calc_second_order, print_to_console=False)\n",
    "        \n",
    "        # --- S2 Extraction & Significance Check ---\n",
    "        s2_raw = Si['S2']\n",
    "        significant_interactions = []\n",
    "        \n",
    "        # S2 is a DxD matrix where only the upper triangle is filled\n",
    "        for i in range(D):\n",
    "            for j in range(i + 1, D):\n",
    "                val = s2_raw[i, j]\n",
    "                if not np.isnan(val) and abs(val) > 0.02:  # Threshold for significance\n",
    "                    significant_interactions.append(f\"{param_names[i]} x {param_names[j]}: {val:.3f}\")\n",
    "        \n",
    "        if significant_interactions:\n",
    "            print(f\"  -> Top Interactions: {', '.join(significant_interactions[:2])}\")\n",
    "\n",
    "        # Store results (clean NaNs for JSON safety)\n",
    "        Si_results[key] = {\n",
    "            \"S1\": np.nan_to_num(Si['S1'], nan=0.0).tolist(),\n",
    "            \"ST\": np.nan_to_num(Si['ST'], nan=0.0).tolist(),\n",
    "            \"S2\": np.nan_to_num(Si['S2'], nan=0.0).tolist()\n",
    "        }\n",
    "        \n",
    "    except Exception as e:\n",
    "        print(f\"  -> Sobol failed for {key}: {e}\")\n",
    "\n",
    "# --- STEP 4: SAVE ---\n",
    "master_data[\"Si_all\"] = Si_results\n",
    "with open(FINAL_OUTPUT, \"w\") as f:\n",
    "    json.dump(master_data, f, indent=4)\n",
    "\n",
    "print(f\"\\nSUCCESS! Recovery and Analysis complete.\")\n",
    "print(f\"Saved results to: {FINAL_OUTPUT}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c6ef81f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import networkx as nx\n",
    "\n",
    "# 1. LOAD THE PROPER DATA\n",
    "FILE_NAME = \"SOBOL_RECOVERED_S2.json\" \n",
    "with open(FILE_NAME, \"r\") as f:\n",
    "    master = json.load(f)\n",
    "\n",
    "si_all = master[\"Si_all\"]\n",
    "param_names = master[\"problem\"][\"names\"]\n",
    "metrics = list(si_all.keys())\n",
    "\n",
    "# --- 2. PLOT 1: TOTAL ORDER (ST) VS FIRST ORDER (S1) ---\n",
    "# This shows direct vs. interaction-based drivers\n",
    "st_df = pd.DataFrame({m: si_all[m][\"ST\"] for m in metrics}, index=param_names).T\n",
    "s1_df = pd.DataFrame({m: si_all[m][\"S1\"] for m in metrics}, index=param_names).T\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)\n",
    "\n",
    "sns.heatmap(s1_df, annot=True, cmap=\"RdBu_r\", ax=axes[0], cbar_kws={'label': 'Index Value'})\n",
    "axes[0].set_title(\"First-Order Sensitivity ($S_1$)\\n(Direct Contribution)\")\n",
    "\n",
    "sns.heatmap(st_df, annot=True, cmap=\"RdBu_r\", ax=axes[1], cbar_kws={'label': 'Index Value'})\n",
    "axes[1].set_title(\"Total-Order Sensitivity ($S_T$)\\n(Including Interactions)\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# --- 3. PLOT 2: S2 INTERACTION MATRIX (Corrected for Object Dtype) ---\n",
    "target_metric = \"acc\" \n",
    "\n",
    "# Convert the list-of-lists directly to a DataFrame and force float conversion\n",
    "# This handles the 'None' to 'NaN' transition correctly\n",
    "s2_matrix = pd.DataFrame(\n",
    "    si_all[target_metric][\"S2\"], \n",
    "    index=param_names, \n",
    "    columns=param_names\n",
    ").astype(float) # <--- This is the magic line that fixes your TypeError\n",
    "\n",
    "plt.figure(figsize=(10, 8))\n",
    "\n",
    "# Mask the lower triangle and the diagonal for a cleaner look\n",
    "mask = np.tril(np.ones_like(s2_matrix, dtype=bool))\n",
    "\n",
    "sns.heatmap(\n",
    "    s2_matrix, \n",
    "    mask=mask, \n",
    "    annot=True, \n",
    "    cmap=\"RdBu_r\", \n",
    "    center=0, \n",
    "    fmt=\".3f\",\n",
    "    cbar_kws={'label': 'Synergy Index'}\n",
    ")\n",
    "\n",
    "plt.title(f\"Second-Order Synergy ($S_2$) for {target_metric.upper()}\\n\"\n",
    "          f\"Positive = Synergy | Negative = Competitive\")\n",
    "plt.show()\n",
    "\n",
    "# --- 4. PLOT 3: THE INTERACTION NETWORK ---\n",
    "# Visualizing the \"binding\" strength between parameters\n",
    "# --- 4. PLOT 3: THE INTERACTION NETWORK (FIXED) ---\n",
    "plt.figure(figsize=(8, 8))\n",
    "G = nx.Graph()\n",
    "\n",
    "# Iterate through the DataFrame we created in Step 3\n",
    "for i, p1 in enumerate(param_names):\n",
    "    for j, p2 in enumerate(param_names):\n",
    "        if i >= j: continue # Only look at the upper triangle\n",
    "        \n",
    "        val = s2_matrix.iloc[i, j] # Use iloc to get values from the DF\n",
    "        \n",
    "        # We only plot significant interactions (> 0.02 or 0.05)\n",
    "        if not np.isnan(val) and abs(val) > 0.02: \n",
    "            color = 'firebrick' if val > 0 else 'royalblue'\n",
    "            G.add_edge(p1, p2, weight=abs(val), color=color)\n",
    "\n",
    "# Draw the graph\n",
    "pos = nx.spring_layout(G, k=1.5) # Spring layout often looks better than circular for networks\n",
    "edges = G.edges()\n",
    "colors = [G[u][v]['color'] for u, v in edges]\n",
    "weights = [G[u][v]['weight'] * 40 for u, v in edges] # Scaled for visibility\n",
    "\n",
    "nx.draw(G, pos, with_labels=True, node_size=3000, node_color=\"#f0f0f0\", \n",
    "        edge_color=colors, width=weights, font_size=10, font_weight=\"bold\")\n",
    "\n",
    "plt.title(f\"Parameter Interaction Network: {target_metric.upper()}\\n\"\n",
    "          f\"(Red = Synergy | Blue = Competition)\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c5bc511",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "\n",
    "# --- 1. CONFIGURATION & MAPPING ---\n",
    "# Ensure this matches your local filename\n",
    "FILE_NAME = \"SOBOL_RECOVERED_S2.json\" \n",
    "\n",
    "# Mapping the \"Problem\" names to the \"Config\" keys found in your JSON\n",
    "CONFIG_MAP = {\n",
    "    \"LAMBDA_SLOW\": \"tau\",\n",
    "    \"BASE_STRENGTH\": \"strength\",\n",
    "    \"JITTER_SCALE\": \"jitter\",\n",
    "    \"PERIOD\": \"period\",\n",
    "    \"H_INERTIA\": \"h_inertia\",\n",
    "    \"LEARNING_RATE\": \"learning rate\"\n",
    "}\n",
    "\n",
    "# List of all measures to plot\n",
    "METRICS = [\n",
    "    (\"acc\", \"Accuracy\"),\n",
    "    (\"loss\", \"Loss\"),\n",
    "    (\"effective_rank\", \"Effective Rank\"),\n",
    "    (\"synchrony\", \"Synchrony\"),\n",
    "    (\"entropy\", \"Entropy\"),\n",
    "    (\"a_corr\", \"Auto-Correlation\"),\n",
    "    (\"interference\", \"Interference\")\n",
    "]\n",
    "\n",
    "def generate_sobol_evolution_plots(file_path):\n",
    "    with open(file_path, \"r\") as f:\n",
    "        data = json.load(f)\n",
    "\n",
    "    problem = data[\"problem\"]\n",
    "    runs = data[\"runs\"]\n",
    "    param_names = problem[\"names\"]\n",
    "    param_bounds = problem[\"bounds\"]\n",
    "\n",
    "    # Iterate through every metric (e.g., Accuracy, then Rank, etc.)\n",
    "    for metric_key, metric_label in METRICS:\n",
    "        print(f\"Generating evolution plots for: {metric_label}...\")\n",
    "        \n",
    "        # Create a figure with 5 subplots (one for each hyperparameter)\n",
    "        fig, axes = plt.subplots(1, 5, figsize=(25, 6), sharey=False)\n",
    "        fig.suptitle(f\"Drift of {metric_label} Across Parameter Space (Epoch 1 → 6)\", \n",
    "                     fontsize=22, fontweight='bold', y=1.08)\n",
    "        \n",
    "        for i, p_name in enumerate(param_names):\n",
    "            ax = axes[i]\n",
    "            c_key = CONFIG_MAP.get(p_name)\n",
    "            bounds = param_bounds[i]\n",
    "            \n",
    "            # Formatting the subplot\n",
    "            ax.set_title(f\"vs {p_name}\", fontsize=14, pad=10)\n",
    "            ax.set_ylabel(f\"Parameter Value: {p_name}\", fontsize=12)\n",
    "            ax.set_xlabel(f\"{metric_label} Value\", fontsize=12)\n",
    "            ax.set_ylim(bounds[0], bounds[1])\n",
    "            ax.grid(True, linestyle=\"--\", alpha=0.3)\n",
    "            \n",
    "            # Plot trajectories for every single run\n",
    "            for run_id, run_data in runs.items():\n",
    "                try:\n",
    "                    if \"history\" not in run_data or \"config\" not in run_data:\n",
    "                        continue\n",
    "                    \n",
    "                    # 1. Get the hyperparameter value (Constant for the run - Y-axis)\n",
    "                    y_val = run_data[\"config\"][c_key]\n",
    "                    \n",
    "                    # 2. Get the metric history (Changes per epoch - X-axis)\n",
    "                    if metric_key in [\"acc\", \"loss\"]:\n",
    "                        x_history = run_data[\"history\"][metric_key]\n",
    "                    else:\n",
    "                        # Extract from the hidden_metrics list\n",
    "                        x_history = [epoch[metric_key] for epoch in run_data[\"history\"][\"hidden_metrics\"]]\n",
    "                    \n",
    "                    # 3. Draw the line (Blue = trajectory, Red Dot = Final State)\n",
    "                    ax.plot(x_history, [y_val] * len(x_history), \n",
    "                            color='#2980b9', alpha=0.1, linewidth=1)\n",
    "                    \n",
    "                    # Mark the final epoch (Epoch 5)\n",
    "                    ax.scatter(x_history[-1], y_val, color='#c0392b', s=8, alpha=0.2)\n",
    "                    \n",
    "                except (KeyError, IndexError, TypeError):\n",
    "                    continue # Skip incomplete data points silently\n",
    "        \n",
    "        plt.tight_layout()\n",
    "        # Save as high-res PNG\n",
    "        plt.savefig(f\"SOBOL_EVOLUTION_{metric_key.upper()}.png\", dpi=150, bbox_inches='tight')\n",
    "        plt.show()\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    generate_sobol_evolution_plots(FILE_NAME)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "883971c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pandas.plotting import parallel_coordinates\n",
    "\n",
    "# --- 1. DATA PROCESSING ---\n",
    "FILE_PATH = \"SOBOL_RECOVERED_S2.json\"\n",
    "with open(FILE_PATH, \"r\") as f:\n",
    "    data = json.load(f)\n",
    "\n",
    "def get_processed_df(data):\n",
    "    runs = data[\"runs\"]\n",
    "    param_names = data[\"problem\"][\"names\"]\n",
    "    # Mapping JSON config keys to readable problem names\n",
    "    config_map = {\"LAMBDA_SLOW\": \"tau\", \"BASE_STRENGTH\": \"strength\", \n",
    "                  \"JITTER_SCALE\": \"jitter\", \"PERIOD\": \"period\", \"H_INERTIA\": \"h_inertia\"}\n",
    "    #config_map = {\"LEARNING_RATE\": \"LR\", \"H_INERTIA\": \"h_inertia\"}\n",
    "    \n",
    "    \n",
    "    records = []\n",
    "    for rid, rdata in runs.items():\n",
    "        if \"history\" not in rdata: continue\n",
    "        h = rdata[\"history\"]\n",
    "        m = h[\"hidden_metrics\"][-1]\n",
    "        \n",
    "        row = {\n",
    "            \"acc\": h[\"acc\"][-1],\n",
    "            \"loss\": h[\"loss\"][-1],\n",
    "            \"rank\": m[\"effective_rank\"],\n",
    "            \"sync\": m[\"synchrony\"],\n",
    "            \"entr\": m[\"entropy\"],\n",
    "            \"acorr\": m[\"a_corr\"],\n",
    "            \"intf\": m[\"interference\"]\n",
    "        }\n",
    "        # Safely map parameters\n",
    "        for p_name in param_names:\n",
    "            c_key = config_map.get(p_name, p_name.lower())\n",
    "            row[p_name] = rdata[\"config\"].get(c_key, 0)\n",
    "        records.append(row)\n",
    "    return pd.DataFrame(records)\n",
    "\n",
    "df = get_processed_df(data)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e90380c",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# --- 2. PLOT 1: RADAR SENSITIVITY (FIXED) ---\n",
    "def plot_radar(si_all, param_names, target_metrics=['acc', 'sync', 'rank', 'entr', 'acorr', 'Intf']):\n",
    "    angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()\n",
    "    angles += angles[:1] # Close the circle\n",
    "    \n",
    "    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))\n",
    "    \n",
    "    for m_key in target_metrics:\n",
    "        if m_key in si_all:\n",
    "            # Convert list to numpy array to ensure it works, then handle the loop\n",
    "            values = np.array(si_all[m_key][\"ST\"]).tolist()\n",
    "            values += values[:1]\n",
    "            ax.plot(angles, values, linewidth=2, label=m_key.upper())\n",
    "            ax.fill(angles, values, alpha=0.1)\n",
    "\n",
    "    ax.set_theta_offset(np.pi / 2)\n",
    "    ax.set_theta_direction(-1)\n",
    "    ax.set_xticks(angles[:-1])\n",
    "    ax.set_xticklabels(param_names)\n",
    "    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))\n",
    "    plt.title(\"Total Sensitivity ($S_T$) across Metrics\", pad=20, fontsize=15)\n",
    "    plt.show()\n",
    "\n",
    "# --- 3. PLOT 2: PHASE PORTRAIT (Synchrony vs Accuracy) ---\n",
    "plt.figure(figsize=(10, 6))\n",
    "# Using 'sync' and 'acc' from our processed dataframe\n",
    "scatter = plt.scatter(df['sync'], df['acc'], c=df['H_INERTIA'], cmap='viridis', alpha=0.6)\n",
    "plt.colorbar(scatter, label='H_INERTIA Value')\n",
    "plt.xlabel('Global Synchrony (Final State)')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.title('Mechanism: Accuracy vs. Synchrony Phase Portrait')\n",
    "plt.grid(True, linestyle='--', alpha=0.5)\n",
    "plt.show()\n",
    "\n",
    "\n",
    "\n",
    "# --- 4. PLOT 3: PARALLEL COORDINATES (The \"Winning Recipe\") ---\n",
    "def plot_winning_pathways(df):\n",
    "    plt.figure(figsize=(15, 8))\n",
    "    \n",
    "    # 1. Prepare and Normalize Data\n",
    "    cols = ['LAMBDA_SLOW', 'BASE_STRENGTH', 'JITTER_SCALE', 'PERIOD', 'H_INERTIA', 'acc']\n",
    "    df_plot = df[cols].copy()\n",
    "    \n",
    "    # Min-Max Normalization for plotting\n",
    "    for col in cols:\n",
    "        df_plot[col] = (df_plot[col] - df_plot[col].min()) / (df_plot[col].max() - df_plot[col].min())\n",
    "    \n",
    "    # 2. Split into groups based on original Accuracy\n",
    "    q90_val = df['acc'].quantile(0.99)\n",
    "    # Get indices for splitting\n",
    "    top_indices = df[df['acc'] >= q90_val].index\n",
    "    other_indices = df[df['acc'] < q90_val].index\n",
    "    \n",
    "    x = np.arange(len(cols))\n",
    "    \n",
    "    # 3. Plot \"Others\" first (Green, Higher Alpha)\n",
    "    for idx in other_indices:\n",
    "        plt.plot(x, df_plot.loc[idx], color=\"#710d0d\", alpha=0.15, linewidth=0.5)\n",
    "        \n",
    "    # 4. Plot \"Top 1%\" (Red, Lower Alpha)\n",
    "    for idx in top_indices:\n",
    "        plt.plot(x, df_plot.loc[idx], color=\"#3ce73c\", alpha=0.6, linewidth=2.0)\n",
    "\n",
    "    # 5. Formatting\n",
    "    plt.xticks(x, cols, rotation=15, fontsize=12)\n",
    "    plt.title(\"Neural Pathway Analysis: Top 10% (Red Ghosting) vs Others (Solid Green)\", fontsize=16, pad=20)\n",
    "    plt.ylabel(\"Normalized Parameter Range [0-1]\", fontsize=12)\n",
    "    \n",
    "    # Custom Legend\n",
    "    from matplotlib.lines import Line2D\n",
    "    legend_elements = [\n",
    "        Line2D([0], [0], color=\"#710d0d\", lw=2, alpha=0.15, label='Others '),\n",
    "        Line2D([0], [0], color=\"#3ce742\", lw=2, alpha=0.6, label='Top 1% ')\n",
    "    ]\n",
    "    plt.legend(handles=legend_elements, loc='upper right')\n",
    "    \n",
    "    plt.grid(axis='x', linestyle='-', alpha=0.1)\n",
    "    plt.grid(axis='y', linestyle='--', alpha=0.2)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "\n",
    "# --- EXECUTE ALL ---\n",
    "plot_radar(data[\"Si_all\"], data[\"problem\"][\"names\"])\n",
    "plot_winning_pathways(df)\n",
    "\n",
    "def plot_rank_pathways(df):\n",
    "    plt.figure(figsize=(15, 8))\n",
    "    \n",
    "    # 1. Prepare and Normalize Data\n",
    "    # We include 'rank' in the columns now to see its final value\n",
    "    cols = ['LAMBDA_SLOW', 'BASE_STRENGTH', 'JITTER_SCALE', 'PERIOD', 'H_INERTIA', 'rank']\n",
    "    df_plot = df[cols].copy()\n",
    "    \n",
    "    # Min-Max Normalization for visual alignment\n",
    "    for col in cols:\n",
    "        df_plot[col] = (df_plot[col] - df_plot[col].min()) / (df_plot[col].max() - df_plot[col].min())\n",
    "    \n",
    "    # 2. Split into groups based on Effective Rank\n",
    "    # Quantile 0.99 targets the most complex high-dimensional models\n",
    "    q99_rank = df['rank'].quantile(0.99)\n",
    "    top_rank_indices = df[df['rank'] >= q99_rank].index\n",
    "    other_indices = df[df['rank'] < q99_rank].index\n",
    "    \n",
    "    x = np.arange(len(cols))\n",
    "    \n",
    "    # 3. Plot \"Others\" (Deep Red/Brown background)\n",
    "    for idx in other_indices:\n",
    "        plt.plot(x, df_plot.loc[idx], color=\"#710d0d\", alpha=0.15, linewidth=0.5)\n",
    "        \n",
    "    # 4. Plot \"Top 1% Rank\" (Bright Green highlight)\n",
    "    # Higher alpha and thicker lines to show the 'complex' pathways\n",
    "    for idx in top_rank_indices:\n",
    "        plt.plot(x, df_plot.loc[idx], color=\"#3ce73c\", alpha=0.6, linewidth=2.5)\n",
    "\n",
    "    # 5. Formatting\n",
    "    plt.xticks(x, cols, rotation=15, fontsize=12)\n",
    "    plt.title(\"Dimensionality Analysis: Top 1% Effective Rank (Green) vs Baseline (Red)\", fontsize=16, pad=20)\n",
    "    plt.ylabel(\"Normalized Parameter Range [0-1]\", fontsize=12)\n",
    "    \n",
    "    # Custom Legend\n",
    "    from matplotlib.lines import Line2D\n",
    "    legend_elements = [\n",
    "        Line2D([0], [0], color=\"#710d0d\", lw=1, alpha=0.3, label='Standard Rank Runs'),\n",
    "        Line2D([0], [0], color=\"#3ce73c\", lw=3, alpha=0.8, label='Max Dimensionality (Top 1%)')\n",
    "    ]\n",
    "    plt.legend(handles=legend_elements, loc='upper right')\n",
    "    \n",
    "    plt.grid(axis='x', linestyle='-', alpha=0.1)\n",
    "    plt.grid(axis='y', linestyle='--', alpha=0.2)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# --- EXECUTE ---\n",
    "plot_rank_pathways(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea4ad141",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from matplotlib.lines import Line2D\n",
    "\n",
    "def plot_combined_pathways(df):\n",
    "    plt.figure(figsize=(16, 9))\n",
    "    \n",
    "    # 1. Prepare and Normalize Data\n",
    "    # We include both outcomes at the end of the axis\n",
    "    cols = ['LAMBDA_SLOW', 'BASE_STRENGTH', 'JITTER_SCALE', 'PERIOD', 'H_INERTIA', 'acc', 'rank']\n",
    "    #cols = ['LEARNING_RATE','H_INERTIA', 'acc', 'rank']\n",
    "    df_plot = df[cols].copy()\n",
    "    \n",
    "    for col in cols:\n",
    "        df_plot[col] = (df_plot[col] - df_plot[col].min()) / (df_plot[col].max() - df_plot[col].min())\n",
    "    \n",
    "    # 2. Define our \"Elite\" groups\n",
    "    q99_acc = df['acc'].quantile(0.95)\n",
    "    q99_rank = df['rank'].quantile(0.95)\n",
    "    \n",
    "    acc_top_idx = df[df['acc'] >= q99_acc].index\n",
    "    rank_top_idx = df[df['rank'] >= q99_rank].index\n",
    "    # Others are anything not in the top 1% of either\n",
    "    other_idx = df.index.difference(acc_top_idx.union(rank_top_idx))\n",
    "    \n",
    "    x = np.arange(len(cols))\n",
    "    \n",
    "    # 3. Plot \"Others\" (Grey/Brown background - Very low alpha)\n",
    "    for idx in other_idx:\n",
    "        plt.plot(x, df_plot.loc[idx], color=\"#710d0d\", alpha=0.15, linewidth=0.8)\n",
    "        \n",
    "    # 4. Plot \"Top 1% Rank\" (Blue highlight - Dimensionality)\n",
    "    for idx in rank_top_idx:\n",
    "        plt.plot(x, df_plot.loc[idx], color=\"#3498db\", alpha=0.6, linewidth=2.5)\n",
    "        \n",
    "    # 5. Plot \"Top 1% Accuracy\" (Green highlight - Performance)\n",
    "    for idx in acc_top_idx:\n",
    "        plt.plot(x, df_plot.loc[idx], color=\"#2ecc71\", alpha=0.6, linewidth=2.5)\n",
    "\n",
    "    # 6. Formatting\n",
    "    plt.xticks(x, cols, rotation=15, fontsize=12)\n",
    "    plt.title(\"Dual-Objective Pathway Analysis: Accuracy (Green) vs. Effective Rank (Blue)\", fontsize=18, pad=25)\n",
    "    plt.ylabel(\"Normalized Range [0-1]\", fontsize=13)\n",
    "    \n",
    "    # Custom Legend\n",
    "    legend_elements = [\n",
    "        Line2D([0], [0], color=\"#2ecc71\", lw=3, alpha=0.7, label='Top 1% Accuracy'),\n",
    "        Line2D([0], [0], color=\"#3498db\", lw=3, alpha=0.7, label='Top 1% Effective Rank'),\n",
    "        Line2D([0], [0], color=\"#710d0d\", lw=1, alpha=0.3, label='Baseline Runs')\n",
    "    ]\n",
    "    plt.legend(handles=legend_elements, loc='upper right', fontsize=11)\n",
    "    \n",
    "    plt.grid(axis='y', linestyle='--', alpha=0.2)\n",
    "    plt.grid(axis='x', linestyle='-', alpha=0.1)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# Run the dual analysis\n",
    "plot_combined_pathways(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd69ebe5",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_bifurcation_sweep(df):\n",
    "    # 1. Sort by H_INERTIA to see the progression\n",
    "    df_sweep = df.sort_values('H_INERTIA')\n",
    "    \n",
    "    fig, ax1 = plt.subplots(figsize=(12, 6))\n",
    "\n",
    "    # Plot Accuracy on the first Y-axis\n",
    "    color_acc = '#2ecc71'\n",
    "    ax1.set_xlabel('H_INERTIA (The Master Parameter)', fontsize=12)\n",
    "    ax1.set_ylabel('Accuracy', color=color_acc, fontsize=12)\n",
    "    # Using a rolling mean to show the trend through the noise\n",
    "    ax1.plot(df_sweep['H_INERTIA'], df_sweep['acc'].rolling(window=10).mean(), \n",
    "             color=color_acc, linewidth=3, label='Accuracy Trend')\n",
    "    ax1.scatter(df_sweep['H_INERTIA'], df_sweep['acc'], color=color_acc, alpha=0.1, s=10)\n",
    "    ax1.tick_params(axis='y', labelcolor=color_acc)\n",
    "\n",
    "    # Create a second Y-axis for Effective Rank\n",
    "    ax2 = ax1.twinx()\n",
    "    color_rank = '#3498db'\n",
    "    ax2.set_ylabel('Effective Rank (Complexity)', color=color_rank, fontsize=12)\n",
    "    ax2.plot(df_sweep['H_INERTIA'], df_sweep['rank'].rolling(window=10).mean(), \n",
    "             color=color_rank, linewidth=3, linestyle='--', label='Rank Trend')\n",
    "    ax2.tick_params(axis='y', labelcolor=color_rank)\n",
    "\n",
    "    plt.title('Bifurcation Analysis: The Dominant Effect of H_INERTIA', fontsize=15, pad=20)\n",
    "    ax1.grid(True, alpha=0.2)\n",
    "    \n",
    "    # Adding a vertical line at the \"Critical Point\"\n",
    "    # (Assuming the peak accuracy happens at a specific inertia)\n",
    "    optimal_h = df.loc[df['acc'].idxmax(), 'H_INERTIA']\n",
    "    plt.axvline(x=optimal_h, color='red', linestyle=':', alpha=0.5, label='Optimal H')\n",
    "\n",
    "    fig.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "plot_bifurcation_sweep(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b467548",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_synchrony_crash(df):\n",
    "    df_sweep = df.sort_values('H_INERTIA')\n",
    "    \n",
    "    fig, ax1 = plt.subplots(figsize=(12, 6))\n",
    "\n",
    "    # Plot Accuracy (The Goal)\n",
    "    color_acc = '#2ecc71' # Green\n",
    "    ax1.set_xlabel('H_INERTIA', fontsize=12)\n",
    "    ax1.set_ylabel('Accuracy', color=color_acc, fontsize=12)\n",
    "    ax1.plot(df_sweep['H_INERTIA'], df_sweep['acc'].rolling(window=15).mean(), \n",
    "             color=color_acc, linewidth=3, label='Accuracy')\n",
    "    ax1.tick_params(axis='y', labelcolor=color_acc)\n",
    "\n",
    "    # Plot Synchrony (The Physical State)\n",
    "    ax2 = ax1.twinx()\n",
    "    color_sync = '#e67e22' # Orange\n",
    "    ax2.set_ylabel('Global Synchrony', color=color_sync, fontsize=12)\n",
    "    ax2.plot(df_sweep['H_INERTIA'], df_sweep['sync'].rolling(window=15).mean(), \n",
    "             color=color_sync, linewidth=3, linestyle='-.', label='Synchrony')\n",
    "    ax2.tick_params(axis='y', labelcolor=color_sync)\n",
    "\n",
    "    plt.title('The Criticality Threshold: When High Synchrony Kills Accuracy', fontsize=15, pad=20)\n",
    "    ax1.grid(True, alpha=0.2)\n",
    "    \n",
    "    # Highlight the \"Goldilocks Zone\"\n",
    "    plt.axvspan(0.85, 0.93, color='yellow', alpha=0.1, label='Functional Regime')\n",
    "\n",
    "    fig.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "plot_synchrony_crash(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc28ea62",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def plot_rank_sync_h_inertia(df):\n",
    "    # 1. Sort by H_INERTIA to visualize the transition\n",
    "    df_sweep = df.sort_values('H_INERTIA')\n",
    "    \n",
    "    fig, ax1 = plt.subplots(figsize=(12, 6))\n",
    "\n",
    "    # --- PRIMARY Y-AXIS: EFFECTIVE RANK ---\n",
    "    color_rank = '#3498db' # Blue\n",
    "    ax1.set_xlabel('H_INERTIA', fontsize=12)\n",
    "    ax1.set_ylabel('Effective Rank (Dimensionality)', color=color_rank, fontsize=12)\n",
    "    \n",
    "    # We use a rolling mean (window=15) to see the trend through the Sobol noise\n",
    "    ax1.plot(df_sweep['H_INERTIA'], df_sweep['rank'].rolling(window=15).mean(), \n",
    "             color=color_rank, linewidth=3, label='Rank (Structural Complexity)')\n",
    "    ax1.scatter(df_sweep['H_INERTIA'], df_sweep['rank'], color=color_rank, alpha=0.1, s=10)\n",
    "    ax1.tick_params(axis='y', labelcolor=color_rank)\n",
    "\n",
    "    # --- SECONDARY Y-AXIS: SYNCHRONY ---\n",
    "    ax2 = ax1.twinx()\n",
    "    color_sync = '#e67e22' # Orange\n",
    "    ax2.set_ylabel('Global Synchrony (Binding)', color=color_sync, fontsize=12)\n",
    "    \n",
    "    ax2.plot(df_sweep['H_INERTIA'], df_sweep['sync'].rolling(window=15).mean(), \n",
    "             color=color_sync, linewidth=3, linestyle='-.', label='Synchrony')\n",
    "    ax2.tick_params(axis='y', labelcolor=color_sync)\n",
    "\n",
    "    # --- FORMATTING ---\n",
    "    plt.title('The Complexity-Synchrony Trade-off in Global Fields', fontsize=15, pad=20)\n",
    "    ax1.grid(True, alpha=0.2)\n",
    "    \n",
    "    # Highlight the \"Critical Window\" where Rank is high but Synchrony is still rising\n",
    "    # Adjust these numbers based on your specific results (0.85-0.93 was your sweet spot)\n",
    "    plt.axvspan(0.85, 0.93, color='grey', alpha=0.1, label='Metastable Regime')\n",
    "\n",
    "    # Combined Legend\n",
    "    lines, labels = ax1.get_legend_handles_labels()\n",
    "    lines2, labels2 = ax2.get_legend_handles_labels()\n",
    "    ax1.legend(lines + lines2, labels + labels2, loc='upper left')\n",
    "\n",
    "    fig.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# Run the analysis\n",
    "plot_rank_sync_h_inertia(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dc85f325",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "\n",
    "def plot_gf_distribution_sweep(df):\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    \n",
    "    # This creates a \"Ridge Plot\" feel by showing how the \n",
    "    # Global Synchrony (the GF's output) spreads out.\n",
    "    sns.kdeplot(data=df, x=\"H_INERTIA\", y=\"sync\", \n",
    "                cmap=\"magma\", fill=True, thresh=0, levels=30)\n",
    "    \n",
    "    plt.title(\"The GF State Transition Map\")\n",
    "    plt.xlabel(\"H_INERTIA (Inertia)\")\n",
    "    plt.ylabel(\"Global Synchrony (Field Strength)\")\n",
    "    plt.show()\n",
    "plot_gf_distribution_sweep(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f6d00ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def plot_acc_landscape_sweep(df):\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    \n",
    "    # We use 'viridis' or 'plasma' for accuracy to distinguish it from the 'magma' sync plot\n",
    "    # This creates a \"Topographical Map\" of your model's performance\n",
    "    sns.kdeplot(data=df, x=\"H_INERTIA\", y=\"acc\", \n",
    "                cmap=\"viridis\", fill=True, thresh=0, levels=30)\n",
    "    \n",
    "    # Overlay individual runs to see the outliers/sobol noise\n",
    "    plt.scatter(df['H_INERTIA'], df['acc'], color='white', s=1, alpha=0.2)\n",
    "    \n",
    "    plt.title(\"Functional Performance Landscape (Accuracy vs. Inertia)\", fontsize=14)\n",
    "    plt.xlabel(\"H_INERTIA (Global Field Inertia)\")\n",
    "    plt.ylabel(\"Accuracy\")\n",
    "    \n",
    "    # Vertical lines to mark the \"Functional Cliff\" you discovered\n",
    "    plt.axvline(x=0.93, color='red', linestyle='--', alpha=0.5, label='Bifurcation Point')\n",
    "    \n",
    "    plt.grid(True, alpha=0.1)\n",
    "    plt.show()\n",
    "\n",
    "plot_acc_landscape_sweep(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f420b783",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def plot_rank_landscape_sweep(df):\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    \n",
    "    # We'll use 'plasma' or 'magma' here to highlight the intensity of complexity\n",
    "    sns.kdeplot(data=df, x=\"H_INERTIA\", y=\"rank\", \n",
    "                cmap=\"plasma\", fill=True, thresh=0, levels=30)\n",
    "    \n",
    "    # Scatter overlay to see the Sobol sampling points\n",
    "    plt.scatter(df['H_INERTIA'], df['rank'], color='white', s=1, alpha=0.15)\n",
    "    \n",
    "    plt.title(\"Structural Complexity Landscape (Rank vs. Inertia)\", fontsize=14)\n",
    "    plt.xlabel(\"H_INERTIA (Stickiness)\")\n",
    "    plt.ylabel(\"Effective Rank (System Dimensionality)\")\n",
    "    \n",
    "    # Mark the collapse point\n",
    "    plt.axvline(x=0.93, color='cyan', linestyle='--', alpha=0.6, label='Complexity Collapse')\n",
    "    \n",
    "    plt.grid(True, alpha=0.1)\n",
    "    plt.show()\n",
    "\n",
    "plot_rank_landscape_sweep(df)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
