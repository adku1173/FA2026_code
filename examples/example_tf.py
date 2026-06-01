"""
CMF data generator — TensorFlow example
=========================================

Shows how to plug the physical data generator into a TensorFlow / Keras
training loop, matching the structure of the LISTA notebook.

The sensing matrix A is a plain tf.constant — pass it into your layer/model
the same way the notebook passes the random A.

Run:
    uv run python examples/example_tf.py
"""

import numpy as np
import tensorflow as tf

from fa2026.generators.generator_tf import make_tf_dataset
from fa2026.physical import build_sensing_matrix

# ── Sensing matrix ───────────────────────────────────────────────────────────
A_np = build_sensing_matrix(he=16., r_diag=False)   # (M², N) = (256, 256)
A    = tf.constant(A_np)                             # tf.Tensor, shape (M_eff, N)
M_eff, N = A.shape

# ── Dataset ──────────────────────────────────────────────────────────────────
ds = make_tf_dataset(
    A          = A_np,
    nsources   = 3,
    batch_size = 32,
    r_diag     = False,
    snr_db     = 20.0,
    seed       = 42,
)

# ── Minimal model (replace with your LISTA layer) ────────────────────────────
class DummyModel(tf.keras.Model):
    """One-step gradient descent step: x̂ = y A  (pseudo-inverse warm-start)."""
    def call(self, y, A):
        return y @ A                    # (B, N)

model     = DummyModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# ── Training loop ────────────────────────────────────────────────────────────
for step, (y, x) in enumerate(ds.take(5)):
    # y: (32, M_eff)  x: (32, N)

    with tf.GradientTape() as tape:
        x_hat = model(y, A)
        loss  = tf.reduce_mean(tf.square(x_hat - x))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(f"step {step:03d}  loss = {loss.numpy():.4e}")
