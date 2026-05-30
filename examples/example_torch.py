"""
CMF data generator — PyTorch example
======================================

Shows how to plug the physical data generator into a PyTorch training loop.
The sensing matrix A_tensor is available directly from the dataset and can be
passed into the model (e.g. as the fixed forward operator in LISTA / FISTA).

Run:
    uv run python examples/example_torch.py
"""

import torch
import torch.nn as nn

from fa2026.generator_torch import CMFTorchDataset
from fa2026.physical import build_sensing_matrix

he = 16

# ── Sensing matrix ───────────────────────────────────────────────────────────
A_np = build_sensing_matrix(he=he, r_diag=False)   # (M², N) = (256, 4096)

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = CMFTorchDataset(
    A          = A_np,
    nsources   = 3,
    batch_size = 32,
    r_diag     = False,
    snr_db     = 20.0,
    seed       = 42,
)

# Sensing matrix as a torch tensor — pass this into your model.
A = dataset.A_tensor                    # (M_eff, N)

# ── Minimal model (replace with your LISTA / unfolded network) ───────────────
# This toy model has one learnable weight matrix W initialised to A^T.
class LinearStep(nn.Module):
    """Single learned linear step: x̂ = y W."""
    def __init__(self, M_eff, N):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(M_eff, N))

    def forward(self, y):
        return y @ self.W               # (B, N)

M_eff, N = A.shape
model     = LinearStep(M_eff, N)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── Training loop ────────────────────────────────────────────────────────────
g = iter(dataset)                       # single iterator over the infinite stream

for step in range(5):
    y, x = next(g)                      # y: (32, M_eff), x: (32, N)

    x_hat = model(y)
    loss  = torch.nn.functional.mse_loss(x_hat, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"step {step:03d}  loss = {loss.item():.4e}")
