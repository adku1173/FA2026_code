"""
CMF data generator — FISTA example
==================================

Solves y = A x with FISTA and plots the result.

Run:  uv run python examples/example_torch.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from fa2026.generators.generator_torch import CMFTorchDataset
from fa2026.physical import build_sensing_matrix, vogel_subarray
from fa2026.optim import FISTA

# ── Setup ────────────────────────────────────────────────────────────────────
he = 16
n_grid = 16
nsources = 5
lambda_val = 0.1

mics = vogel_subarray(n_mics=16)
A = torch.as_tensor(build_sensing_matrix(he=he, n_grid=n_grid, r_diag=False))

# ── Generate data ────────────────────────────────────────────────────────────
dataset = CMFTorchDataset(A=A, nsources=nsources, batch_size=1, snr_db=20.0, seed=42)
y, x_gt = next(iter(dataset))  # y: (1, M_eff), x: (1, N)

# ── Solve with FISTA ────────────────────────────────────────────────────────
fista = FISTA(norm_A=False, positive=True)
with torch.no_grad():
    x_hat = fista(y=y, A=A.unsqueeze(0), lam=lambda_val, t=100)
x_hat = x_hat.squeeze(0)
x_gt = x_gt.squeeze(0)

# ── Plot ────────────────────────────────────────────────────────────────────
cmap = plt.cm.hot_r
cmap._init()
cmap._lut[-3, -1] = 0  # transparent background

grid_size = int(np.sqrt(x_gt.numel()))
extent = [-0.5 * mics.aperture, 0.5 * mics.aperture] * 2

x_gt_map = x_gt.numpy().reshape(grid_size, grid_size)
x_hat_map = x_hat.numpy().reshape(grid_size, grid_size)

L_gt = 10 * np.log10(x_gt_map + 1e-20)
L_hat = 10 * np.log10(x_hat_map + 1e-20)

vmax = max(L_gt.max(), L_hat.max())
delta_Lp = 20

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, L, title in zip(axes, [L_gt, L_hat], ['Ground Truth', 'FISTA']):
    im = ax.imshow(L.T, origin='lower', cmap=cmap, vmin=vmax - delta_Lp,
                   vmax=vmax, extent=extent, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('$x$ / m')
    ax.set_ylabel('$y$ / m')

fig.colorbar(im, ax=axes, label='SPL / dB')
plt.suptitle(f'λ = {lambda_val}, t = 100')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fista_sourcemap.png', dpi=150, bbox_inches='tight')
plt.show()
