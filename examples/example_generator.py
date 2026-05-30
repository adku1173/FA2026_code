"""
CMF data generator — drop-in replacement for the random dictionary notebook
============================================================================

Copy the cells below into the LISTA notebook in place of the random A / x / y
generation block.

Requirements:  pip/uv install fa2026
"""

# ── Cell 1: imports ─────────────────────────────────────────────────────────
from fa2026 import CMFDataGenerator
from fa2026.physical import build_sensing_matrix

# ── Cell 2: sensing matrix (physical, acoular) ──────────────────────────────
# 64-mic TUB Vogel array · innermost 16 channels · 64×64 focus grid · he = 16
# r_diag=False → A shape (M², N) = (256, 4096)
A = build_sensing_matrix(he=16, r_diag=False)

# ── Cell 3: generator ───────────────────────────────────────────────────────
gen = CMFDataGenerator(
    A          = A,
    nsources   = 3,       # expected number of active sources
    batch_size = 32,
    r_diag     = False,   # must match the flag used to build A
    snr_db     = 20.0,    # sensor noise; set None for clean signal
    seed       = 42,
)
g = gen.generate()

# ── Cell 4: draw a batch ────────────────────────────────────────────────────
y, x = next(g)            # call next(g) again each training step

print(f"A : {A.shape}")   # (4096, 4096)
print(f"x : {x.shape}")   # (32, 4096)  — source power maps
print(f"y : {y.shape}")   # (32, 4096)  — vectorised CSM
