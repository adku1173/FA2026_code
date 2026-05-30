"""PyTorch IterableDataset wrapping CMFDataGenerator.

Usage
-----
    from fa2026.generator_torch import CMFTorchDataset

    dataset = CMFTorchDataset(A=A, nsources=3, batch_size=32, r_diag=False)

    # Option 1 — iterate directly (recommended; dataset already yields batches)
    g = iter(dataset)
    for step in range(max_steps):
        y, x = next(g)          # tensors of shape (32, M_eff) and (32, N)
        ...

    # Option 2 — DataLoader in pass-through mode (no additional batching)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=None)
    for y, x in loader:
        ...
"""

import numpy as np
import torch
import torch.utils.data as data

from .generator import CMFDataGenerator


class CMFTorchDataset(data.IterableDataset):
    """PyTorch IterableDataset that wraps :class:`~fa2026.generator.CMFDataGenerator`.

    Each iteration step yields one pre-batched ``(y, x)`` tuple of tensors.
    Parameters are identical to :class:`~fa2026.generator.CMFDataGenerator`.

    Attributes
    ----------
    A_tensor : torch.Tensor, shape (M_eff, N)
        The sensing matrix as a torch tensor (convenient for model initialisation).

    Notes
    -----
    The dataset yields *batches*, not individual samples. When using
    ``torch.utils.data.DataLoader``, set ``batch_size=None`` to disable
    DataLoader's own collation and avoid an extra batch dimension.
    """

    def __init__(
        self,
        A,
        nsources: int,
        batch_size: int,
        r_diag: bool = True,
        snr_db=None,
        seed=None,
        precision: str = "single",
    ):
        super().__init__()
        self._core = CMFDataGenerator(
            A=A,
            nsources=nsources,
            batch_size=batch_size,
            r_diag=r_diag,
            snr_db=snr_db,
            seed=seed,
            precision=precision,
        )
        self._dtype = torch.float32 if precision == "single" else torch.float64

    @property
    def A_tensor(self) -> torch.Tensor:
        """The sensing matrix as a torch.Tensor."""
        return torch.as_tensor(self._core.A, dtype=self._dtype)

    @property
    def M_eff(self) -> int:
        return self._core.M_eff

    @property
    def N(self) -> int:
        return self._core.N

    def __iter__(self):
        for y_np, x_np in self._core.generate():
            yield (
                torch.as_tensor(y_np, dtype=self._dtype),
                torch.as_tensor(x_np, dtype=self._dtype),
            )
