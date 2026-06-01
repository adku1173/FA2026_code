"""
Shared numpy core for CMF data generation.

Signal model (infinite-snapshot, single frequency)
---------------------------------------------------
    CSM  = H diag(x) H^H  +  σ² I_M
    y    = A @ x           +  n

where
    x   : source power vector  (N,),   Bernoulli × Rayleigh, non-negative
    y   : vectorized CSM       (M_eff,)
    A   : sensing matrix       (M_eff, N),  injected by caller
    n   : noise vector — σ² at the CSM-diagonal positions, 0 elsewhere
          (only relevant when r_diag=False)

Vectorisation convention (PhysicalModel / BeamformerCMF)
---------------------------------------------------------
With r_diag=False  →  M_eff = M²
    y = [real parts of lower-tri(CSM.T),  imag parts of off-diagonal lower-tri(CSM.T)]
    Diagonal entries of CSM appear at positions  i*(i+3)//2  for i = 0 … M-1.

With r_diag=True   →  M_eff = M(M-1)
    Diagonal rows/columns are stripped from both y and A.
    Sensor noise n vanishes entirely (no diagonal positions left in y).
"""

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_M(M_eff: int, r_diag: bool) -> int:
    """Infer the number of microphones M from the measurement dimension M_eff.

    Parameters
    ----------
    M_eff : int
        Number of rows in the sensing matrix / length of the vectorized CSM.
    r_diag : bool
        Whether the CSM diagonal has been removed.

    Returns
    -------
    int
        Number of microphones M.

    Raises
    ------
    ValueError
        If M_eff is not consistent with the given r_diag convention.
    """
    if r_diag:
        # M*(M-1) = M_eff  →  M² - M - M_eff = 0  →  M = (1 + √(1 + 4·M_eff)) / 2
        M = int(round((1.0 + np.sqrt(1.0 + 4.0 * M_eff)) / 2.0))
        if M * (M - 1) != M_eff:
            raise ValueError(
                f"M_eff={M_eff} is not of the form M*(M-1) for any integer M. "
                "When r_diag=True, A must have been built with diagonal removal "
                "(e.g. BeamformerCMF._calc_sensing_matrix with r_diag=True)."
            )
    else:
        # M² = M_eff
        M = int(round(np.sqrt(float(M_eff))))
        if M * M != M_eff:
            raise ValueError(
                f"M_eff={M_eff} is not a perfect square. "
                "When r_diag=False, A must have been built without diagonal removal "
                "(e.g. PhysicalModel.get_reduced_A())."
            )
    return M


def _diagonal_noise_indices(M: int) -> np.ndarray:
    """Return the positions in the vectorized y (r_diag=False) that correspond
    to the real parts of the CSM diagonal entries.

    Under the BeamformerCMF / PhysicalModel vectorisation, the i-th diagonal
    element of CSM appears at position  i*(i+3)//2  in the first (real) block
    of y.  The imaginary parts of diagonal elements are zero (CSM is Hermitian)
    and are excluded from y by the off-diagonal imaginary mask.

    Parameters
    ----------
    M : int
        Number of microphones.

    Returns
    -------
    ndarray of int, shape (M,)
    """
    return np.array([i * (i + 3) // 2 for i in range(M)])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CMFDataGenerator:
    """Core data generator for Covariance Matrix Fitting (CMF).

    Generates batches of ``(y, x)`` pairs representing the infinite-snapshot
    signal model at a single frequency:

        CSM = H diag(x) H^H  (+  σ² I  when snr_db is given and r_diag=False)
        y   = A @ x           (+  n     at CSM-diagonal positions)

    Parameters
    ----------
    A : array-like, shape (M_eff, N)
        Pre-built sensing matrix injected by the caller. Must be consistent
        with *r_diag*:

        - ``r_diag=False``: build via ``PhysicalModel.get_reduced_A()`` or
          ``BeamformerCMF._calc_sensing_matrix()`` with ``r_diag=False``.
          Then ``M_eff = M²``.
        - ``r_diag=True``: build via ``BeamformerCMF._calc_sensing_matrix()``
          with ``r_diag=True``. Then ``M_eff = M(M-1)``.

        A random matrix of the correct shape also works (colleague's notebook
        style).
    nsources : int
        Expected number of active sources. Controls sparsity via
        ``pnz = nsources / N``.
    batch_size : int
        Number of ``(y, x)`` samples per generated batch.
    r_diag : bool, optional
        If ``True`` (default), the CSM diagonal has been stripped from ``y``
        and ``A``; sensor noise cannot be injected.  If ``False``, the diagonal
        is retained and sensor noise is added when *snr_db* is given.
    snr_db : float or None, optional
        Signal-to-noise ratio in dB.  Per-sample noise power is
        ``σ² = sum(x) / 10^(snr_db/10)``, added only at the CSM-diagonal
        positions of ``y``.  Ignored (with a ``UserWarning``) when
        ``r_diag=True``.  ``None`` (default) → clean signal, no noise.
    seed : int or None, optional
        Random seed for reproducibility.
    precision : {'single', 'double'}, optional
        Floating-point precision.  Defaults to ``'single'`` (float32).

    Examples
    --------
    Random sensing matrix (colleague's notebook style):

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> M, N = 15, 30
    >>> A = rng.standard_normal((M * M, N)).astype(np.float32)  # r_diag=False
    >>> gen = CMFDataGenerator(A=A, nsources=3, batch_size=16,
    ...                        r_diag=False, snr_db=20.0, seed=42)
    >>> y, x = next(gen.generate())
    >>> y.shape, x.shape
    ((16, 225), (16, 30))
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
        self._float_dtype = np.float32 if precision == "single" else np.float64
        self.A = np.asarray(A, dtype=self._float_dtype)
        self.nsources = nsources
        self.batch_size = batch_size
        self.r_diag = r_diag
        self.snr_db = snr_db
        self.seed = seed
        self.precision = precision

        self.M_eff, self.N = self.A.shape
        self.pnz = nsources / self.N

        # Pre-compute diagonal noise injection indices.
        # Only needed when r_diag=False and snr_db is provided.
        if snr_db is not None:
            if r_diag:
                warnings.warn(
                    "snr_db is set but r_diag=True: the CSM diagonal is stripped, "
                    "so sensor noise has no effect on y. snr_db will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
                self._diag_idx = None
            else:
                M = _infer_M(self.M_eff, r_diag=False)
                self._diag_idx = _diagonal_noise_indices(M)
        else:
            self._diag_idx = None

    # ------------------------------------------------------------------
    # Private sampling helpers
    # ------------------------------------------------------------------

    def _rayleigh(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw Rayleigh(σ=1) samples via inverse-CDF, clipped away from 0."""
        u = rng.uniform(2.220446e-16, 1.0, size=size).astype(self._float_dtype)
        return np.sqrt(-2.0 * np.log(u)).astype(self._float_dtype)

    def _sample_x(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a batch of sparse non-negative source vectors.

        Returns
        -------
        ndarray, shape (batch_size, N), dtype = self._float_dtype
            Every row contains at least one non-zero entry.
        """
        B, N = self.batch_size, self.N
        size = B * N

        mask = rng.uniform(0.0, 1.0, size=size) < self.pnz
        amp = self._rayleigh(rng, size=size)
        x = np.where(mask, amp, self._float_dtype(0.0)).reshape(B, N)

        # Re-sample any all-zero rows (can happen when pnz is very small).
        all_zero = np.where(np.all(x == 0.0, axis=1))[0]
        while all_zero.size > 0:
            n_bad = all_zero.size
            mask_bad = rng.uniform(0.0, 1.0, size=n_bad * N) < self.pnz
            amp_bad = self._rayleigh(rng, size=n_bad * N)
            x[all_zero] = np.where(
                mask_bad, amp_bad, self._float_dtype(0.0)
            ).reshape(n_bad, N)
            all_zero = np.where(np.all(x == 0.0, axis=1))[0]

        return x

    # ------------------------------------------------------------------
    # Public generator
    # ------------------------------------------------------------------

    def generate(self):
        """Infinite generator of ``(y, x)`` batches.

        Yields
        ------
        y : ndarray, shape (batch_size, M_eff), dtype = float32 or float64
            Vectorized CSM (with optional diagonal sensor noise).
        x : ndarray, shape (batch_size, N), dtype = float32 or float64
            Source power vector (sparse, non-negative, un-normalised).

        Notes
        -----
        Call ``next(gen.generate())`` for a single batch, or iterate in a
        training loop::

            g = generator.generate()
            for step in range(max_steps):
                y, x = next(g)
        """
        rng = np.random.default_rng(seed=self.seed)

        while True:
            x = self._sample_x(rng)           # (B, N)
            y = x @ self.A.T                  # (B, M_eff)  — same as (A @ x.T).T

            if self._diag_idx is not None:
                # σ² = sum(x) / 10^(snr_db / 10),  shape (B,)
                signal_power = x.sum(axis=1).astype(self._float_dtype)
                sigma2 = signal_power / self._float_dtype(10.0 ** (self.snr_db / 10.0))
                # Broadcast: sigma2[:, None] is (B, 1) → adds σ² to each diagonal slot
                y[:, self._diag_idx] += sigma2[:, np.newaxis]

            yield y, x
