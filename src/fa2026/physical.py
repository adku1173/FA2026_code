"""
Physical sensing matrix construction from acoular steering vectors.

Follows the PhysicalModel / BeamformerCMF vectorisation convention:
  y = [real parts of lower-tri(CSM.T),  imag parts of off-diagonal lower-tri(CSM.T)]

so that  y = A @ x  with x the source power vector.
"""

from pathlib import Path

import acoular as ac
import numpy as np


def vogel_subarray(n_mics: int = 16):
    """Return a MicGeom containing the *n_mics* innermost channels of the
    TUB Vogel 64 array, selected by ascending radial distance from the centre.

    This is equivalent to iteratively removing the outermost microphone until
    *n_mics* channels remain.

    Positions are in physical metres (no aperture normalisation).

    Parameters
    ----------
    n_mics : int, optional
        Number of microphones to keep.  Must satisfy ``1 <= n_mics <= 64``.
        Default is 16.

    Returns
    -------
    ac.MicGeom
        Microphone geometry with *n_mics* channels, positions in metres.

    """
    if not 1 <= n_mics <= 64:
        raise ValueError(f"n_mics must be between 1 and 64, got {n_mics}")

    full = ac.MicGeom(file=Path(ac.__file__).parent / "xml" / "tub_vogel64.xml")

    pos  = full.pos_total             # (3, 64), metres
    r    = np.linalg.norm(pos[:2], axis=0)   # radial distance in the xy-plane
    keep = np.argsort(r)[:n_mics]    # indices of the n_mics closest channels

    sub           = ac.MicGeom()
    sub.pos_total = pos[:, keep]      # raw physical positions, no normalisation
    return sub


def build_sensing_matrix(
    he,
    mics=None,
    n_grid: int = 16,
    env=None,
    r_diag: bool = False,
    precision: str = "single",
) -> np.ndarray:
    """Build the physical CMF sensing matrix A from acoular steering vectors.

    Implements the same vectorisation as ``PhysicalModel.get_reduced_A()``
    (``r_diag=False``) and ``BeamformerCMF._calc_sensing_matrix()``
    (``r_diag=True``).

    Parameters
    ----------
    he : float
        Helmholtz number  ``he = f * aperture / c``, where ``aperture`` is
        the physical aperture of *mics* in metres.
    mics : ac.MicGeom, optional
        Microphone geometry in physical metres.  Defaults to the 16-channel
        innermost subarray of the TUB Vogel 64 array
        (see :func:`vogel_subarray`).
    n_grid : int, optional
        Number of grid points per axis.  The focus grid always spans
        [−0.5·ap, 0.5·ap] × [−0.5·ap, 0.5·ap] at z = 0.75 m, where
        ``ap = mics.aperture``.  Increment is set to ``ap / (n_grid − 1)``.
        Default is 16 (→ 16 × 16 = 256 grid points, matching the default
        16-mic array so that A is square when ``r_diag=False``).
    env : ac.Environment, optional
        Acoustic environment (speed of sound etc.).
        Defaults to ``ac.Environment()`` (c = 343 m/s, dry air).
    r_diag : bool, optional
        If ``False`` (default): keep CSM diagonal → ``M_eff = M²``.
        If ``True``:  strip CSM diagonal → ``M_eff = M(M−1)``.
    precision : {'single', 'double'}, optional
        Output dtype.  Default ``'single'`` (float32).

    Returns
    -------
    A : ndarray, shape (M_eff, N)
        Real-valued sensing matrix ready for injection into
        :class:`~fa2026.generator.CMFDataGenerator`.

    """
    # ── defaults ───────────────────────────────────────────────────────────
    if mics is None:
        mics = vogel_subarray(n_mics=16)

    if env is None:
        env = ac.Environment()

    ap = mics.aperture
    grid = ac.RectGrid(
        x_min=-0.5 * ap, x_max=0.5 * ap,
        y_min=-0.5 * ap, y_max=0.5 * ap,
        z=0.75,                          # 0.5 × full Vogel 64 aperture (≈ 1.5 m)
        increment=ap / (n_grid - 1),
    )

    steer = ac.SteeringVector(grid=grid, mics=mics, env=env)
    f = he * steer.env.c / mics.aperture   # he = f · aperture / c

    nc = mics.num_mics
    N  = grid.size

    # ── Kronecker product of steering vectors (shape: nc*nc × N) ───────────
    h  = steer.transfer(f).T                          # (nc, N)
    Bc = (
        h[:, :, np.newaxis] * h.conjugate().T[np.newaxis, :, :]
    ).transpose(2, 0, 1)                              # (N, nc, nc)
    Ac = Bc.reshape(nc * nc, N)                       # (nc², N)

    # ── Vectorisation indices ───────────────────────────────────────────────
    ind     = np.reshape(np.tril(np.ones((nc, nc))), (nc * nc,)) > 0
    ind_im0 = (np.reshape(np.eye(nc), (nc * nc,)) == 0)[ind]

    if r_diag:
        # Strip diagonal → M_eff = nc*(nc-1)
        ind_reim = np.hstack([ind_im0, ind_im0])
    else:
        # Keep diagonal → M_eff = nc²
        ind_reim = np.hstack([np.ones(ind_im0.size, dtype=bool), ind_im0])
        ind_reim[0] = True

    A_complex = Ac[ind, :]                            # lower-tri block
    A = np.vstack([A_complex.real, A_complex.imag])[ind_reim, :]

    dtype = np.float32 if precision == "single" else np.float64
    return A.astype(dtype)
