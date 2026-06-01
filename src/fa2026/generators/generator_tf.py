"""TensorFlow tf.data.Dataset factory wrapping CMFDataGenerator.

Usage
-----
    from fa2026.generator_tf import make_tf_dataset

    ds = make_tf_dataset(A=A, nsources=3, batch_size=32, r_diag=False)

    # The dataset already yields pre-batched (y, x) tensors.
    for y, x in ds.take(max_steps):
        ...

    # Access the sensing matrix as a tf.Tensor
    A_tf = tf.constant(A)
"""

import numpy as np
import tensorflow as tf

from ..generators.generator import CMFDataGenerator


def make_tf_dataset(
    A,
    nsources: int,
    batch_size: int,
    r_diag: bool = True,
    snr_db=None,
    seed=None,
    precision: str = "single",
) -> tf.data.Dataset:
    """Create an infinite ``tf.data.Dataset`` yielding ``(y, x)`` batches.

    Parameters are identical to :class:`~fa2026.generator.CMFDataGenerator`.

    Parameters
    ----------
    A : array-like, shape (M_eff, N)
        Pre-built sensing matrix. See :class:`~fa2026.generator.CMFDataGenerator`
        for the ``r_diag`` conventions.
    nsources : int
        Expected number of active sources.
    batch_size : int
        Number of samples per yielded batch.
    r_diag : bool, optional
        If ``True`` (default), CSM diagonal is stripped from ``y`` and ``A``.
    snr_db : float or None, optional
        SNR in dB. Ignored (with a warning) when ``r_diag=True``.
    seed : int or None, optional
        Random seed.
    precision : {'single', 'double'}, optional
        Floating-point precision. Defaults to ``'single'``.

    Returns
    -------
    tf.data.Dataset
        Infinite dataset yielding ``(y, x)`` tuples of tensors with shapes
        ``(batch_size, M_eff)`` and ``(batch_size, N)``.

    Notes
    -----
    The dataset already yields *pre-batched* tensors — do **not** call
    ``.batch()`` on the returned dataset unless you want a batch-of-batches.
    """
    core = CMFDataGenerator(
        A=A,
        nsources=nsources,
        batch_size=batch_size,
        r_diag=r_diag,
        snr_db=snr_db,
        seed=seed,
        precision=precision,
    )

    tf_dtype = tf.float32 if precision == "single" else tf.float64
    M_eff = core.M_eff
    N = core.N

    def _gen():
        for y, x in core.generate():
            yield y, x

    return tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, M_eff), dtype=tf_dtype),
            tf.TensorSpec(shape=(batch_size, N),     dtype=tf_dtype),
        ),
    )
