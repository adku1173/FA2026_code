"""
Tests for fa2026.generator (core), fa2026.generator_torch, fa2026.generator_tf.

Run with:
    pip install -e ".[dev]"
    pytest
"""

import numpy as np
import pytest

from fa2026.generators.generator import (
    CMFDataGenerator,
    _diagonal_noise_indices,
    _infer_M,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_A(M_eff: int, N: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((M_eff, N)).astype(np.float32)


def A_no_rdiag(M: int, N: int, seed: int = 0) -> np.ndarray:
    """Sensing matrix consistent with r_diag=False  (M_eff = M²)."""
    return _make_A(M * M, N, seed)


def A_rdiag(M: int, N: int, seed: int = 0) -> np.ndarray:
    """Sensing matrix consistent with r_diag=True  (M_eff = M*(M-1))."""
    return _make_A(M * (M - 1), N, seed)


# ---------------------------------------------------------------------------
# _infer_M
# ---------------------------------------------------------------------------

class TestInferM:
    @pytest.mark.parametrize("M", [4, 8, 16, 32])
    def test_r_diag_false(self, M):
        assert _infer_M(M * M, r_diag=False) == M

    @pytest.mark.parametrize("M", [4, 8, 16, 32])
    def test_r_diag_true(self, M):
        assert _infer_M(M * (M - 1), r_diag=True) == M

    def test_invalid_r_diag_false_raises(self):
        with pytest.raises(ValueError, match="perfect square"):
            _infer_M(15, r_diag=False)

    def test_invalid_r_diag_true_raises(self):
        with pytest.raises(ValueError, match=r"M\*\(M-1\)"):
            _infer_M(16, r_diag=True)  # 16 = 4*4, not M*(M-1) for any M


# ---------------------------------------------------------------------------
# _diagonal_noise_indices
# ---------------------------------------------------------------------------

class TestDiagonalNoiseIndices:
    def test_known_values(self):
        # M=4: lower-tri of CSM.T (row-major) diagonal positions are [0, 2, 5, 9]
        np.testing.assert_array_equal(_diagonal_noise_indices(4), [0, 2, 5, 9])

    def test_known_values_m3(self):
        np.testing.assert_array_equal(_diagonal_noise_indices(3), [0, 2, 5])

    @pytest.mark.parametrize("M", [2, 4, 8, 16])
    def test_count(self, M):
        assert len(_diagonal_noise_indices(M)) == M

    @pytest.mark.parametrize("M", [2, 4, 8])
    def test_within_bounds(self, M):
        M_eff = M * M
        idx = _diagonal_noise_indices(M)
        assert np.all(idx >= 0)
        assert np.all(idx < M * (M + 1) // 2)  # within the real block


# ---------------------------------------------------------------------------
# CMFDataGenerator — output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_r_diag_false(self):
        M, N, B = 4, 16, 8
        gen = CMFDataGenerator(A=A_no_rdiag(M, N), nsources=2, batch_size=B, r_diag=False)
        y, x = next(gen.generate())
        assert y.shape == (B, M * M)
        assert x.shape == (B, N)

    def test_r_diag_true(self):
        M, N, B = 4, 16, 8
        gen = CMFDataGenerator(A=A_rdiag(M, N), nsources=2, batch_size=B, r_diag=True)
        y, x = next(gen.generate())
        assert y.shape == (B, M * (M - 1))
        assert x.shape == (B, N)


# ---------------------------------------------------------------------------
# CMFDataGenerator — x properties
# ---------------------------------------------------------------------------

class TestXProperties:
    def test_nonnegative(self):
        gen = CMFDataGenerator(A=A_no_rdiag(4, 16), nsources=2, batch_size=32, r_diag=False)
        y, x = next(gen.generate())
        assert np.all(x >= 0.0)

    def test_no_all_zero_rows(self):
        # Very sparse (nsources=1, N=100) stresses the re-sampling loop.
        gen = CMFDataGenerator(
            A=A_no_rdiag(4, 100), nsources=1, batch_size=32, r_diag=False, seed=0
        )
        g = gen.generate()
        for _ in range(5):
            _, x = next(g)
            assert not np.any(np.all(x == 0.0, axis=1))

    def test_sparsity_approx(self):
        M, N, B = 4, 50, 256
        nsources = 3
        gen = CMFDataGenerator(
            A=A_no_rdiag(M, N), nsources=nsources, batch_size=B, r_diag=False, seed=1
        )
        _, x = next(gen.generate())
        actual_pnz = (x > 0).mean()
        expected_pnz = nsources / N
        # Large batch → should be within 10 percentage points
        assert abs(actual_pnz - expected_pnz) < 0.10


# ---------------------------------------------------------------------------
# CMFDataGenerator — clean signal model (snr_db=None)
# ---------------------------------------------------------------------------

class TestCleanSignal:
    def test_y_equals_Ax(self):
        M, N, B = 4, 16, 8
        A = A_no_rdiag(M, N)
        gen = CMFDataGenerator(A=A, nsources=2, batch_size=B, r_diag=False, snr_db=None, seed=7)
        y, x = next(gen.generate())
        np.testing.assert_allclose(y, x @ A.T, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# CMFDataGenerator — noise injection (snr_db given, r_diag=False)
# ---------------------------------------------------------------------------

class TestNoiseInjection:
    def test_noise_only_at_diagonal_positions(self):
        M, N, B = 4, 16, 8
        A = A_no_rdiag(M, N)
        kwargs = dict(A=A, nsources=2, batch_size=B, r_diag=False, seed=42)
        y_clean, x_clean = next(CMFDataGenerator(**kwargs, snr_db=None).generate())
        y_noisy, x_noisy = next(CMFDataGenerator(**kwargs, snr_db=20.0).generate())

        # Same seed → same x
        np.testing.assert_array_equal(x_clean, x_noisy)

        diag_idx = _diagonal_noise_indices(M)
        off_mask = np.ones(M * M, dtype=bool)
        off_mask[diag_idx] = False

        diff = y_noisy - y_clean
        np.testing.assert_array_equal(diff[:, off_mask], 0.0)
        assert np.all(diff[:, diag_idx] > 0.0)

    def test_snr_value_correct(self):
        M, N, B = 4, 16, 8
        snr_db = 20.0
        A = A_no_rdiag(M, N)
        kwargs = dict(A=A, nsources=2, batch_size=B, r_diag=False, seed=42)
        y_noisy, x = next(CMFDataGenerator(**kwargs, snr_db=snr_db).generate())
        y_clean, _ = next(CMFDataGenerator(**kwargs, snr_db=None).generate())

        diag_idx = _diagonal_noise_indices(M)
        sigma2_actual   = y_noisy[:, diag_idx[0]] - y_clean[:, diag_idx[0]]
        sigma2_expected = x.sum(axis=1) / 10.0 ** (snr_db / 10.0)

        np.testing.assert_allclose(sigma2_actual, sigma2_expected, rtol=1e-5)

    def test_all_diag_entries_get_same_sigma2(self):
        """All M diagonal slots of the same sample receive the same σ²."""
        M, N, B = 4, 16, 8
        snr_db = 15.0
        A = A_no_rdiag(M, N)
        kwargs = dict(A=A, nsources=2, batch_size=B, r_diag=False, seed=42)
        y_noisy, x = next(CMFDataGenerator(**kwargs, snr_db=snr_db).generate())
        y_clean, _ = next(CMFDataGenerator(**kwargs, snr_db=None).generate())

        diag_idx = _diagonal_noise_indices(M)
        noise_at_diag = y_noisy[:, diag_idx] - y_clean[:, diag_idx]   # (B, M)
        # All M columns of noise_at_diag should be identical (same σ² per sample)
        # Compare each column against the first using explicit broadcasting
        for col in range(1, M):
            np.testing.assert_allclose(
                noise_at_diag[:, col], noise_at_diag[:, 0], rtol=1e-5
            )

    def test_snr_ignored_when_r_diag_true(self):
        A = A_rdiag(4, 16)
        with pytest.warns(UserWarning, match="r_diag=True"):
            gen = CMFDataGenerator(A=A, nsources=2, batch_size=8, r_diag=True, snr_db=20.0)
        # No noise applied: y must equal A @ x
        y, x = next(gen.generate())
        np.testing.assert_allclose(y, x @ A.T, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# CMFDataGenerator — reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_output(self):
        A = A_no_rdiag(4, 16)
        kw = dict(A=A, nsources=2, batch_size=8, r_diag=False, seed=99)
        g1 = CMFDataGenerator(**kw).generate()
        g2 = CMFDataGenerator(**kw).generate()
        for _ in range(3):
            y1, x1 = next(g1)
            y2, x2 = next(g2)
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        A = A_no_rdiag(4, 16)
        y1, x1 = next(CMFDataGenerator(A=A, nsources=2, batch_size=32, r_diag=False, seed=1).generate())
        y2, x2 = next(CMFDataGenerator(A=A, nsources=2, batch_size=32, r_diag=False, seed=2).generate())
        assert not np.array_equal(x1, x2)


# ---------------------------------------------------------------------------
# CMFDataGenerator — precision / dtype
# ---------------------------------------------------------------------------

class TestPrecision:
    def test_single(self):
        A = A_no_rdiag(4, 16)
        gen = CMFDataGenerator(A=A, nsources=2, batch_size=8, r_diag=False, precision="single")
        y, x = next(gen.generate())
        assert y.dtype == np.float32
        assert x.dtype == np.float32

    def test_double(self):
        A = A_no_rdiag(4, 16).astype(np.float64)
        gen = CMFDataGenerator(A=A, nsources=2, batch_size=8, r_diag=False, precision="double")
        y, x = next(gen.generate())
        assert y.dtype == np.float64
        assert x.dtype == np.float64


# ---------------------------------------------------------------------------
# Cross-framework: PyTorch vs TensorFlow must give identical output
# ---------------------------------------------------------------------------

def test_torch_tf_identical():
    """Given the same seed, PyTorch and TF variants must produce bit-identical output."""
    try:
        from fa2026.generators.generator_torch import CMFTorchDataset
    except ImportError:
        pytest.skip("torch not installed")

    try:
        from fa2026.generators.generator_tf import make_tf_dataset
    except ImportError:
        pytest.skip("tensorflow not installed")

    A = A_no_rdiag(M=4, N=16, seed=7)
    kwargs = dict(A=A, nsources=2, batch_size=8, r_diag=False, snr_db=20.0, seed=42)

    # PyTorch
    y_torch, x_torch = next(iter(CMFTorchDataset(**kwargs)))
    y_torch = y_torch.numpy()
    x_torch = x_torch.numpy()

    # TensorFlow
    y_tf, x_tf = next(iter(make_tf_dataset(**kwargs)))
    y_tf = y_tf.numpy()
    x_tf = x_tf.numpy()

    np.testing.assert_array_equal(x_torch, x_tf)
    np.testing.assert_array_equal(y_torch, y_tf)
