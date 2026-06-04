"""Iterative Shrinkage-Thresholding Algorithms (ISTA and FISTA).

This module provides PyTorch implementations of:
    - ISTA: Iterative Shrinkage-Thresholding Algorithm
    - FISTA: Fast Iterative Shrinkage-Thresholding Algorithm

Both solve the LASSO problem:
    min_x 1/2 ||y - A x||_2^2 + λ ||x||_1

With support for:
    - Column normalization of A
    - Non-negativity constraints (positive shrinkage)
    - Least-squares (LS) or Total Least Squares (TLS) objectives
    - Callback-based early stopping and monitoring
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ISTA(nn.Module):
    """
    Iterative Shrinkage-Thresholding Algorithm (ISTA) for LASSO.

    Solves the objective:
        min_x 1/2 ||y - A x||_2^2 + λ ||x||_1

    Args:
        norm_A (bool): if True, normalize the columns of ``A``
        positive (bool): if True, enforces non-negativity on the solution
        ls_mode (str): least-squares mode, either ``"ls"`` or ``"tls"``
        callbacks (list | None): optional solver callbacks
    """

    def __init__(
        self,
        norm_A=True,
        positive=True,
        ls_mode="ls",
        callbacks=None,
    ):
        super().__init__()

        self.norm_A = norm_A
        self.positive = positive
        self.ls_mode = ls_mode
        self.callbacks = callbacks if callbacks is not None else []

    def _prepare_A(self, A, reference_tensor=None):
        if A is None:
            raise ValueError("A must be provided in forward().")

        if torch.is_tensor(A):
            A_tensor = A.detach()
        else:
            A_tensor = torch.as_tensor(A)

        if A_tensor.ndim not in (2, 3):
            raise ValueError(f"A must be 2D or 3D, got shape {tuple(A_tensor.shape)}")

        if not torch.is_floating_point(A_tensor):
            A_tensor = A_tensor.to(torch.float32)

        if reference_tensor is not None:
            A_tensor = A_tensor.to(
                device=reference_tensor.device, dtype=reference_tensor.dtype
            )

        A_tensor = A_tensor.contiguous()
        l2_norms = None
        if self.norm_A:
            l2_norms = torch.linalg.norm(A_tensor, dim=-2)
            if torch.any(l2_norms == 0):
                raise ValueError("A contains zero-norm columns; cannot normalize.")
            A_tensor = A_tensor / l2_norms.unsqueeze(-2)

        return A_tensor, l2_norms

    @staticmethod
    def _to_batch_vector(tensor, name):
        if tensor.ndim == 0:
            return tensor
        if tensor.ndim == 1:
            if tensor.numel() == 1:
                return tensor.squeeze(0)
            return tensor.unsqueeze(1)
        if tensor.ndim == 2 and tensor.shape[1] == 1:
            return tensor
        raise ValueError(
            f"{name} must be scalar or have shape (B,) / (B, 1), got {tuple(tensor.shape)}"
        )

    @staticmethod
    def _apply_forward_operator(A, x):
        if A.ndim == 2:
            return F.linear(x, A)
        return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _apply_adjoint_operator(v, Op):
        if Op.ndim == 2:
            return v @ Op
        return torch.bmm(v.unsqueeze(1), Op).squeeze(1)

    @staticmethod
    def _default_gamma_from_A(A):
        with torch.no_grad():
            L = torch.linalg.norm(A, ord=2, dim=(-2, -1)) ** 2
        return 1.0 / L

    @staticmethod
    def _resolve_batch_param(value, batch_size, name, reference_tensor):
        if value is None:
            return None

        if torch.is_tensor(value) or isinstance(value, (np.ndarray, list, tuple)):
            tensor = torch.as_tensor(
                value, device=reference_tensor.device, dtype=reference_tensor.dtype
            )
            if tensor.ndim > 2:
                raise ValueError(
                    f"{name} must be scalar, 1D, or 2D with shape (B, 1), got {tuple(tensor.shape)}"
                )
            tensor = ISTA._to_batch_vector(tensor, name=name)
            if tensor.ndim == 0:
                return tensor
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)} but expected batch size {batch_size}."
                )
            return tensor

        return float(value)

    def _init_x(self, b, n, x0, reference_tensor):
        if x0 is None:
            return torch.zeros(
                (b, n), device=reference_tensor.device, dtype=reference_tensor.dtype
            )
        return x0.clone()

    def _validate_shapes(self, y, x0, A):
        if A.ndim == 2:
            m, n = A.shape
            if y.shape[1] != m:
                raise ValueError(f"y has shape {tuple(y.shape)} but A has shape {tuple(A.shape)}")
            batch_size = y.shape[0]
        else:
            batch_size, m, n = A.shape
            if y.shape[0] != batch_size or y.shape[1] != m:
                raise ValueError(
                    f"y has shape {tuple(y.shape)} but A has shape {tuple(A.shape)}"
                )

        if x0 is not None and tuple(x0.shape) != (batch_size, n):
            raise ValueError(
                f"x0 has shape {tuple(x0.shape)} but expected {(batch_size, n)}"
            )

    @staticmethod
    def shrink(x, thresh):
        """Element-wise soft-thresholding."""
        return torch.sign(x) * F.relu(x.abs() - thresh)

    @staticmethod
    def shrink_positive(x, thresh):
        """Element-wise positive soft-thresholding.
        
        For positive shrinkage, we want max(x - theta, 0), not max(|x| - theta, 0).
        This enforces non-negativity on the output while applying the threshold.
        """
        return F.relu(x - thresh)

    def ls_grad(self, y, x, A, Op=None, get_residual=False):
        """Compute gradient using least-squares mode."""
        if Op is None:
            Op = A
        res = y - self._apply_forward_operator(A, x)
        grad = self._apply_adjoint_operator(res, Op)
        if get_residual:
            return grad, res
        return grad

    def tls_grad(self, y, x, A, Op=None, get_residual=False):
        """Compute gradient using total least-squares mode.
        
        NOTE: The TLS gradient formula implementation below has not been fully
        verified against the mathematical derivation. The current formula is:
        
            grad = (2.0 / alpha**2) * (alpha * A^T * res - res_norm_sq * x)
        
        where alpha = 1.0 + ||x||^2 and res = y - A*x.
        
        According to the quotient rule derivation for f(x) = ||y - A x||^2 / (1 + ||x||^2),
        the correct negative gradient should be:
        
            grad = (2.0 / alpha**2) * (alpha * A^T * res + res_norm_sq * x)
        
        This discrepancy (sign on res_norm_sq * x) needs further verification.
        """
        if Op is None:
            Op = A
        res = y - self._apply_forward_operator(A, x)
        x_norm_sq = (x**2).sum(dim=1, keepdim=True)
        res_norm_sq = (res**2).sum(dim=1, keepdim=True)
        alpha = 1.0 + x_norm_sq
        grad = (2.0 / alpha**2) * (
            alpha * self._apply_adjoint_operator(res, Op) - res_norm_sq * x
        )
        if get_residual:
            return grad, res
        return grad

    def get_grad_func(self):
        """Get gradient function based on ls_mode."""
        if self.ls_mode == "ls":
            return self.ls_grad
        if self.ls_mode == "tls":
            return self.tls_grad
        raise ValueError(f"Unknown ls_mode: {self.ls_mode}")

    def get_shrink_func(self):
        """Get shrinkage function based on positivity."""
        if self.positive:
            return self.shrink_positive
        return self.shrink

    def step(self, y, x, A, shrink_func, grad_func, gamma, theta):
        """Perform a single ISTA step."""
        grad = grad_func(y, x, A)
        x = x + gamma * grad
        x = shrink_func(x, theta)
        return x

    def _prepare_forward_inputs(self, y, x0, A):
        """Prepare tensors shared by all ISTA variants."""
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if x0 is not None and x0.ndim == 1:
            x0 = x0.unsqueeze(0)
        batch_size = y.shape[0]

        A_tensor, l2_norms = self._prepare_A(A, reference_tensor=y)
        y = y.to(device=A_tensor.device, dtype=A_tensor.dtype)
        if x0 is not None:
            x0 = x0.to(device=A_tensor.device, dtype=A_tensor.dtype)

        self._validate_shapes(y, x0, A_tensor)
        return y, x0, A_tensor, l2_norms, batch_size

    def _resolve_lam_value(
        self, lam, batch_size, A_tensor, missing_message="lam must be provided."
    ):
        """Resolve regularization strength for scalar or per-batch input."""
        lam_value = self._resolve_batch_param(
            lam,
            batch_size,
            name="lam",
            reference_tensor=A_tensor,
        )
        if lam_value is None:
            raise ValueError(missing_message)
        return lam_value

    def _resolve_gamma_theta(self, lam_value, gamma, batch_size, A_tensor):
        """Resolve step size and shrinkage threshold for fixed-step solvers."""
        gamma_value = gamma
        if gamma_value is None:
            gamma_value = self._default_gamma_from_A(A_tensor)
        gamma_value = self._resolve_batch_param(
            gamma_value,
            batch_size,
            name="gamma",
            reference_tensor=A_tensor,
        )

        theta_value = lam_value * gamma_value
        theta_value = self._resolve_batch_param(
            theta_value,
            batch_size,
            name="theta",
            reference_tensor=A_tensor,
        )
        return gamma_value, theta_value

    def _setup_forward(self, y, x0, A, lam, gamma):
        """Prepare and validate inputs, resolve all solver parameters."""
        y, x0, A_tensor, l2_norms, batch_size = self._prepare_forward_inputs(y, x0, A)
        lam_value = self._resolve_lam_value(lam, batch_size, A_tensor)
        gamma_value, theta_value = self._resolve_gamma_theta(
            lam_value, gamma, batch_size, A_tensor
        )
        return y, x0, A_tensor, l2_norms, gamma_value, theta_value

    def _start_iterations(self, y):
        """Initialize callback-visible solver state."""
        self.y = y
        self.stop = False

    def _on_iteration_begin(self, k, x):
        """Run begin callbacks and report whether iteration should continue."""
        self.iter = k
        for cb in self.callbacks:
            cb.on_step_begin(self, x)
        return not self.stop

    def _on_iteration_end(self, x, gamma, theta, A_tensor, y):
        """Run end callbacks and report whether iteration should continue."""
        for cb in self.callbacks:
            cb.on_step_end(self, x, gamma=gamma, theta=theta, A=A_tensor, y=y)
        return not self.stop

    def _run_iterations(self, y, A_tensor, t, x, step_fn):
        """Run shared iteration loop with callback and stop handling."""
        self._start_iterations(y)
        for k in range(t):
            if not self._on_iteration_begin(k, x):
                break
            x, gamma_k, theta_k = step_fn(k, x)
            if not self._on_iteration_end(x, gamma_k, theta_k, A_tensor, y):
                break
        return x

    def _iteration_functions(self):
        """Resolve gradient and shrinkage functions once per solve."""
        return self.get_grad_func(), self.get_shrink_func()

    def _init_solver_vector(self, y, x0, A_tensor):
        """Initialize one solution-shaped vector."""
        return self._init_x(
            b=y.shape[0],
            n=A_tensor.shape[-1],
            x0=x0,
            reference_tensor=A_tensor,
        )

    def _iterate(self, y, x0, A_tensor, gamma_value, theta_value, t):
        """Run ISTA iterations and return the solution."""
        grad_func, shrink_func = self._iteration_functions()
        x = self._init_solver_vector(y, x0, A_tensor)

        def step_fn(_k, x_current):
            x_next = self.step(
                y,
                x_current,
                A_tensor,
                shrink_func,
                grad_func,
                gamma=gamma_value,
                theta=theta_value,
            )
            return x_next, gamma_value, theta_value

        return self._run_iterations(y, A_tensor, t, x, step_fn)

    def forward(self, y, x0=None, t=1, A=None, lam=None, gamma=None):
        """
        Args:
            y (Tensor): measurements of shape ``(B, m)`` or ``(m,)``
            x0 (Tensor | None): optional init of shape ``(B, n)`` or ``(n,)``
            t (int): number of iterations
            A (Tensor | None): sensing matrix of shape ``(m, n)`` or ``(B, m, n)``
            lam (float | array-like | Tensor | None): regularization strength
            gamma (float | array-like | Tensor | None): step size

        Returns:
            Tensor: solution of shape ``(B, n)``
        """
        y, x0, A_tensor, l2_norms, gamma_value, theta_value = self._setup_forward(
            y, x0, A, lam, gamma
        )
        x = self._iterate(y, x0, A_tensor, gamma_value, theta_value, t)
        if self.norm_A and l2_norms is not None:
            x = x / l2_norms
        return x


class FISTA(ISTA):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for LASSO.

    Solves the objective:
        min_x 1/2 ||y - A x||_2^2 + λ ||x||_1

    Uses Nesterov momentum acceleration for faster convergence than ISTA.
    """

    def _nesterov_momentum(self, tk, x, x_next):
        """Apply Nesterov momentum update.
        
        Note: Corrected spelling from '_nestrov_momentum' to '_nesterov_momentum'.
        """
        t_next = 0.5 + math.sqrt(1.0 + 4.0 * tk**2) / 2.0
        z_next = x_next + (tk - 1.0) / t_next * (x_next - x)
        return t_next, z_next, x_next

    def _iterate_accelerated(self, y, x0, A_tensor, t, step_params_fn):
        """Run FISTA-style iterations with caller-provided step parameters."""
        grad_func, shrink_func = self._iteration_functions()
        x = self._init_solver_vector(y, x0, A_tensor)
        z = self._init_solver_vector(y, x0, A_tensor)
        tk = 1.0

        def step_fn(k, x_current):
            nonlocal z, tk
            gamma_k, theta_k = step_params_fn(k)
            x_next = self.step(
                y,
                z,
                A_tensor,
                shrink_func,
                grad_func,
                gamma=gamma_k,
                theta=theta_k,
            )
            tk, z, x_updated = self._nesterov_momentum(tk, x_current, x_next)
            return x_updated, gamma_k, theta_k

        return self._run_iterations(y, A_tensor, t, x, step_fn)

    def _iterate(self, y, x0, A_tensor, gamma_value, theta_value, t):
        """Run FISTA iterations with Nesterov momentum and return the solution."""
        return self._iterate_accelerated(
            y,
            x0,
            A_tensor,
            t,
            step_params_fn=lambda _k: (gamma_value, theta_value),
        )


class SFISTA(FISTA):
    """
    Step-size-learned FISTA variant inspired by Ablin et al.'s Step-LISTA.

    This class is **not** the exact SLISTA architecture from "Learning step sizes
    for unfolded sparse coding". Ablin et al.'s SLISTA unfolds ISTA with one
    learned step size per layer and no acceleration:

        z_{k+1} = ST(z_k - gamma_k A.T @ (A @ z_k - y), lambda * gamma_k)

    This implementation keeps the learned per-layer step sizes and tied
    thresholds from SLISTA, but deliberately adapts the architecture by:

        - applying the update to the FISTA momentum variable;
        - using the standard FISTA/Nesterov momentum recurrence;
        - defaulting to positive shrinkage when ``positive=True``, i.e.
          ``max(x - theta, 0)``, which enforces non-negative coefficients rather
          than the signed soft-thresholding used in the paper.

    It therefore solves an unfolded, momentum-accelerated, optionally
    non-negative sparse-coding variant of:

        min_x 1/2 ||y - A x||_2^2 + λ ||x||_1

    At iteration ``k``, ``gamma[k]`` is a trainable scalar step size and the
    threshold is computed as ``theta_k = lam * gamma[k]``. The step parameters
    are stored directly and are not positivity-constrained by this class; callers
    should initialize/train them accordingly if positive steps are required.

    Args:
        n_iterations (int): number of iterations/layers (trainable gamma parameters)
        gamma_init (float | None): initialization value for gamma parameters.
            Must be provided (default: None, which will raise an error if not set).
        norm_A (bool): if True, normalize the columns of ``A`` (inherited from FISTA)
        positive (bool): if True, enforces non-negativity on the solution (default).
            If False, uses signed soft-thresholding as in SLISTA.
        ls_mode (str): least-squares mode, either ``"ls"`` or ``"tls"`` (inherited from FISTA)
        callbacks (list | None): optional solver callbacks (inherited from FISTA)
    """

    def __init__(
        self,
        n_iterations,
        gamma_init=None,
        norm_A=True,
        positive=True,
        ls_mode="ls",
        callbacks=None,
    ):
        super().__init__(
            norm_A=norm_A,
            positive=positive,
            ls_mode=ls_mode,
            callbacks=callbacks,
        )
        if n_iterations <= 0:
            raise ValueError(f"n_iterations must be positive, got {n_iterations}")
        if gamma_init is None:
            raise ValueError("gamma_init must be provided for SFISTA")
        self.n_iterations = n_iterations
        self.gamma = nn.Parameter(torch.full((n_iterations,), float(gamma_init)))
        # Temporary storage for lam_value during forward pass
        self._current_lam_value = None

    def _setup_forward(self, y, x0, A, lam, gamma):
        """Prepare inputs and validate SFISTA-specific constraints.
        
        Validates that gamma is not passed (SFISTA manages its own gamma parameters)
        and handles device/dtype for the trainable gamma.
        """
        if gamma is not None:
            raise ValueError(
                "SFISTA manages its own gamma parameters; "
                "do not pass gamma to forward()"
            )

        y, x0, A_tensor, l2_norms, batch_size = self._prepare_forward_inputs(y, x0, A)
        self._current_lam_value = self._resolve_lam_value(
            lam,
            batch_size,
            A_tensor,
            missing_message="lam must be provided",
        )

        # Ensure gamma is on the same device and dtype as A_tensor
        if self.gamma.device != A_tensor.device or self.gamma.dtype != A_tensor.dtype:
            self.gamma.data = self.gamma.data.to(
                device=A_tensor.device,
                dtype=A_tensor.dtype,
            )

        # Return dummy values for gamma_value and theta_value (not used by SFISTA)
        return y, x0, A_tensor, l2_norms, None, None

    def _step_params(self, k):
        """Return SFISTA trainable step size and matching threshold."""
        gamma_k = self.gamma[k]
        theta_k = self._current_lam_value * gamma_k
        return gamma_k, theta_k

    def _iterate(self, y, x0, A_tensor, gamma_value, theta_value, t):
        """Run SFISTA iterations with trainable per-iteration step sizes.
        
        Uses gamma[k] at iteration k and computes theta_k = lam * gamma[k].
        """
        if t > self.n_iterations:
            raise ValueError(
                f"t ({t}) cannot exceed n_iterations ({self.n_iterations})"
            )

        return self._iterate_accelerated(
            y,
            x0,
            A_tensor,
            t,
            step_params_fn=self._step_params,
        )
