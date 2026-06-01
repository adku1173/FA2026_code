"""Callback infrastructure for iterative solvers.

Classes
-------
IterationCallback
    Base class for solver callbacks.
SolutionChangeStopper
    Stop solver when iterate changes fall below tolerance.
StepParamCollector
    Record step parameters (gamma, theta) at each iteration.
IntermediateOutputCollector
    Record intermediate iterates with computational graph attached.
"""

import torch


class IterationCallback:
    """Base class for callbacks of the solver

    Subclass and override `on_step_begin`/`on_step_end`.
    Methods receive the solver instance and the current iterate `x`.
    """

    def on_step_begin(self, obj, x, **kwargs):
        """Called at the beginning of each iteration."""
        pass

    def on_step_end(self, obj, x, **kwargs):
        """Called at the end of each iteration."""
        pass


class StepParamCollector(IterationCallback):
    """Callback that records the step sizes and thresholds at each iteration.

    Attach to a solver via ``solver.callbacks = [collector]``.  The solver
    passes ``gamma`` and ``theta`` as keyword arguments to :meth:`on_step_end`;
    the callback stores a detached copy of each tensor.

    After the forward pass:

    - :attr:`gammas_tensor` — shape ``(T, B, 1)`` step sizes per iteration per sample.
    - :attr:`thetas_tensor` — shape ``(T, B, 1)`` thresholds per iteration per sample.

    Call :meth:`reset` before reusing the same collector for a new forward pass.

    Example::

        collector = StepParamCollector()
        solver.callbacks = [collector]
        x_hat = solver(y, A=A, t=25)
        gammas = collector.gammas_tensor  # (25, B, 1)
        thetas = collector.thetas_tensor  # (25, B, 1)
        solver.callbacks = []
    """

    def __init__(self):
        self.gammas = []
        self.thetas = []

    def reset(self):
        """Clear accumulated gamma and theta lists."""
        self.gammas = []
        self.thetas = []

    def on_step_end(self, solver, x, gamma=None, theta=None, **kwargs):
        """Append detached copies of ``gamma`` and ``theta`` for this iteration.

        Silently skips when called without ``gamma``/``theta`` keyword arguments
        (e.g. when attached to a plain ISTA solver).
        """
        if gamma is not None:
            self.gammas.append(gamma.detach().clone())
        if theta is not None:
            self.thetas.append(theta.detach().clone())

    @property
    def gammas_tensor(self):
        """Stack collected gammas into a ``(T, B, 1)`` tensor."""
        if not self.gammas:
            raise RuntimeError("No gamma values collected. Run a forward pass first.")
        return torch.stack(self.gammas)

    @property
    def thetas_tensor(self):
        """Stack collected thetas into a ``(T, B, 1)`` tensor."""
        if not self.thetas:
            raise RuntimeError("No theta values collected. Run a forward pass first.")
        return torch.stack(self.thetas)


class IntermediateOutputCollector(IterationCallback):
    """Callback that records the iterate ``x`` at every step WITH its computational graph.

    Unlike metric collectors, tensors are stored **without** detaching so
    that gradients can flow through them during ``backward()``.  This is required
    by objectives that compute a loss over all intermediate iterates.

    The internal buffer is automatically cleared at the beginning of the first
    iteration of each forward pass (detected via ``solver.iter == 0``), so the
    collector does not need to be manually reset between training steps.

    After a forward pass, :attr:`intermediates` is a list of ``T`` tensors each
    of shape ``(B, n)``.

    Example::

        collector = IntermediateOutputCollector()
        solver.callbacks = [collector]
        x_hat = solver(y, A=A, t=10)
        # collector.intermediates contains 10 (B, n) tensors (with grad)
    """

    def __init__(self):
        self.intermediates = []

    def reset(self):
        """Manually clear the collected iterates."""
        self.intermediates = []

    def on_step_begin(self, solver, x, **kwargs):
        """Clear the buffer at the start of the first iteration of a forward pass."""
        if getattr(solver, "iter", None) == 0:
            self.intermediates = []

    def on_step_end(self, solver, x, **kwargs):
        """Append the current iterate ``x`` (with gradient) to the buffer."""
        self.intermediates.append(x)


class SolutionChangeStopper(IterationCallback):
    """Stop an iterative solver once consecutive iterates no longer change.

    By default the criterion is relative, which makes the tolerance independent
    of the source amplitude:

    ``||x_k - x_{k-1}|| / max(||x_k||, ||x_{k-1}||, eps) < tol``.

    For batched solvers, all batch elements must satisfy the criterion before
    the callback sets ``solver.stop = True``.
    """

    def __init__(self, tol=1e-6, relative=True, eps=1e-12):
        self.tol = float(tol)
        self.relative = bool(relative)
        self.eps = float(eps)
        self.previous = None
        self.last_change = None
        self.converged = False
        self.stop_iter = None
        self.n_steps = 0

    def reset(self):
        """Clear state from a previous solver run."""
        self.previous = None
        self.last_change = None
        self.converged = False
        self.stop_iter = None
        self.n_steps = 0

    @staticmethod
    def _batch_norm(x):
        if x.ndim == 1:
            return torch.linalg.vector_norm(x).reshape(1)
        return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1)

    def on_step_begin(self, solver, x, **kwargs):
        """Store the pre-step iterate for comparison at step end."""
        if getattr(solver, "iter", None) == 0:
            self.reset()
        self.previous = x.detach().clone()

    def on_step_end(self, solver, x, **kwargs):
        """Set ``solver.stop`` when the iterate change is below tolerance."""
        if self.previous is None:
            self.previous = x.detach().clone()
            return

        x_det = x.detach()
        diff = self._batch_norm(x_det - self.previous)
        if self.relative:
            scale = torch.maximum(self._batch_norm(x_det), self._batch_norm(self.previous))
            diff = diff / scale.clamp_min(self.eps)

        self.last_change = diff.detach().clone()
        self.n_steps = int(getattr(solver, "iter", 0)) + 1

        if torch.all(diff < self.tol):
            self.converged = True
            self.stop_iter = int(getattr(solver, "iter", self.n_steps - 1))
            solver.stop = True
