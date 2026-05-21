from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class FitResult:
    """Return type for the JAX fitting functions."""

    success: bool
    coefficients: jnp.ndarray
    predictions: jnp.ndarray
    cost: float
    n_residuals: int
    n_params: int
    message: str = ""
    nfev: int = 0


def _residual(model_predict, coefs: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    return model_predict(coefs) - y_true


def fit_levenberg_marquardt(
    model_predict,
    y_true: jnp.ndarray,
    x0: jnp.ndarray | None = None,
    jac=None,
    max_iter: int = 200,
    tol: float = 1e-4,
    damping: float = 1e-6,
    damping_factor: float = 2.0,
    damping_decrease_thresh: float = 1e-4,
    damping_increase_thresh: float = 5.0,
) -> FitResult:
    """Levenberg-Marquardt optimiser implemented entirely in JAX.

    Solves :math:`\\min_\\beta \\sum_i r_i(\\beta)^2` where
    :math:`r_i` are the residuals of *model_predict*.

    Parameters
    ----------
    model_predict:
        Callable ``(coefs) -> predictions``.  Must be JAX-transformable.
    y_true:
        Target values, shape ``(n_samples,)``.
    x0:
        Initial coefficient guess.  Defaults to zeros.
    jac:
        Optional Jacobian callable ``(coefs) -> J`` of shape
        ``(n_samples, n_coeffs)``.  If provided the analytic
        Jacobian is used; otherwise it is computed via
        ``jax.jacfwd``.
    max_iter:
        Maximum number of LM iterations.
    tol:
        Convergence tolerance on the relative cost change.
    damping:
        Initial damping parameter :math:`\\mu`.
    damping_factor:
        Multiplier for increasing / decreasing damping.
    damping_decrease_thresh:
        If the relative cost reduction exceeds this threshold,
        damping is divided by *damping_factor*.
    damping_increase_thresh:
        If the relative cost reduction is below this threshold,
        damping is multiplied by *damping_factor*.

    Returns
    -------
    FitResult
    """
    y_true = jnp.asarray(y_true, dtype=jnp.float64)
    n_samples = y_true.shape[0]

    if x0 is None:
        probe = jnp.ones(1)
        _ = model_predict(probe)
        x0 = jnp.zeros(1)

    x0 = jnp.asarray(x0, dtype=jnp.float64)
    n_params = int(x0.shape[0])

    # Compute initial Jacobian if not provided
    if jac is None:
        jac = jax.jacfwd(model_predict, argnums=0)

    cost = jnp.sum(_residual(model_predict, x0, y_true) ** 2)
    J0 = jac(x0)
    mu = damping * jnp.max(jnp.diag(J0.T @ J0) + 1e-10)

    # Check if we already have a perfect fit
    if cost < tol:
        return FitResult(
            success=True,
            coefficients=x0,
            predictions=model_predict(x0),
            cost=float(cost),
            n_residuals=n_samples,
            n_params=n_params,
            message="Cost already below tolerance",
            nfev=1,
        )

    message = ""
    success = False
    nfev = 1

    def _step(c: jnp.ndarray, mu_val: float) -> tuple[jnp.ndarray, float, bool, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        J = jac(c) if callable(jac) else jac  # type: ignore
        r = _residual(model_predict, c, y_true)
        r_cost = jnp.sum(r**2)

        A = J.T @ J  # type: ignore
        g = J.T @ r  # type: ignore

        I_mat = jnp.eye(n_params)
        d = jnp.linalg.lstsq(A + mu_val * I_mat, -g, rcond=None)[0]

        new_c = c + d
        new_r_cost = jnp.sum(_residual(model_predict, new_c, y_true) ** 2)

        denom = jnp.abs(r_cost) + 1e-30
        improvement = (r_cost - new_r_cost) / denom
        improvement = jnp.where(jnp.isfinite(improvement), improvement, 0.0)
        improved = new_r_cost < r_cost

        return new_c, mu_val, bool(improved), new_r_cost, improvement, d

    c = x0
    mu_val = mu
    prev_cost = cost

    for i in range(max_iter):
        new_c, mu_val, improved, new_r_cost, improvement, d = _step(c, mu_val)
        nfev += 1

        if improved:
            c = new_c
            cost = new_r_cost
            if improvement > damping_decrease_thresh:
                mu_val = max(mu_val / damping_factor, 1e-15)
            if jnp.abs(prev_cost - cost) < tol * jnp.abs(prev_cost + 1e-30):
                success = True
                message = f"Converged at iteration {i + 1}"
                break
            if cost < tol:
                success = True
                message = f"Cost below tolerance at iteration {i + 1}"
                break
        else:
            mu_val *= damping_factor
            if mu_val > 1e12:
                message = f"Damping blew up at iteration {i + 1}"
                break

        prev_cost = cost

        # Also check if coefficient change is tiny
        coef_change = jnp.max(jnp.abs(d))
        if coef_change < tol * 100:
            success = True
            message = f"Coefficient change below threshold at iteration {i + 1}"
            break

    predictions = model_predict(c)
    return FitResult(
        success=success,
        coefficients=c,
        predictions=predictions,
        cost=float(cost),
        n_residuals=n_samples,
        n_params=n_params,
        message=message,
        nfev=nfev,
    )


def fit_least_squares(
    model_predict,
    y_true: jnp.ndarray,
    x0: jnp.ndarray | None = None,
    bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    max_iter: int = 200,
) -> FitResult:
    """Wrapper around ``jax.scipy.optimize.minimize`` for least-squares.

    Minimises ``sum((model_predict(coefs) - y_true)^2)``.  JAX computes
    gradients automatically, so no explicit Jacobian is needed.
    """
    from jax.scipy.optimize import minimize

    y_true = jnp.asarray(y_true, dtype=jnp.float64)

    if x0 is None:
        x0 = jnp.zeros(1)
    x0 = jnp.asarray(x0, dtype=jnp.float64)

    def cost_fn(coefs: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((model_predict(coefs) - y_true) ** 2)

    if bounds is not None:
        lb, ub = bounds

        def cost_fn_bounded(coefs: jnp.ndarray) -> jnp.ndarray:
            penalty = 1e6 * jnp.sum(
                jnp.where(coefs < lb, (lb - coefs) ** 2, 0.0) + jnp.where(coefs > ub, (coefs - ub) ** 2, 0.0)
            )
            return cost_fn(coefs) + penalty

        result = minimize(cost_fn_bounded, x0, method="BFGS", options={"maxiter": max_iter})
    else:
        result = minimize(cost_fn, x0, method="BFGS", options={"maxiter": max_iter})

    return FitResult(
        success=bool(result.success),
        coefficients=result.x,
        predictions=model_predict(result.x),
        cost=float(result.fun),
        n_residuals=len(y_true),
        n_params=int(len(x0)),
        message=str(result.message) if hasattr(result, "message") else "",
        nfev=result.nit if hasattr(result, "nit") else 0,
    )
