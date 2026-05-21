from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from pge_jax.metrics import (
    aic,
    bic,
    chisqr,
    explained_variance,
    mae,
    mse,
    r2,
    redchi,
    rmae,
    rmse,
)
from pge_jax.model import JAXModel
from pge_jax.optimize import FitResult, fit_levenberg_marquardt


@dataclass
class EvalResult:
    """Full evaluation result for a single model."""

    fit: FitResult
    score: float  # primary error metric (RMSE)
    r2: float
    evar: float
    aic: float
    bic: float
    chisqr: float
    redchi: float
    mae: float
    rmae: float
    predictions: jnp.ndarray


def fit_model(
    model: JAXModel,
    y_true: jnp.ndarray,
    *x_inputs: jnp.ndarray,
    max_iter: int = 200,
    method: str = "lm",
) -> FitResult:
    """Fit the coefficients of *model* to minimise prediction error.

    Parameters
    ----------
    model:
        A :class:`JAXModel` whose ``jax_fun`` will be used.
    y_true:
        Target values, shape ``(n_samples,)``.
    *x_inputs:
        Input-variable arrays, each shape ``(n_samples,)``.
    max_iter:
        Maximum optimisation iterations.
    method:
        Optimiser to use: ``"lm"`` (Levenberg-Marquardt, default) or
        ``"ls"`` (``jax.scipy.optimize.least_squares``).

    Returns
    -------
    FitResult
    """
    y_true = jnp.asarray(y_true, dtype=jnp.float64)

    # Build a zero-arg predict that captures x_inputs
    def predict(coefs: jnp.ndarray) -> jnp.ndarray:
        return model.jax_fun(coefs, *x_inputs)

    # Initialize coefficients to zeros with the right size
    x0 = jnp.zeros(model.n_coeffs, dtype=jnp.float64)

    if method == "lm":
        return fit_levenberg_marquardt(predict, y_true, x0=x0, max_iter=max_iter)
    elif method == "ls":
        from pge_jax.optimize import fit_least_squares

        return fit_least_squares(predict, y_true, x0=x0, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown method: {method!r}")


def predict(
    model: JAXModel,
    coefs: jnp.ndarray,
    *x_inputs: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the fitted model at the given inputs."""
    return model.jax_fun(jnp.asarray(coefs, dtype=jnp.float64), *x_inputs)


def evaluate(
    model: JAXModel,
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    n_params: int | None = None,
) -> EvalResult:
    """Compute all standard regression metrics.

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred:
        Model predictions.
    n_params:
        Number of free parameters (for AIC / BIC).  Defaults to
        ``model.n_coeffs``.
    """
    y_true = jnp.asarray(y_true, dtype=jnp.float64)
    y_pred = jnp.asarray(y_pred, dtype=jnp.float64)

    if n_params is None:
        n_params = model.n_coeffs if hasattr(model, "n_coeffs") else 1

    return EvalResult(
        score=float(rmse(y_true, y_pred)),
        r2=float(r2(y_true, y_pred)),
        evar=float(explained_variance(y_true, y_pred)),
        aic=float(aic(y_true, y_pred, n_params)),
        bic=float(bic(y_true, y_pred, n_params)),
        chisqr=float(chisqr(y_true, y_pred, n_params)),
        redchi=float(redchi(y_true, y_pred, n_params)),
        mae=float(mae(y_true, y_pred)),
        rmae=float(rmae(y_true, y_pred)),
        predictions=y_pred,
        fit=FitResult(  # placeholder — caller should pass FitResult
            success=True,
            coefficients=jnp.array([]),
            predictions=y_pred,
            cost=float(mse(y_true, y_pred)),
            n_residuals=len(y_true),
            n_params=n_params,
        ),
    )
