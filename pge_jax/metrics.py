from __future__ import annotations

import jax.numpy as jnp


def rmse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Root mean squared error."""
    return jnp.sqrt(jnp.mean((y_true - y_pred) ** 2))


def mae(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute error."""
    return jnp.mean(jnp.abs(y_true - y_pred))


def mse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error."""
    return jnp.mean((y_true - y_pred) ** 2)


def r2(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """R-squared (coefficient of determination)."""
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-30)


def explained_variance(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Explained variance score."""
    var = jnp.var(y_true)
    return 1.0 - jnp.var(y_true - y_pred) / (var + 1e-30)


def aic(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    n_params: int,
) -> jnp.ndarray:
    """Akaike information criterion.

    Assumes Gaussian errors:
    ``AIC = n * log(RSS/n) + 2 * k``
    """
    n = len(y_true)
    rss = jnp.sum((y_true - y_pred) ** 2)
    return n * jnp.log(rss / n + 1e-30) + 2.0 * n_params


def bic(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    n_params: int,
) -> jnp.ndarray:
    """Bayesian information criterion.

    ``BIC = n * log(RSS/n) + k * log(n)``
    """
    n = len(y_true)
    rss = jnp.sum((y_true - y_pred) ** 2)
    return n * jnp.log(rss / n + 1e-30) + n_params * jnp.log(n + 1e-30)


def chisqr(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    n_params: int,
) -> jnp.ndarray:
    """Chi-squared statistic (unnormalised)."""
    return jnp.sum((y_true - y_pred) ** 2)


def redchi(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    n_params: int,
) -> jnp.ndarray:
    """Reduced chi-squared (chi-squared per degree of freedom)."""
    n = len(y_true)
    dof = n - n_params
    return chisqr(y_true, y_pred, n_params) / (dof + 1e-30)


def rmae(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """Relative mean absolute error (normalised by mean of y_true)."""
    mean_y = jnp.mean(jnp.abs(y_true))
    return jnp.mean(jnp.abs(y_true - y_pred)) / (mean_y + 1e-30)
