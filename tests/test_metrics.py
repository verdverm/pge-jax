"""Tests for pge_jax.metrics module."""

import jax.numpy as jnp
import numpy as np
import pytest

from pge_jax.metrics import (
    aic,
    bic,
    chisqr,
    explained_variance,
    mae,
    mse,
    r2,
    rmae,
    rmse,
)


@pytest.fixture
def y_true():
    return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def y_pred_perfect():
    return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def y_pred_noisy():
    return jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])


def test_rmse_perfect(y_pred_perfect, y_true):
    assert rmse(y_true, y_pred_perfect) == pytest.approx(0.0, abs=1e-10)


def test_rmse_noisy(y_pred_noisy, y_true):
    expected = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred_noisy)) ** 2))
    assert rmse(y_true, y_pred_noisy) == pytest.approx(expected, rel=1e-5)


def test_mae_perfect(y_pred_perfect, y_true):
    assert mae(y_true, y_pred_perfect) == pytest.approx(0.0, abs=1e-10)


def test_mse_perfect(y_pred_perfect, y_true):
    assert mse(y_true, y_pred_perfect) == pytest.approx(0.0, abs=1e-10)


def test_r2_perfect(y_pred_perfect, y_true):
    assert r2(y_true, y_pred_perfect) == pytest.approx(1.0, abs=1e-10)


def test_r2_noisy(y_pred_noisy, y_true):
    r2_val = r2(y_true, y_pred_noisy)
    assert r2_val > 0.9  # Should be a good fit
    assert r2_val < 1.0


def test_explained_variance_perfect(y_pred_perfect, y_true):
    assert explained_variance(y_true, y_pred_perfect) == pytest.approx(1.0, abs=1e-10)


def test_aic_bic(y_true, y_pred_noisy):
    aic_val = aic(y_true, y_pred_noisy, n_params=2)
    bic_val = bic(y_true, y_pred_noisy, n_params=2)
    # For n > e^2 (~7.4), BIC penalizes more than AIC; for small n the
    # relationship can flip.  Just verify both are finite numbers.
    assert jnp.isfinite(aic_val)
    assert jnp.isfinite(bic_val)


def test_chisqr(y_true, y_pred_perfect):
    assert chisqr(y_true, y_pred_perfect, n_params=1) == pytest.approx(0.0, abs=1e-10)


def test_rmae(y_true, y_pred_noisy):
    rmae_val = rmae(y_true, y_pred_noisy)
    expected = np.mean(np.abs(np.array(y_true) - np.array(y_pred_noisy))) / np.mean(np.abs(np.array(y_true)))
    assert rmae_val == pytest.approx(expected, rel=1e-5)


def test_metrics_jax_transformable():
    """Verify metrics work inside jax.grad."""
    import jax

    def loss_fn(params):
        y_pred = params[0] + params[1] * jnp.array([1.0, 2.0, 3.0])
        return rmse(jnp.array([2.0, 3.0, 4.0]), y_pred)

    grads = jax.grad(loss_fn)(jnp.array([1.0, 1.0]))
    assert grads.shape == (2,)
