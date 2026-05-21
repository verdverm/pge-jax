"""Tests for pge_jax.optimize module."""

import jax.numpy as jnp
import numpy as np
import pytest

from pge_jax.optimize import fit_levenberg_marquardt, fit_least_squares, FitResult


class TestFitLevenbergMarquardt:
    def test_linear_single_coeff(self):
        """Fit y = C_0 * x to synthetic data."""
        x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true_coef = 2.5
        y_data = true_coef * x_data + jnp.array([0.01, -0.02, 0.01, -0.01, 0.02])

        def predict(coefs):
            return coefs[0] * x_data

        result = fit_levenberg_marquardt(predict, y_data, x0=jnp.array([0.0]))

        assert result.success
        assert result.coefficients[0] == pytest.approx(true_coef, rel=1e-3)
        assert result.nfev > 0

    def test_linear_two_coeffs(self):
        """Fit y = C_0 * x + C_1 to synthetic data."""
        x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true_c0, true_c1 = 2.0, 3.0
        y_data = true_c0 * x_data + true_c1 + jnp.array([0.01, -0.02, 0.01, -0.01, 0.02])

        def predict(coefs):
            return coefs[0] * x_data + coefs[1]

        result = fit_levenberg_marquardt(predict, y_data, x0=jnp.array([0.0, 0.0]))

        assert result.success
        assert result.coefficients[0] == pytest.approx(true_c0, rel=5e-3)
        assert result.coefficients[1] == pytest.approx(true_c1, rel=5e-3)

    def test_quadratic(self):
        """Fit y = C_0 * x^2 + C_1 * x + C_2."""
        x_data = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        true = jnp.array([1.5, -2.0, 0.5])
        y_data = true[0] * x_data**2 + true[1] * x_data + true[2]

        def predict(coefs):
            return coefs[0] * x_data**2 + coefs[1] * x_data + coefs[2]

        result = fit_levenberg_marquardt(predict, y_data, x0=jnp.zeros(3))

        assert result.success
        assert jnp.allclose(result.coefficients, true, atol=5e-3)

    def test_trig_model(self):
        """Fit y = C_0 * sin(x) + C_1 * cos(x)."""
        x_data = jnp.array([0.0, jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4, jnp.pi])
        true_c0, true_c1 = 3.0, -1.0
        sin_vals = jnp.sin(x_data)
        cos_vals = jnp.cos(x_data)
        y_data = true_c0 * sin_vals + true_c1 * cos_vals

        def predict(coefs):
            return coefs[0] * sin_vals + coefs[1] * cos_vals

        result = fit_levenberg_marquardt(predict, y_data, x0=jnp.array([0.0, 0.0]))

        assert result.success
        assert result.coefficients[0] == pytest.approx(true_c0, rel=1e-3)
        assert result.coefficients[1] == pytest.approx(true_c1, rel=1e-3)

    def test_with_jacobian(self):
        """LM with analytic Jacobian should match finite-diff quality."""
        x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true_c0 = 2.5
        y_data = true_c0 * x_data

        def predict(coefs):
            return coefs[0] * x_data

        def jac(coefs):
            return jnp.column_stack([x_data])

        result = fit_levenberg_marquardt(predict, y_data, x0=jnp.array([0.0]), jac=jac)

        assert result.success
        assert result.coefficients[0] == pytest.approx(true_c0, rel=1e-5)

    def test_noisy_data(self):
        """Should still converge with noise, though not perfectly."""
        x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        true_c0 = 1.5
        noise = jnp.array([0.3, -0.2, 0.1, -0.4, 0.2, 0.1, -0.3, 0.2, -0.1, 0.4])
        y_data = true_c0 * x_data + noise

        def predict(coefs):
            return coefs[0] * x_data

        result = fit_levenberg_marquardt(predict, y_data, x0=jnp.array([0.0]))

        assert result.coefficients[0] == pytest.approx(true_c0, rel=0.1)
        assert result.cost < jnp.sum((y_data - true_c0 * x_data) ** 2) * 1.5

    def test_fitresult_attributes(self):
        """FitResult should have all expected attributes."""
        x_data = jnp.array([1.0, 2.0, 3.0])
        y_data = jnp.array([2.0, 4.0, 6.0])

        def predict(coefs):
            return coefs[0] * x_data

        result = fit_levenberg_marquardt(predict, y_data, x0=jnp.array([0.0]))

        assert isinstance(result, FitResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.coefficients, jnp.ndarray)
        assert isinstance(result.predictions, jnp.ndarray)
        assert isinstance(result.cost, float)
        assert result.n_residuals == 3
        assert result.n_params == 1
        assert result.nfev > 0


class TestFitLeastSquares:
    def test_linear(self):
        x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true_c0 = 2.5
        y_data = true_c0 * x_data

        def predict(coefs):
            return coefs[0] * x_data

        result = fit_least_squares(predict, y_data, x0=jnp.array([0.0]))

        assert result.success
        assert result.coefficients[0] == pytest.approx(true_c0, rel=1e-3)

    def test_with_bounds(self):
        x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true_c0 = 2.5
        y_data = true_c0 * x_data

        def predict(coefs):
            return coefs[0] * x_data

        result = fit_least_squares(predict, y_data, x0=jnp.array([0.0]), bounds=(jnp.array([0.0]), jnp.array([10.0])))

        assert result.success
        assert result.coefficients[0] >= 0.0
        assert result.coefficients[0] <= 10.0
