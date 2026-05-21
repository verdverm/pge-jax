"""Tests for pge_jax.evaluate module."""

import jax.numpy as jnp
import numpy as np
import pytest
import sympy

from pge_jax.model import JAXModel
from pge_jax.evaluate import fit_model, predict, evaluate


class TestFitModel:
    def test_linear_fit(self):
        """Full pipeline: create model -> fit -> check coefficients."""
        x = sympy.Symbol("x")
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 * x + c1

        x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true_c0, true_c1 = 2.0, 3.0
        y_data = true_c0 * x_data + true_c1

        model = JAXModel(expr)
        result = fit_model(model, y_data, x_data)

        assert result.success
        assert result.coefficients[0] == pytest.approx(true_c0, rel=1e-3)
        assert result.coefficients[1] == pytest.approx(true_c1, rel=1e-3)

    def test_quadratic_fit(self):
        x = sympy.Symbol("x")
        c0, c1, c2 = sympy.symbols("C_0 C_1 C_2")
        expr = c0 * x**2 + c1 * x + c2

        x_data = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        true = jnp.array([1.0, -1.0, 2.0])
        y_data = true[0] * x_data**2 + true[1] * x_data + true[2]

        model = JAXModel(expr)
        result = fit_model(model, y_data, x_data)

        assert result.success
        assert jnp.allclose(result.coefficients, true, atol=5e-3)

    def test_trig_fit(self):
        x = sympy.Symbol("x")
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 * sympy.sin(x) + c1 * sympy.cos(x)

        x_data = jnp.array([0.0, jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4, jnp.pi])
        true_c0, true_c1 = 3.0, -2.0
        y_data = true_c0 * jnp.sin(x_data) + true_c1 * jnp.cos(x_data)

        model = JAXModel(expr)
        result = fit_model(model, y_data, x_data)

        assert result.success
        assert result.coefficients[0] == pytest.approx(true_c0, rel=1e-3)
        assert result.coefficients[1] == pytest.approx(true_c1, rel=1e-3)

    def test_method_ls(self):
        """Test with least_squares method instead of LM."""
        x = sympy.Symbol("x")
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 * x + c1

        x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = 2.0 * x_data + 3.0

        model = JAXModel(expr)
        result = fit_model(model, y_data, x_data, method="ls")

        assert result.success
        assert result.coefficients[0] == pytest.approx(2.0, rel=1e-2)


class TestPredict:
    def test_predict_linear(self):
        x = sympy.Symbol("x")
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 * x + c1
        model = JAXModel(expr)

        coefs = jnp.array([2.0, 3.0])
        x_input = jnp.array([1.0, 2.0, 3.0])
        pred = predict(model, coefs, x_input)

        expected = 2.0 * np.array([1.0, 2.0, 3.0]) + 3.0
        assert jnp.allclose(pred, expected)


class TestEvaluate:
    def test_evaluate_perfect(self):
        x = sympy.Symbol("x")
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 * x + c1
        model = JAXModel(expr)

        y_true = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        result = evaluate(model, y_true, y_pred)

        assert result.score == pytest.approx(0.0, abs=1e-10)
        assert result.r2 == pytest.approx(1.0, abs=1e-10)
        assert result.mae == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_noisy(self):
        x = sympy.Symbol("x")
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 * x + c1
        model = JAXModel(expr)

        y_true = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])

        result = evaluate(model, y_true, y_pred)

        assert result.score > 0
        assert result.r2 < 1.0
        assert result.r2 > 0.9
        # BIC vs AIC ordering depends on n; just check both are finite
        assert jnp.isfinite(result.bic)
        assert jnp.isfinite(result.aic)

    def test_evaluate_returns_all_metrics(self):
        x = sympy.Symbol("x")
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 * x + c1
        model = JAXModel(expr)

        y_true = jnp.array([1.0, 2.0, 3.0])
        y_pred = jnp.array([1.0, 2.0, 3.0])

        result = evaluate(model, y_true, y_pred)

        assert hasattr(result, "score")
        assert hasattr(result, "r2")
        assert hasattr(result, "evar")
        assert hasattr(result, "aic")
        assert hasattr(result, "bic")
        assert hasattr(result, "chisqr")
        assert hasattr(result, "redchi")
        assert hasattr(result, "mae")
        assert hasattr(result, "rmae")
        assert hasattr(result, "predictions")
        assert hasattr(result, "fit")
