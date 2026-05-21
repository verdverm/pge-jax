"""Tests for pge_jax.model module."""

import jax.numpy as jnp
import numpy as np
import sympy

from pge_jax.model import JAXModel, _extract_coeffs_and_vars


class TestExtractCoeffsAndVars:
    def test_simple_linear(self):
        x, y, c0, c1 = sympy.symbols("x y C_0 C_1")
        expr = c0 * x + c1 * y
        coeffs, vars_ = _extract_coeffs_and_vars(expr)
        assert set(coeffs) == {c0, c1}
        assert set(vars_) == {x, y}

    def test_no_coefficients(self):
        x, y = sympy.symbols("x y")
        expr = x + y
        coeffs, vars_ = _extract_coeffs_and_vars(expr)
        assert coeffs == []
        assert set(vars_) == {x, y}

    def test_no_variables(self):
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 + c1
        coeffs, vars_ = _extract_coeffs_and_vars(expr)
        assert set(coeffs) == {c0, c1}
        assert vars_ == []

    def test_mixed_expression(self):
        x = sympy.Symbol("x")
        c0, c1, c2 = sympy.symbols("C_0 C_1 C_2")
        expr = c0 * x**2 + c1 * sympy.sin(x) + c2
        coeffs, vars_ = _extract_coeffs_and_vars(expr)
        assert set(coeffs) == {c0, c1, c2}
        assert vars_ == [x]


class TestJAXModel:
    def test_linear_model_basic(self):
        """C_0 * x + C_1 should be expressible as a JAX function."""
        x = sympy.Symbol("x")
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 * x + c1
        model = JAXModel(expr)

        assert model.n_coeffs == 2
        assert model.n_vars == 1

        coefs = jnp.array([2.0, 3.0])
        x_input = jnp.array([1.0, 2.0, 3.0])
        pred = model.predict(coefs, x_input)

        expected = 2.0 * np.array([1.0, 2.0, 3.0]) + 3.0
        assert pred.shape == (3,)
        assert jnp.allclose(pred, expected)

    def test_quadratic_model(self):
        x = sympy.Symbol("x")
        c0, c1, c2 = sympy.symbols("C_0 C_1 C_2")
        expr = c0 * x**2 + c1 * x + c2
        model = JAXModel(expr)

        assert model.n_coeffs == 3
        assert model.n_vars == 1

        coefs = jnp.array([1.0, 2.0, 3.0])
        x_input = jnp.array([0.0, 1.0, 2.0])
        pred = model.predict(coefs, x_input)

        expected = 1.0 * np.array([0.0, 1.0, 4.0]) + 2.0 * np.array([0.0, 1.0, 2.0]) + 3.0
        assert jnp.allclose(pred, expected)

    def test_trig_model(self):
        """Model with sin/cos should work via sympy.lambdify(modules='jax')."""
        x = sympy.Symbol("x")
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 * sympy.sin(x) + c1 * sympy.cos(x)
        model = JAXModel(expr)

        coefs = jnp.array([1.0, 1.0])
        x_input = jnp.array([0.0, jnp.pi / 2, jnp.pi])
        pred = model.predict(coefs, x_input)

        expected = np.sin(np.array([0.0, np.pi / 2, np.pi])) + np.cos(np.array([0.0, np.pi / 2, np.pi]))
        assert jnp.allclose(pred, expected, atol=1e-5)

    def test_jacobian_linear(self):
        """For a linear model the Jacobian should be constant."""
        x = sympy.Symbol("x")
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 * x + c1
        model = JAXModel(expr)

        coefs = jnp.array([2.0, 3.0])
        x_input = jnp.array([1.0, 2.0, 3.0])
        J = model.jacobian(coefs, x_input)

        assert J.shape == (3, 2)
        # d/dC_0 = x, d/dC_1 = 1
        expected = jnp.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        assert jnp.allclose(J, expected)

    def test_jacobian_quadratic(self):
        x = sympy.Symbol("x")
        c0, c1, c2 = sympy.symbols("C_0 C_1 C_2")
        expr = c0 * x**2 + c1 * x + c2
        model = JAXModel(expr)

        coefs = jnp.array([1.0, 2.0, 3.0])
        x_input = jnp.array([1.0, 2.0])
        J = model.jacobian(coefs, x_input)

        assert J.shape == (2, 3)
        # d/dC_0 = x^2, d/dC_1 = x, d/dC_2 = 1
        expected = jnp.array([[1.0, 1.0, 1.0], [4.0, 2.0, 1.0]])
        assert jnp.allclose(J, expected)

    def test_no_coefficients(self):
        """Expression with no C_* symbols should still work."""
        x = sympy.Symbol("x")
        expr = x**2 + 2 * x + 1
        model = JAXModel(expr)

        assert model.n_coeffs == 0
        assert model.n_vars == 1

        x_input = jnp.array([1.0, 2.0, 3.0])
        pred = model.predict(jnp.array([]), x_input)
        expected = np.array([4.0, 9.0, 16.0])
        assert jnp.allclose(pred, expected)

    def test_no_variables(self):
        """Constant expression (no input variables) should work."""
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 + c1
        model = JAXModel(expr)

        assert model.n_coeffs == 2
        assert model.n_vars == 0

        coefs = jnp.array([2.0, 3.0])
        # No input variables — should return scalar broadcast
        pred = model.predict(coefs)
        assert pred.shape == ()
        assert pred == 5.0

    def test_pretty_expr(self):
        x = sympy.Symbol("x")
        c0, c1 = sympy.symbols("C_0 C_1")
        expr = c0 * x + c1
        model = JAXModel(expr)

        coefs = jnp.array([2.5, 3.5])
        pretty = model.pretty_expr(coefs)
        assert str(pretty) == "2.5*x + 3.5"

    def test_multivariate(self):
        """Model with multiple input variables."""
        x1, x2 = sympy.symbols("x1 x2")
        c0, c1, c2 = sympy.symbols("C_0 C_1 C_2")
        expr = c0 * x1 + c1 * x2 + c2
        model = JAXModel(expr)

        assert model.n_vars == 2

        coefs = jnp.array([1.0, 2.0, 3.0])
        x1_in = jnp.array([1.0, 2.0])
        x2_in = jnp.array([3.0, 4.0])
        pred = model.predict(coefs, x1_in, x2_in)

        expected = 1.0 * np.array([1.0, 2.0]) + 2.0 * np.array([3.0, 4.0]) + 3.0
        assert jnp.allclose(pred, expected)
