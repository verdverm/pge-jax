from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import sympy


def _extract_coeffs_and_vars(
    expr: sympy.Expr,
) -> tuple[list[sympy.Symbol], list[sympy.Symbol]]:
    """Pull coefficient (C_*) and variable symbols from *expr*.

    Convention: symbols whose name starts with ``C`` (but is not
    exactly ``'C'``) are treated as coefficients.  Everything else
    that appears free in the expression is an input variable.

    Both lists are sorted by string name for deterministic ordering.
    """
    coeffs: list[sympy.Symbol] = []
    vars_: list[sympy.Symbol] = []
    for sym in expr.free_symbols:
        name = str(sym)
        if name.startswith("C_") or name.startswith("C["):
            coeffs.append(sym)
        else:
            vars_.append(sym)
    coeffs.sort(key=lambda s: str(s))
    vars_.sort(key=lambda s: str(s))
    return coeffs, vars_


class JAXModel:
    """Wraps a sympy expression and provides JAX-compatible evaluation.

    This is the JAX equivalent of pypge's ``Model`` class, but replaces
    the ``sympy.lambdify`` + ``lmfit`` evaluation pipeline with
    JAX-native code.

    Parameters
    ----------
    expr:
        A sympy expression.  Bare ``sympy.Symbol('C')`` leaves are
        treated as optimisable coefficients; all other free symbols
        are treated as input variables.
    cs:
        Explicit list of coefficient symbols.  If *None* they are
        inferred from *expr*.
    xs:
        Explicit list of input-variable symbols.  If *None* they are
        inferred from *expr*.
    c_values:
        Initial coefficient values (used only to build the JAX
        function; the optimiser will overwrite them).

    Attributes
    ----------
    orig : sympy.Expr
        The original expression before expansion.
    expr : sympy.Expr
        ``sympy.expand(orig)``.
    cs : list[sympy.Symbol]
        Coefficient symbols, ordered consistently.
    xs : list[sympy.Symbol]
        Input-variable symbols, ordered consistently.
    n_coeffs : int
        Number of optimisable coefficients.
    n_vars : int
        Number of input variables.
    jax_fun : callable
        JAX-traceable function ``(cs, *xs) -> prediction``.
    jac_fun : callable
        JAX-traceable function ``(cs, *xs) -> Jacobian matrix``
        of shape ``(n_samples, n_coeffs)``.
    """

    def __init__(
        self,
        expr: sympy.Expr,
        cs: Sequence[sympy.Symbol] | None = None,
        xs: Sequence[sympy.Symbol] | None = None,
        c_values: Sequence[float] | None = None,
    ):
        self.orig = expr
        self.expr = sympy.expand(expr)

        if cs is None:
            cs, xs = _extract_coeffs_and_vars(self.expr)
        self.cs = list(cs)
        self.xs = list(xs) if xs is not None else []
        self.n_coeffs = len(self.cs)
        self.n_vars = len(self.xs)

        if c_values is not None:
            c_vals = np.asarray(c_values, dtype=np.float64)
            if self.n_coeffs > 0 and len(c_vals) != self.n_coeffs:
                raise ValueError(f"c_values length ({len(c_vals)}) != n_coeffs ({self.n_coeffs})")
            self.c_values = c_vals
        else:
            self.c_values = np.zeros(self.n_coeffs, dtype=np.float64)

        self._build_jax_functions()

    # ------------------------------------------------------------------
    # JAX function construction
    # ------------------------------------------------------------------

    def _build_jax_functions(self) -> None:
        """Lambdify the expression with the ``jax`` module and wrap it."""
        all_syms = self.cs + self.xs

        if len(all_syms) == 0:
            # Truly constant expression
            val = float(self.expr)
            const_val = jnp.array(val)

            def jax_fun(coefs: jnp.ndarray, *inputs: jnp.ndarray) -> jnp.ndarray:
                shape = inputs[0].shape if inputs else ()
                return jnp.broadcast_to(const_val, shape)

            def jac_fun(coefs: jnp.ndarray, *inputs: jnp.ndarray) -> jnp.ndarray:
                shape = inputs[0].shape if inputs else (1,)
                return jnp.zeros((len(shape) if isinstance(shape, tuple) else 1, 0))

            self.jax_fun = jax_fun
            self.jac_fun = jac_fun
            return

        # Lambdify with JAX backend
        try:
            raw_fun = sympy.lambdify(all_syms, self.expr, modules="jax")
        except Exception:
            raw_fun = sympy.lambdify(all_syms, self.expr, modules="numpy")

        # Wrap so the first arg is the coefficient array, the rest are
        # input-variable arrays.  This matches the signature expected
        # by the optimiser and Jacobian builder.
        def jax_fun(coefs: jnp.ndarray, *inputs: jnp.ndarray) -> jnp.ndarray:
            args: list[jnp.ndarray] = []
            if self.n_coeffs > 0:
                # Each coefficient symbol needs its own argument
                for i in range(self.n_coeffs):
                    args.append(coefs[i])
            args.extend(inputs)
            result = raw_fun(*args)
            result = jnp.asarray(result, dtype=jnp.float64)
            # Ensure output shape matches input data shape
            if result.ndim == 0:
                if inputs:
                    result = jnp.broadcast_to(result, inputs[0].shape)
                # else keep as scalar
            return result

        # Jacobian via jax.jacfwd + vmap
        if self.n_coeffs > 0:

            def _pred_single(coefs: jnp.ndarray, x0: jnp.ndarray) -> jnp.ndarray:
                if self.n_vars == 0:
                    return jnp.zeros_like(x0)
                return jax_fun(coefs, x0)

            jac_raw = jax.jacfwd(_pred_single, argnums=0)

            def jac_fun(coefs: jnp.ndarray, *inputs: jnp.ndarray) -> jnp.ndarray:
                if self.n_vars == 0:
                    n = inputs[0].shape[0] if inputs else 1
                    return jnp.zeros((n, self.n_coeffs))
                # vmap over data points
                jac_sampled = jax.vmap(jac_raw, in_axes=(None, 0))
                return jac_sampled(coefs, inputs[0])

            self.jac_fun = jac_fun
        else:

            def jac_fun(coefs: jnp.ndarray, *inputs: jnp.ndarray) -> jnp.ndarray:
                if inputs:
                    n = inputs[0].shape[0]
                else:
                    n = 1
                return jnp.zeros((n, 0))

        self.jax_fun = jax_fun
        self.jac_fun = jac_fun

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def predict(self, coefs: jnp.ndarray, *inputs: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the expression at *inputs* with given coefficients."""
        return self.jax_fun(coefs, *inputs)

    def jacobian(self, coefs: jnp.ndarray, *inputs: jnp.ndarray) -> jnp.ndarray:
        """Jacobian of predictions w.r.t. coefficients.

        Returns an array of shape ``(n_samples, n_coeffs)``.
        """
        return self.jac_fun(coefs, *inputs)

    def pretty_expr(self, coefs: jnp.ndarray) -> sympy.Expr:
        """Substitute coefficient *values* back into the sympy expression."""
        subs = dict(zip(self.cs, coefs.tolist() if self.n_coeffs > 0 else []))
        return self.expr.subs(subs)

    def __repr__(self) -> str:
        return f"JAXModel(coeffs={self.n_coeffs}, vars={self.n_vars}, expr={self.expr})"
