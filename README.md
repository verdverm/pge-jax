# pge-jax

JAX implementation of the PGE algorithm (Prioritized Grammar Enumeration) for symbolic regression.

## Overview

This project is a JAX-native reimplementation of the PGE algorithm, ported from the original Python (pypge) and Go (go-pge) implementations. PGE is a symbolic regression method that enumerates candidate expressions from a grammar, prioritizing them by fitness to efficiently search the space of possible mathematical formulas.

The key innovation here is replacing the evaluation and optimization pipeline with JAX, enabling:
- GPU/TPU acceleration of model evaluation
- Automatic differentiation for Jacobian computation
- JIT compilation for fast execution
- Differentiable programming capabilities

## Architecture

### Module Structure

```
pge_jax/
├── __init__.py          # Public API exports
├── model.py             # JAXModel - sympy expression wrapper with JAX evaluation
├── optimize.py          # Levenberg-Marquardt and BFGS optimizers in JAX
├── metrics.py           # JAX-native metric functions (RMSE, R², AIC, etc.)
└── evaluate.py          # High-level fit/evaluate pipeline
```

### Core Components

#### `JAXModel` (model.py)

Wraps a sympy expression and provides JAX-compatible evaluation. This is the JAX equivalent of pypge's `Model` class.

**Key responsibilities:**
- Parse sympy expressions to extract coefficient symbols (`C_*`) and input variables
- Build JAX-traceable prediction functions via `sympy.lambdify(modules="jax")`
- Compute Jacobian matrices using `jax.jacfwd` + `jax.vmap`
- Provide `predict()`, `jacobian()`, and `pretty_expr()` methods

**Symbol convention:**
- Symbols starting with `C_` or `C[` are treated as optimizable coefficients
- All other free symbols in the expression are input variables
- Both lists are sorted by string name for deterministic ordering

**Example:**
```python
import sympy
from pge_jax import JAXModel

x = sympy.Symbol("x")
c0, c1 = sympy.symbols("C_0 C_1")
expr = c0 * x + c1

model = JAXModel(expr)
# model.n_coeffs == 2, model.n_vars == 1

import jax.numpy as jnp
pred = model.predict(jnp.array([2.0, 3.0]), jnp.array([1.0, 2.0, 3.0]))
# Returns: [5.0, 7.0, 9.0]
```

#### Optimizers (optimize.py)

##### `fit_levenberg_marquardt()`

Custom Levenberg-Marquardt optimizer implemented entirely in JAX. Solves:

$$\min_\beta \sum_i r_i(\beta)^2$$

where $r_i$ are the residuals of the model prediction.

**Algorithm:**
1. Compute initial Jacobian via `jax.jacfwd`
2. Initialize damping parameter $\mu = \text{damping} \times \max(\text{diag}(J^T J))$
3. Iterate:
   - Solve damped normal equations: $(J^T J + \mu I) d = -J^T r$
   - Accept step if cost decreases
   - Adjust damping based on improvement ratio
4. Converge when cost change or coefficient change falls below tolerance

**Parameters:**
- `model_predict`: Callable `(coefs) -> predictions`, must be JAX-transformable
- `y_true`: Target values, shape `(n_samples,)`
- `x0`: Initial coefficient guess (defaults to zeros)
- `jac`: Optional analytic Jacobian (defaults to `jax.jacfwd`)
- `max_iter`: Maximum iterations (default: 200)
- `tol`: Convergence tolerance (default: 1e-4)
- `damping`: Initial damping $\mu$ (default: 1e-6)
- `damping_factor`: Multiplier for increasing/decreasing damping (default: 2.0)

**Returns:** `FitResult` dataclass with coefficients, predictions, cost, success flag, and metadata.

##### `fit_least_squares()`

Wrapper around `jax.scipy.optimize.minimize` using BFGS method. Useful when you prefer a battle-tested solver over custom LM.

**Features:**
- Automatic gradient computation via JAX autodiff
- Optional bounds via penalty method (JAX's minimize doesn't support bounds natively)

#### Metrics (metrics.py)

JAX-native implementations of standard regression metrics:

| Function | Formula | Notes |
|----------|---------|-------|
| `rmse` | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ | Root mean squared error |
| `mae` | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ | Mean absolute error |
| `mse` | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | Mean squared error |
| `r2` | $1 - \frac{SS_{res}}{SS_{tot}}$ | R-squared coefficient of determination |
| `explained_variance` | $1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$ | Explained variance score |
| `aic` | $n \log(\text{RSS}/n) + 2k$ | Akaike information criterion |
| `bic` | $n \log(\text{RSS}/n) + k \log(n)$ | Bayesian information criterion |
| `chisqr` | $\sum (y_i - \hat{y}_i)^2$ | Chi-squared statistic |
| `redchi` | $\chi^2 / (n - k)$ | Reduced chi-squared |
| `rmae` | $\text{MAE} / \overline{|y|}$ | Relative MAE |

All metrics are JAX-transformable and work with `jax.grad`, `jax.vmap`, etc.

#### Evaluation Pipeline (evaluate.py)

High-level functions for fitting and evaluating models:

- `fit_model(model, y_true, *x_inputs, max_iter=200, method="lm")`: Fit model coefficients
- `predict(model, coefs, *x_inputs)`: Evaluate model with given coefficients
- `evaluate(model, y_true, y_pred, n_params=None)`: Compute all metrics, returns `EvalResult`

**Example:**
```python
from pge_jax import JAXModel, fit_model, evaluate
import jax.numpy as jnp
import sympy

x = sympy.Symbol("x")
c0, c1 = sympy.symbols("C_0 C_1")
expr = c0 * x + c1

model = JAXModel(expr)
x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_data = 2.0 * x_data + 3.0

result = fit_model(model, y_data, x_data)
print(result.coefficients)  # [2.0, 3.0]

eval_result = evaluate(model, y_data, result.predictions)
print(f"RMSE: {eval_result.score:.6f}")
print(f"R²: {eval_result.r2:.6f}")
```

## Setup

### Prerequisites

- Python 3.11+
- macOS, Linux, or Windows with CUDA/ROCm for GPU support

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with all dependencies
pip install -e ".[dev,benchmark]"
```

### Optional Dependencies

- `dev`: pytest, ruff, mypy, pre-commit (for development)
- `benchmark`: scikit-learn, lmfit, pandas (for compatibility with pypge benchmarks)
- `notebook`: jupyter, matplotlib (for interactive exploration)

## Development

### Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

### Code Quality

```bash
# Lint with ruff
ruff check pge_jax/ tests/

# Format with ruff
ruff format pge_jax/ tests/

# Type check with mypy
mypy pge_jax/
```

### Project Configuration

- **Line length:** 120 characters
- **Target Python:** 3.11+
- **Linter:** ruff (selects E, F, I, N, W, NPY rules)
- **Type checker:** mypy (non-strict mode)

## Design Decisions

### Why sympy + JAX?

Sympy is retained for expression representation, tree manipulation, expansion, and simplification. The JAX layer replaces only the evaluation and optimization pipeline, providing:

1. **Performance:** GPU acceleration and JIT compilation
2. **Differentiability:** Automatic Jacobian computation via `jax.jacfwd`
3. **Compatibility:** Sympy expressions can still be used with the existing PGE search logic

### Jacobian Computation

The Jacobian matrix $J_{ij} = \partial \hat{y}_i / \partial \beta_j$ is computed using:
- `jax.jacfwd` for forward-mode autodiff (efficient when #inputs < #outputs)
- `jax.vmap` to vectorize over data points

This replaces the symbolic Jacobian computation in pypge (`sympy.diff`) with automatic differentiation.

### Coefficient Extraction

Coefficients are identified by symbol name convention:
- `C_*` or `C[*]` → coefficient
- Everything else → input variable

This matches the pypge convention and allows automatic parsing of sympy expressions.

### Optimizer Choice

Levenberg-Marquardt is the default because:
- It's well-suited for least-squares problems (natural fit for symbolic regression)
- It uses the Jacobian, which we compute efficiently via JAX
- It's the method used in the original pypge implementation (via lmfit)

## Integration with PGE Search

The `pge_jax` module is designed to integrate with the PGE search algorithm (to be implemented). The typical workflow:

1. **Generate expressions** using sympy tree manipulation (from pypge's expand.py)
2. **Create JAXModel** for each expression
3. **Fit coefficients** using `fit_model()` or `fit_levenberg_marquardt()`
4. **Evaluate fitness** using metrics from metrics.py
5. **Select best models** using NSGA-II or other multi-objective selection
6. **Expand selected models** to generate children
7. **Repeat** until convergence or iteration limit

## Prior Art

- **pypge**: Original Python implementation - https://github.com/verdverm/pypge
- **go-pge**: Go implementation with performance optimizations - https://github.com/verdverm/go-pge
- **PySR**: Symbolic regression with JAX - https://github.com/MilesCranmer/pysr
- **FFX**: Fast feature selection for symbolic regression - https://github.com/ffx-org/ffx

## Citation

If you use this library in academic work, please cite the original PGE paper:

> "Prioritized Grammar Enumeration" - Best Paper, GECCO 2013
> http://dl.acm.org/citation.cfm?id=2463486
