import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from pge_jax import PGE

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 2)
Y = 3.0 * X[:, 0] + 1.5 * X[:, 1] ** 2 - 0.5 * np.sin(X[:, 0])

# Run PGE search
pge = PGE(
    usable_vars=["x0", "x1"],
    usable_funcs=["sin", "cos", "exp", "log"],
    max_iter=3,
    pop_count=5,
    max_size=16,
    # peek_count=20,
    # peek_fraction=0.25,  # fraction of data for fast partial evaluation
)
pge.fit(X, Y)

# Get results programmatically
best = pge.get_best_model()
print(best.pretty_expr())  # e.g. "3.0*x0 + 1.5*x1**2 - 0.5*sin(x0)"

paretos = pge.get_final_paretos()  # list of Pareto fronts
