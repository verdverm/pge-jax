"""Benchmark PGE search — CPU vs GPU with memory and GPU utilization tracking."""

import jax
import time
import tracemalloc
import sys

jax.config.update("jax_enable_x64", True)

import numpy as np
from pge_jax import PGE


def generate_data(seed=42, n=500):
    np.random.seed(seed)
    X = np.random.randn(n, 2)
    Y = 3.0 * X[:, 0] + 1.5 * X[:, 1] ** 2 - 0.5 * np.sin(X[:, 0])
    return X, Y


def benchmark_pge(X, Y, config, label):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Devices: {jax.devices()}")
    print(f"{'=' * 60}")

    pge = PGE(
        usable_vars=["x0", "x1"],
        usable_funcs=["sin", "cos", "exp", "log"],
        max_iter=config.get("max_iter", 5),
        pop_count=config.get("pop_count", 20),
        max_size=config.get("max_size", 32),
        peek_count=config.get("peek_count", 40),
        peek_fraction=config.get("peek_fraction", 0.0),
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    pge.fit(X, Y)
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n  Time:      {elapsed:.2f}s")
    print(f"  Memory:    {peak / 1e6:.1f} MB (peak)")
    print(f"  Best expr: {pge.get_best_model().pretty_expr()}")
    print(f"  Paretos:   {len(pge.get_final_paretos())} fronts")

    return elapsed, peak


if __name__ == "__main__":
    X, Y = generate_data()

    # Small run for quick validation
    small = {"max_iter": 3, "pop_count": 10, "max_size": 20, "peek_count": 20, "peek_fraction": 0.05}

    # Full run
    full = {"max_iter": 10, "pop_count": 30, "max_size": 48, "peek_count": 60, "peek_fraction": 0.15}

    run = sys.argv[1] if len(sys.argv) > 1 else "small"
    config = small if run == "small" else full

    benchmark_pge(X, Y, config, f"PGE ({run} run, {len(jax.devices())} device(s))")
