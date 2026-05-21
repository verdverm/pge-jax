from pge_jax.model import JAXModel
from pge_jax.optimize import fit_levenberg_marquardt, fit_least_squares
from pge_jax.metrics import rmse, mae, mse, r2, explained_variance, aic, bic
from pge_jax.evaluate import fit_model, predict, evaluate
from pge_jax.search_model import SearchModel
from pge_jax.filters import filter_models, default_filters
from pge_jax.algebra import manip_model, do_simp
from pge_jax.memoize import Memoizer
from pge_jax.selection import selNSGA2, selSPEA2, sortLogNondominated, isDominated, assignCrowdingDist
from pge_jax.fitness_funcs import build_fitness_calc, build_fitness_weights, build_value_extractor
from pge_jax.expand import Grower, map_names_to_funcs
from pge_jax.search import PGE, ExpanderConfig

__all__ = [
    # Evaluation backend
    "JAXModel",
    "fit_levenberg_marquardt",
    "fit_least_squares",
    "rmse",
    "mae",
    "mse",
    "r2",
    "explained_variance",
    "aic",
    "bic",
    "fit_model",
    "predict",
    "evaluate",
    # Search loop
    "SearchModel",
    "filter_models",
    "default_filters",
    "manip_model",
    "do_simp",
    "Memoizer",
    "selNSGA2",
    "selSPEA2",
    "sortLogNondominated",
    "isDominated",
    "assignCrowdingDist",
    "build_fitness_calc",
    "build_fitness_weights",
    "build_value_extractor",
    "Grower",
    "map_names_to_funcs",
    "PGE",
    "ExpanderConfig",
]
