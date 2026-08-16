"""
Statistical tooling for retrieval evaluation: bootstrap confidence intervals and
paired significance tests over per-query metric scores.

All randomness is seeded (NumPy Generator) so intervals and p-values are
reproducible for a given seed.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

DEFAULT_RESAMPLES = 10000
DEFAULT_ALPHA = 0.05


def bootstrap_ci(
    values: list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of `values`."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    if n == 1:
        v = float(arr[0])
        return {"mean": v, "lo": v, "hi": v}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return {"mean": float(arr.mean()), "lo": lo, "hi": hi}


def paired_bootstrap_test(
    a: list[float],
    b: list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> dict[str, float]:
    """
    Two-sided paired bootstrap test on the mean difference (a - b).

    p_value is the achieved significance level under H0: mean(a - b) = 0, computed
    by resampling the mean-centered differences. Also returns a percentile CI for
    the observed mean difference.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape or a_arr.size == 0:
        return {"mean_diff": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0, "n": int(a_arr.size)}

    diff = a_arr - b_arr
    n = diff.size
    obs = float(diff.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diff[idx].mean(axis=1)

    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    # H0 distribution: center resample means at 0, measure how extreme obs is.
    centered = boot_means - obs
    p_value = float(np.mean(np.abs(centered) >= abs(obs)))
    return {"mean_diff": obs, "lo": lo, "hi": hi, "p_value": p_value, "n": int(n)}


def wilcoxon_signed_rank(a: list[float], b: list[float]) -> Optional[dict[str, Any]]:
    """
    Wilcoxon signed-rank test on paired samples (requires SciPy).

    Returns None if SciPy is unavailable so callers can skip gracefully.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None

    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape or a_arr.size == 0 or np.allclose(a_arr, b_arr):
        return {"statistic": 0.0, "p_value": 1.0}
    try:
        stat, p = wilcoxon(a_arr, b_arr)
        return {"statistic": float(stat), "p_value": float(p)}
    except ValueError:
        # All differences zero, or too few samples.
        return {"statistic": 0.0, "p_value": 1.0}
