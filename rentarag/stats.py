"""Pooled DirBucket test statistics, support gate and Benjamini-Hochberg.

These are exact ports of the functions used by the paper's audit runner
(``run_dirbucket_quick_experiments.py``: ``pooled_z``,
``gate_significance_by_min_pooled_T``, ``bh_adjust``).

Under the null (no watermark or wrong key) each anchored score
``<u, v>`` is roughly centred at 0 with variance ``1/d``, so for a provider
with pooled score sum ``S`` over ``T`` aligned windows
``Z = S / sqrt(T/d) = mean * sqrt(d*T)`` is approximately standard normal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["norm_sf", "pooled_z", "bh_adjust", "GatedTest", "gate_by_min_support"]


def norm_sf(z: float) -> float:
    """One-sided upper-tail p-value of a standard normal."""
    try:
        from scipy.stats import norm

        return float(norm.sf(z))
    except ImportError:
        return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def pooled_z(sum_proj: Sequence[float], support: Sequence[int], dim: int) -> list[float]:
    """Pooled ``Z_i = (S_i / T_i) * sqrt(d * T_i)``; 0 for providers with no support."""
    out: list[float] = []
    for s, T in zip(sum_proj, support):
        T = int(T)
        out.append(0.0 if T <= 0 else float(s / T) * math.sqrt(dim * T))
    return out


def bh_adjust(pvals: Sequence[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg step-up procedure.

    Returns BH-adjusted p-values (q-values) and rejection decisions at level
    ``alpha`` over all ``m = len(pvals)`` hypotheses.
    """
    m = len(pvals)
    idx = np.argsort(pvals)
    p_sorted = np.array(pvals, dtype=float)[idx]
    q_sorted = np.empty(m, dtype=float)
    running = 1.0
    for j in range(m, 0, -1):
        running = min(running, (p_sorted[j - 1] * m) / j)
        q_sorted[j - 1] = running
    qvals = np.empty(m, dtype=float)
    qvals[idx] = q_sorted

    below = p_sorted <= (np.arange(1, m + 1) / m) * alpha
    k = int(np.max(np.where(below)[0]) + 1) if np.any(below) else 0
    sig = np.zeros(m, dtype=bool)
    if k > 0:
        sig[idx[:k]] = True
    return [float(x) for x in qvals.tolist()], [bool(x) for x in sig.tolist()]


@dataclass
class GatedTest:
    """Result of :func:`gate_by_min_support` (one entry per provider)."""

    z: list[float]
    pvals: list[float]
    qvals: list[float]
    significant: list[bool]
    eligible: list[bool]


def gate_by_min_support(z_raw: Sequence[float], support: Sequence[int], *, min_support: int, alpha: float = 0.05) -> GatedTest:
    """Support gate followed by BH.

    Providers with fewer than ``min_support`` (``T_min``) aligned windows are
    not tested: their ``z`` is set to 0 and ``p`` to 1, but they still count
    towards the BH family size (conservative). The others keep ``p = sf(z)``.
    """
    min_t = max(0, int(min_support))
    z_out: list[float] = []
    p_out: list[float] = []
    eligible: list[bool] = []
    for z, T in zip(z_raw, support):
        ok = int(T) >= min_t
        eligible.append(ok)
        z_out.append(float(z) if ok else 0.0)
        p_out.append(norm_sf(float(z)) if ok else 1.0)
    q, sig = bh_adjust(p_out, alpha=alpha)
    return GatedTest(z=z_out, pvals=p_out, qvals=q, significant=sig, eligible=eligible)
