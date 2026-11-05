"""Pooled Z, support gate and BH against the paper's audit runner."""

from types import SimpleNamespace

import numpy as np
import pytest

from rentarag import bh_adjust, gate_by_min_support, pooled_z


@pytest.fixture(scope="module")
def runner():
    return pytest.importorskip("run_dirbucket_quick_experiments")


def _random_pvals(rng: np.random.Generator, m: int) -> list[float]:
    p = rng.uniform(0, 1, m) ** rng.uniform(0.2, 4)
    p[rng.random(m) < 0.2] = 1.0
    if m > 2:
        p[1] = p[0]
    return p.tolist()


@pytest.mark.parametrize("seed", range(40))
def test_bh_matches_runner(runner, seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 25))
    pvals = _random_pvals(rng, m)
    alpha = float(rng.choice([0.01, 0.05, 0.1]))
    assert bh_adjust(pvals, alpha) == runner.bh_adjust(pvals, alpha)


@pytest.mark.parametrize("seed", range(40))
def test_gate_and_pooled_z_match_runner(runner, seed):
    rng = np.random.default_rng(100 + seed)
    P, dim = int(rng.integers(2, 12)), int(rng.choice([64, 384, 768]))
    T = rng.integers(0, 80, P).astype(np.int64)
    S = rng.normal(0, 1, P) * np.sqrt(T / dim) + rng.uniform(0, 0.08, P) * T
    det = SimpleNamespace(embedder=SimpleNamespace(st=SimpleNamespace(get_sentence_embedding_dimension=lambda: dim)))
    z_ref = runner.pooled_z(det, T, S.astype(np.float64))
    z = pooled_z(S.tolist(), T.tolist(), dim)
    assert z == z_ref
    t_min = int(rng.integers(0, 60))
    gated = gate_by_min_support(z, T.tolist(), min_support=t_min, alpha=0.05)
    zg, p, q, sig, elig = runner.gate_significance_by_min_pooled_T(z_ref, T, min_pooled_T=t_min, alpha=0.05)
    assert (gated.z, gated.pvals, gated.qvals, gated.significant, gated.eligible) == (zg, p, q, sig, elig)


def test_bh_basic_properties():
    q, sig = bh_adjust([0.001, 0.2, 0.04, 0.03], alpha=0.05)
    assert sig == [True, False, False, False]
    assert q[0] == pytest.approx(0.004)
    assert bh_adjust([0.01, 0.02], 0.05)[1] == [True, True]


def test_gate_excludes_low_support():
    g = gate_by_min_support([10.0, 10.0], [24, 25], min_support=25)
    assert g.eligible == [False, True]
    assert g.z[0] == 0.0 and g.pvals[0] == 1.0
    assert g.significant == [False, True]


def test_pooled_z_zero_support():
    assert pooled_z([0.0, 1.0], [0, 100], 768) == [0.0, 0.01 * np.sqrt(768 * 100)]
