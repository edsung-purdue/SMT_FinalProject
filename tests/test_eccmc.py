import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eccmc import analytic as A               # noqa: E402
from eccmc import montecarlo as MC            # noqa: E402
from eccmc import radiation as R              # noqa: E402
from eccmc.codes import (DAEC, DECTED, SECDED, SNCDND, TAEC, TEC,  # noqa: E402
                         WordErrors)

MB = 2 ** 20 * 8


def _we(bit_lists):
    k = np.array([len(b) for b in bit_lists])
    first = np.array([b[0] for b in bit_lists])
    last = np.array([b[-1] for b in bit_lists])
    starts = np.concatenate([[0], np.cumsum(k)[:-1]])
    z = np.zeros(len(bit_lists), dtype=int)
    return WordErrors(z, z, k, first, last, starts, np.concatenate(bit_lists))


def test_code_rules():
    we = _we([[3], [3, 4], [3, 9], [3, 4, 5], [3, 4, 9], [0, 1, 2, 3], [0, 4], [8, 9]])
    assert SECDED(16, 8).fails(we).tolist() == [0, 1, 1, 1, 1, 1, 1, 1]
    assert DAEC(16, 8).fails(we).tolist() == [0, 0, 1, 1, 1, 1, 1, 0]
    assert DECTED(16, 8).fails(we).tolist() == [0, 0, 0, 1, 1, 1, 0, 0]
    assert TAEC(16, 8).fails(we).tolist() == [0, 0, 1, 0, 1, 1, 1, 0]
    assert TEC(16, 8).fails(we).tolist() == [0, 0, 0, 0, 0, 1, 0, 0]
    assert SNCDND(16, 8, 4).fails(we).tolist() == [0, 1, 1, 1, 1, 0, 1, 0]


def test_mc_matches_exact_counts():
    LA, W, m = 2 ** 12, 8, 1500
    r = MC.simulate(LA, W, m, 4000, nibbles=(4,), seed=3)
    for k in (2, 3, 4):
        mean, ci = r.nfm(k)
        assert abs(mean - A.nfm_exact(k, m, LA, W)) < 2.5 * ci + 1e-9
    mean, ci = r.nfm_dn(4)
    assert abs(mean - A.nfm_dn_exact(m, LA, W, 4)) < 2.5 * ci
    mean, ci = r.nfm_na(2)
    assert abs(mean - A.nfm_exact(2, m, LA, W) * (1 - 2 / W)) < 2.5 * ci


def test_paper_eqs_close_to_exact():
    LA, W = 2 ** 21, 8
    for m in (1e3, 1e5, 1e6):
        for k in (2, 3, 4):
            assert A.nfm(k, m, LA, W) == pytest.approx(A.nfm_exact(k, m, LA, W), rel=2e-3)


def test_nfm_dn_stable_equals_naive():
    LA, W = 2 ** 20, 16
    for m in (1e3, 1e5, 1e6):
        assert A.nfm_dn(m, LA, W, 4) == pytest.approx(A.nfm_dn_naive(m, LA, W, 4), rel=1e-6)


@pytest.mark.parametrize("code,col", [(SECDED(22, 16), [1682, 531, 169, 54, 18, 6, 3]),
                                      (SECDED(137, 128), [584, 185, 59, 19, 7, 3, 2]),
                                      (SNCDND(144, 128, 4), [590, 187, 60, 20, 7, 3, 2])])
def test_table2(code, col):
    LA = 256 * MB // code.k
    nines = [0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999]
    got = [A.max_bitflips_for_reliability(code, LA, x) for x in nines]
    assert got == col


def test_fig7_quoted_numbers():
    c = SECDED(39, 32)
    LA = 128 * 1024
    assert A.p_failure(c, 55, LA) > 0.01 > A.p_failure(c, 54, LA) * 0.9
    assert A.p_failure(c, 170, LA) > 0.10


def test_scrubbing_mc():
    LA, W, m = 128 * 1024, 32, 3000
    r = MC.simulate(LA, W, m, 400, n_scrubs=3, seed=4)
    mean, ci = r.nfm_plus(2)
    assert abs(mean - A.nfm_plus_scrubbed(2, m, LA, W, 3)) < 2.5 * ci


def test_geometry_bijective():
    g = R.Geometry(W=72, rows=16, cols=1000, interleave=4)
    rows, cols = np.meshgrid(np.arange(g.rows), np.arange(g.cols), indexing="ij")
    cells = g.cell_index(0, rows.ravel(), cols.ravel())
    cells = cells[cells >= 0]
    assert cells.size == g.n_words * g.W
    assert np.unique(cells).size == cells.size
    # physically adjacent columns belong to different words
    c = g.cell_index(0, np.zeros(4, int), np.arange(4))
    assert np.unique(c // g.W).size == 4


def test_nonuniform_collision_factor():
    LA, W, m = 2 ** 18, 72, 1500
    s = R.NonUniformSampler(LA, W, block_words=256, block_sigma=0.8, device_seed=2)
    kappa = s.collision_factor()
    assert kappa > 1.5
    r = MC.simulate(LA, W, m, 2000, sampler=s, seed=5)
    mean, ci = r.nfm(2)
    assert abs(mean - A.nfm(2, m, LA / kappa, W)) < 3 * ci


def test_tid_persists_through_scrubs():
    LA, W = 2 ** 14, 72
    tid = R.TIDModel(5e-4, MC.UniformSampler(LA, W))
    r0 = MC.simulate(LA, W, 0, 200, tid=tid, n_scrubs=0, seed=1)
    r3 = MC.simulate(LA, W, 0, 200, tid=tid, n_scrubs=3, seed=1)
    # same persistent errors are seen in each of the 4 intervals
    assert r3.mean_nbf()[0] == pytest.approx(4 * r0.mean_nbf()[0], rel=0.1)


def test_device_sefi_corrected_by_snc():
    LA, W = 2 ** 14, 72
    sefi = R.SEFIModel(None, rate=1, kind="device", span=200, p_flip=0.6, device_bits=4,
                       W=W, n_words=LA, poisson=False)
    codes = (SECDED(72, 64), SNCDND(72, 64, 4))
    r = MC.simulate(LA, W, 0, 50, codes=codes, sefi=sefi, flip_mode="set", seed=2)
    assert r.p_failure(codes[0].name)[0] == 1.0
    assert r.p_failure(codes[1].name)[0] == 0.0
