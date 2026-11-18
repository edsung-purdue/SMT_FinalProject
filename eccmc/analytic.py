"""Closed-form predictions from Clemente, Rezaei & Franco, "Reliability of Error
Correction Codes Against Multiple Events by Accumulation", IEEE TNS 69(2), 2022.

Notation (paper Section II):
    LA : number of addresses (words)
    W  : word width in bits (for an (n, k) code this is the codeword width n)
    L  : LA * W memory size in bits
    m  : number of bitflips that actually occurred (particle hits on cells)
    NBF: number of observed flipped cells (double hits on one cell cancel)

Every function is vectorised over ``m`` (numpy arrays are accepted).

Besides the paper's equations, ``nfm_exact`` gives an exact reference
(binomial word occupancy x Markov chain for the number of odd-hit cells),
used to check both the Monte Carlo engine and the paper's approximations.
"""
from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Eq. (1) and Eq. (2)
# ---------------------------------------------------------------------------


def m_from_nbf(nbf, L):
    """Eq. (1): actual bitflips m from observed bitflips NBF."""
    nbf = np.asarray(nbf, dtype=float)
    return np.round(nbf + nbf ** 2 / L)


def expected_nbf(m, L):
    """Exact expectation of observed flipped cells after m uniform toggles."""
    m = np.asarray(m, dtype=float)
    return L * (1.0 - (1.0 - 2.0 / L) ** m) / 2.0


def _binom_pmf(k, m, p):
    """Binomial pmf that also accepts non-integer m (needed for m/(s+1))."""
    m = np.asarray(m, dtype=float)
    k = np.asarray(k, dtype=float)
    valid = (k >= 0) & (k <= m)
    mm = np.where(valid, m, k)
    logc = gammaln(mm + 1) - gammaln(k + 1) - gammaln(mm - k + 1)
    with np.errstate(divide="ignore"):
        logp = logc + k * np.log(p) + (mm - k) * np.log1p(-p)
    return np.where(valid, np.exp(logp), 0.0)


def n_hit(k, m, LA):
    """Eq. (2): expected number of addresses hit exactly k times.

    N_H(k) = C(m, k) * LA * (1/LA)^k * (1 - 1/LA)^(m-k) = LA * Binom(k; m, 1/LA).
    """
    return LA * _binom_pmf(k, m, 1.0 / LA)


def n_hit_atleast(k, m, LA, extra=80):
    """Expected number of addresses hit k or more times.

    Summed term by term (no 1 - cdf cancellation), so it stays accurate when
    the result is ~1e-10 of LA, as needed for Table II.
    """
    m = np.asarray(m, dtype=float)
    lam = float(np.max(m)) / LA
    jmax = int(k + extra + 20 * lam)
    return sum(n_hit(j, m, LA) for j in range(k, jmax + 1))


# ---------------------------------------------------------------------------
# Eqs. (3)-(6) and (22): number of false k-bit MBUs
# ---------------------------------------------------------------------------


def _falling(W, k):
    """W (W-1) ... (W-k+1)."""
    out = 1.0
    for i in range(k):
        out *= W - i
    return out


def nfm2(m, LA, W):
    """Eq. (3)."""
    return (W - 1) / W * (n_hit(2, m, LA) + 6 * (W - 2) / W ** 2 * n_hit(4, m, LA))


def nfm3(m, LA, W):
    """Eq. (4)."""
    return (W - 1) * (W - 2) / W ** 2 * (
        n_hit(3, m, LA) + 10 * (W - 3) / W ** 2 * n_hit(5, m, LA))


def nfm4(m, LA, W):
    """Eq. (5)."""
    return (W - 1) * (W - 2) * (W - 3) / W ** 3 * (
        n_hit(4, m, LA) + 5 * (3 * W - 8) / W ** 2 * n_hit(6, m, LA))


def nfm_eq6(k, m, LA, W):
    """Eq. (6): first-order estimate from the authors' earlier work [16]."""
    return _falling(W, k) / W ** k * n_hit(k, m, LA)


def nfm_eq22(k, m, LA, W):
    """Eq. (22): N_FM,A(k) + N_FM,B(k) for any multiplicity k (Appendix B)."""
    a = _falling(W, k) / W ** k * n_hit(k, m, LA)
    b = comb(k + 2, 2) * _falling(W, k + 1) / W ** (k + 2) * n_hit(k + 2, m, LA)
    return a + b


def nfm(k, m, LA, W, method="paper"):
    """Expected number of words showing exactly k flipped bits.

    method:
        "paper" : Eqs. (3)-(5) for k = 2..4 and Eq. (22) otherwise
        "eq6"   : Eq. (6)
        "eq22"  : Eq. (22) for all k
        "exact" : exact combinatorial reference (see ``nfm_exact``)
    """
    if method == "paper":
        if k == 2:
            return nfm2(m, LA, W)
        if k == 3:
            return nfm3(m, LA, W)
        if k == 4:
            return nfm4(m, LA, W)
        return nfm_eq22(k, m, LA, W)
    if method == "eq6":
        return nfm_eq6(k, m, LA, W)
    if method == "eq22":
        return nfm_eq22(k, m, LA, W)
    if method == "exact":
        return nfm_exact(k, m, LA, W)
    raise ValueError(method)


@lru_cache(maxsize=None)
def _odd_cells_given_hits(W, hmax):
    """P[#cells with odd hit count = j | h hits in the word], h = 0..hmax.

    Each hit toggles a uniformly chosen cell, so the number of odd cells is a
    Markov chain: j -> j-1 with prob j/W, j -> j+1 with prob (W-j)/W.
    """
    P = np.zeros((hmax + 1, W + 1))
    P[0, 0] = 1.0
    j = np.arange(W + 1)
    down = j / W
    up = (W - j) / W
    for h in range(1, hmax + 1):
        prev = P[h - 1]
        cur = np.zeros(W + 1)
        cur[:-1] += prev[1:] * down[1:]
        cur[1:] += prev[:-1] * up[:-1]
        P[h] = cur
    return P


def nfm_exact(k, m, LA, W, hmax=None):
    """Exact expected number of words with exactly k observed bitflips."""
    m_arr = np.atleast_1d(np.asarray(m, dtype=float))
    lam = m_arr.max() / LA
    if hmax is None:
        hmax = int(max(30, k + 12, np.ceil(lam + 12 * np.sqrt(lam + 1) + 12)))
    P = _odd_cells_given_hits(W, hmax)
    h = np.arange(hmax + 1)
    # LA * Binom(h; m, 1/LA) for all h, shape (len(m), hmax+1)
    pmf = _binom_pmf(h[None, :], m_arr[:, None], 1.0 / LA)
    out = LA * pmf @ P[:, k]
    return out if np.ndim(m) else float(out[0])


def nfm_plus(k, m, LA, W, method="paper", kmax=None):
    """Eq. (13): expected number of false MBUs with multiplicity >= k."""
    if kmax is None:
        kmax = W
    kmax = min(kmax, W, k + 16)  # higher terms are negligible
    return sum(nfm(i, m, LA, W, method) for i in range(k, kmax + 1))


# ---------------------------------------------------------------------------
# Eqs. (7)-(11): non-adjacent false MBUs
# ---------------------------------------------------------------------------


def adjacent_fraction(k, W):
    """Fraction of k-bit patterns whose bits are all consecutive (Eqs. 7, 9, 11)."""
    if k < 2:
        return 1.0
    den = 1.0
    for i in range(k - 1):
        den *= W - i
    return float(np.prod(np.arange(1, k + 1))) / den


def nfm_na(k, m, LA, W, method="paper"):
    """Eq. (11) (Eqs. (8) and (10) for k = 2, 3): k-bit MBUs not all adjacent."""
    return nfm(k, m, LA, W, method) * (1.0 - adjacent_fraction(k, W))


# ---------------------------------------------------------------------------
# Eq. (12) and (23): MBUs affecting different b-bit nibbles (SNC-DND)
# ---------------------------------------------------------------------------


def nfm_dn(m, LA, W, b, method="paper"):
    """Eq. (12): words with bitflips in two or more different b-bit nibbles.

    Computed as  sum_{h>=2} N_H(h) - N_H(2)/W - (3W-2)/W^2 N_H(3) - NUA2 - NUA3 - NUA4,
    which is algebraically identical to L_A - (NUA0 + ... + NUA4) but avoids
    catastrophic cancellation for very small values (Table II needs ~1e-8).
    """
    nua2 = (b - 1) / (W - 1) * nfm(2, m, LA, W, method)
    nua3 = (b - 1) * (b - 2) / ((W - 1) * (W - 2)) * nfm(3, m, LA, W, method)
    nua4 = (b - 1) * (b - 2) * (b - 3) / ((W - 1) * (W - 2) * (W - 3)) * nfm(4, m, LA, W, method)
    return (n_hit_atleast(2, m, LA) - n_hit(2, m, LA) / W
            - (3 * W - 2) / W ** 2 * n_hit(3, m, LA) - nua2 - nua3 - nua4)


def nfm_dn_naive(m, LA, W, b, method="paper"):
    """Eq. (12) exactly as printed (subject to cancellation for tiny values)."""
    nua0 = n_hit(0, m, LA) + n_hit(2, m, LA) / W
    nua1 = n_hit(1, m, LA) + (3 * W - 2) / W ** 2 * n_hit(3, m, LA)
    nua2 = (b - 1) / (W - 1) * nfm(2, m, LA, W, method)
    nua3 = (b - 1) * (b - 2) / ((W - 1) * (W - 2)) * nfm(3, m, LA, W, method)
    nua4 = (b - 1) * (b - 2) * (b - 3) / ((W - 1) * (W - 2) * (W - 3)) * nfm(4, m, LA, W, method)
    return LA - (nua0 + nua1 + nua2 + nua3 + nua4)


def same_nibble_fraction(k, W, b):
    """P(k distinct uniform bit positions all fall in one b-bit nibble)."""
    if k > b:
        return 0.0
    return (W // b) * comb(b, k) / comb(W, k)


def nfm_dn_exact(m, LA, W, b, kmax=None):
    """Exact reference for Eq. (12)."""
    kmax = min(W, kmax or 16)
    return sum(nfm_exact(k, m, LA, W) * (1 - same_nibble_fraction(k, W, b))
               for k in range(2, kmax + 1))


# ---------------------------------------------------------------------------
# Eqs. (14)-(21): Poisson model of the probability of data-integrity loss
# ---------------------------------------------------------------------------


def poisson_pmf(lam, i):
    """Eq. (14)."""
    lam = np.asarray(lam, dtype=float)
    return np.exp(-lam + i * np.log(np.where(lam > 0, lam, 1e-300)) - gammaln(i + 1))


def poisson_cdf(lam, i):
    """Eq. (15)."""
    return sum(poisson_pmf(lam, j) for j in range(i + 1))


def failure_rate(code, m, LA, method="paper"):
    """Expected number of uncorrectable words (lambda) for an ``ECCCode``."""
    W = code.n
    kind = code.kind
    if kind == "SEC-DED":
        return nfm_plus(2, m, LA, W, method)
    if kind in ("DEC-TED", "DEC"):
        return nfm_plus(3, m, LA, W, method)
    if kind == "TEC":
        return nfm_plus(4, m, LA, W, method)
    if kind == "DAEC":
        return nfm_na(2, m, LA, W, method) + nfm_plus(3, m, LA, W, method)
    if kind == "TAEC":
        if code.taec_rule == "eq20":
            return nfm_na(3, m, LA, W, method) + nfm_plus(4, m, LA, W, method)
        # rule used for the plotted curves (Fig. 9): 2-bit non-adjacent also fails
        return (nfm_na(2, m, LA, W, method) + nfm_na(3, m, LA, W, method)
                + nfm_plus(4, m, LA, W, method))
    if kind == "SNC-DND":
        if method == "exact":
            return nfm_dn_exact(m, LA, W, code.b)
        return nfm_dn(m, LA, W, code.b, method)
    raise ValueError(kind)


def p_failure(code, m, LA, method="paper"):
    """Eqs. (16)-(21): P(data integrity loss) = 1 - exp(-lambda)."""
    return -np.expm1(-failure_rate(code, m, LA, method))


def max_bitflips_for_reliability(code, LA, reliability, method="paper", m_hi=10**8,
                                 convention="first_violation"):
    """Table II: number of bitflips at which ``code`` stops meeting ``reliability``.

    convention="first_violation" (default, matches the printed Table II):
        smallest m with 1 - P_fail(m) < reliability.
    convention="last_ok": largest m with 1 - P_fail(m) >= reliability (one less).
    """
    target = 1.0 - reliability
    lo, hi = 0, 1
    while p_failure(code, hi, LA, method) <= target:
        lo, hi = hi, hi * 2
        if hi > m_hi:
            return np.inf
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p_failure(code, mid, LA, method) <= target:
            lo = mid
        else:
            hi = mid
    return hi if convention == "first_violation" else lo


# ---------------------------------------------------------------------------
# Section III-B: scrubbing (Fig. 10)
# ---------------------------------------------------------------------------


def nfm_plus_scrubbed(k, m, LA, W, n_scrubs, method="paper"):
    """Expected false MBUs (multiplicity >= k) with ``n_scrubs`` equally spaced
    scrubbing cycles: the m bitflips are split into n_scrubs + 1 intervals and
    the memory is clean at the start of each one."""
    parts = n_scrubs + 1
    return parts * nfm_plus(k, np.asarray(m, dtype=float) / parts, LA, W, method)
