"""Sparse, vectorised Monte Carlo engine.

The paper (Section II-A) simulates the memory as a boolean matrix of LA x W
cells, flips m randomly chosen cells (a cell hit twice flips back), counts the
resulting multi-bit words and repeats the experiment many times.

Here the memory is never allocated.  Each trial draws m cell indices, the
indices of a whole batch of trials are tagged with the trial number, sorted
once, and cells hit an even number of times are dropped.  That makes a
256 MB memory as cheap to simulate as a 2 Mbit one and allows ~10^7 hits per
second on one core.

Extensions (see ``eccmc.radiation``):
  * ``sampler``    non-uniform cell susceptibility (leakage / weak cells / data pattern)
  * ``tid``        persistent TID-induced errors that survive scrubbing
  * ``sefi``       correlated multi-cell events (row / column / device lines)
  * ``n_scrubs``   equally spaced scrubbing cycles
  * ``flip_mode``  "toggle" (paper, SRAM) or "set" (DRAM charge loss: a second
                   hit cannot restore the cell)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t as student_t

from .codes import ECCCode, WordErrors


class UniformSampler:
    """Every cell is equally likely to be hit (paper 1 assumption)."""

    def __init__(self, n_words: int, W: int):
        self.n_words = int(n_words)
        self.W = int(W)
        self.n_cells = self.n_words * self.W

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.integers(0, self.n_cells, size=n, dtype=np.int64)

    def collision_factor(self) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# Analysis of a sorted set of flipped cells
# ---------------------------------------------------------------------------


def _flipped_cells(keys: np.ndarray, flip_mode: str) -> np.ndarray:
    keys.sort(kind="stable" if keys.size < 1000 else "quicksort")
    if keys.size == 0:
        return keys
    new = np.empty(keys.size, dtype=bool)
    new[0] = True
    np.not_equal(keys[1:], keys[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    uniq = keys[starts]
    if flip_mode == "set":
        return uniq
    counts = np.diff(np.append(starts, keys.size))
    return uniq[(counts & 1) == 1]


def analyse_cells(keys: np.ndarray, W: int, n_words: int) -> WordErrors:
    """Group sorted unique flipped-cell keys (trial*L + cell) into words."""
    wk = keys // W
    bits = keys - wk * W
    if keys.size == 0:
        e = np.empty(0, dtype=np.int64)
        return WordErrors(e, e, e, e, e, e, e)
    new = np.empty(keys.size, dtype=bool)
    new[0] = True
    np.not_equal(wk[1:], wk[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    k = np.diff(np.append(starts, keys.size))
    first = bits[starts]
    last = bits[starts + k - 1]
    w = wk[starts]
    trial = w // n_words
    return WordErrors(trial=trial, word=w - trial * n_words, k=k, first=first,
                      last=last, starts=starts, bits=bits)


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------


def t95(n: int) -> float:
    return float(student_t.ppf(0.975, max(n - 1, 1)))


@dataclass
class MCResult:
    n_words: int
    W: int
    m: int
    n_trials: int
    n_scrubs: int
    kmax: int
    nbf: np.ndarray                                  # observed flipped bits per trial
    n_k: np.ndarray                                  # (trials, kmax+2): words with k bits; last col = >kmax
    n_na: np.ndarray                                 # words with k bits not all adjacent
    n_dn: dict = field(default_factory=dict)         # b -> words spanning >1 nibble
    fail_words: dict = field(default_factory=dict)   # code name -> failing words per trial
    failed: dict = field(default_factory=dict)       # code name -> bool per trial

    @staticmethod
    def _stats(x):
        x = np.asarray(x, dtype=float)
        n = x.size
        mean = x.mean()
        sd = x.std(ddof=1) if n > 1 else 0.0
        return mean, t95(n) * sd / np.sqrt(n)

    def nfm(self, k):
        """Mean and 95% half-width of the number of words with exactly k flips."""
        return self._stats(self.n_k[:, k])

    def nfm_plus(self, k):
        return self._stats(self.n_k[:, k:].sum(axis=1))

    def nfm_na(self, k):
        return self._stats(self.n_na[:, k])

    def nfm_dn(self, b):
        return self._stats(self.n_dn[b])

    def mean_nbf(self):
        return self._stats(self.nbf)

    def failure_rate(self, name):
        """Mean number of uncorrectable words (lambda)."""
        return self._stats(self.fail_words[name])

    def p_failure(self, name):
        """Direct estimate of P(at least one uncorrectable word) with a Wilson 95% CI."""
        x = self.failed[name]
        n = x.size
        p = x.mean()
        z = 1.959963984540054
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return p, max(centre - half, 0.0), min(centre + half, 1.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def simulate(n_words: int, W: int, m: int, n_trials: int, *,
             codes: tuple[ECCCode, ...] = (), nibbles: tuple[int, ...] = (),
             sampler=None, tid=None, sefi=None, n_scrubs: int = 0,
             flip_mode: str = "toggle", kmax: int = 8, seed=None,
             rng: np.random.Generator | None = None,
             max_batch_hits: int = 8_000_000) -> MCResult:
    """Run ``n_trials`` independent irradiations of ``m`` bitflips each.

    With ``n_scrubs`` > 0 the m bitflips are split evenly over n_scrubs + 1
    intervals; the memory is scrubbed (all transient errors removed) between
    them.  Counts are summed over the intervals and a code is counted as
    failed if any interval holds an uncorrectable word, i.e. the Fig. 10 set-up.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if sampler is None:
        sampler = UniformSampler(n_words, W)
    for c in codes:
        if c.n != W:
            raise ValueError(f"code {c.name} has n={c.n} but memory word is W={W}")
    if flip_mode not in ("toggle", "set"):
        raise ValueError(flip_mode)
    L = n_words * W
    parts = n_scrubs + 1
    m_part = np.full(parts, m // parts, dtype=np.int64)
    m_part[: m % parts] += 1

    nbf = np.zeros(n_trials, dtype=np.int64)
    n_k = np.zeros((n_trials, kmax + 2), dtype=np.int64)
    n_na = np.zeros((n_trials, kmax + 2), dtype=np.int64)
    n_dn = {b: np.zeros(n_trials, dtype=np.int64) for b in nibbles}
    fail_words = {c.name: np.zeros(n_trials, dtype=np.int64) for c in codes}
    failed = {c.name: np.zeros(n_trials, dtype=bool) for c in codes}

    per_trial = max(int(m), 1) + (tid.expected_count() * parts if tid is not None else 0)
    batch = int(max(1, min(n_trials, max_batch_hits // max(per_trial, 1))))

    for t0 in range(0, n_trials, batch):
        nt = min(batch, n_trials - t0)
        n_sub = nt * parts
        reps = np.tile(m_part, nt)
        sub = np.repeat(np.arange(n_sub, dtype=np.int64), reps)
        keys = sub * L + sampler.sample(rng, int(reps.sum()))
        keys = _flipped_cells(keys, flip_mode)

        extra = []
        if tid is not None:      # persistent: same cells in every interval of a trial
            tr, cells = tid.sample(rng, nt)
            for p in range(parts):
                extra.append((tr * parts + p) * L + cells)
        if sefi is not None:
            sb, cells = sefi.sample(rng, n_sub, parts)
            extra.append(sb * L + cells)
        if extra:
            keys = np.union1d(keys, np.concatenate(extra))

        we = analyse_cells(keys, W, n_words)
        tr_of_sub = we.trial // parts  # local trial index

        def per_trial_sum(mask=None, weights=None):
            idx = tr_of_sub if mask is None else tr_of_sub[mask]
            return np.bincount(idx, weights=weights, minlength=nt)[:nt].astype(np.int64)

        nbf[t0:t0 + nt] = np.bincount(tr_of_sub, weights=we.k, minlength=nt)[:nt].astype(np.int64)
        kk = np.minimum(we.k, kmax + 1)
        nonadj = ~we.adjacent
        flat = tr_of_sub * (kmax + 2) + kk
        n_k[t0:t0 + nt] = np.bincount(flat, minlength=nt * (kmax + 2)).reshape(nt, kmax + 2)
        n_na[t0:t0 + nt] = np.bincount(flat[nonadj], minlength=nt * (kmax + 2)).reshape(nt, kmax + 2)
        for b in nibbles:
            span = (we.k >= 2) & (we.first // b != we.last // b)
            n_dn[b][t0:t0 + nt] = per_trial_sum(span)
        for c in codes:
            f = c.fails(we)
            fail_words[c.name][t0:t0 + nt] = per_trial_sum(f)
            failed[c.name][t0:t0 + nt] = per_trial_sum(f) > 0

    return MCResult(n_words=n_words, W=W, m=int(m), n_trials=n_trials, n_scrubs=n_scrubs,
                    kmax=kmax, nbf=nbf, n_k=n_k, n_na=n_na, n_dn=n_dn,
                    fail_words=fail_words, failed=failed)


def trials_for(m: int, budget_hits: float = 5e7, lo: int = 200, hi: int = 100_000) -> int:
    """Adaptive number of trials: the paper used 1e5 everywhere; we keep 1e5
    for small m (where events are rare) and cap total work for large m."""
    return int(np.clip(budget_hits // max(m, 1), lo, hi))
