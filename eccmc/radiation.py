"""Extensions motivated by Ryu et al., "Single-Event Upset and Total Ionizing
Dose Effects on DDR4 DRAM Due to Proton Irradiation Under Different
Temperatures", IEEE TED 72(9), 2025.

Paper 1 assumes every cell is equally likely to flip and that every particle
flips at most one cell of a word.  Paper 2 shows three things that break this:

1. Non-uniform susceptibility (leakage / GIDL).  SEU error density in DDR4
   rises from 9.77e-10 (153 K) to ~1.4e-7 (300 K) at 3e10 p/cm^2, driven by
   gate-induced drain leakage that degrades retention.  Leakage is not
   uniform: a small population of weak (leaky) cells, and rows/regions with
   worse retention, dominate.  -> ``NonUniformSampler``.
2. TID-induced errors.  After 4.8 krad(SiO2) the error density measured at
   room temperature is 2.5e-7 (300 K irradiation), 2.3e-7 (373 K) and 2.8e-8
   (153 K) with a 128 ms refresh.  These are interface-trap / retention
   errors: they are persistent (reappear after every refresh/scrub, only
   slowly annealing) and they live preferentially in the leaky cells.
   -> ``TIDModel``.
3. SEFI.  At 373 K, SEFIs produced lines of errors aligned along one
   address of the bitmap (Fig. 3(a)), adding 1.87e-6 (S2) and 3.47e-6 (S1)
   error density, i.e. ~1000-1900 bits in a 2^29-bit bank.  These errors are correlated and aligned
   with the array geometry, so whether they defeat an ECC depends on how
   logical words are mapped to the physical array.  -> ``SEFIModel``.

Numbers marked "digitised" below were read from the paper's text and
figures; the shape parameters of the susceptibility distributions are model
assumptions (paper 2 does not report a per-cell distribution) and are exposed
as arguments so they can be swept.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Paper 2 measured data (DDR4, 48 MeV protons, fluence 3e10 cm^-2, 4.8 krad)
# ---------------------------------------------------------------------------

PAPER2_FLUENCE = 3e10          # p/cm^2
PAPER2_DOSE_KRAD = 4.8         # krad(SiO2) at the final fluence
DDR4_BANK_ROWS = 65536         # Fig. 3(a) bitmap axes
DDR4_BANK_COLS = 8192

#: SEU error density (errors / bank bit) at the end of irradiation, Fig. 2 / text.
SEU_ERROR_DENSITY = {
    # (T [K], refresh [ms]) : density
    (373, 64): 5.59e-6,   # S1, includes SEFI contributions
    (373, 32): 1.97e-6,   # S2, includes one SEFI (~1.87e-6)
    (300, 64): 1.45e-7,   # S3
    (300, 32): 1.34e-7,   # S4
    (153, 64): 9.77e-10,  # S5
    (153, 128): 9.77e-10,  # S6 (refresh had no effect at 153 K)
}

#: SEU-only density at 373 K once the SEFI steps are removed (digitised, Fig. 2(a)).
SEU_ERROR_DENSITY_NO_SEFI = {
    (373, 64): 5.59e-6 - 3.47e-6,  # ~2.1e-6: S1 total minus its SEFI contribution (3.47e-6)
    (373, 32): 1.97e-6 - 1.87e-6,  # ~1.0e-7: S2 total minus its SEFI step (1.87e-6)
    (300, 64): 1.45e-7,
    (300, 32): 1.34e-7,
    (153, 64): 9.77e-10,
    (153, 128): 9.77e-10,
}

#: TID-induced (post-irradiation, read at 300 K) error density, Case 1 of Fig. 4.
TID_ERROR_DENSITY = {
    300: 2.5e-7,   # S3+S4, highest at every refresh period
    373: 2.3e-7,   # S1+S2 at 128 ms refresh
    153: 2.8e-8,   # S5+S6 at 128 ms refresh
}

#: SEFI observations at 373 K (Fig. 2(a)): one SEFI per DUT over the full fluence;
#: SEFI error density 1.87e-6 (S2, 32 ms) and 3.47e-6 (S1, 64 ms) per bank bit,
#: i.e. ~1000 and ~1860 error bits in a 2^29-bit bank.
SEFI_373K = dict(events_per_fluence=1.0 / PAPER2_FLUENCE,
                 bits_per_event={32: int(1.87e-6 * DDR4_BANK_ROWS * DDR4_BANK_COLS),
                                 64: int(3.47e-6 * DDR4_BANK_ROWS * DDR4_BANK_COLS)})


def seu_cross_section(T: int, refresh_ms: int = 64, include_sefi: bool = False) -> float:
    """Per-bit SEU cross-section (cm^2/bit) implied by paper 2's densities."""
    table = SEU_ERROR_DENSITY if include_sefi else SEU_ERROR_DENSITY_NO_SEFI
    return table[(T, refresh_ms)] / PAPER2_FLUENCE


def expected_bitflips(fluence: float, n_bits: int, T: int, refresh_ms: int = 64) -> float:
    """Expected number of SEU bitflips m for a memory of n_bits at a given fluence."""
    return seu_cross_section(T, refresh_ms) * fluence * n_bits


def tid_density(T: int, dose_krad: float) -> float:
    """TID error density per bit, linear in dose (paper 2 has a single dose point)."""
    return TID_ERROR_DENSITY[T] * dose_krad / PAPER2_DOSE_KRAD


#: Susceptibility shape presets (model assumptions, see module docstring).
#: Leakage tails widen with temperature (GIDL / band-to-band tunnelling).
LEAKAGE_PRESETS = {
    153: dict(block_sigma=0.2, weak_fraction=1e-4, weak_ratio=10.0),
    300: dict(block_sigma=0.6, weak_fraction=1e-3, weak_ratio=50.0),
    373: dict(block_sigma=0.9, weak_fraction=3e-3, weak_ratio=100.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _splitmix64(x: np.ndarray, seed: int) -> np.ndarray:
    z = (x.astype(np.uint64) + np.uint64((seed * 0x9E3779B97F4A7C15) % 2**64)) * np.uint64(1)
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


# ---------------------------------------------------------------------------
# Physical geometry and word interleaving
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    """Banks of ``rows`` x ``cols`` cells holding W-bit codewords.

    Each row is cut into segments of W*interleave columns.  Inside a segment,
    bit j of word i (0 <= i < interleave) sits at column j*interleave + i, so
    physically adjacent cells in a row belong to different words (bit
    interleaving).  Columns left over at the end of a row are unused.
    """
    W: int
    rows: int = DDR4_BANK_ROWS
    cols: int = DDR4_BANK_COLS
    banks: int = 1
    interleave: int = 1

    @property
    def seg_cols(self):
        return self.W * self.interleave

    @property
    def words_per_row(self):
        return (self.cols // self.seg_cols) * self.interleave

    @property
    def n_words(self):
        return self.banks * self.rows * self.words_per_row

    def cell_index(self, bank, row, col):
        """Logical cell index (word*W + bit) of physical cells; -1 if unused column."""
        bank = np.asarray(bank, dtype=np.int64)
        row = np.asarray(row, dtype=np.int64)
        col = np.asarray(col, dtype=np.int64)
        seg, off = np.divmod(col, self.seg_cols)
        bit, sub = np.divmod(off, self.interleave)
        word = ((bank * self.rows + row) * (self.cols // self.seg_cols) + seg) * self.interleave + sub
        valid = seg < (self.cols // self.seg_cols)
        return np.where(valid, word * self.W + bit, -1)


# ---------------------------------------------------------------------------
# 1. Non-uniform susceptibility
# ---------------------------------------------------------------------------


class NonUniformSampler:
    """Draw hit cells with probability proportional to a per-cell susceptibility.

    susceptibility(cell) = B[block(cell)] * (weak_ratio if cell is weak else 1)
                           * (1 if cell is vulnerable else 0)

    * ``block_words`` / ``block_sigma``: memory split into blocks of
      ``block_words`` consecutive words (e.g. one DRAM row); each block has a
      lognormal(0, block_sigma) weight - regional retention / leakage spread.
    * ``weak_fraction`` / ``weak_ratio``: a fixed random set of weak (leaky)
      cells that are ``weak_ratio`` times more sensitive (GIDL tail).
    * ``vulnerable_fraction``: fraction of cells whose stored value can be
      upset (DRAM: only charged cells lose data; paper 2 saw no upsets with
      "all 0").  1.0 reproduces the "all 1" pattern.

    The weak set and block weights are drawn once (``device_seed``): one
    sampler represents one physical device.
    """

    def __init__(self, n_words: int, W: int, *, block_words: int | None = None,
                 block_sigma: float = 0.0, weak_fraction: float = 0.0,
                 weak_ratio: float = 1.0, vulnerable_fraction: float = 1.0,
                 device_seed: int = 0):
        self.n_words = int(n_words)
        self.W = int(W)
        self.n_cells = self.n_words * self.W
        self.vulnerable_fraction = float(vulnerable_fraction)
        self.hash_seed = device_seed + 12345
        drng = np.random.default_rng(device_seed)

        self.block_words = int(block_words or self.n_words)
        self.n_blocks = -(-self.n_words // self.block_words)
        if block_sigma > 0 and self.n_blocks > 1:
            bw = drng.lognormal(0.0, block_sigma, self.n_blocks)
        else:
            bw = np.ones(self.n_blocks)
        self.block_weight = bw / bw.mean()
        sizes = np.full(self.n_blocks, self.block_words * self.W, dtype=float)
        sizes[-1] = self.n_cells - (self.n_blocks - 1) * self.block_words * self.W
        self._block_cells = sizes
        self._block_cdf = np.cumsum(self.block_weight * sizes)

        self.weak_ratio = float(weak_ratio)
        n_weak = int(round(weak_fraction * self.n_cells)) if weak_ratio != 1.0 else 0
        self.n_weak = n_weak
        # Large weak populations (e.g. 3e8 cells in a 32 GB DIMM) are not stored:
        # weak cell j is (A*j + off) mod L, a fixed pseudo-random spread-out set.
        self.implicit_weak = n_weak > self.MAX_EXPLICIT_WEAK
        if self.implicit_weak:
            A = int(drng.integers(2**20, 2**31)) | 1   # A * j < 2^63 for j < 2^32
            while np.gcd(A, self.n_cells) != 1:
                A += 2
            self._weak_A, self._weak_off = A, int(drng.integers(self.n_cells))
            probe = self._weak_cell(drng.integers(0, n_weak, 200_000))
            self._weak_bmean = float(self.block_weight[(probe // self.W) // self.block_words].mean())
            self._bmax = float(self.block_weight.max())
            self.weak_cells = np.empty(0, np.int64)
            extra_total = (self.weak_ratio - 1.0) * n_weak * self._weak_bmean
            self._weak_cdf = np.zeros(1)
        else:
            weak = (np.unique(drng.integers(0, self.n_cells, n_weak, dtype=np.int64))
                    if n_weak else np.empty(0, np.int64))
            self.weak_cells = weak
            if weak.size:
                wblk = (weak // self.W) // self.block_words
                self._weak_cdf = np.cumsum(self.block_weight[wblk] * (self.weak_ratio - 1.0))
                extra_total = self._weak_cdf[-1]
            else:
                self._weak_cdf = np.zeros(1)
                extra_total = 0.0
        base_total = self._block_cdf[-1]
        self._p_weak = extra_total / (base_total + extra_total)
        self._drng = drng

    MAX_EXPLICIT_WEAK = 4_000_000

    def _weak_cell(self, j):
        j = np.asarray(j, dtype=np.int64)
        # (A*j + off) mod L without int64 overflow, via Python ints in object dtype when needed
        if self._weak_A * max(self.n_weak, 1) < 2**63:
            return (j * self._weak_A + self._weak_off) % self.n_cells
        jo = j.astype(object)
        return np.array((jo * self._weak_A + self._weak_off) % self.n_cells, dtype=np.int64)

    def _draw_weak(self, rng, n):
        if not self.implicit_weak:
            idx = np.searchsorted(self._weak_cdf, rng.random(n) * self._weak_cdf[-1], side="right")
            return self.weak_cells[np.minimum(idx, self.weak_cells.size - 1)]
        parts, have = [], 0
        while have < n:   # rejection on the block weight
            k = max(64, int((n - have) * self._bmax / max(self._weak_bmean, 1e-9) * 1.2))
            k = min(k, 5_000_000)
            c = self._weak_cell(rng.integers(0, self.n_weak, k))
            b = self.block_weight[(c // self.W) // self.block_words]
            c = c[rng.random(k) * self._bmax < b][: n - have]
            parts.append(c)
            have += c.size
        return np.concatenate(parts)

    # -- sampling ---------------------------------------------------------
    def is_vulnerable(self, cells: np.ndarray) -> np.ndarray:
        if self.vulnerable_fraction >= 1.0:
            return np.ones(cells.shape, dtype=bool)
        h = _splitmix64(cells, self.hash_seed)
        return h < np.uint64(int(self.vulnerable_fraction * 2**64 - 1))

    def _draw(self, rng, n):
        out = np.empty(n, dtype=np.int64)
        use_weak = rng.random(n) < self._p_weak
        nw = int(use_weak.sum())
        if nw:
            out[use_weak] = self._draw_weak(rng, nw)
        nb = n - nw
        if nb:
            blk = np.searchsorted(self._block_cdf, rng.random(nb) * self._block_cdf[-1], side="right")
            blk = np.minimum(blk, self.n_blocks - 1)
            start = blk * self.block_words * self.W
            off = (rng.random(nb) * self._block_cells[blk]).astype(np.int64)
            out[~use_weak] = start + off
        return out

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.vulnerable_fraction >= 1.0:
            return self._draw(rng, n)
        parts, have = [], 0
        while have < n:
            need = n - have
            c = self._draw(rng, int(need / self.vulnerable_fraction * 1.1) + 16)
            c = c[self.is_vulnerable(c)][:need]
            parts.append(c)
            have += c.size
        return np.concatenate(parts)

    # -- analytic bridge ---------------------------------------------------
    def collision_factor(self) -> float:
        """kappa such that the paper's equations hold with LA_eff = LA / kappa.

        Two hits form a false 2-bit MBU when they land in the same word but in
        different cells, so to first order
            N_FM(2) ~ C(m,2) * sum_w (P_w^2 - sum_{c in w} P_c^2),
        with P_w / P_c the probability that a hit lands in word w / cell c.
        For uniform hits this is C(m,2)/LA * (W-1)/W, hence
            kappa = LA * sum_w (P_w^2 - sum_c P_c^2) / ((W-1)/W).
        """
        W, bw = self.W, self.block_words
        bweight = self.block_weight
        words_per_block = np.full(self.n_blocks, bw, dtype=float)
        words_per_block[-1] = self.n_words - (self.n_blocks - 1) * bw
        # words without weak cells: word weight B*W, sum of squared cell weights B^2*W
        s_pair = np.sum(bweight ** 2 * words_per_block * (W ** 2 - W))
        s1 = np.sum(bweight * words_per_block * W)
        scale = 1.0
        if self.implicit_weak:
            j = (np.arange(self.n_weak) if self.n_weak <= 1_000_000
                 else self._drng.integers(0, self.n_weak, 1_000_000))
            wcells = self._weak_cell(j)
            scale = self.n_weak / wcells.size
        else:
            wcells = self.weak_cells
        if wcells.size:
            wword, n = np.unique(wcells // W, return_counts=True)
            b = bweight[wword // bw]
            r = self.weak_ratio
            heavy = W + (r - 1.0) * n
            cells_sq = (W - n) + n * r ** 2
            s_pair += scale * np.sum(b ** 2 * ((heavy ** 2 - cells_sq) - (W ** 2 - W)))
            s1 += scale * np.sum(b * (heavy - W))
        kappa = self.n_words * s_pair / s1 ** 2 / ((W - 1) / W)
        # A data-dependent vulnerable fraction q does not change kappa: with
        # V ~ Binomial(W, q) vulnerable cells per word, E[V(V-1)] / (qW)^2 equals
        # W(W-1)/W^2, the same within-word pair probability as the uniform case.
        return float(kappa)

    def effective_n_words(self) -> float:
        return self.n_words / self.collision_factor()


def leakage_sampler(n_words: int, W: int, T: int, *, block_words: int | None = None,
                    device_seed: int = 0, **overrides) -> NonUniformSampler:
    """NonUniformSampler with the temperature preset of ``LEAKAGE_PRESETS``."""
    kw = dict(LEAKAGE_PRESETS[T])
    kw.update(overrides)
    return NonUniformSampler(n_words, W, block_words=block_words, device_seed=device_seed, **kw)


# ---------------------------------------------------------------------------
# 2. TID: persistent errors
# ---------------------------------------------------------------------------


class TIDModel:
    """Persistent TID-induced errors.

    Each trial gets Poisson(density * L) error cells, drawn from ``sampler``
    (so with a leakage sampler they sit in the leaky cells, as interface-trap
    induced retention loss would).  They are present in every scrub interval
    of the trial: scrubbing rewrites the data but the cell keeps failing.
    """

    def __init__(self, density: float, sampler):
        self.density = float(density)
        self.sampler = sampler

    def expected_count(self) -> int:
        return int(np.ceil(self.density * self.sampler.n_cells))

    def sample(self, rng, n_trials):
        counts = rng.poisson(self.density * self.sampler.n_cells, n_trials)
        cells = self.sampler.sample(rng, int(counts.sum()))
        trial = np.repeat(np.arange(n_trials, dtype=np.int64), counts)
        # one cell can only fail once per trial
        key = np.unique(trial * self.sampler.n_cells + cells)
        return key // self.sampler.n_cells, key % self.sampler.n_cells


# ---------------------------------------------------------------------------
# 3. SEFI: correlated, geometrically aligned errors
# ---------------------------------------------------------------------------


class SEFIModel:
    """Correlated multi-cell errors produced by single-event functional interrupts.

    kind:
      "column" : one column, ``span`` consecutive rows (bitmap line, paper 2 Fig. 3(a))
      "row"    : one row, ``span`` consecutive columns
      "device" : chip-level burst in a DIMM - in ``span`` consecutive words the
                 ``device_bits``-wide slice owned by one chip is corrupted
                 (the failure SNC-DND / chip-kill is designed for)
    Every cell on the line flips with probability ``p_flip`` (only charged
    cells read wrong, so a line is rarely solid).

    ``rate`` is the expected number of SEFIs per trial (whole exposure); with
    scrubbing the events are spread over the intervals.  With poisson=False
    every trial gets exactly round(rate) SEFIs (placed in random intervals).
    """

    def __init__(self, geometry: Geometry | None, rate: float, kind: str = "column",
                 span: int = 4096, p_flip: float = 0.2, device_bits: int = 4,
                 W: int | None = None, n_words: int | None = None, poisson: bool = True):
        self.poisson = poisson
        self.g = geometry
        self.rate = float(rate)
        self.kind = kind
        self.span = int(span)
        self.p_flip = float(p_flip)
        self.device_bits = int(device_bits)
        self.W = geometry.W if geometry else int(W)
        self.n_words = geometry.n_words if geometry else int(n_words)
        if kind in ("row", "column") and geometry is None:
            raise ValueError("row/column SEFIs need a Geometry")

    def expected_bits(self) -> float:
        return self.span * self.p_flip * (self.device_bits if self.kind == "device" else 1)

    def _one_event(self, rng):
        g = self.g
        if self.kind == "column":
            bank = rng.integers(g.banks)
            col = rng.integers(g.cols)
            r0 = rng.integers(g.rows)
            rows = (r0 + np.arange(self.span)) % g.rows
            rows = rows[rng.random(rows.size) < self.p_flip]
            cells = g.cell_index(bank, rows, np.full(rows.size, col))
        elif self.kind == "row":
            bank = rng.integers(g.banks)
            row = rng.integers(g.rows)
            c0 = rng.integers(g.cols)
            cols = (c0 + np.arange(self.span)) % g.cols
            cols = cols[rng.random(cols.size) < self.p_flip]
            cells = g.cell_index(bank, np.full(cols.size, row), cols)
        elif self.kind == "device":
            n_dev = self.W // self.device_bits
            d = rng.integers(n_dev)
            w0 = rng.integers(self.n_words)
            words = (w0 + np.arange(self.span)) % self.n_words
            bits = d * self.device_bits + np.arange(self.device_bits)
            cells = (words[:, None] * self.W + bits[None, :]).ravel()
            cells = cells[rng.random(cells.size) < self.p_flip]
        else:
            raise ValueError(self.kind)
        return cells[cells >= 0]

    def sample(self, rng, n_sub, parts):
        if self.poisson:
            counts = rng.poisson(self.rate / parts, n_sub)
        else:
            n_trials = n_sub // parts
            counts = np.zeros(n_sub, dtype=np.int64)
            for _ in range(int(round(self.rate))):
                np.add.at(counts, np.arange(n_trials) * parts + rng.integers(0, parts, n_trials), 1)
        subs, cells = [], []
        for s in np.flatnonzero(counts):
            for _ in range(counts[s]):
                c = self._one_event(rng)
                cells.append(c)
                subs.append(np.full(c.size, s, dtype=np.int64))
        if not cells:
            e = np.empty(0, dtype=np.int64)
            return e, e
        return np.concatenate(subs), np.concatenate(cells).astype(np.int64)


# ---------------------------------------------------------------------------
# Analytic bridges for the extensions
# ---------------------------------------------------------------------------


def tid_seu_failure_rate(m, n_tid, n_words, W, kappa=1.0):
    """Expected SEC-DED failures from SEUs landing in words that already hold
    a persistent TID error (first order).  ``kappa`` > 1 when SEUs and TID
    errors concentrate in the same leaky regions."""
    m = np.asarray(m, dtype=float)
    return kappa * m * n_tid * (W - 1) / W / n_words


def secded_rate_with_tid(m, n_tid, n_words, W, n_scrubs=0, kappa=1.0):
    """lambda for SEC-DED with s scrubs and n_tid persistent errors.

    Three terms: SEU-SEU pairs (cut by scrubbing, paper 1 Fig. 10), SEU-TID
    pairs and TID-TID pairs.  The last two are not removed by scrubbing.
    """
    from . import analytic as A
    parts = n_scrubs + 1
    LA_eff = n_words / kappa
    seu = parts * A.nfm_plus(2, np.asarray(m, float) / parts, LA_eff, W)
    tid_tid = A.nfm_plus(2, n_tid, LA_eff, W) if n_tid > 0 else 0.0
    return seu + tid_seu_failure_rate(m, n_tid, n_words, W, kappa) + tid_tid
