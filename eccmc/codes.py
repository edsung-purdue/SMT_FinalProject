"""ECC models: which error patterns inside one codeword are uncorrectable.

A codeword error pattern is described by the sorted bit positions of its
flipped bits.  The Monte Carlo engine hands each code a ``WordErrors``
summary (multiplicity ``k``, first and last flipped bit, and the full bit
lists) and the code returns a boolean "data integrity lost" mask.

Rules follow Clemente et al. 2022, Section I:
    SEC-DED : any word with k >= 2 flipped bits fails
    DAEC    : fails for 2-bit non-adjacent and any k >= 3
    DEC-TED : fails for k >= 3          (paper also calls it "DEC")
    TAEC    : fails for 2- or 3-bit non-adjacent and any k >= 4
              ("eq20" rule: only 3-bit non-adjacent and k >= 4, as in Eq. (20))
    TEC     : fails for k >= 4
    SNC-DND : fails if the flipped bits span two or more b-bit nibbles
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ECCCode:
    kind: str          # "SEC-DED", "DAEC", "DEC-TED", "DEC", "TAEC", "TEC", "SNC-DND"
    n: int             # codeword bits (W in the paper)
    k: int             # data bits
    b: int = 0         # nibble size for SNC-DND
    taec_rule: str = "figure"  # "figure" (Fig. 9 behaviour) or "eq20"

    @property
    def name(self) -> str:
        s = f"({self.n},{self.k}) {self.kind}"
        if self.kind == "SNC-DND":
            s += f" b={self.b}"
        return s

    def fails(self, we: "WordErrors") -> np.ndarray:
        k = we.k
        adj = we.adjacent
        kind = self.kind
        if kind == "SEC-DED":
            return k >= 2
        if kind in ("DEC-TED", "DEC"):
            return k >= 3
        if kind == "TEC":
            return k >= 4
        if kind == "DAEC":
            return (k >= 3) | ((k == 2) & ~adj)
        if kind == "TAEC":
            if self.taec_rule == "eq20":
                return (k >= 4) | ((k == 3) & ~adj)
            return (k >= 4) | ((k >= 2) & ~adj)
        if kind == "SNC-DND":
            return (k >= 2) & (we.first // self.b != we.last // self.b)
        raise ValueError(kind)


def SECDED(n, k):
    return ECCCode("SEC-DED", n, k)


def DAEC(n, k):
    return ECCCode("DAEC", n, k)


def DECTED(n, k):
    return ECCCode("DEC-TED", n, k)


def TAEC(n, k, rule="figure"):
    return ECCCode("TAEC", n, k, taec_rule=rule)


def TEC(n, k):
    return ECCCode("TEC", n, k)


def SNCDND(n, k, b):
    return ECCCode("SNC-DND", n, k, b=b)


@dataclass
class WordErrors:
    """All words that hold at least one flipped bit, over a batch of trials."""
    trial: np.ndarray    # trial index of each affected word
    word: np.ndarray     # word address
    k: np.ndarray        # number of flipped bits in the word
    first: np.ndarray    # lowest flipped bit position
    last: np.ndarray     # highest flipped bit position
    starts: np.ndarray   # offsets into ``bits``
    bits: np.ndarray     # concatenated sorted bit positions

    @property
    def adjacent(self) -> np.ndarray:
        """True when all flipped bits are consecutive (k=1 counts as adjacent)."""
        return (self.last - self.first) == (self.k - 1)
