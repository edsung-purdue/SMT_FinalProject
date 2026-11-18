#!/usr/bin/env python3
"""Reproduce every figure and table of Clemente et al., IEEE TNS 69(2), 2022.

Writes
  results/paper1_reproduction.xlsx   one sheet per figure / table
  results/figures/paper1_*.png       quick-look plots for visual comparison

Usage:
  python scripts/reproduce_paper1.py            # full run (~10-20 min on 4 cores)
  python scripts/reproduce_paper1.py --quick    # small trial counts, ~1-2 min
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eccmc import analytic as A                       # noqa: E402
from eccmc import montecarlo as MC                    # noqa: E402
from eccmc.codes import DAEC, DECTED, SECDED, SNCDND, TAEC, TEC, ECCCode  # noqa: E402
from eccmc.report import write_workbook, plot_sheet   # noqa: E402

MB = 2 ** 20 * 8     # bits in one MB
BUDGET = 4e7         # hits per Monte Carlo point (overridden by --quick)
MAX_TRIALS = 100_000


def ntrials(m):
    return MC.trials_for(m, BUDGET, lo=100 if BUDGET < 1e7 else 200, hi=MAX_TRIALS)


# ---------------------------------------------------------------------------
# Monte Carlo worker (top level so it can be pickled)
# ---------------------------------------------------------------------------

def _mc_task(args):
    tag, LA, W, m, codes, nibbles, n_scrubs, seed = args
    r = MC.simulate(LA, W, int(m), ntrials(m), codes=codes, nibbles=nibbles,
                    n_scrubs=n_scrubs, kmax=9, seed=seed)
    out = {"tag": tag, "m": int(m), "trials": r.n_trials}
    for k in range(2, 9):
        out[f"NFM{k}"], out[f"NFM{k}_ci"] = r.nfm(k)
        out[f"NFM{k}_NA"], out[f"NFM{k}_NA_ci"] = r.nfm_na(k)
    out["NFM+2"], out["NFM+2_ci"] = r.nfm_plus(2)
    for b in nibbles:
        out[f"DN{b}"], out[f"DN{b}_ci"] = r.nfm_dn(b)
    for c in codes:
        p, lo, hi = r.p_failure(c.name)
        out[f"P[{c.name}]"] = p
        out[f"P[{c.name}]_lo"] = lo
        out[f"P[{c.name}]_hi"] = hi
    return out


def run_tasks(tasks, workers):
    t = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_mc_task, tasks, chunksize=1))
    print(f"  {len(tasks)} MC points in {time.time() - t:.0f}s", flush=True)
    return res


def mc_frame(results, tag):
    return pd.DataFrame([r for r in results if r["tag"] == tag]).drop(columns="tag")


# ---------------------------------------------------------------------------
# Figure definitions
# ---------------------------------------------------------------------------

FIG1_LA, FIG1_W = 2 ** 21, 8
FIG1_MC = [10, 50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 800_000, 900_000, 1_000_000]

FIG4_LA, FIG4_W = 2 ** 20, 16
FIG4_MC = [10, 100, 1000, 10_000, 100_000, 300_000, 500_000, 700_000, 1_000_000]

FIG7_LA = 128 * 1024
FIG7_CODES = [SECDED(39, 32), DAEC(39, 32), DECTED(45, 32)]

FIG8_BITS = 24 * MB
FIG8_CODES = [SECDED(266, 256), DECTED(275, 256), SECDED(137, 128), DECTED(145, 128),
              SECDED(72, 64), DECTED(79, 64), SECDED(39, 32), DECTED(45, 32),
              SECDED(22, 16), DECTED(27, 16)]

FIG9_BITS = 256 * MB
FIG9_CODES = [SECDED(137, 128), TAEC(138, 128), SNCDND(144, 128, 4), SNCDND(152, 128, 8),
              SNCDND(80, 64, 4), DECTED(145, 128), TEC(64, 45)]

TABLE2_CODES = [SECDED(22, 16), SECDED(39, 32), SECDED(72, 64), SECDED(137, 128),
                DECTED(27, 16), DECTED(45, 32), DECTED(79, 64), DECTED(145, 128),
                SNCDND(144, 128, 4), SNCDND(152, 128, 8), TEC(64, 45)]
TABLE2_PAPER = {  # values printed in the paper's Table II
    "(22,16) SEC-DED": [1682, 531, 169, 54, 18, 6, 3],
    "(39,32) SEC-DED": [1178, 372, 118, 38, 13, 5, 2],
    "(72,64) SEC-DED": [828, 262, 83, 27, 9, 4, 2],
    "(137,128) SEC-DED": [584, 185, 59, 19, 7, 3, 2],
    "(27,16) DEC-TED": [106813, 49505, 22975, 10664, 4951, 2299, 1068],
    "(45,32) DEC-TED": [66250, 30705, 14250, 6615, 3071, 1426, 663],
    "(79,64) DEC-TED": [41328, 19154, 8890, 4127, 1916, 890, 414],
    "(145,128) DEC-TED": [25886, 11997, 5568, 2585, 1201, 558, 260],
    "(144,128) SNC-DND b=4": [590, 187, 60, 20, 7, 3, 2],
    "(152,128) SNC-DND b=8": [598, 189, 61, 20, 7, 3, 2],
    "(64,45) TEC": [147488, 99477, 66480, 43581, 27638, 16772, 9786],
}
NINES = [0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999]

FIG10_MEMS = {"a_128Kx32": (128 * 1024, 32, 5000), "b_24Mx8": (24 * 2 ** 20, 8, 70000)}


def la_for(code: ECCCode, data_bits: int, tec_includes_check=True) -> int:
    """Number of words: effective data size / data bits per word.  For the
    (64,45) TEC the paper counts 256 MB *including* check bits."""
    if code.kind == "TEC" and tec_includes_check:
        return data_bits // code.n
    return data_bits // code.k


def mc_grid(lo, hi, n):
    return sorted(set(int(round(x)) for x in np.logspace(np.log10(lo), np.log10(hi), n)))


def build_tasks():
    tasks = []
    seed = 1000
    for m in FIG1_MC:                              # Figs. 1-3 share the 2M x 8 memory
        tasks.append(("fig1", FIG1_LA, FIG1_W, m, (), (), 0, seed)); seed += 1
    for m in FIG4_MC:
        tasks.append(("fig4", FIG4_LA, FIG4_W, m, (), (4, 8), 0, seed)); seed += 1
    for m in mc_grid(5, 3000, 9):                  # Fig. 5: 2M x 8 SRAM, low m
        tasks.append(("fig5", FIG1_LA, FIG1_W, m, (), (4,), 0, seed)); seed += 1
    for m in mc_grid(10, 20000, 14):
        for W in (39, 45):
            codes = tuple(c for c in FIG7_CODES if c.n == W)
            tasks.append((f"fig7_{W}", FIG7_LA, W, m, codes, (), 0, seed)); seed += 1
    for c in FIG8_CODES:
        lo, hi = (30, 6000) if c.kind == "SEC-DED" else (1500, 2e5)
        for m in mc_grid(lo, hi, 10):
            tasks.append((f"fig8_{c.name}", la_for(c, FIG8_BITS), c.n, m, (c,), (), 0, seed)); seed += 1
    for c in FIG9_CODES:
        lo, hi = {"DEC-TED": (2000, 1e5), "TEC": (50_000, 1e6)}.get(c.kind, (100, 15000))
        for m in mc_grid(lo, hi, 9):
            tasks.append((f"fig9_{c.name}", la_for(c, FIG9_BITS), c.n, m, (c,), (), 0, seed)); seed += 1
    for key, (LA, W, mmax) in FIG10_MEMS.items():
        for s in range(8):
            for m in np.linspace(mmax / 5, mmax, 5).astype(int):
                tasks.append((f"fig10_{key}_s{s}", LA, W, m, (), (), s, seed)); seed += 1
    return tasks


# ---------------------------------------------------------------------------
# Sheet builders: analytic curve on a dense grid + MC points
# ---------------------------------------------------------------------------

def sheet_fig1(res):
    LA, W = FIG1_LA, FIG1_W
    m = np.unique(np.concatenate([np.logspace(1, 6, 101).round(), FIG1_MC]))
    df = pd.DataFrame({"m": m})
    for k, eq in ((2, "3"), (3, "4"), (4, "5")):
        df[f"NFM{k} eq({eq})"] = A.nfm(k, m, LA, W)
        df[f"NFM{k} eq(6)"] = A.nfm_eq6(k, m, LA, W)
        df[f"NFM{k} exact"] = A.nfm_exact(k, m, LA, W)
    mc = mc_frame(res, "fig1")
    keep = ["m", "trials"] + [c for k in (2, 3, 4) for c in (f"NFM{k}", f"NFM{k}_ci")]
    mc = mc[keep].rename(columns=lambda c: c if c in ("m", "trials") else "MC " + c)
    return df.merge(mc, on="m", how="left")


def sheet_fig2(res):
    LA, W = FIG1_LA, FIG1_W
    m = np.unique(np.concatenate([np.linspace(7e5, 1e6, 61).round(), [800_000, 900_000, 1_000_000]]))
    df = pd.DataFrame({"m": m})
    for k, eq in ((2, "3"), (3, "4"), (4, "5")):
        df[f"NFM{k} eq({eq})"] = A.nfm(k, m, LA, W)
        df[f"NFM{k} eq(6)"] = A.nfm_eq6(k, m, LA, W)
        df[f"NFM{k} exact"] = A.nfm_exact(k, m, LA, W)
    mc = mc_frame(res, "fig1")
    mc = mc[mc.m >= 7e5]
    keep = ["m", "trials"] + [c for k in (2, 3, 4) for c in (f"NFM{k}", f"NFM{k}_ci")]
    mc = mc[keep].rename(columns=lambda c: c if c in ("m", "trials") else "MC " + c)
    return df.merge(mc, on="m", how="left")


def sheet_fig3(res):
    LA, W = FIG1_LA, FIG1_W
    m = np.unique(np.concatenate([np.logspace(1, 6, 101).round(), FIG1_MC]))
    df = pd.DataFrame({"m": m})
    df["NFM2 eq(3)"] = A.nfm(2, m, LA, W)
    df["NFM2_NA eq(8)"] = A.nfm_na(2, m, LA, W)
    df["NFM3 eq(4)"] = A.nfm(3, m, LA, W)
    df["NFM3_NA eq(10)"] = A.nfm_na(3, m, LA, W)
    for k in range(4, 8):
        df[f"NFM{k}_NA eq(11)"] = A.nfm_na(k, m, LA, W)
    mc = mc_frame(res, "fig1")
    cols = ["m", "trials", "NFM2", "NFM2_ci", "NFM3", "NFM3_ci"]
    for k in range(2, 8):
        cols += [f"NFM{k}_NA", f"NFM{k}_NA_ci"]
    mc = mc[cols].rename(columns=lambda c: c if c in ("m", "trials") else "MC " + c)
    return df.merge(mc, on="m", how="left")


def sheet_fig4(res):
    LA, W = FIG4_LA, FIG4_W
    m = np.unique(np.concatenate([np.logspace(1, 6, 101).round(), FIG4_MC]))
    df = pd.DataFrame({"m": m})
    for b in (4, 8):
        df[f"NFMDN b={b} eq(12)"] = A.nfm_dn(m, LA, W, b)
        df[f"NFMDN b={b} exact"] = A.nfm_dn_exact(m, LA, W, b)
    mc = mc_frame(res, "fig4")[["m", "trials", "DN4", "DN4_ci", "DN8", "DN8_ci"]]
    mc = mc.rename(columns={"DN4": "MC NFMDN b=4", "DN4_ci": "MC NFMDN b=4_ci",
                            "DN8": "MC NFMDN b=8", "DN8_ci": "MC NFMDN b=8_ci"})
    return df.merge(mc, on="m", how="left")


def sheet_fig5(res):
    """Fig. 5 plots predictions against the authors' experimental MBU counts,
    which are not tabulated in the paper; only the prediction curves (vs NBF,
    the observed number of bitflips) are reproduced here, plus MC."""
    LA, W = FIG1_LA, FIG1_W
    L = LA * W
    nbf = np.unique(np.concatenate([np.logspace(0.7, 3.5, 60).round(), np.arange(100, 3001, 100)]))
    m = A.m_from_nbf(nbf, L)
    df = pd.DataFrame({"NBF (observed)": nbf, "m eq(1)": m})
    df["NFM2 eq(3)"] = A.nfm(2, m, LA, W)
    df["NFM2_NA eq(8)"] = A.nfm_na(2, m, LA, W)
    df["NFMDN b=4 eq(12)"] = A.nfm_dn(m, LA, W, 4)
    mc = mc_frame(res, "fig5")
    mc = pd.DataFrame({"m": mc.m, "MC NBF": [None] * len(mc), "MC trials": mc.trials,
                       "MC NFM2": mc.NFM2, "MC NFM2_ci": mc.NFM2_ci,
                       "MC NFM2_NA": mc.NFM2_NA, "MC NFM2_NA_ci": mc.NFM2_NA_ci,
                       "MC NFMDN b=4": mc.DN4, "MC NFMDN b=4_ci": mc.DN4_ci})
    mc["MC NBF"] = A.expected_nbf(mc.m.values, L)
    return pd.concat([df, mc.rename(columns={"m": "MC m"})], axis=1)


def sheet_fig6():
    i = np.arange(0, 13)
    return pd.DataFrame({"i (false MBUs)": i, "PMF eq(14) lambda=2": A.poisson_pmf(2.0, i),
                         "CDF eq(15) lambda=2": [A.poisson_cdf(2.0, j) for j in i]})


def _pfail_sheet(res, codes, la_of, tag_of, mlo, mhi):
    m = np.unique(np.logspace(np.log10(mlo), np.log10(mhi), 121).round())
    df = pd.DataFrame({"m": m})
    for c in codes:
        df[f"{c.name} analytic"] = A.p_failure(c, m, la_of(c))
    frames = []
    for c in codes:
        mc = mc_frame(res, tag_of(c))
        cols = {"m": "m", "trials": f"MC {c.name} trials", f"P[{c.name}]": f"MC {c.name}",
                f"P[{c.name}]_lo": f"MC {c.name} lo95", f"P[{c.name}]_hi": f"MC {c.name} hi95"}
        mc = mc[list(cols)].rename(columns=cols)
        frames.append(mc)
    mcall = frames[0]
    for f in frames[1:]:
        mcall = mcall.merge(f, on="m", how="outer")
    return df.merge(mcall, on="m", how="outer").sort_values("m").reset_index(drop=True)


def sheet_fig7(res):
    return _pfail_sheet(res, FIG7_CODES, lambda c: FIG7_LA, lambda c: f"fig7_{c.n}", 10, 2e4)


def sheet_fig8(res):
    return _pfail_sheet(res, FIG8_CODES, lambda c: la_for(c, FIG8_BITS), lambda c: f"fig8_{c.name}", 10, 2e5)


def sheet_fig9(res):
    return _pfail_sheet(res, FIG9_CODES, lambda c: la_for(c, FIG9_BITS), lambda c: f"fig9_{c.name}", 10, 1e6)


def sheet_table2(res):
    rows = []
    for c in TABLE2_CODES:
        LA = la_for(c, 256 * MB)
        for nines, paper in zip(NINES, TABLE2_PAPER[c.name]):
            ours = A.max_bitflips_for_reliability(c, LA, nines)
            ours_exact = A.max_bitflips_for_reliability(c, LA, nines, method="exact")
            rows.append({"code": c.name, "LA (words)": LA, "reliability": f"{nines*100:.6g}%",
                         "paper m": paper, "reproduced m (paper eqs)": ours,
                         "reproduced m (exact)": ours_exact,
                         "diff": ours - paper,
                         "P_fail at paper m": float(A.p_failure(c, paper, LA)),
                         "target P_fail": 1 - nines})
    return pd.DataFrame(rows)


def sheet_table2_mc(res):
    """MC spot-check of Table II: simulate at the paper's m for the 99% and
    99.9% rows and compare the observed failure probability to the target."""
    rows = [r for r in res if r["tag"].startswith("t2_")]
    out = []
    for r in rows:
        code = r["tag"][3:]
        p, lo, hi = r[f"P[{code}]"], r[f"P[{code}]_lo"], r[f"P[{code}]_hi"]
        out.append({"code": code, "m (paper)": r["m"], "trials": r["trials"], "target P_fail": r.get("target"),
                    "MC P_fail": p, "MC lo95": lo, "MC hi95": hi})
    return pd.DataFrame(out)


def sheet_fig10(res, key):
    LA, W, mmax = FIG10_MEMS[key]
    m = np.linspace(0, mmax, 101).round()
    df = pd.DataFrame({"m": m})
    for s in range(8):
        lab = "no scrubbing" if s == 0 else f"{s} scrub" + ("s" if s > 1 else "")
        df[f"NFM+(2) {lab}"] = A.nfm_plus_scrubbed(2, m, LA, W, s)
    for s in range(8):
        mc = mc_frame(res, f"fig10_{key}_s{s}")[["m", "NFM+2", "NFM+2_ci"]]
        mc = mc.rename(columns={"NFM+2": f"MC {s} scrubs", "NFM+2_ci": f"MC {s} scrubs_ci"})
        df = df.merge(mc, on="m", how="outer")
    return df.sort_values("m").reset_index(drop=True)


def main():
    global BUDGET, MAX_TRIALS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()
    if args.quick:
        BUDGET, MAX_TRIALS = 2e6, 5000
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    tasks = build_tasks()
    # Table II MC spot checks at 99% and 99.9%
    seed = 90_000
    for c in TABLE2_CODES:
        for nines in NINES[:2]:
            m = TABLE2_PAPER[c.name][NINES.index(nines)]
            tasks.append((f"t2_{c.name}", la_for(c, 256 * MB), c.n, m, (c,), (), 0, seed)); seed += 1
    print(f"Running {len(tasks)} Monte Carlo points on {args.workers} workers "
          f"(budget {BUDGET:.0e} hits/point)", flush=True)
    res = run_tasks(tasks, args.workers)
    # attach targets to table2 MC rows
    for r in res:
        if r["tag"].startswith("t2_"):
            code = r["tag"][3:]
            idx = TABLE2_PAPER[code].index(r["m"]) if r["m"] in TABLE2_PAPER[code] else 0
            r["target"] = 1 - NINES[min(idx, 1)]

    sheets = {
        "Fig1": sheet_fig1(res),
        "Fig2": sheet_fig2(res),
        "Fig3": sheet_fig3(res),
        "Fig4": sheet_fig4(res),
        "Fig5": sheet_fig5(res),
        "Fig6": sheet_fig6(),
        "Fig7": sheet_fig7(res),
        "Fig8": sheet_fig8(res),
        "Fig9": sheet_fig9(res),
        "Fig10a": sheet_fig10(res, "a_128Kx32"),
        "Fig10b": sheet_fig10(res, "b_24Mx8"),
        "TableII": sheet_table2(res),
        "TableII_MC": sheet_table2_mc(res),
    }
    notes = PAPER1_NOTES
    path = os.path.join(args.out, "paper1_reproduction.xlsx")
    write_workbook(path, sheets, notes, title="Reproduction of Clemente et al., IEEE TNS 2022")
    for name, df in sheets.items():
        plot_sheet(name, df, os.path.join(args.out, "figures", f"paper1_{name}.png"))
    print("wrote", os.path.abspath(path))


PAPER1_NOTES = {
    "Fig1": "Number of false 2-, 3-, 4-bit MBUs vs m. Memory LA=2^21 words, W=8. Eqs. (3)-(5) and (6), exact reference, and Monte Carlo (mean, 95% half-width t95*sd/sqrt(N)).",
    "Fig2": "Zoom of Fig. 1 for m in [7e5, 1e6] (linear scale).",
    "Fig3": "False k-bit MBUs whose bits are not all adjacent, Eqs. (8), (10), (11); LA=2^21, W=8.",
    "Fig4": "Words with bitflips in >=2 different b-bit nibbles, Eq. (12); LA=2^20, W=16, b=4 and 8.",
    "Fig5": "Prediction curves vs observed bitflips NBF for a 2M x 8 SRAM (Eq. (1) converts NBF to m). The paper's experimental points are not tabulated, so only predictions and MC are given.",
    "Fig6": "Poisson PMF and CDF for lambda = 2 (Eqs. (14), (15)).",
    "Fig7": "P(data integrity loss) for a 128K x 32-bit memory: (39,32) SEC-DED, (39,32) DAEC, (45,32) DEC-TED. Eqs. (16), (19), (17); MC is a direct estimate of P(any uncorrectable word) with Wilson 95% bounds.",
    "Fig8": "P(data integrity loss) for 24 MB of data with SEC-DED and DEC-TED from (22,16) to (275,256). LA = 24 MB / data bits.",
    "Fig9": "P(data integrity loss) for 256 MB: SNC-DND, SEC-DED, TAEC, DEC, TEC. TEC uses LA = 256 MB / 64 (check bits included, as in Table II). TAEC fails on non-adjacent 2- and 3-bit MBUs (the behaviour plotted in Fig. 9; Eq. (20) omits the 2-bit term).",
    "Fig10a": "Expected false MBUs (k>=2) with 0-7 equally spaced scrubs, 128K x 32-bit memory.",
    "Fig10b": "Expected false MBUs (k>=2) with 0-7 equally spaced scrubs, 24M x 8-bit memory.",
    "TableII": "Largest m keeping reliability above each threshold, 256 MB memory. 'paper m' is copied from the paper; 'reproduced' uses Eqs. (13)-(21) (and the exact reference).",
    "TableII_MC": "Monte Carlo at the paper's m for the 99% and 99.9% rows: MC P_fail should be at or just below the target.",
}


if __name__ == "__main__":
    main()
