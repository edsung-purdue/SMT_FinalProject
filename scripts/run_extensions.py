#!/usr/bin/env python3
"""Extension studies: paper 1's ECC reliability model under the DDR4 effects
reported by Ryu et al., IEEE TED 72(9), 2025 (paper 2).

Writes results/paper2_extensions.xlsx (one sheet per study) and quick-look PNGs.

  E1_Temperature  P(fail) of a 32 GB (72,64) SEC-DED DIMM vs proton fluence at
                  153 / 300 / 373 K, uniform vs leakage-weighted hits
  E2_NonUniform   how leakage non-uniformity inflates false MBUs, and the
                  effective-LA correction that restores paper 1's equations
  E3_TID_Scrub    persistent TID errors put a floor under the benefit of scrubbing
  E4_SEFI         geometric alignment of SEFI errors vs interleaving and ECC type
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
from eccmc import analytic as A                                   # noqa: E402
from eccmc import montecarlo as MC                                # noqa: E402
from eccmc import radiation as R                                  # noqa: E402
from eccmc.codes import SECDED, SNCDND                            # noqa: E402

from eccmc.report import write_workbook                            # noqa: E402

SCALE = 1.0   # trial multiplier (--quick sets 0.15)


def T(n):
    return max(50, int(n * SCALE))


# ---------------------------------------------------------------------------
# E1: temperature / leakage, 32 GB DIMM
# ---------------------------------------------------------------------------

DIMM_DATA_BITS = 32 * 2 ** 30 * 8
DIMM_W = 72
DIMM_LA = DIMM_DATA_BITS // 64
DIMM_ROW_WORDS = 1024        # one DIMM row = 8 KB of data = 1024 64-bit words
CODE72 = SECDED(72, 64)
FLUENCES = [2e9, 5e9, 1e10, 2e10, 3e10, 5e10, 1e11]
E1_CASES = [(153, 64), (300, 64), (300, 32), (373, 64), (373, 32)]


def e1_task(args):
    T_K, refresh, fluence, model, seed = args
    m = int(round(R.expected_bitflips(fluence, DIMM_LA * DIMM_W, T_K, refresh)))
    if model == "uniform":
        sampler = None
        kappa = 1.0
    else:
        sampler = R.leakage_sampler(DIMM_LA, DIMM_W, T_K, block_words=DIMM_ROW_WORDS, device_seed=seed)
        kappa = sampler.collision_factor()
    n = T(2000) if m < 2e4 else T(600) if m < 2e5 else T(150)
    if m == 0:
        return dict(T=T_K, refresh=refresh, fluence=fluence, model=model, m=0, trials=0,
                    kappa=kappa, P_mc=0.0, lo=0.0, hi=0.0)
    r = MC.simulate(DIMM_LA, DIMM_W, max(m, 1), n, codes=(CODE72,), sampler=sampler,
                    flip_mode="set", seed=seed)
    p, lo, hi = r.p_failure(CODE72.name)
    return dict(T=T_K, refresh=refresh, fluence=fluence, model=model, m=m, trials=n, kappa=kappa,
                P_mc=p, lo=lo, hi=hi,
                P_paper_eq=float(A.p_failure(CODE72, m, DIMM_LA)),
                P_eq_LAeff=float(A.p_failure(CODE72, m, DIMM_LA / kappa)))


def run_e1(pool):
    tasks, seed = [], 10
    for T_K, ref in E1_CASES:
        for f in FLUENCES:
            for model in ("uniform", "leakage"):
                tasks.append((T_K, ref, f, model, seed)); seed += 1
    rows = list(pool.map(e1_task, tasks, chunksize=1))
    df = pd.DataFrame(rows)
    df.insert(0, "case", [f"{t} K, {r} ms" for t, r in zip(df["T"], df["refresh"])])
    return df.rename(columns={"P_mc": "MC P_fail", "lo": "MC lo95", "hi": "MC hi95",
                              "P_paper_eq": "Paper-1 eq (uniform LA)",
                              "P_eq_LAeff": "Paper-1 eq with LA_eff=LA/kappa",
                              "m": "m (SEU bitflips)"})


# ---------------------------------------------------------------------------
# E2: non-uniformity sweep
# ---------------------------------------------------------------------------

E2_LA, E2_W, E2_M = 2 ** 22, 72, 4000


def e2_task(args):
    label, kw, flip_mode, seed = args
    s = R.NonUniformSampler(E2_LA, E2_W, block_words=1024, device_seed=seed, **kw)
    kappa = s.collision_factor()
    r = MC.simulate(E2_LA, E2_W, E2_M, T(4000), sampler=s, flip_mode=flip_mode, seed=seed)
    n2, c2 = r.nfm(2)
    return dict(scenario=label, flip_mode=flip_mode, **{k: v for k, v in kw.items()},
                kappa=kappa, LA_eff=E2_LA / kappa, **{"MC NFM2": n2, "MC NFM2_ci": c2},
                **{"Paper-1 eq(3) NFM2": float(A.nfm(2, E2_M, E2_LA, E2_W)),
                   "eq(3) with LA_eff": float(A.nfm(2, E2_M, E2_LA / kappa, E2_W)),
                   "MC/eq(3)": n2 / float(A.nfm(2, E2_M, E2_LA, E2_W))})


def run_e2(pool):
    sc = [("uniform", {}),
          ("data pattern 50% vulnerable", dict(vulnerable_fraction=0.5))]
    for sig in (0.3, 0.6, 0.9, 1.2):
        sc.append((f"row lognormal sigma={sig}", dict(block_sigma=sig)))
    for f, r in ((1e-4, 10), (1e-3, 50), (3e-3, 100), (1e-2, 100)):
        sc.append((f"weak cells f={f:g} x{r}", dict(weak_fraction=f, weak_ratio=r)))
    for T_K, kw in R.LEAKAGE_PRESETS.items():
        sc.append((f"preset {T_K} K", dict(kw)))
    tasks, seed = [], 200
    for label, kw in sc:
        for fm in ("toggle", "set"):
            tasks.append((label, kw, fm, seed)); seed += 1
    return pd.DataFrame(list(pool.map(e2_task, tasks, chunksize=1)))


# ---------------------------------------------------------------------------
# E3: TID + scrubbing
# ---------------------------------------------------------------------------

E3_LA = 2 ** 27            # 1 GiB of data, (72,64) SEC-DED
E3_W = 72
E3_FLUENCE = 1e11          # ~16 krad(SiO2)
E3_TK = 300


def e3_task(args):
    dose_scale, model, n_scrubs, seed = args
    dose = R.PAPER2_DOSE_KRAD * E3_FLUENCE / R.PAPER2_FLUENCE * dose_scale
    m = int(round(R.expected_bitflips(E3_FLUENCE, E3_LA * E3_W, E3_TK, 64)))
    if model == "uniform":
        sampler = MC.UniformSampler(E3_LA, E3_W)
        kappa = 1.0
    else:
        sampler = R.leakage_sampler(E3_LA, E3_W, E3_TK, block_words=1024, device_seed=seed % 7)
        kappa = sampler.collision_factor()
    dens = R.tid_density(E3_TK, dose)
    tid = R.TIDModel(dens, sampler) if dens > 0 else None
    r = MC.simulate(E3_LA, E3_W, m, T(3000), codes=(CODE72,), sampler=sampler, tid=tid,
                    n_scrubs=n_scrubs, flip_mode="set", seed=seed)
    p, lo, hi = r.p_failure(CODE72.name)
    n_tid = dens * E3_LA * E3_W
    lam = R.secded_rate_with_tid(m, n_tid, E3_LA, E3_W, n_scrubs, kappa)
    return dict(dose_krad=dose, model=model, n_scrubs=n_scrubs, m=m, n_tid_expected=n_tid,
                kappa=kappa, trials=r.n_trials, **{"MC P_fail": p, "MC lo95": lo, "MC hi95": hi,
                "analytic P_fail": float(-np.expm1(-lam)),
                "analytic, SEU-only (paper 1)": float(-np.expm1(-(n_scrubs + 1) * A.nfm_plus(
                    2, m / (n_scrubs + 1), E3_LA, E3_W)))})


def run_e3(pool):
    tasks, seed = [], 400
    for dose_scale in (0.0, 0.25, 1.0):
        for model in ("uniform", "leakage"):
            for s in (0, 1, 3, 7, 15, 31):
                tasks.append((dose_scale, model, s, seed)); seed += 1
    return pd.DataFrame(list(pool.map(e3_task, tasks, chunksize=1)))


# ---------------------------------------------------------------------------
# E4: SEFI alignment
# ---------------------------------------------------------------------------

def e4_task(args):
    label, kind, W, interleave, code, span, p_flip, seed = args
    g = R.Geometry(W=W, interleave=interleave)
    m = int(round(R.expected_bitflips(R.PAPER2_FLUENCE, g.n_words * W, 373, 64)))
    if kind == "scattered":
        # same number of error bits as one SEFI, but uniformly scattered
        extra = int(span * p_flip)
        sefi = None
        m_tot = m + extra
    else:
        sefi = R.SEFIModel(g, rate=1.0, kind=kind, span=span, p_flip=p_flip, device_bits=4, poisson=False)
        m_tot = m
    r = MC.simulate(g.n_words, W, m_tot, T(1500), codes=(code,), sefi=sefi, flip_mode="set", seed=seed)
    p, lo, hi = r.p_failure(code.name)
    fw, fwc = r.failure_rate(code.name)
    return {"scenario": label, "SEFI kind": kind, "code": code.name, "interleave": interleave,
            "SEU bitflips m": m, "SEFI span": span, "SEFI p_flip": p_flip,
            "expected SEFI bits": span * p_flip * (4 if kind == "device" else 1),
            "MC P_fail": p, "MC lo95": lo, "MC hi95": hi,
            "MC failing words": fw, "MC failing words_ci": fwc,
            "paper-1 eq, SEUs only": float(A.p_failure(code, m, g.n_words)), "trials": r.n_trials}


def run_e4(pool):
    sec = SECDED(72, 64)
    snc4 = SNCDND(72, 64, 4)
    snc8 = SNCDND(72, 64, 8)
    tasks, seed = [], 600
    col_span, col_p = 4096, 0.25    # ~1000 bits, like S2's SEFI
    for code in (sec, snc8):
        tasks.append(("no SEFI (SEUs only)", "scattered", 72, 1, code, 0, 0.0, seed)); seed += 1
        tasks.append(("SEFI bits scattered uniformly", "scattered", 72, 1, code, col_span, col_p, seed)); seed += 1
        tasks.append(("column-aligned SEFI (4096 rows)", "column", 72, 1, code, col_span, col_p, seed)); seed += 1
        tasks.append(("row-aligned SEFI (4096 cols), interleave 1", "row", 72, 1, code, col_span, col_p, seed)); seed += 1
        tasks.append(("row-aligned SEFI (4096 cols), interleave 16", "row", 72, 16, code, col_span, col_p, seed)); seed += 1
        for il in (1, 2, 4, 8, 16, 32):
            tasks.append((f"short row SEFI (32 cols), interleave {il}", "row", 72, il, code, 32, 0.5, seed)); seed += 1
    for code in (sec, snc4, snc8):
        tasks.append(("device (chip) SEFI, x4 slice, 512 words", "device", 72, 1, code, 512, 0.5, seed)); seed += 1
    out = list(pool.map(e4_task, tasks, chunksize=1))
    return pd.DataFrame(out)


NOTES = {
    "E1_Temperature": ("32 GB DIMM, (72,64) SEC-DED, DRAM 'set' flips. m = sigma_SEU(T, refresh) x fluence x bits, "
                       "sigma from paper 2's error densities (SEFI steps removed at 373 K). 'leakage' draws hits with "
                       "the LEAKAGE_PRESETS susceptibility (row-level lognormal spread + weak-cell tail); kappa is the "
                       "resulting collision factor and LA_eff = LA/kappa restores paper 1's equations."),
    "E2_NonUniform": ("LA=2^22 words of 72 bits, m=4000. False 2-bit MBUs vs susceptibility model. MC/eq(3) shows how far "
                      "paper 1's uniform assumption under-predicts; 'eq(3) with LA_eff' is the corrected prediction."),
    "E3_TID_Scrub": ("1 GiB data, (72,64) SEC-DED, 300 K, fluence 1e11 p/cm^2. Persistent TID errors (density from paper 2, "
                     "Case 1, scaled linearly with dose) survive scrubbing; SEUs are cleared at each scrub. The "
                     "SEU-only column is paper 1's scrubbing model, which predicts that scrubbing alone fixes reliability."),
    "E4_SEFI": ("One 2^29-bit DDR4 bank (65536 x 8192), 373 K, fluence 3e10. One SEFI per trial (~1000 error bits, as S2) "
                "plus background SEUs. Exactly one SEFI per trial. Column- and row-aligned lines vs the same bits scattered; row lines vs bit "
                "interleaving degree; chip-level (device) SEFI vs SEC-DED and SNC-DND."),
}


def main():
    global SCALE
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--only", default="E1,E2,E3,E4")
    args = ap.parse_args()
    if args.quick:
        SCALE = 0.15
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    sheets = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for key, fn, name in (("E1", run_e1, "E1_Temperature"), ("E2", run_e2, "E2_NonUniform"),
                              ("E3", run_e3, "E3_TID_Scrub"), ("E4", run_e4, "E4_SEFI")):
            if key in args.only.split(","):
                t = time.time()
                sheets[name] = fn(pool)
                print(f"{name}: {time.time() - t:.0f}s", flush=True)
    path = os.path.join(args.out, "paper2_extensions.xlsx")
    write_workbook(path, sheets, NOTES, title="Paper 1 ECC model with paper 2 DDR4 effects")
    plot_extensions(sheets, os.path.join(args.out, "figures"))
    print("wrote", os.path.abspath(path))


def plot_extensions(sheets, outdir):
    import matplotlib.pyplot as plt
    if "E1_Temperature" in sheets:
        df = sheets["E1_Temperature"]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for (case, model), d in df.groupby(["case", "model"], sort=False):
            d = d.sort_values("fluence")
            ln = ax.plot(d.fluence, d["MC P_fail"], "o-" if model == "leakage" else "s--", ms=4,
                         label=f"{case} {model}")
        ax.set_xscale("log"); ax.set_xlabel("fluence (p/cm^2)"); ax.set_ylabel("P(data integrity loss)")
        ax.set_title("E1: 32 GB (72,64) SEC-DED DIMM vs fluence"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "paper2_E1.png"), dpi=110); plt.close(fig)
    if "E3_TID_Scrub" in sheets:
        df = sheets["E3_TID_Scrub"]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for (dose, model), d in df.groupby(["dose_krad", "model"]):
            ax.errorbar(d.n_scrubs + 1, d["MC P_fail"], yerr=[(d["MC P_fail"] - d["MC lo95"]).clip(lower=0),
                        (d["MC hi95"] - d["MC P_fail"]).clip(lower=0)], fmt="o", ms=4, capsize=2,
                        label=f"MC {dose:.1f} krad {model}")
            ax.plot(d.n_scrubs + 1, d["analytic P_fail"], "-", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("scrub intervals (n_scrubs + 1)"); ax.set_ylabel("P(data integrity loss)")
        ax.set_title("E3: TID floor under scrubbing (lines: analytic)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "paper2_E3.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
