# ECC reliability Monte Carlo simulator

A Python Monte Carlo simulator that reproduces

> J. A. Clemente, M. Rezaei, F. J. Franco, "Reliability of Error Correction Codes
> Against Multiple Events by Accumulation", *IEEE Trans. Nucl. Sci.* 69(2), 2022 ("paper 1")

and extends it with the DDR4 radiation effects reported in

> M. Ryu et al., "Single-Event Upset and Total Ionizing Dose Effects on DDR4 DRAM Due
> to Proton Irradiation Under Different Temperatures", *IEEE Trans. Electron Devices*
> 72(9), 2025 ("paper 2"):
> non-uniform error distributions (leakage / TID) and correlated errors (SEFI alignment).

## Quick start

```bash
pip install -r requirements.txt
python scripts/reproduce_paper1.py          # every figure + Table II  -> results/paper1_reproduction.xlsx
python scripts/run_extensions.py            # paper-2 studies          -> results/paper2_extensions.xlsx
python -m pytest -q tests                   # 13 checks, ~5 s
```

Both scripts take `--quick` (small trial counts, about a minute) and `--workers N`.
Each workbook has a README sheet and one sheet per figure/table. On every figure
sheet the analytic curves sit on a dense grid of `m` and the Monte Carlo points
(mean plus 95% half-width, or Wilson 95% bounds for probabilities) are in the
`MC ...` columns on the rows where they were simulated. Quick-look plots are
written to `results/figures/`.

## Using the simulator

```python
from eccmc import montecarlo as MC, analytic as A, radiation as R
from eccmc.codes import SECDED, DECTED, DAEC, TAEC, TEC, SNCDND

code = SECDED(39, 32)
LA = 128 * 1024                                   # words (W = code.n = 39 bits each)
r = MC.simulate(LA, code.n, m=170, n_trials=20000, codes=(code,), seed=1)
r.p_failure(code.name)           # (p, lo95, hi95), direct estimate of P(data integrity loss)
r.nfm(2), r.nfm_na(2)            # false 2-bit MBUs, and those that are not adjacent
A.p_failure(code, 170, LA)       # paper 1, Eq. (16)
```

`simulate()` options:

| option | meaning |
|---|---|
| `codes` | ECCs to evaluate: SEC-DED, DAEC, DEC-TED, TAEC, TEC, SNC-DND(b) |
| `nibbles` | also count words whose flips span more than one b-bit nibble (Eq. 12) |
| `n_scrubs` | equally spaced scrubbing cycles (Fig. 10) |
| `flip_mode` | `"toggle"` (paper 1: a second hit restores the cell) or `"set"` (DRAM charge loss) |
| `sampler` | `R.NonUniformSampler` / `R.leakage_sampler(...)` for non-uniform susceptibility |
| `tid` | `R.TIDModel`: persistent TID errors that survive scrubbing |
| `sefi` | `R.SEFIModel`: row, column or chip (device) aligned SEFI lines on a `R.Geometry` |

## How the Monte Carlo works

Paper 1 simulates the memory as an LA x W boolean matrix, flips m random cells
(a cell hit twice flips back), counts the multi-bit words and repeats the run
10^5 times. Here the matrix is never allocated. Each trial draws m cell indices.
The indices of a batch of trials are tagged with the trial number and sorted once.
Cells hit an even number of times are dropped, and the rest are grouped into words.
For each word the engine keeps its multiplicity, its first and last flipped bit,
and the full bit list, which is enough to decide every ECC rule. A 256 MB memory
then costs the same as a 2 Mbit one, at roughly 10^7 hits per second per core.

## Package layout

```
eccmc/analytic.py    paper 1 Eqs. (1)-(23), Poisson failure model, Table II solver, exact reference
eccmc/codes.py       ECC failure rules
eccmc/montecarlo.py  sparse vectorised Monte Carlo engine
eccmc/radiation.py   paper 2 data and models: leakage sampler, TID, SEFI, geometry/interleaving
eccmc/report.py      Excel and plot export
scripts/             reproduce_paper1.py, run_extensions.py
tests/               unit and statistical checks
```

## Results: paper 1 reproduction (`results/paper1_reproduction.xlsx`)

Sheets: Fig1 to Fig10b, TableII and TableII_MC, one per figure or table.

* **Figs. 1 to 4 (false MBU counts).** Monte Carlo matches Eqs. (3) to (12)
  across 10 <= m <= 10^6. In the far right of Fig. 2, MC sits about 0.1% above
  Eq. (3) (131,066 vs 130,877 at m = 10^6). An exact reference
  (`analytic.nfm_exact`) matches MC there. The difference comes from Eq. (3)
  ignoring 4-hit words where one cell is hit three times. Eq. (5) includes that
  case for k = 4, but Eqs. (3) and (4) leave it out for k = 2 and 3.
* **Figs. 7 to 9 (probability of data-integrity loss).** MC failure
  probabilities fall on the Poisson curves of Eqs. (16) to (21). The quoted
  numbers reproduce: (39,32) SEC-DED reaches 1% at m = 55 and 10% at m = 170;
  for 24 MB, (22,16) SEC-DED reaches 1% at 520 and (266,256) at 125.
* **Table II.** All 28 SEC-DED and all 14 SNC-DND entries reproduce exactly.
  Table II reports the first m at which the reliability target is missed.
  DEC-TED entries agree to within 15 bitflips (at most 0.014%). The
  **(64,45) TEC column cannot be reproduced** by Eq. (18), whatever memory size
  is assumed. Its 8-nines value is close (9,996 vs 9,786), but its 99% value is
  147,488 against 317,041 here. The printed column grows as roughly m^5 to m^6,
  while a code that fails at 4 bitflips has to grow as m^4. The Fig. 9 TEC
  curve, by contrast, is consistent with Eq. (18). TableII_MC confirms the other
  columns: at the paper's m, simulated failure probabilities sit on the 1% and
  0.1% targets.
* **TAEC.** Eq. (20) counts only non-adjacent triples and quadruples, which would
  make (138,128) TAEC about as strong as DEC. Fig. 9 instead plots TAEC almost on
  top of SEC-DED. That only happens if non-adjacent doubles also break TAEC, so
  the default rule here follows the figure. `TAEC(..., rule="eq20")` gives the
  equation's version.
* **Fig. 5.** The paper's experimental MBU counts are not tabulated, so this
  sheet has the prediction curves (vs observed NBF) and MC only.

## Results: paper 2 extensions (`results/paper2_extensions.xlsx`)

Paper 2's measured inputs live in `eccmc/radiation.py`: SEU error densities vs
temperature and refresh, TID error densities after 4.8 krad, and SEFI sizes.
The SEU-only density at 373 K subtracts the SEFI steps read from Fig. 2(a). The
shape of the susceptibility distribution is a model assumption (`LEAKAGE_PRESETS`),
because paper 2 reports no per-cell distribution.

* **E1, temperature.** For a 32 GB (72,64) SEC-DED DIMM at 3e10 p/cm^2,
  P(fail) is about 0 at 153 K and about 0.19 at 300 K (64 ms, uniform). At
  373 K/64 ms it is already 0.75 at 5e9 p/cm^2. Leakage-weighted hits raise
  P(fail) further, for example 0.19 to 0.27 at 300 K and 3e10, and 0.17 to 0.35
  at 373 K and 2e9.
* **E2, non-uniform errors.** Non-uniformity matters when it is regional (row or
  block-level spread in leakage). It raises false 2-bit MBUs by a collision
  factor kappa = LA * sum_w (P_w^2 - sum_c P_c^2) / ((W-1)/W), from 1.09 for
  sigma = 0.3 to 3.97 for sigma = 1.2. A sparse population of very weak cells, or
  a 50% data pattern, leaves kappa at about 1. In every case paper 1's
  equations are restored by replacing LA with **LA_eff = LA / kappa**, and MC
  agrees within its error bars.
* **E3, TID with scrubbing.** TID errors persist through scrubbing. Paper 1's
  model says 31 scrubs cut P(fail) of a 1 GiB (72,64) SEC-DED memory from 0.077 to 0.0025.
  With the TID density paper 2 measured after 4.8 krad (scaled to the dose
  here), P(fail) is stuck near 0.08 to 0.12 at about 4 krad and 0.40 to 0.55 at
  about 16 krad, however often you scrub. The analytic model
  (`radiation.secded_rate_with_tid`) adds SEU-TID and TID-TID pair terms and
  matches MC.
* **E4, SEFI alignment.** One SEFI of about 1000 bits (like S2), added to the
  SEUs, raises P(fail) from 0.08 to 0.21 when the bits run along a bitline
  (column), because each word gets at most one bit. The same number of bits
  along a wordline (row) gives P(fail) = 1, whatever the interleaving. A short
  row burst (32 columns) is only absorbed once interleaving reaches about the
  burst length (P = 0.066 at 32-way interleave, vs about 1 below 16-way).
  Chip-level (device) SEFIs always defeat SEC-DED and are fully corrected by
  SNC-DND (chip-kill), which leaves only the SEU baseline.
