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

## Results

Full results are being generated; see `results/` once committed.
