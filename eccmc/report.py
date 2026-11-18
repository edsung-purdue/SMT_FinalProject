"""Excel export and quick-look plots."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

_SKIP_SUFFIX = ("_ci", "lo95", "hi95", "trials")


def write_workbook(path, sheets: dict, notes: dict, title: str = ""):
    """One sheet per figure/table plus a README sheet describing each."""
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        readme = pd.DataFrame({"sheet": list(sheets), "contents": [notes.get(k, "") for k in sheets]})
        readme.to_excel(xw, sheet_name="README", index=False, startrow=2)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False, startrow=2)
        wb = xw.book
        ws = wb["README"]
        ws["A1"] = title
        ws["A1"].font = Font(bold=True, size=13)
        ws.column_dimensions["A"].width = 14
        ws.column_dimensions["B"].width = 140
        for row in ws.iter_rows(min_row=4):
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
        for name in sheets:
            ws = wb[name]
            ws["A1"] = notes.get(name, name)
            ws["A1"].font = Font(italic=True)
            hdr = PatternFill("solid", fgColor="DDEBF7")
            for cell in ws[3]:
                cell.font = Font(bold=True)
                cell.fill = hdr
                cell.alignment = Alignment(wrap_text=True)
            for i, col in enumerate(sheets[name].columns, start=1):
                ws.column_dimensions[get_column_letter(i)].width = max(11, min(28, len(str(col)) + 2))
            ws.freeze_panes = "B4"


_LOGX = {"Fig1", "Fig3", "Fig4", "Fig5", "Fig7", "Fig8", "Fig9"}
_LOGY = {"Fig1", "Fig3", "Fig4", "Fig5"}


def plot_sheet(name, df: pd.DataFrame, path, logx=None, logy=None, xcol=None):
    if name.startswith("Table") or name == "Fig6" and False:
        return
    logx = name in _LOGX if logx is None else logx
    logy = name in _LOGY if logy is None else logy
    xcol = xcol or df.columns[0]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    lines = [c for c in df.columns if c != xcol and not c.startswith("MC") and not c.endswith(_SKIP_SUFFIX)
             and pd.api.types.is_numeric_dtype(df[c]) and c not in ("m eq(1)",)]
    colors = {}
    for i, c in enumerate(lines):
        d = df[[xcol, c]].dropna()
        style = "--" if ("eq(6)" in c or "exact" in c) else "-"
        ln, = ax.plot(d[xcol], d[c], style, lw=1.2, label=c)
        colors[c] = ln.get_color()
    mcs = [c for c in df.columns if c.startswith("MC") and not c.endswith(_SKIP_SUFFIX)
           and c not in ("MC m", "MC NBF")]
    mx = "MC NBF" if "MC NBF" in df.columns else xcol
    for c in mcs:
        d = df[[mx, c]].copy()
        if c + "_ci" in df.columns:
            d["err"] = df[c + "_ci"]
            d = d.dropna()
            ax.errorbar(d[mx], d[c], yerr=d["err"], fmt="o", ms=4, capsize=2, label=c)
        elif c + " lo95" in df.columns:
            lo, hi = df[c + " lo95"], df[c + " hi95"]
            d["lo"], d["hi"] = (d[c] - lo).clip(lower=0), (hi - d[c]).clip(lower=0)
            d = d.dropna()
            ax.errorbar(d[mx], d[c], yerr=[d["lo"], d["hi"]], fmt="o", ms=4, capsize=2, label=c)
        else:
            d = d.dropna()
            ax.plot(d[mx], d[c], "o", ms=4, label=c)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
        ymin = np.nanmax([1e-6, np.nanmin(df[lines].replace(0, np.nan).values) if lines else 1e-6])
        ax.set_ylim(bottom=ymin)
    ax.set_xlabel(xcol)
    ax.set_title(name)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
