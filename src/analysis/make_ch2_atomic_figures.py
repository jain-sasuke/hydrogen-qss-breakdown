#!/usr/bin/env python
"""
make_ch2_atomic_figures.py
==========================
The three Chapter 2 figures for the CCC / RMPS comparison.

  fig2_2_ccc_rmps_benchmark   the benchmark itself: agreement against the
                              independent calculation, resolved by upper shell
                              and across temperature
  fig2_3_rydberg_series       the test that localises the n=5 disagreement,
                              using the shells only CCC reaches
  fig2_4_atomic_propagation   what the disagreement costs: the shift in the
                              timescale separation over the (Te, ne) grid, and
                              which coefficient each dataset moves

Every number plotted is read from a CSV written by a verify_* script; nothing
is recomputed here, so a figure cannot disagree with the text that quotes it.

Run:  python src/analysis/make_ch2_atomic_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
COL = ROOT / "data/processed/collisions"
OUT = ROOT / "figures"

# ---- thesis figure style (match the document, not matplotlib defaults) ------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# colourblind-safe, ordered by upper shell
CSHELL = {2: "#0072B2", 3: "#009E73", 4: "#E69F00", 5: "#D55E00"}


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf and .png")


# ============================================================ figure 2.2
def fig_benchmark():
    g = pd.read_csv(COL / "ccc_vs_anderson2002_thesis_Te_grid.csv")
    f = pd.read_csv(COL / "ccc_vs_anderson2002_full_Te_range.csv")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.9))

    # (a) the RATIO, not the two values against each other: over twelve
    #     decades a +-20% band is thinner than the plotted line, and every
    #     dataset looks perfect on a 1:1 log-log plot.
    xlim = [g.K_And2002.min() * 0.4, g.K_And2002.max() * 2.5]
    axa.fill_between(xlim, 0.8, 1.2, color="0.86", lw=0, zorder=1,
                     label="within 20%")
    axa.axhline(1.0, color="0.35", lw=0.8, zorder=2)
    for n in (2, 3, 4, 5):
        s = g[g.n_upper == n]
        axa.scatter(s.K_And2002, s.K_CCC_stored / s.K_And2002, s=2.0,
                    alpha=0.55, color=CSHELL[n], lw=0, label=f"$n'={n}$",
                    zorder=6 - n)
    axa.set(xscale="log", yscale="log", xlim=xlim, ylim=(0.18, 4.0),
            xlabel=r"$K$ from RMPS  [cm$^3$/s]",
            ylabel=r"$K_{\mathrm{CCC}} \, / \, K_{\mathrm{RMPS}}$")
    axa.set_yticks([0.25, 0.5, 1, 2, 4])
    axa.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    axa.set_title("(a) all 85 transitions, 50 temperatures", loc="left")
    h, l = axa.get_legend_handles_labels()
    axa.legend(h[1:] + h[:1], l[1:] + l[:1], loc="upper left", markerscale=3.2,
               handletextpad=0.3, borderpad=0.3, labelspacing=0.22,
               frameon=False, ncol=2, columnspacing=0.8)

    # (b) agreement against temperature, over the full RMPS range
    axb.axvspan(1.0, 10.0, color="0.90", lw=0, zorder=0)
    axb.text(3.1, 30, "range used\nin this work", ha="center", va="center",
             fontsize=7, color="0.35")
    for sel, lab, n in (((f.n_upper <= 3), r"$n' \leq 3$", 3),
                        ((f.n_upper == 4), r"$n' = 4$", 4),
                        ((f.n_upper == 5), r"$n' = 5$", 5)):
        m = f[sel].groupby("Te_eV").pct_err.apply(lambda x: x.abs().mean())
        axb.plot(m.index, m.values, "o-", ms=3, color=CSHELL[n], label=lab)
    axb.axhline(20, color="0.35", ls=":", lw=0.8)
    axb.text(22, 21.5, "20%", fontsize=7, color="0.35", ha="right")
    axb.set(xscale="log", xlim=(0.45, 27), ylim=(0, 50),
            xlabel=r"$T_e$  [eV]", ylabel=r"mean $|$error$|$  [%]")
    axb.set_xticks([0.5, 1, 3, 10, 25])
    axb.set_xticklabels(["0.5", "1", "3", "10", "25"])
    axb.set_title("(b) by upper shell", loc="left")
    axb.legend(frameon=False, handletextpad=0.4, labelspacing=0.25)

    fig.tight_layout(pad=0.4)
    save(fig, "fig2_2_ccc_rmps_benchmark")


# ============================================================ figure 2.3
def fig_rydberg():
    d = pd.read_csv(COL / "anderson_validity_dipole_series.csv")
    series = [("1s->np", r"$1s \rightarrow np$"),
              ("2p->nd", r"$2p \rightarrow nd$"),
              ("2s->np", r"$2s \rightarrow np$")]

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharex=True)
    for ax, (key, lab) in zip(axes, series):
        for Te, alpha, ls in ((1.0, 1.0, "-"), (10.0, 0.45, "-")):
            s = d[(d.series == key) & (d.Te_eV == Te)].sort_values("n_upper")
            if len(s) == 0:
                continue
            ax.plot(s.n_upper, s.R_CCC, "o" + ls, ms=3.4, color="#0072B2",
                    alpha=alpha, label="CCC" if Te == 1.0 else None)
            a = s.dropna(subset=["R_And"])
            ax.plot(a.n_upper, a.R_And, "s--", ms=3.8, color="#D55E00",
                    alpha=alpha, label="RMPS" if Te == 1.0 else None)
        # mark the last shell the RMPS basis contains
        ax.axvline(5, color="0.6", ls=":", lw=0.8, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel(r"$n_{\mathrm{upper}}$")
        ax.set_title(lab, loc="left")
        ax.set_xticks(range(2, 11, 2))
    axes[0].set_ylabel(r"$K \, n_{\mathrm{upper}}^{3}$  [cm$^3$/s]")
    axes[0].legend(frameon=False, handletextpad=0.4, labelspacing=0.25,
                   loc="upper right")
    axes[2].text(5.2, axes[2].get_ylim()[1] * 0.5,
                 "top of the\nRMPS basis", fontsize=7, color="0.4", va="top")
    axes[1].text(0.97, 0.94, r"upper curves $T_e=10$ eV," "\n"
                 r"lower $T_e=1$ eV", transform=axes[1].transAxes,
                 fontsize=6.5, color="0.4", ha="right", va="top")
    fig.tight_layout(pad=0.4)
    save(fig, "fig2_3_rydberg_series")


# ============================================================ figure 2.4
def fig_propagation():
    gi = pd.read_csv(COL / "ccc_anderson_grid_impact.csv")
    rs = pd.read_csv(ROOT / "validation/recombination_substitution"
                            "/recombination_substitution.csv")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.8))

    # (a) |dM| over the grid
    Te = np.sort(gi.Te_eV.unique())
    ne = np.sort(gi.ne_cm3.unique())
    Z = (gi.pivot_table(index="ne_cm3", columns="Te_eV",
                        values="pct_d_M").abs().values)
    pc = axa.pcolormesh(Te, ne, Z, shading="nearest", cmap="magma",
                        rasterized=True)
    axa.plot(2.947, 1.3895e14, "*", ms=9, mfc="none", mec="w", mew=1.0)
    axa.text(3.4, 1.3895e14, "benchmark", color="w", fontsize=7, va="center")
    axa.set(yscale="log", xscale="log", xlabel=r"$T_e$  [eV]",
            ylabel=r"$n_e$  [cm$^{-3}$]")
    axa.set_xticks([1, 2, 3, 5, 10])
    axa.set_xticklabels(["1", "2", "3", "5", "10"])
    axa.set_title(r"(a) shift in $M$ when CCC $\rightarrow$ RMPS", loc="left")
    cb = fig.colorbar(pc, ax=axa, pad=0.02)
    cb.set_label(r"$|\Delta M|$  [%]", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # (b) which coefficient each dataset moves
    ok = rs[rs.window_ok & (rs.Te >= 2.0)]
    quants = [("Delta", r"$\Delta$"), ("cap", "cap"),
              ("Sbar", r"$\overline{S}$"), ("G", "$G$"),
              ("eps", r"$\epsilon_{\mathrm{plateau}}$")]
    x = np.arange(len(quants))
    w = 0.38
    for off, nm, lab, c in ((-w / 2, "exc", "excitation $\\rightarrow$ RMPS", "#0072B2"),
                            (+w / 2, "ion", "ionisation $\\rightarrow$ Lotz", "#D55E00")):
        vals = [((ok[f"{q}_{nm}"].abs() / ok[q].abs() - 1) * 100).abs().mean()
                for q, _ in quants]
        axb.bar(x + off, vals, w, color=c, label=lab, lw=0)
    axb.set_xticks(x)
    axb.set_xticklabels([l for _, l in quants])
    axb.set_ylabel(r"mean $|$change$|$  [%]")
    axb.set_title("(b) effect of each substitution", loc="left")
    axb.set_ylim(0, 21.5)          # headroom so the legend clears the bars
    axb.legend(frameon=False, handletextpad=0.5, labelspacing=0.25,
               loc="upper left", ncol=1)
    axb.grid(axis="y", alpha=0.25, lw=0.5)
    axb.set_axisbelow(True)

    fig.tight_layout(pad=0.4)
    save(fig, "fig2_4_atomic_propagation")


if __name__ == "__main__":
    print("Chapter 2 atomic-data figures")
    fig_benchmark()
    fig_rydberg()
    fig_propagation()
    print("done.")
