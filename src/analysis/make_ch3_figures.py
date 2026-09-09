"""
make_ch3_figures.py
===================
Generates the four figures for Chapter 3 from the current CR matrix.

File names keep their original numbering; the FIGURE numbers in the chapter
follow first reference, so they differ:

  fig3_4_partition.pdf   -> Figure 3.1  CR network and slow/fast partition
  fig3_1_spectrum.pdf    -> Figure 3.2  eigenvalue ladder: one gap, dense band
  fig3_2_R_of_b1.pdf     -> Figure 3.3  R^QSS(b_1) at the benchmark, both limits
  fig3_3_ground_fed.pdf  -> Figure 3.4  f_3, f_4 and f_3 - f_4 against b_1

Every figure is stamped with a combined SHA-8 over L_grid, S_grid and the state
index -- not L_grid alone, since Figures 3.3 and 3.4 depend on all three and a
change in S_grid would otherwise leave the stamp identical. Vector PDF is what
goes in the thesis; a PNG preview is written alongside each for inspection
only.

The b_1 range shown is b_1^CRE, the CR-EQUILIBRIUM departure coefficient at
each grid point. It is NOT the transient b_1(t) realised after a temperature
step. Distinguishing the two is the subject of the chapter and the figures must
not blur them.

All quantities are computed here, not read from cached arrays, so the figures
cannot silently disagree with the text. The numbers printed to stdout are the
ones quoted in Chapter 3; check them.

Report only: writes to figures/, modifies nothing else.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
from cr_context import CRContext  # noqa: E402

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

H_PLANCK, M_E, EV_TO_J = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19
CHI_H = 13.605693


def saha_Z(p: int, te_ev: float) -> float:
    """Saha-Boltzmann coefficient of level p, in cm^3, with Te in eV."""
    return (1e6 * p ** 2
            * (H_PLANCK ** 2 / (2 * np.pi * M_E * EV_TO_J * te_ev)) ** 1.5
            * np.exp((CHI_H / p ** 2) / te_ev))


def main():
    ctx = CRContext.load()
    root = ctx.root
    Lp = root / "data/processed/cr_matrix/L_grid.npy"
    S = np.load(root / "data/processed/cr_matrix/S_grid.npy")
    L, Te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    nv = np.asarray(ctx.n_values)
    g = int(ctx.ground_index)
    E = np.array([k for k in range(ctx.n_states) if k != g])
    pos = {s: k for k, s in enumerate(E)}

    Sp = root / "data/processed/cr_matrix/S_grid.npy"
    h = hashlib.sha256()
    for path in (Lp, Sp, Path(ctx.state_index_path)):
        h.update(Path(path).read_bytes())
    sha8 = h.hexdigest()[:8]
    stamp = f"CR data {sha8} · {datetime.now():%Y-%m-%d}"
    outdir = root / "figures"
    outdir.mkdir(exist_ok=True)

    def provenance(fig):
        # negative y puts it outside the axes; bbox_inches="tight" expands to
        # include it, so it can never overlap an axis label
        fig.text(1.0, -0.035, stamp, ha="right", va="top",
                 fontsize=5, color="0.55")

    # ---- grid-wide eigenvalue sanity check --------------------------------
    # Chapter 3 defines tau_QSS and tau_relax from real negative eigenvalues at
    # EVERY grid point, not only at the benchmark. This check belongs here
    # rather than in a one-off verification script: this is the script that
    # gets re-run when the matrix changes, so the guard must live where the
    # regeneration happens.
    # The loop already diagonalises every point, so Eq. (3.14)'s grid-wide
    # ranges and the "one gap, dense band" structure cost nothing extra here.
    # Section 3.3.4 currently quotes them from a separate script; computing
    # them inside the script that draws the spectrum is what stops the two
    # from drifting apart.
    max_real, max_imag_rel = -np.inf, 0.0
    M_gr = np.zeros((len(Te), len(ne)))
    tq_gr = np.zeros_like(M_gr)
    tr_gr = np.zeros_like(M_gr)
    iso_gr = np.zeros_like(M_gr)
    for i in range(len(Te)):
        for j in range(len(ne)):
            e_ij = np.linalg.eigvals(L[i, j])
            max_real = max(max_real, e_ij.real.max())
            t_ij = np.sort(1.0 / np.abs(e_ij.real))[::-1]
            tq_gr[i, j], tr_gr[i, j] = t_ij[0], t_ij[1]
            M_gr[i, j] = t_ij[0] / t_ij[1]
            r_ij = t_ij[:-1] / t_ij[1:]
            iso_gr[i, j] = r_ij[0] / r_ij[1:].max()
            # PER-EIGENVALUE, not global-max over global-max. The spectrum
            # spans nine orders of magnitude and lambda_0 is the smallest, so
            # a global ratio can hide a complex slow mode entirely behind a
            # large fast one -- and lambda_0 is exactly what tau_QSS is
            # defined from.
            max_imag_rel = max(max_imag_rel,
                               float(np.max(np.abs(e_ij.imag)
                                            / np.maximum(np.abs(e_ij),
                                                         1e-300))))
    if max_real >= 0:
        raise RuntimeError(f"non-decaying mode somewhere on the grid: "
                           f"max Re(lambda) = {max_real:.3e}")
    if max_imag_rel > 1e-10:
        raise RuntimeError(f"complex eigenvalues somewhere on the grid: "
                           f"max relative imaginary part {max_imag_rel:.3e}")
    print(f"all-grid eigenvalue check: max Re(lambda) = {max_real:.3e}, "
          f"max imaginary ratio = {max_imag_rel:.3e}")
    aM, bM = np.unravel_index(np.argmin(M_gr), M_gr.shape)
    cM, dM = np.unravel_index(np.argmax(M_gr), M_gr.shape)
    aI, bI = np.unravel_index(np.argmin(iso_gr), iso_gr.shape)
    print(f"all-grid tau_relax  {tr_gr.min():.4e} .. {tr_gr.max():.4e} s")
    print(f"all-grid tau_QSS    {tq_gr.min():.4e} .. {tq_gr.max():.4e} s")
    print(f"all-grid M          {M_gr.min():.4g} at [{aM},{bM}] "
          f"(Te={Te[aM]:.3g} eV, ne={ne[bM]:.3g}) .. {M_gr.max():.4g} at "
          f"[{cM},{dM}] (Te={Te[cM]:.3g} eV, ne={ne[dM]:.3g})")
    print(f"all-grid isolation  min {iso_gr.min():.0f}x at [{aI},{bI}] "
          f"(Te={Te[aI]:.3g} eV, ne={ne[bI]:.3g})  "
          f"-- first gap vs largest of the other 41")
    if iso_gr.min() < 10.0:
        raise RuntimeError(f"the slow mode is not isolated everywhere: "
                           f"min isolation {iso_gr.min():.2f}x at "
                           f"[{aI},{bI}]")

    ib, jb = 23, 5          # the thesis benchmark point, fixed by definition
    assert np.isclose(Te[ib], 2.947, rtol=5e-4), f"Te[{ib}]={Te[ib]}"
    assert np.isclose(ne[jb], 1.389e14, rtol=5e-4), f"ne[{jb}]={ne[jb]}"
    print(f"benchmark grid [{ib},{jb}]  Te={Te[ib]:.4f} eV  ne={ne[jb]:.4e} cm^-3")
    print(f"CR data sha8 {sha8}  (L_grid + S_grid + state_index)\n")

    # =======================================================================
    # FIG 3.1 -- the spectrum: one gap, then a continuum
    # =======================================================================
    eig = np.linalg.eigvals(L[ib, jb])
    if np.any(eig.real >= 0):
        raise RuntimeError(f"non-decaying eigenvalue: max Re={eig.real.max():.3e}")
    imag_rel = float(np.max(np.abs(eig.imag)
                            / np.maximum(np.abs(eig), 1e-300)))
    if imag_rel > 1e-10:
        raise RuntimeError(f"complex eigenvalues need explicit treatment: "
                           f"max relative imaginary part {imag_rel:.3e}")
    ev = np.sort(eig.real)[::-1]
    tau = 1.0 / np.abs(ev)
    print(f"        eigenvalues real to {imag_rel:.2e}, all Re < 0")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.9))

    axa.semilogy(np.arange(len(tau)), tau, "o", ms=3.5, color="#1f4e79",
                 markeredgewidth=0)
    axa.semilogy([0], [tau[0]], "o", ms=6, mfc="none", mec="#c0392b", mew=1.2)
    axa.annotate(r"$\tau_{\rm QSS}$", (0, tau[0]), xytext=(4, tau[0] * 0.35),
                 fontsize=8, color="#c0392b")
    axa.annotate(r"$\tau_{\rm relax}$", (1, tau[1]), xytext=(4, tau[1] * 2.2),
                 fontsize=8, color="#1f4e79")
    axa.annotate("", xy=(0.4, tau[0]), xytext=(0.4, tau[1]),
                 arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.35"))
    axa.text(0.9, np.sqrt(tau[0] * tau[1]), rf"$M = {tau[0]/tau[1]:.0f}$",
             fontsize=8, color="0.25")
    axa.set_xlabel("mode index $k$")
    axa.set_ylabel(r"$\tau_k = 1/|\lambda_k|$   [s]")
    axa.set_title("(a) all 43 modes", loc="left")
    axa.grid(alpha=0.25, lw=0.4)

    # ratio between consecutive modes -- shows the gap is unique
    ratio = tau[:-1] / tau[1:]
    axb.semilogy(np.arange(len(ratio)), ratio, "s", ms=3.2, color="#1f4e79",
                 markeredgewidth=0)
    axb.axhline(1, color="0.6", lw=0.6)
    axb.set_xlabel("gap between modes $k$ and $k+1$")
    axb.set_ylabel(r"$\tau_k / \tau_{k+1}$")
    axb.set_title("(b) one isolated gap, then a dense band", loc="left")
    axb.grid(alpha=0.25, lw=0.4)
    axb.annotate(f"{ratio[0]:.0f}", (0, ratio[0]), xytext=(2.5, ratio[0] * 0.4),
                 fontsize=8, color="#c0392b",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#c0392b"))

    provenance(fig)
    fig.savefig(outdir / "fig3_1_spectrum.pdf")
    fig.savefig(outdir / "fig3_1_spectrum.png", dpi=150)
    plt.close(fig)
    print(f"fig3_1_spectrum      tau_QSS={tau[0]:.4e}  tau_relax={tau[1]:.4e}  "
          f"M={tau[0]/tau[1]:.1f}  next gap={ratio[1]:.2f}  "
          f"largest competing gap={ratio[1:].max():.2f} "
          f"at k={int(np.argmax(ratio[1:]))+1}  "
          f"isolation={ratio[0]/ratio[1:].max():.0f}x")

    # =======================================================================
    # channel decomposition, reused by figs 3.2 and 3.3
    # =======================================================================
    def channels(i, j):
        LEE = L[i, j][np.ix_(E, E)]
        LEg = L[i, j][np.ix_(E, [g])].ravel()
        a = np.linalg.solve(LEE, -LEg)
        c = np.linalg.solve(LEE, -S[i, j][E])
        ra = (np.linalg.norm(LEE @ a + LEg)
              / max(np.linalg.norm(LEg), 1e-300))
        rc = (np.linalg.norm(LEE @ c + S[i, j][E])
              / max(np.linalg.norm(S[i, j][E]), 1e-300))
        if ra > 1e-10 or rc > 1e-10:
            raise RuntimeError(f"channel solve failed at ({i},{j}): "
                               f"res_a={ra:.3e} res_c={rc:.3e}")
        n_full = np.linalg.solve(L[i, j], -S[i, j])
        sup = (np.abs(a * n_full[g] + c - n_full[E]).max()
               / max(np.abs(n_full[E]).max(), 1e-300))
        if sup > 1e-8:
            raise RuntimeError(f"superposition n_E = a n_g + c broken at "
                               f"({i},{j}): {sup:.3e}")
        # Section 3.5.4 claims 0 <= f_m <= 1, which needs a, c >= 0. That
        # follows from -L_EE^-1 being elementwise non-negative (M-matrix),
        # but the figure script should verify rather than inherit it.
        if a.min() < -1e-10 * max(np.abs(a).max(), 1e-300):
            raise RuntimeError(f"negative ground-fed response at ({i},{j}): "
                               f"min a = {a.min():.3e}")
        if c.min() < -1e-10 * max(np.abs(c).max(), 1e-300):
            raise RuntimeError(f"negative recombination-fed response at "
                               f"({i},{j}): min c = {c.min():.3e}")
        am = {m: a[[pos[s] for s in np.where(nv == m)[0]]].sum() for m in (3, 4)}
        cm = {m: c[[pos[s] for s in np.where(nv == m)[0]]].sum() for m in (3, 4)}
        return am, cm

    def b1_cre(i, j):
        """The CR-EQUILIBRIUM departure coefficient: solves L n + S = 0, i.e.
        the ground-state CR balance against the FIXED ion reservoir has
        settled. Not "ionisation balance" in the two-way sense: n_ion is
        prescribed and never evolved, so nothing here balances the ion
        inventory. This is also NOT the transient b_1(t) realised after a
        temperature step -- distinguishing the two is the subject of the
        chapter, and the figures must not blur them."""
        n = np.linalg.solve(L[i, j], -S[i, j])      # n_ion = 1
        return n[g] / (saha_Z(1, Te[i]) * ne[j])

    b1_cre_grid = np.array([[b1_cre(i, j) for j in range(len(ne))]
                            for i in range(len(Te))])
    if not np.all(np.isfinite(b1_cre_grid)) or np.any(b1_cre_grid <= 0):
        raise RuntimeError("non-positive or non-finite b_1^CRE on the grid")
    print(f"        b_1^CRE grid range {b1_cre_grid.min():.3e}"
          f" .. {b1_cre_grid.max():.3e}   benchmark {b1_cre_grid[ib,jb]:.3e}")

    # =======================================================================
    # FIG 3.2 -- R^QSS(b_1) at the benchmark point ONLY.
    #   The multi-condition comparison belongs in Chapter 5, which asks WHERE
    #   the error occurs. Chapter 3 asks only whether the dependence exists,
    #   and one curve answers that without a legend competing for the
    #   transition region. The curve is a HYPOTHETICAL family: what R would be
    #   if b_1 took other values at fixed (Te, ne). The marker shows b_1^CRE,
    #   the single value the equilibrium closure selects.
    # =======================================================================
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bb = np.logspace(-2, 8, 600)
    am, cm = channels(ib, jb)
    Zn = saha_Z(1, Te[ib]) * ne[jb]
    u = bb * Zn
    R = (am[3] * u + cm[3]) / (am[4] * u + cm[4])
    R_rec, R_grd = cm[3] / cm[4], am[3] / am[4]
    b1b = b1_cre_grid[ib, jb]

    ax.axvspan(b1_cre_grid.min(), b1_cre_grid.max(), color="0.88",
               alpha=0.7, zorder=0, lw=0)
    ax.loglog(bb, R, color="#c0392b", lw=1.6, zorder=3)
    ax.axhline(R_rec, color="#8e44ad", lw=0.9, ls="--")
    ax.axhline(R_grd, color="#2e7d32", lw=0.9, ls="--")
    ax.plot([b1b], [(am[3] * b1b * Zn + cm[3]) / (am[4] * b1b * Zn + cm[4])],
            "o", ms=6, color="#c0392b", mec="w", mew=0.9, zorder=4)

    ax.set_yticks([R_rec, 1.0, 2.0, R_grd])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.text(bb[5], R_rec * 0.90,
            r"recombination-fed limit $c_3/c_4$", fontsize=7.5,
            color="#8e44ad", va="top")
    ax.text(bb[-1], R_grd * 1.06, r"ground-fed limit $a_3/a_4$",
            fontsize=7.5, color="#2e7d32", ha="right", va="bottom")
    ax.annotate(r"$b_1^{\rm CRE}$", (b1b, (am[3] * b1b * Zn + cm[3])
                                    / (am[4] * b1b * Zn + cm[4])),
                xytext=(b1b / 60, R_rec * 1.9), fontsize=8,
                color="#c0392b",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#c0392b"))
    ax.text(np.sqrt(b1_cre_grid.min() * b1_cre_grid.max()),
            R_grd * 1.45, r"$b_1^{\rm CRE}$ over the grid",
            ha="center", fontsize=7, color="0.35")

    ax.set_xlabel(r"ground-state departure coefficient $b_1$")
    ax.set_ylabel(r"$R^{\rm QSS} = n_3/n_4$")
    ax.set_ylim(R_rec * 0.72, R_grd * 1.9)
    ax.grid(alpha=0.25, lw=0.4, which="both")
    provenance(fig)
    fig.savefig(outdir / "fig3_2_R_of_b1.pdf")
    fig.savefig(outdir / "fig3_2_R_of_b1.png", dpi=150)
    plt.close(fig)
    print(f"fig3_2_R_of_b1      R_rec={R_rec:.4f}  R_grd={R_grd:.4f}  "
          f"span={R_grd/R_rec:.3f}x  b1_CRE={b1b:.3e}")

    # =======================================================================
    # FIG 3.3 -- the ground-fed fractions and their difference
    # =======================================================================
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    am, cm = channels(ib, jb)
    Zn = saha_Z(1, Te[ib]) * ne[jb]
    f3 = am[3] * bb * Zn / (am[3] * bb * Zn + cm[3])
    f4 = am[4] * bb * Zn / (am[4] * bb * Zn + cm[4])
    ax.semilogx(bb, f3, color="#1f4e79", label=r"$f_3$")
    ax.semilogx(bb, f4, color="#2e7d32", ls="--", label=r"$f_4$")
    ax.semilogx(bb, f3 - f4, color="#c0392b", lw=1.6,
                label=r"$f_3 - f_4 = \mathrm{d}\ln R/\mathrm{d}\ln b_1$")
    ax.fill_between(bb, 0, f3 - f4, color="#c0392b", alpha=0.10)
    sens = f3 - f4
    k = int(np.argmax(np.abs(sens)))
    ax.plot([bb[k]], [(f3 - f4)[k]], "o", ms=5, color="#c0392b", mec="w", mew=0.7)
    ax.annotate(rf"$\max|f_3-f_4| = {abs(sens[k]):.2f}$",
                (bb[k], sens[k]),
                xytext=(bb[k] * 12, (f3 - f4)[k] + 0.04), fontsize=8,
                color="#c0392b")
    ax.axvline(b1_cre_grid[ib, jb], color="0.4", lw=0.7, ls=":")
    # "$b_1^{CRE}$ at benchmark" is 19 characters rotated downward from
    # y = 0.98; it reached y ~ 0.65 and crossed the f_3 curve. Short form on
    # the figure, "at the benchmark point" in the caption.
    ax.text(b1_cre_grid[ib, jb] * 1.4, 0.99, r"$b_1^{\rm CRE}$", rotation=90,
            fontsize=7, color="0.4", va="top")
    ax.text(1e-3, 0.06, "recombination-fed\n$f_3=f_4=0$", fontsize=7, color="0.35")
    # was (1e7, 0.80): f_4 is still climbing through 0.9 there. Drop into
    # the band between the decayed sensitivity and the merged fractions.
    ax.text(bb[-1], 0.72, "ground-fed\n$f_3=f_4=1$", fontsize=7, color="0.35",
            ha="right")
    ax.set_xlabel(r"ground-state departure coefficient $b_1$")
    ax.set_ylabel(r"ground-fed fraction $f_m$,"
                  "\n"
                  r"and sensitivity $f_3-f_4$")
    ax.set_ylim(-0.04, 1.06)
    ax.legend(frameon=False, loc="center left")
    ax.grid(alpha=0.25, lw=0.4)
    provenance(fig)
    fig.savefig(outdir / "fig3_3_ground_fed.pdf")
    fig.savefig(outdir / "fig3_3_ground_fed.png", dpi=150)
    plt.close(fig)
    # ---- closed form for the peak, and the area sum rule -------------------
    # In x = ln(b_1 Z n_e) each f_m is an exact logistic centred at
    # x_m = ln(c_m/a_m), all of unit width; they differ only by a shift
    #     Delta = x_4 - x_3 = ln[(a_3/a_4)/(c_3/c_4)] = ln(R_grd/R_rec).
    # f_3 - f_4 is then even about the midpoint, so its maximum sits there and
    #     max|f_3 - f_4| = 2*sigma(Delta/2) - 1 = tanh(Delta/4),
    # while its area is Delta itself. Both are fixed by the two asymptotes of
    # Figure 3.3 alone -- no property of the 42-state network survives into
    # either. Section 3.5.6 quotes these; they are computed here so the text
    # and the figure cannot drift apart.
    Delta = np.log((am[3] / am[4]) / (cm[3] / cm[4]))
    # tanh(|Delta|/4): max|f_3-f_4| is non-negative whatever the order of
    # the shell pair, while Delta changes sign if the pair is reversed.
    peak_closed = np.tanh(abs(Delta) / 4.0)
    area_num = np.trapezoid(sens, np.log(bb))
    b1_peak_closed = np.sqrt((cm[3] / am[3]) * (cm[4] / am[4])) / Zn
    for name, num, closed in (("peak", abs(sens[k]), peak_closed),
                              ("area", area_num, Delta)):
        rel = abs(num - closed) / max(abs(closed), 1e-300)
        if rel > 5e-3:
            raise RuntimeError(f"logistic closed form broken for {name}: "
                               f"numerical {num:.6e} vs closed {closed:.6e} "
                               f"(rel {rel:.2e})")
    # argmax cannot resolve the peak location better than one grid step, so
    # this one is toleranced against the sampling of bb, not against 5e-3.
    # Each shell is half ground-fed at b_1 = c_m/(a_m Z n_e). Their ratio is
    # exp(Delta) = R_grd/R_rec exactly -- a fourth, independent route to the
    # same number, and the one that shows the two sigmoids are a single curve
    # displaced rather than two unrelated ones.
    b1_half = {m: cm[m] / (am[m] * Zn) for m in (3, 4)}
    ratio_half = b1_half[4] / b1_half[3]
    if abs(np.log(ratio_half / np.exp(Delta))) > 1e-10:
        raise RuntimeError(f"switching-point ratio {ratio_half:.6f} does not "
                           f"equal R_grd/R_rec = {np.exp(Delta):.6f}")
    f_b = {m: am[m] * b1b * Zn / (am[m] * b1b * Zn + cm[m]) for m in (3, 4)}
    # R at the benchmark, three ways: directly, and through each identity
    R_id_g = R_grd * f_b[4] / f_b[3]
    R_id_r = R_rec * (1 - f_b[4]) / (1 - f_b[3])
    R_dir = (am[3] * b1b * Zn + cm[3]) / (am[4] * b1b * Zn + cm[4])
    for nm, v in (("via a", R_id_g), ("via c", R_id_r)):
        if abs(v - R_dir) / R_dir > 1e-10:
            raise RuntimeError(f"benchmark R identity {nm} broken: "
                               f"{v:.8e} vs {R_dir:.8e}")
    print(f"        half-ground-fed at b_1 = {b1_half[3]:.4g} (n=3), "
          f"{b1_half[4]:.4g} (n=4); ratio {ratio_half:.4f} = R_grd/R_rec")
    print(f"        benchmark f_3={f_b[3]:.4f} f_4={f_b[4]:.4f}  "
          f"f_3-f_4={f_b[3]-f_b[4]:.4f} = "
          f"{100*(f_b[3]-f_b[4])/peak_closed:.1f}% of peak;  R={R_dir:.4f}")
    dlnb = np.log(bb[1] / bb[0])
    if abs(np.log(bb[k] / b1_peak_closed)) > 2.0 * dlnb:
        raise RuntimeError(f"peak of f_3-f_4 at b_1={bb[k]:.4e}, closed form "
                           f"predicts {b1_peak_closed:.4e} "
                           f"({abs(np.log(bb[k]/b1_peak_closed))/dlnb:.1f} "
                           f"grid steps apart)")
    print(f"        Delta=ln(R_grd/R_rec)={Delta:.6f}  "
          f"tanh(Delta/4)={peak_closed:.6f} vs numerical {abs(sens[k]):.6f}  "
          f"area={area_num:.6f} vs {Delta:.6f}")
    print(f"fig3_3_ground_fed   max|f3-f4|={abs(sens[k]):.4f} at b_1={bb[k]:.3e}; "
          f"at benchmark f3={am[3]*b1_cre_grid[ib,jb]*Zn/(am[3]*b1_cre_grid[ib,jb]*Zn+cm[3]):.4f} "
          f"f4={am[4]*b1_cre_grid[ib,jb]*Zn/(am[4]*b1_cre_grid[ib,jb]*Zn+cm[4]):.4f}")

    # =======================================================================
    # FIG 3.4 -- SCHEMATIC of the network and the slow/fast partition
    #   NOT drawn on a true energy scale. On -13.6/n^2 everything above n~4
    #   collapses into a sliver below the continuum and no label fits. Level
    #   rows are therefore evenly spaced and the axis is unlabelled, which is
    #   standard for a Grotrian-style schematic. The figure states this.
    # =======================================================================
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axis("off")

    ROW = {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0,
           "gap": 3.9, 9: 4.6, 12: 5.3, 15: 6.0, "cont": 7.0}
    XL, XR = 0.14, 0.68          # level-column box edges
    XA = 0.72                    # annotation column, always ha="left"

    # FIX 1: axhline takes AXES fractions, not data coords -- it ran under the
    # continuum label and struck it through. Draw in data coords.
    ax.plot([XL - 0.02, XR - 0.02], [ROW["cont"]] * 2, color="0.15", lw=1.4)
    ax.text(XA, ROW["cont"], r"continuum (H$^+$, fixed reservoir)",
            fontsize=7.5, va="center", ha="left")

    subl = {1: ["1s"], 2: ["2s", "2p"], 3: ["3s", "3p", "3d"],
            4: ["4s", "4p", "4d", "4f"]}
    xs = {1: [0.34], 2: [0.26, 0.42], 3: [0.22, 0.34, 0.46],
          4: [0.18, 0.30, 0.42, 0.54]}
    for n, labs in subl.items():
        for x, lab in zip(xs[n], labs):
            ax.plot([x - 0.045, x + 0.045], [ROW[n]] * 2, color="0.15", lw=1.4)
            ax.text(x, ROW[n] + 0.13, lab, fontsize=7, ha="center")
    ax.text(XA, ROW[3] + 0.5, r"$n \leq 8$: $\ell$-resolved", fontsize=7.5,
            va="center", ha="left", color="0.3")

    ax.text(0.36, ROW["gap"], r"$\vdots$", fontsize=9, ha="center", color="0.45")
    ax.text(XA, ROW["gap"], r"$n = 5$--$8$: $\ell$-resolved, not drawn",
            fontsize=7, va="center", ha="left", color="0.45")

    for n in (9, 12, 15):
        ax.plot([0.24, 0.48], [ROW[n]] * 2, color="0.45", lw=1.2)
        ax.text(0.50, ROW[n], rf"$n={n}$", fontsize=7, va="center", color="0.45")
    ax.text(XA, ROW[9], r"$n = 9$--$15$: bundled", fontsize=7.5,
            va="center", ha="left", color="0.45")

    # l-mixing acts in every resolved shell, but arrows are drawn only for
    # n = 2 and 3. The 4l sublevels sit 0.12 apart with 0.09 of level line,
    # leaving 0.03 of clear space for a 0.02-wide arrow: drawn, it renders as
    # a blob that welds the n=4 row into one continuous bar. The label says
    # "all resolved n" and the caption states which shells are drawn.
    for n in (2, 3):
        for x0, x1 in zip(xs[n][:-1], xs[n][1:]):
            ax.add_patch(FancyArrowPatch((x0 + 0.05, ROW[n]), (x1 - 0.05, ROW[n]),
                                         arrowstyle="<|-|>", mutation_scale=7,
                                         lw=1.0, color="#e67e22"))
    ax.text(XA, ROW[2], r"proton $\ell$-mixing ($\Delta n = 0$, all resolved $n$)",
            fontsize=7.5, va="center", ha="left", color="#e67e22")

    # radiative 2p -> 1s
    ax.add_patch(FancyArrowPatch((0.42, ROW[2] - 0.08), (0.36, ROW[1] + 0.10),
                                 arrowstyle="-|>", mutation_scale=8, lw=1.1,
                                 color="0.35", connectionstyle="arc3,rad=-0.2"))
    ax.text(0.455, (ROW[1] + ROW[2]) / 2 - 0.08, r"$A_{2p\to1s}$", fontsize=7,
            color="0.35", ha="left", va="center")
    # collisional 2s -> 1s only
    ax.add_patch(FancyArrowPatch((0.26, ROW[2] - 0.08), (0.32, ROW[1] + 0.10),
                                 arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                 color="#2e7d32", ls=(0, (4, 2)),
                                 connectionstyle="arc3,rad=0.2"))
    # FIX 3: was ha="right" at x=0.235, so it ran leftward across the fast-set
    # box border and collided with the l-mixing label. Annotation column instead.
    ax.text(XA, ROW[1] + 0.35, r"$2s \to 1s$: collisional only, no E1",
            fontsize=7.5, va="center", ha="left", color="#2e7d32")

    # the partition
    ax.add_patch(Rectangle((XL, ROW[1] - 0.34), XR - XL, 0.60, fill=False,
                           ec="#c0392b", lw=1.1, ls="--"))
    ax.text(XL + 0.01, ROW[1] - 0.52, r"slow: $\mathcal{S} = \{1s\}$",
            fontsize=8, color="#c0392b", va="top")
    ax.add_patch(Rectangle((XL, ROW[2] - 0.40), XR - XL,
                           ROW[15] - ROW[2] + 0.72, fill=False,
                           ec="#1f4e79", lw=1.1, ls="--"))
    ax.text(XL + 0.01, ROW[15] + 0.44, r"fast: $\mathcal{F}$, 42 excited levels",
            fontsize=8, color="#1f4e79", va="bottom")

    # a and c: network responses
    ax.add_patch(FancyArrowPatch((0.08, ROW[1]), (0.08, ROW[4]),
                                 arrowstyle="-|>", mutation_scale=12,
                                 lw=2.6, color="#2e7d32", alpha=0.5))
    ax.text(0.055, (ROW[1] + ROW[4]) / 2, "ground-fed" "\n" r"response $\mathbf{a}$",
            fontsize=7.5, color="#2e7d32", ha="right", va="center")
    # FIX 4: the c arrow sat at x=0.71 with a ha="right" label that ran back over
    # the n=15 level and its label. Moved into the free channel at x=0.60, label
    # in the annotation column.
    ax.add_patch(FancyArrowPatch((0.60, ROW["cont"] - 0.12), (0.60, ROW[9]),
                                 arrowstyle="-|>", mutation_scale=12,
                                 lw=2.6, color="#8e44ad", alpha=0.5))
    ax.text(XA, 5.75,
            "recombination-fed" "\n" r"response $\mathbf{c}$",
            fontsize=7.5, color="#8e44ad", ha="left", va="center")

    ax.text(0.02, ROW[1] - 0.95, "schematic: vertical spacing not to scale",
            fontsize=6.5, color="0.45", va="top", style="italic")

    ax.set_xlim(-0.02, 1.16)
    ax.set_ylim(ROW[1] - 1.3, ROW["cont"] + 0.5)
    provenance(fig)
    fig.savefig(outdir / "fig3_4_partition.pdf")
    fig.savefig(outdir / "fig3_4_partition.png", dpi=150)
    plt.close(fig)

    print("\nwrote:")
    for f in ("fig3_1_spectrum", "fig3_2_R_of_b1",
              "fig3_3_ground_fed", "fig3_4_partition"):
        print(f"  {outdir / (f + '.pdf')}")


if __name__ == "__main__":
    main()