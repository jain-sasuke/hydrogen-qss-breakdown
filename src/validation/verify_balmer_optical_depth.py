#!/usr/bin/env python
"""
verify_balmer_optical_depth.py
==============================
Line-centre optical depth of H_alpha and H_beta across a 10 cm slab at every
grid point, under two absorber conventions, with the population escape factor
and the shift it puts on the observed H_alpha/H_beta ratio (backlog item N10).

WHY THIS EXISTS
---------------
Chapter 4 (thesis_tex/chapter4.tex ~803-812, "And is the plasma transparent
to its own Balmer light?") says: the optical depth of H_alpha across a 10 cm
slab is 6.3e-7 at the cold corner, 1.0e-3 at [0,4] and 0.263 at [0,7], the
single worst cell on the grid; even there the escape factor stays above 0.85,
which moves the observed ratio by at most about 10 % in that one cell and by
a negligible amount everywhere else. Those numbers rest on findings_10
ADDENDUM D.8 (outputs/findings_10_four_agent_review.md ~1209-1212), a working
note with no producing script, no stated absorber population, no stated
oscillator strength and no stated slab convention. Backlog G7 grades the item
verified on the strength of the same note. The rule is that every number in
the thesis comes from a run; this is that run.

PHYSICS AND METHOD
------------------
Line-centre absorption cross-section for a Doppler profile, in the form
escape_factor.lyman_alpha_sigma0 uses (Rybicki and Lightman eq. 10.6 with
phi(0) = 1/(sqrt(pi) dnu_D); Mihalas 9-2; ADAS214 eq. 3.14.5):

    sigma_0 = (pi e^2 / m_e c) f_lu / (sqrt(pi) dnu_D)
    dnu_D   = v_th / lambda_0,   v_th = sqrt(2 k T_n / m_H),   T_n = Te

kappa_0 = sum over absorbing lower levels of sigma_0(l) n_l is the
line-centre absorption coefficient [1/cm]. Populations are the thin CRE
ones, n = -L^{-1} S per unit n_ion, times n_ion = n_e (quasineutral pure
hydrogen): the expression verify_lyman_optical_depth.py and
make_story_figures.py use for n(1s), applied to n(2s), n(2p). Slab D = 10 cm:

    tau_full = kappa_0 D        the literal "across a 10 cm slab"
    tau_half = kappa_0 D / 2    escape_factor.py's tau_c, a photon born at
                                the slab centre

Escape factor Theta = escape_factor.escape_factor_quadrature(tau), the
ADAS214 population escape factor by quadrature, evaluated at tau_half (the
module's documented convention) and at tau_full (the literal reading); both
are reported. Observed-ratio shift: I(H_alpha)/I(H_beta) is multiplied by
Theta(H_alpha)/Theta(H_beta), so the shift is Theta_a/Theta_b - 1.

Two absorber conventions, carried side by side:
(a) shell level. n(n=2) = n_2s + n_2p with the total multiplet oscillator
    strengths of Wiese and Fuhr, J. Phys. Chem. Ref. Data 38, 565 (2009),
    Table 4 (H I, allowed transitions, average values), rows 40 and 41:
    f(2-3) = 0.64108, lambda_vac = 6564.64 A; f(2-4) = 0.11938,
    lambda_vac = 4862.70 A. The script reads those two rows from
    data/raw/wiese_fuhr.pdf (page index 8) with pypdf and STOPS if the parsed
    values differ from the transcribed ones; if pypdf is unavailable it says
    so and uses the transcription. The g-weighted mean of the pipeline's own
    component f values, sum_l (g_l/8) f_l, is 0.64075 / 0.11932 (the
    0.6407 / 0.1193 the task statement attributes to Wiese and Fuhr); it
    sits 0.05 % below Wiese-Fuhr because the pipeline's A and wavelengths
    are the infinite-nuclear-mass hydrogenic ones (chapter 2 ~718-730).
    Both are printed; (a) uses the Wiese-Fuhr values.
(b) l-resolved. Each E1 component with its own lower-level population and
    an absorption oscillator strength derived from the pipeline's A:

        f_lu = A_ul (g_u / g_l) m_e c lambda^2 / (8 pi^2 e^2)

    A_ul from data/processed/Radiative/A_resolved.npy ([lower, upper]),
    g = 2(2l+1) from state_index.csv, lambda = h c / (I_l - I_u) from the
    I_eV column of state_index.csv (13.6058/n^2 eV, infinite-mass Rydberg).
    H_alpha: 2s-3p, 2p-3s, 2p-3d; H_beta: 2s-4p, 2p-4s, 2p-4d. The three
    components are placed at one line centre and summed, an upper bound on
    the blended peak (the pipeline is LS-coupled without fine structure; the
    fine-structure spread of H_alpha, 0.1-0.2 A, is below the 1 eV Doppler
    1/e half-width of 0.30 A, so the bound is not loose).
Constants, Gaussian CGS, the values escape_factor.py uses so that G1 pins
them: e = 4.80326e-10 esu, m_e = 9.10938e-28 g, c = 2.99792e10 cm/s,
m_H = 1.67262e-24 g (proton mass), 1 eV = 1.60218e-12 erg. Only lambda in
(b) needs h = 6.62607e-27 erg s (CODATA 2018).

GATES (the run stops if any fails)
----------------------------------
G1  sigma_0 for Lyman-alpha rebuilt here (f = 0.4162, lambda = 1215.67 A,
    the module's own inputs) equals escape_factor.lyman_alpha_sigma0(T) to
    1e-6 relative at T = 1, 1.5, 2, 3, 5, 10 eV.
G2  u_CRE = [-L^{-1} S]_1s reproduces validation/molecular_channel/
    molecular_channel.csv u_CRE at 400/400 points to 1e-8 relative.
G3  the six f_lu derived from A_resolved.npy reproduce the f_abs column of
    data/processed/Radiative/H_A_E1_LS_n1_15_physical.csv (the pipeline's
    own, not used by the matrix) to 1e-3 relative. A wrong g ratio (x1.67)
    or a swapped wavelength (x1.8) fails this.
G0  A_resolved orientation: A[lower, upper] > 0 and A[upper, lower] = 0 for
    all six components; the Wiese-Fuhr transcription matches the PDF.

PREDICTIONS (written before the run)
------------------------------------
P1  one convention reproduces 6.3e-7 / 1.0e-3 / 0.263 at [0,0], [0,4], [0,7]
    to two significant figures; the note's likely convention is (a) with
    tau_full. Hand estimate for the worst cell: sigma_0(H_alpha, a, 1 eV)
    = 2.654e-2 x 0.641 x 6.565e-5 / (1.772 x 1.384e6) = 4.55e-13 cm^2, so
    tau_full = 0.263 needs n(n=2) = 5.8e10 cm^-3 at ne = 1e15, i.e.
    n_2/n_ion = 5.8e-5.
P1b (b)/(a) for tau(H_alpha) lies in [0.68, 1] everywhere and is closest to 1
    at ne = 1e15: 2p decays at 6.27e8 1/s while 2s has only collisional
    exits, so n_2p/n_2s < 3 (under-statistical) at low density, where (b)
    tends to n_2s f(2s-3p) = 0.435 n_2s against (a) 0.641 n_2s, ratio 0.68;
    at 1e15 l-mixing makes n=2 statistical and sum_l (g_l/8) f_l = f_n makes
    the two conventions coincide to 0.05 %.
P2  the worst cell on the grid for tau(H_alpha) is [0,7] under both
    conventions.
P3  Theta at [0,7] >= 0.85 and the H_alpha/H_beta shift there <= 10 %.
    Hand estimate under (a): tau_half = 0.13, Theta ~ 1 - 0.13/sqrt2 = 0.91;
    tau(H_beta)/tau(H_alpha) = (0.1194 x 4862.7)/(0.6411 x 6564.6) = 0.138,
    Theta_b(half) ~ 0.987, shift ~ -7.8 %: P3 holds for tau_half. Fed
    tau_full = 0.263 instead, Theta ~ 0.83 and the shift ~ -15 %: P3 fails
    for tau_full. So the chapter's "above 0.85" and "about 10 %" are
    predicted to be half-slab numbers attached to a full-slab tau.
P4  no cell with Te >= 2 eV has tau_full(H_alpha) > 0.1; grid-wide,
    tau_full > 1 at 0 cells and > 0.1 at a handful, all at Te < 1.3 eV and
    ne >= 3.7e14 cm^-3, under both conventions.
RESULT 3 (the sensitivity table at the named points: sqrt(kT/m) width,
    deuterium, T_n = 3 eV, pipeline f, (b), statistical n=2 from 2s or 2p
    alone, tau_half) carries no prediction. It was added after the first run
    on 21 Sep 2026 showed P1 failing under every convention, to locate the
    discrepancy with D.8; nothing above it was changed.

REFUTING OBSERVATION (for the chapter's "optically thin over the region
where this thesis makes quantitative claims")
------------------------------------------------------------------------
Any cell with Te >= 2 eV where tau_full(H_alpha) > 0.1 under either
convention, or where the ratio shift exceeds 10 % under either convention
and either tau definition.

OUTPUTS (with --write)
----------------------
validation/balmer_optical_depth/balmer_optical_depth.csv          per point
validation/balmer_optical_depth/balmer_optical_depth_summary.csv  quoted numbers
validation/balmer_optical_depth/balmer_optical_depth.txt          this run's log
Each carries a '#' header with the script name, date, interpreter and the
sha256 of L_grid.npy, S_grid.npy, state_index.csv, A_resolved.npy, the
radiative CSV, molecular_channel.csv and wiese_fuhr.pdf.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root  # noqa: E402

ROOT = find_repo_root(_HERE)
sys.path.insert(0, str(ROOT / "src" / "analysis"))
from escape_factor import (  # noqa: E402
    escape_factor_quadrature,
    lyman_alpha_sigma0,
)

# -- constants, Gaussian CGS; the first five are escape_factor.py's -----------
E_ESU = 4.80326e-10        # statcoulomb
M_E = 9.10938e-28          # g
C_CGS = 2.99792e10         # cm/s
M_H = 1.67262e-24          # g, proton mass (escape_factor.py's m_H)
EV_TO_ERG = 1.60218e-12
H_PLANCK = 6.62607e-27     # erg s, CODATA 2018; only for lambda in (b)
F_LYA, LAMBDA_LYA = 0.4162, 1.21567e-5     # escape_factor.py's Ly-alpha inputs
D_CM = 10.0

# Wiese and Fuhr, J. Phys. Chem. Ref. Data 38, 565 (2009), Table 4, H I
# allowed transitions, average values. Transcribed from data/raw/wiese_fuhr.pdf
# page index 8; checked against the PDF at run time (G0).
WF = {"Halpha": dict(row=40, label="2-3", f=0.64108, lam_vac_A=6564.64, A_1e8=0.44101),
      "Hbeta":  dict(row=41, label="2-4", f=0.11938, lam_vac_A=4862.70, A_1e8=0.084193)}
COMPONENTS = {"Halpha": [("2S", "3P"), ("2P", "3S"), ("2P", "3D")],
              "Hbeta":  [("2S", "4P"), ("2P", "4S"), ("2P", "4D")]}
# chapter 4 ~806-808 and findings_10 D.8: tau(H_alpha, 10 cm)
QUOTED = {(0, 0): 6.3e-7, (0, 4): 1.0e-3, (0, 7): 0.263}
NAMED = [("cold corner", (0, 0)), ("[0,4]", (0, 4)), ("[0,7]", (0, 7))]
CH4_THETA_MIN, CH4_SHIFT_MAX = 0.85, 0.10
BENCH_TE, BENCH_NE = 2.947, 1.389e14


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sigma0(f_lu: float, lam_cm: float, T_eV: float, mass_g: float = M_H,
           width_factor: float = 2.0) -> float:
    """Doppler line-centre cross-section [cm^2], escape_factor.py's form.
    width_factor = 2 is the 1/e half-width sqrt(2kT/m); 1 is the superseded
    sqrt(kT/m) width behind chapter 5's Lyman numbers (RESULT 3 only)."""
    v_th = np.sqrt(width_factor * T_eV * EV_TO_ERG / mass_g)
    dnu_D = v_th / lam_cm
    return (np.pi * E_ESU ** 2 / (M_E * C_CGS)) * f_lu / (np.sqrt(np.pi) * dnu_D)


def f_from_A(A_ul: float, g_u: float, g_l: float, lam_cm: float) -> float:
    """Absorption oscillator strength from the emission A (Gaussian CGS)."""
    return A_ul * (g_u / g_l) * M_E * C_CGS * lam_cm ** 2 / (8.0 * np.pi ** 2 * E_ESU ** 2)


def round_sig(v: float, n: int) -> float:
    return float(f"{v:.{n}g}")


def read_wiese_fuhr(pdf_path: Path):
    """Parse Table 4 rows 40 (2-3) and 41 (2-4) from page index 8 of the PDF.
    Returns (dict | None, reason)."""
    try:
        import pypdf
    except ImportError:
        return None, "pypdf not importable in this interpreter"
    if not pdf_path.is_file():
        return None, f"{pdf_path} is missing"
    text = pypdf.PdfReader(str(pdf_path)).pages[8].extract_text() or ""
    out = {}
    for name, tag in (("Halpha", "40 2–3 "), ("Hbeta", "41 2–4 ")):
        ln = next((l for l in text.splitlines() if l.startswith(tag)), None)
        if ln is None:
            return None, f"row '{tag.strip()}' not found on page index 8"
        sci = re.findall(r"(\d\.\d{4})e\s*([−+-])\s*(\d{2})", ln)
        lam = re.search(r"(\d) (\d{3}\.\d{2}) (\d) (\d{3}\.\d{2})", ln)
        if len(sci) < 2 or lam is None:
            return None, f"row '{tag.strip()}' did not parse: {ln!r}"

        def conv(m, s, e):
            return float(m) * 10.0 ** ((-1 if s in "−-" else 1) * int(e))

        out[name] = dict(A_1e8=conv(*sci[0]), f=conv(*sci[1]),
                         lam_air_A=float(lam.group(1) + lam.group(2)),
                         lam_vac_A=float(lam.group(3) + lam.group(4)), raw=ln)
    return out, "parsed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--slab", type=float, default=D_CM, help="full slab thickness D [cm]")
    args = ap.parse_args()
    D = float(args.slab)

    ctx = CRContext.load(root=ROOT)
    root = ctx.root
    L, TeL, neL = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    nT, nN, nS, _ = L.shape
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    L_path = root / "data/processed/cr_matrix/L_grid.npy"
    A_path = root / "data/processed/Radiative/A_resolved.npy"
    rad_csv = root / "data/processed/Radiative/H_A_E1_LS_n1_15_physical.csv"
    mol_path = root / "validation/molecular_channel/molecular_channel.csv"
    pdf_path = root / "data/raw/wiese_fuhr.pdf"
    for p in (S_path, A_path, rad_csv, mol_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    S = np.load(S_path)
    if S.shape != (nT, nN, nS):
        raise ValueError(f"S_grid shape {S.shape} does not match L_grid {L.shape}")
    A = np.load(A_path)
    if A.shape != (36, 36):
        raise ValueError(f"A_resolved shape {A.shape}, expected (36, 36)")
    si = pd.read_csv(ctx.state_index_path)
    for col in ("label", "n", "l", "g", "I_eV"):
        if col not in si.columns:
            raise ValueError(f"{ctx.state_index_path} lacks column {col!r}")
    lab = {str(r.label): int(r.idx) for r in si.itertuples()}
    g_of = {str(r.label): float(r.g) for r in si.itertuples()}
    I_of = {str(r.label): float(r.I_eV) for r in si.itertuples()}
    g = ctx.ground_index
    if ctx.labels[g] != "1S":
        raise RuntimeError(f"ground state label is {ctx.labels[g]!r}, not '1S'")
    i2s, i2p = lab["2S"], lab["2P"]
    if not (g_of["2S"] == 2 and g_of["2P"] == 6):
        raise RuntimeError("state_index g for 2S/2P is not 2/6")
    ti, ni = ctx.nearest_point(BENCH_TE, BENCH_NE)
    if (ti, ni) != (23, 5):
        raise ValueError(f"benchmark resolves to [{ti},{ni}], expected [23,5]")

    log: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        log.append(s)

    summary: list[tuple] = []

    say("=" * 78)
    say("BALMER OPTICAL DEPTH -- H_alpha and H_beta across a 10 cm slab, two conventions")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}")
    say(f"interpreter   {sys.executable}   numpy {np.__version__}")
    say(ctx.describe())
    say(f"  S_grid         : {S.shape}   ({S_path.relative_to(root)})")
    say(f"  A_resolved     : {A.shape}   ({A_path.relative_to(root)}), A[lower, upper]")
    say(f"  radiative csv  : {rad_csv.relative_to(root)}  (f_abs column, G3)")
    say(f"  G2 reference   : {mol_path.relative_to(root)}")
    say(f"  slab D         : {D:g} cm   tau_full = kappa_0 D,  tau_half = kappa_0 D/2")
    say(f"  neutral temp   : T_n = Te,  hydrogen m_H = {M_H:.5e} g,  n_ion = n_e")
    say("=" * 78)

    # -- G0a: A orientation ---------------------------------------------------
    say("\nG0a A_resolved orientation for the six Balmer components:")
    for line, comps in COMPONENTS.items():
        for lo, up in comps:
            fwd, back = A[lab[lo], lab[up]], A[lab[up], lab[lo]]
            say(f"    {line:6s} {lo}<-{up}: A[{lo},{up}] = {fwd:.5e}  A[{up},{lo}] = {back:.1e}")
            if not (fwd > 0 and back == 0):
                raise AssertionError("G0a FAILED: A_resolved is not [lower, upper] for this component")
    say("    G0a PASSED")

    # -- G1: sigma_0 constants pinned to the module ---------------------------
    say("\nG1  sigma_0(Ly-alpha) rebuilt here vs escape_factor.lyman_alpha_sigma0:")
    worst = 0.0
    for T in (1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
        s_mod = float(lyman_alpha_sigma0(T))
        s_ind = sigma0(F_LYA, LAMBDA_LYA, T)
        r = abs(s_ind / s_mod - 1)
        worst = max(worst, r)
        say(f"    T = {T:5.2f} eV  module {s_mod:.6e}  rebuilt {s_ind:.6e}  rel {r:.2e}")
    if worst > 1e-6:
        raise AssertionError(f"G1 FAILED: sigma_0 rebuild differs from the module by {worst:.2e}")
    say(f"    G1 PASSED (worst {worst:.1e} < 1e-6)")

    # -- G0b: Wiese-Fuhr values read from the PDF -----------------------------
    say("\nG0b Wiese and Fuhr (2009) Table 4 rows 40, 41 from data/raw/wiese_fuhr.pdf:")
    wf_pdf, why = read_wiese_fuhr(pdf_path)
    wf_source = "transcribed only"
    if wf_pdf is None:
        say(f"    WARNING: could not read the PDF ({why}); using the transcribed values")
    else:
        for name in ("Halpha", "Hbeta"):
            p, t = wf_pdf[name], WF[name]
            say(f"    row {t['row']} ({t['label']}): parsed f = {p['f']:.5f}  lambda_vac = {p['lam_vac_A']:.2f} A"
                f"  lambda_air = {p['lam_air_A']:.2f} A  A = {p['A_1e8']:.5g}e8 1/s"
                f"   | transcribed f = {t['f']:.5f}  lambda_vac = {t['lam_vac_A']:.2f} A  A = {t['A_1e8']:.5g}e8")
            for k in ("f", "lam_vac_A", "A_1e8"):
                if abs(p[k] / t[k] - 1) > 1e-9:
                    raise AssertionError(f"G0b FAILED: transcribed {name} {k} = {t[k]} but the PDF says {p[k]}")
        wf_source = "read from data/raw/wiese_fuhr.pdf page index 8 and equal to the transcription"
        say(f"    G0b PASSED: {wf_source}")

    # -- convention (b): f_lu from the pipeline's A; G3 -----------------------
    say("\n(b) l-resolved oscillator strengths from A_resolved.npy:")
    say(f"    f_lu = A_ul (g_u/g_l) m_e c lambda^2 / (8 pi^2 e^2);  m_e c/(8 pi^2 e^2) = "
        f"{M_E * C_CGS / (8 * np.pi ** 2 * E_ESU ** 2):.6f} s/cm^2;  lambda = h c/(I_l - I_u)")
    rad = pd.read_csv(rad_csv)
    lam_b: dict[str, float] = {}
    comp_f: dict[str, list[tuple[str, str, float, float]]] = {}
    g3_worst = 0.0
    for line, comps in COMPONENTS.items():
        lo0, up0 = comps[0]
        dE = (I_of[lo0] - I_of[up0]) * EV_TO_ERG
        lam_b[line] = H_PLANCK * C_CGS / dE
        comp_f[line] = []
        for lo, up in comps:
            if abs((I_of[lo] - I_of[up]) * EV_TO_ERG - dE) > 1e-12 * dE:
                raise RuntimeError(f"{lo}-{up} has a different level spacing from {lo0}-{up0}")
            fl = f_from_A(A[lab[lo], lab[up]], g_of[up], g_of[lo], lam_b[line])
            n_lo, n_up = int(si.n[lab[lo]]), int(si.n[lab[up]])
            l_lo, l_up = int(si.l[lab[lo]]), int(si.l[lab[up]])
            row = rad[(rad.nu == n_up) & (rad.lu == l_up) & (rad.nl == n_lo) & (rad.ll == l_lo)]
            if len(row) != 1:
                raise RuntimeError(f"radiative csv has {len(row)} rows for {lo}-{up}")
            f_csv = float(row.f_abs.iloc[0])
            r = abs(fl / f_csv - 1)
            g3_worst = max(g3_worst, r)
            comp_f[line].append((lo, up, fl, f_csv))
            say(f"    {line:6s} {lo}->{up}: A = {A[lab[lo], lab[up]]:.5e}  g_u/g_l = {g_of[up]:.0f}/{g_of[lo]:.0f}"
                f"  f_lu = {fl:.5f}   csv f_abs = {f_csv:.5f}  rel {r:.1e}")
        say(f"    {line:6s} lambda(state_index) = {lam_b[line] * 1e8:.2f} A   "
            f"(Wiese-Fuhr lambda_vac = {WF[line]['lam_vac_A']:.2f} A, ratio {lam_b[line] * 1e8 / WF[line]['lam_vac_A']:.5f})")
    say(f"\nG3  derived f_lu vs the pipeline's f_abs column, worst relative difference: {g3_worst:.2e}")
    if g3_worst > 1e-3:
        raise AssertionError("G3 FAILED: the oscillator strengths derived from A do not match the pipeline's own")
    say("    G3 PASSED")
    say("\n    g-weighted shell means sum_l (g_l/8) f_l of the derived set vs Wiese-Fuhr multiplet f:")
    f_shell_pipe: dict[str, float] = {}
    for line in COMPONENTS:
        f_shell_pipe[line] = sum(g_of[lo] / 8.0 * fl for lo, _, fl, _ in comp_f[line])
        say(f"    {line:6s} pipeline g-mean {f_shell_pipe[line]:.5f}   Wiese-Fuhr {WF[line]['f']:.5f}"
            f"   ratio {f_shell_pipe[line] / WF[line]['f']:.5f}")
        summary.append(("(a)", f"f shell {line} Wiese-Fuhr", WF[line]["f"], "", wf_source, ""))
        summary.append(("(a)", f"f shell {line} pipeline g-weighted mean", f_shell_pipe[line], "", "A_resolved.npy", ""))

    # -- populations and G2 ---------------------------------------------------
    say("\nCRE populations n = -L^{-1} S per unit n_ion at 400 points")
    u = np.empty((nT, nN)); n2s = np.empty((nT, nN)); n2p = np.empty((nT, nN))
    for i in range(nT):
        for j in range(nN):
            n = np.linalg.solve(L[i, j], -S[i, j])
            if not np.all(np.isfinite(n)) or n[g] <= 0 or n[i2s] <= 0 or n[i2p] <= 0:
                raise RuntimeError(f"non-positive CRE population at [{i},{j}]")
            u[i, j], n2s[i, j], n2p[i, j] = n[g], n[i2s], n[i2p]
    m = pd.read_csv(mol_path, comment="#").sort_values(["i", "j"])
    if len(m) != nT * nN:
        raise ValueError(f"molecular_channel.csv has {len(m)} rows, expected {nT * nN}")
    if np.any(m.i.values != np.repeat(np.arange(nT), nN)) or np.any(m.j.values != np.tile(np.arange(nN), nT)):
        raise ValueError("molecular_channel.csv (i, j) ordering is not the grid's")
    g2 = np.abs(u.ravel() / m.u_CRE.values - 1)
    say(f"G2  u_CRE vs molecular_channel.csv: worst relative {g2.max():.2e}, "
        f"{int((g2 <= 1e-8).sum())}/{nT * nN} within 1e-8")
    if g2.max() > 1e-8:
        raise AssertionError("G2 FAILED: the CRE ground fraction does not reproduce the stamped artifact")
    say("    G2 REPRODUCED")
    n2 = (n2s + n2p) * neL[None, :]                 # cm^-3, n_ion = n_e
    n2s_cm3, n2p_cm3 = n2s * neL[None, :], n2p * neL[None, :]
    ratio_2p_2s = n2p / n2s
    say(f"    n_2p/n_2s over the grid: {ratio_2p_2s.min():.3f} to {ratio_2p_2s.max():.3f} "
        f"(statistical = 3); at ne = {neL[-1]:.1e}: {ratio_2p_2s[:, -1].min():.3f} to {ratio_2p_2s[:, -1].max():.3f}")
    say(f"    n(n=2)/n_ion: {(n2s + n2p).min():.3e} to {(n2s + n2p).max():.3e};  "
        f"n(n=2) [cm^-3]: {n2.min():.3e} to {n2.max():.3e}, max at "
        f"[{','.join(str(int(k)) for k in np.unravel_index(n2.argmax(), n2.shape))}]")

    # -- kappa_0, tau, Theta, shift ----------------------------------------------
    conv_names = {"a": "(a) shell, Wiese-Fuhr f", "b": "(b) l-resolved, f from A"}
    kap = {c: {} for c in conv_names}
    for line in COMPONENTS:
        lam_a = WF[line]["lam_vac_A"] * 1e-8
        sig_a = np.array([sigma0(WF[line]["f"], lam_a, T) for T in TeL])
        kap["a"][line] = sig_a[:, None] * n2
        kb = np.zeros((nT, nN))
        for lo, _, fl, _ in comp_f[line]:
            sig_l = np.array([sigma0(fl, lam_b[line], T) for T in TeL])
            kb += sig_l[:, None] * ((n2s_cm3 if lo == "2S" else n2p_cm3))
        kap["b"][line] = kb
        say(f"    sigma_0({line}, a) at 1 eV = {sig_a[0]:.4e} cm^2;  components (b) at 1 eV: "
            + ", ".join(f"{lo}->{up} {sigma0(fl, lam_b[line], TeL[0]):.3e}" for lo, up, fl, _ in comp_f[line]))
    tau = {c: {ln: dict(full=kap[c][ln] * D, half=kap[c][ln] * D / 2) for ln in COMPONENTS} for c in conv_names}
    say("\n    escape factors by escape_factor.escape_factor_quadrature at tau_half and tau_full ...")
    theta = {c: {ln: {d: np.array([[float(escape_factor_quadrature(t)) for t in row] for row in tau[c][ln][d]])
                      for d in ("full", "half")} for ln in COMPONENTS} for c in conv_names}
    shift = {c: {d: theta[c]["Halpha"][d] / theta[c]["Hbeta"][d] - 1 for d in ("full", "half")} for c in conv_names}

    # -- RESULT 1: named points -------------------------------------------------
    say("\n" + "-" * 78)
    say(f"RESULT 1: tau(H_alpha), tau(H_beta) across D = {D:g} cm at the three named points")
    say("-" * 78)
    for name, (i, j) in NAMED:
        say(f"\n  {name} [{i},{j}]  Te = {TeL[i]:.4f} eV  ne = {neL[j]:.3e} cm^-3  u_CRE = {u[i, j]:.4e}"
            f"  n_2s/n_ion = {n2s[i, j]:.3e}  n_2p/n_ion = {n2p[i, j]:.3e}  n_2p/n_2s = {ratio_2p_2s[i, j]:.3f}"
            f"  n(n=2) = {n2[i, j]:.3e} cm^-3")
        say(f"    {'convention':26s} {'tau_full(Ha)':>13s} {'tau_half(Ha)':>13s} {'tau_full(Hb)':>13s}"
            f" {'Th_a(half)':>11s} {'Th_a(full)':>11s} {'Th_b(half)':>11s} {'Th_b(full)':>11s} {'shift(half)':>12s} {'shift(full)':>12s}")
        for c, cname in conv_names.items():
            say(f"    {cname:26s} {tau[c]['Halpha']['full'][i, j]:13.4e} {tau[c]['Halpha']['half'][i, j]:13.4e}"
                f" {tau[c]['Hbeta']['full'][i, j]:13.4e} {theta[c]['Halpha']['half'][i, j]:11.5f}"
                f" {theta[c]['Halpha']['full'][i, j]:11.5f} {theta[c]['Hbeta']['half'][i, j]:11.5f}"
                f" {theta[c]['Hbeta']['full'][i, j]:11.5f} {100 * shift[c]['half'][i, j]:+11.3f}% {100 * shift[c]['full'][i, j]:+11.3f}%")
            for ln in COMPONENTS:
                for d in ("full", "half"):
                    summary.append((f"R1 {c}", f"tau_{d}({ln}) [{i},{j}]", tau[c][ln][d][i, j], "", "computed", ""))
                    summary.append((f"R1 {c}", f"Theta_{d}({ln}) [{i},{j}]", theta[c][ln][d][i, j], "", "computed", ""))
            for d in ("full", "half"):
                summary.append((f"R1 {c}", f"ratio shift Theta_a/Theta_b - 1 ({d}) [{i},{j}]", shift[c][d][i, j], "", "computed", ""))
        say(f"    (b)/(a) for tau(H_alpha): {tau['b']['Halpha']['full'][i, j] / tau['a']['Halpha']['full'][i, j]:.4f}")

    # -- P1 -------------------------------------------------------------------------
    say("\n  P1  which convention reproduces the quoted 6.3e-7 / 1.0e-3 / 0.263 to two significant figures:")
    p1_hits = []
    for c, cname in conv_names.items():
        for d in ("full", "half"):
            oks = []
            for (i, j), q in QUOTED.items():
                v = tau[c]["Halpha"][d][i, j]
                ok = round_sig(v, 2) == round_sig(q, 2)
                oks.append(ok)
                summary.append((f"P1 {c} {d}", f"tau_{d}(H_alpha) [{i},{j}]", v, q, "chapter4 ~806-808 / D.8",
                                "REPRODUCED (2 s.f.)" if ok else "NOT REPRODUCED"))
            line = f"      {cname:26s} tau_{d}: " + "  ".join(
                f"[{i},{j}] {tau[c]['Halpha'][d][i, j]:.3e} vs {q:.2g} {'ok' if ok else 'NO'}"
                for ((i, j), q), ok in zip(QUOTED.items(), oks))
            say(line + f"   -> {'ALL THREE' if all(oks) else str(sum(oks)) + ' of 3'}")
            if all(oks):
                p1_hits.append(f"{cname} tau_{d}")
    say(f"      P1 {'as predicted' if p1_hits else 'NOT as predicted'}: reproduced by "
        f"{p1_hits if p1_hits else 'no convention'}")
    if p1_hits:
        say(f"      three-significant-figure check at [0,7] for the reproducing convention(s): "
            + ", ".join(f"{h}: {round_sig(tau['a' if h.startswith('(a)') else 'b']['Halpha'][h.split('_')[-1]][0, 7], 3):.3g} vs 0.263"
                        for h in p1_hits))
    summary.append(("P1", "conventions reproducing all three quoted tau", len(p1_hits), 1, "task", "; ".join(p1_hits) or "NONE"))

    # -- P1b ----------------------------------------------------------------------
    rba = tau["b"]["Halpha"]["full"] / tau["a"]["Halpha"]["full"]
    say(f"\n  P1b (b)/(a) for tau(H_alpha) over the grid: {rba.min():.4f} at "
        f"[{','.join(str(int(k)) for k in np.unravel_index(rba.argmin(), rba.shape))}] to {rba.max():.4f} at "
        f"[{','.join(str(int(k)) for k in np.unravel_index(rba.argmax(), rba.shape))}];"
        f"  at ne = {neL[-1]:.0e}: {rba[:, -1].min():.4f} to {rba[:, -1].max():.4f};"
        f"  identity limit f_pipe/f_WF = {f_shell_pipe['Halpha'] / WF['Halpha']['f']:.5f}")
    p1b = (rba.min() >= 0.68 - 0.01) and (rba.max() <= 1.0 + 1e-3) and (rba[:, -1].min() >= rba[:, :-1].max() - 0.05)
    say(f"      P1b {'as predicted' if p1b else 'NOT as predicted'} (range in [0.68, 1] and closest to 1 at ne = 1e15)")
    summary.append(("P1b", "(b)/(a) tau(H_alpha) min", rba.min(), 0.68, "hand estimate", "as predicted" if p1b else "NOT"))
    summary.append(("P1b", "(b)/(a) tau(H_alpha) max", rba.max(), 1.0, "hand estimate", "as predicted" if p1b else "NOT"))

    # -- P2: worst cell -------------------------------------------------------------
    say("\n  P2  worst cell for tau(H_alpha):")
    p2 = True
    worst_cells = {}
    for c, cname in conv_names.items():
        t = tau[c]["Halpha"]["full"]
        wi, wj = (int(k) for k in np.unravel_index(t.argmax(), t.shape))
        worst_cells[c] = (wi, wj)
        ok = (wi, wj) == (0, 7)
        p2 &= ok
        # second-worst for context
        flat = np.argsort(t.ravel())[::-1]
        s2i, s2j = (int(k) for k in np.unravel_index(flat[1], t.shape))
        say(f"      {cname:26s} worst [{wi},{wj}] tau_full = {t[wi, wj]:.4e}  (next [{s2i},{s2j}] {t[s2i, s2j]:.4e})"
            f"  -> {'as predicted' if ok else 'NOT as predicted'}")
        summary.append((f"P2 {c}", "worst cell tau_full(H_alpha)", f"[{wi},{wj}]", "[0,7]", "chapter4", "as predicted" if ok else "NOT"))

    # -- P3: escape factor and shift at [0,7] and at the worst cell ------------------
    say(f"\n  P3  Theta(H_alpha) >= {CH4_THETA_MIN} and |shift| <= {100 * CH4_SHIFT_MAX:.0f} % at [0,7] "
        f"(and at each convention's worst cell):")
    p3 = {}
    for c, cname in conv_names.items():
        for (i, j) in sorted({(0, 7), worst_cells[c]}):
            for d in ("half", "full"):
                th = theta[c]["Halpha"][d][i, j]; sh = shift[c][d][i, j]
                ok = (th >= CH4_THETA_MIN) and (abs(sh) <= CH4_SHIFT_MAX)
                p3[(c, d)] = ok
                say(f"      {cname:26s} [{i},{j}] tau_{d}(Ha) = {tau[c]['Halpha'][d][i, j]:.4f}  Theta_a = {th:.4f}"
                    f"  Theta_b = {theta[c]['Hbeta'][d][i, j]:.4f}  shift = {100 * sh:+.2f} %  -> "
                    f"{'holds' if ok else 'FAILS'}")
                summary.append((f"P3 {c} {d}", f"Theta(H_alpha) [{i},{j}]", th, CH4_THETA_MIN, "chapter4 ~809", "holds" if th >= CH4_THETA_MIN else "FAILS"))
                summary.append((f"P3 {c} {d}", f"ratio shift [{i},{j}]", sh, -CH4_SHIFT_MAX, "chapter4 ~810", "holds" if abs(sh) <= CH4_SHIFT_MAX else "FAILS"))
    say(f"      P3 as written in the chapter holds under: "
        + (", ".join(f"{conv_names[c]} tau_{d}" for (c, d), ok in p3.items() if ok) or "NONE"))
    say(f"      small-tau check: 1 - tau_half/sqrt2 at [0,7] (a) = {1 - tau['a']['Halpha']['half'][0, 7] / np.sqrt(2):.4f}, "
        f"quadrature {theta['a']['Halpha']['half'][0, 7]:.4f}")

    # -- P4 and census -----------------------------------------------------------------
    say("\n" + "-" * 78)
    say("RESULT 2: the whole grid -- census of tau(H_alpha) and the defended set Te >= 2 eV")
    say("-" * 78)
    defended = (TeL >= 2.0)[:, None] & np.ones((1, nN), bool)
    say(f"  defended set: {int(defended.sum())} of {nT * nN} cells (Te rows {int(np.argmax(TeL >= 2.0))}..{nT - 1})")
    p4 = True
    refuter = False
    for c, cname in conv_names.items():
        for d in ("full", "half"):
            t = tau[c]["Halpha"][d]
            n01, n1 = int((t > 0.1).sum()), int((t > 1.0).sum())
            cells01 = [f"[{a},{b}]" for a, b in zip(*np.where(t > 0.1))]
            dmax = t[defended].max()
            di, dj = (int(k) for k in np.unravel_index(np.where(defended, t, -np.inf).argmax(), t.shape))
            smax = np.abs(shift[c][d][defended]).max()
            si_, sj_ = (int(k) for k in np.unravel_index(np.where(defended, np.abs(shift[c][d]), -np.inf).argmax(), t.shape))
            say(f"  {cname:26s} tau_{d}: > 0.1 at {n01} cells {cells01 if n01 <= 12 else ''}, > 1 at {n1};"
                f"  defended max tau = {dmax:.3e} at [{di},{dj}] (Te = {TeL[di]:.3f});"
                f"  defended max |shift| = {100 * smax:.3f} % at [{si_},{sj_}]")
            summary.append((f"R2 {c} {d}", "cells with tau(H_alpha) > 0.1", n01, "", "computed", ""))
            summary.append((f"R2 {c} {d}", "cells with tau(H_alpha) > 1", n1, "", "computed", ""))
            summary.append((f"R2 {c} {d}", "defended max tau(H_alpha)", dmax, 0.1, "refuter", "below" if dmax <= 0.1 else "REFUTER FIRED"))
            summary.append((f"R2 {c} {d}", "defended max |ratio shift|", smax, 0.1, "refuter", "below" if smax <= 0.1 else "REFUTER FIRED"))
            if d == "full":
                p4 &= dmax <= 0.1
                if n01 > 0:
                    hot = TeL[np.where(t > 0.1)[0]].max(); thin = neL[np.where(t > 0.1)[1]].min()
                    say(f"      cells above 0.1 lie at Te <= {hot:.3f} eV and ne >= {thin:.2e} cm^-3")
                    p4 &= (hot < 1.3) and (thin >= 3.7e14) and (n1 == 0)
            refuter |= (dmax > 0.1) or (smax > 0.1)
    say(f"  P4 {'as predicted' if p4 else 'NOT as predicted'} (no defended cell with tau_full > 0.1; > 0.1 only at Te < 1.3 eV, "
        f"ne >= 3.7e14; none > 1)")
    say(f"  REFUTER (defended cell with tau_full(H_alpha) > 0.1, or |shift| > 10 % on the defended set, either convention, "
        f"either tau): {'FIRED' if refuter else 'did not appear'}")
    say(f"  largest defended-set escape-factor reduction 1 - Theta_half(H_alpha): "
        + ", ".join(f"{conv_names[c]} {1 - theta[c]['Halpha']['half'][defended].min():.2e}" for c in conv_names))
    tb = tau["a"]["Hbeta"]["full"]
    say(f"  H_beta for reference, (a) tau_full: grid max {tb.max():.4e} at "
        f"[{','.join(str(int(k)) for k in np.unravel_index(tb.argmax(), tb.shape))}], defended max {tb[defended].max():.3e}")
    say(f"  benchmark [23,5]: (a) tau_full(Ha) = {tau['a']['Halpha']['full'][23, 5]:.3e}, (b) {tau['b']['Halpha']['full'][23, 5]:.3e},"
        f" shift(half, a) = {100 * shift['a']['half'][23, 5]:+.2e} %")
    summary.append(("bench", "tau_full(H_alpha) [23,5] (a)", tau["a"]["Halpha"]["full"][23, 5], "", "computed", ""))

    # -- RESULT 3: sensitivity to definitions at the named points (no prediction) ----
    # Added after the first run (21 Sep 2026) showed P1 failing under every
    # convention, to locate the discrepancy; the predictions above are unchanged.
    say("\n" + "-" * 78)
    say("RESULT 3 (no prediction; block added after the first run showed P1 failing): the D.8 numbers")
    say("against this run, and the factor each alternative convention applies to tau_full(H_alpha) under (a)")
    say("-" * 78)
    M_D = 3.34358e-24   # deuteron mass, g (as in verify_lyman_optical_depth.py)
    fa, la = WF["Halpha"]["f"], WF["Halpha"]["lam_vac_A"] * 1e-8
    alts = {}
    for name, (i, j) in NAMED:
        s0 = sigma0(fa, la, TeL[i])
        alts[(i, j)] = {
            "sqrt(kT/m) width": sigma0(fa, la, TeL[i], width_factor=1.0) / s0,
            "deuterium mass": sigma0(fa, la, TeL[i], mass_g=M_D) / s0,
            "T_n = 3 eV": sigma0(fa, la, 3.0) / s0,
            "pipeline f, not WF": f_shell_pipe["Halpha"] / fa,
            "(b) l-resolved": rba[i, j],
            "n2 = 4 n_2s": 4 * n2s[i, j] / (n2s[i, j] + n2p[i, j]),
            "n2 = (4/3) n_2p": (4.0 / 3.0) * n2p[i, j] / (n2s[i, j] + n2p[i, j]),
            "tau_half": 0.5,
        }
    keys = list(next(iter(alts.values())).keys())
    say(f"    {'point':12s} {'run (a) full':>13s} {'D.8':>9s} {'D.8/run':>8s}  " + "  ".join(f"{k:>18s}" for k in keys))
    for name, (i, j) in NAMED:
        q = QUOTED[(i, j)]; base = tau["a"]["Halpha"]["full"][i, j]
        say(f"    [{i},{j}]{'':7s} {base:13.4e} {q:9.3g} {q / base:8.4f}  " + "  ".join(f"{alts[(i, j)][k]:18.4f}" for k in keys))
        summary.append(("R3", f"D.8 / run tau_full(H_alpha) (a) [{i},{j}]", q / base, "", "diagnostic", ""))
    explained = [k for k in keys if all(abs(alts[ij][k] / (QUOTED[ij] / tau["a"]["Halpha"]["full"][ij]) - 1) <= 0.02 for _, ij in NAMED)]
    say(f"    single alternative reproducing D.8/run within 2 % at all three points: {explained if explained else 'NONE'}")
    say(f"    D.8/run is not a constant factor ({min(QUOTED[ij] / tau['a']['Halpha']['full'][ij] for _, ij in NAMED):.3f} to "
        f"{max(QUOTED[ij] / tau['a']['Halpha']['full'][ij] for _, ij in NAMED):.3f}), so it is not a cross-section convention alone")
    summary.append(("R3", "alternatives reproducing D.8/run at all three points", len(explained), 0, "diagnostic", "; ".join(explained) or "NONE"))

    # -- reading --------------------------------------------------------------------
    say("\nREADING: the chapter-4 numbers are to be requoted from this run with the convention named")
    say("(absorber population, oscillator strength and source, slab definition for tau and for Theta).")
    say("Assumptions: thin CRE populations (no Lyman trapping feedback on n=2), T_n = Te, hydrogen mass,")
    say("n_ion = n_e, Doppler profile only, components of one line summed at a single line centre (upper bound).")

    # -- write ----------------------------------------------------------------------
    if args.write:
        out = Path(args.out) if args.out else root / "validation/balmer_optical_depth"
        out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}",
               f"# interpreter {sys.executable}  numpy {np.__version__}",
               f"# L_grid.npy      sha256 {sha256_file(L_path)}",
               f"# S_grid.npy      sha256 {sha256_file(S_path)}",
               f"# state_index.csv sha256 {sha256_file(ctx.state_index_path)}  ({ctx.state_index_path.relative_to(root)})",
               f"# A_resolved.npy  sha256 {sha256_file(A_path)}",
               f"# H_A_E1_LS_n1_15_physical.csv sha256 {sha256_file(rad_csv)}  (G3)",
               f"# molecular_channel.csv sha256 {sha256_file(mol_path)}  (G2 gate)",
               f"# wiese_fuhr.pdf  sha256 {sha256_file(pdf_path) if pdf_path.is_file() else 'MISSING'}  ({wf_source})",
               f"# D = {D:g} cm; tau_full = kappa_0 D; tau_half = kappa_0 D/2; T_n = Te; m_H = {M_H:.5e} g; n_ion = n_e",
               f"# (a) n(n=2) = n_2s + n_2p, f = {WF['Halpha']['f']} (2-3), {WF['Hbeta']['f']} (2-4), lambda_vac = "
               f"{WF['Halpha']['lam_vac_A']}, {WF['Hbeta']['lam_vac_A']} A (Wiese and Fuhr 2009 Table 4 rows 40, 41)",
               f"# (b) per component f_lu = A_ul (g_u/g_l) m_e c lambda^2/(8 pi^2 e^2) from A_resolved.npy, lambda from state_index I_eV",
               f"# Theta = escape_factor.escape_factor_quadrature(tau); shift = Theta(H_alpha)/Theta(H_beta) - 1",
               f"# benchmark point [{ti},{ni}]  Te={TeL[ti]:.4f} eV  ne={neL[ni]:.4e} cm^-3"]
        rows = []
        for i in range(nT):
            for j in range(nN):
                r = dict(i=i, j=j, Te=TeL[i], ne=neL[j], u_CRE=u[i, j], n2s_per_ion=n2s[i, j], n2p_per_ion=n2p[i, j],
                         n2_cm3=n2[i, j], n2p_over_n2s=ratio_2p_2s[i, j], defended=bool(defended[i, j]))
                for c in conv_names:
                    for ln in COMPONENTS:
                        r[f"kappa0_{ln}_{c}"] = kap[c][ln][i, j]
                        for d in ("half", "full"):
                            r[f"tau_{d}_{ln}_{c}"] = tau[c][ln][d][i, j]
                            r[f"Theta_{d}_{ln}_{c}"] = theta[c][ln][d][i, j]
                    for d in ("half", "full"):
                        r[f"shift_{d}_{c}"] = shift[c][d][i, j]
                r["tau_b_over_a_Halpha"] = rba[i, j]
                rows.append(r)
        with open(out / "balmer_optical_depth.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n"); pd.DataFrame(rows).to_csv(fh, index=False)
        with open(out / "balmer_optical_depth_summary.csv", "w") as fh:
            fh.write("\n".join(hdr) + "\n")
            pd.DataFrame(summary, columns=["item", "quantity", "value", "recorded", "source", "status"]).to_csv(fh, index=False)
        with open(out / "balmer_optical_depth.txt", "w") as fh:
            fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(root)}/balmer_optical_depth{{.csv,_summary.csv,.txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
