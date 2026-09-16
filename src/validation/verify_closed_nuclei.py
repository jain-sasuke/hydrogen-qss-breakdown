"""
verify_closed_nuclei.py
=======================
Self-consistent closed-nuclei electron density across one temperature step,
against the fixed-n_e inference used in chapter 5, section 5.10.1.

WHAT THE CHAPTER DOES
---------------------
Section 5.10.1 (thesis_tex/chapter5.tex, table tab:quasineutrality) infers
the electron-density change a closed hydrogen parcel would need across a
+4.81 percent Te step from two FIXED-n_e collisional-radiative-equilibrium
(CRE) solves:

    Delta n_e / n_e  :=  | u_new - u_old |,     u = n_g / n_ion,

with u_old from L(Te_i, ne_j) and u_new from L(Te_k, ne_j) at the SAME n_e
(src/analysis/make_story_figures.py, Step.dne_over_ne_ground and
.dne_over_ne_total).  Every neutral that ionises delivers one electron, so
the change in the neutral population per ion is read as the change in n_e
per ion.  That is not a closed-nuclei solution, for two separate reasons:

  (a) bookkeeping: if u falls from 28 to 14 per ion, the ion count itself
      roughly doubles, so the neutrals released per NEW ion is not 14 -- the
      conserved quantity is n_e (1 + sum_states b), and solving that for the
      new n_e with the fixed-n_e populations already gives a factor ~2, not
      ~15;
  (b) rate feedback: a changed n_e changes every collisional rate in L and
      S, so the post-step populations per ion are not those of the
      fixed-n_e solve.

WHAT THIS SCRIPT DOES
---------------------
Hydrogen only.  Quasineutrality n_e = n_i.  Nuclei conservation
n_g + n_i + (all excited) = const.

  1. Establish that at fixed Te the matrix L is EXACTLY linear in n_e and
     the source vector S is EXACTLY quadratic in n_e across the 8 density
     columns (two-point fit through columns 0 and 7, residual on all 8
     columns; three-point fit through columns 0, 7 and the column nearest
     the log-midpoint density, to show the quadratic coefficient of L and
     the constant term of S vanish).  Tolerance 1e-12
     relative; the script RAISES if either fails, because everything after
     it evaluates L and S at off-grid n_e by these forms.  The structural
     reason is that every collisional rate in L is n_e times a Te-only
     coefficient, radiative decay is n_e-independent, and S (per ion) is
     radiative recombination (n_e) plus three-body recombination (n_e^2).

  2. For each Te row i, density column j and both directions (heat: k is
     the grid index nearest Te_i(1+frac), exactly as verify_divertor_map.py
     snaps it; cool: nearest Te_i(1-frac)):
       pre-step CRE at (Te_i, ne_j) with n_ion = ne_j:
           b_old = solve(L, -S)       (43 states per ion, ground included)
           N     = ne_j (1 + sum b_old)              total nuclei
       post-step: find ne+ by brentq such that
           ne+ (1 + sum b(Te_k, ne+)) = N
       and report ne+/ne_j - 1 against the chapter's |u_new - u_old|.

  3. At (Te_k, ne+) versus (Te_k, ne_j): tau_slow = 1/|least-negative
     eigenvalue|, and eps_plateau by the partial-equilibrium construction
     of verify_divertor_map.py (excited manifold in equilibrium with the
     FROZEN pre-step ground density on the post-step operator, compared
     with the post-step CRE shell ratio n3/n4).

     On which eps_plateau is "self-consistent": the plateau itself is
     already self-consistent under closed nuclei, because during the
     plateau neither n_g nor n_e has moved -- the operator at (Te_k, ne_j)
     with the frozen n_g IS the closed parcel at that stage.  What the
     closed-nuclei solution changes is the ENDPOINT the plateau is compared
     with: the parcel relaxes to CRE at (Te_k, ne+), not (Te_k, ne_j).  So
     three numbers are reported per pair:
         eps_plateau_fixed      plateau(k, ne_j) vs CRE(k, ne_j)   [chapter]
         eps_plateau_endpoint   plateau(k, ne_j) vs CRE(k, ne+)    [closed parcel]
         eps_plateau_at_neplus  plateau(k, ne+)  vs CRE(k, ne+)    [table read at ne+]
     the last with the frozen ABSOLUTE ground density expressed per new ion.

PREDICTION (written before the run)
-----------------------------------
  P1  Above Te = 2 eV (the 448 window_ok pairs with Te >= 2 in
      validation/divertor_map/divertor_map.csv) the self-consistent
      correction |ne+/ne_j - 1| is at most 5.3e-3, and the relative change
      in eps_plateau (either closed-nuclei reading against the chapter's)
      is under 0.2 percent.
  P2  At Te = 1 eV the self-consistent rise in n_e is a factor of about 2
      (ne+/ne_j - 1 of order 1), not the factor 15 to 21 the fixed-n_e
      inference gives (13.9 at [0,4], 20.0 at [0,0]).
  P3  The chapter's fixed-n_e numbers reproduce from this script's own
      fixed-n_e solves: 13.9 at [0,4]; max 20.0 at 1 eV; max 5.4e-3 for
      Te >= 2 (at 2.02 eV); first below 1e-3 above 2.56 eV; 68 of the 392
      heating pairs above 10 percent and 36 above 100 percent; 10 percent
      contour at Te = 1.41 to 1.50 eV.
  Scratch values to reproduce (cr-physicist, check2.py):
      heat [15,0]: inferred 5.40e-3, self-consistent 5.34e-3
      heat [15,3]: 4.32e-3 / 4.28e-3
      heat [23,5]: 2.08e-4 / 2.09e-4
      heat [0,4] : 13.9 / 1.05 (n_e doubles; Delta ln n_g = -0.038;
                   tau_slow 233 ms -> 86 ms)

REFUTING OBSERVATIONS
---------------------
  P1 is refuted by any warm pair with |ne+/ne_j - 1| > 5.35e-3, or with a
     relative change in eps_plateau above 0.2 percent.
  P2 is refuted if at Te = 1 eV any column gives ne+/ne_j - 1 > 5 (closer
     to the inferred 14-20 than to a doubling), or < 0.3.
  P3 is refuted if any listed chapter number disagrees beyond its quoted
     digits; that is reported as NOT REPRODUCED, never raised, because the
     disagreement is the finding.
  The linearity assertion (step 1) is the one hard gate: if it fails the
  off-grid evaluation is not exact and no later number may be used.

SCOPE / CAVEATS
---------------
  - Isothermal atoms and ions; Te steps, T_i and T_atom do not enter.
  - The parcel is closed: no transport, no molecular source.  The chapter
    argues that a closed parcel and a fixed-n_e parcel cannot both hold;
    this script supplies the closed side of that pair properly.
  - ne+ can leave the grid range [ne_0, ne_7] (it does at 1 eV).  The
    evaluation there is exact for the model as built (step 1 shows the
    dependence is structural, not fitted), but the underlying rate
    coefficients were only ever tabulated on the grid; flagged per pair.
  - Observable is the n=3/n=4 shell population ratio, as in
    verify_divertor_map.py.

Report only.  Writes to validation/closed_nuclei/ with --write; modifies
nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_context import CRContext  # noqa: E402

LIN_TOL = 1e-12          # relative residual for exact linearity / quadraticity
ROOT_TOL = 1e-12         # relative residual of nuclei balance at the root
CSV_MATCH_TOL = 1e-9     # agreement with divertor_map.csv values

# Chapter 5 section 5.10.1 / make_story_figures.py recorded values (fixed n_e)
REC = dict(
    qn_0_4=13.9, qn_row0_max=20.0, qn_warm_max=5.4e-3, qn_warm_max_Te=2.02,
    qn_first_below_1e3_Te=2.56, n_over_10=68, n_over_100=36, n_total=392,
    contour_lo=1.41, contour_hi=1.50,
)
# cr-physicist scratch run (check2.py), heat direction, [i, j]
SCRATCH = {(15, 0): (5.40e-3, 5.34e-3), (15, 3): (4.32e-3, 4.28e-3),
           (23, 5): (2.08e-4, 2.09e-4), (0, 4): (13.9, 1.05)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frac", type=float, default=0.05,
                   help="fractional Te step, snapped to the grid as in "
                        "verify_divertor_map.py (default 0.05)")
    p.add_argument("--win-lo", type=float, default=30.0)
    p.add_argument("--win-hi", type=float, default=30.0)
    p.add_argument("--write", action="store_true",
                   help="write csv/txt to the output directory")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--divertor-map", type=Path, default=None,
                   help="divertor_map.csv defining the warm/window_ok set")
    return p.parse_args()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_commented_csv(path: Path):
    import csv as _csv
    hdr = []
    with open(path) as fh:
        body = []
        for line in fh:
            if line.startswith("#"):
                hdr.append(line.rstrip("\n"))
            else:
                body.append(line)
    rows = list(_csv.DictReader(body))
    return hdr, rows


class DensityForms:
    """Exact n_e dependence of L and S at each Te row, established from the
    grid and asserted, then used to evaluate at arbitrary n_e."""

    def __init__(self, L, S, ne, say):
        nT, nN = L.shape[:2]
        j0, j1 = 0, nN - 1
        jm = int(np.argmin(np.abs(np.log(ne) - 0.5 * (np.log(ne[j0]) + np.log(ne[j1])))))
        self.A = np.empty_like(L[:, 0])
        self.K = np.empty_like(L[:, 0])
        self.sa = np.empty_like(S[:, 0])
        self.sb = np.empty_like(S[:, 0])
        worst = dict(L2=0.0, S2=0.0, L3q=0.0, S3c=0.0, L3=0.0, S3=0.0)
        x = ne / ne[j1]                       # scaled density for conditioning
        for i in range(nT):
            # --- two-point forms (used for evaluation) ----------------------
            K = (L[i, j1] - L[i, j0]) / (ne[j1] - ne[j0])
            A = L[i, j0] - ne[j0] * K
            M2 = np.array([[ne[j0], ne[j0] ** 2], [ne[j1], ne[j1] ** 2]])
            ab = np.linalg.solve(M2, np.vstack([S[i, j0], S[i, j1]]))
            self.A[i], self.K[i], self.sa[i], self.sb[i] = A, K, ab[0], ab[1]
            rL = max(np.abs(A + ne[j] * K - L[i, j]).max() / np.abs(L[i, j]).max()
                     for j in range(nN))
            rS = max(np.abs(ne[j] * ab[0] + ne[j] ** 2 * ab[1] - S[i, j]).max()
                     / np.abs(S[i, j]).max() for j in range(nN))
            # --- three-point general quadratics (structure check) -----------
            V = np.array([[1.0, x[j], x[j] ** 2] for j in (j0, jm, j1)])
            cL = np.linalg.solve(V, np.stack([L[i, j0], L[i, jm], L[i, j1]])
                                 .reshape(3, -1))
            cS = np.linalg.solve(V, np.stack([S[i, j0], S[i, jm], S[i, j1]]))
            scaleL = max(np.abs(cL[0]).max(), np.abs(cL[1]).max())
            scaleS = max(np.abs(cS[1]).max(), np.abs(cS[2]).max())
            q_rel = np.abs(cL[2]).max() / scaleL          # quadratic coeff of L
            c_rel = np.abs(cS[0]).max() / scaleS          # constant term of S
            rL3 = max(np.abs((cL[0] + x[j] * cL[1] + x[j] ** 2 * cL[2])
                             .reshape(L.shape[2:]) - L[i, j]).max()
                      / np.abs(L[i, j]).max() for j in range(nN))
            rS3 = max(np.abs(cS[0] + x[j] * cS[1] + x[j] ** 2 * cS[2] - S[i, j]).max()
                      / np.abs(S[i, j]).max() for j in range(nN))
            for key, val in (("L2", rL), ("S2", rS), ("L3q", q_rel),
                             ("S3c", c_rel), ("L3", rL3), ("S3", rS3)):
                worst[key] = max(worst[key], float(val))
        self.worst = worst
        say("n_e-DEPENDENCE OF THE OPERATOR (all Te rows, all 8 columns)")
        say(f"  two-point fit cols {j0},{j1}; three-point fit cols {j0},{jm},{j1}")
        say(f"  L = A + ne K       max rel residual on 8 cols   {worst['L2']:.2e}")
        say(f"  S = a ne + b ne^2  max rel residual on 8 cols   {worst['S2']:.2e}")
        say(f"  3-pt quadratic through L: |quad coeff|/|lin,const| {worst['L3q']:.2e}"
            f"   residual {worst['L3']:.2e}")
        say(f"  3-pt quadratic through S: |const|/|lin,quad|       {worst['S3c']:.2e}"
            f"   residual {worst['S3']:.2e}")
        bad = [k for k in ("L2", "S2", "L3", "S3") if worst[k] > LIN_TOL]
        if bad:
            raise RuntimeError(
                f"L is not exactly linear / S not exactly quadratic in n_e: "
                f"residuals {[(k, worst[k]) for k in bad]} exceed {LIN_TOL:g}. "
                f"Off-grid evaluation would not be exact; nothing below may run.")
        say(f"  ASSERTED: all residuals < {LIN_TOL:g}  (L linear, S quadratic in n_e)")

    def L(self, i, n):
        return self.A[i] + n * self.K[i]

    def S(self, i, n):
        return n * self.sa[i] + n * n * self.sb[i]


def main():
    a = parse_args()
    ctx = CRContext.load()
    root = ctx.root
    L_path = root / "data/processed/cr_matrix/L_grid.npy"
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    if not S_path.exists():
        raise FileNotFoundError(f"missing source vector: {S_path}")
    L, S = ctx.L_grid, np.load(S_path)
    Te, ne = ctx.te_grid, ctx.ne_grid
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")
    nT, nN, nS = L.shape[:3]

    g = int(ctx.ground_index)
    E = np.array([s for s in range(nS) if s != g], dtype=int)
    N3 = np.where(np.asarray(ctx.n_values) == 3)[0]
    N4 = np.where(np.asarray(ctx.n_values) == 4)[0]
    if len(N3) != 3 or len(N4) != 4:
        raise ValueError(f"shell membership wrong: n=3 -> {N3}, n=4 -> {N4}")
    pos = {s: q for q, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3])
    n4E = np.array([pos[s] for s in N4])

    dm_path = a.divertor_map or (root / "validation/divertor_map/divertor_map.csv")
    if not dm_path.exists():
        raise FileNotFoundError(
            f"{dm_path} not found; the warm/window_ok set is defined by it. "
            f"Run verify_divertor_map.py first.")
    dm_hdr, dm_rows = read_commented_csv(dm_path)
    dm = {(r["direction"], int(r["i"]), int(r["j"])): r for r in dm_rows}
    dm_L_sha = next((h.split()[-1] for h in dm_hdr if "L_grid sha256" in h), None)

    out = a.out or (root / "validation" / "closed_nuclei")
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    shaL, shaS, shaI = sha256(L_path), sha256(S_path), sha256(ctx.state_index_path)
    prov = [
        f"# script       {Path(__file__).resolve()}",
        f"# generated    {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"# interpreter  {sys.executable}  python {sys.version.split()[0]}  "
        f"numpy {np.__version__}  scipy {scipy.__version__}",
        f"# L_grid       {L_path}  sha256 {shaL}",
        f"# S_grid       {S_path}  sha256 {shaS}",
        f"# state_index  {ctx.state_index_path}  sha256 {shaI}",
        f"# divertor_map {dm_path}",
        f"# frac {a.frac}  win_lo {a.win_lo}  win_hi {a.win_hi}",
    ]
    say("=" * 78)
    say("CLOSED-NUCLEI SELF-CONSISTENT n_e ACROSS ONE Te STEP")
    for p in prov:
        say(p[2:])
    say(ctx.describe())
    say(f"ground index {g} ({ctx.labels[g]}); n=3 {[ctx.labels[s] for s in N3]}; "
        f"n=4 {[ctx.labels[s] for s in N4]}")
    if dm_L_sha != shaL:
        say(f"*** WARNING: divertor_map.csv was generated from L_grid sha {dm_L_sha}, "
            f"current L_grid sha is {shaL}; cross-checks against it may fail ***")
    else:
        say("divertor_map.csv L_grid sha256 matches the current L_grid")
    say("=" * 78)

    forms = DensityForms(L, S, ne, say)

    # ---------------------------------------------------------------- helpers
    def cre(i, n):
        b = np.linalg.solve(forms.L(i, n), -forms.S(i, n))
        if b.min() < -1e-12 * b.max():
            raise RuntimeError(
                f"negative CRE population at Te[{i}]={Te[i]:.4g} eV, "
                f"ne={n:.4g}: min {b.min():.3e} (max {b.max():.3e}); the solve "
                f"is not physical and nothing built on it may be used")
        return b

    def spectrum(i, n):
        lam = np.linalg.eigvals(forms.L(i, n))
        lam = lam[np.argsort(lam.real)[::-1]]
        if lam[0].real >= 0 or lam[1].real >= 0:
            raise RuntimeError(f"unstable operator at Te[{i}], ne={n:.4g}")
        return -1.0 / lam[0].real, -1.0 / lam[1].real

    def plateau_ratio(k, n_op, ng_per_ion):
        """n3/n4 with the excited manifold in equilibrium with a frozen
        ground density (per ion of the operator's n_e) on L(Te_k, n_op)."""
        A = forms.L(k, n_op)
        s = forms.S(k, n_op)
        LEE = A[np.ix_(E, E)]
        LEg = A[np.ix_(E, [g])].ravel()
        n0 = np.linalg.solve(LEE, -s[E])
        n1 = np.linalg.solve(LEE, -LEg * ng_per_ion)
        nE = n0 + n1
        return nE[n3E].sum() / nE[n4E].sum(), n0, n1

    def shell_ratio(b):
        return b[N3].sum() / b[N4].sum()

    def tau_schur(A):
        """1 / (effective ground-state loss rate with the excited manifold
        slaved): a well-conditioned linear solve, used only to size the
        conditioning of the least-negative eigenvalue."""
        LEE = A[np.ix_(E, E)]
        aa = np.linalg.solve(LEE, -A[np.ix_(E, [g])].ravel())
        return -1.0 / (A[g, g] + A[g, E] @ aa)

    def tau_direct(k, j):
        lam = np.linalg.eigvals(L[k, j])
        return -1.0 / np.sort(lam.real)[::-1][0]

    def solve_closed(i, k, n_old):
        b_old = cre(i, n_old)
        Ntot = n_old * (1.0 + b_old.sum())

        def f(n):
            return n * (1.0 + cre(k, n).sum()) - Ntot

        lo, hi = n_old / 2.0, n_old * 2.0
        flo, fhi = f(lo), f(hi)
        it = 0
        while flo > 0 and lo > n_old * 1e-4:
            lo /= 2.0
            flo = f(lo)
            it += 1
        while fhi < 0 and hi < n_old * 1e4:
            hi *= 2.0
            fhi = f(hi)
            it += 1
        if not (flo < 0 < fhi):
            raise RuntimeError(
                f"could not bracket the closed-nuclei root at [{i}->{k}], "
                f"ne={n_old:.4g}: f({lo:.3g})={flo:.3g}, f({hi:.3g})={fhi:.3g}")
        n_new = brentq(f, lo, hi, xtol=1e-15 * n_old, rtol=4 * np.finfo(float).eps,
                       maxiter=500)
        resid = abs(f(n_new)) / Ntot
        if resid > ROOT_TOL:
            raise RuntimeError(f"root residual {resid:.2e} > {ROOT_TOL:g} at [{i}->{k}]")
        return b_old, Ntot, n_new, resid

    # ------------------------------------------------------------- main loop
    rows = []
    for sgn, dlab in ((+1, "heat"), (-1, "cool")):
        for i in range(nT):
            k = int(np.argmin(np.abs(Te - Te[i] * (1 + sgn * a.frac))))
            if k == i:
                continue
            for j in range(nN):
                n_old = float(ne[j])
                b_old, Ntot, n_new, resid = solve_closed(i, k, n_old)
                b_fix = cre(k, n_old)
                b_sc = cre(k, n_new)
                u_old, u_fix, u_sc = b_old[g], b_fix[g], b_sc[g]

                tQ_fix, tR_fix = spectrum(k, n_old)
                tQ_sc, tR_sc = spectrum(k, n_new)
                win = (a.win_lo * tR_fix) < (tQ_fix / a.win_hi)

                # plateau at (Te_k, ne_j) with frozen pre-step n_g (chapter)
                R_pe_fix, n0, n1 = plateau_ratio(k, n_old, u_old)
                x_fix = u_fix / u_old
                sup = (np.abs(n0 + x_fix * n1 - b_fix[E]).max()
                       / np.abs(b_fix[E]).max())
                if sup > 1e-8:
                    raise RuntimeError(f"two-channel superposition fails at "
                                       f"[{i},{j}]: {sup:.3e}")
                # plateau on the operator at ne+, same absolute frozen n_g
                R_pe_plus, _, _ = plateau_ratio(k, n_new, u_old * n_old / n_new)
                R_cre_fix = shell_ratio(b_fix)
                R_cre_sc = shell_ratio(b_sc)
                eps_fix = abs(R_pe_fix / R_cre_fix - 1.0)
                eps_end = abs(R_pe_fix / R_cre_sc - 1.0)
                eps_plus = abs(R_pe_plus / R_cre_sc - 1.0)

                dm_r = dm.get((dlab, i, j))
                if dm_r is None:
                    raise RuntimeError(f"({dlab},{i},{j}) missing from {dm_path}")
                dm_tQ = float(dm_r["tau_QSS"])
                dm_eps = float(dm_r["eps_plateau"])
                dm_win = dm_r["window_ok"] == "True"

                rows.append(dict(
                    direction=dlab, i=i, j=j, k=k, Te_pre=Te[i], Te_post=Te[k],
                    frac_achieved=Te[k] / Te[i] - 1.0, ne_pre=n_old,
                    u_old=u_old, u_new_fixed=u_fix, u_new_sc=u_sc,
                    sumb_old=b_old.sum(), sumb_new_fixed=b_fix.sum(),
                    sumb_new_sc=b_sc.sum(), nuclei_per_ne_pre=Ntot / n_old,
                    dne_inferred_ground=abs(u_fix - u_old),
                    dne_inferred_total=abs(b_fix.sum() - b_old.sum()),
                    dne_bookkeeping=(1 + b_old.sum()) / (1 + b_fix.sum()) - 1.0,
                    dne_sc=n_new / n_old - 1.0, ne_post_sc=n_new,
                    ne_post_off_grid=bool(n_new < ne[0] or n_new > ne[-1]),
                    root_resid_rel=resid,
                    ln_x_fixed=np.log(x_fix),
                    dln_ng_closed=np.log(u_sc * n_new / (u_old * n_old)),
                    tau_slow_fixed=tQ_fix, tau_slow_sc=tQ_sc,
                    tau_slow_rel_change=tQ_sc / tQ_fix - 1.0,
                    tau_relax_fixed=tR_fix, tau_relax_sc=tR_sc,
                    M_fixed=tQ_fix / tR_fix, window_ok=bool(win),
                    dm_window_ok=dm_win,
                    eps_plateau_fixed=eps_fix, eps_plateau_endpoint=eps_end,
                    eps_plateau_at_neplus=eps_plus,
                    rel_change_eps_endpoint=eps_end / eps_fix - 1.0,
                    rel_change_eps_at_neplus=eps_plus / eps_fix - 1.0,
                    superposition_err=sup,
                    dm_tau_QSS_reldiff=abs(tQ_fix / dm_tQ - 1.0),
                    dm_tau_QSS_reldiff_gridmatrix=abs(tau_direct(k, j) / dm_tQ - 1.0),
                    tau_slow_schur_fixed=tau_schur(forms.L(k, n_old)),
                    tau_slow_schur_sc=tau_schur(forms.L(k, n_new)),
                    dm_eps_plateau_reldiff=abs(eps_fix / dm_eps - 1.0),
                ))

    A = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    heat = A["direction"] == "heat"
    cool = ~heat
    ok = A["window_ok"].astype(bool)
    warm = ok & (A["Te_pre"] >= 2.0)
    summary = []

    def rec(key, val, comment=""):
        summary.append((key, val, comment))

    say()
    say(f"evaluated {len(rows)} (point, direction) pairs: {heat.sum()} heat, "
        f"{cool.sum()} cool; window_ok {ok.sum()}; warm (window_ok & Te>=2) "
        f"{warm.sum()}")
    rec("n_pairs", len(rows)); rec("n_heat", int(heat.sum()))
    rec("n_window_ok", int(ok.sum())); rec("n_warm", int(warm.sum()),
                                             "window_ok and Te_pre >= 2 eV")
    say(f"achieved fractional step {A['frac_achieved'].min():+.4f} .. "
        f"{A['frac_achieved'].max():+.4f}")
    say(f"max root residual (nuclei balance)      {A['root_resid_rel'].max():.2e}")
    say(f"max two-channel superposition error     {A['superposition_err'].max():.2e}")
    rec("max_root_resid_rel", float(A["root_resid_rel"].max()))

    # ---- cross-checks against divertor_map.csv -----------------------------
    say()
    say("CROSS-CHECK against divertor_map.csv (fixed-n_e quantities)")
    dtq = A["dm_tau_QSS_reldiff"].max()
    dep = A["dm_eps_plateau_reldiff"].max()
    nwin = int((A["window_ok"] != A["dm_window_ok"]).sum())
    say(f"  tau_QSS(k, ne_j)     max rel diff {dtq:.2e}  "
        f"{'REPRODUCED' if dtq < CSV_MATCH_TOL else 'NOT REPRODUCED'}"
        f"   (eig of the two-point form L(k, ne_j))")
    dtg = A["dm_tau_QSS_reldiff_gridmatrix"].max()
    say(f"  tau_QSS(k, ne_j)     max rel diff {dtg:.2e}  "
        f"{'REPRODUCED' if dtg < CSV_MATCH_TOL else 'NOT REPRODUCED'}"
        f"   (eig of the grid matrix L[k,j] itself)")
    cond_e = np.abs(A["tau_slow_fixed"] / A["tau_slow_schur_fixed"] - 1.0)
    qc = int(np.argmax(A["dm_tau_QSS_reldiff"]))
    say(f"  the two-point form reproduces L[k,j] to {forms.worst['L2']:.1e}; the eigenvalue "
        f"nevertheless moves by up to {dtq:.1e}")
    say(f"  because the least-negative eigenvalue of L is ill-conditioned where M is huge: "
        f"worst at {A['direction'][qc]} [{A['i'][qc]},{A['j'][qc]}] (Te_post="
        f"{A['Te_post'][qc]:.3f}, M={A['M_fixed'][qc]:.2e}).")
    # the Schur rate differs from lambda_0 analytically by O(1/M), so only
    # pairs with M > 1e7 (O(1/M) < 1e-7) isolate the eigenvalue conditioning
    big = A["M_fixed"] > 1e7
    say(f"  eig vs Schur effective ground-loss rate (well-conditioned solve; differs from "
        f"lambda_0 by O(1/M) analytically), restricted to the {int(big.sum())} pairs with "
        f"M > 1e7: max rel diff {cond_e[big].max():.2e}  (all pairs: {cond_e.max():.2e}, "
        f"dominated by the O(1/M) term at warm points)")
    say(f"  -> divertor_map.csv tau_QSS carries a ~{dtq:.0e} relative numerical uncertainty "
        f"in the cold corner; no ratio reported here depends on it at that level")
    rec("dm_tau_QSS_max_reldiff_gridmatrix", float(dtg))
    rec("max_reldiff_tau_eig_vs_schur_Mgt1e7", float(cond_e[big].max()))
    say(f"  eps_plateau(k, ne_j) max rel diff {dep:.2e}  "
        f"{'REPRODUCED' if dep < CSV_MATCH_TOL else 'NOT REPRODUCED'}")
    say(f"  window_ok flags disagree on {nwin} pairs")
    rec("dm_tau_QSS_max_reldiff", float(dtq)); rec("dm_eps_plateau_max_reldiff", float(dep))
    rec("dm_window_ok_mismatches", nwin)
    if warm.sum() != 448:
        say(f"*** warm-pair count {warm.sum()} differs from the expected 448 ***")

    # ---- P3: chapter's fixed-n_e numbers -----------------------------------
    say()
    say("P3  CHAPTER 5.10.1 FIXED-n_e NUMBERS, RECOMPUTED (heat, k = i+1)")
    qn = np.full((nT, nN), np.nan)
    qn_tot = np.full((nT, nN), np.nan)
    sc = np.full((nT, nN), np.nan)
    for r in rows:
        if r["direction"] == "heat":
            if r["k"] != r["i"] + 1:
                raise RuntimeError(f"heat snap at i={r['i']} landed on k={r['k']}, "
                                   f"not i+1; the 392-pair census does not apply")
            qn[r["i"], r["j"]] = r["dne_inferred_ground"]
            qn_tot[r["i"], r["j"]] = r["dne_inferred_total"]
            sc[r["i"], r["j"]] = abs(r["dne_sc"])
    fin = np.isfinite(qn)

    def crossing(field, level):
        outc = np.full(nN, np.nan)
        for j in range(nN):
            y = np.log(field[:, j] / level)
            m = np.isfinite(y)
            idx = np.where(np.diff(np.sign(y[m])))[0]
            if len(idx) == 0:
                continue
            tt = np.log(Te[m])
            q = idx[0]
            f = -y[m][q] / (y[m][q + 1] - y[m][q])
            outc[j] = float(np.exp(tt[q] + f * (tt[q + 1] - tt[q])))
        return outc

    def report(name, got, want, digits_rel):
        rel = abs(got - want) / abs(want)
        verdict = "REPRODUCED" if rel <= digits_rel else "NOT REPRODUCED"
        say(f"  {name:<44s} {got:<12.4g} recorded {want:<8.4g} "
            f"({rel*100:.3g}% apart)  {verdict}")
        rec(f"P3_{name.replace(' ', '_')}", float(got), f"recorded {want}; {verdict}")
        return verdict == "REPRODUCED"

    p3 = []
    warm_mask = fin & (Te[:, None] >= 2.0)
    iw = np.where(warm_mask)[0][int(np.argmax(qn[warm_mask]))]
    first_below = Te[[i for i in range(nT - 1) if np.nanmax(qn[i]) < 1e-3][0]]
    p3.append(report("inferred dne/ne at [0,4]", qn[0, 4], REC["qn_0_4"], 5e-3))
    p3.append(report("inferred max at Te=1 eV row", np.nanmax(qn[0]), REC["qn_row0_max"], 5e-3))
    p3.append(report("inferred max for Te>=2", qn[warm_mask].max(), REC["qn_warm_max"], 1e-2))
    p3.append(report("Te of that max", Te[iw], REC["qn_warm_max_Te"], 5e-3))
    p3.append(report("Te above which inferred < 1e-3", first_below,
                     REC["qn_first_below_1e3_Te"], 5e-3))
    p3.append(report("pairs with inferred > 10%", (qn[fin] > 0.10).sum(), REC["n_over_10"], 0))
    p3.append(report("pairs with inferred > 100%", (qn[fin] > 1.00).sum(), REC["n_over_100"], 0))
    p3.append(report("pairs in census", fin.sum(), REC["n_total"], 0))
    xc_inf = crossing(qn, 0.10)
    xc_tot = crossing(qn_tot, 0.10)
    p3.append(report("10% contour, inferred, min Te", np.nanmin(xc_inf), REC["contour_lo"], 5e-3))
    p3.append(report("10% contour, inferred, max Te", np.nanmax(xc_inf), REC["contour_hi"], 5e-3))
    say(f"  ground-only vs whole-manifold inference: max rel diff "
        f"{np.nanmax(np.abs(qn_tot[fin]/qn[fin]-1)):.2e}; contour shift "
        f"{np.nanmax(np.abs(xc_tot/xc_inf-1))*100:.3g}%")
    say(f"  P3 overall: {'REPRODUCED' if all(p3) else 'NOT REPRODUCED in ' + str(sum(not v for v in p3)) + ' items'}")
    rec("P3_all_reproduced", bool(all(p3)))

    # ---- scratch values --------------------------------------------------
    say()
    say("SCRATCH VALUES (cr-physicist check2.py), heat direction")
    say(f"  {'pair':<8} {'inferred':>10} {'scratch':>9}   {'self-cons':>10} {'scratch':>9}"
        f"   {'bookkeep':>9} {'dln_ng_fix':>10} {'dln_ng_sc':>10} {'tau_slow fix->sc':>22}")
    for (i, j), (s_inf, s_sc) in SCRATCH.items():
        r = next(r for r in rows if r["direction"] == "heat" and r["i"] == i and r["j"] == j)
        ok_inf = abs(r["dne_inferred_ground"] / s_inf - 1) < 1e-2
        ok_sc = abs(r["dne_sc"] / s_sc - 1) < 1e-2
        say(f"  [{i:>2},{j}]  {r['dne_inferred_ground']:>10.3e} {s_inf:>9.3g} "
            f"{'ok' if ok_inf else 'XX'} {r['dne_sc']:>10.3e} {s_sc:>9.3g} "
            f"{'ok' if ok_sc else 'XX'} {r['dne_bookkeeping']:>9.3e} "
            f"{r['ln_x_fixed']:>+10.4f} {r['dln_ng_closed']:>+10.4f} "
            f"{r['tau_slow_fixed']:>10.3e} -> {r['tau_slow_sc']:.3e}")
        rec(f"scratch_heat_{i}_{j}_inferred", r["dne_inferred_ground"], f"scratch {s_inf}")
        rec(f"scratch_heat_{i}_{j}_selfconsistent", r["dne_sc"], f"scratch {s_sc}")

    # ---- P1: warm set ----------------------------------------------------
    say()
    say("P1  WARM SET (window_ok & Te_pre >= 2 eV), both directions")
    w_sc = np.abs(A["dne_sc"][warm])
    w_inf = A["dne_inferred_ground"][warm]
    w_e1 = np.abs(A["rel_change_eps_endpoint"][warm])
    w_e2 = np.abs(A["rel_change_eps_at_neplus"][warm])
    w_t = np.abs(A["tau_slow_rel_change"][warm])
    qw = np.where(warm)[0]

    def where(arr):
        q = qw[int(np.argmax(arr))]
        return (f"{A['direction'][q]} [{A['i'][q]},{A['j'][q]}] "
                f"Te={A['Te_pre'][q]:.3f} ne={A['ne_pre'][q]:.3g}")

    say(f"  max |ne+/ne - 1| self-consistent   {w_sc.max():.3e}  at {where(w_sc)}")
    say(f"  max |u_new - u_old| inferred        {w_inf.max():.3e}  at {where(w_inf)}")
    say(f"  max |rel change eps_plateau|, endpoint read at ne+   {w_e1.max():.3e}  at {where(w_e1)}")
    say(f"  max |rel change eps_plateau|, table read at ne+      {w_e2.max():.3e}  at {where(w_e2)}")
    say(f"  max |rel change tau_slow|                            {w_t.max():.3e}  at {where(w_t)}")
    say(f"  max |eps_plateau| itself on the warm set             {A['eps_plateau_fixed'][warm].max():.4f}")
    p1 = (w_sc.max() <= 5.35e-3) and (max(w_e1.max(), w_e2.max()) < 2e-3)
    say(f"  P1 ({'ne <= 5.3e-3 and eps change < 0.2%'}): "
        f"{'HOLDS' if p1 else 'REFUTED'}")
    rec("warm_max_abs_dne_sc", float(w_sc.max()), where(w_sc))
    rec("warm_max_dne_inferred", float(w_inf.max()), where(w_inf))
    rec("warm_max_rel_change_eps_endpoint", float(w_e1.max()), where(w_e1))
    rec("warm_max_rel_change_eps_at_neplus", float(w_e2.max()), where(w_e2))
    rec("warm_max_rel_change_tau_slow", float(w_t.max()), where(w_t))
    rec("P1_holds", bool(p1))

    # ---- P2: 1 eV row ----------------------------------------------------
    say()
    say("P2  Te = 1 eV ROW, heating: fixed-n_e inference vs closed nuclei")
    say(f"  {'j':>2} {'ne':>9} {'u_old':>8} {'u_new_fix':>9} {'inferred':>9} "
        f"{'bookkeep':>9} {'self-cons':>9} {'ne+':>9} {'ln x_fix':>9} {'dln ng_sc':>9} "
        f"{'tau_slow fix':>12} {'tau_slow sc':>12} {'eps_fix':>8} {'eps_end':>8} {'eps_ne+':>8}")
    r0 = [r for r in rows if r["direction"] == "heat" and r["i"] == 0]
    for r in r0:
        say(f"  {r['j']:>2} {r['ne_pre']:>9.3g} {r['u_old']:>8.3f} {r['u_new_fixed']:>9.3f} "
            f"{r['dne_inferred_ground']:>9.3f} {r['dne_bookkeeping']:>9.3f} "
            f"{r['dne_sc']:>9.3f} {r['ne_post_sc']:>9.3g} {r['ln_x_fixed']:>+9.4f} "
            f"{r['dln_ng_closed']:>+9.4f} {r['tau_slow_fixed']:>12.4e} "
            f"{r['tau_slow_sc']:>12.4e} {r['eps_plateau_fixed']:>8.4f} "
            f"{r['eps_plateau_endpoint']:>8.4f} {r['eps_plateau_at_neplus']:>8.4f}")
    sc0 = np.array([r["dne_sc"] for r in r0])
    p2 = bool(np.all((sc0 <= 5.0) & (sc0 >= 0.3)))
    say(f"  self-consistent ne+/ne - 1 at 1 eV: {sc0.min():.3f} .. {sc0.max():.3f}; "
        f"inferred: {min(r['dne_inferred_ground'] for r in r0):.3g} .. "
        f"{max(r['dne_inferred_ground'] for r in r0):.3g}")
    say(f"  P2 (factor ~2, not 15): {'HOLDS' if p2 else 'REFUTED'}")
    rec("row0_heat_dne_sc_min", float(sc0.min())); rec("row0_heat_dne_sc_max", float(sc0.max()))
    rec("P2_holds", p2)

    # ---- distribution over all pairs --------------------------------------
    say()
    say("DISTRIBUTION over all pairs (self-consistent |ne+/ne - 1| vs inferred)")
    for lab, m in (("heat (392)", heat), ("cool (392)", cool), ("all (784)", np.ones_like(heat))):
        s_ = np.abs(A["dne_sc"][m]); q_ = A["dne_inferred_ground"][m]
        say(f"  {lab:<11} self-cons: median {np.median(s_):.3e} p90 {np.percentile(s_,90):.3e} "
            f"max {s_.max():.3f}  >10%: {(s_>0.1).sum():>3d}  >100%: {(s_>1).sum():>3d}   |  "
            f"inferred: median {np.median(q_):.3e} max {q_.max():.3f}  >10%: {(q_>0.1).sum():>3d}  "
            f">100%: {(q_>1).sum():>3d}")
        rec(f"{lab.split()[0]}_sc_over_10pct", int((s_ > 0.1).sum()))
        rec(f"{lab.split()[0]}_sc_over_100pct", int((s_ > 1).sum()))
        rec(f"{lab.split()[0]}_sc_max", float(s_.max()))
    off = A["ne_post_off_grid"].astype(bool)
    say(f"  pairs whose ne+ leaves the tabulated density range: {off.sum()} "
        f"(Te_pre up to {A['Te_pre'][off].max() if off.any() else float('nan'):.3f} eV)")
    rec("n_ne_post_off_grid", int(off.sum()))
    e1 = np.abs(A["rel_change_eps_endpoint"]); e2 = np.abs(A["rel_change_eps_at_neplus"])
    t_ = np.abs(A["tau_slow_rel_change"])
    say(f"  all pairs: max |rel change eps_plateau| endpoint {e1.max():.3f}, "
        f"table-at-ne+ {e2.max():.3f}; max |rel change tau_slow| {t_.max():.3f}")
    rec("all_max_rel_change_eps_endpoint", float(e1.max()))
    rec("all_max_rel_change_eps_at_neplus", float(e2.max()))
    rec("all_max_rel_change_tau_slow", float(t_.max()))
    # the physicist's [0,4] eps triple, for the record
    r04 = next(r for r in rows if r["direction"] == "heat" and r["i"] == 0 and r["j"] == 4)
    rec("heat_0_4_eps_plateau_fixed", r04["eps_plateau_fixed"])
    rec("heat_0_4_eps_plateau_endpoint", r04["eps_plateau_endpoint"])
    rec("heat_0_4_eps_plateau_at_neplus", r04["eps_plateau_at_neplus"])
    rec("heat_0_4_tau_slow_fixed_s", r04["tau_slow_fixed"])
    rec("heat_0_4_tau_slow_sc_s", r04["tau_slow_sc"])
    rec("heat_0_4_dln_ng_closed", r04["dln_ng_closed"])

    # ---- 10 percent contour ----------------------------------------------
    say()
    say("10 PERCENT CONTOUR IN Te PER DENSITY COLUMN (heating, log-log interpolation)")
    xc_sc = crossing(sc, 0.10)
    say(f"  {'j':>2} {'ne':>10} {'inferred (chapter)':>19} {'self-consistent':>16}")
    for j in range(nN):
        say(f"  {j:>2} {ne[j]:>10.3g} {xc_inf[j]:>19.4f} {xc_sc[j]:>16.4f}")
    say(f"  inferred      : {np.nanmin(xc_inf):.3f} .. {np.nanmax(xc_inf):.3f} eV "
        f"(chapter: {REC['contour_lo']} .. {REC['contour_hi']})")
    say(f"  self-consist. : {np.nanmin(xc_sc):.3f} .. {np.nanmax(xc_sc):.3f} eV")
    for j in range(nN):
        rec(f"contour10_inferred_Te_col{j}", float(xc_inf[j]))
        rec(f"contour10_selfconsistent_Te_col{j}", float(xc_sc[j]))
    rec("contour10_inferred_Te_min", float(np.nanmin(xc_inf)))
    rec("contour10_inferred_Te_max", float(np.nanmax(xc_inf)))
    rec("contour10_selfconsistent_Te_min", float(np.nanmin(xc_sc)))
    rec("contour10_selfconsistent_Te_max", float(np.nanmax(xc_sc)))
    # where does the self-consistent measure first fall below 1e-3 (heating)?
    below = [i for i in range(nT - 1) if np.nanmax(sc[i]) < 1e-3]
    if below:
        say(f"  self-consistent heating correction first < 1e-3 above Te = {Te[below[0]]:.3f} eV "
            f"(inferred: {first_below:.3f} eV)")
        rec("selfconsistent_first_below_1e3_Te", float(Te[below[0]]))

    say()
    say("VERDICT  P1 " + ("HOLDS" if p1 else "REFUTED") + "   P2 " +
        ("HOLDS" if p2 else "REFUTED") + "   P3 " +
        ("REPRODUCED" if all(p3) else "NOT REPRODUCED"))

    # ---- write -------------------------------------------------------------
    def fmt(v):
        # numpy 2 repr of np.float64 is "np.float64(x)"; cast first so the
        # csv holds plain full-precision decimals
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (float, np.floating)):
            return repr(float(v))
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)

    if a.write:
        out.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0])
        with (out / "closed_nuclei.csv").open("w") as f:
            f.write("\n".join(prov) + "\n")
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(fmt(r[kk]) for kk in keys) + "\n")
        with (out / "closed_nuclei_summary.csv").open("w") as f:
            f.write("\n".join(prov) + "\n")
            f.write("key,value,comment\n")
            for kk, v, c in summary:
                f.write(f"{kk},{fmt(v)},\"{c}\"\n")
        (out / "closed_nuclei.txt").write_text("\n".join(prov) + "\n" + "\n".join(lines) + "\n")
        say(f"\nwrote {out/'closed_nuclei.csv'}")
        say(f"wrote {out/'closed_nuclei_summary.csv'}")
        say(f"wrote {out/'closed_nuclei.txt'}")
    else:
        say("\n(dry run; pass --write to save outputs)")


if __name__ == "__main__":
    main()
