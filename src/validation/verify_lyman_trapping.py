#!/usr/bin/env python
"""
verify_lyman_trapping.py
========================
Quantify the effect of Lyman-series radiation trapping on the plateau error
map, closing open item 1 of `outputs/findings_10_four_agent_review.md`.

WHY THIS EXISTS
---------------
`thesis_ready.md` A11 leads with a 38.7% breakdown at Te = 1.0 eV,
ne = 5.18e13. Section 3.5.3 restricts quantitative results to Te >~ 2 eV
because the Lyman-alpha escape factor falls to ~3e-5 on the Te = 1 eV edge.
The headline therefore sits inside the region the thesis itself excludes, and
nobody had computed what trapping does to f3, f4 or eps_plateau.

This script computes it.

WHAT IT DOES NOT DO
-------------------
It does not touch `data/processed/cr_matrix/`. The canonical L_grid is read
for comparison only. Trapped matrices are built in memory via
`assemble_cr_matrix.precompute_L_grid(rates=..., out_dir=...)` with a modified
rate dict and are written, if at all, only under `validation/lyman_trapping/`.

METHOD
------
1. Lyman transitions are located in the repo's own A-value arrays: the
   resolved decays into the ground state, A_resolved[g, j], and the bundled
   decays A_bund_res[g, b]. No transition list is hardcoded.

2. Line-centre absorption cross sections come from the repo's own A-values and
   spectroscopic energies via the Ladenburg relation inverted through

       A_ul = (8 pi^2 e^2 nu^2 / m_e c^3) (g_l/g_u) f_lu     [Gaussian CGS]
       sigma_0 = (pi e^2 / m_e c) f_lu / (sqrt(pi) dnu_D)

   The Lyman-alpha value is checked against `escape_factor.lyman_alpha_sigma0`,
   which is an independent implementation reading f_12 = 0.4162 from
   literature. Disagreement above 1% raises: that check is the reason this
   script may be trusted to extend beyond Lyman-alpha.

3. Theta_P per line from `escape_factor.escape_factor_quadrature`, the
   ADAS214 population escape factor by direct quadrature. Not a fitting
   formula, and validated in that module against the Holstein asymptote.

4. Trapping is SELF-CONSISTENT. Theta_P depends on n(1s); n(1s) depends on
   Theta_P through the CR balance. The script iterates

       Theta -> A' -> L' -> n = -L'^{-1} S -> n(1s) -> tau_c -> Theta

   to a fixed point. A run that fails to converge raises rather than
   returning the last iterate.

5. The slab thickness D is a NEW free parameter that the 0-D model does not
   contain. It is swept, never fixed. Every trapped number in the output
   carries its D.

CONVENTIONS
-----------
- n_ion = n_e (quasineutrality, pure hydrogen), matching the rest of the repo.
- T_at = Te by default (the project's T_i = T_e convention); --t-at overrides
  with a fixed neutral temperature for a Franck-Condon sensitivity check.
- tau_c is the line-centre optical depth over the HALF slab, tau_c = kappa0*D/2,
  which is the convention `escape_factor.py` documents and requires.
- eps_plateau is computed by the same algebra as
  `verify_plateau_gridmap.py:200-209`; see `eps_plateau_map` below.

OUTPUTS
-------
`validation/lyman_trapping/lyman_trapping.txt` and `.csv` (with --write).
Without --write nothing is persisted.

Provenance is printed at the top of every run: the SHA-256 of every input
array, the grid shapes, and the interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

# ── repo wiring ────────────────────────────────────────────────────────────────
# cr_context finds the repo root; everything else is addressed relative to it.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent))

from cr_context import CRContext, find_repo_root          # noqa: E402

ROOT = find_repo_root(_HERE)
sys.path.insert(0, str(ROOT / "src" / "rates"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import assemble_cr_matrix as acm                          # noqa: E402
from escape_factor import (                               # noqa: E402
    escape_factor_quadrature,
    lyman_alpha_sigma0,
)

# ── physical constants (CGS), same values escape_factor.py uses ───────────────
E_ESU = 4.80326e-10      # statcoulomb
M_E = 9.10938e-28        # g
C_CGS = 2.99792e10       # cm/s
M_H = 1.67262e-24        # g
EV_TO_ERG = 1.60218e-12
H_PLANCK = 6.62607e-27   # erg s


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Locate the Lyman transitions and build their cross sections
# ══════════════════════════════════════════════════════════════════════════════

def lyman_channels(rates: dict, si, g_idx: int):
    """
    Find every radiative decay into the ground state, resolved and bundled.

    Returns two lists of dicts with keys:
        kind ('res'|'bund'), col (column index into A_resolved / A_bund_res),
        A (s^-1), E_ph (eV), g_u (upper statistical weight), label.

    Nothing here is hardcoded: the channels are whichever entries of
    A_resolved[g,:] and A_bund_res[g,:] are non-zero.
    """
    A_res = rates["A_resolved"]
    A_br = rates["A_bund_res"]
    n_res = A_res.shape[0]

    I_eV = si["I_eV"].to_numpy(float)
    g_stat = si["g"].to_numpy(float)
    labels = si["label"].tolist()

    res = []
    for j in np.nonzero(A_res[g_idx, :])[0]:
        # photon energy = binding(ground) - binding(upper), spectroscopic
        E_ph = I_eV[g_idx] - I_eV[j]
        if E_ph <= 0:
            raise RuntimeError(
                f"non-positive photon energy for {labels[j]}->{labels[g_idx]}: "
                f"{E_ph} eV; state_index I_eV column is inconsistent")
        res.append(dict(kind="res", col=int(j), A=float(A_res[g_idx, j]),
                        E_ph=float(E_ph), g_u=float(g_stat[j]),
                        label=f"{labels[j]}->{labels[g_idx]}"))

    bund = []
    for b in np.nonzero(A_br[g_idx, :])[0]:
        k = n_res + b                     # index into the full state list
        E_ph = I_eV[g_idx] - I_eV[k]
        if E_ph <= 0:
            raise RuntimeError(
                f"non-positive photon energy for bundled {labels[k]}: {E_ph} eV")
        bund.append(dict(kind="bund", col=int(b), A=float(A_br[g_idx, b]),
                         E_ph=float(E_ph), g_u=float(g_stat[k]),
                         label=f"{labels[k]}->{labels[g_idx]}"))

    if not res:
        raise RuntimeError("no resolved decays into the ground state found; "
                           "A_resolved or the ground index is wrong")
    return res, bund


def sigma0_from_A(A: float, E_ph_eV: float, g_u: float, g_l: float,
                  T_at_eV: float) -> float:
    """
    Line-centre absorption cross section [cm^2] from an Einstein A coefficient.

    Inverts the Gaussian-CGS Einstein relation

        A_ul = (8 pi^2 e^2 nu^2 / m_e c^3) (g_l/g_u) f_lu

    for f_lu, then applies the Ladenburg relation
              sigma_0 = (pi e^2 / m_e c) f_lu / (sqrt(pi) dnu_D).

    Deriving f from the repo's own A-values means no oscillator strength is
    imported; the Lyman-alpha result is cross-checked against the independent
    literature value in escape_factor.lyman_alpha_sigma0.
    """
    lam = H_PLANCK * C_CGS / (E_ph_eV * EV_TO_ERG)     # cm
    nu0 = C_CGS / lam
    # Gaussian CGS: the coefficient is 8*pi^2, not the 2*pi of the SI form.
    # Using the SI coefficient here overstates f by exactly 4*pi; the
    # Lyman-alpha cross-check below is what catches that.
    f_lu = A * M_E * C_CGS**3 * g_u / (8.0 * np.pi**2 * E_ESU**2 * nu0**2 * g_l)

    v_th = np.sqrt(2.0 * T_at_eV * EV_TO_ERG / M_H)
    dnu_D = (nu0 / C_CGS) * v_th
    prefactor = np.pi * E_ESU**2 / (M_E * C_CGS)
    return prefactor * f_lu / (np.sqrt(np.pi) * dnu_D), f_lu


def check_sigma0_against_module(res_channels, g_l: float, T_at: float,
                                tol: float = 0.01) -> dict:
    """
    Severity check: the Lyman-alpha sigma_0 derived from the repo's A-value
    must reproduce escape_factor.lyman_alpha_sigma0, which uses an independent
    literature f_12. If it does not, the derivation is wrong and every higher
    Lyman line built the same way is wrong too. Raises on failure.
    """
    lya = max(res_channels, key=lambda c: c["A"])      # strongest = Ly-alpha
    s_derived, f_derived = sigma0_from_A(lya["A"], lya["E_ph"], lya["g_u"],
                                         g_l, T_at)
    s_module = lyman_alpha_sigma0(T_at)
    rel = abs(s_derived - s_module) / s_module
    if rel > tol:
        raise RuntimeError(
            f"Lyman-alpha sigma_0 cross-check FAILED at T_at={T_at} eV: "
            f"derived {s_derived:.4e} cm^2 from A={lya['A']:.5e} s^-1 "
            f"(f={f_derived:.4f}) vs escape_factor module {s_module:.4e} cm^2, "
            f"{100*rel:.2f}% apart (tolerance {100*tol:.0f}%). "
            f"The A->sigma_0 inversion is wrong; do not trust any line.")
    return dict(line=lya["label"], A=lya["A"], f_derived=f_derived,
                sigma_derived=s_derived, sigma_module=s_module, rel=rel)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Apply escape factors to the rate dict
# ══════════════════════════════════════════════════════════════════════════════

def apply_trapping(rates: dict, res_ch, bund_ch, theta_res, theta_bund,
                   g_idx: int) -> dict:
    """
    Return a copy of `rates` with the Lyman A-values scaled by their escape
    factors and the total decay rates gamma reduced consistently.

    A_eff = Theta_P * A. Because gamma[j] is the SUM of all decays out of j,
    removing (1-Theta)*A from one channel must remove exactly the same amount
    from gamma[j], or the matrix stops being a rate matrix.
    """
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in rates.items()}

    for ch, th in zip(res_ch, theta_res):
        j, A = ch["col"], ch["A"]
        out["A_resolved"][g_idx, j] = A * th
        out["gamma_resolved"][j] -= A * (1.0 - th)

    for ch, th in zip(bund_ch, theta_bund):
        b, A = ch["col"], ch["A"]
        out["A_bund_res"][g_idx, b] = A * th
        out["gamma_bundled"][b] -= A * (1.0 - th)

    if np.any(out["gamma_resolved"] < -1e-9) or np.any(out["gamma_bundled"] < -1e-9):
        raise RuntimeError("negative total decay rate after trapping; "
                           "gamma bookkeeping is wrong")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Self-consistent trapped matrix
# ══════════════════════════════════════════════════════════════════════════════

def solve_selfconsistent(rates, res_ch, bund_ch, g_idx, te_grid, ne_grid,
                         D_cm, t_at_fixed=None, tol=1e-8, itmax=200):
    """
    Fixed-point iteration on Theta_P over the whole grid.

    n(1s) = n_ion * [ -L^{-1} S ]_g  with n_ion = n_e, S given per unit n_ion.
    tau_c = n(1s) * sigma_0 * D/2.

    Returns L_grid, S_grid, theta_grid (n_Te, n_ne, n_lines), n1s_grid,
    and the per-point iteration count. Raises if any point fails to converge.
    """
    n_Te, n_ne = len(te_grid), len(ne_grid)
    n_lines = len(res_ch) + len(bund_ch)
    all_ch = list(res_ch) + list(bund_ch)

    n_states = rates["A_resolved"].shape[0] + rates["A_bund_res"].shape[1]
    L_grid = np.zeros((n_Te, n_ne, n_states, n_states))
    S_grid = np.zeros((n_Te, n_ne, n_states))
    theta_grid = np.ones((n_Te, n_ne, n_lines))
    n1s_grid = np.zeros((n_Te, n_ne))
    iters = np.zeros((n_Te, n_ne), dtype=int)
    g_l = 2.0                                    # ground-state weight, 1S

    for i in range(n_Te):
        T_at = t_at_fixed if t_at_fixed is not None else float(te_grid[i])
        sig = np.array([sigma0_from_A(c["A"], c["E_ph"], c["g_u"], g_l, T_at)[0]
                        for c in all_ch])
        for j in range(n_ne):
            ne = float(ne_grid[j])
            theta = np.ones(n_lines)
            for it in range(1, itmax + 1):
                r = apply_trapping(rates, res_ch, bund_ch,
                                   theta[:len(res_ch)], theta[len(res_ch):],
                                   g_idx)
                L = acm.build_L(i, ne, r)
                S = acm.build_source(i, ne, r, n_ion=1.0)
                n = np.linalg.solve(L, -S)          # per unit n_ion
                if not np.all(np.isfinite(n)):
                    raise RuntimeError(f"non-finite CRE solution at [{i},{j}]")
                if np.any(n < 0):
                    raise RuntimeError(
                        f"negative CRE population at [{i},{j}]: min {n.min():.3e}")
                n1s = n[g_idx] * ne                  # n_ion = n_e
                tau_c = n1s * sig * D_cm / 2.0
                theta_new = np.array([escape_factor_quadrature(t) for t in tau_c])
                if np.max(np.abs(theta_new - theta)) < tol:
                    theta = theta_new
                    break
                # damped update: the map is contracting but stiff at large tau
                theta = 0.5 * theta + 0.5 * theta_new
            else:
                raise RuntimeError(
                    f"escape-factor fixed point did not converge at [{i},{j}] "
                    f"(Te={te_grid[i]:.4g} eV, ne={ne:.4g}, D={D_cm} cm) "
                    f"after {itmax} iterations; residual "
                    f"{np.max(np.abs(theta_new - theta)):.3e}")

            r = apply_trapping(rates, res_ch, bund_ch,
                               theta[:len(res_ch)], theta[len(res_ch):], g_idx)
            L_grid[i, j] = acm.build_L(i, ne, r)
            S_grid[i, j] = acm.build_source(i, ne, r, n_ion=1.0)
            theta_grid[i, j] = theta
            n1s_grid[i, j] = n1s
            iters[i, j] = it

    return L_grid, S_grid, theta_grid, n1s_grid, iters


# ══════════════════════════════════════════════════════════════════════════════
# 4. eps_plateau, by the same algebra as verify_plateau_gridmap.py:200-209
# ══════════════════════════════════════════════════════════════════════════════

def eps_plateau_map(L, S, te_grid, ne_grid, g_idx, E, N3, N4, n3E, n4E,
                    frac=0.05, win_lo=30.0, win_hi=30.0):
    """
    Reproduces verify_plateau_gridmap.py's heating/cooling scan. Returns a list
    of dicts with the same field names, so the two can be compared directly.
    """
    rows = []
    for sgn, dlab in ((+1, "heat"), (-1, "cool")):
        for i in range(len(te_grid)):
            k = int(np.argmin(np.abs(te_grid - te_grid[i] * (1 + sgn * frac))))
            if k == i:
                continue
            for j in range(len(ne_grid)):
                lam = np.linalg.eigvals(L[k, j])
                lam = lam[np.argsort(lam.real)[::-1]]
                if lam[0].real >= 0 or lam[1].real >= 0:
                    raise RuntimeError(
                        f"unstable operator at Te={te_grid[k]:g} ne={ne_grid[j]:g}")
                tQ, tR = -1.0 / lam[0].real, -1.0 / lam[1].real
                window_ok = (win_lo * tR) < (tQ / win_hi)

                n_old = np.linalg.solve(L[i, j], -S[i, j])
                n_new = np.linalg.solve(L[k, j], -S[k, j])

                LEE = L[k, j][np.ix_(E, E)]
                LEg = L[k, j][np.ix_(E, [g_idx])].ravel()
                n0 = np.linalg.solve(LEE, -S[k, j][E])
                n1 = np.linalg.solve(LEE, -LEg * n_old[g_idx])
                x_new = n_new[g_idx] / n_old[g_idx]

                sup = (np.abs(n0 + x_new * n1 - n_new[E]).max()
                       / np.abs(n_new[E]).max())
                if sup > 1e-8:
                    raise RuntimeError(
                        f"superposition residual {sup:.3e} at [{i},{j}] {dlab}; "
                        f"the two-channel split is not exact for this operator")

                a3_0, a4_0 = n0[n3E].sum(), n0[n4E].sum()
                a3_1, a4_1 = n1[n3E].sum(), n1[n4E].sum()
                f3 = a3_1 / (a3_0 + a3_1)
                f4 = a4_1 / (a4_0 + a4_1)

                Rq = n_new[N3].sum() / n_new[N4].sum()
                R_pe = (a3_0 + a3_1) / (a4_0 + a4_1)
                R_old = n_old[N3].sum() / n_old[N4].sum()
                d_step = R_old / Rq - 1.0
                d_pe = R_pe / Rq - 1.0

                rows.append(dict(
                    direction=dlab, i=i, j=j, Te=float(te_grid[i]),
                    ne=float(ne_grid[j]), tau_QSS=tQ, tau_relax=tR, M=tQ / tR,
                    window_ok=bool(window_ok), x_new=x_new,
                    eps_step=abs(d_step), eps_plateau=abs(d_pe),
                    f3=f3, f4=f4, sens=f3 - f4,
                    abs_ln_x=abs(np.log(x_new)),
                ))
    return rows


def elm_lower_bound(tau_QSS, eps_plateau, tau_d):
    """A11's lower bound: eps * (tau_QSS/tau_d) * (1 - exp(-tau_d/tau_QSS))."""
    return eps_plateau * (tau_QSS / tau_d) * (1.0 - np.exp(-tau_d / tau_QSS))


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--slab", type=float, nargs="+",
                   default=[1.0, 5.0, 20.0],
                   help="slab thickness D in cm; swept, never assumed")
    p.add_argument("--t-at", type=float, default=None,
                   help="fixed neutral temperature [eV]; default T_at = Te")
    p.add_argument("--frac", type=float, default=0.05)
    p.add_argument("--tau-d", type=float, default=1e-4, help="ELM duration [s]")
    p.add_argument("--te-report", type=float, default=2.0,
                   help="report threshold matching thesis section 3.5.3")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--write", action="store_true")
    a = p.parse_args()

    # tee stdout to the run log when persisting, so the .txt this module's
    # docstring promises actually exists and carries the full provenance block
    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()

    log_fh = None
    if a.write:
        _out = Path(a.out) if a.out else ROOT / "validation" / "lyman_trapping"
        _out.mkdir(parents=True, exist_ok=True)
        log_fh = (_out / "lyman_trapping.txt").open("w")
        sys.stdout = _Tee(sys.__stdout__, log_fh)

    ctx = CRContext.load()
    ctx.validate()
    print(ctx.describe())
    print()

    # the state table itself: cr_context resolved the path, we read the columns
    # (g, I_eV) it does not expose. Never re-derive the path here.
    import pandas as pd
    si = pd.read_csv(ctx.state_index_path)

    te_grid = ctx.te_grid
    ne_grid = ctx.ne_grid
    g_idx = ctx.ground_index
    n_states = ctx.n_states

    nv = ctx.n_values
    if len(si) != n_states:
        raise RuntimeError(
            f"state_index has {len(si)} rows but L_grid has {n_states} states")
    if not np.array_equal(si["n"].to_numpy(float), nv):
        raise RuntimeError("state_index 'n' column disagrees with ctx.n_values")
    E = np.array([i for i in range(n_states) if i != g_idx])
    N3 = np.where(nv == 3)[0]
    N4 = np.where(nv == 4)[0]
    pos = {s: k for k, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3])
    n4E = np.array([pos[s] for s in N4])

    rates = acm.load_rates({k: str(ROOT / v) for k, v in acm.PATHS.items()})

    print("=" * 78)
    print("INPUT PROVENANCE (sha256, first 16 hex)")
    print("=" * 78)
    for k in sorted(rates):
        print(f"  {k:<16s} {str(rates[k].shape):<16s} {sha256(rates[k])}")
    print(f"  {'L_grid (canon)':<16s} {str(ctx.L_grid.shape):<16s} "
          f"{sha256(ctx.L_grid)}")
    print(f"  interpreter      {sys.executable}")
    print(f"  numpy            {np.__version__}")
    print()

    res_ch, bund_ch = lyman_channels(rates, si, g_idx)
    all_ch = list(res_ch) + list(bund_ch)
    print("=" * 78)
    print(f"LYMAN CHANNELS FOUND: {len(res_ch)} resolved + {len(bund_ch)} bundled")
    print("=" * 78)
    print(f"  {'line':<14s} {'A [s^-1]':>12s} {'E_ph [eV]':>10s} {'g_u':>5s} "
          f"{'f_lu':>9s} {'sigma0(1eV)':>12s}")
    for c in all_ch:
        s, f = sigma0_from_A(c["A"], c["E_ph"], c["g_u"], 2.0, 1.0)
        print(f"  {c['label']:<14s} {c['A']:12.4e} {c['E_ph']:10.4f} "
              f"{c['g_u']:5.0f} {f:9.4f} {s:12.4e}")
    print()

    chk = check_sigma0_against_module(res_ch, 2.0, 1.0)
    print(f"  CROSS-CHECK vs escape_factor.lyman_alpha_sigma0 at T_at = 1 eV:")
    print(f"    {chk['line']}: derived f = {chk['f_derived']:.4f} "
          f"(literature 0.4162)")
    print(f"    sigma0 derived {chk['sigma_derived']:.4e} vs module "
          f"{chk['sigma_module']:.4e}  -> {100*chk['rel']:.3f}% apart  [PASS]")
    print()

    # ── baseline: untrapped, from the same builder, to prove the builder ──────
    print("=" * 78)
    print("BASELINE CHECK: untrapped rebuild vs canonical L_grid")
    print("=" * 78)
    L0 = np.zeros_like(ctx.L_grid)
    S0 = np.zeros((len(te_grid), len(ne_grid), n_states))
    for i in range(len(te_grid)):
        for j, ne in enumerate(ne_grid):
            L0[i, j] = acm.build_L(i, float(ne), rates)
            S0[i, j] = acm.build_source(i, float(ne), rates, n_ion=1.0)
    dmax = np.abs(L0 - ctx.L_grid).max()
    rel = dmax / np.abs(ctx.L_grid).max()
    print(f"  max |rebuilt - canonical| = {dmax:.6e}  (relative {rel:.3e})")
    if rel > 1e-12:
        raise RuntimeError(
            "untrapped rebuild does not reproduce the canonical L_grid; "
            "the rate dict or builder has drifted. Every trapped number below "
            "would be meaningless. STOPPING.")
    print("  [PASS] builder reproduces the canonical matrix to machine precision")
    print()

    base_rows = eps_plateau_map(L0, S0, te_grid, ne_grid, g_idx, E, N3, N4,
                                n3E, n4E, frac=a.frac)

    def summarise(rows, tag, D=None):
        ok = [r for r in rows if r["window_ok"]]
        lb = np.array([elm_lower_bound(r["tau_QSS"], r["eps_plateau"], a.tau_d)
                       for r in ok])
        eps = np.array([r["eps_plateau"] for r in ok])
        te = np.array([r["Te"] for r in ok])
        ne = np.array([r["ne"] for r in ok])
        hot = te >= a.te_report
        dense = ne >= 1e14
        return dict(
            tag=tag, D=D, n_ok=len(ok),
            n_break=int((lb > 0.10).sum()), worst=float(lb.max()),
            n_break_hot=int((lb[hot] > 0.10).sum()), n_hot=int(hot.sum()),
            worst_hot=float(lb[hot].max()) if hot.any() else float("nan"),
            n_break_div=int((lb[hot & dense] > 0.10).sum()),
            n_div=int((hot & dense).sum()),
            worst_div=float(lb[hot & dense].max()) if (hot & dense).any() else float("nan"),
            eps_max=float(eps.max()),
        )

    results = [summarise(base_rows, "untrapped (canonical)")]

    # ── trapped runs ─────────────────────────────────────────────────────────
    trapped_rows = {}
    for D in a.slab:
        print("=" * 78)
        print(f"SELF-CONSISTENT TRAPPED RUN: D = {D} cm, "
              f"T_at = {'Te' if a.t_at is None else f'{a.t_at} eV'}")
        print("=" * 78)
        Lt, St, th, n1s, iters = solve_selfconsistent(
            rates, res_ch, bund_ch, g_idx, te_grid, ne_grid, D,
            t_at_fixed=a.t_at)
        print(f"  fixed point converged everywhere; iterations "
              f"min {iters.min()} median {int(np.median(iters))} max {iters.max()}")
        lya = int(np.argmax([c["A"] for c in all_ch]))
        print(f"  Theta_P(Ly-alpha): min {th[..., lya].min():.4e}  "
              f"max {th[..., lya].max():.4e}")
        print()
        print(f"  {'Te':>7s} {'ne':>10s} {'n(1s)':>11s} {'Theta_Lya':>11s} "
              f"{'tau_QSS':>11s} {'tau_relax':>11s} {'M':>11s}")
        for i in (0, 10, 23):
            for j in (0, 3, 4, 5):
                lam = np.sort(np.linalg.eigvals(Lt[i, j]).real)[::-1]
                neg = lam[lam < 0]
                tQ, tR = 1 / abs(neg[0]), 1 / abs(neg[1])
                print(f"  {te_grid[i]:7.3f} {ne_grid[j]:10.3e} "
                      f"{n1s[i, j]:11.4e} {th[i, j, lya]:11.4e} "
                      f"{tQ:11.4e} {tR:11.4e} {tQ/tR:11.4e}")
        print()
        rows = eps_plateau_map(Lt, St, te_grid, ne_grid, g_idx, E, N3, N4,
                               n3E, n4E, frac=a.frac)
        trapped_rows[D] = rows
        results.append(summarise(rows, f"trapped D={D} cm", D))

    # ── comparison ───────────────────────────────────────────────────────────
    print("=" * 78)
    print(f"A11 RECOUNT  (ELM tau_d = {a.tau_d:g} s, lower bound > 10%)")
    print("=" * 78)
    print(f"  {'run':<24s} {'all window_ok':>16s} {'Te>=2 eV':>16s} "
          f"{'Te>=2 & ne>=1e14':>18s}")
    for r in results:
        print(f"  {r['tag']:<24s} "
              f"{r['n_break']:>5d}/{r['n_ok']:<5d} {r['worst']:6.4f} "
              f"{r['n_break_hot']:>5d}/{r['n_hot']:<5d} {r['worst_hot']:6.4f} "
              f"{r['n_break_div']:>6d}/{r['n_div']:<5d} {r['worst_div']:6.4f}")
    print()

    print("=" * 78)
    print("RIDGE: eps_plateau by density column, heating, per Te row")
    print("=" * 78)
    for tag, rows in [("untrapped", base_rows)] + \
                     [(f"D={D} cm", trapped_rows[D]) for D in a.slab]:
        print(f"\n  --- {tag} ---")
        print(f"  {'Te':>7s} " + " ".join(f"{n:>9.2e}" for n in ne_grid)
              + "   argmax")
        for i in (0, 2, 10, 23, 35):
            vals = []
            for j in range(len(ne_grid)):
                m = [r for r in rows
                     if r["i"] == i and r["j"] == j and r["direction"] == "heat"]
                vals.append(m[0]["eps_plateau"] if m else np.nan)
            vals = np.array(vals)
            am = int(np.nanargmax(vals))
            print(f"  {te_grid[i]:7.3f} "
                  + " ".join(f"{v:9.5f}" for v in vals)
                  + f"   j={am} ({ne_grid[am]:.2e})")
    print()

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "lyman_trapping"
        out.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with (out / "lyman_trapping.csv").open("w", newline="") as fh:
            keys = list(base_rows[0].keys()) + ["run", "D_cm"]
            w = _csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in base_rows:
                w.writerow({**r, "run": "untrapped", "D_cm": 0.0})
            for D in a.slab:
                for r in trapped_rows[D]:
                    w.writerow({**r, "run": f"trapped", "D_cm": D})
        print(f"  wrote {out/'lyman_trapping.csv'}")
        print(f"  wrote {out/'lyman_trapping.txt'}")

    if log_fh is not None:
        sys.stdout = sys.__stdout__
        log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
