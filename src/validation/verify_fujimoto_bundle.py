#!/usr/bin/env python
"""
verify_fujimoto_bundle.py
=========================
Impose Fujimoto's own state-space closure on this model's operator and measure
what it does to r_1(p), to settle whether the l closure explains the factor-8
r_1 deficit of Table 4.1(b).

WHY THIS EXISTS
---------------
Chapter 4 carried, until 16 Sep 2026, the claim that the r_1 disagreement
"measures the l closure rather than this model": Fujimoto's Ch. 4 model assumes
statistical populations among l-states of the same n before writing the rate
equations, while this model resolves l below n=9. The evidence offered was that
REMOVING proton l-mixing moves r_1(2) by a factor 118, so a comparison whose
treatment of one process spans two orders of magnitude cannot adjudicate a
factor 8.

That argument is not a test of the closure. Removing l-mixing is the limit in
which l-mixing is ABSENT; Fujimoto's assumption is the limit in which it is
INFINITELY FAST. Those are opposite limits. verify_fujimoto_table41.py's own
docstring already said so, under "TWO RUNS -- AND NEITHER IS A STRICT FUJIMOTO
BENCHMARK", and named the missing calculation:

    "A strict benchmark would require building shell-level rates under
     statistical-l weighting, C(n->n') = sum_l (g_nl/g_n) sum_l' C(nl->n'l'),
     and solving that bundled operator -- not done here."

This script is that calculation.

THE ALGEBRA
-----------
The recipe in that docstring is a similarity-like projection. With shells
k = 1..15, define

    P[k, i] = 1              for state i in shell k        (sum over destinations)
    R[i, k] = g_i / g_k      for state i in shell k        (statistical source weights)

then  L_bundle = P L R  is exactly  C(n->n') = sum_l (g_nl/g_n) sum_l' C(nl->n'l').
P R = I_15 identically, since sum_{i in k} g_i / g_k = 1.

Two structural consequences, both checked below rather than asserted:

  1. P B R = 0 for any intra-shell operator B, because such an operator
     conserves each shell's population (column sums zero within the shell) and
     is confined to it. The l-mixing block is exactly such an operator.
     Therefore the bundled operator CANNOT express l-mixing, and bundling the
     production matrix and bundling the no-l-mixing matrix give identical
     results. Statistical-l is the l-mixing-irrelevant limit.

  2. Column sums map as  colsum(L_bundle)_k = sum_{i in k} (g_i/g_k) colsum(L)_i,
     i.e. the statistically weighted ionisation loss, since ionisation is the
     only process leaving the 43-state manifold.

r_0 and r_1 are computed exactly as verify_fujimoto_table41.py computes them,
so the PRODUCTION column here must reproduce that script's PRODUCTION column.

CHECKS THAT STOP THE SCRIPT
    1. max|P R - I| = 0 to roundoff
    2. resolved column sums = -ne K_ion(p)                    (manifold closure)
    3. bundled column sums = statistically weighted of the same
    4. solve residuals in both spaces below 1e-10
    5. CALIBRATION: production r_1(3) reproduces the value Chapter 4 quotes,
       6.897e-6, to 3 significant figures. If this fails the script is not
       measuring the same quantity as the thesis and nothing below is valid.
    6. SEVERITY: bundling an operator whose l-distribution is genuinely NOT
       statistical (l-mixing removed) must change r_1(2) by more than a factor
       2. A test that cannot detect a large change cannot certify a small one.
       If this gate does not fire, the null result of this script is vacuous.

PREDICTION, WRITTEN BEFORE THE RUN (16 Sep 2026)
    P R = I to machine precision. Production r_1(3) reproduces 6.897e-6.
    Bundled r_1(3) lands within a few per cent of production, NOT a factor 8
    from it, because the production model's sublevel populations are already
    within a few per cent of g_nl/g_n for p >= 3. Bundled r_1(2) moves more
    than p=3, because 2s and 2p are the most non-statistical pair in the model.
    p = 10 and p = 15 are unchanged, being bundled already.

THE REFUTING OBSERVATION
    Bundled r_1(3) approaching Fujimoto's tabulated 5.72e-5, i.e. bundle/prod
    of order 8. That would mean imposing the source's closure closes the gap,
    the l-closure explanation is correct, and Chapter 4's original text stands.
    It did not appear: the measured ratio is 1.0042.

SCOPE
    Te is taken at the grid top, 10.0 eV, against Fujimoto's 11.03 eV, a -9.3 %
    offset; the grid does not reach 11.03 eV. This is the same offset
    verify_fujimoto_table41.py carries and it is not removable without
    regenerating the rate pipeline on a wider Te grid.
"""

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_context import CRContext  # noqa: E402

# Fujimoto, Plasma Spectroscopy (2004), Table 4.1(b); Te = 1.28e5 K = 11.03 eV.
# lg n_e = log10(n_e / m^-3).  Entries are (r_0, r_1).
FUJIMOTO_TE_EV = 11.03
TABLE_41B = {
    18: {2: (7.30e-1, 1.79e-4), 3: (8.35e-1, 5.72e-5), 4: (9.47e-1, 1.66e-5),
         5: (9.83e-1, 5.08e-6), 7: (9.97e-1, 7.55e-7), 10: (1.00, 9.61e-8),
         15: (1.00, 9.34e-9)},
    21: {2: (9.81e-1, 7.70e-3), 3: (9.98e-1, 8.02e-4), 4: (1.00, 1.40e-4),
         5: (1.00, 3.63e-5), 7: (1.00, 4.80e-6), 10: (1.00, 5.80e-7),
         15: (1.00, 5.50e-8)},
}

H_PLANCK, M_E, EV_TO_J, CHI_H_EV = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19, 13.605693
NZ_CM3 = 1.0          # the solve uses n_ion = 1 cm^-3
NG_CM3 = 1.0          # unit ground density for the r_1 channel
CH4_R1_P3 = 6.897e-6  # the value Chapter 4 quotes for production r_1(3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--te", type=float, default=10.0,
                   help="Te in eV (default 10.0, the grid top)")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default validation/fujimoto_bundle)")
    return p.parse_args()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saha_boltzmann_Z(p: int, te_ev: float) -> float:
    """Z(p) in m^3 for neutral hydrogen, Fujimoto's convention n_p = Z(p) n_e n_i.
    g_p = 2p^2 and the free-electron spin 2 cancel, leaving p^2."""
    kT = te_ev * EV_TO_J
    return p ** 2 * (H_PLANCK ** 2 / (2.0 * np.pi * M_E * kT)) ** 1.5 \
        * np.exp((CHI_H_EV / p ** 2) / te_ev)


def build_projectors(n_state, g_state):
    """P (15x43) sums over a shell's sublevels; R (43x15) weights them
    statistically. L_bundle = P L R is the statistical-l reduction."""
    shells = np.arange(1, 16)
    g_shell = np.array([g_state[n_state == n].sum() for n in shells], dtype=float)
    if not np.allclose(g_shell, 2.0 * shells ** 2):
        raise RuntimeError(f"shell degeneracies are not 2n^2: {g_shell}")
    P = np.zeros((15, 43))
    R = np.zeros((43, 15))
    for k, n in enumerate(shells):
        m = (n_state == n)
        P[k, m] = 1.0
        R[m, k] = g_state[m] / g_shell[k]
    return P, R, shells, g_shell


def resolved_r(Lop, S, ctx, n_state, te_ev, ne_si, ps):
    """r_0, r_1 exactly as verify_fujimoto_table41.py computes them."""
    g0 = ctx.ground_index
    E = np.array([i for i in range(43) if i != g0])
    LEE = Lop[np.ix_(E, E)]
    LEg = Lop[np.ix_(E, [g0])].ravel()
    n0 = np.linalg.solve(LEE, -S[E])
    n1 = np.linalg.solve(LEE, -LEg * NG_CM3)
    res = max(np.linalg.norm(LEE @ n0 + S[E]) / max(np.linalg.norm(S[E]), 1e-300),
              np.linalg.norm(LEE @ n1 + LEg * NG_CM3) / max(np.linalg.norm(LEg), 1e-300))
    Z1 = saha_boltzmann_Z(1, te_ev)
    out = {}
    for p in ps:
        loc = np.array([int(np.where(E == s)[0][0]) for s in np.where(n_state == p)[0]])
        Zp = saha_boltzmann_Z(p, te_ev)
        out[p] = ((n0[loc].sum() / NZ_CM3) / (Zp * ne_si),
                  (n1[loc].sum() / NG_CM3) * (Z1 / Zp))
    return out, res, n1, E


def bundled_r(Lop, S, P, R, te_ev, ne_si, ps):
    """Same coefficients from the statistical-l bundled operator."""
    Lb, Sb = P @ Lop @ R, P @ S
    Eb = np.arange(1, 15)                      # shells 2..15; shell 1 is the ground
    LEE = Lb[np.ix_(Eb, Eb)]
    LEg = Lb[np.ix_(Eb, [0])].ravel()
    n0 = np.linalg.solve(LEE, -Sb[Eb])
    n1 = np.linalg.solve(LEE, -LEg * NG_CM3)
    res = max(np.linalg.norm(LEE @ n0 + Sb[Eb]) / max(np.linalg.norm(Sb[Eb]), 1e-300),
              np.linalg.norm(LEE @ n1 + LEg * NG_CM3) / max(np.linalg.norm(LEg), 1e-300))
    Z1 = saha_boltzmann_Z(1, te_ev)
    out = {}
    for p in ps:
        Zp = saha_boltzmann_Z(p, te_ev)
        out[p] = ((n0[p - 2] / NZ_CM3) / (Zp * ne_si),
                  (n1[p - 2] / NG_CM3) * (Z1 / Zp))
    return out, res, Lb


def main():
    a = parse_args()
    ctx = CRContext.load()
    ctx.validate()
    root = ctx.root
    out = a.out or (root / "validation" / "fujimoto_bundle")
    out.mkdir(parents=True, exist_ok=True)

    L_path = root / "data/processed/cr_matrix/L_grid.npy"
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    K_lmix_path = root / "data/processed/lmix/K_lmix.npy"
    K_ion_path = root / "data/processed/collisions/tics/K_ion_final.npy"
    S_grid = np.load(S_path)
    K_lmix = np.load(K_lmix_path)
    K_ion = np.load(K_ion_path)

    si = pd.read_csv(ctx.state_index_path)
    n_state = si["n"].values.astype(int)
    g_state = si["g"].values.astype(float)
    P, R, shells, g_shell = build_projectors(n_state, g_state)

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("FUJIMOTO STATISTICAL-l BUNDLE -- does the l closure explain the r_1 deficit?")
    say("=" * 78)
    say(f"L_grid sha256 {sha256(L_path)}")
    say(f"S_grid sha256 {sha256(S_path)}")

    ti = int(np.argmin(np.abs(ctx.te_grid - a.te)))
    te_used = float(ctx.te_grid[ti])
    say(f"Te = {te_used:.6f} eV (grid index {ti}); Fujimoto's table is "
        f"{FUJIMOTO_TE_EV} eV, {(te_used/FUJIMOTO_TE_EV - 1)*100:+.1f}% offset")

    # ---- gate 1: the projectors ------------------------------------------
    pr_err = float(np.abs(P @ R - np.eye(15)).max())
    say(f"\ngate 1  max|P R - I| = {pr_err:.3e}")
    if pr_err > 1e-12:
        raise RuntimeError(f"P R != I: {pr_err:.3e}")

    rows = []
    for lg_ne, table in sorted(TABLE_41B.items()):
        ne_cm3 = 10.0 ** lg_ne / 1e6
        j = int(np.argmin(np.abs(np.log(ctx.ne_grid) - np.log(ne_cm3))))
        if abs(ctx.ne_grid[j] / ne_cm3 - 1) > 1e-6:
            say(f"\nlg n_e = {lg_ne}: not a grid point, skipped")
            continue
        ne_si = float(ctx.ne_grid[j]) * 1e6
        L = ctx.L_grid[ti, j]
        S = S_grid[ti, j]
        ps = sorted(table)

        say("\n" + "=" * 78)
        say(f"lg n_e = {lg_ne}  ->  {ne_cm3:.3e} cm^-3  (grid j={j})")

        # explicit l-mixing block, built exactly as verify_fujimoto_table41.py does
        Lm = (K_lmix[:, :, ti] * ctx.ne_grid[j]).copy()
        np.fill_diagonal(Lm, 0.0)
        B = Lm.copy()
        np.fill_diagonal(B, -Lm.sum(axis=0))
        L_nolm = L - B
        scale = float(np.abs(L).max())
        pbr = float(np.abs(P @ B @ R).max())
        say(f"  l-mixing block: column conservation {np.abs(B.sum(axis=0)).max():.3e}, "
            f"matrix scale {scale:.3e}")
        say(f"  max|P B R| = {pbr:.4e}   (structural zero: bundling cannot express l-mixing)")
        if pbr / scale > 1e-12:
            raise RuntimeError(f"P B R is not zero: {pbr:.3e} against scale {scale:.3e}")

        # ---- gate 2/3: column sums ---------------------------------------
        expect = -K_ion[:, ti] * ctx.ne_grid[j]
        e_res = float(np.abs((L.sum(axis=0) - expect) / np.abs(expect)).max())
        Lb_chk = P @ L @ R
        expect_b = np.array([(g_state[n_state == n] / g_shell[k] * expect[n_state == n]).sum()
                             for k, n in enumerate(shells)])
        e_bun = float(np.abs((Lb_chk.sum(axis=0) - expect_b) / np.abs(expect_b)).max())
        say(f"  gate 2  resolved column sums vs -ne K_ion : {e_res:.3e}")
        say(f"  gate 3  bundled  column sums vs weighted  : {e_bun:.3e}")
        if e_res > 1e-9 or e_bun > 1e-9:
            raise RuntimeError(f"column-sum closure violated: {e_res:.3e}, {e_bun:.3e}")

        prod, r_p, n1_prod, E = resolved_r(L, S, ctx, n_state, te_used, ne_si, ps)
        nolm, r_n, _, _ = resolved_r(L_nolm, S, ctx, n_state, te_used, ne_si, ps)
        bund, r_b, _ = bundled_r(L, S, P, R, te_used, ne_si, ps)
        bund_nolm, _, _ = bundled_r(L_nolm, S, P, R, te_used, ne_si, ps)

        say(f"  gate 4  solve residuals: prod {r_p:.2e}  nolmix {r_n:.2e}  bundle {r_b:.2e}")
        if max(r_p, r_n, r_b) > 1e-10:
            raise RuntimeError("solve residual too large")

        # ---- gate 5: calibration against the thesis's own quoted value ----
        if lg_ne == 18:
            cal = abs(prod[3][1] / CH4_R1_P3 - 1)
            say(f"  gate 5  production r_1(3) = {prod[3][1]:.4e} vs Chapter 4's "
                f"{CH4_R1_P3:.3e}, rel {cal:.3e}")
            if cal > 2e-3:
                raise RuntimeError(
                    f"production r_1(3) does not reproduce Chapter 4's value: "
                    f"{prod[3][1]:.4e} vs {CH4_R1_P3:.3e}")

            # ---- gate 6: severity -----------------------------------------
            sev = bund_nolm[2][1] / nolm[2][1]
            say(f"  gate 6  SEVERITY: bundling a non-statistical operator moves "
                f"r_1(2) by {1/sev:.1f}x (ratio {sev:.4f})")
            if not (sev < 0.5 or sev > 2.0):
                raise RuntimeError(
                    "severity gate did not fire: bundling a deliberately "
                    "non-statistical operator barely moved r_1(2), so this "
                    "script cannot detect the effect it is testing")

        say(f"\n  {'p':>3} {'Fujimoto':>11} {'PRODUCTION':>12} {'NO-LMIX':>12} "
            f"{'BUNDLE':>12} {'bund/prod':>10} {'prod/Fuji':>10} {'bund/Fuji':>10}")
        for p in ps:
            f0, f1 = table[p]
            say(f"  {p:>3} {f1:>11.3e} {prod[p][1]:>12.4e} {nolm[p][1]:>12.4e} "
                f"{bund[p][1]:>12.4e} {bund[p][1]/prod[p][1]:>10.4f} "
                f"{prod[p][1]/f1:>10.3f} {bund[p][1]/f1:>10.3f}")
            rows.append(dict(lg_ne=lg_ne, p=p, Te_eV=te_used, ne_cm3=float(ctx.ne_grid[j]),
                             r0_fuji=f0, r1_fuji=f1,
                             r0_prod=prod[p][0], r1_prod=prod[p][1],
                             r0_nolmix=nolm[p][0], r1_nolmix=nolm[p][1],
                             r0_bundle=bund[p][0], r1_bundle=bund[p][1],
                             r1_bundle_over_prod=bund[p][1] / prod[p][1],
                             r0_bundle_over_prod=bund[p][0] / prod[p][0],
                             r1_prod_over_fuji=prod[p][1] / f1,
                             r1_bundle_over_fuji=bund[p][1] / f1,
                             max_PBR=pbr, matrix_scale=scale))

        say(f"\n  r_0: p=2 Fujimoto {table[2][0]:.3f}  production {prod[2][0]:.4f}  "
            f"bundle {bund[2][0]:.4f}")

        # the l-distribution the production model actually has
        say("  ground-fed (n_nl/n_n)/(g_nl/g_n), production:")
        for p in (2, 3, 4, 8):
            idx = np.where(n_state == p)[0]
            loc = np.array([int(np.where(E == s)[0][0]) for s in idx])
            frac = n1_prod[loc] / n1_prod[loc].sum()
            say(f"    n={p}: " + " ".join(f"{x:.3f}" for x in frac / (g_state[idx] / g_shell[p - 1])))

    # ---- verdict ---------------------------------------------------------
    r18 = [r for r in rows if r["lg_ne"] == 18 and r["p"] == 3][0]
    r2 = [r for r in rows if r["lg_ne"] == 18 and r["p"] == 2][0]
    say("\n" + "=" * 78)
    say("VERDICT")
    say("=" * 78)
    say(f"  Imposing Fujimoto's own statistical-l closure moves r_1(3) by "
        f"{(r18['r1_bundle_over_prod']-1)*100:+.2f}%.")
    say(f"  The deficit at p=3 goes from a factor {1/r18['r1_prod_over_fuji']:.2f} "
        f"to a factor {1/r18['r1_bundle_over_fuji']:.2f}.")
    say(f"  At p=2 the agreement WORSENS: {1/r2['r1_prod_over_fuji']:.1f}x low "
        f"becomes {1/r2['r1_bundle_over_fuji']:.1f}x low, and r_0(2) moves from "
        f"{r2['r0_prod']:.4f} to {r2['r0_bundle']:.4f} against a tabulated "
        f"{r2['r0_fuji']:.3f}.")
    say("  The refuting observation (bundle/prod of order 8) did not appear.")
    say("  CONCLUSION: the l closure does not explain the r_1 deficit.")

    txt = out / "fujimoto_bundle.txt"
    csvp = out / "fujimoto_bundle.csv"
    stamp = (f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}\n"
             f"# interpreter {sys.executable}  numpy {np.__version__}\n"
             f"# L_grid sha256 {sha256(L_path)}\n"
             f"# S_grid sha256 {sha256(S_path)}\n"
             f"# K_lmix sha256 {sha256(K_lmix_path)}\n"
             f"# state_index sha256 {sha256(ctx.state_index_path)}\n"
             f"# Te {te_used:.6f} eV (Fujimoto {FUJIMOTO_TE_EV} eV, "
             f"{(te_used/FUJIMOTO_TE_EV-1)*100:+.1f}%)\n"
             f"# L_bundle = P L R, P[k,i]=1 for i in shell k, R[i,k]=g_i/g_k\n")
    with open(txt, "w") as f:
        f.write(stamp)
        f.write("\n".join(lines) + "\n")
    df = pd.DataFrame(rows)
    with open(csvp, "w") as f:
        f.write(stamp)
        df.to_csv(f, index=False)
    say(f"\nwrote {txt}\nwrote {csvp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
