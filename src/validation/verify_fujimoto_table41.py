"""
verify_fujimoto_table41.py
==========================
External consistency benchmark of the model's QSS population coefficients
r_0(p), r_1(p) against Fujimoto, *Plasma Spectroscopy* (2004), Table 4.1(b).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It IS: a comparison of the same coefficient definition, computed from this
model's own matrix, against independently published values.

It is NOT a validation of the atomic data, and not a reproduction of
Fujimoto's model. His calculation uses a different atomic-data stack
(excitation as well as ionization), a different state resolution, and a
different level cutoff. Exact 1%-level agreement is not expected and would be
suspicious. See KNOWN OFFSETS below.

Draft 1 of this script (23 Aug 2026) was reviewed externally and carried a
critical units error plus two conceptual errors. All are corrected here; the
draft is kept for the record. Corrections listed at the end of this docstring.

THE COEFFICIENTS
----------------
Fujimoto eq. (4.20) decomposes the excited populations into two supply
channels:

    n(p) = r_0(p) Z(p) n_e n_z  +  r_1(p) [Z(p)/Z(1)] n(1)

The same split falls out of this model's matrix. Setting dn_E/dt = 0,

    0 = L_EE n_E + L_Eg n_g + S_E
    n_E = -L_EE^{-1} S_E  +  -L_EE^{-1} L_Eg n_g
        =      n^(0)      +        n^(1)

so, summing over l within each shell (Fujimoto's n(p) is a shell population):

    r_0(p) = [n^(0)_p / n_z] / [Z(p) n_e]        both factors dimensionless
    r_1(p) = [n^(1)_p / n_g] * [Z(1) / Z(p)]

Saha-Boltzmann coefficient for neutral hydrogen. g_p = 2p^2 (p^2 orbital
states x 2 for spin); g_i = 1 for a bare proton; the 2 in the denominator is
free-electron spin -- the two 2s cancel, leaving p^2:

    Z(p) = p^2 (h^2 / 2 pi m_e k T_e)^{3/2} exp[ (13.605/p^2) / T_e(eV) ]

The exponent is POSITIVE: Fujimoto writes n_p = Z(p) n_e n_i, the inverse of
the usual Saha form n_e n_i / n_p ~ exp(-chi/kT).

Units: n^(0), n^(1), n_z, n_g all in cm^-3 (the solve uses n_ion = 1 cm^-3,
confirmed: S_grid = n_e(alpha_RR + n_e alpha_3BR) in s^-1, so b = S * n_ion).
Z(p) is in m^3, so n_e must be converted to m^-3. Writing r_0 as a ratio of
two dimensionless groups makes the cm/m boundary explicit and is what draft 1
got wrong -- it divided by n_z = 1 without converting, giving r_0 too large by
exactly 10^6.

TWO RUNS -- AND NEITHER IS A STRICT FUJIMOTO BENCHMARK
------------------------------------------------------
Fujimoto's Ch. 4 model omits ion-induced transitions AND assumes statistical
populations among l-states of the same n *before* writing the rate equations.
His dynamical variable is therefore the shell, not the sublevel.

This model keeps 43 independent nl states for n <= 8. So:

  NO-PROTON-LMIX  proton l-mixing subtracted. Process list closer to
                  Fujimoto, but the l-distribution is then free to go
                  non-statistical -- which he does not permit.
  PRODUCTION      proton l-mixing retained. Contains a process he omits, but
                  drives the resolved states toward the statistical-l
                  distribution he assumes.

Neither reproduces his state-space assumption. Draft 1 called the first
"BENCHMARK -- matches Fujimoto's process set"; that claim is withdrawn. A
strict benchmark would require building shell-level rates under statistical-l
weighting, C(n->n') = sum_l (g_nl/g_n) sum_l' C(nl->n'l'), and solving that
bundled operator -- not done here.

p = 10 AND p = 15 ARE THE CLEANEST COMPARISON
---------------------------------------------
This model bundles n = 9-15 with statistical l-equilibrium -- exactly
Fujimoto's assumption. At p = 10 and p = 15 the state-space mismatch
disappears. And r_1 is the discriminating quantity there, since the tabulated
r_0 = 1.00 is rounded and cannot distinguish anything.

THE l-MIXING SUBTRACTION
------------------------
assemble_cr_matrix.py (lines 179-183) adds l-mixing as a separate additive
block conserving column sums. It is reconstructed explicitly here and
subtracted, with two genuine invariants checked:
  reconstruction: max|(L_nolm + B) - L|          must be ~0
  conservation:   max|colsum(B)|                  must be ~0
Draft 1 instead checked that entries l-mixing touched became zero after
subtraction. That is not a valid invariant -- another process may contribute
to the same (i,k) entry.

PREDICTION, recorded 23 Aug 2026 before the corrected run
----------------------------------------------------------
Student: the l-mixing on/off difference is SMALL. Grounds: backlog A6
measured the 4F fraction of the n=4 shell at 0.4361-0.4375 against the
statistical 0.4375, and the l-mixing F(U_m) correction moved tau_relax by
0.12% while changing the underlying rates by factors of 3-7 -- i.e. the
observable is saturated.

Draft-1 result: confirmed at lg n_e = 21 (r_0 ratios 0.912 to 0.99999,
improving with p) and REFUTED at lg n_e = 18 (factor 140 at p = 2).

The low-density divergence is NOT YET EXPLAINED. Draft 1 attributed it to
Fujimoto's statistical-l assumption failing there; that is a different
question -- Fujimoto does not appear in a comparison of this model against
itself. Note also that the l-mixing block scales as n_e, so its absolute
dynamical weight FALLS with density. A large on/off difference at low density
therefore requires its own mechanism, not yet identified.

Claude's Te-direction prediction (r_0 below the tabulated values because this
grid is 9.3% cooler) held at lg n_e = 21 -- every ratio below 1, monotone in
p. Recorded as a prediction that held, NOT as an acceptance criterion: r_0 is
a normalized CR coefficient involving recombination, collisional
redistribution, ionization and Z_p(T), and the sign of a 10 -> 11.03 eV change
does not follow from a one-process argument.

KNOWN OFFSETS -- state these with any number this script produces
----------------------------------------------------------------
  1. TEMPERATURE. Table 4.1(b) is at T_e = 1.28e5 K = 11.03 eV; this grid tops
     out at 10.0 eV, 9.3% lower. A clean comparison would assemble one extra
     operator at exactly 11.03 eV. Not done.
  2. ATOMIC DATA. Fujimoto uses a different excitation dataset as well as a
     different ionization dataset. This is not "the same model, independently
     computed".
  3. STATE RESOLUTION. His n(p) is a shell population under an assumed
     statistical l-distribution; this model resolves l for n <= 8.
  4. LEVEL CUTOFF. This model terminates at n = 15. Population coefficients of
     lower shells can depend on cascade and recombination flux through omitted
     Rydberg states; Fujimoto notes substantial contributions from p > 10 even
     at n_e = 1e18 m^-3. An n_max convergence test is required before any
     agreement here is claimed as validation.
  5. DENSITY ALIGNMENT. Table rows run lg n_e = 12, 14, 16, 17, ..., 24 in
     m^-3. Only 18 (1e12 cm^-3, grid j=0) and 21 (1e15 cm^-3, grid j=7)
     coincide with grid points.
  6. TABLE COVERAGE. Only 4.1(a) at 1e3 K = 0.086 eV and 4.1(b) at 1.28e5 K =
     11.03 eV exist. 4.1(a) is far below this grid. There is no tabulated case
     inside 1-10 eV.

WHAT THE TWO DENSITIES TEST
---------------------------
Measured 3BR share of the recombination source at T_e = 10 eV:
  lg n_e = 18: 0.000 (1S) to 0.096 (4F) -- essentially pure RADIATIVE
               recombination. Tests alpha_RR and the cascade through L_EE^-1.
  lg n_e = 21: 0.006 (1S) to 0.991 (4F) -- ~99% THREE-BODY. Tests alpha_3BR.

CORRECTIONS TO DRAFT 1 (external review, 23 Aug 2026)
-----------------------------------------------------
  critical  r_0 normalization off by 10^6 (n_z = 1 cm^-3 not converted)
  major     "BENCHMARK matches Fujimoto's process set" -- withdrawn
  major     low-density l-mixing divergence mis-explained -- now open
  medium    r_1 extracted via the full solve -- now from unit n_g directly
  medium    superposition check was near-tautological -- now channel residuals
  medium    l-mixing diagnostic was not an invariant -- now reconstruction
            and column conservation of the explicit block
  medium    n_max = 15 offset missing from the list
  medium    atomic-data mismatch understated as "ionization only"
  minor     header printed the requested Te, not the snapped grid Te
  minor     EV_TO_J was misnamed K_B_J_PER_EV
  added     p = 10 and p = 15 (bundled shells -- the cleanest comparison)

Table values transcribed from the scan and re-read at full resolution
23 Aug 2026. All 14 entries verified, including the coronal row used for the
unit check.

RE-CHECK AGAINST THE PRINTED TABLE, 10 Sep 2026: DONE, and the transcription is
exact. The lg n_e = 18 row reads r_0 = 0.730 / 0.835 / 0.947 / 0.983 and
r_1 = 1.79e-4 / 5.72e-5 / 1.66e-5 / 5.08e-6 at p = 2, 3, 4, 5, matching the
values below. Three further things the source settles:

  * lg n_e IS log10(n_e / m^-3). The table proves it internally: r_0(2) reaches
    0.981 at lg n_e = 21, and Griem's criterion puts LTE for n=2 at 11.03 eV at
    1.74e16 cm^-3. Read as m^-3 that onset is 1e15 cm^-3, the right order; read
    as cm^-3 it would sit five orders above Griem. The UNIT CHECK below reaches
    the same conclusion by a circular route and should be replaced by this one.

  * p IS A SHELL. The columns are p = 2, 3, 4, 5, 7, 10, 15, principal quantum
    numbers, with no l label anywhere in either table. Fujimoto's coefficients
    are bundled in l; this model resolves l below n=9. That is the whole of the
    low-p r_1 discrepancy: removing proton l-mixing here moves r_1(2) by 118x
    at this density, so the comparison tests the l-closure and cannot adjudicate
    a factor of 8.

  * NO SECOND USABLE CASE. 4.1(b) is at 1.28e5 K = 11.03 eV against a grid that
    stops at 10 eV; 4.1(a) is at 1e3 K = 0.0862 eV, two orders below it.

Report only. Writes to the output directory; modifies nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_context import CRContext  # noqa: E402

# ---------------------------------------------------------------------------
# Fujimoto Table 4.1(b), T_e = 1.28e5 K = 11.03 eV.  lg n_e = log10(n_e / m^-3)
# entries are (r_0, r_1)
# ---------------------------------------------------------------------------
FUJIMOTO_TE_K = 1.28e5
FUJIMOTO_TE_EV = 11.03
TABLE_41B = {
    18: {2: (7.30e-1, 1.79e-4), 3: (8.35e-1, 5.72e-5), 4: (9.47e-1, 1.66e-5),
         5: (9.83e-1, 5.08e-6), 7: (9.97e-1, 7.55e-7), 10: (1.00, 9.61e-8),
         15: (1.00, 9.34e-9)},
    21: {2: (9.81e-1, 7.70e-3), 3: (9.98e-1, 8.02e-4), 4: (1.00, 1.40e-4),
         5: (1.00, 3.63e-5), 7: (1.00, 4.80e-6), 10: (1.00, 5.80e-7),
         15: (1.00, 5.50e-8)},
}
# coronal asymptote, n_e -> 0 row, p = 2:  r_1(2) -> 1.93e-22 * n_e [m^-3]
CORONAL_R1_P2 = 1.93e-22
CORONAL_CHECK_LG = 12          # the table's own lg n_e = 12 row
CORONAL_CHECK_VAL = 1.93e-10   # what that row reads

H_PLANCK = 6.62607015e-34
M_E = 9.1093837015e-31
EV_TO_J = 1.602176634e-19
CHI_H_EV = 13.605693

NZ_CM3 = 1.0   # the solve uses n_ion = 1 cm^-3
NG_CM3 = 1.0   # unit ground density for the r_1 channel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--te", type=float, default=10.0,
                   help="Te in eV (default 10.0, the grid top, nearest to "
                        "Fujimoto's 11.03 eV)")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saha_boltzmann_Z(p: int, te_ev: float) -> float:
    """Z(p) in m^3 for neutral hydrogen: p^2 (h^2/2 pi m_e kT)^{3/2}
    exp(chi_p / kT). The 2s from g_p = 2p^2 and free-electron spin cancel."""
    kT = te_ev * EV_TO_J
    thermal = (H_PLANCK ** 2 / (2.0 * np.pi * M_E * kT)) ** 1.5
    return p ** 2 * thermal * np.exp((CHI_H_EV / p ** 2) / te_ev)


def main():
    a = parse_args()
    ctx = CRContext.load()
    root = ctx.root
    L_path = root / "data/processed/cr_matrix/L_grid.npy"
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    K_path = root / "data/processed/lmix/K_lmix.npy"
    for q in (S_path, K_path):
        if not q.exists():
            raise FileNotFoundError(f"missing: {q}")

    L_grid = ctx.L_grid
    S_grid = np.load(S_path)
    K_lmix = np.load(K_path)
    Te, ne = ctx.te_grid, ctx.ne_grid
    n_vals = np.asarray(ctx.n_values)
    g = int(ctx.ground_index)
    E = np.array([i for i in range(ctx.n_states) if i != g], dtype=int)

    ti = int(np.argmin(np.abs(Te - a.te)))
    te_used = float(Te[ti])

    out = a.out or (root / "validation" / "fujimoto_table41")
    out.mkdir(parents=True, exist_ok=True)
    lines, rows = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("FUJIMOTO TABLE 4.1(b) -- external consistency benchmark of r_0, r_1")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"repo root      {root}")
    say(f"L_grid sha256  {sha256(L_path)}")
    say(f"S_grid sha256  {sha256(S_path)}")
    say(f"K_lmix sha256  {sha256(K_path)}")
    say(f"state idx sha  {sha256(ctx.state_index_path)}")
    say("")
    say("NOT a validation of the atomic data and NOT a reproduction of")
    say("Fujimoto's model: different excitation and ionization datasets,")
    say("different state resolution, different level cutoff. Exact agreement")
    say("would be suspicious.")
    say("")
    say(f"Table 4.1(b): Te = {FUJIMOTO_TE_K:.2e} K = {FUJIMOTO_TE_EV} eV")
    say(f"model evaluated at Te = {te_used:.6f} eV (grid index {ti}), "
        f"{(te_used/FUJIMOTO_TE_EV - 1)*100:+.1f}% offset")
    say("=" * 78)

    # ---- unit check, at the table's own deep-coronal row -------------------
    say("\nUNIT CHECK on Table 4.1(b) lg n_e")
    pred = CORONAL_R1_P2 * 10.0 ** CORONAL_CHECK_LG
    say(f"  coronal asymptote r_1(2) -> {CORONAL_R1_P2:.3e} * n_e")
    say(f"  at lg n_e = {CORONAL_CHECK_LG}: predicts {pred:.3e}, "
        f"table reads {CORONAL_CHECK_VAL:.3e}  "
        f"({(pred/CORONAL_CHECK_VAL - 1)*100:+.2f}%)")
    say(f"  (the lg n_e = 18 row is {(CORONAL_R1_P2*1e18/TABLE_41B[18][2][1]-1)*100:+.1f}% "
        f"off the asymptote -- 1e18 m^-3 is no longer coronal, which is why")
    say("   the units must be settled at lg n_e = 12, not 18)")
    say("  => lg n_e = log10( n_e / m^-3 );  1e18 m^-3 = 1e12 cm^-3")

    for lg_ne, table in sorted(TABLE_41B.items()):
        ne_cm3 = 10.0 ** lg_ne / 1e6
        j = int(np.argmin(np.abs(np.log(ne) - np.log(ne_cm3))))
        rel = abs(ne[j] / ne_cm3 - 1)
        say("\n" + "=" * 78)
        say(f"lg n_e = {lg_ne}  ->  {ne_cm3:.3e} cm^-3   grid j={j} at "
            f"{ne[j]:.6e} cm^-3   (offset {rel*100:.4f}%)")
        if rel > 1e-6:
            say("  SKIP: does not coincide with a grid point.")
            continue

        L = L_grid[ti, j]
        S = S_grid[ti, j]
        ne_si = ne[j] * 1e6

        # ---- explicit l-mixing block, with genuine invariants -------------
        Lm_off = (K_lmix[:, :, ti] * ne[j]).copy()
        np.fill_diagonal(Lm_off, 0.0)
        B = Lm_off.copy()
        np.fill_diagonal(B, -Lm_off.sum(axis=0))
        L_nolm = L - B
        recon = float(np.abs((L_nolm + B) - L).max())
        conserv = float(np.abs(B.sum(axis=0)).max())
        scale = float(np.abs(L).max())
        say(f"  l-mixing block: reconstruction {recon:.3e}  "
            f"column conservation {conserv:.3e}  (matrix scale {scale:.3e})")
        if conserv / scale > 1e-12:
            say("  *** the l-mixing block does not conserve column sums -- "
                "the subtraction is WRONG ***")

        for tag, Lop in (("NO-PROTON-LMIX", L_nolm), ("PRODUCTION", L)):
            LEE = Lop[np.ix_(E, E)]
            LEg = Lop[np.ix_(E, [g])].ravel()

            n0 = np.linalg.solve(LEE, -S[E])            # recombination-fed
            n1 = np.linalg.solve(LEE, -LEg * NG_CM3)    # ground-fed, unit n_g

            res0 = (np.linalg.norm(LEE @ n0 + S[E])
                    / max(np.linalg.norm(S[E]), 1e-300))
            res1 = (np.linalg.norm(LEE @ n1 + LEg * NG_CM3)
                    / max(np.linalg.norm(LEg * NG_CM3), 1e-300))
            cond = float(np.linalg.cond(LEE))
            neg0 = int((n0 < -1e-10 * np.abs(n0).max()).sum())
            neg1 = int((n1 < -1e-10 * np.abs(n1).max()).sum())

            say(f"\n  --- {tag} ---")
            say(f"      channel residuals  r0 {res0:.3e}   r1 {res1:.3e}   "
                f"cond(L_EE) {cond:.3e}")
            if neg0 or neg1:
                say(f"      *** negative populations: n0 {neg0}, n1 {neg1} ***")
            say(f"      {'p':>3} {'r0 model':>10} {'r0 Fuji':>8} {'ratio':>7}"
                f"   {'r1 model':>11} {'r1 Fuji':>10} {'ratio':>7}")

            Z1 = saha_boltzmann_Z(1, te_used)
            for p in sorted(table):
                idx = np.where(n_vals == p)[0]
                if len(idx) == 0:
                    say(f"      {p:>3}  (no states with n={p} in this model)")
                    continue
                loc = np.array([int(np.where(E == s)[0][0]) for s in idx])
                n0p, n1p = float(n0[loc].sum()), float(n1[loc].sum())
                Zp = saha_boltzmann_Z(p, te_used)

                r0 = (n0p / NZ_CM3) / (Zp * ne_si)
                r1 = (n1p / NG_CM3) * (Z1 / Zp)
                f0, f1 = table[p]
                say(f"      {p:>3} {r0:>10.4f} {f0:>8.3f} {r0/f0:>7.4f}"
                    f"   {r1:>11.4e} {f1:>10.3e} {r1/f1:>7.4f}")
                if r0 > 1.05:
                    # NOT unphysical, and an earlier version of this line said
                    # it was. r_0 > 1 is expected once the radiative sink is
                    # removed: three-body recombination and collisional
                    # ionisation balance to exactly Saha, so radiative
                    # recombination is an unbalanced source and
                    # r_0(p) = 1 + alpha_RR / (n_e Z(p) S_ion). In the
                    # NO-PROTON-LMIX variant 2s additionally has no radiative
                    # exit in this dataset, so r_0(2) diverges as n_e -> 0.
                    # It is a property of a deliberately broken variant.
                    say(f"          note: r_0 = {r0:.4f} > 1. Expected for an "
                        f"unbalanced radiative-recombination source; see the "
                        f"module docstring.")
                rows.append(dict(lg_ne=lg_ne, ne_cm3=ne[j], Te_eV=te_used,
                                 variant=tag, p=p, n_sublevels=len(idx),
                                 r0_model=r0, r0_fuji=f0, r0_ratio=r0/f0,
                                 r1_model=r1, r1_fuji=f1, r1_ratio=r1/f1,
                                 res0=res0, res1=res1, cond_LEE=cond))

    # ---- l-mixing effect --------------------------------------------------
    say("\n" + "=" * 78)
    say("l-MIXING EFFECT -- production / no-proton-lmix")
    say("Student's prediction: SMALL. Note the l-mixing block scales as n_e,")
    say("so its absolute weight FALLS with density; a large low-density")
    say("difference needs its own mechanism and is currently UNEXPLAINED.")
    say(f"  {'lg ne':>6} {'p':>3} {'r0 prod/nolmix':>16} {'r1 prod/nolmix':>16}")
    for lg in sorted(TABLE_41B):
        for p in sorted(TABLE_41B[lg]):
            b = [r for r in rows if r["lg_ne"] == lg and r["p"] == p
                 and r["variant"] == "NO-PROTON-LMIX"]
            q = [r for r in rows if r["lg_ne"] == lg and r["p"] == p
                 and r["variant"] == "PRODUCTION"]
            if b and q and b[0]["r0_model"] and b[0]["r1_model"]:
                say(f"  {lg:>6} {p:>3} "
                    f"{q[0]['r0_model']/b[0]['r0_model']:>16.6f} "
                    f"{q[0]['r1_model']/b[0]['r1_model']:>16.6f}")

    say("\nNOTE p=10 and p=15 are BUNDLED shells in this model, with")
    say("statistical l-equilibrium assumed -- the same closure Fujimoto uses.")
    say("The state-space mismatch vanishes there, so those rows are the")
    say("cleanest comparison. r_1 is the discriminating quantity, since the")
    say("tabulated r_0 = 1.00 is rounded.")

    txt, csv = out / "fujimoto_table41.txt", out / "fujimoto_table41.csv"
    txt.write_text("\n".join(lines) + "\n")
    if rows:
        keys = list(rows[0])
        with csv.open("w") as f:
            f.write(f"# generated {datetime.now():%Y-%m-%d %H:%M} by "
                    f"{Path(__file__).name}\n")
            f.write(f"# L_grid sha256 {sha256(L_path)}\n")
            f.write(f"# S_grid sha256 {sha256(S_path)}\n")
            f.write(f"# Fujimoto Table 4.1(b) Te = {FUJIMOTO_TE_EV} eV; "
                    f"model at Te = {te_used:.4f} eV\n")
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
    say(f"\nwrote {txt}")
    say(f"wrote {csv}")


if __name__ == "__main__":
    main()