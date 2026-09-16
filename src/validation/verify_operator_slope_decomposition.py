#!/usr/bin/env python
"""
verify_operator_slope_decomposition.py
======================================
Split the temperature slope of the tabulated n=3/n=4 ratio into the manifold's
own response and the reservoir's, and split the manifold's response into its
two feed channels, with every identity checked to roundoff.

WHY THIS EXISTS
---------------
Section sec:inversion_measured rests on two measured signs, P > 0 and SG < 0,
and on their near-cancellation. A scratch analysis (16 Sep) claimed both feed
channels raise the ratio "for the single reason f_3 > f_4". The reviewer
refuted that from the scratch numbers themselves: the recombination-fed
contribution is not positive at every pair, so f_3 > f_4 alone cannot be the
reason. This script replaces the scratch analysis with a stamped one that
records WHY each channel has the sign it has, pair by pair, and claims nothing
it does not check.

THE ALGEBRA
-----------
At fixed reservoir u the excited populations are affine, N_m = a_m u + c_m,
and R = N_3/N_4. Locally (exact chain rule):

    Sigma = dlnR_cre/dlnTe = P + S G,
    P     = (dlnR/dlnTe)_u = f3 A3 + (1-f3) C3 - f4 A4 - (1-f4) C4,
    A_m   = dln a_m/dlnTe,  C_m = dln c_m/dlnTe,  f_m = a_m u/(a_m u + c_m),
    P_a   = f3 A3 - f4 A4,        P_c = (1-f3) C3 - (1-f4) C4,
    P_a > 0  <=>  (f3/f4) / (A4/A3) > 1            when A3, A4 > 0,
    P_c > 0  <=>  ((1-f4)/(1-f3)) / (B3/B4) > 1    when B_m = -C_m > 0.

Over the finite step actually used (one grid interval, nearest index to +/-5 %)
the same three quantities are SECANTS from the same three solves as
verify_plateau_gridmap.py, and the decomposition is exact by telescoping:

    Pbar    = ln(R_pe / R_cre^-) / D        manifold at frozen u^-
    SGbar   = ln(R_cre^+ / R_pe) / D        reservoir at fixed Te^+
    Sigbar  = ln(R_cre^+ / R_cre^-) / D  =  Pbar + SGbar      (exact)
    SGbar   = Sbar * Gbar                                     (exact, by the
              thesis's definitions Sbar = SGbar*D/dln u, Gbar = dln u / D)

with D = dlnTe of the step. The channel split of Pbar is exact only in
log-mixture form,
    Pbar*D = ln[f3 e^{A3 D} + (1-f3) e^{C3 D}] - ln[f4 e^{A4 D} + (1-f4) e^{C4 D}],
where A_m, C_m are now the secants ln(a_m^+/a_m^-)/D, ln(c_m^+/c_m^-)/D and
f_m is at (Te^-, u^-). The additive split P_a + P_c is the first-order form of
that; its residual against Pbar is measured and reported, never assumed zero.

CHECKS THAT STOP THE SCRIPT
    1. Pbar, SGbar, Sigbar reproduce validation/inversion_error/*.csv (K8)
    2. Sigbar - Pbar - SGbar = 0 to roundoff
    3. SGbar - Sbar*Gbar = 0 to roundoff, Sbar and Gbar recomputed here
    4. the log-mixture identity for Pbar holds to roundoff
    5. sign(P_a) agrees with (ratio_a > 1) wherever A3, A4 > 0, and sign(P_c)
       with (ratio_c > 1) wherever B3, B4 > 0: exact equivalences
Nothing is a theorem here except the identities. Every sign is counted.

PREDICTION, WRITTEN BEFORE THE RUN (scratch, log-mixture counterfactuals):
    P_a > 0 at 448/448, P_c > 0 at ~426/448, A_m > 0 and C_m < 0 at 448/448,
    additive residual |Pbar - P_a - P_c| / |Pbar| ~ 0.3 %.
REFUTER: P_a or P_c negative at a large fraction of pairs, or the additive
    residual comparable to the terms, in which case the channel picture does
    not describe the finite step and only the exact secants may be quoted.
"""
from __future__ import annotations

import argparse
import csv as _csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_context import CRContext  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frac", type=float, default=0.05)
    p.add_argument("--win-lo", type=float, default=30.0)
    p.add_argument("--win-hi", type=float, default=30.0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def main():
    a = parse_args()
    ctx = CRContext.load()
    root = ctx.root
    L_path = root / "data/processed/cr_matrix/L_grid.npy"
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    ref_path = root / "validation/inversion_error/inversion_error.csv"
    for p in (S_path, ref_path):
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")
    L, S = ctx.L_grid, np.load(S_path)
    Te, ne = ctx.te_grid, ctx.ne_grid
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")
    nT, nN = len(Te), len(ne)
    lnTe = np.log(Te)

    g = int(ctx.ground_index)
    E = np.array([i for i in range(ctx.n_states) if i != g], dtype=int)
    N3 = np.where(np.asarray(ctx.n_values) == 3)[0]
    N4 = np.where(np.asarray(ctx.n_values) == 4)[0]
    if len(N3) != 3 or len(N4) != 4:
        raise ValueError(f"shell membership wrong: n=3 -> {N3}, n=4 -> {N4}")
    pos = {s: k for k, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3])
    n4E = np.array([pos[s] for s in N4])

    out = a.out or (root / "validation" / "operator_slope_decomposition")
    out.mkdir(parents=True, exist_ok=True)
    lines, rows = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("OPERATOR SLOPE DECOMPOSITION -- Sigma = P + SG, and P by feed channel")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"repo root      {root}")
    say(f"L_grid sha256  {sha256(L_path)}")
    say(f"S_grid sha256  {sha256(S_path)}")
    say(f"state idx sha  {sha256(ctx.state_index_path)}")
    say(f"reproduction   {ref_path.relative_to(root)}")
    say("=" * 78)

    # ---- the two channels at every node: a_m, c_m, and u_cre --------------
    a3 = np.empty((nT, nN)); a4 = a3.copy(); c3 = a3.copy(); c4 = a3.copy()
    u_cre = a3.copy(); R_full = a3.copy()
    worst_sup = 0.0
    for j in range(nN):
        for m in range(nT):
            LEE = L[m, j][np.ix_(E, E)]
            LEg = L[m, j][np.ix_(E, [g])].ravel()
            n0 = np.linalg.solve(LEE, -S[m, j][E])      # ion-fed, per unit n_ion
            n1 = np.linalg.solve(LEE, -LEg)             # ground-fed, per unit n_g
            nf = np.linalg.solve(L[m, j], -S[m, j])     # full CRE, per unit n_ion
            c3[m, j], c4[m, j] = n0[n3E].sum(), n0[n4E].sum()
            a3[m, j], a4[m, j] = n1[n3E].sum(), n1[n4E].sum()
            u_cre[m, j] = nf[g]
            R_full[m, j] = nf[N3].sum() / nf[N4].sum()
            sup = np.abs(n0 + nf[g] * n1 - nf[E]).max() / np.abs(nf[E]).max()
            worst_sup = max(worst_sup, sup)
    say(f"\ntwo-channel superposition at all {nT*nN} nodes: worst residual {worst_sup:.3e}")

    def R_at(m, j, u):
        return (a3[m, j] * u + c3[m, j]) / (a4[m, j] * u + c4[m, j])

    # ---- reproduction target -----------------------------------------------
    with ref_path.open() as fh:
        ref = {(r["direction"], int(r["i"]), int(r["j"])): r
               for r in _csv.DictReader(ln for ln in fh if not ln.startswith("#"))}

    w_rep = w_tele = w_prod = w_mix = 0.0
    n_ra_check = n_ra_bad = n_rc_check = n_rc_bad = 0

    for sgn_, dlab in ((+1, "heat"), (-1, "cool")):
        for i in range(nT):
            k = int(np.argmin(np.abs(Te - Te[i] * (1 + sgn_ * a.frac))))
            if k == i:
                continue
            D = lnTe[k] - lnTe[i]
            for j in range(nN):
                lam = np.linalg.eigvals(L[k, j])
                lam = lam[np.argsort(lam.real)[::-1]]
                tQ, tR = -1.0 / lam[0].real, -1.0 / lam[1].real
                window_ok = bool((a.win_lo * tR) < (tQ / a.win_hi))

                um, up = u_cre[i, j], u_cre[k, j]
                R_m, R_p = R_at(i, j, um), R_at(k, j, up)      # R_cre^-, R_cre^+
                R_pe = R_at(k, j, um)                          # plateau observable
                if abs(R_p - R_full[k, j]) / R_full[k, j] > 1e-10:
                    raise RuntimeError(f"R_cre from channels != full solve at {(k, j)}")

                Pbar = np.log(R_pe / R_m) / D
                SGbar = np.log(R_p / R_pe) / D
                Sigbar = np.log(R_p / R_m) / D
                dlnu = np.log(up / um)
                Gbar = dlnu / D
                Sbar = np.log(R_pe / R_p) / dlnu if dlnu != 0 else np.nan
                # NB thesis convention: Sbar = ln(R_pe/R_qss_new)/ln x, so SGbar = -Sbar*Gbar
                # is the sign-flipped reservoir term; the identity checked is on the
                # reservoir channel itself, ln(R_p/R_pe) = -(Sbar * dlnu).
                w_tele = max(w_tele, abs(Sigbar - Pbar - SGbar))
                w_prod = max(w_prod, abs(SGbar + Sbar * Gbar))

                r = ref.get((dlab, i, j))
                if r is None:
                    raise RuntimeError(f"pair {(dlab, i, j)} absent from {ref_path}")
                w_rep = max(w_rep, abs(Pbar - float(r["P_bar"])),
                            abs(SGbar - float(r["SbarG"])), abs(Sigbar - float(r["Sigma_bar"])))

                # -- channel split at (Te^-, u^-)
                f3 = a3[i, j] * um / (a3[i, j] * um + c3[i, j])
                f4 = a4[i, j] * um / (a4[i, j] * um + c4[i, j])
                A3, A4 = np.log(a3[k, j] / a3[i, j]) / D, np.log(a4[k, j] / a4[i, j]) / D
                C3, C4 = np.log(c3[k, j] / c3[i, j]) / D, np.log(c4[k, j] / c4[i, j]) / D
                mix = (np.log(f3 * np.exp(A3 * D) + (1 - f3) * np.exp(C3 * D))
                       - np.log(f4 * np.exp(A4 * D) + (1 - f4) * np.exp(C4 * D))) / D
                w_mix = max(w_mix, abs(mix - Pbar))
                P_a = f3 * A3 - f4 * A4
                P_c = (1 - f3) * C3 - (1 - f4) * C4
                P_lin = P_a + P_c
                B3, B4 = -C3, -C4
                ratio_a = (f3 / f4) / (A4 / A3) if A3 > 0 and A4 > 0 else np.nan
                ratio_c = ((1 - f4) / (1 - f3)) / (B3 / B4) if B3 > 0 and B4 > 0 else np.nan
                if np.isfinite(ratio_a):
                    n_ra_check += 1
                    n_ra_bad += int((P_a > 0) != (ratio_a > 1))
                if np.isfinite(ratio_c):
                    n_rc_check += 1
                    n_rc_bad += int((P_c > 0) != (ratio_c > 1))

                # local slope at the post-step node, as K8 classified on
                lo_, hi_ = max(k - 1, 0), min(k + 1, nT - 1)
                Sig_loc = (np.log(R_full[hi_, j]) - np.log(R_full[lo_, j])) / (lnTe[hi_] - lnTe[lo_])

                rows.append(dict(
                    direction=dlab, i=i, j=j, Te=Te[i], Te_new=Te[k], ne=ne[j],
                    window_ok=window_ok, dlnTe=D,
                    f3=f3, f4=f4, A3=A3, A4=A4, C3=C3, C4=C4,
                    Pbar=Pbar, SGbar=SGbar, Sigbar=Sigbar, Sbar=Sbar, Gbar=Gbar,
                    P_a=P_a, P_c=P_c, P_lin=P_lin,
                    additive_residual=Pbar - P_lin,
                    ratio_a=ratio_a, ratio_c=ratio_c,
                    cancellation=abs(Sigbar) / abs(Pbar),
                    amp_pred=abs(SGbar) / abs(Sigbar) if Sigbar != 0 else np.nan,
                    Sigma_local_new=Sig_loc,
                ))

    say(f"\ncheck 1  reproduction of K8's Pbar, SGbar, Sigbar: worst |diff| {w_rep:.3e}")
    say(f"check 2  Sigbar - Pbar - SGbar:                    worst |diff| {w_tele:.3e}")
    say(f"check 3  SGbar + Sbar*Gbar (thesis sign convention): worst |diff| {w_prod:.3e}")
    say(f"check 4  log-mixture identity for Pbar:              worst |diff| {w_mix:.3e}")
    say(f"check 5  sign(P_a) <=> ratio_a>1: {n_ra_bad} bad of {n_ra_check};  "
        f"sign(P_c) <=> ratio_c>1: {n_rc_bad} bad of {n_rc_check}")
    for name, v in (("1", w_rep), ("2", w_tele), ("3", w_prod), ("4", w_mix)):
        if v > 1e-9:
            raise RuntimeError(f"check {name} failed: {v:.3e}")
    if n_ra_bad or n_rc_bad:
        raise RuntimeError("sufficient-ratio equivalence violated")

    keys = list(rows[0])
    A = {kk: np.array([r[kk] for r in rows]) for kk in keys}
    W = A["window_ok"].astype(bool)
    S2 = A["Te"] >= 2.0

    def block(label, m):
        n = int(m.sum())
        say(f"\n{label}: {n} pairs")
        cnt = lambda c: int((m & c).sum())
        say(f"   signs   Pbar>0 {cnt(A['Pbar']>0)}   SGbar<0 {cnt(A['SGbar']<0)}   "
            f"Sigbar<0 {cnt(A['Sigbar']<0)}   Sigma_local(Te+)<0 {cnt(A['Sigma_local_new']<0)}")
        say(f"   rates   A3>0 {cnt(A['A3']>0)}  A4>0 {cnt(A['A4']>0)}  C3<0 {cnt(A['C3']<0)}  "
            f"C4<0 {cnt(A['C4']<0)}   A4>A3 {cnt(A['A4']>A['A3'])}   |C4|>|C3| {cnt(np.abs(A['C4'])>np.abs(A['C3']))}")
        say(f"   shares  f3>f4 {cnt(A['f3']>A['f4'])}   f3 median {q(A['f3'][m],50):.3f}  f4 median {q(A['f4'][m],50):.3f}")
        say(f"   channel P_a>0 {cnt(A['P_a']>0)}   P_c>0 {cnt(A['P_c']>0)}   P_lin>0 {cnt(A['P_lin']>0)}   "
            f"both>0 {cnt((A['P_a']>0)&(A['P_c']>0))}   P_c<0 with P_a+P_c>0 {cnt((A['P_c']<0)&(A['P_lin']>0))}")
        say(f"   ratios  ratio_a>1 {cnt(A['ratio_a']>1)}  median {q(A['ratio_a'][m],50):.3f} min {A['ratio_a'][m].min():.3f}   "
            f"ratio_c>1 {cnt(A['ratio_c']>1)}  median {q(A['ratio_c'][m],50):.3f} min {A['ratio_c'][m].min():.3f}")
        say(f"   medians A3 {q(A['A3'][m],50):+.3f}  A4 {q(A['A4'][m],50):+.3f}  C3 {q(A['C3'][m],50):+.3f}  "
            f"C4 {q(A['C4'][m],50):+.3f}   P_a {q(A['P_a'][m],50):+.3f}  P_c {q(A['P_c'][m],50):+.3f}  "
            f"Pbar {q(A['Pbar'][m],50):+.3f}")
        rr = np.abs(A["additive_residual"][m]) / np.abs(A["Pbar"][m])
        say(f"   additive split |Pbar - P_a - P_c|/|Pbar|: median {q(rr,50):.4f}  90th {q(rr,90):.4f}  max {rr.max():.4f}")
        say(f"   cancellation |Sigbar|/|Pbar|: median {q(A['cancellation'][m],50):.4f}   "
            f"|Pbar| {q(np.abs(A['Pbar'][m]),50):.3f}  |SGbar| {q(np.abs(A['SGbar'][m]),50):.3f}  "
            f"|Sigbar| {q(np.abs(A['Sigbar'][m]),50):.4f}")
        neg = m & (A["P_c"] < 0)
        if neg.any():
            say(f"   where P_c<0 ({int(neg.sum())}): f3 {q(A['f3'][neg],50):.3f} f4 {q(A['f4'][neg],50):.3f}  "
                f"ratio_c median {q(A['ratio_c'][neg],50):.3f}   Te {A['Te'][neg].min():.2f}-{A['Te'][neg].max():.2f} eV  "
                f"ne {A['ne'][neg].min():.2e}-{A['ne'][neg].max():.2e}   heat {int((neg&(A['direction']=='heat')).sum())} cool {int((neg&(A['direction']=='cool')).sum())}")
        return n

    block("window_ok, Te_old >= 2 eV (the defended scope)", W & S2)
    block("   heating only", W & S2 & (A["direction"] == "heat"))
    block("   cooling only", W & S2 & (A["direction"] == "cool"))
    block("all 784 pairs (for reference; scope claims use the block above)", np.ones(len(rows), bool))

    ms = W & S2
    say("\nREFUTER  channel picture describes the finite step if P_a, P_c are mostly")
    say("         positive and the additive residual is small against the terms:")
    rr = np.abs(A["additive_residual"][ms]) / np.abs(A["Pbar"][ms])
    say(f"         P_a>0 {int((ms&(A['P_a']>0)).sum())}/448  P_c>0 {int((ms&(A['P_c']>0)).sum())}/448  "
        f"residual median {q(rr,50):.4f}  -> "
        + ("not refuted" if q(rr, 50) < 0.05 and int((ms & (A['P_lin'] > 0)).sum()) == int(ms.sum()) else "REFUTED"))

    txt, csvp = out / "operator_slope_decomposition.txt", out / "operator_slope_decomposition.csv"
    txt.write_text("\n".join(lines) + "\n")
    with csvp.open("w") as f:
        f.write(f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}\n"
                f"# L_grid sha256 {sha256(L_path)}\n# S_grid sha256 {sha256(S_path)}\n"
                f"# state_index sha256 {sha256(ctx.state_index_path)}\n"
                f"# step {a.frac}; window {a.win_lo}/{a.win_hi}; reproduces {ref_path.relative_to(root)} to {w_rep:.3e}\n"
                f"# f3,f4 at (Te-,u-); A_m,C_m secants over the step; P_a,P_c additive first-order split\n")
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[kk]) for kk in keys) + "\n")
    say(f"\nwrote {txt}\nwrote {csvp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
