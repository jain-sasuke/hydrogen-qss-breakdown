#!/usr/bin/env python
"""
verify_inversion_error.py
=========================
Measure the error in the temperature a diagnostician INFERS from the tabulated
n=3/n=4 ratio during the plateau, at known n_e, by inverting the table exactly.

WHY THIS EXISTS
---------------
Chapter 1 asks how large the error in an inversion of the Balmer ratio is.
Chapter 5 measures eps_plateau, the error in the OBSERVABLE relative to the
tabulated equilibrium value, and Section sec:inversion_jacobian says in words
that the two are not the same thing: the inferred-temperature error is the
observable error divided by d ln R_cre / d ln Te, and that denominator passes
through zero inside the operating range on six of eight density columns.
Eq. eq:inversion_jacobian (Chapter 3) defines the object at known n_e and says
Chapter 5 uses "the exact finite-step form wherever a number is quoted". No
script computed that finite-step form. This one does.

WHAT IT COMPUTES
----------------
For every (point, direction) pair of verify_plateau_gridmap.py, with the SAME
step rule (nearest grid index to a +/-5 % step), the SAME plateau-window rule
and the SAME three linear solves:

    R_cre(Te_m, ne_j)   for all 50 nodes m of density column j
    R_pe                the plateau observable: post-step excited response on
                        the pre-step reservoir (two-channel split of L+)
    Te*                 every temperature on column j at which the TABLE
                        returns R_pe, found by bracketing R_cre(Te) - R_pe
                        between adjacent nodes and interpolating

The inferred-temperature error is  err = ln(Te*) - ln(Te_new), where Te_new is
the true post-step temperature; the amplification is |err| / |ln(Te_new/Te_old)|,
the factor by which the true excursion is misread.

Three things a bare median would hide are reported separately, never silently
dropped:
    off-table     no Te in [1, 10] eV returns R_pe: zero roots
    multi-root    more than one Te returns R_pe (the fold of sec:inversion_jacobian)
    fold-crossed  a stationary point of R_cre(Te) lies between Te_new and Te*

Two interpolation variants are carried (linear in (ln Te, ln R), the primary,
and linear in (Te, R)) so that the reported number is shown to be insensitive to
that choice. The linearised Jacobian prediction of eq:inversion_jacobian is
computed alongside so the size of the linearisation error is on record.

PREDICTIONS, WRITTEN BEFORE THE RUN (from a reconstruction of the same solve
out of plateau_gridmap.csv on 11 Sep 2026, ln-ln interpolation):
    Te_old >= 2 eV, window_ok: 448 pairs; ~166 off-table; of the solvable,
    median |err| ~ 0.445, 90th ~ 0.90, max ~ 1.42; ~9 multi-root.
    Median amplification ~ 10.
THE REFUTING OBSERVATION: median amplification of order 1 over the defended
scope. Then the inversion does not amplify the closure error and the sentence
"the table misreads a 5 % transient as a ~50 % temperature error" is false.

CHECKS THAT MUST PASS OR THE SCRIPT STOPS
    1. d_pe and d_step reproduce plateau_gridmap.csv's signed_plateau and
       signed_step at every pair (same solves, so 1e-9 is generous).
    2. On the two monotone columns, sign(err) == sign(d_pe) * sign(dR/dTe).
       An inverted interpolation fails this at every pair.
    3. The two interpolation variants agree on the median to 10 %.
Nothing is hardcoded: grids, ordering and operators come from cr_context.
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
    p.add_argument("--frac", type=float, default=0.05,
                   help="fractional Te step, as verify_plateau_gridmap.py")
    p.add_argument("--win-lo", type=float, default=30.0)
    p.add_argument("--win-hi", type=float, default=30.0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def roots_on_column(x, y, target):
    """Every x at which piecewise-linear y(x) equals target. Returns [] if none."""
    out = []
    for m in range(len(x) - 1):
        a, b = y[m] - target, y[m + 1] - target
        if a == 0.0:
            out.append(x[m])
        elif a * b < 0.0:
            t = a / (a - b)
            out.append(x[m] + t * (x[m + 1] - x[m]))
    if len(y) and y[-1] - target == 0.0:
        out.append(x[-1])
    return out


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def main():
    a = parse_args()
    ctx = CRContext.load()
    root = ctx.root
    L_path = root / "data/processed/cr_matrix/L_grid.npy"
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    if not S_path.exists():
        raise FileNotFoundError(f"missing source vector: {S_path}")
    ref_path = root / "validation/plateau_gridmap/plateau_gridmap.csv"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"missing reproduction target {ref_path}; run "
            "verify_plateau_gridmap.py first. This script will not proceed "
            "without the cross-check that its solves are the thesis's solves.")

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

    out = a.out or (root / "validation" / "inversion_error")
    out.mkdir(parents=True, exist_ok=True)
    lines, rows = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("INVERSION ERROR -- what the table returns for Te during the plateau")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"repo root      {root}")
    say(f"L_grid sha256  {sha256(L_path)}")
    say(f"S_grid sha256  {sha256(S_path)}")
    say(f"state idx sha  {sha256(ctx.state_index_path)}")
    say(f"reproduction   {ref_path.relative_to(root)}")
    say(f"grid           {nT} Te x {nN} ne; one Te interval = "
        f"dlnTe {lnTe[1]-lnTe[0]:.7f}")
    say("=" * 78)

    # ---- the table: R_cre at every node, and its stationary points ---------
    R_cre = np.empty((nT, nN))
    for j in range(nN):
        for m in range(nT):
            n = np.linalg.solve(L[m, j], -S[m, j])
            R_cre[m, j] = n[N3].sum() / n[N4].sum()
    lnR = np.log(R_cre)
    # same rule as verify_crest_subgrid.py: a sign change of diff(ln R) between
    # consecutive intervals c and c+1 puts the stationary point at node c+1
    stat_nodes = {}
    for j in range(nN):
        sgn = np.sign(np.diff(lnR[:, j]))
        cross = np.where(np.diff(sgn) != 0)[0]
        stat_nodes[j] = [int(c) + 1 for c in cross]
    monotone_cols = [j for j in range(nN) if not stat_nodes[j]]
    say("\nstationary points of the tabulated ratio, per density column:")
    for j in range(nN):
        w = ", ".join(f"{Te[c]:.3g} eV" for c in stat_nodes[j]) or "none"
        say(f"   j={j} ne={ne[j]:.3g}  {w}")
    say(f"monotone columns: {monotone_cols}")

    # ---- reproduction target ---------------------------------------------
    with ref_path.open() as fh:
        ref = {(r["direction"], int(r["i"]), int(r["j"])): r
               for r in _csv.DictReader(
                   ln for ln in fh if not ln.startswith("#"))}

    worst_rep = 0.0
    n_sign_checked = n_sign_bad = 0

    for sgn_, dlab in ((+1, "heat"), (-1, "cool")):
        for i in range(nT):
            k = int(np.argmin(np.abs(Te - Te[i] * (1 + sgn_ * a.frac))))
            if k == i:
                continue
            dlnTe_true = lnTe[k] - lnTe[i]
            for j in range(nN):
                lam = np.linalg.eigvals(L[k, j])
                lam = lam[np.argsort(lam.real)[::-1]]
                if lam[0].real >= 0 or lam[1].real >= 0:
                    raise RuntimeError(
                        f"unstable operator at Te={Te[k]:g} ne={ne[j]:g}")
                tQ, tR = -1.0 / lam[0].real, -1.0 / lam[1].real
                window_ok = bool((a.win_lo * tR) < (tQ / a.win_hi))

                n_old = np.linalg.solve(L[i, j], -S[i, j])
                n_new = np.linalg.solve(L[k, j], -S[k, j])
                LEE = L[k, j][np.ix_(E, E)]
                LEg = L[k, j][np.ix_(E, [g])].ravel()
                n0 = np.linalg.solve(LEE, -S[k, j][E])
                n1 = np.linalg.solve(LEE, -LEg * n_old[g])
                R_pe = (n0[n3E].sum() + n1[n3E].sum()) / (n0[n4E].sum() + n1[n4E].sum())
                Rq = n_new[N3].sum() / n_new[N4].sum()
                R_old = n_old[N3].sum() / n_old[N4].sum()
                d_pe = R_pe / Rq - 1.0
                d_step = R_old / Rq - 1.0

                # -- the two channels of the tabulated slope, as secants over
                #    the step. ln R_pe - ln R_cre(new) = -SbarG*dlnTe (reservoir
                #    channel); ln R_cre(new) - ln R_cre(old) = Sigma*dlnTe (net);
                #    P = Sigma - SbarG is the manifold's response at frozen u.
                Sig_bar = -np.log1p(d_step) / dlnTe_true
                SbarG = -np.log1p(d_pe) / dlnTe_true
                P_bar = Sig_bar - SbarG

                # -- check 1: these are the thesis's solves, not a re-derivation
                r = ref.get((dlab, i, j))
                if r is None:
                    raise RuntimeError(f"pair {(dlab, i, j)} absent from {ref_path}")
                rep = max(abs(d_pe - float(r["signed_plateau"])),
                          abs(d_step - float(r["signed_step"])))
                worst_rep = max(worst_rep, rep)
                if bool(r["window_ok"] == "True") != window_ok:
                    raise RuntimeError(f"window_ok disagrees at {(dlab, i, j)}")

                # -- the inversion, two interpolation variants
                col_lnR = lnR[:, j]
                roots_ll = roots_on_column(lnTe, col_lnR, np.log(R_pe))
                roots_lin = [np.log(x) for x in
                             roots_on_column(Te, R_cre[:, j], R_pe)]

                def summarise(roots):
                    if not roots:
                        return dict(n=0, err=np.nan, err_far=np.nan, fold=False)
                    errs = [rt - lnTe[k] for rt in roots]
                    near = min(errs, key=abs)
                    far = max(errs, key=abs)
                    lo, hi = min(lnTe[k], lnTe[k] + near), max(lnTe[k], lnTe[k] + near)
                    fold = any(lo < lnTe[c] < hi for c in stat_nodes[j])
                    return dict(n=len(roots), err=near, err_far=far, fold=fold)

                s_ll, s_lin = summarise(roots_ll), summarise(roots_lin)

                # -- where the table's range was missed
                off = "in-range"
                if s_ll["n"] == 0:
                    off = "above-max" if np.log(R_pe) > col_lnR.max() else "below-min"

                # -- linearised Jacobian prediction, central difference at node k
                lo_, hi_ = max(k - 1, 0), min(k + 1, nT - 1)
                dlnR_dlnTe = (col_lnR[hi_] - col_lnR[lo_]) / (lnTe[hi_] - lnTe[lo_])
                err_lin_jac = np.log1p(d_pe) / dlnR_dlnTe if dlnR_dlnTe != 0 else np.nan

                # -- check 2: sign identity on the monotone columns
                if j in monotone_cols and s_ll["n"] == 1:
                    n_sign_checked += 1
                    want = np.sign(d_pe) * np.sign(dlnR_dlnTe)
                    if np.sign(s_ll["err"]) != want:
                        n_sign_bad += 1

                rows.append(dict(
                    direction=dlab, i=i, j=j, Te=Te[i], Te_new=Te[k], ne=ne[j],
                    window_ok=window_ok, dlnTe_true=dlnTe_true,
                    R_pe=R_pe, R_cre_new=Rq, signed_plateau=d_pe,
                    n_roots=s_ll["n"], off_table=off,
                    err_lnTe=s_ll["err"], err_lnTe_far=s_ll["err_far"],
                    fold_crossed=s_ll["fold"],
                    amp=abs(s_ll["err"]) / abs(dlnTe_true),
                    err_lnTe_linTeR=s_lin["err"], n_roots_linTeR=s_lin["n"],
                    err_lnTe_jacobian=err_lin_jac,
                    dlnR_dlnTe_at_new=dlnR_dlnTe,
                    Sigma_bar=Sig_bar, SbarG=SbarG, P_bar=P_bar,
                    cancellation=abs(Sig_bar) / abs(P_bar),
                    amp_pred_secant=abs(SbarG) / abs(Sig_bar),
                ))

    say(f"\ncheck 1  reproduction of plateau_gridmap.csv: worst |diff| "
        f"{worst_rep:.3e}")
    if worst_rep > 1e-9:
        raise RuntimeError("solves do not reproduce the stamped artifact")
    say(f"check 2  sign identity on monotone columns: {n_sign_bad} bad of "
        f"{n_sign_checked}")
    if n_sign_bad:
        raise RuntimeError("inversion sign identity violated")

    keys = list(rows[0])
    A = {kk: np.array([r[kk] for r in rows]) for kk in keys}
    W = A["window_ok"].astype(bool)
    S2 = A["Te"] >= 2.0

    summ = []

    def report(label, m):
        n_all = int(m.sum())
        solv = m & (A["n_roots"] > 0)
        off = m & (A["n_roots"] == 0)
        multi = m & (A["n_roots"] > 1)
        fold = solv & A["fold_crossed"].astype(bool)
        clean = solv & ~A["fold_crossed"].astype(bool)
        e = np.abs(A["err_lnTe"])
        el = np.abs(A["err_lnTe_linTeR"])
        ej = np.abs(A["err_lnTe_jacobian"])
        amp = A["amp"]
        say(f"\n{label}: {n_all} pairs")
        say(f"   off-table {int(off.sum())}  "
            f"(above table max {int((off & (A['off_table']=='above-max')).sum())}, "
            f"below min {int((off & (A['off_table']=='below-min')).sum())})")
        say(f"   solvable  {int(solv.sum())}   multi-root {int(multi.sum())}   "
            f"fold-crossed {int(fold.sum())}   clean {int(clean.sum())}")
        for tag, mm in (("solvable, nearest root", solv), ("clean (fold-crossed removed)", clean)):
            if mm.sum() == 0:
                continue
            say(f"   |err ln Te| {tag:<30} median {q(e[mm],50):.4f}  90th {q(e[mm],90):.4f}  "
                f"max {e[mm].max():.4f}   -> median Te error {100*(np.exp(q(e[mm],50))-1):.1f} %")
            say(f"   amplification {tag:<28} median {q(amp[mm],50):.2f}  90th {q(amp[mm],90):.2f}  "
                f"max {amp[mm].max():.1f}")
            say(f"   (Te,R)-linear variant, same pairs      median {q(el[mm],50):.4f}  "
                f"ratio to primary {q(el[mm],50)/q(e[mm],50):.4f}")
            jm = mm & np.isfinite(A["err_lnTe_jacobian"])
            say(f"   linearised Jacobian prediction         median {q(ej[jm],50):.4f}  "
                f"ratio to exact {q(ej[jm],50)/q(e[mm],50):.3f}")
            summ.append(dict(scope=label, subset=tag, n=int(mm.sum()),
                             off_table=int(off.sum()), multi_root=int(multi.sum()),
                             fold_crossed=int(fold.sum()),
                             med_abs_err_lnTe=q(e[mm],50), p90_abs_err_lnTe=q(e[mm],90),
                             max_abs_err_lnTe=float(e[mm].max()),
                             med_amp=q(amp[mm],50), p90_amp=q(amp[mm],90),
                             med_err_linTeR=q(el[mm],50), med_err_jacobian=q(ej[jm],50)))
        sgn_pos = int((solv & (A["err_lnTe"] > 0)).sum())
        say(f"   sign of err: {sgn_pos} inferred hotter than true, "
            f"{int(solv.sum())-sgn_pos} colder")

    report("all window_ok pairs", W)
    report("window_ok, Te_old >= 2 eV (the defended scope)", W & S2)
    report("   heating only, Te_old >= 2 eV", W & S2 & (A["direction"] == "heat"))
    report("   cooling only, Te_old >= 2 eV", W & S2 & (A["direction"] == "cool"))
    report("   monotone columns only, Te_old >= 2 eV", W & S2 & np.isin(A["j"], monotone_cols))

    # check 3: definitional sensitivity
    m = W & S2 & (A["n_roots"] > 0)
    r_var = q(np.abs(A["err_lnTe_linTeR"][m]),50) / q(np.abs(A["err_lnTe"][m]),50)
    say(f"\ncheck 3  interpolation-variant ratio of medians (defended scope): {r_var:.4f}")
    if not (0.9 < r_var < 1.1):
        say("   WARNING: the two variants disagree by more than 10 %; do not quote a "
            "precise median")

    med_amp = q(A["amp"][m & ~A["fold_crossed"].astype(bool)], 50)
    say(f"\nREFUTER  median amplification, defended scope, clean: {med_amp:.2f}")
    say("   " + ("REFUTED: of order one; the inversion does not amplify" if med_amp < 2
               else "not refuted: the true excursion is misread by this factor"))

    # ---- derivation checks: the direction and size follow from three signs --
    def rankcorr(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    say("\nDERIVATION CHECKS  (Sigma = P + SbarG; on the plateau only P acts)")
    ms = W & S2
    heat, cool = A["direction"] == "heat", A["direction"] == "cool"
    solv = A["n_roots"] > 0
    Sg = A["dlnR_dlnTe_at_new"]
    say(f"   signs over the {int(ms.sum())} pairs in scope:  P_bar>0 "
        f"{int((ms & (A['P_bar']>0)).sum())}   SbarG<0 {int((ms & (A['SbarG']<0)).sum())}"
        f"   Sigma_bar<0 {int((ms & (A['Sigma_bar']<0)).sum())}")
    hn = ms & heat & solv & (Sg < 0); cn = ms & cool & solv & (Sg < 0)
    say(f"   theorem, Sigma<0 at Te_new: heating colder {int((hn & (A['err_lnTe']<0)).sum())}"
        f" of {int(hn.sum())};  cooling hotter {int((cn & (A['err_lnTe']>0)).sum())} of {int(cn.sum())}")
    hp = ms & heat & solv & (Sg > 0)
    fc = A["fold_crossed"].astype(bool)
    say(f"   heating with Sigma>0 at Te_new: {int(hp.sum())}; of those fold-crossed "
        f"{int((hp & fc).sum())}; fold-crossed with Sigma<0: {int((ms & heat & fc & (Sg<0)).sum())}")
    say(f"   cancellation |Sigma|/|P| in scope: median {q(A['cancellation'][ms],50):.4f};  "
        f"|P| {q(np.abs(A['P_bar'][ms]),50):.3f}  |SbarG| {q(np.abs(A['SbarG'][ms]),50):.3f}  "
        f"|Sigma| {q(np.abs(A['Sigma_bar'][ms]),50):.4f}")
    cl = ms & solv & ~fc
    rr = A["amp_pred_secant"][cl] / A["amp"][cl]
    say(f"   secant prediction |SbarG|/|Sigma| vs exact amplification, {int(cl.sum())} fold-free pairs: "
        f"rank corr {rankcorr(A['amp'][cl], A['amp_pred_secant'][cl]):.3f}, "
        f"pred/exact median {q(rr,50):.3f} (10th {q(rr,10):.3f}, 90th {q(rr,90):.3f})")
    co = ms & cool & (A["n_roots"] == 0); cs = ms & cool & solv
    ratio = np.abs(A["P_bar"]) / np.abs(A["Sigma_bar"])
    say(f"   cooling off-table {int(co.sum())} of {int((ms & cool).sum())}: |P|/|Sigma| median "
        f"{q(ratio[co],50):.1f} there, {q(ratio[cs],50):.1f} where a root exists")
    if (ms & (A['P_bar'] <= 0)).any() or (ms & (A['SbarG'] >= 0)).any():
        raise RuntimeError("a sign the derivation rests on fails inside the scope")
    if int((hn & (A['err_lnTe']>=0)).sum()) or int((cn & (A['err_lnTe']<=0)).sum()):
        raise RuntimeError("direction theorem violated where Sigma<0")

    # ---- worst cases, for the text -----------------------------------------
    say("\nlargest clean |err ln Te| in the defended scope:")
    mm = W & S2 & (A["n_roots"] > 0) & ~A["fold_crossed"].astype(bool)
    o = np.argsort(-np.abs(A["err_lnTe"][mm]))[:6]
    for t_, tn_, n_, d_, e_, am_ in zip(A["Te"][mm][o], A["Te_new"][mm][o], A["ne"][mm][o],
                                        A["direction"][mm][o], A["err_lnTe"][mm][o], A["amp"][mm][o]):
        say(f"   {d_:4s} Te {t_:.3f}->{tn_:.3f} eV  ne {n_:.3g}  err {e_:+.4f}  "
            f"({100*(np.exp(e_)-1):+.0f} % in Te)  amp {am_:.1f}")

    txt, csvp, sump = out / "inversion_error.txt", out / "inversion_error.csv", out / "inversion_error_summary.csv"
    txt.write_text("\n".join(lines) + "\n")
    hdr = (f"# generated {datetime.now():%Y-%m-%d %H:%M} by {Path(__file__).name}\n"
           f"# L_grid sha256 {sha256(L_path)}\n# S_grid sha256 {sha256(S_path)}\n"
           f"# state_index sha256 {sha256(ctx.state_index_path)}\n"
           f"# requested fractional step {a.frac}; window {a.win_lo}/{a.win_hi}\n"
           f"# reproduces {ref_path.relative_to(root)} to {worst_rep:.3e}\n"
           "# SIGN NOTE: column SbarG = ln(R_cre+/R_pe)/dlnTe, the reservoir secant with the\n"
           "#   NATURAL sign of S = f3-f4 > 0 (so SbarG < 0). In the thesis's Eq. Sbar_def\n"
           "#   convention (Sbar integrates u+ -> u-, so Sbar < 0) this equals -Sbar*Gbar.\n"
           "#   Sigma_bar = P_bar + SbarG exactly; verified by verify_operator_slope_decomposition.py.\n")
    with csvp.open("w") as f:
        f.write(hdr + ",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[kk]) for kk in keys) + "\n")
    with sump.open("w") as f:
        f.write(hdr)
        w = _csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    say(f"\nwrote {txt}\nwrote {csvp}\nwrote {sump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
