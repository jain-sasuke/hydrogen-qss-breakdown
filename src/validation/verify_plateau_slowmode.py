"""
verify_plateau_slowmode.py
==========================
Supersedes verify_plateau_window.py (22 Aug 2026).

WHY THIS REWRITE
----------------
The previous script's docstring claimed the cold corner shows no decay because
"tau_QSS is 4e5 x longer than the window". That reasoning is WRONG: the window
is defined as 30*tau_relax < t < tau_QSS/30, so its upper edge is always
tau_QSS/30 and it always spans exactly 1/30 of tau_QSS. Under a single
exponential exp(-t/tau_QSS) BOTH points would show the same ~1.5% deficit.

Measured: ITER window/analytic = 0.985267, cold corner = 0.999983. So the cold
corner genuinely does not decay on tau_QSS, and "no time to decay" cannot be
the explanation. (Caught by an independent review, 22 Aug 2026.)

HYPOTHESIS UNDER TEST (restated)
--------------------------------
After the fast modes have died, is the residual n=3/n=4 shell-ratio error
carried by the slow eigenmode lambda_0, and does the observable's projection
onto lambda_0 explain why the benchmark point decays over the window while the
cold corner does not?

Sub-claims, each tested separately below:

  H1  The analytic partial-equilibrium (PE) state reproduces the plateau.
      n_E^PE = -L_EE^{-1} (S_E + L_Eg n_g^old)

  H2  Over the plateau window, ln|d(t)| is linear in t with slope -1/tau_QSS.
      Tested by R^2 and max residual, not by eyeballing tau_fit.

  H3  The window deficit is exactly what sampling that decay predicts:
      <eps>_window / eps_PE  ==  mean_i exp(-t_i / tau_QSS)  over the SAME
      sample times. (Not a hand-estimated 0.985.)

  H4  Where H2/H3 fail, the observable has little projection onto lambda_0.
      Measured directly via left eigenvectors.

  H5  The local sensitivity dlnR/dln b_1 at the OLD ground density predicts
      which points decay. Deep in an asymptotic Fujimoto limit this is ~0
      even though the FINITE excursion to the new ground density still
      produces a large plateau error. Local slope and finite excursion are
      different quantities.

EXACT ONE-PARAMETER FAMILY (the key algebraic simplification)
------------------------------------------------------------
Write the excited block under the NEW operator with the ground density scaled
by x relative to its old value:

    n_E(x) = n0 + x * n1,
      n0 = -L_EE^{-1} S_E              (recombination-fed channel)
      n1 = -L_EE^{-1} L_Eg n_g^old     (ground-fed channel)

Then EXACTLY:
    x = 1                     -> the partial-equilibrium (plateau) state
    x = n_g^new / n_g^old     -> the new QSS state

because the new QSS excited block satisfies the same equation with n_g^new.
So the whole plateau-to-target path is one straight line in x, and

    dlnR/dlnx |_x = f3(x) - f4(x),    f_p = x*n1_p_sum / (n0_p_sum + x*n1_p_sum)

This is Fujimoto eq. (4.20)'s two-channel split, realised in the model's own
matrix. f_p is the ground-fed fraction of shell p.

WHAT WOULD REFUTE THE WHOLE PICTURE
-----------------------------------
  - superposition check fails: n0 + x_new*n1 != direct new-QSS solve
  - H1 fails: analytic PE differs from the integrated plateau by >1%
  - a point with near-zero slow-mode projection that nevertheless decays on
    tau_QSS, or vice versa

SCOPE / CAVEATS
---------------
  - The ion density n_z is a FIXED RESERVOIR in this model: it lives in S and
    is never evolved. "Partial equilibrium" freezes n_1 while n_z was already
    frozen. Any statement about the slow subsystem is therefore about the
    ground state only. This is the open-system caveat.
  - The observable is the n=3/n=4 SHELL POPULATION RATIO. Calling it the
    Balmer H-alpha/H-beta ratio additionally requires the l-distribution to
    make the A-coefficients cancel (backlog C8, unverified).

Report only. Writes to the output directory; modifies nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_context import CRContext  # noqa: E402


# ----------------------------------------------------------------------------
# configuration (exposed, not buried)
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--dte", type=float, default=0.6,
                   help="absolute Te step in eV (default 0.6, matching the "
                        "original measurement)")
    p.add_argument("--win-lo", type=float, default=30.0,
                   help="window starts at WIN_LO * tau_relax")
    p.add_argument("--win-hi", type=float, default=30.0,
                   help="window ends at tau_QSS / WIN_HI")
    p.add_argument("--tol", type=float, default=0.005,
                   help="agreement tolerance for H3")
    p.add_argument("--tau-tol", type=float, default=0.2,
                   help="H2 requires |tau_fit/tau_QSS - 1| < TAU_TOL")
    p.add_argument("--lin-tol", type=float, default=0.10,
                   help="modal decomposition suppressed above this "
                        "linearisation error")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default <root>/validation/"
                        "plateau_slowmode)")
    p.add_argument("--points", type=str,
                   default="ITER ref:2.95,1.4e14;cold corner:1.0,1.0e12",
                   help="semicolon-separated LABEL:Te_eV,ne_cm3 — resolved to "
                        "the nearest grid point, never hardcoded indices")
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = parse_args()

    ctx = CRContext.load()
    root = ctx.root
    L_path = root / "data/processed/cr_matrix/L_grid.npy"
    S_path = root / "data/processed/cr_matrix/S_grid.npy"
    if not S_path.exists():
        raise FileNotFoundError(f"missing source vector: {S_path}")

    L = ctx.L_grid
    S = np.load(S_path)
    Te, ne = ctx.te_grid, ctx.ne_grid
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")

    g = int(ctx.ground_index)
    E = np.array([i for i in range(ctx.n_states) if i != g], dtype=int)
    N3 = np.where(np.asarray(ctx.n_values) == 3)[0]
    N4 = np.where(np.asarray(ctx.n_values) == 4)[0]
    if len(N3) != 3 or len(N4) != 4:
        raise ValueError(f"shell membership wrong: n=3 -> {N3}, n=4 -> {N4}")

    # local indices of the shells within the excited block
    pos = {s: k for k, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3])
    n4E = np.array([pos[s] for s in N4])

    out_dir = args.out or (root / "validation" / "plateau_slowmode")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines, rows = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("PLATEAU SLOW-MODE CHECK")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {Path(__file__).name}")
    say(f"repo root         {root}")
    say(f"L_grid            {L_path}")
    say(f"  sha256          {sha256(L_path)}")
    say(f"S_grid            {S_path}")
    say(f"  sha256          {sha256(S_path)}")
    say(f"state index       {ctx.state_index_path}")
    say(f"  sha256          {sha256(ctx.state_index_path)}")
    say(f"n=3 states {[int(i) for i in N3]} {[ctx.labels[i] for i in N3]}")
    say(f"n=4 states {[int(i) for i in N4]} {[ctx.labels[i] for i in N4]}")
    say(f"ground index      {g} ({ctx.labels[g]})")
    say(f"step dTe          +{args.dte} eV (absolute)")
    say(f"window            {args.win_lo:g}*tau_relax .. tau_QSS/{args.win_hi:g}")
    say(f"H2 requires       R^2 > 0.9999, one sign, "
        f"|tau_fit/tau_QSS - 1| < {args.tau_tol}")
    say(f"H4 suppressed if  linearisation error > {args.lin_tol:.0%}")
    say("NOTE: the ion density is a fixed reservoir in S and is never evolved.")
    say("      'partial equilibrium' freezes n(1) only.")
    say("OBSERVABLE: n=3 / n=4 shell population ratio (NOT yet the Balmer")
    say("      line ratio -- that additionally needs the C8 l-distribution check)")
    say("=" * 78)

    def R_of(v):
        return v[N3].sum() / v[N4].sum()

    for spec in args.points.split(";"):
        lab, coords = spec.split(":")
        te_t, ne_t = (float(v) for v in coords.split(","))
        i = int(np.argmin(np.abs(Te - te_t)))
        j = int(np.argmin(np.abs(np.log(ne) - np.log(ne_t))))
        k = int(np.argmin(np.abs(Te - (Te[i] + args.dte))))
        if k == i:
            raise ValueError(f"{lab}: step does not move a grid index")

        say("\n" + "-" * 78)
        say(f"{lab}   requested Te={te_t:g} eV ne={ne_t:.3g} cm^-3")
        say(f"  resolved to grid [{i},{j}]: Te={Te[i]:.6f} -> {Te[k]:.6f} eV, "
            f"ne={ne[j]:.6e} cm^-3")

        # --- eigenvalues: raise rather than abs() ---------------------------
        lam = np.linalg.eigvals(L[k, j])
        lam = lam[np.argsort(lam.real)[::-1]]
        if lam[0].real >= 0 or lam[1].real >= 0:
            raise RuntimeError(f"{lab}: operator not stable; "
                               f"lam0={lam[0]}, lam1={lam[1]}")
        imag_frac = np.abs(lam.imag).max() / np.abs(lam.real).max()
        tQ, tR = -1.0 / lam[0].real, -1.0 / lam[1].real
        say(f"  lambda_0 = {lam[0].real:.9e} {lam[0].imag:+.3e}j -> "
            f"tau_QSS   = {tQ:.6e} s")
        say(f"  lambda_1 = {lam[1].real:.9e} {lam[1].imag:+.3e}j -> "
            f"tau_relax = {tR:.6e} s")
        say(f"  max|Im|/max|Re| over spectrum = {imag_frac:.3e}")
        say(f"  M = tau_QSS/tau_relax = {tQ/tR:.6g}")

        # --- window validity guard (H2/H3 are meaningless without it) -------
        lo, hi = args.win_lo * tR, tQ / args.win_hi
        if lo >= hi:
            say(f"  SKIP: no timescale-separated plateau window "
                f"(needs M > {args.win_lo*args.win_hi:g}, have {tQ/tR:.4g})")
            continue

        # --- states ---------------------------------------------------------
        n_old = np.linalg.solve(L[i, j], -S[i, j])
        n_new = np.linalg.solve(L[k, j], -S[k, j])

        LEE = L[k, j][np.ix_(E, E)]
        LEg = L[k, j][np.ix_(E, [g])].ravel()
        n0 = np.linalg.solve(LEE, -S[k, j][E])            # recombination-fed
        n1 = np.linalg.solve(LEE, -LEg * n_old[g])        # ground-fed (at x=1)

        x_new = n_new[g] / n_old[g]

        # superposition / consistency check: does the family reproduce new QSS?
        n_new_E_pred = n0 + x_new * n1
        sup_err = (np.abs(n_new_E_pred - n_new[E]).max()
                   / np.abs(n_new[E]).max())
        say(f"  n(1)_old/n(1)_new = {1/x_new:.6f}   (x_new = {x_new:.6e})")
        say(f"  superposition check |n0 + x_new*n1 - n_new_E|_inf / "
            f"|n_new_E|_inf = {sup_err:.3e}")
        if sup_err > 1e-8:
            say("  *** SUPERPOSITION FAILS -- the two-channel split is wrong ***")

        n_pe = np.empty(ctx.n_states)
        n_pe[g] = n_old[g]
        n_pe[E] = n0 + n1

        Rq = R_of(n_new)
        d_pe = R_of(n_pe) / Rq - 1.0
        e_pe = abs(d_pe)
        d_step = R_of(n_old) / Rq - 1.0

        say(f"  eps_step   (old QSS vs new QSS)  {abs(d_step):.8f}  "
            f"(signed {d_step:+.8f})")
        say(f"  eps_plateau ANALYTIC (H1)        {e_pe:.8f}  "
            f"(signed {d_pe:+.8f})")

        # --- H5: local sensitivity and the two channels ---------------------
        def shell_sums(x):
            a3 = n0[n3E].sum() + x * n1[n3E].sum()
            a4 = n0[n4E].sum() + x * n1[n4E].sum()
            return a3, a4

        def sens(x):
            a3, a4 = shell_sums(x)
            f3 = x * n1[n3E].sum() / a3
            f4 = x * n1[n4E].sum() / a4
            return f3, f4, f3 - f4

        f3_o, f4_o, s_old = sens(1.0)
        f3_n, f4_n, s_new = sens(x_new)
        say(f"  ground-fed fraction at x_old : f3={f3_o:.6f}  f4={f4_o:.6f}  "
            f"dlnR/dlnx = {s_old:+.6e}")
        say(f"  ground-fed fraction at x_new : f3={f3_n:.6f}  f4={f4_n:.6f}  "
            f"dlnR/dlnx = {s_new:+.6e}")
        say(f"  linearised estimate of eps_plateau = |s_old * ln(x_new)| = "
            f"{abs(s_old*np.log(x_new)):.8f}   vs actual {e_pe:.8f}")

        # --- H4: does the observable project onto lambda_0? -----------------
        w, V = np.linalg.eig(L[k, j])
        order = np.argsort(w.real)[::-1]
        w, V = w[order], V[:, order]
        Wl = np.linalg.inv(V).T          # left eigenvectors, biorthogonal
        dn = n_pe - n_new
        c = Wl.T @ dn                    # modal amplitudes
        gradR = np.zeros(ctx.n_states)
        gradR[N3] = 1.0 / n_new[N4].sum()
        gradR[N4] = -n_new[N3].sum() / n_new[N4].sum() ** 2
        contrib = np.array([np.real(c[m] * (gradR @ V[:, m]))
                            for m in range(len(w))])
        tot = contrib.sum()
        direct = R_of(n_pe) - Rq
        lin_err = abs(tot / direct - 1) if direct != 0 else np.inf
        lin_ok = lin_err < args.lin_tol
        frac0 = contrib[0] / tot if (lin_ok and tot != 0) else np.nan
        say(f"  linearised dR from all modes = {tot:.6e}   "
            f"(direct dR = {direct:.6e})   linearisation error {lin_err:.3e}")
        if lin_ok:
            say(f"  H4 slow-mode (lambda_0) share of residual = {frac0:.6f}")
        else:
            say(f"  H4 SUPPRESSED: linearisation error {lin_err:.3e} exceeds "
                f"{args.lin_tol:.0%}; modal decomposition is not valid here")

        # --- integrate, with a tolerance convergence check -------------------
        t = np.logspace(np.log10(tR), np.log10(tQ * 3), 4000)
        sols = {}
        for rtol, atol in ((1e-9, 1e-30), (1e-12, 1e-32)):
            s = solve_ivp(lambda _, y: L[k, j] @ y + S[k, j], (0, t[-1]),
                          n_old, t_eval=t, method="LSODA",
                          rtol=rtol, atol=atol)
            if not s.success:
                raise RuntimeError(f"{lab}: integration failed -- {s.message}")
            sols[(rtol, atol)] = s
        keys = list(sols)
        dref = sols[keys[-1]].y[N3].sum(0) / sols[keys[-1]].y[N4].sum(0) / Rq - 1
        dlow = sols[keys[0]].y[N3].sum(0) / sols[keys[0]].y[N4].sum(0) / Rq - 1
        conv = np.abs(dref - dlow).max() / np.abs(dref).max()
        say(f"  tolerance convergence rtol 1e-9 vs 1e-12: max rel diff "
            f"{conv:.3e}")

        sol = sols[keys[-1]]
        d = dref
        m = (sol.t > lo) & (sol.t < hi)
        say(f"  window {lo:.4e} .. {hi:.4e} s  ({m.sum()} log-spaced samples)")
        say(f"    window upper edge / tau_QSS = {hi/tQ:.6f}  "
            f"(identical at every grid point by construction)")

        # --- H2: sign check, then fit quality on ln|d| ----------------------
        same_sign = bool(np.all(np.sign(d[m]) == np.sign(d[m][0])))
        say(f"  residual keeps one sign across the window: {same_sign}")
        if not same_sign:
            say("  *** zero crossing inside the window -- single exponential "
                "cannot apply ***")
        y = np.log(np.abs(d[m]))
        cf = np.polyfit(sol.t[m], y, 1)
        yf = np.polyval(cf, sol.t[m])
        ss_res = float(((y - yf) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        max_res = float(np.abs(np.exp(y) / np.exp(yf) - 1).max())
        tau_fit = -1.0 / cf[0]
        say(f"  H2 fit ln|d| vs t : tau_fit = {tau_fit:.6e} s = "
            f"{tau_fit/tQ:.4f} x tau_QSS")
        say(f"     R^2 = {r2:.8f}   max relative residual = {max_res:.3e}")

        # --- H3: predicted vs observed window deficit ------------------------
        obs_ratio = np.abs(d[m]).mean() / e_pe
        pred_ratio = float(np.exp(-sol.t[m] / tQ).mean())
        say(f"  H3 window/analytic OBSERVED  {obs_ratio:.8f}")
        say(f"     window/analytic PREDICTED {pred_ratio:.8f}  "
            f"(mean of exp(-t_i/tau_QSS) over the SAME samples)")
        say(f"     difference {obs_ratio - pred_ratio:+.3e}")
        say("     (both are means over LOG-SPACED samples, not time averages)")

        # --- slow-manifold intercept (NOT n(0+), which is n_old) ------------
        e_int = float(np.exp(cf[1]))
        say(f"  slow-manifold intercept extrapolated to t=0: {e_int:.8f}  "
            f"ratio to analytic {e_int/e_pe:.6f}")
        say("     NB this is the slow-manifold amplitude continued back to the "
            "step, NOT n(0+) -- n(0+) = n_old exactly.")

        # --- verdict ---------------------------------------------------------
        h1 = abs(np.abs(d[m])[0] / e_pe - 1) < 0.05
        h2 = (r2 > 0.9999) and same_sign and abs(tau_fit / tQ - 1) < args.tau_tol
        h3_power = abs(1.0 - pred_ratio) > 0.005
        h3 = (abs(obs_ratio - pred_ratio) < args.tol) and h3_power
        if not h3_power:
            say(f"     H3 HAS NO DISCRIMINATING POWER here: predicted deficit "
                f"is only {1-pred_ratio:.2e}; any result would 'agree'")
        if not h2 and same_sign and r2 > 0.9999:
            say(f"     -> H2 fails on tau only: tau_fit = {tau_fit/tQ:.1f} x "
                f"tau_QSS. Clean fit, but NOT lambda_0 -- a pinned observable, "
                f"not a decaying one.")
        say(f"  VERDICT  H1 plateau==PE: {'PASS' if h1 else 'FAIL'}"
            f"   H2 single exponential at lambda_0: {'PASS' if h2 else 'FAIL'}"
            f"   H3 sampling explains deficit: "
            f"{'PASS' if h3 else ('NO POWER' if not h3_power else 'FAIL')}")
        if h3 and not h2:
            say("     -> deficit matches sampling but the decay is not a clean "
                "single exponential; report both.")
        if not h2 and frac0 < 0.5:
            say(f"     -> H4: observable carries little lambda_0 "
                f"({frac0:.4f}); non-decay is a projection effect, "
                f"NOT lack of time.")

        rows.append(dict(
            label=lab, i=i, j=j, Te=Te[i], Te_new=Te[k], ne=ne[j],
            tau_QSS=tQ, tau_relax=tR, M=tQ / tR,
            eps_step=abs(d_step), eps_plateau=e_pe,
            amplification=e_pe / abs(d_step) if d_step != 0 else np.nan,
            x_new=x_new, f3_old=f3_o, f4_old=f4_o, sens_old=s_old,
            f3_new=f3_n, f4_new=f4_n, sens_new=s_new,
            slowmode_share=frac0, superposition_err=sup_err,
            lin_err=lin_err, lin_ok=lin_ok, h3_power=h3_power,
            H1=h1, H2=h2, H3=h3,
            tau_fit_over_tauQSS=tau_fit / tQ, r2=r2, max_resid=max_res,
            obs_ratio=obs_ratio, pred_ratio=pred_ratio,
            tol_convergence=conv, same_sign=same_sign,
        ))

    txt = out_dir / "plateau_slowmode.txt"
    csv = out_dir / "plateau_slowmode.csv"
    txt.write_text("\n".join(lines) + "\n")
    if rows:
        keys = list(rows[0])
        with csv.open("w") as f:
            f.write(f"# generated {datetime.now():%Y-%m-%d %H:%M} by "
                    f"{Path(__file__).name}\n")
            f.write(f"# L_grid sha256 {sha256(L_path)}\n")
            f.write(f"# S_grid sha256 {sha256(S_path)}\n")
            f.write(f"# state_index sha256 {sha256(ctx.state_index_path)}\n")
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
    say(f"\nwrote {txt}")
    say(f"wrote {csv}")


if __name__ == "__main__":
    main()