#!/usr/bin/env python
"""
verify_open_reservoir.py
========================
Does tau_esc < tau_slow make the chapter 5 magnitudes UPPER estimates?

WHY THIS EXISTS
---------------
Chapter 6, Section 6.2 (thesis_tex/chapter6.tex, sec:transport_selection and
sec:transport_partition) finds that the charge-exchange-limited neutral escape
time tau_esc is shorter than tau_slow at every one of the 45 defended breakdown
pairs (validation/transport_selection/transport_selection.csv, written by
verify_transport_selection.py), and concludes that because recycling can renew
the ground-state reservoir faster than local ionisation balance can, the
closed-parcel numbers of chapter 5 are "upper estimates of an error whose true
size this model cannot compute".

That inference does not follow, and this script tests it with the smallest
model that contains the missing physics. A reservoir that exchanges neutrals
with its surroundings on tau_esc relaxes on tau_esc, but it relaxes TO the
value set by the balance of recycling source against escape plus ionisation,
not to the local CRE value. Whether the resulting steady error is smaller or
larger than the closed-parcel plateau error depends on what the recycling
source does during the event, which the chapter does not model. An ELM raises
the target particle flux, so the source goes up, not down.

THE MODEL, FROM THE OPERATOR ITSELF
-----------------------------------
Everything below is built from L_grid.npy and S_grid.npy through the same
excited-block elimination that verify_divertor_map.py uses for its two-channel
split, so the toy shares its coefficients with the chapter 5 result rather
than introducing a second model.

Write the CR system per ion as dn/dt = L n + S with n = (n_g, n_E), where g is
the ground state and E the 42 excited states. Eliminating the excited block
at quasi-steady state (n_E = a u + c, with a = -L_EE^{-1} L_Eg and
c = -L_EE^{-1} S_E, both per ion) gives the exact one-dimensional slow dynamics
of the ground-state fraction u = n_g / n_ion:

    du/dt = alpha_eff - S_eff u

    S_eff     = -(L_gg + L_gE a)    effective loss frequency of the ground
                                    state through the excited manifold, s^-1
    alpha_eff =  S_g + L_gE c       effective recombination source into the
                                    ground state, per ion, s^-1

This is the Schur complement of L on the ground state at eigenvalue zero.
Two checks pin it to the full operator:

    alpha_eff / S_eff  must equal  u_CRE = [L^{-1}(-S)]_g   (exact identity;
                                    a failure is a coding error and raises)
    1 / S_eff          must equal  tau_slow = 1/|lambda_0| of the full L
                                    (the QSS reduction of the slow eigenvalue;
                                    the relative discrepancy is O(1/M), so it
                                    is reported, not asserted)

The observable is the n=3/n=4 shell population ratio of verify_divertor_map.py,
R(u) = (a3 u + c3) / (a4 u + c4), evaluated with the POST-step coefficients
a3 = sum of a over n=3 states, etc. The closed-parcel plateau error is
eps_plateau = |R(u_CRE-) / R(u_CRE+) - 1|, recomputed here and compared with
the value stored in divertor_map.csv.

Now open the reservoir. Add an escape sink nu_esc = 1/tau_esc, with tau_esc
read from transport_selection.csv for this pair at L = 10 cm and 20 cm, and a
recycling source Gamma (per ion, s^-1):

    du/dt = alpha_eff + Gamma - (S_eff + nu_esc) u

Gamma is fixed by requiring that the PRE-step open steady state equal the
pre-step local CRE value, so that the table is exactly right before the event
and the toy cannot manufacture a disagreement out of nothing:

    Gamma_0 = (S_eff- + nu_esc) u_CRE- - alpha_eff-  =  nu_esc u_CRE-

(the second equality because S_eff- u_CRE- = alpha_eff- identically: the
recycling source that closes the pre-step balance is the escape flux at CRE).

After the Te step, with the source multiplied by m in {0, 1, 2, 5}:

    u_open+        = (alpha_eff+ + m Gamma_0) / (S_eff+ + nu_esc)
    steady error   = |R(u_open+) / R(u_CRE+) - 1|
    relaxation     = 1 / (S_eff+ + nu_esc)

and the steady error is compared with eps_plateau and with lo_ELM_crash, the
100 us time-averaged lower bound that produced the census.

PREDICTIONS, WRITTEN BEFORE RUNNING
-----------------------------------
P1  Gamma_0 / (nu_esc u_CRE-) = 1 to machine precision at every pair.
P2  |1/S_eff+ / tau_QSS - 1| is of order 1/M: below 1e-4 at all 45 pairs
    (M >= 2e5 there) and marginal at the benchmark (M = 8243, 1/M = 1.2e-4).
    A discrepancy not of order 1/M would mean the Schur reduction is wrong.
P3  With nu_esc >> S_eff+, u_open+ ~= m u_CRE- + alpha_eff+/nu_esc. So at
    m = 1 the steady error sits just BELOW eps_plateau, the deficit being of
    order S_eff+ tau_esc (about 1% at [15,3], L = 10 cm; about 4% at 20 cm),
    and it does not decay: the transient of the closed parcel becomes a
    permanent offset. At m = 2 and m = 5 the steady error exceeds eps_plateau
    at every pair. At m = 0 the reservoir empties to alpha_eff+/nu_esc and the
    error is set by |(c3/c4)/R_CRE+ - 1|, the pure-recombination ratio, which
    is large wherever the ground-fed fraction is.
P4  The relaxation time equals tau_esc to within the same S_eff+ tau_esc
    fraction, so the open reservoir reaches its steady error within the
    100 us ELM at every pair where tau_esc < 100 us.
P5  Reproduction of the physicist's scratch at heat [15,3], L = 10 cm:
    S_eff 493 / 669 s^-1 (pre / post), u_CRE 1.4122e-2 / 9.8039e-3,
    tau_esc 13.3 us; m=1: u_open+ 1.408e-2, steady error 0.179 against a
    closed-parcel plateau of 0.181; m=2: 0.60; m=5: 1.20; m=0: 0.56.

OUTCOME (recorded after the first run, 11 Sep 2026; predictions above left
as written)
-----------------------------------------------------------------------------
P1 held (4.3e-13). P2 held in form (rel discrepancy x M = 1.4 .. 2.8 at all 46
pairs) but the benchmark missed 4 digits: 2.4e-4 against 1/M = 1.2e-4. P3 held
at m = 1 (ratio 0.87 .. 0.9997, none above 1) and at m = 5 (45/45 above 1 at
both L) but NOT "at every pair" for m = 2: 6/45 at L = 10 cm and 9/45 at
L = 20 cm sit at ratio 0.79 .. 1.00, all of them COOLING steps, where u_CRE
rises by exp(0.38) across the step and a doubled source overshoots it by only
exp(0.31). The direction of the effect (rising source -> larger error) is
unbroken at the heating pairs. P4 held (45/45 at 10 cm, 41/45 at 20 cm relax
inside 100 us). P5 reproduced at every stated digit.

WHAT WOULD REFUTE THE CONCLUSION
--------------------------------
If at m = 1 the steady error were well below eps_plateau across the 45 pairs
(say median ratio < 0.5), or if at m >= 2 it stayed below eps_plateau, then a
fast-exchanging reservoir would indeed reduce the error and the "upper
estimate" reading would stand. If P2 failed by more than O(1/M), the reduced
dynamics would not be the slow dynamics of the operator and nothing built on
S_eff could be trusted.

WHAT THE TEST SHOWS AND DOES NOT SHOW
-------------------------------------
Shows: tau_esc < tau_slow makes the reservoir relax in tau_esc to the
recycling-set value, not to local CRE. A source that is fixed through the
event reproduces the closed-parcel error as a permanent offset (the plateau
never decays, so the time-averaged error over the event is the plateau, i.e.
the UPPER bound of verify_divertor_map.py, not the lower bound that produced
the census). A source that rises with the event, as the ELM target flux does,
gives an error LARGER than the closed-parcel plateau. The chapter 5 magnitudes
are therefore CONDITIONAL on the reservoir history, not upper bounds on it.

Does not show: the actual size of the divertor error. The toy is 0-D; it uses
a single escape time for both pre- and post-step conditions (the CSV value,
evaluated at the pre-step Te; the 5% Te step changes tau_diff by about 5%);
the recycling source is independent of u (no feedback of the local neutral
density on the target flux); there is no ionisation-front motion, no
molecular channel, no spatial profile of the source, and the multipliers
{0, 1, 2, 5} are illustrative, not measured. What the ELM does to the local
source at a given parcel is an input this model cannot supply. The toy fixes
the SIGN of the chapter's inference, not the number.

Read-only. Nothing outside validation/open_reservoir/ is written, and only
with --write.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root                  # noqa: E402

ROOT = find_repo_root(_HERE)

# Physicist's scratch values at heat [15,3], L = 10 cm, for reproduction only
# (check3.py, section (d), 11 Sep 2026). Recorded, not trusted; the run says
# whether they come back.
SCRATCH_15_3 = dict(
    S_eff_pre=493.0, S_eff_post=669.0, u_pre=1.4122e-2, u_post=9.8039e-3,
    tau_esc_us=13.3, u_open_m1=1.408e-2, err_m1=0.179, eps_plateau=0.181,
    err_m2=0.60, err_m5=1.20, err_m0=0.56,
)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_commented_csv(path: Path) -> tuple[list[str], list[dict]]:
    """CSV with leading '#' provenance lines. Returns (header lines, rows)."""
    header, rows = [], []
    with path.open() as fh:
        line = fh.readline()
        while line.startswith("#"):
            header.append(line.rstrip("\n"))
            line = fh.readline()
        rdr = csv.DictReader(fh, fieldnames=line.rstrip("\n").split(","))
        rows = list(rdr)
    return header, rows


def schur_reduce(A: np.ndarray, s: np.ndarray, g: int, E: np.ndarray):
    """
    Eliminate the excited block of dn/dt = A n + s at quasi-steady state.

    Returns S_eff, alpha_eff, a, c with n_E = a u + c (per ion), so that
    du/dt = alpha_eff - S_eff u exactly at the slow-manifold fixed point.
    """
    LEE = A[np.ix_(E, E)]
    a = np.linalg.solve(LEE, -A[E, g])          # excited pop per unit u
    c = np.linalg.solve(LEE, -s[E])             # excited pop from recombination
    S_eff = -(A[g, g] + A[g, E] @ a)
    alpha_eff = s[g] + A[g, E] @ c
    return S_eff, alpha_eff, a, c


def slow_time(A: np.ndarray) -> tuple[float, float]:
    """tau_slow = 1/|lambda_0| and tau_relax = 1/|lambda_1| of the full operator."""
    lam = np.linalg.eigvals(A)
    lam = lam[np.argsort(lam.real)[::-1]]
    if lam[0].real >= 0 or lam[1].real >= 0:
        raise RuntimeError("operator has a non-negative eigenvalue")
    return -1.0 / lam[0].real, -1.0 / lam[1].real


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--map", type=str, default=None,
                    help="divertor_map.csv (default: the canonical one)")
    ap.add_argument("--transport", type=str, default=None,
                    help="transport_selection.csv (default: the canonical one)")
    ap.add_argument("--lengths", type=float, nargs="+", default=[10.0, 20.0])
    ap.add_argument("--multipliers", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 5.0])
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--te-floor", type=float, default=2.0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    # ---- load the pipeline's own files ----------------------------------
    ctx = CRContext.load(root=ROOT)
    L_path = ROOT / "data/processed/cr_matrix/L_grid.npy"
    S_path = ROOT / "data/processed/cr_matrix/S_grid.npy"
    if not S_path.exists():
        raise FileNotFoundError(f"missing source vector: {S_path}")
    L, S = ctx.L_grid, np.load(S_path)
    if S.shape != L.shape[:3]:
        raise ValueError(f"S_grid {S.shape} incompatible with L_grid {L.shape}")
    Te, ne = ctx.te_grid, ctx.ne_grid

    g = int(ctx.ground_index)
    E = np.array([i for i in range(ctx.n_states) if i != g], dtype=int)
    N3 = np.where(np.asarray(ctx.n_values) == 3)[0]
    N4 = np.where(np.asarray(ctx.n_values) == 4)[0]
    if len(N3) != 3 or len(N4) != 4:
        raise ValueError(f"shell membership wrong: n=3 -> {N3}, n=4 -> {N4}")
    pos = {s: k for k, s in enumerate(E)}
    n3E = np.array([pos[s] for s in N3])
    n4E = np.array([pos[s] for s in N4])

    mp = Path(a.map) if a.map else ROOT / "validation/divertor_map/divertor_map.csv"
    tp = (Path(a.transport) if a.transport
          else ROOT / "validation/transport_selection/transport_selection.csv")
    for p in (mp, tp):
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}. This script reads the census and the escape "
                f"times that produced the chapter 6 claim; it does not "
                f"recompute them and it does not substitute anything.")

    hashes = {
        "L_grid.npy": sha256(L_path),
        "S_grid.npy": sha256(S_path),
        "state_index.csv": sha256(ctx.state_index_path),
        "divertor_map.csv": sha256(mp),
        "transport_selection.csv": sha256(tp),
    }

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 78)
    say("OPEN-RESERVOIR TEST OF THE 'UPPER ESTIMATE' INFERENCE (ch6 sec 6.2)")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}")
    say(f"interpreter {sys.executable}   numpy {np.__version__}")
    say(f"repo root {ROOT}")
    for k, v in hashes.items():
        say(f"  sha256 {k:<24} {v}")
    say(f"state ordering {ctx.state_index_path.relative_to(ROOT)}; "
        f"n=3 {[ctx.labels[i] for i in N3]}  n=4 {[ctx.labels[i] for i in N4]}")
    say("=" * 78)

    # ---- the census file: is it built on the operator we just loaded? ----
    map_hdr, map_rows = read_commented_csv(mp)
    recorded = {}
    for h in map_hdr:
        for key in ("L_grid", "S_grid", "state_index"):
            if h.startswith(f"# {key} sha256 "):
                recorded[key] = h.split()[-1]
    for key, mine in (("L_grid", hashes["L_grid.npy"]),
                      ("S_grid", hashes["S_grid.npy"]),
                      ("state_index", hashes["state_index.csv"])):
        if key not in recorded:
            raise RuntimeError(f"{mp} carries no {key} sha256 header line")
        if recorded[key] != mine:
            raise RuntimeError(
                f"{mp} was generated from a {key} with sha256 {recorded[key]}, "
                f"but the file on disk has {mine}. The census is stale "
                f"relative to the operator; regenerate one or the other "
                f"before anything here can be compared.")
    frac_line = next((h for h in map_hdr if "fractional step" in h), None)
    if frac_line is None:
        raise RuntimeError(f"{mp} header does not record the fractional step")
    frac = float(frac_line.split("fractional step")[1].split(",")[0])
    say(f"divertor_map.csv provenance matches the operator on disk; "
        f"fractional step {frac}")

    # ---- the transport file and the defended set --------------------------
    with tp.open() as fh:
        tr_rows = list(csv.DictReader(fh))
    need = {"L_cm", "direction", "i", "j", "tau_esc", "tau_QSS",
            "lo_ELM_crash", "breakdown"}
    if need - set(tr_rows[0]):
        raise RuntimeError(f"{tp} lacks columns {sorted(need - set(tr_rows[0]))}")
    tau_esc_of = {}
    brk_from_transport = {L: set() for L in a.lengths}
    for r in tr_rows:
        Lc = float(r["L_cm"])
        key = (r["direction"], int(r["i"]), int(r["j"]))
        tau_esc_of[(Lc, *key)] = float(r["tau_esc"])
        if Lc in brk_from_transport and r["breakdown"].strip().lower() == "true":
            brk_from_transport[Lc].add(key)
    for Lc in a.lengths:
        if not any(abs(float(r["L_cm"]) - Lc) < 1e-12 for r in tr_rows):
            raise RuntimeError(f"{tp} has no rows at L = {Lc} cm; the escape "
                               f"time at that length was never computed")

    map_of = {}
    brk_from_map = set()
    for r in map_rows:
        key = (r["direction"], int(r["i"]), int(r["j"]))
        map_of[key] = r
        ok = r["window_ok"].strip().lower() == "true"
        if (ok and float(r["Te"]) >= a.te_floor
                and float(r["lo_ELM_crash"]) > a.threshold):
            brk_from_map.add(key)
    for Lc in a.lengths:
        if brk_from_transport[Lc] != brk_from_map:
            raise RuntimeError(
                f"the breakdown set flagged in {tp.name} at L = {Lc} cm "
                f"({len(brk_from_transport[Lc])} pairs) differs from the one "
                f"recomputed from {mp.name} (window_ok, Te >= {a.te_floor}, "
                f"lo_ELM_crash > {a.threshold}: {len(brk_from_map)} pairs). "
                f"The two files disagree about which pairs carry the result.")
    pairs = sorted(brk_from_map, key=lambda t: (t[0], t[1], t[2]))
    say(f"defended breakdown set (window_ok, Te >= {a.te_floor} eV, "
        f"lo_ELM_crash > {a.threshold}): {len(pairs)} pairs, "
        f"{sum(p[0] == 'heat' for p in pairs)} heating, "
        f"{sum(p[0] == 'cool' for p in pairs)} cooling; identical in both files")
    if len(pairs) != 45:
        say(f"  NOTE chapter 6 speaks of 45 pairs; this file yields {len(pairs)}")

    i_bm, j_bm = ctx.nearest_point(3.0, 1e14)
    bm_key = ("heat", i_bm, j_bm)
    if bm_key not in map_of:
        raise RuntimeError(f"benchmark pair {bm_key} absent from {mp.name}")
    say(f"benchmark pair {bm_key}: Te = {Te[i_bm]:.4f} eV, ne = {ne[j_bm]:.4e} "
        f"cm^-3, in the breakdown set: {bm_key in brk_from_map}")

    # ---- per-pair reduction ---------------------------------------------
    say()
    say("=" * 78)
    say("STEP 1  Schur reduction against the full operator")
    say("=" * 78)
    say("  columns: pair | M | alpha/S vs u_CRE (rel) | 1/S_eff+ vs tau_QSS "
        "(rel) | eps_plateau recomputed vs csv (rel)")

    per_pair = {}
    worst_u, worst_tau, worst_eps, worst_frac = 0.0, 0.0, 0.0, 0.0
    n_tau_fail = 0
    for key in pairs + [bm_key]:
        d, i, j = key
        sgn = +1 if d == "heat" else -1
        k = int(np.argmin(np.abs(Te - Te[i] * (1 + sgn * frac))))
        if k == i:
            raise RuntimeError(f"step did not move a grid index at {key}")
        r = map_of[key]
        fa_csv = float(r["frac_achieved"])
        fa = Te[k] / Te[i] - 1.0
        worst_frac = max(worst_frac, abs(fa - fa_csv))
        if abs(fa - fa_csv) > 1e-12:
            raise RuntimeError(
                f"{key}: reconstructed step {fa:+.6e} differs from the "
                f"frac_achieved {fa_csv:+.6e} recorded in {mp.name}; the "
                f"post-step operator would not be the one the census used")

        Sm, am, _, _ = schur_reduce(L[i, j], S[i, j], g, E)
        Sp, ap_, a_vec, c_vec = schur_reduce(L[k, j], S[k, j], g, E)
        u_m = float(np.linalg.solve(L[i, j], -S[i, j])[g])
        u_p = float(np.linalg.solve(L[k, j], -S[k, j])[g])
        rel_u = max(abs(am / Sm / u_m - 1.0), abs(ap_ / Sp / u_p - 1.0))
        worst_u = max(worst_u, rel_u)
        if rel_u > 1e-8:
            raise RuntimeError(
                f"{key}: alpha_eff/S_eff differs from the full CRE ground "
                f"fraction by {rel_u:.2e}. This is an exact identity; the "
                f"reduction in this script is wrong.")

        tQ_op, tR_op = slow_time(L[k, j])
        tQ_csv = float(r["tau_QSS"])
        if abs(tQ_op / tQ_csv - 1.0) > 1e-9:
            raise RuntimeError(
                f"{key}: tau_QSS of L[{k},{j}] is {tQ_op:.6e} s but {mp.name} "
                f"records {tQ_csv:.6e}; the census is not built on this operator")
        rel_tau = abs((1.0 / Sp) / tQ_op - 1.0)
        worst_tau = max(worst_tau, rel_tau)
        M = tQ_op / tR_op
        if rel_tau > 1e-4:
            n_tau_fail += 1

        a3, a4 = a_vec[n3E].sum(), a_vec[n4E].sum()
        c3, c4 = c_vec[n3E].sum(), c_vec[n4E].sum()

        def R(u):
            return (a3 * u + c3) / (a4 * u + c4)

        R_p = R(u_p)
        eps_pl = abs(R(u_m) / R_p - 1.0)
        eps_csv = float(r["eps_plateau"])
        rel_eps = abs(eps_pl / eps_csv - 1.0)
        worst_eps = max(worst_eps, rel_eps)
        if rel_eps > 1e-9:
            raise RuntimeError(
                f"{key}: eps_plateau recomputed {eps_pl:.9e} vs csv "
                f"{eps_csv:.9e}; the two-channel coefficients here are not "
                f"the ones verify_divertor_map.py used")

        per_pair[key] = dict(
            direction=d, i=i, j=j, k=k, Te_pre=Te[i], Te_post=Te[k], ne=ne[j],
            frac_achieved=fa, M=M, tau_QSS=tQ_op, tau_relax=tR_op,
            S_eff_pre=Sm, S_eff_post=Sp, alpha_eff_pre=am, alpha_eff_post=ap_,
            u_CRE_pre=u_m, u_CRE_post=u_p,
            rel_err_u_identity=rel_u, rel_err_invS_vs_tauQSS=rel_tau,
            a3=a3, a4=a4, c3=c3, c4=c4, R_CRE_post=R_p, R_zero_u=c3 / c4,
            eps_plateau=eps_pl, eps_plateau_csv=eps_csv,
            lo_ELM_crash=float(r["lo_ELM_crash"]),
            in_breakdown_set=key in brk_from_map,
        )
        tag = "" if rel_tau <= 1e-4 else "   <-- 1/S_eff misses tau_QSS at 4 digits"
        say(f"  {d:<4} [{i:>2},{j}] -> [{k:>2},{j}]  M={M:9.3e}  "
            f"{rel_u:8.1e}  {rel_tau:8.1e}  {rel_eps:8.1e}{tag}")

    say()
    say(f"  worst alpha/S vs u_CRE         {worst_u:.2e}   (identity; raise if > 1e-8)")
    say(f"  worst 1/S_eff+ vs tau_QSS       {worst_tau:.2e}   "
        f"({n_tau_fail} of {len(per_pair)} pairs beyond 1e-4)")
    say(f"  worst eps_plateau vs csv        {worst_eps:.2e}")
    say(f"  worst frac_achieved vs csv      {worst_frac:.2e}")
    Ms = np.array([v["M"] for v in per_pair.values()])
    rts = np.array([v["rel_err_invS_vs_tauQSS"] for v in per_pair.values()])
    say(f"  P2 check: rel discrepancy x M ranges "
        f"{(rts * Ms).min():.3f} .. {(rts * Ms).max():.3f}  "
        f"(O(1) means the discrepancy is the expected 1/M QSS correction)")

    # ---- the open reservoir ----------------------------------------------
    say()
    say("=" * 78)
    say("STEP 2  Open reservoir: escape sink + recycling source")
    say("=" * 78)
    say("  du/dt = alpha_eff + m*Gamma_0 - (S_eff + nu_esc) u,  "
        "Gamma_0 = (S_eff- + nu_esc) u_CRE- - alpha_eff-")
    say("  steady error = |R(u_open+)/R(u_CRE+) - 1| with post-step "
        "two-channel coefficients")

    out_rows = []
    worst_gamma = 0.0
    for Lc in a.lengths:
        for key, v in per_pair.items():
            tkey = (Lc, *key)
            if tkey not in tau_esc_of:
                raise RuntimeError(f"{tp.name} has no tau_esc for {key} at "
                                   f"L = {Lc} cm")
            tau_esc = tau_esc_of[tkey]
            nu = 1.0 / tau_esc
            Gamma0 = (v["S_eff_pre"] + nu) * v["u_CRE_pre"] - v["alpha_eff_pre"]
            worst_gamma = max(worst_gamma,
                              abs(Gamma0 / (nu * v["u_CRE_pre"]) - 1.0))
            u_open_pre = (v["alpha_eff_pre"] + Gamma0) / (v["S_eff_pre"] + nu)
            if abs(u_open_pre / v["u_CRE_pre"] - 1.0) > 1e-10:
                raise RuntimeError(f"{key} L={Lc}: pre-step open steady state "
                                   f"does not equal pre-step CRE; the source "
                                   f"was mis-set")
            relax = 1.0 / (v["S_eff_post"] + nu)
            a3, a4, c3, c4 = v["a3"], v["a4"], v["c3"], v["c4"]
            for m in a.multipliers:
                u_open = (v["alpha_eff_post"] + m * Gamma0) / (v["S_eff_post"] + nu)
                R_open = (a3 * u_open + c3) / (a4 * u_open + c4)
                err = abs(R_open / v["R_CRE_post"] - 1.0)
                out_rows.append(dict(
                    set="benchmark" if key == bm_key and not v["in_breakdown_set"]
                    else "breakdown45",
                    L_cm=Lc, source_multiplier=m,
                    direction=v["direction"], i=v["i"], j=v["j"], k_post=v["k"],
                    Te_pre=v["Te_pre"], Te_post=v["Te_post"], ne=v["ne"],
                    M=v["M"], tau_QSS=v["tau_QSS"], tau_relax=v["tau_relax"],
                    tau_esc=tau_esc, tau_esc_over_tau_QSS=tau_esc / v["tau_QSS"],
                    S_eff_pre=v["S_eff_pre"], S_eff_post=v["S_eff_post"],
                    alpha_eff_pre=v["alpha_eff_pre"],
                    alpha_eff_post=v["alpha_eff_post"],
                    u_CRE_pre=v["u_CRE_pre"], u_CRE_post=v["u_CRE_post"],
                    rel_err_u_identity=v["rel_err_u_identity"],
                    rel_err_invS_vs_tauQSS=v["rel_err_invS_vs_tauQSS"],
                    a3=a3, a4=a4, c3=c3, c4=c4,
                    R_CRE_post=v["R_CRE_post"], R_zero_u=v["R_zero_u"],
                    Gamma0=Gamma0, nu_esc=nu,
                    u_open_post=u_open,
                    ln_u_open_over_u_CRE_post=np.log(u_open / v["u_CRE_post"]),
                    steady_err=err,
                    eps_plateau=v["eps_plateau"],
                    steady_err_over_eps_plateau=err / v["eps_plateau"],
                    lo_ELM_crash=v["lo_ELM_crash"],
                    steady_err_over_lo_ELM_crash=err / v["lo_ELM_crash"],
                    relax_time=relax, relax_over_tau_esc=relax / tau_esc,
                ))
    say(f"  P1 check: worst |Gamma_0/(nu_esc u_CRE-) - 1| = {worst_gamma:.2e}")
    say(f"  pre-step open steady state equals pre-step CRE at every pair "
        f"(asserted to 1e-10)")

    # ---- reproduction of the scratch at heat [15,3], L = 10 cm ------------
    say()
    say("=" * 78)
    say("STEP 3  Reproduction of the physicist's scratch (check3.py (d)) at "
        "heat [15,3], L = 10 cm")
    say("=" * 78)
    k153 = ("heat", 15, 3)
    if k153 in per_pair and 10.0 in a.lengths:
        v = per_pair[k153]
        rows153 = {r["source_multiplier"]: r for r in out_rows
                   if (r["direction"], r["i"], r["j"]) == k153 and r["L_cm"] == 10.0}
        sc = SCRATCH_15_3

        def cmp(label, got, rec, digits):
            ok = abs(got - rec) <= 0.5 * 10.0 ** (np.floor(np.log10(abs(rec))) - digits + 1)
            say(f"    {label:<28} got {got:<14.6g} scratch {rec:<10.5g} "
                f"{'agrees' if ok else 'DISAGREES'} at {digits} sig. fig.")
            return ok

        all_ok = True
        all_ok &= cmp("S_eff pre  [s^-1]", v["S_eff_pre"], sc["S_eff_pre"], 3)
        all_ok &= cmp("S_eff post [s^-1]", v["S_eff_post"], sc["S_eff_post"], 3)
        all_ok &= cmp("u_CRE pre", v["u_CRE_pre"], sc["u_pre"], 5)
        all_ok &= cmp("u_CRE post", v["u_CRE_post"], sc["u_post"], 5)
        all_ok &= cmp("tau_esc [us]", rows153[1.0]["tau_esc"] * 1e6, sc["tau_esc_us"], 3)
        all_ok &= cmp("eps_plateau (closed parcel)", v["eps_plateau"], sc["eps_plateau"], 3)
        all_ok &= cmp("m=1  u_open+", rows153[1.0]["u_open_post"], sc["u_open_m1"], 4)
        all_ok &= cmp("m=1  steady error", rows153[1.0]["steady_err"], sc["err_m1"], 3)
        all_ok &= cmp("m=2  steady error", rows153[2.0]["steady_err"], sc["err_m2"], 2)
        all_ok &= cmp("m=5  steady error", rows153[5.0]["steady_err"], sc["err_m5"], 3)
        all_ok &= cmp("m=0  steady error", rows153[0.0]["steady_err"], sc["err_m0"], 2)
        say(f"  scratch reproduced at the stated digits: {all_ok}")
        say(f"  1/S_eff+ = {1e3 / v['S_eff_post']:.4f} ms vs tau_QSS "
            f"{v['tau_QSS'] * 1e3:.4f} ms   relax time at L=10 cm "
            f"{rows153[1.0]['relax_time'] * 1e6:.2f} us")
    else:
        say("  heat [15,3] is not in the defended set of this file, or 10 cm "
            "not requested; scratch comparison skipped")

    # ---- summary over the 45 pairs ---------------------------------------
    say()
    say("=" * 78)
    say("STEP 4  Summary over the defended breakdown set")
    say("=" * 78)
    say(f"  {'L':>4} {'m':>4} {'n':>3} | {'steady err':^28} | "
        f"{'err / eps_plateau':^28} | {'>1':>5} | {'err / lo_ELM':^20} | "
        f"{'relax [us]':^20} | {'ln(u+/uCRE+)':^14}")
    say(f"  {'':>4} {'':>4} {'':>3} | {'median':>8} {'min':>9} {'max':>9} | "
        f"{'median':>8} {'min':>9} {'max':>9} | {'':>5} | {'median':>9} {'max':>10} | "
        f"{'median':>9} {'max':>10} | {'median':>14}")
    summary = []
    for Lc in a.lengths:
        for m in a.multipliers:
            sel = [r for r in out_rows if r["set"] == "breakdown45"
                   and r["L_cm"] == Lc and r["source_multiplier"] == m]
            if not sel:
                raise RuntimeError(f"no breakdown rows at L={Lc}, m={m}")
            err = np.array([r["steady_err"] for r in sel])
            rat = np.array([r["steady_err_over_eps_plateau"] for r in sel])
            rlo = np.array([r["steady_err_over_lo_ELM_crash"] for r in sel])
            rx = np.array([r["relax_time"] for r in sel]) * 1e6
            lnu = np.array([r["ln_u_open_over_u_CRE_post"] for r in sel])
            row = dict(L_cm=Lc, source_multiplier=m, n=len(sel),
                       steady_err_median=np.median(err), steady_err_min=err.min(),
                       steady_err_max=err.max(),
                       ratio_to_eps_plateau_median=np.median(rat),
                       ratio_to_eps_plateau_min=rat.min(),
                       ratio_to_eps_plateau_max=rat.max(),
                       n_ratio_above_1=int((rat > 1.0).sum()),
                       n_steady_err_above_threshold=int((err > a.threshold).sum()),
                       ratio_to_lo_ELM_median=np.median(rlo),
                       ratio_to_lo_ELM_max=rlo.max(),
                       relax_time_us_median=np.median(rx),
                       relax_time_us_max=rx.max(),
                       relax_over_tau_esc_median=np.median(
                           [r["relax_over_tau_esc"] for r in sel]),
                       ln_u_ratio_median=np.median(lnu),
                       ln_u_ratio_min=lnu.min(), ln_u_ratio_max=lnu.max())
            summary.append(row)
            say(f"  {Lc:>4g} {m:>4g} {len(sel):>3} | {np.median(err):8.4f} "
                f"{err.min():9.4f} {err.max():9.4f} | {np.median(rat):8.4f} "
                f"{rat.min():9.4f} {rat.max():9.4f} | {int((rat > 1).sum()):>2}/"
                f"{len(sel):<2} | {np.median(rlo):9.3f} {rlo.max():10.3f} | "
                f"{np.median(rx):9.2f} {rx.max():10.2f} | {np.median(lnu):+14.4f}")
    say()
    say("  Rows with m > 1 whose steady error stays BELOW the closed-parcel "
        "plateau (P3 said none):")
    below = [r for r in out_rows if r["set"] == "breakdown45"
             and r["source_multiplier"] > 1.0
             and r["steady_err_over_eps_plateau"] <= 1.0]
    if not below:
        say("    none")
    for r in below:
        say(f"    {r['direction']:<4} [{r['i']:>2},{r['j']}] L={r['L_cm']:>4g} "
            f"m={r['source_multiplier']:g}  ln(u+/uCRE+)={r['ln_u_open_over_u_CRE_post']:+.4f} "
            f"(closed parcel: ln(uCRE-/uCRE+)={np.log(r['u_CRE_pre'] / r['u_CRE_post']):+.4f})  "
            f"err={r['steady_err']:.4f}  eps_plateau={r['eps_plateau']:.4f}  "
            f"ratio={r['steady_err_over_eps_plateau']:.3f}")
    n_cool_below = sum(r["direction"] == "cool" for r in below)
    say(f"    {len(below)} rows, {n_cool_below} of them cooling steps. On a cooling "
        f"step u_CRE rises, so a source that")
    say("    rises by a similar factor moves u_open+ past u_CRE+ by less than "
        "the stale value fell short of it.")
    say()
    say("  Reading: at m = 1 the ratio err/eps_plateau is 1 - O(S_eff+ tau_esc)")
    say("  (the plateau persists as a permanent offset; since it never decays,")
    say("  the time-averaged error over the event is the plateau itself, i.e.")
    say("  above the lo_ELM_crash bound that produced the census). At m >= 2 it")
    say("  exceeds the plateau. At m = 0 the reservoir empties and the error is")
    say("  |(c3/c4)/R_CRE+ - 1|, the pure-recombination ratio.")
    for Lc in a.lengths:
        sel = [r for r in out_rows if r["set"] == "breakdown45"
               and r["L_cm"] == Lc and r["source_multiplier"] == a.multipliers[0]]
        rx = np.array([r["relax_time"] for r in sel])
        say(f"  L = {Lc:g} cm: relaxation time of the open reservoir "
            f"{rx.min() * 1e6:.2f} .. {rx.max() * 1e6:.2f} us; "
            f"{int((rx < 1e-4).sum())}/{len(sel)} pairs reach steady state "
            f"inside a 100 us ELM")

    # ---- the benchmark row ------------------------------------------------
    say()
    say("=" * 78)
    say("STEP 5  Benchmark pair heat [%d,%d]  (Te = %.4f eV, ne = %.4e cm^-3)"
        % (i_bm, j_bm, Te[i_bm], ne[j_bm]))
    say("=" * 78)
    v = per_pair[bm_key]
    say(f"  in defended breakdown set: {v['in_breakdown_set']}   "
        f"lo_ELM_crash = {v['lo_ELM_crash']:.4f}   eps_plateau = {v['eps_plateau']:.5f}")
    say(f"  M = {v['M']:.4g}   tau_QSS = {v['tau_QSS'] * 1e6:.3f} us   "
        f"1/S_eff+ = {1e6 / v['S_eff_post']:.3f} us   rel diff "
        f"{v['rel_err_invS_vs_tauQSS']:.2e}  (1/M = {1 / v['M']:.2e})")
    say(f"  S_eff pre/post = {v['S_eff_pre']:.4e} / {v['S_eff_post']:.4e} s^-1   "
        f"u_CRE pre/post = {v['u_CRE_pre']:.5e} / {v['u_CRE_post']:.5e}")
    for r in out_rows:
        if (r["direction"], r["i"], r["j"]) == bm_key:
            say(f"  L={r['L_cm']:>4g} cm  tau_esc={r['tau_esc'] * 1e6:7.2f} us "
                f"(tau_esc/tau_QSS={r['tau_esc_over_tau_QSS']:.3f})  "
                f"m={r['source_multiplier']:g}: u_open+={r['u_open_post']:.4e}  "
                f"ln(u+/uCRE+)={r['ln_u_open_over_u_CRE_post']:+.4f}  "
                f"steady err={r['steady_err']:.4f}  "
                f"/eps_plateau={r['steady_err_over_eps_plateau']:.3f}  "
                f"relax={r['relax_time'] * 1e6:.2f} us")
    say("  Note: at the benchmark tau_esc is NOT short against tau_QSS "
        "(ratio 3.6 and 14), so the")
    say("  reservoir is closer to closed there; the m = 1 error falls below the "
        "plateau by the")
    say("  fraction S_eff+ tau_esc, which is no longer small. That is what the "
        "toy predicts")
    say("  when the chapter 6 condition tau_esc < tau_slow does not hold.")

    # ---- verdict ----------------------------------------------------------
    say()
    say("=" * 78)
    say("VERDICT")
    say("=" * 78)
    s1 = [s for s in summary if s["source_multiplier"] == 1.0]
    s2 = [s for s in summary if s["source_multiplier"] > 1.0]
    med1 = min(s["ratio_to_eps_plateau_median"] for s in s1) if s1 else np.nan
    n_above = sum(s["n_ratio_above_1"] for s in s2)
    n_tot = sum(s["n"] for s in s2)
    n_below = n_tot - n_above
    say(f"  m = 1: smallest median err/eps_plateau over the lengths = {med1:.4f}; "
        f"refuter (median < 0.5) {'APPEARED' if med1 < 0.5 else 'did not appear'}")
    say(f"  m > 1: err/eps_plateau > 1 at {n_above}/{n_tot} (pair, L, m) rows and "
        f"<= 1 at {n_below}; refuter (stays below at most pairs) "
        f"{'APPEARED' if n_above < n_tot / 2 else 'did not appear'}")
    say(f"  P3 as written ('at every pair') is refuted by the {n_below} rows listed "
        f"in STEP 4; the direction of the")
    say("  effect (rising source -> larger error) holds at every heating pair and "
        "at m = 5 everywhere.")
    say("  A fast-exchanging reservoir with a steady source freezes the closed-")
    say("  parcel error rather than removing it; a rising source enlarges it at")
    say("  the heating pairs and at large multipliers everywhere. 'Upper estimate'")
    say("  is therefore not what tau_esc < tau_slow implies. The chapter 5")
    say("  magnitudes are conditional on the reservoir history.")
    say()
    say("  Assumptions of this toy (all stated, none tested here): 0-D parcel;")
    say("  one escape time per pair, taken at the pre-step Te and held through")
    say("  the step; recycling source independent of u; no ionisation-front")
    say("  motion, no molecular channel, no spatial source profile; source")
    say("  multipliers illustrative. The toy fixes the sign of the inference,")
    say("  not the magnitude of the divertor error.")

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "open_reservoir"
        out.mkdir(parents=True, exist_ok=True)
        prov = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}",
                f"# interpreter {sys.executable}  numpy {np.__version__}"]
        prov += [f"# {k} sha256 {v}" for k, v in hashes.items()]
        prov += [f"# fractional step {frac}; threshold {a.threshold}; "
                 f"te_floor {a.te_floor}; lengths {a.lengths}; "
                 f"multipliers {a.multipliers}"]
        with (out / "open_reservoir.csv").open("w", newline="") as fh:
            fh.write("\n".join(prov) + "\n")
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        with (out / "open_reservoir_summary.csv").open("w", newline="") as fh:
            fh.write("\n".join(prov) + "\n")
            w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        (out / "open_reservoir.txt").write_text("\n".join(lines) + "\n")
        print(f"\nwrote {out / 'open_reservoir.csv'}  ({len(out_rows)} rows)")
        print(f"wrote {out / 'open_reservoir_summary.csv'}  ({len(summary)} rows)")
        print(f"wrote {out / 'open_reservoir.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
