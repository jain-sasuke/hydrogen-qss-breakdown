#!/usr/bin/env python
"""
diagnose_gate_d.py
==================
Why does Gate D fail at 400 of 400 points, and is the model or the gate wrong?

WHY THIS EXISTS
---------------
validate_gates.py reports Gate D, the comparison against ADAS effective
coefficients, as FAIL at 0% of points. The recorded eta = SCD_model/SCD_ADAS
runs from about 6 to 5600 and grows almost exactly in proportion to n_e. A
model error would not produce a clean power law in a variable the coefficient
is supposed to be nearly independent of. That pattern is the reason to suspect
the comparison before suspecting the model.

Gate D is also half built: it loads ACD96 and never uses it, so the
recombination side has never been compared at all.

THE HYPOTHESIS, STATED BEFORE IT IS TESTED
------------------------------------------
Gate D computes

    SCD_model = sum_p K_ion(p) n_p^CRE / sum_p n_p^CRE

from the FULL collisional-radiative equilibrium, which contains both supply
channels: population fed from the ground state and population fed by
recombination from the ion. ADAS SCD is defined for the ground-fed channel
alone; the recombination-fed population is the subject of ACD, which is a
separate coefficient precisely so that the two are not mixed. At low Te and
high n_e the CRE state is recombination-dominated and three-body recombination
carries an extra factor of n_e, which is where the observed n_e scaling would
come from.

If that is the explanation, then rebuilding SCD on the ground-fed channel alone
should collapse eta toward unity, and it should do so WITHOUT any tolerance
being touched.

The refuting observation, stated in advance: if the ground-fed SCD is still
orders of magnitude from ADAS, or still scales with n_e, the hypothesis is
wrong and the model has a real disagreement with ADAS that must be reported as
one.

THE TWO CHANNELS ARE THIS THESIS'S OWN DECOMPOSITION
----------------------------------------------------
With g the ground index and E the 42 excited states,

    n_E^(0) = -L_EE^{-1} S_E n_ion      recombination-fed
    n_E^(1) = -L_EE^{-1} L_Eg n_g       ground-fed
    n_E^CRE = n_E^(0) + n_E^(1)         exact, no approximation

so

    SCD = K_ion(g) + sum_{p in E} K_ion(p) n_p^(1) / n_g
    ACD = [ S_g n_ion + sum_{p in E} L_gp n_p^(0) ] / (n_e n_ion)

SCD is the ionisation rate per ground-state atom, counting direct ionisation
and ionisation out of levels the ground state populated. ACD is the net flux
into the ground state carried by the recombination-fed channel, per unit
n_e n_ion: direct radiative recombination into 1s plus everything that cascades
or is collisionally transferred down into it.

REPORT, DO NOT REPAIR
---------------------
This script does not modify validate_gates.py and does not change any
tolerance. It computes both definitions side by side and says which one matches
ADAS. Deciding what Gate D should test is a separate act, and it belongs to
whoever owns the gate.

Read-only. Writes only to validation/gate_D_diagnosis/, and only with --write.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root          # noqa: E402

ROOT = find_repo_root(_HERE)

ADAS_SCD = ROOT / "data/processed/adas/SCD96_interpolated.csv"
ADAS_ACD = ROOT / "data/processed/adas/ACD96_interpolated.csv"
K_ION = ROOT / "data/processed/collisions/tics/K_ion_final.npy"
S_GRID = ROOT / "data/processed/cr_matrix/S_grid.npy"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_adas(path: Path, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Te, ne, value) columns. No interpolation here; the caller does
    it, so that the interpolation is visible rather than hidden in a loader."""
    if not path.exists():
        raise RuntimeError(
            f"missing {path}. Gate D cannot be diagnosed without the ADAS "
            f"table it compares against, and no stand-in is acceptable.")
    import csv as _csv
    te, ne, v = [], [], []
    with path.open() as fh:
        for row in _csv.DictReader(fh):
            te.append(float(row["Te_eV"]))
            ne.append(float(row["ne_cm3"]))
            v.append(float(row[col]))
    return np.array(te), np.array(ne), np.array(v)


def adas_at(te_tab, ne_tab, v_tab, Te, ne, band=(0.5, 2.0)) -> float:
    """Interpolate in Te among the ADAS rows whose density is within `band` of
    ne. This is the same selection Gate D makes, kept deliberately identical so
    that any difference found below is a difference in the MODEL side and not
    in how the table was read."""
    r = ne_tab / ne
    m = (r >= band[0]) & (r <= band[1])
    if m.sum() < 2:
        return float("nan")
    o = np.argsort(te_tab[m])
    return float(np.interp(Te, te_tab[m][o], v_tab[m][o]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    ctx = CRContext.load()
    ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid
    g = ctx.ground_index
    E = np.array([i for i in range(ctx.n_states) if i != g])

    for p in (K_ION, S_GRID):
        if not p.exists():
            raise RuntimeError(f"missing {p}")
    K_ion = np.load(K_ION)          # (n_states, n_Te), cm^3/s
    S = np.load(S_GRID)             # (n_Te, n_ne, n_states), s^-1 per unit n_ion
    if K_ion.shape != (ctx.n_states, len(te)):
        raise RuntimeError(f"K_ion shape {K_ion.shape} does not match the "
                           f"{ctx.n_states}-state grid of {len(te)} temperatures")

    te_s, ne_s, scd_s = load_adas(ADAS_SCD, "SCD_cm3_s")
    te_a, ne_a, acd_a = load_adas(ADAS_ACD, "ACD_cm3_s")

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    print(f"  L_grid  {L.shape}  sha256 "
          f"{sha256(ROOT/'data/processed/cr_matrix/L_grid.npy')[:32]}...")
    print(f"  S_grid  {S.shape}  sha256 {sha256(S_GRID)[:32]}...")
    print(f"  K_ion   {K_ion.shape}  sha256 {sha256(K_ION)[:32]}...")
    print(f"  SCD96   {len(scd_s)} rows  sha256 {sha256(ADAS_SCD)[:32]}...")
    print(f"  ACD96   {len(acd_a)} rows  sha256 {sha256(ADAS_ACD)[:32]}...")
    print(f"  ground index {g} ({ctx.labels[g]})   interpreter {sys.executable}")
    print()

    nT, nN = len(te), len(ne)
    eta_cre = np.full((nT, nN), np.nan)      # Gate D's definition
    eta_gf = np.full((nT, nN), np.nan)       # ground-fed only
    eta_acd = np.full((nT, nN), np.nan)      # the half that was never built
    scd_gf_arr = np.full((nT, nN), np.nan)
    acd_arr = np.full((nT, nN), np.nan)
    f_rec = np.full((nT, nN), np.nan)        # recombination-fed share of the
                                             # ionisation flux, the suspect

    for i in range(nT):
        for j in range(nN):
            A = L[i, j]
            n_ion = ne[j]                    # quasi-neutral, as Gate D assumes
            Sv = S[i, j] * n_ion

            n_cre = np.linalg.solve(A, -Sv)
            if np.any(n_cre <= 0):
                raise RuntimeError(f"non-positive CRE population at [{i},{j}]")

            LEE = A[np.ix_(E, E)]
            n0 = np.linalg.solve(LEE, -Sv[E])                   # recomb-fed
            n1 = np.linalg.solve(LEE, -A[np.ix_(E, [g])].ravel() * n_cre[g])
            resid = np.abs(n0 + n1 - n_cre[E]).max() / np.abs(n_cre[E]).max()
            if resid > 1e-8:
                raise RuntimeError(
                    f"the two-channel split does not reproduce the CRE state "
                    f"at [{i},{j}]: relative residual {resid:.3e}")

            # Gate D's quantity, recomputed here to prove it is the same one.
            scd_cre = float(K_ion[:, i] @ n_cre) / float(n_cre.sum())
            # The ground-fed coefficient.
            scd_gf = float(K_ion[g, i] + (K_ion[E, i] @ n1) / n_cre[g])
            # The recombination side, which Gate D loads and never uses.
            into_g = float(A[g, E] @ n0) + float(Sv[g])
            acd = into_g / (ne[j] * n_ion)

            f_rec[i, j] = float(K_ion[E, i] @ n0) / float(K_ion[:, i] @ n_cre)
            scd_gf_arr[i, j] = scd_gf
            acd_arr[i, j] = acd

            s_ref = adas_at(te_s, ne_s, scd_s, te[i], ne[j])
            a_ref = adas_at(te_a, ne_a, acd_a, te[i], ne[j])
            if np.isfinite(s_ref) and s_ref > 0:
                eta_cre[i, j] = scd_cre / s_ref
                eta_gf[i, j] = scd_gf / s_ref
            if np.isfinite(a_ref) and a_ref > 0:
                eta_acd[i, j] = acd / a_ref

    def band(name, arr, lo=0.5, hi=2.0):
        f = np.isfinite(arr)
        if not f.any():
            raise RuntimeError(f"{name}: no point could be compared to ADAS")
        v = arr[f]
        inside = ((v >= lo) & (v <= hi)).mean()
        print(f"  {name:<34} min {v.min():10.4g}  median {np.median(v):10.4g}"
              f"  max {v.max():10.4g}   within a factor 2: "
              f"{100*inside:5.1f}%  ({int(((v>=lo)&(v<=hi)).sum())}/{v.size})")
        return inside

    print("=" * 78)
    print("SCD: TWO DEFINITIONS AGAINST THE SAME ADAS TABLE")
    print("=" * 78)
    in_cre = band("eta, Gate D's CRE definition", eta_cre)
    in_gf = band("eta, ground-fed channel only", eta_gf)
    print()
    print("  Gate D passes at 70% agreement within a factor 2. The two rows")
    print("  above use the same ADAS reading, the same grid and the same")
    print("  tolerance; only the model-side definition differs.")
    print()

    print("=" * 78)
    print("IS THE n_e SCALING THE RECOMBINATION CHANNEL?")
    print("=" * 78)
    print("  If it is, the recombination-fed share of the ionisation flux must")
    print("  rise with n_e and fall with Te, and eta must track it.")
    for j in range(nN):
        f = np.isfinite(eta_cre[:, j])
        print(f"    ne = {ne[j]:9.3g}   recomb-fed share of sum K_ion n_p: "
              f"{f_rec[0, j]*100:6.2f}% at 1 eV, {f_rec[-1, j]*100:6.2f}% at "
              f"{te[-1]:.3g} eV   |   median eta(CRE) "
              f"{np.median(eta_cre[f, j]):9.4g}, eta(ground-fed) "
              f"{np.median(eta_gf[f, j]):7.4g}")
    print()

    print("=" * 78)
    print("ACD: THE HALF OF GATE D THAT WAS NEVER BUILT")
    print("=" * 78)
    band("eta, ACD from the recomb-fed channel", eta_acd)
    ib, jb = 23, 5
    print(f"    at the benchmark [{ib},{jb}]: ACD_model "
          f"{acd_arr[ib,jb]:.4e} cm^3/s, ADAS "
          f"{adas_at(te_a, ne_a, acd_a, te[ib], ne[jb]):.4e}, ratio "
          f"{eta_acd[ib,jb]:.4f}")
    print(f"    at the benchmark [{ib},{jb}]: SCD_model (ground-fed) "
          f"{scd_gf_arr[ib,jb]:.4e} cm^3/s, ADAS "
          f"{adas_at(te_s, ne_s, scd_s, te[ib], ne[jb]):.4e}, ratio "
          f"{eta_gf[ib,jb]:.4f}")
    print()

    # ---- where does the ground-fed ionisation flux come from? ------------
    print("=" * 78)
    print("THE SCD DEFICIT AND THE TOP OF THE LADDER")
    print("=" * 78)
    print("  The ground-fed SCD sits below ADAS everywhere and the gap closes")
    print("  as Te rises. The obvious suspect is the truncation at n = 15:")
    print("  ADAS96 carries levels this model does not, and the levels nearest")
    print("  the continuum ionise most easily. If that is the cause, the")
    print("  deficit must track the share of the flux carried by the top of")
    print("  the retained ladder.")
    nv = ctx.n_values
    top = np.array([k for k, s in enumerate(E) if nv[s] >= 12])
    share = np.full((nT, nN), np.nan)
    for i in range(nT):
        for j in range(nN):
            A = L[i, j]
            Sv = S[i, j] * ne[j]
            n_cre = np.linalg.solve(A, -Sv)
            LEE = A[np.ix_(E, E)]
            n1 = np.linalg.solve(LEE, -A[np.ix_(E, [g])].ravel() * n_cre[g])
            tot = float(K_ion[g, i] * n_cre[g] + K_ion[E, i] @ n1)
            share[i, j] = float(K_ion[E[top], i] @ n1[top]) / tot
    fin = np.isfinite(eta_gf)
    x, y = share[fin], eta_gf[fin]
    r = float(np.corrcoef(np.log(x), np.log(1.0 / y))[0, 1])
    print(f"    share of the ground-fed ionisation flux carried by n >= 12: "
          f"{x.min()*100:.2f}% to {x.max()*100:.2f}%, median {np.median(x)*100:.2f}%")
    print(f"    correlation of log(share) with log(ADAS/model): r = {r:+.4f}")
    print("    A positive r is consistent with truncation; it is not proof,")
    print("    because Te drives both. The n_max convergence study in")
    print("    chapter 6 is the test that would settle it, and it is not this.")
    print()

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if in_gf > in_cre:
        print(f"  The ground-fed definition agrees with ADAS at "
              f"{100*in_gf:.1f}% of points against {100*in_cre:.1f}% for the")
        print("  CRE definition Gate D uses. The gate is comparing a mixed")
        print("  quantity against a table that separates the two channels.")
        print("  REPORTED, NOT REPAIRED: validate_gates.py is unchanged.")
    else:
        print("  The ground-fed definition does NOT do better. The hypothesis")
        print("  in this script's docstring is refuted and the disagreement")
        print("  with ADAS is a disagreement about the physics, not about the")
        print("  definition. It must be reported as such.")
    print()

    if a.write:
        out = Path(a.out) if a.out else ROOT / "validation" / "gate_D_diagnosis"
        out.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with (out / "gate_D_diagnosis.csv").open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["i", "j", "Te", "ne", "eta_SCD_CRE", "eta_SCD_groundfed",
                        "SCD_groundfed", "eta_ACD", "ACD_model",
                        "recomb_fed_share_of_ionisation"])
            for i in range(nT):
                for j in range(nN):
                    w.writerow([i, j, f"{te[i]:.6g}", f"{ne[j]:.6g}",
                                f"{eta_cre[i,j]:.6e}", f"{eta_gf[i,j]:.6e}",
                                f"{scd_gf_arr[i,j]:.6e}", f"{eta_acd[i,j]:.6e}",
                                f"{acd_arr[i,j]:.6e}", f"{f_rec[i,j]:.6e}"])
        print(f"  wrote {out/'gate_D_diagnosis.csv'}  ({nT*nN} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
