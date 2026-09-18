#!/usr/bin/env python
"""
verify_bundling_total_loss.py
=============================
Proton l-mixing out of np in the bundled shells n = 9..15 against the TOTAL loss rate -L[k,k], decomposed by channel.

WHY THIS EXISTS
---------------
Section 6.8.3 (sec:bundling_check) and outputs/bundling/bundling_report.txt
test whether the bundled shells n = 9..15 are statistically populated in l by
comparing the proton l-mixing rate out of np with the RADIATIVE rate
A(np_total) only, and find a margin >= 1538 at all 2800 (shell, grid)
combinations. At these densities the relevant escape from a high shell is
collisional, not radiative: the diagonal -L[k,k] of a bundled state at
1e15 cm^-3 is five orders of magnitude above gamma_bundled. This script
compares the same l-mixing rate with the total loss rate out of the bundled
state, -L[k,k], and decomposes that loss into its three channels so the text
can say which loss the mixing has to beat, and by how much.

METHOD
------
  K_lmix(n; Te, ne)  exactly as verify_bundling_psm20.lmix_rate_np builds it:
                     q_down(n, l_> = 1) + q_up(n, l = 1) from compute_lmix.py
                     (Badnell 2021 PSM20 Debye formula, Debye cutoff FROZEN at
                     compute_lmix.NE_DEFAULT = 1e14 cm^-3), times the local
                     n_p = ne. Recomputed here from the module's own functions
                     (read-only import) and required to reproduce
                     outputs/bundling/K_lmix_per_shell.npy.
  A(np_total)        imported from verify_bundling_psm20.A_np_total (the
                     hardcoded NIST A(np->1s) x 1.3), so that G1 reproduces
                     bundling_ratio.npy with the same numbers, not a re-typing.
  -L[k,k]            total loss out of bundled state k, from L_grid.npy.
  L = R + ne C       at each Te, R and C are built from the two lowest
                     densities (C = (L[1]-L[0])/(ne1-ne0), R = L[0] - ne0 C) and
                     the identity is then required to hold at all eight
                     densities to 1e-10 relative. L is linear in ne by
                     construction (assemble_cr_matrix.py: radiative + ne x
                     collisional; three-body recombination lives in S).
  decomposition      radiative out    = sum_{i!=k} R[i,k]      (should = gamma_bundled)
                     collisional b-b  = ne sum_{i!=k} C[i,k]
                     ionisation       = -sum_i L[i,k]          (should = K_ion ne)
                     each as a fraction of -L[k,k]; the three must sum to 1.
  Ratio_frozen       K_lmix_frozen / (-L[k,k])
  Ratio_local        K_lmix with the Debye cutoff at the LOCAL density, through
                     the same two-channel call path with ne_cm3 = ne_local,
                     divided by -L[k,k]. Also reported, Ratio_local_F1: the
                     rescale K_frozen x F(U_m(n,1,Te,ne)) / F(U_m(n,1,Te,1e14))
                     using only the l_> = 1 channel's Coulomb factor
                     (compute_lmix._psm20_Um, _psm20_F, read-only).
  F variation        F(U_m(n, l_> = 1, Te, ne)) / F(U_m(n, l_> = 1, Te, 1e14))
                     for n = 9..15 at Te = 1 and 10 eV across the eight
                     densities, to settle compute_lmix.py's comment "F(U_m)
                     varies ~30% across the full grid" against
                     bundling_report.txt's x20 / /30 at n = 15. The report's
                     own number (two-channel rate, Te index n_Te//2) is
                     reproduced alongside, and the same table is given for
                     n = 2, 5, 8 to show where the 30 % statement does hold.
  adjacent fraction  of the collisional bound-bound loss out of shell n, the
                     part going to n-1 and n+1 versus farther, at [0,7] and
                     [23,5]. Every bound-bound entry is linear in ne, so this
                     split is a function of Te alone; the density index is
                     carried only to label the requested points.

GATES (all must pass before any ratio is reported)
-------------------------------------------------
  G1  own K_lmix reproduces K_lmix_per_shell.npy to 1e-12, and own
      K_lmix / A(np_total) reproduces bundling_ratio.npy to 1e-12, with its
      minimum 1537.58 at n = 9, Te index 0, ne index 0
  G2  L = R + ne C exact at all eight densities, every Te, to 1e-10 relative
  G3  frac_rad + frac_coll + frac_ion = 1 at every (shell, Te, ne) to 1e-10
  G4  R[k,k] = -gamma_bundled[k] and the off-diagonal radiative column sum
      equals gamma_bundled[k], to 1e-10 relative (names the radiative piece)
  G5  the column-sum deficit equals K_ion[k] ne to 1e-10 relative (names the
      ionisation piece; it is the only true loss from the bound system)

PREDICTIONS (written before the run)
-----------------------------------
P1  per-shell minimum of Ratio_frozen over the grid about 26.5, 21.6, 12.0,
    6.9, 4.2, 2.85, 3.85 for n = 9..15, all at grid point [0,0] (1 eV, 1e12)
P2  at [23,5] Ratio_frozen about 125, 120, 84, 57, 39, 28, 41
P3  escape composition 90-98 % collisional n-changing, 2-10 % ionisation,
    <= 0.2 % radiative, over the 2800 (shell, grid) points
P4  Ratio_local falls to about 0.09-0.86 at [0,7] (1 eV, 1e15)
P5  the rate for n = 15 varies by roughly x20 at 1e12 and /30 at 1e15
    relative to the frozen 1e14 cutoff, not 30 %
REFUTER for the "statistical" assumption as currently argued: Ratio_frozen < 1
anywhere. Ratio_local < 1 anywhere is reported with its location; it is a
finding about the frozen cutoff, not a failure of this script.

OUTPUTS (with --write): validation/bundling_total_loss/bundling_total_loss.{csv,txt}
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

_HERE = Path(__file__).resolve(); sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root  # noqa: E402
ROOT = find_repo_root(_HERE)
REPORT_MIN = 1537.58          # bundling_report.txt, worst offender n = 9, Te = 1 eV, ne = 1e12
P1 = [26.5, 21.6, 12.0, 6.9, 4.2, 2.85, 3.85]; P2 = [125, 120, 84, 57, 39, 28, 41]


def _load(name: str, path: Path):
    """Import a pipeline module by path, read-only (nothing in it is edited or re-run)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = CRContext.load(); ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid; nT, nN, nS, _ = L.shape
    P = lambda *p: ROOT.joinpath(*p)
    paths = dict(L_grid=P("data/processed/cr_matrix/L_grid.npy"), Te_grid_L=P("data/processed/cr_matrix/Te_grid_L.npy"),
                 ne_grid_L=P("data/processed/cr_matrix/ne_grid_L.npy"), state_index=ctx.state_index_path,
                 gamma_bundled=P("data/processed/Radiative/gamma_bundled.npy"), K_ion_final=P("data/processed/collisions/tics/K_ion_final.npy"),
                 K_lmix_per_shell=P("outputs/bundling/K_lmix_per_shell.npy"), bundling_ratio=P("outputs/bundling/bundling_ratio.npy"),
                 compute_lmix=P("src/rates/compute_lmix.py"), verify_bundling_psm20=_HERE.parent / "verify_bundling_psm20.py")
    for q in paths.values():
        if not q.is_file(): raise FileNotFoundError(f"required input missing: {q}")
    lm = _load("compute_lmix", paths["compute_lmix"]); vb = _load("verify_bundling_psm20", paths["verify_bundling_psm20"])
    gam = np.load(paths["gamma_bundled"]); K_ion = np.load(paths["K_ion_final"])
    K_shell_ref = np.load(paths["K_lmix_per_shell"]); ratio_ref = np.load(paths["bundling_ratio"])
    si = pd.read_csv(paths["state_index"])
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 78); say("BUNDLED-SHELL l-MIXING AGAINST TOTAL LOSS -- -L[k,k] decomposed, n = 9..15")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}"); say(f"interpreter {sys.executable}"); say(ctx.describe()); say("=" * 78)

    # ---- which states are bundled: from state_index.csv, cross-checked against the hardcoded layout in verify_bundling_psm20.py
    bmap = si.bundled.map({True: True, False: False, "True": True, "False": False, 1: True, 0: False})
    if bmap.isna().any(): raise RuntimeError(f"state_index.csv 'bundled' column has values this script cannot read: {si.bundled.unique()}")
    bund = si[bmap.values.astype(bool)].sort_values("idx"); bidx = bund.idx.values.astype(int); nb = bund.n.values.astype(int); NB = len(bidx)
    if not (np.array_equal(bidx, np.arange(bidx[0], bidx[0] + NB)) and np.array_equal(nb, np.arange(nb[0], nb[0] + NB))):
        raise RuntimeError(f"bundled states are not one contiguous run of consecutive n: idx {bidx}, n {nb}")
    if not (np.array_equal(bidx, vb.BUND_IDX) and np.array_equal(nb, vb.N_BUND)):
        raise RuntimeError(f"verify_bundling_psm20.py hardcodes BUND_IDX {vb.BUND_IDX} / N_BUND {vb.N_BUND}; state_index.csv says {bidx} / {nb}")
    if not np.array_equal(ctx.n_values[bidx].astype(int), nb): raise RuntimeError("cr_context n_values disagree with state_index.csv on the bundled rows")
    if gam.shape != (NB,): raise RuntimeError(f"gamma_bundled.npy shape {gam.shape}, expected ({NB},)")
    if K_ion.shape != (nS, nT): raise RuntimeError(f"K_ion_final.npy shape {K_ion.shape}, expected ({nS}, {nT})")
    if K_shell_ref.shape != (NB, nT, nN) or ratio_ref.shape != (NB, nT, nN):
        raise RuntimeError(f"outputs/bundling arrays have shapes {K_shell_ref.shape}, {ratio_ref.shape}; expected ({NB}, {nT}, {nN})")
    say(f"\nbundled states: idx {bidx[0]}..{bidx[-1]} = {', '.join(ctx.labels[k] for k in bidx)}  (n = {nb[0]}..{nb[-1]}), from {paths['state_index'].relative_to(ROOT)}")

    # ---- the l-mixing rate out of np, the call path of verify_bundling_psm20.lmix_rate_np, evaluated here from compute_lmix directly
    def q_np(n: int, ne_debye: float) -> np.ndarray:
        """Rate coefficient [cm^3/s] out of (n, l=1) over all Te: np->ns (l_> = 1) plus np->nd (l_> = 2). n >= 3 throughout."""
        q = lm._psm20_q_down(n, 1, te, ne_cm3=ne_debye)
        if n >= 3: q = q + lm._psm20_q_up(n, 1, te, ne_cm3=ne_debye)
        return q
    q_frozen = np.array([q_np(int(n), lm.NE_DEFAULT) for n in nb])                                   # (NB, nT)  cm^3/s
    K_frozen = q_frozen[:, :, None] * ne[None, None, :]                                               # (NB, nT, nN)  s^-1
    K_local = np.array([[q_np(int(n), ne[j]) * ne[j] for j in range(nN)] for n in nb]).transpose(0, 2, 1)
    F_fro = np.array([lm._psm20_F(lm._psm20_Um(int(n), 1, te, lm.NE_DEFAULT)) for n in nb])          # (NB, nT)
    F_loc = np.array([[lm._psm20_F(lm._psm20_Um(int(n), 1, te, ne[j])) for j in range(nN)] for n in nb]).transpose(0, 2, 1)
    F1 = F_loc / F_fro[:, :, None]; K_local_F1 = K_frozen * F1
    A_np = np.array([vb.A_np_total(int(n)) for n in nb]); ratio_Anp = K_frozen / A_np[:, None, None]

    # ---- G1
    d1 = np.abs(K_frozen / K_shell_ref - 1).max(); d1b = np.abs(ratio_Anp / ratio_ref - 1).max()
    imin = tuple(int(v) for v in np.unravel_index(ratio_Anp.argmin(), ratio_Anp.shape))
    say(f"\n  G1  own K_lmix (compute_lmix._psm20_q_down + _psm20_q_up, l_> = 1 and 2, Debye cutoff at NE_DEFAULT = {lm.NE_DEFAULT:.3g} cm^-3, x ne)")
    say(f"      vs K_lmix_per_shell.npy: max rel diff {d1:.3e}")
    say(f"      own K_lmix / A(np_total)  [A_np_total IMPORTED from verify_bundling_psm20.py: {vb.A_NP_TOTAL_FACTOR} x A(np->1s) table]")
    say(f"      vs bundling_ratio.npy: max rel diff {d1b:.3e};  minimum {ratio_Anp.min():.2f} at n = {nb[imin[0]]}, Te index {imin[1]}, ne index {imin[2]}  (report: {REPORT_MIN} at n = 9, [0,0])")
    if d1 > 1e-12 or d1b > 1e-12 or imin != (0, 0, 0) or abs(ratio_Anp.min() / REPORT_MIN - 1) > 1e-5: raise RuntimeError("Gate G1")

    # ---- G2: L = R + ne C from the two lowest densities, tested at all eight
    R = np.empty((nT, nS, nS)); C = np.empty_like(R)
    for i in range(nT):
        C[i] = (L[i, 1] - L[i, 0]) / (ne[1] - ne[0]); R[i] = L[i, 0] - ne[0] * C[i]
    recon = R[:, None] + ne[None, :, None, None] * C[:, None]
    d2 = (np.abs(recon - L).max(axis=(2, 3)) / np.abs(L).max(axis=(2, 3))).max()
    say(f"  G2  L = R + ne C (R, C from ne indices 0,1; tested at all {nN} densities, {nT} Te): max rel residual {d2:.3e}")
    if d2 > 1e-10: raise RuntimeError("Gate G2")

    # ---- decomposition of -L[k,k] for the bundled columns
    off = ~np.eye(nS, dtype=bool)
    tot = np.stack([-L[:, :, k, k] for k in bidx])                                    # (NB, nT, nN)
    rad = np.stack([R[:, off[:, k], k].sum(axis=1) for k in bidx])                    # (NB, nT)  radiative gains to other states
    coll_coef = np.stack([C[:, off[:, k], k].sum(axis=1) for k in bidx])              # (NB, nT)  cm^3/s, bound-bound collisional out
    coll = coll_coef[:, :, None] * ne[None, None, :]
    ion = np.stack([-L[:, :, :, k].sum(axis=2) for k in bidx])                        # (NB, nT, nN) column-sum deficit
    Rdiag = np.stack([R[:, k, k] for k in bidx])                                      # (NB, nT)
    frac_rad = rad[:, :, None] / tot; frac_coll = coll / tot; frac_ion = ion / tot
    d3 = np.abs(frac_rad + frac_coll + frac_ion - 1).max()
    say(f"  G3  frac_rad + frac_coll + frac_ion = 1 at every (shell, Te, ne): max |sum - 1| {d3:.3e}")
    if d3 > 1e-10: raise RuntimeError("Gate G3")
    d4a = np.abs(Rdiag / (-gam[:, None]) - 1).max(); d4b = np.abs(rad / gam[:, None] - 1).max()
    say(f"  G4  R[k,k] = -gamma_bundled: max rel diff {d4a:.3e};  off-diagonal radiative column sum = gamma_bundled: max rel diff {d4b:.3e}")
    if d4a > 1e-10 or d4b > 1e-10: raise RuntimeError("Gate G4")
    ion_exp = np.stack([K_ion[k][:, None] * ne[None, :] for k in bidx])
    d5 = np.abs(ion / ion_exp - 1).max()
    say(f"  G5  column-sum deficit = K_ion[k] ne (ionisation identity): max rel diff {d5:.3e}")
    if d5 > 1e-10: raise RuntimeError("Gate G5")
    # R and C are reconstructed by subtraction, so an entry whose true value is zero (no radiative A upward, for instance) comes back as
    # a rounding residual of order 1e-16 x the column's largest entry, of either sign. A real negative gain would be of order the entries.
    negC = min(C[:, off[:, k], k].min() for k in bidx); negR = min(R[:, off[:, k], k].min() for k in bidx)
    scaleC = max(np.abs(C[:, :, k]).max() for k in bidx); scaleR = max(np.abs(L[:, 0, :, k]).max() for k in bidx)
    say(f"      (smallest off-diagonal entry in the bundled columns: C {negC:.3e} cm^3/s = {negC/scaleC:.1e} of the largest, R {negR:.3e} s^-1 = {negR/scaleR:.1e} of the largest L[.,0] entry;")
    say(f"       a negative beyond -1e-12 of that scale would mean a piece below is not a rate)")
    if negC < -1e-12 * scaleC or negR < -1e-12 * scaleR: raise RuntimeError("negative gain entry in a bundled column of R or C beyond rounding")
    say("\n  ALL GATES PASSED.")

    # ---- results
    ratio_frozen = K_frozen / tot; ratio_local = K_local / tot; ratio_local_F1 = K_local_F1 / tot; ratio_gam = K_frozen / rad[:, :, None]
    i5, j5 = 23, 5; i0, j7 = 0, 7
    say("\n" + "=" * 78); say("RESULT 1: Ratio_frozen = K_lmix(np, frozen Debye cutoff) / (-L[k,k]), per bundled shell over the 400 grid points"); say("=" * 78)
    say(f"  {'n':>3} {'min':>8} {'at':>8} {'Te eV':>6} {'ne':>9} {'K_lmix':>10} {'-L[k,k]':>10} {'rad%':>7} {'coll%':>7} {'ion%':>7} | {'[23,5]':>8} {'[0,7]':>8} | {'min vs A(np)':>12} {'min vs gamma_n':>14}")
    mins = []
    for b, n in enumerate(nb):
        m = tuple(int(v) for v in np.unravel_index(ratio_frozen[b].argmin(), ratio_frozen[b].shape)); i, j = m; mins.append((ratio_frozen[b][m], m))
        say(f"  {n:>3} {ratio_frozen[b][m]:>8.2f} {str(m):>8} {te[i]:>6.2f} {ne[j]:>9.2e} {K_frozen[b, i, j]:>10.3e} {tot[b, i, j]:>10.3e} "
            f"{100*frac_rad[b, i, j]:>7.3f} {100*frac_coll[b, i, j]:>7.2f} {100*frac_ion[b, i, j]:>7.2f} | {ratio_frozen[b, i5, j5]:>8.1f} {ratio_frozen[b, i0, j7]:>8.2f} | "
            f"{ratio_Anp[b].min():>12.1f} {ratio_gam[b].min():>14.1f}")
    n_ref = int((ratio_frozen < 1).sum())
    say(f"  grid minimum {ratio_frozen.min():.3f} at n = {nb[np.unravel_index(ratio_frozen.argmin(), ratio_frozen.shape)[0]]}, {tuple(int(v) for v in np.unravel_index(ratio_frozen.argmin(), ratio_frozen.shape)[1:])};"
        f"  points with Ratio_frozen < 1: {n_ref} of {ratio_frozen.size};  points with Ratio_frozen < 10: {int((ratio_frozen < 10).sum())}")
    say(f"  the original test's quantity, K_lmix / A(np_total), has grid minimum {ratio_Anp.min():.1f}; against the shell-average radiative loss gamma_n it is {ratio_gam.min():.1f}")

    say("\n" + "=" * 78); say("RESULT 2: what -L[k,k] is made of (fractions of the total loss, over the 2800 points)"); say("=" * 78)
    for lab, f in (("radiative", frac_rad), ("collisional bound-bound", frac_coll), ("ionisation", frac_ion)):
        lo = tuple(int(v) for v in np.unravel_index(f.argmin(), f.shape)); hi = tuple(int(v) for v in np.unravel_index(f.argmax(), f.shape))
        say(f"  {lab:24s} {100*f.min():8.4f} % at n={nb[lo[0]]} {lo[1:]}   to {100*f.max():8.4f} % at n={nb[hi[0]]} {hi[1:]}   median {100*np.median(f):7.3f} %")
    for (i, j, lab) in ((0, 0, "[0,0] 1 eV, 1e12"), (i5, j5, "[23,5] benchmark"), (i0, j7, "[0,7] 1 eV, 1e15"), (nT - 1, j7, f"[{nT-1},7] 10 eV, 1e15")):
        say(f"  at {lab:>18}: " + "  ".join(f"n{n}: r{100*frac_rad[b, i, j]:.3f}/c{100*frac_coll[b, i, j]:.1f}/i{100*frac_ion[b, i, j]:.1f}%" for b, n in enumerate(nb)))

    say("\n" + "=" * 78); say("RESULT 3: Ratio_local = K_lmix with the Debye cutoff at the LOCAL density / (-L[k,k])"); say("=" * 78)
    say(f"  {'n':>3} {'min':>8} {'at':>8} {'Te eV':>6} {'ne':>9} | {'[0,7]':>8} {'[0,7] F1':>9} {'[23,5]':>8} {'[0,0]':>9} | {'# < 1':>6} {'# < 10':>6}")
    for b, n in enumerate(nb):
        m = tuple(int(v) for v in np.unravel_index(ratio_local[b].argmin(), ratio_local[b].shape)); i, j = m
        say(f"  {n:>3} {ratio_local[b][m]:>8.3f} {str(m):>8} {te[i]:>6.2f} {ne[j]:>9.2e} | {ratio_local[b, i0, j7]:>8.3f} {ratio_local_F1[b, i0, j7]:>9.3f} {ratio_local[b, i5, j5]:>8.1f} {ratio_local[b, 0, 0]:>9.1f} | "
            f"{int((ratio_local[b] < 1).sum()):>6} {int((ratio_local[b] < 10).sum()):>6}")
    below = np.argwhere(ratio_local < 1)
    if len(below):
        js = sorted(set(int(j) for j in below[:, 2])); its = sorted(set(int(i) for i in below[:, 1]))
        say(f"  Ratio_local < 1 at {len(below)} of {ratio_local.size} points: shells n = {sorted(set(int(nb[b]) for b in below[:, 0]))}, ne indices {js} ({', '.join(f'{ne[j]:.1e}' for j in js)}), Te indices {its[0]}..{its[-1]} ({te[its[0]]:.2f}..{te[its[-1]]:.2f} eV)")
    else: say("  Ratio_local < 1 nowhere on the grid")
    dF = np.abs(ratio_local_F1 / ratio_local - 1).max()
    say(f"  exact two-channel local rate vs the l_> = 1 F-rescale: max rel diff {dF:.3f} (they differ because the np->nd channel has its own U_m)")

    say("\n" + "=" * 78); say("RESULT 4: how much F(U_m) actually varies with density (settles the '~30%' comment in compute_lmix.py)"); say("=" * 78)
    imid = nT // 2
    loc_over_fro = K_local / K_frozen
    say(f"  bundling_report.txt's number, two-channel rate local/frozen at Te index {imid} ({te[imid]:.2f} eV), n = 15: "
        f"x{loc_over_fro[-1, imid, 0]:.2f} at {ne[0]:.0e},  /{1/loc_over_fro[-1, imid, -1]:.2f} at {ne[-1]:.0e}   (report says x20 and /30)")
    say(f"  over all shells n = 9..15 and the whole grid, two-channel local/frozen ranges {loc_over_fro.min():.4f} to {loc_over_fro.max():.2f}")
    hdr = f"  {'n':>3} {'Te eV':>6} {'U_m(1e14)':>10} | " + " ".join(f"{ne[j]:>8.1e}" for j in range(nN))
    for Tlab, T in (("1 eV", 1.0), ("10 eV", 10.0)):
        i = int(np.argmin(np.abs(te - T)))
        say(f"\n  F(U_m(n, l_> = 1, Te = {te[i]:.3g} eV, ne)) / F(U_m(n, 1, Te, 1e14)), bundled shells:"); say(hdr)
        for b, n in enumerate(nb):
            say(f"  {n:>3} {te[i]:>6.2f} {lm._psm20_Um(int(n), 1, te[i:i+1], lm.NE_DEFAULT)[0]:>10.3e} | " + " ".join(f"{F1[b, i, j]:>8.3f}" for j in range(nN)))
    say(f"\n  the same ratio for resolved shells, l_> = 1, where compute_lmix.py's K_lmix is actually used:"); say(hdr)
    for n in (2, 5, 8):
        for T in (1.0, 10.0):
            i = int(np.argmin(np.abs(te - T)))
            Ff = lm._psm20_F(lm._psm20_Um(n, 1, te[i:i+1], lm.NE_DEFAULT))[0]
            say(f"  {n:>3} {te[i]:>6.2f} {lm._psm20_Um(n, 1, te[i:i+1], lm.NE_DEFAULT)[0]:>10.3e} | " + " ".join(f"{lm._psm20_F(lm._psm20_Um(n, 1, te[i:i+1], ne[j]))[0]/Ff:>8.3f}" for j in range(nN)))
    say("  Reading: U_m = E_min/kT grows as n^4 (through D_ji) and as ne (through 1/lambda_D^2). For n = 2..8 it stays well below 1 and F is\n"
        "  logarithmic in ne, which is the regime the '~30%' comment describes. For n >= 9 at ne >= 1e14, U_m exceeds 1 and F falls as\n"
        "  U_m^-3/2, i.e. as ne^-3/2: the Debye screening is cutting the collision off, and the frozen cutoff is then not a mild approximation.")

    say("\n" + "=" * 78); say("RESULT 5: where the collisional bound-bound loss out of shell n goes (fractions of the bound-bound collisional loss)"); say("=" * 78)
    nv = ctx.n_values.astype(int); adj = {}
    for b, (k, n) in enumerate(zip(bidx, nb)):
        colC = C[:, :, k].copy(); colC[:, k] = 0.0; totC = colC.sum(axis=1)
        f = lambda sel: colC[:, sel].sum(axis=1) / totC
        adj[b] = dict(m1=f(nv == n - 1), p1=f(nv == n + 1) if (nv == n + 1).any() else np.full(nT, np.nan), lo=f(nv < n - 1), hi=f(nv > n + 1))
    say("  (independent of ne because every bound-bound entry is ne x C; the density index only labels the requested point)")
    for (i, j, lab) in ((i0, j7, "[0,7]"), (i5, j5, "[23,5]")):
        say(f"\n  at {lab} Te = {te[i]:.3f} eV, ne = {ne[j]:.3e}:")
        say(f"  {'n':>3} {'to n-1':>8} {'to n+1':>8} {'adjacent':>9} {'to <n-1':>8} {'to >n+1':>8}  {'coll b-b out s^-1':>18}")
        for b, n in enumerate(nb):
            d = adj[b]; p1 = d['p1'][i]
            say(f"  {n:>3} {100*d['m1'][i]:>7.2f}% {100*p1:>7.2f}% {100*(d['m1'][i] + (0 if np.isnan(p1) else p1)):>8.2f}% {100*d['lo'][i]:>7.2f}% {100*d['hi'][i]:>7.2f}%  {coll[b, i, j]:>18.3e}")
        say(f"  (n = {nb[-1]} has no n+1: the model is truncated there, so its 'to n+1' is undefined and its adjacent share is the n-1 share alone)")

    # ---- predictions and refuter
    say("\n" + "=" * 78); say("PREDICTIONS (written before the run) against what came out"); say("=" * 78)
    missed = []
    p1ok = all(abs(mins[b][0] / P1[b] - 1) < 0.10 and mins[b][1] == (0, 0) for b in range(NB))
    say(f"  P1 per-shell minima {', '.join(f'{mins[b][0]:.2f}' for b in range(NB))} at {[mins[b][1] for b in range(NB)]}  (predicted {P1} all at (0, 0)): {'reproduced' if p1ok else 'NOT reproduced'}")
    if not p1ok: missed.append("P1")
    v5 = ratio_frozen[:, i5, j5]; p2ok = all(abs(v5[b] / P2[b] - 1) < 0.10 for b in range(NB))
    say(f"  P2 at [23,5]: {', '.join(f'{v:.1f}' for v in v5)}  (predicted {P2}): {'reproduced' if p2ok else 'NOT reproduced'}")
    if not p2ok: missed.append("P2")
    p3ok = (frac_coll.min() >= 0.88) and (frac_coll.max() <= 0.99) and (frac_ion.min() >= 0.01) and (frac_ion.max() <= 0.12) and (frac_rad.max() <= 0.0025)
    say(f"  P3 composition: coll {100*frac_coll.min():.1f}-{100*frac_coll.max():.1f} %, ion {100*frac_ion.min():.1f}-{100*frac_ion.max():.1f} %, rad <= {100*frac_rad.max():.3f} %"
        f"  (predicted 90-98 / 2-10 / <= 0.2 %; judged with 2 points slack): {'reproduced' if p3ok else 'NOT reproduced'}")
    if not p3ok: missed.append("P3")
    v7 = ratio_local[:, i0, j7]; p4ok = (abs(v7.min() / 0.09 - 1) < 0.25) and (abs(v7.max() / 0.86 - 1) < 0.25)
    say(f"  P4 Ratio_local at [0,7]: {v7.min():.3f} to {v7.max():.3f}  (predicted about 0.09-0.86): {'reproduced' if p4ok else 'NOT reproduced'}")
    if not p4ok: missed.append("P4")
    up, dn = loc_over_fro[-1, imid, 0], 1 / loc_over_fro[-1, imid, -1]; p5ok = abs(up / 20 - 1) < 0.25 and abs(dn / 30 - 1) < 0.25
    say(f"  P5 n = 15 local/frozen at Te index {imid}: x{up:.1f} at 1e12, /{dn:.1f} at 1e15  (predicted about x20, /30; '~30%' would be 0.7-1.3): {'reproduced' if p5ok else 'NOT reproduced'}")
    if not p5ok: missed.append("P5")
    say("\nPREDICTIONS: " + ("P1-P5 all reproduced" if not missed else "not reproduced as written: " + ", ".join(missed)))
    say("REFUTER (Ratio_frozen < 1 anywhere): " + (f"APPEARED at {n_ref} points" if n_ref else "did not appear") +
        f";  grid minimum {ratio_frozen.min():.2f}, so the mixing beats the TOTAL loss by at least that factor with the model's own frozen cutoff.")
    say("FINDING (Ratio_local < 1): " + (f"appears at {len(below)} points (see RESULT 3); with the cutoff at the local density the mixing does not beat the total\n"
        f"  collisional loss at the top of the density range for shells n >= {int(nb[below[:, 0].min()])}. It is not the quantity the frozen-cutoff model uses." if len(below) else "does not appear"))
    say("\nWhat this does and does not say. The original test compared the mixing with the l-DIFFERENTIAL loss (radiative decay of np, the channel\n"
        "that would de-populate one l preferentially); that margin is >= 1538. Against the TOTAL loss the margin shrinks to a few because the\n"
        "dominant loss at these densities is collisional n-changing, which RESULT 5 shows goes overwhelmingly to the adjacent shells. Whether\n"
        "an n-changing collision at high n disturbs the l-distribution is a physics question this script does not answer; it supplies both numbers.")

    if a.write:
        out = Path(a.out) if a.out else P("validation/bundling_total_loss"); out.mkdir(parents=True, exist_ok=True)
        B, I, J = np.meshgrid(np.arange(NB), np.arange(nT), np.arange(nN), indexing="ij"); Bf, If, Jf = B.ravel(), I.ravel(), J.ravel()
        bc = lambda x: np.broadcast_to(x[:, :, None], tot.shape).ravel()
        adj_m1 = np.stack([adj[b]['m1'] for b in range(NB)]); adj_p1 = np.stack([adj[b]['p1'] for b in range(NB)])
        df = pd.DataFrame(dict(n=nb[Bf], k=bidx[Bf], i=If, j=Jf, Te_eV=te[If], ne_cm3=ne[Jf],
                               K_lmix_frozen=K_frozen.ravel(), K_lmix_local=K_local.ravel(), F1_local_over_frozen=F1.ravel(), A_np_total=A_np[Bf],
                               gamma_bundled=gam[Bf], total_loss=tot.ravel(), rad_out=bc(rad), coll_bb_out=coll.ravel(), ion_out=ion.ravel(),
                               frac_rad=frac_rad.ravel(), frac_coll=frac_coll.ravel(), frac_ion=frac_ion.ravel(),
                               ratio_Anp=ratio_Anp.ravel(), ratio_gamma=ratio_gam.ravel(), ratio_frozen=ratio_frozen.ravel(),
                               ratio_local=ratio_local.ravel(), ratio_local_F1=ratio_local_F1.ravel(),
                               frac_coll_to_nminus1=bc(adj_m1), frac_coll_to_nplus1=bc(adj_p1), frac_coll_adjacent=bc(adj_m1 + np.nan_to_num(adj_p1))))
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}", f"# interpreter {sys.executable}"]
        hdr += [f"# {q.relative_to(ROOT) if q.is_relative_to(ROOT) else q} sha256 {sha(q)}" for q in paths.values()]
        hdr += [f"# Debye cutoff frozen at compute_lmix.NE_DEFAULT = {lm.NE_DEFAULT:.4g} cm^-3; A_np_total = {vb.A_NP_TOTAL_FACTOR} x A(np->1s); channels l_> = 1 and 2 out of np",
                f"# ratio_frozen = K_lmix_frozen / total_loss; ratio_local uses the two-channel rate with the cutoff at ne_cm3; ratio_local_F1 rescales by F(U_m, l_>=1) only"]
        with open(out / "bundling_total_loss.csv", "w") as fh: fh.write("\n".join(hdr) + "\n"); df.to_csv(fh, index=False)
        with open(out / "bundling_total_loss.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}/bundling_total_loss.{{csv,txt}}")
    return 0


if __name__ == "__main__": sys.exit(main())
