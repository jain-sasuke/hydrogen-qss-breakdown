#!/usr/bin/env python
"""
verify_l_blindness.py
=====================
Is the l-distribution of a high shell statistical where proton l-mixing no longer
wins? The argument chapter 6 (sec:bundling_check) leaves untested, put to the
model's own l-resolved data.

WHY THIS EXISTS
---------------
verify_bundling_total_loss.py found that with the Debye cutoff evaluated at the
local density the l-mixing rate out of np falls BELOW the total loss out of the
bundled shell at 121 of 2800 (shell, grid) combinations: every bundled shell
n = 9..15, the two highest densities (3.7e14 and 1e15 cm^-3), Te <= 2.95 eV.
Chapter 6 says that there the l-distribution "is set by the feeds, and
n-changing collisions between adjacent shells, which carry 71-95 % of the
collisional loss, are close to l-blind at high n, so the statistical assumption
is plausible there for a different reason, one this thesis has not tested."

This script tests it with what the repository already holds:

  (i)   CCC electron-impact excitation rates resolved in BOTH (n_i, l_i) and
        (n_f, l_f) up to n_f = 10, including every (9, l) -> (10, l') pair
        (data/processed/collisions/ccc/K_CCC_metadata.csv rows with n_f = 9, 10;
        the model itself bundles n >= 9 and only ever uses these rows summed
        over l_f, see assemble_K_exc.py Block 2), and their detailed-balance
        de-excitation partners;
  (ii)  the model's assembled operator L, from which the l-mixing block can be
        removed exactly (it is the Delta-n = 0 off-diagonal block, equal to
        K_lmix.npy x ne) or replaced by the local-cutoff rate through
        compute_lmix.compute_K_lmix(ne_cm3=ne_local), read-only.

Two meanings of "l-blind" are measured separately, because the argument needs both:
  LOSS blindness    the total adjacent-shell transfer OUT of (n, l) does not depend on l;
  FEED statisticality  the transfer INTO (n, l') from a statistically populated
                       neighbour shell lands in proportion to 2l'+1.
If both hold, the steady-state l-distribution set by the feeds alone is statistical.
If feeds are statistical but losses are not, p_l tracks 1/loss_l and the departure
from statistical equals the loss spread.

METHOD
------
  Departure coefficient  b_l = (p_l / g_l) / (sum_l p_l / sum_l g_l), g_l = 2(2l+1);
                         statistical <=> b_l = 1 for every l. Reported as
                         max_l |b_l - 1| and the population-weighted mean |b_l - 1|.
  R1  loss spread: for n = 1..9, K_up(n, l) = sum_l' K_CCC(n l -> n+1 l'); for
      n = 2..10, K_down(n, l) = sum_l' K_CCC_deexc(n l -> n-1 l'); spread = max_l / min_l
      at every Te; the l that carries the max and the min.
  R2  feed statisticality: from a statistically populated shell n_i (weights
      (2l_i+1)/n_i^2), gain(l_f) = sum_l_i w_i K(n_i l_i -> n_f l_f), n_f = n_i +- 1;
      b(l_f) = [gain / sum gain] / [(2l_f+1)/n_f^2].
  R3  the first bundled shell, n = 9, from the two neighbours the data resolves:
      p_l(9) proportional to [N_8 sum_l' w_8(l') K(8l'->9l) + N_10 sum_l'' w_10(l'') K_deexc(10l''->9l)]
                          / [sum_l' K_deexc(9l->8l') + sum_l'' K(9l->10l'')]
      with N_8, N_10 the model's own CRE shell populations at the grid point
      (p = -L^-1 S). Ionisation and Delta-n >= 2 transfer out of n = 9 are
      l-summed in the model and are OMITTED from the headline denominator; a
      second column adds them as one l-blind constant taken from
      validation/bundling_total_loss/ for n = 9 at the point WITHOUT the l-blind
      feeds that balance them at LTE (an inconsistent extreme, a bound only).
      Radiative decay out of n = 9 is l-DEPENDENT (A(9p_total) is 8.9 x gamma_9)
      and uncompensated by any feed; its effect on b(9p) is estimated separately
      (R3c). Feeds from n >= 11 and from recombination are omitted.
      With detailed balance exact (G1) the headline is an identity in one number:
      p_l/omega_l = [alpha K_down(l) + beta K_up(l)]/[K_down(l) + K_up(l)], so
      b_l - 1 = (theta - 1)[f_dn(l) - mean f_dn], f_dn = K_down/(K_down + K_up),
      with theta = alpha/beta the Boltzmann departure of the neighbour pair. The
      physical content is theta: R3 measures it from the model's populations,
      R3b for every bundled pair, and R5 re-measures it under the bundled
      radiative rescalings, so that theta is shown not to be an artefact of the
      bundled block's own assumptions. The Saha departure of every high shell
      against the model's own ionisation/three-body pair is printed beside it.
  R4  the highest resolved shells inside the model itself: solve p = -L'^-1 S with
      (a) L as built (frozen cutoff), (b) L with the l-mixing block removed,
      (c) L with the block rebuilt at the local density; b_l for n = 2..8 at every
      grid point. (b) is the feeds-and-losses-only distribution the argument is
      about, computed with every channel the model has (CCC into and out of
      n = 8, de-excitation from n = 9 and 10 by detailed balance, ionisation,
      three-body and radiative recombination, radiative cascade). Also: the
      gain into shell 8 decomposed by source, so the reader sees how much of the
      answer is the model's own statistical-9 assumption coming back through
      detailed balance; and the mixing-to-loss ratio of 8p at the local cutoff,
      to place n = 8 relative to the 121 points.
  R5  what the assumption costs where it is unlicensed: at the 34 grid points
      the 121 combinations occupy, scale the bundled shells' radiative columns of
      L (off-diagonal A_bund_*, diagonal -gamma_bundled; column sums stay zero)
      by s = 0 (no radiative decay from the bundled block) and by
      s = A(np_total)/gamma_bundled (the shell's radiative rate raised to that
      of its brightest sublevel np, the statistical branching kept; A(np_total)
      imported from verify_bundling_psm20.py); recompute
      u_CRE, f_3 - f_4 at u_CRE, tau_slow with the same functions
      verify_nmax_downward_scan.py uses. The l-distribution of the bundled block
      reaches L through these columns and through the detailed-balance split of
      de-excitation into the resolved shells; the second path assumes statistical
      l by construction and cannot be varied without new data. Stated, not scanned.

GATES (all must pass before any result is printed)
--------------------------------------------------
  G0  data/processed/collisions/ccc/Te_grid.npy equals the L grid's Te exactly;
      state_index.csv (n, l) -> idx agrees with the (n, l) walk compute_lmix.py
      hardcodes; every CCC row with n_f <= 8 equals K_exc_full[si, sf] and
      K_deexc_full[sf, si] to the bit; the n_f = 9 rows summed over l_f equal
      K_exc_full[si, 36]; the n_f = 10 rows summed over l_f equal K_exc_full[si, 37]
      to 1e-12 (the l-resolved data tested is the data L was built from)
  G1  the CCC de-excitation table obeys K_deexc = K_exc (omega_i/omega_f) exp(dE/kT)
      to 1e-10 at every row and Te (names the balance and its orientation)
  G2  L = R + ne C at all eight densities to 1e-10; the Delta-n = 0 off-diagonal
      block of C equals K_lmix.npy to 1e-12; compute_K_lmix(te, NE_DEFAULT)
      reproduces K_lmix.npy to 1e-12 (so the local-cutoff rebuild is the same code)
  G3  u_CRE, a3, a4, c3, c4 from -L^-1 S reproduce validation/molecular_channel/
      molecular_channel.csv at 400/400 points to 1e-8
  G4  bundled columns of R: off-diagonal sum equals gamma_bundled and the diagonal
      equals -gamma_bundled to 1e-10 (so the R5 scaling is of the radiative part
      alone and conserves particles)
  G5  the 121 combinations re-derived from bundling_total_loss.csv (ratio_local < 1)
      number 121 and occupy 34 grid points

PREDICTIONS (written before the run)
------------------------------------
P1  R1 upward adjacent spread max/min at 1 eV falls from n = 2 to n = 9 but stays
    above 1.3 at n = 9; for n >= 5 the largest K_up(n, l) is at the highest l
    (the yrast sublevel), the smallest at low l.
P2  R1 downward adjacent spread at n = 9 and 10 is also between 1.3 and 3.
P3  R2 feed from a statistical lower shell: max |b - 1| > 0.5 at n_f = 3,
    < 0.5 at n_f = 9 and n_f = 10, with b(l_f = n_f - 1) > 1 (yrast over-fed).
P4  R3 at [0,7] (1 eV, 1e15): max_l |b_l(9) - 1| < 0.5 and weighted mean < 0.2
    in the headline (adjacent-only) column.
P5  R4 with the mixing block removed, n = 8 at [0,7]: max |b - 1| < 0.5. The
    three-body feed into n = 8 alone is non-statistical by a factor 1.28 at
    1 eV (alpha_3BR/g tracks K_ion by Saha balance; measured before the run).
P6  R5 at the 34 points: |Delta u_CRE|, |Delta(f_3 - f_4)|, |Delta tau_slow| all
    below 0.5 % for both s = 0 and s = A(np_total)/gamma_bundled.
REFUTER of the chapter's sentence ("close to l-blind at high n, so the
statistical assumption is plausible there"): max_l |b_l - 1| > 1, i.e. some
sublevel's departure coefficient off by a factor of two or more, at n = 9 in R3
(headline column) or at n = 8 in R4(b), at any of the 34 grid points the 121
combinations occupy. A spread max/min > 3 in R1 at n = 9 or 10 refutes the
"close to l-blind" clause on its own.

OUTPUTS (with --write): validation/l_blindness/
  l_blindness.txt                  this log
  l_blindness_loss_spread.csv      R1, one row per (direction, n, Te index)
  l_blindness_feed.csv             R2, one row per (direction, n_f, l_f, Te index)
  l_blindness_n9.csv               R3, one row per (grid point, l)
  l_blindness_resolved.csv         R4, one row per (grid point, operator, n)
  l_blindness_consequence.csv      R5, one row per (grid point, s)
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, sys, contextlib
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

_HERE = Path(__file__).resolve(); sys.path.insert(0, str(_HERE.parent))
from cr_context import CRContext, find_repo_root  # noqa: E402
ROOT = find_repo_root(_HERE)


def _load(name: str, path: Path):
    """Import a pipeline or validation module by path, read-only (nothing in it is edited or re-run)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()


def departure(p: np.ndarray, g: np.ndarray):
    """b_l = (p_l/g_l)/(P/G); returns (b, max|b-1|, population-weighted mean |b-1|)."""
    P = p.sum(); b = (p / g) / (P / g.sum()); dev = np.abs(b - 1.0)
    return b, dev.max(), float((p * dev).sum() / P)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--write", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = CRContext.load(); ctx.validate()
    L, te, ne = ctx.L_grid, ctx.te_grid, ctx.ne_grid; nT, nN, nS, _ = L.shape
    P = lambda *p: ROOT.joinpath(*p)
    paths = dict(L_grid=P("data/processed/cr_matrix/L_grid.npy"), S_grid=P("data/processed/cr_matrix/S_grid.npy"),
                 Te_grid_L=P("data/processed/cr_matrix/Te_grid_L.npy"), ne_grid_L=P("data/processed/cr_matrix/ne_grid_L.npy"),
                 state_index=ctx.state_index_path,
                 K_CCC_exc=P("data/processed/collisions/ccc/K_CCC_exc_table.npy"), K_CCC_deexc=P("data/processed/collisions/ccc/K_CCC_deexc_table.npy"),
                 K_CCC_meta=P("data/processed/collisions/ccc/K_CCC_metadata.csv"), Te_grid_ccc=P("data/processed/collisions/ccc/Te_grid.npy"),
                 K_exc_full=P("data/processed/collisions/K_exc_full/K_exc_full.npy"), K_deexc_full=P("data/processed/collisions/K_exc_full/K_deexc_full.npy"),
                 K_lmix=P("data/processed/lmix/K_lmix.npy"), K_ion_final=P("data/processed/collisions/tics/K_ion_final.npy"),
                 alpha_3BR_res=P("data/processed/recombination/alpha_3BR_resolved.npy"), alpha_3BR_bund=P("data/processed/recombination/alpha_3BR_bundled.npy"),
                 A_bund_res=P("data/processed/Radiative/A_bund_res.npy"), A_bund_bund=P("data/processed/Radiative/A_bund_bund.npy"),
                 gamma_resolved=P("data/processed/Radiative/gamma_resolved.npy"), gamma_bundled=P("data/processed/Radiative/gamma_bundled.npy"),
                 molecular_channel=P("validation/molecular_channel/molecular_channel.csv"),
                 bundling_total_loss=P("validation/bundling_total_loss/bundling_total_loss.csv"),
                 compute_lmix=P("src/rates/compute_lmix.py"), verify_bundling_psm20=_HERE.parent / "verify_bundling_psm20.py",
                 verify_nmax_downward_scan=_HERE.parent / "verify_nmax_downward_scan.py")
    for q in paths.values():
        if not q.is_file(): raise FileNotFoundError(f"required input missing: {q}")
    lm = _load("compute_lmix", paths["compute_lmix"]); vb = _load("verify_bundling_psm20", paths["verify_bundling_psm20"])
    ns = _load("verify_nmax_downward_scan", paths["verify_nmax_downward_scan"])
    S = np.load(paths["S_grid"]); Kx = np.load(paths["K_CCC_exc"]); Kdx = np.load(paths["K_CCC_deexc"])
    meta = pd.read_csv(paths["K_CCC_meta"]); te_ccc = np.load(paths["Te_grid_ccc"])
    Kef = np.load(paths["K_exc_full"]); Kdf = np.load(paths["K_deexc_full"]); Kl = np.load(paths["K_lmix"])
    K_ion = np.load(paths["K_ion_final"]); a3br = np.load(paths["alpha_3BR_res"]); a3bb = np.load(paths["alpha_3BR_bund"])
    A_br = np.load(paths["A_bund_res"]); A_bb = np.load(paths["A_bund_bund"])
    gam_res = np.load(paths["gamma_resolved"]); gam_bund = np.load(paths["gamma_bundled"])
    si = pd.read_csv(paths["state_index"]); mc = pd.read_csv(paths["molecular_channel"], comment="#")
    btl = pd.read_csv(paths["bundling_total_loss"], comment="#")
    log = []; say = lambda s="": (print(s), log.append(s))
    say("=" * 100); say("l-BLINDNESS OF ADJACENT-SHELL COLLISIONS AND THE l-DISTRIBUTION WHERE MIXING LOSES")
    say(f"generated {datetime.now():%Y-%m-%d %H:%M}  by {_HERE.name}"); say(f"interpreter {sys.executable}"); say(ctx.describe()); say("=" * 100)

    # ---------------------------------------------------------------- G0: indexing and the identity of the tested data with L's inputs
    nv = si.n.values.astype(int); lv = si.l.values.astype(int); gv = si.g.values.astype(float)
    if not np.array_equal(nv, ctx.n_values.astype(int)): raise RuntimeError("state_index n column disagrees with cr_context")
    byn = {(int(n), int(l)): int(k) for k, (n, l) in enumerate(zip(nv, lv))}
    for (n, l), k in lm.NL_TO_IDX.items():
        if byn.get((n, l)) != k: raise RuntimeError(f"compute_lmix NL_TO_IDX {(n, l)} -> {k} disagrees with state_index.csv {byn.get((n, l))}")
    if S.shape != (nT, nN, nS): raise RuntimeError(f"S_grid shape {S.shape}")
    res_idx = np.where(lv >= 0)[0]
    if not np.array_equal(res_idx, np.arange(len(res_idx))) or a3br.shape[0] != len(res_idx) or gam_res.shape[0] != len(res_idx):
        raise RuntimeError("resolved states are not the leading block of state_index, or the resolved-only arrays disagree in length; the [:n_res] slices below would mislabel")
    n_res_states = len(res_idx)
    if te_ccc.shape != te.shape or not np.array_equal(te_ccc, te): raise RuntimeError("CCC Te grid differs from the L grid's Te")
    if Kx.shape != (len(meta), nT) or Kdx.shape != Kx.shape: raise RuntimeError(f"CCC tables {Kx.shape}, {Kdx.shape} vs metadata {len(meta)} rows")
    row = {(int(r.n_i), int(r.l_i), int(r.n_f), int(r.l_f)): int(k) for k, r in enumerate(meta.itertuples(index=False))}
    if len(row) != len(meta): raise RuntimeError("duplicate (n_i,l_i,n_f,l_f) in CCC metadata")
    d_res = 0.0; n_res = 0
    for (ni, li, nf, lf), k in row.items():
        if nf <= 8:
            si_, sf_ = byn[(ni, li)], byn[(nf, lf)]
            d_res = max(d_res, np.abs(Kx[k] - Kef[si_, sf_]).max(), np.abs(Kdx[k] - Kdf[sf_, si_]).max()); n_res += 1
    b9 = byn[(9, -1)]; b10 = byn[(10, -1)]; d9 = 0.0; d10 = 0.0
    for (n, l), s_ in byn.items():
        if n > 8: continue
        s9 = sum(Kx[row[(n, l, 9, lf)]] for lf in range(9)); s10 = sum(Kx[row[(n, l, 10, lf)]] for lf in range(10))
        d9 = max(d9, np.abs(s9 / Kef[s_, b9] - 1).max()); d10 = max(d10, np.abs(s10 / Kef[s_, b10] - 1).max())
    say(f"\n  G0  CCC rows n_f <= 8 ({n_res}) equal K_exc_full / K_deexc_full: max abs diff {d_res:.1e};  sum over l_f of n_f = 9 rows vs K_exc_full[., n9]: {d9:.1e};  n_f = 10: {d10:.1e}")
    if d_res != 0.0 or d9 > 1e-12 or d10 > 1e-12: raise RuntimeError("Gate G0")

    # ---------------------------------------------------------------- G1: detailed balance of the CCC de-excitation table
    om_i = meta.omega_i.values[:, None]; om_f = meta.omega_f.values[:, None]; dE = meta.dE_eV.values[:, None]
    d1 = np.abs(Kdx / (Kx * (om_i / om_f) * np.exp(dE / te[None, :])) - 1).max()
    say(f"  G1  K_deexc = K_exc (omega_i/omega_f) exp(dE/kT) on all {len(meta)} rows x {nT} Te: max rel diff {d1:.1e}")
    if d1 > 1e-10: raise RuntimeError("Gate G1")

    # ---------------------------------------------------------------- G2: L = R + ne C; the mixing block
    R = np.empty((nT, nS, nS)); C = np.empty_like(R)
    for i in range(nT):
        C[i] = (L[i, 1] - L[i, 0]) / (ne[1] - ne[0]); R[i] = L[i, 0] - ne[0] * C[i]
    recon = R[:, None] + ne[None, :, None, None] * C[:, None]
    d2 = (np.abs(recon - L).max(axis=(2, 3)) / np.abs(L).max(axis=(2, 3))).max()
    same = (nv[:, None] == nv[None, :]) & ~np.eye(nS, dtype=bool)
    Kl_t = Kl.transpose(2, 0, 1)                                     # (nT, nS, nS), K_lmix[i, j, t] = j -> i
    d2b = np.abs(C[:, same] - Kl_t[:, same]).max() / np.abs(Kl).max()
    if np.abs(Kl_t[:, ~same]).max() != 0.0: raise RuntimeError("K_lmix.npy has entries outside the Delta-n = 0 off-diagonal block")
    with contextlib.redirect_stdout(io.StringIO()):
        Kl_re = lm.compute_K_lmix(te_grid=te, ne_cm3=lm.NE_DEFAULT)
    d2c = np.abs(Kl_re - Kl).max() / np.abs(Kl).max()
    say(f"  G2  L = R + ne C at all {nN} densities: max rel residual {d2:.1e};  Delta-n = 0 off-diagonal block of C = K_lmix.npy: {d2b:.1e};  compute_K_lmix(te, NE_DEFAULT = {lm.NE_DEFAULT:.0e}) = K_lmix.npy: {d2c:.1e}")
    if d2 > 1e-10 or d2b > 1e-12 or d2c > 1e-12: raise RuntimeError("Gate G2")

    # ---------------------------------------------------------------- G3: the two-channel quantities reproduce molecular_channel.csv
    gpos = ctx.ground_index; i3 = [k for k in range(nS) if nv[k] == 3]; i4 = [k for k in range(nS) if nv[k] == 4]
    Q = {q: np.empty((nT, nN)) for q in ("a3", "a4", "c3", "c4", "u", "tau")}
    pops = np.empty((nT, nN, nS))
    for i in range(nT):
        for j in range(nN):
            vals = ns.quantities(L[i, j], S[i, j], gpos, i3, i4)
            for q, v in zip(("a3", "a4", "c3", "c4", "u", "tau"), vals): Q[q][i, j] = v
            pops[i, j] = -np.linalg.solve(L[i, j], S[i, j])
    if len(mc) != nT * nN or not (np.array_equal(mc.i.values, np.repeat(np.arange(nT), nN)) and np.array_equal(mc.j.values, np.tile(np.arange(nN), nT))):
        raise RuntimeError("molecular_channel.csv row order is not (i major, j minor) over the full grid")
    d3 = max(np.abs(Q["u"].ravel() / mc.u_CRE.values - 1).max(), *[np.abs(Q[q].ravel() / mc[q].values - 1).max() for q in ("a3", "a4", "c3", "c4")])
    say(f"  G3  u_CRE, a3, a4, c3, c4 from -L^-1 S vs molecular_channel.csv at {len(mc)} points: max rel diff {d3:.1e}")
    if d3 > 1e-8: raise RuntimeError("Gate G3")
    if pops.min() <= 0: raise RuntimeError("a CRE population is not positive")

    # ---------------------------------------------------------------- G4: the bundled radiative columns
    bidx = np.array([k for k in range(nS) if lv[k] < 0]); nb = nv[bidx]
    off = ~np.eye(nS, dtype=bool)
    d4 = max(np.abs(np.stack([R[:, off[:, k], k].sum(axis=1) for k in bidx]) / gam_bund[:, None] - 1).max(),
             np.abs(np.stack([R[:, k, k] for k in bidx]) / (-gam_bund[:, None]) - 1).max())
    say(f"  G4  bundled columns of R: off-diagonal sum = gamma_bundled and diagonal = -gamma_bundled: max rel diff {d4:.1e}")
    if d4 > 1e-10: raise RuntimeError("Gate G4")

    # ---------------------------------------------------------------- G5: the 121 combinations and their 34 grid points
    b121 = btl[btl.ratio_local < 1]; pts = sorted(set(zip(b121.i.astype(int), b121.j.astype(int))))
    say(f"  G5  bundling_total_loss.csv: ratio_local < 1 at {len(b121)} combinations occupying {len(pts)} grid points "
        f"(ne indices {sorted(set(j for _, j in pts))}, Te indices {min(i for i, _ in pts)}..{max(i for i, _ in pts)})")
    if len(b121) != 121 or len(pts) != 34: raise RuntimeError("Gate G5")
    say("\n  ALL GATES PASSED.")

    tsel = [0, ctx.nearest_point(2.947, 1.389e14)[0], nT - 1]      # 1 eV, the benchmark row, 10 eV
    P5_3BR_SPREAD = 1.2833   # measured 22 Sep 2026 before the run: max/min over l of alpha_3BR_resolved[n = 8]/g at Te index 0; R4 prints the same quantity
    w_stat = lambda n: (2 * np.arange(n) + 1) / n**2
    rows_loss, rows_feed, rows_n9, rows_res, rows_cons = [], [], [], [], []

    # ================================================================ R1: loss blindness
    say("\n" + "=" * 100); say("RESULT 1: adjacent-shell transfer OUT of (n, l), summed over l' of the neighbour shell, across l  (CCC, cm^3/s)"); say("=" * 100)
    K_up = {}; K_dn = {}
    for n in range(1, 10):
        K_up[n] = np.array([sum(Kx[row[(n, l, n + 1, lf)]] for lf in range(n + 1)) for l in range(n)])          # (n, nT)
    for n in range(2, 11):
        K_dn[n] = np.array([sum(Kdx[row[(n - 1, li, n, l)]] for li in range(n - 1)) for l in range(n)])         # (n, nT)
    for lab, KD in (("up n -> n+1", K_up), ("down n -> n-1", K_dn)):
        say(f"\n  {lab}:  spread = max_l / min_l;  argmax / argmin l;  at Te = {te[tsel[0]]:.2f}, {te[tsel[1]]:.3f}, {te[tsel[2]]:.1f} eV;  then the max spread over all 50 Te;")
        say(f"  then the (2l+1)-weighted coefficient of variation and weighted mean |K/K_mean - 1| at 1 eV (the population-weighted measures of the same l-dependence)")
        say(f"  {'n':>3} " + " ".join(f"{'spread':>7} {'lmax':>4} {'lmin':>4} |" for _ in tsel) + f" {'max over Te':>11} {'at Te':>6} | {'wCV':>6} {'wMAD':>6}")
        for n, Kn in KD.items():
            cells = []
            for t in tsel:
                v = Kn[:, t]; cells.append(f"{v.max()/v.min():>7.3f} {int(v.argmax()):>4d} {int(v.argmin()):>4d} |")
            sp = Kn.max(axis=0) / Kn.min(axis=0); tm = int(sp.argmax())
            w = w_stat(n); Km = (w[:, None] * Kn).sum(axis=0); wcv = np.sqrt((w[:, None] * (Kn - Km) ** 2).sum(axis=0)) / Km; wmad = (w[:, None] * np.abs(Kn / Km - 1)).sum(axis=0)
            say(f"  {n:>3} " + " ".join(cells) + f" {sp.max():>11.3f} {te[tm]:>6.2f} | {wcv[0]:>6.3f} {wmad[0]:>6.3f}")
            for t in range(nT):
                v = Kn[:, t]
                rows_loss.append(dict(direction=lab.split()[0], n=n, t=t, Te_eV=te[t], spread=v.max() / v.min(), l_max=int(v.argmax()), l_min=int(v.argmin()),
                                      K_min=v.min(), K_max=v.max(), K_stat_mean=float(Km[t]), weighted_cv=float(wcv[t]), weighted_mad=float(wmad[t])))
        say(f"  per-l values at 1 eV for the highest shells:")
        for n in list(KD.keys())[-3:]:
            say(f"    n = {n:>2}: " + " ".join(f"l{l}={KD[n][l, 0]:.3e}" for l in range(n)))

    # ================================================================ R2: feed statisticality
    say("\n" + "=" * 100); say("RESULT 2: l' distribution of the transfer INTO shell n_f from a STATISTICALLY populated neighbour shell; b(l') = share / [(2l'+1)/n_f^2]"); say("=" * 100)
    feed_up = {}; feed_dn = {}
    for nf in range(2, 11):
        ni = nf - 1; w = w_stat(ni)
        feed_up[nf] = np.array([sum(w[li] * Kx[row[(ni, li, nf, lf)]] for li in range(ni)) for lf in range(nf)])   # (nf, nT)
    for nf in range(1, 10):
        ni = nf + 1; w = w_stat(ni)
        feed_dn[nf] = np.array([sum(w[li] * Kdx[row[(nf, lf, ni, li)]] for li in range(ni)) for lf in range(nf)])   # (nf, nT)
    for lab, F in (("from n_f - 1 (excitation)", feed_up), ("from n_f + 1 (de-excitation)", feed_dn)):
        say(f"\n  {lab}:  max_l' |b - 1|  and  feed-weighted mean |b - 1|  at Te = {te[tsel[0]]:.2f}, {te[tsel[1]]:.3f}, {te[tsel[2]]:.1f} eV;  b(l'=0) and b(yrast) at 1 eV")
        say(f"  {'n_f':>3} " + " ".join(f"{'max|b-1|':>8} {'mean':>6} |" for _ in tsel) + f" {'b(0)':>7} {'b(yrast)':>8}  {'b at 1 eV':}")
        for nf, Fm in F.items():
            if nf < 2: continue
            cells = []; bvec = None
            for t in tsel:
                b, mx, mn = departure(Fm[:, t], 2 * (2 * np.arange(nf) + 1.0)); cells.append(f"{mx:>8.3f} {mn:>6.3f} |")
                if t == 0: bvec = b
            say(f"  {nf:>3} " + " ".join(cells) + f" {bvec[0]:>7.3f} {bvec[-1]:>8.3f}  " + " ".join(f"{x:.2f}" for x in bvec))
            for t in range(nT):
                b, mx, mn = departure(Fm[:, t], 2 * (2 * np.arange(nf) + 1.0))
                for lf in range(nf):
                    rows_feed.append(dict(direction="up" if F is feed_up else "down", n_f=nf, l_f=lf, t=t, Te_eV=te[t], b=b[lf], max_abs_dev=mx, mean_abs_dev=mn))

    # ================================================================ R3: n = 9 from its two resolved neighbours
    say("\n" + "=" * 100); say("RESULT 3: the first bundled shell, n = 9, populated by collisions from n = 8 and n = 10 alone (both statistical), losses to 8 and 10 only"); say("=" * 100)
    g9 = 2 * (2 * np.arange(9) + 1.0); w8 = w_stat(8); w10 = w_stat(10)
    feed8 = np.array([sum(w8[l_] * Kx[row[(8, l_, 9, l)]] for l_ in range(8)) for l in range(9)])       # (9, nT) per unit N_8 ne
    feed10 = np.array([sum(w10[l_] * Kdx[row[(9, l, 10, l_)]] for l_ in range(10)) for l in range(9)])  # (9, nT) per unit N_10 ne
    loss9 = np.array([sum(Kdx[row[(8, l_, 9, l)]] for l_ in range(8)) + sum(Kx[row[(9, l, 10, l_)]] for l_ in range(10)) for l in range(9)])  # (9, nT) cm^3/s
    sh8 = nv == 8; k9 = byn[(9, -1)]; k10 = byn[(10, -1)]
    btl9 = btl[btl.n == 9].set_index(["i", "j"])
    # Detailed balance makes the outcome depend on one number. With K_deexc = K_exc (omega_i/omega_f) exp(dE/kT) (G1),
    #   feed8(l)  = (omega_9l / G_8) exp(-dE_89/kT) K_down(9, l)   and   feed10(l) = (omega_9l / G_10) exp(+dE_910/kT) K_up(9, l),
    # so p_l / omega_l = [alpha K_down(l) + beta K_up(l)] / [K_down(l) + K_up(l)] with alpha = N_8 exp(-dE_89/kT)/G_8, beta = N_10 exp(dE_910/kT)/G_10.
    # alpha = beta exactly when N_8/N_10 is the Boltzmann ratio; then p_l/omega_l is constant whatever the l-dependence of the rates.
    # theta = alpha/beta is that Boltzmann departure of the neighbour pair; b_l - 1 is of order (theta - 1) x [K_down/(K_down+K_up) - its mean].
    dE89 = float(meta.dE_eV[row[(8, 0, 9, 0)]]); dE910 = float(meta.dE_eV[row[(9, 0, 10, 0)]]); G8 = gv[nv == 8].sum(); G10 = gv[nv == 10].sum()
    B9 = np.empty((nT, nN, 9)); dev9 = np.empty((nT, nN)); mean9 = np.empty((nT, nN)); dev9f = np.empty((nT, nN)); share_blind = np.empty((nT, nN)); theta = np.empty((nT, nN))
    for i in range(nT):
        for j in range(nN):
            N8 = pops[i, j, sh8].sum(); N10 = pops[i, j, k10]
            theta[i, j] = (N8 * np.exp(-dE89 / te[i]) / G8) / (N10 * np.exp(dE910 / te[i]) / G10)
            fd = N8 * feed8[:, i] + N10 * feed10[:, i]; ls = loss9[:, i]
            b, mx, mn = departure(fd / ls, g9); B9[i, j] = b; dev9[i, j] = mx; mean9[i, j] = mn
            r = btl9.loc[(i, j)]
            blind = r.total_loss * (1.0 - r.frac_coll * r.frac_coll_adjacent - r.frac_rad)     # ionisation + Delta-n >= 2 transfer, l-blind in the model  [s^-1]
            share_blind[i, j] = blind / r.total_loss
            _, mxf, _ = departure(fd / (ls + blind / ne[j]), g9); dev9f[i, j] = mxf
            for l in range(9):
                rows_n9.append(dict(i=i, j=j, Te_eV=te[i], ne_cm3=ne[j], l=l, b_adjacent_only=b[l], max_abs_dev=mx, mean_abs_dev=mn, max_abs_dev_with_blind_loss=mxf,
                                    N8_over_N10=N8 / N10, theta_boltzmann_departure=theta[i, j], feed_from_8_share=float((N8 * feed8[:, i]).sum() / fd.sum()), blind_loss_share=share_blind[i, j]))
    say(f"  by detailed balance p_l/omega_l = [alpha K_down(l) + beta K_up(l)]/[K_down(l) + K_up(l)]; theta = alpha/beta = 1 when N_8/N_10 is Boltzmann, and then b_l = 1 for ANY l-dependence of the rates")
    say(f"  {'point':>8} {'Te eV':>6} {'ne':>9} {'theta-1':>9} {'max|b-1|':>9} {'mean|b-1|':>9} {'+blind':>7} {'blind%':>6} {'feed8%':>6}  b_l(9), l = 0..8")
    named = [(0, 7), (0, 6), (10, 7), (23, 7), (23, 5), (0, 0), (49, 7)]
    for (i, j) in named:
        say(f"  {str((i, j)):>8} {te[i]:>6.2f} {ne[j]:>9.2e} {theta[i, j]-1:>+9.1e} {dev9[i, j]:>9.4f} {mean9[i, j]:>9.4f} {dev9f[i, j]:>7.3f} {100*share_blind[i, j]:>6.1f} "
            f"{100*float((pops[i, j, sh8].sum()*feed8[:, i]).sum()/((pops[i, j, sh8].sum()*feed8[:, i]) + pops[i, j, k10]*feed10[:, i]).sum()):>6.1f}  " + " ".join(f"{x:.3f}" for x in B9[i, j]))
    d34 = np.array([dev9[i, j] for (i, j) in pts]); m34 = np.array([mean9[i, j] for (i, j) in pts]); f34 = np.array([dev9f[i, j] for (i, j) in pts])
    t34 = np.array([abs(theta[i, j] - 1) for (i, j) in pts])
    say(f"  over the 34 points of the 121 combinations: |theta-1| {t34.min():.1e} to {t34.max():.1e}; max|b-1| {d34.min():.4f} to {d34.max():.4f} (at {pts[int(d34.argmax())]}), mean|b-1| up to {m34.max():.4f};"
        f" with the l-blind loss in the denominator and no l-blind feed (an inconsistent extreme, kept as a bound) up to {f34.max():.3f}")
    say(f"  over the whole grid: |theta-1| up to {np.abs(theta-1).max():.3f} at {tuple(int(v) for v in np.unravel_index(np.abs(theta-1).argmax(), theta.shape))}; max|b-1| {dev9.min():.4f} to {dev9.max():.4f} at {tuple(int(v) for v in np.unravel_index(dev9.argmax(), dev9.shape))}")
    say(f"  the l-dependence that theta - 1 multiplies: K_down/(K_down + K_up) at 1 eV runs from {(loss9[:, 0] - K_up[9][:, 0]).min()/loss9[:, 0][np.argmin(loss9[:, 0] - K_up[9][:, 0])]:.3f} to {((loss9[:, 0] - K_up[9][:, 0])/loss9[:, 0]).max():.3f} across l")
    say(f"  the l that departs most at [0,7]: l = {int(np.abs(B9[0, 7] - 1).argmax())} (b = {B9[0, 7][int(np.abs(B9[0, 7] - 1).argmax())]:.3f});  b(9s) = {B9[0, 7][0]:.3f}, b(9p) = {B9[0, 7][1]:.3f}, b(yrast l=8) = {B9[0, 7][8]:.3f}")
    # the identity, checked numerically: b_l - 1 = (theta - 1) [f_dn(l) - mean f_dn] to first order
    f_dn = (loss9 - K_up[9]) / loss9                                                     # (9, nT)
    ident = max(np.abs((B9[i, j] - 1) - (theta[i, j] - 1) * (f_dn[:, i] - (g9 * f_dn[:, i]).sum() / g9.sum())).max() / max(abs(theta[i, j] - 1), 1e-300) for (i, j) in pts + [(49, 0)])
    say(f"  identity check: |(b_l - 1) - (theta - 1)[f_dn(l) - mean f_dn]| / |theta - 1| <= {ident:.1e} over the 34 points and [49,0]  (so max|b-1| <= {np.abs(f_dn[:, 0] - (g9 * f_dn[:, 0]).sum() / g9.sum()).max():.3f} |theta - 1| at 1 eV: R3's headline is an identity in theta, not a test of the rates)")
    # R3c: radiative decay out of 9p is l-dependent and has no compensating feed; first-order estimate of the 9p departure it alone would cause
    A9p_excess = vb.A_np_total(9) - gam_bund[0]                                          # s^-1, A(9p_total) minus the shell average
    rad9p = -A9p_excess / (ne[None, :] * loss9[1][:, None])                              # (nT, nN): b(9p) - 1 from the uncompensated radiative loss
    say(f"  R3c  radiative loss out of 9p exceeds the shell average by A(9p_total) - gamma_9 = {A9p_excess:.3e} s^-1 (A_np_total from verify_bundling_psm20.py); against the collisional loss of 9p this shifts b(9p) by "
        f"{rad9p[0, 7]:+.1e} at [0,7], at most {max(abs(rad9p[i, j]) for (i, j) in pts):.1e} over the 34 points, and {rad9p[0, 0]:+.3f} at [0,0] (where mixing at 50x the loss covers it)")
    # R3b: the same neighbour-pair Boltzmann departure for every bundled shell, from the model's own CRE populations and state_index energies
    say(f"\n  R3b  theta_n - 1 = Boltzmann departure of the neighbour pair (n-1, n+1) from the model's CRE populations, for every bundled shell (I_eV, g from state_index.csv):")
    n_lo, n_hi = int(nb.min()), int(nb.max())
    I_eV = si.I_eV.values.astype(float); Ntot = {n: pops[:, :, nv == n].sum(axis=2) for n in range(n_lo - 1, n_hi + 1)}
    Gn = {n: gv[nv == n].sum() for n in range(n_lo - 1, n_hi + 1)}; In = {n: float(I_eV[nv == n][0]) for n in range(n_lo - 1, n_hi + 1)}
    say(f"  (check: dE(8->9) from CCC metadata {dE89:.6f} eV vs state_index {In[8]-In[9]:.6f} eV;  dE(9->10) {dE910:.6f} vs {In[9]-In[10]:.6f})")
    say(f"  {'n':>3} {'max|theta-1| over 34 pts':>24} {'at':>8} {'[0,7]':>10} {'[23,7]':>10} {'[23,5]':>10} {'[0,0]':>10} {'grid max':>10} {'at':>8}")
    theta_n = {}
    for n in range(n_lo, n_hi):
        th = (Ntot[n - 1] / Ntot[n + 1]) / ((Gn[n - 1] / Gn[n + 1]) * np.exp((In[n - 1] - In[n + 1]) / te[:, None])); theta_n[n] = th
        t34n = np.array([abs(th[i, j] - 1) for (i, j) in pts]); k = int(t34n.argmax()); gm = tuple(int(v) for v in np.unravel_index(np.abs(th - 1).argmax(), th.shape))
        say(f"  {n:>3} {t34n.max():>24.1e} {str(pts[k]):>8} {th[0, 7]-1:>+10.1e} {th[23, 7]-1:>+10.1e} {th[23, 5]-1:>+10.1e} {th[0, 0]-1:>+10.1e} {np.abs(th-1).max():>10.3f} {str(gm):>8}")
        for (i, j) in pts: rows_n9.append(dict(i=i, j=j, Te_eV=te[i], ne_cm3=ne[j], l=-1, b_adjacent_only=np.nan, max_abs_dev=np.nan, mean_abs_dev=np.nan, max_abs_dev_with_blind_loss=np.nan,
                                              N8_over_N10=np.nan, theta_boltzmann_departure=th[i, j], feed_from_8_share=np.nan, blind_loss_share=np.nan, shell_n=n))
    say(f"  n = {n_hi} has no upper neighbour in the model (terminal shell, see sec:convergence); its pair cannot be formed.")
    # Saha departure of each high shell against the continuum, using the model's own ionisation / three-body pair:
    # alpha_3BR = K_ion (g/2) lambda^3 exp(I/kT) (recombination_rates.alpha_3BR_from_Kion), so N_Saha per unit n_ion = ne alpha_3BR / K_ion.
    say(f"\n  Saha departure b_Saha(n) = N_n / [ne alpha_3BR(n)/K_ion(n)] (per unit n_ion) from the model's own pair, at the 34 points (range) and at [0,0]:")
    a3_all = np.vstack([a3br, a3bb])                                                    # (nS, nT), resolved then bundled: the leading-block gate above licenses this
    if a3_all.shape != K_ion.shape: raise RuntimeError("alpha_3BR arrays do not stack to K_ion's shape")
    N_saha = ne[None, :, None] * (a3_all / K_ion).T[:, None, :]                          # (nT, nN, nS)
    bsaha = {}
    for n in range(n_lo - 1, n_hi + 1):
        selN = nv == n; bs = pops[:, :, selN].sum(axis=2) / N_saha[:, :, selN].sum(axis=2); bsaha[n] = bs
        v34 = np.array([bs[i, j] for (i, j) in pts])
        say(f"    n = {n:>2}: b_Saha - 1 over the 34 points {v34.min()-1:+.1e} to {v34.max()-1:+.1e};  [0,0] {bs[0, 0]-1:+.3f};  [49,0] {bs[49, 0]-1:+.3f};  grid range {bs.min()-1:+.3f} to {bs.max()-1:+.3f}")
    # complementarity: where the neighbour pair is NOT in Boltzmann ratio, does mixing at the local cutoff cover the shell?
    minloc = btl.groupby(["i", "j"]).ratio_local.min()
    for thr in (1e-2, 1e-3):
        bad = [(i, j) for i in range(nT) for j in range(nN) if abs(theta[i, j] - 1) > thr]
        both = [(i, j) for (i, j) in bad if minloc.loc[(i, j)] < 1]
        say(f"  complementarity: |theta_9 - 1| > {thr:g} at {len(bad)} of 400 points ({sum(1 for (i, j) in bad if te[i] >= 2.0)} at Te >= 2 eV), ne up to {max((ne[j] for (_, j) in bad), default=float('nan')):.1e}; "
            f"at those the smallest local mixing-to-loss ratio over the bundled shells is {min((minloc.loc[(i, j)] for (i, j) in bad), default=float('nan')):.0f}; points failing BOTH criteria: {len(both)}")

    # ================================================================ R4: the model's resolved shells with the mixing block removed or localised
    say("\n" + "=" * 100); say("RESULT 4: CRE l-distribution of the resolved shells from the model's own operator: as built / mixing removed / mixing at the local cutoff"); say("=" * 100)
    Lm_loc = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for j in range(nN): Lm_loc[j] = lm.compute_K_lmix(te_grid=te, ne_cm3=ne[j]).transpose(2, 0, 1)   # (nT, nS, nS) cm^3/s
    def strip_mixing(Lij, i, j):
        M = Kl_t[i] * ne[j]; return Lij - M + np.diag(M.sum(axis=0))
    def add_mixing(Lij, M):
        return Lij + M - np.diag(M.sum(axis=0))
    ops = ("built", "nomix", "local")
    DEV = {o: np.empty((nT, nN, 9)) for o in ops}; MEAN = {o: np.empty((nT, nN, 9)) for o in ops}; BL = {o: {} for o in ops}
    gam_eff = np.empty((nT, nN, 3)); ratio8p = np.empty((nT, nN, 2)); k8p = byn[(8, 1)]
    for i in range(nT):
        for j in range(nN):
            L0 = L[i, j]; L1 = strip_mixing(L0, i, j); L2 = add_mixing(L1, Lm_loc[j][i] * ne[j])
            for o, Lo in zip(ops, (L0, L1, L2)):
                p = -np.linalg.solve(Lo, S[i, j])
                if p.min() <= 0: raise RuntimeError(f"non-positive population under operator {o} at {(i, j)}")
                for n in range(2, 9):
                    sel = nv == n; b, mx, mn = departure(p[sel], gv[sel]); DEV[o][i, j, n] = mx; MEAN[o][i, j, n] = mn
                    if n >= 6: BL[o][(i, j, n)] = b
                    rows_res.append(dict(i=i, j=j, Te_eV=te[i], ne_cm3=ne[j], operator=o, n=n, max_abs_dev=mx, mean_abs_dev=mn, b_l=" ".join(f"{x:.5f}" for x in b)))
                if o == "nomix":
                    sel = nv == 8; gam_eff[i, j, 0] = (p[sel] * gam_res[sel[:n_res_states]]).sum() / p[sel].sum()
            sel = nv == 8; gam_eff[i, j, 1] = (gv[sel] * gam_res[sel[:n_res_states]]).sum() / gv[sel].sum(); gam_eff[i, j, 2] = gam_eff[i, j, 0] / gam_eff[i, j, 1]
            ratio8p[i, j, 0] = (Kl_t[i][:, k8p].sum() * ne[j]) / (-L1[k8p, k8p]); ratio8p[i, j, 1] = (Lm_loc[j][i][:, k8p].sum() * ne[j]) / (-L1[k8p, k8p])
    say(f"  max_l |b_l - 1| for n = 8 (n = 7 in brackets):")
    say(f"  {'point':>8} {'Te eV':>6} {'ne':>9} | {'built':>14} {'no mixing':>14} {'local cutoff':>14} | {'8p mix/loss frozen':>18} {'local':>7} | {'gamma_eff/gamma_stat (n=8, no mix)':>20}")
    for (i, j) in named:
        say(f"  {str((i, j)):>8} {te[i]:>6.2f} {ne[j]:>9.2e} | " + " ".join(f"{DEV[o][i, j, 8]:>6.3f} ({DEV[o][i, j, 7]:>5.3f})" for o in ops)
            + f" | {ratio8p[i, j, 0]:>18.2f} {ratio8p[i, j, 1]:>7.3f} | {gam_eff[i, j, 2]:>20.4f}")
    for o in ops:
        d = np.array([DEV[o][i, j, 8] for (i, j) in pts]); d7 = np.array([DEV[o][i, j, 7] for (i, j) in pts])
        say(f"  {o:>6}: over the 34 points, n = 8 max|b-1| {d.min():.4f} to {d.max():.4f} (at {pts[int(d.argmax())]}); n = 7 up to {d7.max():.4f};  grid-wide n = 8 up to {DEV[o][:, :, 8].max():.4f} at {tuple(int(v) for v in np.unravel_index(DEV[o][:, :, 8].argmax(), (nT, nN)))}")
    say(f"  no mixing, b_l for n = 8 at [0,7]: " + " ".join(f"{x:.3f}" for x in BL['nomix'][(0, 7, 8)]) + f";  at [23,7]: " + " ".join(f"{x:.3f}" for x in BL['nomix'][(23, 7, 8)]))
    say(f"  no mixing, b_l for n = 8 at [0,0]: " + " ".join(f"{x:.3f}" for x in BL['nomix'][(0, 0, 8)]) + "   (low density: radiative decay competes, the l-distribution is not the collisional one)")
    say(f"  no mixing, max|b-1| by shell at [0,7]: " + "  ".join(f"n{n}: {DEV['nomix'][0, 7, n]:.3f}" for n in range(2, 9)))
    say(f"  8p mixing-to-loss ratio at the local cutoff over the 34 points: {min(ratio8p[i, j, 1] for (i, j) in pts):.3f} to {max(ratio8p[i, j, 1] for (i, j) in pts):.3f}  (n = 9 in bundling_total_loss: 0.86 at [0,7])")
    # the three-body feed into n = 8, l-resolved, versus statistical (P5's second clause)
    sel = nv == 8
    for t in (0, tsel[1], nT - 1):
        b3, mx3, _ = departure(a3br[sel[:n_res_states], t], gv[sel])
        say(f"  three-body recombination into n = 8 at Te = {te[t]:.2f} eV: b_l = " + " ".join(f"{x:.3f}" for x in b3) + f"  max|b-1| = {mx3:.3f}  (alpha_3BR/g follows K_ion by Saha balance)")
    # gain into shell 8 by source, no-mixing operator, at [0,7]
    for (i, j) in ((0, 7), (23, 5)):
        L1 = strip_mixing(L[i, j], i, j); p = -np.linalg.solve(L1, S[i, j]); Rr = R[i]; Cc = C[i] * ne[j]
        groups = {"n=7 coll": (nv == 7, Cc), "n<=6 coll": (nv <= 6, Cc), "n=9 coll (DB of l-summed CCC)": (nv == 9, Cc), "n=10 coll (DB of l-summed CCC)": (nv == 10, Cc),
                  "n>=11 coll (VS)": (nv >= 11, Cc), "radiative cascade from above": (nv > 8, Rr)}
        gains = {k: float((M[np.ix_(sel, m)] @ p[m]).sum()) for k, (m, M) in groups.items()}
        gains["recombination (S)"] = float(S[i, j][sel].sum()); tot = sum(gains.values())
        say(f"  gain into shell 8 by source at {(i, j)}, no-mixing operator: " + ", ".join(f"{k} {100*v/tot:.1f} %" for k, v in gains.items()))

    # ================================================================ R5: consequence bracket
    say("\n" + "=" * 100); say("RESULT 5: scaling the bundled shells' radiative columns by s (0: dark; s_max: shell rate raised to A(np_total), statistical branching), effect on u_CRE, f_3 - f_4, tau_slow"); say("=" * 100)
    A_np = np.array([vb.A_np_total(int(n)) for n in nb]); s_max = A_np / gam_bund
    say(f"  s_max = A(np_total)/gamma_bundled per shell n = {nb.tolist()}: " + " ".join(f"{x:.2f}" for x in s_max))
    # what the bundled radiative decay is, at the 34 points: its share of the bundled loss and its size against the net flow into the ground state
    fr34 = btl[[((i, j) in pts) for (i, j) in zip(btl.i, btl.j)]].frac_rad
    say(f"  radiative share of the bundled shells' loss at the 34 points: {fr34.min():.1e} to {fr34.max():.1e} (the 0.19 % of RESULT 2 in bundling_total_loss is the grid maximum, at 1e12 cm^-3)")
    say(f"  flux budget per unit n_ion at CRE: net flow into the ground state = N_1 K_ion(1s) ne (balanced by ground ionisation); bundled radiative flux = sum_k gamma_k N_k, split by destination:")
    say(f"  {'point':>8} {'S_tot':>10} {'ground flux':>11} {'ground/S_tot':>12} {'bundled rad':>11} {'rad/ground':>10} | {'to 1s':>6} {'n=2':>6} {'n=3':>6} {'n=4-8':>6} {'bundled':>7}")
    budget = {}
    for (i, j) in [(9, 6), (0, 7), (23, 7)] + [pt for pt in pts if pt not in ((9, 6), (0, 7), (23, 7))]:
        pp = pops[i, j]; Stot = S[i, j].sum(); gflux = pp[gpos] * K_ion[gpos, i] * ne[j]; brad = (gam_bund * pp[bidx]).sum()
        dest = {"1s": A_br[gpos][None, :] @ pp[bidx], "n=2": A_br[nv[:n_res_states] == 2] @ pp[bidx], "n=3": A_br[nv[:n_res_states] == 3] @ pp[bidx],
                "n=4-8": A_br[(nv[:n_res_states] >= 4)] @ pp[bidx], "bundled": A_bb @ pp[bidx]}
        dest = {k: float(np.sum(v)) for k, v in dest.items()}; budget[(i, j)] = (Stot, gflux, brad, dest)
        if (i, j) in ((9, 6), (0, 7), (23, 7)):
            say(f"  {str((i, j)):>8} {Stot:10.4e} {gflux:11.4e} {gflux/Stot:12.1e} {brad:11.4e} {brad/gflux:10.3f} | " + " ".join(f"{100*dest[k]/brad:6.1f}" for k in ("1s", "n=2", "n=3", "n=4-8", "bundled")) + " %")
    rg = np.array([budget[pt][2] / budget[pt][1] for pt in pts])
    say(f"  bundled radiative flux / net ground flux over the 34 points: {rg.min():.3f} to {rg.max():.3f};  destination split check at (9,6): sum of destinations / bundled radiative flux = {sum(budget[(9, 6)][3].values())/budget[(9, 6)][2]:.6f}")
    def scaled(Lij, i, s_vec):
        Ls = Lij.copy()
        for b, k in enumerate(bidx): Ls[:, k] += (s_vec[b] - 1.0) * R[i][:, k]
        return Ls
    worst = {}; du_s0_96 = None; dq_s0_96 = None
    for lab, svec in (("s = 0", np.zeros(len(bidx))), ("s = s_max", s_max)):
        say(f"\n  {lab}:  {'point':>8} {'Te eV':>6} {'ne':>9} {'du_CRE %':>9} {'dS %':>8} {'dtau %':>8}")
        recs = []
        for (i, j) in pts + [(23, 5), (0, 0), (0, 4)]:
            a3_, a4_, c3_, c4_, u_, tau_ = ns.quantities(scaled(L[i, j], i, svec), S[i, j], gpos, i3, i4)
            S0 = ns.fdiff(Q["a3"][i, j], Q["a4"][i, j], Q["c3"][i, j], Q["c4"][i, j], Q["u"][i, j]); S1 = ns.fdiff(a3_, a4_, c3_, c4_, u_)
            du, dS, dt = 100 * (u_ / Q["u"][i, j] - 1), 100 * (S1 / S0 - 1), 100 * (tau_ / Q["tau"][i, j] - 1)
            if lab == "s = 0" and (i, j) == (9, 6):
                du_s0_96 = du; dq_s0_96 = dict(a3=100 * (a3_ / Q["a3"][i, j] - 1), a4=100 * (a4_ / Q["a4"][i, j] - 1), c3=100 * (c3_ / Q["c3"][i, j] - 1), c4=100 * (c4_ / Q["c4"][i, j] - 1))
            recs.append(((i, j), du, dS, dt)); rows_cons.append(dict(i=i, j=j, Te_eV=te[i], ne_cm3=ne[j], scaling=lab, in_121=(i, j) in pts, du_CRE_pct=du, dS_pct=dS, dtau_slow_pct=dt, S_base=S0, u_base=Q["u"][i, j]))
        in34 = [r for r in recs if r[0] in pts]
        for r in sorted(in34, key=lambda r: -max(abs(r[1]), abs(r[2]), abs(r[3])))[:4] + [r for r in recs if r[0] not in pts]:
            say(f"  {'':>8}  {str(r[0]):>8} {te[r[0][0]]:>6.2f} {ne[r[0][1]]:>9.2e} {r[1]:>+9.4f} {r[2]:>+8.4f} {r[3]:>+8.4f}" + ("" if r[0] in pts else "   (outside the 121, for scale)"))
        worst[lab] = max(max(abs(r[1]), abs(r[2]), abs(r[3])) for r in in34)
        say(f"  {'':>8}  largest |change| over the 34 points: {worst[lab]:.4f} %")
        # theta_9 under the same rescaling: is the neighbour-pair Boltzmann ratio an artefact of the bundled block's radiative assumptions?
        th_s = []
        for (i, j) in pts:
            ps = -np.linalg.solve(scaled(L[i, j], i, svec), S[i, j]); N8 = ps[sh8].sum(); N10 = ps[k10]
            th_s.append((N8 * np.exp(-dE89 / te[i]) / G8) / (N10 * np.exp(dE910 / te[i]) / G10))
        th_s = np.array(th_s); say(f"  {'':>8}  theta_9 - 1 over the 34 points under this scaling: {th_s.min()-1:+.1e} to {th_s.max()-1:+.1e} (built: {t34.min():.1e} to {t34.max():.1e})")
    # destination-resolved removal at (9,6): remove only the bundled decays landing in one destination group (column sums kept zero)
    i, j = 9, 6
    say(f"\n  destination-resolved removal at (9,6), du_CRE when only the bundled decays into one group are removed (their loss removed from the diagonal too):")
    parts = {}
    for lab2, rows_sel in (("1s", nv == 1), ("n=2", nv == 2), ("n=3", nv == 3), ("n=4-8", (nv >= 4) & (nv <= 8)), ("bundled", nv >= 9)):
        Ls = L[i, j].copy()
        for b, k in enumerate(bidx):
            col = R[i][:, k].copy(); col[k] = 0.0; take = np.where(rows_sel, col, 0.0); take[k] = 0.0
            Ls[:, k] -= take; Ls[k, k] += take.sum()
        u_ = (-np.linalg.solve(Ls, S[i, j]))[gpos]; parts[lab2] = 100 * (u_ / Q["u"][i, j] - 1)
    say("    " + ", ".join(f"{k} {v:+.3f} %" for k, v in parts.items()) + f";  sum {sum(parts.values()):+.3f} % (s = 0 gave {du_s0_96:+.3f} %)")
    say(f"  at (9,6), s = 0, the two-channel coefficients move by a3 {dq_s0_96['a3']:+.3f} %, a4 {dq_s0_96['a4']:+.3f} %, c3 {dq_s0_96['c3']:+.3f} %, c4 {dq_s0_96['c4']:+.3f} % against u_CRE {du_s0_96:+.3f} %: the effect is on u itself")

    # ================================================================ predictions and refuter
    say("\n" + "=" * 100); say("PREDICTIONS (written before the run) against what came out"); say("=" * 100)
    missed = []
    sp = {n: K_up[n][:, 0].max() / K_up[n][:, 0].min() for n in K_up}
    yr = all(int(K_up[n][:, 0].argmax()) == n - 1 for n in range(5, 10))
    p1 = sp[2] > sp[9] > 1.3 and yr
    say(f"  P1 up-spread at 1 eV: n=2 {sp[2]:.3f} ... n=9 {sp[9]:.3f} (falls: {sp[2] > sp[9]}, stays > 1.3: {sp[9] > 1.3}); yrast carries the max at n = 5..9: {yr}: {'reproduced' if p1 else 'NOT reproduced'}")
    if not p1: missed.append("P1")
    spd = {n: K_dn[n].max(axis=0) / K_dn[n].min(axis=0) for n in (9, 10)}
    p2 = all(1.3 <= spd[n].min() and spd[n].max() <= 3 for n in (9, 10))
    say(f"  P2 down-spread over Te at n = 9: {spd[9].min():.3f} to {spd[9].max():.3f}; n = 10: {spd[10].min():.3f} to {spd[10].max():.3f} (predicted within 1.3 to 3): {'reproduced' if p2 else 'NOT reproduced'}")
    if not p2: missed.append("P2")
    fdev = {nf: departure(feed_up[nf][:, 0], 2 * (2 * np.arange(nf) + 1.0)) for nf in (3, 9, 10)}
    p3 = fdev[3][1] > 0.5 and fdev[9][1] < 0.5 and fdev[10][1] < 0.5 and fdev[9][0][-1] > 1 and fdev[10][0][-1] > 1
    say(f"  P3 feed from below at 1 eV, max|b-1|: n_f=3 {fdev[3][1]:.3f}, n_f=9 {fdev[9][1]:.3f}, n_f=10 {fdev[10][1]:.3f}; b(yrast) n_f=9 {fdev[9][0][-1]:.3f}, n_f=10 {fdev[10][0][-1]:.3f}: {'reproduced' if p3 else 'NOT reproduced'}")
    if not p3: missed.append("P3")
    p4 = dev9[0, 7] < 0.5 and mean9[0, 7] < 0.2
    say(f"  P4 n = 9 two-neighbour estimate at [0,7]: max|b-1| {dev9[0, 7]:.3f} (< 0.5), mean {mean9[0, 7]:.3f} (< 0.2): {'reproduced' if p4 else 'NOT reproduced'}")
    if not p4: missed.append("P4")
    b3, mx3, _ = departure(a3br[sel[:n_res_states], 0], gv[sel]); p5 = DEV["nomix"][0, 7, 8] < 0.5 and abs(b3.max() / b3.min() / P5_3BR_SPREAD - 1) < 0.01
    say(f"  P5 n = 8 without mixing at [0,7]: max|b-1| {DEV['nomix'][0, 7, 8]:.3f} (< 0.5); three-body feed spread at 1 eV {b3.max()/b3.min():.4f} ({P5_3BR_SPREAD}): {'reproduced' if p5 else 'NOT reproduced'}")
    if not p5: missed.append("P5")
    p6 = all(v < 0.5 for v in worst.values())
    say(f"  P6 consequence at the 34 points: largest change {worst['s = 0']:.4f} % (s = 0), {worst['s = s_max']:.4f} % (s = s_max) (< 0.5 %): {'reproduced' if p6 else 'NOT reproduced'}")
    if not p6: missed.append("P6")
    say("\nPREDICTIONS: " + ("P1-P6 all reproduced" if not missed else "not reproduced as written: " + ", ".join(missed)))
    ref_a = d34.max() > 1 or max(DEV["nomix"][i, j, 8] for (i, j) in pts) > 1
    ref_b = max(spd[9].max(), spd[10].max(), (K_up[9].max(axis=0) / K_up[9].min(axis=0)).max()) > 3
    say(f"REFUTER (a factor-two departure, max|b-1| > 1, at n = 9 in R3 or n = 8 in R4 no-mixing, over the 34 points): {'APPEARED' if ref_a else 'did not appear'}"
        f"  [R3 max {d34.max():.3f}, R4 max {max(DEV['nomix'][i, j, 8] for (i, j) in pts):.3f}]")
    say(f"REFUTER of 'close to l-blind' (adjacent transfer spread > 3 at n = 9 or 10 at any Te): {'APPEARED' if ref_b else 'did not appear'}"
        f"  [up n=9 {(K_up[9].max(axis=0) / K_up[9].min(axis=0)).max():.3f}, down n=9 {spd[9].max():.3f}, down n=10 {spd[10].max():.3f}]")
    wcv_up9 = rows_loss and next(r for r in rows_loss if r["direction"] == "up" and r["n"] == 9 and r["t"] == 0); wcv_dn9 = next(r for r in rows_loss if r["direction"] == "down" and r["n"] == 9 and r["t"] == 0)
    bs34 = {n: max(abs(bsaha[n][i, j] - 1) for (i, j) in pts) for n in bsaha}
    say("\nREADING. The chapter's mechanism is refuted and its conclusion survives for a different reason.\n"
        f"  (1) The rates are not l-blind. Adjacent upward transfer out of (n, l) varies across l by a factor that GROWS with n (1.22 at\n"
        f"      n = 2, 2.5-2.8 at n = 9) and downward transfer by 5-6 at n = 7-10, largest at the yrast sublevel in every case (R1); in\n"
        f"      population-weighted terms the scatter at n = 9 and 1 eV is a coefficient of variation {wcv_up9['weighted_cv']:.2f} upward and {wcv_dn9['weighted_cv']:.2f} downward\n"
        f"      (weighted mean |K/K_mean - 1| {wcv_up9['weighted_mad']:.2f} and {wcv_dn9['weighted_mad']:.2f}), so the edge-to-edge factors overstate the typical departure by about\n"
        f"      two, and 27-46 % scatter is still not 'close to l-blind'. The transfer into a shell from a statistically populated neighbour\n"
        f"      is not statistical either: from below, b runs from 0.28 (l' = 0) to 1.66 (yrast) at n_f = 9 and 10; from above, 0.54 to\n"
        f"      1.37 (R2). P1-P3 predicted the opposite trend and are not reproduced.\n"
        f"  (2) The statistical distribution is nevertheless the fixed point wherever the neighbour shells are in mutual Boltzmann ratio,\n"
        f"      because every l-resolved rate obeys detailed balance (G1): b_l - 1 = (theta - 1)[f_dn(l) - mean f_dn] with |f_dn - mean| <= 0.16,\n"
        f"      an identity (checked to {ident:.0e}), so R3's b_l = 1 is not a test of the rates; its content is theta. At the 34 grid points the\n"
        f"      121 combinations occupy, theta_9 - 1 lies within {t34.max():.0e} (R3) and every bundled pair within {max(max(abs(theta_n[n][i, j] - 1) for (i, j) in pts) for n in theta_n):.0e} (R3b), and theta_9 stays\n"
        f"      within 3e-4 of 1 when the bundled shells' radiative rates are set to zero or raised to A(np_total) (R5), so it is not an\n"
        f"      artefact of the bundled block's own assumptions. The stronger statement is the one the pipeline's own ionisation/three-body\n"
        f"      pair supports: the high shells are in Saha-Boltzmann equilibrium there, b_Saha(n) - 1 within {max(bs34.values()):.0e} for n = {min(bs34)}..{max(bs34)} at all 34\n"
        f"      points, which covers n = 11..15 (LTE sublevels are statistical whatever rates exist) and covers the ionisation channel that\n"
        f"      R3's 'inconsistent extreme' column (up to {f34.max():.2f}) left out; a consistent estimate is (b_Saha - 1) x 0.16 x f_blind, of order 1e-6.\n"
        f"      The model's own n = 8 with its mixing block removed is statistical to 1e-4 at every one of the 34 points (R4). P4 and P5\n"
        f"      are reproduced, but not for the reason the predictions assumed (near-statistical feeds and near-blind losses); the\n"
        f"      operative reason is thermalisation of the high shells. Complementarity: |theta_9 - 1| exceeds 0.01 only at low density\n"
        f"      (ne <= 2.7e12, 55 points, 42 of them inside the Te >= 2 eV scope), and at every such point the local-cutoff mixing beats the\n"
        f"      total loss in every bundled shell by at least 341 (bundling_total_loss); no grid point fails both criteria. Where neither\n"
        f"      criterion is met in principle, at the low shells, the collisional distribution is far from statistical: at [0,7] without\n"
        f"      mixing n = 2 departs by 2.3 and n = 3 by 0.22 (R4), though there radiative decay, not collisions, sets the distribution.\n"
        f"  (3) The stake (R5). Had the bundled block been non-statistical at these points, the consequence would not have been small:\n"
        f"      removing the bundled shells' radiative decay moves u_CRE by up to 1.9 % and f_3 - f_4 by 1.2 % at the 34 points, and raising\n"
        f"      every bundled shell's radiative rate to A(np_total) (branching kept) moves them by 21 % and 13 %; tau_slow is unmoved\n"
        f"      (< 0.01 %). P6 predicted < 0.5 % and is not reproduced, for a wrong premise: the 0.19 % radiative share of the bundled loss is\n"
        f"      the grid maximum at 1e12 cm^-3; at the 34 points the share is {fr34.min():.0e} to {fr34.max():.0e}, so the amplification is not 10 but 1e4-1e6. The\n"
        f"      budget explains it: at CRE the net flow into the ground state is a tiny fraction of the recombination source (2e-5 at (9,6)),\n"
        f"      because nearly every recombination is re-ionised from an excited state, and the bundled radiative flux is {rg.min():.2f}-{rg.max():.2f} of that\n"
        f"      net flow (0.43 at (9,6)), landing 23 % in 1s, 13 % in n = 2 and the rest in n >= 3; removing the decays into 1s and n = 2 alone\n"
        f"      accounts for {parts['1s'] + parts['n=2']:+.2f} of the {sum(parts.values()):+.2f} % (the rest re-routes through resolved-shell decays). The effect is on u itself,\n"
        f"      not on a_3, a_4, c_3, c_4 (at (9,6) they move by {max(abs(v) for v in dq_s0_96.values()):.2f} % at most); f_3 - f_4 follows through f = au/(au + c).\n"
        f"  Caveats. R3 omits ionisation, Delta-n >= 2 transfer, feeds from n >= 11 and recombination (l-summed in the model); radiative loss\n"
        f"  out of 9p is l-dependent and uncompensated, shifting b(9p) by {max(abs(rad9p[i, j]) for (i, j) in pts):.0e} at most over the 34 points (R3c; {rad9p[0, 0]:+.3f} at [0,0], where\n"
        f"  mixing covers). In R4, 61-63 % of the n = 8 gain at the 34 points arrives from n = 9 by detailed balance of l-summed CCC rates,\n"
        f"  statistical by construction; R3 and the Saha departures close that loop. Nothing here resolves l inside n = 11..15, where only\n"
        f"  l-summed Vriens-Smeets rates exist; for them the statement rests on R3b and b_Saha. Electron impact only; proton-impact\n"
        f"  n-changing collisions are absent from the model.")

    if a.write:
        out = Path(a.out) if a.out else P("validation/l_blindness"); out.mkdir(parents=True, exist_ok=True)
        hdr = [f"# generated {datetime.now():%Y-%m-%d %H:%M} by {_HERE.name}", f"# interpreter {sys.executable}  numpy {np.__version__}"]
        hdr += [f"# {q.relative_to(ROOT) if q.is_relative_to(ROOT) else q} sha256 {sha(q)}" for q in paths.values()]
        hdr += ["# b_l = (p_l/g_l)/(sum p/sum g), g_l = 2(2l+1); departure from statistical is |b_l - 1|",
                "# R3 headline omits l-summed losses and feeds and is an identity in theta (see .txt); max_abs_dev_with_blind_loss adds the l-blind loss constant from bundling_total_loss without the balancing feed (a bound only)",
                f"# R4 operators: built = L_grid; nomix = L minus the Delta-n = 0 off-diagonal block (K_lmix x ne) with the diagonal restored; local = nomix plus compute_K_lmix(ne_cm3 = ne_local) x ne",
                f"# R5 s_max = A(np_total)/gamma_bundled with A_np_total from verify_bundling_psm20.py ({vb.A_NP_TOTAL_FACTOR} x A(np->1s)); only the bundled radiative columns are scaled"]
        for name, rows in (("l_blindness_loss_spread.csv", rows_loss), ("l_blindness_feed.csv", rows_feed), ("l_blindness_n9.csv", rows_n9),
                           ("l_blindness_resolved.csv", rows_res), ("l_blindness_consequence.csv", rows_cons)):
            with open(out / name, "w") as fh: fh.write("\n".join(hdr) + "\n"); pd.DataFrame(rows).to_csv(fh, index=False)
        with open(out / "l_blindness.txt", "w") as fh: fh.write("\n".join(hdr) + "\n" + "\n".join(log) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}/l_blindness.txt and five csv files")
    return 0


if __name__ == "__main__": sys.exit(main())
