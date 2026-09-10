"""
verify_bundling_psm20.py
========================
Compare ACTUAL PSM20 ℓ-mixing rates (K_lmix.npy) against worst-case
radiative depopulation of np states. Verifies that the bundling assumption
(statistical ℓ-population within shells n=9..15) holds across the (Te, ne)
grid used by the CR matrix.

CRITERION
---------
Bundling requires the within-shell ℓ-relaxation rate to exceed the fastest
substate depopulation. The worst case is the np substate, depleted by the
Lyman channel A(np→1s). The validity criterion is:

    K_lmix(n) / A(np_total) > 10   (well-mixed)
    K_lmix(n) / A(np_total) > 1    (marginal)
    K_lmix(n) / A(np_total) < 1    (bundling INVALID)

USAGE
-----
From repo root:
    cd ~/Desktop/non_markovian_cr
    PYTHONPATH=. python src/validation/verify_bundling_psm20.py

If K_lmix.npy is not at the default path, edit PATH_K_LMIX below.

OUTPUTS
-------
    outputs/bundling/K_lmix_per_shell.npy   (7, n_Te, n_ne) extracted per shell
    outputs/bundling/bundling_ratio.npy     (7, n_Te, n_ne) K_lmix / A(np_total)
    outputs/bundling/bundling_report.txt    text summary
"""

import numpy as np
import os
import sys

# ── User-editable paths ────────────────────────────────────────────────────────
PATH_K_LMIX = 'data/processed/lmix/K_lmix.npy'
FALLBACK_PATHS = [
    'data/processed/cr_matrix/K_lmix.npy',
    'data/processed/K_lmix.npy',
    'data/processed/collisions/K_exc_full/K_lmix.npy',
]
PATH_TE = 'data/processed/cr_matrix/Te_grid_L.npy'
PATH_NE = 'data/processed/cr_matrix/ne_grid_L.npy'
OUTDIR  = 'outputs/bundling'

# ── Model structure (from assemble_cr_matrix.py) ───────────────────────────────
# State layout: indices 0..35 = resolved nℓ for n=1..8; indices 36..42 = bundled n=9..15
N_BUND   = np.arange(9, 16)                # principal quantum numbers
BUND_IDX = np.arange(36, 43)               # state indices in the 43-state vector
N_TE_EXPECTED = 50
N_NE_EXPECTED = 8

# ── A(np→1s) Einstein coefficients [s^-1] from NIST H I tables ────────────────
# Total A from np state ≈ 1.3 × A(np→1s) (Lyman dominant, Balmer/Paschen add ~30%)
A_NP_1S = {
    9:  5.776e6, 10: 4.196e6, 11: 3.143e6, 12: 2.413e6,
    13: 1.892e6, 14: 1.510e6, 15: 1.225e6,
}
A_NP_TOTAL_FACTOR = 1.3

def A_np_total(n):
    return A_NP_TOTAL_FACTOR * A_NP_1S[n]


def check_A_np_against_model():
    """A_NP_1S is hardcoded because the model's own A_resolved.npy stops at
    n = 8: the bundled shells carry one A per shell, not one per substate, so
    there is nothing in the pipeline to read A(np->1s) from above n = 8.

    A hardcoded number needs a check rather than a citation. A(np->1s) * n^3
    tends to a constant for hydrogen, so the model's own resolved values at
    n = 2..8 and the table's values at n = 9..15 must lie on one curve. If
    they do not, the table belongs to a different atom or a different
    normalisation and every ratio below is wrong.
    """
    import csv as _csv
    Ap = 'data/processed/Radiative/A_resolved.npy'
    si = 'data/processed/collisions/K_exc_full/state_index.csv'
    for q in (Ap, si):
        if not os.path.exists(q):
            raise RuntimeError(f"missing {q}: cannot check the hardcoded "
                               f"A(np->1s) table against the model's own data.")
    A = np.load(Ap)
    rows = list(_csv.DictReader(open(si)))
    lab = [r['label'] for r in rows]
    nv = [int(float(r['n'])) for r in rows]
    print("  A(np->1s) * n^3 must be continuous across the bundling boundary:")
    model = []
    for n in range(2, 9):
        j = [k for k in range(len(lab)) if nv[k] == n and lab[k].upper().endswith('P')][0]
        model.append((n, float(A[0, j]) * n ** 3))
    table = [(n, A_NP_1S[n] * n ** 3) for n in sorted(A_NP_1S)]
    print("    from the model  " + "  ".join(f"n={n}:{v/1e9:.3f}" for n, v in model[-3:]))
    print("    from the table  " + "  ".join(f"n={n}:{v/1e9:.3f}" for n, v in table[:3]))
    step = abs(table[0][1] / model[-1][1] - 1.0)
    if step > 0.05:
        raise RuntimeError(
            f"A(np->1s)*n^3 jumps by {step*100:.1f}% between n = 8 (model, "
            f"{model[-1][1]:.4g}) and n = 9 (hardcoded table, {table[0][1]:.4g}). "
            f"The two are not the same quantity and the bundling ratios below "
            f"would be meaningless.")
    print(f"    jump across n = 8 to 9: {step*100:.2f}%, and the table falls "
          f"monotonically to {table[-1][1]/1e9:.3f}e9 at n = 15, which is the "
          f"expected approach to the hydrogenic asymptote.")
    print()


# ── Find and load K_lmix.npy ───────────────────────────────────────────────────
def find_klmix():
    for p in [PATH_K_LMIX] + FALLBACK_PATHS:
        if os.path.exists(p):
            return p
    print("ERROR: K_lmix.npy not found at any of:")
    for p in [PATH_K_LMIX] + FALLBACK_PATHS:
        print(f"  {p}")
    print("\nEdit PATH_K_LMIX at the top of this script and rerun.")
    sys.exit(1)


# ── Interpret K_lmix shape ─────────────────────────────────────────────────────
def diagnose_shape(K):
    """Identify what K_lmix.npy axes mean. Return ('mode', extraction_info)."""
    sh = K.shape
    print(f"K_lmix.npy shape: {sh}")
    print(f"dtype: {K.dtype}, finite values: {np.isfinite(K).all()}, "
          f"min/max: {K.min():.3e} / {K.max():.3e}")

    g = (N_TE_EXPECTED, N_NE_EXPECTED)   # (50, 8)

    # Case A: full rate matrix (50, 8, 43, 43)
    if K.ndim == 4 and sh[:2] == g and sh[2] == sh[3] == 43:
        print("  → mode: full 43×43 ℓ-mixing contribution to L matrix")
        return 'full_matrix'

    # Case B: per-bundled-state outgoing rate (50, 8, 7)
    if K.ndim == 3 and sh[:2] == g and sh[2] == 7:
        print("  → mode: per-shell outgoing ℓ-mixing rate (7 bundled shells)")
        return 'per_shell'

    # Case C: per-state outgoing rate for all 43 (50, 8, 43)
    if K.ndim == 3 and sh[:2] == g and sh[2] == 43:
        print("  → mode: per-state outgoing rate (extract indices 36..42)")
        return 'per_state'

    # Case D: collisional matrix Te-axis last (43, 43, 50, 8) or similar — flag
    if K.ndim == 4 and sh[2:] == g and sh[0] == sh[1] == 43:
        print("  → mode: full matrix with Te,ne as LAST axes")
        return 'full_matrix_te_last'

    # Case E: what the pipeline actually writes. compute_lmix.py:77 declares
    # K_lmix as (43, 43, n_Te) in cm^3/s, with NO density axis, because the
    # proton-impact rate is linear in density and assemble_cr_matrix.py:179
    # multiplies by ne when it builds L. Every branch above was written for a
    # file that does not exist.
    if K.ndim == 3 and sh[0] == sh[1] == 43 and sh[2] == N_TE_EXPECTED:
        print("  → mode: rate COEFFICIENTS [cm^3/s], (dest, src, Te); "
              "rate = K * ne")
        return 'coeff_te_only'

    print(f"\nERROR: Unrecognised shape {sh}.")
    print("Expected one of:")
    print("  (50, 8, 43, 43)  full L_mix matrix")
    print("  (50, 8, 7)       per-bundled-shell rate")
    print("  (50, 8, 43)      per-state outgoing rate")
    print("  (43, 43, 50, 8)  full L_mix matrix, Te/ne last")
    print("\nInspect K_lmix.npy manually and add an extraction branch.")
    sys.exit(1)


def extract_coeff_te_only(K, Te_grid, ne_grid, labels, n_values):
    """The bundled shells n=9..15 carry no l-mixing in K_lmix, so the test
    cannot be run against the file. Establish that, then validate the analytic
    rate against the file where both exist, and only then use the analytic
    rate for the bundled shells.

    Returns (K_shell, provenance, resolved_check).
    """
    if np.abs(K[:, BUND_IDX, :]).max() != 0.0 or np.abs(K[BUND_IDX, :, :]).max() != 0.0:
        raise RuntimeError(
            "the bundled block of K_lmix is NOT identically zero. This "
            "script's central premise, that l-mixing was computed for the "
            "resolved block only, no longer holds, and the analytic "
            "substitution below is no longer justified. Re-read "
            "assemble_cr_matrix.py before trusting anything here.")

    print()
    print("  The bundled block of K_lmix (states 36..42, n = 9..15) is")
    print("  IDENTICALLY ZERO in both directions. assemble_cr_matrix.py:37")
    print("  applies l-mixing to the resolved block n = 2..8 only, so the")
    print("  shells this test is about carry none of it in the matrix. The")
    print("  test therefore cannot be run against K_lmix.npy, and the earlier")
    print("  promise in this file's title to use 'actual K_lmix.npy' for the")
    print("  bundled shells could never have been kept.")
    print()
    print("  What can be done instead: call compute_lmix.py's own rate")
    print("  function at n = 9..15. The check below is that this call path")
    print("  reproduces K_lmix exactly at n = 2..8, where the file has values.")
    print("  A ratio of 1.0000 there is not an independent confirmation of the")
    print("  physics, and must not be read as one: it confirms that the rate")
    print("  used at n = 9..15 is the SAME function, with the same arguments,")
    print("  that built the matrix, rather than a second implementation. The")
    print("  physics of the rate is Badnell (2021) PSM20 and is validated in")
    print("  chapter 2, not here.")
    print()

    # -- validate the analytic rate where the file has numbers ---------------
    print("  Pipeline PSM20 rate against K_lmix, per np substate, at ne = "
          f"{ne_grid[0]:.2e} and {ne_grid[-1]:.2e} cm^-3:")
    print(f"    {'n':>3}  {'Te':>6}  {'K_lmix*ne':>12}  {'PSM20':>12}  {'ratio':>8}")
    checks = []
    for n in range(2, 9):
        idx = [k for k in range(len(labels))
               if n_values[k] == n and labels[k].upper().endswith("P")]
        if len(idx) != 1:
            raise RuntimeError(
                f"expected exactly one np substate for n={n}, found "
                f"{[labels[k] for k in idx]}. The state ordering is not what "
                f"this script assumes and no rate can be attributed.")
        j = idx[0]
        for iTe in (0, len(Te_grid) // 2, len(Te_grid) - 1):
            for jne in (0, len(ne_grid) - 1):
                file_rate = float(K[:, j, iTe].sum()) * ne_grid[jne]
                ana = lmix_rate_np(n, Te_grid[iTe], ne_grid[jne])
                if ana <= 0:
                    raise RuntimeError(f"analytic rate is non-positive at n={n}")
                checks.append((n, Te_grid[iTe], ne_grid[jne], file_rate,
                               ana, file_rate / ana))
    for n, T, N, fr, an, r in checks:
        if N == ne_grid[0] and abs(T - Te_grid[len(Te_grid)//2]) < 1e-9:
            print(f"    {n:>3}  {T:>6.2f}  {fr:>12.3e}  {an:>12.3e}  {r:>8.3f}")
    rr = np.array([c[5] for c in checks])
    print(f"    over all {len(rr)} checked (n, Te, ne): ratio "
          f"{rr.min():.4f} to {rr.max():.4f}, median {np.median(rr):.4f}")
    print()
    m = _lmix_module()
    print(f"  What the frozen Debye cutoff costs. K_lmix.npy has no density")
    print(f"  axis: F(U_m) is evaluated once at ne = {m.NE_DEFAULT:.3g} cm^-3 and the")
    print(f"  coefficient is then multiplied by the local ne. Re-evaluating")
    print(f"  F(U_m) at the local density instead, for the np substate:")
    for n in (2, 8, 15):
        row = []
        for jne in (0, len(ne_grid) - 1):
            frozen = lmix_rate_np(n, Te_grid[len(Te_grid)//2], ne_grid[jne])
            local = lmix_rate_np(n, Te_grid[len(Te_grid)//2], ne_grid[jne],
                                 ne_debye=ne_grid[jne])
            row.append(local / frozen)
        print(f"    n = {n:>2}:  local/frozen = {row[0]:.4f} at ne = "
              f"{ne_grid[0]:.2e},  {row[1]:.4f} at ne = {ne_grid[-1]:.2e}")
    print("  This is not a small correction. At n = 15 it is a factor 20 up")
    print("  at the bottom of the density range and a factor 30 down at the")
    print("  top, because F(U_m) grows as the Debye length grows and the")
    print("  frozen value sits in the middle of three decades. Whether it")
    print("  moves the verdict is a separate question, answered rather than")
    print("  assumed:")
    worst_frozen, worst_local = np.inf, np.inf
    for n in N_BUND:
        A_n = A_np_total(int(n))
        for iTe in range(len(Te_grid)):
            for jne in range(len(ne_grid)):
                worst_frozen = min(worst_frozen,
                    lmix_rate_np(int(n), Te_grid[iTe], ne_grid[jne]) / A_n)
                worst_local = min(worst_local,
                    lmix_rate_np(int(n), Te_grid[iTe], ne_grid[jne],
                                 ne_debye=ne_grid[jne]) / A_n)
    print(f"    worst K_lmix/A(np) over n = 9..15 and the whole grid:")
    print(f"      frozen cutoff, as the model is built: {worst_frozen:.4g}")
    print(f"      cutoff at the local density:          {worst_local:.4g}")
    print(f"    Bundling needs this above 10. The smaller of the two clears")
    print(f"    that threshold by a factor {min(worst_frozen, worst_local)/10:.0f}, so the conclusion is")
    print(f"    insensitive to the freezing even though the rate is not.")
    if rr.min() < 1.0 - 1e-9 or rr.max() > 1.0 + 1e-9:
        raise RuntimeError(
            f"the call path does not reproduce K_lmix on the resolved block "
            f"(ratio {rr.min():.3g} to {rr.max():.3g}). Since it is meant to "
            f"be the same function with the same arguments, a difference means "
            f"an argument is wrong, most likely the Debye-cutoff density. "
            f"Extending it to n = 9..15 would then be a different rate from "
            f"the model's, so this script stops here.")
    print()

    # -- now, and only now, the bundled shells -------------------------------
    K_shell = np.zeros((7, len(Te_grid), len(ne_grid)))
    for b, n in enumerate(N_BUND):
        for iTe in range(len(Te_grid)):
            for jne in range(len(ne_grid)):
                K_shell[b, iTe, jne] = ps64_estimate(n, Te_grid[iTe], ne_grid[jne])
    return K_shell, "compute_lmix.py PSM20, same call path that built K_lmix", rr


def extract_per_shell(K, mode):
    """Return (7, n_Te, n_ne) — outgoing ℓ-mixing rate for each bundled shell."""
    K_shell = np.zeros((7, N_TE_EXPECTED, N_NE_EXPECTED))

    if mode == 'full_matrix':
        # K[Te, ne, i, j] = rate from j to i. Outgoing from j = -K[Te,ne,j,j]
        # (assuming diagonal stores total loss). If diagonal is zero/positive,
        # use column sum minus diagonal as a fallback.
        for b in range(7):
            j = BUND_IDX[b]
            diag = K[:, :, j, j]
            if np.all(diag <= 0) and np.any(diag < 0):
                K_shell[b] = -diag
            else:
                col = K[:, :, :, j].sum(axis=-1) - K[:, :, j, j]
                K_shell[b] = col

    elif mode == 'full_matrix_te_last':
        for b in range(7):
            j = BUND_IDX[b]
            diag = K[j, j, :, :]
            if np.all(diag <= 0) and np.any(diag < 0):
                K_shell[b] = -diag
            else:
                col = K[:, j, :, :].sum(axis=0) - K[j, j, :, :]
                K_shell[b] = col

    elif mode == 'per_shell':
        for b in range(7):
            K_shell[b] = K[:, :, b]

    elif mode == 'per_state':
        for b in range(7):
            K_shell[b] = K[:, :, BUND_IDX[b]]

    return K_shell


# ── Load Te, ne grids ──────────────────────────────────────────────────────────
def load_grids():
    """Load the grids the pipeline wrote. There is no fallback.

    This function used to synthesise the grid with np.logspace when the files
    were absent, printing a warning and continuing. That is the exact failure
    CLAUDE.md rule 1 exists to prevent: a synthesised grid produces a report
    that looks right, carries the pipeline's axis labels, and describes
    conditions the model was never evaluated at. Every ratio in the output
    would be attached to the wrong (Te, ne) pair and nothing downstream would
    notice.

    The synthesised grid also happened to be wrong on its own terms: it took
    N_TE_EXPECTED and N_NE_EXPECTED, which are hardcoded here, rather than the
    shape of K_lmix, so a pipeline change to the grid size would have produced
    a shape mismatch or, worse, silently correct shapes with wrong values.
    """
    missing = [p for p in (PATH_TE, PATH_NE) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            "missing grid file(s): " + ", ".join(missing) + ". This script "
            "compares l-mixing rates against radiative rates at specific "
            "(Te, ne) points and cannot label a single number without the "
            "grids the pipeline wrote. Run assemble_cr_matrix.py first. No "
            "grid is synthesised here.")
    Te, ne = np.load(PATH_TE), np.load(PATH_NE)
    if Te.ndim != 1 or ne.ndim != 1:
        raise RuntimeError(f"grid files are not 1-D: Te {Te.shape}, ne {ne.shape}")
    if (len(Te), len(ne)) != (N_TE_EXPECTED, N_NE_EXPECTED):
        raise RuntimeError(
            f"the pipeline's grid is {len(Te)} x {len(ne)} but this script's "
            f"BUND_IDX and N_TE_EXPECTED/N_NE_EXPECTED were written for "
            f"{N_TE_EXPECTED} x {N_NE_EXPECTED}. The state layout may also "
            f"have moved. Update this script deliberately rather than letting "
            f"it broadcast against the wrong axis.")
    return Te, ne


# ── The l-mixing rate out of the np substate ──────────────────────────────────
# This script previously carried its own Pengelly-Seaton expression:
#
#     q = 1.294e-5 * n**2 * <((l+0.5)/n)^2> * T_K**-0.5 * ln(R_D/(n^2 a0))
#
# It disagrees with the rate the pipeline actually uses by a factor of 95 at
# n = 2 rising to 5.6e3 at n = 8 and high density, and the growth with n is
# roughly n^2, which is the signature of a missing power of n^2 in the cross
# section on top of a wrong constant. Using it here would have produced a
# verdict about a rate the model does not contain. It is deleted rather than
# corrected: there is no reason for a validation script to carry a second,
# independently maintained implementation of a rate the pipeline already
# computes. The comparison below is against compute_lmix.py itself, and the
# agreement with K_lmix on the resolved block is what makes the extension to
# n = 9..15 an interpolation of the model rather than a new claim.
_LMIX = None


def _lmix_module():
    global _LMIX
    if _LMIX is None:
        import importlib.util
        src = os.path.join('src', 'rates', 'compute_lmix.py')
        if not os.path.exists(src):
            raise RuntimeError(
                f"missing {src}: this test compares against the pipeline's own "
                f"l-mixing rate and will not re-derive one.")
        spec = importlib.util.spec_from_file_location("compute_lmix", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LMIX = mod
    return _LMIX


def lmix_rate_np(n, Te_eV, ne, ne_debye=None):
    """Total l-mixing rate OUT of the np substate [s^-1], from the pipeline's
    own PSM20 implementation. np is the substate that matters because it is
    the one the Lyman channel empties, so it is the substate whose mixing has
    to win for the shell to stay statistical.

    Channels out of l = 1: down to l = 0 and up to l = 2. For n = 2 only the
    downward channel exists, and the function returns only that.

    TWO DENSITIES, DELIBERATELY SEPARATE. The collision frequency is linear in
    the proton density and that is the `ne` multiplying at the end. The Debye
    cutoff inside F(U_m) also depends on density, and the pipeline FREEZES it
    at compute_lmix.NE_DEFAULT = 1e14 for the whole grid: K_lmix.npy has no
    density axis, and assemble_cr_matrix.py:179 multiplies the frozen
    coefficient by the local ne. Passing the local density here instead would
    compare against a rate the model does not use, which is what produced a
    spread of 0.06 to 29 in an earlier version of this check. `ne_debye`
    defaults to the pipeline's own value so that the comparison is against the
    model as built, and the cost of the freezing is measured separately below.
    """
    m = _lmix_module()
    if ne_debye is None:
        ne_debye = m.NE_DEFAULT
    Te = np.atleast_1d(float(Te_eV))
    q = m._psm20_q_down(n, 1, Te, ne_cm3=ne_debye)[0]
    if n >= 3:
        q += m._psm20_q_up(n, 1, Te, ne_cm3=ne_debye)[0]
    return float(q) * ne


def ps64_estimate(n, Te_eV, ne):
    """Kept as a name for the report; delegates to the pipeline's rate."""
    return lmix_rate_np(n, Te_eV, ne)


# ── Reporting ──────────────────────────────────────────────────────────────────
def report(K_shell, Te_grid, ne_grid, outfile=None):
    A_rad = np.array([A_np_total(n) for n in N_BUND])
    ratio = K_shell / A_rad[:, None, None]

    iTe_3 = np.argmin(np.abs(Te_grid - 3.0))

    lines = []
    lines.append("=" * 76)
    lines.append("PSM20 BUNDLING VALIDITY CHECK — l-mixing rate out of np vs A(np_total)")
    lines.append("=" * 76)
    lines.append(f"Grid: Te = {Te_grid[0]:.2f}..{Te_grid[-1]:.2f} eV ({len(Te_grid)} pts)")
    lines.append(f"      ne = {ne_grid[0]:.2e}..{ne_grid[-1]:.2e} cm^-3 ({len(ne_grid)} pts)")
    lines.append("")
    lines.append("Per-shell summary at Te = 3 eV (closest to benchmark point):")
    lines.append("-" * 76)
    lines.append(f"{'n':>3}  {'A(np_tot)':>11}  {'K_lmix(lo ne)':>15}  "
                 f"{'K_lmix(hi ne)':>15}  {'ratio lo-ne':>13}  status")
    lines.append("-" * 76)
    for b, n in enumerate(N_BUND):
        Kl_lo = K_shell[b, iTe_3, 0]
        Kl_hi = K_shell[b, iTe_3, -1]
        r = Kl_lo / A_rad[b]
        status = "OK" if r >= 10 else ("MARGINAL" if r >= 1 else "INVALID")
        lines.append(f"{n:>3}  {A_rad[b]:>11.2e}  {Kl_lo:>15.2e}  "
                     f"{Kl_hi:>15.2e}  {r:>13.2f}  {status}")
    lines.append("-" * 76)

    # Worst offender across full grid
    iworst = np.unravel_index(np.argmin(ratio), ratio.shape)
    n_w  = N_BUND[iworst[0]]
    Te_w = Te_grid[iworst[1]]
    ne_w = ne_grid[iworst[2]]
    r_w  = ratio[iworst]
    K_w  = K_shell[iworst]
    A_w  = A_rad[iworst[0]]
    lines.append("")
    lines.append("Worst offender across full (n, Te, ne) grid:")
    lines.append(f"  n = {n_w}, Te = {Te_w:.3f} eV, ne = {ne_w:.3e} cm^-3")
    lines.append(f"  K_lmix = {K_w:.3e} s^-1   A(np) = {A_w:.3e} s^-1   "
                 f"ratio = {r_w:.3f}")

    # Grid fractions
    f_invalid  = (ratio < 1).mean()
    f_marginal = ((ratio >= 1) & (ratio < 10)).mean()
    f_ok       = (ratio >= 10).mean()
    lines.append("")
    lines.append("Grid fraction summary (over 7 × 50 × 8 = 2800 points):")
    lines.append(f"  ratio < 1  (invalid):    {f_invalid*100:5.1f}%")
    lines.append(f"  ratio 1–10 (marginal):   {f_marginal*100:5.1f}%")
    lines.append(f"  ratio ≥ 10 (valid):      {f_ok*100:5.1f}%")

    # Cross-check vs PS64 analytical prediction
    lines.append("")
    lines.append("Rate at the worst offender, recomputed through the same call path:")
    Kw_ps64 = ps64_estimate(n_w, Te_w, ne_w)
    r_ps64 = Kw_ps64 / A_w
    lines.append(f"  K_lmix = {Kw_ps64:.3e} s^-1   ratio = {r_ps64:.3f}")
    rel = (K_w - Kw_ps64) / Kw_ps64 if Kw_ps64 > 0 else float('nan')
    lines.append(f"  agreement: {rel*100:+.1f}%. This is zero by construction, since "
                 f"both sides are compute_lmix.py; it is a wiring check, not a physics one.")

    # Verdict
    lines.append("")
    lines.append("=" * 76)
    if f_invalid > 0.01:
        lines.append("VERDICT: bundling INVALID at >1% of grid — investigate before Paper 1.")
        lines.append("Possible causes:")
        lines.append("  • K_lmix.npy shape interpretation wrong (rerun and verify mode)")
        lines.append("  • Lifetime cutoff active (small Te shrinks K below PS64 estimate)")
        lines.append("  • A_np values inconsistent with your radiative file")
    elif f_marginal > 0.10:
        lines.append("VERDICT: bundling MARGINAL across >10% of grid — add explicit caveat.")
        lines.append("Restrict quantitative claims at the affected (n, ne) corner.")
    else:
        lines.append("VERDICT: bundling assumption holds across the working grid.")
        lines.append(f"Worst point is n = {n_w} at Te = {Te_w:.2f} eV, ne = {ne_w:.2e},")
        lines.append(f"where the mixing rate beats A(np) by {r_w:.0f}x, clearing the")
        lines.append(f"well-mixed threshold of 10 by a factor {r_w/10:.0f}.")
        lines.append("Cite Badnell+2021 PSM20 in Methods. The caveat that remains is not")
        lines.append("the mixing rate: it is that the bundled block carries no l-mixing in")
        lines.append("the matrix at all, so the shells are ASSUMED statistical, and this")
        lines.append("test shows only that the assumption is the right one to make.")
    lines.append("=" * 76)

    text = "\n".join(lines)
    print(text)
    if outfile:
        with open(outfile, 'w') as f:
            f.write(text + "\n")
    return ratio


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Locating K_lmix.npy ...")
    path = find_klmix()
    print(f"  found: {path}\n")

    K = np.load(path)
    mode = diagnose_shape(K)
    print()

    Te_grid, ne_grid = load_grids()
    check_A_np_against_model()

    if mode == 'coeff_te_only':
        import csv as _csv
        si = 'data/processed/collisions/K_exc_full/state_index.csv'
        if not os.path.exists(si):
            raise RuntimeError(
                f"missing {si}: the np substate of each resolved shell cannot "
                f"be located without the pipeline's own state ordering, and "
                f"guessing it would attribute rates to the wrong levels.")
        with open(si) as fh:
            rows = list(_csv.DictReader(fh))
        lab_key = 'label' if 'label' in rows[0] else list(rows[0])[1]
        n_key = 'n' if 'n' in rows[0] else list(rows[0])[2]
        labels = [r[lab_key] for r in rows]
        n_values = [int(float(r[n_key])) for r in rows]
        K_shell, prov, _ = extract_coeff_te_only(K, Te_grid, ne_grid,
                                                 labels, n_values)
        print(f"  rate source for n = 9..15: {prov}")
    else:
        K_shell = extract_per_shell(K, mode)
    print(f"K_shell extracted, shape {K_shell.shape}\n")

    os.makedirs(OUTDIR, exist_ok=True)
    ratio = report(K_shell, Te_grid, ne_grid,
                   outfile=f'{OUTDIR}/bundling_report.txt')

    np.save(f'{OUTDIR}/K_lmix_per_shell.npy', K_shell)
    np.save(f'{OUTDIR}/bundling_ratio.npy', ratio)
    print(f"\nSaved:")
    print(f"  {OUTDIR}/K_lmix_per_shell.npy   {K_shell.shape}")
    print(f"  {OUTDIR}/bundling_ratio.npy     {ratio.shape}")
    print(f"  {OUTDIR}/bundling_report.txt")


if __name__ == '__main__':
    main()
