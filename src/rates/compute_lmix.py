"""
compute_lmix.py
===============
Proton-impact ℓ-mixing rate coefficients for the hydrogen CR model.

Implements the Badnell et al. (2021) PSM20 Debye-cutoff formula for
adjacent (Δℓ = ±1) ℓ-changing collisions within a fixed n-shell.

PHYSICS:
  Process: H(n,ℓ) + p → H(n,ℓ±1) + p   (intra-shell, Δn = 0)

  Downward rate (ℓ → ℓ-1, high-ℓ to low-ℓ):
    q_down(n,ℓ→ℓ-1; T) from PSM20 Debye formula

  Upward rate (ℓ → ℓ+1, low-ℓ to high-ℓ):
    q_up(n,ℓ→ℓ+1; T) = q_down(n,ℓ+1→ℓ; T) × (2ℓ+3)/(2ℓ+1)
    (reciprocity, Badnell 2021 eq. recommendation)

PSM20 DEBYE-CUTOFF FORMULA (Badnell 2021, Eq. 9):
  For the "downward" transition (ℓ_> = max(ℓ,ℓ')):

    D_ji = 6 n² ℓ_> (n² - ℓ_>²) / z²         [dipole coupling, z = proton charge]

    q_ji = (a_0³/τ_0) × sqrt(π μ I_H / kT) × (D_ji / ω_ℓ) × F(U_m)

    F(U_m) = (sqrt(π)/2) × U_m^{-3/2} × erf(sqrt(U_m))
             - exp(-U_m)/U_m
             + E1(U_m)

  with U_m = E_min / kT, and E_min from Badnell Eqs. 5+8:
    E_min = a_0² × μ × I_H × D_ji / (2 × P1 × ω_ℓ × R_c²)
    R_c   = λ_D (Debye length, two-species T_i = T_e assumption)
    P1    = small-impact matching probability (PS64/PSM convention, ~0.5)

  F(U_m) IS the Coulomb-logarithm factor of the PSM20 rate coefficient.
  Limits: F(U_m -> 0) diverges logarithmically (dense-plasma / small-U_m
  regime); F(U_m -> infinity) -> 0 (screening kills the collision).
  At ITER divertor conditions (T_e ~ 3 eV, n_e ~ 1e14 cm^-3) F(U_m) is
  typically 3-7 depending on (n, ℓ_>), NOT 1.

  where:
    μ   = reduced mass ratio = m_H / m_e ≈ 918.0764  (proton/electron mass)
    I_H = 13.605693 eV   (Rydberg energy)
    ω_ℓ = 2ℓ+1           (statistical weight of lower state, ℓ = ℓ_> - 1)
    a_0 = Bohr radius    = 5.29177e-9 cm
    τ_0 = a_0 / (α c)   = 2.4189e-17 s  (atomic unit of time)
    z   = 1              (proton charge)
    kT  in eV

HISTORICAL NOTE (see derivation_04b §7.2 for full audit):
  A previous version of this file set F(U_m) = 1, justified as "low-density
  limit U_m -> 0, F -> 1". This was doubly incorrect: at ITER density the
  true U_m is small (order 1e-3 to 1e-1), and F(U_m -> 0) diverges
  logarithmically, not to 1. The old code under-counted rates by ×3-7.
  τ_relax is insensitive (<0.2%) to this correction because ℓ-mixing is
  already the fastest process in the CR spectrum, but absolute ℓ-resolved
  populations and line ratios do depend on the correction. Corrected here.

APPROXIMATIONS:
  - T_i = T_e  (proton temperature = electron temperature)
  - n_p = n_e  (quasi-neutrality)
  - Δℓ = ±1 only (adjacent sublevels, Badnell minimal implementation)
  - n=1 has no ℓ-mixing (only 1s state)
  - Bundled n=9–15: statistical ℓ-equilibrium assumed (not computed here)

STATE INDEXING (must match assemble_cr_matrix.py):
  Index 0       : 1s  (n=1, ℓ=0)
  Index 1–2     : 2s, 2p
  Index 3–5     : 3s, 3p, 3d
  Index 6–9     : 4s, 4p, 4d, 4f
  Index 10–14   : 5s, 5p, 5d, 5f, 5g
  Index 15–20   : 6s, 6p, 6d, 6f, 6g, 6h
  Index 21–27   : 7s, 7p, 7d, 7f, 7g, 7h, 7i
  Index 28–35   : 8s, 8p, 8d, 8f, 8g, 8h, 8i, 8k

OUTPUT:
  K_lmix.npy  : (43, 43, 50) float64  [cm³/s]
    K_lmix[i, j, Te_idx] = rate coefficient for ℓ-mixing j→i
    Non-zero only for same-n adjacent-ℓ pairs within n=2–8
    Upper triangle: downward (higher ℓ → lower ℓ)
    Lower triangle: upward   (lower ℓ → higher ℓ)

  Multiply by n_p (= n_e) to get rate [s⁻¹].

REFERENCE:
  Badnell, N. R. et al. (2021). MNRAS, 507, 2922–2929.
  DOI: 10.1093/mnras/stab2086
"""

import numpy as np
from scipy.special import erf, exp1   # erf and E1 (exponential integral)
import os

# ── Physical constants ──────────────────────────────────────────────────────────
A0       = 5.291772e-9    # Bohr radius [cm]
TAU0     = 2.418884e-17   # atomic unit of time [s]  = a0 / (alpha*c)
IH       = 13.605693      # Rydberg energy [eV]
MU       = 918.07635      # m_p / m_e  (proton/electron mass ratio)
A0_3     = A0**3          # [cm³]

# ── Temperature grid (must match assemble_cr_matrix.py) ────────────────────────
TE_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 50)   # eV

# ── State indexing ──────────────────────────────────────────────────────────────
# Build (n, ℓ) lookup tables for the 36 resolved states (indices 0–35)

def _build_state_tables():
    """Return arrays nl_to_idx[(n,ℓ)] and idx_to_nl[idx] = (n,ℓ)."""
    nl_to_idx = {}
    idx_to_nl = []
    idx = 0
    for n in range(1, 9):          # n = 1..8
        for ell in range(n):       # ℓ = 0..n-1
            nl_to_idx[(n, ell)] = idx
            idx_to_nl.append((n, ell))
            idx += 1
    assert idx == 36
    return nl_to_idx, idx_to_nl

NL_TO_IDX, IDX_TO_NL = _build_state_tables()


# ── PSM20 Debye-cutoff formula ──────────────────────────────────────────────────
def _psm20_D(n, ell_upper):
    """
    Dipole coupling factor D_ji for (n, ℓ_upper) → (n, ℓ_upper - 1).

    D_ji = 6 n² ℓ_> (n² - ℓ_>²) / z²
    where ℓ_> = max(ℓ, ℓ') = ℓ_upper for downward transition.
    z = 1 (proton charge).

    Parameters
    ----------
    n         : int  principal quantum number
    ell_upper : int  larger ℓ of the pair  (= ℓ_> = max(ℓ, ℓ'))

    Returns
    -------
    D : float  [dimensionless]
    """
    l = ell_upper
    return 6.0 * n**2 * l * (n**2 - l**2)


# Default electron density used for the Debye cutoff when computing the
# density-dependent rate coefficient. The scale of the ITER divertor grid
# (10^12 – 10^15 cm^-3) makes any single value a compromise; F(U_m) is only
# logarithmically sensitive to ne, so this is a mild dependence. For strict
# density-resolved rates, pass ne explicitly.
NE_DEFAULT = 1.0e14   # cm^-3   (mid-grid; F(U_m) varies ~30% across the full grid)

# Small-impact-parameter matching probability (PS64/PSM convention).
# Enters U_m through Badnell Eq. 5+8; sets the R_1 matching radius.
P1_DEFAULT = 0.5


def _psm20_Um(n, ell_upper, Te_arr, ne_cm3, P1=P1_DEFAULT):
    """
    Compute U_m = E_min / kT_e for the PSM20 Debye-cutoff form.

    Badnell (2021) Eq. 8:  E_min = a0^2 * mu * I_H * D_ji / (2 * P1 * omega_l * R_c^2)
    with R_c = lambda_D (Debye length; Eq. 10 setting Debye limit).

    Working in atomic units for lengths (R_c/a0 dimensionless):
        E_min / I_H = mu * D_ji / (2 * P1 * omega_l * (R_c/a0)^2)

    Parameters
    ----------
    n         : int
    ell_upper : int    ℓ_> = max(ℓ, ℓ')
    Te_arr    : (N,) float [eV]
    ne_cm3    : float [cm^-3]
    P1        : float, matching probability (Badnell Eq. 4)

    Returns
    -------
    U_m : (N,) float, dimensionless
    """
    ell_lo = ell_upper - 1
    omega  = 2 * ell_lo + 1                          # weight of LOWER state
    D      = _psm20_D(n, ell_upper)

    # Two-species Debye length (T_e = T_i, quasi-neutral).
    # In SI, then converted to cm and finally to atomic units (a0).
    # lambda_D^2 = eps0 * kT / (2 * ne * e^2)   [two species, T_i = T_e]
    # In eV units and cm^-3:
    #   lambda_D [cm] = 7.4340e2 * sqrt(T_e[eV] / (2 * ne[cm^-3]))
    #   (numerical prefactor from eps0, k_B, e in SI, converting to cm)
    # Derivation: sqrt(eps0 * eV / e^2) = 7.4340e2 cm when using n [cm^-3]
    lamD_cm = 7.4340e2 * np.sqrt(Te_arr / (2.0 * ne_cm3))
    Rc_over_a0 = lamD_cm / A0

    # E_min / kT_e = (mu * D_ji / (2 * P1 * omega * Rc_a0^2)) * (I_H / kT_e)
    Um = (MU * D / (2.0 * P1 * omega * Rc_over_a0**2)) * (IH / Te_arr)
    return Um


def _psm20_F(Um):
    """
    Badnell (2021) Eq. 9 bracket F(U_m):

        F(U_m) = (sqrt(pi)/2) * U_m^{-3/2} * erf(sqrt(U_m))
                 - exp(-U_m) / U_m
                 + E1(U_m)

    This is the Coulomb-logarithm factor of the PSM20 Debye rate coefficient.

    Limits:
      U_m -> 0   : F -> +infinity (logarithmic) ~ ln(R_c/R_1) at leading order
      U_m -> inf : F -> 0   (screening kills the collision)

    Notes
    -----
    In §7.2 of derivation_04b, this factor was verified against Badnell's
    published Eq. 9 (source PDF). At ITER divertor densities (T ~ 3 eV,
    ne ~ 1e14 cm^-3) it takes values ~3 to ~7 depending on (n, ℓ_>), NOT 1.
    Setting F = 1 as a "low-density limit" is INCORRECT — the low-density
    limit is F -> infinity, not F -> 1.
    """
    Um = np.asarray(Um, dtype=np.float64)
    # Guard against U_m very close to zero where the individual terms diverge
    # but F remains finite (they cancel). Use a small floor consistent with
    # double precision.
    Um = np.where(Um < 1e-30, 1e-30, Um)
    term1 = 0.5 * np.sqrt(np.pi) * Um**(-1.5) * erf(np.sqrt(Um))
    term2 = -np.exp(-Um) / Um
    term3 = exp1(Um)
    return term1 + term2 + term3


def _psm20_q_down(n, ell_upper, Te_arr, ne_cm3=NE_DEFAULT, P1=P1_DEFAULT):
    """
    PSM20 Debye-cutoff downward rate coefficient [cm³/s]:
      q(n, ℓ_upper → ℓ_upper - 1; T_e, n_e)

    Full Badnell (2021) Eq. 9:
      q_ji = (a0^3/tau0) * sqrt(pi * mu * I_H / kT) * (D/omega_l) * F(U_m)

    Parameters
    ----------
    n         : int     principal quantum number
    ell_upper : int     ℓ of the higher-ℓ state (1 ≤ ℓ_upper ≤ n-1)
    Te_arr    : (N,) float  electron/proton temperatures [eV]
    ne_cm3    : float   electron density [cm^-3] used for the Debye cutoff
                        (defaults to NE_DEFAULT = 1e14)
    P1        : float   small-impact-parameter matching probability

    Returns
    -------
    q : (N,) float  [cm³/s]  downward rate coefficient at (T_e, n_e)

    Notes
    -----
    The dependence on n_e is only through F(U_m), which is a slowly-varying
    (roughly logarithmic) function; a factor-of-1000 change in n_e over the
    grid moves F by roughly ~30%. For a strictly (T_e, n_e)-resolved rate
    coefficient table, pass n_e explicitly per grid point.
    """
    ell_lo = ell_upper - 1
    omega  = 2 * ell_lo + 1

    D   = _psm20_D(n, ell_upper)

    kT  = Te_arr                          # [eV]
    prefactor = A0_3 / TAU0               # [cm³/s]
    sqrt_term = np.sqrt(np.pi * MU * IH / kT)

    Um = _psm20_Um(n, ell_upper, Te_arr, ne_cm3, P1=P1)
    F_um = _psm20_F(Um)

    q = prefactor * sqrt_term * (D / omega) * F_um

    return q


def _psm20_q_up(n, ell_lower, Te_arr, ne_cm3=NE_DEFAULT, P1=P1_DEFAULT):
    """
    Upward rate coefficient via reciprocity:
      q(n, ℓ_lower → ℓ_lower + 1; T) = q_down(n, ℓ_lower+1 → ℓ_lower; T)
                                        × (2*ℓ_lower + 3) / (2*ℓ_lower + 1)

    Parameters
    ----------
    n         : int
    ell_lower : int    ℓ of the lower state (0 ≤ ℓ_lower ≤ n-2)
    Te_arr    : (N,) float  [eV]
    ne_cm3    : float  [cm^-3] Debye-cutoff electron density
    P1        : float  matching probability

    Returns
    -------
    q : (N,) float  [cm³/s]  upward rate coefficient
    """
    ell_upper = ell_lower + 1
    q_down = _psm20_q_down(n, ell_upper, Te_arr, ne_cm3=ne_cm3, P1=P1)
    # reciprocity: detailed balance at LTE gives this ratio
    ratio  = (2 * ell_upper + 1) / (2 * ell_lower + 1)
    return q_down * ratio


# ── Build K_lmix table ──────────────────────────────────────────────────────────
def compute_K_lmix(te_grid=None, ne_cm3=NE_DEFAULT, P1=P1_DEFAULT, out_dir=None):
    """
    Compute proton-impact ℓ-mixing rate coefficients for all adjacent
    (n, ℓ) → (n, ℓ±1) pairs in the resolved block n = 2–8.

    Parameters
    ----------
    te_grid : (N_Te,) array [eV], optional
    ne_cm3  : float [cm^-3], Debye-cutoff reference density.
              F(U_m) varies logarithmically with n_e, so a single density
              is a good approximation across the grid; use a per-point
              (T_e, n_e) call site if strict density resolution is needed.
    P1      : float, small-impact matching probability (Badnell Eq. 4)
    out_dir : str, output directory for K_lmix.npy

    Returns
    -------
    K_lmix : (43, 43, n_Te) float64  [cm³/s]
        K_lmix[i, j, t] = rate coefficient for transition j → i
        Multiply by n_p (= n_e) to get rate [s⁻¹].

    Non-zero entries only for:
        - same n (intra-shell)
        - |ℓ_i - ℓ_j| = 1  (adjacent sublevels)
        - n = 2..8 (resolved block; n=1 has no ℓ-mixing partner)
    """
    if te_grid is None:
        te_grid = TE_GRID
    n_Te = len(te_grid)

    K = np.zeros((43, 43, n_Te), dtype=np.float64)

    pairs_added = 0

    for n in range(2, 9):           # n = 2..8 resolved
        for ell in range(n - 1):    # ℓ = 0..n-2  (has ℓ+1 partner)

            idx_lo = NL_TO_IDX[(n, ell)]        # lower-ℓ state index
            idx_hi = NL_TO_IDX[(n, ell + 1)]    # higher-ℓ state index

            # Downward rate: hi → lo  (ℓ+1 → ℓ)
            q_down = _psm20_q_down(n, ell + 1, te_grid,
                                   ne_cm3=ne_cm3, P1=P1)   # (n_Te,)

            # Upward rate: lo → hi  (ℓ → ℓ+1)
            q_up   = _psm20_q_up(n, ell, te_grid,
                                 ne_cm3=ne_cm3, P1=P1)     # (n_Te,)

            # K[destination, source, Te]
            K[idx_lo, idx_hi, :] = q_down    # hi → lo
            K[idx_hi, idx_lo, :] = q_up      # lo → hi

            pairs_added += 1

    print(f"  ℓ-mixing pairs added (n=2–8, Δℓ=±1): {pairs_added}")
    print(f"  Non-zero elements: {(K > 0).sum()}")
    print(f"  Expected: {2 * pairs_added} non-zero × {n_Te} Te points "
          f"= {2 * pairs_added * n_Te}")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = f'{out_dir}/K_lmix.npy'
        np.save(path, K)
        print(f"  Saved: {path}  shape={K.shape}  "
              f"{K.nbytes/1024:.0f} KB")

    return K


# ── QC checks ──────────────────────────────────────────────────────────────────
def qc_K_lmix(K, te_grid=None):
    """
    Physics QC checks on the K_lmix table.

    Check 1: Non-zero elements are only same-n, adjacent-ℓ, n=2–8.
    Check 2: Reciprocity — q_up/q_down = (2ℓ+3)/(2ℓ+1) for each pair.
    Check 3: Temperature scaling — q ∝ T^(-0.5) approximately.
    Check 4: n-scaling — rates increase with n (D ∝ n^4 scaling).
    Check 5: Conservative redistribution — column sums are zero
             (ℓ-mixing does not change the column sum of L).
    """
    if te_grid is None:
        te_grid = TE_GRID

    print("\nQC: K_lmix")
    print("=" * 55)
    all_pass = True

    # Check 1: structure — non-zero only where expected
    print("\nCheck 1 — Non-zero structure (same-n, Δℓ=±1, n=2–8):")
    for i in range(36):
        for j in range(36):
            if K[i, j, 0] > 0:
                ni, li = IDX_TO_NL[i]
                nj, lj = IDX_TO_NL[j]
                ok = (ni == nj) and (abs(li - lj) == 1) and (ni >= 2)
                if not ok:
                    print(f"  FAIL: unexpected non-zero at ({ni},{li})←({nj},{lj})")
                    all_pass = False
    # Also check bundled block is zero
    if K[36:, :, 0].any() or K[:, 36:, 0].any():
        print("  FAIL: bundled block has non-zero ℓ-mixing (should be zero)")
        all_pass = False
    else:
        print("  PASS")

    # Check 2: reciprocity q_up/q_down = (2ℓ_hi+1)/(2ℓ_lo+1)
    print("\nCheck 2 — Reciprocity ratio (upward/downward = g_hi/g_lo):")
    max_err = 0.0
    for n in range(2, 9):
        for ell in range(n - 1):
            idx_lo = NL_TO_IDX[(n, ell)]
            idx_hi = NL_TO_IDX[(n, ell + 1)]
            q_up   = K[idx_hi, idx_lo, :]    # lo → hi
            q_down = K[idx_lo, idx_hi, :]    # hi → lo
            expected_ratio = (2 * (ell + 1) + 1) / (2 * ell + 1)
            actual_ratio   = q_up / q_down
            err = np.abs(actual_ratio - expected_ratio).max()
            max_err = max(max_err, err)
    print(f"  Max ratio error: {max_err:.2e}  "
          f"{'PASS' if max_err < 1e-10 else 'FAIL'}")
    if max_err >= 1e-10:
        all_pass = False

    # Check 3: temperature scaling — with F(U_m) correction, the pure -0.5
    # slope from sqrt(pi mu I_H / kT) is modified because F(U_m) grows
    # weakly with T (since U_m ∝ 1/T and F ∝ ln(1/U_m) at small U_m).
    # The net slope drifts from ~-0.2 (n=2) toward ~+0.1 (n=8). All are
    # physical; only slopes far outside [-0.5, +0.2] indicate a problem.
    print("\nCheck 3 — Temperature scaling (with F(U_m) correction):")
    # Use 2s→2p as test case
    idx_2s = NL_TO_IDX[(2, 0)]
    idx_2p = NL_TO_IDX[(2, 1)]
    q_test = K[idx_2p, idx_2s, :]    # 2s → 2p
    # Fit log-log slope
    slope = np.polyfit(np.log(te_grid), np.log(q_test), 1)[0]
    print(f"  2s→2p log-log slope = {slope:.3f}  "
          f"(range with F: [-0.5, +0.2])")
    ok = -0.5 <= slope <= 0.2
    print(f"  {'PASS' if ok else 'FAIL — slope outside physical range'}")
    if not ok:
        all_pass = False

    # Check 4: n-scaling — compare q for n=2 vs n=7 at same Te
    print("\nCheck 4 — n-scaling (higher n → higher rate):")
    ti = np.argmin(np.abs(te_grid - 3.0))
    q_n2 = K[NL_TO_IDX[(2,1)], NL_TO_IDX[(2,0)], ti]    # 2s→2p
    q_n7 = K[NL_TO_IDX[(7,1)], NL_TO_IDX[(7,0)], ti]    # 7s→7p
    print(f"  q(2s→2p) at 3 eV = {q_n2:.3e} cm³/s")
    print(f"  q(7s→7p) at 3 eV = {q_n7:.3e} cm³/s")
    print(f"  Ratio q(n=7)/q(n=2) = {q_n7/q_n2:.1f}  (expect >> 1)")
    if q_n7 <= q_n2:
        print("  FAIL: n-scaling wrong")
        all_pass = False
    else:
        print("  PASS")

    # Check 5: conservative redistribution — check L_mix column sums to 0
    # Build L_mix for a test ne=1e14 and check column sums RELATIVE to diagonal
    print("\nCheck 5 — Conservative redistribution (column sums = 0, relative):")
    ne_test = 1e14
    L_mix = np.zeros((43, 43))
    for i in range(36):
        for j in range(36):
            if K[i, j, ti] > 0:
                L_mix[i, j] += K[i, j, ti] * ne_test   # off-diagonal gain
    # diagonal: minus outgoing sum
    for j in range(36):
        L_mix[j, j] -= L_mix[:, j].sum()
    col_sums = L_mix.sum(axis=0)
    max_col_err = np.abs(col_sums).max()
    max_diag    = np.abs(np.diag(L_mix)).max()
    # With F(U_m) correction, entries are ~1e12; absolute test would fail
    # at machine round-off. Use relative tolerance instead.
    rel_err = max_col_err / max_diag if max_diag > 0 else 0.0
    print(f"  Max |column sum| = {max_col_err:.2e}  "
          f"(relative: {rel_err:.2e})  "
          f"{'PASS' if rel_err < 1e-12 else 'FAIL'}")
    if rel_err >= 1e-12:
        all_pass = False

    # Print sample rates
    print("\nSample rates at Te=3 eV:")
    print(f"  {'Transition':<15s}  {'q [cm³/s]':>14s}  "
          f"{'n_e*q [s⁻¹] at 1e14':>22s}")
    print("  " + "-" * 55)
    sample_pairs = [
        (2, 0, 2, 1,  "2s→2p"),
        (2, 1, 2, 0,  "2p→2s"),
        (3, 1, 3, 0,  "3p→3s"),
        (3, 0, 3, 1,  "3s→3p"),
        (3, 2, 3, 1,  "3d→3p"),
        (7, 1, 7, 0,  "7p→7s"),
        (8, 7, 8, 6,  "8k→8i"),
    ]
    for n_to, l_to, n_fr, l_fr, label in sample_pairs:
        i = NL_TO_IDX[(n_to, l_to)]
        j = NL_TO_IDX[(n_fr, l_fr)]
        q = K[i, j, ti]
        print(f"  {label:<15s}  {q:>14.4e}  {q*1e14:>22.4e}")

    print(f"\nOverall QC: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ── add_lmix_to_L helper (for use in assemble_cr_matrix.py) ────────────────────
def add_lmix_to_L(L, K_lmix, Te_idx, ne):
    """
    Add proton-impact ℓ-mixing to an existing 43×43 rate matrix L.

    This is a CONSERVATIVE redistribution: column sums of L are unchanged.

    Parameters
    ----------
    L       : (43,43) ndarray  existing rate matrix [s⁻¹], modified in place
    K_lmix  : (43,43,n_Te) ndarray  ℓ-mixing rate coefficients [cm³/s]
    Te_idx  : int   index into Te grid
    ne      : float electron density [cm⁻³]  (= n_p under quasi-neutrality)

    Returns
    -------
    L : (43,43) ndarray  modified in place (also returned for convenience)
    """
    # Rate matrix for ℓ-mixing at this (Te, ne)
    # K_lmix[i,j,Te_idx] * ne = rate j→i [s⁻¹]
    Lm = K_lmix[:, :, Te_idx] * ne   # (43,43)

    # Off-diagonal gains: L[i,j] += rate j→i
    # Zero the diagonal of Lm before adding (diagonal set separately)
    Lm_off = Lm.copy()
    np.fill_diagonal(Lm_off, 0.0)
    L += Lm_off

    # Diagonal: loss = sum of all outgoing ℓ-mixing rates from each state
    # outgoing from j = sum_i K[i,j]*ne  for i≠j
    outgoing = Lm_off.sum(axis=0)    # (43,) — column sums of off-diagonal
    np.fill_diagonal(L, np.diag(L) - outgoing)

    return L


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Computing proton-impact ℓ-mixing rate coefficients (PSM20)")
    print("=" * 60)

    OUT_DIR = 'data/processed/lmix'

    K_lmix = compute_K_lmix(te_grid=TE_GRID, out_dir=OUT_DIR)
    qc_K_lmix(K_lmix, te_grid=TE_GRID)

    print("\nDone.")
    print(f"  K_lmix.npy saved to {OUT_DIR}/")
    print(f"  Use add_lmix_to_L() in assemble_cr_matrix.py to add to L.")