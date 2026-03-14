"""
radiative_rates.py
==================
Einstein A coefficients for the hydrogen CR model.

State space:
  n=1-8  ℓ-resolved  (36 neutral states)
  n=9-15 ℓ-bundled   (7 bundled states)
  H+     ion         (1 state)
  Total: 44 states

Source: Hoang-Binh (1993) via ADUU v1.0
File:   data/processed/Radiative/H_A_E1_LS_n1_15_physical.csv
  Columns: nu, lu, nl, ll, A_s-1, ...
  Convention: nu,lu = UPPER state; nl,ll = LOWER state
  All |Δl|=1, all nu>nl (downward only), complete for nu≤15

Outputs:
  A_resolved[i, j]   — A coefficient [s⁻¹] from resolved state j → resolved state i
                        (i is lower, j is upper; gain into i, loss from j)
  A_bundled[n, n']   — Effective A [s⁻¹] from bundled shell n → bundled or resolved n'
                        statistically weighted over ℓ within shell n
  state_index        — dict mapping (n, l) → row index for resolved states
  bundled_index      — dict mapping n → row index for bundled states (n=9..15)

Usage:
  from radiative_rates import load_radiative_rates
  A_res, A_bund, si, bi = load_radiative_rates()
"""

import numpy as np
import pandas as pd
import os

# ── Constants ─────────────────────────────────────────────────────────────────
N_RESOLVED_MAX = 8      # n=1..8 are ℓ-resolved
N_BUNDLED_MIN  = 9      # n=9..15 are ℓ-bundled
N_BUNDLED_MAX  = 15
L_CHAR = ['S','P','D','F','G','H','I','K']

# ── State indexing ────────────────────────────────────────────────────────────
def build_state_index():
    """
    Build ordered index for resolved states n=1..8.
    Order: (1,0), (2,0),(2,1), (3,0),(3,1),(3,2), ...
    Returns dict: (n,l) -> row_index  [0-indexed, 0..35]
    """
    idx = {}
    i = 0
    for n in range(1, N_RESOLVED_MAX + 1):
        for l in range(n):
            idx[(n, l)] = i
            i += 1
    return idx   # 36 entries

def build_bundled_index():
    """
    Index for bundled shells n=9..15.
    Returns dict: n -> row_index  [0-indexed, 0..6]
    """
    return {n: (n - N_BUNDLED_MIN) for n in range(N_BUNDLED_MIN, N_BUNDLED_MAX + 1)}

# ── Load and validate raw data ────────────────────────────────────────────────
def load_raw(filepath):
    df = pd.read_csv(filepath)
    required = {'nu','lu','nl','ll','A_s-1'}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    # Physics checks
    assert (df.nu > df.nl).all(),          "All transitions must be downward (nu > nl)"
    assert ((df.lu - df.ll).abs() == 1).all(), "All transitions must have |Δl|=1 (E1)"
    assert (df['A_s-1'] > 0).all(),        "All A coefficients must be positive"

    # Completeness for nu≤15
    missing = []
    for nu in range(2, N_BUNDLED_MAX + 1):
        for lu in range(nu):
            for nl in range(1, nu):
                for ll in [lu - 1, lu + 1]:
                    if ll < 0 or ll >= nl:
                        continue
                    if not ((df.nu==nu)&(df.lu==lu)&(df.nl==nl)&(df.ll==ll)).any():
                        missing.append((nu,lu,nl,ll))
    assert len(missing) == 0, f"Missing E1 transitions for nu≤15: {missing[:5]}"

    print(f"  Loaded {len(df)} rows, nu={df.nu.min()}..{df.nu.max()}")
    print(f"  Completeness check for nu≤15: PASS")
    return df

# ── Build resolved A matrix ───────────────────────────────────────────────────
def build_A_resolved(df, state_index):
    """
    A_resolved[i, j] = A coefficient for transition from state j → state i
    where j is the UPPER state and i is the LOWER state.

    In the CR equation:
      gain into state i:  sum_j A_resolved[i,j] * N[j]  (j > i in energy)
      loss from state j:  sum_i A_resolved[i,j] * N[j]  (equivalently: col sum)

    Shape: (36, 36), units: s⁻¹
    Off-diagonal only (no self-transitions).
    """
    n_res = len(state_index)
    A = np.zeros((n_res, n_res))

    df_res = df[(df.nu <= N_RESOLVED_MAX) & (df.nl <= N_RESOLVED_MAX)]

    for _, row in df_res.iterrows():
        nu, lu = int(row.nu), int(row.lu)
        nl, ll = int(row.nl), int(row.ll)
        j = state_index[(nu, lu)]   # upper state — source of decay
        i = state_index[(nl, ll)]   # lower state — destination
        A[i, j] = row['A_s-1']

    return A

# ── Build bundled effective A ─────────────────────────────────────────────────
def _stat_frac(l, n):
    """Statistical fraction of level (n,l) within shell n: (2l+1)/n²."""
    return (2*l + 1) / n**2

def build_A_bundled_to_resolved(df, state_index, bundled_index):
    """
    Effective A from bundled shell n (n=9..15) into resolved state (nl, ll).

    A_bund_res[i, b] = Σ_l [(2l+1)/n²] × A(n,l → nl,ll)
    where sum is over all l in bundled shell n that can decay to (nl,ll).

    Shape: (36, 7), units: s⁻¹
    Row i = resolved lower state index
    Col b = bundled shell index (0=n9, 1=n10, ..., 6=n15)
    """
    n_res  = len(state_index)
    n_bund = len(bundled_index)
    A = np.zeros((n_res, n_bund))

    for n_up, b in bundled_index.items():
        df_n = df[(df.nu == n_up) & (df.nl <= N_RESOLVED_MAX)]
        for _, row in df_n.iterrows():
            lu = int(row.lu)
            nl, ll = int(row.nl), int(row.ll)
            i = state_index[(nl, ll)]
            A[i, b] += _stat_frac(lu, n_up) * row['A_s-1']

    return A

def build_A_bundled_to_bundled(df, bundled_index):
    """
    Effective A from bundled shell n → bundled shell n' (both ≥ 9).

    A_bund_bund[b', b] = Σ_l [(2l+1)/n²] × Σ_{l'} A(n,l → n',l')
    where b=upper, b'=lower.

    Shape: (7, 7), units: s⁻¹
    """
    n_bund = len(bundled_index)
    A = np.zeros((n_bund, n_bund))

    bund_ns = list(bundled_index.keys())

    for n_up, b in bundled_index.items():
        # Transitions into lower bundled shells only (n_low < n_up, both ≥ 9)
        df_n = df[(df.nu == n_up) &
                  (df.nl >= N_BUNDLED_MIN) &
                  (df.nl < n_up)]
        for _, row in df_n.iterrows():
            lu   = int(row.lu)
            n_lo = int(row.nl)
            bp   = bundled_index[n_lo]   # lower bundled shell index
            A[bp, b] += _stat_frac(lu, n_up) * row['A_s-1']

    return A

def build_A_resolved_to_bundled(df, bundled_index):
    """
    A from resolved state (nu, lu) into bundled shell n' (n'=9..15).
    This only exists if nu > N_RESOLVED_MAX, which cannot happen since
    resolved states have nu≤8 and bundled shells start at n=9.
    Radiative decay always goes downward, so resolved states (nu≤8)
    CANNOT radiatively decay INTO bundled shells (n≥9).
    This matrix is identically zero — included for completeness.

    Shape: (7, 36)
    """
    return np.zeros((len(bundled_index), 36))

# ── Total radiative loss rate per state ──────────────────────────────────────
def total_loss_rates(A_res, A_bund_res, A_bund_bund, state_index, bundled_index):
    """
    Γ_rad[state] = sum of all A out of that state [s⁻¹]
    Useful for QC and for computing radiative lifetimes.
    """
    n_res  = len(state_index)
    n_bund = len(bundled_index)

    gamma_res  = np.zeros(n_res)
    gamma_bund = np.zeros(n_bund)

    # Resolved states: sum over all lower resolved states (col sum of A_res)
    gamma_res = A_res.sum(axis=0)          # shape (36,)

    # Bundled states: sum over resolved destinations + lower bundled destinations
    gamma_bund = (A_bund_res.sum(axis=0) +   # → resolved
                  A_bund_bund.sum(axis=0))    # → lower bundled

    return gamma_res, gamma_bund

# ── QC checks ────────────────────────────────────────────────────────────────
def run_qc(A_res, A_bund_res, A_bund_bund,
           gamma_res, gamma_bund, state_index, bundled_index):

    print("\n  QC Check A — No negative A coefficients:")
    assert (A_res >= 0).all()
    assert (A_bund_res >= 0).all()
    assert (A_bund_bund >= 0).all()
    print("    PASS")

    print("  QC Check B — Diagonal of A_res is zero (no self-decay):")
    assert np.diag(A_res).sum() == 0
    print("    PASS")

    print("  QC Check C — Key transition values vs NIST:")
    checks = [
        ((2,1),(1,0), 6.268e8, "2P→1S Lyman-α"),
        ((3,2),(2,1), 6.469e7, "3D→2P Balmer-α"),
        ((3,1),(2,0), 2.246e7, "3P→2S"),
        ((3,0),(2,1), 6.317e6, "3S→2P"),
    ]
    all_pass = True
    for (nu,lu),(nl,ll), expected, label in checks:
        j = state_index[(nu,lu)]
        i = state_index[(nl,ll)]
        got = A_res[i,j]
        err = abs(got/expected - 1)*100
        flag = "PASS" if err < 1.0 else "FAIL"
        if flag == "FAIL": all_pass = False
        print(f"    {label:20s}: A={got:.4e} s⁻¹  err={err:.2f}%  {flag}")
    assert all_pass

    print("  QC Check D — Radiative lifetimes (selected states):")
    print(f"    {'State':6s}  {'Γ_rad [s⁻¹]':>14s}  {'τ [ns]':>10s}")
    spot = [(2,1),(3,0),(3,1),(3,2),(4,1),(4,3)]
    for (n,l) in spot:
        idx = state_index[(n,l)]
        g = gamma_res[idx]
        tau_ns = 1e9/g if g > 0 else np.inf
        print(f"    {n}{L_CHAR[l]}      {g:14.4e}  {tau_ns:10.3f}")

    print("  QC Check E — 2S has very small Γ_rad (metastable, only 2S→1S forbidden):")
    idx_2s = state_index[(2,0)]
    # 2S cannot decay by E1 to 1S (Δl=0 forbidden). Only A(2S→...) involves
    # transitions to n<2 with Δl=±1 — but there is only 1s with l=0,
    # and 2s→1s requires Δl=0: FORBIDDEN. So A_res col for 2S must be zero.
    assert A_res[:, idx_2s].sum() == 0.0, \
        f"2S should have zero E1 decay but got {A_res[:,idx_2s].sum():.3e}"
    print(f"    2S total E1 decay rate = 0 (correctly metastable)  PASS")

    print("  QC Check F — Bundled A_eff n=10→n=9 is nonzero and physical:")
    b10 = bundled_index[10]
    b9  = bundled_index[9]
    A_10_9 = A_bund_bund[b9, b10]
    assert A_10_9 > 0, "A_eff(10→9) should be nonzero"
    print(f"    A_eff(n=10 → n=9) = {A_10_9:.4e} s⁻¹  PASS")

    print("\n  All QC checks PASSED ✓")

# ── Main loader ───────────────────────────────────────────────────────────────
def load_radiative_rates(filepath=None):
    """
    Main entry point. Returns:
      A_res       (36,36)  resolved→resolved A matrix [s⁻¹]
      A_bund_res  (36, 7)  bundled→resolved  A matrix [s⁻¹]
      A_bund_bund ( 7, 7)  bundled→bundled   A matrix [s⁻¹]
      gamma_res   (36,)    total rad loss rate per resolved state [s⁻¹]
      gamma_bund  ( 7,)    total rad loss rate per bundled state  [s⁻¹]
      state_index          dict (n,l) → int [0..35]
      bundled_index        dict n     → int [0..6]
    """
    if filepath is None:
        # Try standard repo path relative to repo root
        candidates = [
            'data/processed/Radiative/H_A_E1_LS_n1_15_physical.csv',
            '/mnt/user-data/uploads/H_A_E1_LS_n1_15_physical.csv',
        ]
        for c in candidates:
            if os.path.exists(c):
                filepath = c
                break
        if filepath is None:
            raise FileNotFoundError(
                "H_A_E1_LS_n1_15_physical.csv not found. "
                "Pass filepath explicitly or run from repo root.")

    print(f"Loading radiative rates from: {filepath}")

    df            = load_raw(filepath)
    state_index   = build_state_index()
    bundled_index = build_bundled_index()

    A_res       = build_A_resolved(df, state_index)
    A_bund_res  = build_A_bundled_to_resolved(df, state_index, bundled_index)
    A_bund_bund = build_A_bundled_to_bundled(df, bundled_index)

    gamma_res, gamma_bund = total_loss_rates(
        A_res, A_bund_res, A_bund_bund, state_index, bundled_index)

    print("\nRunning QC checks...")
    run_qc(A_res, A_bund_res, A_bund_bund,
           gamma_res, gamma_bund, state_index, bundled_index)

    print("\nOutput shapes:")
    print(f"  A_res       : {A_res.shape}   resolved→resolved")
    print(f"  A_bund_res  : {A_bund_res.shape}    bundled→resolved")
    print(f"  A_bund_bund : {A_bund_bund.shape}     bundled→bundled")
    print(f"  gamma_res   : {gamma_res.shape}    total loss [s⁻¹] per resolved state")
    print(f"  gamma_bund  : {gamma_bund.shape}     total loss [s⁻¹] per bundled state")

    return (A_res, A_bund_res, A_bund_bund,
            gamma_res, gamma_bund,
            state_index, bundled_index)


# ── Standalone run ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    (A_res, A_bund_res, A_bund_bund,
     gamma_res, gamma_bund,
     state_index, bundled_index) = load_radiative_rates()

    # Print summary table
    print("\n" + "="*55)
    print("RADIATIVE RATE SUMMARY")
    print("="*55)
    print("\nResolved states — total radiative decay rate:")
    print(f"  {'State':6s}  {'Γ_rad [s⁻¹]':>14s}  {'τ_rad [ns]':>12s}")
    print("  " + "-"*38)
    L_CHAR_LOCAL = ['S','P','D','F','G','H','I','K']
    for (n,l), idx in sorted(state_index.items()):
        g = gamma_res[idx]
        if g == 0:
            print(f"  {n}{L_CHAR_LOCAL[l]}      {'0 (metastable)':>14s}  {'∞':>12s}")
        else:
            tau_ns = 1e9/g
            print(f"  {n}{L_CHAR_LOCAL[l]}      {g:14.4e}  {tau_ns:12.3f}")

    print("\nBundled states — effective total radiative decay rate:")
    print(f"  {'Shell':6s}  {'Γ_eff [s⁻¹]':>14s}  {'τ_eff [ns]':>12s}")
    print("  " + "-"*38)
    for n, b in bundled_index.items():
        g = gamma_bund[b]
        tau_ns = 1e9/g if g > 0 else np.inf
        print(f"  n={n}      {g:14.4e}  {tau_ns:12.3f}")

    print("\nNon-zero elements in A_res:", (A_res > 0).sum(),
          f"/ {A_res.size} (sparsity: {100*(A_res==0).mean():.1f}%)")
    print("Non-zero elements in A_bund_res:", (A_bund_res > 0).sum())
    print("Non-zero elements in A_bund_bund:", (A_bund_bund > 0).sum())

    # ── Save outputs ──────────────────────────────────────────────────────────
    import os
    out_dir = 'data/processed/Radiative'
    os.makedirs(out_dir, exist_ok=True)

    # 1. Dense numpy arrays
    np.save(f'{out_dir}/A_resolved.npy',     A_res)
    np.save(f'{out_dir}/A_bund_res.npy',     A_bund_res)
    np.save(f'{out_dir}/A_bund_bund.npy',    A_bund_bund)
    np.save(f'{out_dir}/gamma_resolved.npy', gamma_res)
    np.save(f'{out_dir}/gamma_bundled.npy',  gamma_bund)
    print(f"\nSaved .npy arrays to {out_dir}/")

    # 2. Human-readable CSV: all non-zero A values (long format)
    L_CHAR_LOCAL = ['S','P','D','F','G','H','I','K']
    rows = []

    # Resolved → resolved
    for (nu,lu), j in state_index.items():
        for (nl,ll), i in state_index.items():
            if A_res[i,j] > 0:
                rows.append({
                    'type':         'res_to_res',
                    'n_upper':      nu, 'l_upper': lu,
                    'label_upper':  f"{nu}{L_CHAR_LOCAL[lu]}",
                    'n_lower':      nl, 'l_lower': ll,
                    'label_lower':  f"{nl}{L_CHAR_LOCAL[ll]}",
                    'A_s-1':        A_res[i,j],
                    'idx_upper':    j,  'idx_lower': i,
                })

    # Bundled → resolved
    for n_up, b in bundled_index.items():
        for (nl,ll), i in state_index.items():
            if A_bund_res[i,b] > 0:
                rows.append({
                    'type':         'bund_to_res',
                    'n_upper':      n_up, 'l_upper': -1,
                    'label_upper':  f"n{n_up}(bund)",
                    'n_lower':      nl,   'l_lower': ll,
                    'label_lower':  f"{nl}{L_CHAR_LOCAL[ll]}",
                    'A_s-1':        A_bund_res[i,b],
                    'idx_upper':    b,    'idx_lower': i,
                })

    # Bundled → bundled
    for n_up, b in bundled_index.items():
        for n_lo, bp in bundled_index.items():
            if A_bund_bund[bp,b] > 0:
                rows.append({
                    'type':         'bund_to_bund',
                    'n_upper':      n_up, 'l_upper': -1,
                    'label_upper':  f"n{n_up}(bund)",
                    'n_lower':      n_lo, 'l_lower': -1,
                    'label_lower':  f"n{n_lo}(bund)",
                    'A_s-1':        A_bund_bund[bp,b],
                    'idx_upper':    b,    'idx_lower': bp,
                })

    df_out = pd.DataFrame(rows).sort_values(
        ['type','n_upper','n_lower']).reset_index(drop=True)
    csv_path = f'{out_dir}/radiative_rates.csv'
    df_out.to_csv(csv_path, index=False)
    print(f"Saved {len(df_out)}-row CSV to {csv_path}")
    print(f"  res_to_res  : {(df_out.type=='res_to_res').sum()} rows")
    print(f"  bund_to_res : {(df_out.type=='bund_to_res').sum()} rows")
    print(f"  bund_to_bund: {(df_out.type=='bund_to_bund').sum()} rows")

    # 3. State index CSV
    idx_rows = []
    for (n,l), i in state_index.items():
        idx_rows.append({'idx': i, 'n': n, 'l': l,
                         'label': f"{n}{L_CHAR_LOCAL[l]}",
                         'type': 'resolved',
                         'gamma_rad_s-1': gamma_res[i]})
    for n, b in bundled_index.items():
        idx_rows.append({'idx': 36+b, 'n': n, 'l': -1,
                         'label': f"n{n}(bund)",
                         'type': 'bundled',
                         'gamma_rad_s-1': gamma_bund[b]})
    df_idx = pd.DataFrame(idx_rows).sort_values('idx').reset_index(drop=True)
    idx_path = f'{out_dir}/state_index.csv'
    df_idx.to_csv(idx_path, index=False)
    print(f"Saved state index to {idx_path} ({len(df_idx)} states)")