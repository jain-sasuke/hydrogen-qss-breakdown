"""
parse_tics.py
=============
Parse CCC Total Ionization Cross Section (TICS) files from Bray (2026).

File naming convention: TICS.INITIAL
  e.g. TICS.2P  = ionization cross section from 2P state
       TICS.2   = ionization cross section from n=2 (all ℓ bundled)
       TICS.1S  = ionization cross section from 1S ground state

File format (same as excitation files):
  # comment lines
  E(eV)   sigma(a0^2)   [asym]   [metadata]
  E is KE of incident electron measured from initial state
  First data row E ≈ I_n = 13.6058/n² eV (ionization threshold)

Two types:
  ℓ-resolved: TICS.nL  (e.g. TICS.2P, TICS.9K)  — 45 files
  n-bundled:  TICS.n   (e.g. TICS.2, TICS.9)    —  9 files (QC only)

Output:
  data/processed/collisions/tics/tics_crosssections.csv
    Columns: n, l, l_char, type, E_eV, sigma_a0sq, filename
    type = 'resolved' or 'bundled'
"""

import os
import re
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
RAW_DIR = 'data/raw/ccc/e-H_XSEC_LS'
OUT_DIR = 'data/processed/collisions/tics'
OUT_CSV = f'{OUT_DIR}/tics_crosssections.csv'

IH_EV   = 13.6058
L_MAP   = {'S':0,'P':1,'D':2,'F':3,'G':4,'H':5,'I':6,'J':7,'K':8}
L_CHAR  = ['S','P','D','F','G','H','I','J','K']

# ── Filename parser ───────────────────────────────────────────────────────────
def parse_tics_filename(filename):
    """
    Parse TICS filename → (n, l, l_char, type)

    TICS.2P  → (2, 1, 'P', 'resolved')
    TICS.9K  → (9, 8, 'K', 'resolved')
    TICS.2   → (2, None, None, 'bundled')
    TICS.1S  → (1, 0, 'S', 'resolved')

    Returns None if not a valid TICS file.
    """
    if not filename.startswith('TICS.'):
        return None

    suffix = filename[5:]   # everything after 'TICS.'

    # n-bundled: TICS.N where N is a single integer
    if re.match(r'^\d+$', suffix):
        n = int(suffix)
        return (n, None, None, 'bundled')

    # ℓ-resolved: TICS.NL where L is a letter
    m = re.match(r'^(\d+)([SPDFGHIJK])$', suffix)
    if m:
        n = int(m.group(1))
        l_char = m.group(2)
        l = L_MAP[l_char]
        # Validate: l must be < n
        if l >= n:
            return None
        return (n, l, l_char, 'resolved')

    return None

# ── File reader ───────────────────────────────────────────────────────────────
def read_tics_file(filepath):
    """
    Read TICS data file.
    Returns (E_eV array, sigma_a0sq array) or raises on format error.
    """
    E_vals = []
    sig_vals = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                E   = float(parts[0])
                sig = float(parts[1])
                if E > 0 and sig >= 0:
                    E_vals.append(E)
                    sig_vals.append(sig)
            except ValueError:
                continue

    if len(E_vals) < 5:
        raise ValueError(f"Too few data points ({len(E_vals)}) in {filepath}")

    return np.array(E_vals), np.array(sig_vals)

# ── Physics validation ────────────────────────────────────────────────────────
def validate_tics_file(n, l, l_char, E_vals, sig_vals, filename):
    """
    Physics checks for a single TICS file.
    Returns list of warning strings (empty = all good).
    """
    warnings = []
    I_n = IH_EV / n**2

    # Check 1: E_min should be near I_n
    E_min = E_vals[0]
    if abs(E_min - I_n) / I_n > 0.5:
        warnings.append(
            f"E_min={E_min:.4f} eV far from I_n={I_n:.4f} eV "
            f"(ratio={E_min/I_n:.2f})")

    # Check 2: sigma must be non-negative
    if (sig_vals < 0).any():
        warnings.append(f"Negative sigma values: {(sig_vals<0).sum()} points")

    # Check 3: sigma should go to zero at threshold (or rise from zero)
    # First few points near threshold should be small
    if sig_vals[0] > 1e6:
        warnings.append(f"sigma[0]={sig_vals[0]:.3e} a0² suspiciously large at threshold")

    # Check 4: E values must be monotonically increasing
    if not np.all(np.diff(E_vals) > 0):
        warnings.append("E values not monotonically increasing")

    # Check 5: sigma should decrease at high E (ionization cross section peaks
    # near threshold and falls as ~1/E at high energy)
    if len(E_vals) > 10:
        if sig_vals[-1] > sig_vals[0] * 10:
            warnings.append(
                f"sigma rises too much at high E: "
                f"sig[-1]={sig_vals[-1]:.3e} >> sig[0]={sig_vals[0]:.3e}")

    return warnings

# ── Main parser ───────────────────────────────────────────────────────────────
def parse_all_tics(raw_dir=RAW_DIR, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # Find all TICS files
    all_files = sorted(os.listdir(raw_dir))
    tics_files = [f for f in all_files if f.startswith('TICS.')]
    print(f"Found {len(tics_files)} TICS files in {raw_dir}")

    records = []
    skipped = []
    warnings_log = []

    for filename in tics_files:
        parsed = parse_tics_filename(filename)
        if parsed is None:
            skipped.append(filename)
            continue

        n, l, l_char, ftype = parsed
        filepath = os.path.join(raw_dir, filename)

        try:
            E_vals, sig_vals = read_tics_file(filepath)
        except Exception as e:
            skipped.append(f"{filename} (read error: {e})")
            continue

        # Physics validation
        if ftype == 'resolved':
            warns = validate_tics_file(n, l, l_char, E_vals, sig_vals, filename)
            for w in warns:
                warnings_log.append(f"  {filename}: {w}")

        # Store each data point as a row
        l_val = l if l is not None else -1
        l_str = l_char if l_char is not None else 'bundled'
        for E, sig in zip(E_vals, sig_vals):
            records.append({
                'n':        n,
                'l':        l_val,
                'l_char':   l_str,
                'type':     ftype,
                'E_eV':     E,
                'sigma_a0sq': sig,
                'filename': filename,
            })

    df = pd.DataFrame(records)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_resolved = df[df.type=='resolved']['filename'].nunique()
    n_bundled  = df[df.type=='bundled']['filename'].nunique()
    print(f"\nParsed:")
    print(f"  ℓ-resolved files : {n_resolved}  (expected 45)")
    print(f"  n-bundled files  : {n_bundled}   (expected 9)")
    print(f"  Total data rows  : {len(df)}")
    print(f"  Skipped          : {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"    {s}")

    if warnings_log:
        print(f"\nPhysics warnings ({len(warnings_log)}):")
        for w in warnings_log:
            print(w)
    else:
        print("\nNo physics warnings.")

    # ── Coverage check ────────────────────────────────────────────────────────
    print("\nℓ-resolved coverage (n=1..9):")
    print(f"  {'n':3s}  {'l values found':30s}  {'expected':20s}  status")
    print("  " + "-"*70)
    all_ok = True
    for n in range(1, 10):
        sub = df[(df.type=='resolved') & (df.n==n)]
        found_l = sorted(sub.l.unique())
        expected_l = list(range(n))
        ok = found_l == expected_l
        if not ok:
            all_ok = False
        print(f"  {n:3d}  {str(found_l):30s}  {str(expected_l):20s}  "
              f"{'✓' if ok else '✗ MISSING '+str(set(expected_l)-set(found_l))}")

    if all_ok:
        print("  Coverage: COMPLETE ✓")
    else:
        print("  Coverage: INCOMPLETE ✗")

    # ── Threshold check ───────────────────────────────────────────────────────
    print("\nThreshold energy check (E_min vs I_n):")
    print(f"  {'File':12s}  {'E_min (eV)':>12s}  {'I_n (eV)':>10s}  {'ratio':>8s}  status")
    print("  " + "-"*58)
    spot_checks = ['TICS.1S','TICS.2P','TICS.3D','TICS.4F','TICS.8K','TICS.9S']
    for fname in spot_checks:
        sub = df[df.filename==fname]
        if len(sub) == 0:
            print(f"  {fname:12s}  NOT FOUND")
            continue
        n_val = sub.n.iloc[0]
        I_n   = IH_EV / n_val**2
        E_min = sub.E_eV.min()
        ratio = E_min / I_n
        flag  = "✓" if 0.5 < ratio < 2.0 else "✗"
        print(f"  {fname:12s}  {E_min:12.4f}  {I_n:10.4f}  {ratio:8.4f}  {flag}")

    # ── Data points per file ──────────────────────────────────────────────────
    pts_per_file = df[df.type=='resolved'].groupby('filename').size()
    print(f"\nData points per ℓ-resolved file: "
          f"min={pts_per_file.min()}, max={pts_per_file.max()}, "
          f"median={pts_per_file.median():.0f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}  ({len(df)} rows)")

    return df


if __name__ == '__main__':
    df = parse_all_tics()

    print("\nSample rows (first 3 of each type):")
    for ftype in ['resolved', 'bundled']:
        sub = df[df.type==ftype].head(3)
        print(f"\n  type={ftype}:")
        print(sub[['n','l','l_char','type','E_eV','sigma_a0sq','filename']
                  ].to_string(index=False))