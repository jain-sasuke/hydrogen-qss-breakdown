"""
parse_ccc.py
============
Parser for CCC (Convergent Close-Coupling) electron-impact cross-section data
from Prof. Igor Bray's database.

FILE CONVENTION (Bray, personal communication 2026):
  Filename = FINAL.INITIAL
  e.g. 2P.1S  →  transition is 1S → 2P  (stored as de-excitation: 2P → 1S)
  Column 1 = incident electron KE relative to INITIAL state (eV)
  Column 2 = cross-section (a0^2)
  Column 3 = asymmetry parameter (not used)
  Column 4 = internal bookkeeping (not used)

EXCLUSIONS:
  - Same-n transitions (Δn=0): non-relativistic divergence, non-physical
  - n-bundled files (e.g. 1S.2, 3.2): not ℓ-resolved
  - TICS files: ionization, not excitation
  - Colon files (n=10): handled separately if needed

OUTPUT:
  - ccc_crosssections.h5   (HDF5, primary)
  - ccc_crosssections.csv  (CSV, human-readable QC)

STORED DIRECTION:
  All data stored as de-excitation (higher-n initial → lower-n final AS IN FILE).
  Use detailed balance in excitation_rates.py to get K_exc from K_deexc.
"""

import os
import re
import numpy as np
import pandas as pd
import h5py
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────
L_MAP = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7, 'K': 8}
L_MAP_INV = {v: k for k, v in L_MAP.items()}

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_state(token):
    """
    Parse a state token like '2P' into (n, l).
    Returns (n:int, l:int) or raises ValueError.
    """
    token = token.strip()
    if len(token) < 2:
        raise ValueError(f"Token too short: '{token}'")
    n_str = token[:-1]
    l_char = token[-1].upper()
    if not n_str.isdigit():
        raise ValueError(f"Non-integer n in token: '{token}'")
    if l_char not in L_MAP:
        raise ValueError(f"Unknown ℓ character '{l_char}' in token: '{token}'")
    n = int(n_str)
    l = L_MAP[l_char]
    if l >= n:
        raise ValueError(f"Unphysical state: ℓ={l} >= n={n} in token '{token}'")
    return n, l


def classify_filename(filename):
    """
    Classify a CCC filename.

    Returns one of:
      'valid_lr'   - ℓ-resolved, Δn≠0  → KEEP
      'same_n'     - ℓ-resolved, Δn=0  → EXCLUDE
      'bundled'    - n-bundled          → SKIP
      'tics'       - ionization         → SKIP
      'colon'      - n=10 files         → SKIP
      'unknown'    - unrecognised       → SKIP

    For 'valid_lr' and 'same_n', also returns (n_final, l_final, n_initial, l_initial).
    Otherwise returns None for the state tuple.
    """
    name = os.path.basename(filename)

    if 'TICS' in name:
        return 'tics', None
    if name.startswith(':'):
        return 'colon', None

    if '.' not in name:
        return 'unknown', None

    parts = name.split('.')
    if len(parts) != 2:
        return 'unknown', None

    final_tok, initial_tok = parts[0], parts[1]

    # ℓ-resolved pattern: digit(s) + letter
    lr_pattern = re.compile(r'^\d+[SPDFGHIJK]$', re.IGNORECASE)

    if lr_pattern.match(final_tok) and lr_pattern.match(initial_tok):
        try:
            n_f, l_f = parse_state(final_tok)
            n_i, l_i = parse_state(initial_tok)
        except ValueError:
            return 'unknown', None
        if n_f == n_i:
            return 'same_n', (n_f, l_f, n_i, l_i)
        return 'valid_lr', (n_f, l_f, n_i, l_i)

    # n-bundled: at least one side is digits-only
    digit_only = re.compile(r'^\d+$')
    if digit_only.match(final_tok) or digit_only.match(initial_tok):
        return 'bundled', None

    return 'unknown', None


def read_ccc_file(filepath):
    """
    Read a single CCC data file.
    Returns (energies, sigmas) as numpy arrays, or raises on failure.

    Skips comment lines (starting with #) and any line that doesn't
    parse as two leading floats.
    """
    energies = []
    sigmas = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            try:
                e = float(tokens[0])
                s = float(tokens[1])
            except ValueError:
                continue
            if e < 0 or s < 0:
                continue          # unphysical, skip
            energies.append(e)
            sigmas.append(s)

    return np.array(energies), np.array(sigmas)


# ── Main parser ────────────────────────────────────────────────────────────────

def parse_ccc_database(data_dir, output_dir='.'):
    """
    Parse all CCC files in data_dir.

    Parameters
    ----------
    data_dir  : str or Path  — directory containing CCC files
    output_dir: str or Path  — where to write output files

    Returns
    -------
    df : pd.DataFrame with columns:
         n_i, l_i, n_f, l_f, n_points, E_min_eV, E_max_eV, sigma_max_a0sq
         (one row per transition — summary level)

    Full (E, sigma) data written to HDF5 and CSV.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect all files ──────────────────────────────────────────────────────
    all_files = sorted(data_dir.iterdir())
    print(f"Found {len(all_files)} files in {data_dir}")

    # ── Counters ───────────────────────────────────────────────────────────────
    counts = {'valid_lr': 0, 'same_n': 0, 'bundled': 0,
              'tics': 0, 'colon': 0, 'unknown': 0, 'read_error': 0}

    records_summary = []   # one row per transition
    records_full    = []   # one row per (transition, energy point) → CSV

    h5_path  = output_dir / 'ccc_crosssections.h5'
    csv_path = output_dir / 'ccc_crosssections.csv'

    with h5py.File(h5_path, 'w') as h5f:

        # Store metadata
        h5f.attrs['description'] = 'CCC electron-impact de-excitation cross sections'
        h5f.attrs['source']      = 'Bray, personal communication 2026'
        h5f.attrs['energy_unit'] = 'eV (relative to initial state)'
        h5f.attrs['sigma_unit']  = 'a0^2 (Bohr radius squared)'
        h5f.attrs['convention']  = 'FINAL.INITIAL filename; data = de-excitation'

        for fpath in all_files:
            cat, states = classify_filename(fpath.name)
            counts[cat] = counts.get(cat, 0) + 1

            if cat != 'valid_lr':
                continue

            n_f, l_f, n_i, l_i = states

            # Read data
            try:
                energies, sigmas = read_ccc_file(fpath)
            except Exception as e:
                print(f"  READ ERROR: {fpath.name} — {e}")
                counts['read_error'] += 1
                continue

            if len(energies) == 0:
                print(f"  EMPTY: {fpath.name}")
                counts['read_error'] += 1
                continue

            # ── Write to HDF5 ──────────────────────────────────────────────────
            # Group structure: /ni_li/nf_lf/  with datasets 'energy' and 'sigma'
            l_i_char = L_MAP_INV[l_i]
            l_f_char = L_MAP_INV[l_f]
            grp_name = f"{n_i}{l_i_char}_to_{n_f}{l_f_char}"

            grp = h5f.create_group(grp_name)
            grp.attrs['n_initial'] = n_i
            grp.attrs['l_initial'] = l_i
            grp.attrs['n_final']   = n_f
            grp.attrs['l_final']   = l_f
            grp.attrs['direction'] = 'de-excitation (as stored in file)'
            grp.attrs['filename']  = fpath.name
            grp.create_dataset('energy_eV', data=energies, compression='gzip')
            grp.create_dataset('sigma_a0sq', data=sigmas,  compression='gzip')

            # ── Accumulate for CSV ─────────────────────────────────────────────
            for e, s in zip(energies, sigmas):
                records_full.append({
                    'n_i': n_i, 'l_i': l_i,
                    'n_f': n_f, 'l_f': l_f,
                    'l_i_char': l_i_char, 'l_f_char': l_f_char,
                    'E_eV': e,
                    'sigma_a0sq': s,
                    'filename': fpath.name,
                })

            # ── Summary record ─────────────────────────────────────────────────
            records_summary.append({
                'n_i': n_i, 'l_i': l_i, 'l_i_char': l_i_char,
                'n_f': n_f, 'l_f': l_f, 'l_f_char': l_f_char,
                'n_points':     len(energies),
                'E_min_eV':     energies.min(),
                'E_max_eV':     energies.max(),
                'sigma_max_a0sq': sigmas.max(),
                'filename':     fpath.name,
            })

    # ── Write CSV ──────────────────────────────────────────────────────────────
    df_full = pd.DataFrame(records_full)
    col_order = ['n_i', 'l_i', 'l_i_char', 'n_f', 'l_f', 'l_f_char',
                 'E_eV', 'sigma_a0sq', 'filename']
    df_full = df_full[col_order]
    df_full.to_csv(csv_path, index=False)
    print(f"\nCSV written: {csv_path}  ({len(df_full):,} rows)")

    # ── Summary DataFrame ──────────────────────────────────────────────────────
    df_summary = pd.DataFrame(records_summary)

    # ── Print report ───────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  CCC PARSE REPORT")
    print("="*55)
    print(f"  Total files scanned   : {len(all_files)}")
    print(f"  Valid ℓ-resolved Δn≠0 : {counts['valid_lr']}  ← PARSED")
    print(f"  Same-n excluded (Δn=0): {counts['same_n']}  ← EXCLUDED")
    print(f"  n-bundled skipped     : {counts['bundled']}")
    print(f"  TICS skipped          : {counts['tics']}")
    print(f"  Colon (n=10) skipped  : {counts['colon']}")
    print(f"  Unknown skipped       : {counts['unknown']}")
    print(f"  Read errors           : {counts['read_error']}")
    print(f"  Total data points     : {len(df_full):,}")
    print(f"\n  Output:")
    print(f"    HDF5 : {h5_path}")
    print(f"    CSV  : {csv_path}")
    print("="*55)

    return df_summary, df_full


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_ccc.py <ccc_data_dir> [output_dir]")
        print("Example: python parse_ccc.py ./ccc_data ./output")
        sys.exit(1)

    data_dir   = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './output'

    df_summary, df_full = parse_ccc_database(data_dir, output_dir)

    print("\nSample summary (first 10 transitions):")
    print(df_summary.head(10).to_string(index=False))