"""
CCC Cross Section Database Builder

Parses all CCC files from Bray and creates a structured database.

Usage:
    python -m src.parsers.parse_ccc

Output:
    - ccc_database.pkl: Complete cross section database
    - ccc_transitions_summary.csv: Human-readable summary

Author: Week 2, Task 2.1
Date: 2026-02-22
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config.paths import PATHS, ensure_dir


def decode_filename(filename):
    """
    Decode CCC filename to quantum numbers.
    
    Bray's notation:
    - S, P, D, F, G, H, I, J, K, L = l = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    - : = n = 10
    - Numbers 1-9 = n = 1-9
    
    Examples:
        '1S.2P' -> (n=1, l=0) -> (n=2, l=1)
        '2P.3D' -> (n=2, l=1) -> (n=3, l=2)
        ':P.1'  -> (n=10, l=1) -> ionization
        
    Args:
        filename: CCC filename (e.g., '1S.2P')
    
    Returns:
        ni, li, nf, lf: Initial and final quantum numbers
        or None if parsing fails
    """
    
    l_map = {
        'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4,
        'H': 5, 'I': 6, 'J': 7, 'K': 8, 'L': 9
    }
    
    parts = filename.split('.')
    if len(parts) != 2:
        return None, None, None, None
    
    initial, final = parts
    
    # Parse initial state
    if initial.startswith(':'):
        ni = 10
        l_letter = initial[1:]
        if l_letter in l_map:
            li = l_map[l_letter]
        else:
            return None, None, None, None
    else:
        digit_part = ''
        l_letter = ''
        for char in initial:
            if char.isdigit():
                digit_part += char
            else:
                l_letter = char
                break
        
        if digit_part and l_letter in l_map:
            ni = int(digit_part)
            li = l_map[l_letter]
        else:
            return None, None, None, None
    
    # Parse final state
    if final.startswith(':'):
        nf = 10
        l_letter = final[1:]
        if l_letter in l_map:
            lf = l_map[l_letter]
        else:
            return None, None, None, None
    else:
        digit_part = ''
        l_letter = ''
        for char in final:
            if char.isdigit():
                digit_part += char
            else:
                l_letter = char
                break
        
        if digit_part:
            nf = int(digit_part)
            if l_letter in l_map:
                lf = l_map[l_letter]
            else:
                lf = -1  # Continuum
        else:
            return None, None, None, None
    
    return ni, li, nf, lf


def parse_ccc_file(filepath):
    """
    Parse a single CCC cross section file.
    
    Args:
        filepath: Path to CCC file
    
    Returns:
        E_eV: Energy above threshold [eV]
        sigma_cm2: Cross section [cm^2]
        threshold_eV: Threshold energy [eV]
        n_points: Number of data points
    """
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    energy_list = []
    sigma_list = []
    
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        
        columns = line.split()
        
        try:
            E = float(columns[0])
            sigma_a0sq = float(columns[1])
            energy_list.append(E)
            sigma_list.append(sigma_a0sq)
        except:
            continue
    
    E_eV = np.array(energy_list)
    sigma_a0sq = np.array(sigma_list)
    
    # Convert units: a0^2 to cm^2
    a0_cm = 5.29177e-9
    sigma_cm2 = sigma_a0sq * (a0_cm ** 2)
    
    # Extract threshold from first data line
    threshold_eV = None
    for line in lines:
        if not line.startswith('#') and line.strip() != '':
            columns = line.split()
            if len(columns) >= 4:
                last_col = columns[-1]
                threshold_str = last_col.split('eV')[0]
                try:
                    threshold_eV = float(threshold_str)
                except:
                    pass
                break
    
    if threshold_eV is None:
        threshold_eV = 0.0
    
    n_points = len(E_eV)
    
    return E_eV, sigma_cm2, threshold_eV, n_points


def build_ccc_database(ccc_directory):
    """
    Parse all CCC files and build structured database.
    
    Args:
        ccc_directory: Path to directory containing CCC files
    
    Returns:
        database: Nested dictionary with cross section data
        summary: List of dictionaries with metadata
    """
    
    print("=" * 70)
    print("BUILDING CCC DATABASE")
    print("=" * 70)
    print()
    
    ccc_dir = Path(ccc_directory)
    
    if not ccc_dir.exists():
        raise FileNotFoundError(f"CCC directory not found: {ccc_dir}")
    
    all_files = [f for f in os.listdir(ccc_dir) 
                 if not f.startswith('.') and 
                 os.path.isfile(ccc_dir / f) and
                 f != 'file_list.txt' and
                 not f.endswith('.csv') and
                 not f.endswith('.py')]
    
    print(f"Found {len(all_files)} files in:")
    print(f"  {ccc_dir}")
    print()
    
    database = {}
    summary = []
    
    n_parsed = 0
    n_failed = 0
    
    for filename in all_files:
        
        ni, li, nf, lf = decode_filename(filename)
        
        if ni is None:
            n_failed += 1
            continue
        
        # Skip ionization files for now
        if lf == -1:
            continue
        
        filepath = ccc_dir / filename
        
        try:
            E_eV, sigma_cm2, threshold_eV, n_points = parse_ccc_file(filepath)
            
            if ni not in database:
                database[ni] = {}
            if li not in database[ni]:
                database[ni][li] = {}
            if nf not in database[ni][li]:
                database[ni][li][nf] = {}
            
            database[ni][li][nf][lf] = {
                'E_eV': E_eV,
                'sigma_cm2': sigma_cm2,
                'threshold_eV': threshold_eV,
                'filename': filename,
                'n_points': n_points
            }
            
            summary.append({
                'ni': ni,
                'li': li,
                'nf': nf,
                'lf': lf,
                'threshold_eV': threshold_eV,
                'n_points': n_points,
                'sigma_max_cm2': sigma_cm2.max(),
                'filename': filename
            })
            
            n_parsed += 1
            
            if n_parsed % 100 == 0:
                print(f"  Parsed {n_parsed} files...")
            
        except Exception as e:
            n_failed += 1
            continue
    
    print()
    print("=" * 70)
    print("PARSING COMPLETE")
    print("=" * 70)
    print(f"Successfully parsed: {n_parsed}")
    print(f"Failed to parse: {n_failed}")
    print(f"Total transitions: {n_parsed}")
    print()
    
    return database, summary


def save_database(database, summary, output_dir):
    """
    Save database to pickle file and summary to CSV.
    
    Args:
        database: Nested dictionary with cross sections
        summary: List of transition metadata
        output_dir: Directory to save files
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    db_file = output_path / 'ccc_database.pkl'
    with open(db_file, 'wb') as f:
        pickle.dump(database, f)
    print(f"Saved database: {db_file.relative_to(PATHS['root'])}")
    print(f"  Size: {os.path.getsize(db_file) / 1024 / 1024:.2f} MB")
    
    csv_file = output_path / 'ccc_transitions_summary.csv'
    with open(csv_file, 'w') as f:
        f.write("ni,li,nf,lf,threshold_eV,n_points,sigma_max_cm2,filename\n")
        
        for item in summary:
            f.write(f"{item['ni']},{item['li']},{item['nf']},{item['lf']},"
                   f"{item['threshold_eV']:.6f},{item['n_points']},"
                   f"{item['sigma_max_cm2']:.6e},{item['filename']}\n")
    
    print(f"Saved summary: {csv_file.relative_to(PATHS['root'])}")
    print(f"  Rows: {len(summary)}")
    print()


def print_database_stats(database, summary):
    """Print useful statistics about the database."""
    
    print("=" * 70)
    print("DATABASE STATISTICS")
    print("=" * 70)
    print()
    
    transitions_from_1s = 0
    transitions_from_2s = 0
    transitions_from_2p = 0
    
    for item in summary:
        if item['ni'] == 1 and item['li'] == 0:
            transitions_from_1s += 1
        elif item['ni'] == 2 and item['li'] == 0:
            transitions_from_2s += 1
        elif item['ni'] == 2 and item['li'] == 1:
            transitions_from_2p += 1
    
    print(f"Transitions from 1s (ground state): {transitions_from_1s}")
    print(f"Transitions from 2s (metastable): {transitions_from_2s}")
    print(f"Transitions from 2p: {transitions_from_2p}")
    print()
    
    print("Key transitions found:")
    print("-" * 50)
    
    key_transitions = [
        (1, 0, 2, 1, "1s -> 2p (Lyman alpha)"),
        (1, 0, 3, 1, "1s -> 3p"),
        (1, 0, 4, 1, "1s -> 4p"),
        (2, 0, 2, 1, "2s -> 2p (l-mixing)"),
        (2, 1, 3, 2, "2p -> 3d"),
        (2, 1, 3, 0, "2p -> 3s"),
    ]
    
    for ni, li, nf, lf, description in key_transitions:
        if (ni in database and li in database[ni] and 
            nf in database[ni][li] and lf in database[ni][li][nf]):
            data = database[ni][li][nf][lf]
            print(f"[FOUND] {description:25s} | "
                  f"Threshold: {data['threshold_eV']:7.3f} eV | "
                  f"Points: {data['n_points']:3d}")
        else:
            print(f"[ MISS] {description:25s} | NOT FOUND")
    
    print()


if __name__ == "__main__":
    
    print()
    print("=" * 70)
    print(" CCC DATABASE BUILDER ".center(70))
    print("=" * 70)
    print()
    
    # Use paths from config
    ccc_input = PATHS['ccc_raw']
    ccc_output = PATHS['ccc_processed']
    
    # Build database
    database, summary = build_ccc_database(ccc_input)
    
    # Print statistics
    print_database_stats(database, summary)
    
    # Save to disk
    save_database(database, summary, ccc_output)
    
    print("=" * 70)
    print("DATABASE BUILD COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Load database: pickle.load(open('ccc_database.pkl', 'rb'))")
    print("  2. Access data: database[ni][li][nf][lf]['sigma_cm2']")
    print("  3. Week 2 Task 2.2: Maxwellian averaging")
    print()