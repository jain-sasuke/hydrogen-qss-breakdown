"""
Verify CCC Database Structure

Quick checks to ensure the database loaded correctly.

Usage:
    python src/parsers/verify_ccc_database.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config.paths import get_file


def verify_database():
    """
    Load and verify CCC database structure.
    """
    
    print("=" * 70)
    print("CCC DATABASE VERIFICATION")
    print("=" * 70)
    print()
    
    # Load database
    db_file = get_file('ccc_database_pkl')
    
    print(f"Loading: {db_file.relative_to(project_root)}")
    
    with open(db_file, 'rb') as f:
        database = pickle.load(f)
    
    print("Database loaded successfully!")
    print()
    
    # Check structure
    print("=" * 70)
    print("DATABASE STRUCTURE")
    print("=" * 70)
    print()
    
    total_transitions = 0
    for ni in database:
        for li in database[ni]:
            for nf in database[ni][li]:
                for lf in database[ni][li][nf]:
                    total_transitions += 1
    
    print(f"Total transitions: {total_transitions}")
    print(f"Initial n values: {sorted(database.keys())}")
    print()
    
    # Check 1s -> 2p in detail
    print("=" * 70)
    print("DETAILED CHECK: 1s -> 2p (Lyman alpha)")
    print("=" * 70)
    print()
    
    if 1 in database and 0 in database[1] and 2 in database[1][0] and 1 in database[1][0][2]:
        data = database[1][0][2][1]
        
        print(f"Transition: 1s -> 2p")
        print(f"Filename: {data['filename']}")
        print(f"Threshold: {data['threshold_eV']:.4f} eV")
        print(f"Data points: {data['n_points']}")
        print()
        
        E = data['E_eV']
        sigma = data['sigma_cm2']
        
        print("Energy grid:")
        print(f"  Min: {E.min():.6f} eV (above threshold)")
        print(f"  Max: {E.max():.2f} eV (above threshold)")
        print(f"  Shape: {E.shape}")
        print()
        
        print("Cross section:")
        print(f"  Min: {sigma.min():.3e} cm^2")
        print(f"  Max: {sigma.max():.3e} cm^2")
        print(f"  At threshold: {sigma[0]:.3e} cm^2")
        print(f"  Shape: {sigma.shape}")
        print()
        
        # Check near your regime
        # Te = 2-10 eV means electrons have E ~ 0-10 eV above threshold
        mask_regime = E <= 10.0
        if mask_regime.any():
            E_regime = E[mask_regime]
            sigma_regime = sigma[mask_regime]
            print(f"In your regime (E < 10 eV above threshold):")
            print(f"  Energy points: {len(E_regime)}")
            print(f"  Sigma range: {sigma_regime.min():.3e} - {sigma_regime.max():.3e} cm^2")
        print()
        
    else:
        print("ERROR: 1s -> 2p transition not found!")
        print()
    
    # Check a few more transitions
    print("=" * 70)
    print("SAMPLE TRANSITIONS")
    print("=" * 70)
    print()
    
    sample_transitions = [
        (1, 0, 3, 1, "1s -> 3p"),
        (1, 0, 4, 1, "1s -> 4p"),
        (2, 0, 2, 1, "2s -> 2p"),
        (2, 1, 3, 2, "2p -> 3d"),
    ]
    
    for ni, li, nf, lf, label in sample_transitions:
        if (ni in database and li in database[ni] and 
            nf in database[ni][li] and lf in database[ni][li][nf]):
            data = database[ni][li][nf][lf]
            print(f"[OK] {label:15s} | Threshold: {data['threshold_eV']:7.3f} eV | "
                  f"Points: {data['n_points']:3d} | "
                  f"Sigma_max: {data['sigma_cm2'].max():.2e} cm^2")
        else:
            print(f"[MISS] {label:15s} | NOT FOUND")
    
    print()
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print()
    
    return database


if __name__ == "__main__":
    database = verify_database()
    
    print("Database is ready for use!")
    print()
    print("Example usage:")
    print("  data = database[1][0][2][1]  # 1s -> 2p")
    print("  E_eV = data['E_eV']")
    print("  sigma_cm2 = data['sigma_cm2']")
    print()