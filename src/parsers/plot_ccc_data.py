"""
CCC Cross Section Visualization

Creates publication-quality plots of CCC excitation cross sections.

Generates 3 figures:
1. Comparison of 1s -> np transitions (n=2,3,4,5)
2. Near-threshold detail for 1s -> 2p
3. Metastable vs ground state: 2s->2p vs 1s->2p

Usage:
    python src/parsers/plot_ccc_data.py

Output:
    figures/week2/ccc_*.png

Author: Week 2, Task 2.1
Date: 2026-02-22
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config.paths import get_file, get_path, ensure_dir


def load_database():
    """Load CCC database from pickle file."""
    db_file = get_file('ccc_database_pkl')
    print(f"Loading database: {db_file.name}")
    
    with open(db_file, 'rb') as f:
        database = pickle.load(f)
    
    print(f"  Loaded {sum(len(database[ni][li][nf]) for ni in database for li in database[ni] for nf in database[ni][li])} transitions")
    return database


def plot_1s_np_comparison(database, output_dir):
    """
    Plot 1: Compare 1s -> np cross sections for n=2,3,4,5.
    
    Shows how cross section decreases with increasing n.
    """
    
    print("\nCreating Plot 1: 1s -> np comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Transitions to plot: 1s -> np for n=2,3,4,5
    transitions = [
        (2, 'Lyman alpha (1s->2p)', 'blue', '-'),
        (3, '1s->3p', 'red', '--'),
        (4, '1s->4p', 'green', '-.'),
        (5, '1s->5p', 'orange', ':'),
    ]
    
    for n, label, color, linestyle in transitions:
        # Check if transition exists
        if 1 in database and 0 in database[1] and n in database[1][0] and 1 in database[1][0][n]:
            data = database[1][0][n][1]
            E = data['E_eV']
            sigma = data['sigma_cm2']
            
            # Convert to more readable units (a0^2)
            a0_cm = 5.29177e-9
            sigma_a0sq = sigma / (a0_cm ** 2)
            
            ax.loglog(E, sigma_a0sq, color=color, linestyle=linestyle, 
                     linewidth=2, label=label)
            
            print(f"  Plotted: 1s -> {n}p")
        else:
            print(f"  Missing: 1s -> {n}p")
    
    ax.set_xlabel('Energy above threshold (eV)', fontsize=13)
    ax.set_ylabel('Cross section (a₀²)', fontsize=13)
    ax.set_title('Ground State Excitation: 1s → np', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='best')
    
    # Mark your regime
    ax.axvspan(0, 10, alpha=0.1, color='cyan', label='Your regime (E<10 eV)')
    
    plt.tight_layout()
    outfile = output_dir / 'ccc_1s_np_comparison.png'
    plt.savefig(outfile, dpi=300)
    print(f"  Saved: {outfile.relative_to(project_root)}")
    plt.close()


def plot_1s2p_threshold(database, output_dir):
    """
    Plot 2: Near-threshold detail for 1s -> 2p.
    
    Shows resonance structure near threshold (important for Maxwellian averaging).
    """
    
    print("\nCreating Plot 2: 1s->2p near-threshold detail...")
    
    if not (1 in database and 0 in database[1] and 2 in database[1][0] and 1 in database[1][0][2]):
        print("  Error: 1s->2p not found!")
        return
    
    data = database[1][0][2][1]
    E = data['E_eV']
    sigma = data['sigma_cm2']
    threshold = data['threshold_eV']
    
    # Convert to a0^2
    a0_cm = 5.29177e-9
    sigma_a0sq = sigma / (a0_cm ** 2)
    
    # Focus on E < 50 eV above threshold
    mask = E < 50.0
    E_zoom = E[mask]
    sigma_zoom = sigma_a0sq[mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Linear scale
    ax1.plot(E_zoom, sigma_zoom, 'b-', linewidth=2)
    ax1.set_xlabel('Energy above threshold (eV)', fontsize=12)
    ax1.set_ylabel('Cross section (a₀²)', fontsize=12)
    ax1.set_title('1s → 2p: Linear Scale', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(0, 10, alpha=0.15, color='cyan', label='Te=2-10 eV regime')
    ax1.legend(fontsize=10)
    
    # Add annotation for threshold behavior
    ax1.annotate(f'Threshold: {threshold:.3f} eV', 
                xy=(0, sigma_zoom[0]), xytext=(15, sigma_zoom[0]*0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red')
    
    # Right: Log-log scale
    ax2.loglog(E_zoom, sigma_zoom, 'b-', linewidth=2)
    ax2.set_xlabel('Energy above threshold (eV)', fontsize=12)
    ax2.set_ylabel('Cross section (a₀²)', fontsize=12)
    ax2.set_title('1s → 2p: Log Scale', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.axvspan(0.001, 10, alpha=0.15, color='cyan')
    
    plt.tight_layout()
    outfile = output_dir / 'ccc_1s2p_threshold_detail.png'
    plt.savefig(outfile, dpi=300)
    print(f"  Saved: {outfile.relative_to(project_root)}")
    plt.close()


def plot_metastable_vs_ground(database, output_dir):
    """
    Plot 3: Compare 2s->2p (metastable) vs 1s->2p (ground state).
    
    Shows that metastable excitation has MUCH larger cross section.
    """
    
    print("\nCreating Plot 3: Metastable vs ground state...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 1s -> 2p (ground state excitation)
    if 1 in database and 0 in database[1] and 2 in database[1][0] and 1 in database[1][0][2]:
        data_1s2p = database[1][0][2][1]
        E_1s = data_1s2p['E_eV']
        sigma_1s = data_1s2p['sigma_cm2']
        thresh_1s = data_1s2p['threshold_eV']
        
        a0_cm = 5.29177e-9
        sigma_1s_a0 = sigma_1s / (a0_cm ** 2)
        
        ax.loglog(E_1s, sigma_1s_a0, 'b-', linewidth=2.5, 
                 label=f'1s->2p (Threshold: {thresh_1s:.2f} eV)')
        print("  Plotted: 1s->2p")
    
    # 2s -> 2p (metastable l-mixing)
    if 2 in database and 0 in database[2] and 2 in database[2][0] and 1 in database[2][0][2]:
        data_2s2p = database[2][0][2][1]
        E_2s = data_2s2p['E_eV']
        sigma_2s = data_2s2p['sigma_cm2']
        thresh_2s = data_2s2p['threshold_eV']
        
        sigma_2s_a0 = sigma_2s / (a0_cm ** 2)
        
        ax.loglog(E_2s, sigma_2s_a0, 'r--', linewidth=2.5, 
                 label=f'2s->2p (Threshold: {thresh_2s:.2f} eV)')
        print("  Plotted: 2s->2p")
        
        # Calculate ratio at a few points
        print("\n  Cross section ratio (2s->2p / 1s->2p):")
        test_energies = [0.01, 0.1, 1.0, 10.0]
        for E_test in test_energies:
            idx_1s = np.argmin(np.abs(E_1s - E_test))
            idx_2s = np.argmin(np.abs(E_2s - E_test))
            ratio = sigma_2s_a0[idx_2s] / sigma_1s_a0[idx_1s]
            print(f"    E = {E_test:6.2f} eV: ratio = {ratio:8.1f}x larger")
    
    ax.set_xlabel('Energy above threshold (eV)', fontsize=13)
    ax.set_ylabel('Cross section (a₀²)', fontsize=13)
    ax.set_title('Metastable vs Ground State Excitation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12, loc='best')
    
    # Add text box explaining
    textstr = 'Metastable 2s has MUCH larger\ncross section due to smaller\nenergy gap (l-mixing)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    outfile = output_dir / 'ccc_metastable_vs_ground.png'
    plt.savefig(outfile, dpi=300)
    print(f"  Saved: {outfile.relative_to(project_root)}")
    plt.close()


def create_summary_table(database, output_dir):
    """
    Create a summary table of key transitions.
    """
    
    print("\nCreating summary table...")
    
    key_transitions = [
        (1, 0, 2, 1, "1s -> 2p"),
        (1, 0, 3, 1, "1s -> 3p"),
        (1, 0, 4, 1, "1s -> 4p"),
        (1, 0, 5, 1, "1s -> 5p"),
        (2, 0, 2, 1, "2s -> 2p"),
        (2, 1, 3, 2, "2p -> 3d"),
        (2, 1, 3, 0, "2p -> 3s"),
        (3, 1, 4, 2, "3p -> 4d"),
    ]
    
    outfile = output_dir / 'ccc_key_transitions.txt'
    
    with open(outfile, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CCC KEY TRANSITIONS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Transition':<15s} | {'Threshold (eV)':<15s} | {'Points':<8s} | "
                f"{'Sigma_max (cm^2)':<18s} | {'Sigma_max (a0^2)':<15s}\n")
        f.write("-" * 80 + "\n")
        
        a0_cm = 5.29177e-9
        
        for ni, li, nf, lf, label in key_transitions:
            if (ni in database and li in database[ni] and 
                nf in database[ni][li] and lf in database[ni][li][nf]):
                data = database[ni][li][nf][lf]
                thresh = data['threshold_eV']
                n_pts = data['n_points']
                sigma_max = data['sigma_cm2'].max()
                sigma_max_a0 = sigma_max / (a0_cm ** 2)
                
                f.write(f"{label:<15s} | {thresh:15.4f} | {n_pts:<8d} | "
                       f"{sigma_max:18.3e} | {sigma_max_a0:15.3e}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"  Saved: {outfile.relative_to(project_root)}")


if __name__ == "__main__":
    
    print("=" * 70)
    print(" CCC DATA VISUALIZATION ".center(70))
    print("=" * 70)
    
    # Load database
    database = load_database()
    
    # Ensure output directory exists
    output_dir = ensure_dir('figures_week2')
    print(f"\nOutput directory: {output_dir.relative_to(project_root)}")
    
    # Create plots
    plot_1s_np_comparison(database, output_dir)
    plot_1s2p_threshold(database, output_dir)
    plot_metastable_vs_ground(database, output_dir)
    
    # Create summary table
    create_summary_table(database, output_dir)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. ccc_1s_np_comparison.png      - Compare 1s->2p,3p,4p,5p")
    print("  2. ccc_1s2p_threshold_detail.png - Near-threshold detail")
    print("  3. ccc_metastable_vs_ground.png  - 2s->2p vs 1s->2p")
    print("  4. ccc_key_transitions.txt       - Summary table")
    print()
    print("Ready for Task 2.2: Maxwellian Averaging")
    print()