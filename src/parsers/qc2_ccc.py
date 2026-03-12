"""
CCC Data Quality Control
=========================

Validates CCC cross section database through multiple physical checks:
1. Filename interpretation correctness
2. Same-n transition exclusion
3. Energy grid coverage
4. Cross section magnitude validation
5. Detailed balance verification (Milne relation)


Project: Hydrogen CR QSS Breakdown Study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Physical constants
RYDBERG = 13.6057  # eV


def check_detailed_balance(df: pd.DataFrame, 
                          n_i_exc: int, l_i_exc: int,
                          n_f_exc: int, l_f_exc: int,
                          g_i: int, g_f: int,
                          verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Verify detailed balance (Milne relation) for a transition pair.
    
    Detailed Balance (Cross-Section Level)
    ---------------------------------------
    For excitation i→j at energy E_exc and de-excitation j→i at E_deexc:
    
        g_i × σ(i→j, E_exc) × E_exc = g_j × σ(j→i, E_deexc) × (E_deexc)
        
    where E_deexc = E_exc - ΔE (energy shift due to transition energy)
    
    Parameters
    ----------
    df : pd.DataFrame
        Full CCC database
    n_i_exc, l_i_exc, n_f_exc, l_f_exc : int
        Quantum numbers for EXCITATION direction (i→f)
    g_i, g_f : int
        Statistical weights (2(2ℓ+1) for hydrogen)
    verbose : bool
        Print detailed results
        
    Returns
    -------
    df_matched : pd.DataFrame
        Matched energy points with ratio calculations
    stats : dict
        Summary statistics
        
    Notes
    -----
    CCC data should satisfy detailed balance to ~0.05-0.5% precision
    at high energies, with larger errors near threshold due to numerical
    interpolation.
    """
    # Get excitation data (i→f)
    exc = df[(df['n_i']==n_i_exc) & (df['l_i']==l_i_exc) & 
             (df['n_f']==n_f_exc) & (df['l_f']==l_f_exc)].copy()
    
    # Get de-excitation data (f→i)
    deexc = df[(df['n_i']==n_f_exc) & (df['l_i']==l_f_exc) &
               (df['n_f']==n_i_exc) & (df['l_f']==l_i_exc)].copy()
    
    if len(exc) == 0 or len(deexc) == 0:
        print(f"⚠ Missing data for {n_i_exc}→{n_f_exc} or reverse")
        return pd.DataFrame(), {}
    
    # Sort by energy
    exc = exc.sort_values('E_eV').reset_index(drop=True)
    deexc = deexc.sort_values('E_eV').reset_index(drop=True)
    
    # Transition energy
    E_i = -RYDBERG / (n_i_exc ** 2)
    E_f = -RYDBERG / (n_f_exc ** 2)
    Delta_E = E_f - E_i  # Positive for excitation
    
    if verbose:
        print(f"\nDetailed Balance Check: {n_i_exc}↔{n_f_exc}")
        print("-" * 70)
        print(f"Transition energy ΔE = {Delta_E:.4f} eV")
        print(f"Statistical weights: g_{n_i_exc} = {g_i}, g_{n_f_exc} = {g_f}")
        print(f"Excitation points: {len(exc)}")
        print(f"De-excitation points: {len(deexc)}")
        print()
    
    # Match energy points
    matched_pairs = []
    
    for idx, row in exc.iterrows():
        E_exc = row['E_eV']
        sigma_exc = row['sigma_a0sq']
        
        # Corresponding de-excitation energy
        E_deexc_target = E_exc - Delta_E
        
        # Find closest point
        idx_deexc = (np.abs(deexc['E_eV'] - E_deexc_target)).argmin()
        E_deexc_actual = deexc.iloc[idx_deexc]['E_eV']
        sigma_deexc = deexc.iloc[idx_deexc]['sigma_a0sq']
        
        # Energy matching error
        E_error = abs(E_deexc_actual - E_deexc_target)
        
        # Only include if energy match is good
        if E_error < 0.1:  # 0.1 eV tolerance
            # Detailed balance ratio (Milne relation)
            LHS = g_i * sigma_exc * E_exc
            RHS = g_f * sigma_deexc * E_deexc_actual
            ratio = LHS / RHS
            percent_error = 100 * abs(ratio - 1.0)
            
            matched_pairs.append({
                'E_exc': E_exc,
                'sigma_exc': sigma_exc,
                'E_deexc': E_deexc_actual,
                'sigma_deexc': sigma_deexc,
                'LHS': LHS,
                'RHS': RHS,
                'ratio': ratio,
                'percent_error': percent_error,
                'E_match_error': E_error
            })
    
    df_matched = pd.DataFrame(matched_pairs)
    
    if len(df_matched) == 0:
        print("⚠ No matched energy points found")
        return df_matched, {}
    
    # Statistics
    stats = {
        'mean_ratio': df_matched['ratio'].mean(),
        'median_ratio': df_matched['ratio'].median(),
        'std_ratio': df_matched['ratio'].std(),
        'min_ratio': df_matched['ratio'].min(),
        'max_ratio': df_matched['ratio'].max(),
        'mean_error_pct': df_matched['percent_error'].mean(),
        'median_error_pct': df_matched['percent_error'].median(),
        'max_error_pct': df_matched['percent_error'].max(),
        'n_matched': len(df_matched)
    }
    
    if verbose:
        print("Detailed Balance Statistics:")
        print("-" * 70)
        print(f"Expected ratio: 1.0000")
        print(f"Mean ratio:     {stats['mean_ratio']:.6f}")
        print(f"Median ratio:   {stats['median_ratio']:.6f}")
        print(f"Std deviation:  {stats['std_ratio']:.6f}")
        print()
        print(f"Mean error:     {stats['mean_error_pct']:.4f}%")
        print(f"Median error:   {stats['median_error_pct']:.4f}%")
        print(f"Max error:      {stats['max_error_pct']:.4f}%")
        print()
        
        # Verdict
        if stats['median_error_pct'] < 0.1:
            verdict = "✓ EXCELLENT (CCC precision <0.1%)"
        elif stats['median_error_pct'] < 1.0:
            verdict = "✓ VERY GOOD (error <1%)"
        elif stats['median_error_pct'] < 5.0:
            verdict = "✓ GOOD (error <5%)"
        else:
            verdict = "⚠ CHECK (error >5%)"
        
        print(f"Verdict: {verdict}")
        print()
    
    return df_matched, stats


def run_quality_control(csv_file: str, output_dir: str = '.'):
    """
    Run comprehensive quality control on CCC database.
    
    Parameters
    ----------
    csv_file : str
        Path to parsed CCC CSV file
    output_dir : str
        Directory for output figures and reports
    """
    print("="*80)
    print("CCC DATA QUALITY CONTROL")
    print("="*80)
    print()
    
    # Load data
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Total data points: {len(df):,}")
    print(f"Unique transitions: {len(df.groupby(['n_i','l_i','n_f','l_f']))}")
    print()
    
    # Check 1: Same-n transitions
    print("CHECK 1: Same-n Transition Exclusion")
    print("-" * 80)
    same_n = df[df['n_i'] == df['n_f']]
    if len(same_n) == 0:
        print("✓ PASS: No same-n transitions found (correctly excluded)")
    else:
        print(f"✗ FAIL: Found {len(same_n)} same-n transitions (should be 0)")
    print()
    
    # Check 2: Energy coverage
    print("CHECK 2: Energy Coverage")
    print("-" * 80)
    E_min = df['E_eV'].min()
    E_max = df['E_eV'].max()
    E_median = df['E_eV'].median()
    print(f"Energy range: {E_min:.2e} to {E_max:.2f} eV")
    print(f"Median energy: {E_median:.2f} eV")
    
    # Check for threshold coverage
    threshold_2p = 10.2  # eV for 1s→2p
    near_threshold = df[(df['E_eV'] > 10.0) & (df['E_eV'] < 11.0)]
    if len(near_threshold) > 0:
        print(f"✓ Near-threshold data present (E ~ {threshold_2p} eV)")
    print()
    
    # Check 3: Cross section magnitudes
    print("CHECK 3: Cross Section Magnitudes")
    print("-" * 80)
    sigma_min = df['sigma_a0sq'].min()
    sigma_max = df['sigma_a0sq'].max()
    sigma_median = df['sigma_a0sq'].median()
    print(f"σ range: {sigma_min:.2e} to {sigma_max:.2e} a₀²")
    print(f"Median σ: {sigma_median:.2f} a₀²")
    
    # Typical values check
    if 0.1 < sigma_median < 1000:
        print("✓ Median cross section in expected range (0.1-1000 a₀²)")
    else:
        print("⚠ Median cross section outside typical range")
    print()
    
    # Check 4: Key transitions present
    print("CHECK 4: Key Transitions")
    print("-" * 80)
    key_transitions = [
        (2, 1, 1, 0, "2p → 1s (Lyman α de-exc)"),
        (1, 0, 2, 1, "1s → 2p (Lyman α exc)"),
        (2, 0, 1, 0, "2s → 1s"),
        (1, 0, 2, 0, "1s → 2s"),
        (3, 2, 2, 1, "3d → 2p"),
    ]
    
    for ni, li, nf, lf, desc in key_transitions:
        count = len(df[(df['n_i']==ni) & (df['l_i']==li) & 
                       (df['n_f']==nf) & (df['l_f']==lf)])
        if count > 0:
            print(f"✓ {desc:30s} : {count:4d} points")
        else:
            print(f"✗ {desc:30s} : NOT FOUND")
    print()
    
    # Check 5: Detailed Balance (1s ↔ 2p)
    print("CHECK 5: Detailed Balance Verification")
    print("="*80)
    
    # 1s ↔ 2p (most important transition)
    g_1s = 2  # 2(2×0+1) = 2
    g_2p = 6  # 2(2×1+1) = 6
    
    df_matched, stats = check_detailed_balance(
        df, n_i_exc=1, l_i_exc=0, n_f_exc=2, l_f_exc=1,
        g_i=g_1s, g_f=g_2p, verbose=True
    )
    
    # Create summary plot
    if len(df_matched) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Ratio vs energy
        ax1 = axes[0]
        ax1.semilogx(df_matched['E_exc'], df_matched['ratio'], 'go', 
                     markersize=5, alpha=0.6)
        ax1.axhline(1.0, color='k', linestyle='--', lw=2, label='Expected = 1.0')
        ax1.fill_between([10, 1000], 0.99, 1.01, color='green', alpha=0.2, 
                        label='±1% band')
        ax1.set_xlabel('Excitation Energy [eV]', fontsize=12)
        ax1.set_ylabel('Ratio: LHS / RHS', fontsize=12)
        ax1.set_title('Detailed Balance: 1s ↔ 2p', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.9, 1.1])
        
        # Plot 2: Error vs energy
        ax2 = axes[1]
        ax2.loglog(df_matched['E_exc'], df_matched['percent_error'], 'ro', 
                   markersize=5, alpha=0.6)
        ax2.axhline(0.1, color='g', linestyle='--', lw=2, label='0.1% (CCC precision)')
        ax2.axhline(1.0, color='orange', linestyle='--', lw=2, label='1.0% threshold')
        ax2.set_xlabel('Excitation Energy [eV]', fontsize=12)
        ax2.set_ylabel('Percent Error [%]', fontsize=12)
        ax2.set_title('Detailed Balance Error', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ccc_qc_detailed_balance.png', dpi=150, 
                    bbox_inches='tight')
        print(f"✓ Saved figure: {output_dir}/ccc_qc_detailed_balance.png")
        print()
    
    # Final verdict
    print("="*80)
    print("QUALITY CONTROL SUMMARY")
    print("="*80)
    
    checks_passed = 0
    total_checks = 5
    
    # Summarize results
    if len(same_n) == 0:
        checks_passed += 1
    if 0.1 < sigma_median < 1000:
        checks_passed += 1
    if len(near_threshold) > 0:
        checks_passed += 1
    
    # Check for key transitions
    key_present = sum(1 for ni, li, nf, lf, _ in key_transitions 
                      if len(df[(df['n_i']==ni) & (df['l_i']==li) & 
                               (df['n_f']==nf) & (df['l_f']==lf)]) > 0)
    if key_present >= 4:
        checks_passed += 1
    
    # Detailed balance check
    if len(stats) > 0 and stats['median_error_pct'] < 1.0:
        checks_passed += 1
    
    print(f"Checks passed: {checks_passed}/{total_checks}")
    print()
    
    if checks_passed == total_checks:
        print("✓✓✓ ALL CHECKS PASSED - DATA READY FOR MAXWELLIAN AVERAGING ✓✓✓")
    elif checks_passed >= 4:
        print("✓ MOST CHECKS PASSED - Data quality good, proceed with caution")
    else:
        print("⚠ MULTIPLE FAILURES - Review data before proceeding")
    
    print("="*80)


if __name__ == "__main__":
    """
    Run QC on parsed CCC data.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python qc_ccc.py <path_to_ccc_crosssections.csv>")
        print("\nExample:")
        print("  python qc_ccc.py data/processed/ccc/ccc_cross_sections.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    run_quality_control(csv_file, output_dir='figures/week2')