"""
Maxwellian Averaging for Electron-Impact Excitation Cross Sections

Converts CCC cross sections sigma(E) to rate coefficients K(Te).

Theory:
    K(Te) = sqrt(8/(pi*me)) * (1/(kTe)^(3/2)) * 
            integral[E * sigma(E) * exp(-E/kTe) dE]

Where:
    E = electron kinetic energy
    Te = electron temperature
    sigma(E) = excitation cross section
    K(Te) = rate coefficient [cm^3/s]

Author: Week 2, Task 2.2
Date: 2026-02-22
"""

import sys
import pickle
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config.paths import get_file, get_path, ensure_dir


# Physical constants
ME_KG = 9.1093837015e-31        # Electron mass [kg]
KB_J_K = 1.380649e-23           # Boltzmann constant [J/K]
EV_TO_J = 1.602176634e-19       # eV to Joules conversion
KB_EV_K = KB_J_K / EV_TO_J      # Boltzmann constant [eV/K]


def calculate_excitation_energy(ni, nf):
    """
    Calculate excitation energy for hydrogen transition.
    
    Args:
        ni: Initial principal quantum number
        nf: Final principal quantum number
    
    Returns:
        Delta_E: Excitation energy [eV]
    """
    RY_EV = 13.598  # Rydberg constant [eV]
    E_i = -RY_EV / (ni**2)
    E_f = -RY_EV / (nf**2)
    Delta_E = E_f - E_i
    return Delta_E


def maxwellian_integrand(E, sigma_interp, Te_eV):
    """
    Compute integrand for Maxwellian averaging.
    
    Integrand = E * sigma(E) * exp(-E/kTe)
    
    Args:
        E: Electron kinetic energy [eV]
        sigma_interp: Interpolated cross section function
        Te_eV: Electron temperature [eV]
    
    Returns:
        Integrand value
    """
    if E <= 0:
        return 0.0
    
    try:
        sigma = sigma_interp(E)
        if sigma < 0:
            sigma = 0.0
    except:
        return 0.0
    
    # Integrand: E * sigma(E) * exp(-E/kTe)
    integrand = E * sigma * np.exp(-E / Te_eV)
    
    return integrand


def maxwellian_average_single(E_eV, sigma_cm2, Te_eV, 
                              energy_mode='kinetic',
                              Delta_E=None):
    """
    Compute Maxwellian-averaged rate coefficient for single transition.
    
    Args:
        E_eV: Energy grid from CCC file [eV]
        sigma_cm2: Cross section [cm^2]
        Te_eV: Electron temperature [eV]
        energy_mode: 'kinetic' or 'corrected'
        Delta_E: Excitation energy [eV], required if energy_mode='corrected'
    
    Returns:
        K: Rate coefficient [cm^3/s]
    
    Theory:
        K(Te) = sqrt(8/(pi*me)) * (1/(kTe)^(3/2)) * 
                integral[E * sigma(E) * exp(-E/kTe) dE]
        
        Prefactor = sqrt(8/(pi*me*kTe)) * 1/kTe
    """
    
    # Handle energy convention
    if energy_mode == 'kinetic':
        # E_eV is already kinetic energy
        E_kinetic = E_eV.copy()
    elif energy_mode == 'corrected':
        # E_eV is energy above threshold, add Delta_E
        if Delta_E is None:
            raise ValueError("Delta_E required for corrected mode")
        E_kinetic = E_eV + Delta_E
    else:
        raise ValueError(f"Unknown energy_mode: {energy_mode}")
    
    # Remove any negative energies (shouldn't happen but be safe)
    mask = E_kinetic > 0
    E_kinetic = E_kinetic[mask]
    sigma_cm2 = sigma_cm2[mask]
    
    if len(E_kinetic) == 0:
        return 0.0
    
    # Create interpolator for cross section
    # Use log-log for better behavior, then convert back
    log_E = np.log10(E_kinetic + 1e-10)  # Avoid log(0)
    log_sigma = np.log10(sigma_cm2 + 1e-30)  # Avoid log(0)
    
    # Interpolator (extrapolate as constant outside range)
    log_sigma_interp = interp1d(log_E, log_sigma, 
                                kind='linear',
                                bounds_error=False,
                                fill_value=(log_sigma[0], log_sigma[-1]))
    
    def sigma_interp(E):
        """Interpolated cross section in linear space."""
        if E <= 0:
            return 0.0
        return 10**log_sigma_interp(np.log10(E + 1e-10))
    
    # Integration limits
    E_min = E_kinetic[0]
    E_max = max(E_kinetic[-1], 50 * Te_eV)  # Integrate to ~50 kTe
    
    # Numerical integration
    try:
        integral, error = integrate.quad(
            maxwellian_integrand,
            E_min, E_max,
            args=(sigma_interp, Te_eV),
            limit=100,
            epsrel=1e-4
        )
    except:
        # If integration fails, try simpler approach
        E_grid = np.logspace(np.log10(E_min), np.log10(E_max), 1000)
        integrand_vals = [maxwellian_integrand(E, sigma_interp, Te_eV) 
                         for E in E_grid]
        integral = np.trapz(integrand_vals, E_grid)
    
    # CORRECT formula from first principles
    #
    # K = <sigma × v> = ∫ sigma(E) × v(E) × f(E) dE
    #
    # where:
    #   v(E) = sqrt(2E/me) 
    #   f(E) = 2/sqrt(pi) × (1/Te)^(3/2) × sqrt(E) × exp(-E/Te)
    #
    # Substituting:
    # K = sqrt(8/(pi×me)) × (1/Te)^(3/2) × ∫ E × sigma(E) × exp(-E/Te) dE
    #
    # In CGS-eV units (Te in eV, E in eV, sigma in cm²):
    # 
    # Prefactor = sqrt(2*eV_to_J/me_kg) × 100 cm/m × 2/sqrt(pi)
    #           = 5.931e7 cm/s/sqrt(eV) × 1.1284
    #           = 6.692e7 cm·sqrt(eV)/s
    
    PREFACTOR_CGS_EV = 6.692e7  # cm·sqrt(eV)/s  [NOT 6.692e-7!]
    
    # Temperature factor
    temp_factor = Te_eV**(-1.5)  # eV^(-3/2)
    
    # Rate coefficient
    # Units: [cm·sqrt(eV)/s] × [1/eV^(3/2)] × [eV·cm²]
    #      = [cm³/s × eV^(1/2) / eV^(3/2) × eV]
    #      = [cm³/s]  ✓
    K_cm3_s = PREFACTOR_CGS_EV * temp_factor * integral
    
    return K_cm3_s


def compute_rate_coefficient_vs_temperature(E_eV, sigma_cm2, 
                                           Te_range_eV,
                                           energy_mode='kinetic',
                                           Delta_E=None):
    """
    Compute rate coefficient over a range of temperatures.
    
    Args:
        E_eV: Energy grid [eV]
        sigma_cm2: Cross section [cm^2]
        Te_range_eV: Array of temperatures [eV]
        energy_mode: 'kinetic' or 'corrected'
        Delta_E: Excitation energy [eV]
    
    Returns:
        K_array: Rate coefficients [cm^3/s]
    """
    
    K_array = np.zeros_like(Te_range_eV)
    
    for i, Te in enumerate(Te_range_eV):
        K_array[i] = maxwellian_average_single(
            E_eV, sigma_cm2, Te,
            energy_mode=energy_mode,
            Delta_E=Delta_E
        )
    
    return K_array


def load_and_compute_rates(database, ni, li, nf, lf, 
                           Te_range_eV,
                           energy_mode='kinetic'):
    """
    Load CCC data and compute rate coefficients.
    
    Args:
        database: CCC database dictionary
        ni, li: Initial state quantum numbers
        nf, lf: Final state quantum numbers
        Te_range_eV: Temperature range [eV]
        energy_mode: Energy handling mode
    
    Returns:
        Te_range_eV: Temperature grid [eV]
        K_array: Rate coefficients [cm^3/s]
        metadata: Dictionary with transition info
    """
    
    # Check if transition exists
    if not (ni in database and li in database[ni] and 
            nf in database[ni][li] and lf in database[ni][li][nf]):
        raise ValueError(f"Transition {ni},{li} -> {nf},{lf} not found")
    
    # Load data
    data = database[ni][li][nf][lf]
    E_eV = data['E_eV']
    sigma_cm2 = data['sigma_cm2']
    
    # Calculate excitation energy
    Delta_E = calculate_excitation_energy(ni, nf)
    
    # Compute rates
    K_array = compute_rate_coefficient_vs_temperature(
        E_eV, sigma_cm2, Te_range_eV,
        energy_mode=energy_mode,
        Delta_E=Delta_E
    )
    
    # Metadata
    metadata = {
        'ni': ni, 'li': li,
        'nf': nf, 'lf': lf,
        'Delta_E': Delta_E,
        'threshold_CCC': data['threshold_eV'],
        'filename': data['filename']
    }
    
    return Te_range_eV, K_array, metadata


def plot_rate_coefficient(Te_eV, K_cm3s, metadata, output_dir):
    """
    Plot rate coefficient vs temperature.
    """
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Convert Te to both eV and K for dual x-axis
    Te_K = Te_eV / KB_EV_K
    
    ax.loglog(Te_eV, K_cm3s, 'b-', linewidth=2.5)
    
    ax.set_xlabel('Electron Temperature (eV)', fontsize=13)
    ax.set_ylabel('Rate Coefficient (cm$^3$/s)', fontsize=13)
    
    transition_label = (f"{metadata['ni']}"
                       f"{'spdfgh'[metadata['li']]} -> "
                       f"{metadata['nf']}"
                       f"{'spdfgh'[metadata['lf']]}")
    
    ax.set_title(f'Excitation Rate Coefficient: {transition_label}',
                fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, which='both')
    
    # Add text box with info
    textstr = (f"Transition: {transition_label}\n"
              f"$\\Delta E$ = {metadata['Delta_E']:.3f} eV\n"
              f"CCC threshold = {metadata['threshold_CCC']:.3f} eV")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', bbox=props)
    
    # Mark your regime
    ax.axvspan(2, 10, alpha=0.15, color='cyan', 
              label='Your regime (2-10 eV)')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    filename = f"rate_coeff_{metadata['ni']}{chr(metadata['li']+115)}_to_{metadata['nf']}{chr(metadata['lf']+115)}.png"
    outfile = output_dir / filename
    plt.savefig(outfile, dpi=300)
    print(f"  Saved: {outfile.relative_to(project_root)}")
    
    plt.close()


if __name__ == "__main__":
    
    print("=" * 70)
    print(" MAXWELLIAN AVERAGING - TASK 2.2 ".center(70))
    print("=" * 70)
    print()
    
    # Load database
    print("Loading CCC database...")
    db_file = get_file('ccc_database_pkl')
    with open(db_file, 'rb') as f:
        database = pickle.load(f)
    print(f"  Loaded from: {db_file.name}")
    print()
    
    # Temperature range for your regime
    Te_range_eV = np.logspace(np.log10(0.2), np.log10(100), 100)
    
    # Output directory
    output_dir = ensure_dir('figures_week2')
    
    # Test transitions
    test_transitions = [
        (1, 0, 2, 1, "1s -> 2p (Lyman alpha)"),
        (1, 0, 3, 1, "1s -> 3p"),
        (2, 0, 2, 1, "2s -> 2p (l-mixing)"),
    ]
    
    print("Computing rate coefficients...")
    print()
    
    results = {}
    
    for ni, li, nf, lf, label in test_transitions:
        print(f"Processing: {label}")
        
        try:
            Te, K, metadata = load_and_compute_rates(
                database, ni, li, nf, lf,
                Te_range_eV,
                energy_mode='kinetic'  # Can change to 'corrected' later
            )
            
            results[(ni, li, nf, lf)] = (Te, K, metadata)
            
            # Print some values
            print(f"  Delta_E = {metadata['Delta_E']:.3f} eV")
            print(f"  K(2 eV) = {K[np.argmin(np.abs(Te-2))]:10.3e} cm^3/s")
            print(f"  K(5 eV) = {K[np.argmin(np.abs(Te-5))]:10.3e} cm^3/s")
            print(f"  K(10 eV) = {K[np.argmin(np.abs(Te-10))]:10.3e} cm^3/s")
            
            # Plot
            plot_rate_coefficient(Te, K, metadata, output_dir)
            
        except Exception as e:
            print(f"  Error: {e}")
        
        print()
    
    print("=" * 70)
    print("MAXWELLIAN AVERAGING COMPLETE")
    print("=" * 70)
    print()
    print("Generated rate coefficient plots for:")
    for label in [t[4] for t in test_transitions]:
        print(f"  - {label}")
    print()
    print("Next: Compare with literature values (Anderson et al.)")
    print()