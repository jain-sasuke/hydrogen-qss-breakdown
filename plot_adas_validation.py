import os
import numpy as np
import matplotlib.pyplot as plt

# Use your parser (the one that already works)
from src.parser_adasf11 import parse_adf11

# ----------------------------
# Constants (commented; you can verify exact later)
# ----------------------------
K_B_EV_PER_K = 8.617333262145e-5      # eV/K
K_B_SI = 1.380649e-23                 # J/K
H_SI = 6.62607015e-34                 # J*s
M_E = 9.1093837015e-31                # kg
E_ION_EV = 13.6                       # eV
E_ION_J = E_ION_EV * 1.602176634e-19  # J

OUTDIR = "figures/week1"
os.makedirs(OUTDIR, exist_ok=True)


def load_adf11(filepath):
    Z, IDMAXD, ITMAXD, logne, logTe, logK = parse_adf11(filepath)

    # Hydrogen file -> 1 block
    logK = logK[0]  # (ITMAXD, IDMAXD)

    Te_eV = 10 ** logTe
    ne_cm3 = 10 ** logne
    K = 10 ** logK  # cm^3/s

    return Te_eV, ne_cm3, K, logK


def saha_ratio_ni_over_n0(Te_eV, ne_cm3):
    """
    Saha for hydrogen (rough, LTE):  ni*ne/n0 = 2*(2*pi*m_e*kT/h^2)^(3/2)*exp(-Eion/kT)
    => ni/n0 = [2*(2*pi*m_e*kT/h^2)^(3/2)*exp(-Eion/kT)] / ne

    NOTE: ADAS ADF11 is CR-effective, not strict LTE. This comparison is only a sanity trend check.
    """
    Te_K = Te_eV / K_B_EV_PER_K
    ne_m3 = ne_cm3 * 1e6

    saha_pref = 2.0 * (2.0 * np.pi * M_E * K_B_SI * Te_K / (H_SI ** 2)) ** (1.5)
    saha = saha_pref * np.exp(-E_ION_J / (K_B_SI * Te_K))  # ni*ne/n0 in SI (m^-3)
    ni_over_n0 = saha / ne_m3
    return ni_over_n0


def plot_scd_vs_T(Te_eV, ne_cm3, SCD):
    idx = [0, 6, 12, 18, len(ne_cm3) - 1]
    plt.figure(figsize=(8, 6))
    for j in idx:
        plt.loglog(Te_eV, SCD[:, j], label=f"ne={ne_cm3[j]:.1e} cm^-3")

    plt.axvline(2, linestyle="--")
    plt.axvline(10, linestyle="--")

    plt.xlabel("Te (eV)")
    plt.ylabel("SCD (cm^3/s)")
    plt.title("ADAS SCD96 Hydrogen: Ionization vs Temperature")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot1_scd_vs_T.png", dpi=300)
    plt.close()


def plot_acd_vs_T(Te_eV, ne_cm3, ACD):
    idx = [0, 6, 12, 18, len(ne_cm3) - 1]
    plt.figure(figsize=(8, 6))
    for j in idx:
        plt.loglog(Te_eV, ACD[:, j], label=f"ne={ne_cm3[j]:.1e} cm^-3")

    plt.axvline(2, linestyle="--")
    plt.axvline(10, linestyle="--")

    plt.xlabel("Te (eV)")
    plt.ylabel("ACD (cm^3/s)")
    plt.title("ADAS ACD96 Hydrogen: Recombination vs Temperature")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot2_acd_vs_T.png", dpi=300)
    plt.close()


def plot_heatmaps(Te_eV, ne_cm3, logSCD, logACD):
    # pcolormesh expects edges-ish; using centers is ok with shading="auto"
    Te_grid, ne_grid = np.meshgrid(Te_eV, ne_cm3, indexing="ij")

    # SCD heatmap
    plt.figure(figsize=(9, 6))
    plt.pcolormesh(Te_grid, ne_grid, logSCD, shading="auto")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Te (eV)")
    plt.ylabel("ne (cm^-3)")
    plt.title("log10(SCD) heatmap")
    cb = plt.colorbar()
    cb.set_label("log10(cm^3/s)")
    # thesis box: Te 2-10, ne 1e13-1e15
    plt.plot([2, 10, 10, 2, 2], [1e13, 1e13, 1e15, 1e15, 1e13], linewidth=2)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot3a_heatmap_logSCD.png", dpi=300)
    plt.close()

    # ACD heatmap
    plt.figure(figsize=(9, 6))
    plt.pcolormesh(Te_grid, ne_grid, logACD, shading="auto")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Te (eV)")
    plt.ylabel("ne (cm^-3)")
    plt.title("log10(ACD) heatmap")
    cb = plt.colorbar()
    cb.set_label("log10(cm^3/s)")
    plt.plot([2, 10, 10, 2, 2], [1e13, 1e13, 1e15, 1e15, 1e13], linewidth=2)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot3b_heatmap_logACD.png", dpi=300)
    plt.close()


def plot_ratio_scd_over_acd_vs_saha(Te_eV, ne_cm3, SCD, ACD):
    # pick one density (middle of your thesis regime is good)
    # choose closest to 1e14
    j = int(np.argmin(np.abs(np.log10(ne_cm3) - 14.0)))
    ne0 = ne_cm3[j]

    ratio_rates = SCD[:, j] / ACD[:, j]  # should correspond to ni/n0 at LTE balance
    ratio_saha = saha_ratio_ni_over_n0(Te_eV, ne0)

    plt.figure(figsize=(8, 6))
    plt.loglog(Te_eV, ratio_rates, label=f"SCD/ACD @ ne={ne0:.1e}")
    plt.loglog(Te_eV, ratio_saha, label="Saha ni/n0 (LTE, rough)")

    plt.axvline(2, linestyle="--")
    plt.axvline(10, linestyle="--")

    plt.xlabel("Te (eV)")
    plt.ylabel("Ratio")
    plt.title("SCD/ACD vs Saha (trend sanity check)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/plot4_ratio_scd_acd_vs_saha.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Load from raw ADF11 (not CSV) -> avoids CSV parsing issues
    Te_eV, ne_cm3, SCD, logSCD = load_adf11("data/raw/adas/scd96_h.dat")
    Te_eV2, ne_cm3_2, ACD, logACD = load_adf11("data/raw/adas/acd96_h.dat")

    # basic consistency
    assert np.allclose(Te_eV, Te_eV2)
    assert np.allclose(ne_cm3, ne_cm3_2)

    # Plot 1 & 2
    plot_scd_vs_T(Te_eV, ne_cm3, SCD)
    plot_acd_vs_T(Te_eV, ne_cm3, ACD)

    # Plot 3
    plot_heatmaps(Te_eV, ne_cm3, logSCD, logACD)

    # Plot 4
    plot_ratio_scd_over_acd_vs_saha(Te_eV, ne_cm3, SCD, ACD)

    print(f"Saved plots to: {OUTDIR}/")




