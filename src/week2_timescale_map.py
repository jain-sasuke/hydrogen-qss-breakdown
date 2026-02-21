"""
Week 2 - Task 2.1
Ionization fraction relaxation timescale tau(Te, ne) using ADAS ADF11 SCD/ACD.

Physics:
  dx/dt = ne * [ S(Te,ne) * (1-x) - alpha(Te,ne) * x ]

Closed-form:
  tau = 1 / (ne * (S + alpha))
  x_inf = S / (S + alpha)

Units:
  Te input: eV
  ne input: cm^-3
  S, alpha: cm^3/s
  tau: seconds

Notes:
- No constants needed besides unit conversions for plotting labels.
- If you later want Te in K: Te[K] = Te[eV] / kB_eV_per_K
  where kB_eV_per_K ~ 8.61733e-5 (you can verify later).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adas_interpolator import ADASRateInterpolator  # uses your existing class


def tau_and_xinf(Te_eV: float, ne_cm3: float, scd, acd):
    """Return (S, alpha, tau_s, x_inf) at a single (Te, ne)."""
    S = float(scd(Te_eV, ne_cm3))      # cm^3/s
    alpha = float(acd(Te_eV, ne_cm3))  # cm^3/s

    # Guard (should not happen if data is sane)
    denom = ne_cm3 * (S + alpha)
    if denom <= 0:
        tau = np.nan
        x_inf = np.nan
    else:
        tau = 1.0 / denom
        x_inf = S / (S + alpha)

    return S, alpha, tau, x_inf


def main():
    # Paths (relative to repo root when running `python src/week2_timescale_map.py`)
    scd_path = "data/raw/adas/scd96_h.dat"
    acd_path = "data/raw/adas/acd96_h.dat"

    # Build interpolators
    scd = ADASRateInterpolator(scd_path)  # SCD
    acd = ADASRateInterpolator(acd_path)  # ACD

    # Thesis regime grid (you can densify later)
    Te_list = np.array([2, 3, 5, 8, 10], dtype=float)  # eV
    ne_list = np.array([1e12, 1e13, 1e14, 1e15, 2e15], dtype=float)  # cm^-3

    rows = []
    for Te in Te_list:
        for ne in ne_list:
            S, alpha, tau, x_inf = tau_and_xinf(Te, ne, scd, acd)
            rows.append({
                "Te_eV": Te,
                "ne_cm3": ne,
                "SCD_cm3_s": S,
                "ACD_cm3_s": alpha,
                "tau_s": tau,
                "x_inf": x_inf
            })

    df = pd.DataFrame(rows)

    # Save table
    out_csv = "results/week2_tau_map_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"Written: {out_csv}")

    # --- Heatmap for tau ---
    # Reshape to (len(ne), len(Te)) for plotting
    tau_grid = df.pivot(index="ne_cm3", columns="Te_eV", values="tau_s").values
    xinf_grid = df.pivot(index="ne_cm3", columns="Te_eV", values="x_inf").values

    # Plot tau heatmap (log scale)
    plt.figure(figsize=(8, 5))
    plt.imshow(np.log10(tau_grid), aspect="auto", origin="lower")
    plt.xticks(np.arange(len(Te_list)), Te_list)
    plt.yticks(np.arange(len(ne_list)), [f"{n:.0e}" for n in ne_list])
    plt.xlabel("Te (eV)")
    plt.ylabel("ne (cm^-3)")
    plt.title("log10(tau [s])  where tau = 1/(ne*(SCD+ACD))")
    plt.colorbar(label="log10(tau_s)")
    plt.tight_layout()
    tau_png = "figures/week2/week2_tau_heatmap.png"
    plt.savefig(tau_png, dpi=300)
    plt.close()
    print(f"Written: {tau_png}")

    # Plot x_inf heatmap (linear)
    plt.figure(figsize=(8, 5))
    plt.imshow(xinf_grid, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    plt.xticks(np.arange(len(Te_list)), Te_list)
    plt.yticks(np.arange(len(ne_list)), [f"{n:.0e}" for n in ne_list])
    plt.xlabel("Te (eV)")
    plt.ylabel("ne (cm^-3)")
    plt.title("x_inf = SCD/(SCD+ACD)")
    plt.colorbar(label="x_inf")
    plt.tight_layout()
    xinf_png = "figures/week2/week2_xinf_heatmap.png"
    plt.savefig(xinf_png, dpi=300)
    plt.close()
    print(f"Written: {xinf_png}")

    # Print a quick “sanity” slice
    print("\nSample rows:")
    print(df.sort_values(["ne_cm3", "Te_eV"]).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
