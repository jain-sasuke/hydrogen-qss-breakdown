"""
verify_grid_coverage.py
=======================
What region of ITER divertor parameter space does the model grid actually
cover, and which parts of it are citable?

This exists because Chapter 1 asserts an operating range on five figures with
no source, Section 3.5.3 now restricts the quantitative results on optical
grounds, and the epsilon ridge sits at a density that may or may not be inside
the region any published source describes. Those three restrictions are
different shapes and they have never been drawn on the same axes.

The grid is read from the pipeline, not assumed. The literature boxes are
hardcoded with their source and DOI attached to each, because a number without
its source is what this whole exercise exists to prevent.

CITED REGIONS
-------------
Guillemaut, Pitts, Kukushkin & O'Mullane (2011), Fus. Eng. Des. 86, 2954,
DOI 10.1016/j.fusengdes.2011.07.008, Sec. 4 first paragraph:
    "in the divertor SOL the plasma electron temperature, Te, in the most
    intense radiative regions ... is in the range 1-10 eV and the particle
    density is very high (ne = 1e20-1e21 m^-3), as expected for a detached
    divertor plasma."
  -> Te 1-10 eV, ne 1e20-1e21 m^-3.  DETACHED TARGET REGION.
  -> Caveat: these SOLPS runs are the CFC/carbon-impurity era; the all-W
     baseline had not been simulated to the same level of detail.

Stangeby, Lore, Pitts, Canik & Bonnin (2022), Nucl. Fusion 62,
DOI 10.1088/1741-4326/ac9917 (part B; part A is ac9916):
    upstream separatrix density for the ITER Q=10 baseline SOLPS-4.3 database
    peaks at ne,sep ~ 6e19 m^-3, which is 0.5 n_GW for the 15 MA baseline, and
    the JET/AUG H-mode operational boundary is ne,sep/n_GW < 0.4-0.5.
  -> ne,sep ~ 3e19 - 6e19 m^-3.  UPSTREAM SEPARATRIX, not the target.
  -> This is a DENSITY only; it carries no Te for the divertor.

Lore et al. (2022), Nucl. Fusion 62, 106017,
DOI 10.1088/1741-4326/ac8a5f:
    at the highest neon puff levels with steady-state solutions the electron
    temperature falls below 1 eV across 50 cm of each divertor target.
  -> supports Te < 1 eV at strong detachment, i.e. the model's Te floor is a
     real operating point rather than an arbitrary grid edge.

Report only. Writes nothing.

    python src/validation/verify_grid_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "validation", HERE.parent / "analysis"):
    sys.path.insert(0, str(p))
from cr_context import CRContext  # noqa: E402

CM3_TO_M3 = 1.0e6

# (label, Te_lo, Te_hi, ne_lo_m3, ne_hi_m3, source)
REGIONS = [
    ("detached target",
     1.0, 10.0, 1.0e20, 1.0e21,
     "Guillemaut 2011 Sec.4  DOI 10.1016/j.fusengdes.2011.07.008"),
    ("upstream separatrix",
     None, None, 3.0e19, 6.0e19,
     "Stangeby 2022 B        DOI 10.1088/1741-4326/ac9917"),
]

TE_FLOOR_OPTICAL = 2.0      # Sec 3.5.3
NE_RIDGE_M3 = 1.931e19      # measured, verify_eps_gridmap.py


def main() -> None:
    ctx = CRContext.load()
    Te, ne = ctx.te_grid, ctx.ne_grid
    ne_m3 = ne * CM3_TO_M3
    nT, nN = len(Te), len(ne)

    print("=" * 78)
    print("1. THE GRID, AS READ FROM THE PIPELINE")
    print("=" * 78)
    print(f"  Te : {nT} points, {Te[0]:.4g} to {Te[-1]:.4g} eV, "
          f"ratio {Te[1]/Te[0]:.6f} per step")
    print(f"  ne : {nN} points, {ne[0]:.4g} to {ne[-1]:.4g} cm^-3 "
          f"= {ne_m3[0]:.4g} to {ne_m3[-1]:.4g} m^-3")
    print()
    print(f"  {'j':>3s} {'ne (cm^-3)':>12s} {'ne (m^-3)':>12s}   region")
    for j in range(nN):
        tags = []
        for lab, _, _, lo, hi, _ in REGIONS:
            if lo <= ne_m3[j] <= hi:
                tags.append(lab)
        if abs(ne_m3[j] - NE_RIDGE_M3) / NE_RIDGE_M3 < 0.05:
            tags.append("<<< eps ridge")
        print(f"  {j:3d} {ne[j]:12.4g} {ne_m3[j]:12.4g}   "
              f"{', '.join(tags) if tags else '-'}")

    print()
    print("=" * 78)
    print("2. COVERAGE: does the grid contain each cited region?")
    print("=" * 78)
    for lab, tlo, thi, nlo, nhi, src in REGIONS:
        cols = [j for j in range(nN) if nlo <= ne_m3[j] <= nhi]
        rows = ([i for i in range(nT) if tlo <= Te[i] <= thi]
                if tlo is not None else list(range(nT)))
        ne_ok = ne_m3[0] <= nlo and nhi <= ne_m3[-1]
        te_ok = True if tlo is None else (Te[0] <= tlo and thi <= Te[-1])
        print(f"  {lab}")
        print(f"      source        {src}")
        if tlo is None:
            print(f"      Te            not specified by this source")
        else:
            print(f"      Te  {tlo:g}-{thi:g} eV      "
                  f"spanned by grid: {te_ok}   ({len(rows)} of {nT} rows)")
        print(f"      ne  {nlo:.0e}-{nhi:.0e} m^-3  "
              f"spanned by grid: {ne_ok}   ({len(cols)} of {nN} columns: {cols})")
        if len(cols) < 2:
            print(f"      !! {len(cols)} column(s) inside this region -- the grid "
                  f"barely resolves it")
        print()

    print("=" * 78)
    print("3. WHERE THE RIDGE SITS RELATIVE TO EACH REGION")
    print("=" * 78)
    j_ridge = int(np.argmin(np.abs(ne_m3 - NE_RIDGE_M3)))
    print(f"  ridge at j={j_ridge}: {ne[j_ridge]:.4g} cm^-3 = {ne_m3[j_ridge]:.4g} m^-3")
    for lab, _, _, nlo, nhi, _ in REGIONS:
        if ne_m3[j_ridge] < nlo:
            print(f"    below the {lab} band by a factor "
                  f"{nlo/ne_m3[j_ridge]:.1f}")
        elif ne_m3[j_ridge] > nhi:
            print(f"    above the {lab} band by a factor "
                  f"{ne_m3[j_ridge]/nhi:.1f}")
        else:
            print(f"    INSIDE the {lab} band")

    print()
    print("=" * 78)
    print("4. THE THREE RESTRICTIONS, ON ONE MAP")
    print("=" * 78)
    print("  rows = Te (every 4th shown), columns = ne")
    print("  legend:  O = optically thin (Te >= %.1f, Sec 3.5.3)" % TE_FLOOR_OPTICAL)
    print("           G = inside Guillemaut detached-target box")
    print("           . = in the grid, outside both")
    print()
    hdr = "         " + "".join(f"{ne_m3[j]:>10.1e}" for j in range(nN))
    print(hdr)
    for i in range(0, nT, 4):
        cells = []
        for j in range(nN):
            optical = Te[i] >= TE_FLOOR_OPTICAL
            guil = (1.0 <= Te[i] <= 10.0) and (1e20 <= ne_m3[j] <= 1e21)
            cells.append(("O" if optical else ".") + ("G" if guil else " "))
        print(f"  {Te[i]:6.2f} " + "".join(f"{c:>10s}" for c in cells))
    print()
    n_both = sum(1 for i in range(nT) for j in range(nN)
                 if Te[i] >= TE_FLOOR_OPTICAL and 1e20 <= ne_m3[j] <= 1e21)
    print(f"  points satisfying BOTH the optical restriction and the cited "
          f"detached-target box: {n_both} of {nT*nN}")
    print()
    print("  Report only. Nothing was written.")


if __name__ == "__main__":
    main()