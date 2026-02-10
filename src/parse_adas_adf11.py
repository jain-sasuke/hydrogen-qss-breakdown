#!/usr/bin/env python3
"""
ADAS ADF11 Parser + Interpolator + Basic Physical Sanity Checks (Reviewer-Grade)

Purpose
-------
Parse ADAS ADF11 "stage-to-stage" effective coefficients (e.g., SCD, ACD),
interpolate them onto a user-defined (Te, ne) grid, export tidy CSVs,
and generate validation plots.

This script is designed for:
- Reproducible thesis / paper workflows
- GitHub publication (clean CLI, logs, docstrings, defensible assumptions)

Supported files
---------------
Typical Hydrogen ADF11 files such as:
- scdXX_h.dat  (effective ionization coefficients)
- acdXX_h.dat  (effective recombination coefficients)

Core assumptions (explicit)
--------------------------
1) ADF11 contains, in order after header lines:
   - Te grid: nTe values of log10(Te[eV])
   - ne grid: nne values of log10(ne[cm^-3])
   - coefficient table: nTe*nne values of log10(coeff[cm^3/s]) in row-major order
   This is consistent with standard ADF11 documentation and common ADAS distributions.

2) Interpolation is performed in log10-space:
   log10(coeff) interpolated over (log10 Te, log10 ne) using bicubic splines.
   This is standard practice for tabulated CR coefficients spanning many decades.

3) Coverage is strictly enforced:
   Points outside ADAS bounds are set to NaN unless `--allow-extrapolation` is passed.

4) Density dependence of "ACD includes 3BR" is NOT guaranteed by a simple ne^2 fit.
   ACD is an "effective recombination coefficient" from a bundled CR model; its ne-dependence
   can reflect collisional mixing, suppression, and bundling assumptions, not only 3BR.
   This script provides a diagnostic, not a proof.

References (for thesis citation)
--------------------------------
- ADAS ADF11 format is described in ADAS manuals / Summers et al. (ADAS documentation).
  Cite the relevant ADAS/Summers documentation version you used (e.g., ADAS User Manual,
  Summers' ADF data format notes). Do not cite this script.

Usage
-----
Example (your repo structure):
  non_markovian_cr/
    data/Raw/adas/      (input .dat)
    data/Processed/     (outputs)
    figures/            (plots)
    src/                (this script)

Run:
  python src/parse_adas_adf11.py \
    --scd data/Raw/adas/scd96_h.dat \
    --acd data/Raw/adas/acd96_h.dat \
    --out-dir data/Processed \
    --fig-dir figures \
    --te-grid "1,1.2,1.5,1.8,2,2.5,3.2,4,5,6.3,8,10" \
    --ne-grid "1e12,3e12,1e13,3e13,1e14,3e14,1e15,2e15"

Author
------
Hanagaki / Non-Markovian CR thesis workflow

License
-------
Choose a license in your repo (MIT/BSD-3-Clause recommended). This file is license-agnostic.
"""


from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("adas_adf11")


def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), None)
    if not isinstance(lvl, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -----------------------------------------------------------------------------
# Data structure
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ADF11Data:
    file_path: Path
    header_line: str
    nTe: int
    nne: int
    Te_eV: np.ndarray        # shape (nTe,)
    ne_cm3: np.ndarray       # shape (nne,)
    coeff_cm3s: np.ndarray   # shape (nTe, nne)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def project_root_from_this_file(this_file: Path) -> Path:
    """
    Assumes this script is in <repo>/src/.
    """
    return this_file.parent.parent


def parse_csv_list(s: str) -> np.ndarray:
    items = [x.strip() for x in s.split(",") if x.strip()]
    if not items:
        raise ValueError("Empty grid specification.")
    return np.array([float(x) for x in items], dtype=float)


def extract_numeric_tokens(lines: List[str], start_line: int = 2) -> List[float]:
    """
    Collect all numeric tokens after start_line. Skips non-numeric fragments robustly.
    """
    tokens: List[float] = []
    for line in lines[start_line:]:
        for t in line.split():
            try:
                tokens.append(float(t))
            except ValueError:
                continue
    return tokens


def parse_header_counts(first_line: str) -> Tuple[int, int]:
    """
    Parse (nne, nTe) from the first header line.

    Your files show:
      "1  24  29  1  1  /HYDROGEN ..."
    which corresponds to:
      nne = 24, nTe = 29

    We only rely on positions [1] and [2] as integers.
    """
    parts = first_line.split()
    if len(parts) < 3:
        raise ValueError(f"Cannot parse header counts from: {first_line}")
    try:
        nne = int(parts[1])
        nTe = int(parts[2])
    except ValueError as e:
        raise ValueError(f"Header counts are not integers in: {first_line}") from e
    if nne <= 0 or nTe <= 0:
        raise ValueError(f"Non-positive header counts parsed: nne={nne}, nTe={nTe}")
    return nne, nTe


def ensure_strictly_increasing(x: np.ndarray, name: str) -> np.ndarray:
    """
    Ensure x is strictly increasing. If decreasing, reverse it.
    If unsorted, sort (with warning).
    """
    dx = np.diff(x)
    if np.all(dx > 0):
        return x
    if np.all(dx < 0):
        logger.warning("%s grid is decreasing; reversing to increasing order.", name)
        return x[::-1]
    logger.warning("%s grid is not monotone; sorting it. Check file integrity.", name)
    return np.sort(x)


# -----------------------------------------------------------------------------
# ADF11 parser (deterministic for your format)
# -----------------------------------------------------------------------------
def parse_adf11(file_path: Path, element_guard: Optional[str] = "HYDROGEN") -> ADF11Data:
    """
    Deterministic ADF11 parsing for your H scd/acd files:

    Layout after header lines (typical for your H 96 files):
      - ne grid: nne values of log10(ne[cm^-3])
      - Te grid: nTe values of log10(Te[eV])
      - data: nTe * nne values of log10(coeff[cm^3/s])
        stored row-major as a matrix with shape (nTe, nne):
          rows correspond to Te points
          columns correspond to ne points

    This matches your header excerpt: the first block (7.69897..15.30103) is ne-log,
    the second block (-0.69897..4.00000) is Te-log.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ADF11 file not found: {file_path}")

    text = file_path.read_text(errors="replace")
    lines = text.splitlines(True)
    if not lines:
        raise ValueError(f"Empty file: {file_path}")

    header0 = lines[0].strip()
    if element_guard:
        if element_guard.upper() not in header0.upper():
            raise ValueError(
                f"Header guard failed. Expected '{element_guard}' in first line.\n"
                f"File: {file_path}\nHeader: {header0}"
            )

    nne, nTe = parse_header_counts(header0)

    tokens = extract_numeric_tokens(lines, start_line=2)
    needed = nne + nTe + (nTe * nne)
    if len(tokens) < needed:
        raise ValueError(
            f"Not enough numeric tokens in {file_path.name}: found {len(tokens)}, need {needed} "
            f"(nne={nne}, nTe={nTe})."
        )

    # IMPORTANT: ne first, then Te
    ne_log = np.array(tokens[:nne], dtype=float)
    Te_log = np.array(tokens[nne:nne + nTe], dtype=float)

    data_start = nne + nTe
    coeff_log = np.array(tokens[data_start:data_start + (nTe * nne)], dtype=float)

    ne_cm3 = 10.0 ** ne_log
    Te_eV = 10.0 ** Te_log
    coeff_cm3s = 10.0 ** coeff_log.reshape(nTe, nne)

    # Enforce increasing grids and keep matrix consistent if we reverse
    if np.all(np.diff(ne_cm3) < 0):
        ne_cm3 = ne_cm3[::-1]
        coeff_cm3s = coeff_cm3s[:, ::-1]
    elif not np.all(np.diff(ne_cm3) > 0):
        # sort and reorder columns accordingly
        idx = np.argsort(ne_cm3)
        ne_cm3 = ne_cm3[idx]
        coeff_cm3s = coeff_cm3s[:, idx]

    if np.all(np.diff(Te_eV) < 0):
        Te_eV = Te_eV[::-1]
        coeff_cm3s = coeff_cm3s[::-1, :]
    elif not np.all(np.diff(Te_eV) > 0):
        idx = np.argsort(Te_eV)
        Te_eV = Te_eV[idx]
        coeff_cm3s = coeff_cm3s[idx, :]

    # Final sanity logs
    logger.info(
        "Parsed %s | Te: %.3g..%.3g eV (%d) | ne: %.3g..%.3g cm^-3 (%d) | coeff: %.3e..%.3e",
        file_path.name,
        float(np.min(Te_eV)), float(np.max(Te_eV)), Te_eV.size,
        float(np.min(ne_cm3)), float(np.max(ne_cm3)), ne_cm3.size,
        float(np.min(coeff_cm3s)), float(np.max(coeff_cm3s)),
    )

    return ADF11Data(
        file_path=file_path,
        header_line=header0,
        nTe=nTe,
        nne=nne,
        Te_eV=Te_eV,
        ne_cm3=ne_cm3,
        coeff_cm3s=coeff_cm3s,
    )


# -----------------------------------------------------------------------------
# Interpolation
# -----------------------------------------------------------------------------
def interpolate_to_target_grid(
    adf: ADF11Data,
    Te_target: np.ndarray,
    ne_target: np.ndarray,
    allow_extrapolation: bool = False,
    kx: int = 3,
    ky: int = 3,
) -> pd.DataFrame:
    """
    Interpolate log10(coeff) over (log10 Te, log10 ne) using bicubic splines.

    Returns tidy DataFrame with columns:
      Te_eV, ne_cm3, coeff_cm3s
    """
    Te_target = np.asarray(Te_target, dtype=float)
    ne_target = np.asarray(ne_target, dtype=float)

    if np.any(Te_target <= 0) or np.any(ne_target <= 0):
        raise ValueError("Target Te and ne must be positive.")

    Te_src = adf.Te_eV
    ne_src = adf.ne_cm3
    coeff_src = adf.coeff_cm3s

    # Build interpolator in log-space
    interp = RectBivariateSpline(
        np.log10(Te_src),
        np.log10(ne_src),
        np.log10(coeff_src),
        kx=kx,
        ky=ky,
    )

    Te_min, Te_max = float(np.min(Te_src)), float(np.max(Te_src))
    ne_min, ne_max = float(np.min(ne_src)), float(np.max(ne_src))

    rows: List[Dict[str, Any]] = []
    for Te in Te_target:
        for ne in ne_target:
            in_bounds = (Te_min <= Te <= Te_max) and (ne_min <= ne <= ne_max)
            if (not in_bounds) and (not allow_extrapolation):
                coeff = np.nan
            else:
                # .ev returns a scalar value (not a 1x1 array)
                logc = float(interp.ev(np.log10(Te), np.log10(ne)))
                coeff = 10.0 ** logc
            rows.append({"Te_eV": Te, "ne_cm3": ne, "coeff_cm3s": coeff})

    df = pd.DataFrame(rows)

    n_valid = int(df["coeff_cm3s"].notna().sum())
    n_total = int(df.shape[0])
    logger.info("Interpolation coverage: %d/%d valid (%.1f%%).",
                n_valid, n_total, 100.0 * n_valid / n_total if n_total else 0.0)

    if n_valid == 0:
        logger.error("No valid points. Your target grid is outside the parsed ADAS bounds.")
        logger.error("Parsed bounds: Te=%.3g..%.3g eV, ne=%.3g..%.3g cm^-3",
                     Te_min, Te_max, ne_min, ne_max)

    return df


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def make_validation_plot(
    scd_df: pd.DataFrame,
    acd_df: pd.DataFrame,
    fig_path: Path,
    thesis_te_span: Tuple[float, float],
    thesis_ne_span: Tuple[float, float],
) -> None:
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Helper to pick only values that exist exactly in the DataFrame
    def pick_existing(series: pd.Series, candidates: List[float]) -> List[float]:
        existing = set(np.unique(series.values))
        return [c for c in candidates if c in existing]

    # Panel 1: SCD vs Te at selected ne
    ax = axes[0, 0]
    for ne in pick_existing(scd_df["ne_cm3"], [1e12, 1e13, 1e14, 1e15, 2e15]):
        sub = scd_df[(scd_df["ne_cm3"] == ne) & (scd_df["SCD_cm3s"].notna())]
        if len(sub) > 0:
            ax.loglog(sub["Te_eV"], sub["SCD_cm3s"], marker="o", linestyle="-", label=f"ne={ne:.0e}")
    ax.set_xlabel("Te [eV]")
    ax.set_ylabel("SCD [cm^3/s]")
    ax.set_title("Ionization Coefficient (SCD)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 2: ACD vs Te at selected ne
    ax = axes[0, 1]
    for ne in pick_existing(acd_df["ne_cm3"], [1e12, 1e13, 1e14, 1e15, 2e15]):
        sub = acd_df[(acd_df["ne_cm3"] == ne) & (acd_df["ACD_cm3s"].notna())]
        if len(sub) > 0:
            ax.loglog(sub["Te_eV"], sub["ACD_cm3s"], marker="s", linestyle="-", label=f"ne={ne:.0e}")
    ax.set_xlabel("Te [eV]")
    ax.set_ylabel("ACD [cm^3/s]")
    ax.set_title("Recombination Coefficient (ACD)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 3: ACD vs ne at selected Te
    ax = axes[1, 0]
    for Te in pick_existing(acd_df["Te_eV"], [1.0, 2.0, 3.0, 5.0, 10.0]):
        sub = acd_df[(acd_df["Te_eV"] == Te) & (acd_df["ACD_cm3s"].notna())]
        if len(sub) > 0:
            ax.loglog(sub["ne_cm3"], sub["ACD_cm3s"], marker="o", linestyle="-", label=f"Te={Te:g}")
    ax.set_xlabel("ne [cm^-3]")
    ax.set_ylabel("ACD [cm^3/s]")
    ax.set_title("ACD vs ne (diagnostic)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 4: coverage map + thesis regime shading
    ax = axes[1, 1]
    scd_valid = scd_df[scd_df["SCD_cm3s"].notna()]
    acd_valid = acd_df[acd_df["ACD_cm3s"].notna()]
    ax.scatter(scd_valid["Te_eV"], scd_valid["ne_cm3"], marker="o", s=55, alpha=0.7, label="SCD valid")
    ax.scatter(acd_valid["Te_eV"], acd_valid["ne_cm3"], marker="s", s=40, alpha=0.7, label="ACD valid")

    ax.axvspan(thesis_te_span[0], thesis_te_span[1], alpha=0.15, label="Target Te span")
    ax.axhspan(thesis_ne_span[0], thesis_ne_span[1], alpha=0.15, label="Target ne span")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Te [eV]")
    ax.set_ylabel("ne [cm^-3]")
    ax.set_title("Coverage Map")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved figure: %s", fig_path)


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def run(
    scd_path: Path,
    acd_path: Path,
    out_dir: Path,
    fig_dir: Path,
    Te_target: np.ndarray,
    ne_target: np.ndarray,
    allow_extrapolation: bool = False,
    element_guard: Optional[str] = "HYDROGEN",
) -> None:
    out_dir = Path(out_dir)
    fig_dir = Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Input SCD: %s", scd_path)
    logger.info("Input ACD: %s", acd_path)
    logger.info("Output dir: %s", out_dir)
    logger.info("Figure dir: %s", fig_dir)

    scd = parse_adf11(scd_path, element_guard=element_guard)
    acd = parse_adf11(acd_path, element_guard=element_guard)

    scd_df = interpolate_to_target_grid(scd, Te_target, ne_target, allow_extrapolation=allow_extrapolation)
    scd_df = scd_df.rename(columns={"coeff_cm3s": "SCD_cm3s"})

    acd_df = interpolate_to_target_grid(acd, Te_target, ne_target, allow_extrapolation=allow_extrapolation)
    acd_df = acd_df.rename(columns={"coeff_cm3s": "ACD_cm3s"})

    collision_dir = out_dir / "collisions"
    collision_dir.mkdir(parents=True, exist_ok=True)

    scd_out = collision_dir / "SCD_interpolated.csv"
    acd_out = collision_dir / "ACD_interpolated.csv"

    scd_df.to_csv(scd_out, index=False)
    acd_df.to_csv(acd_out, index=False)

    logger.info("Saved collision data:")
    logger.info("  %s", scd_out)
    logger.info("  %s", acd_out)

    fig_path = fig_dir / "ADAS_validation.png"
    make_validation_plot(
        scd_df=scd_df,
        acd_df=acd_df,
        fig_path=fig_path,
        thesis_te_span=(float(np.min(Te_target)), float(np.max(Te_target))),
        thesis_ne_span=(float(np.min(ne_target)), float(np.max(ne_target))),
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse ADAS ADF11 (SCD/ACD), interpolate to target grid, export CSVs and a validation figure."
    )

    # Defaults that match your repo structure (so it runs easily in VS Code)
    default_scd = repo_root / "data" / "Raw" / "adas" / "scd96_h.dat"
    default_acd = repo_root / "data" / "Raw" / "adas" / "acd96_h.dat"
    default_out = repo_root / "data" / "Processed"
    default_fig = repo_root / "figures"

    default_te = "1,1.2,1.5,1.8,2,2.5,3.2,4,5,6.3,8,10"
    default_ne = "1e12,3e12,1e13,3e13,1e14,3e14,1e15,2e15"

    p.add_argument("--scd", type=str, default=str(default_scd), help=f"SCD ADF11 file (default: {default_scd})")
    p.add_argument("--acd", type=str, default=str(default_acd), help=f"ACD ADF11 file (default: {default_acd})")

    p.add_argument("--out-dir", type=str, default=str(default_out), help=f"Output directory (default: {default_out})")
    p.add_argument("--fig-dir", type=str, default=str(default_fig), help=f"Figure directory (default: {default_fig})")

    p.add_argument("--te-grid", type=str, default=default_te, help=f'Te grid in eV (default: "{default_te}")')
    p.add_argument("--ne-grid", type=str, default=default_ne, help=f'ne grid in cm^-3 (default: "{default_ne}")')

    p.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="Allow spline extrapolation outside ADAS bounds (not recommended for publication).",
    )

    p.add_argument(
        "--element-guard",
        type=str,
        default="HYDROGEN",
        help="Require this substring in the first header line (case-insensitive). Set empty to disable.",
    )

    p.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")

    return p


def main() -> None:
    this_file = Path(__file__).resolve()
    repo_root = project_root_from_this_file(this_file)

    parser = build_parser(repo_root)
    args = parser.parse_args()

    setup_logging(args.log_level)

    scd_path = Path(args.scd)
    acd_path = Path(args.acd)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)

    Te_target = parse_csv_list(args.te_grid)
    ne_target = parse_csv_list(args.ne_grid)

    if np.any(Te_target <= 0) or np.any(ne_target <= 0):
        raise ValueError("Te and ne must be positive.")

    element_guard = args.element_guard.strip()
    if element_guard == "":
        element_guard_opt: Optional[str] = None
    else:
        element_guard_opt = element_guard

    run(
        scd_path=scd_path,
        acd_path=acd_path,
        out_dir=out_dir,
        fig_dir=fig_dir,
        Te_target=Te_target,
        ne_target=ne_target,
        allow_extrapolation=bool(args.allow_extrapolation),
        element_guard=element_guard_opt,
    )


if __name__ == "__main__":
    main()
