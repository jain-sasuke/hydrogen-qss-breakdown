#!/usr/bin/env python3
"""
cr_context.py — Load the CR model's grids and state ordering from the codebase
==============================================================================

WHY THIS EXISTS
---------------
Verification scripts must not redefine the temperature grid, density grid, or
state ordering. If a verification script hardcodes them and the pipeline later
changes, the script keeps running and silently mislabels every result — which
is exactly the failure mode verification is supposed to catch.

Everything here is loaded from files the pipeline itself writes:

    data/processed/cr_matrix/Te_grid_L.npy      <- written by assemble_cr_matrix.py
    data/processed/cr_matrix/ne_grid_L.npy      <- written by assemble_cr_matrix.py
    data/processed/cr_matrix/L_grid.npy         <- written by assemble_cr_matrix.py
    data/processed/collisions/K_exc_full/state_index.csv   <- state ordering

If a file is missing, this module raises rather than falling back to a guess.
A loud failure is correct here: a silent fallback would reintroduce the very
problem this module solves.

USAGE
-----
    from cr_context import CRContext
    ctx = CRContext.load()                    # auto-discovers the repo root
    ctx = CRContext.load(root="/path/to/repo")

    ctx.te_grid      # (n_Te,)  eV
    ctx.ne_grid      # (n_ne,)  cm^-3
    ctx.L_grid       # (n_Te, n_ne, n_states, n_states)  s^-1
    ctx.labels       # list[str]   e.g. ['1S','2S','2P',...]
    ctx.n_values     # (n_states,) principal quantum number of each state
    ctx.ground_index # index of the ground state
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Locations relative to the repo root, matching the pipeline's own layout.
REL_CR_MATRIX = Path("data/processed/cr_matrix")
REL_STATE_INDEX_CANDIDATES = [
    Path("data/processed/collisions/K_exc_full/state_index.csv"),
    Path("data/processed/Radiative/state_index.csv"),
]


def find_repo_root(start: Path | None = None) -> Path:
    """
    Walk upward from `start` (or cwd) looking for the directory that contains
    data/processed/cr_matrix. This lets the script be run from anywhere in the
    tree without the caller having to know the depth.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / REL_CR_MATRIX).is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate a repo root containing {REL_CR_MATRIX} "
        f"searching upward from {here}. Pass --root explicitly."
    )


def _read_state_index(path: Path) -> tuple[list[str], np.ndarray]:
    """
    Read the pipeline's state_index.csv.

    The file is expected to have a header row and one row per state. This
    parser does not assume specific column names: it looks for a column that
    names the state and (if present) a column giving n. It reports what it
    found so the caller can sanity-check the interpretation rather than
    trusting it blindly.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")

    fieldnames = [f.strip() for f in rows[0].keys()]
    lower = {f.lower(): f for f in fieldnames}

    # Column holding a human-readable state label.
    label_col = next(
        (lower[c] for c in ("label", "state", "name", "term", "config") if c in lower),
        None,
    )
    # Column holding the principal quantum number, if the file provides it.
    n_col = next(
        (lower[c] for c in ("n", "n_principal", "principal", "shell") if c in lower),
        None,
    )
    l_col = next(
        (lower[c] for c in ("l", "ell", "l_orbital", "orbital") if c in lower),
        None,
    )

    if label_col is None and n_col is None:
        raise ValueError(
            f"{path}: could not identify a state-label or n column. "
            f"Columns present: {fieldnames}"
        )

    labels: list[str] = []
    n_values: list[float] = []

    for i, row in enumerate(rows):
        if label_col is not None:
            lab = str(row[label_col]).strip()
        else:
            lab = f"idx{i}"
        labels.append(lab)

        if n_col is not None:
            n_values.append(float(str(row[n_col]).strip()))
        else:
            # Derive n from the leading digits of the label, e.g. '3D' -> 3,
            # 'n12' -> 12, '10' -> 10.
            digits = ""
            for ch in lab:
                if ch.isdigit():
                    digits += ch
                elif digits:
                    break
            if not digits:
                raise ValueError(
                    f"{path} row {i}: cannot determine n from label {lab!r} "
                    f"and no n column is present."
                )
            n_values.append(float(digits))

    return labels, np.asarray(n_values, dtype=float)


@dataclass
class CRContext:
    """Everything a verification script needs, loaded from the pipeline."""

    root: Path
    te_grid: np.ndarray          # (n_Te,)  eV
    ne_grid: np.ndarray          # (n_ne,)  cm^-3
    L_grid: np.ndarray           # (n_Te, n_ne, n_states, n_states)  s^-1
    labels: list                 # length n_states
    n_values: np.ndarray         # (n_states,)
    state_index_path: Path

    @property
    def n_states(self) -> int:
        return self.L_grid.shape[-1]

    @property
    def ground_index(self) -> int:
        """Index of the lowest state (smallest n; ties broken by first occurrence)."""
        return int(np.argmin(self.n_values))

    @classmethod
    def load(cls, root: str | Path | None = None,
             lgrid: str | Path | None = None) -> "CRContext":
        root_path = Path(root).resolve() if root else find_repo_root()
        cr_dir = root_path / REL_CR_MATRIX

        te_path = cr_dir / "Te_grid_L.npy"
        ne_path = cr_dir / "ne_grid_L.npy"
        L_path = Path(lgrid).resolve() if lgrid else cr_dir / "L_grid.npy"

        for p in (te_path, ne_path, L_path):
            if not p.is_file():
                raise FileNotFoundError(
                    f"Required file not found: {p}\n"
                    f"Run assemble_cr_matrix.py first, or pass --root / --lgrid."
                )

        te_grid = np.load(te_path)
        ne_grid = np.load(ne_path)
        L_grid = np.load(L_path)

        si_path = next(
            (root_path / c for c in REL_STATE_INDEX_CANDIDATES
             if (root_path / c).is_file()),
            None,
        )
        if si_path is None:
            raise FileNotFoundError(
                "Could not find state_index.csv in any of:\n  "
                + "\n  ".join(str(root_path / c) for c in REL_STATE_INDEX_CANDIDATES)
                + "\nThe state ordering must come from the pipeline, not be assumed."
            )
        labels, n_values = _read_state_index(si_path)

        ctx = cls(root=root_path, te_grid=te_grid, ne_grid=ne_grid, L_grid=L_grid,
                  labels=labels, n_values=n_values, state_index_path=si_path)
        ctx.validate()
        return ctx

    def validate(self) -> None:
        """
        Cross-check that the loaded pieces are mutually consistent. These are
        the checks that catch a stale file being silently combined with a fresh
        one.
        """
        n_te, n_ne, n_a, n_b = self.L_grid.shape
        problems = []
        if n_a != n_b:
            problems.append(f"L_grid last two dims differ: {n_a} vs {n_b}")
        if len(self.te_grid) != n_te:
            problems.append(
                f"Te_grid has {len(self.te_grid)} points but L_grid has {n_te}")
        if len(self.ne_grid) != n_ne:
            problems.append(
                f"ne_grid has {len(self.ne_grid)} points but L_grid has {n_ne}")
        if len(self.labels) != n_a:
            problems.append(
                f"state_index.csv has {len(self.labels)} states but L_grid has {n_a}")
        if problems:
            raise ValueError(
                "Loaded files are mutually inconsistent — one of them is stale:\n  "
                + "\n  ".join(problems)
            )

    def nearest_point(self, te_ev: float, ne_cm3: float) -> tuple[int, int]:
        """Return (ti, ni) for the grid point nearest the requested conditions."""
        ti = int(np.argmin(np.abs(self.te_grid - te_ev)))
        ni = int(np.argmin(np.abs(np.log(self.ne_grid) - np.log(ne_cm3))))
        return ti, ni

    def describe(self) -> str:
        lines = [
            f"  repo root      : {self.root}",
            f"  L_grid         : {self.L_grid.shape}",
            f"  Te grid        : {len(self.te_grid)} points, "
            f"{self.te_grid.min():.3g} to {self.te_grid.max():.3g} eV",
            f"  ne grid        : {len(self.ne_grid)} points, "
            f"{self.ne_grid.min():.3g} to {self.ne_grid.max():.3g} cm^-3",
            f"  state ordering : {self.state_index_path.relative_to(self.root)}",
            f"  states         : {self.n_states}  "
            f"(first: {', '.join(map(str, self.labels[:4]))} ... "
            f"last: {', '.join(map(str, self.labels[-3:]))})",
            f"  n range        : {int(self.n_values.min())} to {int(self.n_values.max())}",
            f"  ground index   : {self.ground_index} ({self.labels[self.ground_index]})",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print the loaded CR context.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--lgrid", default=None)
    args = ap.parse_args()

    ctx = CRContext.load(root=args.root, lgrid=args.lgrid)
    print("\nCR context loaded from the pipeline's own files:\n")
    print(ctx.describe())
    print()
