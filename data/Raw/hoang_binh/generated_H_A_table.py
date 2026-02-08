"""

Output file:
  H_A_E1_LS_n1_15_physical.csv
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd

# -------------------- User-editable defaults --------------------
WORKDIR = Path("/Users/nikhiljain/Desktop/non_markovian_cr/data/Raw/hoang_binh")          # folder containing ba5, ba5.in/out, and this script
BA5_EXE = Path("ba5")        # ba5 executable filename (in WORKDIR) OR absolute path
Z = 1.0
M_AU = 1836.152673           # proton mass in electron-mass units
N_MAX = 15

FINAL_OUT = "/Users/nikhiljain/Desktop/non_markovian_cr/data/Processed/Radiative/H_A_E1_LS_n1_15_physical.csv"
# ---------------------------------------------------------------

# ba5 row format: "lu ll R2 f A" (scientific notation)
ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([0-9.]+E[+\-]\d+)\s+([0-9.]+E[+\-]\d+)\s+([0-9.]+E[+\-]\d+)\s*$"
)

COLS_ORDER = ["nu", "lu", "nl", "ll", "A_s-1", "f_abs", "R2_au2", "Z", "M_au", "source"]


def _resolve_executable(ba5: Path, workdir: Path) -> Path:
    """
    Ensure the executable path contains a slash (absolute or ./),
    otherwise macOS will search only $PATH (not current directory).
    """
    if ba5.is_absolute():
        exe = ba5
    else:
        # resolve relative to the working directory
        exe = (workdir / ba5).resolve()
    if not exe.exists():
        raise FileNotFoundError(f"Cannot find ba5 executable at: {exe}")
    return exe


def run_ba5(nu: int, nl: int, *, z: float, m_au: float, ba5: Path, workdir: Path) -> str:
    """Run ba5 for (nu->nl). Returns ba5.out text."""
    (workdir / "ba5.in").write_text(f"{nu} {nl} {z} {m_au}\n")

    exe = _resolve_executable(ba5, workdir)

    # run ba5 inside workdir so ba5.out is created there
    subprocess.run([str(exe)], check=True, cwd=str(workdir))

    out_path = workdir / "ba5.out"
    if not out_path.exists():
        raise RuntimeError(f"ba5 ran but did not produce {out_path.resolve()}")
    return out_path.read_text(errors="replace")


def parse_ba5_out(text: str, nu: int, nl: int, *, z: float, m_au: float) -> List[Dict[str, object]]:
    """Parse ba5.out into row dicts."""
    rows: List[Dict[str, object]] = []
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue

        lu = int(m.group(1))
        ll = int(m.group(2))
        R2 = float(m.group(3))
        f_abs = float(m.group(4))
        A = float(m.group(5))

        rows.append(
            {
                "nu": nu,
                "lu": lu,
                "nl": nl,
                "ll": ll,
                "A_s-1": A,
                "f_abs": f_abs,
                "R2_au2": R2,
                "Z": z,
                "M_au": m_au,
                "source": "Hoang-Binh ba5 (ADUU v1.0)",
            }
        )
    return rows


def generate_raw_dataframe(
    *,
    n_max: int = N_MAX,
    z: float = Z,
    m_au: float = M_AU,
    ba5: Path = BA5_EXE,
    workdir: Path = WORKDIR,
) -> pd.DataFrame:
    """
    Generate the raw table in-memory (this is what the old intermediate CSV contained).
    """
    # early check: executable exists
    _resolve_executable(ba5, workdir)

    all_rows: List[Dict[str, object]] = []
    for nu in range(2, n_max + 1):
        for nl in range(1, nu):
            out_text = run_ba5(nu, nl, z=z, m_au=m_au, ba5=ba5, workdir=workdir)
            rows = parse_ba5_out(out_text, nu, nl, z=z, m_au=m_au)
            if not rows:
                raise RuntimeError(
                    f"No transition rows parsed for nu={nu}, nl={nl}. "
                    "Check ba5.out formatting or ROW_RE."
                )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = df[COLS_ORDER]  # enforce stable column order
    return df


def apply_physical_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match your physical.py intent:
    Keep only bound-state l ranges:
      0 <= lu <= nu-1
      0 <= ll <= nl-1
    """
    return df[
        (df.lu >= 0)
        & (df.lu <= df.nu - 1)
        & (df.ll >= 0)
        & (df.ll <= df.nl - 1)
    ].copy()


def main() -> None:
    df_raw = generate_raw_dataframe()
    df_phys = apply_physical_filter(df_raw)

    # QC prints (safe; does not change output)
    print("Original rows:", len(df_raw))
    print("Physical rows:", len(df_phys))
    print("Removed rows:", len(df_raw) - len(df_phys))

    bad_dl = df_phys[(df_phys.lu - df_phys.ll).abs() != 1]
    print("Bad Δl after filter:", len(bad_dl))

    print("\nTop-5 A after physical filtering:")
    print(df_phys.sort_values("A_s-1", ascending=False)[["nu", "lu", "nl", "ll", "A_s-1"]].head())

    df_phys.to_csv(FINAL_OUT, index=False)
    print(f"\nSaved: {FINAL_OUT}")


if __name__ == "__main__":
    main()
