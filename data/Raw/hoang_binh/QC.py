import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- File paths ----------------
SCRIPT_DIR = Path(__file__).parent.resolve()

# Try to find project root (has 'data' folder)
def find_project_root(start_path: Path) -> Path:
    """Find project root by looking for 'data' directory."""
    current = start_path
    for _ in range(5):  # Search up to 5 levels
        if (current / "data").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (folder with 'data/')")

PROJECT_ROOT = find_project_root(SCRIPT_DIR)

CSV_PATH = PROJECT_ROOT / "data" / "Processed" / "Radiative" / "H_A_E1_LS_n1_15_physical.csv"
OUT_TXT = PROJECT_ROOT / "data" / "Processed" / "Radiative" / "validation.txt"
# -------------------------------------------

df = pd.read_csv(CSV_PATH)

with open(OUT_TXT, "w") as f:

    # -------- Physical l-range constraints --------
    bad1 = df[(df.lu < 0) | (df.ll < 0)]
    bad2 = df[df.lu > df.nu - 1]
    bad3 = df[df.ll > df.nl - 1]

    f.write("PHYSICAL l-RANGE CHECK\n")
    f.write("----------------------\n")
    f.write(f"bad l < 0: {len(bad1)}\n")
    f.write(f"bad lu > nu-1: {len(bad2)}\n")
    f.write(f"bad ll > nl-1: {len(bad3)}\n\n")

    if len(bad2) or len(bad3):
        f.write("Examples of invalid rows:\n")
        if len(bad2):
            f.write("\nbad2 head:\n")
            f.write(bad2.head().to_string())
            f.write("\n")
        if len(bad3):
            f.write("\nbad3 head:\n")
            f.write(bad3.head().to_string())
            f.write("\n")

    # -------- Statistical weights --------
    df["g_upper"] = 2 * (2 * df["lu"] + 1)
    df["g_lower"] = 2 * (2 * df["ll"] + 1)

    f.write("\n\nSTATISTICAL WEIGHTS ADDED\n")
    f.write("-------------------------\n")
    f.write("Columns added: g_upper, g_lower\n")

    # -------- Sum-rule check Σ A_ul --------
    f.write("\n\nTOTAL DECAY RATE CHECK  Σ A_ul\n")
    f.write("------------------------------\n")

    for n in range(2, 16):
        for l in range(n):
            from_state = df[(df.nu == n) & (df.lu == l)]
            if len(from_state) > 0:
                total_A = from_state["A_s-1"].sum()
                f.write(f"({n},{l}): Total A = {total_A:.6e} s^-1\n")

print(f"Validation written to: {OUT_TXT}")
