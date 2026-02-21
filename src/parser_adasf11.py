import re
import numpy as np

# ----------------------------
# Constants (verify later)
# ----------------------------
K_B_EV_PER_K = 8.617333262145e-5  # eV/K (CODATA). Te[K] = Te[eV]/kB

# Regex for floats, supports Fortran D exponent
_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][+-]?\d+)?")


def _floats(line: str):
    """Extract floats from a line; supports Fortran D exponents."""
    out = []
    for tok in _FLOAT_RE.findall(line):
        tok = tok.replace("D", "E").replace("d", "e")
        out.append(float(tok))
    return out


def parse_adf11(filepath: str):
    """
    Parse an ADAS ADF11 file into:
      Z, IDMAXD, ITMAXD, logne, logTe, logK

    Manual-consistent meaning:
      logne = log10(ne [cm^-3]) length IDMAXD
      logTe = log10(Te [eV])    length ITMAXD
      logK  = log10(K [cm^3/s]) shaped (nblocks, ITMAXD, IDMAXD)

    Robustness:
      - Starts reading data after first dashed separator line
      - Stops reading at Fortran comment block ('C' lines)
      - Skips metadata lines containing IPRT/IGRD/Z1/DATE that include numbers but are NOT data
    """
    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"Empty file: {filepath}")

    # Header ints: IZMAX, IDMAXD, ITMAXD, IZ1MIN, IZ1MAX
    h = lines[0].split()
    if len(h) < 5:
        raise ValueError(f"Header too short: {lines[0]!r}")

    Z = int(h[0])
    IDMAXD = int(h[1])
    ITMAXD = int(h[2])

    # Collect numeric stream
    nums = []
    started = False

    for line in lines[1:]:
        s = line.strip()
        upper = s.upper()

        # Start after first dashed separator
        if not started:
            if s.startswith("-") and len(s) >= 10:
                started = True
            continue

        # Stop at comment section
        if s.startswith(("C", "c")):
            break

        # Skip metadata/separator lines that contain numbers but are not data
        if ("IPRT=" in upper) or ("IGRD=" in upper) or ("Z1=" in upper) or ("DATE=" in upper):
            continue
        # Often looks like "-----/ IPRT= 1 / ... / DATE= 19/04/99"
        if s.startswith("-") and "/" in s:
            continue

        nums.extend(_floats(line))

    nums = np.array(nums, dtype=float)

    need = IDMAXD + ITMAXD
    if nums.size < need:
        raise ValueError(f"Not enough numbers for grids: need {need}, got {nums.size}")

    logne = nums[:IDMAXD]
    logTe = nums[IDMAXD:IDMAXD + ITMAXD]
    rest = nums[IDMAXD + ITMAXD:]

    per_block = IDMAXD * ITMAXD
    if rest.size < per_block:
        raise ValueError(f"Not enough numbers for coefficients: need {per_block}, got {rest.size}")

    # Keep only full blocks
    usable = (rest.size // per_block) * per_block
    rest = rest[:usable]

    nblocks = rest.size // per_block
    logK = rest.reshape((nblocks, ITMAXD, IDMAXD))

    return Z, IDMAXD, ITMAXD, logne, logTe, logK


def validate_adas_data(Z, IDMAXD, ITMAXD, logne, logTe, logK, coeff_type="scd"):
    """
    Validation gate for ADF11 parse integrity + physics sanity.

    References:
      - ADF11 format: ADAS Manual Appendix A11
      - Order-of-magnitude sanity: Fujimoto (2004), Table 3.1 (rough)

    IMPORTANT:
      This is a *sanity check*, not a proof of LTE/detailed balance.
    """
    print("\n" + "=" * 70)
    print(f"VALIDATING ADF11 {coeff_type.upper()}")
    print("=" * 70)

    # 1) Element check
    if Z != 1:
        raise ValueError(f"Expected hydrogen (Z=1), got Z={Z}")
    print("✓ Z=1 (hydrogen)")

    # 2) Grid size check (warn, don't hard fail)
    print(f"Grid points: IDMAXD={IDMAXD} (density), ITMAXD={ITMAXD} (temperature)")
    if IDMAXD <= 0 or ITMAXD <= 0:
        raise ValueError("Invalid grid sizes.")

    # 3) Convert grids
    Te_eV = 10 ** logTe
    ne_cm3 = 10 ** logne

    # 4) Range checks
    Te_min, Te_max = Te_eV.min(), Te_eV.max()
    ne_min, ne_max = ne_cm3.min(), ne_cm3.max()

    print(f"Te range [eV]: {Te_min:.3g} → {Te_max:.3g}")
    print(f"ne range [cm^-3]: {ne_min:.3g} → {ne_max:.3g}")

    # Must cover thesis Te regime (2–10 eV)
    if not (Te_min <= 2.0 and Te_max >= 10.0):
        raise ValueError("Te grid does NOT cover 2-10 eV thesis regime.")

    # 5) Coefficient integrity in log-space (catches your earlier bug)
    if not np.all(np.isfinite(logK)):
        raise ValueError("Non-finite logK values found.")
    # Real ADF11 rates are log10(K) typically negative in most of this domain
    frac_positive = np.mean(logK > 0)
    if frac_positive > 0.01:
        raise ValueError(
            f"logK has too many positive values ({frac_positive:.2%}). "
            "Likely metadata contamination / mis-parse."
        )
    print("logK looks sane (mostly negative)")

    # 6) Convert to linear for magnitude sanity
    K = 10 ** logK
    if np.any(K <= 0):
        raise ValueError("Non-positive K found (non-physical).")

    K_min, K_med, K_max = float(K.min()), float(np.median(K)), float(K.max())
    print(f"K range [cm^3/s]: min={K_min:.3e}, median={K_med:.3e}, max={K_max:.3e}")

    # Loose magnitude windows (sanity, not strict)
    if coeff_type.lower() == "scd":
        # Ionization often ~1e-10 to 1e-7 depending on Te, ne (rough)
        if not (1e-14 < K_med < 1e-5):
            print("WARNING: SCD median magnitude looks unusual (sanity window failed).")
        else:
            print("SCD median magnitude within loose sanity window")
    else:
        # Recombination often ~1e-16 to 1e-11 depending on Te, ne (rough)
        if not (1e-20 < K_med < 1e-8):
            print("WARNING: ACD median magnitude looks unusual (sanity window failed).")
        else:
            print("ACD median magnitude within loose sanity window")

    # 7) Block count
    nblocks = logK.shape[0]
    print(f"Blocks: {nblocks}  | logK shape: {logK.shape}")
    print("=" * 70)
    print("VALIDATION PASS (sanity)")
    print("=" * 70 + "\n")

    return True

if __name__ == "__main__":
    Z, IDMAXD, ITMAXD, logne, logTe, logK = parse_adf11("./data/raw/adas/scd96_h.dat")
    validate_adas_data(Z, IDMAXD, ITMAXD, logne, logTe, logK, coeff_type="scd")
    Z, IDMAXD, ITMAXD, logne, logTe, logK = parse_adf11("./data/raw/adas/acd96_h.dat")
    validate_adas_data(Z, IDMAXD, ITMAXD, logne, logTe, logK, coeff_type="acd")


    import csv
import numpy as np

def export_csv(infile, outfile, coeff_type):
    Z, IDMAXD, ITMAXD, logne, logTe, logK = parse_adf11(infile)
    validate_adas_data(Z, IDMAXD, ITMAXD, logne, logTe, logK, coeff_type)

    ne = 10 ** logne
    Te = 10 ** logTe
    K = 10 ** logK

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Te_eV", "ne_cm3", "log10_K", "K_cm3_s"])

        for it in range(ITMAXD):
            for idd in range(IDMAXD):
                writer.writerow([
                    Te[it],
                    ne[idd],
                    logK[0, it, idd],   # hydrogen has single block
                    K[0, it, idd]
                ])

    print("Written:", outfile)


if __name__ == "__main__":

    export_csv("data/raw/adas/scd96_h.dat",
               "data/processed/adas/scd96_h_long.csv",
               "scd")

    export_csv("data/raw/adas/acd96_h.dat",
               "data/processed/adas/acd96_h_long.csv",
               "acd")